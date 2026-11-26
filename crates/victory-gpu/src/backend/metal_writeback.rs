//! GPU-written ocean textures: private Metal texture -> linear staging ->
//! console-tiled title memory, before the producer's completion is published.

use std::sync::OnceLock;

use victory_formats::texture::{Layout, Texture as ConsoleTexture};
use victory_metal::sys::WMTPixelFormat as Pixel;
use victory_metal::texture::Texture;
use victory_metal::{Buffer, CommandBuffer, Compute, ComputePipeline, Device, Error, Result};

use super::metal::MetalBackend;
use super::metal_detile::{Params, kernel_params};
use super::metal_texture::MetalTexture;
use crate::resource::{ResourceDesc, texture_layout};

const TEXTURE_2D: u32 = 3;
const DEPTH_STENCIL: u32 = 2;
const UNORDERED_ACCESS: u32 = 4;

fn shape(desc: &ResourceDesc) -> Option<(Pixel, u32)> {
    if desc.dimension != TEXTURE_2D
        || desc.sample_count != 1
        || desc.flags & UNORDERED_ACCESS == 0
        || desc.flags & DEPTH_STENCIL != 0
    {
        return None;
    }
    match (desc.width, desc.height, desc.format) {
        (256, 256, 41) => Some((Pixel::WMTPixelFormatR32Float, 4)),
        (64, 64, 16) | (256, 256, 16) => Some((Pixel::WMTPixelFormatRG32Float, 8)),
        (128, 128, 36) => Some((Pixel::WMTPixelFormatRG16Uint, 4)),
        _ => None,
    }
}

fn single_plane(layout: &ConsoleTexture, element_bytes: u32) -> Option<&Layout> {
    let [plane] = layout.planes.as_slice() else {
        return None;
    };
    (plane.offset == 0
        && !plane.stencil
        && plane.layout.addressable()
        && plane.layout.desc.element_bytes == element_bytes
        && plane.layout.desc.block == 1)
        .then_some(&plane.layout)
}

fn writeback_layout(desc: &ResourceDesc) -> Option<(ConsoleTexture, Pixel, u32)> {
    let (pixel, bytes) = shape(desc)?;
    let layout = texture_layout(desc)?;
    single_plane(&layout, bytes)?;
    Some((layout, pixel, bytes))
}

fn enabled(value: Option<&std::ffi::OsStr>) -> bool {
    value != Some(std::ffi::OsStr::new("0"))
}

fn writeback_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| enabled(std::env::var_os("VICTORY_TEXTURE_BACK").as_deref()))
}

pub(crate) fn wants_writeback(desc: &ResourceDesc) -> bool {
    writeback_enabled() && writeback_layout(desc).is_some()
}

/// A resource kept alive by the submission, with the unique (slice, mip)
/// pairs whose UAVs the submission may have written.
pub(crate) struct WritebackResource<'a> {
    pub(crate) texture: &'a MetalTexture,
    pub(crate) va: u64,
    pub(crate) subresources: &'a [(u32, u32)],
}

impl MetalBackend {
    /// Encodes a writeback after the producing encoders have ended. `Ok(false)`
    /// means disabled, outside the whitelist, or there is nothing to copy.
    /// On error, discard the command buffer instead of signalling completion.
    ///
    /// # Safety
    /// The caller must hold the resource and exclude overlapping CPU access,
    /// frees, remaps and residency invalidation through command completion.
    /// `va` must still be the writable backing of `texture`. The mapping
    /// protocol must keep cached no-copy windows current and allow creation
    /// of a window without wrapping pages from a stale resident buffer.
    pub(crate) unsafe fn writeback(
        &self,
        cb: &CommandBuffer,
        resource: &WritebackResource<'_>,
    ) -> Result<bool> {
        let tex = resource.texture;
        if resource.subresources.is_empty() || !writeback_enabled() {
            return Ok(false);
        }
        let Some((console, pixel, bytes)) = writeback_layout(&tex.desc) else {
            return Ok(false);
        };
        if tex.pixel != pixel {
            return Err(Error("Metal writeback: incompatible texture format".into()));
        }
        let layout = &console.planes[0].layout;
        let plan = Plan::new(layout, tex.slices, tex.mips, resource.subresources)?;
        let (dst, offset) = self
            .chunks()
            .window_buffer(resource.va, layout.size)
            .ok_or_else(|| Error("Metal writeback: no title memory window".into()))?;

        // The backend is a process singleton, with one Metal device.
        static TILER: OnceLock<Result<Tiler>> = OnceLock::new();
        let tiler = TILER
            .get_or_init(|| Tiler::new(&self.device))
            .as_ref()
            .map_err(|e| Error(e.0.clone()))?;
        tiler.encode(&self.device, cb, &tex.texture, &dst, offset, &plan, bytes)?;
        Ok(true)
    }
}

// F25's tile kernel, restricted to the 4- and 8-byte ocean formats.
const TILE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Params {
    ulong base, slice_stride;
    uint w, h, d, pitch_blocks, log2blk, linear, tailx, taily, pitch, bw, bh, bz, nx, ny, nz;
    uint xb[8], yb[8], zb[8];
    uint px[256], py[256], pz[256];
};

struct Linear {
    ulong row, image;
};

static inline ulong tiled_offset(constant Params& p, uint3 g, uint log2bytes) {
    if (p.linear) {
        return p.base + ulong(g.z) * p.slice_stride
             + ((ulong(g.y) * p.pitch + g.x) << log2bytes);
    }
    uint x = g.x + p.tailx, y = g.y + p.taily;
    ulong block = ulong(y / p.bh) * p.pitch_blocks + x / p.bw;
    return p.base + ulong(g.z / p.bz) * p.slice_stride + (block << p.log2blk)
         + p.px[x % p.bw] + p.py[y % p.bh] + p.pz[g.z % p.bz];
}

#define TILE(N, T, LOG) \
kernel void tile##N(device const uchar* src [[buffer(0)]], device uchar* dst [[buffer(1)]], \
                    constant Params& p [[buffer(2)]], constant Linear& l [[buffer(3)]], \
                    uint3 g [[thread_position_in_grid]]) { \
    if (g.x >= p.w || g.y >= p.h || g.z >= p.d) return; \
    ulong from = ulong(g.z) * l.image + ulong(g.y) * l.row + (ulong(g.x) << LOG); \
    *(device T*)(dst + tiled_offset(p, g, LOG)) = *(device const T*)(src + from); \
}

TILE(4, uint, 2)
TILE(8, uint2, 3)
"#;

const PARAM_DATA: usize = core::mem::offset_of!(Params, pz) + size_of::<[u32; 256]>();
const PARAM_SIZE: usize = size_of::<Params>();
const _: () = assert!(PARAM_DATA == 3244 && PARAM_SIZE == 3248);

struct Copy {
    slice: u32,
    mip: u32,
    offset: u64,
    row: u32,
    image: u32,
    params: Params,
}

struct Plan {
    copies: Vec<Copy>,
    staging_size: usize,
    tiled_size: u64,
}

impl Plan {
    fn new(layout: &Layout, slices: u32, mips: u32, subresources: &[(u32, u32)]) -> Result<Self> {
        let invalid = || Error("Metal writeback: invalid subresource or layout".into());
        let mut copies = Vec::with_capacity(subresources.len());
        let mut offset = 0u64;
        for &(slice, mip) in subresources {
            if slice >= slices || mip >= mips {
                return Err(invalid());
            }
            let params = kernel_params(layout, mip, slice).ok_or_else(invalid)?;
            let row = params
                .w
                .checked_mul(layout.desc.element_bytes)
                .and_then(|n| n.checked_next_multiple_of(256))
                .ok_or_else(invalid)?;
            let image = row.checked_mul(params.h).ok_or_else(invalid)?;
            copies.push(Copy {
                slice,
                mip,
                offset,
                row,
                image,
                params,
            });
            offset = offset.checked_add(u64::from(image)).ok_or_else(invalid)?;
        }
        Ok(Self {
            copies,
            staging_size: usize::try_from(offset).map_err(|_| invalid())?,
            tiled_size: layout.size,
        })
    }
}

struct Tiler {
    tile4: ComputePipeline,
    tile8: ComputePipeline,
}

impl Tiler {
    fn new(device: &Device) -> Result<Self> {
        let library = device.library_msl(TILE_MSL)?;
        Ok(Self {
            tile4: device.compute_pipeline_from_library(&library, "tile4")?,
            tile8: device.compute_pipeline_from_library(&library, "tile8")?,
        })
    }

    fn encode(
        &self,
        device: &Device,
        cb: &CommandBuffer,
        texture: &Texture,
        dst: &Buffer,
        offset: u64,
        plan: &Plan,
        bytes: u32,
    ) -> Result<()> {
        let pipeline = match bytes {
            4 => &self.tile4,
            8 => &self.tile8,
            _ => return Err(Error("Metal writeback: unsupported element size".into())),
        };
        if !offset.is_multiple_of(u64::from(bytes))
            || offset
                .checked_add(plan.tiled_size)
                .is_none_or(|end| end > dst.len)
        {
            return Err(Error("Metal writeback: invalid destination range".into()));
        }
        let _pool = victory_metal::Pool::new();
        // Owned storage is safe to keep in flight via the retaining command
        // buffer; it never wraps or aliases the title's memory.
        let staging = device.shared_buffer(&vec![0; plan.staging_size])?;
        let mut blit = cb.blit_encoder()?;
        for c in &plan.copies {
            blit.copy_from_texture_to_buffer(
                texture,
                c.slice,
                c.mip,
                [0; 3],
                [u64::from(c.params.w), u64::from(c.params.h), 1],
                &staging,
                c.offset,
                c.row,
                c.image,
            );
        }
        blit.finish();

        // Do not expose the Rust struct's uninitialized tail padding to Metal.
        let args: Vec<_> = plan
            .copies
            .iter()
            .map(|c| {
                let mut params = [0; PARAM_SIZE];
                // SAFETY: the homogeneous repr(C) fields before the tail are initialized.
                let fields = unsafe {
                    core::slice::from_raw_parts(
                        core::ptr::from_ref(&c.params).cast::<u8>(),
                        PARAM_DATA,
                    )
                };
                params[..PARAM_DATA].copy_from_slice(fields);
                let mut pitch = [0; 16];
                pitch[..8].copy_from_slice(&u64::from(c.row).to_le_bytes());
                pitch[8..].copy_from_slice(&u64::from(c.image).to_le_bytes());
                (params, pitch)
            })
            .collect();
        let mut steps = Vec::with_capacity(2 + 4 * plan.copies.len());
        steps.push(Compute::Pipeline {
            pipeline,
            threads_per_group: [8, 8, 1],
        });
        steps.push(Compute::Buffer {
            buffer: dst,
            offset,
            index: 1,
        });
        for (c, (params, pitch)) in plan.copies.iter().zip(&args) {
            steps.push(Compute::Buffer {
                buffer: &staging,
                offset: c.offset,
                index: 0,
            });
            steps.push(Compute::Bytes {
                bytes: params,
                index: 2,
            });
            steps.push(Compute::Bytes {
                bytes: pitch,
                index: 3,
            });
            steps.push(Compute::Dispatch {
                groups: [
                    u64::from(c.params.w.div_ceil(8)),
                    u64::from(c.params.h.div_ceil(8)),
                    1,
                ],
            });
        }
        cb.compute(&steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_formats::texture::{Dimension, Swizzle, TextureDesc};

    const CASES: [(u32, u32); 4] = [(256, 41), (256, 16), (64, 16), (128, 36)];

    fn desc(side: u32, format: u32) -> ResourceDesc {
        // SAFETY: ResourceDesc is a C record containing only integers.
        let mut d: ResourceDesc = unsafe { std::mem::zeroed() };
        d.dimension = TEXTURE_2D;
        d.width = u64::from(side);
        d.height = side;
        d.depth_or_array_size = 2;
        d.mip_levels = 4;
        d.format = format;
        d.sample_count = 1;
        d.flags = UNORDERED_ACCESS;
        d
    }

    #[test]
    fn whitelist_and_layout() {
        for (side, format) in CASES {
            let d = desc(side, format);
            assert!(writeback_layout(&d).is_some());
            let mut bad = d;
            bad.width += 1;
            assert!(writeback_layout(&bad).is_none());
            bad = d;
            bad.format = 42;
            assert!(writeback_layout(&bad).is_none());
            for flags in [0, 5, 6] {
                // not UAV, default RT XOR, and depth respectively
                bad = d;
                bad.flags = flags;
                assert!(writeback_layout(&bad).is_none());
            }
            bad = d;
            bad.sample_count = 4;
            assert!(writeback_layout(&bad).is_none());
            bad = d;
            bad.dimension = 4;
            assert!(writeback_layout(&bad).is_none());
        }
        let mut mixed = desc(256, 16);
        mixed.width = 64;
        assert!(writeback_layout(&mixed).is_none());
        let (mut c, _, bytes) = writeback_layout(&desc(256, 41)).unwrap();
        c.planes.push(c.planes[0].clone());
        assert!(single_plane(&c, bytes).is_none());
        c.planes.pop();
        assert!(single_plane(&c, 8).is_none());
        c.planes[0].offset = 256;
        assert!(single_plane(&c, bytes).is_none());
    }

    #[test]
    fn switch_and_planner() {
        assert!(enabled(None));
        assert!(enabled(Some("1".as_ref())));
        assert!(!enabled(Some("0".as_ref())));
        let (c, _, _) = writeback_layout(&desc(64, 16)).unwrap();
        let l = &c.planes[0].layout;
        let p = Plan::new(l, 2, 4, &[(1, 3), (0, 0)]).unwrap();
        let (mip, base) = (&p.copies[0], &p.copies[1]);
        assert_eq!((mip.row, mip.image), (256, 2048));
        assert_eq!(base.offset, u64::from(mip.image));
        assert_eq!(p.staging_size as u64, base.offset + u64::from(base.image));
        assert!(Plan::new(l, 2, 4, &[(2, 0)]).is_err());
        assert!(Plan::new(l, 2, 4, &[(0, 4)]).is_err());
    }

    const FILL_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define FILL(NAME, TYPE) \
kernel void NAME(texture2d<TYPE, access::write> dst [[texture(0)]], \
                 constant uint& tag [[buffer(0)]], uint2 pos [[thread_position_in_grid]]) { \
    if (pos.x >= dst.get_width() || pos.y >= dst.get_height()) return; \
    vec<TYPE, 4> value(TYPE(1 + pos.x * 3 + pos.y * 5 + tag * 100), \
                       TYPE(2 + pos.x * 7 + pos.y * 2 + tag * 209), 0, 1); \
    dst.write(value, pos); \
}

FILL(fill_f, float)
FILL(fill_u, uint)
"#;

    fn encode_fill(
        cb: &CommandBuffer,
        pso: &ComputePipeline,
        view: &Texture,
        tag: u32,
        w: u32,
        h: u32,
    ) {
        // victory-metal has the native texture-bind opcode but not a safe
        // Compute step for it. Use the shim directly for this test producer.
        use victory_metal::sys::{self, WMTComputeCommandType as Op, WMTMemoryPointer as Ptr};
        let next = |p: *const core::ffi::c_void| Ptr { ptr: p.cast_mut() };
        let dispatch = sys::wmtcmd_compute_dispatch {
            type_: Op::WMTComputeCommandDispatch,
            reserved: [0; 3],
            next: next(core::ptr::null()),
            size: sys::WMTSize {
                width: u64::from(w.div_ceil(8)),
                height: u64::from(h.div_ceil(8)),
                depth: 1,
            },
        };
        let bytes = sys::wmtcmd_compute_setbytes {
            type_: Op::WMTComputeCommandSetBytes,
            reserved: [0; 3],
            next: next(core::ptr::from_ref(&dispatch).cast()),
            bytes: next(core::ptr::from_ref(&tag).cast()),
            length: 4,
            index: 0,
        };
        let texture = sys::wmtcmd_compute_settexture {
            type_: Op::WMTComputeCommandSetTexture,
            reserved: [0; 3],
            next: next(core::ptr::from_ref(&bytes).cast()),
            texture: view.handle(),
            index: 0,
        };
        let pipeline = sys::wmtcmd_compute_setpso {
            type_: Op::WMTComputeCommandSetPSO,
            reserved: [0; 3],
            next: next(core::ptr::from_ref(&texture).cast()),
            pso: pso.handle(),
            threadgroup_size: sys::WMTSize {
                width: 8,
                height: 8,
                depth: 1,
            },
        };
        // SAFETY: the linked records, argument bytes and handles are all live
        // until encoding completes; the caller keeps the view through wait.
        unsafe {
            let encoder = sys::MTLCommandBuffer_computeCommandEncoder(cb.handle(), false);
            assert_ne!(encoder, 0);
            sys::MTLComputeCommandEncoder_encodeCommands(
                encoder,
                core::ptr::from_ref(&pipeline).cast(),
            );
            sys::MTLCommandEncoder_endEncoding(encoder);
        }
    }

    struct Pages(*mut core::ffi::c_void);

    impl Drop for Pages {
        fn drop(&mut self) {
            use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
            // SAFETY: this test allocation outlives its sole Metal wrapper.
            if !self.0.is_null() {
                unsafe { VirtualFree(self.0, 0, MEM_RELEASE) };
            }
        }
    }

    #[test]
    #[ignore = "requires a private Wine prefix with victorymetal and a real Metal device"]
    fn gpu_texture_writeback_matches_cpu_tile() {
        use victory_metal::sys::{
            self, WMTResourceOptions, WMTTextureType as Type, WMTTextureUsage,
        };
        use victory_metal::texture::TextureDescriptor;
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
        };

        let _pool = victory_metal::Pool::new();
        let device = Device::first().unwrap();
        let queue = device.new_queue(4).unwrap();
        let tiler = Tiler::new(&device).unwrap();
        let library = device.library_msl(FILL_MSL).unwrap();
        let fill_f = device
            .compute_pipeline_from_library(&library, "fill_f")
            .unwrap();
        let fill_u = device
            .compute_pipeline_from_library(&library, "fill_u")
            .unwrap();
        let subresources = [(0, 0), (0, 3), (1, 1)];
        for (side, format) in CASES {
            let d = desc(side, format);
            let (pixel, bytes) = shape(&d).unwrap();
            for swizzle in [Swizzle::SW_4KB_S, Swizzle::LINEAR] {
                let console = ConsoleTexture::new(&TextureDesc {
                    dimension: Dimension::Tex2d,
                    width: side,
                    height: side,
                    depth_or_array: 2,
                    mips: 4,
                    format,
                    samples: 1,
                    swizzle,
                })
                .unwrap();
                let layout = single_plane(&console, bytes).unwrap();
                let plan = Plan::new(layout, 2, 4, &subresources).unwrap();
                let texture = device
                    .new_texture(&TextureDescriptor {
                        pixel_format: pixel,
                        width: side,
                        height: side,
                        depth: 1,
                        array_length: 2,
                        type_: Type::WMTTextureType2DArray,
                        mipmap_level_count: 4,
                        sample_count: 1,
                        usage: WMTTextureUsage(2 | 16),
                        options: WMTResourceOptions::WMTResourceStorageModePrivate,
                    })
                    .unwrap();
                let views: Vec<_> = subresources
                    .iter()
                    .map(|&(slice, mip)| {
                        let mut id = 0;
                        // SAFETY: mip and slice exist in this private array texture.
                        let raw = unsafe {
                            sys::MTLTexture_newTextureView(
                                texture.handle(),
                                pixel,
                                Type::WMTTextureType2D,
                                mip as u16,
                                1,
                                slice as u16,
                                1,
                                super::super::metal_texture::identity(),
                                &mut id,
                            )
                        };
                        Texture::from_object(victory_metal::Object::adopt(raw).unwrap())
                    })
                    .collect();
                const OFFSET: usize = 256;
                const GUARD: u8 = 0xa7;
                let size = (OFFSET * 2 + layout.size as usize).next_multiple_of(64 << 10);
                let mut expected = vec![GUARD; size];
                // SAFETY: isolated test pages, each wrapped only once and
                // never remapped or added to any residency set.
                let pages = Pages(unsafe {
                    VirtualAlloc(
                        core::ptr::null(),
                        size,
                        MEM_RESERVE | MEM_COMMIT,
                        PAGE_READWRITE,
                    )
                });
                assert!(!pages.0.is_null());
                // SAFETY: freshly allocated range; no GPU work yet.
                unsafe { pages.0.cast::<u8>().write_bytes(GUARD, size) };
                // Drain all autoreleased holders of the wrapper before Pages drops.
                let _buffer_pool = victory_metal::Pool::new();
                let dst = unsafe { device.wrap(pages.0, size as u64) }.unwrap();
                {
                    let cb = queue.command_buffer().unwrap();
                    for (i, (c, view)) in plan.copies.iter().zip(&views).enumerate() {
                        let tag = i as u32 + 1;
                        let fill = if format == 36 { &fill_u } else { &fill_f };
                        let (w, h) = (c.params.w, c.params.h);
                        let row = (w * bytes) as usize;
                        let mut linear = Vec::with_capacity(row * h as usize);
                        for y in 0..h {
                            for x in 0..w {
                                let a = 1 + x * 3 + y * 5 + tag * 100;
                                let b = 2 + x * 7 + y * 2 + tag * 209;
                                if format == 36 {
                                    linear.extend_from_slice(&(a as u16).to_le_bytes());
                                    linear.extend_from_slice(&(b as u16).to_le_bytes());
                                } else {
                                    linear.extend_from_slice(&(a as f32).to_le_bytes());
                                    if format == 16 {
                                        linear.extend_from_slice(&(b as f32).to_le_bytes());
                                    }
                                }
                            }
                        }
                        layout
                            .tile(
                                &mut expected[OFFSET..][..layout.size as usize],
                                c.mip,
                                c.slice,
                                &linear,
                                row,
                                linear.len(),
                            )
                            .unwrap();
                        encode_fill(&cb, fill, view, tag, w, h);
                    }
                    tiler
                        .encode(&device, &cb, &texture, &dst, OFFSET as u64, &plan, bytes)
                        .unwrap();
                    cb.commit();
                    cb.wait().unwrap();
                }
                // SAFETY: GPU completion precedes this read; the pages are mapped.
                let actual = unsafe { core::slice::from_raw_parts(pages.0.cast::<u8>(), size) };
                let diff = actual.iter().zip(&expected).position(|(a, b)| a != b);
                assert!(
                    diff.is_none(),
                    "{side}x{side}f{format}, {swizzle:?}: mismatch at {diff:?}"
                );
                println!("{side}x{side}f{format} {swizzle:?}: {size} bytes match");
            }
        }
    }
}

impl MetalBackend {
    /// After a submission's dispatches: the whitelisted textures shaders
    /// can write (UNORDERED) written back into title memory, all their
    /// subresources (F44). Textures the CPU fills are never written back:
    /// that would put a stale copy over the title's data.
    pub(crate) fn writeback_all(&self, cb: &CommandBuffer) {
        const UNORDERED: u32 = 4;
        if !writeback_enabled() {
            return;
        }
        let candidates: Vec<(usize, std::sync::Arc<MetalTexture>)> = self
            .textures
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter_map(|(&k, t)| t.as_ref().map(|t| (k, t.clone())))
            .filter(|(_, t)| t.desc.flags & UNORDERED != 0 && wants_writeback(&t.desc))
            .collect();
        for (key, tex) in candidates {
            let Some((memory, _)) = crate::resource::texture_memory(key as crate::object::This) else { continue };
            let subresources: Vec<(u32, u32)> =
                (0..tex.slices.max(1)).flat_map(|s| (0..tex.mips.max(1)).map(move |m| (s, m))).collect();
            let resource = WritebackResource { texture: &tex, va: memory as u64, subresources: &subresources };
            // SAFETY: the texture is held (Arc) through the submission, which
            // is waited for before the title's fences are signalled; remaps
            // keep wrappers showing their old pages (re-wrapping is off).
            if let Err(e) = unsafe { self.writeback(cb, &resource) } {
                static SEEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                    tracing::info!(error = %e, "Metal: a texture write-back left out");
                }
            }
        }
    }
}
