//! Ordered Metal copy helpers (F50): turns D3D12 / console copy operations
//! (`CopyBufferRegion`, `CopyTextureRegion`, `CopyResource`, `CopyMemoryToMemoryX`)
//! into ordered operations on a caller-supplied [`victory_metal::CommandBuffer`].
//!
//! Two halves:
//! - Pure planning & compatibility helpers ([`CopyBox`], [`FootprintSlice`],
//!   [`TexCopyPlan`], [`decode_subresource`], [`plan_footprint_slice`],
//!   [`plan_texture_copy`], [`ranges_overlap`]): portable, tested natively.
//! - Windows/Wine GPU encoders ([`Tiler`], [`encode_buffer_copy`],
//!   [`encode_linear_buffer_to_texture`], [`encode_tiled_buffer_to_texture`],
//!   [`encode_texture_to_texture`], [`encode_texture_to_linear_buffer`],
//!   [`encode_texture_to_tiled_buffer`]): encode Metal compute/blit steps onto
//!   the caller's command buffer without hidden commits or waits.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopyBox {
    pub left: u32,
    pub top: u32,
    pub front: u32,
    pub right: u32,
    pub bottom: u32,
    pub back: u32,
}

/// Dimensions `[width, height, depth]` of mip `mip` (clamped to at least 1).
pub fn subresource_dims(width: u64, height: u32, depth: u32, mip: u32, is_3d: bool) -> [u64; 3] {
    let mw = (width >> mip).max(1);
    let mh = u64::from((height >> mip).max(1));
    let md = if is_3d { u64::from((depth >> mip).max(1)) } else { 1 };
    [mw, mh, md]
}

/// Validates `src_box` against `mip_dims` (or builds the full-subresource box
/// when `None`), returning `(clamped_box, size)`.
pub fn resolve_box(src_box: Option<CopyBox>, mip_dims: [u64; 3]) -> Option<(CopyBox, [u64; 3])> {
    let b = match src_box {
        Some(b) => b,
        None => CopyBox {
            left: 0,
            top: 0,
            front: 0,
            right: u32::try_from(mip_dims[0]).ok()?,
            bottom: u32::try_from(mip_dims[1]).ok()?,
            back: u32::try_from(mip_dims[2]).ok()?,
        },
    };
    if b.right <= b.left || b.bottom <= b.top || b.back <= b.front {
        return None;
    }
    if u64::from(b.right) > mip_dims[0]
        || u64::from(b.bottom) > mip_dims[1]
        || u64::from(b.back) > mip_dims[2]
    {
        return None;
    }
    let size = [
        u64::from(b.right - b.left),
        u64::from(b.bottom - b.top),
        u64::from(b.back - b.front),
    ];
    Some((b, size))
}

/// Decodes a D3D12 subresource index into `(plane, mip, slice)`.
pub fn decode_subresource(index: u32, mips: u32, slices: u32) -> Option<(u32, u32, u32)> {
    let mips = mips.max(1);
    let slices = slices.max(1);
    let per_plane = mips.checked_mul(slices)?;
    let plane = index / per_plane;
    let rem = index % per_plane;
    let mip = rem % mips;
    let slice = rem / mips;
    Some((plane, mip, slice))
}

/// Linear buffer slice parameters for a Metal `copyFromBuffer:toTexture:` or
/// `copyFromTexture:toBuffer:` blit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FootprintSlice {
    pub offset: u64,
    pub bytes_per_row: u32,
    pub bytes_per_image: u32,
    pub size: [u64; 3],
    pub needed_bytes: u64,
}

/// Converts a D3D12 `PlacedFootprint` + optional source box into Metal blit
/// parameters, accounting for block-compressed (BC) block units.
#[allow(clippy::too_many_arguments)]
pub fn plan_footprint_slice(
    offset: u64,
    row_pitch: u32,
    fp_width: u32,
    fp_height: u32,
    fp_depth: u32,
    src_box: Option<CopyBox>,
    block_w: u32,
    block_h: u32,
    block_bytes: u32,
) -> Option<FootprintSlice> {
    if block_w == 0 || block_h == 0 || block_bytes == 0 {
        return None;
    }
    let (b, size) = resolve_box(
        src_box,
        [u64::from(fp_width), u64::from(fp_height), u64::from(fp_depth)],
    )?;
    if b.left % block_w != 0 || b.top % block_h != 0 {
        return None;
    }
    let bw = u64::from(block_w);
    let bh = u64::from(block_h);
    let bpe = u64::from(block_bytes);
    let rows_per_slice = u64::from(fp_height).div_ceil(bh);
    let bytes_per_image = u64::from(row_pitch).checked_mul(rows_per_slice)?;
    let start_byte = offset
        .checked_add(u64::from(b.front).checked_mul(bytes_per_image)?)?
        .checked_add((u64::from(b.top) / bh).checked_mul(u64::from(row_pitch))?)?
        .checked_add((u64::from(b.left) / bw).checked_mul(bpe)?)?;
    let copy_blocks_x = size[0].div_ceil(bw);
    let copy_blocks_y = size[1].div_ceil(bh);
    let copy_row_bytes = copy_blocks_x.checked_mul(bpe)?;
    if u64::from(row_pitch) < copy_row_bytes {
        return None;
    }
    let total_rows = (size[2] - 1)
        .checked_mul(rows_per_slice)?
        .checked_add(copy_blocks_y)?;
    let needed_bytes = u64::from(row_pitch)
        .checked_mul(total_rows.saturating_sub(1))?
        .checked_add(copy_row_bytes)?;
    Some(FootprintSlice {
        offset: start_byte,
        bytes_per_row: row_pitch,
        bytes_per_image: u32::try_from(bytes_per_image).ok()?,
        size,
        needed_bytes,
    })
}

/// Whether `[a_start, a_start + a_len)` and `[b_start, b_start + b_len)` overlap.
pub fn ranges_overlap(a_start: u64, a_len: u64, b_start: u64, b_len: u64) -> bool {
    if a_len == 0 || b_len == 0 {
        return false;
    }
    let a_end = a_start.saturating_add(a_len);
    let b_end = b_start.saturating_add(b_len);
    a_start < b_end && b_start < a_end
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TexCopyPlan {
    DirectWhole,
    DirectRegion,
    ViewReinterpret { view_format: u32 },
    StagingBounce { block_w: u32, block_h: u32, block_bytes: u32 },
    Unsupported(&'static str),
}

/// Metal pixel-format properties needed for texture-to-texture copy planning.
/// Values follow Apple `MTLPixelFormat.h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MtlFmtClass {
    bytes: u32,
    block_w: u32,
    block_h: u32,
    family: u8, // 0=other, 1=8-bit, 2=16-bit, 3=32-bit, 4=64-bit, 5=128-bit, 6=bc64, 7=bc128, 8=depth
}

fn mtl_fmt_class(fmt: u32) -> Option<MtlFmtClass> {
    Some(match fmt {
        // 8-bit (1 B)
        10 | 12 | 13 | 14 => MtlFmtClass { bytes: 1, block_w: 1, block_h: 1, family: 1 },
        // 16-bit (2 B)
        20 | 22 | 23 | 24 | 25 | 30 | 32 | 33 | 34 | 40 | 41 | 42 | 43 => {
            MtlFmtClass { bytes: 2, block_w: 1, block_h: 1, family: 2 }
        }
        // 32-bit (4 B)
        53 | 54 | 55 | 60 | 62 | 63 | 64 | 65 | 70 | 71 | 72 | 73 | 74 | 80 | 81 | 90 | 91 | 92 | 93 => {
            MtlFmtClass { bytes: 4, block_w: 1, block_h: 1, family: 3 }
        }
        // 64-bit (8 B)
        103 | 104 | 105 | 110 | 112 | 113 | 114 | 115 => {
            MtlFmtClass { bytes: 8, block_w: 1, block_h: 1, family: 4 }
        }
        // 128-bit (16 B)
        123 | 124 | 125 => MtlFmtClass { bytes: 16, block_w: 1, block_h: 1, family: 5 },
        // BC1 (130, 131), BC4 (140, 141): 8 B per 4x4 block
        130 | 131 | 140 | 141 => MtlFmtClass { bytes: 8, block_w: 4, block_h: 4, family: 6 },
        // BC2 (132, 133), BC3 (134, 135), BC5 (142, 143), BC6H (150, 151), BC7 (152, 153): 16 B per 4x4 block
        132 | 133 | 134 | 135 | 142 | 143 | 150 | 151 | 152 | 153 => {
            MtlFmtClass { bytes: 16, block_w: 4, block_h: 4, family: 7 }
        }
        // Depth / Stencil
        250 | 252 | 253 | 255 | 260 | 261 => MtlFmtClass { bytes: 4, block_w: 1, block_h: 1, family: 8 },
        _ => return None,
    })
}

/// Classifies how a Metal texture-to-texture copy should be encoded.
pub fn plan_texture_copy(
    src_mtl_fmt: u32,
    dst_mtl_fmt: u32,
    src_samples: u32,
    dst_samples: u32,
    whole_resource: bool,
    src_size: [u64; 3],
    dst_size: [u64; 3],
) -> TexCopyPlan {
    if src_samples.max(1) != dst_samples.max(1) {
        return TexCopyPlan::Unsupported("sample count mismatch");
    }
    if src_mtl_fmt == dst_mtl_fmt {
        if whole_resource && src_size == dst_size {
            return TexCopyPlan::DirectWhole;
        }
        return TexCopyPlan::DirectRegion;
    }
    let (Some(sc), Some(dc)) = (mtl_fmt_class(src_mtl_fmt), mtl_fmt_class(dst_mtl_fmt)) else {
        return TexCopyPlan::Unsupported("unknown Metal pixel format");
    };
    if sc.family == 8 || dc.family == 8 {
        return TexCopyPlan::Unsupported("mismatched depth/stencil pixel formats");
    }
    if sc.block_w != dc.block_w || sc.block_h != dc.block_h {
        // Compressed <-> uncompressed copy of matching element/block byte width (e.g. BC1 <-> RG32Uint).
        if sc.bytes == dc.bytes {
            return TexCopyPlan::StagingBounce {
                block_w: sc.block_w.max(dc.block_w),
                block_h: sc.block_h.max(dc.block_h),
                block_bytes: sc.bytes,
            };
        }
        return TexCopyPlan::Unsupported("incompatible block sizes");
    }
    if sc.bytes != dc.bytes {
        return TexCopyPlan::Unsupported("texel byte width mismatch");
    }
    // Same block dimensions & same bit width -> Metal texture view reinterpretation.
    TexCopyPlan::ViewReinterpret { view_format: dst_mtl_fmt }
}

/// F25 tile compute shader (`tile.metal`): linear Metal blit rows -> console tiled layout.
#[cfg(windows)]
pub const TILE_MSL: &str = r#"
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

TILE(1, uchar, 0)
TILE(2, ushort, 1)
TILE(4, uint, 2)
TILE(8, uint2, 3)
TILE(16, uint4, 4)
"#;

#[cfg(windows)]
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct TileLinear {
    row: u64,
    image: u64,
}

#[cfg(windows)]
const TILE_GROUP_X: u64 = 32;
#[cfg(windows)]
const TILE_GROUP_Y: u64 = 8;

/// GPU tiling pipelines (`tile1`..`tile16`), inverse of [`super::metal_detile::Detiler`].
#[cfg(windows)]
pub(crate) struct Tiler {
    kernels: [victory_metal::ComputePipeline; 5],
}

#[cfg(windows)]
fn params_element_bytes(p: &super::metal_detile::Params) -> u32 {
    if p.linear != 0 {
        256 / p.bw
    } else {
        (1u32 << p.log2blk) / (p.bw * p.bh * p.bz)
    }
}

#[cfg(windows)]
impl Tiler {
    pub(crate) fn new(device: &victory_metal::Device) -> victory_metal::Result<Self> {
        let names = ["tile1", "tile2", "tile4", "tile8", "tile16"];
        let mut kernels = Vec::with_capacity(5);
        for name in names {
            kernels.push(device.compute_pipeline_msl(TILE_MSL, name)?);
        }
        Ok(Self {
            kernels: kernels
                .try_into()
                .map_err(|_| victory_metal::Error("internal: tile kernel count".into()))?,
        })
    }

    /// Encodes one tile dispatch into `steps`:
    /// `src[src_offset + linear_row_pitch/image_pitch]` -> `dst[dst_offset + console tiled]`.
    pub(crate) fn encode<'a>(
        &'a self,
        steps: &mut Vec<victory_metal::Compute<'a>>,
        src: &'a victory_metal::Buffer,
        src_offset: u64,
        dst: &'a victory_metal::Buffer,
        dst_offset: u64,
        params: &'a super::metal_detile::Params,
        linear: &'a TileLinear,
    ) {
        let bpe = params_element_bytes(params);
        let log2 = bpe.trailing_zeros() as usize;
        let groups = [
            params.w.div_ceil(TILE_GROUP_X as u32) as u64,
            params.h.div_ceil(TILE_GROUP_Y as u32) as u64,
            params.d.max(1) as u64,
        ];
        steps.push(victory_metal::Compute::Pipeline {
            pipeline: &self.kernels[log2],
            threads_per_group: [TILE_GROUP_X, TILE_GROUP_Y, 1],
        });
        steps.push(victory_metal::Compute::Buffer { buffer: src, offset: src_offset, index: 0 });
        steps.push(victory_metal::Compute::Buffer { buffer: dst, offset: dst_offset, index: 1 });
        // SAFETY: `params` and `linear` are `#[repr(C)]` plain structs borrowed for `'a`;
        // `Compute::Bytes` copies them immediately during `CommandBuffer::compute`.
        let p_bytes = unsafe {
            core::slice::from_raw_parts(
                core::ptr::from_ref(params).cast::<u8>(),
                core::mem::size_of::<super::metal_detile::Params>(),
            )
        };
        let l_bytes = unsafe {
            core::slice::from_raw_parts(
                core::ptr::from_ref(linear).cast::<u8>(),
                core::mem::size_of::<TileLinear>(),
            )
        };
        steps.push(victory_metal::Compute::Bytes { bytes: p_bytes, index: 2 });
        steps.push(victory_metal::Compute::Bytes { bytes: l_bytes, index: 3 });
        steps.push(victory_metal::Compute::Dispatch { groups });
    }
}

/// Ordered buffer-to-buffer copy on `cb`. Uses `scratch` bounce when `src` and
/// `dst` alias overlapping byte ranges.
#[cfg(windows)]
pub(crate) fn encode_buffer_copy(
    cb: &victory_metal::CommandBuffer,
    src: &victory_metal::Buffer,
    src_offset: u64,
    dst: &victory_metal::Buffer,
    dst_offset: u64,
    length: u64,
    scratch: Option<(&victory_metal::Buffer, u64)>,
) -> victory_metal::Result<()> {
    if length == 0 {
        return Ok(());
    }
    let mut blit = cb.blit_encoder()?;
    if src.handle() == dst.handle() && ranges_overlap(src_offset, length, dst_offset, length) {
        let (tmp, tmp_off) = scratch.ok_or_else(|| {
            victory_metal::Error("overlapping buffer copy requires scratch buffer".into())
        })?;
        blit.copy_buffer(src, src_offset, tmp, tmp_off, length);
        blit.copy_buffer(tmp, tmp_off, dst, dst_offset, length);
    } else {
        blit.copy_buffer(src, src_offset, dst, dst_offset, length);
    }
    blit.finish();
    Ok(())
}

/// Ordered linear buffer -> Metal texture blit on `cb`.
#[cfg(windows)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_linear_buffer_to_texture(
    cb: &victory_metal::CommandBuffer,
    src: &victory_metal::Buffer,
    src_base_offset: u64,
    slice_info: &FootprintSlice,
    dst: &victory_metal::texture::Texture,
    dst_slice: u32,
    dst_level: u32,
    dst_origin: [u64; 3],
    options: u16,
) -> victory_metal::Result<()> {
    let mut blit = cb.blit_encoder()?;
    let offset = src_base_offset + slice_info.offset;
    if options == victory_metal::blit::BLIT_OPTION_NONE {
        blit.copy_from_buffer_to_texture(
            src,
            offset,
            slice_info.bytes_per_row,
            slice_info.bytes_per_image,
            slice_info.size,
            dst,
            dst_slice,
            dst_level,
            dst_origin,
        );
    } else {
        blit.copy_from_buffer_to_texture_with_options(
            src,
            offset,
            slice_info.bytes_per_row,
            slice_info.bytes_per_image,
            slice_info.size,
            dst,
            dst_slice,
            u16::try_from(dst_level).map_err(|_| victory_metal::Error("mip level overflow".into()))?,
            dst_origin,
            options,
        );
    }
    blit.finish();
    Ok(())
}

/// Ordered console-tiled buffer -> staging detile -> Metal texture blit on `cb`.
#[cfg(windows)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_tiled_buffer_to_texture(
    cb: &victory_metal::CommandBuffer,
    detiler: &super::metal_detile::Detiler,
    src_tiled: &victory_metal::Buffer,
    src_offset: u64,
    params: &super::metal_detile::Params,
    staging: &victory_metal::Buffer,
    staging_offset: u64,
    staging_row_pitch: u32,
    staging_image_pitch: u32,
    dst: &victory_metal::texture::Texture,
    dst_slice: u32,
    dst_level: u32,
    dst_origin: [u64; 3],
    texel_size: [u64; 3],
) -> victory_metal::Result<()> {
    let mut steps = Vec::new();
    detiler.encode(&mut steps, src_tiled, src_offset, staging, staging_offset, params);
    cb.compute(&steps)?;
    let mut blit = cb.blit_encoder()?;
    blit.copy_from_buffer_to_texture(
        staging,
        staging_offset,
        staging_row_pitch,
        staging_image_pitch,
        texel_size,
        dst,
        dst_slice,
        dst_level,
        dst_origin,
    );
    blit.finish();
    Ok(())
}

/// Ordered Metal texture -> Metal texture copy on `cb` (direct whole, direct region,
/// view reinterpretation, or staging bounce for BC <-> uncompressed).
#[cfg(windows)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_texture_to_texture(
    cb: &victory_metal::CommandBuffer,
    src: &victory_metal::texture::Texture,
    src_slice: u32,
    src_level: u32,
    src_origin: [u64; 3],
    src_size: [u64; 3],
    dst: &victory_metal::texture::Texture,
    dst_slice: u32,
    dst_level: u32,
    dst_origin: [u64; 3],
    whole_resource: bool,
    staging: Option<(&victory_metal::Buffer, u64)>,
) -> victory_metal::Result<()> {
    let src_fmt = src.pixel_format().0;
    let dst_fmt = dst.pixel_format().0;
    let plan = plan_texture_copy(
        src_fmt,
        dst_fmt,
        1,
        1,
        whole_resource,
        [src.width(), src.height(), src.depth()],
        [dst.width(), dst.height(), dst.depth()],
    );
    match plan {
        TexCopyPlan::DirectWhole => {
            let mut blit = cb.blit_encoder()?;
            blit.copy_texture(src, dst);
            blit.finish();
        }
        TexCopyPlan::DirectRegion => {
            let mut blit = cb.blit_encoder()?;
            blit.copy_from_texture_to_texture(
                src, src_slice, src_level, src_origin, src_size, dst, dst_slice, dst_level, dst_origin,
            );
            blit.finish();
        }
        TexCopyPlan::ViewReinterpret { view_format } => {
            let _pool = victory_metal::Pool::new();
            let swizzle = victory_metal::sys::WMTTextureSwizzleChannels {
                r: victory_metal::sys::WMTTextureSwizzle::WMTTextureSwizzleRed,
                g: victory_metal::sys::WMTTextureSwizzle::WMTTextureSwizzleGreen,
                b: victory_metal::sys::WMTTextureSwizzle::WMTTextureSwizzleBlue,
                a: victory_metal::sys::WMTTextureSwizzle::WMTTextureSwizzleAlpha,
            };
            let mut unused_id = 0u64;
            let tex_type = if src.depth() > 1 {
                victory_metal::sys::WMTTextureType::WMTTextureType3D
            } else if src.array_length() > 1 {
                victory_metal::sys::WMTTextureType::WMTTextureType2DArray
            } else {
                victory_metal::sys::WMTTextureType::WMTTextureType2D
            };
            // SAFETY: live texture `src`; view keeps the underlying allocation alive for the command buffer.
            let view_handle = unsafe {
                victory_metal::sys::MTLTexture_newTextureView(
                    src.handle(),
                    victory_metal::sys::WMTPixelFormat(view_format),
                    tex_type,
                    src_level as u16,
                    1,
                    src_slice as u16,
                    1,
                    swizzle,
                    &mut unused_id,
                )
            };
            let obj = victory_metal::Object::adopt(view_handle)
                .ok_or_else(|| victory_metal::Error("newTextureView for copy failed".into()))?;
            let view = victory_metal::texture::Texture::from_object(obj);
            let mut blit = cb.blit_encoder()?;
            blit.copy_from_texture_to_texture(
                &view, 0, 0, src_origin, src_size, dst, dst_slice, dst_level, dst_origin,
            );
            blit.finish();
        }
        TexCopyPlan::StagingBounce { block_w, block_h, block_bytes } => {
            let (buf, off) = staging.ok_or_else(|| {
                victory_metal::Error("BC <-> integer texture copy requires staging buffer".into())
            })?;
            let sc = mtl_fmt_class(src_fmt).ok_or_else(|| victory_metal::Error("src fmt".into()))?;
            let dc = mtl_fmt_class(dst_fmt).ok_or_else(|| victory_metal::Error("dst fmt".into()))?;
            // Src texel extent in `src_size` is in src texture texels; dst texel extent scales by block ratio.
            let blocks_x = src_size[0].div_ceil(u64::from(sc.block_w));
            let blocks_y = src_size[1].div_ceil(u64::from(sc.block_h));
            let row_bytes = u32::try_from(blocks_x * u64::from(block_bytes))
                .map_err(|_| victory_metal::Error("row_bytes overflow".into()))?;
            let img_bytes = u32::try_from(u64::from(row_bytes) * blocks_y)
                .map_err(|_| victory_metal::Error("img_bytes overflow".into()))?;
            let dst_size = [
                blocks_x * u64::from(dc.block_w),
                blocks_y * u64::from(dc.block_h),
                src_size[2],
            ];
            let _ = (block_w, block_h);
            let mut blit = cb.blit_encoder()?;
            blit.copy_from_texture_to_buffer(
                src, src_slice, src_level, src_origin, src_size, buf, off, row_bytes, img_bytes,
            );
            blit.copy_from_buffer_to_texture(
                buf, off, row_bytes, img_bytes, dst_size, dst, dst_slice, dst_level, dst_origin,
            );
            blit.finish();
        }
        TexCopyPlan::Unsupported(why) => {
            return Err(victory_metal::Error(format!("unsupported texture copy: {why}")));
        }
    }
    Ok(())
}

/// Ordered Metal texture -> linear buffer blit on `cb`.
#[cfg(windows)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_texture_to_linear_buffer(
    cb: &victory_metal::CommandBuffer,
    src: &victory_metal::texture::Texture,
    src_slice: u32,
    src_level: u32,
    src_origin: [u64; 3],
    slice_info: &FootprintSlice,
    dst: &victory_metal::Buffer,
    dst_base_offset: u64,
    options: u16,
) -> victory_metal::Result<()> {
    let mut blit = cb.blit_encoder()?;
    let offset = dst_base_offset + slice_info.offset;
    if options == victory_metal::blit::BLIT_OPTION_NONE {
        blit.copy_from_texture_to_buffer(
            src,
            src_slice,
            src_level,
            src_origin,
            slice_info.size,
            dst,
            offset,
            slice_info.bytes_per_row,
            slice_info.bytes_per_image,
        );
    } else {
        blit.copy_from_texture_to_buffer_with_options(
            src,
            src_slice,
            u16::try_from(src_level).map_err(|_| victory_metal::Error("mip level overflow".into()))?,
            src_origin,
            slice_info.size,
            dst,
            offset,
            slice_info.bytes_per_row,
            slice_info.bytes_per_image,
            options,
        );
    }
    blit.finish();
    Ok(())
}

/// Ordered Metal texture -> staging blit -> GPU tile into console-tiled buffer on `cb`.
#[cfg(windows)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_texture_to_tiled_buffer(
    cb: &victory_metal::CommandBuffer,
    tiler: &Tiler,
    src: &victory_metal::texture::Texture,
    src_slice: u32,
    src_level: u32,
    src_origin: [u64; 3],
    texel_size: [u64; 3],
    staging: &victory_metal::Buffer,
    staging_offset: u64,
    staging_row_pitch: u32,
    staging_image_pitch: u32,
    dst_tiled: &victory_metal::Buffer,
    dst_offset: u64,
    params: &super::metal_detile::Params,
) -> victory_metal::Result<()> {
    let mut blit = cb.blit_encoder()?;
    blit.copy_from_texture_to_buffer(
        src,
        src_slice,
        src_level,
        src_origin,
        texel_size,
        staging,
        staging_offset,
        staging_row_pitch,
        staging_image_pitch,
    );
    blit.finish();
    let linear = TileLinear {
        row: u64::from(staging_row_pitch),
        image: u64::from(staging_image_pitch),
    };
    let mut steps = Vec::new();
    tiler.encode(&mut steps, staging, staging_offset, dst_tiled, dst_offset, params, &linear);
    cb.compute(&steps)?;
    Ok(())
}
