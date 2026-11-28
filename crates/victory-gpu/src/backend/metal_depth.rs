//! Depth-only DSV aliases of single-channel colour textures, and colour
//! twins of depth-stencil textures shaders write ([`StencilTwin`]).

use std::collections::HashMap;

use victory_metal::ir::DescriptorEntry;
use victory_metal::render::{
    DepthStencilDesc, DepthStencilState, PassDepth, Render, RenderPass, RenderPipeline,
    RenderPipelineDesc,
};
use victory_metal::sys::{
    WMTLoadAction, WMTPixelFormat as P, WMTPrimitiveType, WMTResourceOptions, WMTStoreAction,
    WMTTextureType as T, WMTTextureUsage, WMTViewport,
};
use victory_metal::texture::{Texture, TextureDescriptor};
use victory_metal::{Buffer, CommandBuffer, Compute, ComputePipeline, Device, Error, Result};

use super::metal_texture::{MetalTexture, base_type};
use crate::resource::ResourceDesc;

#[derive(Clone)]
pub(crate) struct DepthCompanion {
    pub(crate) texture: Texture,
    colour: Texture,
    bytes: u32,
}

impl DepthCompanion {
    pub(crate) fn format(pixel: P) -> Option<P> {
        Some(match pixel {
            P::WMTPixelFormatR16Unorm
            | P::WMTPixelFormatR16Snorm
            | P::WMTPixelFormatR16Uint
            | P::WMTPixelFormatR16Sint
            | P::WMTPixelFormatR16Float => P::WMTPixelFormatDepth16Unorm,
            P::WMTPixelFormatR32Uint | P::WMTPixelFormatR32Sint | P::WMTPixelFormatR32Float => {
                P::WMTPixelFormatDepth32Float
            }
            _ => return None,
        })
    }

    pub(crate) fn new(device: &Device, colour: &MetalTexture, pixel: P) -> Result<Self> {
        let type_ = base_type(colour);
        if Self::format(colour.pixel) != Some(pixel)
            || !matches!(type_, T::WMTTextureType2D | T::WMTTextureType2DArray)
        {
            return Err(Error(
                "incompatible depth companion format or texture type".into(),
            ));
        }
        let width = colour.texture.width() as u32;
        let height = colour.texture.height() as u32;
        let texture = device.new_texture(&TextureDescriptor {
            pixel_format: pixel,
            width,
            height,
            depth: 1,
            array_length: colour.slices,
            type_,
            mipmap_level_count: colour.mips,
            sample_count: 1,
            usage: WMTTextureUsage(1 | 4), // shader read | render target
            options: WMTResourceOptions(0x20), // private, hazard tracked
        })?;
        let bytes = if pixel == P::WMTPixelFormatDepth16Unorm {
            2
        } else {
            4
        };
        Ok(Self {
            texture,
            colour: colour.texture.clone(),
            bytes,
        })
    }

    fn copy(
        &self,
        device: &Device,
        cb: &CommandBuffer,
        scratch: &mut Option<Buffer>,
        level: u16,
        slice: u16,
        to_depth: bool,
    ) -> Result<()> {
        let _pool = victory_metal::Pool::new();
        let (src, dst) = if to_depth {
            (&self.colour, &self.texture)
        } else {
            (&self.texture, &self.colour)
        };
        let width = (src.width() >> level).max(1);
        let height = (src.height() >> level).max(1);
        let row = (width as u32 * self.bytes).next_multiple_of(256);
        let pitch = row * height as u32;
        if scratch.as_ref().is_none_or(|b| b.len < u64::from(pitch)) {
            *scratch = Some(device.private_buffer(u64::from(pitch))?);
        }
        let staging = scratch.as_ref().unwrap();
        let size = [width, height, 1];
        let mut blit = cb.blit_encoder()?;
        // A DSV reinterprets the storage bits. Do not numerically convert
        // R16Float to D16_UNORM or R32Uint to D32_FLOAT.
        blit.copy_from_texture_to_buffer(
            src,
            slice.into(),
            level.into(),
            [0; 3],
            size,
            staging,
            0,
            row,
            pitch,
        );
        blit.copy_from_buffer_to_texture(
            staging,
            0,
            row,
            pitch,
            size,
            dst,
            slice.into(),
            level.into(),
            [0; 3],
        );
        blit.finish();
        Ok(())
    }
}

/// Replay-local: a failed or abandoned submission must not publish freshness.
/// Bindless shaders may write colour without a per-command resource list.
/// Advancing the epoch conservatively invalidates all cached depth contents.
#[derive(Default)]
pub(crate) struct DepthSync {
    epoch: u64,
    synced: HashMap<(u64, u16, u16), u64>,
    /// As `synced`, for twinned depth textures ([`StencilTwin`]).
    twins: HashMap<(u64, u16, u16), u64>,
    scratch: Option<Buffer>,
}

impl DepthSync {
    pub(crate) fn colour_may_change(&mut self) {
        self.epoch += 1;
    }

    pub(crate) fn load(
        &mut self,
        device: &Device,
        cb: &CommandBuffer,
        d: &DepthCompanion,
        level: u16,
        slice: u16,
    ) -> Result<()> {
        let key = (d.texture.handle(), level, slice);
        if self.synced.get(&key) != Some(&self.epoch) {
            d.copy(device, cb, &mut self.scratch, level, slice, true)?;
            self.preserve(d, level, slice);
        }
        Ok(())
    }

    pub(crate) fn store(
        &mut self,
        device: &Device,
        cb: &CommandBuffer,
        d: &DepthCompanion,
        level: u16,
        slice: u16,
    ) -> Result<()> {
        d.copy(device, cb, &mut self.scratch, level, slice, false)?;
        self.preserve(d, level, slice);
        Ok(())
    }

    /// Before a pass on (or a copy from) twinned depth texture `t`: gives it
    /// what its twins hold, unless it has it since nothing changed.
    pub(crate) fn load_twin(
        &mut self,
        device: &Device,
        cb: &CommandBuffer,
        t: &StencilTwin,
        level: u16,
        slice: u16,
    ) -> Result<()> {
        let key = (t.depth.handle(), level, slice);
        if self.twins.get(&key) != Some(&self.epoch) {
            t.copy_plane(device, cb, &mut self.scratch, level, slice, true)?;
            self.twins.insert(key, self.epoch);
        }
        Ok(())
    }

    /// After a pass that may have written twinned depth texture `t`: its
    /// twins are given what it holds.
    pub(crate) fn store_twin(
        &mut self,
        device: &Device,
        cb: &CommandBuffer,
        t: &StencilTwin,
        level: u16,
        slice: u16,
    ) -> Result<()> {
        t.copy_plane(device, cb, &mut self.scratch, level, slice, false)?;
        self.preserve_twin(t, level, slice);
        Ok(())
    }

    /// After a pass that did not write `t`: it still holds what its twins do.
    pub(crate) fn preserve_twin(&mut self, t: &StencilTwin, level: u16, slice: u16) {
        self.twins.insert((t.depth.handle(), level, slice), self.epoch);
    }

    /// The active DSV subresource cannot also be written through a colour view.
    pub(crate) fn preserve(&mut self, d: &DepthCompanion, level: u16, slice: u16) {
        self.synced
            .insert((d.texture.handle(), level, slice), self.epoch);
    }
}

/// `VICTORY_METAL_STENCIL_TWIN=0`: no twins (shaders write the
/// depth-stencil texture itself, which Metal does not allow).
pub(crate) fn stencil_twin_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_METAL_STENCIL_TWIN").as_deref() != Ok("0"))
}

const TWIN_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
vertex float4 twin_vs(uint i [[vertex_id]]) {
    return float4(float(i & 1) * 4.0 - 1.0, float(i >> 1) * 4.0 - 1.0, 0.0, 1.0);
}
struct TwinDepth { float depth [[depth(any)]]; };
fragment TwinDepth twin_load16(float4 p [[position]],
                               texture2d<float, access::read> twin [[texture(0)]]) {
    TwinDepth o;
    float step = rint(twin.read(uint2(p.xy)).x * 65535.0);
    o.depth = step * as_type<float>(0x37800080u);
    return o;
}
kernel void twin_store16(depth2d<float, access::read> depth [[texture(0)]],
                         texture2d<float, access::write> twin [[texture(1)]],
                         uint2 g [[thread_position_in_grid]]) {
    if (g.x >= twin.get_width() || g.y >= twin.get_height()) {
        return;
    }
    float step = clamp(rint(depth.read(g) / as_type<float>(0x37800080u)), 0.0, 65535.0);
    twin.write(float4(step / 65535.0), g);
}
"#;

struct TwinPasses {
    load16: RenderPipeline,
    always: DepthStencilState,
    store16: ComputePipeline,
}

fn twin_passes(device: &Device) -> Result<&'static TwinPasses> {
    static PASSES: std::sync::OnceLock<Result<TwinPasses>> = std::sync::OnceLock::new();
    PASSES
        .get_or_init(|| {
            let _pool = victory_metal::Pool::new();
            let library = device.library_msl(TWIN_MSL)?;
            let load16 = device.render_pipeline_desc(
                (&library, "twin_vs"),
                Some((&library, "twin_load16")),
                &RenderPipelineDesc {
                    depth_format: P::WMTPixelFormatDepth32Float_Stencil8.0,
                    stencil_format: P::WMTPixelFormatDepth32Float_Stencil8.0,
                    sample_count: 1,
                    topology_class: 3,
                    ..Default::default()
                },
            )?;
            let always = device.depth_stencil_state(&DepthStencilDesc {
                depth_compare: 7,
                depth_write: true,
                ..Default::default()
            })?;
            let store16 = device.compute_pipeline_from_library(&library, "twin_store16")?;
            Ok(TwinPasses { load16, always, store16 })
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// One level and slice of `t` as a 2D texture.
fn view_2d(t: &Texture, pixel: P, level: u16, slice: u16) -> Result<Texture> {
    let mut id = 0u64;
    // SAFETY: a live texture; the view retains it.
    let view = unsafe {
        victory_metal::sys::MTLTexture_newTextureView(
            t.handle(),
            pixel,
            T::WMTTextureType2D,
            level,
            1,
            slice,
            1,
            super::metal_texture::identity(),
            &mut id,
        )
    };
    victory_metal::Object::adopt(view)
        .map(Texture::from_object)
        .ok_or_else(|| Error("twin view".into()))
}

/// The colour twins of a Depth32Float_Stencil8 texture (the console's
/// depth-stencil formats) that shaders also write (A3,
/// depth-probe-scan-20261128). Metal lets no shader write a depth-stencil
/// texture, and the title's tiled lighting writes each pixel's material
/// class into the depth plane and masks into the stencil plane, then draws
/// each class with the depth test "equal" (the rectangles' depth is class
/// × 0x37800080, the float nearest 1/65535). As on D3D12 (`resource::Twin`): the stencil plane's twin
/// is R8Uint, the depth plane's R16Unorm for the console's 16-bit formats
/// (118 to 120: shaders write 16-bit steps) and R32Float otherwise.
///
/// The twins are the truth, as a [`DepthCompanion`]'s colour texture is:
/// every shader view of the texture is a view of a twin; the texture is
/// given the twins before it is a depth target ([`DepthSync::load_twin`])
/// and gives them what a pass wrote ([`DepthSync::store_twin`]). The
/// 16-bit plane goes in by a pass of ours writing step × 0x37800080, so a
/// rectangle's depth and the plane's are the same float.
pub(crate) struct StencilTwin {
    pub(crate) stencil: MetalTexture,
    pub(crate) plane: MetalTexture,
    depth: Texture,
    sixteen: bool,
}

impl StencilTwin {
    /// Whether a texture of description `desc` made in format `pixel` gets
    /// twins: depth and unordered access both, single-sampled 2D.
    pub(crate) fn wanted(desc: &ResourceDesc, pixel: P) -> bool {
        const DEPTH_AND_UNORDERED: u32 = 6;
        const TEXTURE_3D: u32 = 4;
        stencil_twin_on()
            && pixel == P::WMTPixelFormatDepth32Float_Stencil8
            && desc.flags & DEPTH_AND_UNORDERED == DEPTH_AND_UNORDERED
            && desc.sample_count <= 1
            && desc.dimension != TEXTURE_3D
    }

    pub(crate) fn new(
        device: &Device,
        depth: &Texture,
        desc: &ResourceDesc,
        mips: u32,
        slices: u32,
    ) -> Result<Self> {
        let sixteen = (118..=120).contains(&desc.format);
        let twin = |pixel: P, format: u32| -> Result<MetalTexture> {
            let mut d = *desc;
            d.format = format;
            d.flags = 4;
            let texture = device.new_texture(&TextureDescriptor {
                pixel_format: pixel,
                width: depth.width() as u32,
                height: depth.height() as u32,
                depth: 1,
                array_length: slices,
                type_: if slices > 1 { T::WMTTextureType2DArray } else { T::WMTTextureType2D },
                mipmap_level_count: mips,
                sample_count: 1,
                usage: WMTTextureUsage(1 | 2 | 0x10), // read | write | format views
                options: WMTResourceOptions(0x20), // private, hazard tracked
            })?;
            Ok(MetalTexture {
                texture,
                pixel,
                desc: d,
                mips,
                slices,
                companion: Default::default(),
                twin: None,
            })
        };
        // R8_UINT; R16_UNORM or R32_FLOAT.
        let stencil = twin(P::WMTPixelFormatR8Uint, 62)?;
        let plane = if sixteen {
            twin(P::WMTPixelFormatR16Unorm, 56)?
        } else {
            twin(P::WMTPixelFormatR32Float, 41)?
        };
        Ok(Self { stencil, plane, depth: depth.clone(), sixteen })
    }

    /// The twins' textures (to make resident).
    pub(crate) fn handles(&self) -> [victory_metal::sys::obj_handle_t; 2] {
        [self.stencil.texture.handle(), self.plane.texture.handle()]
    }

    /// The entry of a shader view (`words` as for `make_texture_entry`) of
    /// the twinned texture: of the stencil twin if it reads the stencil
    /// plane (plane slice 1 or a stencil format: 22, 47, 62), else of the
    /// depth plane's.
    pub(crate) fn entry(
        &self,
        words: Option<&[u32]>,
        unordered: bool,
    ) -> Option<(victory_metal::Object, DescriptorEntry)> {
        use super::metal_texture::make_texture_entry;
        let Some(w) = words else {
            return make_texture_entry(&self.plane, None, unordered);
        };
        // Plane slice: UAV 2D (dimension 4) word 3, 2D array (5) word 5;
        // SRV 2D word 6, 2D array word 8.
        let at = match (unordered, w.get(1)) {
            (true, Some(4)) => 3,
            (true, Some(5)) => 5,
            (false, Some(4)) => 6,
            (false, Some(5)) => 8,
            _ => usize::MAX,
        };
        let plane = w.get(at).copied().unwrap_or(0);
        let mut w = w.to_vec();
        if let Some(p) = w.get_mut(at) {
            *p = 0;
        }
        if matches!(w[0], 22 | 47 | 62) || plane == 1 {
            // D3D's G8 views read stencil in green, R8Uint gives it in red.
            if !unordered && matches!(w[0], 22 | 47) && w.len() > 2 {
                w[2] = stencil_mapping(w[2]);
            }
            w[0] = 62;
            make_texture_entry(&self.stencil, Some(&w), unordered)
        } else {
            make_texture_entry(&self.plane, Some(&w), unordered)
        }
    }

    fn copy_plane(
        &self,
        device: &Device,
        cb: &CommandBuffer,
        scratch: &mut Option<Buffer>,
        level: u16,
        slice: u16,
        to_depth: bool,
    ) -> Result<()> {
        use victory_metal::blit::{BLIT_OPTION_DEPTH, BLIT_OPTION_STENCIL};
        let width = (self.depth.width() >> level).max(1);
        let height = (self.depth.height() >> level).max(1);
        let size = [width, height, 1];
        let mut planes = vec![(&self.stencil.texture, 1u32, BLIT_OPTION_STENCIL)];
        if !self.sixteen {
            planes.push((&self.plane.texture, 4, BLIT_OPTION_DEPTH));
        }
        let most = (width as u32 * 4).next_multiple_of(256) * height as u32;
        if scratch.as_ref().is_none_or(|b| b.len < u64::from(most)) {
            *scratch = Some(device.private_buffer(u64::from(most))?);
        }
        let staging = scratch.as_ref().unwrap();
        for (twin, bytes, option) in planes {
            let row = (width as u32 * bytes).next_multiple_of(256);
            let pitch = row * height as u32;
            let mut blit = cb.blit_encoder()?;
            if to_depth {
                blit.copy_from_texture_to_buffer(twin, slice.into(), level.into(), [0; 3], size, staging, 0, row, pitch);
                blit.copy_from_buffer_to_texture_with_options(staging, 0, row, pitch, size, &self.depth, slice.into(), level, [0; 3], option);
            } else {
                blit.copy_from_texture_to_buffer_with_options(&self.depth, slice.into(), level, [0; 3], size, staging, 0, row, pitch, option);
                blit.copy_from_buffer_to_texture(staging, 0, row, pitch, size, twin, slice.into(), level.into(), [0; 3]);
            }
            blit.finish();
        }
        if !self.sixteen {
            return Ok(());
        }
        let passes = twin_passes(device)?;
        let twin = view_2d(&self.plane.texture, self.plane.pixel, level, slice)?;
        if to_depth {
            let pass = RenderPass {
                width: width as u32,
                height: height as u32,
                samples: 1,
                depth: Some(PassDepth {
                    texture: &self.depth,
                    level,
                    slice,
                    load: WMTLoadAction::WMTLoadActionDontCare,
                    store: WMTStoreAction::WMTStoreActionStore,
                    clear_depth: 0.0,
                    clear_stencil: 0,
                }),
                // The pipeline's attachments are those of the texture (depth and stencil);
                // the stencil plane, given by the blit above, is kept.
                stencil: Some(PassDepth {
                    texture: &self.depth,
                    level,
                    slice,
                    load: WMTLoadAction::WMTLoadActionLoad,
                    store: WMTStoreAction::WMTStoreActionStore,
                    clear_depth: 0.0,
                    clear_stencil: 0,
                }),
                ..Default::default()
            };
            let viewport = WMTViewport {
                originX: 0.0,
                originY: 0.0,
                width: width as f64,
                height: height as f64,
                znear: 0.0,
                zfar: 1.0,
            };
            cb.render_pass_with(
                &pass,
                &[
                    Render::Pipeline { pipeline: &passes.load16 },
                    Render::Viewports { viewports: std::slice::from_ref(&viewport) },
                    Render::DepthStencil { state: &passes.always, stencil_ref: 0 },
                    Render::FragmentTexture { texture: &twin, index: 0 },
                    Render::Draw {
                        primitive_type: WMTPrimitiveType::WMTPrimitiveTypeTriangle,
                        vertex_start: 0,
                        vertex_count: 3,
                        instance_count: 1,
                        base_instance: 0,
                    },
                ],
            )
        } else {
            let depth = view_2d(&self.depth, P::WMTPixelFormatDepth32Float_Stencil8, level, slice)?;
            cb.compute(&[
                Compute::Pipeline { pipeline: &passes.store16, threads_per_group: [8, 8, 1] },
                Compute::Texture { texture: &depth, index: 0 },
                Compute::Texture { texture: &twin, index: 1 },
                Compute::DispatchThreads { threads: [width, height, 1] },
            ])
        }
    }
}

/// A D3D12 component mapping (3 bits a channel: 0–3 a source channel, 4
/// zero, 5 one) of a G8 stencil view, for a view of the R8 stencil twin:
/// stencil (green) from red, as `stencil_in_green` gives red too.
fn stencil_mapping(mapping: u32) -> u32 {
    let mut out = mapping & !0xfff;
    for c in 0..4 {
        let source = match (mapping >> (3 * c)) & 7 {
            0 | 1 => 0,
            2 => 4,
            3 => 5,
            n => n,
        };
        out |= source << (3 * c);
    }
    out
}

#[cfg(test)]
#[path = "metal_depth_tests.rs"]
mod tests;
