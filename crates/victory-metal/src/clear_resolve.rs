//! Partial-rect render target / depth-stencil clears and MSAA resolves on Metal.
//!
//! - [`ClearResolve::clear_rects`]: clears one or more pixel rectangles on a
//!   color or depth/stencil attachment using a `LoadAction::Load` render pass,
//!   a cached format-specific fullscreen-triangle pipeline, and per-rect scissor
//!   rects.
//! - [`ClearResolve::resolve`]: resolves an MSAA source texture into a
//!   single-sample destination subresource. Uses a hardware render pass with
//!   `resolveTexture` + `StoreActionStoreAndMultisampleResolve` where Metal
//!   allows it (float/unorm/snorm/sRGB color formats), and falls back to
//!   sample-0 / averaging shaders for integer, depth, and 32-bit float formats.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::render::{
    ColorDesc, DepthStencilDesc, DepthStencilState, PassColor, PassDepth, Render, RenderPass,
    RenderPipeline, RenderPipelineDesc, StencilFace,
};
use crate::sys::{
    self, WMTPixelFormat, WMTScissorRect, WMTTextureSwizzle, WMTTextureSwizzleChannels,
    WMTTextureType, WMTViewport,
};
use crate::texture::{Texture, TextureDescriptor};
use crate::{CommandBuffer, Compute, ComputePipeline, Device, Error, Object, Pool, Result};

/// A D3D12-style pixel rectangle (`[left, right) x [top, bottom)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClearRect {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

/// Resolve a rectangle at `src_origin` into the destination mip at `dst_origin`.
#[derive(Clone, Copy, Debug)]
pub struct ResolveRegion {
    pub src_origin: [u32; 2],
    pub dst_origin: [u32; 2],
    pub size: [u32; 2],
}

/// Target subresource for [`ClearResolve::clear_rects`].
#[derive(Clone, Copy, Debug)]
pub struct ClearTarget<'a> {
    pub texture: &'a Texture,
    /// Metal pixel format of the view/attachment (usually `texture.pixel_format()`).
    pub pixel_format: WMTPixelFormat,
    pub level: u16,
    pub slice: u16,
    /// Sample count of `texture` (1 for non-MSAA, 2/4/8 for MSAA).
    pub samples: u8,
}

impl ClearTarget<'_> {
    /// Pixel dimensions `(width, height)` of `self.level`.
    pub fn mip_size(&self) -> (u32, u32) {
        let w = (self.texture.width() as u32 >> self.level).max(1);
        let h = (self.texture.height() as u32 >> self.level).max(1);
        (w, h)
    }
}

/// Value to clear with in [`ClearResolve::clear_rects`].
#[derive(Clone, Copy, Debug)]
pub enum ClearValue {
    /// Four floats as passed to D3D12 `ClearRenderTargetView` (converted to
    /// `uint4` / `int4` by truncation when the target is an integer format).
    Color([f32; 4]),
    /// Depth and/or stencil values as passed to D3D12 `ClearDepthStencilView`.
    DepthStencil { depth: Option<f32>, stencil: Option<u8> },
}

/// How an attachment pixel format is written by a shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColorKind {
    Float,
    Uint,
    Sint,
}

/// Format properties relevant to clear and resolve pipelines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FormatPlanes {
    pub has_depth: bool,
    pub has_stencil: bool,
    pub color: Option<ColorKind>,
}

/// Classifies a Metal pixel format into color (`Float`/`Uint`/`Sint`) or
/// depth/stencil planes.
pub fn classify_format(format: WMTPixelFormat) -> FormatPlanes {
    match format {
        WMTPixelFormat::WMTPixelFormatDepth16Unorm | WMTPixelFormat::WMTPixelFormatDepth32Float => {
            FormatPlanes { has_depth: true, has_stencil: false, color: None }
        }
        WMTPixelFormat::WMTPixelFormatStencil8
        | WMTPixelFormat::WMTPixelFormatX32_Stencil8
        | WMTPixelFormat::WMTPixelFormatX24_Stencil8 => {
            FormatPlanes { has_depth: false, has_stencil: true, color: None }
        }
        WMTPixelFormat::WMTPixelFormatDepth24Unorm_Stencil8
        | WMTPixelFormat::WMTPixelFormatDepth32Float_Stencil8 => {
            FormatPlanes { has_depth: true, has_stencil: true, color: None }
        }
        WMTPixelFormat::WMTPixelFormatR8Uint
        | WMTPixelFormat::WMTPixelFormatR16Uint
        | WMTPixelFormat::WMTPixelFormatRG8Uint
        | WMTPixelFormat::WMTPixelFormatR32Uint
        | WMTPixelFormat::WMTPixelFormatRG16Uint
        | WMTPixelFormat::WMTPixelFormatRGBA8Uint
        | WMTPixelFormat::WMTPixelFormatRGB10A2Uint
        | WMTPixelFormat::WMTPixelFormatRG32Uint
        | WMTPixelFormat::WMTPixelFormatRGBA16Uint
        | WMTPixelFormat::WMTPixelFormatRGBA32Uint => {
            FormatPlanes { has_depth: false, has_stencil: false, color: Some(ColorKind::Uint) }
        }
        WMTPixelFormat::WMTPixelFormatR8Sint
        | WMTPixelFormat::WMTPixelFormatR16Sint
        | WMTPixelFormat::WMTPixelFormatRG8Sint
        | WMTPixelFormat::WMTPixelFormatR32Sint
        | WMTPixelFormat::WMTPixelFormatRG16Sint
        | WMTPixelFormat::WMTPixelFormatRGBA8Sint
        | WMTPixelFormat::WMTPixelFormatRG32Sint
        | WMTPixelFormat::WMTPixelFormatRGBA16Sint
        | WMTPixelFormat::WMTPixelFormatRGBA32Sint => {
            FormatPlanes { has_depth: false, has_stencil: false, color: Some(ColorKind::Sint) }
        }
        _ => FormatPlanes { has_depth: false, has_stencil: false, color: Some(ColorKind::Float) },
    }
}

/// Whether `format` supports hardware `StoreActionMultisampleResolve` via a
/// Metal render pass color attachment on Apple7 (M1 Max).
///
/// Integer formats (`Uint`/`Sint`), depth/stencil formats, and 32-bit float
/// color formats (`R32Float`, `RG32Float`, `RGBA32Float`) are not resolved via
/// hardware `resolveTexture` on Apple7 and must use the shader fallback.
pub fn supports_hw_color_resolve(format: WMTPixelFormat) -> bool {
    let planes = classify_format(format);
    if planes.has_depth || planes.has_stencil {
        return false;
    }
    if !matches!(planes.color, Some(ColorKind::Float)) {
        return false;
    }
    !matches!(
        format,
        WMTPixelFormat::WMTPixelFormatR32Float
            | WMTPixelFormat::WMTPixelFormatRG32Float
            | WMTPixelFormat::WMTPixelFormatRGBA32Float
    )
}

fn resolve_color_bytes(format: WMTPixelFormat) -> Option<u8> {
    // Uncompressed MSAA colour formats, grouped by Metal view width.
    match format.0 {
        1 | 10..=14 => Some(1),
        20 | 22..=25 | 30..=34 | 40..=43 => Some(2),
        53..=55 | 60 | 62..=65 | 70..=74 | 80..=81 | 90..=94 => Some(4),
        103..=105 | 110 | 112..=115 => Some(8),
        123..=125 => Some(16),
        _ => None,
    }
}

/// Clamps `rects` to `[0, width) x [0, height)` and drops empty rectangles.
pub fn clamp_clear_rects(rects: &[ClearRect], width: u32, height: u32) -> Vec<WMTScissorRect> {
    let w = width as i32;
    let h = height as i32;
    let mut out = Vec::with_capacity(rects.len());
    for r in rects {
        let left = r.left.clamp(0, w);
        let top = r.top.clamp(0, h);
        let right = r.right.clamp(0, w);
        let bottom = r.bottom.clamp(0, h);
        if right > left && bottom > top {
            out.push(WMTScissorRect {
                x: left as u64,
                y: top as u64,
                width: (right - left) as u64,
                height: (bottom - top) as u64,
            });
        }
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ClearPipelineKey {
    Color { format: u32, samples: u8 },
    DepthStencil { depth_format: u32, stencil_format: u32, samples: u8 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ResolveComputeKind {
    AvgFloat,
    Sample0Float,
    Sample0Uint,
    Sample0Sint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct DepthResolveKey {
    depth_format: u32,
    stencil_format: u32,
}

/// Cached helper pipelines for partial-rect clears and MSAA resolves.
pub struct ClearResolve {
    device: Device,
    library: Object,
    clear_psos: Mutex<HashMap<ClearPipelineKey, RenderPipeline>>,
    clear_dss: Mutex<HashMap<(bool, bool), DepthStencilState>>,
    resolve_compute_psos: Mutex<HashMap<ResolveComputeKind, ComputePipeline>>,
    resolve_depth_psos: Mutex<HashMap<DepthResolveKey, RenderPipeline>>,
    resolve_depth_dss: DepthStencilState,
}

const CLEAR_RESOLVE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct ClearOut {
    float4 position [[position]];
};

vertex ClearOut clear_vs(uint vid [[vertex_id]], constant float &depth_z [[buffer(0)]]) {
    float2 uv = float2((vid << 1) & 2, vid & 2);
    ClearOut out;
    out.position = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), depth_z, 1.0);
    return out;
}

fragment float4 clear_fs_float(constant float4 &color [[buffer(0)]]) {
    return color;
}

fragment uint4 clear_fs_uint(constant uint4 &color [[buffer(0)]]) {
    return color;
}

fragment int4 clear_fs_sint(constant int4 &color [[buffer(0)]]) {
    return color;
}

kernel void resolve_avg_float(
    texture2d_ms<float, access::read> src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant uint2 &origin [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) {
        return;
    }
    uint n = src.get_num_samples();
    float4 sum = float4(0.0);
    for (uint s = 0; s < n; s++) {
        sum += src.read(origin + gid, s);
    }
    dst.write(sum / float(n), gid);
}

kernel void resolve_sample0_float(
    texture2d_ms<float, access::read> src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant uint2 &origin [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) {
        return;
    }
    dst.write(src.read(origin + gid, 0), gid);
}

kernel void resolve_sample0_uint(
    texture2d_ms<uint, access::read> src [[texture(0)]],
    texture2d<uint, access::write> dst [[texture(1)]],
    constant uint2 &origin [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) {
        return;
    }
    dst.write(src.read(origin + gid, 0), gid);
}

kernel void resolve_sample0_sint(
    texture2d_ms<int, access::read> src [[texture(0)]],
    texture2d<int, access::write> dst [[texture(1)]],
    constant uint2 &origin [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) {
        return;
    }
    dst.write(src.read(origin + gid, 0), gid);
}

struct DepthResolveOut {
    float depth [[depth(any)]];
};

fragment DepthResolveOut resolve_depth_fs(
    float4 pos [[position]],
    depth2d_ms<float, access::read> src [[texture(0)]],
    constant uint *params [[buffer(0)]])
{
    DepthResolveOut out;
    uint2 xy = uint2(pos.xy) - uint2(params[2], params[3]) + uint2(params[0], params[1]);
    uint n = params[4] ? src.get_num_samples() : 1;
    float sum = 0.0;
    for (uint s = 0; s < n; s++) {
        sum += src.read(xy, s);
    }
    out.depth = sum / float(n);
    return out;
}
"#;

fn identity_swizzle() -> WMTTextureSwizzleChannels {
    WMTTextureSwizzleChannels {
        r: WMTTextureSwizzle::WMTTextureSwizzleRed,
        g: WMTTextureSwizzle::WMTTextureSwizzleGreen,
        b: WMTTextureSwizzle::WMTTextureSwizzleBlue,
        a: WMTTextureSwizzle::WMTTextureSwizzleAlpha,
    }
}

fn view_2d_slice(tex: &Texture, format: WMTPixelFormat, level: u16, slice: u16) -> Result<Texture> {
    let mut id = 0u64;
    // SAFETY: `tex` is a live MTLTexture; level/slice are validated by caller or Metal.
    let handle = unsafe {
        sys::MTLTexture_newTextureView(
            tex.handle(),
            format,
            WMTTextureType::WMTTextureType2D,
            level,
            1,
            slice,
            1,
            identity_swizzle(),
            &mut id,
        )
    };
    Object::adopt(handle)
        .map(Texture::from_object)
        .ok_or_else(|| Error(format!("newTextureView 2D failed (level={level}, slice={slice}, fmt={})", format.0)))
}

fn view_ms_slice(tex: &Texture, format: WMTPixelFormat, slice: u16) -> Result<Texture> {
    let mut id = 0u64;
    // SAFETY: `tex` is a live MSAA MTLTexture.
    let handle = unsafe {
        sys::MTLTexture_newTextureView(
            tex.handle(),
            format,
            WMTTextureType::WMTTextureType2DMultisample,
            0,
            1,
            slice,
            1,
            identity_swizzle(),
            &mut id,
        )
    };
    Object::adopt(handle)
        .map(Texture::from_object)
        .ok_or_else(|| Error(format!("newTextureView 2DMS failed (slice={slice}, fmt={})", format.0)))
}

impl ClearResolve {
    /// Compiles the shared clear/resolve MSL library and prepares caches for `device`.
    pub fn new(device: &Device) -> Result<Self> {
        let library = device.library_msl(CLEAR_RESOLVE_MSL)?;
        let resolve_depth_dss = device.depth_stencil_state(&DepthStencilDesc {
            depth_compare: sys::WMTCompareFunction::WMTCompareFunctionAlways.0,
            depth_write: true,
            stencil: false,
            front: Default::default(),
            back: Default::default(),
        })?;
        Ok(Self {
            device: device.clone(),
            library,
            clear_psos: Mutex::new(HashMap::new()),
            clear_dss: Mutex::new(HashMap::new()),
            resolve_compute_psos: Mutex::new(HashMap::new()),
            resolve_depth_psos: Mutex::new(HashMap::new()),
            resolve_depth_dss,
        })
    }

    fn clear_color_pso(&self, format: WMTPixelFormat, samples: u8) -> Result<RenderPipeline> {
        let key = ClearPipelineKey::Color { format: format.0, samples: samples.max(1) };
        if let Some(pso) = self.clear_psos.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
            return Ok(pso.clone());
        }
        let kind = classify_format(format).color.unwrap_or(ColorKind::Float);
        let frag_fn = match kind {
            ColorKind::Float => "clear_fs_float",
            ColorKind::Uint => "clear_fs_uint",
            ColorKind::Sint => "clear_fs_sint",
        };
        let mut desc = RenderPipelineDesc {
            topology_class: 3, // Triangle
            sample_count: samples.max(1),
            ..Default::default()
        };
        desc.colors[0] = ColorDesc {
            pixel_format: format.0,
            write_mask: 0xf,
            ..Default::default()
        };
        let pso = self.device.render_pipeline_desc(
            (&self.library, "clear_vs"),
            Some((&self.library, frag_fn)),
            &desc,
        )?;
        self.clear_psos.lock().unwrap_or_else(|e| e.into_inner()).insert(key, pso.clone());
        Ok(pso)
    }

    fn clear_ds_pso(&self, depth_format: u32, stencil_format: u32, samples: u8) -> Result<RenderPipeline> {
        let key = ClearPipelineKey::DepthStencil {
            depth_format,
            stencil_format,
            samples: samples.max(1),
        };
        if let Some(pso) = self.clear_psos.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
            return Ok(pso.clone());
        }
        let desc = RenderPipelineDesc {
            topology_class: 3,
            depth_format,
            stencil_format,
            sample_count: samples.max(1),
            ..Default::default()
        };
        let pso = self.device.render_pipeline_desc((&self.library, "clear_vs"), None, &desc)?;
        self.clear_psos.lock().unwrap_or_else(|e| e.into_inner()).insert(key, pso.clone());
        Ok(pso)
    }

    fn clear_ds_state(&self, depth_write: bool, stencil_write: bool) -> Result<DepthStencilState> {
        let key = (depth_write, stencil_write);
        if let Some(dss) = self.clear_dss.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
            return Ok(dss.clone());
        }
        let always = sys::WMTCompareFunction::WMTCompareFunctionAlways.0;
        let replace = sys::WMTStencilOperation::WMTStencilOperationReplace.0;
        let keep = sys::WMTStencilOperation::WMTStencilOperationKeep.0;
        let face = if stencil_write {
            StencilFace {
                compare: always,
                stencil_fail: keep,
                depth_fail: replace,
                pass: replace,
                read_mask: 0xff,
                write_mask: 0xff,
            }
        } else {
            StencilFace {
                compare: always,
                stencil_fail: keep,
                depth_fail: keep,
                pass: keep,
                read_mask: 0xff,
                write_mask: 0,
            }
        };
        let dss = self.device.depth_stencil_state(&DepthStencilDesc {
            depth_compare: always,
            depth_write,
            stencil: stencil_write,
            front: face,
            back: face,
        })?;
        self.clear_dss.lock().unwrap_or_else(|e| e.into_inner()).insert(key, dss.clone());
        Ok(dss)
    }

    fn resolve_compute_pso(&self, kind: ResolveComputeKind) -> Result<ComputePipeline> {
        if let Some(pso) = self.resolve_compute_psos.lock().unwrap_or_else(|e| e.into_inner()).get(&kind) {
            return Ok(pso.clone());
        }
        let name = match kind {
            ResolveComputeKind::AvgFloat => "resolve_avg_float",
            ResolveComputeKind::Sample0Float => "resolve_sample0_float",
            ResolveComputeKind::Sample0Uint => "resolve_sample0_uint",
            ResolveComputeKind::Sample0Sint => "resolve_sample0_sint",
        };
        let pso = self.device.compute_pipeline_from_library(&self.library, name)?;
        self.resolve_compute_psos
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(kind, pso.clone());
        Ok(pso)
    }

    fn resolve_depth_pso(&self, depth_format: u32, stencil_format: u32) -> Result<RenderPipeline> {
        let key = DepthResolveKey { depth_format, stencil_format };
        if let Some(pso) = self.resolve_depth_psos.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
            return Ok(pso.clone());
        }
        let desc = RenderPipelineDesc {
            topology_class: 3,
            depth_format,
            stencil_format,
            sample_count: 1,
            ..Default::default()
        };
        let pso = self.device.render_pipeline_desc(
            (&self.library, "clear_vs"),
            Some((&self.library, "resolve_depth_fs")),
            &desc,
        )?;
        self.resolve_depth_psos
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, pso.clone());
        Ok(pso)
    }

    /// Clears `rects` in `target` inside a `LoadAction::Load` render pass.
    pub fn clear_rects(
        &self,
        cb: &CommandBuffer,
        target: &ClearTarget<'_>,
        rects: &[ClearRect],
        value: ClearValue,
    ) -> Result<()> {
        let (width, height) = target.mip_size();
        let scissors = clamp_clear_rects(rects, width, height);
        if scissors.is_empty() {
            return Ok(());
        }
        let samples = target.samples.max(1);
        let load = sys::WMTLoadAction::WMTLoadActionLoad;
        let store = sys::WMTStoreAction::WMTStoreActionStore;
        let viewport = WMTViewport {
            originX: 0.0,
            originY: 0.0,
            width: width.into(),
            height: height.into(),
            znear: 0.0,
            zfar: 1.0,
        };
        let planes = classify_format(target.pixel_format);
        let mut pass = RenderPass { width, height, samples, ..Default::default() };

        let (pso, dss, stencil_ref, depth_z, frag_bytes_buf, depth_clip): (
            RenderPipeline,
            DepthStencilState,
            u8,
            f32,
            Option<[u8; 16]>,
            u8,
        ) = match value {
            ClearValue::Color(rgba) => {
                if planes.has_depth || planes.has_stencil {
                    return Err(Error("color clear requested on depth/stencil format".into()));
                }
                let pso = self.clear_color_pso(target.pixel_format, samples)?;
                let dss = self.clear_ds_state(false, false)?;
                pass.colors[0] = Some(PassColor {
                    texture: target.texture,
                    level: target.level,
                    slice: target.slice,
                    load,
                    store,
                    clear: [0.0; 4],
                });
                let frag = match planes.color.unwrap_or(ColorKind::Float) {
                    ColorKind::Float => rgba.map(f32::to_le_bytes),
                    ColorKind::Uint => rgba.map(|v| (v as u32).to_le_bytes()),
                    ColorKind::Sint => rgba.map(|v| (v as i32).to_le_bytes()),
                };
                let mut flat = [0u8; 16];
                for (i, word) in frag.iter().enumerate() {
                    flat[i * 4..i * 4 + 4].copy_from_slice(word);
                }
                (pso, dss, 0, 0.0, Some(flat), 0)
            }
            ClearValue::DepthStencil { depth, stencil } => {
                let clear_depth = depth.filter(|_| planes.has_depth);
                let clear_stencil = stencil.filter(|_| planes.has_stencil);
                if clear_depth.is_none() && clear_stencil.is_none() {
                    return Ok(());
                }
                let depth_fmt = if planes.has_depth { target.pixel_format.0 } else { 0 };
                let stencil_fmt = if planes.has_stencil { target.pixel_format.0 } else { 0 };
                let pso = self.clear_ds_pso(depth_fmt, stencil_fmt, samples)?;
                let dss = self.clear_ds_state(clear_depth.is_some(), clear_stencil.is_some())?;
                let pd = PassDepth {
                    texture: target.texture,
                    level: target.level,
                    slice: target.slice,
                    load,
                    store,
                    clear_depth: 1.0,
                    clear_stencil: 0,
                };
                if planes.has_depth {
                    pass.depth = Some(pd);
                }
                if planes.has_stencil {
                    pass.stencil = Some(pd);
                }
                (pso, dss, clear_stencil.unwrap_or(0), clear_depth.unwrap_or(0.0), None, 1)
            }
        };

        let depth_bytes = depth_z.to_le_bytes();
        let mut steps = Vec::with_capacity(6 + scissors.len() * 2);
        steps.push(Render::Pipeline { pipeline: &pso });
        steps.push(Render::Viewports { viewports: std::slice::from_ref(&viewport) });
        steps.push(Render::Rasterizer {
            fill: 0,
            cull: 0,
            depth_clip,
            winding: 1,
            depth_bias: 0.0,
            slope_scale: 0.0,
            clamp: 0.0,
        });
        steps.push(Render::DepthStencil { state: &dss, stencil_ref });
        steps.push(Render::VertexBytes { bytes: &depth_bytes, index: 0 });
        if let Some(frag) = &frag_bytes_buf {
            steps.push(Render::FragmentBytes { bytes: frag, index: 0 });
        }
        for s in &scissors {
            steps.push(Render::Scissors { rects: std::slice::from_ref(s) });
            steps.push(Render::Draw {
                primitive_type: sys::WMTPrimitiveType::WMTPrimitiveTypeTriangle,
                vertex_start: 0,
                vertex_count: 3,
                instance_count: 1,
                base_instance: 0,
            });
        }
        cb.render_pass_with(&pass, &steps)
    }

    /// Resolves an MSAA subresource or a rectangle within it. `None`
    /// requires equal source/destination mip dimensions. The source must be a
    /// shader-readable/renderable 2D MSAA texture and the destination a
    /// single-sample 2D texture or array. A format override requires compatible
    /// backing colour formats created with pixel-format-view usage. Depth
    /// resolves require matching depth formats and a renderable destination;
    /// they preserve destination stencil.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve_region(
        &self,
        cb: &CommandBuffer,
        src: &Texture,
        src_slice: u16,
        dst: &Texture,
        dst_level: u16,
        dst_slice: u16,
        format: WMTPixelFormat,
        region: Option<ResolveRegion>,
    ) -> Result<()> {
        let _pool = Pool::new();
        if u64::from(src_slice) >= src.array_length()
            || u64::from(dst_slice) >= dst.array_length()
            || u64::from(dst_level) >= dst.mipmap_level_count()
        {
            return Err(Error("resolve subresource out of bounds".into()));
        }
        let src_size = [src.width() as u32, src.height() as u32];
        let dst_size = [(dst.width() as u32 >> dst_level).max(1), (dst.height() as u32 >> dst_level).max(1)];
        let region = match region {
            Some(region) => region,
            None if src_size == dst_size => ResolveRegion { src_origin: [0; 2], dst_origin: [0; 2], size: src_size },
            None => return Err(Error("whole resolve requires equal mip sizes".into())),
        };
        for i in 0..2 {
            if region.src_origin[i].checked_add(region.size[i]).is_none_or(|v| v > src_size[i])
                || region.dst_origin[i].checked_add(region.size[i]).is_none_or(|v| v > dst_size[i])
            {
                return Err(Error("resolve rectangle out of bounds".into()));
            }
        }
        if region.size.contains(&0) {
            return Ok(());
        }

        if classify_format(format).has_depth && src.pixel_format() == format && dst.pixel_format() == format {
            return self.resolve_depth(cb, src, src_slice, ResolveSlice { texture: dst, level: dst_level, slice: dst_slice }, region, true);
        }
        let bytes = resolve_color_bytes(format);
        if bytes.is_none() || resolve_color_bytes(src.pixel_format()) != bytes || resolve_color_bytes(dst.pixel_format()) != bytes {
            return Err(Error("incompatible or unsupported MSAA resolve format".into()));
        }

        let src_view = (src.pixel_format() != format || src.array_length() > 1)
            .then(|| view_ms_slice(src, format, src_slice)).transpose()?;
        let src = src_view.as_ref().unwrap_or(src);
        let dst_view = (dst.pixel_format() != format)
            .then(|| view_2d_slice(dst, format, dst_level, dst_slice)).transpose()?;
        let dst = if let Some(view) = &dst_view {
            ResolveSlice { texture: view, level: 0, slice: 0 }
        } else {
            ResolveSlice { texture: dst, level: dst_level, slice: dst_slice }
        };
        let hardware = supports_hw_color_resolve(format);
        if hardware && src_size == dst_size && region.src_origin == [0; 2]
            && region.dst_origin == [0; 2] && region.size == src_size
        {
            return resolve_render_pass(cb, src, 0, dst, src_size[0], src_size[1]);
        }

        // A resolve store action has no rectangle. For narrower regions crop
        // its result; compute can resolve just the requested source rectangle.
        // Scratch also allows ordinary resolve destinations without ShaderWrite.
        let [width, height] = if hardware { src_size } else { region.size };
        let scratch = self.device.new_texture(&TextureDescriptor {
            pixel_format: format,
            width,
            height,
            usage: sys::WMTTextureUsage(if hardware { 4 } else { 2 }),
            options: sys::WMTResourceOptions::WMTResourceStorageModePrivate,
            ..Default::default()
        })?;
        let src_origin = if hardware {
            resolve_render_pass(cb, src, 0, ResolveSlice { texture: &scratch, level: 0, slice: 0 }, width, height)?;
            region.src_origin
        } else {
            self.resolve_compute(cb, src, &scratch, color_resolve_kind(format), region.src_origin)?;
            [0; 2]
        };
        let [sx, sy] = src_origin.map(u64::from);
        let [dx, dy] = region.dst_origin.map(u64::from);
        let [w, h] = region.size.map(u64::from);
        let mut blit = cb.blit_encoder()?;
        blit.copy_from_texture_to_texture(&scratch, 0, 0, [sx, sy, 0], [w, h, 1],
            dst.texture, dst.slice.into(), dst.level.into(), [dx, dy, 0]);
        blit.finish();
        Ok(())
    }

    /// Resolves `src_msaa` (mip 0, `src_slice`) into `dst` (`dst_level`, `dst_slice`)
    /// using `format` (or `dst.pixel_format()` if `format == WMTPixelFormatInvalid`).
    ///
    /// - Color float/unorm/snorm/sRGB formats supported by Apple7 hardware MSAA
    ///   resolve use a render pass with `resolveTexture` and
    ///   `WMTStoreActionStoreAndMultisampleResolve` (preserving `src_msaa`).
    /// - Other float color formats (e.g. `R32Float`, `RG32Float`, `RGBA32Float`)
    ///   use the `resolve_avg_float` compute shader.
    /// - Integer formats (`Uint`/`Sint`) use sample-0 compute shaders (D3D12
    ///   semantics; requires `dst` to have `ShaderWrite` usage).
    /// - Depth formats (`Depth16Unorm`, `Depth32Float`, `Depth32Float_Stencil8`)
    ///   use a fragment-depth render pass reading sample 0 (Metal depth textures
    ///   cannot have `ShaderWrite` usage).
    #[allow(clippy::too_many_arguments)]
    pub fn resolve(
        &self,
        cb: &CommandBuffer,
        src_msaa: &Texture,
        src_slice: u16,
        dst: &Texture,
        dst_level: u16,
        dst_slice: u16,
        format: WMTPixelFormat,
    ) -> Result<()> {
        let fmt = if format == WMTPixelFormat::WMTPixelFormatInvalid {
            dst.pixel_format()
        } else {
            format
        };
        let planes = classify_format(fmt);
        let dst_w = (dst.width() as u32 >> dst_level).max(1);
        let dst_h = (dst.height() as u32 >> dst_level).max(1);
        let src_w = src_msaa.width() as u32;
        let src_h = src_msaa.height() as u32;
        if dst_w != src_w || dst_h != src_h {
            return Err(Error(format!(
                "resolve size mismatch: src {src_w}x{src_h}, dst mip {dst_level} is {dst_w}x{dst_h}"
            )));
        }

        let dst_sub = ResolveSlice { texture: dst, level: dst_level, slice: dst_slice };
        if planes.has_depth {
            if src_msaa.pixel_format() != fmt || dst.pixel_format() != fmt {
                return Err(Error("depth resolve requires matching depth formats".into()));
            }
            let region = ResolveRegion { src_origin: [0; 2], dst_origin: [0; 2], size: [dst_w, dst_h] };
            return self.resolve_depth(cb, src_msaa, src_slice, dst_sub, region, false);
        }
        if planes.has_stencil {
            return Err(Error("stencil-only MSAA resolve is not supported".into()));
        }

        if supports_hw_color_resolve(fmt) {
            // Ensure src and dst views share `fmt` if either differs (e.g. typeless/sRGB).
            let src_view = if src_msaa.pixel_format() != fmt {
                Some(view_ms_slice(src_msaa, fmt, src_slice)?)
            } else {
                None
            };
            let dst_view = if dst.pixel_format() != fmt {
                Some(view_2d_slice(dst, fmt, dst_level, dst_slice)?)
            } else {
                None
            };
            let (src_tex, s_slice) = match &src_view {
                Some(v) => (v, 0),
                None => (src_msaa, src_slice),
            };
            let resolved_dst = match &dst_view {
                Some(v) => ResolveSlice { texture: v, level: 0, slice: 0 },
                None => dst_sub,
            };
            return resolve_render_pass(cb, src_tex, s_slice, resolved_dst, dst_w, dst_h);
        }

        // Compute resolve fallback: 32-bit float average, or Uint/Sint sample-0.
        let src_planes = classify_format(src_msaa.pixel_format());
        let kind = if src_planes.has_depth && planes.color == Some(ColorKind::Float) {
            ResolveComputeKind::Sample0Float
        } else {
            color_resolve_kind(fmt)
        };
        // Bind a 2DMS view of `src_msaa` if it's an array or has a different format
        // (unless reading a depth texture as depth2d_ms/texture2d_ms directly).
        let src_view = if !src_planes.has_depth && (src_msaa.array_length() > 1 || src_msaa.pixel_format() != fmt) {
            Some(view_ms_slice(src_msaa, fmt, src_slice)?)
        } else {
            None
        };
        let dst_view = if dst_level != 0 || dst_slice != 0 || dst.array_length() > 1 || dst.pixel_format() != fmt {
            Some(view_2d_slice(dst, fmt, dst_level, dst_slice)?)
        } else {
            None
        };
        let src_tex = src_view.as_ref().unwrap_or(src_msaa);
        let dst_tex = dst_view.as_ref().unwrap_or(dst);

        self.resolve_compute(cb, src_tex, dst_tex, kind, [0; 2])
    }

    fn resolve_compute(&self, cb: &CommandBuffer, src: &Texture, dst: &Texture, kind: ResolveComputeKind, origin: [u32; 2]) -> Result<()> {
        let pso = self.resolve_compute_pso(kind)?;
        let bytes = origin.map(u32::to_le_bytes);
        let groups = [dst.width().div_ceil(8), dst.height().div_ceil(8), 1];
        cb.compute(&[
            Compute::Pipeline { pipeline: &pso, threads_per_group: [8, 8, 1] },
            Compute::Texture { texture: src, index: 0 },
            Compute::Texture { texture: dst, index: 1 },
            Compute::Bytes { bytes: bytes.as_flattened(), index: 0 },
            Compute::Dispatch { groups },
        ])
    }

    fn resolve_depth(
        &self,
        cb: &CommandBuffer,
        src_msaa: &Texture,
        src_slice: u16,
        dst: ResolveSlice<'_>,
        region: ResolveRegion,
        average: bool,
    ) -> Result<()> {
        let w = (dst.texture.width() as u32 >> dst.level).max(1);
        let h = (dst.texture.height() as u32 >> dst.level).max(1);
        let fmt = dst.texture.pixel_format();
        let has_stencil = classify_format(fmt).has_stencil;
        let depth_fmt = fmt.0;
        let stencil_fmt = if has_stencil { fmt.0 } else { 0 };
        let pso = self.resolve_depth_pso(depth_fmt, stencil_fmt)?;

        let src_view = if src_msaa.array_length() > 1 {
            Some(view_ms_slice(src_msaa, src_msaa.pixel_format(), src_slice)?)
        } else {
            None
        };
        let src_tex = src_view.as_ref().unwrap_or(src_msaa);

        let load_preserve = sys::WMTLoadAction::WMTLoadActionLoad;
        let load_dont_care = sys::WMTLoadAction::WMTLoadActionDontCare;
        let store = sys::WMTStoreAction::WMTStoreActionStore;
        let mut pass = RenderPass { width: w, height: h, samples: 1, ..Default::default() };
        pass.depth = Some(PassDepth {
            texture: dst.texture,
            level: dst.level,
            slice: dst.slice,
            load: if region.dst_origin == [0; 2] && region.size == [w, h] { load_dont_care } else { load_preserve },
            store,
            clear_depth: 1.0,
            clear_stencil: 0,
        });
        if has_stencil {
            pass.stencil = Some(PassDepth {
                texture: dst.texture,
                level: dst.level,
                slice: dst.slice,
                load: load_preserve,
                store,
                clear_depth: 1.0,
                clear_stencil: 0,
            });
        }
        let viewport = WMTViewport {
            originX: 0.0,
            originY: 0.0,
            width: w.into(),
            height: h.into(),
            znear: 0.0,
            zfar: 1.0,
        };
        let scissor = WMTScissorRect {
            x: region.dst_origin[0].into(), y: region.dst_origin[1].into(),
            width: region.size[0].into(), height: region.size[1].into(),
        };
        let depth_z = 0.0f32.to_le_bytes();
        let [sx, sy] = region.src_origin;
        let [dx, dy] = region.dst_origin;
        let params = [sx, sy, dx, dy, u32::from(average)].map(u32::to_le_bytes);
        cb.render_pass_with(
            &pass,
            &[
                Render::Pipeline { pipeline: &pso },
                Render::Viewports { viewports: std::slice::from_ref(&viewport) },
                Render::Scissors { rects: std::slice::from_ref(&scissor) },
                Render::Rasterizer {
                    fill: 0,
                    cull: 0,
                    depth_clip: 0,
                    winding: 1,
                    depth_bias: 0.0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
                Render::DepthStencil { state: &self.resolve_depth_dss, stencil_ref: 0 },
                Render::VertexBytes { bytes: &depth_z, index: 0 },
                Render::FragmentTexture { texture: src_tex, index: 0 },
                Render::FragmentBytes { bytes: params.as_flattened(), index: 0 },
                Render::Draw {
                    primitive_type: sys::WMTPrimitiveType::WMTPrimitiveTypeTriangle,
                    vertex_start: 0,
                    vertex_count: 3,
                    instance_count: 1,
                    base_instance: 0,
                },
            ],
        )
    }
}

fn color_resolve_kind(format: WMTPixelFormat) -> ResolveComputeKind {
    match classify_format(format).color {
        Some(ColorKind::Uint) => ResolveComputeKind::Sample0Uint,
        Some(ColorKind::Sint) => ResolveComputeKind::Sample0Sint,
        _ => ResolveComputeKind::AvgFloat,
    }
}

#[derive(Clone, Copy)]
struct ResolveSlice<'a> {
    texture: &'a Texture,
    level: u16,
    slice: u16,
}

/// Hardware MSAA color resolve via `MTLRenderPassColorAttachmentDescriptor`
/// `resolveTexture` + `WMTStoreActionStoreAndMultisampleResolve`.
fn resolve_render_pass(
    cb: &CommandBuffer,
    src_msaa: &Texture,
    src_slice: u16,
    dst: ResolveSlice<'_>,
    width: u32,
    height: u32,
) -> Result<()> {
    let _pool = Pool::new();
    // SAFETY: all-zero descriptor; fields populated below.
    let mut pass: sys::WMTRenderPassInfo = unsafe { std::mem::zeroed() };
    pass.render_target_width = width;
    pass.render_target_height = height;
    pass.colors[0].texture = src_msaa.handle();
    pass.colors[0].level = 0;
    pass.colors[0].slice = src_slice;
    pass.colors[0].load_action = sys::WMTLoadAction::WMTLoadActionLoad;
    pass.colors[0].store_action = sys::WMTStoreAction::WMTStoreActionStoreAndMultisampleResolve;
    pass.colors[0].resolve_texture = dst.texture.handle();
    pass.colors[0].resolve_level = dst.level;
    pass.colors[0].resolve_slice = dst.slice;
    {
        let mut work = cb.1.lock().unwrap_or_else(|e| e.into_inner());
        work.resolve_pixels = work.resolve_pixels.saturating_add(u64::from(width) * u64::from(height));
    }

    // SAFETY: `cb` and both textures are live for the encoding call.
    let encoder = unsafe { sys::MTLCommandBuffer_renderCommandEncoder(cb.handle(), &mut pass) };
    if encoder == 0 {
        return Err(Error("renderCommandEncoder (resolve)".into()));
    }
    // SAFETY: `encoder` is a live render command encoder.
    unsafe { sys::MTLCommandEncoder_endEncoding(encoder) };
    Ok(())
}

#[cfg(all(test, windows))]
#[path = "clear_resolve/f80_tests.rs"]
mod f80_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_rects_clips_and_drops_empty() {
        let rects = [
            ClearRect { left: -4, top: 2, right: 10, bottom: 8 },
            ClearRect { left: 20, top: 20, right: 40, bottom: 50 },
            ClearRect { left: 12, top: 5, right: 12, bottom: 10 }, // empty width
            ClearRect { left: 10, top: 10, right: 5, bottom: 20 }, // inverted
            ClearRect { left: 100, top: 0, right: 120, bottom: 10 }, // outside
        ];
        let out = clamp_clear_rects(&rects, 32, 24);
        assert_eq!(out.len(), 2);
        assert_eq!((out[0].x, out[0].y, out[0].width, out[0].height), (0, 2, 10, 6));
        assert_eq!((out[1].x, out[1].y, out[1].width, out[1].height), (20, 20, 12, 4));
    }

    #[test]
    fn format_classification_matches_metal_rules() {
        assert_eq!(
            classify_format(WMTPixelFormat::WMTPixelFormatBGRA8Unorm),
            FormatPlanes { has_depth: false, has_stencil: false, color: Some(ColorKind::Float) }
        );
        assert_eq!(
            classify_format(WMTPixelFormat::WMTPixelFormatR32Uint),
            FormatPlanes { has_depth: false, has_stencil: false, color: Some(ColorKind::Uint) }
        );
        assert_eq!(
            classify_format(WMTPixelFormat::WMTPixelFormatRGBA16Sint),
            FormatPlanes { has_depth: false, has_stencil: false, color: Some(ColorKind::Sint) }
        );
        assert_eq!(
            classify_format(WMTPixelFormat::WMTPixelFormatDepth32Float),
            FormatPlanes { has_depth: true, has_stencil: false, color: None }
        );
        assert_eq!(
            classify_format(WMTPixelFormat::WMTPixelFormatDepth32Float_Stencil8),
            FormatPlanes { has_depth: true, has_stencil: true, color: None }
        );
        assert!(supports_hw_color_resolve(WMTPixelFormat::WMTPixelFormatBGRA8Unorm));
        assert!(supports_hw_color_resolve(WMTPixelFormat::WMTPixelFormatBGRA8Unorm_sRGB));
        assert!(supports_hw_color_resolve(WMTPixelFormat::WMTPixelFormatRGBA16Float));
        assert!(!supports_hw_color_resolve(WMTPixelFormat::WMTPixelFormatR32Float));
        assert!(!supports_hw_color_resolve(WMTPixelFormat::WMTPixelFormatRGBA32Float));
        assert!(!supports_hw_color_resolve(WMTPixelFormat::WMTPixelFormatR32Uint));
        assert!(!supports_hw_color_resolve(WMTPixelFormat::WMTPixelFormatDepth32Float));
    }
}
