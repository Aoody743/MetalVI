//! A stand-in for the indirect light the title traces rays for, on the
//! Metal path (Lead B of `depth-probe-scan-20261128`).
//!
//! With the ray-tracing structure builder off (`crate::pipeline::builder_runs`)
//! the title's rays find nothing, and its program that resolves the diffuse
//! light of its screen probes (`CS_ScreenProbes_TemporalResolve_*`) leaves
//! nothing indoors and, in blocks of tiles, what the probes' memory held.
//! Right after a dispatch of that program a small kernel of ours writes a
//! constant light into the image it just wrote: root parameter 1, dword 176
//! of its table, a `RWTexture2D<uint>` of R9G9B9E5 (upstream `ambient.rs`,
//! after beloooo's find; the constant mode is theirs).
//!
//! `VICTORY_AMBIENT=constant:<level>` turns it on (default off; any other
//! value is off here: the upstream estimate mode is not on Metal yet). The
//! colour is `(1, 0.98, 0.94) * level`, packed on the GPU as the upstream
//! HLSL packs it.

use std::sync::OnceLock;

use victory_metal::sys::{self, WMTPixelFormat, WMTTextureType};
use victory_metal::texture::Texture;
use victory_metal::{CommandBuffer, Compute, ComputePipeline, Device};

use super::metal_texture::MetalTexture;

/// The program the stand-in follows, by a name in its container.
pub(crate) const PROBE_RESOLVE: &[u8] = b"CS_ScreenProbes_TemporalResolve";

/// Where the resolve writes the diffuse light: root parameter and place in
/// its table, in dwords.
pub(crate) const LIGHT: (u8, u32) = (1, 176);

/// The tint of the constant (upstream's).
pub(crate) const TINT: [f32; 3] = [1.0, 0.98, 0.94];

/// What `VICTORY_AMBIENT` asks for.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Set {
    Off,
    Constant(f32),
}

pub(crate) fn parse(set: &str) -> Set {
    let level = set.trim().strip_prefix("constant:").and_then(|t| t.trim().parse::<f32>().ok());
    match level {
        Some(v) if v.is_finite() && v > 0.0 => Set::Constant(v),
        _ => Set::Off,
    }
}

fn set() -> Set {
    static SET: OnceLock<Set> = OnceLock::new();
    *SET.get_or_init(|| {
        let raw = std::env::var("VICTORY_AMBIENT").unwrap_or_default();
        let s = parse(&raw);
        if s == Set::Off && !raw.trim().is_empty() && !matches!(raw.trim(), "0" | "off") {
            tracing::warn!(value = %raw, "VICTORY_AMBIENT: only constant:<level> is on the Metal path; left off");
        }
        s
    })
}

/// The constant's level if the stand-in is on: asked for, and the title's
/// rays find nothing (the builder does not run).
pub(crate) fn level() -> Option<f32> {
    match set() {
        Set::Constant(level) if !crate::pipeline::builder_runs() => Some(level),
        _ => None,
    }
}

/// Whether a compute container is the probes' resolve.
pub(crate) fn is_probe_resolve(container: &[u8]) -> bool {
    container.windows(PROBE_RESOLVE.len()).any(|w| w == PROBE_RESOLVE)
}

/// R9G9B9E5 as the upstream HLSL `pack` makes it (the reference for the
/// kernel's).
pub(crate) fn pack_rgb9e5(c: [f32; 3]) -> u32 {
    let c = c.map(|v| v.clamp(0.0, 65408.0));
    let top = c[0].max(c[1]).max(c[2]);
    let mut e = ((top.max(1e-9).log2().floor() as i32) + 16).clamp(0, 31);
    let mut scale = (24 - e) as f32;
    scale = scale.exp2();
    if top * scale >= 511.5 {
        e = (e + 1).min(31);
        scale = ((24 - e) as f32).exp2();
    }
    let m = c.map(|v| ((v * scale + 0.5) as u32).min(511));
    m[0] | m[1] << 9 | m[2] << 18 | (e as u32) << 27
}

pub(crate) fn unpack_rgb9e5(v: u32) -> [f32; 3] {
    let scale = ((v >> 27) as f32 - 24.0).exp2();
    [(v & 511) as f32 * scale, ((v >> 9) & 511) as f32 * scale, ((v >> 18) & 511) as f32 * scale]
}

const MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Params {
    float4 colour;
    uint2 size;
};

static uint pack_rgb9e5(float3 c) {
    c = clamp(c, 0.0f, 65408.0f);
    float top = max(c.r, max(c.g, c.b));
    int e = clamp(int(floor(log2(max(top, 1e-9f)))) + 16, 0, 31);
    float scale = exp2(float(24 - e));
    if (top * scale >= 511.5f) { e = min(e + 1, 31); scale = exp2(float(24 - e)); }
    uint3 m = min(uint3(c * scale + 0.5f), uint3(511u));
    return m.r | (m.g << 9) | (m.b << 18) | (uint(e) << 27);
}

kernel void raise_constant(texture2d<uint, access::write> light [[texture(0)]],
                           constant Params& p [[buffer(0)]],
                           uint2 at [[thread_position_in_grid]]) {
    uint2 size = min(p.size, uint2(light.get_width(), light.get_height()));
    if (at.x >= size.x || at.y >= size.y) return;
    light.write(uint4(pack_rgb9e5(p.colour.rgb), 0u, 0u, 0u), at);
}
"#;

/// Bytes of the kernel's `Params`.
fn params(colour: [f32; 3], size: [u32; 2]) -> [u8; 32] {
    let mut b = [0u8; 32];
    for (i, v) in colour.into_iter().chain([0.0]).enumerate() {
        b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    b[16..20].copy_from_slice(&size[0].to_le_bytes());
    b[20..24].copy_from_slice(&size[1].to_le_bytes());
    b
}

/// The kernel, built once per process (`None` if it did not build).
fn kernel(device: &Device) -> Option<&'static ComputePipeline> {
    static KERNEL: OnceLock<Option<ComputePipeline>> = OnceLock::new();
    KERNEL
        .get_or_init(|| {
            let built = device.library_msl(MSL).and_then(|l| device.compute_pipeline_from_library(&l, "raise_constant"));
            built.inspect_err(|e| tracing::warn!(error = %e, "Metal: no ambient stand-in kernel")).ok()
        })
        .as_ref()
}

/// One run of the stand-in, recorded after the resolve's dispatch.
pub(crate) struct MetalAmbient {
    /// An R32Uint 2D view of the image's level and slice the resolve wrote.
    pub(crate) view: Texture,
    /// Its size.
    pub(crate) size: [u32; 2],
    /// The linear colour written.
    pub(crate) colour: [f32; 3],
}

impl MetalAmbient {
    /// The stand-in for UAV `words` (D3D12_UNORDERED_ACCESS_VIEW_DESC; none:
    /// the whole texture) of `t` at `level`. None unless it is a 2D view of
    /// a 32-bit texture Metal lets us view as R32Uint (R9G9B9E5, R32_UINT).
    pub(crate) fn new(t: &MetalTexture, words: Option<&[u32]>, level: f32) -> Option<Self> {
        const UAV_TEXTURE2D: u32 = 4;
        const UAV_TEXTURE2D_ARRAY: u32 = 5;
        if !matches!(t.pixel, WMTPixelFormat::WMTPixelFormatRGB9E5Float | WMTPixelFormat::WMTPixelFormatR32Uint)
            || t.desc.sample_count > 1
            || t.desc.flags & 4 == 0
        {
            return None;
        }
        let (mip, slice) = match words {
            None => (0, 0),
            Some(w) if w.len() >= 4 && w[1] == UAV_TEXTURE2D => (w[2], 0),
            Some(w) if w.len() >= 5 && w[1] == UAV_TEXTURE2D_ARRAY && w[4] == 1 => (w[2], w[3]),
            Some(_) => return None,
        };
        if mip >= t.mips || slice >= t.slices {
            return None;
        }
        let mut id = 0u64;
        let _pool = victory_metal::Pool::new();
        // SAFETY: a live texture; R32Uint is in RGB9E5's view class (group 3
        // of metal_texture's `mtl_view_group`), the texture has PixelFormatView.
        let view = unsafe {
            sys::MTLTexture_newTextureView(
                t.texture.handle(),
                WMTPixelFormat::WMTPixelFormatR32Uint,
                WMTTextureType::WMTTextureType2D,
                mip as u16,
                1,
                slice as u16,
                1,
                super::metal_texture::identity(),
                &mut id,
            )
        };
        let view = Texture::from_object(victory_metal::Object::adopt(view)?);
        let size = [view.width() as u32, view.height() as u32];
        Some(MetalAmbient { view, size, colour: TINT.map(|c| c * level) })
    }

    /// Records the kernel into `cb`, in a compute pass of its own.
    pub(crate) fn encode(&self, device: &Device, cb: &CommandBuffer) -> victory_metal::Result<()> {
        let Some(pso) = kernel(device) else {
            return Err(victory_metal::Error("ambient kernel unavailable".into()));
        };
        let bytes = params(self.colour, self.size);
        cb.compute(&[
            Compute::Pipeline { pipeline: pso, threads_per_group: [8, 8, 1] },
            Compute::Texture { texture: &self.view, index: 0 },
            Compute::Bytes { bytes: &bytes, index: 0 },
            Compute::Dispatch { groups: [u64::from(self.size[0].div_ceil(8)), u64::from(self.size[1].div_ceil(8)), 1] },
        ])
    }
}

// Tests: `metal.rs` `ambient_tests` (they need the backend's private constructor).
