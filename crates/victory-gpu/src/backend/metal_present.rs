//! P2-5: presenting from Metal (`VICTORY_METAL_PRESENT=1`, with
//! `VICTORY_METAL_DRAW=1`): the title's presented texture as Metal drew it,
//! drawn into the window's `CAMetalLayer` (F18), scaled to fit with its
//! aspect kept. The D3D12 present is then skipped (no swap chain is made
//! over the window).

use std::sync::{Mutex, OnceLock};

use victory_metal::Object;
use victory_metal::present::MetalLayer;
use victory_metal::render::{PassColor, Render, RenderPass, RenderPipeline, RenderPipelineDesc};
use victory_metal::sys;
use victory_metal::texture::Texture;

use super::metal::MetalBackend;
use crate::object::This;

/// Whether presents go through Metal.
pub(crate) fn present_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_METAL_PRESENT").is_ok_and(|v| v == "1")) && super::metal_draw::draws_on()
}

const BLIT_MSL: &str = r"
#include <metal_stdlib>
using namespace metal;
struct V { float4 p [[position]]; float2 uv; };
vertex V present_vs(uint id [[vertex_id]]) {
    float2 uv = float2((id << 1) & 2, id & 2);
    V o;
    o.p = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    o.uv = uv;
    return o;
}
fragment float4 present_fs(V in [[stage_in]], texture2d<float> t [[texture(0)]]) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    return float4(t.sample(s, in.uv).rgb, 1.0);
}
";

struct Presenter {
    layer: MetalLayer,
    /// F55's colour conversion (sRGB, scRGB tone map); the plain blit below
    /// is the fallback when it refuses a source.
    converter: Option<victory_metal::present::convert::PresentConverter>,
    library: Object,
    /// The blit pipeline for the drawable's pixel format.
    pipeline: Option<(u32, RenderPipeline)>,
    size: (u32, u32),
}

// Only used under the presenter lock.
unsafe impl Send for Presenter {}

impl MetalBackend {
    /// Shows title texture `resource` (as Metal drew it) in the window.
    /// False if it could not (the caller may present the D3D12 way).
    pub(crate) fn present_metal(&self, resource: This) -> bool {
        static PRESENTER: Mutex<Option<Presenter>> = Mutex::new(None);
        let Some(source) = self.texture_for(resource) else { return fell_back("no Metal texture for the presented resource") };
        let _pool = victory_metal::Pool::new();
        let mut guard = PRESENTER.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            let Some(hwnd) = crate::host::title_window() else { return fell_back("no title window") };
            let layer = match MetalLayer::for_window(&self.device, hwnd.0 as isize) {
                Ok(l) => l,
                Err(e) => {
                    warn_once("no layer over the window", &e);
                    return false;
                }
            };
            let library = match self.device.library_msl(BLIT_MSL) {
                Ok(l) => l,
                Err(e) => {
                    warn_once("the blit shader did not compile", &e);
                    return false;
                }
            };
            let converter = victory_metal::present::convert::PresentConverter::new(&self.device).ok();
            if converter.is_some() {
                let _ = victory_metal::present::convert::PresentConverter::configure_sdr_layer(&layer);
            }
            tracing::info!(converter = converter.is_some(), "Metal present: drawing into the window's layer");
            *guard = Some(Presenter { layer, converter, library, pipeline: None, size: (0, 0) });
        }
        let p = guard.as_mut().expect("made above");
        // The drawable follows the window's client area.
        let (cw, ch) = client_size(crate::host::title_window());
        if (cw, ch) != p.size && cw > 0 && ch > 0 {
            let _ = p.layer.set_drawable_size(f64::from(cw), f64::from(ch));
            p.size = (cw, ch);
        }
        let Ok(drawable) = p.layer.next_drawable() else { return fell_back("no drawable from the layer") };
        let Ok(target) = drawable.texture() else { return fell_back("the drawable has no texture") };
        let target = Texture::from_object(target);
        // F55 first: the source's typed format decides the transfer.
        if let Some(conv) = p.converter.as_mut() {
            let dxgi = crate::resource::pc_format(source.desc.format);
            if let Ok(cb) = self.queue.command_buffer() {
                match conv.draw(&cb, &source.texture, dxgi, &target, Default::default()) {
                    Ok(()) => {
                        cb.present(&drawable);
                        cb.commit();
                        return true;
                    }
                    Err(e) => warn_once("colour conversion refused the source, plain blit instead", &e),
                }
            }
        }
        // SAFETY: a live texture.
        let format = unsafe { sys::MTLTexture_pixelFormat(target.handle()) }.0;
        if p.pipeline.as_ref().is_none_or(|(f, _)| *f != format) {
            let mut desc = RenderPipelineDesc::default();
            desc.colors[0].pixel_format = format;
            desc.colors[0].write_mask = 0xf;
            desc.colors[0].src_rgb = 1;
            desc.colors[0].src_alpha = 1;
            desc.sample_count = 1;
            desc.topology_class = 3;
            match self.device.render_pipeline_desc((&p.library, "present_vs"), Some((&p.library, "present_fs")), &desc) {
                Ok(pso) => p.pipeline = Some((format, pso)),
                Err(e) => {
                    warn_once("no blit pipeline", &e);
                    return false;
                }
            }
        }
        let (_, pso) = p.pipeline.as_ref().expect("made above");
        let (tw, th) = (target.width() as f64, target.height() as f64);
        let (sw, sh) = (f64::from(source.desc.width as u32), f64::from(source.desc.height.max(1)));
        // Fit with the aspect kept (black bars).
        let scale = (tw / sw).min(th / sh);
        let (vw, vh) = (sw * scale, sh * scale);
        let viewport = sys::WMTViewport { originX: (tw - vw) / 2.0, originY: (th - vh) / 2.0, width: vw, height: vh, znear: 0.0, zfar: 1.0 };
        let mut pass = RenderPass { width: tw as u32, height: th as u32, samples: 1, ..Default::default() };
        pass.colors[0] = Some(PassColor {
            texture: &target,
            level: 0,
            slice: 0,
            load: sys::WMTLoadAction::WMTLoadActionClear,
            store: sys::WMTStoreAction::WMTStoreActionStore,
            clear: [0.0, 0.0, 0.0, 1.0],
        });
        let steps = [
            Render::Pipeline { pipeline: pso },
            Render::Viewports { viewports: std::slice::from_ref(&viewport) },
            Render::FragmentTexture { texture: &source.texture, index: 0 },
            Render::Draw {
                primitive_type: sys::WMTPrimitiveType::WMTPrimitiveTypeTriangle,
                vertex_start: 0,
                vertex_count: 3,
                instance_count: 1,
                base_instance: 0,
            },
        ];
        let Ok(cb) = self.queue.command_buffer() else { return fell_back("no command buffer") };
        if cb.render_pass_with(&pass, &steps).is_err() {
            return fell_back("the blit pass did not encode");
        }
        cb.present(&drawable);
        cb.commit();
        true
    }
}

/// The window's client area in pixels (0×0 if unknown).
fn client_size(hwnd: Option<windows::Win32::Foundation::HWND>) -> (u32, u32) {
    let Some(hwnd) = hwnd else { return (0, 0) };
    let mut r = windows::Win32::Foundation::RECT::default();
    // SAFETY: a live window handle; `r` is writable.
    if unsafe { windows::Win32::UI::WindowsAndMessaging::GetClientRect(hwnd, &mut r) }.is_err() {
        return (0, 0);
    }
    ((r.right - r.left).max(0) as u32, (r.bottom - r.top).max(0) as u32)
}

/// A present Metal could not make, so the D3D12 picture shows this frame:
/// logged the first time per reason, then every 1024th.
fn fell_back(why: &'static str) -> bool {
    static SEEN: Mutex<Vec<(&'static str, u64)>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let n = match seen.iter_mut().find(|(w, _)| *w == why) {
        Some((_, n)) => {
            *n += 1;
            *n
        }
        None => {
            seen.push((why, 1));
            1
        }
    };
    if n == 1 || n.is_multiple_of(1024) {
        tracing::warn!(times = n, "Metal present fell back to the D3D12 picture: {why}");
    }
    false
}

/// Logs a present failure the first few times only (presents retry every frame).
fn warn_once(what: &str, e: &dyn std::fmt::Display) {
    static SEEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
        tracing::warn!(error = %e, "Metal present: {what}");
    }
}
