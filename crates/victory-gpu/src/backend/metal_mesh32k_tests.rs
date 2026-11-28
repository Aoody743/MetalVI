//! The title's mesh pipelines over Metal's 32 KiB of outputs, offline: each mesh shader
//! translated the way `pipeline::host` does for its pixel shader (outputs it does not
//! read left out, the rest in the order (and, still over, components it does not read) `victory_shader` `mesh_order` picks), its size
//! as Metal counts it (`metal_mesh_bytes`), and for those that now fit, Metal Shader
//! Converter and a Metal mesh render pipeline.
//!
//! `VICTORY_MESH_PAIRS` = `<mesh container>|<pixel container>` pairs separated by `;`
//! (Wine paths of `*.cdxil`). One line a pair: `MESH32K <mesh> <pixel> bytes <before> ->
//! <after> <fits|over> [pipeline ok|ERR ...]`. Pipelines over the limit are never made
//! (D3DMetal never finishes compiling one).

use super::*;

fn translate(path: &str) -> Result<victory_shader::emit::HostShader, String> {
    let bytes = crate::read_file(std::path::Path::new(path)).map_err(|e| format!("{path}: {e}"))?;
    victory_shader::emit::translate(&bytes).map_err(|e| format!("{path}: {e}"))
}

fn short(path: &str) -> &str {
    path.rsplit(['\\', '/']).next().unwrap_or(path).split('.').next().unwrap_or(path)
}

#[test]
#[ignore = "needs VICTORY_MESH_PAIRS (title containers) and Wine"]
fn the_title_mesh_pipelines_fit_and_build() {
    use victory_metal::msc::{Stage, convert};
    let _pool = victory_metal::Pool::new();
    let backend = MetalBackend::new().unwrap();
    victory_shader::emit::set_host_max_wave(32);
    let list = std::env::var("VICTORY_MESH_PAIRS").unwrap_or_default();
    let (mut fit, mut built, mut total) = (0, 0, 0);
    for pair in list.split(';').filter(|p| !p.is_empty()) {
        let Some((mesh_path, pixel_path)) = pair.split_once('|') else { continue };
        total += 1;
        let run = || -> Result<String, String> {
            let mut ps = translate(pixel_path)?;
            let plain = translate(mesh_path)?;
            let before = crate::pipeline::host::metal_mesh_bytes(&plain.dxil).ok_or("no size")?;
            let keep = victory_shader::emit::input_semantics(&ps.dxil).ok_or("no pixel inputs")?;
            let old = victory_shader::emit::set_mesh_outputs_keep(Some(keep));
            let old_reads = victory_shader::emit::set_mesh_output_reads(victory_shader::emit::input_reads(&ps.dxil));
            let mut ms = translate(mesh_path);
            let alias = victory_shader::emit::last_clip_alias();
            let mut after = ms.as_ref().ok().and_then(|m| crate::pipeline::host::metal_mesh_bytes(&m.dxil)).unwrap_or(0);
            let mut how = "";
            // As pipeline::host: still over, clip distances to the pixel shader.
            if after > 32 * 1024
                && let Some(alias) = alias
            {
                let old_drop = victory_shader::emit::set_mesh_drop_clip(true);
                ms = translate(mesh_path);
                victory_shader::emit::set_mesh_drop_clip(old_drop);
                let old_clip = victory_shader::emit::set_pixel_clip_inputs(Some(vec![alias]));
                ps = translate(pixel_path)?;
                victory_shader::emit::set_pixel_clip_inputs(old_clip);
                after = ms.as_ref().ok().and_then(|m| crate::pipeline::host::metal_mesh_bytes(&m.dxil)).unwrap_or(0);
                how = " (clip distances discarded in the pixel shader)";
            }
            victory_shader::emit::set_mesh_output_reads(old_reads);
            victory_shader::emit::set_mesh_outputs_keep(old);
            let ms = ms?;
            let mut line = format!("bytes {before} -> {after}{how}");
            if after > 32 * 1024 {
                return Ok(line + " over");
            }
            line += " fits";
            let m = convert(&ms.dxil, Stage::Mesh, &root_params(&ms), &static_samplers(&ms)).map_err(|e| format!("{line} MSC mesh: {e}"))?;
            let p = convert(&ps.dxil, Stage::Fragment, &root_params(&ps), &static_samplers(&ps)).map_err(|e| format!("{line} MSC pixel: {e}"))?;
            let mlib = backend.device.library(&m.metallib).map_err(|e| format!("{line} mesh library: {e}"))?;
            let plib = backend.device.library(&p.metallib).map_err(|e| format!("{line} pixel library: {e}"))?;
            let targets = victory_shader::emit::output_semantics(&ps.dxil).map_or(0, |o| o.iter().filter(|e| e.2 == 16).count());
            let mut desc = victory_metal::render::RenderPipelineDesc { sample_count: 1, topology_class: 3, ..Default::default() };
            for c in desc.colors.iter_mut().take(targets.max(1)) {
                c.pixel_format = victory_metal::sys::WMTPixelFormat::WMTPixelFormatRGBA16Float.0;
                c.write_mask = 0xf;
            }
            desc.depth_format = victory_metal::sys::WMTPixelFormat::WMTPixelFormatDepth32Float.0;
            backend
                .device
                .mesh_render_pipeline_desc((&mlib, &m.entry), Some((&plib, &p.entry)), &desc)
                .map_err(|e| format!("{line} pipeline: {e}"))?;
            Ok(line + " pipeline ok")
        };
        let line = run().unwrap_or_else(|e| format!("ERR {e}"));
        fit += usize::from(line.contains(" fits"));
        built += usize::from(line.contains("pipeline ok"));
        println!("MESH32K {} {} {line}", short(mesh_path), short(pixel_path));
    }
    println!("MESH32K {fit} of {total} pairs fit, {built} pipelines made");
    assert_eq!(fit, built, "a pair that fits did not build");
}

/// The premise of `clip_discard`: a triangle in perspective clipped by the hardware
/// (clip distances) covers the pixels where the same distances, interpolated with
/// perspective as a varying, are not below zero. Two 128x128 renders, compared pixel
/// by pixel: at most a few pixels on the clipping edge may differ (the rasterizer's
/// rule for the new edge against a test at the pixel centre).
#[test]
fn clip_distances_and_a_discard_cover_the_same_pixels() {
    use victory_metal::render::{ColorDesc, PassColor, Render, RenderPass, RenderPipelineDesc};
    use victory_metal::sys::{WMTLoadAction, WMTPixelFormat as P, WMTPrimitiveType, WMTStoreAction, WMTViewport};
    const MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
// A triangle leaning away (w from 1 to 4), two clip planes through it.
constant float4 corners[3] = { float4(-0.9, -0.9, 0.1, 1.0), float4(3.6, -1.2, 0.4, 4.0), float4(-0.3, 3.2, 0.3, 4.0) };
static float2 planes(float4 p) { return float2(p.x + 0.3 * p.y - 0.2 * p.w, 0.5 * p.w - p.y + 0.1 * p.x); }
struct Hw { float4 p [[position]]; float clip [[clip_distance]] [2]; };
vertex Hw hw_vs(uint i [[vertex_id]]) {
    Hw o; o.p = corners[i]; float2 d = planes(o.p); o.clip[0] = d.x; o.clip[1] = d.y; return o;
}
struct Sw { float4 p [[position]]; float2 d [[user(clip)]]; };
vertex Sw sw_vs(uint i [[vertex_id]]) { Sw o; o.p = corners[i]; o.d = planes(o.p); return o; }
fragment float4 hit() { return float4(1); }
fragment float4 sw_ps(Sw in [[stage_in]]) { if (any(in.d < 0.0)) discard_fragment(); return float4(1); }
"#;
    let _pool = victory_metal::Pool::new();
    let backend = MetalBackend::new().unwrap();
    let lib = backend.device.library_msl(MSL).unwrap();
    const N: u32 = 128;
    let mut desc = RenderPipelineDesc { sample_count: 1, topology_class: 3, ..Default::default() };
    desc.colors[0] = ColorDesc { pixel_format: P::WMTPixelFormatR32Float.0, write_mask: 0xf, ..Default::default() };
    let render = |vs: &str, ps: &str| -> Vec<f32> {
        let pso = backend.device.render_pipeline_desc((&lib, vs), Some((&lib, ps)), &desc).unwrap();
        let target = backend
            .device
            .new_texture(&victory_metal::texture::TextureDescriptor {
                pixel_format: P::WMTPixelFormatR32Float,
                width: N,
                height: N,
                usage: victory_metal::sys::WMTTextureUsage(1 | 4),
                options: victory_metal::sys::WMTResourceOptions(0x20),
                ..Default::default()
            })
            .unwrap();
        let cb = backend.queue.command_buffer().unwrap();
        let mut pass = RenderPass { width: N, height: N, samples: 1, ..Default::default() };
        pass.colors[0] = Some(PassColor {
            texture: &target,
            level: 0,
            slice: 0,
            load: WMTLoadAction::WMTLoadActionClear,
            store: WMTStoreAction::WMTStoreActionStore,
            clear: [0.0; 4],
        });
        let viewport = WMTViewport { originX: 0.0, originY: 0.0, width: N.into(), height: N.into(), znear: 0.0, zfar: 1.0 };
        cb.render_pass_with(
            &pass,
            &[
                Render::Pipeline { pipeline: &pso },
                Render::Viewports { viewports: std::slice::from_ref(&viewport) },
                Render::Draw { primitive_type: WMTPrimitiveType::WMTPrimitiveTypeTriangle, vertex_start: 0, vertex_count: 3, instance_count: 1, base_instance: 0 },
            ],
        )
        .unwrap();
        let (buffer, ptr) = backend.device.shared_buffer_mapped(&vec![0u8; (N * N * 4) as usize]).unwrap();
        let mut blit = cb.blit_encoder().unwrap();
        blit.copy_from_texture_to_buffer(&target, 0, 0, [0; 3], [N.into(), N.into(), 1], &buffer, 0, N * 4, N * N * 4);
        blit.finish();
        cb.commit();
        cb.wait().unwrap();
        // SAFETY: the copy completed; N*N floats.
        unsafe { std::slice::from_raw_parts(ptr.cast::<f32>(), (N * N) as usize) }.to_vec()
    };
    let hw = render("hw_vs", "hit");
    let sw = render("sw_vs", "sw_ps");
    let covered = hw.iter().filter(|&&v| v > 0.0).count();
    let differ = hw.iter().zip(&sw).filter(|(a, b)| (**a > 0.0) != (**b > 0.0)).count();
    println!("CLIP hardware covers {covered} pixels, discard {} pixels, {differ} differ", sw.iter().filter(|&&v| v > 0.0).count());
    assert!(covered > 1000, "the clipped triangle covers too little to tell");
    assert!(differ <= 8, "{differ} pixels differ");
}
