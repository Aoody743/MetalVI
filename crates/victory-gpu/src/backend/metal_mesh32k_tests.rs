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
            let ps = translate(pixel_path)?;
            let plain = translate(mesh_path)?;
            let before = crate::pipeline::host::metal_mesh_bytes(&plain.dxil).ok_or("no size")?;
            let keep = victory_shader::emit::input_semantics(&ps.dxil).ok_or("no pixel inputs")?;
            let old = victory_shader::emit::set_mesh_outputs_keep(Some(keep));
            let old_reads = victory_shader::emit::set_mesh_output_reads(victory_shader::emit::input_reads(&ps.dxil));
            let ms = translate(mesh_path);
            victory_shader::emit::set_mesh_output_reads(old_reads);
            victory_shader::emit::set_mesh_outputs_keep(old);
            let ms = ms?;
            let after = crate::pipeline::host::metal_mesh_bytes(&ms.dxil).ok_or("no size")?;
            let mut line = format!("bytes {before} -> {after}");
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
