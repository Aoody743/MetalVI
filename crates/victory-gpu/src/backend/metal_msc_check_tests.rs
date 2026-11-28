//! Offline check that Metal Shader Converter takes the translation of shader containers
//! (`*.cdxil`, as the shader cache keeps them) and Metal builds the pipeline. Ignored: run
//! by hand with `VICTORY_MSC_CHECK` = container paths separated by `;` (Wine paths, say
//! `Z:\...\all-shaders\96ccb14ef981fd85.compute.cdxil`). One line a container on stdout:
//! `MSC <path> ok` or `MSC <path> ERR <why>`. MSC aborts the process on some DXIL
//! (`Fatal: ...`), so check one container a process when that matters.

use super::*;

#[test]
#[ignore]
fn msc_takes_listed_containers() {
    let _pool = victory_metal::Pool::new();
    let backend = MetalBackend::new().unwrap();
    victory_shader::emit::set_host_max_wave(32);
    let list = std::env::var("VICTORY_MSC_CHECK").unwrap_or_default();
    let mut failed = 0;
    for path in list.split(';').filter(|p| !p.is_empty()) {
        let result = crate::read_file(std::path::Path::new(path))
            .map_err(|e| e.to_string())
            .and_then(|bytes| victory_shader::emit::translate(&bytes).map_err(|e| e.to_string()))
            .and_then(|host| {
                let (params, samplers) = (root_params(&host), static_samplers(&host));
                backend.device.compute_pipeline_dxil(&host.dxil, &params, &samplers).map(|_| ()).map_err(|e| e.to_string())
            });
        match result {
            Ok(()) => println!("MSC {path} ok"),
            Err(e) => {
                failed += 1;
                println!("MSC {path} ERR {e}");
            }
        }
    }
    assert_eq!(failed, 0, "containers MSC or Metal refused");
}
