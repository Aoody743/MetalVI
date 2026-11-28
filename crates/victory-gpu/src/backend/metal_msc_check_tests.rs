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
    // The settings a Metal run translates with (`victory_gpu::host`); with
    // doubles left on (`VICTORY_MSC_CHECK_F64=1`) MSC aborts on the wave
    // sums of doubles in 6871/926c/a101/5dc3/6f22.
    victory_shader::emit::set_metal_host();
    if std::env::var("VICTORY_MSC_CHECK_F64").is_ok_and(|v| v == "1") {
        victory_shader::emit::set_host_f64(true);
    }
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

/// The compute shaders MSC 4.0.3 crashes on are kept from it by the console
/// shader's hash; the ones it takes with the Metal settings are not.
#[test]
fn known_converter_crashes_are_not_converted() {
    for crash in ["926c46bab3b86bd6", "a10179ab7eda0dc3"] {
        assert!(msc_known_crash(crash).is_some_and(|why| why.contains(crash)), "{crash}");
    }
    for fine in ["6871d6e4b3ac9a10", "5dc3b25986622949", "6f225b641c5b0708", "2561af9577c47289", ""] {
        assert!(msc_known_crash(fine).is_none(), "{fine}");
    }
}
