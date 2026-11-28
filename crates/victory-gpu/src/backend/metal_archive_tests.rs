//! P8 through the backend: `MetalBackend::new` opens the archive `VICTORY_METAL_ARCHIVE`
//! names, `metal_archive::tick` saves it once enough is new, and the next
//! process finds the pipeline in it.

use super::*;

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void archived(device uint* b [[buffer(0)]], uint i [[thread_position_in_grid]]) { b[i] += 7; }
"#;

/// Run twice (two processes) with `VICTORY_METAL_ARCHIVE` = a `Z:\` path that does not
/// exist yet and `VICTORY_METAL_ARCHIVE_SAVE_EVERY=1`: the first run compiles and saves,
/// the second loads and hits.
#[test]
#[ignore = "run twice under Wine with VICTORY_METAL_ARCHIVE=<Z: path>, VICTORY_METAL_ARCHIVE_SAVE_EVERY=1"]
fn the_backend_archive_is_saved_and_found_again() {
    let Some(path) = std::env::var_os("VICTORY_METAL_ARCHIVE") else { return };
    let _pool = victory_metal::Pool::new();
    let backend = MetalBackend::new().unwrap();
    let archive = victory_metal::archive::installed().expect("the backend opened no archive");
    let loaded = archive.stats().loaded;
    let library = backend.device.library_msl(SOURCE).unwrap();
    backend.device.compute_pipeline_from_library(&library, "archived").unwrap();
    crate::backend::metal_archive::tick();
    let s = archive.stats();
    println!("BACKEND ARCHIVE loaded={loaded} hits={} misses={}", s.hits, s.misses);
    if loaded {
        assert_eq!((s.hits, s.misses), (1, 0));
    } else {
        assert_eq!((s.hits, s.misses), (0, 1));
        // The save runs on a thread of its own.
        let file = std::path::PathBuf::from(&path);
        let start = std::time::Instant::now();
        while archive.stats().saves == 0 && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert_eq!(archive.stats().saves, 1, "no save within 10 s");
        assert!(std::fs::metadata(&file).is_ok_and(|m| m.len() > 0));
    }
    victory_metal::archive::install(None);
}
