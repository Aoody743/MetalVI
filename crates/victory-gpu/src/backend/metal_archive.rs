//! The Metal backend's pipeline binary archive (PLAN-NATIVE-METAL P8):
//! opened when the backend starts, saved every so often while pipelines are
//! being added, so the next run finds the GPU binaries instead of compiling.
//!
//! `VICTORY_METAL_ARCHIVE` (opt in):
//! - unset, empty or `0`: no archive (pipelines compile as before);
//! - `1`: `<metal cache dir>/pipelines-<device>.metalar` (the directory of
//!   `VICTORY_METAL_CACHE`, else `<VICTORY_CACHE_DIR>/metal`);
//! - anything else: that file (`Z:\...` or a unix path).
//!
//! `VICTORY_METAL_ARCHIVE_SAVE_EVERY` (default 64) pipelines added, or any
//! added and `VICTORY_METAL_ARCHIVE_SAVE_SECS` (default 20) passed, start a
//! save on a thread of its own (pipelines made in the last seconds before the
//! process ends are not saved: they compile again next run).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use victory_metal::Device;
use victory_metal::archive::{self, PipelineArchive};

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

/// The archive file `VICTORY_METAL_ARCHIVE` asks for, if any.
fn wanted_path(device: &Device) -> Option<std::path::PathBuf> {
    let v = std::env::var_os("VICTORY_METAL_ARCHIVE")?;
    let s = v.to_str()?.trim().to_owned();
    if s.is_empty() || s == "0" {
        return None;
    }
    if s != "1" {
        return Some(s.into());
    }
    let dir = std::env::var_os("VICTORY_METAL_CACHE")
        .filter(|v| !v.is_empty())
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("VICTORY_CACHE_DIR").filter(|v| !v.is_empty()).map(|d| std::path::PathBuf::from(d).join("metal"))
        })?;
    let _ = crate::make_dir(&dir);
    let name: String =
        device.name().chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' }).collect();
    Some(dir.join(format!("pipelines-{name}.metalar")))
}

/// Opens and installs the archive `VICTORY_METAL_ARCHIVE` asks for.
pub(crate) fn open(device: &Device) {
    let Some(path) = wanted_path(device) else { return };
    match PipelineArchive::open(device, &path) {
        Ok((a, why)) => {
            if let Some(why) = why {
                tracing::warn!(path = %path.display(), error = %why, "Metal pipeline archive did not load (moved aside); starting empty");
            }
            let bytes = std::fs::metadata(a.path()).map(|m| m.len()).unwrap_or(0);
            tracing::info!(path = %a.path().display(), loaded = a.stats().loaded, bytes,
                bounds_policy = ?victory_metal::msc::bounds_policy(),
                bounds_policy_supported = victory_metal::msc::bounds_policy_supported(),
                "Metal pipeline archive");
            archive::install(Some(Arc::new(a)));
        }
        Err(e) => tracing::warn!(path = %path.display(), error = %e, "no Metal pipeline archive"),
    }
}

static SAVING: AtomicBool = AtomicBool::new(false);

fn save(a: &PipelineArchive) {
    let start = std::time::Instant::now();
    match a.save() {
        Ok(bytes) => {
            let s = a.stats();
            tracing::info!(hits = s.hits, misses = s.misses, failed = s.failed, add_failures = s.add_failures,
                saves = s.saves, bytes, ms = start.elapsed().as_millis() as u64, "Metal pipeline archive saved");
        }
        Err(e) => tracing::warn!(error = %e, "Metal pipeline archive save failed"),
    }
}

/// After a pipeline was made: starts a save when enough is new.
pub(crate) fn tick() {
    let Some(a) = archive::installed() else { return };
    let every = env_u64("VICTORY_METAL_ARCHIVE_SAVE_EVERY", 64).max(1);
    let after = Duration::from_secs(env_u64("VICTORY_METAL_ARCHIVE_SAVE_SECS", 20));
    if !a.should_save(every, after) || SAVING.swap(true, Ordering::AcqRel) {
        return;
    }
    let spawned = std::thread::Builder::new().name("metal-archive-save".into()).spawn(move || {
        let _pool = victory_metal::Pool::new();
        save(&a);
        SAVING.store(false, Ordering::Release);
    });
    if spawned.is_err() {
        SAVING.store(false, Ordering::Release);
    }
}
