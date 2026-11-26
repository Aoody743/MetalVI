//! Offline Metal Shader Converter cache warm-up (`victory-warm-metal`).
//!
//! Reads translated [`HostShader`] entries (`VSH1`) from the shader disk cache
//! (`shader_cache::decode`) and pre-populates the Metal disk cache (`VICTORY_METAL_CACHE`)
//! using the exact same root signature layout and cache path functions as the game
//! (`metal::root_params`, `metal::static_samplers`, `metal::msc_cache_path`,
//! `metal::convert_cached`).

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use victory_metal::Pool;
use victory_metal::msc::{Converted, Stage, convert};
use victory_shader::HwStage;
use victory_shader::emit::HostShader;

use super::metal::{convert_cached, msc_cache_path, root_params, static_samplers};

/// Shader stages that the Metal backend caches as `.vmsc` files.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum WarmStage {
    Compute,
    Vertex,
    Fragment,
    Mesh,
}

impl WarmStage {
    /// Maps a translated shader's [`HwStage`] to its Metal cache stage, if cached.
    pub fn from_hw_stage(stage: HwStage) -> Option<Self> {
        match stage {
            HwStage::Cs => Some(Self::Compute),
            HwStage::Vs => Some(Self::Vertex),
            HwStage::Ps => Some(Self::Fragment),
            HwStage::Ms => Some(Self::Mesh),
            HwStage::Ls | HwStage::Hs | HwStage::Es | HwStage::Gs => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Compute => "compute",
            Self::Vertex => "vertex",
            Self::Fragment => "fragment",
            Self::Mesh => "mesh",
        }
    }

    /// Computes the exact `.vmsc` path that the live Metal backend checks for `host`.
    ///
    /// Compute pipelines in `MetalBackend::make_compute` read `<hash>.vmsc`, whereas
    /// graphics stages in `convert_cached` read `<hash>.<Stage>.vmsc`.
    pub fn cache_path(self, host: &HostShader) -> Option<PathBuf> {
        let params = root_params(host);
        let samplers = static_samplers(host);
        let base = msc_cache_path(&host.dxil, &params, &samplers)?;
        Some(match self {
            Self::Compute => base,
            Self::Vertex => base.with_extension(format!("{:?}.vmsc", Stage::Vertex)),
            Self::Fragment => base.with_extension(format!("{:?}.vmsc", Stage::Fragment)),
            Self::Mesh => base.with_extension(format!("{:?}.vmsc", Stage::Mesh)),
        })
    }
}

/// Options for [`warm_shader_cache`].
#[derive(Clone, Copy, Debug)]
pub struct WarmOptions {
    pub jobs: usize,
    pub limit: Option<usize>,
    pub verify_sample: usize,
    pub progress_every: usize,
}

impl Default for WarmOptions {
    fn default() -> Self {
        Self {
            jobs: std::thread::available_parallelism().map_or(4, |n| n.get()),
            limit: None,
            verify_sample: 32,
            progress_every: 1000,
        }
    }
}

/// Per-stage conversion counts.
#[derive(Clone, Copy, Debug, Default)]
pub struct StageCounts {
    pub eligible: usize,
    pub converted: usize,
    pub already_cached: usize,
    pub failed: usize,
}

/// Summary produced by [`warm_shader_cache`].
#[derive(Debug)]
pub struct WarmReport {
    pub files_scanned: usize,
    pub decode_failed: usize,
    pub unsupported_stage: usize,
    pub duplicate_keys: usize,
    pub stages: BTreeMap<&'static str, StageCounts>,
    pub failures: BTreeMap<String, (usize, String)>,
    pub output_files: usize,
    pub output_bytes: u64,
    pub verified_ok: usize,
    pub verified_failed: usize,
    pub elapsed: Duration,
}

impl WarmReport {
    pub fn total_eligible(&self) -> usize {
        self.stages.values().map(|s| s.eligible).sum()
    }
    pub fn total_converted(&self) -> usize {
        self.stages.values().map(|s| s.converted).sum()
    }
    pub fn total_already_cached(&self) -> usize {
        self.stages.values().map(|s| s.already_cached).sum()
    }
    pub fn total_failed(&self) -> usize {
        self.stages.values().map(|s| s.failed).sum()
    }
}

/// Collects all non-hidden shader cache files under `dir`, sorted deterministically.
pub fn collect_shader_files(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    collect_recursive(dir, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_recursive(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        if name.to_string_lossy().starts_with('.') {
            continue;
        }
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            collect_recursive(&path, out)?;
        } else {
            out.push(path);
        }
    }
    Ok(())
}

/// Whether `victorymetal.dll` is loaded/available for offline MSC warm-up.
pub fn available() -> bool {
    victory_metal::sys::available()
}

/// Converts one decoded [`HostShader`] into its Metal cache `.vmsc` file.
///
/// Callers invoking this from multiple threads must ensure no two threads
/// target the same cache path concurrently (since `convert_cached` uses a
/// per-PID staging extension).
pub fn warm_host_shader(host: &HostShader, thread_id: usize) -> Result<WarmOutcome, String> {
    let Some(stage) = WarmStage::from_hw_stage(host.stage) else {
        return Ok(WarmOutcome::UnsupportedStage);
    };
    let path = stage
        .cache_path(host)
        .ok_or_else(|| "no Metal cache directory configured (set VICTORY_METAL_CACHE or VICTORY_CACHE_DIR)".to_owned())?;
    if let Ok(bytes) = crate::read_file(&path)
        && Converted::from_bytes(&bytes).is_some()
    {
        return Ok(WarmOutcome::AlreadyCached(path));
    }
    match stage {
        WarmStage::Compute => {
            let params = root_params(host);
            let samplers = static_samplers(host);
            let converted = convert(&host.dxil, Stage::Compute, &params, &samplers).map_err(|e| e.to_string())?;
            let tmp = path.with_extension(format!("tmp{}.{thread_id}", std::process::id()));
            std::fs::write(&tmp, converted.to_bytes()).map_err(|e| format!("write {}: {e}", tmp.display()))?;
            if let Err(e) = std::fs::rename(&tmp, &path) {
                let _ = std::fs::remove_file(&tmp);
                return Err(format!("rename {}: {e}", path.display()));
            }
        }
        WarmStage::Vertex => {
            convert_cached(host, Stage::Vertex)?;
        }
        WarmStage::Fragment => {
            convert_cached(host, Stage::Fragment)?;
        }
        WarmStage::Mesh => {
            convert_cached(host, Stage::Mesh)?;
        }
    }
    Ok(WarmOutcome::Converted(path))
}

#[derive(Debug, PartialEq, Eq)]
pub enum WarmOutcome {
    Converted(PathBuf),
    AlreadyCached(PathBuf),
    UnsupportedStage,
}

/// Warms the Metal shader cache (`VICTORY_METAL_CACHE` or `VICTORY_CACHE_DIR/metal`)
/// from every `VSH1` shader entry in `shader_cache_dir` using `opts.jobs` worker threads.
pub fn warm_shader_cache(shader_cache_dir: &Path, opts: WarmOptions) -> anyhow::Result<WarmReport> {
    let started = Instant::now();
    let mut files = collect_shader_files(shader_cache_dir)?;
    if let Some(limit) = opts.limit {
        files.truncate(limit);
    }
    let files_scanned = files.len();

    // Probe the cache path once single-threaded before fanning out, so `OnceLock`
    // in `msc_cache_path` initializes cleanly and we fail fast if unconfigured.
    let probe = HostShader {
        stage: HwStage::Cs,
        dxil: Vec::new(),
        user_data: Vec::new(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        root_indexed_views: Vec::new(),
    };
    let probe_path = WarmStage::Compute
        .cache_path(&probe)
        .ok_or_else(|| anyhow::anyhow!("VICTORY_METAL_CACHE (or VICTORY_CACHE_DIR) is not set"))?;
    let metal_cache_dir = probe_path
        .parent()
        .expect("msc_cache_path always joins onto cache dir")
        .to_path_buf();

    let next = AtomicUsize::new(0);
    let processed = AtomicUsize::new(0);
    let decode_failed = AtomicUsize::new(0);
    let unsupported_stage = AtomicUsize::new(0);
    let duplicate_keys = AtomicUsize::new(0);

    let seen: Mutex<HashSet<PathBuf>> = Mutex::new(HashSet::new());
    let sample_paths: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());
    let failures: Mutex<BTreeMap<String, (usize, String)>> = Mutex::new(BTreeMap::new());
    let stage_counts: Mutex<BTreeMap<&'static str, StageCounts>> = Mutex::new(BTreeMap::new());

    let jobs = opts.jobs.max(1).min(files.len().max(1));
    std::thread::scope(|scope| {
        for thread_id in 0..jobs {
            let files = &files;
            let next = &next;
            let processed = &processed;
            let decode_failed = &decode_failed;
            let unsupported_stage = &unsupported_stage;
            let duplicate_keys = &duplicate_keys;
            let seen = &seen;
            let sample_paths = &sample_paths;
            let failures = &failures;
            let stage_counts = &stage_counts;
            scope.spawn(move || {
                let _pool = Pool::new();
                let mut local_stages: BTreeMap<&'static str, StageCounts> = BTreeMap::new();
                let mut local_failures: BTreeMap<String, (usize, String)> = BTreeMap::new();
                let mut local_samples: Vec<PathBuf> = Vec::new();
                loop {
                    let idx = next.fetch_add(1, Ordering::Relaxed);
                    let Some(file) = files.get(idx) else { break };

                    let done = processed.fetch_add(1, Ordering::Relaxed) + 1;
                    if opts.progress_every > 0 && done.is_multiple_of(opts.progress_every) {
                        eprintln!(
                            "[{:6.1} s] {done}/{files_scanned} shader files processed",
                            started.elapsed().as_secs_f64()
                        );
                    }

                    let Ok(bytes) = crate::read_file(file) else {
                        decode_failed.fetch_add(1, Ordering::Relaxed);
                        continue;
                    };
                    let Some(host) = crate::shader_cache::decode(&bytes) else {
                        decode_failed.fetch_add(1, Ordering::Relaxed);
                        continue;
                    };
                    let Some(stage) = WarmStage::from_hw_stage(host.stage) else {
                        unsupported_stage.fetch_add(1, Ordering::Relaxed);
                        continue;
                    };
                    let Some(path) = stage.cache_path(&host) else {
                        continue;
                    };
                    {
                        let mut guard = seen.lock().unwrap_or_else(|e| e.into_inner());
                        if !guard.insert(path.clone()) {
                            duplicate_keys.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                    }

                    let entry = local_stages.entry(stage.label()).or_default();
                    entry.eligible += 1;

                    match warm_host_shader(&host, thread_id) {
                        Ok(WarmOutcome::Converted(p)) => {
                            entry.converted += 1;
                            if local_samples.len() < opts.verify_sample {
                                local_samples.push(p);
                            }
                        }
                        Ok(WarmOutcome::AlreadyCached(p)) => {
                            entry.already_cached += 1;
                            if local_samples.len() < opts.verify_sample {
                                local_samples.push(p);
                            }
                        }
                        Ok(WarmOutcome::UnsupportedStage) => {
                            unsupported_stage.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(err) => {
                            entry.failed += 1;
                            let e = local_failures
                                .entry(err)
                                .or_insert_with(|| (0, file.display().to_string()));
                            e.0 += 1;
                        }
                    }
                }

                let mut g_stages = stage_counts.lock().unwrap_or_else(|e| e.into_inner());
                for (k, v) in local_stages {
                    let dst = g_stages.entry(k).or_default();
                    dst.eligible += v.eligible;
                    dst.converted += v.converted;
                    dst.already_cached += v.already_cached;
                    dst.failed += v.failed;
                }
                drop(g_stages);

                let mut g_fail = failures.lock().unwrap_or_else(|e| e.into_inner());
                for (k, (cnt, ex)) in local_failures {
                    let dst = g_fail.entry(k).or_insert((0, ex));
                    dst.0 += cnt;
                }
                drop(g_fail);

                let mut g_samples = sample_paths.lock().unwrap_or_else(|e| e.into_inner());
                for p in local_samples {
                    if g_samples.len() >= opts.verify_sample {
                        break;
                    }
                    g_samples.push(p);
                }
            });
        }
    });

    let samples = sample_paths.into_inner().unwrap_or_else(|e| e.into_inner());
    let (mut verified_ok, mut verified_failed) = (0usize, 0usize);
    for path in samples.into_iter().take(opts.verify_sample) {
        let valid = crate::read_file(&path)
            .ok()
            .and_then(|b| Converted::from_bytes(&b))
            .is_some_and(|c| !c.metallib.is_empty() && !c.entry.is_empty());
        if valid {
            verified_ok += 1;
        } else {
            verified_failed += 1;
        }
    }

    let (output_files, output_bytes) = measure_vmsc_dir(&metal_cache_dir);

    Ok(WarmReport {
        files_scanned,
        decode_failed: decode_failed.into_inner(),
        unsupported_stage: unsupported_stage.into_inner(),
        duplicate_keys: duplicate_keys.into_inner(),
        stages: stage_counts.into_inner().unwrap_or_else(|e| e.into_inner()),
        failures: failures.into_inner().unwrap_or_else(|e| e.into_inner()),
        output_files,
        output_bytes,
        verified_ok,
        verified_failed,
        elapsed: started.elapsed(),
    })
}

fn measure_vmsc_dir(dir: &Path) -> (usize, u64) {
    let mut count = 0usize;
    let mut bytes = 0u64;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "vmsc")
                && let Ok(data) = crate::read_file(&path)
            {
                count += 1;
                bytes += data.len() as u64;
            }
        }
    }
    (count, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn dummy_host(stage: HwStage, dxil: &[u8]) -> HostShader {
        HostShader {
            stage,
            dxil: dxil.to_vec(),
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            root_uavs: Vec::new(),
            root_indexed_views: Vec::new(),
        }
    }

    #[test]
    fn stage_mapping_covers_all_hw_stages() {
        assert_eq!(WarmStage::from_hw_stage(HwStage::Cs), Some(WarmStage::Compute));
        assert_eq!(WarmStage::from_hw_stage(HwStage::Vs), Some(WarmStage::Vertex));
        assert_eq!(WarmStage::from_hw_stage(HwStage::Ps), Some(WarmStage::Fragment));
        assert_eq!(WarmStage::from_hw_stage(HwStage::Ms), Some(WarmStage::Mesh));
        for unsupported in [HwStage::Ls, HwStage::Hs, HwStage::Es, HwStage::Gs] {
            assert_eq!(WarmStage::from_hw_stage(unsupported), None);
        }
    }

    #[test]
    fn cache_path_matches_live_backend_naming() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        if std::env::var_os("VICTORY_METAL_CACHE").is_none() {
            // SAFETY: serialized by ENV_LOCK in unit tests.
            unsafe { std::env::set_var("VICTORY_METAL_CACHE", dir.path()) };
        }
        let cs = dummy_host(HwStage::Cs, b"dxil-cs");
        let vs = dummy_host(HwStage::Vs, b"dxil-vs");
        let ps = dummy_host(HwStage::Ps, b"dxil-ps");
        let ms = dummy_host(HwStage::Ms, b"dxil-ms");

        let base_cs = msc_cache_path(&cs.dxil, &root_params(&cs), &static_samplers(&cs)).unwrap();
        assert_eq!(WarmStage::Compute.cache_path(&cs).unwrap(), base_cs);
        assert!(base_cs.file_name().unwrap().to_string_lossy().ends_with(".vmsc"));
        assert!(!base_cs.file_name().unwrap().to_string_lossy().ends_with(".Compute.vmsc"));

        let base_vs = msc_cache_path(&vs.dxil, &root_params(&vs), &static_samplers(&vs)).unwrap();
        assert_eq!(WarmStage::Vertex.cache_path(&vs).unwrap(), base_vs.with_extension("Vertex.vmsc"));

        let base_ps = msc_cache_path(&ps.dxil, &root_params(&ps), &static_samplers(&ps)).unwrap();
        assert_eq!(WarmStage::Fragment.cache_path(&ps).unwrap(), base_ps.with_extension("Fragment.vmsc"));

        let base_ms = msc_cache_path(&ms.dxil, &root_params(&ms), &static_samplers(&ms)).unwrap();
        assert_eq!(WarmStage::Mesh.cache_path(&ms).unwrap(), base_ms.with_extension("Mesh.vmsc"));
    }

    #[test]
    fn collect_shader_files_ignores_temp_files_and_sorts() {
        let dir = tempfile::tempdir().unwrap();
        let d0 = dir.path().join("00");
        let d1 = dir.path().join("0a");
        std::fs::create_dir_all(&d0).unwrap();
        std::fs::create_dir_all(&d1).unwrap();
        std::fs::write(d1.join("bbb"), b"1").unwrap();
        std::fs::write(d0.join("aaa"), b"2").unwrap();
        std::fs::write(d0.join(".tmp-1-0"), b"3").unwrap();
        let found = collect_shader_files(dir.path()).unwrap();
        assert_eq!(found, vec![d0.join("aaa"), d1.join("bbb")]);
    }
}
