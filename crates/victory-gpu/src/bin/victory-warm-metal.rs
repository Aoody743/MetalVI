//! Offline Metal Shader Converter cache warm-up CLI (`victory-warm-metal`).
//!
//! Runs under Wine with `victorymetal.dll` installed. Reads translated `VSH1`
//! entries from the shader cache and pre-populates `VICTORY_METAL_CACHE` with
//! `.vmsc` files under the exact names used by the game's Metal backend.

#[cfg(windows)]
fn main() -> std::process::ExitCode {
    match run() {
        Ok(ok) if ok => std::process::ExitCode::SUCCESS,
        Ok(_) => std::process::ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e:#}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(windows)]
fn usage() -> &'static str {
    "usage: victory-warm-metal [--shader-cache <dir>] [--metal-cache <dir>] [--jobs <n>] [--limit <n>] [--verify <n>] [--progress <n>]"
}

#[cfg(windows)]
fn run() -> anyhow::Result<bool> {
    use std::path::PathBuf;
    use victory_gpu::metal_warm::{WarmOptions, warm_shader_cache};

    let mut shader_cache: Option<PathBuf> = None;
    let mut metal_cache: Option<PathBuf> = None;
    let mut opts = WarmOptions::default();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{}", usage());
                return Ok(true);
            }
            "--shader-cache" => {
                shader_cache = Some(PathBuf::from(args.next().ok_or_else(|| anyhow::anyhow!("missing value for --shader-cache"))?));
            }
            "--metal-cache" => {
                metal_cache = Some(PathBuf::from(args.next().ok_or_else(|| anyhow::anyhow!("missing value for --metal-cache"))?));
            }
            "--jobs" | "-j" => {
                let v = args.next().ok_or_else(|| anyhow::anyhow!("missing value for --jobs"))?;
                opts.jobs = v.parse()?;
            }
            "--limit" => {
                let v = args.next().ok_or_else(|| anyhow::anyhow!("missing value for --limit"))?;
                opts.limit = Some(v.parse()?);
            }
            "--verify" => {
                let v = args.next().ok_or_else(|| anyhow::anyhow!("missing value for --verify"))?;
                opts.verify_sample = v.parse()?;
            }
            "--progress" => {
                let v = args.next().ok_or_else(|| anyhow::anyhow!("missing value for --progress"))?;
                opts.progress_every = v.parse()?;
            }
            other if !other.starts_with('-') && shader_cache.is_none() => {
                shader_cache = Some(PathBuf::from(other));
            }
            other if !other.starts_with('-') && metal_cache.is_none() => {
                metal_cache = Some(PathBuf::from(other));
            }
            other => anyhow::bail!("unknown argument {other}\n{}", usage()),
        }
    }

    if let Some(dir) = &metal_cache {
        // SAFETY: called single-threaded at startup before worker threads start.
        unsafe { std::env::set_var("VICTORY_METAL_CACHE", dir) };
    }

    let shader_dir = shader_cache
        .or_else(|| {
            std::env::var_os("VICTORY_CACHE_DIR")
                .filter(|v| !v.is_empty())
                .map(|d| PathBuf::from(d).join("shaders"))
        })
        .ok_or_else(|| anyhow::anyhow!("missing --shader-cache <dir> (or VICTORY_CACHE_DIR)\n{}", usage()))?;

    println!(
        "warming Metal cache from {} with {} worker thread(s) (limit={:?})",
        shader_dir.display(),
        opts.jobs,
        opts.limit
    );
    let report = warm_shader_cache(&shader_dir, opts)?;

    println!("--- summary ---");
    println!("files scanned      : {}", report.files_scanned);
    println!("decode failed      : {}", report.decode_failed);
    println!("unsupported stage  : {}", report.unsupported_stage);
    println!("duplicate keys     : {}", report.duplicate_keys);
    println!("eligible shaders   : {}", report.total_eligible());
    println!("converted new      : {}", report.total_converted());
    println!("already cached     : {}", report.total_already_cached());
    println!("conversion failed  : {}", report.total_failed());
    for (stage, c) in &report.stages {
        println!(
            "  {:<8} : eligible={} converted={} cached={} failed={}",
            stage, c.eligible, c.converted, c.already_cached, c.failed
        );
    }
    println!("output .vmsc files : {}", report.output_files);
    println!(
        "output disk size   : {} bytes ({:.2} MiB)",
        report.output_bytes,
        report.output_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "sample verify      : {}/{} valid via Converted::from_bytes",
        report.verified_ok,
        report.verified_ok + report.verified_failed
    );
    println!("elapsed            : {:.2} s", report.elapsed.as_secs_f64());

    if !report.failures.is_empty() {
        println!("--- top failure reasons ---");
        let mut sorted: Vec<_> = report.failures.iter().collect();
        sorted.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
        for (reason, (count, example)) in sorted.into_iter().take(20) {
            println!("  {count:>5}  {reason}  (e.g. {example})");
        }
    }

    Ok(report.verified_failed == 0 && report.total_eligible() > 0)
}

#[cfg(not(windows))]
fn main() {
    eprintln!("victory-warm-metal runs on Windows (Wine) only");
}
