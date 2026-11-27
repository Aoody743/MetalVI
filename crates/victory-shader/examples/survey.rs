//! Translation survey: every `*.cdxil` in a directory, natively, on a host of
//! `SURVEY_HOST_WAVE` lanes (default 32). One line per container on stdout:
//! `<file>\tok\t<wave>` or `<file>\terr\t<error>`.
//!
//! `cargo run --release -p victory-shader --example survey -- <dir> [filter]`
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = args.first().expect("directory");
    let filter = args.get(1).cloned().unwrap_or_default();
    let wave: u32 = std::env::var("SURVEY_HOST_WAVE").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
    victory_shader::emit::set_host_max_wave(wave);
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .expect("read dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "cdxil"))
        .filter(|p| p.file_name().unwrap().to_string_lossy().contains(&filter))
        .collect();
    files.sort();
    let files = Arc::new(files);
    let next = Arc::new(AtomicUsize::new(0));
    let out = Arc::new(Mutex::new(Vec::new()));
    let threads: usize = std::thread::available_parallelism().map_or(4, |n| n.get());
    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let (files, next, out) = (files.clone(), next.clone(), out.clone());
            std::thread::Builder::new()
                .stack_size(256 << 20)
                .spawn(move || loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(path) = files.get(i) else { break };
                    let name = path.file_name().unwrap().to_string_lossy().into_owned();
                    let bytes = std::fs::read(path).expect("read");
                    let line = match std::panic::catch_unwind(|| victory_shader::emit::translate(&bytes)) {
                        Ok(Ok(_)) => format!("{name}\tok\t{:?}", victory_shader::emit::last_wave()),
                        Ok(Err(e)) => format!("{name}\terr\t{e}"),
                        Err(_) => format!("{name}\terr\tPANIC"),
                    };
                    out.lock().unwrap().push(line);
                })
                .unwrap()
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let mut out = out.lock().unwrap();
    out.sort();
    for l in out.iter() {
        println!("{l}");
    }
}
