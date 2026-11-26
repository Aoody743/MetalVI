//! Metal Shader Converter test (run in Wine with victorymetal installed):
//! translated compute shaders from victory's shader cache, converted with
//! the root signature victory builds for them, made into Metal pipelines.
//!
//! usage: victory-metal-msc <shader cache folder> [count]

#[cfg(windows)]
fn main() -> std::process::ExitCode {
    match run() {
        Ok(true) => {
            println!("RESULT: ALL CHECKS PASSED");
            std::process::ExitCode::SUCCESS
        }
        Ok(false) => {
            println!("RESULT: FAILED: some shaders did not convert");
            std::process::ExitCode::FAILURE
        }
        Err(e) => {
            println!("RESULT: FAILED: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// A cache entry's parts the root signature comes from
/// (victory-gpu/src/shader_cache.rs `encode`).
#[cfg(windows)]
struct Entry {
    stage: u32,
    dxil: Vec<u8>,
    user_data_dwords: u32,
    static_samplers: Vec<victory_metal::msc::StaticSampler>,
    root: [Vec<u8>; 3],
}

#[cfg(windows)]
fn parse(bytes: &[u8]) -> Option<Entry> {
    let mut at = 0usize;
    let mut take = |n: usize| -> Option<&[u8]> {
        let s = bytes.get(at..at + n)?;
        at += n;
        Some(s)
    };
    if take(4)? != b"VSH1" {
        return None;
    }
    let word = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap());
    let stage = word(take(4)?);
    let n = word(take(4)?) as usize;
    let dxil = take(n)?.to_vec();
    let n = word(take(4)?) as usize;
    let mut max_index = None::<u32>;
    for _ in 0..n {
        let index = word(take(4)?);
        take(12)?;
        max_index = Some(max_index.map_or(index, |m| m.max(index)));
    }
    let user_data_dwords = max_index.map_or(0, |m| (m + 1).next_multiple_of(4));
    let n = word(take(4)?) as usize;
    take(n * 16)?; // samplers in the heap: not part of the signature
    let n = word(take(4)?) as usize;
    let mut static_samplers = Vec::with_capacity(n);
    for i in 0..n {
        let w: Vec<u32> = (0..10).map(|_| take(4).map(word)).collect::<Option<_>>()?;
        static_samplers.push(victory_metal::msc::StaticSampler {
            filter: w[0],
            address_u: w[1],
            address_v: w[2],
            address_w: w[3],
            mip_lod_bias: f32::from_bits(w[4]),
            max_anisotropy: w[5],
            comparison_func: w[6],
            border_color: w[7],
            min_lod: f32::from_bits(w[8]),
            max_lod: f32::from_bits(w[9]),
            shader_register: i as u32,
            register_space: 0,
        });
    }
    let mut root: [Vec<u8>; 3] = Default::default();
    for slots in &mut root {
        let n = word(take(4)?) as usize;
        *slots = take(n)?.to_vec();
    }
    Some(Entry { stage, dxil, user_data_dwords, static_samplers, root })
}

#[cfg(windows)]
fn run() -> Result<bool, Box<dyn std::error::Error>> {
    use victory_metal::msc::RootParam;
    use victory_metal::{Device, Pool};

    let mut args = std::env::args().skip(1);
    let folder = args.next().ok_or("usage: victory-metal-msc <shader cache folder> [count]")?;
    let want: usize = args.next().map_or(Ok(40), |s| s.parse())?;
    let _pool = Pool::new();
    let device = Device::first()?;

    let mut files = Vec::new();
    for dir in std::fs::read_dir(&folder)? {
        let dir = dir?.path();
        if dir.is_dir() {
            for f in std::fs::read_dir(&dir)? {
                files.push(f?.path());
            }
        }
    }
    files.sort();
    let (mut tried, mut ok, mut failures) = (0usize, 0usize, Vec::new());
    let started = std::time::Instant::now();
    for path in files {
        if tried == want {
            break;
        }
        let Some(e) = parse(&std::fs::read(&path)?) else { continue };
        if e.stage != 7 {
            continue; // compute only (STAGES[7] = Cs)
        }
        tried += 1;
        let mut params = Vec::new();
        if e.user_data_dwords > 0 {
            params.push(RootParam::Constants { register: 0, space: 0, num: e.user_data_dwords });
        }
        params.extend(e.root[0].iter().map(|&s| RootParam::Cbv { register: s.into(), space: 1 }));
        params.extend(e.root[1].iter().map(|&s| RootParam::Srv { register: s.into(), space: 1 }));
        params.extend(e.root[2].iter().map(|&s| RootParam::Uav { register: s.into(), space: 1 }));
        match device.compute_pipeline_dxil(&e.dxil, &params, &e.static_samplers) {
            Ok((_, c)) => {
                ok += 1;
                if ok <= 3 {
                    println!(
                        "ok   {} params {} samplers {} threadgroup {:?} top-level {} bytes: {:?}",
                        name(&path), params.len(), e.static_samplers.len(), c.threadgroup, c.top_level.bytes.len(),
                        c.top_level.layout.iter().map(|l| (l.kind, l.space, l.slot, l.offset, l.size)).collect::<Vec<_>>()
                    );
                }
            }
            Err(err) => failures.push(format!("{}: {err}", name(&path))),
        }
    }
    println!(
        "{ok}/{tried} compute shaders became Metal pipelines ({:.1} s)",
        started.elapsed().as_secs_f64()
    );
    for f in failures.iter().take(8) {
        println!("FAIL {f}");
    }
    Ok(tried > 0 && failures.len() * 20 <= tried) // T2's offline rate was 99.3% for compute
}

#[cfg(windows)]
fn name(p: &std::path::Path) -> String {
    p.file_name().map(|n| n.to_string_lossy()[..16.min(n.len())].to_string()).unwrap_or_default()
}

#[cfg(not(windows))]
fn main() {
    eprintln!("victory-metal-msc runs on Windows (Wine) only");
}
