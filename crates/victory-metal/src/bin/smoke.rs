//! victory-metal smoke test (run it in Wine with victorymetal installed):
//! zero-copy compute on VirtualAlloc'd memory, the way title memory will be
//! used, through the safe wrappers.

#[cfg(windows)]
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => {
            println!("RESULT: ALL CHECKS PASSED");
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            println!("RESULT: FAILED: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(windows)]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use victory_metal::{Compute, Device, Pool};
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};

    const LEN: usize = 2 << 20; // one chunk, as DESIGN.md cuts the page pool
    const WORDS: u64 = 1 << 18; // the first 1 MiB is written by the GPU
    let _pool = Pool::new();

    let device = Device::first()?;
    println!("device: {}", device.name());

    // SAFETY: a fresh allocation.
    let mem = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
    if mem.is_null() {
        return Err("VirtualAlloc".into());
    }
    // SAFETY: LEN bytes just committed.
    let words = unsafe { std::slice::from_raw_parts_mut(mem.cast::<u32>(), LEN / 4) };
    words.fill(0xCCCC_CCCC);

    // SAFETY: the allocation lives until the end of run(), after the GPU is done.
    let buffer = unsafe { device.wrap(mem, LEN as u64)? };
    println!("cpu {mem:p} gpu {:#x}", buffer.gpu_address);

    let pso = device.compute_pipeline_msl(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         kernel void fill(device uint *out [[buffer(0)]], uint i [[thread_position_in_grid]]) { out[i] = i * 2654435761u; }\n",
        "fill",
    )?;
    let queue = device.new_queue(4)?;
    let cb = queue.command_buffer()?;
    cb.compute(&[
        Compute::Pipeline { pipeline: &pso, threads_per_group: [64, 1, 1] },
        Compute::Buffer { buffer: &buffer, offset: 0, index: 0 },
        Compute::Dispatch { groups: [WORDS / 64, 1, 1] },
    ])?;
    cb.commit();
    cb.wait()?;

    let bad = (0..WORDS as usize).filter(|&i| words[i] != (i as u32).wrapping_mul(2654435761)).count();
    let canary = words[WORDS as usize..].iter().filter(|&&w| w != 0xCCCC_CCCC).count();
    println!("{WORDS} words written by the GPU, {bad} wrong, {canary} canary words changed");
    if bad != 0 || canary != 0 {
        return Err("GPU results wrong".into());
    }
    Ok(())
}

#[cfg(not(windows))]
fn main() {
    eprintln!("victory-metal-smoke runs on Windows (Wine) only");
}
