//! Chunk manager test (run in Wine with victorymetal installed): title-like
//! memory (a section mapped in 64 KiB views, as xmem does) read and written
//! by the GPU through the chunk table, then remapped and read again.

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
mod win {
    use core::ffi::c_void;
    use windows_sys::Win32::Foundation::{HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Memory::*;

    pub const PAGE: usize = 64 << 10;

    pub fn section(pages: usize) -> HANDLE {
        let bytes = (pages * PAGE) as u64;
        // SAFETY: an anonymous, committed section.
        unsafe {
            CreateFileMappingA(INVALID_HANDLE_VALUE, core::ptr::null(), PAGE_READWRITE | SEC_COMMIT, (bytes >> 32) as u32, bytes as u32, core::ptr::null())
        }
    }

    /// Reserves address space for `pages` and releases it (to map views there).
    pub fn hole(pages: usize) -> usize {
        // SAFETY: a reservation released at once: the address is free after.
        unsafe {
            let p = VirtualAlloc(core::ptr::null(), pages * PAGE, MEM_RESERVE, PAGE_NOACCESS);
            VirtualFree(p, 0, MEM_RELEASE);
            p as usize
        }
    }

    /// Maps section pages `[first, first+n)` at `at`.
    pub fn map(s: HANDLE, first: usize, n: usize, at: usize) -> Result<(), String> {
        let off = (first * PAGE) as u64;
        // SAFETY: `at` is free address space of ours.
        let v = unsafe { MapViewOfFileEx(s, FILE_MAP_ALL_ACCESS, (off >> 32) as u32, off as u32, n * PAGE, at as *const c_void) };
        if v.Value as usize == at { Ok(()) } else { Err(format!("MapViewOfFileEx at {at:#x} got {:p}", v.Value)) }
    }

    pub fn unmap(at: usize) {
        // SAFETY: a view we mapped at `at`.
        unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: at as *mut c_void }) };
    }

    pub fn word(at: usize) -> *mut u32 {
        at as *mut u32
    }
}

#[cfg(windows)]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use victory_metal::chunks::Chunks;
    use victory_metal::{Buffer, Compute, Device, Pool};
    use win::*;

    const PAGES: usize = 64; // 4 MiB: two chunks
    let _pool = Pool::new();
    let device = Device::first()?;
    let s = section(PAGES + 8);
    if s.is_null() {
        return Err("CreateFileMapping".into());
    }
    let base = hole(PAGES);
    if base % (2 << 20) != 0 {
        println!("note: base {base:#x} not chunk aligned (chunks start below it)");
    }
    // Title-like layout: section pages mapped one view per 64 KiB page.
    for p in 0..PAGES {
        map(s, p, 1, base + p * PAGE)?;
    }
    // Mark every section page (including the spare ones past PAGES).
    let marker = |page: usize| 0x1000 + page as u32;
    let spare = hole(8);
    map(s, PAGES, 8, spare)?;
    for p in 0..PAGES {
        // SAFETY: mapped above.
        unsafe { *word(base + p * PAGE) = marker(p) };
    }
    for p in 0..8 {
        // SAFETY: mapped above.
        unsafe { *word(spare + p * PAGE) = marker(PAGES + p) };
    }

    let mut chunks = Chunks::new(&device, base as u64, (PAGES * PAGE) as u64)?;
    chunks.wrap(base as u64, (PAGES * PAGE) as u64);
    println!("wrapped: {:?}", chunks.stats);

    // The kernel: for each title address in `vas`, find its page through the
    // table, read the word there into `out`, and add 1 to it in title memory.
    let pso = device.compute_pipeline_msl(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         kernel void k(device const ulong *table [[buffer(0)]], device const ulong *args [[buffer(1)]],\n\
                       device uint *out [[buffer(2)]], uint i [[thread_position_in_grid]]) {\n\
           ulong base = args[0]; ulong n = args[1]; if (i >= n) return;\n\
           ulong va = args[2 + i];\n\
           ulong page = table[(va - base) >> 16];\n\
           if (page == 0) { out[i] = 0xDEADu; return; }\n\
           device uint *p = (device uint *)(page + (va & 0xFFFFul));\n\
           out[i] = *p; *p = *p + 1u;\n\
         }\n",
        "k",
    )?;
    let queue = device.new_queue(4)?;
    queue.add_residency_set(chunks.residency());

    // Scratch memory of our own for args and results.
    let scratch = hole(2);
    // SAFETY: committing our own reservation-free range.
    let scratch = unsafe {
        windows_sys::Win32::System::Memory::VirtualAlloc(scratch as *const _, 2 * PAGE, windows_sys::Win32::System::Memory::MEM_RESERVE | windows_sys::Win32::System::Memory::MEM_COMMIT, windows_sys::Win32::System::Memory::PAGE_READWRITE)
    } as usize;
    let args_buf: Buffer = unsafe { device.wrap(scratch as *mut _, PAGE as u64)? };
    let out_buf: Buffer = unsafe { device.wrap((scratch + PAGE) as *mut _, PAGE as u64)? };
    let args = scratch as *mut u64;
    let out = (scratch + PAGE) as *mut u32;

    let probe = |chunks: &Chunks, pages: &[usize]| -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let _pool = Pool::new();
        // SAFETY: our scratch memory; the GPU is idle between probes.
        unsafe {
            *args = chunks.base;
            *args.add(1) = pages.len() as u64;
            for (i, &p) in pages.iter().enumerate() {
                *args.add(2 + i) = (base + p * PAGE) as u64;
            }
        }
        let cb = queue.command_buffer()?;
        cb.compute(&[
            Compute::Pipeline { pipeline: &pso, threads_per_group: [32, 1, 1] },
            Compute::Buffer { buffer: chunks.table().expect("dense table"), offset: 0, index: 0 },
            Compute::Buffer { buffer: &args_buf, offset: 0, index: 1 },
            Compute::Buffer { buffer: &out_buf, offset: 0, index: 2 },
            Compute::Dispatch { groups: [pages.len().div_ceil(32) as u64, 1, 1] },
        ])?;
        cb.commit();
        cb.wait()?;
        // SAFETY: written by the GPU, which is done.
        Ok((0..pages.len()).map(|i| unsafe { *out.add(i) }).collect())
    };

    // 1. Every page read through the table, then +1 written back.
    let all: Vec<usize> = (0..PAGES).collect();
    let got = probe(&chunks, &all)?;
    let bad = all.iter().filter(|&&p| got[p] != marker(p)).count();
    // SAFETY: mapped.
    let back = all.iter().filter(|&&p| unsafe { *word(base + p * PAGE) } != marker(p) + 1).count();
    for &p in &[0usize, 1, 16, 40, 63] {
        println!("   page {p}: va {:#x} table->gpu {:#x} got {:#x} want {:#x}", base + p * PAGE, chunks.gpu_address((base + p * PAGE) as u64), got[p], marker(p));
    }
    println!("1. read {} pages through the table: {bad} wrong; GPU writes seen by the CPU: {} wrong", PAGES, back);
    if bad + back != 0 {
        return Err("step 1".into());
    }

    // 2. Remap page 3 to spare section page PAGES+2 (xmem's XMemMapPhysicalPages).
    // SAFETY: probe waited, dropped its command buffer, and no other queue
    // or retained view uses these chunks.
    unsafe { chunks.invalidate_quiesced((base + 3 * PAGE) as u64, PAGE as u64) };
    unmap(base + 3 * PAGE);
    map(s, PAGES + 2, 1, base + 3 * PAGE)?;
    chunks.wrap((base + 3 * PAGE) as u64, PAGE as u64);
    let got = probe(&chunks, &[3, 4])?;
    println!("2. after remapping page 3: read {:#x} (want {:#x}), page 4 {:#x} (want {:#x}); {:?}", got[0], marker(PAGES + 2), got[1], marker(4) + 1, chunks.stats);
    if got[0] != marker(PAGES + 2) || got[1] != marker(4) + 1 {
        return Err("step 2".into());
    }

    // 3. Unmap page 5 (a hole): the chunk falls back to per-page buffers.
    // SAFETY: same quiescence as step 2.
    unsafe { chunks.invalidate_quiesced((base + 5 * PAGE) as u64, PAGE as u64) };
    unmap(base + 5 * PAGE);
    chunks.wrap((base + 5 * PAGE) as u64, PAGE as u64);
    let got = probe(&chunks, &[5, 6])?;
    println!("3. after unmapping page 5: page 5 {:#x} (want 0xdead), page 6 {:#x} (want {:#x}); {:?}", got[0], got[1], marker(6) + 1, chunks.stats);
    if got[0] != 0xDEAD || got[1] != marker(6) + 1 {
        return Err("step 3".into());
    }

    // 5. A window across a chunk boundary: one contiguous GPU range.
    let boundary = (chunks.base + (2 << 20)) as usize; // the end of the first chunk
    let words_at = boundary - 8;
    for k in 0..4u32 {
        // SAFETY: mapped (pages around the boundary are pages 15/16 region).
        unsafe { *((words_at + k as usize * 4) as *mut u32) = 0xA000 + k };
    }
    // SAFETY: the previous probe is complete and its pool drained. Retire
    // both chunks before making a span over any of the same pages.
    unsafe { chunks.invalidate_quiesced(words_at as u64, 16) };
    let w = chunks.window(words_at as u64, 16);
    if w == 0 {
        return Err("no cross-chunk window".into());
    }
    let raw = device.compute_pipeline_msl(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         kernel void r(device const ulong *args [[buffer(0)]], device uint *out [[buffer(1)]], uint i [[thread_position_in_grid]]) {\n\
           device const uint *p = (device const uint *)args[0]; out[i] = p[i];\n\
         }\n",
        "r",
    )?;
    // SAFETY: our scratch memory, the GPU is idle.
    unsafe { *args = w };
    let cb = queue.command_buffer()?;
    cb.compute(&[
        Compute::Pipeline { pipeline: &raw, threads_per_group: [4, 1, 1] },
        Compute::Buffer { buffer: &args_buf, offset: 0, index: 0 },
        Compute::Buffer { buffer: &out_buf, offset: 0, index: 1 },
        Compute::Dispatch { groups: [1, 1, 1] },
    ])?;
    cb.commit();
    cb.wait()?;
    // SAFETY: written by the GPU, which is done.
    let got: Vec<u32> = (0..4).map(|i| unsafe { *out.add(i) }).collect();
    println!("5. window across the chunk boundary at {boundary:#x}: gpu {w:#x}, read {got:x?} (want [a000, a001, a002, a003]); {:?}", chunks.stats);
    if got != [0xA000, 0xA001, 0xA002, 0xA003] {
        return Err("step 5".into());
    }

    Ok(())
}

#[cfg(not(windows))]
fn main() {
    eprintln!("victory-metal-chunks runs on Windows (Wine) only");
}
