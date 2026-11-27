//! Shader diagnostic counters (`victory_shader::emit::SLOT_DIAG`): one
//! host-visible buffer every translated shader that may need it binds as
//! `u248, space1`. Shaders write it only when something went wrong (a
//! wave64-on-wave32 half-wave sync that timed out, a loop budget that stopped a
//! wave), so the normal path costs nothing. After each command buffer the
//! completion path reads it, reports what it found against the operations of
//! that buffer, and zeroes it. With one buffer in flight
//! (`metal_completion::QUEUE`) the report names the operations exactly.

use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
use victory_shader::emit::{DIAG_DWORDS, DIAG_HALF_SYNC_TIMEOUTS, DIAG_LOOP_BUDGET, DIAG_LOOP_DWORDS, DIAG_LOOPS, DIAG_SITES};

/// The mapping of the buffer, once made.
static PTR: AtomicPtr<u32> = AtomicPtr::new(std::ptr::null_mut());
/// Totals over the run (host waves).
pub(crate) static HALF_SYNC_TIMEOUTS: AtomicU64 = AtomicU64::new(0);
pub(crate) static LOOP_BUDGET_STOPS: AtomicU64 = AtomicU64::new(0);
static BUDGET_REPORTS: AtomicU64 = AtomicU64::new(0);
/// GPU time of the last completed command buffer (for tests).
#[cfg(test)]
pub(crate) static LAST_GPU_NS: AtomicU64 = AtomicU64::new(0);
/// The last non-empty counts reported (for tests).
#[cfg(test)]
pub(crate) static LAST: std::sync::Mutex<Vec<(u32, u32)>> = std::sync::Mutex::new(Vec::new());

/// Makes the buffer on first use: its GPU address (0 when it could not be made).
pub(crate) fn gpu(backend: &super::MetalBackend) -> u64 {
    backend
        .diag
        .get_or_init(|| {
            use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
            let bytes = (DIAG_DWORDS as usize * 4).max(4096);
            // SAFETY: a fresh zeroed page-aligned allocation kept for the run.
            let mem = unsafe { VirtualAlloc(std::ptr::null(), bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
            if mem.is_null() {
                return None;
            }
            // SAFETY: lives for the run (never freed).
            let buffer = unsafe { backend.device.wrap(mem, bytes as u64) }.ok()?;
            backend.add_resident(&[buffer.handle()]);
            PTR.store(mem.cast(), Ordering::Release);
            Some(buffer)
        })
        .as_ref()
        .map_or(0, |b| b.gpu_address)
}

/// What a completed command buffer's shaders counted, as (dword, count),
/// zeroing what it read. Empty when nothing was counted.
pub(crate) fn take() -> Vec<(u32, u32)> {
    let p = PTR.load(Ordering::Acquire);
    if p.is_null() {
        return Vec::new();
    }
    // SAFETY: the mapping lives for the run; the GPU work that writes it has
    // completed (the caller waited for the command buffer). Volatile: the GPU
    // writes it behind the compiler's back.
    let words = unsafe { std::slice::from_raw_parts_mut(p, DIAG_DWORDS as usize) };
    let total = |i: u32| unsafe { std::ptr::read_volatile(&words[i as usize]) };
    if total(DIAG_HALF_SYNC_TIMEOUTS) == 0 && total(DIAG_LOOP_BUDGET) == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (i, w) in words.iter_mut().enumerate() {
        // SAFETY: as above.
        let v = unsafe { std::ptr::read_volatile(w) };
        if v != 0 {
            out.push((i as u32, v));
            unsafe { std::ptr::write_volatile(w, 0) };
        }
    }
    out
}

/// Reports what the shaders of a completed command buffer counted.
/// A half-wave sync timeout is always an error: the two halves of a console
/// wave then exchanged nothing, and what they computed is wrong.
pub(crate) fn report(label: &str, command: usize, describe: impl FnOnce() -> String) {
    let counts = take();
    if counts.is_empty() {
        return;
    }
    #[cfg(test)]
    {
        *LAST.lock().unwrap_or_else(|e| e.into_inner()) = counts.clone();
    }
    let get = |d: u32| counts.iter().find(|c| c.0 == d).map_or(0, |c| c.1);
    let (timeouts, budget) = (get(DIAG_HALF_SYNC_TIMEOUTS), get(DIAG_LOOP_BUDGET));
    let ops = describe();
    if timeouts != 0 {
        let sites: Vec<String> = counts
            .iter()
            .filter(|c| (DIAG_SITES..DIAG_LOOPS).contains(&c.0))
            .map(|&(d, n)| format!("{}:{n}", d - DIAG_SITES))
            .collect();
        let total = HALF_SYNC_TIMEOUTS.fetch_add(u64::from(timeouts), Ordering::Relaxed) + u64::from(timeouts);
        tracing::error!(%label, command, timeouts, total, sites = %sites.join(","), %ops, "Metal: half-wave sync timed out (wave64 halves diverged)");
    }
    if budget != 0 {
        let total = LOOP_BUDGET_STOPS.fetch_add(u64::from(budget), Ordering::Relaxed) + u64::from(budget);
        let n = BUDGET_REPORTS.fetch_add(1, Ordering::Relaxed);
        if n < 64 || n.is_power_of_two() {
            // per loop: index, threads cut, header offset, the header's scalar joins
            let mut loops = Vec::new();
            for &(d, n) in counts.iter().filter(|c| c.0 >= DIAG_LOOPS && (c.0 - DIAG_LOOPS) % DIAG_LOOP_DWORDS == 0) {
                let at = |i: u32| get(d + i);
                let values: Vec<String> = (2..DIAG_LOOP_DWORDS).map(|i| format!("{:#x}", at(i))).collect();
                loops.push(format!("loop{}:{n}@{:#x}[{}]", (d - DIAG_LOOPS) / DIAG_LOOP_DWORDS, at(1), values.join(" ")));
            }
            tracing::warn!(%label, command, threads = budget, total, loops = %loops.join(","), %ops, "Metal: a loop budget cut shader threads");
        }
    }
}
