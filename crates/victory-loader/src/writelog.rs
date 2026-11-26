//! A ring of the writes we make into title memory from outside the title's
//! own code (file reads, DirectStorage destinations, ...) and of the memory
//! we hand out and take back, so that a crash on corrupted title data can
//! list what touched that memory last (the crash report does, see
//! `crash::report`). Lock-free and fixed in size: noting costs a few
//! atomic stores.
//!
//! `VICTORY_WRITE_LOG=0` turns it off.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// What made an entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Source {
    ReadFile = 1,
    Fread = 2,
    DStorageMemory = 3,
    DStoragePages = 4,
    XMemVirtualAlloc = 5,
    XMemVirtualFree = 6,
    XMemAlloc = 7,
    XMemFree = 8,
    PagesWritten = 9,
    Other = 10,
    BottomOfPipe = 11,
}

impl Source {
    fn name(v: u8) -> &'static str {
        match v {
            1 => "ReadFile",
            2 => "fread",
            3 => "dstorage-memory",
            4 => "dstorage-pages",
            5 => "XMemVirtualAlloc",
            6 => "XMemVirtualFree",
            7 => "XMemAlloc",
            8 => "XMemFree",
            9 => "pages-written",
            11 => "bottom-of-pipe",
            _ => "other",
        }
    }
}

const N: usize = 1 << 16;

struct Slot {
    /// Sequence number + 1 (0: never written), written last.
    seq: AtomicU64,
    address: AtomicU64,
    /// Length (low 48 bits), source (next 8).
    len_source: AtomicU64,
    /// Thread id (low 32 bits), milliseconds since the first note (high 32).
    thread_time: AtomicU64,
}

static SLOTS: [Slot; N] = [const {
    Slot { seq: AtomicU64::new(0), address: AtomicU64::new(0), len_source: AtomicU64::new(0), thread_time: AtomicU64::new(0) }
}; N];
static NEXT: AtomicU64 = AtomicU64::new(0);

fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WRITE_LOG").is_none_or(|v| v != "0"))
}

fn now_ms() -> u64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

fn thread_id() -> u64 {
    #[cfg(windows)]
    {
        // SAFETY: no arguments.
        u64::from(unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() })
    }
    #[cfg(not(windows))]
    {
        0
    }
}

/// Notes `[address, address + len)` written (or handed out / taken back).
pub fn note(address: usize, len: usize, source: Source) {
    if address == 0 || !on() {
        return;
    }
    let seq = NEXT.fetch_add(1, Relaxed);
    let slot = &SLOTS[seq as usize % N];
    slot.seq.store(0, Relaxed);
    slot.address.store(address as u64, Relaxed);
    slot.len_source.store((len as u64 & 0xffff_ffff_ffff) | (source as u64) << 48, Relaxed);
    slot.thread_time.store(thread_id() | now_ms() << 32, Relaxed);
    slot.seq.store(seq + 1, std::sync::atomic::Ordering::Release);
}

/// The entries overlapping `[lo, hi)`, newest first, at most `max`: one
/// line each.
pub fn near(lo: u64, hi: u64, max: usize) -> Vec<String> {
    let end = NEXT.load(Relaxed);
    let now = now_ms();
    let mut out = Vec::new();
    let first = end.saturating_sub(N as u64);
    for seq in (first..end).rev() {
        let slot = &SLOTS[seq as usize % N];
        if slot.seq.load(std::sync::atomic::Ordering::Acquire) != seq + 1 {
            continue;
        }
        let address = slot.address.load(Relaxed);
        let ls = slot.len_source.load(Relaxed);
        let len = ls & 0xffff_ffff_ffff;
        if address >= hi || address.saturating_add(len.max(1)) <= lo {
            continue;
        }
        let tt = slot.thread_time.load(Relaxed);
        out.push(format!(
            "{} {:#x}+{:#x} thread {} {} ms ago ({} notes ago)",
            Source::name((ls >> 48) as u8),
            address,
            len,
            tt & 0xffff_ffff,
            now.saturating_sub(tt >> 32),
            end - seq - 1
        ));
        if out.len() >= max {
            break;
        }
    }
    out
}

/// Notes taken so far.
pub fn total() -> u64 {
    NEXT.load(Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_overlapping_entries_newest_first() {
        note(0x1_0000_0000, 0x100, Source::ReadFile);
        note(0x2_0000_0000, 0x100, Source::Fread);
        note(0x1_0000_0080, 0x10, Source::DStorageMemory);
        let hits = near(0x1_0000_0000, 0x1_0000_4000, 10);
        assert!(hits.len() >= 2);
        assert!(hits[0].starts_with("dstorage-memory 0x100000080+0x10"));
        assert!(hits[1].starts_with("ReadFile 0x100000000+0x100"));
        assert!(near(0x3_0000_0000, 0x3_0000_1000, 10).is_empty());
    }
}
