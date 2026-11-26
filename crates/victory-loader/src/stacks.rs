//! Where every thread is: `VICTORY_STACKS=<seconds>[,<seconds>...]` prints
//! each thread's unwound stack as `module+rva` that many seconds after
//! [`dump_stacks_on_schedule`], to find what a title waits on.

use std::ffi::c_void;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use windows_sys::Wdk::System::Threading::{
    NtQueryInformationThread, ThreadBasicInformation, ThreadQuerySetWin32StartAddress,
};
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE, LocalFree};
use windows_sys::Win32::System::Diagnostics::Debug::{
    CONTEXT, CONTEXT_FULL_AMD64, GetThreadContext,
};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
};
use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
use windows_sys::Win32::System::Threading::{
    GetCurrentProcessId, GetCurrentThreadId, GetThreadDescription, OpenThread, ResumeThread,
    SuspendThread, THREAD_GET_CONTEXT, THREAD_QUERY_INFORMATION, THREAD_SUSPEND_RESUME,
};

use crate::crash::{Aligned16, locate, unwind_into};

/// Starts the diagnostics-only present watchdog. `0` disables it. On Wine,
/// suspension is opt-in because the host can be stopped holding its VM lock.
pub fn watch_present_stalls(dir: PathBuf, present: fn() -> u64, locks: fn() -> Vec<(&'static str, usize)>) {
    let threshold = match std::env::var("VICTORY_STALL_DUMP_S") {
        Ok(value) => match value.parse::<u64>() {
            Ok(0) => return,
            Ok(seconds) => Duration::from_secs(seconds),
            Err(_) => {
                eprintln!("VIctory: VICTORY_STALL_DUMP_S must be an integer number of seconds");
                return;
            }
        },
        Err(_) if on_wine() => return,
        Err(_) => Duration::from_secs(30),
    };
    let _ = std::thread::Builder::new().name("present-watchdog".into()).spawn(move || {
        let mut stall = Stall::new(present(), Instant::now(), threshold);
        let mut serial = 0;
        loop {
            std::thread::sleep(threshold.min(Duration::from_secs(1)));
            let frame = present();
            if !stall.poll(frame, Instant::now()) {
                continue;
            }
            loop {
                serial += 1;
                let path = dir.join(format!("stall-{serial}.txt"));
                let mut file = match std::fs::File::create_new(&path) {
                    Ok(file) => file,
                    Err(err) if err.kind() == io::ErrorKind::AlreadyExists => continue,
                    Err(err) => {
                        eprintln!("VIctory: {}: {err}", path.display());
                        break;
                    }
                };
                let result = writeln!(file, "VIctory: present stalled at frame {frame}")
                    .and_then(|_| write_stacks(&mut file, &locks()));
                if let Err(err) = result {
                    eprintln!("VIctory: {}: {err}", path.display());
                }
                break;
            }
        }
    });
}

fn on_wine() -> bool {
    // SAFETY: ntdll stays loaded; Wine supplies this marker export.
    unsafe {
        let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
        !ntdll.is_null() && GetProcAddress(ntdll, c"wine_get_version".as_ptr().cast()).is_some()
    }
}

struct Stall {
    frame: u64,
    since: Instant,
    threshold: Duration,
    dumped: bool,
}

impl Stall {
    fn new(frame: u64, since: Instant, threshold: Duration) -> Self {
        Self { frame, since, threshold, dumped: false }
    }

    fn poll(&mut self, frame: u64, now: Instant) -> bool {
        if self.frame != frame {
            *self = Self::new(frame, now, self.threshold);
        }
        if self.frame == 0 || self.dumped || now.duration_since(self.since) < self.threshold {
            return false;
        }
        self.dumped = true;
        true
    }
}

/// Reads `VICTORY_STACKS` and, if set, starts a thread that prints every
/// thread's stack at each listed time (seconds from now).
pub fn dump_stacks_on_schedule() {
    let Ok(list) = std::env::var("VICTORY_STACKS") else {
        return;
    };
    let mut times: Vec<f64> = list
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    times.sort_by(f64::total_cmp);
    let start = std::time::Instant::now();
    std::thread::spawn(move || {
        for t in times {
            let at = std::time::Duration::from_secs_f64(t.max(0.0));
            std::thread::sleep(at.saturating_sub(start.elapsed()));
            dump_stacks();
        }
    });
}

/// `THREAD_BASIC_INFORMATION` (ntddk).
#[repr(C)]
struct BasicInfo {
    exit_status: i32,
    teb: *const usize,
    client_id: [usize; 2],
    affinity: usize,
    priority: i32,
    base_priority: i32,
}

/// Prints every other thread's stack to stderr. Each thread is suspended
/// only while its context is read and unwound into a buffer allocated
/// beforehand (it may hold the heap lock).
pub fn dump_stacks() {
    let _ = write_stacks(&mut io::stderr(), &[]);
}

fn write_stacks(out: &mut impl Write, locks: &[(&str, usize)]) -> io::Result<()> {
    // SAFETY: plain queries and a snapshot of this process's threads.
    let (me, process, snapshot) = unsafe {
        (
            GetCurrentThreadId(),
            GetCurrentProcessId(),
            CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0),
        )
    };
    if snapshot == INVALID_HANDLE_VALUE {
        return Err(io::Error::last_os_error());
    }
    let mut ids = Vec::new();
    // SAFETY: zeroed is a valid THREADENTRY32; its size is set before use.
    let mut entry: THREADENTRY32 = unsafe { std::mem::zeroed() };
    entry.dwSize = size_of::<THREADENTRY32>() as u32;
    // SAFETY: `snapshot` is open; `entry` is sized.
    let mut more = unsafe { Thread32First(snapshot, &mut entry) } != 0;
    while more {
        if entry.th32OwnerProcessID == process && entry.th32ThreadID != me {
            ids.push(entry.th32ThreadID);
        }
        // SAFETY: as above.
        more = unsafe { Thread32Next(snapshot, &mut entry) } != 0;
    }
    // SAFETY: we opened it.
    unsafe { CloseHandle(snapshot) };
    writeln!(out, "VIctory: stacks of {} threads", ids.len())?;
    for (name, address) in locks {
        writeln!(out, "  mutex {name} at {address:#x}")?;
    }
    for id in ids {
        // SAFETY: opens a thread of our own process.
        let thread = unsafe {
            OpenThread(
                THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                0,
                id,
            )
        };
        if thread.is_null() {
            continue;
        }
        let (frames, (priority, base)) = frames_of(thread);
        let name = name_of(thread);
        let start = start_of(thread);
        // SAFETY: we opened it.
        unsafe { CloseHandle(thread) };
        writeln!(out, "  thread {id} \"{name}\" start {} priority {priority} (base {base})", locate(start))?;
        if frames.is_empty() {
            writeln!(out, "    stack unavailable")?;
        }
        for (i, pc) in frames.into_iter().enumerate() {
            writeln!(out, "    #{i:<2} {}", locate(pc as usize))?;
        }
    }
    Ok(())
}

/// The thread's frames and its current and base scheduling priority (0-31).
fn frames_of(thread: HANDLE) -> (Vec<u64>, (i32, i32)) {
    let mut frames = Vec::with_capacity(64);
    let mut info: BasicInfo = BasicInfo {
        exit_status: 0,
        teb: std::ptr::null(),
        client_id: [0; 2],
        affinity: 0,
        priority: 0,
        base_priority: 0,
    };
    // SAFETY: writes `info`, whose layout and size are the system's.
    let status = unsafe {
        NtQueryInformationThread(
            thread,
            ThreadBasicInformation,
            (&raw mut info).cast(),
            size_of::<BasicInfo>() as u32,
            std::ptr::null_mut(),
        )
    };
    if status != 0 || info.teb.is_null() {
        return (frames, (0, 0));
    }
    // SAFETY: zeroed is a valid CONTEXT.
    let mut context = Aligned16(unsafe { std::mem::zeroed::<CONTEXT>() });
    context.0.ContextFlags = CONTEXT_FULL_AMD64;
    // SAFETY: suspends, reads and resumes a thread that is not this one;
    // nothing between suspend and resume allocates or takes a lock the
    // thread could hold besides the unwinder's own.
    unsafe {
        if SuspendThread(thread) == u32::MAX {
            return (frames, (info.priority, info.base_priority));
        }
        if GetThreadContext(thread, &mut context.0) != 0 {
            // The suspended thread's NT_TIB has the current stack bounds.
            let (high, low) = (info.teb.add(1).read(), info.teb.add(2).read());
            unwind_into(context.0, low, high, &mut frames);
        }
        ResumeThread(thread);
    }
    (frames, (info.priority, info.base_priority))
}

fn start_of(thread: HANDLE) -> usize {
    let mut start = 0usize;
    // SAFETY: writes one pointer-sized start address.
    unsafe {
        NtQueryInformationThread(
            thread,
            ThreadQuerySetWin32StartAddress,
            (&raw mut start).cast(),
            size_of::<usize>() as u32,
            std::ptr::null_mut(),
        )
    };
    start
}

fn name_of(thread: HANDLE) -> String {
    let mut name = std::ptr::null_mut();
    // SAFETY: the system allocates the string; we free it with LocalFree.
    unsafe {
        if GetThreadDescription(thread, &mut name) < 0 || name.is_null() {
            return String::new();
        }
        let len = (0..).take_while(|&i| *name.add(i) != 0).count();
        let text = String::from_utf16_lossy(std::slice::from_raw_parts(name, len));
        LocalFree(name.cast::<c_void>());
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dumps_once_per_stall_after_first_present() {
        let now = Instant::now();
        let seconds = Duration::from_secs;
        let mut timer = Stall::new(0, now, seconds(30));
        assert!(!timer.poll(0, now + seconds(60)));
        assert!(!timer.poll(1, now + seconds(61)));
        assert!(!timer.poll(1, now + seconds(90)));
        assert!(timer.poll(1, now + seconds(91)));
        assert!(!timer.poll(1, now + seconds(120)));
        assert!(!timer.poll(2, now + seconds(121)));
        assert!(timer.poll(2, now + seconds(151)));
    }

    #[test]
    fn snapshots_two_mutually_blocked_threads() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use windows_sys::Win32::System::Threading::{WaitOnAddress, WakeByAddressAll};

        static LOCK_A: AtomicU32 = AtomicU32::new(1);
        static LOCK_B: AtomicU32 = AtomicU32::new(1);
        let (tx, rx) = std::sync::mpsc::channel();
        let mut threads = Vec::new();
        for (name, other) in [("holders-a-waits-b", &LOCK_B), ("holders-b-waits-a", &LOCK_A)] {
            let tx = tx.clone();
            let thread = std::thread::Builder::new().name(name.into()).spawn(move || {
                // SAFETY: plain query and a WaitOnAddress on a live, aligned word.
                tx.send(unsafe { GetCurrentThreadId() }).unwrap();
                let locked = 1u32;
                unsafe { WaitOnAddress(other.as_ptr().cast(), (&raw const locked).cast(), 4, 5000) };
            }).unwrap();
            threads.push(thread);
        }
        let ids = [rx.recv().unwrap(), rx.recv().unwrap()];
        std::thread::sleep(Duration::from_millis(50));
        let mut dump = Vec::new();
        let outcome = write_stacks(&mut dump, &[
            ("lock-a", LOCK_A.as_ptr() as usize),
            ("lock-b", LOCK_B.as_ptr() as usize),
        ]);
        for word in [&LOCK_A, &LOCK_B] {
            word.store(0, Ordering::Release);
            // SAFETY: wakes any waiter on this word.
            unsafe { WakeByAddressAll(word.as_ptr().cast()) };
        }
        for thread in threads {
            thread.join().unwrap();
        }
        outcome.unwrap();
        let dump = String::from_utf8(dump).unwrap();
        for (id, name) in ids.into_iter().zip(["holders-a-waits-b", "holders-b-waits-a"]) {
            // The two workers may send their IDs in either order.
            assert!(dump.contains(&format!("thread {id} ")), "{dump}");
            let start = dump.find(name).expect(&dump);
            let stack = dump[start..].split("\n  thread ").next().unwrap();
            assert!(stack.contains("#0 ") && stack.contains("#2 "), "{stack}");
        }
        assert!(dump.contains(&format!("mutex lock-a at {:#x}", LOCK_A.as_ptr() as usize)));
        assert!(dump.contains(&format!("mutex lock-b at {:#x}", LOCK_B.as_ptr() as usize)));
    }

    #[test]
    fn another_threads_stack_unwinds_to_its_start() {
        let (tx, rx) = std::sync::mpsc::channel();
        let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
        let parked = std::thread::spawn(move || {
            // SAFETY: plain query.
            tx.send(unsafe { GetCurrentThreadId() }).unwrap();
            stop_rx.recv().ok();
        });
        let id = rx.recv().unwrap();
        // SAFETY: a thread of this process.
        let thread = unsafe {
            OpenThread(
                THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                0,
                id,
            )
        };
        assert!(!thread.is_null());
        let (frames, _) = frames_of(thread);
        // Waiting in the kernel, several frames down from the thread start.
        assert!(frames.len() >= 3, "{frames:?}");
        assert!(start_of(thread) != 0);
        // SAFETY: opened above.
        unsafe { CloseHandle(thread) };
        dump_stacks();
        stop_tx.send(()).unwrap();
        parked.join().unwrap();
    }
}
