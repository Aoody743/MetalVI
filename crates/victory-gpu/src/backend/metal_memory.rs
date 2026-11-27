//! Title memory by address on Metal (`victory_shader::emit::SLOT_MEMORY`),
//! the contract `crate::memory` keeps for D3D12: a directory with, per
//! 64 KiB page of title address space, the number of the view that holds
//! the page (0: none) and the page's byte offset in it; view `n` at view
//! heap slot `base + n` (slot `base` reads zeros); the pages shaders found
//! empty in the faults buffer, read once a frame.
//!
//! Zero copy: a view is a span over title memory itself
//! (`Chunks::window_buffer`, which wraps xmem's physical pages through
//! their persistent section view), so what a shader stores lands in title
//! memory and nothing is copied either way.
//!
//! What a dispatch names is made ready before it, as `crate::memory` does:
//! the 4 MiB window around each root view, around each pair of root
//! constants that is a title address and around each root constant that is
//! one in 256-byte units, and the window after each.
//! shortcut: what a stage reaches only through pointers it reads out of
//! memory gets its view from the fault list, a frame late.

use super::MetalBackend;
use std::collections::HashMap;
use std::sync::Mutex;
use victory_metal::Buffer;
use victory_metal::ir::DescriptorEntry;
use victory_shader::emit::{MEMORY_FAULTS, MEMORY_PAGES};

/// Views that can be numbered (the run of heap slots is one longer).
pub(crate) const COPIES: u32 = 1024;
/// A window named around an address (as `crate::memory`).
const WINDOW: u64 = 4 << 20;
/// A directory page.
const PAGE: u64 = 1 << 16;

struct View {
    /// Keeps the span alive (it is in the backend's residency set).
    _span: Buffer,
    entry: DescriptorEntry,
}

struct State {
    directory: (Buffer, *mut u32),
    faults: (Buffer, *mut u32),
    /// View `n` is `views[n - 1]`.
    views: Vec<View>,
    /// Per Metal view heap: how many views have their slot written.
    heaps: HashMap<usize, u32>,
    frame: u64,
    refused: u64,
}

// SAFETY: the pointers are into allocations the state owns; all access is
// under the mutex.
unsafe impl Send for State {}

static STATE: Mutex<Option<State>> = Mutex::new(None);

/// What a dispatch binds.
pub(crate) struct Ready {
    pub(crate) directory: u64,
    pub(crate) faults: u64,
    /// The view heap slot of the null view; view `n` is `n` further.
    pub(crate) base: u32,
}

impl MetalBackend {
    /// Zeroed memory of our own wrapped for the GPU and made resident.
    fn own_buffer(&self, bytes: usize) -> Option<(Buffer, *mut u32)> {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        let bytes = bytes.next_multiple_of(64 << 10);
        // SAFETY: a fresh allocation, zeroed, never freed.
        let mem = unsafe { VirtualAlloc(std::ptr::null(), bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        if mem.is_null() {
            return None;
        }
        // SAFETY: as above.
        let buffer = unsafe { self.device.wrap(mem, bytes as u64) }.ok()?;
        self.add_resident(&[buffer.handle()]);
        Some((buffer, mem.cast()))
    }

    /// Before a dispatch whose shader reads title memory by address: views
    /// of what its root arguments name (`addresses`: its root views,
    /// `constants`: its root constants), every numbered view in the run of
    /// slots after the `count` slots of view heap `heap`. `None` if the
    /// backend cannot hold them; the dispatch is then dropped.
    pub(crate) fn memory_ready(&self, heap: usize, count: u32, addresses: &[u64], constants: &[u32]) -> Option<Ready> {
        let mut guard = STATE.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            let directory = self.own_buffer(MEMORY_PAGES as usize * 8)?;
            let faults = self.own_buffer(MEMORY_FAULTS as usize * 4)?;
            *guard = Some(State { directory, faults, views: Vec::new(), heaps: HashMap::new(), frame: u64::MAX, refused: 0 });
        }
        let st = guard.as_mut()?;
        let frame = crate::frame::current();
        if st.frame != frame {
            st.frame = frame;
            self.take_faults(st);
        }
        for &address in addresses {
            self.name(st, address);
        }
        // A title address in two constants (low dword, then bits 32 to
        // 39), or in one in 256-byte units (as `crate::memory`).
        for (i, &c) in constants.iter().enumerate() {
            if let Some(&high) = constants.get(i + 1)
                && (1..=0xFF).contains(&high)
                && c >= 0x1_0000
            {
                self.name(st, u64::from(c) | u64::from(high) << 32);
            }
            if c >= 0x0100_0000 {
                self.name(st, u64::from(c) << 8);
            }
        }
        let made = st.heaps.get(&heap).copied();
        let entries: Vec<DescriptorEntry> = st.views.iter().skip(made.unwrap_or(0) as usize).map(|v| v.entry).collect();
        let base = self.memory_run(heap, count, made.is_none(), made.unwrap_or(0), &entries)?;
        st.heaps.insert(heap, st.views.len() as u32);
        Some(Ready { directory: st.directory.0.gpu_address, faults: st.faults.0.gpu_address, base })
    }

    /// The pages shaders found without a view since the last look: their
    /// views are made now, for the frames to come.
    fn take_faults(&self, st: &mut State) {
        let mut pages = Vec::new();
        for i in 1..MEMORY_FAULTS as usize {
            // SAFETY: inside the faults buffer; the GPU may be writing, a
            // number missed now is written again.
            let at = unsafe { st.faults.1.add(i) };
            // SAFETY: as above.
            let page = unsafe { at.read_volatile() };
            if page != 0 {
                pages.push(page);
                // SAFETY: as above.
                unsafe { at.write_volatile(0) };
            }
        }
        if !pages.is_empty() {
            static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 50 {
                tracing::warn!(
                    pages = pages.len(),
                    first = format_args!("{:#x}", u64::from(pages[0]) * PAGE),
                    views = st.views.len(),
                    "Metal: title memory by address: shaders touched pages without a view (those accesses read 0 or were dropped); views are made now"
                );
            }
        }
        for page in pages {
            self.name(st, u64::from(page) * PAGE);
        }
    }

    /// The window around `address` and the one after it.
    fn name(&self, st: &mut State, address: u64) {
        self.map(st, address);
        self.map(st, (address & !(WINDOW - 1)) + WINDOW);
    }

    /// A view of the committed memory of `address`'s window from its page on
    /// (from the window's start where that is committed too), its pages
    /// entered in the directory where they have none yet.
    fn map(&self, st: &mut State, address: u64) {
        let page = address / PAGE;
        if address == 0 || page >= u64::from(MEMORY_PAGES) - 1 {
            return;
        }
        // SAFETY: page < MEMORY_PAGES, the directory holds 2 dwords each.
        if unsafe { st.directory.1.add(2 * page as usize).read() } != 0 {
            return;
        }
        if st.views.len() as u32 >= COPIES {
            static FULL: std::sync::Once = std::sync::Once::new();
            FULL.call_once(|| tracing::warn!(COPIES, "Metal: title memory by address: every view number is taken; further pages read 0"));
            return;
        }
        let window = address & !(WINDOW - 1);
        let from_page = address & !(PAGE - 1);
        let (start, len) = match victory_metal::chunks::committed_extent(window, WINDOW) {
            0 => (from_page, victory_metal::chunks::committed_extent(from_page, window + WINDOW - from_page)),
            n => (window, n),
        };
        let len = len & !(PAGE - 1);
        if len < PAGE {
            st.refused += 1;
            return;
        }
        let Some((span, offset)) = self.chunks().window_buffer(start, len) else {
            st.refused += 1;
            return;
        };
        let entry = DescriptorEntry::buffer(span.gpu_address + offset, len);
        st.views.push(View { _span: span, entry });
        let n = st.views.len() as u32;
        for p in start / PAGE..(start + len) / PAGE {
            let at = 2 * p as usize;
            // SAFETY: p < MEMORY_PAGES - 1 (checked for the first page; the
            // window is 4 MiB), inside the directory.
            unsafe {
                if p < u64::from(MEMORY_PAGES) && st.directory.1.add(at).read() == 0 {
                    st.directory.1.add(at + 1).write((p * PAGE - start) as u32);
                    st.directory.1.add(at).write(n);
                }
            }
        }
    }
}
