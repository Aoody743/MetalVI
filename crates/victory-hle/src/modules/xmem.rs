//! `xmem.dll`: console memory, served from host memory. The GPU side works
//! from CPU addresses (unified memory), so memory types and bandwidth
//! classes change nothing here; what constrains addresses (alignment,
//! 2 MiB pages, address-bit limits) is honoured.
#![allow(non_snake_case)]
// Title code calls these through its import table, passing pointers the real
// API would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::{Layout, alloc, dealloc};
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard};

use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_COMMITMENT_LIMIT, HANDLE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER,
    MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE,
    MEM_RESERVE_PLACEHOLDER, MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3,
    MemExtendedParameterAddressRequirements, PAGE_NOACCESS, PAGE_READWRITE, SEC_COMMIT,
    SEC_RESERVE, UnmapViewOfFile2, VirtualAlloc, VirtualAlloc2, VirtualFree, VirtualQueryEx,
};
use windows_sys::Win32::System::ProcessStatus::{
    K32GetPerformanceInfo, K32GetProcessMemoryInfo, PERFORMANCE_INFORMATION,
    PROCESS_MEMORY_COUNTERS,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;
use windows_sys::core::BOOL;

use crate::{HleModule, hle_export};

const MIB: usize = 1 << 20;

/// Whether we run under Wine (its ntdll exports `wine_get_version`).
pub(crate) fn on_wine() -> bool {
    static WINE: LazyLock<bool> = LazyLock::new(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        // SAFETY: plain lookups in ntdll, always loaded.
        unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            !ntdll.is_null() && GetProcAddress(ntdll, c"wine_get_version".as_ptr().cast()).is_some()
        }
    });
    *WINE
}

/// Standard memory the title has handed to the system.
static TRANSFERRED: AtomicUsize = AtomicUsize::new(0);

// Console additions to the host's flags.
/// 2 MiB pages; the host reads this bit as large pages, which need a
/// privilege.
const PAGES_2M: u32 = 0x2000_0000;
/// With `PAGES_2M`: 64 KiB pages; the host reads it as AWE memory.
const PAGES_64K_BIT: u32 = 0x40_0000;
const PAGES_64K: u32 = PAGES_2M | PAGES_64K_BIT;
/// The host's protection constants and `PAGE_GUARD`. Above them the console
/// adds GPU access bits (0x800-0x40000). Write-combining and uncached are
/// dropped too: our GPU side reads this memory through the CPU.
const HOST_PROTECTION: u32 = 0x1ff;
/// XMem flags of memory both processors reach: all of ours.
const GRAPHICS: u64 = 3;
/// XMem flag bit of 32-bit graphics memory.
const ADDRESS_32BIT: u64 = 4;
/// The host's write watching (not in windows-sys' memory module).
const MEM_WRITE_WATCH: u32 = 0x20_0000;

/// The addresses XMem `flags` allow, first and last: the low 4 GiB for
/// 32-bit memory; else from 4 GiB (kept for 32-bit requests, which the
/// title's big pools would otherwise crowd out) up to 46 down to 40 bits
/// from the field at bit 12, or 40 bits without one: the console keeps
/// title memory there, and its ray-tracing structures must be (
/// the title's BVH builder refuses one above; the title asks for its GPU heap
/// with no field). Decision 0001.
fn address_range(flags: u64) -> (usize, usize) {
    let bits = match (flags & ADDRESS_32BIT, flags >> 12 & 7) {
        (0, 0) => 40,
        (0, field) => 47 - field,
        _ => return (0, u32::MAX as usize),
    };
    (1 << 32, (1 << bits) - 1)
}

pub extern "system" fn XMemVirtualAlloc(
    base: *mut c_void,
    size: usize,
    kind: u32,
    flags: u64,
    protect: u32,
) -> *mut c_void {
    let (lowest, highest) = address_range(flags);
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: lowest as *mut c_void,
        HighestEndingAddress: highest as *mut c_void,
        Alignment: if kind & (PAGES_2M | PAGES_64K_BIT) == PAGES_2M {
            2 * MIB
        } else {
            0
        },
    };
    let mut protect = match protect & HOST_PROTECTION {
        0 => PAGE_NOACCESS,
        p => p,
    };
    // A reservation of 64 KiB pages takes physical pages later
    // (`XMemMapPhysicalPages`); on the host only a placeholder can.
    // shortcut: committing part of such a reservation later would fail
    // (the title's 23 calls never do); replace that part of the
    // placeholder then.
    let placeholder = kind & PAGES_64K == PAGES_64K && kind & MEM_COMMIT == 0;
    if placeholder {
        if kind & MEM_REPLACE_PLACEHOLDER != 0 {
            // Part of a placeholder the title split: it stays one.
            return base;
        }
        protect = PAGE_NOACCESS;
    }
    let mut kind = kind & !(PAGES_2M | PAGES_64K_BIT);
    if placeholder {
        kind |= MEM_RESERVE | MEM_RESERVE_PLACEHOLDER;
    }
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // Address requirements apply to reservations at an address of our choice.
    let constrained = base.is_null();
    if constrained {
        kind |= MEM_RESERVE;
        // The GPU side copies only the pages the title wrote since its last
        // copy (`victory_gpu`, decision 0007); placeholders cannot be
        // watched (their physical-page views are copied whole).
        if kind & MEM_RESERVE_PLACEHOLDER == 0 && victory_gpu::fault_tracking() {
            kind |= MEM_WRITE_WATCH;
        }
    }
    // SAFETY: the title's own arguments, cleaned of console bits; the
    // parameter points at `requirements`, which outlives the call.
    let at = unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            base,
            size,
            kind,
            protect,
            if constrained {
                &mut parameter
            } else {
                std::ptr::null_mut()
            },
            constrained.into(),
        )
    };
    if !at.is_null() {
        victory_gpu::note_memory_changed();
        victory_loader::writelog::note(at as usize, size, victory_loader::writelog::Source::XMemVirtualAlloc);
    }
    if base.is_null() {
        tracing::debug!(
            size,
            kind = format_args!("{kind:#x}"),
            flags = format_args!("{flags:#x}"),
            at = format_args!("{at:p}"),
            "XMemVirtualAlloc"
        );
    }
    at
}

/// `MEMORY_BASIC_INFORMATION` (the console's first 0x30 bytes are the
/// same), then the XMem flags and tag.
#[repr(C)]
pub struct BasicInformation {
    host: MEMORY_BASIC_INFORMATION,
    flags: u64,
    tag_code: u32,
    tag: [u8; 4],
}

// shortcut: every committed page reports graphics memory and no tag, since
// all of it is GPU-reachable here (the title's BVH builder, the
// only caller, wants graphics memory); remember the flags of
// `XMemVirtualAlloc` regions if a title tells CPU-only memory apart.
pub extern "system" fn XMemVirtualQuery(
    process: HANDLE,
    address: *const c_void,
    out: *mut BasicInformation,
) -> BOOL {
    let size = size_of::<MEMORY_BASIC_INFORMATION>();
    // SAFETY: the title passes room for the whole structure; the host
    // fails on a null one.
    unsafe {
        if VirtualQueryEx(process, address, out.cast(), size) == 0 {
            return 0;
        }
        let info = &mut *out;
        info.flags = if info.host.State == MEM_COMMIT {
            GRAPHICS
        } else {
            0
        };
        info.tag_code = 0;
        info.tag = [0; 4];
    }
    1
}

// `XMemAlloc` attributes: alignment as log2 (5 bits at 24, 4 bytes and up
// from 2), then the memory type, then bit 33: a 32-bit address.
const ALIGNMENT_SHIFT: u64 = 24;
const ALLOC_32BIT: u64 = 1 << 33;
/// Our block header: the block's size and alignment, just below the
/// pointer handed out.
const HEADER: usize = 2 * size_of::<usize>();

fn layout(size: usize, attributes: u64) -> Option<Layout> {
    // An alignment of at least 16 leaves room for the header.
    let align = 1usize << (attributes >> ALIGNMENT_SHIFT & 0x1f).max(4);
    Layout::from_size_align(size.checked_add(align)?, align).ok()
}

type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
type FreeRoutine = extern "system" fn(*mut c_void, u64);

/// The allocation and free routines the title's exe exports as
/// `__XMemAllocationHooks__` (the title does): `XMemAlloc` and
/// `XMemFree` call them, for the title's own calls and its middleware's;
/// the `…Default` exports do not.
fn hooks() -> [usize; 2] {
    static AT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let at = match AT.get() {
        Some(&at) => at,
        // Looked up once the title is mapped, which is before it runs.
        None => match crate::runtime::main_image() {
            Some(base) => *AT.get_or_init(|| {
                use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
                // SAFETY: a mapped image and a NUL-terminated name.
                let at = unsafe {
                    GetProcAddress(base as _, c"__XMemAllocationHooks__".as_ptr().cast())
                }
                .map_or(0, |f| f as usize);
                tracing::info!(found = at != 0, "xmem allocation hooks");
                at
            }),
            None => 0,
        },
    };
    if at == 0 {
        return [0; 2];
    }
    // SAFETY: the export is two routine pointers.
    unsafe { (at as *const [usize; 2]).read() }
}

/// The title's general allocator, for middleware too: its hook if it has
/// one, else [`XMemAllocDefault`].
pub extern "system" fn XMemAlloc(size: usize, attributes: u64) -> *mut c_void {
    match hooks()[0] {
        0 => XMemAllocDefault(size, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let alloc: AllocRoutine = unsafe { std::mem::transmute(hook) };
            let at = alloc(size, attributes);
            victory_loader::writelog::note(at as usize, size, victory_loader::writelog::Source::XMemAlloc);
            at
        }
    }
}

/// See [`XMemAlloc`].
pub extern "system" fn XMemFree(address: *mut c_void, attributes: u64) {
    match hooks()[1] {
        0 => XMemFreeDefault(address, attributes),
        hook => {
            victory_loader::writelog::note(address as usize, 1, victory_loader::writelog::Source::XMemFree);
            // SAFETY: the title's own routine of this type.
            let free: FreeRoutine = unsafe { std::mem::transmute(hook) };
            free(address, attributes)
        }
    }
}

/// The system allocator behind [`XMemAlloc`].
pub extern "system" fn XMemAllocDefault(size: usize, attributes: u64) -> *mut c_void {
    if attributes & ALLOC_32BIT != 0 {
        return XMemVirtualAlloc(
            std::ptr::null_mut(),
            size,
            MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | ADDRESS_32BIT,
            PAGE_READWRITE,
        );
    }
    let Some(layout) = layout(size, attributes) else {
        return std::ptr::null_mut();
    };
    // shortcut: the host heap's addresses lie above 2^40, unlike the
    // console's; nothing seen builds ray-tracing structures in this memory
    // (the title uses its GPU heap). Sub-allocate from a
    // reservation below 2^40 if one does.
    // SAFETY: the layout has a non-zero size.
    let block = unsafe { alloc(layout) };
    if block.is_null() {
        return block.cast();
    }
    // SAFETY: the block holds the alignment (at least the header) and then
    // `size` bytes.
    unsafe {
        let user = block.add(layout.align());
        user.sub(HEADER)
            .cast::<[usize; 2]>()
            .write([layout.size(), layout.align()]);
        user.cast()
    }
}

/// Frees what [`XMemAllocDefault`] returned, given the same attributes.
pub extern "system" fn XMemFreeDefault(address: *mut c_void, attributes: u64) {
    if address.is_null() {
        return;
    }
    // SAFETY: the address came from `XMemAlloc` with these attributes: a
    // whole reservation, or a header-prefixed block.
    unsafe {
        if attributes & ALLOC_32BIT != 0 {
            let _changing = victory_gpu::title_memory_remapping();
            victory_loader::writelog::note(address as usize, 1, victory_loader::writelog::Source::XMemVirtualFree);
            VirtualFree(address, 0, MEM_RELEASE);
            victory_gpu::note_memory_changed();
            return;
        }
        let user = address.cast::<u8>();
        let [size, align] = user.sub(HEADER).cast::<[usize; 2]>().read();
        dealloc(
            user.sub(align),
            Layout::from_size_align_unchecked(size, align),
        );
    }
}

/// A retail console has no tools memory.
pub extern "system" fn XMemIsToolsMemAvailable() -> BOOL {
    0
}

// Physical pages: on the console, 64 KiB frames a title maps into its own
// reservations, at more than one address if it likes. Here each allocation
// is a pagefile-backed section, its pages are 64 KiB slices of it, and a
// mapping is a view of consecutive slices in place of a host placeholder.

const PAGE: usize = 64 << 10;

struct Section {
    handle: HANDLE,
    persistent: Option<View>,
    windows: Mutex<Vec<Range<usize>>>,
    window_closed: Condvar,
}

// SAFETY: the handle can be shared and windows serialize overlapping access
// through the persistent virtual address; disjoint ranges can stream in parallel.
unsafe impl Send for Section {}
// SAFETY: as above.
unsafe impl Sync for Section {}

impl Drop for Section {
    fn drop(&mut self) {
        if let Some(view) = &self.persistent {
            victory_gpu::title_memory_persistent_gone(view.at.Value as usize, view.len);
        }
        drop(self.persistent.take());
        // SAFETY: our handle; title views keep the section alive on their own.
        unsafe { CloseHandle(self.handle) };
    }
}

impl Section {
    fn new(handle: HANDLE, size: usize) -> Self {
        static PERSISTENT: LazyLock<bool> = LazyLock::new(|| {
            on_wine() && !std::env::var_os("VICTORY_XMEM_PERSISTENT").is_some_and(|v| v == "0")
        });
        Self {
            handle,
            persistent: if *PERSISTENT {
                View::persistent(handle, size)
            } else {
                None
            },
            windows: Mutex::new(Vec::new()),
            window_closed: Condvar::new(),
        }
    }

    fn open_window(&self, range: &Range<usize>) {
        let windows = self.windows.lock().unwrap_or_else(|e| e.into_inner());
        self.window_closed
            .wait_while(windows, |active| {
                active.iter().any(|r| r.start < range.end && range.start < r.end)
            })
            .unwrap_or_else(|e| e.into_inner())
            .push(range.clone());
    }

    fn close_window(&self, range: &Range<usize>) {
        let mut active = self.windows.lock().unwrap_or_else(|e| e.into_inner());
        let index = active.iter().position(|r| r == range).expect("open physical window");
        active.swap_remove(index);
        drop(active);
        self.window_closed.notify_all();
    }
}

/// Where a page lives, or what a view starts with: a section and a page
/// index in it.
#[derive(Clone)]
struct Slice {
    section: Arc<Section>,
    index: usize,
}

#[derive(Default)]
struct Physical {
    next_page: usize,
    pages: HashMap<usize, Slice>,
    /// Mapped views by address: their length and what they show.
    views: BTreeMap<usize, (usize, Slice)>,
    /// Where each page is shown, by section and index in it.
    shown: HashMap<(usize, usize), Vec<usize>>,
}

impl Slice {
    /// The page `k` pages on, as [`Physical::shown`] names it.
    fn key(&self, k: usize) -> (usize, usize) {
        (Arc::as_ptr(&self.section) as usize, self.index + k)
    }
}

static PHYSICAL: LazyLock<Mutex<Physical>> = LazyLock::new(|| {
    Mutex::new(Physical {
        // Page numbers are opaque to titles; 0 stays unused.
        next_page: 0x10_0000,
        ..Default::default()
    })
});

fn physical() -> MutexGuard<'static, Physical> {
    PHYSICAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// A mapped view as [`physical_alias`] sees it: its length, its section
/// (by address, and a weak reference for [`physical_mirror`]), the index of
/// its first page and the section's persistent view (address and length; 0
/// for none).
#[derive(Clone)]
struct AliasView {
    len: usize,
    section: usize,
    owner: std::sync::Weak<Section>,
    index: usize,
    persistent: usize,
    persistent_len: usize,
}

/// [`Physical::views`] again, behind a lock of its own that is held for
/// nothing but its own update: [`physical_alias`] runs under the GPU side's
/// locks, and the page table's lock is held while waiting for GPU-side
/// readers of title memory (`title_memory_remapping_range`).
static ALIASES: std::sync::RwLock<BTreeMap<usize, AliasView>> = std::sync::RwLock::new(BTreeMap::new());

fn alias_insert(base: usize, len: usize, slice: &Slice) {
    let (persistent, persistent_len) = slice
        .section
        .persistent
        .as_ref()
        .map_or((0, 0), |v| (v.at.Value as usize, v.len));
    let view = AliasView {
        len,
        section: Arc::as_ptr(&slice.section) as usize,
        owner: Arc::downgrade(&slice.section),
        index: slice.index,
        persistent,
        persistent_len,
    };
    ALIASES.write().unwrap_or_else(|e| e.into_inner()).insert(base, view);
}

fn alias_remove(base: usize) {
    ALIASES.write().unwrap_or_else(|e| e.into_inner()).remove(&base);
}

/// Where the bytes of title range `[va, va+len)` lie in their section's
/// persistent view, when the range is in mapped views of consecutive pages
/// of one section that has one: the address the GPU side wraps instead of
/// the title's, which the title remaps while streaming (see
/// `victory_metal::chunks::Alias`). Takes [`ALIASES`] only.
pub fn physical_alias(va: u64, len: u64) -> Option<u64> {
    let (va, len) = (usize::try_from(va).ok()?, usize::try_from(len).ok()?);
    let end = va.checked_add(len)?;
    let views = ALIASES.read().unwrap_or_else(|e| e.into_inner());
    let (&start, first) = views.range(..=va).next_back()?;
    if va >= start + first.len || first.persistent == 0 {
        return None;
    }
    let offset = first.index * PAGE + (va - start);
    // On through views that show the next pages of the same section.
    let (mut covered, mut next) = (start + first.len, first.index + first.len / PAGE);
    while covered < end {
        let (&s, v) = views.range(covered..).next()?;
        if s != covered || v.section != first.section || v.index != next {
            return None;
        }
        covered += v.len;
        next += v.len / PAGE;
    }
    (offset + len <= first.persistent_len).then(|| (first.persistent + offset) as u64)
}

/// The most bytes and pieces one mirror maps ([`physical_mirror`]).
const MIRROR_MAX_BYTES: usize = 64 << 20;
const MIRROR_MAX_PIECES: usize = 1024;

/// Mirrors made, by what they show: runs of consecutive slices as (section
/// address, first slice, slices), in title order. Each keeps the sections
/// it was made of (weakly: a section address can be reused once freed) and
/// its address. Mirrors are never unmapped: the GPU side keeps no-copy
/// buffers over them for the process's life.
struct Mirror {
    at: usize,
    owners: Vec<std::sync::Weak<Section>>,
}

static MIRRORS: LazyLock<Mutex<HashMap<Vec<(usize, usize, usize)>, Mirror>>> = LazyLock::new(Default::default);

/// Mirrors at most (`VICTORY_XMEM_MIRRORS`, default 8192; 0: none, ranges
/// across discontiguous views then get spans at title addresses as before).
fn mirror_cap() -> usize {
    static CAP: LazyLock<usize> = LazyLock::new(|| {
        std::env::var("VICTORY_XMEM_MIRRORS").ok().and_then(|v| v.parse().ok()).unwrap_or(8192)
    });
    *CAP
}

/// Mirrors made, refused at the cap, and failed to map (for the log).
static MIRROR_COUNTS: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];

/// Where the bytes of title range `[va, va+len)` lie in a contiguous view
/// of our own of the same physical pages, when the range runs through
/// mapped views of slices that do not follow on (pages out of order, or of
/// several sections), so that [`physical_alias`] has no answer: the GPU
/// side wraps that view instead of the title addresses, which the title
/// remaps while streaming (a no-copy buffer there keeps the old pages: the
/// GPU then reads stale data, 2026-11-28, `stale_uses` 73–5,506 per run).
///
/// The mirror shows the views that hold the range whole, in title order,
/// and is made once per such list of slices: after the title remaps part
/// of the range, the same call names other slices and gets another mirror.
/// `None` when the range is in one view (or none), crosses a gap, or is
/// larger than [`MIRROR_MAX_BYTES`]. Takes [`ALIASES`] (read), then
/// [`MIRRORS`]; never the page table's lock.
pub fn physical_mirror(va: u64, len: u64) -> Option<u64> {
    let (va, len) = (usize::try_from(va).ok()?, usize::try_from(len).ok()?);
    let end = va.checked_add(len)?;
    if len == 0 || mirror_cap() == 0 {
        return None;
    }
    let (start, runs, owners) = {
        let views = ALIASES.read().unwrap_or_else(|e| e.into_inner());
        let (&start, first) = views.range(..=va).next_back()?;
        if va >= start + first.len {
            return None;
        }
        let mut runs: Vec<(usize, usize, usize)> = Vec::new();
        let mut owners: Vec<std::sync::Weak<Section>> = Vec::new();
        let (mut at, mut view) = (start, first);
        loop {
            let pages = view.len / PAGE;
            match runs.last_mut() {
                // A view whose slices follow on from the last run's: one run.
                Some(last) if last.0 == view.section && last.1 + last.2 == view.index => last.2 += pages,
                _ => {
                    runs.push((view.section, view.index, pages));
                    owners.push(view.owner.clone());
                }
            }
            let covered = at + view.len;
            if covered >= end {
                break;
            }
            if covered - start > MIRROR_MAX_BYTES || runs.len() > MIRROR_MAX_PIECES {
                return None;
            }
            let (&s, v) = views.range(covered..).next()?;
            if s != covered {
                return None; // a gap: not all physical memory
            }
            (at, view) = (s, v);
        }
        // One run: the persistent view's alias (or nothing to mirror).
        if runs.len() < 2 {
            return None;
        }
        (start, runs, owners)
    };
    let total: usize = runs.iter().map(|r| r.2 * PAGE).sum();
    if total > MIRROR_MAX_BYTES {
        return None;
    }
    let mut mirrors = MIRRORS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(m) = mirrors.get(&runs) {
        // The same sections still (an address can be a new section's).
        if m.owners.len() == owners.len()
            && m.owners.iter().zip(&owners).all(|(a, b)| a.ptr_eq(b) && a.strong_count() > 0)
        {
            return Some((m.at + (va - start)) as u64);
        }
        // Left mapped: a buffer may still be over it.
        mirrors.remove(&runs);
    }
    if mirrors.len() >= mirror_cap() {
        let n = MIRROR_COUNTS[1].fetch_add(1, Ordering::Relaxed) + 1;
        if n.is_power_of_two() {
            tracing::warn!(n, cap = mirror_cap(), "xmem: a range across discontiguous views left at title addresses (mirror cap)");
        }
        return None;
    }
    let sections: Option<Vec<Arc<Section>>> = owners.iter().map(std::sync::Weak::upgrade).collect();
    let Some(at) = sections.and_then(|s| map_mirror(&runs, &s, total)) else {
        let n = MIRROR_COUNTS[2].fetch_add(1, Ordering::Relaxed) + 1;
        if n.is_power_of_two() {
            tracing::warn!(n, total, pieces = runs.len(), error = %std::io::Error::last_os_error(), "xmem: mapping a mirror failed");
        }
        return None;
    };
    let n = MIRROR_COUNTS[0].fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 16 || n.is_power_of_two() {
        tracing::info!(n, at = format_args!("{at:#x}"), kb = total >> 10, pieces = runs.len(), "xmem: a mirror of discontiguous physical views");
    }
    mirrors.insert(runs, Mirror { at, owners });
    Some((at + (va - start)) as u64)
}

/// Maps `runs` (slices of `sections`, one each) one after another in a new
/// reservation of `total` bytes above the title's address space. All or
/// nothing.
fn map_mirror(runs: &[(usize, usize, usize)], sections: &[Arc<Section>], total: usize) -> Option<usize> {
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: (1usize << 40) as *mut c_void,
        ..Default::default()
    };
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // SAFETY: a new placeholder of our own; the parameter outlives the call.
    let base = unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            std::ptr::null(),
            total,
            MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
            PAGE_NOACCESS,
            &mut parameter,
            1,
        )
    } as usize;
    if base == 0 {
        return None;
    }
    let mut offset = 0;
    for (k, (&(_, index, pages), section)) in runs.iter().zip(sections).enumerate() {
        let (at, len) = (base + offset, pages * PAGE);
        // SAFETY: splits our placeholder (all but its last piece), then maps
        // the section's slices in place of the piece.
        let view = unsafe {
            if k + 1 < runs.len() {
                VirtualFree(at as *mut c_void, len, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
            }
            MapViewOfFile3(
                section.handle,
                GetCurrentProcess(),
                at as *const c_void,
                (index * PAGE) as u64,
                len,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        if view.Value as usize != at || !commit(at, len) {
            if view.Value as usize == at {
                // SAFETY: the view just mapped, back to a placeholder.
                unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) };
            }
            // SAFETY: the pieces mapped so far, then every placeholder of
            // our reservation (released piece by piece).
            unsafe {
                let mut o = 0;
                for &(_, _, p) in &runs[..k] {
                    UnmapViewOfFile2(GetCurrentProcess(), MEMORY_MAPPED_VIEW_ADDRESS { Value: (base + o) as *mut c_void }, MEM_PRESERVE_PLACEHOLDER);
                    o += p * PAGE;
                }
                let mut o = 0;
                for &(_, _, p) in runs {
                    VirtualFree((base + o) as *mut c_void, 0, MEM_RELEASE);
                    o += p * PAGE;
                }
            }
            return None;
        }
        offset += len;
    }
    Some(base)
}

/// `count` is in and out (pages wanted, pages given); all or nothing here.
pub extern "system" fn XMemAllocatePhysicalPages(
    flags: u64,
    count: *mut usize,
    pages: *mut usize,
) -> BOOL {
    if count.is_null() || pages.is_null() {
        return 0;
    }
    victory_gpu::set_physical_alias(physical_alias);
    victory_gpu::set_physical_mirror(physical_mirror);
    // SAFETY: checked non-null.
    let n = unsafe { *count };
    let Some(size) = n.checked_mul(PAGE).filter(|&s| s > 0) else {
        return 0;
    };
    // A pagefile-backed section of `size` bytes, its pages committed when
    // first mapped or written (`commit`): the title asks for 6.1
    // GiB at once, which a busy host often cannot commit (error 1450).
    // SAFETY: plain section creation.
    let handle = unsafe {
        CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            std::ptr::null(),
            // Under Wine a committed section is a sparse file, free until
            // touched, and committing a reserved one page by page costs a
            // wineserver request each that scans every range committed so
            // far: commit it whole there.
            PAGE_READWRITE | if on_wine() { SEC_COMMIT } else { SEC_RESERVE },
            (size as u64 >> 32) as u32,
            size as u32,
            std::ptr::null(),
        )
    };
    if handle.is_null() {
        let error = std::io::Error::last_os_error();
        tracing::warn!(n, %error, "XMemAllocatePhysicalPages failed");
        // SAFETY: checked non-null.
        unsafe { count.write(0) };
        return 0;
    }
    tracing::debug!(
        n,
        flags = format_args!("{flags:#x}"),
        "XMemAllocatePhysicalPages"
    );
    // For the GPU side's in-place census: only a committed section can be
    // opened as a host heap.
    victory_gpu::title_memory_section(handle as usize, size as u64, on_wine());
    let section = Arc::new(Section::new(handle, size));
    // Its never-remapped view, which the GPU side may wire whole ahead.
    if let Some(view) = &section.persistent {
        victory_gpu::title_memory_persistent(view.at.Value as usize, view.len);
    }
    let mut physical = physical();
    let first = physical.next_page;
    physical.next_page += n * FRAMES;
    for index in 0..n {
        let slice = Slice {
            section: section.clone(),
            index,
        };
        let number = first + index * FRAMES;
        physical.pages.insert(number, slice);
        // SAFETY: the title passes room for `n` page numbers.
        unsafe { pages.add(index).write(number) };
    }
    1
}

/// Page numbers count 4 KiB frames, so a 64 KiB page's is a multiple of
/// 16: the title keeps them as 21-bit `number >> 4` and rebuilds
/// them with `<< 4` (128 GiB of pages at most).
const FRAMES: usize = PAGE >> 12;

/// Pages still mapped stay readable through their views until unmapped.
pub extern "system" fn XMemFreePhysicalPages(n: usize, pages: *const usize) -> BOOL {
    if pages.is_null() {
        return 0;
    }
    let mut released = Vec::new();
    {
        let mut physical = physical();
        for i in 0..n {
            // SAFETY: the title passes `n` page numbers.
            if let Some(slice) = physical.pages.remove(&unsafe { *pages.add(i) })
                && !released.last().is_some_and(|s| Arc::ptr_eq(s, &slice.section))
            {
                released.push(slice.section);
            }
        }
    }
    // A last reference can unmap an entire streaming view.
    drop(released);
    1
}

/// A view of our own on consecutive slices of one section, committed and
/// writable, unmapped when dropped.
struct View {
    at: MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
}

impl View {
    fn persistent(handle: HANDLE, len: usize) -> Option<Self> {
        let mut requirements = MEM_ADDRESS_REQUIREMENTS {
            LowestStartingAddress: (1usize << 40) as *mut c_void,
            ..Default::default()
        };
        let mut parameter = MEM_EXTENDED_PARAMETER::default();
        parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
        parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
        // SAFETY: this Wine section is already SEC_COMMIT. Keep the view
        // out of the low address space used by the title's own reservations.
        let at = unsafe {
            MapViewOfFile3(
                handle, GetCurrentProcess(), std::ptr::null(), 0, len, 0,
                PAGE_READWRITE, &mut parameter, 1,
            )
        };
        if at.Value.is_null() {
            tracing::debug!(len, error = %std::io::Error::last_os_error(), "xmem: using temporary streaming views");
            return None;
        }
        static FIRST: std::sync::Once = std::sync::Once::new();
        FIRST.call_once(|| tracing::info!(len, "xmem: persistent Wine streaming views active"));
        Some(Self { at, len })
    }

    /// Maps `n` slices from `slice` on.
    fn map(slice: &Slice, n: usize) -> Option<View> {
        use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
        let (offset, len) = ((slice.index * PAGE) as u64, n * PAGE);
        // SAFETY: slices of a section we own; the view is ours until dropped.
        let at = unsafe {
            MapViewOfFile(
                slice.section.handle,
                FILE_MAP_WRITE,
                (offset >> 32) as u32,
                offset as u32,
                len,
            )
        };
        if at.Value.is_null() {
            return None;
        }
        let view = View { at, len };
        commit(at.Value as usize, len).then_some(view)
    }

    fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: the view holds `len` committed read-write bytes, ours
        // alone while `self` lives (other views of the slices only alias
        // them, as the console's do).
        unsafe { std::slice::from_raw_parts_mut(self.at.Value.cast(), self.len) }
    }
}

impl Drop for View {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::UnmapViewOfFile;
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(self.at) };
    }
}

/// The pages behind bytes `[offset, offset + len)` of `pages` (64 KiB
/// each), as runs of consecutive slices of one section: each run's first
/// slice and its length in pages. One look at the page table, which the
/// title's own mapping calls share with every DirectStorage worker. `None`
/// if a page is unknown or `pages` ends early.
fn runs(pages: &[usize], offset: usize, len: usize) -> Option<Vec<(Slice, usize)>> {
    let end = offset.checked_add(len)?;
    let pages = pages.get(offset / PAGE..end.div_ceil(PAGE))?;
    let physical = physical();
    let mut runs: Vec<(Slice, usize)> = Vec::new();
    for page in pages {
        let slice = physical.pages.get(page)?;
        match runs.last_mut() {
            Some((first, n))
                if Arc::ptr_eq(&first.section, &slice.section)
                    && slice.index == first.index + *n =>
            {
                *n += 1;
            }
            _ => runs.push((slice.clone(), 1)),
        }
    }
    Some(runs)
}

/// Whether every page under `[offset, offset + len)` is known (a write that
/// fails with them known is a host failure).
pub(crate) fn pages_known(pages: &[usize], offset: usize, len: usize) -> bool {
    runs(pages, offset, len).is_some()
}

/// Copies `data` into physical pages, from `offset` bytes into the first,
/// whether or not they are mapped anywhere (DirectStorage destinations).
/// Reuse the streaming view on Wine, or map one temporary view per run.
/// False if a page is unknown or the host fails.
pub(crate) fn write_physical(pages: &[usize], offset: usize, data: &[u8]) -> bool {
    if data.is_empty() {
        return true;
    }
    let Some(runs) = runs(pages, offset, data.len()) else {
        return false;
    };
    let (mut within, mut done) = (offset % PAGE, 0);
    for (slice, n) in runs {
        let k = (n * PAGE - within).min(data.len() - done);
        let Some(mut window) = PageWindow::new(slice, n, within, k) else {
            return false;
        };
        window.bytes().copy_from_slice(&data[done..done + k]);
        (within, done) = (0, done + k);
    }
    true
}

/// After writes into `n` pages from `slice` on through a view of our own:
/// where the title sees those pages, the GPU side has to copy them again (it
/// does not see such writes).
fn written(slice: &Slice, n: usize) {
    let mut shown: Vec<usize> = {
        let physical = physical();
        (0..n)
            .filter_map(|k| physical.shown.get(&slice.key(k)))
            .flatten()
            .copied()
            .collect()
    };
    shown.sort_unstable();
    for run in shown.chunk_by(|a, b| a + PAGE == *b) {
        victory_gpu::kernel_will_write(run[0], run.len() * PAGE);
    }
}

/// Bytes `[offset, offset + len)` of physical pages, writable in place
/// until dropped: DirectStorage decodes into it instead of a buffer it
/// would then copy.
pub(crate) struct PageWindow {
    temporary: Option<View>,
    range: Range<usize>,
    slice: Slice,
    pages: usize,
}

impl Drop for PageWindow {
    fn drop(&mut self) {
        if self.temporary.is_none() {
            self.slice.section.close_window(&self.range);
        }
        written(&self.slice, self.pages);
    }
}

impl PageWindow {
    fn new(slice: Slice, pages: usize, within: usize, len: usize) -> Option<Self> {
        let (temporary, start) = if slice.section.persistent.is_some() {
            let start = slice.index * PAGE + within;
            slice.section.open_window(&(start..start + len));
            (None, start)
        } else {
            (Some(View::map(&slice, pages)?), within)
        };
        Some(Self { temporary, range: start..start + len, slice, pages })
    }

    pub(crate) fn bytes(&mut self) -> &mut [u8] {
        if let Some(view) = &mut self.temporary {
            return &mut view.bytes()[self.range.clone()];
        }
        let view = self.slice.section.persistent.as_ref().expect("persistent physical view");
        // SAFETY: the section outlives the window; the byte range is committed
        // and excluded from other streaming windows at this virtual address.
        unsafe {
            std::slice::from_raw_parts_mut(
                view.at.Value.cast::<u8>().add(self.range.start),
                self.range.len(),
            )
        }
    }
}

/// A window on bytes `[offset, offset + len)` of `pages` when they are
/// consecutive pages of one allocation (how titles usually hand them out);
/// `None` otherwise (pages of several allocations, out of order, or
/// unknown, or a host failure), which `write_physical` then handles or
/// reports.
pub(crate) fn page_window(pages: &[usize], offset: usize, len: usize) -> Option<PageWindow> {
    if len == 0 {
        return None;
    }
    let mut runs = runs(pages, offset, len)?;
    if runs.len() != 1 {
        return None;
    }
    let (slice, n) = runs.pop()?;
    PageWindow::new(slice, n, offset % PAGE, len)
}

/// Maps `n` pages at `address`, in place of what is mapped there; null
/// `pages` unmaps. Returns `address`, or null if a page or the host fails.
pub extern "system" fn XMemMapPhysicalPages(
    address: *mut c_void,
    n: usize,
    pages: *const usize,
) -> *mut c_void {
    if n == 0 {
        return address;
    }
    let base = address as usize;
    let Some(len) = n.checked_mul(PAGE).filter(|&len| base.checked_add(len).is_some()) else {
        return std::ptr::null_mut();
    };
    let mut physical = physical();
    if !pages.is_null() {
        // SAFETY: the title passes `n` page numbers.
        let mapped = unsafe { std::slice::from_raw_parts(pages, n) };
        if mapped.iter().zip((base..).step_by(PAGE)).all(|(page, at)| {
            physical
                .pages
                .get(page)
                .is_some_and(|slice| shows(&physical, at, slice))
        }) {
            return address;
        }
    }
    // Until the pages are mapped again: `unmap` takes whole views away and
    // maps what stays of them again, and a mapped view is a reservation
    // until it is committed. A read of title memory in place must not fall
    // into that gap (`victory_gpu::title_memory_remapping`).
    let _changing = changing_views(&physical, base, len);
    unmap(&mut physical, base, len);
    if pages.is_null() {
        return address;
    }
    // SAFETY: the title passes `n` page numbers.
    let pages = unsafe { std::slice::from_raw_parts(pages, n) };
    let mut i = 0;
    while i < n {
        let Some(first) = physical.pages.get(&pages[i]).cloned() else {
            tracing::warn!(page = pages[i], "XMemMapPhysicalPages: unknown page");
            return std::ptr::null_mut();
        };
        // One view per run of consecutive slices of one section.
        let run = 1 + pages[i + 1..]
            .iter()
            .zip(1..)
            .take_while(|&(page, k)| {
                physical.pages.get(page).is_some_and(|s| {
                    Arc::ptr_eq(&s.section, &first.section) && s.index == first.index + k
                })
            })
            .count();
        if !map(&mut physical, base + i * PAGE, run * PAGE, first) {
            return std::ptr::null_mut();
        }
        i += run;
    }
    address
}

/// One page at each address; null `pages`, or a page number 0, unmaps
/// (the title passes 0 for addresses to clear). Only adjacent clears are
/// batched: later remapping one page must not unmap its neighbours.
pub extern "system" fn XMemMapPhysicalPagesScatter(
    addresses: *const *mut c_void,
    n: usize,
    pages: *const usize,
) -> BOOL {
    if addresses.is_null() {
        return 0;
    }
    // SAFETY: the title passes `n` addresses and, unless null, `n` pages.
    let addresses = unsafe { std::slice::from_raw_parts(addresses, n) };
    let pages = (!pages.is_null()).then(|| unsafe { std::slice::from_raw_parts(pages, n) });
    let page_at = |i| pages.map_or(0, |p| p[i]);
    let mut physical = physical();
    let mut i = 0;
    while i < n {
        let base = addresses[i] as usize;
        let Some(mut end) = base.checked_add(PAGE).filter(|_| base != 0) else {
            return 0;
        };
        let page = page_at(i);
        let mut run = i + 1;
        while page == 0 && run < n && addresses[run] as usize == end && page_at(run) == 0 {
            let Some(next) = end.checked_add(PAGE) else { break };
            end = next;
            run += 1;
        }
        let len = end - base;
        let _changing = changing_views(&physical, base, len);
        unmap(&mut physical, base, len);
        if page != 0 {
            let Some(slice) = physical.pages.get(&page).cloned() else {
                tracing::warn!(page, "XMemMapPhysicalPagesScatter: unknown page");
                return 0;
            };
            if !map(&mut physical, base, PAGE, slice) {
                return 0;
            }
        }
        i = run;
    }
    1
}

/// Maps the physical pages shown at `[from, from + len)` at `to` as well,
/// in place of what `to` shows; `from` 0 unmaps. The GPU's page-mapping
/// copies (`victory_gpu`, queue `CopyPageMappingsBatchX`):
/// the title maps pages into a pool view, then has the GPU map them into its tiled
/// heap. Page by page; false if a source page is not mapped physical memory.
pub fn map_alias(to: u64, from: u64, len: u64) -> bool {
    let (to, from, len) = (to as usize, from as usize, len as usize);
    if len == 0 {
        return true;
    }
    if to.checked_add(len).is_none() || from.checked_add(len).is_none() {
        return false;
    }
    let mut physical = physical();
    if from != 0
        && (0..len).step_by(PAGE).all(|k| {
            slice_at(&physical, from + k).is_some_and(|slice| shows(&physical, to + k, &slice))
        })
    {
        return true;
    }
    // As in `XMemMapPhysicalPages`: no read in place meets the gap.
    let _changing = changing_views(&physical, to, len);
    unmap(&mut physical, to, len);
    for k in (0..len).step_by(PAGE) {
        if from == 0 {
            break;
        }
        let Some(slice) = slice_at(&physical, from + k) else {
            return false;
        };
        if !map(&mut physical, to + k, PAGE, slice) {
            return false;
        }
    }
    true
}

/// The page a title view shows at `at`, if any.
fn slice_at(physical: &Physical, at: usize) -> Option<Slice> {
    physical
        .views
        .range(..=at)
        .next_back()
        .filter(|(base, (size, _))| at < *base + size)
        .map(|(base, (_, s))| Slice {
            index: s.index + (at - base) / PAGE,
            section: s.section.clone(),
        })
}

/// Whether a title view shows `slice` at `at` already. Mapping it there
/// again would change nothing, and would cost an unmap and a map, which
/// under Wine each hold the process's one virtual-memory lock across a
/// wineserver request: every page fault and protection change on other
/// threads waits for them (#50).
fn shows(physical: &Physical, at: usize, slice: &Slice) -> bool {
    slice_at(physical, at).is_some_and(|s| s.key(0) == slice.key(0))
}

/// A partial unmap temporarily removes both end views in their entirety.
/// Hold their ranges as well as the requested range until they are rebuilt.
fn changing_views(physical: &Physical, base: usize, len: usize) -> victory_gpu::Remapping {
    let end = base + len;
    let containing = |address: usize| {
        physical.views.range(..=address).next_back().and_then(|(&at, &(bytes, _))| {
            let stop = at + bytes;
            (address < stop).then_some((at, stop))
        })
    };
    let (lo, hi) = if len == 0 {
        (base, end)
    } else {
        (
            containing(base).map_or(base, |(at, _)| at),
            containing(end - 1).map_or(end, |(_, stop)| stop),
        )
    };
    victory_gpu::title_memory_remapping_range(lo, hi - lo)
}

/// Maps `slice` and the slices after it over `[base, base + len)`, which
/// must lie in one placeholder.
fn map(physical: &mut Physical, base: usize, len: usize, slice: Slice) -> bool {
    // SAFETY: splits the placeholder around the range (failing harmlessly
    // when the range is the whole placeholder), then maps the section
    // there; both act only on placeholders.
    let view = unsafe {
        VirtualFree(
            base as *mut c_void,
            len,
            MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
        );
        MapViewOfFile3(
            slice.section.handle,
            GetCurrentProcess(),
            base as *const c_void,
            (slice.index * PAGE) as u64,
            len,
            MEM_REPLACE_PLACEHOLDER,
            PAGE_READWRITE,
            std::ptr::null_mut(),
            0,
        )
    };
    if view.Value as usize != base {
        tracing::warn!(
            address = format_args!("{base:#x}"),
            len,
            error = %std::io::Error::last_os_error(),
            "mapping physical pages failed (not a placeholder?)"
        );
        return false;
    }
    if !commit(base, len) {
        // SAFETY: the view just mapped, back to a placeholder.
        unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) };
        return false;
    }
    // New contents, read-write: the GPU side's copies no longer guard it.
    victory_gpu::title_memory_remapped(base, len);
    // What the range shows now, for the GPU side's in-place census.
    victory_gpu::title_memory_view(base, len, slice.section.handle as usize, (slice.index * PAGE) as u64);
    for k in 0..len / PAGE {
        let shown = physical.shown.entry(slice.key(k)).or_default();
        shown.push(base + k * PAGE);
    }
    alias_insert(base, len, &slice);
    physical.views.insert(base, (len, slice));
    true
}

/// Commits the section pages a view shows (once per page, whichever view).
fn commit(base: usize, len: usize) -> bool {
    if on_wine() {
        return true; // sections are committed whole there
    }
    // SAFETY: `[base, base + len)` is a view of a reserved section.
    if !unsafe { VirtualAlloc(base as *const c_void, len, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
        victory_gpu::note_memory_changed();
        return true;
    }
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(ERROR_COMMITMENT_LIMIT as i32) {
        let mut info = PERFORMANCE_INFORMATION {
            cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
            ..Default::default()
        };
        // SAFETY: the buffer and size describe a writable PERFORMANCE_INFORMATION.
        let queried = unsafe { K32GetPerformanceInfo(&mut info, info.cb) } != 0;
        victory_loader::log_crash(&commit_limit_message(len, &error, queried.then_some(&info)));
    } else {
        tracing::warn!(len, %error, "committing physical pages failed");
    }
    false
}

fn commit_limit_message(
    len: usize,
    error: &std::io::Error,
    info: Option<&PERFORMANCE_INFORMATION>,
) -> String {
    let mut message = format!(
        "VIctory: Windows commit limit reached while committing {len} bytes of physical pages: {error}.\n"
    );
    if let Some(info) = info {
        use std::fmt::Write;

        let committed = info.CommitTotal * info.PageSize / MIB;
        let limit = info.CommitLimit * info.PageSize / MIB;
        let _ = writeln!(
            message,
            "System commit: {committed} MiB of {limit} MiB in use."
        );
    }
    message.push_str(
        "Close other applications or increase the Windows paging file size, then retry.\n",
    );
    message
}

/// Returns `[base, base + len)` to placeholders; views reaching past it
/// are mapped again outside it.
fn unmap(physical: &mut Physical, base: usize, len: usize) {
    // No bytes: no view to return. The search below would still find the
    // view holding `base` and cut it in two there, and its range of views
    // starting inside would run backwards (which `BTreeMap` panics on).
    if len == 0 {
        return;
    }
    let end = base + len;
    // Views do not overlap: the one holding `base`, then those starting
    // inside (the GPU's page mappings make one view per page).
    let first = physical
        .views
        .range(..=base)
        .next_back()
        .filter(|(at, (size, _))| *at + size > base)
        .map(|(at, _)| *at);
    let hit: Vec<usize> = first
        .into_iter()
        .chain(physical.views.range(base + 1..end).map(|(at, _)| *at))
        .collect();
    for at in hit {
        let Some((size, slice)) = physical.views.remove(&at) else {
            continue;
        };
        alias_remove(at);
        for k in 0..size / PAGE {
            if let std::collections::hash_map::Entry::Occupied(mut o) =
                physical.shown.entry(slice.key(k))
            {
                o.get_mut().retain(|&shown| shown != at + k * PAGE);
                if o.get().is_empty() {
                    o.remove();
                }
            }
        }
        victory_gpu::title_memory_remapped(at, size);
        // SAFETY: a view we mapped; it becomes a placeholder again.
        unsafe {
            UnmapViewOfFile2(
                GetCurrentProcess(),
                MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: at as *mut c_void,
                },
                MEM_PRESERVE_PLACEHOLDER,
            )
        };
        if at < base {
            map(physical, at, base - at, slice.clone());
        }
        if at + size > end {
            let index = slice.index + (end - at) / PAGE;
            map(physical, end, at + size - end, Slice { index, ..slice });
        }
    }
}

/// The oldest layout we know; newer titles pass other flag values (Minecraft
/// 0x20000001) and read the same leading fields.
#[repr(C)]
#[derive(Debug, Default)]
pub struct WorkingSet {
    pub title_limit: usize,
    pub title_used: usize,
    pub tools_limit: usize,
    pub tools_used: usize,
    pub stack_used: u32,
    pub physical_pages: u32,
    pub gpu_optimal_limit: usize,
    pub gpu_optimal_used: usize,
    pub tools_gpu_optimal_limit: usize,
    pub tools_gpu_optimal_used: usize,
    pub kernel_pool_used: u32,
    pub kernel_pool_limit: u32,
}

/// The title's budget and our process's commit as what it uses. No tools
/// memory: a retail console.
// shortcut: all fields are filled whatever the flags ask for, and use is
// not split by bandwidth class (GPU-optimal use reads 0); track
// allocations per class if a title budgets from that field.
pub extern "system" fn XMemGetWorkingSetStatistics(_which: u32, out: *mut WorkingSet) -> BOOL {
    if out.is_null() {
        return 0;
    }
    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    let (gpu_optimal, standard) = victory_gpu::console::current().title_memory();
    let stats = WorkingSet {
        title_limit: gpu_optimal + standard - TRANSFERRED.load(Ordering::Relaxed),
        title_used: counters.PagefileUsage,
        gpu_optimal_limit: gpu_optimal,
        ..Default::default()
    };
    // What the title is told, logged at the first asking and at every
    // 1,024th: "used" is all the process has committed, the layer's own
    // memory and the card's included (27 GB in run 60's gameplay against a
    // limit of 13.5 GB). A title that budgets its streaming from these two
    // figures believes it is far over its limit.
    // shortcut: the figure is not yet the title's own memory alone (its
    // heaps and the pages it mapped); count those if this line shows the
    // title asking.
    static ASKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let asked = ASKED.fetch_add(1, Ordering::Relaxed);
    if asked.is_multiple_of(1024) {
        tracing::info!(
            asked = asked + 1,
            which = _which,
            limit_mb = stats.title_limit >> 20,
            used_mb = stats.title_used >> 20,
            "the title asks how much memory it uses"
        );
    }
    // SAFETY: checked non-null; the title passes room for the structure.
    unsafe { out.write(stats) };
    1
}

/// Hands title memory (standard, in 2 MiB pages) to the system. The other
/// transfers involve tools memory, which a retail console has none of.
pub extern "system" fn XMemTransferMemory(kind: u32, size: usize) -> BOOL {
    const TITLE_TO_SYSTEM: u32 = 0;
    tracing::info!(kind, size, "XMemTransferMemory");
    if size == 0 {
        return 1;
    }
    let Some(size) = size.checked_next_multiple_of(2 * MIB) else {
        return 0;
    };
    let standard = victory_gpu::console::current().title_memory().1;
    let moved = kind == TITLE_TO_SYSTEM
        && TRANSFERRED
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| {
                t.checked_add(size).filter(|&t| t <= standard)
            })
            .is_ok();
    moved.into()
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "xmem",
        dlls: &["xmem.dll"],
        exports: vec![
            hle_export!(XMemAlloc),
            hle_export!(XMemFree),
            hle_export!(XMemAllocDefault),
            hle_export!(XMemFreeDefault),
            hle_export!(XMemIsToolsMemAvailable),
            hle_export!(XMemAllocatePhysicalPages),
            hle_export!(XMemFreePhysicalPages),
            hle_export!(XMemMapPhysicalPages),
            hle_export!(XMemMapPhysicalPagesScatter),
            hle_export!(XMemVirtualAlloc),
            hle_export!(XMemVirtualQuery),
            hle_export!(XMemGetWorkingSetStatistics),
            hle_export!(XMemTransferMemory),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(address: *const c_void) -> BasicInformation {
        // SAFETY: plain data; zero is a valid value.
        let mut info: BasicInformation = unsafe { std::mem::zeroed() };
        // SAFETY: a pseudo handle needs no closing.
        let process = unsafe { GetCurrentProcess() };
        assert_eq!(XMemVirtualQuery(process, address, &mut info), 1);
        info
    }

    #[test]
    fn alloc_honours_alignment_and_frees() {
        for (field, align) in [(0, 16), (4, 16), (0xc, 4096), (0x10, 64 * 1024)] {
            let attributes = field << ALIGNMENT_SHIFT | 1 << 29; // a graphics type
            let p = XMemAlloc(100, attributes);
            assert!(!p.is_null());
            assert_eq!(p as usize % align, 0, "field {field:#x}");
            // SAFETY: 100 bytes were allocated.
            unsafe { p.cast::<u8>().write_bytes(0xab, 100) };
            XMemFree(p, attributes);
        }
        let p = XMemAlloc(4096, ALLOC_32BIT);
        assert!(!p.is_null() && (p as usize) < 1 << 32);
        XMemFree(p, ALLOC_32BIT);
    }

    #[test]
    fn virtual_alloc_meets_console_constraints() {
        const GPU_READ_WRITE: u32 = 0x2000;
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * MIB,
            PAGES_2M | MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | 7 << 12, // 40-bit addresses
            PAGE_READWRITE | GPU_READ_WRITE,
        );
        assert!(!p.is_null());
        assert_eq!(p as usize % (2 * MIB), 0);
        assert!((p as usize) < 1 << 40);
        let info = query(p);
        assert_eq!(info.host.State, MEM_COMMIT);
        assert_eq!(info.host.Protect, PAGE_READWRITE);
        assert_eq!(info.flags & GRAPHICS, GRAPHICS);
        // Write-watched, for the GPU side's copies.
        let (mut found, mut count, mut granularity) = ([0usize; 4], 4, 0);
        // SAFETY: room for `count` addresses.
        let watched = unsafe {
            windows_sys::Win32::System::Memory::GetWriteWatch(
                0,
                p,
                4 * MIB,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        };
        if victory_gpu::fault_tracking() {
            assert_eq!(watched, 0);
        }
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
        assert_eq!(query(p).flags, 0);
        // No address field: above the 32-bit range and below 2^40, where
        // the console keeps it.
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            16 << 30,
            MEM_RESERVE,
            GRAPHICS,
            PAGE_NOACCESS,
        );
        assert!(!p.is_null() && (p as usize) >= 1 << 32, "{p:p}");
        assert!((p as usize) + (16 << 30) <= 1 << 40, "{p:p}");
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn transfers_shrink_the_title_budget() {
        let mut before = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0x2000_0001, &mut before), 1);
        assert!(before.title_used > 0);
        assert_eq!(XMemTransferMemory(0, 128 * MIB - 1), 1);
        assert_eq!(XMemTransferMemory(2, MIB), 0, "no tools memory");
        let standard = victory_gpu::console::current().title_memory().1;
        assert_eq!(XMemTransferMemory(0, standard), 0, "over budget");
        let mut after = WorkingSet::default();
        XMemGetWorkingSetStatistics(0x2000_0001, &mut after);
        assert_eq!(before.title_limit - after.title_limit, 128 * MIB);
    }

    #[test]
    fn statistics_have_the_console_layout() {
        assert_eq!(size_of::<WorkingSet>(), 0x50);
        assert_eq!(size_of::<BasicInformation>(), 0x40);
        assert_eq!(std::mem::offset_of!(WorkingSet, gpu_optimal_limit), 0x28);
    }

    #[test]
    fn commit_limit_report_includes_request_and_system_commit() {
        let error = std::io::Error::from_raw_os_error(ERROR_COMMITMENT_LIMIT as i32);
        let info = PERFORMANCE_INFORMATION {
            CommitTotal: 512,
            CommitLimit: 768,
            PageSize: 4096,
            ..Default::default()
        };
        let report = commit_limit_message(4 * MIB, &error, Some(&info));
        assert!(report.contains("4194304 bytes"));
        assert!(report.contains("2 MiB of 3 MiB in use"));
        let fallback = commit_limit_message(4 * MIB, &error, None);
        for report in [&report, &fallback] {
            assert!(report.contains("os error 1455"));
            assert!(report.contains("Windows paging file size"));
        }
        assert!(!fallback.contains("System commit:"));
    }

    #[test]
    fn physical_pages_map_alias_and_unmap() {
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        assert_eq!(n, 3);
        // Packed into 21 bits as `number >> 4`, as the title keeps them.
        for p in pages {
            assert_eq!((p >> 4 & 0x1f_ffff) << 4, p);
        }
        // Reservations of 64 KiB pages, as titles make them for mapping.
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);

        // The same page at a second address shows the same bytes.
        assert_eq!(
            XMemMapPhysicalPages(at(b, 1), 1, pages[2..].as_ptr()),
            at(b, 1)
        );
        // SAFETY: both pages are mapped read-write.
        unsafe {
            at(a, 2).cast::<u8>().write(7);
            assert_eq!(at(b, 1).cast::<u8>().read(), 7);
            a.cast::<u8>().write(1);
        }
        // Writes into pages (DirectStorage's) show through their views.
        assert!(write_physical(&pages[2..], 1, &[9]));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { at(b, 1).cast::<u8>().add(1).read() }, 9);

        // Unmapping the middle page keeps its neighbours.
        assert_eq!(
            XMemMapPhysicalPages(at(a, 1), 1, std::ptr::null()),
            at(a, 1)
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(at(a, 2)).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { at(a, 2).cast::<u8>().read() }, 7);

        // A scattered page lands in the hole.
        let addresses = [at(a, 1)];
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, pages.as_ptr()),
            1
        );
        // SAFETY: page 0 is now mapped at both `a` and `a + PAGE`.
        assert_eq!(unsafe { at(a, 1).cast::<u8>().read() }, 1);
        // Page number 0 unmaps.
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, [0usize].as_ptr()),
            1
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        // The GPU's page-mapping copy: what `a + 2 pages` shows, at `b`.
        assert!(map_alias(b as u64, at(a, 2) as u64, PAGE as u64));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { b.cast::<u8>().read() }, 7);
        assert!(map_alias(b as u64, 0, PAGE as u64));
        assert_eq!(query(b).host.State, MEM_RESERVE);
        assert!(!map_alias(b as u64, at(a, 1) as u64, PAGE as u64));
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
        assert_eq!(XMemIsToolsMemAvailable(), 0);
    }

    /// A thread that reads a mapped page in place, as the GPU side reads a
    /// texture, never meets the moment in which the runtime has taken the
    /// page's view away to map it again: mapping the same pages at the same
    /// address unmaps the view and maps a new one, and without the lock the
    /// reader faults on a page "reserved, never committed" (the crashes of
    /// runs 71 and 74).
    #[test]
    fn a_read_in_place_never_meets_the_gap_of_a_remapping() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            3 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        // SAFETY: mapped read-write.
        unsafe { a.cast::<u8>().write(1) };
        let first = a as usize;
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let reader = {
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut reads = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    let _reading = victory_gpu::title_memory_reading_range(first, 1);
                    // SAFETY: the page is mapped whenever no change of
                    // mapping is under way, and none is while `_reading`
                    // is held.
                    assert_eq!(unsafe { (first as *const u8).read_volatile() }, 1);
                    reads += 1;
                }
                reads
            })
        };
        for _ in 0..2000 {
            // The same pages at the same address: the view goes and comes
            // back, with the same bytes.
            assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        }
        stop.store(true, Ordering::Relaxed);
        assert!(reader.join().expect("the reader never faulted") > 0);
        assert_eq!(XMemMapPhysicalPages(a, 3, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn mapping_no_pages_changes_nothing() {
        let pages = allocate(2);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        // SAFETY: both pages are mapped read-write.
        unsafe { a.cast::<u8>().write(5) };
        let second = a.wrapping_byte_add(PAGE);

        // A count of 0 at the start of a view, inside it and past it: with
        // pages, without (an unmap), and as the GPU's page-mapping copy.
        for at in [a, second, a.wrapping_byte_add(2 * PAGE)] {
            assert_eq!(XMemMapPhysicalPages(at, 0, pages.as_ptr()), at);
            assert_eq!(XMemMapPhysicalPages(at, 0, std::ptr::null()), at);
            assert!(map_alias(at as u64, a as u64, 0));
            assert!(map_alias(at as u64, 0, 0));
        }

        // The view is still the one view it was.
        let view = physical().views.get(&(a as usize)).map(|v| v.0);
        assert_eq!(view, Some(2 * PAGE));
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(second).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { a.cast::<u8>().read() }, 5);

        // A count of 1 still unmaps that page alone.
        assert_eq!(XMemMapPhysicalPages(second, 1, std::ptr::null()), second);
        assert_eq!(query(second).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// `n` pages of one new allocation.
    fn allocate(n: usize) -> Vec<usize> {
        let (mut pages, mut given) = (vec![0; n], n);
        assert_eq!(
            XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
            1
        );
        assert_eq!(given, n);
        pages
    }

    /// The bytes of one page, through a view of our own.
    fn page_bytes(page: usize) -> Vec<u8> {
        let slice = physical().pages.get(&page).cloned().expect("known page");
        View::map(&slice, 1).expect("mapped").bytes().to_vec()
    }

    #[test]
    fn runs_group_consecutive_pages_of_one_allocation() {
        let (a, b) = (allocate(3), allocate(1));
        let lengths = |pages: &[usize], offset, len| {
            runs(pages, offset, len).map(|r| r.iter().map(|&(_, n)| n).collect::<Vec<_>>())
        };
        assert_eq!(lengths(&a, 0, 3 * PAGE), Some(vec![3]));
        // Only the pages the bytes reach.
        assert_eq!(lengths(&a, PAGE + 5, 10), Some(vec![1]));
        assert_eq!(lengths(&a, PAGE - 1, 2), Some(vec![2]));
        assert_eq!(lengths(&a, 0, 0), Some(vec![]));
        // Out of order, or across allocations: one run each.
        assert_eq!(
            lengths(&[a[0], a[2], a[1]], 0, 3 * PAGE),
            Some(vec![1, 1, 1])
        );
        assert_eq!(lengths(&[a[1], a[2], b[0]], 0, 3 * PAGE), Some(vec![2, 1]));
        // A slice that follows on in another section starts a run.
        assert_eq!(lengths(&[b[0], a[1]], 0, 2 * PAGE), Some(vec![1, 1]));
        // Too few pages for the bytes, or an unknown one.
        assert_eq!(lengths(&a, 0, 3 * PAGE + 1), None);
        assert_eq!(lengths(&[a[0], 0x10], 0, 2 * PAGE), None);
        assert_eq!(lengths(&a, usize::MAX, 2), None);
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn writes_cross_runs_and_allocations() {
        let (a, b) = (allocate(2), allocate(1));
        let pages = [a[0], a[1], b[0]];
        let data: Vec<u8> = (0..2 * PAGE + 64).map(|i| (i * 7 % 251) as u8).collect();
        let offset = PAGE - 32;
        // Past the third page: the second run is too short.
        assert!(!write_physical(&pages, offset, &data));
        assert!(write_physical(&pages, offset, &data[..2 * PAGE]));
        let seen: Vec<u8> = pages.iter().flat_map(|&p| page_bytes(p)).collect();
        assert!(seen[..offset].iter().all(|&b| b == 0), "before the offset");
        assert_eq!(seen[offset..offset + 2 * PAGE], data[..2 * PAGE]);
        assert!(seen[offset + 2 * PAGE..].iter().all(|&b| b == 0), "after");
        assert!(
            !write_physical(&[a[0], 0x10], 0, &data[..PAGE + 1]),
            "unknown page"
        );
        assert_eq!(XMemFreePhysicalPages(2, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    /// Pages know where they are shown, as views come, are cut and go.
    #[test]
    fn pages_know_where_they_are_shown() {
        let (mut count, mut pages) = (4usize, [0usize; 4]);
        assert_eq!(XMemAllocatePhysicalPages(0, &mut count, pages.as_mut_ptr()), 1);
        let kind = MEM_RESERVE | PAGES_64K;
        let base = XMemVirtualAlloc(std::ptr::null_mut(), 8 * PAGE, kind, 3, PAGE_READWRITE);
        assert!(!base.is_null());
        let at = base as usize;
        let shown = |page: usize| {
            let physical = physical();
            let key = physical.pages[&page].key(0);
            let mut shown = physical.shown.get(&key).cloned().unwrap_or_default();
            shown.sort_unstable();
            shown
        };
        assert_eq!(XMemMapPhysicalPages(base, 4, pages.as_ptr()), base);
        assert_eq!(shown(pages[2]), [at + 2 * PAGE]);
        // Shown twice; the first view cut in two.
        let second = (at + 5 * PAGE) as *mut c_void;
        assert_eq!(XMemMapPhysicalPages(second, 1, &pages[2]), second);
        assert_eq!(shown(pages[2]), [at + 2 * PAGE, at + 5 * PAGE]);
        let gone = (at + 2 * PAGE) as *mut c_void;
        XMemMapPhysicalPages(gone, 1, std::ptr::null());
        assert_eq!(shown(pages[2]), [at + 5 * PAGE]);
        assert_eq!(shown(pages[1]), [at + PAGE]);
        assert_eq!(shown(pages[3]), [at + 3 * PAGE]);
        XMemMapPhysicalPages(base, 8, std::ptr::null());
        assert!(pages.iter().all(|&p| shown(p).is_empty()));
    }

    #[test]
    fn page_windows_cover_one_run_in_place() {
        let (a, b) = (allocate(3), allocate(1));
        let mut window = page_window(&a, PAGE + 3, PAGE).expect("one run");
        assert_eq!(window.bytes().len(), PAGE);
        window.bytes().fill(0x5a);
        drop(window);
        let (second, third) = (page_bytes(a[1]), page_bytes(a[2]));
        assert!(second[..3].iter().all(|&b| b == 0));
        assert!(second[3..].iter().all(|&b| b == 0x5a));
        assert!(third[..3].iter().all(|&b| b == 0x5a));
        assert!(third[3..].iter().all(|&b| b == 0));
        // Several runs, or an unknown page: no window.
        assert!(page_window(&[a[0], b[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[a[1], a[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[0x10], 0, 1).is_none());
        // One page of several is still one run.
        assert!(page_window(&[a[0], b[0]], 0, PAGE).is_some());
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn scatter_clears_neighbours_without_merging_the_kept_views() {
        let pages = allocate(4);
        let base = XMemVirtualAlloc(
            std::ptr::null_mut(), 4 * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE,
        );
        assert!(!base.is_null());
        let addrs = std::array::from_fn::<_, 4, _>(|i| base.wrapping_byte_add(i * PAGE));
        assert_eq!(XMemMapPhysicalPagesScatter(addrs.as_ptr(), 4, pages.as_ptr()), 1);
        {
            let physical = physical();
            for addr in addrs {
                assert_eq!(physical.views.get(&(addr as usize)).unwrap().0, PAGE);
            }
        }
        assert!(write_physical(&pages[3..], 0, &[42]));
        assert_eq!(XMemMapPhysicalPagesScatter(addrs.as_ptr(), 3, std::ptr::null()), 1);
        for addr in &addrs[..3] {
            assert_eq!(query(*addr).host.State, MEM_RESERVE);
        }
        // SAFETY: the fourth mapping is still live and was not part of the clear.
        assert_eq!(unsafe { addrs[3].cast::<u8>().read() }, 42);
        assert_eq!(XMemMapPhysicalPages(base, 4, std::ptr::null()), base);
        // SAFETY: our own reservation is empty.
        assert_ne!(unsafe { VirtualFree(base, 0, MEM_RELEASE) }, 0);
        assert_eq!(XMemFreePhysicalPages(4, pages.as_ptr()), 1);
    }

    /// A title range across views of pages that do not follow on gets a
    /// mirror: one contiguous view of our own of the same pages, in title
    /// order, made once per list of pages; after the title remaps a page of
    /// the range, the same range gets another mirror showing the new page,
    /// and the old mirror still shows the old pages. A range in one view,
    /// or over a hole, gets none.
    #[test]
    fn ranges_across_discontiguous_views_get_a_mirror() {
        let pages = allocate(6);
        let title = XMemVirtualAlloc(std::ptr::null_mut(), 4 * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
        assert!(!title.is_null());
        let at = |page: usize| title.wrapping_byte_add(page * PAGE) as u64;
        // Pages 4, 1, 2 of the allocation at title pages 0..3: two views
        // (4 | 1 2), so no alias in the section view for the whole range.
        assert_eq!(XMemMapPhysicalPages(title, 3, [pages[4], pages[1], pages[2]].as_ptr()), title);
        for (k, fill) in [(0usize, 0x44u8), (1, 0x11), (2, 0x22)] {
            // SAFETY: mapped read-write.
            unsafe { std::ptr::write_bytes(at(k) as *mut u8, fill, PAGE) };
        }
        let (va, len) = (at(0) + 0x100, 2 * PAGE as u64 + 0x200);
        assert_eq!(physical_alias(va, len), None);
        let mirror = physical_mirror(va, len).expect("a mirror");
        assert!(mirror >= 1 << 40, "above the title's address space: {mirror:#x}");
        let bytes = |at: u64, len: u64| {
            // SAFETY: a mirror stays mapped for the process's life.
            unsafe { std::slice::from_raw_parts(at as *const u8, len as usize).to_vec() }
        };
        let shown = bytes(mirror, len);
        assert_eq!(shown, bytes(va, len), "the mirror shows what the title range shows");
        // Made once: the same pages, the same mirror (at their offset).
        assert_eq!(physical_mirror(va, len), Some(mirror));
        assert_eq!(physical_mirror(va + 0x1000, 0x100), None, "inside one view: the alias's case");
        assert_eq!(physical_mirror(at(0) + 0x2000, PAGE as u64), Some(mirror - 0x100 + 0x2000));
        // Writes through the mirror land in the pages.
        // SAFETY: the mirror is mapped read-write.
        unsafe { (mirror as *mut u8).add(PAGE).write(0x5a) };
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { (at(1) as *const u8).add(0x100).read() }, 0x5a);
        // The title remaps its second page to page 5.
        assert_eq!(XMemMapPhysicalPages(at(1) as *mut c_void, 1, [pages[5]].as_ptr()), at(1) as *mut c_void);
        // SAFETY: mapped read-write.
        unsafe { std::ptr::write_bytes(at(1) as *mut u8, 0x55, PAGE) };
        let again = physical_mirror(va, len).expect("a new mirror");
        assert_ne!(again, mirror);
        assert_eq!(bytes(again, len), bytes(va, len));
        assert!(bytes(again + PAGE as u64 - 0x100, 16).iter().all(|&b| b == 0x55));
        assert_eq!(bytes(mirror, len), {
            let mut old = shown.clone();
            old[PAGE] = 0x5a;
            old
        }, "the old mirror keeps the old pages");
        // A hole (title page 3 is not mapped): no mirror.
        assert_eq!(physical_mirror(at(2), 2 * PAGE as u64), None);
        assert_eq!(XMemMapPhysicalPages(title, 3, std::ptr::null()), title);
        assert_eq!(XMemFreePhysicalPages(6, pages.as_ptr()), 1);
    }

    #[test]
    fn streaming_window_reuses_persistent_address() {
        let pages = allocate(2);
        let (first, persistent) = {
            let mut window = page_window(&pages, 4, PAGE).unwrap();
            let persistent = window.slice.section.persistent.as_ref().map(|v| v.at.Value as usize);
            (window.bytes().as_mut_ptr() as usize, persistent)
        };
        if let Some(persistent) = persistent {
            let mut window = page_window(&pages, 8, PAGE).unwrap();
            assert_eq!(first, persistent + 4);
            assert_eq!(window.bytes().as_mut_ptr() as usize, persistent + 8);
        }
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }
}
