//! Title memory as GPU memory, without copies (DESIGN.md, "Revisions after
//! T3–T5"): a range of the title's address space is wrapped in no-copy
//! buffers, and a table gives the GPU address of every 64 KiB page of it.
//!
//! - Buffers are 2 MiB chunks where the whole chunk is mapped (T4: the
//!   cheapest size per byte), else one buffer per mapped 64 KiB page (xmem
//!   maps physical memory in 64 KiB pages, so a chunk can be partly mapped).
//! - The table has one `u64` per 64 KiB page: the GPU address of the page's
//!   first byte, 0 where nothing is wrapped. Shaders turn a title address
//!   `va` into `table[(va - base) >> 16] + (va & 0xFFFF)`.
//! - A remap (xmem changed which physical pages are behind some addresses)
//!   makes the buffers there stale: a buffer keeps the pages it saw when it
//!   was made. Callers must quiesce, invalidate and release wrappers
//!   before any mapping change, then wrap the new mapping separately.
//!
//! Every live buffer (and the table) is in one residency set ([`Chunks::residency`]),
//! which the queues that run title work must have ([`crate::Queue::add_residency_set`]):
//! shaders reach title memory through addresses, not bindings (T4 §3.6).

use std::collections::BTreeMap;

use core::ffi::c_void;

use crate::{Buffer, Device, Error, ResidencySet, Result};

mod evict;

pub const PAGE: u64 = 64 << 10;
pub const CHUNK: u64 = 2 << 20;
/// The least a new span covers (see [`Chunks::span_ranges`]).
pub const SPAN_MIN: u64 = 16 << 20;

/// Lookups that returned a span a remap made stale (`resident_window`
/// takes `&self`).
pub static STALE_USES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
const PAGES_PER_CHUNK: u64 = CHUNK / PAGE;

/// How one chunk is wrapped.
enum Wrapped {
    Whole(Buffer),
    /// Per mapped page (index within the chunk → buffer).
    Pages(BTreeMap<u64, Buffer>),
}

pub struct Chunks {
    device: Device,
    /// The first title address covered, chunk aligned, and the size.
    pub base: u64,
    pub size: u64,
    chunks: BTreeMap<u64, Wrapped>,
    /// Buffers over ranges that cross chunk (or page buffer) boundaries,
    /// by (start, end): a view must be contiguous in GPU addresses, and
    /// neighbouring chunk buffers are not ([`Chunks::window`]).
    spans: BTreeMap<(u64, u64), Buffer>,
    /// One u64 per page, in memory of our own that the GPU reads through
    /// `table_buffer` (no copy: written in place). Null in sparse mode.
    table: *mut u64,
    pages: usize,
    table_buffer: Option<Buffer>,
    /// Sparse mode ([`Chunks::sparse`]): page → GPU address, CPU side only.
    sparse: Option<std::collections::HashMap<u64, u64>>,
    residency: ResidencySet,
    /// Counters (wraps of whole chunks, of single pages, rewraps).
    pub stats: Stats,
    /// No new span once live spans hold this many bytes: every span wires
    /// its pages for the GPU and joins the one residency set every
    /// submission makes resident.
    pub span_cap: u64,
    /// The longest live span's length (how far back a lookup looks).
    longest_span: u64,
    /// Spans whose pages a remap changed: they still show the old ones.
    stale: std::collections::HashSet<(u64, u64)>,
    /// Where else the bytes of a title range are mapped, never remapped
    /// ([`Alias`]): spans are made there instead.
    pub alias: Option<Alias>,
}

/// The host address of the same bytes as title range `[va, va+len)` in a
/// mapping nobody remaps (xmem's persistent views of its physical-page
/// sections), or `None` to wrap the title addresses themselves. Title
/// views of physical pages are remapped thousands of times a second while
/// streaming: a wrapper over them keeps the old pages (the GPU then reads
/// stale data, and the old pages stay wired), and wrapping the new ones
/// beside it is what panicked macOS (2026-11-26). The section's own view
/// shows every page where it always is.
pub type Alias = fn(u64, u64) -> Option<u64>;

#[derive(Default, Debug, Clone, Copy)]
pub struct Stats {
    pub whole: u64,
    pub pages: u64,
    pub rewrapped_chunks: u64,
    pub spans: u64,
    pub refused: u64,
    /// Bytes under live span buffers (they overlap chunks and each other).
    pub span_bytes: u64,
    /// Spans refused by [`Chunks::span_cap`].
    pub capped: u64,
    /// Remaps that changed pages under a live span ([`Chunks::note_remap`]),
    /// the spans they made stale, and lookups that returned a stale span.
    pub remaps_under_spans: u64,
    pub stale_spans: u64,
    pub stale_uses: u64,
}

/// Whether `[at, at+len)` is committed memory one could wrap.
fn mapped(at: u64, len: u64) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery};
    let mut va = at;
    while va < at + len {
        // SAFETY: plain data filled in by the query.
        let mut m: MEMORY_BASIC_INFORMATION = unsafe { core::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { VirtualQuery(va as *const c_void, &mut m, size_of_val(&m)) } == 0 {
            return false;
        }
        if m.State != MEM_COMMIT || m.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0 || m.Protect == 0 {
            return false;
        }
        va = m.BaseAddress as u64 + m.RegionSize as u64;
    }
    true
}

/// How far committed memory runs on from `va` without a gap, at most `max`
/// bytes (0 if `va` itself is not committed): the extent a root view (an
/// unbounded pointer) is given.
pub fn committed_extent(va: u64, max: u64) -> u64 {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery};
    let mut at = va;
    while at < va + max {
        // SAFETY: plain data filled in by the query.
        let mut m: MEMORY_BASIC_INFORMATION = unsafe { core::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { VirtualQuery(at as *const c_void, &mut m, size_of_val(&m)) } == 0
            || m.State != MEM_COMMIT
            || m.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0
            || m.Protect == 0
        {
            break;
        }
        at = m.BaseAddress as u64 + m.RegionSize as u64;
    }
    at.min(va + max).saturating_sub(va)
}

impl Chunks {
    /// Covers `[base, base+size)` (both rounded out to chunks). Nothing is
    /// wrapped yet: see [`Chunks::wrap`].
    pub fn new(device: &Device, base: u64, size: u64) -> Result<Self> {
        let start = base & !(CHUNK - 1);
        let end = (base + size + CHUNK - 1) & !(CHUNK - 1);
        let pages = ((end - start) / PAGE) as usize;
        // Page-aligned memory of our own (zeroed by VirtualAlloc), never
        // freed: one table per title, for the process' life.
        let bytes = (pages * 8).next_multiple_of(PAGE as usize);
        let table_mem = Self::alloc(bytes)?;
        // SAFETY: freshly committed and never released.
        let table_buffer = unsafe { device.wrap(table_mem, bytes as u64)? };
        let table = table_mem.cast::<u64>();
        let residency = ResidencySet::new(device, 4096)?;
        residency.add(&[table_buffer.handle()]);
        residency.commit();
        let table_buffer = Some(table_buffer);
        Ok(Self {
            device: device.clone(),
            base: start,
            size: end - start,
            chunks: BTreeMap::new(),
            spans: BTreeMap::new(),
            table,
            pages,
            table_buffer,
            sparse: None,
            residency,
            stats: Stats::default(),
            span_cap: u64::MAX,
            longest_span: 0,
            stale: std::collections::HashSet::new(),
            alias: None,
        })
    }

    fn alloc(bytes: usize) -> Result<*mut c_void> {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        // SAFETY: a fresh allocation.
        let p = unsafe { VirtualAlloc(core::ptr::null(), bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        if p.is_null() { Err(Error("VirtualAlloc for the chunk table".into())) } else { Ok(p) }
    }

    /// The whole title address space (47 bits) without a GPU table: the
    /// GPU gets addresses through descriptors and root parameters
    /// ([`Chunks::window`]), which is all Metal Shader Converter's shaders
    /// need; page lookups stay on the CPU.
    pub fn sparse(device: &Device) -> Result<Self> {
        let residency = ResidencySet::new(device, 4096)?;
        Ok(Self {
            device: device.clone(),
            base: 0,
            size: 1 << 47,
            chunks: BTreeMap::new(),
            spans: BTreeMap::new(),
            table: core::ptr::null_mut(),
            pages: ((1u64 << 47) / PAGE) as usize,
            table_buffer: None,
            sparse: Some(std::collections::HashMap::new()),
            residency,
            stats: Stats::default(),
            span_cap: u64::MAX,
            longest_span: 0,
            stale: std::collections::HashSet::new(),
            alias: None,
        })
    }

    /// The table, for shaders (bind it, or pass its GPU address); `None` in
    /// sparse mode.
    pub fn table(&self) -> Option<&Buffer> {
        self.table_buffer.as_ref()
    }

    /// The residency set holding every live buffer: attach it to the queues.
    pub fn residency(&self) -> &ResidencySet {
        &self.residency
    }

    /// The GPU address of title address `va`, 0 if nothing is wrapped there.
    pub fn gpu_address(&self, va: u64) -> u64 {
        if va < self.base || va >= self.base + self.size {
            return 0;
        }
        let page = self.entry((va - self.base) / PAGE);
        if page == 0 { 0 } else { page + (va % PAGE) }
    }

    /// The GPU address of `[va, va+len)` as one contiguous range: inside a
    /// single wrapped buffer it is that buffer's address; across buffers a
    /// span buffer over exactly the range (page aligned) is made once and
    /// kept until a remap there. 0 if the memory is not wrapped/mapped.
    pub fn window(&mut self, va: u64, len: u64) -> u64 {
        self.window_buffer(va, len).map_or(0, |(b, off)| b.gpu_address + off)
    }

    /// `[va, va+len)` where spans are made for it: its [`Alias`], else itself.
    fn home(&self, va: u64, len: u64) -> u64 {
        self.alias.and_then(|f| f(va, len)).unwrap_or(va)
    }

    /// A contiguous range in a buffer already in this residency set.
    pub fn resident_window(&self, va: u64, len: u64) -> Option<(Buffer, u64)> {
        // The same bytes can have a span at either address: a longer view
        // that crosses non-consecutive pages has none at the alias.
        let home = self.home(va, len);
        self.lookup(home, len).or_else(|| if home != va { self.lookup(va, len) } else { None })
    }

    fn lookup(&self, va: u64, len: u64) -> Option<(Buffer, u64)> {
        let end = va.checked_add(len)?;
        if len == 0 || va < self.base || end > self.base + self.size {
            return None;
        }
        let first = (va - self.base) / CHUNK;
        let last = (end - 1 - self.base) / CHUNK;
        if first == last {
            let chunk_va = self.base + first * CHUNK;
            match self.chunks.get(&first) {
                Some(Wrapped::Whole(b)) => return Some((b.clone(), va - chunk_va)),
                Some(Wrapped::Pages(pages)) if (va % PAGE) + len <= PAGE => {
                    let page = (va - chunk_va) / PAGE;
                    if let Some(b) = pages.get(&page) {
                        return Some((b.clone(), va % PAGE));
                    }
                }
                _ => {}
            }
        }
        // Any span holding the range, not only the one starting last
        // before it: spans of every start no further back than the longest.
        let floor = va.saturating_sub(self.longest_span);
        self.spans
            .range((floor, 0)..=(va, u64::MAX))
            .rev()
            .find(|((s, e), _)| *s <= va && end <= *e)
            .map(|(&(s, e), b)| {
                if self.stale.contains(&(s, e)) {
                    STALE_USES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                (b.clone(), va - s)
            })
    }

    /// Notes that the pages of `[at, at+len)` changed (counts only: the
    /// spans over them are left as they are).
    pub fn note_remap(&mut self, at: u64, len: u64) {
        let end = at.saturating_add(len);
        let floor = at.saturating_sub(self.longest_span);
        let hit: Vec<(u64, u64)> = self
            .spans
            .range((floor, 0)..(end, 0))
            .map(|(k, _)| *k)
            .filter(|&(s, e)| s < end && at < e)
            .collect();
        if !hit.is_empty() {
            self.stats.remaps_under_spans += 1;
        }
        for k in hit {
            if self.stale.insert(k) {
                self.stats.stale_spans += 1;
            }
        }
    }

    /// The range a new span over `[va, end)` covers: 2 MiB aligned and at
    /// least [`SPAN_MIN`] where the memory there is committed (candidates, widest first), so that the
    /// views of one region share a buffer (exact ranges made a span per
    /// draw: 14,000 of them before the first 3D frame, each wiring its
    /// pages, 2026-11-27); else the committed run from the page, else the
    /// pages of the range.
    fn span_ranges(va: u64, end: u64) -> Vec<(u64, u64)> {
        let need = end.next_multiple_of(PAGE);
        let page = va & !(PAGE - 1);
        let aligned = va & !(CHUNK - 1);
        let want = need.max(aligned + SPAN_MIN).next_multiple_of(CHUNK);
        let mut out = Vec::new();
        for start in [aligned, page] {
            let run = committed_extent(start, want - start) & !(PAGE - 1);
            if start + run >= need && !out.contains(&(start, start + run)) {
                out.push((start, start + run));
            }
        }
        // Metal can refuse a range over mappings of different kinds: the
        // pages of the range itself come last.
        if !out.contains(&(page, need)) && mapped(page, need - page) {
            out.push((page, need));
        }
        out
    }

    /// The buffer holding `[va, va+len)` contiguously and the range's offset
    /// in it (see [`Chunks::window`]); `None` if not wrapped/mapped.
    pub fn window_buffer(&mut self, va: u64, len: u64) -> Option<(Buffer, u64)> {
        let end = va.checked_add(len)?;
        if len == 0 || va < self.base || end > self.base + self.size {
            return None;
        }
        if let Some(window) = self.resident_window(va, len) {
            return Some(window);
        }
        let va = self.home(va, len);
        let end = va.checked_add(len)?;
        for (start, end) in Self::span_ranges(va, end) {
            if self.stats.span_bytes.saturating_add(end - start) > self.span_cap {
                self.stats.capped += 1;
                return None;
            }
            // SAFETY: committed now; the alias, where there is one, is
            // never remapped.
            match unsafe { self.device.wrap(start as *mut c_void, end - start) } {
                Ok(b) => {
                    self.residency.add(&[b.handle()]);
                    self.residency.commit();
                    self.spans.insert((start, end), b.clone());
                    self.stats.spans += 1;
                    self.stats.span_bytes += end - start;
                    self.longest_span = self.longest_span.max(end - start);
                    return Some((b, va - start));
                }
                Err(_) => self.stats.refused += 1,
            }
        }
        None
    }

    /// Wraps every chunk touching `[at, at+len)` that is not wrapped yet.
    pub fn wrap(&mut self, at: u64, len: u64) {
        for chunk in self.chunk_range(at, len) {
            if !self.chunks.contains_key(&chunk) {
                self.wrap_chunk(chunk);
            }
        }
        self.residency.commit();
    }

    /// Removes the resident wrappers for every touched chunk and clears
    /// their GPU-address entries. Call `wrap` only after the mapping changed.
    ///
    /// # Safety
    /// No submitted or recorded work may still use these addresses, and
    /// callers must release external copies of the buffers and derived
    /// views. Exclude new users through the entire mapping change.
    pub unsafe fn invalidate_quiesced(&mut self, at: u64, len: u64) -> usize {
        let chunks = self.chunk_range(at, len);
        if chunks.is_empty() {
            return 0;
        }
        let lo = self.base + chunks.start * CHUNK;
        let hi = self.base + chunks.end * CHUNK;
        let mut old_buffers = Vec::new();
        let stale: Vec<(u64, u64)> = self.spans.keys().copied().filter(|&(s, e)| s < hi && lo < e).collect();
        for key in stale {
            if let Some(b) = self.spans.remove(&key) {
                self.stats.span_bytes -= key.1 - key.0;
                old_buffers.push(b);
            }
        }
        for chunk in chunks {
            if let Some(old) = self.chunks.remove(&chunk) {
                match old {
                    Wrapped::Whole(b) => old_buffers.push(b),
                    Wrapped::Pages(m) => old_buffers.extend(m.into_values()),
                }
                self.stats.rewrapped_chunks += 1;
            }
            for p in chunk * PAGES_PER_CHUNK..(chunk + 1) * PAGES_PER_CHUNK {
                self.set(p as usize, 0);
            }
        }
        evict::remove_then_release(
            old_buffers,
            |buffers| self.residency.remove(&buffers.iter().map(Buffer::handle).collect::<Vec<_>>()),
            || self.residency.commit(),
        )
    }

    fn chunk_range(&self, at: u64, len: u64) -> std::ops::Range<u64> {
        let lo = at.max(self.base).min(self.base + self.size);
        let hi = at.saturating_add(len).min(self.base + self.size);
        let first = (lo - self.base) / CHUNK;
        let last = if hi > lo { (hi - self.base).div_ceil(CHUNK) } else { first };
        first..last
    }

    fn wrap_chunk(&mut self, chunk: u64) {
        let va = self.base + chunk * CHUNK;
        let first_page = (chunk * PAGES_PER_CHUNK) as usize;
        if mapped(va, CHUNK) {
            // SAFETY: committed now; a later unmap is reported through
            // `remapped` before the GPU can use the stale buffer again.
            match unsafe { self.device.wrap(va as *mut c_void, CHUNK) } {
                Ok(b) => {
                    for p in 0..PAGES_PER_CHUNK as usize {
                        self.set(first_page + p, b.gpu_address + p as u64 * PAGE);
                    }
                    self.residency.add(&[b.handle()]);
                    self.chunks.insert(chunk, Wrapped::Whole(b));
                    self.stats.whole += 1;
                    return;
                }
                Err(_) => self.stats.refused += 1,
            }
        }
        let mut pages = BTreeMap::new();
        for p in 0..PAGES_PER_CHUNK {
            let pva = va + p * PAGE;
            self.set(first_page + p as usize, 0);
            if !mapped(pva, PAGE) {
                continue;
            }
            // SAFETY: as above, for one page.
            match unsafe { self.device.wrap(pva as *mut c_void, PAGE) } {
                Ok(b) => {
                    self.set(first_page + p as usize, b.gpu_address);
                    self.residency.add(&[b.handle()]);
                    pages.insert(p, b);
                    self.stats.pages += 1;
                }
                Err(_) => self.stats.refused += 1,
            }
        }
        self.chunks.insert(chunk, Wrapped::Pages(pages));
    }

    fn entry(&self, page: u64) -> u64 {
        assert!((page as usize) < self.pages);
        if let Some(map) = &self.sparse {
            return map.get(&page).copied().unwrap_or(0);
        }
        // SAFETY: in bounds of the table's memory.
        unsafe { self.table.add(page as usize).read_volatile() }
    }

    /// Writes one table entry. The GPU reads the table while work runs:
    /// entries change only between submissions that use them (the caller
    /// orders a remap before the work that needs the new pages).
    fn set(&mut self, page: usize, value: u64) {
        assert!(page < self.pages);
        if let Some(map) = &mut self.sparse {
            if value == 0 {
                map.remove(&(page as u64));
            } else {
                map.insert(page as u64, value);
            }
            return;
        }
        // SAFETY: in bounds of the table's memory.
        unsafe { self.table.add(page).write_volatile(value) };
    }
}

// The raw table pointer is only touched through &mut self / &self.
unsafe impl Send for Chunks {}

#[cfg(test)]
mod span_tests {
    use super::*;

    /// Views at many offsets and lengths in one region share a few spans,
    /// and each resolves to its own bytes.
    #[test]
    fn views_of_one_region_share_spans() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        let Ok(device) = Device::first() else { return };
        let mut chunks = Chunks::sparse(&device).unwrap();
        let bytes = 64usize << 20;
        // SAFETY: a fresh allocation of our own, never freed (test).
        let base = unsafe { VirtualAlloc(core::ptr::null(), bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as u64;
        assert_ne!(base, 0);
        let mut n = 0;
        for k in 0..2000u64 {
            let va = base + (k * 7919 * 256) % (48 << 20);
            let len = 256 + (k * 104_729) % (4 << 20);
            let (b, off) = chunks.window_buffer(va, len).expect("committed");
            assert!(off + len <= b.len, "the view lies inside its span");
            // The span's first byte is the title address `va - off`.
            assert!(va - off >= base && va - off + b.len <= base + bytes as u64);
            n += 1;
        }
        assert_eq!(n, 2000);
        assert!(chunks.stats.spans <= 16, "spans: {}", chunks.stats.spans);
    }

    static TITLE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static HOST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    /// One title view of one 64 KiB page, aliased at `HOST`.
    fn alias(va: u64, len: u64) -> Option<u64> {
        use std::sync::atomic::Ordering::SeqCst;
        let title = TITLE.load(SeqCst);
        (va >= title && va + len <= title + PAGE).then(|| HOST.load(SeqCst) + (va - title))
    }

    /// The GPU copy of `len` bytes at `offset` in `src` (the destination is
    /// in no residency set: the command buffer makes its bound buffers
    /// resident).
    fn gpu_read(device: &Device, queue: &crate::Queue, src: &Buffer, offset: u64, len: u64) -> Vec<u8> {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        let bytes = len.next_multiple_of(PAGE);
        // SAFETY: a fresh allocation of our own, never freed (test).
        let mem = unsafe { VirtualAlloc(core::ptr::null(), bytes as usize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        assert!(!mem.is_null());
        // SAFETY: committed memory of ours that outlives the copy.
        let dst = unsafe { device.wrap(mem, bytes) }.unwrap();
        let cb = queue.command_buffer().unwrap();
        let mut blit = cb.blit_encoder().unwrap();
        blit.copy_buffer(src, offset, &dst, 0, len);
        blit.finish();
        cb.commit();
        cb.wait().unwrap();
        // SAFETY: a shared buffer of `len` bytes the GPU is done with.
        unsafe { std::slice::from_raw_parts(mem.cast::<u8>(), len as usize).to_vec() }
    }

    /// A title view of a physical page is remapped to another page of the
    /// same section: spans go to the section's own view, so the GPU reads
    /// the page the title shows now, the old span still reads the old page
    /// (its memory never moved), and no wrapper is made over remapped
    /// title addresses.
    #[test]
    fn remapped_title_views_read_through_the_section_view() {
        use std::sync::atomic::Ordering::SeqCst;
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, PAGE_READWRITE, SEC_COMMIT, UnmapViewOfFile};
        let Ok(device) = Device::first() else { return };
        let queue = device.new_queue(4).unwrap();
        let mut chunks = Chunks::sparse(&device).unwrap();
        queue.add_residency_set(chunks.residency());
        chunks.alias = Some(alias);
        let pages = 16u64;
        // SAFETY: a fresh pagefile-backed section and views of it (test).
        let (section, host) = unsafe {
            let h = CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE | SEC_COMMIT, 0, (pages * PAGE) as u32, std::ptr::null());
            assert!(!h.is_null());
            let v = MapViewOfFile(h, FILE_MAP_WRITE, 0, 0, (pages * PAGE) as usize);
            assert!(!v.Value.is_null());
            (h, v.Value as u64)
        };
        let title_view = |page: u64, fill: u8| {
            // SAFETY: a view of a page of our section, written whole.
            unsafe {
                let v = MapViewOfFile(section, FILE_MAP_WRITE, 0, (page * PAGE) as u32, PAGE as usize);
                assert!(!v.Value.is_null());
                std::ptr::write_bytes(v.Value.cast::<u8>(), fill, PAGE as usize);
                TITLE.store(v.Value as u64, SeqCst);
                HOST.store(host + page * PAGE, SeqCst);
                v
            }
        };
        let first = title_view(2, 0xA1);
        let (old, old_off) = chunks.window_buffer(TITLE.load(SeqCst) + 256, 4096).expect("aliased");
        assert!(gpu_read(&device, &queue, &old, old_off, 4096).iter().all(|&b| b == 0xA1));
        // The title remaps its address to page 5.
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(first) };
        let second = title_view(5, 0xB2);
        let (now, now_off) = chunks.window_buffer(TITLE.load(SeqCst) + 256, 4096).expect("aliased");
        assert!(gpu_read(&device, &queue, &now, now_off, 4096).iter().all(|&b| b == 0xB2));
        assert!(gpu_read(&device, &queue, &old, old_off, 4096).iter().all(|&b| b == 0xA1));
        assert_eq!(chunks.stats.spans, 1, "both pages are in the section view's one span");
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(second) };
    }
}
