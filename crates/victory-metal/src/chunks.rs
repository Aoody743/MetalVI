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
//!   was made (spike `remap.m`). [`Chunks::remapped`] wraps again at once
//!   and updates the table; the old buffers are kept until the GPU work that
//!   may use them is done ([`Chunks::retire`]), then released (T4 §3.5: a new
//!   wrap sees the new pages even while the old buffer lives).
//!
//! Every live buffer (and the table) is in one residency set ([`Chunks::residency`]),
//! which the queues that run title work must have ([`crate::Queue::add_residency_set`]):
//! shaders reach title memory through addresses, not bindings (T4 §3.6).

use std::collections::BTreeMap;

use core::ffi::c_void;

use crate::{Buffer, Device, Error, ResidencySet, Result};

pub const PAGE: u64 = 64 << 10;
pub const CHUNK: u64 = 2 << 20;
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
    /// Buffers replaced by a remap, with the fence value after which no
    /// GPU work can use them any more.
    retired: Vec<(u64, Buffer)>,
    /// Counters (wraps of whole chunks, of single pages, rewraps).
    pub stats: Stats,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct Stats {
    pub whole: u64,
    pub pages: u64,
    pub rewrapped_chunks: u64,
    pub spans: u64,
    pub refused: u64,
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
            retired: Vec::new(),
            stats: Stats::default(),
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
            retired: Vec::new(),
            stats: Stats::default(),
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

    /// The buffer holding `[va, va+len)` contiguously and the range's offset
    /// in it (see [`Chunks::window`]); `None` if not wrapped/mapped.
    pub fn window_buffer(&mut self, va: u64, len: u64) -> Option<(Buffer, u64)> {
        if len == 0 || va < self.base || va + len > self.base + self.size {
            return None;
        }
        let first = (va - self.base) / CHUNK;
        let last = (va + len - 1 - self.base) / CHUNK;
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
        let start = va & !(PAGE - 1);
        let end = (va + len).next_multiple_of(PAGE);
        if let Some((&(s, _), b)) = self.spans.range(..=(start, u64::MAX)).next_back().filter(|((s, e), _)| *s <= start && end <= *e) {
            return Some((b.clone(), va - s));
        }
        if !mapped(start, end - start) {
            return None;
        }
        // SAFETY: committed now; a remap there retires the span first.
        match unsafe { self.device.wrap(start as *mut c_void, end - start) } {
            Ok(b) => {
                self.residency.add(&[b.handle()]);
                self.residency.commit();
                self.spans.insert((start, end), b.clone());
                self.stats.spans += 1;
                Some((b, va - start))
            }
            Err(_) => {
                self.stats.refused += 1;
                None
            }
        }
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

    /// Memory under `[at, at+len)` was remapped: the chunks there are
    /// wrapped again now; their old buffers are retired behind `fence`, the
    /// value the GPU's fence reaches once work submitted so far is done.
    pub fn remapped(&mut self, at: u64, len: u64, fence: u64) {
        let stale: Vec<(u64, u64)> = self.spans.keys().copied().filter(|&(s, e)| s < at + len && at < e).collect();
        for key in stale {
            if let Some(b) = self.spans.remove(&key) {
                self.retired.push((fence, b));
            }
        }
        for chunk in self.chunk_range(at, len) {
            if let Some(old) = self.chunks.remove(&chunk) {
                match old {
                    Wrapped::Whole(b) => self.retired.push((fence, b)),
                    Wrapped::Pages(m) => self.retired.extend(m.into_values().map(|b| (fence, b))),
                }
                self.stats.rewrapped_chunks += 1;
            }
            self.wrap_chunk(chunk);
        }
        self.residency.commit();
    }

    /// Releases retired buffers whose fence value `completed` has reached.
    pub fn retire(&mut self, completed: u64) -> usize {
        let (done, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.retired).into_iter().partition(|(fence, _)| *fence <= completed);
        self.retired = kept;
        if !done.is_empty() {
            let handles: Vec<_> = done.iter().map(|(_, b)| b.handle()).collect();
            self.residency.remove(&handles);
            self.residency.commit();
        }
        done.len()
    }

    pub fn retired_pending(&self) -> usize {
        self.retired.len()
    }

    fn chunk_range(&self, at: u64, len: u64) -> impl Iterator<Item = u64> + use<> {
        let lo = at.max(self.base);
        let hi = (at + len).min(self.base + self.size);
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
