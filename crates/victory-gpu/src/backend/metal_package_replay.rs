//! Replaying a frame package (F3): title memory put back at its own
//! addresses (no remapping: the memory directory and every title pointer
//! inside buffers need them), heaps and pipelines rebuilt, then each run
//! through [`MetalBackend::run`] in order; the pages the GPU wrote are
//! compared with the package's reference.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use victory_metal::Buffer;
use victory_metal::ir::{DescriptorEntry, TopLevel};
use victory_metal::msc::{RootParam, StaticSampler};

use super::{
    get_params, get_runs, get_samplers, resolve_entry, resolve_top, PAGE, POp, PCompute, Reader, Ref, Resolver, TEntry,
    MEMORY_RECORD, SPECIAL_DIAG, SPECIAL_WAVE_SCRATCH,
};
use crate::backend::metal::{self as metal, MetalBackend, MetalCmd, MetalCompute, MetalDispatchIndirect, MetalOp};

/// One stored page (`memory.idx`, `reference/memory.idx`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct MemRecord {
    pub(crate) va: u64,
    pub(crate) len: u32,
    /// The run before which the content applies (`u32::MAX`: reference).
    pub(crate) epoch: u32,
    pub(crate) offset: u64,
}

pub(crate) enum DefSource {
    Dxil { dxil: Vec<u8>, params: Vec<RootParam>, samplers: Vec<StaticSampler>, vmsc: Option<Vec<u8>> },
    Msl { source: String, function: String },
}

pub(crate) struct PipelineDef {
    pub(crate) id: u32,
    pub(crate) name: String,
    pub(crate) threadgroup: [u64; 3],
    pub(crate) layout: Vec<u8>,
    pub(crate) template: Vec<u8>,
    pub(crate) source: Option<DefSource>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct HeapDef {
    pub(crate) id: u32,
    pub(crate) count: u32,
    pub(crate) capacity: u32,
    pub(crate) samplers: bool,
}

pub(crate) enum HeapUpdate {
    Entries { id: u32, run: u32, entries: Vec<(u32, TEntry)> },
    Strides { id: u32, run: u32, words: Vec<(u32, u32)> },
}

/// A frame package read back.
pub(crate) struct Package {
    pub(crate) dir: PathBuf,
    pub(crate) pipelines: Vec<PipelineDef>,
    pub(crate) runs: Vec<(u32, Vec<POp>)>,
    pub(crate) memory: Vec<MemRecord>,
    pub(crate) reference: Vec<MemRecord>,
    pub(crate) heaps: Vec<HeapDef>,
    pub(crate) updates: Vec<HeapUpdate>,
    pub(crate) samplers: HashMap<u32, [u32; 13]>,
    pub(crate) typed: HashMap<u32, (u64, u64, u32)>,
    pub(crate) textures: HashMap<u32, (u64, [u32; 10], bool)>,
}

fn read(dir: &Path, name: &str) -> Result<Vec<u8>, String> {
    crate::read_file(&dir.join(name)).map_err(|e| format!("{name}: {e}"))
}

fn records(b: &[u8], name: &str) -> Result<Vec<MemRecord>, String> {
    if !b.len().is_multiple_of(MEMORY_RECORD) {
        return Err(format!("{name}: {} bytes is not a whole number of records", b.len()));
    }
    let mut r = Reader::new(b);
    let mut out = Vec::with_capacity(b.len() / MEMORY_RECORD);
    while !r.done() {
        let (va, len, epoch, offset, flags) = (r.u64(), r.u32(), r.u32(), r.u64(), r.u32());
        let (Some(va), Some(len), Some(epoch), Some(offset), Some(flags)) = (va, len, epoch, offset, flags) else {
            return Err(format!("{name}: truncated"));
        };
        if flags != 0 {
            return Err(format!("{name}: flags {flags:#x} (compressed pages) are not read by this build"));
        }
        out.push(MemRecord { va, len, epoch, offset });
    }
    Ok(out)
}

impl Package {
    pub(crate) fn open(dir: &Path) -> Result<Self, String> {
        let runs = get_runs(&read(dir, "ops.bin")?)?;
        let memory = records(&read(dir, "memory.idx")?, "memory.idx")?;
        let reference = records(&read(dir, "reference/memory.idx")?, "reference/memory.idx")?;

        let b = read(dir, "pipelines.bin")?;
        let mut r = Reader::new(&b);
        if r.bytes(8) != Some(b"VPKPIP01".as_slice()) {
            return Err("pipelines.bin: bad magic".into());
        }
        let bad = || "pipelines.bin: truncated".to_string();
        let n = r.u32().ok_or_else(bad)?;
        let mut pipelines = Vec::new();
        for _ in 0..n {
            let id = r.u32().ok_or_else(bad)?;
            let name = r.text().ok_or_else(bad)?;
            let threadgroup = [r.u64().ok_or_else(bad)?, r.u64().ok_or_else(bad)?, r.u64().ok_or_else(bad)?];
            let layout = r.blob().ok_or_else(bad)?.to_vec();
            let template = r.blob().ok_or_else(bad)?.to_vec();
            let source = match r.u8().ok_or_else(bad)? {
                0 => None,
                1 => {
                    let file = r.text().ok_or_else(bad)?;
                    let params = get_params(&mut r).ok_or_else(bad)?;
                    let samplers = get_samplers(&mut r).ok_or_else(bad)?;
                    let vmsc = r.text().ok_or_else(bad)?;
                    let dxil = read(dir, &format!("shaders/{file}"))?;
                    let vmsc = if vmsc.is_empty() { None } else { read(dir, &format!("shaders/{vmsc}")).ok() };
                    Some(DefSource::Dxil { dxil, params, samplers, vmsc })
                }
                2 => Some(DefSource::Msl { source: r.text().ok_or_else(bad)?, function: r.text().ok_or_else(bad)? }),
                k => return Err(format!("pipelines.bin: source kind {k}")),
            };
            pipelines.push(PipelineDef { id, name, threadgroup, layout, template, source });
        }

        let b = read(dir, "heaps.bin")?;
        let mut r = Reader::new(&b);
        if r.bytes(8) != Some(b"VPKHEP01".as_slice()) {
            return Err("heaps.bin: bad magic".into());
        }
        let bad = || "heaps.bin: truncated".to_string();
        let (mut heaps, mut updates) = (Vec::new(), Vec::new());
        while !r.done() {
            match r.u8().ok_or_else(bad)? {
                1 => heaps.push(HeapDef {
                    id: r.u32().ok_or_else(bad)?,
                    count: r.u32().ok_or_else(bad)?,
                    capacity: r.u32().ok_or_else(bad)?,
                    samplers: r.u8().ok_or_else(bad)? != 0,
                }),
                2 => {
                    let (id, run, n) = (r.u32().ok_or_else(bad)?, r.u32().ok_or_else(bad)?, r.u32().ok_or_else(bad)?);
                    let mut entries = Vec::with_capacity(n as usize);
                    for _ in 0..n {
                        entries.push((r.u32().ok_or_else(bad)?, TEntry::get(&mut r).ok_or_else(bad)?));
                    }
                    updates.push(HeapUpdate::Entries { id, run, entries });
                }
                3 => {
                    let (id, run, n) = (r.u32().ok_or_else(bad)?, r.u32().ok_or_else(bad)?, r.u32().ok_or_else(bad)?);
                    let mut words = Vec::with_capacity(n as usize);
                    for _ in 0..n {
                        words.push((r.u32().ok_or_else(bad)?, r.u32().ok_or_else(bad)?));
                    }
                    updates.push(HeapUpdate::Strides { id, run, words });
                }
                t => return Err(format!("heaps.bin: record {t}")),
            }
        }

        let b = read(dir, "resources.bin")?;
        let mut r = Reader::new(&b);
        if r.bytes(8) != Some(b"VPKRES01".as_slice()) {
            return Err("resources.bin: bad magic".into());
        }
        let bad = || "resources.bin: truncated".to_string();
        let (mut samplers, mut typed, mut textures) = (HashMap::new(), HashMap::new(), HashMap::new());
        while !r.done() {
            let tag = r.u8().ok_or_else(bad)?;
            let id = r.u32().ok_or_else(bad)?;
            match tag {
                1 => {
                    let mut w = [0u32; 13];
                    for x in &mut w {
                        *x = r.u32().ok_or_else(bad)?;
                    }
                    samplers.insert(id, w);
                }
                2 => {
                    typed.insert(id, (r.u64().ok_or_else(bad)?, r.u64().ok_or_else(bad)?, r.u32().ok_or_else(bad)?));
                }
                3 => {
                    let resource = r.u64().ok_or_else(bad)?;
                    let mut w = [0u32; 10];
                    for x in &mut w {
                        *x = r.u32().ok_or_else(bad)?;
                    }
                    textures.insert(id, (resource, w, r.u8().ok_or_else(bad)? != 0));
                }
                t => return Err(format!("resources.bin: record {t}")),
            }
        }
        Ok(Self { dir: dir.to_path_buf(), pipelines, runs, memory, reference, heaps, updates, samplers, typed, textures })
    }

    /// The page-aligned ranges a replay needs at their own addresses
    /// (every stored page, merged where they touch).
    pub(crate) fn ranges(&self) -> Vec<(u64, u64)> {
        let mut pages: Vec<u64> = self.memory.iter().chain(&self.reference).map(|m| m.va & !(PAGE - 1)).collect();
        pages.sort_unstable();
        pages.dedup();
        let mut out: Vec<(u64, u64)> = Vec::new();
        for p in pages {
            match out.last_mut() {
                Some((s, n)) if *s + *n == p => *n += PAGE,
                _ => out.push((p, PAGE)),
            }
        }
        out
    }

    fn page_bytes(&self, file: &mut std::fs::File, m: &MemRecord) -> Result<Vec<u8>, String> {
        let mut bytes = vec![0u8; m.len as usize];
        file.seek(SeekFrom::Start(m.offset)).and_then(|_| file.read_exact(&mut bytes)).map_err(|e| format!("page {:#x}: {e}", m.va))?;
        Ok(bytes)
    }
}

/// What is already in this process where the package's memory must go:
/// (start, bytes, what) per occupied region.
pub(crate) fn conflicts(ranges: &[(u64, u64)]) -> Vec<(u64, u64, String)> {
    use windows_sys::Win32::System::Memory::{MEM_FREE, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut out = Vec::new();
    for &(start, len) in ranges {
        let mut at = start;
        while at < start + len {
            // SAFETY: plain data filled in by the query.
            let mut m: MEMORY_BASIC_INFORMATION = unsafe { core::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            if unsafe { VirtualQuery(at as *const core::ffi::c_void, &mut m, size_of_val(&m)) } == 0 {
                out.push((at, start + len - at, "not queryable (outside the user address space?)".into()));
                break;
            }
            let end = (m.BaseAddress as u64 + m.RegionSize as u64).min(start + len);
            if m.State != MEM_FREE {
                out.push((at, end - at, format!("state {:#x} type {:#x} protect {:#x} (allocation at {:#x})", m.State, m.Type, m.Protect, m.AllocationBase as u64)));
            }
            at = end;
        }
    }
    out
}

/// Reserves and commits the package's ranges at their own addresses. Fails,
/// naming every conflicting region, if any of them is taken: nothing is
/// remapped. The memory stays for the life of the process.
pub(crate) fn place(pkg: &Package) -> Result<Vec<(u64, u64)>, String> {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
    let ranges = pkg.ranges();
    let taken = conflicts(&ranges);
    if !taken.is_empty() {
        let list: Vec<String> = taken.iter().map(|(s, n, what)| format!("[{s:#x}, {:#x}) {what}", s + n)).collect();
        return Err(format!("{} title ranges are taken in this process:\n{}", taken.len(), list.join("\n")));
    }
    for &(start, len) in &ranges {
        // SAFETY: the range is free (checked above); a fixed allocation there.
        let p = unsafe { VirtualAlloc(start as *const core::ffi::c_void, len as usize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        if p as u64 != start {
            return Err(format!("VirtualAlloc({start:#x}, {len:#x}) gave {:#x}", p as u64));
        }
    }
    Ok(ranges)
}

/// Writes the pages of `epoch` into place.
fn write_epoch(pkg: &Package, file: &mut std::fs::File, epoch: u32) -> Result<usize, String> {
    let mut n = 0;
    for m in pkg.memory.iter().filter(|m| m.epoch == epoch) {
        let bytes = pkg.page_bytes(file, m)?;
        // SAFETY: placed by `place` (every stored page is in its ranges).
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), m.va as *mut u8, bytes.len()) };
        n += 1;
    }
    Ok(n)
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Options {
    /// Only runs `a..b` go to the GPU (memory and heaps still follow every run).
    pub(crate) runs: Option<(u32, u32)>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Report {
    pub(crate) runs: u32,
    pub(crate) ops_run: u64,
    pub(crate) pipelines_built: u32,
    pub(crate) pipeline_errors: Vec<String>,
    pub(crate) layout_mismatches: u32,
    pub(crate) skipped_unsupported: u64,
    pub(crate) skipped_no_pipeline: u64,
    pub(crate) skipped_memory_directory: u64,
    pub(crate) skipped_unmapped: u64,
    pub(crate) zeroed_unknown: u64,
    pub(crate) zeroed_texture: u64,
    pub(crate) unmapped_title: u64,
    pub(crate) heap_capacity_mismatches: u32,
    pub(crate) pages_written: u64,
    pub(crate) failed_run: Option<u32>,
}

/// Fake title-heap keys for the replayed heaps in `MetalBackend::heaps`.
const HEAP_KEY: usize = 0x7f9a_0000_0000;

struct Replayer<'a> {
    m: &'a MetalBackend,
    pkg: &'a Package,
    heaps: HashMap<u32, (Buffer, Option<Buffer>)>,
    report: Report,
}

impl Resolver for Replayer<'_> {
    fn address(&mut self, r: &Ref) -> u64 {
        match *r {
            Ref::Null | Ref::RingTable | Ref::Template => 0,
            Ref::Title { va, len } => {
                let gpu = self.m.chunks().window(va, u64::from(len.max(1)));
                if gpu == 0 {
                    self.report.unmapped_title += 1;
                }
                gpu
            }
            Ref::Heap { id, offset } => self.heaps.get(&id).map_or(0, |(b, _)| b.gpu_address + offset),
            Ref::HeapStrides { id, offset } => self.heaps.get(&id).and_then(|(_, s)| s.as_ref()).map_or(0, |s| s.gpu_address + offset),
            Ref::Sampler { id } => self.pkg.samplers.get(&id).and_then(|w| self.m.sampler_entry(w)).map_or(0, |e| e.gpu_va),
            Ref::Typed { id } => self.typed(id).map_or(0, |e| e.texture_view_id),
            Ref::Special { kind: SPECIAL_WAVE_SCRATCH, offset } => self.m.wave_scratch_gpu() + offset,
            Ref::Special { kind: SPECIAL_DIAG, offset } => metal::diag::gpu(self.m) + offset,
            Ref::Texture { .. } => {
                self.report.zeroed_texture += 1;
                0
            }
            Ref::Special { .. } | Ref::Unknown { .. } => {
                self.report.zeroed_unknown += 1;
                0
            }
        }
    }

    fn typed(&mut self, id: u32) -> Option<DescriptorEntry> {
        let &(va, bytes, format) = self.pkg.typed.get(&id)?;
        self.m.typed_entry(va, bytes, format)
    }
}

impl Replayer<'_> {
    fn pipeline(&mut self, def: &PipelineDef) -> Option<Arc<MetalCompute>> {
        let made = match def.source.as_ref() {
            None => Err("no source recorded".to_string()),
            Some(DefSource::Dxil { dxil, params, samplers, vmsc }) => {
                let converted = vmsc.as_deref().and_then(victory_metal::msc::Converted::from_bytes);
                let made = match converted {
                    Some(c) => self.m.device.compute_pipeline_converted(&c).map(|p| (p, c)),
                    None => self.m.device.compute_pipeline_dxil(dxil, params, samplers),
                };
                made.map_err(|e| e.to_string()).and_then(|(pso, mut c)| {
                    self.m.bind_static_samplers(&mut c.top_level, params.len(), samplers)?;
                    Ok(MetalCompute { cost_ns: Default::default(), name: def.name.clone(), pso, threadgroup: c.threadgroup.map(u64::from), top_level: c.top_level })
                })
            }
            Some(DefSource::Msl { source, function }) => {
                self.m.device.compute_pipeline_msl(source, function).map_err(|e| e.to_string()).and_then(|pso| {
                    let (mut top, _) = TopLevel::from_layout_bytes(&def.layout).ok_or("bad layout")?;
                    if top.bytes.len() == def.template.len() {
                        top.bytes.clone_from(&def.template);
                    }
                    Ok(MetalCompute { cost_ns: Default::default(), name: def.name.clone(), pso, threadgroup: def.threadgroup, top_level: top })
                })
            }
        };
        match made {
            Ok(p) => {
                if p.top_level.layout_bytes() != def.layout {
                    self.report.layout_mismatches += 1;
                }
                self.report.pipelines_built += 1;
                Some(Arc::new(p))
            }
            Err(e) => {
                self.report.pipeline_errors.push(format!("{} ({}): {e}", def.id, def.name));
                None
            }
        }
    }

    fn heap_buffer(&self, r: Ref) -> Option<Buffer> {
        match r {
            Ref::Heap { id, .. } => self.heaps.get(&id).map(|(b, _)| b.clone()),
            _ => None,
        }
    }

    fn apply_updates(&mut self, run: u32) {
        let pkg = self.pkg;
        let resolved: Vec<(u32, Vec<(u32, DescriptorEntry)>, Vec<(u32, u32)>)> = pkg
            .updates
            .iter()
            .filter_map(|u| match u {
                HeapUpdate::Entries { id, run: r, entries } if *r == run => {
                    Some((*id, entries.iter().map(|(i, e)| (*i, resolve_entry(self, e))).collect(), Vec::new()))
                }
                HeapUpdate::Strides { id, run: r, words } if *r == run => Some((*id, Vec::new(), words.clone())),
                _ => None,
            })
            .collect();
        let heaps = self.m.heaps.lock().unwrap_or_else(|e| e.into_inner());
        for (id, entries, words) in resolved {
            let Some(h) = heaps.get(&(HEAP_KEY + id as usize)) else { continue };
            for (i, e) in entries {
                if i < h.capacity {
                    // SAFETY: inside the heap's `capacity` entries; the GPU is idle
                    // between runs (each run is waited for).
                    unsafe { h.entries.add(i as usize).write(e) };
                }
            }
            if let Some((_, ptr)) = h.strides {
                for (i, w) in words {
                    if i < victory_shader::emit::SLOT_STRIDES_DWORDS {
                        // SAFETY: inside the stride table (`new_heap`).
                        unsafe { ptr.add(i as usize).write(w) };
                    }
                }
            }
        }
    }

    fn compute(&mut self, c: &PCompute, pipelines: &[Option<Arc<MetalCompute>>]) -> Option<(Arc<MetalCompute>, Vec<u8>, Vec<(u32, DescriptorEntry)>)> {
        let Some(pso) = pipelines.get(c.pipeline as usize).cloned().flatten() else {
            self.report.skipped_no_pipeline += 1;
            return None;
        };
        if c.top.refs.iter().any(|(_, r)| matches!(r, Ref::Special { kind: super::SPECIAL_MEMORY_DIRECTORY | super::SPECIAL_FAULTS, .. })) {
            self.report.skipped_memory_directory += 1;
            return None;
        }
        let top = resolve_top(self, &c.top, &pso.top_level.bytes);
        let tables = c.tables.iter().map(|(at, e)| (*at, resolve_entry(self, e))).collect();
        Some((pso, top, tables))
    }

    fn op(&mut self, op: &POp, pipelines: &[Option<Arc<MetalCompute>>]) -> Option<MetalOp> {
        match op {
            POp::Dispatch { c, groups, after_barrier, loop_counts } => {
                let (pso, top, tables) = self.compute(c, pipelines)?;
                let loop_counts = loop_counts.and_then(|vas| {
                    let mut chunks = self.m.chunks();
                    let mut at = |va: u64| chunks.window_buffer(va, 4).map(|(b, o)| (va, b, o));
                    Some([at(vas[0])?, at(vas[1])?])
                });
                Some(MetalOp::Dispatch(MetalCmd {
                    pso,
                    heap: self.heap_buffer(c.heap),
                    sampler_heap: self.heap_buffer(c.sampler_heap),
                    top,
                    groups: *groups,
                    tables,
                    loop_counts,
                    after_barrier: *after_barrier,
                }))
            }
            POp::DispatchIndirect { c, records, stride, max, count, records_va, count_va, group_base_at } => {
                let window = |r: &Ref, m: &MetalBackend| match *r {
                    Ref::Title { va, len } => m.chunks().window_buffer(va, u64::from(len.max(1))),
                    _ => None,
                };
                let Some((records, records_offset)) = window(records, self.m) else {
                    self.report.skipped_unmapped += 1;
                    return None;
                };
                let count = match count {
                    Some(r) => match window(r, self.m) {
                        Some(w) => Some(w),
                        None => {
                            self.report.skipped_unmapped += 1;
                            return None;
                        }
                    },
                    None => None,
                };
                let (pso, top, tables) = self.compute(c, pipelines)?;
                Some(MetalOp::DispatchIndirect(MetalDispatchIndirect {
                    pso,
                    heap: self.heap_buffer(c.heap),
                    sampler_heap: self.heap_buffer(c.sampler_heap),
                    top,
                    tables,
                    records,
                    records_offset,
                    stride: *stride,
                    max: *max,
                    count,
                    records_va: *records_va,
                    count_va: *count_va,
                    group_base_at: group_base_at.map(|v| v as usize),
                }))
            }
            POp::BufferCopy { dst, src, bytes } => Some(MetalOp::BufferCopy { dst: *dst as usize, src: *src as usize, bytes: *bytes as usize }),
            POp::Unsupported { .. } => {
                self.report.skipped_unsupported += 1;
                None
            }
        }
    }
}

/// Replays `pkg` on `m`; its memory must have been placed ([`place`]).
pub(crate) fn replay(m: &MetalBackend, pkg: &Package, opts: &Options) -> Result<Report, String> {
    let _pool = victory_metal::Pool::new();
    let mut r = Replayer { m, pkg, heaps: HashMap::new(), report: Report::default() };
    let mut pipelines: Vec<Option<Arc<MetalCompute>>> = Vec::new();
    for def in &pkg.pipelines {
        let p = r.pipeline(def);
        let at = def.id as usize;
        if pipelines.len() <= at {
            pipelines.resize(at + 1, None);
        }
        pipelines[at] = p;
    }
    for def in &pkg.heaps {
        let h = m.new_heap(def.count, def.samplers).ok_or_else(|| format!("heap {}: could not be made", def.id))?;
        if h.capacity != def.capacity {
            r.report.heap_capacity_mismatches += 1;
        }
        r.heaps.insert(def.id, (h.buffer.clone(), h.strides.as_ref().map(|(b, _)| b.clone())));
        m.heaps.lock().unwrap_or_else(|e| e.into_inner()).insert(HEAP_KEY + def.id as usize, h);
    }
    let mut file = std::fs::File::open(pkg.dir.join("memory.bin")).map_err(|e| format!("memory.bin: {e}"))?;
    for (run, ops) in &pkg.runs {
        r.report.pages_written += write_epoch(pkg, &mut file, *run)? as u64;
        r.apply_updates(*run);
        if opts.runs.is_some_and(|(a, b)| *run < a || *run >= b) {
            continue;
        }
        let ops: Vec<Arc<MetalOp>> = ops.iter().filter_map(|op| r.op(op, &pipelines)).map(Arc::new).collect();
        r.report.ops_run += ops.len() as u64;
        r.report.runs += 1;
        if !m.run(&ops) {
            r.report.failed_run = Some(*run);
            break;
        }
    }
    Ok(r.report)
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Compare {
    pub(crate) pages: u64,
    pub(crate) identical: u64,
    /// (page, first differing byte, differing bytes).
    pub(crate) differing: Vec<(u64, usize, usize)>,
}

/// The pages the GPU wrote while recording, compared with memory now.
pub(crate) fn compare_reference(pkg: &Package) -> Result<Compare, String> {
    let mut file = std::fs::File::open(pkg.dir.join("reference").join("memory.bin")).map_err(|e| format!("reference/memory.bin: {e}"))?;
    let mut out = Compare::default();
    for m in &pkg.reference {
        let want = pkg.page_bytes(&mut file, m)?;
        // SAFETY: placed by `place`; the GPU is done (every run was waited for).
        let got = unsafe { std::slice::from_raw_parts(m.va as *const u8, want.len()) };
        out.pages += 1;
        let diff = want.iter().zip(got).filter(|(a, b)| a != b).count();
        if diff == 0 {
            out.identical += 1;
        } else {
            let first = want.iter().zip(got).position(|(a, b)| a != b).unwrap_or(0);
            out.differing.push((m.va, first, diff));
        }
    }
    Ok(out)
}
