//! T3 frame package (T3-FRAME-PACKAGE-DESIGN.md, steps F1–F3): one frame's
//! Metal compute work recorded to a folder, replayed offline in a Wine test
//! ([`replay`]).
//!
//! Recording: `VICTORY_METAL_PACKAGE_FRAME=<n>` (n >= 2) records every
//! `MetalBackend::run` between the presents of frames n-1 and n into
//! `<VICTORY_METAL_PACKAGE_DIR or .>/framepkg-f<n>-<pid>/`, at most
//! `VICTORY_METAL_PACKAGE_CAP_MB` (default 4096) of memory contents. Off by
//! default: then the hooks cost one `OnceLock` read.
//!
//! What a package holds (all integers little endian):
//! - `ops.bin`: per run a segment, per op a TLV; GPU addresses as [`Ref`]s
//!   (title memory by host address, heaps by id, samplers/typed views by
//!   resource id, ...), never raw addresses;
//! - `memory.idx` / `memory.bin`: the 64 KiB pages the run's ops reference,
//!   by epoch (the run before which the content applies; a page is stored
//!   again only when the CPU changed it since the last run that used it);
//! - `heaps.bin`: heap layouts and, per run, the entries (translated) and
//!   stride words that changed;
//! - `resources.bin`: samplers, typed buffer views, texture views by id;
//! - `pipelines.bin` + `shaders/`: each compute pipeline's translated DXIL
//!   with its root signature (and the MSC cache entry when there is one);
//! - `reference/memory.*`: the pages the GPU wrote, as they were after the
//!   frame's last run (what a replay must reproduce);
//! - `manifest.json`: settings, counts, what was dropped.
//!
//! Compute only (F2): draws, clears, copies, resolves are counted and left
//! out (F4).

use std::collections::{BTreeSet, HashMap};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use victory_metal::ir::{DescriptorEntry, Location, ResourceType};
use victory_metal::msc::{RootParam, StaticSampler};

use super::{MetalBackend, MetalCompute, MetalOp};

#[path = "metal_package_replay.rs"]
pub(crate) mod replay;

#[cfg(test)]
#[path = "metal_package_tests.rs"]
mod tests;

pub(crate) const VERSION: u32 = 1;
/// The unit memory is kept in (the title's page size).
pub(crate) const PAGE: u64 = 64 << 10;
/// How far past a root descriptor's address its bytes are kept (the D3D12
/// path bounds unknown root views the same way).
const ROOT_EXTENT: u64 = 4 << 20;

// ---------------------------------------------------------------------------
// References and the op format (F2)
// ---------------------------------------------------------------------------

/// What a GPU address (or resource id) recorded in an op stood for. 16
/// bytes in files: `tag u8 | 3 reserved | id u32 | offset u64`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Ref {
    Null,
    /// Title memory at host address `va` (the address of the wrapped bytes:
    /// a persistent alias where spans were made there); `len`: the bytes
    /// the view covers (a hint for re-wrapping).
    Title { va: u64, len: u32 },
    /// Byte `offset` in package heap `id`'s entries.
    Heap { id: u32, offset: u64 },
    /// Byte `offset` in package heap `id`'s stride table.
    HeapStrides { id: u32, offset: u64 },
    /// A texture view's `gpuResourceID` (F4: not rebuilt yet).
    Texture { id: u32 },
    /// A sampler state's `gpuResourceID`.
    Sampler { id: u32 },
    /// A typed buffer view (texture buffer view of title memory).
    Typed { id: u32 },
    /// The pipeline's own top-level value here (static sampler table).
    Template,
    /// A buffer of the backend's own ([`SPECIAL_WAVE_SCRATCH`], ...).
    Special { kind: u32, offset: u64 },
    /// A one-view table: its address is the run's ring, written at run time.
    RingTable,
    /// Nothing known holds this address: replayed as 0, and counted.
    Unknown { raw: u64 },
}

pub(crate) const SPECIAL_WAVE_SCRATCH: u32 = 1;
pub(crate) const SPECIAL_DIAG: u32 = 2;
pub(crate) const SPECIAL_MEMORY_DIRECTORY: u32 = 3;
pub(crate) const SPECIAL_FAULTS: u32 = 4;
pub(crate) const SPECIAL_STATIC_SAMPLERS: u32 = 5;
pub(crate) const SPECIAL_RING: u32 = 6;

impl Ref {
    fn parts(&self) -> (u8, u32, u64) {
        match *self {
            Ref::Null => (0, 0, 0),
            Ref::Title { va, len } => (1, len, va),
            Ref::Heap { id, offset } => (2, id, offset),
            Ref::HeapStrides { id, offset } => (3, id, offset),
            Ref::Texture { id } => (5, id, 0),
            Ref::Sampler { id } => (6, id, 0),
            Ref::Typed { id } => (7, id, 0),
            Ref::Template => (8, 0, 0),
            Ref::Special { kind, offset } => (9, kind, offset),
            Ref::RingTable => (10, 0, 0),
            Ref::Unknown { raw } => (255, 0, raw),
        }
    }

    pub(crate) fn put(&self, w: &mut Vec<u8>) {
        let (tag, id, offset) = self.parts();
        w.extend_from_slice(&[tag, 0, 0, 0]);
        w.extend_from_slice(&id.to_le_bytes());
        w.extend_from_slice(&offset.to_le_bytes());
    }

    pub(crate) fn get(r: &mut Reader<'_>) -> Option<Self> {
        let tag = r.u8()?;
        r.skip(3)?;
        let id = r.u32()?;
        let offset = r.u64()?;
        Some(match tag {
            0 => Ref::Null,
            1 => Ref::Title { va: offset, len: id },
            2 => Ref::Heap { id, offset },
            3 => Ref::HeapStrides { id, offset },
            5 => Ref::Texture { id },
            6 => Ref::Sampler { id },
            7 => Ref::Typed { id },
            8 => Ref::Template,
            9 => Ref::Special { kind: id, offset },
            10 => Ref::RingTable,
            255 => Ref::Unknown { raw: offset },
            _ => return None,
        })
    }
}

/// A descriptor entry with its addresses as references (40 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TEntry {
    pub(crate) gpu: Ref,
    pub(crate) view: Ref,
    pub(crate) metadata: u64,
}

impl TEntry {
    pub(crate) const NULL: Self = Self { gpu: Ref::Null, view: Ref::Null, metadata: 0 };

    fn put(&self, w: &mut Vec<u8>) {
        self.gpu.put(w);
        self.view.put(w);
        w.extend_from_slice(&self.metadata.to_le_bytes());
    }

    fn get(r: &mut Reader<'_>) -> Option<Self> {
        Some(Self { gpu: Ref::get(r)?, view: Ref::get(r)?, metadata: r.u64()? })
    }
}

/// A top-level argument buffer: its bytes with every address slot zeroed,
/// and what each slot held (by byte offset).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct PTop {
    pub(crate) raw: Vec<u8>,
    pub(crate) refs: Vec<(u32, Ref)>,
}

/// What dispatches share.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PCompute {
    pub(crate) pipeline: u32,
    pub(crate) heap: Ref,
    pub(crate) sampler_heap: Ref,
    pub(crate) top: PTop,
    pub(crate) tables: Vec<(u32, TEntry)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum POp {
    Dispatch { c: PCompute, groups: [u64; 3], after_barrier: bool, loop_counts: Option<[u64; 2]> },
    DispatchIndirect {
        c: PCompute,
        records: Ref,
        stride: u32,
        max: u32,
        count: Option<Ref>,
        records_va: u64,
        count_va: Option<u64>,
        group_base_at: Option<u64>,
    },
    BufferCopy { dst: u64, src: u64, bytes: u64 },
    /// An op kind not recorded yet (F4): [`KIND_DRAW`] ...
    Unsupported { kind: u16 },
}

pub(crate) const KIND_DISPATCH: u16 = 1;
pub(crate) const KIND_INDIRECT: u16 = 2;
pub(crate) const KIND_BUFFER_COPY: u16 = 3;
pub(crate) const KIND_DRAW: u16 = 16;
pub(crate) const KIND_CLEAR: u16 = 17;
pub(crate) const KIND_COPY: u16 = 18;
pub(crate) const KIND_READBACK: u16 = 19;
pub(crate) const KIND_RESOLVE: u16 = 20;
pub(crate) const KIND_AMBIENT: u16 = 21;

/// A little-endian reader over a file's bytes.
pub(crate) struct Reader<'a> {
    pub(crate) b: &'a [u8],
    pub(crate) at: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(b: &'a [u8]) -> Self {
        Self { b, at: 0 }
    }
    pub(crate) fn bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        let out = self.b.get(self.at..self.at.checked_add(n)?)?;
        self.at += n;
        Some(out)
    }
    pub(crate) fn skip(&mut self, n: usize) -> Option<()> {
        self.bytes(n).map(|_| ())
    }
    pub(crate) fn u8(&mut self) -> Option<u8> {
        self.bytes(1).map(|b| b[0])
    }
    pub(crate) fn u16(&mut self) -> Option<u16> {
        self.bytes(2).map(|b| u16::from_le_bytes(b.try_into().unwrap()))
    }
    pub(crate) fn u32(&mut self) -> Option<u32> {
        self.bytes(4).map(|b| u32::from_le_bytes(b.try_into().unwrap()))
    }
    pub(crate) fn u64(&mut self) -> Option<u64> {
        self.bytes(8).map(|b| u64::from_le_bytes(b.try_into().unwrap()))
    }
    pub(crate) fn blob(&mut self) -> Option<&'a [u8]> {
        let n = self.u32()? as usize;
        self.bytes(n)
    }
    pub(crate) fn text(&mut self) -> Option<String> {
        String::from_utf8(self.blob()?.to_vec()).ok()
    }
    pub(crate) fn done(&self) -> bool {
        self.at >= self.b.len()
    }
}

pub(crate) fn put_u32(w: &mut Vec<u8>, v: u32) {
    w.extend_from_slice(&v.to_le_bytes());
}
pub(crate) fn put_u64(w: &mut Vec<u8>, v: u64) {
    w.extend_from_slice(&v.to_le_bytes());
}
pub(crate) fn put_blob(w: &mut Vec<u8>, b: &[u8]) {
    put_u32(w, b.len() as u32);
    w.extend_from_slice(b);
}

fn put_opt_u64(w: &mut Vec<u8>, v: Option<u64>) {
    w.push(v.is_some() as u8);
    put_u64(w, v.unwrap_or(0));
}
fn get_opt_u64(r: &mut Reader<'_>) -> Option<Option<u64>> {
    let some = r.u8()? != 0;
    let v = r.u64()?;
    Some(some.then_some(v))
}

fn put_compute(w: &mut Vec<u8>, c: &PCompute) {
    put_u32(w, c.pipeline);
    c.heap.put(w);
    c.sampler_heap.put(w);
    put_blob(w, &c.top.raw);
    put_u32(w, c.top.refs.len() as u32);
    for (at, r) in &c.top.refs {
        put_u32(w, *at);
        r.put(w);
    }
    put_u32(w, c.tables.len() as u32);
    for (at, e) in &c.tables {
        put_u32(w, *at);
        e.put(w);
    }
}

fn get_compute(r: &mut Reader<'_>) -> Option<PCompute> {
    let pipeline = r.u32()?;
    let heap = Ref::get(r)?;
    let sampler_heap = Ref::get(r)?;
    let raw = r.blob()?.to_vec();
    let n = r.u32()? as usize;
    let mut refs = Vec::with_capacity(n.min(4096));
    for _ in 0..n {
        refs.push((r.u32()?, Ref::get(r)?));
    }
    let n = r.u32()? as usize;
    let mut tables = Vec::with_capacity(n.min(4096));
    for _ in 0..n {
        tables.push((r.u32()?, TEntry::get(r)?));
    }
    Some(PCompute { pipeline, heap, sampler_heap, top: PTop { raw, refs }, tables })
}

impl POp {
    pub(crate) fn kind(&self) -> u16 {
        match self {
            POp::Dispatch { .. } => KIND_DISPATCH,
            POp::DispatchIndirect { .. } => KIND_INDIRECT,
            POp::BufferCopy { .. } => KIND_BUFFER_COPY,
            POp::Unsupported { kind } => *kind,
        }
    }

    /// `kind u16 | len u32 | payload`.
    pub(crate) fn put(&self, w: &mut Vec<u8>) {
        let mut p = Vec::new();
        match self {
            POp::Dispatch { c, groups, after_barrier, loop_counts } => {
                put_compute(&mut p, c);
                for g in groups {
                    put_u64(&mut p, *g);
                }
                p.push(*after_barrier as u8);
                p.push(loop_counts.is_some() as u8);
                for v in loop_counts.unwrap_or([0; 2]) {
                    put_u64(&mut p, v);
                }
            }
            POp::DispatchIndirect { c, records, stride, max, count, records_va, count_va, group_base_at } => {
                put_compute(&mut p, c);
                records.put(&mut p);
                put_u32(&mut p, *stride);
                put_u32(&mut p, *max);
                p.push(count.is_some() as u8);
                count.unwrap_or(Ref::Null).put(&mut p);
                put_u64(&mut p, *records_va);
                put_opt_u64(&mut p, *count_va);
                put_opt_u64(&mut p, *group_base_at);
            }
            POp::BufferCopy { dst, src, bytes } => {
                put_u64(&mut p, *dst);
                put_u64(&mut p, *src);
                put_u64(&mut p, *bytes);
            }
            POp::Unsupported { .. } => {}
        }
        w.extend_from_slice(&self.kind().to_le_bytes());
        put_blob(w, &p);
    }

    pub(crate) fn get(r: &mut Reader<'_>) -> Option<Self> {
        let kind = r.u16()?;
        let mut p = Reader::new(r.blob()?);
        let op = match kind {
            KIND_DISPATCH => {
                let c = get_compute(&mut p)?;
                let groups = [p.u64()?, p.u64()?, p.u64()?];
                let after_barrier = p.u8()? != 0;
                let some = p.u8()? != 0;
                let lc = [p.u64()?, p.u64()?];
                POp::Dispatch { c, groups, after_barrier, loop_counts: some.then_some(lc) }
            }
            KIND_INDIRECT => {
                let c = get_compute(&mut p)?;
                let records = Ref::get(&mut p)?;
                let stride = p.u32()?;
                let max = p.u32()?;
                let some = p.u8()? != 0;
                let count = Ref::get(&mut p)?;
                let records_va = p.u64()?;
                let count_va = get_opt_u64(&mut p)?;
                let group_base_at = get_opt_u64(&mut p)?;
                POp::DispatchIndirect { c, records, stride, max, count: some.then_some(count), records_va, count_va, group_base_at }
            }
            KIND_BUFFER_COPY => POp::BufferCopy { dst: p.u64()?, src: p.u64()?, bytes: p.u64()? },
            kind => POp::Unsupported { kind },
        };
        p.done().then_some(op)
    }

    /// Every reference the op holds.
    pub(crate) fn refs(&self) -> Vec<Ref> {
        let mut out = Vec::new();
        let mut compute = |c: &PCompute| {
            out.push(c.heap);
            out.push(c.sampler_heap);
            out.extend(c.top.refs.iter().map(|(_, r)| *r));
            for (_, e) in &c.tables {
                out.push(e.gpu);
                out.push(e.view);
            }
        };
        match self {
            POp::Dispatch { c, .. } => compute(c),
            POp::DispatchIndirect { c, records, count, .. } => {
                compute(c);
                out.push(*records);
                out.extend(*count);
            }
            POp::BufferCopy { .. } | POp::Unsupported { .. } => {}
        }
        out
    }
}

/// `ops.bin`: magic, version; then per run `run u32 | ops u32` and the ops.
pub(crate) const OPS_MAGIC: &[u8; 8] = b"VPKOPS01";

pub(crate) fn put_run(w: &mut Vec<u8>, run: u32, ops: &[POp]) {
    put_u32(w, run);
    put_u32(w, ops.len() as u32);
    for op in ops {
        op.put(w);
    }
}

pub(crate) fn get_runs(b: &[u8]) -> Result<Vec<(u32, Vec<POp>)>, String> {
    let mut r = Reader::new(b);
    if r.bytes(8) != Some(OPS_MAGIC.as_slice()) {
        return Err("ops.bin: not a frame package op file".into());
    }
    let version = r.u32().ok_or("ops.bin: truncated")?;
    if version != VERSION {
        return Err(format!("ops.bin: version {version}, this build reads {VERSION}"));
    }
    let mut runs = Vec::new();
    while !r.done() {
        let run = r.u32().ok_or("ops.bin: truncated run header")?;
        let n = r.u32().ok_or("ops.bin: truncated run header")?;
        let mut ops = Vec::with_capacity(n.min(1 << 16) as usize);
        for i in 0..n {
            ops.push(POp::get(&mut r).ok_or_else(|| format!("ops.bin: run {run} op {i} does not decode"))?);
        }
        runs.push((run, ops));
    }
    Ok(runs)
}

// ---------------------------------------------------------------------------
// Address translation (record side)
// ---------------------------------------------------------------------------

/// A backend buffer of known meaning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Own {
    Heap(u32),
    Strides(u32),
    Special(u32),
}

/// Everything a recorded GPU address or resource id can be looked up in.
#[derive(Default)]
pub(crate) struct AddressMap {
    /// Title memory wrappers: (GPU address, bytes, host address), by GPU address.
    pub(crate) title: Vec<(u64, u64, u64)>,
    /// The backend's own buffers: (GPU address, bytes, what), by GPU address.
    pub(crate) own: Vec<(u64, u64, Own)>,
    /// `gpuResourceID` → package id.
    pub(crate) samplers: HashMap<u64, u32>,
    pub(crate) typed: HashMap<u64, u32>,
    pub(crate) textures: HashMap<u64, u32>,
    /// The title buffer holding a host address, as (start, end): bounds a
    /// root view the way the D3D12 path does (`shadow::buffer_extent`).
    pub(crate) extent: Option<fn(u64) -> Option<(u64, u64)>>,
}

fn find<T: Copy>(v: &[(u64, u64, T)], gpu: u64) -> Option<(u64, u64, T)> {
    let i = v.partition_point(|e| e.0 <= gpu).checked_sub(1)?;
    let e = v[i];
    (gpu - e.0 < e.1).then_some(e)
}

impl AddressMap {
    pub(crate) fn sort(&mut self) {
        self.title.sort_unstable_by_key(|e| e.0);
        self.own.sort_unstable_by_key(|e| e.0);
    }

    /// The reference for GPU address `gpu`; `len`: the bytes the view
    /// covers when known (else up to [`ROOT_EXTENT`] inside its buffer).
    pub(crate) fn address(&self, gpu: u64, len: Option<u64>) -> Ref {
        if gpu == 0 {
            return Ref::Null;
        }
        if let Some((g, n, host)) = find(&self.title, gpu) {
            let room = n - (gpu - g);
            let va = host + (gpu - g);
            let buffer = || self.extent.and_then(|f| f(va)).filter(|&(s, e)| s <= va && va < e).map(|(_, e)| e - va);
            let len = len.or_else(buffer).unwrap_or(ROOT_EXTENT).min(room).max(1);
            return Ref::Title { va, len: len.min(u64::from(u32::MAX)) as u32 };
        }
        match find(&self.own, gpu) {
            Some((g, _, Own::Heap(id))) => Ref::Heap { id, offset: gpu - g },
            Some((g, _, Own::Strides(id))) => Ref::HeapStrides { id, offset: gpu - g },
            Some((g, _, Own::Special(kind))) => Ref::Special { kind, offset: gpu - g },
            None => Ref::Unknown { raw: gpu },
        }
    }

    /// A heap or table entry with its addresses as references.
    pub(crate) fn entry(&self, e: &DescriptorEntry, samplers: bool) -> TEntry {
        if *e == DescriptorEntry::NULL {
            return TEntry::NULL;
        }
        let unknown = |raw: u64| if raw == 0 { Ref::Null } else { Ref::Unknown { raw } };
        if samplers {
            let gpu = self.samplers.get(&e.gpu_va).map_or_else(|| unknown(e.gpu_va), |&id| Ref::Sampler { id });
            return TEntry { gpu, view: unknown(e.texture_view_id), metadata: e.metadata };
        }
        if e.gpu_va != 0 {
            let size = e.metadata & 0xffff_ffff;
            let gpu = self.address(e.gpu_va, Some(size));
            let view = if e.metadata >> 63 == 1 {
                self.typed.get(&e.texture_view_id).map_or_else(|| unknown(e.texture_view_id), |&id| Ref::Typed { id })
            } else {
                unknown(e.texture_view_id)
            };
            return TEntry { gpu, view, metadata: e.metadata };
        }
        let view = self.textures.get(&e.texture_view_id).map_or_else(|| unknown(e.texture_view_id), |&id| Ref::Texture { id });
        TEntry { gpu: Ref::Null, view, metadata: e.metadata }
    }

    /// `top` decoded by `layout`: constants stay as bytes; every address
    /// slot becomes a reference (one-view table slots: [`Ref::RingTable`];
    /// the pipeline's own value, as in `template`: [`Ref::Template`]).
    pub(crate) fn top(&self, layout: &[Location], template: &[u8], top: &[u8], tables: &[(u32, DescriptorEntry)]) -> PTop {
        let mut raw = top.to_vec();
        let mut refs = Vec::new();
        for l in layout {
            let address = match l.kind {
                ResourceType::Constant => false,
                ResourceType::Other(_) => l.size == 8,
                _ => true,
            };
            let at = l.offset as usize;
            if !address || at + 8 > top.len() {
                continue;
            }
            let v = u64::from_le_bytes(top[at..at + 8].try_into().unwrap());
            let r = if tables.iter().any(|t| t.0 == l.offset) {
                Ref::RingTable
            } else if v == 0 {
                Ref::Null
            } else if template.get(at..at + 8) == Some(&top[at..at + 8]) {
                Ref::Template
            } else {
                self.address(v, None)
            };
            raw[at..at + 8].fill(0);
            refs.push((l.offset, r));
        }
        PTop { raw, refs }
    }
}

// ---------------------------------------------------------------------------
// Address resolution (replay side)
// ---------------------------------------------------------------------------

/// Turns references back into this process's addresses.
pub(crate) trait Resolver {
    /// The address (or resource id) for `r`; 0 for what cannot be had.
    fn address(&mut self, r: &Ref) -> u64;
    /// The entry of typed view `id`, made anew.
    fn typed(&mut self, id: u32) -> Option<DescriptorEntry>;
}

pub(crate) fn resolve_entry(res: &mut dyn Resolver, e: &TEntry) -> DescriptorEntry {
    if let Ref::Typed { id } = e.view
        && let Some(t) = res.typed(id)
    {
        return DescriptorEntry { gpu_va: t.gpu_va, texture_view_id: t.texture_view_id, metadata: e.metadata };
    }
    DescriptorEntry { gpu_va: res.address(&e.gpu), texture_view_id: res.address(&e.view), metadata: e.metadata }
}

/// The top-level bytes for `top` in this process (`template`: the rebuilt
/// pipeline's own top level).
pub(crate) fn resolve_top(res: &mut dyn Resolver, top: &PTop, template: &[u8]) -> Vec<u8> {
    let mut bytes = top.raw.clone();
    for (at, r) in &top.refs {
        let at = *at as usize;
        if at + 8 > bytes.len() {
            continue;
        }
        let v = match r {
            Ref::Null | Ref::RingTable => 0,
            Ref::Template => template.get(at..at + 8).map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap())),
            r => res.address(r),
        };
        bytes[at..at + 8].copy_from_slice(&v.to_le_bytes());
    }
    bytes
}

// ---------------------------------------------------------------------------
// Pipeline sources
// ---------------------------------------------------------------------------

/// What a compute pipeline was made from.
#[derive(Clone, Debug)]
pub(crate) enum ShaderCode {
    Dxil { dxil: Arc<Vec<u8>>, params: Vec<RootParam>, samplers: Vec<StaticSampler> },
    /// Metal source (synthetic tests).
    Msl { source: String, function: String },
}

#[derive(Clone, Debug)]
pub(crate) struct Source {
    pub(crate) code: ShaderCode,
    /// Its Metal Shader Converter cache file, if any.
    pub(crate) cache: Option<PathBuf>,
}

/// Sources by Metal pipeline object, kept while recording is wanted.
static SOURCES: Mutex<Option<HashMap<u64, Arc<Source>>>> = Mutex::new(None);

/// Keeps what pipeline `pso` was made from (for its package entry).
pub(crate) fn register_source(pso: &victory_metal::ComputePipeline, source: Source) {
    SOURCES.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashMap::new).insert(pso.handle(), Arc::new(source));
}

/// Hook (`MetalBackend::compute_pipeline`): a title compute pipeline was
/// made; its translated DXIL and root signature are kept when recording is
/// wanted.
pub(crate) fn note_compute(made: &Option<Arc<MetalCompute>>, p: &crate::pipeline::host::HostPipeline) {
    if settings().is_none() {
        return;
    }
    let (Some(made), Some(stage)) = (made, p.stages.first()) else { return };
    let host = &stage.translated().host;
    let params = super::root_params(host);
    let samplers = super::static_samplers(host);
    let cache = super::msc_cache_path(&host.dxil, &params, &samplers);
    register_source(&made.pso, Source { code: ShaderCode::Dxil { dxil: Arc::new(host.dxil.clone()), params, samplers }, cache });
}

fn put_params(w: &mut Vec<u8>, params: &[RootParam]) {
    put_u32(w, params.len() as u32);
    for p in params {
        let (tag, a, b, c) = match *p {
            RootParam::Constants { register, space, num } => (0u8, register, space, num),
            RootParam::Cbv { register, space } => (1, register, space, 0),
            RootParam::Srv { register, space } => (2, register, space, 0),
            RootParam::Uav { register, space } => (3, register, space, 0),
            RootParam::TableSrv { register, space } => (4, register, space, 0),
            RootParam::TableUav { register, space } => (5, register, space, 0),
        };
        w.push(tag);
        for v in [a, b, c] {
            put_u32(w, v);
        }
    }
}

pub(crate) fn get_params(r: &mut Reader<'_>) -> Option<Vec<RootParam>> {
    let n = r.u32()?;
    let mut out = Vec::new();
    for _ in 0..n {
        let tag = r.u8()?;
        let (register, space, num) = (r.u32()?, r.u32()?, r.u32()?);
        out.push(match tag {
            0 => RootParam::Constants { register, space, num },
            1 => RootParam::Cbv { register, space },
            2 => RootParam::Srv { register, space },
            3 => RootParam::Uav { register, space },
            4 => RootParam::TableSrv { register, space },
            5 => RootParam::TableUav { register, space },
            _ => return None,
        });
    }
    Some(out)
}

fn put_samplers(w: &mut Vec<u8>, samplers: &[StaticSampler]) {
    put_u32(w, samplers.len() as u32);
    for s in samplers {
        for v in [
            s.filter,
            s.address_u,
            s.address_v,
            s.address_w,
            s.mip_lod_bias.to_bits(),
            s.max_anisotropy,
            s.comparison_func,
            s.border_color,
            s.min_lod.to_bits(),
            s.max_lod.to_bits(),
            s.shader_register,
            s.register_space,
        ] {
            put_u32(w, v);
        }
    }
}

pub(crate) fn get_samplers(r: &mut Reader<'_>) -> Option<Vec<StaticSampler>> {
    let n = r.u32()?;
    let mut out = Vec::new();
    for _ in 0..n {
        let mut v = [0u32; 12];
        for x in &mut v {
            *x = r.u32()?;
        }
        out.push(StaticSampler {
            filter: v[0],
            address_u: v[1],
            address_v: v[2],
            address_w: v[3],
            mip_lod_bias: f32::from_bits(v[4]),
            max_anisotropy: v[5],
            comparison_func: v[6],
            border_color: v[7],
            min_lod: f32::from_bits(v[8]),
            max_lod: f32::from_bits(v[9]),
            shader_register: v[10],
            register_space: v[11],
        });
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Recording (F1)
// ---------------------------------------------------------------------------

pub(crate) struct Settings {
    pub(crate) frame: u64,
    pub(crate) dir: PathBuf,
    pub(crate) cap: u64,
}

/// `VICTORY_METAL_PACKAGE_FRAME` and friends, read once.
pub(crate) fn settings() -> Option<&'static Settings> {
    static S: OnceLock<Option<Settings>> = OnceLock::new();
    S.get_or_init(|| {
        let frame: u64 = std::env::var("VICTORY_METAL_PACKAGE_FRAME").ok()?.parse().ok()?;
        if frame < 2 {
            tracing::warn!(frame, "VICTORY_METAL_PACKAGE_FRAME must be 2 or more: no frame package");
            return None;
        }
        let dir = std::env::var_os("VICTORY_METAL_PACKAGE_DIR").filter(|v| !v.is_empty()).map_or_else(|| PathBuf::from("."), PathBuf::from);
        let cap = std::env::var("VICTORY_METAL_PACKAGE_CAP_MB").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(4096) << 20;
        tracing::warn!(frame, dir = %dir.display(), cap_mb = cap >> 20, "VICTORY_METAL_PACKAGE_FRAME: this frame's Metal compute work is recorded (slow frame)");
        // The variable only: asking the timeline here could start it early.
        if std::env::var("VICTORY_METAL_EXEC").as_deref() == Ok("1") {
            // Its timeline encodes compute (and with timeline draws, the
            // rest) without `queue::run_metal_checked`: those ops are missed.
            tracing::warn!("VICTORY_METAL_PACKAGE_FRAME with VICTORY_METAL_EXEC=1: timeline-encoded ops are not recorded; record with EXEC=0");
        }
        Some(Settings { frame, dir, cap })
    })
    .as_ref()
}

/// The recording in progress (between the presents of frames n-1 and n).
static ACTIVE: Mutex<Option<Recorder>> = Mutex::new(None);

/// Hook (`queue::run_metal_checked`): runs `ops`, recording them first
/// when a package is being recorded.
pub(crate) fn run(m: &MetalBackend, ops: &[Arc<MetalOp>]) -> bool {
    if settings().is_none() {
        return m.run(ops);
    }
    let mut active = ACTIVE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(rec) = active.as_mut() else {
        drop(active);
        return m.run(ops);
    };
    let pages = rec.before(m, ops);
    let ok = m.run(ops);
    rec.after(&pages);
    ok
}

/// Hook (`queue::present`): frame `frame` was presented; starts the
/// recording after frame n-1 and writes it out at frame n.
pub(crate) fn frame_presented(frame: u64) {
    let Some(s) = settings() else { return };
    if frame + 1 == s.frame {
        let dir = s.dir.join(format!("framepkg-f{:06}-{}", s.frame, std::process::id()));
        match Recorder::new(&dir, s.frame, s.cap) {
            Ok(r) => {
                tracing::warn!(dir = %dir.display(), "frame package: recording");
                *ACTIVE.lock().unwrap_or_else(|e| e.into_inner()) = Some(r);
            }
            Err(e) => tracing::error!(dir = %dir.display(), error = %e, "frame package: could not start"),
        }
    } else if frame == s.frame {
        let Some(r) = ACTIVE.lock().unwrap_or_else(|e| e.into_inner()).take() else { return };
        let backend = super::BACKEND.get().and_then(Option::as_ref);
        match r.finish(backend) {
            Ok(st) => tracing::warn!(runs = st.runs, ops = st.ops, unknown = st.unknown, memory_mb = st.memory_bytes >> 20, stopped = ?st.stopped, "frame package: written"),
            Err(e) => tracing::error!(error = %e, "frame package: could not be written"),
        }
    }
}

/// One page as last seen.
struct Page {
    hash: [u8; 32],
    gpu_written: bool,
}

struct HeapShadow {
    id: u32,
    count: u32,
    capacity: u32,
    samplers: bool,
    entries: Vec<TEntry>,
    strides: Vec<u32>,
}

/// Counts for the manifest (and tests).
#[derive(Clone, Debug, Default)]
pub(crate) struct Stats {
    pub(crate) runs: u32,
    pub(crate) ops: u64,
    pub(crate) dispatches: u64,
    pub(crate) indirect: u64,
    pub(crate) buffer_copies: u64,
    /// Ops left out (F4) by kind.
    pub(crate) unsupported: HashMap<u16, u64>,
    pub(crate) unknown: u64,
    /// Heap entries whose address or view id nothing known holds.
    pub(crate) heap_unknown: u64,
    pub(crate) unknown_samples: Vec<u64>,
    pub(crate) texture_refs: u64,
    pub(crate) memory_directory_refs: u64,
    pub(crate) pipelines: u32,
    pub(crate) pipelines_without_source: u32,
    pub(crate) pages: u64,
    pub(crate) page_versions: u64,
    pub(crate) unreadable_pages: u64,
    pub(crate) memory_bytes: u64,
    pub(crate) heap_entries: u64,
    pub(crate) reference_pages: u64,
    pub(crate) cpu_changed_after: u64,
    pub(crate) remaps: u64,
    pub(crate) stopped: Option<String>,
}

pub(crate) struct Recorder {
    dir: PathBuf,
    frame: u64,
    cap: u64,
    run: u32,
    ops: Vec<u8>,
    heaps_bin: Vec<u8>,
    resources_bin: Vec<u8>,
    memory_idx: Vec<u8>,
    memory_bin: std::io::BufWriter<std::fs::File>,
    memory_at: u64,
    pages: HashMap<u64, Page>,
    pipelines: HashMap<u64, u32>,
    pipelines_bin: Vec<u8>,
    shaders: BTreeSet<String>,
    heap_ids: HashMap<usize, u32>,
    heaps: Vec<HeapShadow>,
    samplers: HashMap<[u32; 13], u32>,
    typed: HashMap<(u64, u64, u32), u32>,
    textures: HashMap<(usize, [u32; 10], bool), u32>,
    remaps_at_start: Option<u64>,
    stats: Stats,
}

/// `memory.idx` record: `va u64 | len u32 | epoch u32 | offset u64 | flags u32`.
pub(crate) const MEMORY_RECORD: usize = 28;

fn read_host(va: u64, len: u64) -> Vec<u8> {
    let _reading = crate::watch::title_memory_reading_range(va as usize, len as usize);
    // SAFETY: the caller checked the range is committed and readable; the
    // mapping gate keeps it from being remapped while read.
    unsafe { std::slice::from_raw_parts(va as *const u8, len as usize).to_vec() }
}

fn readable(page: u64) -> u64 {
    victory_metal::chunks::committed_extent(page, PAGE)
}

impl Recorder {
    pub(crate) fn new(dir: &Path, frame: u64, cap: u64) -> std::io::Result<Self> {
        crate::make_dir(dir)?;
        crate::make_dir(&dir.join("shaders"))?;
        crate::make_dir(&dir.join("reference"))?;
        let memory_bin = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(dir.join("memory.bin"))?);
        let mut ops = OPS_MAGIC.to_vec();
        put_u32(&mut ops, VERSION);
        Ok(Self {
            dir: dir.to_path_buf(),
            frame,
            cap,
            run: 0,
            ops,
            heaps_bin: b"VPKHEP01".to_vec(),
            resources_bin: b"VPKRES01".to_vec(),
            memory_idx: Vec::new(),
            memory_bin,
            memory_at: 0,
            pages: HashMap::new(),
            pipelines: HashMap::new(),
            pipelines_bin: Vec::new(),
            shaders: BTreeSet::new(),
            heap_ids: HashMap::new(),
            heaps: Vec::new(),
            samplers: HashMap::new(),
            typed: HashMap::new(),
            textures: HashMap::new(),
            remaps_at_start: None,
            stats: Stats::default(),
        })
    }

    /// The address map of `m` now; ids are given to every heap, sampler,
    /// typed view and texture view the backend has.
    fn address_map(&mut self, m: &MetalBackend) -> AddressMap {
        let mut map = AddressMap::default();
        {
            let chunks = m.chunks();
            map.title = chunks.wrapped();
            map.extent = Some(crate::shadow::buffer_extent);
            if self.remaps_at_start.is_none() {
                self.remaps_at_start = Some(chunks.stats.remaps_under_spans);
            }
        }
        {
            let heaps = m.heaps.lock().unwrap_or_else(|e| e.into_inner());
            for (&key, h) in heaps.iter() {
                let next = self.heap_ids.len() as u32;
                let id = *self.heap_ids.entry(key).or_insert(next);
                map.own.push((h.buffer.gpu_address, h.buffer.len, Own::Heap(id)));
                if let Some((s, _)) = &h.strides {
                    map.own.push((s.gpu_address, s.len, Own::Strides(id)));
                }
            }
        }
        if let Some(Some(b)) = m.wave_scratch.get() {
            map.own.push((b.gpu_address, b.len, Own::Special(SPECIAL_WAVE_SCRATCH)));
        }
        if let Some(Some(b)) = m.diag.get() {
            map.own.push((b.gpu_address, b.len, Own::Special(SPECIAL_DIAG)));
        }
        if let Some([d, f]) = super::memory::own_ranges() {
            map.own.push((d.0, d.1, Own::Special(SPECIAL_MEMORY_DIRECTORY)));
            map.own.push((f.0, f.1, Own::Special(SPECIAL_FAULTS)));
        }
        for b in m.static_sampler_tables.lock().unwrap_or_else(|e| e.into_inner()).values() {
            map.own.push((b.gpu_address, b.len, Own::Special(SPECIAL_STATIC_SAMPLERS)));
        }
        if let Some(r) = m.ring.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            map.own.push((r.buffer.gpu_address, r.buffer.len, Own::Special(SPECIAL_RING)));
        }
        for (words, (_, e)) in m.samplers.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            let next = self.samplers.len() as u32;
            let id = *self.samplers.entry(*words).or_insert_with(|| {
                self.resources_bin.push(1);
                put_u32(&mut self.resources_bin, next);
                for w in words {
                    put_u32(&mut self.resources_bin, *w);
                }
                next
            });
            map.samplers.insert(e.gpu_va, id);
        }
        for (key, (_, e)) in m.typed.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            let next = self.typed.len() as u32;
            let id = *self.typed.entry(*key).or_insert_with(|| {
                self.resources_bin.push(2);
                put_u32(&mut self.resources_bin, next);
                put_u64(&mut self.resources_bin, key.0);
                put_u64(&mut self.resources_bin, key.1);
                put_u32(&mut self.resources_bin, key.2);
                next
            });
            map.typed.insert(e.texture_view_id, id);
        }
        for (key, (_, e)) in m.views.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            let next = self.textures.len() as u32;
            let id = *self.textures.entry(*key).or_insert_with(|| {
                self.resources_bin.push(3);
                put_u32(&mut self.resources_bin, next);
                put_u64(&mut self.resources_bin, key.0 as u64);
                for w in key.1 {
                    put_u32(&mut self.resources_bin, w);
                }
                self.resources_bin.push(key.2 as u8);
                next
            });
            map.textures.insert(e.texture_view_id, id);
        }
        map.sort();
        map
    }

    /// The package id of compute pipeline `p` (written on first sight).
    fn pipeline(&mut self, p: &MetalCompute) -> u32 {
        let handle = p.pso.handle();
        if let Some(&id) = self.pipelines.get(&handle) {
            return id;
        }
        let id = self.pipelines.len() as u32;
        self.pipelines.insert(handle, id);
        self.stats.pipelines += 1;
        let source = SOURCES.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|s| s.get(&handle).cloned());
        let w = &mut self.pipelines_bin;
        put_u32(w, id);
        put_blob(w, p.name.as_bytes());
        for t in p.threadgroup {
            put_u64(w, t);
        }
        put_blob(w, &p.top_level.layout_bytes());
        put_blob(w, &p.top_level.bytes);
        match source.as_deref() {
            None => {
                w.push(0);
                self.stats.pipelines_without_source += 1;
            }
            Some(Source { code: ShaderCode::Dxil { dxil, params, samplers }, cache }) => {
                w.push(1);
                let name = format!("{}.dxil", blake3::hash(dxil).to_hex());
                if self.shaders.insert(name.clone()) {
                    let _ = std::fs::write(self.dir.join("shaders").join(&name), dxil.as_slice());
                }
                put_blob(w, name.as_bytes());
                put_params(w, params);
                put_samplers(w, samplers);
                let vmsc = cache.as_ref().and_then(|path| {
                    let bytes = crate::read_file(path).ok()?;
                    let name = path.file_name()?.to_string_lossy().into_owned();
                    std::fs::write(self.dir.join("shaders").join(&name), bytes).ok()?;
                    Some(name)
                });
                put_blob(w, vmsc.unwrap_or_default().as_bytes());
            }
            Some(Source { code: ShaderCode::Msl { source, function }, .. }) => {
                w.push(2);
                put_blob(w, source.as_bytes());
                put_blob(w, function.as_bytes());
            }
        }
        id
    }

    fn compute(
        &mut self,
        map: &AddressMap,
        pso: &MetalCompute,
        heap: Option<&victory_metal::Buffer>,
        sampler_heap: Option<&victory_metal::Buffer>,
        top: &[u8],
        tables: &[(u32, DescriptorEntry)],
    ) -> PCompute {
        let pipeline = self.pipeline(pso);
        let buffer = |b: Option<&victory_metal::Buffer>| b.map_or(Ref::Null, |b| map.address(b.gpu_address, Some(b.len)));
        PCompute {
            pipeline,
            heap: buffer(heap),
            sampler_heap: buffer(sampler_heap),
            top: map.top(&pso.top_level.layout, &pso.top_level.bytes, top, tables),
            tables: tables.iter().map(|(at, e)| (*at, map.entry(e, false))).collect(),
        }
    }

    /// `ops` as package ops (F2).
    pub(crate) fn translate(&mut self, map: &AddressMap, ops: &[Arc<MetalOp>]) -> Vec<POp> {
        ops.iter()
            .map(|op| match &**op {
                MetalOp::Dispatch(c) => POp::Dispatch {
                    c: self.compute(map, &c.pso, c.heap.as_ref(), c.sampler_heap.as_ref(), &c.top, &c.tables),
                    groups: c.groups,
                    after_barrier: c.after_barrier,
                    loop_counts: c.loop_counts.as_ref().map(|l| [l[0].0, l[1].0]),
                },
                MetalOp::DispatchIndirect(c) => POp::DispatchIndirect {
                    c: self.compute(map, &c.pso, c.heap.as_ref(), c.sampler_heap.as_ref(), &c.top, &c.tables),
                    records: map.address(c.records.gpu_address + c.records_offset, Some(u64::from(c.stride) * u64::from(c.max.max(1)))),
                    stride: c.stride,
                    max: c.max,
                    count: c.count.as_ref().map(|(b, o)| map.address(b.gpu_address + o, Some(4))),
                    records_va: c.records_va,
                    count_va: c.count_va,
                    group_base_at: c.group_base_at.map(|v| v as u64),
                },
                MetalOp::BufferCopy { dst, src, bytes } => POp::BufferCopy { dst: *dst as u64, src: *src as u64, bytes: *bytes as u64 },
                MetalOp::Draw(_) => POp::Unsupported { kind: KIND_DRAW },
                MetalOp::Clear(_) => POp::Unsupported { kind: KIND_CLEAR },
                MetalOp::Copy(_) => POp::Unsupported { kind: KIND_COPY },
                MetalOp::FloatRowReadback { .. } => POp::Unsupported { kind: KIND_READBACK },
                MetalOp::Resolve(_) => POp::Unsupported { kind: KIND_RESOLVE },
                MetalOp::Ambient(_) => POp::Unsupported { kind: KIND_AMBIENT },
            })
            .collect()
    }

    /// Snapshots the heaps `ops` bind: their layout once, then the entries
    /// and stride words that changed. Returns the title ranges their buffer
    /// entries name.
    fn snapshot_heaps(&mut self, m: &MetalBackend, map: &AddressMap, ops: &[POp]) -> Vec<(u64, u64)> {
        let mut wanted = BTreeSet::new();
        for op in ops {
            if let POp::Dispatch { c, .. } | POp::DispatchIndirect { c, .. } = op {
                for r in [c.heap, c.sampler_heap] {
                    if let Ref::Heap { id, .. } = r {
                        wanted.insert(id);
                    }
                }
                for (_, r) in &c.top.refs {
                    if let Ref::HeapStrides { id, .. } = r {
                        wanted.insert(*id);
                    }
                }
            }
        }
        let mut ranges = Vec::new();
        let heaps = m.heaps.lock().unwrap_or_else(|e| e.into_inner());
        for (&key, h) in heaps.iter() {
            let Some(&id) = self.heap_ids.get(&key) else { continue };
            if !wanted.contains(&id) {
                continue;
            }
            let samplers = h.strides.is_none();
            let shadow = match self.heaps.iter().position(|s| s.id == id) {
                Some(i) => i,
                None => {
                    self.heaps_bin.push(1);
                    put_u32(&mut self.heaps_bin, id);
                    put_u32(&mut self.heaps_bin, h.count);
                    put_u32(&mut self.heaps_bin, h.capacity);
                    self.heaps_bin.push(samplers as u8);
                    let strides = if h.strides.is_some() { victory_shader::emit::SLOT_STRIDES_DWORDS as usize } else { 0 };
                    self.heaps.push(HeapShadow {
                        id,
                        count: h.count,
                        capacity: h.capacity,
                        samplers,
                        entries: vec![TEntry::NULL; h.capacity as usize],
                        strides: vec![0; strides],
                    });
                    self.heaps.len() - 1
                }
            };
            let s = &mut self.heaps[shadow];
            // SAFETY: the heap's memory holds `capacity` entries; written only
            // under the heaps lock, which is held.
            let entries = unsafe { std::slice::from_raw_parts(h.entries, s.capacity as usize) };
            let mut changed = Vec::new();
            for (i, e) in entries.iter().enumerate() {
                let t = map.entry(e, s.samplers);
                if s.entries[i] != t {
                    s.entries[i] = t;
                    changed.push(i as u32);
                }
            }
            if !changed.is_empty() {
                self.heaps_bin.push(2);
                put_u32(&mut self.heaps_bin, id);
                put_u32(&mut self.heaps_bin, self.run);
                put_u32(&mut self.heaps_bin, changed.len() as u32);
                for &i in &changed {
                    put_u32(&mut self.heaps_bin, i);
                    let e = s.entries[i as usize];
                    e.put(&mut self.heaps_bin);
                    for r in [e.gpu, e.view] {
                        match r {
                            Ref::Unknown { raw } => {
                                self.stats.heap_unknown += 1;
                                if self.stats.unknown_samples.len() < 16 {
                                    self.stats.unknown_samples.push(raw);
                                }
                            }
                            Ref::Texture { .. } => self.stats.texture_refs += 1,
                            _ => {}
                        }
                    }
                }
                self.stats.heap_entries += changed.len() as u64;
            }
            if let Some((_, ptr)) = h.strides {
                // SAFETY: the stride table holds SLOT_STRIDES_DWORDS words (`new_heap`).
                let words = unsafe { std::slice::from_raw_parts(ptr, s.strides.len()) };
                let changed: Vec<u32> = (0..words.len() as u32).filter(|&i| words[i as usize] != s.strides[i as usize]).collect();
                if !changed.is_empty() {
                    self.heaps_bin.push(3);
                    put_u32(&mut self.heaps_bin, id);
                    put_u32(&mut self.heaps_bin, self.run);
                    put_u32(&mut self.heaps_bin, changed.len() as u32);
                    for &i in &changed {
                        s.strides[i as usize] = words[i as usize];
                        put_u32(&mut self.heaps_bin, i);
                        put_u32(&mut self.heaps_bin, words[i as usize]);
                    }
                }
            }
            for e in &s.entries {
                if let Ref::Title { va, len } = e.gpu {
                    ranges.push((va, u64::from(len)));
                }
            }
        }
        ranges
    }

    /// Before `ops` run: translates them, snapshots the heaps they bind and
    /// the memory they reference (pages new or changed by the CPU since
    /// they were last used). Returns the pages with their hashes now, for
    /// [`Recorder::after`].
    pub(crate) fn before(&mut self, m: &MetalBackend, ops: &[Arc<MetalOp>]) -> Vec<(u64, [u8; 32])> {
        if self.stats.stopped.is_some() {
            return Vec::new();
        }
        let map = self.address_map(m);
        let pops = self.translate(&map, ops);
        let mut ranges = self.snapshot_heaps(m, &map, &pops);
        for op in &pops {
            for r in op.refs() {
                if let Ref::Title { va, len } = r {
                    ranges.push((va, u64::from(len)));
                }
            }
            match op {
                POp::Dispatch { loop_counts: Some(l), .. } => ranges.extend(l.iter().map(|&va| (va, 4))),
                POp::BufferCopy { dst, src, bytes } => {
                    ranges.push((*dst, *bytes));
                    ranges.push((*src, *bytes));
                }
                _ => {}
            }
        }
        let mut pages = BTreeSet::new();
        for (va, len) in ranges {
            if len == 0 {
                continue;
            }
            let mut p = va & !(PAGE - 1);
            while p < va.saturating_add(len) {
                pages.insert(p);
                p += PAGE;
            }
        }
        let mut seen = Vec::with_capacity(pages.len());
        for p in pages {
            let len = readable(p);
            if len == 0 {
                self.stats.unreadable_pages += 1;
                continue;
            }
            let bytes = read_host(p, len);
            let hash = *blake3::hash(&bytes).as_bytes();
            let store = match self.pages.get(&p) {
                None => {
                    self.stats.pages += 1;
                    self.pages.insert(p, Page { hash, gpu_written: false });
                    true
                }
                Some(page) => page.hash != hash,
            };
            if store {
                if self.stats.memory_bytes + len > self.cap {
                    self.stats.stopped = Some(format!("memory cap ({} MiB) reached at run {}", self.cap >> 20, self.run));
                    return Vec::new();
                }
                if let Some(page) = self.pages.get_mut(&p) {
                    page.hash = hash;
                }
                self.write_page(p, &bytes);
            }
            seen.push((p, hash));
        }
        self.count(&pops);
        put_run(&mut self.ops, self.run, &pops);
        self.run += 1;
        self.stats.runs = self.run;
        seen
    }

    fn unknown(&mut self, raw: u64) {
        self.stats.unknown += 1;
        if self.stats.unknown_samples.len() < 16 {
            self.stats.unknown_samples.push(raw);
        }
    }

    /// Counts a recorded run's ops and what they reference.
    fn count(&mut self, pops: &[POp]) {
        for op in pops {
            for r in op.refs() {
                match r {
                    Ref::Unknown { raw } => self.unknown(raw),
                    Ref::Texture { .. } => self.stats.texture_refs += 1,
                    Ref::Special { kind: SPECIAL_MEMORY_DIRECTORY | SPECIAL_FAULTS, .. } => self.stats.memory_directory_refs += 1,
                    _ => {}
                }
            }
            self.stats.ops += 1;
            match op {
                POp::Dispatch { .. } => self.stats.dispatches += 1,
                POp::DispatchIndirect { .. } => self.stats.indirect += 1,
                POp::BufferCopy { .. } => self.stats.buffer_copies += 1,
                POp::Unsupported { kind } => *self.stats.unsupported.entry(*kind).or_default() += 1,
            }
        }
    }

    fn write_page(&mut self, va: u64, bytes: &[u8]) {
        if self.memory_bin.write_all(bytes).is_err() {
            self.stats.stopped = Some("memory.bin could not be written".into());
            return;
        }
        put_u64(&mut self.memory_idx, va);
        put_u32(&mut self.memory_idx, bytes.len() as u32);
        put_u32(&mut self.memory_idx, self.run);
        put_u64(&mut self.memory_idx, self.memory_at);
        put_u32(&mut self.memory_idx, 0);
        self.memory_at += bytes.len() as u64;
        self.stats.memory_bytes += bytes.len() as u64;
        self.stats.page_versions += 1;
    }

    /// After the run: pages whose bytes it changed are the GPU's (part of
    /// the reference), and their new hashes are what the CPU is compared to.
    pub(crate) fn after(&mut self, seen: &[(u64, [u8; 32])]) {
        for &(p, before) in seen {
            let len = readable(p);
            if len == 0 {
                continue;
            }
            let hash = *blake3::hash(&read_host(p, len)).as_bytes();
            if let Some(page) = self.pages.get_mut(&p) {
                page.gpu_written |= hash != before;
                page.hash = hash;
            }
        }
    }

    /// Writes the reference (pages the GPU wrote, unchanged by the CPU
    /// since) and every remaining file.
    pub(crate) fn finish(mut self, m: Option<&MetalBackend>) -> std::io::Result<Stats> {
        self.memory_bin.flush()?;
        let mut ref_idx = Vec::new();
        let mut ref_bin = std::io::BufWriter::new(std::fs::File::create(self.dir.join("reference").join("memory.bin"))?);
        let mut at = 0u64;
        let mut written: Vec<u64> = self.pages.iter().filter(|(_, p)| p.gpu_written).map(|(&va, _)| va).collect();
        written.sort_unstable();
        for va in written {
            let len = readable(va);
            if len == 0 {
                continue;
            }
            let bytes = read_host(va, len);
            if *blake3::hash(&bytes).as_bytes() != self.pages[&va].hash {
                self.stats.cpu_changed_after += 1;
                continue;
            }
            ref_bin.write_all(&bytes)?;
            put_u64(&mut ref_idx, va);
            put_u32(&mut ref_idx, len as u32);
            put_u32(&mut ref_idx, u32::MAX);
            put_u64(&mut ref_idx, at);
            put_u32(&mut ref_idx, 0);
            at += len;
            self.stats.reference_pages += 1;
        }
        ref_bin.flush()?;
        drop(ref_bin);
        if let (Some(m), Some(start)) = (m, self.remaps_at_start) {
            self.stats.remaps = m.chunks().stats.remaps_under_spans.saturating_sub(start);
        }
        let mut pipelines = b"VPKPIP01".to_vec();
        put_u32(&mut pipelines, self.stats.pipelines);
        pipelines.extend_from_slice(&self.pipelines_bin);
        let files: [(&str, &[u8]); 6] = [
            ("ops.bin", &self.ops),
            ("memory.idx", &self.memory_idx),
            ("heaps.bin", &self.heaps_bin),
            ("resources.bin", &self.resources_bin),
            ("pipelines.bin", &pipelines),
            ("reference/memory.idx", &ref_idx),
        ];
        let mut hashes = Vec::new();
        for (name, bytes) in files {
            std::fs::write(self.dir.join(name), bytes)?;
            hashes.push((name, blake3::hash(bytes).to_hex().to_string()));
        }
        std::fs::write(self.dir.join("manifest.json"), self.manifest(&hashes))?;
        Ok(self.stats)
    }

    fn manifest(&self, hashes: &[(&str, String)]) -> String {
        fn esc(s: &str) -> String {
            s.chars()
                .flat_map(|c| match c {
                    '"' => "\\\"".chars().collect::<Vec<_>>(),
                    '\\' => "\\\\".chars().collect(),
                    c if (c as u32) < 0x20 => format!("\\u{:04x}", c as u32).chars().collect(),
                    c => vec![c],
                })
                .collect()
        }
        let s = &self.stats;
        let mut env: Vec<(String, String)> = std::env::vars().filter(|(k, _)| k.starts_with("VICTORY_")).collect();
        env.sort();
        let env = env.iter().map(|(k, v)| format!("    \"{}\": \"{}\"", esc(k), esc(v))).collect::<Vec<_>>().join(",\n");
        let mut unsupported: Vec<_> = s.unsupported.iter().collect();
        unsupported.sort();
        let unsupported = unsupported.iter().map(|(k, v)| format!("\"{k}\": {v}")).collect::<Vec<_>>().join(", ");
        let samples = s.unknown_samples.iter().map(|v| format!("\"{v:#x}\"")).collect::<Vec<_>>().join(", ");
        let hashes = hashes.iter().map(|(n, h)| format!("    \"{n}\": \"{h}\"")).collect::<Vec<_>>().join(",\n");
        format!(
            "{{\n  \"format\": \"victory-frame-package\",\n  \"version\": {VERSION},\n  \"frame\": {},\n  \"crate_version\": \"{}\",\n  \"commit\": \"{}\",\n  \"debug_assertions\": {},\n  \"diagnostics\": {},\n  \"msc_cache_key\": \"msc-4.0.3/bounds-check/v1\",\n  \"runs\": {},\n  \"ops\": {},\n  \"dispatches\": {},\n  \"indirect_dispatches\": {},\n  \"buffer_copies\": {},\n  \"unsupported_ops_by_kind\": {{{unsupported}}},\n  \"unknown_addresses\": {},\n  \"heap_unknown\": {},\n  \"unknown_samples\": [{samples}],\n  \"texture_refs_not_recorded\": {},\n  \"memory_directory_refs\": {},\n  \"pipelines\": {},\n  \"pipelines_without_source\": {},\n  \"pages\": {},\n  \"page_versions\": {},\n  \"unreadable_pages\": {},\n  \"memory_bytes\": {},\n  \"cap_bytes\": {},\n  \"heap_entries_written\": {},\n  \"reference_pages\": {},\n  \"pages_changed_by_cpu_after_last_run\": {},\n  \"remaps_under_spans_during_frame\": {},\n  \"stopped\": {},\n  \"files\": {{\n{hashes}\n  }},\n  \"env\": {{\n{env}\n  }}\n}}\n",
            self.frame,
            env!("CARGO_PKG_VERSION"),
            option_env!("VICTORY_GIT_COMMIT").unwrap_or("unknown"),
            cfg!(debug_assertions),
            cfg!(feature = "diagnostics"),
            s.runs,
            s.ops,
            s.dispatches,
            s.indirect,
            s.buffer_copies,
            s.unknown,
            s.heap_unknown,
            s.texture_refs,
            s.memory_directory_refs,
            s.pipelines,
            s.pipelines_without_source,
            s.pages,
            s.page_versions,
            s.unreadable_pages,
            s.memory_bytes,
            self.cap,
            s.heap_entries,
            s.reference_pages,
            s.cpu_changed_after,
            s.remaps,
            s.stopped.as_deref().map_or_else(|| "null".to_string(), |r| format!("\"{}\"", esc(r))),
        )
    }
}
