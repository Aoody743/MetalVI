//! P1 of plan v3 (stage P): the independent binding layer for compute
//! dispatches, behind `VICTORY_METAL_PREP=off|shadow|on` (default off).
//!
//! What it owns, apart from the D3D12-era record step (`draw.rs`
//! `metal_stage`: `sync_heap`, `refresh_slot`, `shadow::buffer_extent`):
//! - its own Metal descriptor heap per title heap ([`MetalHeap`] shape:
//!   entries, `SLOT_STRIDES` table), written from the title's descriptor
//!   writes ([`written`]: at once for buffer views over memory that is
//!   already wrapped, else at the next dispatch that binds the heap);
//! - reverse indices title memory (2 MiB granules) → slots and resource →
//!   slots, so a remap or a release makes only the slots it touches stale
//!   (texture slots: any remap, conservatively);
//! - the root parameters (user data, root descriptors, one-view tables)
//!   from the title's root state, without `draw.rs`;
//! - title buffers by address ([`buffer_made`]), instead of the shadow
//!   module's table.
//!
//! `shadow`: the old path's result is what runs; this layer's is computed
//! next to it and compared item by item ([`compare`]), mismatches counted
//! by category with the first few logged. `on`: compute dispatches run on
//! this layer's result (draws keep the old path); dispatches it does not
//! support (the title memory directory) fall back to the old path, counted.
//!
//! Locks: [`PREP`] is a leaf (nothing else is taken while it is held, no
//! Metal or Wine call is made under it). Remaps are queued (the runtime
//! calls [`remapped`] holding its page table) and applied at the next bind.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, MutexGuard};

use victory_metal::Buffer;
use victory_metal::ir::{DescriptorEntry, TopLevel};
use victory_shader::RootType;

use super::{Drop, MetalBackend, MetalHeap, RootSlot, root_layout};
use crate::backend::metal_heap::Missing;
use crate::backend::metal_views::{Window, placed_address, srv_window, uav_window};
use crate::descriptor::View;
use crate::draw::{MetalStage, Root};
use crate::object::This;
use crate::pipeline::Translated;

#[cfg(test)]
#[path = "metal_prep_tests.rs"]
mod tests;

/// What `VICTORY_METAL_PREP` asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Off = 0,
    Shadow = 1,
    On = 2,
}

const UNSET: u8 = u8::MAX;
static MODE: AtomicU8 = AtomicU8::new(UNSET);
static EAGER: AtomicU8 = AtomicU8::new(UNSET);

pub(crate) fn mode() -> Mode {
    let mut v = MODE.load(Relaxed);
    if v == UNSET {
        let m = match std::env::var("VICTORY_METAL_PREP").as_deref() {
            Ok("shadow") => Mode::Shadow,
            Ok("on") => Mode::On,
            Ok(other) if !other.is_empty() && other != "off" && other != "0" => {
                tracing::warn!(value = other, "VICTORY_METAL_PREP: unknown value, off");
                Mode::Off
            }
            _ => Mode::Off,
        };
        if m != Mode::Off {
            tracing::warn!(mode = ?m, "VICTORY_METAL_PREP: independent compute bindings (P1)");
        }
        if m == Mode::On && std::env::var_os("VICTORY_METAL_PACKAGE_FRAME").is_some() {
            // The frame package (T3) snapshots the record step's heaps only.
            tracing::warn!("VICTORY_METAL_PREP=on: a frame package does not record this layer's heaps; record with off or shadow");
        }
        let _ = MODE.compare_exchange(UNSET, m as u8, Relaxed, Relaxed);
        v = MODE.load(Relaxed);
    }
    match v {
        1 => Mode::Shadow,
        2 => Mode::On,
        _ => Mode::Off,
    }
}

/// Whether descriptor writes resolve buffer views over wrapped memory at
/// once (`VICTORY_METAL_PREP_EAGER=0`: all at the next bind).
fn eager() -> bool {
    let mut v = EAGER.load(Relaxed);
    if v == UNSET {
        let on = std::env::var("VICTORY_METAL_PREP_EAGER").as_deref() != Ok("0");
        let _ = EAGER.compare_exchange(UNSET, u8::from(on), Relaxed, Relaxed);
        v = EAGER.load(Relaxed);
    }
    v == 1
}

#[cfg(test)]
pub(crate) fn set_mode(m: Mode) {
    MODE.store(m as u8, Relaxed);
}

#[cfg(test)]
pub(crate) fn set_eager(on: bool) {
    EAGER.store(u8::from(on), Relaxed);
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------------------
// Counters.

/// What the layer did (see [`STAT_NAMES`]).
#[derive(Clone, Copy, Debug)]
pub(crate) enum Stat {
    /// Compute stages bound by this layer (shadow or on).
    Bound,
    /// On mode: stages it did not support, bound by the old path instead.
    Fallback,
    /// Title descriptor writes noted.
    Writes,
    /// ... of which resolved at the write (buffer views over wrapped memory).
    WriteResolved,
    /// Slots resolved at a bind, from the heap's written-but-unresolved set.
    FlushResolved,
    /// Slots resolved when a dispatch used them (stale, missing, unwritten).
    UseResolved,
    /// Used slots that were clean: nothing to do.
    Hits,
    /// Clean slots made stale by a remap or a release.
    Invalidated,
    /// Heaps of our own made.
    HeapsMade,
    /// Shadow comparisons made.
    Compared,
}

const STATS: usize = 10;
pub(crate) const STAT_NAMES: [&str; STATS] =
    ["bound", "fallback", "writes", "write_resolved", "flush_resolved", "use_resolved", "hits", "invalidated", "heaps_made", "compared"];
static STAT: [AtomicU64; STATS] = [const { AtomicU64::new(0) }; STATS];

fn bump(s: Stat) -> u64 {
    STAT[s as usize].fetch_add(1, Relaxed) + 1
}

pub(crate) fn stats() -> [u64; STATS] {
    STAT.each_ref().map(|c| c.load(Relaxed))
}

/// Shadow mismatch categories (see [`CAT_NAMES`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Cat {
    /// The user-data constants in the top-level buffer.
    UserData,
    /// A root CBV/SRV/UAV's GPU address.
    RootAddress,
    /// ... the same title memory through another wrapper (informational).
    RootWrapper,
    /// A one-view table entry (title root SRV/UAV).
    TableEntry,
    TableWrapper,
    /// A heap slot a dispatch reads (its table or root-indexed views).
    HeapEntry,
    HeapWrapper,
    /// A used slot's stride table words.
    Stride,
    /// A heap slot no table names (the rotating sample of both heaps).
    Bindless,
    /// The dispatch outcome (empty-slot class) or the refusal.
    Outcome,
    /// Where the program reads its first group's number.
    GroupBase,
    /// A heap bound by one path and not the other.
    HeapBound,
    /// The old path refused, this layer bound.
    OldFailed,
    /// This layer refused, the old path bound.
    PrepFailed,
    /// Other top-level bytes (victory's own root descriptors, static samplers).
    TopOther,
    /// Not compared: this layer does not support the dispatch (reason logged).
    Unsupported,
}

const CATS: usize = 16;
pub(crate) const CAT_NAMES: [&str; CATS] = [
    "user_data",
    "root_address",
    "root_wrapper",
    "table_entry",
    "table_wrapper",
    "heap_entry",
    "heap_wrapper",
    "stride",
    "bindless",
    "outcome",
    "group_base",
    "heap_bound",
    "old_failed",
    "prep_failed",
    "top_other",
    "unsupported",
];
static MISMATCH: [AtomicU64; CATS] = [const { AtomicU64::new(0) }; CATS];
/// Samples logged per category.
const SAMPLES: u64 = 3;

pub(crate) fn mismatches() -> [u64; CATS] {
    MISMATCH.each_ref().map(|c| c.load(Relaxed))
}

/// Mismatches that are not informational (wrappers) or not comparisons.
pub(crate) fn real_mismatches() -> u64 {
    let m = mismatches();
    [Cat::RootWrapper, Cat::TableWrapper, Cat::HeapWrapper, Cat::Unsupported]
        .iter()
        .fold(m.iter().sum::<u64>(), |sum, c| sum - m[*c as usize])
}

fn mismatch(cat: Cat, pipeline: &str, detail: impl FnOnce() -> String) {
    let n = MISMATCH[cat as usize].fetch_add(1, Relaxed) + 1;
    if n <= SAMPLES {
        let detail = detail();
        #[cfg(test)]
        lock(&SAMPLES_SEEN).push(format!("{} #{n}: {detail}", CAT_NAMES[cat as usize]));
        tracing::warn!(category = CAT_NAMES[cat as usize], n, pipeline, %detail, "Metal prep shadow: mismatch");
    }
}

/// The samples logged (tests: no log subscriber there).
#[cfg(test)]
pub(crate) static SAMPLES_SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());

fn named<const N: usize>(names: &[&'static str; N], values: [u64; N]) -> Vec<(&'static str, u64)> {
    names.iter().copied().zip(values).filter(|(_, v)| *v != 0).collect()
}

/// Logs the totals every 2^k (from 1024 on).
fn report(n: u64, what: &str) {
    if n >= 1024 && n.is_power_of_two() {
        tracing::info!(n, what, stats = ?named(&STAT_NAMES, stats()), mismatches = ?named(&CAT_NAMES, mismatches()), "Metal prep: totals");
    }
}

// ---------------------------------------------------------------------------
// Title buffers by address (buffer ownership without `shadow.rs`).

static BUFFERS: Mutex<BTreeMap<(u64, usize), u64>> = Mutex::new(BTreeMap::new());

/// A title buffer exists at `start` (`len` bytes): `resource.rs`, where it
/// tells the shadow module.
pub(crate) fn buffer_made(resource: usize, start: u64, len: u64) {
    if mode() == Mode::Off {
        return;
    }
    lock(&BUFFERS).insert((start, resource), start.saturating_add(len));
}

/// The title released it.
pub(crate) fn buffer_gone(resource: usize, start: u64) {
    if mode() == Mode::Off {
        return;
    }
    lock(&BUFFERS).remove(&(start, resource));
}

/// The title buffer holding `va` as (start, end): the one starting last
/// at or below it among the 64 nearest (as `shadow::buffer_extent`).
pub(crate) fn buffer_extent(va: u64) -> Option<(u64, u64)> {
    lock(&BUFFERS).range(..=(va, usize::MAX)).rev().take(64).find(|(_, end)| **end > va).map(|(k, end)| (k.0, *end))
}

// ---------------------------------------------------------------------------
// Heaps of our own.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum State {
    /// Never written by the title (the entry is null, as the title's slot).
    #[default]
    Empty,
    /// Written, not resolved yet: resolved at the next bind of the heap.
    Pending,
    /// Resolved from the descriptor the title wrote last, nothing missing.
    Clean,
    /// Resolved with something missing: retried when a dispatch uses it.
    Missing,
    /// Clean once, then its memory remapped or its resource released:
    /// resolved again when a dispatch uses it (as the old path's slots,
    /// which only change when the title writes them or a dispatch uses them).
    Stale,
}

#[derive(Clone, Copy, Debug, Default)]
struct Slot {
    /// Bumped by every write and invalidation: a resolve done meanwhile
    /// does not mark the slot.
    seq: u32,
    state: State,
}

/// What a slot's entry depends on besides the descriptor itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dep {
    /// Title memory `[start, end)` (a buffer view; `resource`: its buffer
    /// object, 0 for constant and placed views).
    Range { start: u64, end: u64, resource: usize },
    /// A texture (0: a placed one, known by address only).
    Texture { resource: usize },
}

struct HeapState {
    count: u32,
    samplers: bool,
    gpu: Option<MetalHeap>,
    slots: Vec<Slot>,
    deps: HashMap<u32, Dep>,
    pending: HashSet<u32>,
}

impl HeapState {
    fn new(key: usize) -> Self {
        let heap = crate::descriptor::heap(key as This);
        let count = heap.count();
        Self {
            count,
            samplers: heap.samplers(),
            gpu: None,
            slots: vec![Slot::default(); count as usize],
            deps: HashMap::new(),
            pending: HashSet::new(),
        }
    }
}

/// 2 MiB granules of title memory in the reverse index.
const GRANULE_SHIFT: u32 = 21;
/// Views over more granules than this are checked on every remap.
const HUGE_GRANULES: u64 = 64;

type Key = (usize, u32);

#[derive(Default)]
struct Prep {
    heaps: HashMap<usize, HeapState>,
    granules: HashMap<u64, HashSet<Key>>,
    huge: HashSet<Key>,
    resources: HashMap<usize, HashSet<Key>>,
    textures: HashSet<Key>,
}

static PREP: Mutex<Option<Prep>> = Mutex::new(None);
static REMAPS: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
static REMAPPED: AtomicBool = AtomicBool::new(false);

fn with<R>(f: impl FnOnce(&mut Prep) -> R) -> R {
    let mut guard = lock(&PREP);
    f(guard.get_or_insert_with(Prep::default))
}

fn granules(start: u64, end: u64) -> Option<std::ops::RangeInclusive<u64>> {
    (end > start).then(|| (start >> GRANULE_SHIFT)..=((end - 1) >> GRANULE_SHIFT))
}

impl Prep {
    fn index(&mut self, key: Key, dep: Dep) {
        match dep {
            Dep::Range { start, end, resource } => {
                if let Some(g) = granules(start, end) {
                    if g.end() - g.start() >= HUGE_GRANULES {
                        self.huge.insert(key);
                    } else {
                        for g in g {
                            self.granules.entry(g).or_default().insert(key);
                        }
                    }
                }
                if resource != 0 {
                    self.resources.entry(resource).or_default().insert(key);
                }
            }
            Dep::Texture { resource } => {
                self.textures.insert(key);
                if resource != 0 {
                    self.resources.entry(resource).or_default().insert(key);
                }
            }
        }
    }

    fn unindex(&mut self, key: Key, dep: Dep) {
        let out_of = |map: &mut HashMap<u64, HashSet<Key>>, at: u64| {
            if let Some(set) = map.get_mut(&at) {
                set.remove(&key);
                if set.is_empty() {
                    map.remove(&at);
                }
            }
        };
        match dep {
            Dep::Range { start, end, resource } => {
                if let Some(g) = granules(start, end) {
                    if g.end() - g.start() >= HUGE_GRANULES {
                        self.huge.remove(&key);
                    } else {
                        for g in g {
                            out_of(&mut self.granules, g);
                        }
                    }
                }
                if resource != 0
                    && let Some(set) = self.resources.get_mut(&resource)
                {
                    set.remove(&key);
                    if set.is_empty() {
                        self.resources.remove(&resource);
                    }
                }
            }
            Dep::Texture { resource } => {
                self.textures.remove(&key);
                if resource != 0
                    && let Some(set) = self.resources.get_mut(&resource)
                {
                    set.remove(&key);
                    if set.is_empty() {
                        self.resources.remove(&resource);
                    }
                }
            }
        }
    }

    fn set_dep(&mut self, key: Key, dep: Option<Dep>) {
        let Some(h) = self.heaps.get_mut(&key.0) else { return };
        let old = match dep {
            Some(d) => h.deps.insert(key.1, d),
            None => h.deps.remove(&key.1),
        };
        if old == dep {
            return;
        }
        if let Some(o) = old {
            self.unindex(key, o);
        }
        if let Some(d) = dep {
            self.index(key, d);
        }
    }

    fn invalidate(&mut self, key: Key) {
        if let Some(s) = self.heaps.get_mut(&key.0).and_then(|h| h.slots.get_mut(key.1 as usize)) {
            s.seq = s.seq.wrapping_add(1);
            if s.state == State::Clean {
                s.state = State::Stale;
                bump(Stat::Invalidated);
            }
        }
    }

    fn remapped(&mut self, at: u64, len: u64) {
        let end = at.saturating_add(len);
        let Some(range) = granules(at, end) else { return };
        let mut hit: Vec<Key> = self.textures.iter().copied().collect();
        hit.extend(self.huge.iter().copied());
        for g in range {
            if let Some(set) = self.granules.get(&g) {
                hit.extend(set.iter().copied());
            }
        }
        hit.sort_unstable();
        hit.dedup();
        for key in hit {
            let overlaps = match self.heaps.get(&key.0).and_then(|h| h.deps.get(&key.1)) {
                Some(Dep::Range { start, end: e, .. }) => *start < end && at < *e,
                Some(Dep::Texture { .. }) => true,
                None => false,
            };
            if overlaps {
                self.invalidate(key);
            }
        }
    }

    fn resource_gone(&mut self, resource: usize) {
        if let Some(set) = self.resources.remove(&resource) {
            for key in set {
                self.invalidate(key);
            }
        }
    }

    fn heap_gone(&mut self, heap: usize) {
        if let Some(h) = self.heaps.remove(&heap) {
            for (index, dep) in h.deps {
                self.unindex((heap, index), dep);
            }
        }
    }

    /// Takes a resolve done at `seq` into slot `key`: written into our heap
    /// and marked, unless the slot was written or invalidated meanwhile.
    fn commit(&mut self, key: Key, seq: u32, r: &Resolved) -> bool {
        let Some(h) = self.heaps.get_mut(&key.0) else { return false };
        let Some(gpu) = h.gpu.as_ref() else { return false };
        let Some(slot) = h.slots.get_mut(key.1 as usize) else { return false };
        if slot.seq != seq || key.1 >= gpu.count {
            return false;
        }
        // SAFETY: index < count, the entries' memory holds >= count entries.
        unsafe { gpu.entries.add(key.1 as usize).write(r.entry) };
        if let Some((_, strides)) = gpu.strides {
            // SAFETY: the strides table holds SLOT_STRIDES_DWORDS dwords (`new_heap`).
            unsafe { crate::descriptor::host::write_stride_slot(strides, key.1, r.stride.0, r.stride.1) };
        }
        slot.state = if r.missing.is_none() { State::Clean } else { State::Missing };
        h.pending.remove(&key.1);
        self.set_dep(key, r.dep);
        true
    }
}

/// The title wrote descriptor `index` of shader-visible heap `heap`
/// (`metal::note_slot_dirty`, on the writing thread).
pub(crate) fn written(heap: usize, index: u32, handle: u64) {
    if mode() == Mode::Off || handle == 0 {
        return;
    }
    bump(Stat::Writes);
    let noted = with(|p| {
        let h = p.heaps.entry(heap).or_insert_with(|| HeapState::new(heap));
        let ready = h.gpu.is_some() && !h.samplers;
        let slot = h.slots.get_mut(index as usize)?;
        slot.seq = slot.seq.wrapping_add(1);
        slot.state = State::Pending;
        let seq = slot.seq;
        h.pending.insert(index);
        Some((seq, ready))
    });
    if let Some((seq, true)) = noted
        && eager()
        && let Some(r) = resolve_resident(handle)
    {
        let key = (heap, index);
        if with(|p| p.commit(key, seq, &r)) {
            bump(Stat::WriteResolved);
        }
    }
}

/// Title memory `[at, at + len)` was remapped (`metal::title_memory_remapped`,
/// with the runtime's page table held): queued.
pub(crate) fn remapped(at: u64, len: u64) {
    if mode() == Mode::Off || len == 0 {
        return;
    }
    let mut q = lock(&REMAPS);
    if let Some(last) = q.last_mut()
        && last.0 + last.1 == at
    {
        last.1 += len;
    } else if q.len() < 4096 {
        q.push((at, len));
    } else {
        // Too many to keep apart: one covering range.
        let lo = q.iter().map(|r| r.0).min().unwrap_or(at).min(at);
        let hi = q.iter().map(|r| r.0 + r.1).max().unwrap_or(at + len).max(at + len);
        q.clear();
        q.push((lo, hi - lo));
    }
    REMAPPED.store(true, Relaxed);
}

/// A title resource was released (`metal::forget_resource`).
pub(crate) fn resource_gone(resource: usize) {
    if mode() == Mode::Off {
        return;
    }
    with(|p| p.resource_gone(resource));
}

/// A title heap was released (`metal::forget_heap`).
pub(crate) fn heap_gone(heap: usize) {
    if mode() == Mode::Off {
        return;
    }
    with(|p| p.heap_gone(heap));
}

fn apply_remaps() {
    if !REMAPPED.swap(false, Relaxed) {
        return;
    }
    let q = std::mem::take(&mut *lock(&REMAPS));
    with(|p| {
        for (at, len) in q {
            p.remapped(at, len);
        }
    });
}

// ---------------------------------------------------------------------------
// Resolving.

struct Resolved {
    entry: DescriptorEntry,
    stride: (u32, u64),
    missing: Option<Missing>,
    dep: Option<Dep>,
}

/// The buffer window a buffer view covers: (title address, bytes, typed,
/// buffer object).
fn buffer_window(view: &View) -> Option<(u64, Window, usize)> {
    match view {
        View::Constant(address, size) => Some((*address, Window { offset: 0, bytes: u64::from(*size), typed_format: None }, 0)),
        View::Shader(r, desc) => {
            let (data, width) = crate::resource::buffer_bytes(*r as This)?;
            let w = match desc {
                None => Window { offset: 0, bytes: width, typed_format: None },
                Some(d) => srv_window(d)?,
            };
            Some((data as u64, w, *r))
        }
        View::Unordered(r, _, desc) => {
            let (data, width) = crate::resource::buffer_bytes(*r as This)?;
            let w = match desc {
                None => Window { offset: 0, bytes: width, typed_format: None },
                Some(d) => uav_window(d)?,
            };
            Some((data as u64, w, *r))
        }
        View::Placed(unordered, resource, desc) if resource[0] == 1 => {
            let w = if *unordered { uav_window(&desc[..10]) } else { srv_window(&desc[..10]) }?;
            Some((placed_address(desc), w, 0))
        }
        _ => None,
    }
}

fn dep_of(view: &View) -> Option<Dep> {
    if let Some((base, w, resource)) = buffer_window(view) {
        let start = base.wrapping_add(w.offset);
        return Some(Dep::Range { start, end: start.saturating_add(w.bytes), resource });
    }
    match view {
        View::Shader(r, _) | View::Unordered(r, _, _) => {
            if let Some((data, width)) = crate::resource::buffer_bytes(*r as This) {
                // A buffer view whose description we cannot read: the buffer.
                Some(Dep::Range { start: data as u64, end: data as u64 + width, resource: *r })
            } else {
                Some(Dep::Texture { resource: *r })
            }
        }
        View::Placed(..) => Some(Dep::Texture { resource: 0 }),
        View::Constant(..) | View::Sampler(_) | View::ConsoleSampler(_) => None,
    }
}

fn resolve(m: &MetalBackend, handle: u64) -> Resolved {
    let (entry, stride, missing, view) = m.resolve_view(handle);
    Resolved { entry, stride, missing, dep: view.as_ref().and_then(dep_of) }
}

/// At a descriptor write: a raw or structured buffer view (or a constant
/// buffer) over memory a wrapper already holds, without making one (no
/// span, no Metal call; the chunks lock only if free). The same entry
/// `metal_heap::entry` makes.
fn resolve_resident(handle: u64) -> Option<Resolved> {
    let m = backend()?;
    let view = crate::descriptor::view(handle)?;
    let (base, window, resource) = buffer_window(&view)?;
    if window.typed_format.is_some() {
        return None;
    }
    let va = base + window.offset;
    let entry = if window.bytes == 0 {
        DescriptorEntry::NULL
    } else {
        let chunks = m.chunks.try_lock().ok()?;
        let (buffer, offset) = chunks.resident_window(va, window.bytes)?;
        DescriptorEntry::buffer(buffer.gpu_address + offset, window.bytes)
    };
    let stride = (crate::descriptor::host::view_stride_word(&view), crate::descriptor::host::view_address_word(&view));
    Some(Resolved { entry, stride, missing: None, dep: Some(Dep::Range { start: va, end: va.saturating_add(window.bytes), resource }) })
}

fn backend() -> Option<&'static MetalBackend> {
    #[cfg(test)]
    if let Some(m) = tests::backend() {
        return Some(m);
    }
    super::BACKEND.get().and_then(Option::as_ref)
}

/// Our heap for title heap `key`, made on first use, with every slot the
/// title wrote since resolved: (its buffer, its stride table's address).
fn heap(m: &MetalBackend, key: usize) -> Option<(Buffer, u64)> {
    let need = with(|p| {
        let h = p.heaps.entry(key).or_insert_with(|| HeapState::new(key));
        h.gpu.is_none().then_some((h.count, h.samplers))
    });
    if let Some((count, samplers)) = need {
        // One maker at a time (heaps are never freed: two threads binding
        // a new heap at once must not both make one). Taken before the
        // chunks lock (`new_heap` adds to the residency set), never under it.
        static MAKING: Mutex<()> = Mutex::new(());
        let _making = lock(&MAKING);
        if with(|p| p.heaps.get(&key).is_some_and(|h| h.gpu.is_none())) {
            let made = m.new_heap(count, samplers)?;
            bump(Stat::HeapsMade);
            with(|p| {
                if let Some(h) = p.heaps.get_mut(&key)
                    && h.gpu.is_none()
                {
                    h.gpu = Some(made);
                }
            });
        }
    }
    flush(m, key);
    with(|p| {
        let gpu = p.heaps.get(&key)?.gpu.as_ref()?;
        Some((gpu.buffer.clone(), gpu.strides.as_ref().map_or(0, |(b, _)| b.gpu_address)))
    })
}

/// Resolves the slots of `key` written since they were last resolved.
fn flush(m: &MetalBackend, key: usize) {
    const BATCH: usize = 4096;
    let title = crate::descriptor::heap(key as This);
    loop {
        let batch: Vec<(u32, u32)> = with(|p| {
            let Some(h) = p.heaps.get_mut(&key) else { return Vec::new() };
            let take: Vec<u32> = h.pending.iter().copied().take(BATCH).collect();
            take.into_iter()
                .filter_map(|i| {
                    h.pending.remove(&i);
                    h.slots.get(i as usize).map(|s| (i, s.seq))
                })
                .collect()
        });
        if batch.is_empty() {
            return;
        }
        let resolved: Vec<(u32, u32, Resolved)> = batch
            .into_iter()
            .filter_map(|(i, seq)| Some((i, seq, resolve(m, title.handle(i)?))))
            .collect();
        with(|p| {
            for (i, seq, r) in &resolved {
                if p.commit((key, *i), *seq, r) {
                    bump(Stat::FlushResolved);
                }
            }
        });
    }
}

/// A slot a dispatch reads: nothing to do when clean, else resolved now.
/// Why its entry is empty, if it is.
fn use_slot(m: &MetalBackend, key: Key, handle: u64) -> Option<Missing> {
    let mut missing = None;
    for _ in 0..2 {
        let seq = with(|p| {
            let s = p.heaps.get(&key.0)?.slots.get(key.1 as usize)?;
            (s.state != State::Clean).then_some(s.seq)
        });
        let Some(seq) = seq else {
            bump(Stat::Hits);
            return None;
        };
        let r = resolve(m, handle);
        bump(Stat::UseResolved);
        missing = r.missing;
        if with(|p| p.commit(key, seq, &r)) {
            break;
        }
    }
    missing
}

// ---------------------------------------------------------------------------
// Binding.

/// One compute stage's inputs: the title's root state and the shader.
pub(crate) struct Input<'a> {
    pub(crate) name: &'a str,
    pub(crate) t: &'a Translated,
    pub(crate) roots: &'a [Root],
    /// View heap, sampler heap (title heap objects; 0: none).
    pub(crate) heaps: [usize; 2],
    pub(crate) limit: u32,
    pub(crate) groups_xy: [u32; 2],
}

#[derive(Debug)]
pub(crate) enum Refusal {
    /// The dispatch cannot run (as the old path would refuse it).
    Drop(Drop),
    /// This layer does not bind it; the old path may.
    Unsupported(&'static str),
}

pub(crate) struct Bound {
    pub(crate) stage: MetalStage,
    /// The heap slots it reads (title heap, index).
    used: Vec<Key>,
}

#[cfg(test)]
static FAULT_USER_DATA: AtomicBool = AtomicBool::new(false);

/// The bindings of one compute stage from the title's state alone: the
/// top-level buffer from `top` (the pipeline's empty one), the heaps of our
/// own. Calls nothing of the D3D12-era record step.
pub(crate) fn bind(m: &MetalBackend, input: &Input<'_>, mut top: TopLevel) -> Result<Bound, Refusal> {
    use victory_shader::emit::{
        SLOT_DIAG, SLOT_FAULTS, SLOT_MEMORY, SLOT_STRIDES, SLOT_WAVE_SCRATCH, USER_DATA_SPACE, USER_DISPATCH_THREADS_X, USER_GROUP_BASE,
        USER_GROUP_COUNT_X, USER_GROUP_COUNT_Y, USER_VIEW_ADDRESS, USER_VIEW_BIAS,
    };
    let host = &input.t.host;
    if host.root_srvs.contains(&SLOT_MEMORY) {
        return Err(Refusal::Unsupported("memory_directory"));
    }
    bump(Stat::Bound);
    apply_remaps();
    crate::backend::metal_texture::drain_remap_partial();

    let heap_of = |handle: u64| {
        input.heaps.iter().filter(|&&h| h != 0).map(|&h| crate::descriptor::heap(h as This)).find(|h| h.index(handle).is_some())
    };
    let root = |slot: u8| input.roots.get(usize::from(slot));
    let constant = |slot: u8, dword: u32| match root(slot) {
        Some(Root::Constants(c)) => c.get(dword as usize).copied().unwrap_or(0),
        _ => 0,
    };

    // User data.
    let mut words = vec![0u32; host.user_data_dwords() as usize];
    for u in &host.user_data {
        let v = match u.root {
            RootType::Other(k) if k == USER_GROUP_BASE || k == USER_VIEW_BIAS => 0,
            RootType::Other(k) if k == USER_DISPATCH_THREADS_X => input.limit,
            RootType::Other(k) if k == USER_GROUP_COUNT_X => input.groups_xy[0],
            RootType::Other(k) if k == USER_GROUP_COUNT_Y => input.groups_xy[1],
            RootType::Other(k) if k == USER_VIEW_ADDRESS || k == USER_VIEW_ADDRESS + 1 => {
                // The title address of the buffer view at that dword of the table.
                let address = match root(u.slot) {
                    Some(&Root::Table(table)) => crate::descriptor::view(table + u64::from(u.dword) * 4)
                        .and_then(|v| crate::descriptor::host::buffer_range(&v))
                        .map_or(0, |r| r.0),
                    _ => 0,
                };
                if k == USER_VIEW_ADDRESS { address as u32 } else { (address >> 32) as u32 }
            }
            // The memory base (no directory here) or a host word unknown to
            // this build: the old path refuses the dispatch too.
            RootType::Other(_) => return Err(Refusal::Drop(Drop::MemoryDirectory)),
            RootType::Table => match root(u.slot) {
                Some(&Root::Table(handle)) => heap_of(handle).map_or(0, |h| (handle - h.start) as u32),
                _ => 0,
            },
            RootType::Constants => constant(u.slot, u.dword),
            _ => 0,
        };
        if let Some(w) = words.get_mut(u.index as usize) {
            *w = v;
        }
    }
    #[cfg(test)]
    if FAULT_USER_DATA.load(Relaxed)
        && let Some(w) = words.first_mut()
    {
        *w ^= 1;
    }
    let group_base_at = host
        .user_data
        .iter()
        .find(|u| u.root == RootType::Other(USER_GROUP_BASE))
        .and_then(|u| Some(top.constants_offset(USER_DATA_SPACE, 0)? as usize + u.index as usize * 4));
    if !words.is_empty() && !top.set_constants(USER_DATA_SPACE, 0, &words) {
        return Err(Refusal::Drop(Drop::UserDataLayout));
    }

    // Our heaps, up to date with the title's writes.
    let view_heap = if input.heaps[0] != 0 { heap(m, input.heaps[0]) } else { None };
    let sampler_heap = if input.heaps[1] != 0 { heap(m, input.heaps[1]) } else { None };
    let mut heap_buffer = view_heap.as_ref().map(|(b, _)| b.clone());
    let mut sampler_buffer = sampler_heap.as_ref().map(|(b, _)| b.clone());
    let strides_gpu = view_heap.as_ref().map_or(0, |(_, s)| *s);

    // Root views: title buffers bounded by the buffer that holds them.
    let mut tables = Vec::new();
    let wave_scratch_gpu = if host.root_uavs.contains(&SLOT_WAVE_SCRATCH) { m.wave_scratch_gpu() } else { 0 };
    let diag_gpu = if host.root_uavs.contains(&SLOT_DIAG) { super::diag::gpu(m) } else { 0 };
    {
        let mut chunks = m.chunks();
        for (i, r) in root_layout(host).into_iter().enumerate() {
            let slot = match r {
                RootSlot::Constants(_) => continue,
                RootSlot::Srv(s) if s == SLOT_STRIDES => {
                    top.set_u64(i, strides_gpu);
                    continue;
                }
                RootSlot::Uav(s) if s == SLOT_WAVE_SCRATCH => {
                    top.set_u64(i, wave_scratch_gpu);
                    continue;
                }
                RootSlot::Uav(s) if s == SLOT_DIAG => {
                    top.set_u64(i, diag_gpu);
                    continue;
                }
                RootSlot::Uav(s) if s == SLOT_FAULTS => {
                    top.set_u64(i, 0);
                    continue;
                }
                RootSlot::Cbv(s) | RootSlot::Srv(s) | RootSlot::Uav(s) | RootSlot::TableSrv(s) | RootSlot::TableUav(s) => s,
            };
            let va = match root(slot) {
                Some(Root::View(va)) => *va,
                _ => 0,
            };
            let extent = if va == 0 {
                0
            } else {
                match buffer_extent(va) {
                    Some((_, end)) => end - va,
                    None => victory_metal::chunks::committed_extent(va, 4 << 20),
                }
            };
            let gpu = if extent == 0 { 0 } else { chunks.window(va, extent) };
            match r {
                RootSlot::TableSrv(_) | RootSlot::TableUav(_) => {
                    let entry = if gpu == 0 { DescriptorEntry::NULL } else { DescriptorEntry::buffer(gpu, extent) };
                    if let Some(at) = top.offset_of(i) {
                        tables.push((at, entry));
                    }
                }
                _ => {
                    top.set_u64(i, gpu);
                }
            }
        }
    }

    // The slots the shader reads.
    let mut used = Vec::new();
    let mut worst: Option<Missing> = None;
    let mut read = |handle: u64, heap_buffer: &mut Option<Buffer>, sampler_buffer: &mut Option<Buffer>| {
        let Some(h) = heap_of(handle) else { return };
        let Some(index) = h.index(handle) else { return };
        let key = (std::ptr::from_ref(h) as usize, index);
        used.push(key);
        worst = worst.or(use_slot(m, key, handle));
        let ours = if key.0 == input.heaps[0] { view_heap.as_ref() } else { sampler_heap.as_ref() };
        if let Some((b, _)) = ours {
            if h.samplers() {
                *sampler_buffer = Some(b.clone());
            } else {
                *heap_buffer = Some(b.clone());
            }
        }
    };
    for u in &input.t.used {
        if u.root != RootType::Table {
            continue;
        }
        let Some(&Root::Table(table)) = root(u.slot) else { continue };
        read(table + u64::from(u.table_offset_dwords) * 4, &mut heap_buffer, &mut sampler_buffer);
    }
    // Views picked from the heap by a root constant (32-byte slots).
    for index in &host.root_indexed_views {
        let Some(&Root::Table(table)) = root(index.table_slot) else { continue };
        let c = constant(index.constant_slot, index.constant_dword);
        let slot = index.byte_offset.wrapping_add(c << index.shift) / 32;
        let Some(handle) = table.checked_add(u64::from(slot) * 32) else { continue };
        read(handle, &mut heap_buffer, &mut sampler_buffer);
    }
    let outcome = match worst {
        None => Drop::Ran,
        Some(Missing::Typed { .. }) => Drop::SlotTyped,
        Some(Missing::Unmapped) => Drop::SlotUnmapped,
        Some(_) => Drop::SlotMissing,
    };
    let n = stats()[Stat::Bound as usize];
    if mode() == Mode::On {
        report(n, "bound");
    }
    Ok(Bound {
        stage: MetalStage { top: top.bytes, tables, heap: heap_buffer, sampler_heap: sampler_buffer, outcome, group_base_at },
        used,
    })
}

/// On mode, for `draw.rs`: this layer's bindings, or `None` when it does
/// not support the stage (the caller binds it the old way; counted).
pub(crate) fn bind_on(m: &MetalBackend, input: &Input<'_>, top: TopLevel) -> Option<Result<MetalStage, Drop>> {
    match bind(m, input, top) {
        Ok(b) => Some(Ok(b.stage)),
        Err(Refusal::Drop(d)) => Some(Err(d)),
        Err(Refusal::Unsupported(why)) => {
            let n = bump(Stat::Fallback);
            if n.is_power_of_two() {
                tracing::info!(n, why, pipeline = input.name, "Metal prep: stage bound by the old path");
            }
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Shadow comparison.

fn read_entry(gpu: &MetalHeap, index: u32) -> Option<(DescriptorEntry, Option<[u32; 3]>)> {
    if index >= gpu.count {
        return None;
    }
    // SAFETY: index < count.
    let entry = unsafe { gpu.entries.add(index as usize).read_volatile() };
    let strides = gpu.strides.as_ref().map(|(_, p)| *p).and_then(|p| {
        use victory_shader::emit::{SLOT_STRIDES_COUNT, STRIDES_VIEW_ADDRESSES};
        (index < SLOT_STRIDES_COUNT).then(|| {
            let at = (STRIDES_VIEW_ADDRESSES + 2 * index) as usize;
            // SAFETY: inside the SLOT_STRIDES_DWORDS table.
            unsafe { [p.add(index as usize).read_volatile(), p.add(at).read_volatile(), p.add(at + 1).read_volatile()] }
        })
    });
    Some((entry, strides))
}

/// The old path's heap slot (its entry and stride words).
fn old_slot(m: &MetalBackend, key: Key) -> Option<(DescriptorEntry, Option<[u32; 3]>)> {
    read_entry(lock(&m.heaps).get(&key.0)?, key.1)
}

fn our_slot(key: Key) -> Option<(DescriptorEntry, Option<[u32; 3]>, State)> {
    with(|p| {
        let h = p.heaps.get(&key.0)?;
        let (e, s) = read_entry(h.gpu.as_ref()?, key.1)?;
        Some((e, s, h.slots.get(key.1 as usize)?.state))
    })
}

/// Whether two GPU addresses show the same title memory (diagnostics: a
/// linear walk, only on a mismatch).
fn same_memory(m: &MetalBackend, a: u64, b: u64) -> bool {
    if a == 0 || b == 0 {
        return false;
    }
    let chunks = m.chunks();
    let ta = chunks.title_va_of(a);
    ta.is_some() && ta == chunks.title_va_of(b)
}

fn entries_differ(m: &MetalBackend, a: &DescriptorEntry, b: &DescriptorEntry, real: Cat, wrapper: Cat, name: &str, what: &dyn Fn() -> String) {
    if a == b {
        return;
    }
    let cat = if a.texture_view_id == b.texture_view_id && a.metadata == b.metadata && same_memory(m, a.gpu_va, b.gpu_va) { wrapper } else { real };
    mismatch(cat, name, || format!("{}: old {a:x?} prep {b:x?}", what()));
}

/// Compares the old path's bindings of a compute stage with this layer's
/// (shadow mode; `template`: the pipeline's empty top level).
pub(crate) fn compare(m: &MetalBackend, input: &Input<'_>, template: &TopLevel, old: &Result<MetalStage, Drop>, new: &Result<Bound, Refusal>) {
    let n = bump(Stat::Compared);
    let name = input.name;
    match (old, new) {
        (_, Err(Refusal::Unsupported(why))) => mismatch(Cat::Unsupported, name, || (*why).to_string()),
        (Err(a), Err(Refusal::Drop(b))) => {
            if a != b {
                mismatch(Cat::Outcome, name, || format!("refused: old {a:?} prep {b:?}"));
            }
        }
        (Err(a), Ok(_)) => mismatch(Cat::OldFailed, name, || format!("old refused {a:?}")),
        (Ok(_), Err(Refusal::Drop(b))) => mismatch(Cat::PrepFailed, name, || format!("prep refused {b:?}")),
        (Ok(o), Ok(b)) => compare_bound(m, input, template, o, b, n),
    }
    report(n, "compared");
}

/// Victory's own root descriptors (not title root views).
const OWN_SLOTS: [u8; 5] = [
    victory_shader::emit::SLOT_STRIDES,
    victory_shader::emit::SLOT_MEMORY,
    victory_shader::emit::SLOT_FAULTS,
    victory_shader::emit::SLOT_WAVE_SCRATCH,
    victory_shader::emit::SLOT_DIAG,
];

fn compare_bound(m: &MetalBackend, input: &Input<'_>, template: &TopLevel, o: &MetalStage, b: &Bound, n: u64) {
    use victory_shader::emit::{SLOT_STRIDES, USER_DATA_SPACE};
    let name = input.name;
    let s = &b.stage;
    if o.outcome != s.outcome {
        mismatch(Cat::Outcome, name, || format!("old {:?} prep {:?}", o.outcome, s.outcome));
    }
    if o.group_base_at != s.group_base_at {
        mismatch(Cat::GroupBase, name, || format!("old {:?} prep {:?}", o.group_base_at, s.group_base_at));
    }
    if o.heap.is_some() != s.heap.is_some() || o.sampler_heap.is_some() != s.sampler_heap.is_some() {
        mismatch(Cat::HeapBound, name, || {
            format!("view heap old {} prep {}, sampler heap old {} prep {}", o.heap.is_some(), s.heap.is_some(), o.sampler_heap.is_some(), s.sampler_heap.is_some())
        });
    }

    // The top-level buffer, by root parameter.
    if o.top.len() != s.top.len() {
        mismatch(Cat::TopOther, name, || format!("top bytes old {} prep {}", o.top.len(), s.top.len()));
    } else {
        let mut covered = vec![false; o.top.len()];
        let u64_at = |t: &[u8], at: usize| t.get(at..at + 8).map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap()));
        for (i, r) in root_layout(&input.t.host).into_iter().enumerate() {
            let (at, len) = match r {
                RootSlot::Constants(dwords) => match template.constants_offset(USER_DATA_SPACE, 0) {
                    Some(at) => (at as usize, dwords as usize * 4),
                    None => continue,
                },
                _ => match template.offset_of(i) {
                    Some(at) => (at as usize, 8),
                    None => continue,
                },
            };
            let end = (at + len).min(o.top.len());
            covered.get_mut(at.min(end)..end).into_iter().flatten().for_each(|c| *c = true);
            let (a, c) = (&o.top[at.min(end)..end], &s.top[at.min(end)..end]);
            if a == c {
                continue;
            }
            match r {
                RootSlot::Constants(_) => mismatch(Cat::UserData, name, || format!("old {a:x?} prep {c:x?}")),
                // Our own stride table: compared slot by slot below.
                RootSlot::Srv(slot) if slot == SLOT_STRIDES => {}
                RootSlot::TableSrv(_) | RootSlot::TableUav(_) => mismatch(Cat::TopOther, name, || format!("table address bytes at {at}")),
                RootSlot::Cbv(slot) | RootSlot::Srv(slot) | RootSlot::Uav(slot) if !OWN_SLOTS.contains(&slot) => {
                    let (x, y) = (u64_at(&o.top, at), u64_at(&s.top, at));
                    let cat = if same_memory(m, x, y) { Cat::RootWrapper } else { Cat::RootAddress };
                    mismatch(cat, name, || format!("root parameter {i} ({r:?}): old {x:#x} prep {y:#x}"));
                }
                _ => mismatch(Cat::TopOther, name, || format!("root parameter {i} ({r:?}): old {a:x?} prep {c:x?}")),
            }
        }
        if let Some(at) = (0..o.top.len()).find(|&k| !covered[k] && o.top[k] != s.top[k]) {
            mismatch(Cat::TopOther, name, || format!("unnamed byte {at}: old {:#x} prep {:#x}", o.top[at], s.top[at]));
        }
    }

    // One-view tables.
    if o.tables.len() != s.tables.len() {
        mismatch(Cat::TableEntry, name, || format!("tables old {} prep {}", o.tables.len(), s.tables.len()));
    } else {
        for ((oa, oe), (sa, se)) in o.tables.iter().zip(&s.tables) {
            if oa != sa {
                mismatch(Cat::TableEntry, name, || format!("table at {oa} vs {sa}"));
            } else {
                entries_differ(m, oe, se, Cat::TableEntry, Cat::TableWrapper, name, &|| format!("table at {oa}"));
            }
        }
    }

    // The heap slots it reads.
    let mut used = b.used.clone();
    used.sort_unstable();
    used.dedup();
    for key in used {
        compare_slot(m, key, Cat::HeapEntry, Cat::HeapWrapper, name);
    }
    // A rotating window of both heaps every 256 comparisons: slots read
    // through no table (bindless) must be as up to date as the old heap's.
    if n % 256 == 1 {
        for key in input.heaps.into_iter().filter(|&h| h != 0) {
            let count = crate::descriptor::heap(key as This).count();
            if count == 0 {
                continue;
            }
            let first = ((n / 256) * 64) % u64::from(count);
            for k in 0..64u64.min(u64::from(count)) {
                let index = ((first + k) % u64::from(count)) as u32;
                if matches!(our_slot((key, index)), Some((_, _, State::Pending)) | None) {
                    continue;
                }
                compare_slot(m, (key, index), Cat::Bindless, Cat::HeapWrapper, name);
            }
        }
    }
}

fn compare_slot(m: &MetalBackend, key: Key, real: Cat, wrapper: Cat, name: &str) {
    let (Some((oe, os)), Some((se, ss, state))) = (old_slot(m, key), our_slot(key)) else {
        return;
    };
    entries_differ(m, &oe, &se, real, wrapper, name, &|| format!("slot {} ({state:?})", key.1));
    if os.is_some() && ss.is_some() && os != ss {
        mismatch(Cat::Stride, name, || format!("slot {} ({state:?}): old {os:x?} prep {ss:x?}", key.1));
    }
}

/// Test hook: the user data this layer computes is mutated (the first
/// word flipped), to show the shadow comparison catches it.
#[cfg(test)]
pub(crate) fn fault_user_data(on: bool) {
    FAULT_USER_DATA.store(on, Relaxed);
}
