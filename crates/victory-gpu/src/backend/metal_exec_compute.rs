//! Stage 4 start: a submission's compute ops as execution-timeline segments
//! (`victory_metal::exec`), instead of [`super::MetalBackend::run`].
//!
//! * Direct dispatches: runs of them become [`Work::Compute`] segments, cut
//!   where the old path cuts command buffers (`VICTORY_METAL_PASS_DISPATCHES`,
//!   default 4, and the estimate window: `work::prefix`). Inside a segment the
//!   dispatches share one concurrent encoder, with a barrier only before a
//!   dispatch recorded after a title barrier (or anything that orders like
//!   one: [`super::MetalCmd::after_barrier`]). `VICTORY_METAL_SERIAL=1` puts
//!   one before every dispatch, as the old path does.
//! * Root one-view tables go in the timeline's ring (given back when the
//!   command buffer completed), their addresses into the top-level bytes.
//! * Indirect dispatches: a [`Work::Encode`] segment each, through
//!   `IndirectTranslator::encode_bounded` (GPU clamp, counted as before).
//! * Each segment's measured GPU time trains the same per-pipeline estimates
//!   as the old path (`work::observe`).
//!
//! The record side is unchanged: the ops still come from `draw.rs`'s
//! `metal_stage`, which still syncs the Metal heaps from the D3D12-era
//! descriptor mirror and bounds root views by the shadow copies
//! (`crate::draw::metal_prep` counts those calls).
//!
//! [`plan`] cuts one submission part into runs by how each op can run:
//! dispatches straight into timeline compute command buffers, draws /
//! clears / copies / resolves through `run_batch`'s encoding
//! ([`general_segments`]), and only what the CPU must see finished (buffer
//! copies, read-backs, the ambient stand-in, traced dispatches) through the
//! old synchronous `MetalBackend::run`. V0 (runs v0-r3/r4) logged
//! `timeline_dispatches=0`: the title's compute is recorded into the same
//! submissions as its draws, and every compute part also carried texture
//! checks, so no part was ever "compute only" and every dispatch went
//! through `run_batch`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use victory_metal::exec::{ComputeItem, Dispatch, Encoding, Segment, Work};
use victory_metal::indirect::{ClampCounter, DispatchSegments, IndirectKind, IndirectSource, IndirectTranslator};
use victory_metal::ir::DescriptorEntry;
use victory_metal::{Compute, Error, msc};

use super::{MetalCmd, MetalOp, work};

/// Ops encoded through here: direct dispatches, indirect dispatches,
/// segments made.
/// (and extra parts of indirect dispatches split at `RECORD_LIMIT`).
pub(crate) static ENCODED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Moves Metal ops (and Metal objects without `Send`) to the encoder and
/// completion threads; each is used by one thread at a time.
struct Sendable<T>(T);
// SAFETY: see above.
unsafe impl<T> Send for Sendable<T> {}
unsafe impl<T> Sync for Sendable<T> {}

/// Whether every op is a compute dispatch this path encodes (diagnostic
/// tracing that needs the old path's snapshots keeps it there).
pub(crate) fn compute_only(ops: &[Arc<MetalOp>]) -> bool {
    !ops.is_empty()
        && !super::super::metal_draw::trace_scene()
        && ops.iter().all(|o| match &**o {
            MetalOp::Dispatch(c) => c.loop_counts.is_none() && !work::trace_indirect_writer(&c.pso.name),
            MetalOp::DispatchIndirect(c) => !work::trace_indirect(&c.pso.name),
            _ => false,
        })
}

fn pass_cap() -> usize {
    static CAP: OnceLock<usize> = OnceLock::new();
    *CAP.get_or_init(|| {
        match std::env::var("VICTORY_METAL_PASS_DISPATCHES").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(4) {
            0 => usize::MAX,
            n => n,
        }
    })
}

fn entry_bytes(e: &DescriptorEntry) -> Vec<u8> {
    [e.gpu_va, e.texture_view_id, e.metadata].iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn item(c: &MetalCmd) -> Dispatch {
    let mut buffers = Vec::new();
    if let Some(b) = &c.heap {
        buffers.push((b.clone(), 0, msc::DESCRIPTOR_HEAP_BIND_POINT));
    }
    if let Some(b) = &c.sampler_heap {
        buffers.push((b.clone(), 0, msc::SAMPLER_HEAP_BIND_POINT));
    }
    Dispatch {
        pipeline: c.pso.pso.clone(),
        threads_per_group: c.pso.threadgroup,
        groups: c.groups,
        buffers,
        bytes: (!c.top.is_empty()).then(|| (c.top.clone(), msc::ARGUMENT_BUFFER_BIND_POINT)),
        ring: None,
        patches: c.tables.iter().map(|(at, e)| (*at, entry_bytes(e))).collect(),
        reads: Vec::new(),
        writes: Vec::new(),
    }
}

/// `top` with its one-view tables put in the ring and their addresses
/// written in (as the old path's ring does).
fn patched_top(e: &mut Encoding<'_>, top: &[u8], tables: &[(u32, DescriptorEntry)]) -> victory_metal::Result<Vec<u8>> {
    let mut top = top.to_vec();
    if tables.is_empty() {
        return Ok(top);
    }
    let data: Vec<u8> = tables.iter().flat_map(|(_, en)| entry_bytes(en)).collect();
    let base = e.ring.buffer.gpu_address + e.ring.put(&data).ok_or_else(|| Error("ring slice too small for tables".into()))?;
    for (k, (at, _)) in tables.iter().enumerate() {
        let at = *at as usize;
        if at + 8 <= top.len() {
            top[at..at + 8].copy_from_slice(&(base + 24 * k as u64).to_le_bytes());
        }
    }
    Ok(top)
}

fn observer(ops: Vec<Arc<MetalOp>>) -> Box<dyn FnOnce(u64) + Send> {
    let ops = Sendable(ops);
    Box::new(move |ns| {
        let ops = ops;
        work::observe(&ops.0, ns);
    })
}

/// The segments that run `ops` (all compute: [`compute_only`]), in order.
pub(crate) fn segments(ops: &[Arc<MetalOp>]) -> Vec<Segment> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < ops.len() {
        match &*ops[i] {
            MetalOp::Dispatch(_) => {
                let run = ops[i..].iter().take_while(|o| matches!(&***o, MetalOp::Dispatch(_))).count();
                let k = work::prefix(&ops[i..i + run], pass_cap()).max(1);
                let chunk = &ops[i..i + k];
                let mut items = Vec::with_capacity(2 * k);
                for (j, op) in chunk.iter().enumerate() {
                    let MetalOp::Dispatch(c) = &**op else { unreachable!("a run of dispatches") };
                    if j > 0 && c.after_barrier {
                        items.push(ComputeItem::Barrier);
                    }
                    items.push(ComputeItem::Dispatch(item(c)));
                }
                ENCODED[0].fetch_add(k as u64, Ordering::Relaxed);
                let mut seg = Segment::new("title compute", Work::Compute(items)).estimate(work::reservation(chunk));
                seg.observe = Some(observer(chunk.to_vec()));
                out.push(seg);
                i += k;
            }
            MetalOp::DispatchIndirect(c) => {
                ENCODED[1].fetch_add(1, Ordering::Relaxed);
                // More records than one bounded translation takes: in parts,
                // each a command buffer of its own (not cut off).
                let limit = victory_metal::indirect::RECORD_LIMIT;
                let mut first = 0;
                loop {
                    let part = c.max.saturating_sub(first).min(limit);
                    out.push(indirect(ops[i].clone(), c.tables.len(), first, part));
                    first += part;
                    if part == 0 || first >= c.max {
                        break;
                    }
                    ENCODED[3].fetch_add(1, Ordering::Relaxed);
                }
                i += 1;
            }
            _ => unreachable!("compute_only checked"),
        }
    }
    ENCODED[2].fetch_add(out.len() as u64, Ordering::Relaxed);
    out
}

/// Output records per source record of an indirect dispatch: its group-base
/// segments (main's `DispatchSegments`), or 1.
fn segments_of(op: &MetalOp) -> u64 {
    match op {
        MetalOp::DispatchIndirect(c) if c.group_base_at.is_some() && work::dispatch_segments_on() => u64::from(DispatchSegments::DEFAULT.segments),
        _ => 1,
    }
}

/// Records `first..first + max` of an indirect dispatch.
fn indirect(op: Arc<MetalOp>, tables: usize, first: u32, max: u32) -> Segment {
    static TRANSLATOR: OnceLock<Option<Sendable<IndirectTranslator>>> = OnceLock::new();
    let estimate = work::reservation(std::slice::from_ref(&op));
    let counter: Arc<OnceLock<Sendable<ClampCounter>>> = Arc::new(OnceLock::new());
    let for_encode = counter.clone();
    let for_report = counter;
    let encoded = Sendable(op.clone());
    let encode = Box::new(move |e: &mut Encoding<'_>| -> victory_metal::Result<()> {
        let op = &encoded.0;
        let MetalOp::DispatchIndirect(c) = &**op else { unreachable!("an indirect dispatch") };
        let tr = TRANSLATOR
            .get_or_init(|| IndirectTranslator::new(e.device).ok().map(Sendable))
            .as_ref()
            .ok_or_else(|| Error("indirect translator unavailable".into()))?;
        if for_encode.get().is_none() {
            let _ = for_encode.set(Sendable(ClampCounter::new(e.device)?));
        }
        let counter = &for_encode.get().expect("set above").0;
        let top = patched_top(e, &c.top, &c.tables)?;
        // A program that adds a group base: each record cut in segments, each
        // with its own base in its top bytes (as `run_batch` does).
        let segments = c.group_base_at.filter(|_| work::dispatch_segments_on()).map(|_| DispatchSegments::DEFAULT);
        let seg_tops: Vec<Vec<u8>> = match (segments, c.group_base_at) {
            (Some(d), Some(at)) if at + 4 <= top.len() => (0..d.segments)
                .map(|k| {
                    let mut t = top.clone();
                    t[at..at + 4].copy_from_slice(&(k * d.groups).to_le_bytes());
                    t
                })
                .collect(),
            _ => Vec::new(),
        };
        let source = IndirectSource {
            dispatch_segments: segments.filter(|_| !seg_tops.is_empty()),
            kind: IndirectKind::Dispatch,
            records: (&c.records, c.records_offset + u64::from(first) * u64::from(c.stride)),
            stride: c.stride,
            max_count: max,
            count: c.count.as_ref().map(|(b, o)| (b, *o)),
        };
        let Some(batch) = tr.0.encode_bounded_part(e.device, e.cb, &source, counter, first)? else { return Ok(()) };
        let mut steps = Vec::new();
        for r in 0..batch.record_count() {
            let Some(rec) = batch.record(r) else { continue };
            if !steps.is_empty() {
                steps.push(Compute::Barrier);
            }
            steps.push(Compute::Pipeline { pipeline: &c.pso.pso, threads_per_group: c.pso.threadgroup });
            if let Some(b) = &c.heap {
                steps.push(Compute::Buffer { buffer: b, offset: 0, index: msc::DESCRIPTOR_HEAP_BIND_POINT });
            }
            if let Some(b) = &c.sampler_heap {
                steps.push(Compute::Buffer { buffer: b, offset: 0, index: msc::SAMPLER_HEAP_BIND_POINT });
            }
            let top = if seg_tops.is_empty() { &top } else { &seg_tops[(r as usize) % seg_tops.len()] };
            if !top.is_empty() {
                steps.push(Compute::Bytes { bytes: top, index: msc::ARGUMENT_BUFFER_BIND_POINT });
            }
            steps.push(Compute::DispatchIndirect { arguments: rec.buffer, offset: rec.offset });
        }
        if steps.is_empty() { Ok(()) } else { e.cb.compute(&steps) }
    });
    let groups = u64::from(max).saturating_mul(u64::from(victory_metal::indirect::GROUPS_PER_RECORD)).saturating_mul(segments_of(&op));
    Segment::new("title indirect compute", Work::Encode { ring_bytes: tables * 24, groups, encode })
        .estimate(estimate)
        .write(None, move || {
            if let Some(c) = for_report.get() {
                // SAFETY: the write runs after the command buffer completed.
                let records = unsafe { c.0.completed_count() };
                if records != 0 {
                    static CLAMPED: AtomicU64 = AtomicU64::new(0);
                    let total = CLAMPED.fetch_add(u64::from(records), Ordering::Relaxed) + u64::from(records);
                    tracing::warn!(records, total, "Metal: indirect arguments clamped; not a correctness sample");
                }
            }
        })
}

/// Whether `ops` (not all compute) can be encoded into timeline command
/// buffers through `MetalBackend::encode_external`: nothing in them is read
/// by the CPU right after it ran, and no diagnostic needs the old path's
/// cuts and snapshots.
pub(crate) fn encodable(ops: &[Arc<MetalOp>]) -> bool {
    !ops.is_empty()
        && !super::super::metal_draw::trace_scene()
        && ops.iter().all(|o| match &**o {
            MetalOp::BufferCopy { .. } | MetalOp::FloatRowReadback { .. } => false,
            // The ambient stand-in writes an image by address (untracked)
            // and needs the old path's "ambient order" command buffer cuts.
            MetalOp::Ambient(_) => false,
            MetalOp::Dispatch(c) => c.loop_counts.is_none() && !work::trace_indirect_writer(&c.pso.name),
            MetalOp::DispatchIndirect(c) => !work::trace_indirect(&c.pso.name),
            MetalOp::Draw(_) | MetalOp::Clear(_) | MetalOp::Copy(_) | MetalOp::Resolve(_) => true,
        })
}

/// How an op runs on the timeline ([`plan`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Route {
    /// Straight into timeline compute command buffers ([`segments`]).
    Compute,
    /// `run_batch`'s encoding into timeline command buffers
    /// ([`general_segments`]).
    Encode,
    /// The old synchronous `MetalBackend::run`, on the encoder thread, once
    /// the queue has nothing in flight: the CPU reads or writes what these
    /// touch right after they ran, or a diagnostic needs the old path.
    Host,
}

/// `direct_compute`: dispatches go straight to the timeline
/// (`VICTORY_METAL_EXEC_COMPUTE`, else with `run_batch`'s encoding);
/// `encode`: draws and the like go through `run_batch`'s encoding
/// (`VICTORY_METAL_EXEC_DRAW`, else the old path).
pub(crate) fn route(op: &MetalOp, direct_compute: bool, encode: bool) -> Route {
    let r = match op {
        MetalOp::Dispatch(c) if c.loop_counts.is_none() && !work::trace_indirect_writer(&c.pso.name) => Route::Compute,
        MetalOp::DispatchIndirect(c) if !work::trace_indirect(&c.pso.name) => Route::Compute,
        MetalOp::Draw(_) | MetalOp::Clear(_) | MetalOp::Copy(_) | MetalOp::Resolve(_) => Route::Encode,
        // Buffer copies (the CPU copies after the GPU work before it),
        // read-backs, the ambient stand-in (its own command buffer, in
        // "ambient order"), traced dispatches (snapshots).
        _ => Route::Host,
    };
    match r {
        Route::Compute if !direct_compute => if encode { Route::Encode } else { Route::Host },
        Route::Encode if !encode => Route::Host,
        r => r,
    }
}

/// `ops` cut into maximal runs of one [`Route`], in order. While a frame
/// is traced (`VICTORY_METAL_TRACE_FRAMES`) everything runs on the old
/// path, which takes the snapshots.
pub(crate) fn pieces(ops: &[Arc<MetalOp>], direct_compute: bool, encode: bool) -> Vec<(Route, std::ops::Range<usize>)> {
    if super::super::metal_draw::trace_scene() {
        return if ops.is_empty() { Vec::new() } else { vec![(Route::Host, 0..ops.len())] };
    }
    let mut out: Vec<(Route, std::ops::Range<usize>)> = Vec::new();
    for (i, op) in ops.iter().enumerate() {
        let r = route(op, direct_compute, encode);
        match out.last_mut() {
            Some((last, range)) if *last == r => range.end = i + 1,
            _ => out.push((r, i..i + 1)),
        }
    }
    out
}

/// Ops planned by [`plan`], by route: direct compute, `run_batch` encoding,
/// old path.
pub(crate) static ROUTED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// The timeline segments of one submission part's ops, in order (see the
/// module notes). `before` (the part's deferred texture checks) runs on the
/// encoder thread before the first of them; `host` makes the segment that
/// runs a run of ops on the old path. When dispatches went straight to the
/// timeline and textures are waiting to be written back (F44), a last
/// segment writes them back, as `run_batch` does after a batch's dispatches.
pub(crate) fn plan(
    m: &'static super::MetalBackend,
    ops: &[Arc<MetalOp>],
    direct_compute: bool,
    encode: bool,
    before: Option<Box<dyn FnOnce() + Send>>,
    mut host: impl FnMut(Vec<Arc<MetalOp>>) -> Segment,
) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::new();
    let mut direct = false;
    for (r, range) in pieces(ops, direct_compute, encode) {
        let run = &ops[range];
        let k = match r {
            Route::Compute => 0,
            Route::Encode => 1,
            Route::Host => 2,
        };
        ROUTED[k].fetch_add(run.len() as u64, Ordering::Relaxed);
        match r {
            Route::Compute => {
                direct = true;
                out.extend(segments(run));
            }
            Route::Encode => out.extend(general_segments(m, run, None)),
            Route::Host => out.push(host(run.to_vec())),
        }
    }
    if direct && m.writeback_wanted() {
        out.push(writeback(m));
    }
    if let Some(first) = out.first_mut() {
        first.before = before;
    }
    out
}

/// F44's writeback after dispatches that ran on the timeline. Its command
/// buffer always holds GPU work (a 4-byte fill of its ring region): one that
/// held none (nothing left to write back by the time it is encoded) would
/// read as a GPU fault (no GPU time).
fn writeback(m: &'static super::MetalBackend) -> Segment {
    let encode = Box::new(move |e: &mut Encoding<'_>| -> victory_metal::Result<()> {
        m.writeback_all(e.cb).map_err(|err| Error(format!("writeback: {err}")))?;
        let (_, offset) = e.ring.reserve(4).ok_or_else(|| Error("ring slice too small for the writeback marker".into()))?;
        let mut blit = e.cb.blit_encoder()?;
        blit.fill_buffer(&e.ring.buffer, offset, 4, 0);
        blit.finish();
        Ok(())
    });
    Segment::new("title writeback", Work::Encode { ring_bytes: victory_metal::exec::ring::ALIGN, groups: 0, encode })
}

/// Ops per command buffer through [`general_segments`]
/// (`VICTORY_METAL_EXEC_CHUNK_OPS`, default 256; 0: no count limit). V0 R3
/// encoded one batch of 4,772 ops for 2.8 s as one unit: no GPU work could
/// start before the whole batch was encoded. At the measured 70–100 µs of
/// encoding per op, 256 ops is about 20–25 ms of encoder time.
fn chunk_ops() -> usize {
    static CAP: OnceLock<usize> = OnceLock::new();
    *CAP.get_or_init(|| match std::env::var("VICTORY_METAL_EXEC_CHUNK_OPS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(256) {
        0 => usize::MAX,
        n => n,
    })
}

/// Parts of indirect draws over `RECORD_LIMIT` records translated one
/// after the other by `run_batch` on the timeline (instead of rejected).
pub(crate) static DRAW_PARTS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn note_draw_parts(n: u32) {
    DRAW_PARTS.fetch_add(u64::from(n), Ordering::Relaxed);
}

/// Op counts encoded through [`general_segments`]: ops, chunks.
pub(crate) static GENERAL: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The segments that run `ops` ([`encodable`]) through `run_batch`'s
/// encoding, one command buffer per chunk, cut where the old path cuts: the
/// estimate window (`work::prefix`) and `VICTORY_METAL_PASS_DISPATCHES`
/// dispatches per command buffer (1: every op alone); and at most
/// [`chunk_ops`] ops per chunk, so encoding and GPU work overlap.
/// `before` runs on the encoder thread right before the first chunk is
/// encoded (the part's deferred texture checks).
pub(crate) fn general_segments(m: &'static super::MetalBackend, ops: &[Arc<MetalOp>], before: Option<Box<dyn FnOnce() + Send>>) -> Vec<Segment> {
    general_segments_capped(m, ops, before, chunk_ops())
}

/// [`general_segments`] with `max_ops` ops per chunk at most.
pub(crate) fn general_segments_capped(m: &'static super::MetalBackend, ops: &[Arc<MetalOp>], before: Option<Box<dyn FnOnce() + Send>>, max_ops: usize) -> Vec<Segment> {
    let cap = pass_cap();
    let mut before = before;
    let mut out = Vec::new();
    let mut i = 0;
    while i < ops.len() {
        let fit = work::prefix(&ops[i..], if cap == 1 { 1 } else { max_ops.max(1) }).max(1);
        let mut end = i;
        let mut dispatches = 0usize;
        while end < i + fit {
            let dispatch = matches!(&*ops[end], MetalOp::Dispatch(_) | MetalOp::DispatchIndirect(_));
            if dispatch && dispatches >= cap && end > i {
                break;
            }
            dispatches += usize::from(dispatch);
            end += 1;
        }
        let chunk: Vec<Arc<MetalOp>> = ops[i..end].to_vec();
        i = end;
        GENERAL[0].fetch_add(chunk.len() as u64, Ordering::Relaxed);
        GENERAL[1].fetch_add(1, Ordering::Relaxed);
        let ring_bytes = super::MetalBackend::ring_bytes(&chunk);
        let estimate = work::reservation(&chunk);
        let groups = chunk.iter().map(|o| match &**o {
            MetalOp::Dispatch(c) => c.groups.iter().copied().fold(1u64, u64::saturating_mul),
            MetalOp::DispatchIndirect(c) => u64::from(c.max).saturating_mul(u64::from(victory_metal::indirect::GROUPS_PER_RECORD)).saturating_mul(segments_of(o)),
            _ => 0,
        }).fold(0u64, u64::saturating_add);
        // What run_batch leaves for after completion, and the command buffer
        // to check it against.
        let deferred: Arc<std::sync::Mutex<Option<Sendable<(super::Deferred, victory_metal::CommandBuffer)>>>> = Default::default();
        let for_encode = deferred.clone();
        let encoded_ops = Sendable(chunk.clone());
        let first = before.take();
        let encode = Box::new(move |e: &mut Encoding<'_>| -> victory_metal::Result<()> {
            if let Some(f) = first {
                f();
            }
            let ops = &encoded_ops.0;
            let (cpu, offset) = e.ring.reserve(ring_bytes).ok_or_else(|| Error("ring slice too small for a chunk".into()))?;
            let mut ext = super::External {
                cb: e.cb.clone(),
                ring_buffer: e.ring.buffer.clone(),
                ring_offset: offset,
                ring_cpu: cpu,
                ring_len: ring_bytes,
                done: None,
            };
            if !m.encode_external(ops, &mut ext) {
                return Err(Error("run_batch refused an externally encoded chunk".into()));
            }
            if let Some(d) = ext.done {
                *for_encode.lock().unwrap_or_else(|p| p.into_inner()) = Some(Sendable((d, e.cb.clone())));
            }
            Ok(())
        });
        let report_ops = Sendable(chunk.clone());
        let mut seg = Segment::new("title ops", Work::Encode { ring_bytes, groups, encode })
            .estimate(estimate)
            .write(None, move || {
                let report_ops = report_ops;
                if let Some(Sendable((d, cb))) = deferred.lock().unwrap_or_else(|p| p.into_inner()).take() {
                    d.report(&cb, &report_ops.0);
                }
            });
        seg.observe = Some(observer(chunk));
        out.push(seg);
    }
    out
}

#[cfg(test)]
#[path = "metal_exec_compute_tests.rs"]
mod tests;
