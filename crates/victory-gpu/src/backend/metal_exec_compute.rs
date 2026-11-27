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

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use victory_metal::exec::{ComputeItem, Dispatch, Encoding, Segment, Work};
use victory_metal::indirect::{ClampCounter, IndirectKind, IndirectSource, IndirectTranslator};
use victory_metal::ir::DescriptorEntry;
use victory_metal::{Compute, Error, msc};

use super::{MetalCmd, MetalOp, work};

/// Ops encoded through here: direct dispatches, indirect dispatches,
/// segments made.
pub(crate) static ENCODED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

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
                out.push(indirect(ops[i].clone(), c.tables.len(), c.max));
                i += 1;
            }
            _ => unreachable!("compute_only checked"),
        }
    }
    ENCODED[2].fetch_add(out.len() as u64, Ordering::Relaxed);
    out
}

fn indirect(op: Arc<MetalOp>, tables: usize, max: u32) -> Segment {
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
        let source = IndirectSource {
            kind: IndirectKind::Dispatch,
            records: (&c.records, c.records_offset),
            stride: c.stride,
            max_count: c.max,
            count: c.count.as_ref().map(|(b, o)| (b, *o)),
        };
        let Some(batch) = tr.0.encode_bounded(e.device, e.cb, &source, counter)? else { return Ok(()) };
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
            if !top.is_empty() {
                steps.push(Compute::Bytes { bytes: &top, index: msc::ARGUMENT_BUFFER_BIND_POINT });
            }
            steps.push(Compute::DispatchIndirect { arguments: rec.buffer, offset: rec.offset });
        }
        if steps.is_empty() { Ok(()) } else { e.cb.compute(&steps) }
    });
    let groups = u64::from(max).saturating_mul(u64::from(victory_metal::indirect::GROUPS_PER_RECORD));
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

#[cfg(test)]
#[path = "metal_exec_compute_tests.rs"]
mod tests;
