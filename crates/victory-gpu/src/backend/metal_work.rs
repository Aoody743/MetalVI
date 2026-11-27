use crate::backend::metal_draw::DrawKind;
use super::MetalOp;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

// Diagnostic thresholds, not admission limits or measured GPU time.
const GROUPS: u64 = 4096;
const INPUTS: u64 = 65536;
pub(super) const WINDOW_NS: u64 = 50_000_000;
pub(super) fn report_large(label: &str, work: victory_metal::EncodedWork) {
    if work.groups > GROUPS || work.draw_inputs > INPUTS ||
        work.attachment_samples > 32 * 1024 * 1024 || work.blit_bytes > 16 * 1024 * 1024 ||
        work.blit_texels > 8 * 1024 * 1024 || work.resolve_pixels > 8 * 1024 * 1024
    {
        static LARGE: AtomicU64 = AtomicU64::new(0);
        let total = LARGE.fetch_add(1, Ordering::Relaxed) + 1;
        tracing::info!(label, ?work, total, "Metal: large encoded workload");
    }
}
pub(super) static REJECTIONS: AtomicU64 = AtomicU64::new(0);
pub(super) static SPLITS: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
thread_local! {
    pub(super) static FAULT: std::cell::Cell<Option<&'static str>> = const { std::cell::Cell::new(None) };
}

pub(super) fn reject(reason: &'static str) -> bool {
    super::fuse::stop();
    let rejected = REJECTIONS.fetch_add(1, Ordering::Relaxed) + 1;
    tracing::error!(reason, rejected, "Metal: encoding stopped before commit");
    false
}

fn profile(op: &MetalOp) -> Option<(&AtomicU64, u64)> {
    use victory_metal::indirect::{GROUPS_PER_RECORD, INPUTS_PER_RECORD};
    let product = |g: &[u64; 3]| g.iter().copied().fold(1u64, u64::saturating_mul);
    let (cost, units) = match op {
        MetalOp::Dispatch(c) => (&c.pso.cost_ns, product(&c.groups)),
        MetalOp::DispatchIndirect(c) => (&c.pso.cost_ns, u64::from(c.max) * u64::from(GROUPS_PER_RECORD)),
        MetalOp::Draw(d) => (&d.pso.cost_ns, match &d.kind {
            DrawKind::Mesh { groups } => product(groups),
            DrawKind::Geometry { draw, .. } => product(&draw.groups),
            DrawKind::Indirect { max, .. } => u64::from(*max) * u64::from(if d.pso.mesh || d.pso.geometry.is_some() { GROUPS_PER_RECORD } else { INPUTS_PER_RECORD }),
            DrawKind::Vertices { count, instances, .. } | DrawKind::Indexed { count, instances, .. } => u64::from(*count) * u64::from(*instances),
            DrawKind::Generated { indices, instances, .. } => (indices.len() as u64).saturating_mul(u64::from(*instances)),
        }),
        _ => return None,
    };
    Some((cost, units.max(1)))
}

fn estimate(op: &MetalOp) -> Option<u64> {
    let (cost, units) = profile(op)?;
    let measured = cost.load(Ordering::Relaxed);
    // Bootstrap only. Neither an estimate nor an observed time is a WCET.
    Some(if measured == 0 { WINDOW_NS } else { units.saturating_mul(measured).max(100_000) })
}

pub(super) fn reservation(ops: &[Arc<MetalOp>]) -> u64 {
    if ops.is_empty() { return WINDOW_NS; }
    ops.iter().fold(0u64, |n, op| n.saturating_add(estimate(op).unwrap_or(WINDOW_NS)))
}

pub(super) fn observe(ops: &[Arc<MetalOp>], ns: u64) {
    let Some((cost, _)) = ops.first().and_then(|op| profile(op)) else { return };
    // Do not attribute a mixed command buffer's time to a particular pipeline,
    // or learn from a GPU argument count that was only a reserved maximum.
    let units = ops.iter().try_fold(0u64, |sum, op| {
        if matches!(&**op, MetalOp::DispatchIndirect(_)) || matches!(&**op, MetalOp::Draw(d) if matches!(d.kind, DrawKind::Indirect { .. })) {
            return None;
        }
        let (key, n) = profile(op)?;
        std::ptr::eq(key, cost).then(|| sum.saturating_add(n))
    });
    if let Some(n) = units && ns != 0 { cost.store(ns.div_ceil(n), Ordering::Relaxed); }
}

/// An over-window operation runs alone. Replan after each completion so
/// observed costs can inform the next boundary.
pub(super) fn prefix(ops: &[Arc<MetalOp>], max_ops: usize) -> usize {
    fit(ops.iter().take(max_ops).map(|op| estimate(op).unwrap_or(WINDOW_NS)))
}

pub(super) fn trace_ops(batch: u64, ops: &[Arc<MetalOp>]) {
    if !tracing::enabled!(tracing::Level::DEBUG) { return; }
    let frame = crate::frame::current();
    for (op_index, op) in ops.iter().enumerate() {
        let kind = match &**op {
            MetalOp::Dispatch(c) => {
                tracing::debug!(frame, batch, op_index, kind = "dispatch",
                    pipeline = %c.pso.name, groups = ?c.groups, threads = ?c.pso.threadgroup, "Metal: recorded operation");
                continue;
            }
            MetalOp::DispatchIndirect(c) => {
                tracing::debug!(frame, batch, op_index, kind = "dispatch indirect",
                    pipeline = %c.pso.name, threads = ?c.pso.threadgroup, max_records = c.max,
                    stride = c.stride, counted = c.count.is_some(), records_va = c.records_va,
                    count_va = ?c.count_va, "Metal: recorded operation");
                continue;
            }
            MetalOp::Draw(d) => {
                let (kind, groups, count, instances, max_records) = match &d.kind {
                    DrawKind::Vertices { count, instances, .. } => ("draw", None, Some(u64::from(*count)), Some(*instances), None),
                    DrawKind::Indexed { count, instances, .. } => ("draw indexed", None, Some(u64::from(*count)), Some(*instances), None),
                    DrawKind::Generated { indices, instances, .. } => ("draw generated", None, Some(indices.len() as u64), Some(*instances), None),
                    DrawKind::Mesh { groups } => ("draw mesh", Some(*groups), None, None, None),
                    DrawKind::Geometry { draw, .. } => ("draw geometry", Some(draw.groups),
                        Some(u64::from(u32::from_le_bytes(draw.args[..4].try_into().unwrap()))),
                        Some(u32::from_le_bytes(draw.args[4..8].try_into().unwrap())), None),
                    DrawKind::Indirect { max, .. } => ("draw indirect", None, None, None, Some(*max)),
                };
                tracing::debug!(frame, batch, op_index, kind, pipeline = %d.pso.name,
                    ?groups, ?count, ?instances, ?max_records, width = d.targets.width, height = d.targets.height,
                    layers = 1, samples = d.targets.samples, "Metal: recorded operation");
                continue;
            }
            MetalOp::Clear(_) => "clear",
            MetalOp::Copy(_) => "texture copy",
            MetalOp::FloatRowReadback { .. } => "readback",
            MetalOp::BufferCopy { .. } => "CPU buffer copy",
            MetalOp::Resolve(_) => "resolve",
        };
        tracing::debug!(frame, batch, op_index, kind, "Metal: recorded operation");
    }
}

fn fit(costs: impl Iterator<Item = u64>) -> usize {
    let mut used = 0u64;
    costs.enumerate().take_while(|&(i, cost)| {
        used = used.saturating_add(cost);
        i == 0 || used <= WINDOW_NS
    }).count()
}

pub(super) fn trace_indirect(pipeline: &str) -> bool {
    static FILTER: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    FILTER.get_or_init(|| std::env::var("VICTORY_METAL_TRACE_INDIRECT").unwrap_or_default()
        .split(',').map(str::trim).filter(|p| !p.is_empty()).map(str::to_owned).collect())
        .iter().any(|id| pipeline.contains(id))
}

pub(super) fn trace_indirect_writer(pipeline: &str) -> bool {
    static FILTER: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    FILTER.get_or_init(|| std::env::var("VICTORY_METAL_TRACE_INDIRECT_WRITER").ok().filter(|s| !s.is_empty()))
        .as_ref().is_some_and(|s| pipeline.contains(s))
}

pub(super) fn snapshot_arguments(
    device: &victory_metal::Device,
    queue: &victory_metal::Queue,
    sources: &[(&victory_metal::Buffer, u64)],
) -> victory_metal::Result<Vec<[u32; 3]>> {
    const RECORD: usize = 12;
    let size = sources.len() * RECORD;
    let (dst, mapped) = device.shared_buffer_mapped(&vec![0; size])?;
    let cb = queue.command_buffer()?;
    let mut blit = cb.blit_encoder()?;
    for (i, &(src, offset)) in sources.iter().enumerate() {
        if offset % 4 != 0 || offset.checked_add(RECORD as u64).is_none_or(|end| end > src.len) {
            return Err(victory_metal::Error("invalid indirect snapshot window".into()));
        }
        blit.copy_buffer(src, offset, &dst, (i * RECORD) as u64, RECORD as u64);
    }
    blit.finish();
    super::submit_checked(&cb, &[], true, "indirect writer snapshot")?;
    // SAFETY: the buffer is still owned here and submit_checked waited successfully.
    let bytes = unsafe { std::slice::from_raw_parts(mapped, size) };
    Ok(bytes.chunks_exact(RECORD).map(|record| {
        std::array::from_fn(|i| u32::from_le_bytes(record[i * 4..i * 4 + 4].try_into().unwrap()))
    }).collect())
}

#[test]
fn snapshot_indirect_arguments_reads_gpu_writes() {
    use victory_metal::{Compute, Device, Pool};
    let _pool = Pool::new();
    let device = Device::first().unwrap();
    let queue = device.new_queue(1).unwrap();
    let buffer = device.shared_buffer(&[0; 16]).unwrap();
    let sources = [(&buffer, 4)];
    assert_eq!(snapshot_arguments(&device, &queue, &sources).unwrap(), [[0; 3]]);
    let pipeline = device.compute_pipeline_msl(
        "#include <metal_stdlib>\nkernel void write_args(device uint *p [[buffer(0)]]) { p[1]=2; p[2]=1; p[3]=1; }",
        "write_args",
    ).unwrap();
    let cb = queue.command_buffer().unwrap();
    cb.compute(&[
        Compute::Pipeline { pipeline: &pipeline, threads_per_group: [1, 1, 1] },
        Compute::Buffer { buffer: &buffer, offset: 0, index: 0 },
        Compute::Dispatch { groups: [1, 1, 1] },
    ]).unwrap();
    super::submit_checked(&cb, &[], true, "synthetic argument producer").unwrap();
    assert_eq!(snapshot_arguments(&device, &queue, &sources).unwrap(), [[2, 1, 1]]);
}

pub(super) fn report(cb: &victory_metal::CommandBuffer, counter: Option<victory_metal::indirect::ClampCounter>,
    batch: u64, traces: &mut Vec<(usize, victory_metal::indirect::IndirectTrace)>) {
    if let Some(counter) = counter {
        use victory_metal::sys;
        // SAFETY: the buffer is owned; never read GPU output after an error or non-commit.
        if unsafe { sys::MTLCommandBuffer_status(cb.handle()) } != sys::WMTCommandBufferStatus::WMTCommandBufferStatusCompleted {
            tracing::warn!("Metal: no completed indirect clamp count for failed buffer");
            traces.clear();
            return;
        }
        let records = unsafe { counter.completed_count() };
        if records != 0 {
            static CLAMPED: AtomicU64 = AtomicU64::new(0);
            let total = CLAMPED.fetch_add(u64::from(records), Ordering::Relaxed) + u64::from(records);
            tracing::warn!(records, total, "Metal: indirect arguments clamped; not a correctness sample");
        }
        for (op_index, trace) in traces.drain(..) {
            // SAFETY: the owning command buffer completed successfully above.
            for (record, args) in unsafe { trace.completed_records() }.enumerate() {
                tracing::debug!(batch, op_index, record, raw_count = args.raw_count, active = args.active,
                    before = ?args.before, after = ?args.after, "Metal: indirect trace record");
            }
        }
    }
}

#[test]
fn budget_prefix_never_rejects_oversized_first_operation() {
    assert_eq!(fit(std::iter::empty()), 0);
    assert_eq!(fit([u64::MAX, 1].into_iter()), 1);
    assert_eq!(fit([WINDOW_NS / 2, WINDOW_NS / 2, 1].into_iter()), 2);
    assert_eq!(fit([1, WINDOW_NS + 1].into_iter()), 1);
}
