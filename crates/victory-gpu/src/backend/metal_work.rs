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

fn fit(costs: impl Iterator<Item = u64>) -> usize {
    let mut used = 0u64;
    costs.enumerate().take_while(|&(i, cost)| {
        used = used.saturating_add(cost);
        i == 0 || used <= WINDOW_NS
    }).count()
}

pub(super) fn report(cb: &victory_metal::CommandBuffer, counter: Option<victory_metal::indirect::ClampCounter>) {
    if let Some(counter) = counter {
        use victory_metal::sys;
        // SAFETY: the buffer is owned; never read GPU output after an error or non-commit.
        if unsafe { sys::MTLCommandBuffer_status(cb.handle()) } != sys::WMTCommandBufferStatus::WMTCommandBufferStatusCompleted {
            tracing::warn!("Metal: no completed indirect clamp count for failed buffer");
            return;
        }
        let records = unsafe { counter.completed_count() };
        if records != 0 {
            static CLAMPED: AtomicU64 = AtomicU64::new(0);
            let total = CLAMPED.fetch_add(u64::from(records), Ordering::Relaxed) + u64::from(records);
            tracing::warn!(records, total, "Metal: indirect arguments clamped; not a correctness sample");
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
