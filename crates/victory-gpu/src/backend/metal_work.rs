use crate::backend::metal_draw::DrawKind;
use super::MetalOp;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

// Count limits, not measured GPU time limits. A vertex/index is a conservative
// upper bound for a primitive in the ordinary non-amplifying draw paths.
const GROUPS: u64 = 4096;
const INPUTS: u64 = 65536;
pub(super) const WINDOW_NS: u64 = 50_000_000;
pub(super) fn within_limits(work: victory_metal::EncodedWork) -> bool {
    work.groups <= GROUPS && work.draw_inputs <= INPUTS &&
        work.attachment_samples <= 32 * 1024 * 1024 && work.blit_bytes <= 16 * 1024 * 1024 &&
        work.blit_texels <= 8 * 1024 * 1024 && work.resolve_pixels <= 8 * 1024 * 1024
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

#[derive(Default)]
pub(super) struct Budget {
    groups: u64,
    inputs: u64,
    estimated_ns: u64,
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
    ops.iter().try_fold(0u64, |n, op| Some(n.saturating_add(estimate(op)?))).unwrap_or(WINDOW_NS)
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

impl Budget {
    /// Return the fitting prefix. Never consume the first operation of the next
    /// buffer; the caller re-evaluates it after the preceding buffer completes.
    pub(super) fn prefix(ops: &[Arc<MetalOp>]) -> usize {
        let mut b = Self::default();
        ops.iter().take_while(|op| {
            let fits = match &***op {
                MetalOp::Dispatch(c) => b.charge(c.groups, 0),
                MetalOp::DispatchIndirect(c) => b.reserve_indirect(c.max, true, false),
                MetalOp::Draw(d) => b.draw(&d.kind) && match &d.kind {
                    DrawKind::Indirect { max, index, .. } => b.reserve_indirect(*max,
                        d.pso.mesh || d.pso.geometry.is_some(), !d.pso.mesh || d.pso.geometry.is_some() || index.is_some()),
                    _ => true,
                },
                _ => true,
            };
            fits && b.charge_time(op)
        }).count()
    }
    pub(super) fn charge_time(&mut self, op: &MetalOp) -> bool {
        self.estimated_ns = self.estimated_ns.saturating_add(estimate(op).unwrap_or(WINDOW_NS));
        self.estimated_ns <= WINDOW_NS
    }
    pub(super) fn reserve_indirect(&mut self, count: u32, groups: bool, inputs: bool) -> bool {
        use victory_metal::indirect::{GROUPS_PER_RECORD, INPUTS_PER_RECORD, RECORD_LIMIT};
        let helper = u64::from(count.div_ceil(64));
        count <= RECORD_LIMIT && self.charge(
            [helper + u64::from(count) * u64::from(GROUPS_PER_RECORD) * u64::from(groups), 1, 1],
            u64::from(count) * u64::from(INPUTS_PER_RECORD) * u64::from(inputs),
        )
    }
    pub(super) fn charge(&mut self, dims: [u64; 3], inputs: u64) -> bool {
        let Some(groups) = dims.into_iter().try_fold(1u64, u64::checked_mul) else { return false };
        let limit = GROUPS;
        #[cfg(test)]
        let limit = if FAULT.get() == Some("one-group") { 1 } else { limit };
        if dims.into_iter().any(|d| d > limit) || groups > limit - self.groups || inputs > INPUTS - self.inputs {
            return false;
        }
        self.groups += groups;
        self.inputs += inputs;
        true
    }

    pub(super) fn draw(&mut self, kind: &DrawKind) -> bool {
        match kind {
            DrawKind::Vertices { count, instances, .. } | DrawKind::Indexed { count, instances, .. } =>
                self.charge([0; 3], u64::from(*count) * u64::from(*instances)),
            DrawKind::Generated { indices, instances, .. } =>
                (indices.len() as u64).checked_mul((*instances).into()).is_some_and(|n| self.charge([0; 3], n)),
            DrawKind::Mesh { groups } => self.charge(*groups, 0),
            DrawKind::Geometry { draw, .. } => {
                let count = u32::from_le_bytes(draw.args[..4].try_into().unwrap());
                let instances = u32::from_le_bytes(draw.args[4..8].try_into().unwrap());
                self.charge(draw.groups, u64::from(count) * u64::from(instances))
            }
            // GPU-generated arguments require their own reservation and clamp.
            DrawKind::Indirect { .. } => true,
        }
    }
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
fn cumulative_direct_limits() {
    let mut budget = Budget::default();
    assert!(budget.charge([16, 16, 16], 0));
    assert!(!budget.charge([1; 3], 0));
    assert!(!budget.charge([u64::MAX; 3], 1));
    let draw = |count, instances| DrawKind::Vertices { count, instances, first: 0, first_instance: 0 };
    assert!(budget.draw(&draw(1024, 64)));
    assert!(!budget.draw(&draw(1, 1)));
    assert!(!Budget::default().draw(&draw(u32::MAX, u32::MAX)));
}
