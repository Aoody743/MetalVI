//! Compute passes in concurrent encoders, with barriers only where they are
//! needed (plan v2 D).
//!
//! The title's own barriers (UAV barriers, state transitions, the console
//! extension's waits and flushes) become memory barriers. Dispatches with no
//! barrier between them may overlap on the GPU, as they may on the console.
//! [`BarrierPolicy::Serial`] restores a barrier before every dispatch
//! (`VICTORY_METAL_SERIAL=1`) to bisect synchronization errors.

use crate::events::Fence;
use crate::{Buffer, CommandBuffer, Compute, ComputePipeline, Result};

use super::ring::RingSlice;

/// GPU addresses `[start, end)` a dispatch reads or writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Range {
    pub start: u64,
    pub end: u64,
}

impl Range {
    pub fn of(buffer: &Buffer, offset: u64, len: u64) -> Self {
        let start = buffer.gpu_address + offset;
        Self { start, end: start + len }
    }

    fn overlaps(&self, other: &Range) -> bool {
        self.start < other.end && other.start < self.end
    }
}

/// Where barriers go between the dispatches of one compute pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BarrierPolicy {
    /// Only where the title put one ([`ComputeItem::Barrier`]).
    #[default]
    Title,
    /// The title's, plus one wherever declared ranges conflict (read after
    /// write, write after read, write after write). For callers that know
    /// every access; the title's own lists do not declare ranges.
    Tracked,
    /// Before every dispatch but the first (`VICTORY_METAL_SERIAL=1`).
    Serial,
}

impl BarrierPolicy {
    pub fn from_env() -> Self {
        if std::env::var("VICTORY_METAL_SERIAL").as_deref() == Ok("1") { Self::Serial } else { Self::Title }
    }
}

/// One dispatch of a compute pass.
#[derive(Clone, Debug)]
pub struct Dispatch {
    pub pipeline: ComputePipeline,
    pub threads_per_group: [u64; 3],
    pub groups: [u64; 3],
    pub buffers: Vec<(Buffer, u64, u8)>,
    /// `setBytes` data and its index.
    pub bytes: Option<(Vec<u8>, u8)>,
    /// Data put in the command buffer's ring slice and bound as a buffer at
    /// the index (lives until the command buffer has completed).
    pub ring: Option<(Vec<u8>, u8)>,
    pub reads: Vec<Range>,
    pub writes: Vec<Range>,
}

impl Dispatch {
    pub fn group_count(&self) -> u64 {
        self.groups.iter().copied().fold(1, u64::saturating_mul)
    }
}

#[derive(Clone, Debug)]
pub enum ComputeItem {
    Dispatch(Dispatch),
    /// The title's barrier: later dispatches see earlier ones' writes.
    Barrier,
}

/// What [`plan`] decided.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Plan {
    /// Barriers placed.
    pub barriers: u64,
    /// Dispatch boundaries left without one (each a barrier the serial
    /// encoding had).
    pub elided: u64,
    /// Barriers placed only because declared ranges conflicted.
    pub from_ranges: u64,
}

/// For each dispatch in `items`, whether a barrier goes before it.
pub fn plan(items: &[ComputeItem], policy: BarrierPolicy) -> (Vec<bool>, Plan) {
    let mut before = Vec::new();
    let mut stats = Plan::default();
    let mut title = false;
    let mut reads: Vec<Range> = Vec::new();
    let mut writes: Vec<Range> = Vec::new();
    for item in items {
        let d = match item {
            ComputeItem::Barrier => {
                title = true;
                continue;
            }
            ComputeItem::Dispatch(d) => d,
        };
        let first = before.is_empty();
        let hazard = policy == BarrierPolicy::Tracked
            && (d.reads.iter().any(|r| writes.iter().any(|w| r.overlaps(w)))
                || d.writes.iter().any(|w| writes.iter().chain(&reads).any(|o| w.overlaps(o))));
        let barrier = !first && (policy == BarrierPolicy::Serial || title || hazard);
        if barrier {
            stats.barriers += 1;
            if hazard && !title {
                stats.from_ranges += 1;
            }
            reads.clear();
            writes.clear();
        } else if !first {
            stats.elided += 1;
        }
        reads.extend_from_slice(&d.reads);
        writes.extend_from_slice(&d.writes);
        before.push(barrier);
        title = false;
    }
    (before, stats)
}

/// Ring bytes a compute pass puts in its command buffer's slice.
pub fn ring_need(items: &[ComputeItem]) -> usize {
    RingSlice::need(items.iter().filter_map(|i| match i {
        ComputeItem::Dispatch(Dispatch { ring: Some((b, _)), .. }) => Some(b.len()),
        _ => None,
    }))
}

/// Thread groups a compute pass dispatches (the per-command-buffer work
/// limit counts these).
pub fn groups(items: &[ComputeItem]) -> u64 {
    items.iter().map(|i| match i {
        ComputeItem::Dispatch(d) => d.group_count(),
        ComputeItem::Barrier => 0,
    }).fold(0, u64::saturating_add)
}

/// Encodes `items` as one concurrent compute encoder in `cb`: first a wait
/// for `wait` (the queue's encoder fence, when an earlier encoder of the same
/// command buffer may have written what this one reads), barriers as
/// [`plan`] places them, and an update of `update` at the end.
pub fn encode(
    cb: &CommandBuffer,
    items: &[ComputeItem],
    policy: BarrierPolicy,
    ring: &mut RingSlice,
    wait: Option<&Fence>,
    update: Option<&Fence>,
) -> Result<Plan> {
    let (before, stats) = plan(items, policy);
    let dispatches: Vec<&Dispatch> = items.iter().filter_map(|i| match i {
        ComputeItem::Dispatch(d) => Some(d),
        ComputeItem::Barrier => None,
    }).collect();
    // Ring offsets first: steps borrow the slice's buffer.
    let mut ring_offsets = Vec::with_capacity(dispatches.len());
    for d in &dispatches {
        ring_offsets.push(match &d.ring {
            Some((bytes, _)) => Some(ring.put(bytes).ok_or_else(|| crate::Error("ring slice too small for its pass".into()))?),
            None => None,
        });
    }
    let mut steps: Vec<Compute<'_>> = Vec::new();
    if let Some(f) = wait {
        steps.push(Compute::WaitFence(f));
    }
    for (k, d) in dispatches.iter().enumerate() {
        if before[k] {
            steps.push(Compute::Barrier);
        }
        steps.push(Compute::Pipeline { pipeline: &d.pipeline, threads_per_group: d.threads_per_group });
        for (b, offset, index) in &d.buffers {
            steps.push(Compute::Buffer { buffer: b, offset: *offset, index: *index });
        }
        if let Some((bytes, index)) = &d.bytes {
            steps.push(Compute::Bytes { bytes, index: *index });
        }
        if let (Some(offset), Some((_, index))) = (ring_offsets[k], &d.ring) {
            steps.push(Compute::Buffer { buffer: &ring.buffer, offset, index: *index });
        }
        steps.push(Compute::Dispatch { groups: d.groups });
    }
    if let Some(f) = update {
        steps.push(Compute::UpdateFence(f));
    }
    cb.compute_concurrent(&steps)?;
    Ok(stats)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// A dispatch with only declared ranges (never encoded).
    fn fake(reads: &[(u64, u64)], writes: &[(u64, u64)]) -> ComputeItem {
        let r = |v: &[(u64, u64)]| v.iter().map(|&(start, end)| Range { start, end }).collect();
        ComputeItem::Dispatch(Dispatch {
            pipeline: ComputePipeline(crate::Object(0)),
            threads_per_group: [1; 3],
            groups: [1; 3],
            buffers: vec![],
            bytes: None,
            ring: None,
            reads: r(reads),
            writes: r(writes),
        })
    }

    fn forget(items: Vec<ComputeItem>) {
        // The fake pipelines hold a null handle: do not release it.
        for i in items {
            if let ComputeItem::Dispatch(d) = i {
                std::mem::forget(d.pipeline);
            }
        }
    }

    #[test]
    fn title_policy_only_keeps_title_barriers() {
        let items = vec![fake(&[], &[(0, 4)]), fake(&[], &[(4, 8)]), ComputeItem::Barrier, fake(&[(0, 8)], &[(8, 12)])];
        let (before, stats) = plan(&items, BarrierPolicy::Title);
        assert_eq!(before, [false, false, true]);
        assert_eq!(stats, Plan { barriers: 1, elided: 1, from_ranges: 0 });
        let (before, stats) = plan(&items, BarrierPolicy::Serial);
        assert_eq!(before, [false, true, true]);
        assert_eq!(stats.barriers, 2);
        forget(items);
    }

    #[test]
    fn tracked_policy_finds_raw_war_waw() {
        // read after write
        let raw = vec![fake(&[], &[(0, 4)]), fake(&[(0, 4)], &[(16, 20)])];
        // write after read
        let war = vec![fake(&[(0, 4)], &[(16, 20)]), fake(&[], &[(0, 4)])];
        // write after write
        let waw = vec![fake(&[], &[(0, 4)]), fake(&[], &[(2, 6)])];
        // independent: reads of the same range, disjoint writes
        let none = vec![fake(&[(0, 4)], &[(16, 20)]), fake(&[(0, 4)], &[(20, 24)])];
        for (items, want) in [(raw, true), (war, true), (waw, true), (none, false)] {
            let (before, stats) = plan(&items, BarrierPolicy::Tracked);
            assert_eq!(before, [false, want]);
            assert_eq!(stats.from_ranges, u64::from(want));
            forget(items);
        }
    }

    #[test]
    fn a_barrier_resets_what_is_tracked() {
        let items = vec![fake(&[], &[(0, 4)]), ComputeItem::Barrier, fake(&[(0, 4)], &[]), fake(&[(0, 4)], &[])];
        let (before, stats) = plan(&items, BarrierPolicy::Tracked);
        assert_eq!(before, [false, true, false]);
        assert_eq!(stats, Plan { barriers: 1, elided: 1, from_ranges: 0 });
        forget(items);
    }
}
