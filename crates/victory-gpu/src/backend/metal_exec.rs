//! The stage 3 execution timeline (`victory_metal::exec`) behind the title's
//! queues, for the Metal path with a null host GPU (`VICTORY_GPU_NULL=1`).
//! Opt-in: `VICTORY_METAL_EXEC=1`. Off, nothing here runs and `queue.rs`
//! behaves as before.
//!
//! What it changes when on:
//! * `ExecuteCommandLists` only queues: each part of a submission (as
//!   `list::detach_parts` cuts it at its waits for memory values) becomes a
//!   timeline segment on the queue's timeline (graphics or compute). The game
//!   thread no longer waits for the GPU.
//! * A part's waits for memory values are timeline tag waits; its Metal ops
//!   run on the encoder thread through the existing `MetalBackend::run`
//!   (still one synchronous command buffer at a time inside it), and its
//!   bottom-of-pipe writes are done after that, in queue order.
//! * Queue `Signal` and `Wait` are timeline signals and waits; the title's
//!   fence object is set when the signal completes, and a CPU `Signal` on a
//!   fence is passed to the timeline.
//! * Page-mapping changes and presents first wait for every queue to drain:
//!   nothing in flight may read memory being remapped, and the present reads
//!   what the frame drew (the present itself is not on the timeline yet).
//!
//! * Stage 4 start: a part whose Metal ops are all compute dispatches (and
//!   need no texture checks) is encoded straight into timeline command
//!   buffers (`metal::exec_compute`): no `MetalBackend::run`, no submission
//!   lock, no wait per command buffer, concurrent dispatch between title
//!   barriers, overlap between the queues. `VICTORY_METAL_EXEC_COMPUTE=0`
//!   sends those parts through `MetalBackend::run` too.
//!
//! Other parts (draws, clears, copies, texture checks) still run through
//! `MetalBackend::run` on the encoder thread, one synchronous command buffer
//! at a time. This module has not been run in the game.

use std::sync::OnceLock;
use std::time::Duration;

use victory_metal::exec::{self, Config, FenceId, Hooks, QueueId, Segment, Timeline, Work};

use crate::object::This;

/// Whether the timeline takes the title's queues.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("VICTORY_METAL_EXEC").as_deref() == Ok("1")
            && crate::host::null_gpu()
            && super::metal::backend().is_some()
            && timeline().is_some();
        if on {
            tracing::warn!("VICTORY_METAL_EXEC=1: queues run on the stage 3 execution timeline (not validated in the game)");
        }
        on
    })
}

fn timeline() -> Option<&'static Timeline> {
    static TIMELINE: OnceLock<Option<Timeline>> = OnceLock::new();
    TIMELINE
        .get_or_init(|| {
            let m = super::metal::backend()?;
            let mut cfg = Config::from_env();
            cfg.fuse_ns = super::metal::fuse_limit_ns();
            let hooks = Hooks {
                stopped: Box::new(super::metal::fuse_tripped),
                stop: Box::new(|reason| {
                    tracing::error!(%reason, "Metal safety stop: execution timeline");
                    super::metal::stop_submissions("execution timeline");
                }),
                fence_reached: Box::new(|fence, value| crate::fence::set(fence as This, value)),
                // What `run_batch` does before encoding: pending remaps and
                // uploads, then residency.
                prepare: Box::new(|| {
                    let Some(m) = super::metal::backend() else { return false };
                    super::metal_texture::drain_remap_partial();
                    if !m.flush_uploads() {
                        return false;
                    }
                    m.commit_residency();
                    true
                }),
            };
            // One segment per command buffer: segments are already cut where
            // the old path cuts command buffers.
            cfg.max_segments_per_cb = 1;
            match Timeline::new(m.device.clone(), cfg, hooks) {
                Ok(t) => {
                    for q in 0..2 {
                        t.queue(q).add_residency_set(m.chunks().residency());
                    }
                    // Tables in the ring are read by GPU address.
                    let rings: Vec<_> = t.ring_buffers().iter().map(|b| b.handle()).collect();
                    m.add_resident(&rings);
                    m.commit_residency();
                    Some(t)
                }
                Err(e) => {
                    tracing::error!(error = %e, "Metal: execution timeline could not start");
                    None
                }
            }
        })
        .as_ref()
}

/// Whether compute-only parts go straight into timeline command buffers
/// (`VICTORY_METAL_EXEC_COMPUTE=0`: through `MetalBackend::run` as well).
fn timeline_compute() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_METAL_EXEC_COMPUTE").as_deref() != Ok("0"))
}

/// Whether other encodable parts (draws, clears, copies) go into timeline
/// command buffers too (`VICTORY_METAL_EXEC_DRAW=0`: through
/// `MetalBackend::run`).
fn timeline_draw() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_METAL_EXEC_DRAW").as_deref() != Ok("0"))
}

/// The timeline queue for a title queue type (D3D12_COMMAND_LIST_TYPE):
/// compute (2) runs on the asynchronous compute queue, everything else on
/// the graphics queue.
pub(crate) fn queue_of(kind: u32) -> QueueId {
    if kind == 2 { 1 } else { 0 }
}

/// Moves what the closure owns to the encoder or completion thread. The
/// title's detached host lists hold D3D12 objects, which are not marked
/// `Send`; with a null host GPU they are not used by the host, only kept
/// until the part has run and then recycled, one thread at a time.
struct Owned<T>(T);
// SAFETY: see above; each value is touched by one thread at a time.
unsafe impl<T> Send for Owned<T> {}

/// `ExecuteCommandLists` on the timeline (see the module comment).
pub(crate) fn execute(kind: u32, lists: &[This]) {
    let Some(t) = timeline() else { return };
    let q = queue_of(kind);
    let ops = crate::queue::metal_ops(lists);
    let parts = crate::list::detach_parts(lists);
    let n = parts.len();
    let mut ops = Some(Owned(ops));
    let mut segments = Vec::with_capacity(n);
    for (k, (waits, taken, order)) in parts.into_iter().enumerate() {
        // The submission's Metal ops run with its last part, after every
        // wait (as `queue.rs` does with cuts).
        let mine = if k + 1 == n { ops.take() } else { None };
        // Compute only: straight into timeline command buffers. The part's
        // waits go on its first segment, its writes on its last.
        let mut segs: Vec<Segment> = match mine {
            Some(Owned(ops)) if timeline_compute() && ops.textures.is_empty() && super::metal::exec_compute::compute_only(&ops.commands) => {
                super::metal::exec_compute::segments(&ops.commands)
            }
            // Draws, clears, copies, resolves (and compute with texture
            // checks): run_batch's encoding into timeline command buffers.
            Some(Owned(ops)) if timeline_draw() && super::metal::exec_compute::encodable(&ops.commands)
                && let Some(m) = super::metal::backend() =>
            {
                let textures = Owned(ops.textures.clone());
                let before: Box<dyn FnOnce() + Send> = Box::new(move || {
                    let textures = textures;
                    for t in &textures.0 {
                        t.refresh();
                    }
                });
                super::metal::exec_compute::general_segments(m, &ops.commands, Some(before))
            }
            Some(mine) => vec![Segment::new("title part", Work::Host(Box::new(move || {
                let mine = mine;
                crate::queue::run_metal_checked(&mine.0)
            })))],
            None => vec![Segment::new("title part", Work::None)],
        };
        let mut seg = segs.pop().expect("at least one segment");
        {
            let first = segs.first_mut().unwrap_or(&mut seg);
            for w in waits {
                let (address, value, ge) = w.as_tag();
                first.waits.push(exec::Wait::Tag { tag: address, value, ge, met: std::sync::Arc::new(move || w.met()) });
            }
        }
        for &i in &order {
            for (at, what, bytes) in taken[i].host().bottom_writes() {
                let tag = match what {
                    crate::queue::Written::Value(v) => Some((at as u64, if bytes == 4 { v & 0xFFFF_FFFF } else { v })),
                    _ => None,
                };
                seg = seg.write(tag, move || what.write(at as *mut u8, bytes));
            }
        }
        // The host lists come back once the part has run.
        let lists = Owned((taken, order));
        seg = seg.write(None, move || {
            // The whole wrapper moves in (not its fields, which are not Send).
            let lists = lists;
            let Owned((taken, order)) = lists;
            crate::list::recycle(taken, &order);
        });
        segments.extend(segs);
        segments.push(seg);
    }
    if let Err(e) = t.submit(q, segments) {
        tracing::warn!(error = %e, "Metal: submission refused by the execution timeline");
    }
}

/// A queue's `Signal`.
pub(crate) fn signal(kind: u32, fence: This, value: u64) {
    let Some(t) = timeline() else { return };
    if let Err(e) = t.submit(queue_of(kind), vec![Segment::new("signal", Work::None).signal(fence as FenceId, value)]) {
        tracing::warn!(error = %e, "Metal: signal refused by the execution timeline");
    }
}

/// A queue's `Wait`.
pub(crate) fn wait(kind: u32, fence: This, value: u64) {
    let Some(t) = timeline() else { return };
    if crate::fence::completed(fence) >= value {
        return;
    }
    if let Err(e) = t.submit(queue_of(kind), vec![Segment::new("wait", Work::None).wait_fence(fence as FenceId, value)]) {
        tracing::warn!(error = %e, "Metal: wait refused by the execution timeline");
    }
}

/// The title signalled a fence from the CPU.
pub(crate) fn cpu_signal(fence: This, value: u64) {
    if !on() {
        return;
    }
    if let Some(t) = timeline() {
        let _ = t.cpu_signal(fence as FenceId, value);
    }
}

/// Waits until everything given to the queues has run: before page-mapping
/// changes and presents. `force`: waits not met are passed first (a present:
/// a wait is not kept across frames).
pub(crate) fn drain(force: bool, why: &'static str) {
    if !on() {
        return;
    }
    let Some(t) = timeline() else { return };
    if force {
        let _ = t.force();
    }
    match t.drain(Duration::from_secs(5)) {
        Ok(true) => {}
        // Left at waits on the CPU: none of it is on the GPU.
        Ok(false) => {
            static SEEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
                tracing::info!(why, "Metal: execution timeline holds work at waits (none in flight)");
            }
        }
        // Stopped, or work still on the GPU after 5 s: hung. What comes next
        // (a page remap, a present) must not run beside it, so everything
        // stops; the timeline releases its events so nothing it committed
        // waits forever.
        Err(e) => {
            tracing::error!(why, error = %e, in_flight = ?t.in_flight(), "Metal safety stop: execution timeline did not drain");
            super::metal::stop_submissions("execution timeline did not drain");
            t.stop(why);
        }
    }
    static FRAMES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if force && FRAMES.fetch_add(1, std::sync::atomic::Ordering::Relaxed).is_power_of_two() {
        let s = t.stats();
        tracing::info!(
            committed = ?s.committed,
            host_runs = s.host_runs,
            cpu_held = s.cpu_held,
            forced = s.forced,
            gpu_tag_waits = s.gpu_tag_waits,
            gpu_fence_waits = s.gpu_fence_waits,
            budget_stalls = s.budget_stalls,
            barriers = s.barriers,
            barriers_elided = s.barriers_elided,
            timeline_dispatches = super::metal::exec_compute::ENCODED[0].load(std::sync::atomic::Ordering::Relaxed),
            timeline_indirect = super::metal::exec_compute::ENCODED[1].load(std::sync::atomic::Ordering::Relaxed),
            timeline_other_ops = super::metal::exec_compute::GENERAL[0].load(std::sync::atomic::Ordering::Relaxed),
            d3d12_prep = ?crate::draw::metal_prep(),
            "Metal: execution timeline"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_kinds_map_to_timelines() {
        // DIRECT 0, BUNDLE 1, COMPUTE 2, COPY 3.
        assert_eq!([0, 1, 2, 3].map(queue_of), [0, 0, 1, 0]);
    }
}
