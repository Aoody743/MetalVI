use crate::backend::metal_draw::DrawKind;
use super::MetalOp;
use std::collections::HashMap;
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
    let (groups_per_record, inputs_per_record) = victory_metal::indirect::record_budget();
    let product = |g: &[u64; 3]| g.iter().copied().fold(1u64, u64::saturating_mul);
    let (cost, units) = match op {
        MetalOp::Dispatch(c) => (&c.pso.cost_ns, product(&c.groups)),
        MetalOp::DispatchIndirect(c) => (&c.pso.cost_ns, u64::from(c.max) * u64::from(groups_per_record)),
        MetalOp::Draw(d) => (&d.pso.cost_ns, match &d.kind {
            DrawKind::Mesh { groups } => product(groups),
            DrawKind::Geometry { draw, .. } => product(&draw.groups),
            DrawKind::Indirect { max, .. } => u64::from(*max) * u64::from(if d.pso.mesh || d.pso.geometry.is_some() { groups_per_record } else { inputs_per_record }),
            DrawKind::Vertices { count, instances, .. } | DrawKind::Indexed { count, instances, .. } => u64::from(*count) * u64::from(*instances),
            DrawKind::Generated { indices, instances, .. } => (indices.len() as u64).saturating_mul(u64::from(*instances)),
        }),
        _ => return None,
    };
    Some((cost, units.max(1)))
}

fn indirect(op: &MetalOp) -> bool {
    matches!(op, MetalOp::DispatchIndirect(_)) || matches!(op, MetalOp::Draw(d) if matches!(d.kind, DrawKind::Indirect { .. }))
}

/// A short name of an operation's kind, for the split log.
pub(super) fn kind(op: &MetalOp) -> &'static str {
    match op {
        MetalOp::Dispatch(_) => "dispatch",
        MetalOp::DispatchIndirect(_) => "dispatch indirect",
        MetalOp::Draw(d) if matches!(d.kind, DrawKind::Indirect { .. }) => "draw indirect",
        MetalOp::Draw(_) => "draw",
        MetalOp::Clear(_) => "clear",
        MetalOp::Copy(_) => "texture copy",
        MetalOp::FloatRowReadback { .. } => "readback",
        MetalOp::BufferCopy { .. } => "CPU buffer copy",
        MetalOp::Resolve(_) => "resolve",
        MetalOp::Ambient(_) => "ambient",
    }
}

/// Operations whose cost is not a per-unit time: indirect ones (their count
/// comes from the GPU) by pipeline (its `cost_ns` cell's address), and
/// clears, texture copies, resolves and readbacks by kind (small keys no
/// address takes). Their longest GPU time seen alone in a command buffer.
///
/// Without this they kept the bootstrap estimate (a whole window) for ever
/// and each ran alone in a command buffer, waited for: ~250,000 one-operation
/// command buffers at ~0.6 ms each made the 150 s loading stalls before 3D
/// (stage-r-m4-trace1200-20261128-042456: 76 slow batches, 75 s encoding and
/// waiting, 0.26 s GPU). The estimate is twice the longest time seen; the
/// clamp bounds indirect records and the fuse every measured time.
/// Off by default since 2026-11-28 (`VICTORY_METAL_INDIRECT_LEARN=1` turns
/// it on): with it, command buffers held many operations and ran for tens
/// of ms, and four runs ended with the game stuck in a residency-set commit
/// in the kernel (see `IN_FLIGHT` in metal.rs); ~20 runs before had none.
static SOLO_MAX_NS: std::sync::Mutex<Option<HashMap<usize, u64>>> = std::sync::Mutex::new(None);

fn solo_learn() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_METAL_INDIRECT_LEARN").as_deref() == Ok("1"))
}

fn solo_key(op: &MetalOp) -> Option<usize> {
    if indirect(op) {
        return profile(op).map(|(cost, _)| std::ptr::from_ref(cost) as usize);
    }
    match op {
        MetalOp::Clear(_) => Some(1),
        MetalOp::Copy(_) => Some(2),
        MetalOp::Resolve(_) => Some(3),
        MetalOp::FloatRowReadback { .. } => Some(4),
        MetalOp::Ambient(_) => Some(5),
        _ => None,
    }
}

fn solo_estimate(op: &MetalOp) -> Option<u64> {
    let key = solo_key(op)?;
    Some(if solo_learn() { solo_estimate_of(key) } else { WINDOW_NS })
}

fn solo_estimate_of(key: usize) -> u64 {
    let seen = SOLO_MAX_NS.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|m| m.get(&key).copied());
    seen.map_or(WINDOW_NS, |ns| ns.saturating_mul(2).max(100_000))
}

fn solo_learn_time(key: usize, ns: u64) {
    let mut map = SOLO_MAX_NS.lock().unwrap_or_else(|e| e.into_inner());
    let e = map.get_or_insert_with(HashMap::new).entry(key).or_insert(0);
    *e = (*e).max(ns);
}

fn estimate(op: &MetalOp) -> Option<u64> {
    if let Some(ns) = solo_estimate(op) {
        return Some(ns);
    }
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
    // An operation without a per-unit cost alone in its command buffer: its whole time.
    if let [op] = ops
        && ns != 0
        && solo_learn()
        && let Some(key) = solo_key(op)
    {
        solo_learn_time(key, ns);
        return;
    }
    let Some((cost, _)) = ops.first().and_then(|op| profile(op)) else { return };
    // Do not attribute a mixed command buffer's time to a particular pipeline,
    // or learn from a GPU argument count that was only a reserved maximum.
    let units = ops.iter().try_fold(0u64, |sum, op| {
        if indirect(op) {
            return None;
        }
        let (key, n) = profile(op)?;
        std::ptr::eq(key, cost).then(|| sum.saturating_add(n))
    });
    if let Some(n) = units && ns != 0 { cost.store(ns.div_ceil(n), Ordering::Relaxed); }
}

/// GPU time against wall time, logged every 5 s: whether a slow stretch is
/// the GPU's work or time spent elsewhere (encoding, waiting, the title).
/// Wall time spent in `run_batch` (encoding, submitting, waiting).
pub(super) static RUN_NS: AtomicU64 = AtomicU64::new(0);

pub(super) fn gpu_busy(gpu_ns: u64, ops: &[Arc<MetalOp>]) {
    struct Busy { since: std::time::Instant, gpu_ns: u64, buffers: u64, ops: u64, top: HashMap<&'static str, u64> }
    static BUSY: std::sync::Mutex<Option<Busy>> = std::sync::Mutex::new(None);
    let mut guard = BUSY.lock().unwrap_or_else(|e| e.into_inner());
    let b = guard.get_or_insert_with(|| Busy { since: std::time::Instant::now(), gpu_ns: 0, buffers: 0, ops: 0, top: HashMap::new() });
    b.gpu_ns = b.gpu_ns.saturating_add(gpu_ns);
    b.buffers += 1;
    b.ops += ops.len() as u64;
    // A buffer's time by the kind of its first operation.
    if let Some(op) = ops.first() {
        *b.top.entry(kind(op)).or_insert(0) += gpu_ns;
    }
    let wall = b.since.elapsed();
    if wall >= std::time::Duration::from_secs(5) {
        let mut top: Vec<_> = b.top.iter().map(|(k, ns)| format!("{k}:{}ms", ns / 1_000_000)).collect();
        top.sort();
        tracing::info!(wall_ms = wall.as_millis() as u64, gpu_ms = b.gpu_ns / 1_000_000,
            busy_pct = (b.gpu_ns as f64 / wall.as_nanos().max(1) as f64 * 100.0) as u64,
            run_batch_ms = RUN_NS.swap(0, Ordering::Relaxed) / 1_000_000, buffers = b.buffers, ops = b.ops, by_first = %top.join(","), frame = crate::frame::current(),
            "Metal: GPU busy share");
        *guard = None;
    }
}

/// Host phases by lane, lock-free, for the hang watch: what a thread is
/// doing (a `&'static str`) and since when (ms after the first phase). Lane 0
/// is submission (`run`), lane 1 captures and presents.
pub(super) const LANES: usize = 2;
pub(super) const LANE_RUN: usize = 0;
pub(super) const LANE_CAPTURE: usize = 1;
static PHASE_PTR: [std::sync::atomic::AtomicPtr<u8>; LANES] = [const { std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()) }; LANES];
static PHASE_LEN: [AtomicU64; LANES] = [const { AtomicU64::new(0) }; LANES];
static PHASE_AT: [AtomicU64; LANES] = [const { AtomicU64::new(0) }; LANES];
static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

fn now_ms() -> u64 {
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64 + 1
}

/// Marks `lane` as in `phase` (`""`: idle) from now on.
pub(super) fn phase(lane: usize, phase: &'static str) {
    #[cfg(test)]
    profile::mark(phase);
    PHASE_LEN[lane].store(phase.len() as u64, Ordering::Relaxed);
    PHASE_PTR[lane].store(phase.as_ptr().cast_mut(), Ordering::Relaxed);
    PHASE_AT[lane].store(now_ms(), Ordering::Release);
}

/// The phase `lane` has been in for over `ms`, its age, and when it began.
pub(super) fn stuck_phase(lane: usize, ms: u64) -> Option<(&'static str, u64, u64)> {
    let at = PHASE_AT[lane].load(Ordering::Acquire);
    let (ptr, len) = (PHASE_PTR[lane].load(Ordering::Relaxed), PHASE_LEN[lane].load(Ordering::Relaxed) as usize);
    if at == 0 || ptr.is_null() || len == 0 { return None; }
    let age = now_ms().saturating_sub(at);
    // SAFETY: only `&'static str`s are stored; a torn pair is re-read next second.
    let phase = std::str::from_utf8(unsafe { std::slice::from_raw_parts(ptr, len) }).ok()?;
    (age > ms && PHASE_AT[lane].load(Ordering::Acquire) == at).then_some((phase, age, at))
}

thread_local! {
    static LANE: std::cell::Cell<usize> = const { std::cell::Cell::new(LANE_CAPTURE) };
}

/// This thread's lane until the guard drops (then idle and back to captures).
pub(super) fn lane(lane: usize) -> impl Drop {
    struct Lane(usize);
    impl Drop for Lane {
        fn drop(&mut self) {
            phase(self.0, "");
            LANE.set(LANE_CAPTURE);
        }
    }
    LANE.set(lane);
    Lane(lane)
}

/// Marks this thread's lane as in `p`.
pub(in crate::backend) fn here(p: &'static str) {
    phase(LANE.get(), p);
}

/// `p` until the guard drops, then the phase before it again (finer labels
/// for the stuck-phase watchdog and the test profiler, nothing else).
pub(in crate::backend) fn within(p: &'static str) -> impl Drop {
    struct Restore(usize, *mut u8, u64);
    impl Drop for Restore {
        fn drop(&mut self) {
            if self.1.is_null() {
                return;
            }
            // SAFETY: only `&'static str`s are stored in the phase slots.
            let before = unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(self.1, self.2 as usize)) };
            phase(self.0, before);
        }
    }
    let lane = LANE.get();
    let restore = Restore(lane, PHASE_PTR[lane].load(Ordering::Relaxed), PHASE_LEN[lane].load(Ordering::Relaxed));
    phase(lane, p);
    restore
}

/// After a wait or a capture: idle on the capture lane, encoding on the run lane.
pub(in crate::backend) fn after_wait() {
    let lane = LANE.get();
    phase(lane, if lane == LANE_RUN { "run_batch encode" } else { "" });
}

/// Indirect dispatches of programs with a group base are cut in segments
/// (`victory_metal::indirect::DispatchSegments`); `VICTORY_METAL_DISPATCH_SEGMENTS=0`
/// clamps them to the group budget as before.
pub(super) fn dispatch_segments_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_METAL_DISPATCH_SEGMENTS").as_deref() != Ok("0"))
}

/// Cut a bounded command buffer after every compute pass (as before
/// 0056e07): the default again since 2026-11-28, for the reason given at
/// `SOLO_MAX_NS`. `VICTORY_METAL_CUT_EVERY_PASS=0` cuts by dispatch count.
pub(super) fn cut_every_pass() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_METAL_CUT_EVERY_PASS").as_deref() != Ok("0"))
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
            MetalOp::Ambient(_) => "ambient",
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

pub(super) fn a1f6_count_offsets(root: u32, view_bytes: u64) -> Option<[u64; 2]> {
    let offsets = [4, 0x374].map(|n| u64::from(root.wrapping_add(n)));
    offsets.iter().all(|&n| n % 4 == 0 && n + 4 <= view_bytes).then_some(offsets)
}

#[test]
fn a1f6_loop_count_offsets_match_scalar_loads() {
    assert_eq!(a1f6_count_offsets(0x400, 0x778), Some([0x404, 0x774]));
    assert_eq!(a1f6_count_offsets(0x400, 0x777), None);
    assert_eq!(a1f6_count_offsets(1, 0x800), None);
    assert_eq!(a1f6_count_offsets(u32::MAX - 3, 0x374), Some([0, 0x370]));
}

pub(super) fn snapshot_words<const N: usize>(
    device: &victory_metal::Device,
    queue: &victory_metal::Queue,
    sources: &[(&victory_metal::Buffer, u64)],
) -> victory_metal::Result<Vec<[u32; N]>> {
    let record = N * 4;
    let size = sources.len() * record;
    let (dst, mapped) = device.shared_buffer_mapped(&vec![0; size])?;
    let cb = queue.command_buffer()?;
    let mut blit = cb.blit_encoder()?;
    for (i, &(src, offset)) in sources.iter().enumerate() {
        if offset % 4 != 0 || offset.checked_add(record as u64).is_none_or(|end| end > src.len) {
            return Err(victory_metal::Error("invalid indirect snapshot window".into()));
        }
        blit.copy_buffer(src, offset, &dst, (i * record) as u64, record as u64);
    }
    blit.finish();
    super::submit_checked(&cb, &[], true, "indirect writer snapshot")?;
    // SAFETY: the buffer is still owned here and submit_checked waited successfully.
    let bytes = unsafe { std::slice::from_raw_parts(mapped, size) };
    Ok(bytes.chunks_exact(record).map(|record| {
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
    assert_eq!(snapshot_words::<3>(&device, &queue, &sources).unwrap(), [[0; 3]]);
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
    assert_eq!(snapshot_words::<3>(&device, &queue, &sources).unwrap(), [[2, 1, 1]]);
    assert_eq!(snapshot_words::<1>(&device, &queue, &[(&buffer, 4), (&buffer, 12)]).unwrap(), [[2], [1]]);
}

pub(super) fn report(cb: &victory_metal::CommandBuffer, counter: Option<victory_metal::indirect::ClampCounter>,
    batch: u64, traces: &mut Vec<(usize, String, victory_metal::indirect::IndirectTrace)>, ops: &[Arc<MetalOp>]) {
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
            // Which indirect work the clamp cut (the counter is per buffer):
            // its pipelines, by kind, for the first lines.
            static LINES: AtomicU64 = AtomicU64::new(0);
            let pipelines = if LINES.fetch_add(1, Ordering::Relaxed) < 400 {
                let mut names: Vec<String> = ops.iter().filter_map(|op| match &**op {
                    MetalOp::DispatchIndirect(c) => Some(format!("dispatch:{}:max{}", c.pso.name, c.max)),
                    MetalOp::Draw(d) => match &d.kind {
                        DrawKind::Indirect { max, .. } => Some(format!("{}:{}:max{max}",
                            if d.pso.geometry.is_some() { "rects" } else if d.pso.mesh { "mesh" } else { "draw" }, d.pso.name)),
                        _ => None,
                    },
                    _ => None,
                }).collect();
                names.sort();
                names.dedup();
                names.join(",")
            } else {
                String::new()
            };
            tracing::warn!(records, total, pipelines, "Metal: indirect arguments clamped; not a correctness sample");
        }
        for (op_index, pipeline, trace) in traces.drain(..) {
            // SAFETY: the owning command buffer completed successfully above.
            for (record, args) in unsafe { trace.completed_records() }.enumerate() {
                // Records the clamp changed, at info (`VICTORY_METAL_TRACE_INDIRECT`
                // picks the pipelines); the rest at debug.
                static CHANGED: AtomicU64 = AtomicU64::new(0);
                if args.active && args.before != args.after && CHANGED.fetch_add(1, Ordering::Relaxed) < 4000 {
                    tracing::info!(batch, op_index, %pipeline, record, raw_count = args.raw_count,
                        before = ?args.before, after = ?args.after, "Metal: indirect record clamped");
                } else {
                    tracing::debug!(batch, op_index, %pipeline, record, raw_count = args.raw_count, active = args.active,
                        before = ?args.before, after = ?args.after, "Metal: indirect trace record");
                }
            }
        }
    }
}

#[test]
fn an_operation_seen_alone_is_estimated_from_its_longest_time() {
    // A key no operation kind or pipeline address uses.
    const KEY: usize = 99;
    assert_eq!(solo_estimate_of(KEY), WINDOW_NS);
    solo_learn_time(KEY, 30_000);
    solo_learn_time(KEY, 20_000);
    // Twice the longest, at least 100 us: many such operations share a window.
    assert_eq!(solo_estimate_of(KEY), 100_000);
    solo_learn_time(KEY, 1_000_000);
    assert_eq!(solo_estimate_of(KEY), 2_000_000);
    assert_eq!(fit(std::iter::repeat_n(solo_estimate_of(KEY), 40)), 25);
}

#[test]
fn budget_prefix_never_rejects_oversized_first_operation() {
    assert_eq!(fit(std::iter::empty()), 0);
    assert_eq!(fit([u64::MAX, 1].into_iter()), 1);
    assert_eq!(fit([WINDOW_NS / 2, WINDOW_NS / 2, 1].into_iter()), 2);
    assert_eq!(fit([1, WINDOW_NS + 1].into_iter()), 1);
}

/// Tests only: time spent in each phase on this thread, between
/// [`profile::start`] and [`profile::take`] (stage 4 encode profiling).
#[cfg(test)]
pub(crate) mod profile {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::time::Instant;

    thread_local! {
        static P: RefCell<Option<(&'static str, Instant, HashMap<&'static str, (u64, u64)>)>> = const { RefCell::new(None) };
    }

    pub(crate) fn start() {
        P.with_borrow_mut(|p| *p = Some(("", Instant::now(), HashMap::new())));
    }

    pub(super) fn mark(phase: &'static str) {
        P.with_borrow_mut(|p| {
            if let Some((last, at, map)) = p.as_mut() {
                let now = Instant::now();
                let e = map.entry(*last).or_default();
                e.0 += now.duration_since(*at).as_nanos() as u64;
                e.1 += 1;
                *last = phase;
                *at = now;
            }
        });
    }

    /// (phase, ns, times entered), most time first.
    pub(crate) fn take() -> Vec<(&'static str, u64, u64)> {
        mark("");
        let map = P.with_borrow_mut(|p| p.take().map(|(_, _, m)| m)).unwrap_or_default();
        let mut v: Vec<_> = map.into_iter().map(|(k, (ns, n))| (k, ns, n)).collect();
        v.sort_by_key(|e| std::cmp::Reverse(e.1));
        v
    }
}
