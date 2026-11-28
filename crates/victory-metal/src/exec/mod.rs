//! The execution model of plan v2 stage 3: queue timelines and asynchronous
//! submission (A), one encoding thread (B, single-thread version), and
//! dependency-driven barriers (D), under the in-flight and completion checks
//! of safety model v2 (H).
//!
//! ```text
//!  game threads ─ submit() ─► encoder thread ─ commit ─► MTLCommandQueue ×N
//!   (return at once)            │  waits → GPU event waits     │
//!                               │  ring regions, budget        ▼
//!                               └◄── progress ── completion thread per queue
//!                                                (status, GPU time, fuse,
//!                                                 bottom-of-pipe writes,
//!                                                 fence notices, ring back)
//! ```
//!
//! * Each queue is an `MTLCommandQueue`. Its command buffers are chained by a
//!   [`SharedEvent`] (memory reached by address is not tracked by Metal; a
//!   shared event, not a GPU-only one, so the CPU can release it), and each
//!   ends by signalling the queue's `done` [`SharedEvent`] to its sequence
//!   number, which the completion thread waits on with a timeout.
//! * **No GPU wait is left without a signal.** Every event a committed
//!   command buffer waits on (fence, tag, chain) is one the timeline owns. On
//!   any stop (GPU error, fuse, hang timeout, lost completion thread, outer
//!   stop) the timeline sets all of them to their maximum, so command
//!   buffers already committed behind a failed producer drain instead of
//!   waiting forever. `done` events are not released: a completion thread
//!   whose wait timed out never calls `waitUntilCompleted`, so it cannot
//!   block on a command buffer that never finishes. Every CPU-side wait
//!   ([`Timeline::drain`], completion) has a timeout.
//! * A title fence is a [`SharedEvent`] whose value is the fence value. A
//!   signal is encoded at the end of the command buffer; a wait at the start
//!   of the next one, **but only for a value whose signal is already
//!   committed** (otherwise the queue holds the segment on the CPU, as
//!   `queue.rs` did, and counts it). No GPU queue ever waits for a signal
//!   that is not on a GPU queue yet.
//! * A wait for a memory value ("tag") is a GPU wait on the tag's event when
//!   it compares with `>=` against a value some committed segment writes at
//!   its bottom of pipe, and writes to that tag only ever increased. The
//!   completion thread does the write, then signals the tag event from the
//!   CPU. Anything else (`==`, masks, falling values, no producer) is held on
//!   the CPU until the memory holds the value.
//! * Bottom-of-pipe writes run on the completion thread after the command
//!   buffer completed, in submission order per queue (the same order the
//!   synchronous path produces).
//! * In-flight budget per queue: command buffers and estimated GPU time. When
//!   full, the encoder thread moves on to other queues; game threads are
//!   blocked only by a backlog of [`Config::max_pending`] segments.
//! * The completion thread stops all new commits on: a GPU error, measured
//!   GPU time over [`Config::fuse_ns`], no completion within
//!   [`Config::hang_ms`], or its own exit. A command buffer already running
//!   cannot be interrupted.
//! * [`Config::sync`] (`VICTORY_METAL_SYNC=1`): every command buffer is
//!   waited for on the encoder thread before the next is encoded, waits are
//!   never put on the GPU, and `submit` returns only when the timelines can
//!   go no further: the old lockstep behaviour, kept as the reference.
//! * [`Timeline::drain`] times out on the GPU only: committed command
//!   buffers that stop completing. Time the encoder thread spends encoding
//!   (or running host work) is not a hang; only a much longer stretch with
//!   no progress at all ([`Config::encoder_stall_ms`]) ends a drain.

pub mod barrier;
pub mod ring;
#[cfg(test)]
mod tests;

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, mpsc};
use std::time::{Duration, Instant};

use crate::events::{Fence, SharedEvent};
use crate::{CommandBuffer, Device, Error, Pool, Queue, Result};

pub use barrier::{BarrierPolicy, ComputeItem, Dispatch, Range};
pub use ring::RingSlice;

/// [`Timeline::gpu_idle`]: while held, nothing is committed.
pub struct GpuIdle<'a> {
    inner: &'a Inner,
    idle: bool,
}

impl GpuIdle<'_> {
    /// Whether every command buffer committed before it had finished.
    pub fn idle(&self) -> bool {
        self.idle
    }
}

impl Drop for GpuIdle<'_> {
    fn drop(&mut self) {
        lock(&self.inner.commits).0 -= 1;
        self.inner.commits_changed.notify_all();
    }
}

pub type QueueId = usize;
/// A title fence (its object's address, or any unique key).
pub type FenceId = u64;
/// A memory location waited on and written at the bottom of the pipe.
pub type TagId = u64;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Faults injected by tests (all off by default).
#[derive(Clone, Debug, Default)]
pub struct Faults {
    /// Creating a fence or tag event fails.
    pub fail_event: bool,
    /// The completion thread of this queue exits after this many command
    /// buffers.
    pub completion_exits_after: Option<(QueueId, u64)>,
}

#[derive(Clone, Debug)]
pub struct Config {
    /// Number of queues (0 = graphics, 1 = asynchronous compute, ...).
    pub queues: usize,
    /// The synchronous reference path (`VICTORY_METAL_SYNC=1`).
    pub sync: bool,
    /// Command buffers in flight per queue.
    pub max_in_flight: u32,
    /// Estimated GPU time in flight per queue; a command buffer estimated
    /// over it runs alone.
    pub in_flight_ns: u64,
    /// Measured GPU time per command buffer that stops all submission.
    pub fuse_ns: u64,
    /// How long a command buffer may take to complete (wall clock, from
    /// commit) before it counts as hung.
    pub hang_ms: u64,
    /// [`Timeline::drain`]'s backstop: how long the timeline may make no
    /// progress at all (no work started or committed, no completion, no host
    /// work finished) while a drain waits: one piece of work running this
    /// long counts as stuck. Far longer than any batch takes to encode.
    pub encoder_stall_ms: u64,
    /// Ring bytes per queue.
    pub ring_bytes: usize,
    /// Segments game threads may queue ahead of the encoder.
    pub max_pending: usize,
    /// Segments per command buffer.
    pub max_segments_per_cb: usize,
    /// Thread groups per command buffer (a segment over it runs alone).
    pub max_groups_per_cb: u64,
    pub barriers: BarrierPolicy,
    /// Order consecutive all-compute command buffers of a queue by the
    /// queue's `MTLFence` (GPU-only) instead of a wait on the chain event
    /// (`VICTORY_METAL_EVENT_CHAIN=1`: always the event).
    pub fence_chain: bool,
    pub faults: Faults,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            queues: 2,
            sync: false,
            max_in_flight: 2,
            in_flight_ns: 50_000_000,
            fuse_ns: 250_000_000,
            hang_ms: 2000,
            encoder_stall_ms: 60_000,
            ring_bytes: 4 << 20,
            max_pending: 4096,
            max_segments_per_cb: 64,
            max_groups_per_cb: 1 << 16,
            barriers: BarrierPolicy::Title,
            fence_chain: true,
            faults: Faults::default(),
        }
    }
}

impl Config {
    /// `VICTORY_METAL_SYNC`, `VICTORY_METAL_SERIAL`, `VICTORY_METAL_FUSE_MS`
    /// (default 250), `VICTORY_METAL_INFLIGHT` (2), `VICTORY_METAL_INFLIGHT_MS`
    /// (50), `VICTORY_METAL_HANG_MS` (2000), `VICTORY_METAL_ENCODER_STALL_MS`
    /// (60000), `VICTORY_METAL_PASS_DISPATCHES` (segments per command buffer;
    /// 0 = no count limit).
    pub fn from_env() -> Self {
        let d = Self::default();
        let per_cb = env_u64("VICTORY_METAL_PASS_DISPATCHES", d.max_segments_per_cb as u64);
        Self {
            sync: std::env::var("VICTORY_METAL_SYNC").as_deref() == Ok("1"),
            barriers: BarrierPolicy::from_env(),
            fence_chain: std::env::var("VICTORY_METAL_EVENT_CHAIN").as_deref() != Ok("1"),
            fuse_ns: env_u64("VICTORY_METAL_FUSE_MS", 250).max(1) * 1_000_000,
            max_in_flight: env_u64("VICTORY_METAL_INFLIGHT", 2).clamp(1, 16) as u32,
            in_flight_ns: env_u64("VICTORY_METAL_INFLIGHT_MS", 50).max(1) * 1_000_000,
            hang_ms: env_u64("VICTORY_METAL_HANG_MS", 2000).max(1),
            encoder_stall_ms: env_u64("VICTORY_METAL_ENCODER_STALL_MS", 60_000).max(1),
            max_segments_per_cb: if per_cb == 0 { usize::MAX } else { per_cb as usize },
            ..d
        }
    }
}

/// What the timeline tells its owner (the D3D12 layer).
pub struct Hooks {
    /// An outer stop (the process-wide fuse): no commit while it holds.
    pub stopped: Box<dyn Fn() -> bool + Send + Sync>,
    /// The timeline stopped all submission, with why.
    pub stop: Box<dyn Fn(&str) + Send + Sync>,
    /// A fence signal completed on its queue (after the bottom-of-pipe writes
    /// before it): wake what waits for the fence on the CPU.
    pub fence_reached: Box<dyn Fn(FenceId, u64) + Send + Sync>,
    /// Run on the encoder thread before each command buffer is encoded
    /// (pending uploads, residency commits). False stops submission.
    pub prepare: Box<dyn Fn() -> bool + Send + Sync>,
}

impl Default for Hooks {
    fn default() -> Self {
        Self { stopped: Box::new(|| false), stop: Box::new(|_| {}), fence_reached: Box::new(|_, _| {}), prepare: Box::new(|| true) }
    }
}

/// A wait at the start of a segment.
#[derive(Clone)]
pub enum Wait {
    Fence { fence: FenceId, value: u64 },
    /// Memory at `tag` compared with `value`. `ge`: the comparison is
    /// "at least `value`" on a value producers only increase; only then can
    /// it become a GPU wait. `met` reads memory now.
    Tag { tag: TagId, value: u64, ge: bool, met: Arc<dyn Fn() -> bool + Send + Sync> },
}

/// A bottom-of-pipe write: done on the completion thread once the segment's
/// command buffer completed. `tag`: the location and value, when other
/// segments may wait for it.
pub struct Write {
    pub tag: Option<(TagId, u64)>,
    pub action: Box<dyn FnOnce() + Send>,
}

/// What a command buffer is given to encode.
pub struct Encoding<'a> {
    pub device: &'a Device,
    pub cb: &'a CommandBuffer,
    pub ring: &'a mut RingSlice,
}

pub enum Work {
    /// Only waits, signals or writes.
    None,
    /// A compute pass (concurrent encoder, barriers per [`BarrierPolicy`]).
    Compute(Vec<ComputeItem>),
    /// Anything else, encoded into the command buffer by the closure, in a
    /// command buffer of its own. `groups` counts toward the work limit.
    Encode {
        ring_bytes: usize,
        groups: u64,
        encode: Box<dyn FnOnce(&mut Encoding<'_>) -> Result<()> + Send>,
    },
    /// Work that submits and waits for itself (the existing synchronous
    /// backend path, run on the encoder thread so game threads do not wait):
    /// runs once the queue has nothing in flight and its waits are met on
    /// the CPU. False: it failed, and submission stops.
    Host(Box<dyn FnOnce() -> bool + Send>),
}

/// A piece of a queue's timeline: waits, then work, then signals and
/// bottom-of-pipe writes.
pub struct Segment {
    pub label: &'static str,
    pub waits: Vec<Wait>,
    pub work: Work,
    pub signals: Vec<(FenceId, u64)>,
    pub writes: Vec<Write>,
    /// Estimated GPU time (budgeting only; never a reason to refuse).
    pub estimate_ns: u64,
    /// Told the measured GPU time of the segment's command buffer once it
    /// completed (the whole buffer's: exact when it holds one segment).
    pub observe: Option<Box<dyn FnOnce(u64) + Send>>,
    forced: bool,
    held_counted: bool,
}

impl Segment {
    pub fn new(label: &'static str, work: Work) -> Self {
        Self { label, waits: Vec::new(), work, signals: Vec::new(), writes: Vec::new(), estimate_ns: 0, observe: None, forced: false, held_counted: false }
    }
    pub fn wait_fence(mut self, fence: FenceId, value: u64) -> Self {
        self.waits.push(Wait::Fence { fence, value });
        self
    }
    pub fn wait_tag(mut self, tag: TagId, value: u64, ge: bool, met: Arc<dyn Fn() -> bool + Send + Sync>) -> Self {
        self.waits.push(Wait::Tag { tag, value, ge, met });
        self
    }
    pub fn signal(mut self, fence: FenceId, value: u64) -> Self {
        self.signals.push((fence, value));
        self
    }
    pub fn write(mut self, tag: Option<(TagId, u64)>, action: impl FnOnce() + Send + 'static) -> Self {
        self.writes.push(Write { tag, action: Box::new(action) });
        self
    }
    pub fn estimate(mut self, ns: u64) -> Self {
        self.estimate_ns = ns;
        self
    }

    fn groups(&self) -> u64 {
        match &self.work {
            Work::Compute(items) => barrier::groups(items),
            Work::Encode { groups, .. } => *groups,
            Work::None | Work::Host(_) => 0,
        }
    }
    fn ring_bytes(&self) -> usize {
        match &self.work {
            Work::Compute(items) => barrier::ring_need(items),
            Work::Encode { ring_bytes, .. } => ring_bytes.next_multiple_of(ring::ALIGN),
            Work::None | Work::Host(_) => 0,
        }
    }
    /// Whether the command buffer ends after this segment: signals and
    /// writes are due when it completes, and closures get one of their own.
    fn ends_cb(&self) -> bool {
        !self.signals.is_empty() || !self.writes.is_empty() || matches!(self.work, Work::Encode { .. })
    }
}

/// One completed command buffer's GPU interval.
#[derive(Clone, Debug)]
pub struct Interval {
    pub queue: QueueId,
    pub seq: u64,
    pub label: &'static str,
    pub gpu_start: u64,
    pub gpu_end: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    /// Command buffers committed, per queue.
    pub committed: Vec<u64>,
    /// Highest number in flight seen, per queue.
    pub max_in_flight: Vec<u32>,
    /// Fence waits put on the GPU.
    pub gpu_fence_waits: u64,
    /// Tag waits put on the GPU.
    pub gpu_tag_waits: u64,
    /// Segments held on the CPU at a wait (the `Held` fallback), counted once
    /// each.
    pub cpu_held: u64,
    /// Segments run at a forced point with a wait not met.
    pub forced: u64,
    pub barriers: u64,
    pub barriers_elided: u64,
    pub barriers_from_ranges: u64,
    /// Fence signals given by the completion thread after bottom-of-pipe
    /// writes (instead of on the GPU).
    pub cpu_signals: u64,
    /// `MTLFence` waits between encoders of one command buffer.
    pub encoder_fences: u64,
    /// Command buffers ordered after the one before them by the queue's
    /// `MTLFence` instead of a chain event wait.
    pub chain_waits_skipped: u64,
    /// Times a queue's in-flight budget held its next command buffer back.
    pub budget_stalls: u64,
    /// Times ring space held a command buffer back.
    pub ring_stalls: u64,
    pub ring_high_water: Vec<usize>,
    pub host_runs: u64,
    /// Segments dropped after a stop.
    pub discarded: u64,
    /// Times every event was released after a stop.
    pub released: u64,
    /// Encoder thread time spent encoding and committing.
    pub encode_ns: u64,
    /// GPU time of completed command buffers, per queue.
    pub gpu_ns: Vec<u64>,
    /// The last completed command buffers' GPU intervals (at most 4096).
    pub intervals: VecDeque<Interval>,
}

struct FenceState {
    event: SharedEvent,
    /// The highest value a committed command buffer (or the CPU) signals.
    enqueued: u64,
}

struct TagState {
    event: SharedEvent,
    /// The highest value a committed segment writes.
    enqueued: u64,
    /// Every write so far was at least the one before.
    rising: bool,
}

struct QueueShared {
    queue: Queue,
    done: SharedEvent,
    chain: SharedEvent,
    fence: Fence,
    in_flight: Mutex<(u32, u64)>,
    /// When each command buffer in flight was committed (oldest first; the
    /// completion thread finishes them in order), and when the last one
    /// completed: a drain's measure of GPU progress.
    committed_at: Mutex<VecDeque<Instant>>,
    done_at: Mutex<Instant>,
    /// Committed command buffers whose bottom-of-pipe writes have not run
    /// yet. While any are, the queue's fence signals are given by the
    /// completion thread after the writes, not encoded on the GPU: a GPU
    /// signal could let another queue read memory before the write.
    writes_out: std::sync::atomic::AtomicU64,
    ring: Mutex<ring::Ring>,
}

struct Inner {
    device: Device,
    cfg: Config,
    hooks: Hooks,
    stopped: AtomicBool,
    reason: Mutex<Option<String>>,
    queues: Vec<QueueShared>,
    fences: Mutex<HashMap<FenceId, FenceState>>,
    tags: Mutex<HashMap<TagId, TagState>>,
    stats: Mutex<Stats>,
    pending: Mutex<usize>,
    pending_changed: Condvar,
    shutting_down: AtomicBool,
    /// Wakes the encoder thread (after a stop, so waiters are answered).
    wake: Mutex<Option<mpsc::Sender<Msg>>>,
    /// [`Timeline::gpu_idle`]: how many idle guards are held (no command
    /// buffer is committed while any is) and the last sequence committed per
    /// queue. Held by the encoder thread across each commit.
    commits: Mutex<(u32, Vec<u64>)>,
    commits_changed: Condvar,
    /// The last time anything moved: work started or committed, a
    /// completion, host work finished ([`Config::encoder_stall_ms`]).
    progress_at: Mutex<Instant>,
}

enum Msg {
    Submit(QueueId, Vec<Segment>),
    Progress,
    Force,
    /// Answer once nothing more can be encoded now (sync submit).
    Quiesce(mpsc::Sender<Result<()>>),
    /// Answer once everything submitted has completed.
    Drain(mpsc::Sender<Result<bool>>),
    Shutdown,
}

/// An encoded command buffer: it, whether it has GPU work, per segment its
/// signals / CPU-signal flag / writes, the signals and tag writes it
/// commits, and the segments' GPU-time observers.
type Encoded = (
    CommandBuffer,
    bool,
    Vec<(Vec<(FenceId, u64)>, bool, Vec<Write>)>,
    Vec<(FenceId, u64)>,
    Vec<(TagId, u64)>,
    Vec<Box<dyn FnOnce(u64) + Send>>,
);

/// What the completion thread finishes for one command buffer.
struct Flight {
    seq: u64,
    cb: CommandBuffer,
    label: &'static str,
    estimate_ns: u64,
    has_work: bool,
    /// Per segment: its signals, whether the CPU signals them (after the
    /// writes), and its bottom-of-pipe writes.
    done: Vec<(Vec<(FenceId, u64)>, bool, Vec<Write>)>,
    observers: Vec<Box<dyn FnOnce(u64) + Send>>,
    has_writes: bool,
    ring: ring::Token,
}

impl Inner {
    fn fault(&self, reason: String) {
        if self.stopped.swap(true, Ordering::AcqRel) {
            return;
        }
        *lock(&self.reason) = Some(reason.clone());
        self.release();
        (self.hooks.stop)(&reason);
        self.pending_changed.notify_all();
        if let Some(w) = lock(&self.wake).as_ref() {
            let _ = w.send(Msg::Progress);
        }
    }

    /// Sets every fence, tag and chain event to its maximum: command buffers
    /// already committed that wait on one of them can then finish.
    fn release(&self) {
        for f in lock(&self.fences).values() {
            f.event.signal(u64::MAX);
        }
        for t in lock(&self.tags).values() {
            t.event.signal(u64::MAX);
        }
        for q in &self.queues {
            q.chain.signal(u64::MAX);
        }
        lock(&self.stats).released += 1;
    }

    fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::Acquire)
    }

    fn progressed(&self) {
        *lock(&self.progress_at) = Instant::now();
    }

    /// A queue whose committed work has made no progress for longer than
    /// `limit`: the oldest command buffer in flight neither completed nor
    /// was preceded by a completion within it. With how long.
    fn gpu_stalled(&self, limit: Duration) -> Option<(QueueId, Duration)> {
        let now = Instant::now();
        self.queues.iter().enumerate().find_map(|(q, qs)| {
            let oldest = *lock(&qs.committed_at).front()?;
            let since = oldest.max(*lock(&qs.done_at));
            let stalled = now.saturating_duration_since(since);
            (stalled > limit).then_some((q, stalled))
        })
    }

    fn new_event(&self) -> Result<SharedEvent> {
        if self.cfg.faults.fail_event {
            return Err(Error("injected event creation failure".into()));
        }
        SharedEvent::new(&self.device)
    }

    /// The fence's event (made on first use) and its enqueued value.
    fn fence(&self, fence: FenceId) -> Result<(SharedEvent, u64)> {
        let mut fences = lock(&self.fences);
        if let Some(f) = fences.get(&fence) {
            return Ok((f.event.clone(), f.enqueued));
        }
        let event = self.new_event()?;
        fences.insert(fence, FenceState { event: event.clone(), enqueued: 0 });
        Ok((event, 0))
    }

    fn tag(&self, tag: TagId) -> Result<(SharedEvent, u64, bool)> {
        let mut tags = lock(&self.tags);
        if let Some(t) = tags.get(&tag) {
            return Ok((t.event.clone(), t.enqueued, t.rising));
        }
        let event = self.new_event()?;
        tags.insert(tag, TagState { event: event.clone(), enqueued: 0, rising: true });
        Ok((event, 0, true))
    }

    fn pending_done(&self, n: usize) {
        let mut p = lock(&self.pending);
        *p = p.saturating_sub(n);
        self.pending_changed.notify_all();
    }

    /// Finishes one command buffer: waits for it (with the hang timeout),
    /// checks status and GPU time, then runs its bottom-of-pipe writes and
    /// fence notices, and gives back its ring region and budget.
    fn complete(&self, q: QueueId, flight: Flight) {
        let _pool = Pool::new();
        let qs = &self.queues[q];
        let mut measured = 0;
        let ok = (|| -> std::result::Result<(), String> {
            if !qs.done.wait_until_signaled(flight.seq, self.cfg.hang_ms) {
                let reason = format!(
                    "queue {q} command buffer {} ({}) did not complete within {} ms",
                    flight.seq, flight.label, self.cfg.hang_ms
                );
                // Stop and release every event first: if this command buffer
                // waits behind another that failed, it can finish now. Then
                // one more bounded wait.
                self.fault(reason.clone());
                if !qs.done.wait_until_signaled(flight.seq, self.cfg.hang_ms) {
                    return Err(reason);
                }
            }
            flight.cb.wait().map_err(|e| format!("queue {q} command buffer {} ({}) failed: {e}", flight.seq, flight.label))?;
            let gpu = flight.cb.gpu_time_ns();
            measured = gpu;
            let (start, end) = flight.cb.gpu_interval_ns();
            {
                let mut st = lock(&self.stats);
                st.gpu_ns[q] += gpu;
                if st.intervals.len() >= 4096 {
                    st.intervals.pop_front();
                }
                st.intervals.push_back(Interval { queue: q, seq: flight.seq, label: flight.label, gpu_start: start, gpu_end: end });
            }
            if flight.has_work && (gpu == 0 || gpu > self.cfg.fuse_ns) {
                return Err(format!(
                    "queue {q} command buffer {} ({}) GPU time {gpu} ns over the fuse ({} ns, estimate {} ns)",
                    flight.seq, flight.label, self.cfg.fuse_ns, flight.estimate_ns
                ));
            }
            Ok(())
        })();
        if let Err(reason) = ok {
            self.fault(reason);
        }
        if !self.is_stopped() {
            for observe in flight.observers {
                observe(measured);
            }
            for (signals, cpu, writes) in flight.done {
                for w in writes {
                    (w.action)();
                    if let Some((tag, value)) = w.tag
                        && let Some(t) = lock(&self.tags).get(&tag)
                        && t.event.signaled_value() < value
                    {
                        t.event.signal(value);
                    }
                }
                for (fence, value) in signals {
                    if cpu
                        && let Some(f) = lock(&self.fences).get(&fence)
                        && f.event.signaled_value() < value
                    {
                        f.event.signal(value);
                    }
                    (self.hooks.fence_reached)(fence, value);
                }
            }
        }
        if flight.has_writes {
            qs.writes_out.fetch_sub(1, Ordering::AcqRel);
        }
        lock(&qs.committed_at).pop_front();
        *lock(&qs.done_at) = Instant::now();
        self.progressed();
        lock(&qs.ring).fifo.give_back(flight.ring);
        {
            let mut f = lock(&qs.in_flight);
            f.0 -= 1;
            f.1 = f.1.saturating_sub(flight.estimate_ns);
        }
    }
}

/// The queues, their completion threads and the encoder thread.
pub struct Timeline {
    inner: Arc<Inner>,
    tx: Mutex<mpsc::Sender<Msg>>,
    encoder: Option<std::thread::JoinHandle<()>>,
}

impl Timeline {
    pub fn new(device: Device, cfg: Config, hooks: Hooks) -> Result<Self> {
        let _pool = Pool::new();
        let n = cfg.queues.max(1);
        let mut queues = Vec::with_capacity(n);
        for _ in 0..n {
            let queue = device.new_queue(64)?;
            let done = SharedEvent::new(&device)?;
            queues.push(QueueShared {
                queue,
                done,
                chain: SharedEvent::new(&device)?,
                fence: Fence::new(&device)?,
                in_flight: Mutex::new((0, 0)),
                committed_at: Mutex::new(VecDeque::new()),
                done_at: Mutex::new(Instant::now()),
                writes_out: std::sync::atomic::AtomicU64::new(0),
                ring: Mutex::new(ring::Ring::new(&device, cfg.ring_bytes)?),
            });
        }
        let stats = Stats {
            committed: vec![0; n],
            max_in_flight: vec![0; n],
            ring_high_water: vec![0; n],
            gpu_ns: vec![0; n],
            ..Default::default()
        };
        let inner = Arc::new(Inner {
            device,
            cfg: Config { queues: n, ..cfg },
            hooks,
            stopped: AtomicBool::new(false),
            reason: Mutex::new(None),
            queues,
            fences: Mutex::new(HashMap::new()),
            tags: Mutex::new(HashMap::new()),
            stats: Mutex::new(stats),
            pending: Mutex::new(0),
            pending_changed: Condvar::new(),
            shutting_down: AtomicBool::new(false),
            wake: Mutex::new(None),
            commits: Mutex::new((0, vec![0; n])),
            commits_changed: Condvar::new(),
            progress_at: Mutex::new(Instant::now()),
        });
        let (tx, rx) = mpsc::channel();
        *lock(&inner.wake) = Some(tx.clone());
        let mut completion = Vec::with_capacity(n);
        let mut joins = Vec::with_capacity(n);
        if !inner.cfg.sync {
            for q in 0..n {
                let (ctx, crx) = mpsc::channel::<Flight>();
                let inner = inner.clone();
                let progress = tx.clone();
                let exits_after = inner.cfg.faults.completion_exits_after.filter(|&(fq, _)| fq == q).map(|(_, k)| k);
                let join = std::thread::Builder::new()
                    .name(format!("metal-completion-{q}"))
                    .spawn(move || {
                        // A completion thread that ends while work may still
                        // come stops submission: nobody would finish it.
                        struct Exit(Arc<Inner>, QueueId);
                        impl Drop for Exit {
                            fn drop(&mut self) {
                                if !self.0.shutting_down.load(Ordering::Acquire) {
                                    self.0.fault(format!("completion thread of queue {} exited", self.1));
                                }
                            }
                        }
                        let _exit = Exit(inner.clone(), q);
                        let mut done = 0u64;
                        for flight in crx {
                            if exits_after == Some(done) {
                                return;
                            }
                            inner.complete(q, flight);
                            done += 1;
                            let _ = progress.send(Msg::Progress);
                        }
                    })
                    .map_err(|e| Error(format!("completion thread: {e}")))?;
                completion.push(Some(ctx));
                joins.push(join);
            }
        }
        let encoder = {
            let inner = inner.clone();
            std::thread::Builder::new()
                .name("metal-encoder".into())
                .spawn(move || {
                    let mut e = Encoder { inner, pending: (0..n).map(|_| VecDeque::new()).collect(), seq: vec![0; n], fenced: vec![false; n], completion, quiesce: Vec::new(), drains: Vec::new() };
                    e.run(rx);
                    drop(e);
                    for j in joins {
                        let _ = j.join();
                    }
                })
                .map_err(|e| Error(format!("encoder thread: {e}")))?
        };
        Ok(Self { inner, tx: Mutex::new(tx), encoder: Some(encoder) })
    }

    fn send(&self, msg: Msg) -> Result<()> {
        lock(&self.tx).send(msg).map_err(|_| Error("encoder thread gone".into()))
    }

    /// Hands a queue's segments to the encoder thread. Returns at once,
    /// unless more than [`Config::max_pending`] segments wait, or in sync
    /// mode, where it returns once nothing more can be encoded.
    pub fn submit(&self, queue: QueueId, segments: Vec<Segment>) -> Result<()> {
        if let Some(r) = self.stopped() {
            return Err(Error(format!("submission stopped: {r}")));
        }
        if queue >= self.inner.queues.len() {
            return Err(Error(format!("no queue {queue}")));
        }
        {
            let mut p = lock(&self.inner.pending);
            while *p >= self.inner.cfg.max_pending && !self.inner.is_stopped() {
                p = self.inner.pending_changed.wait(p).unwrap_or_else(|e| e.into_inner());
            }
            *p += segments.len();
        }
        self.send(Msg::Submit(queue, segments))?;
        if self.inner.cfg.sync {
            let (tx, rx) = mpsc::channel();
            self.send(Msg::Quiesce(tx))?;
            return rx.recv().map_err(|_| Error("encoder thread gone".into()))?;
        }
        Ok(())
    }

    /// The title signalled a fence from the CPU.
    pub fn cpu_signal(&self, fence: FenceId, value: u64) -> Result<()> {
        let (event, _) = self.inner.fence(fence).inspect_err(|e| self.inner.fault(format!("fence event: {e}")))?;
        if event.signaled_value() < value {
            event.signal(value);
        }
        if let Some(f) = lock(&self.inner.fences).get_mut(&fence) {
            f.enqueued = f.enqueued.max(value);
        }
        self.send(Msg::Progress)
    }

    /// The fence's completed value (0 if never signalled).
    pub fn fence_value(&self, fence: FenceId) -> u64 {
        lock(&self.inner.fences).get(&fence).map_or(0, |f| f.event.signaled_value())
    }

    /// Runs what queues hold at waits that are not met (a present: a wait is
    /// not kept across frames), counting each.
    pub fn force(&self) -> Result<()> {
        self.send(Msg::Force)
    }

    /// Waits until nothing submitted so far is in flight: `Ok(true)` all of
    /// it ran; `Ok(false)` what is left is held at waits on the CPU (nothing
    /// of it is on the GPU). An error: stopped; or a committed command buffer
    /// made no progress on the GPU for `gpu_timeout` (it neither completed
    /// nor followed a completion within it: hung); or the timeline made no
    /// progress at all for [`Config::encoder_stall_ms`]. Time the encoder
    /// thread spends encoding a large batch or running host work, with
    /// nothing in flight, is not a hang (run V0 R3 was stopped for a 2.8 s
    /// batch with `in_flight=[0, 0]`).
    pub fn drain(&self, gpu_timeout: Duration) -> Result<bool> {
        let (tx, rx) = mpsc::channel();
        self.send(Msg::Drain(tx))?;
        let poll = (gpu_timeout / 4).clamp(Duration::from_millis(1), Duration::from_millis(50));
        let backstop = Duration::from_millis(self.inner.cfg.encoder_stall_ms);
        loop {
            match rx.recv_timeout(poll) {
                Ok(answer) => return answer,
                Err(mpsc::RecvTimeoutError::Disconnected) => return Err(Error("encoder thread gone".into())),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
            if let Some(r) = self.stopped() {
                return Err(Error(format!("submission stopped: {r}")));
            }
            if let Some((q, stalled)) = self.inner.gpu_stalled(gpu_timeout) {
                return Err(Error(format!(
                    "drain timed out: queue {q} has committed work that made no GPU progress for {} ms",
                    stalled.as_millis()
                )));
            }
            let idle = lock(&self.inner.progress_at).elapsed();
            if idle > backstop {
                return Err(Error(format!("drain timed out: the timeline made no progress for {} ms", idle.as_millis())));
            }
        }
    }

    /// Stops all submission now (an outer safety stop) and releases every
    /// event committed work may wait on.
    pub fn stop(&self, reason: &str) {
        self.inner.fault(format!("stopped: {reason}"));
    }

    /// Command buffers in flight per queue.
    pub fn in_flight(&self) -> Vec<u32> {
        self.inner.queues.iter().map(|q| lock(&q.in_flight).0).collect()
    }

    /// Pauses commits and waits until the GPU has finished every command
    /// buffer committed so far (each signals its queue's `done` event at its
    /// end, so this does not depend on the completion threads). Commits
    /// resume when the guard drops. For work that must never overlap this
    /// timeline's GPU work (residency-set wiring). Callable from any thread,
    /// the encoder thread included (a `prepare` hook): nothing it waits for
    /// needs the encoder. Bounded: past `timeout` the timeline is stopped
    /// (every event released, so waiters drain) and waited for once more;
    /// the guard says whether the GPU was idle.
    pub fn gpu_idle(&self, timeout: Duration) -> GpuIdle<'_> {
        let targets = {
            let mut c = lock(&self.inner.commits);
            c.0 += 1;
            c.1.clone()
        };
        let ms = u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX).max(1);
        let reached = |ms| self.inner.queues.iter().zip(&targets).all(|(qs, &seq)| seq == 0 || qs.done.wait_until_signaled(seq, ms));
        let mut idle = reached(ms);
        if !idle {
            self.inner.fault(format!("the GPU did not go idle within {ms} ms (for work that must not overlap it)"));
            idle = reached(ms);
        }
        GpuIdle { inner: &self.inner, idle }
    }

    pub fn stopped(&self) -> Option<String> {
        if self.inner.is_stopped() { Some(lock(&self.inner.reason).clone().unwrap_or_default()) } else { None }
    }

    pub fn stats(&self) -> Stats {
        let mut s = lock(&self.inner.stats).clone();
        for (q, qs) in self.inner.queues.iter().enumerate() {
            s.ring_high_water[q] = lock(&qs.ring).fifo.high_water;
        }
        s
    }

    /// The Metal queue of a timeline queue (to attach residency sets).
    pub fn queue(&self, queue: QueueId) -> &Queue {
        &self.inner.queues[queue].queue
    }

    /// The ring buffers (to make them resident).
    pub fn ring_buffers(&self) -> Vec<crate::Buffer> {
        self.inner.queues.iter().map(|q| lock(&q.ring).buffer.clone()).collect()
    }

    pub fn sync(&self) -> bool {
        self.inner.cfg.sync
    }
}

impl Drop for Timeline {
    fn drop(&mut self) {
        self.inner.shutting_down.store(true, Ordering::Release);
        let _ = self.send(Msg::Shutdown);
        if let Some(j) = self.encoder.take() {
            let _ = j.join();
        }
    }
}

/// The encoder thread's state.
struct Encoder {
    inner: Arc<Inner>,
    pending: Vec<VecDeque<Segment>>,
    seq: Vec<u64>,
    /// Whether the queue's last command buffer was compute only: its last
    /// encoder updated the queue's fence, which the next one can wait on.
    fenced: Vec<bool>,
    completion: Vec<Option<mpsc::Sender<Flight>>>,
    quiesce: Vec<mpsc::Sender<Result<()>>>,
    drains: Vec<mpsc::Sender<Result<bool>>>,
}

/// How a segment's waits stand.
enum Ready {
    /// Every wait met, or put on the GPU as these event waits.
    Now(Vec<(SharedEvent, u64, bool)>),
    Held,
}

impl Encoder {
    fn run(&mut self, rx: mpsc::Receiver<Msg>) {
        while let Ok(first) = rx.recv() {
            let mut msg = Some(first);
            while let Some(m) = msg.take() {
                match m {
                    Msg::Submit(q, segs) => self.pending[q].extend(segs),
                    Msg::Progress => {}
                    Msg::Force => {
                        for q in &mut self.pending {
                            for s in q.iter_mut() {
                                s.forced = true;
                            }
                        }
                    }
                    Msg::Quiesce(tx) => self.quiesce.push(tx),
                    Msg::Drain(tx) => self.drains.push(tx),
                    Msg::Shutdown => {
                        self.discard();
                        self.completion.clear();
                        return;
                    }
                }
                msg = rx.try_recv().ok();
            }
            self.pump();
            self.answer();
        }
    }

    fn discard(&mut self) {
        let n: usize = self.pending.iter().map(VecDeque::len).sum();
        if n > 0 {
            for q in &mut self.pending {
                q.clear();
            }
            lock(&self.inner.stats).discarded += n as u64;
            self.inner.pending_done(n);
        }
    }

    fn idle(&self) -> bool {
        self.pending.iter().all(VecDeque::is_empty) && self.inner.queues.iter().all(|q| lock(&q.in_flight).0 == 0)
    }

    fn answer(&mut self) {
        let stopped = self.inner.is_stopped();
        let err = || Error(format!("submission stopped: {}", lock(&self.inner.reason).clone().unwrap_or_default()));
        for tx in self.quiesce.drain(..) {
            let _ = tx.send(if stopped { Err(err()) } else { Ok(()) });
        }
        // Held: work is left, nothing is in flight, and every queue's next
        // segment stands at a wait (not merely behind a budget that a
        // completion freed after the pump looked: its progress message
        // pumps again).
        let held = self.pending.iter().any(|q| !q.is_empty())
            && self.inner.queues.iter().all(|q| lock(&q.in_flight).0 == 0)
            && self.pending.iter().all(|q| q.front().is_none_or(|s| matches!(self.ready(s), Ready::Held)));
        if stopped || self.idle() || held {
            for tx in self.drains.drain(..) {
                let _ = tx.send(if stopped { Err(err()) } else { Ok(!held) });
            }
        }
    }

    fn pump(&mut self) {
        loop {
            if self.inner.is_stopped() || (self.inner.hooks.stopped)() {
                if !self.inner.is_stopped() {
                    self.inner.fault("outer safety stop".into());
                }
                self.discard();
                return;
            }
            let mut moved = false;
            for q in 0..self.pending.len() {
                while self.step(q) {
                    moved = true;
                }
            }
            if !moved {
                return;
            }
        }
    }

    /// The waits' state; in sync mode only waits met on the CPU pass.
    fn ready(&self, s: &Segment) -> Ready {
        if s.forced {
            return Ready::Now(Vec::new());
        }
        let sync = self.inner.cfg.sync;
        let mut gpu = Vec::new();
        for w in &s.waits {
            match w {
                Wait::Fence { fence, value } => {
                    let (event, enqueued) = match self.inner.fence(*fence) {
                        Ok(f) => f,
                        Err(e) => {
                            self.inner.fault(format!("fence event: {e}"));
                            return Ready::Held;
                        }
                    };
                    if event.signaled_value() >= *value {
                        continue;
                    }
                    if !sync && enqueued >= *value {
                        gpu.push((event, *value, false));
                    } else {
                        return Ready::Held;
                    }
                }
                Wait::Tag { tag, value, ge, met } => {
                    if met() {
                        continue;
                    }
                    let (event, enqueued, rising) = match self.inner.tag(*tag) {
                        Ok(t) => t,
                        Err(e) => {
                            self.inner.fault(format!("tag event: {e}"));
                            return Ready::Held;
                        }
                    };
                    if !sync && *ge && rising && enqueued >= *value {
                        gpu.push((event, *value, true));
                    } else {
                        return Ready::Held;
                    }
                }
            }
        }
        Ready::Now(gpu)
    }

    /// Encodes and commits the queue's next command buffer, if its waits,
    /// budget and ring space allow. Whether anything moved.
    fn step(&mut self, q: QueueId) -> bool {
        let inner = self.inner.clone();
        let cfg = &inner.cfg;
        let qs = &inner.queues[q];
        let Some(front) = self.pending[q].front() else { return false };
        let (front_estimate, front_host) = (front.estimate_ns, matches!(front.work, Work::Host(_)));
        let (in_flight, in_flight_ns) = *lock(&qs.in_flight);
        if !cfg.sync
            && (in_flight >= cfg.max_in_flight
                || (in_flight > 0 && in_flight_ns + front_estimate.min(cfg.in_flight_ns) > cfg.in_flight_ns))
        {
            lock(&inner.stats).budget_stalls += 1;
            return false;
        }
        let ready = match self.ready(front) {
            // Host work cannot carry GPU waits: it waits until they are met.
            Ready::Now(w) if front_host && !w.is_empty() => Ready::Held,
            r => r,
        };
        let gpu_waits = match ready {
            Ready::Now(w) => w,
            Ready::Held => {
                if let Some(front) = self.pending[q].front_mut()
                    && !front.held_counted
                {
                    front.held_counted = true;
                    lock(&inner.stats).cpu_held += 1;
                }
                return false;
            }
        };
        let started = Instant::now();
        // Starting work counts as progress (the backstop measures how long
        // one piece of work takes, not how long the timeline sat idle).
        inner.progressed();
        if front_host {
            if in_flight > 0 {
                return false;
            }
            return self.host(q);
        }
        // Gather segments for one command buffer.
        let mut segs = vec![self.pending[q].pop_front().expect("front exists")];
        let mut groups = segs[0].groups();
        let mut estimate = segs[0].estimate_ns;
        let mut ring_bytes = segs[0].ring_bytes();
        while !segs.last().expect("one").ends_cb() && segs.len() < cfg.max_segments_per_cb {
            let Some(next) = self.pending[q].front() else { break };
            if matches!(next.work, Work::Host(_) | Work::Encode { .. }) {
                break;
            }
            // Later segments join only if their waits are met on the CPU.
            if !next.waits.is_empty() && !matches!(self.ready(next), Ready::Now(ref w) if w.is_empty()) {
                break;
            }
            let g = next.groups();
            if groups.saturating_add(g) > cfg.max_groups_per_cb || estimate + next.estimate_ns > cfg.in_flight_ns {
                break;
            }
            groups += g;
            estimate += next.estimate_ns;
            ring_bytes += next.ring_bytes();
            segs.push(self.pending[q].pop_front().expect("front exists"));
        }
        let n = segs.len();
        let label = segs[0].label;
        let put_back = |this: &mut Self, segs: Vec<Segment>| {
            for s in segs.into_iter().rev() {
                this.pending[q].push_front(s);
            }
        };
        // Ring space.
        let token = {
            let mut ring = lock(&qs.ring);
            if !ring.fifo.fits_ever(ring_bytes) {
                drop(ring);
                inner.fault(format!("queue {q}: a command buffer needs {ring_bytes} ring bytes, more than the ring holds"));
                put_back(self, segs);
                return false;
            }
            match ring.fifo.take(ring_bytes) {
                Some(t) => t,
                None => {
                    drop(ring);
                    lock(&inner.stats).ring_stalls += 1;
                    put_back(self, segs);
                    return false;
                }
            }
        };
        if !(inner.hooks.prepare)() {
            lock(&qs.ring).fifo.give_back(token);
            inner.fault(format!("queue {q}: preparing a command buffer failed"));
            inner.pending_done(n);
            return false;
        }
        let mut slice = if ring_bytes == 0 { RingSlice::empty(lock(&qs.ring).buffer.clone()) } else { lock(&qs.ring).slice(&token) };
        let _pool = Pool::new();
        let compute = |s: &Segment| matches!(&s.work, Work::Compute(items) if !items.is_empty());
        let this_fenced = segs.iter().all(compute);
        let fence_first = cfg.fence_chain && self.fenced[q] && compute(&segs[0]);
        let encoded = (|| -> Result<Encoded> {
            let cb = qs.queue.command_buffer()?;
            let seq = self.seq[q] + 1;
            if seq > 1 && !fence_first {
                cb.encode_wait(&qs.chain, seq - 1);
            } else if seq > 1 {
                lock(&inner.stats).chain_waits_skipped += 1;
            }
            {
                let mut st = lock(&inner.stats);
                for (event, value, tag) in &gpu_waits {
                    cb.encode_wait(event, *value);
                    if *tag { st.gpu_tag_waits += 1 } else { st.gpu_fence_waits += 1 }
                }
            }
            let mut has_work = false;
            let mut done = Vec::with_capacity(n);
            let mut signalled = Vec::new();
            let mut tagged = Vec::new();
            let mut encoders = 0usize;
            let mut observers = Vec::new();
            for mut s in segs {
                observers.extend(s.observe.take());
                if s.forced && !s.waits.is_empty() {
                    lock(&inner.stats).forced += 1;
                }
                match s.work {
                    Work::None | Work::Host(_) => {}
                    Work::Compute(items) => {
                        has_work |= !items.is_empty();
                        // Encoders of one command buffer are ordered by the
                        // queue's fence (untracked memory).
                        let wait = (encoders > 0 || fence_first).then_some(&qs.fence);
                        let plan = barrier::encode(&cb, &items, cfg.barriers, &mut slice, wait, Some(&qs.fence))?;
                        let mut st = lock(&inner.stats);
                        st.barriers += plan.barriers;
                        st.barriers_elided += plan.elided;
                        st.barriers_from_ranges += plan.from_ranges;
                        st.encoder_fences += u64::from(wait.is_some());
                        encoders += 1;
                    }
                    Work::Encode { encode, .. } => {
                        has_work = true;
                        encode(&mut Encoding { device: &inner.device, cb: &cb, ring: &mut slice })?;
                        encoders += 1;
                    }
                }
                let cpu = !s.writes.is_empty() || qs.writes_out.load(Ordering::Acquire) > 0;
                for &(fence, value) in &s.signals {
                    let (event, _) = inner.fence(fence)?;
                    if cpu {
                        lock(&inner.stats).cpu_signals += 1;
                    } else {
                        cb.encode_signal(&event, value);
                    }
                    signalled.push((fence, value));
                }
                tagged.extend(s.writes.iter().filter_map(|w| w.tag));
                done.push((s.signals, cpu, s.writes));
            }
            cb.encode_signal(&qs.chain, seq);
            cb.encode_signal(&qs.done, seq);
            Ok((cb, has_work, done, signalled, tagged, observers))
        })();
        let (cb, has_work, done, signalled, tagged, observers) = match encoded {
            Ok(e) => e,
            Err(e) => {
                lock(&qs.ring).fifo.give_back(token);
                inner.fault(format!("queue {q}: encoding failed: {e}"));
                inner.pending_done(n);
                return false;
            }
        };
        if inner.is_stopped() || (inner.hooks.stopped)() {
            lock(&qs.ring).fifo.give_back(token);
            inner.fault("outer safety stop".into());
            inner.pending_done(n);
            return false;
        }
        // No commit while a `gpu_idle` guard is held (bounded: a stop ends
        // the wait, and the command buffer is committed to be completed as
        // a stopped one).
        let mut commits = lock(&inner.commits);
        while commits.0 > 0 && !inner.is_stopped() {
            commits = inner.commits_changed.wait_timeout(commits, Duration::from_millis(50)).unwrap_or_else(|e| e.into_inner()).0;
        }
        self.seq[q] += 1;
        self.fenced[q] = this_fenced;
        let seq = self.seq[q];
        commits.1[q] = seq;
        lock(&qs.committed_at).push_back(Instant::now());
        {
            let mut f = lock(&qs.in_flight);
            f.0 += 1;
            f.1 += estimate.min(cfg.in_flight_ns);
            let mut st = lock(&inner.stats);
            st.max_in_flight[q] = st.max_in_flight[q].max(f.0);
            st.committed[q] += 1;
        }
        let has_writes = done.iter().any(|d| !d.2.is_empty());
        if has_writes {
            qs.writes_out.fetch_add(1, Ordering::AcqRel);
        }
        let flight = Flight { seq, cb: cb.clone(), label, estimate_ns: estimate.min(cfg.in_flight_ns), has_work, done, observers, has_writes, ring: token };
        if cfg.sync {
            cb.commit();
            drop(commits);
            self.enqueued(&signalled, &tagged);
            inner.complete(q, flight);
        } else {
            // Hand over before commit: a command buffer nobody finishes is
            // never committed.
            let handed = self.completion[q].as_ref().is_some_and(|c| c.send(flight).is_ok());
            if !handed {
                commits.1[q] = seq - 1;
                drop(commits);
                lock(&qs.in_flight).0 -= 1;
                lock(&qs.committed_at).pop_back();
                inner.fault(format!("completion thread of queue {q} is gone"));
                inner.pending_done(n);
                return false;
            }
            cb.commit();
            drop(commits);
            self.enqueued(&signalled, &tagged);
        }
        lock(&inner.stats).encode_ns += started.elapsed().as_nanos() as u64;
        inner.progressed();
        inner.pending_done(n);
        true
    }

    /// Signals and tag writes of a command buffer just committed may now be
    /// waited for on the GPU.
    fn enqueued(&self, signalled: &[(FenceId, u64)], tagged: &[(TagId, u64)]) {
        let mut fences = lock(&self.inner.fences);
        for &(fence, value) in signalled {
            if let Some(f) = fences.get_mut(&fence) {
                f.enqueued = f.enqueued.max(value);
            }
        }
        drop(fences);
        let mut tags = lock(&self.inner.tags);
        for &(tag, value) in tagged {
            if let Some(t) = tags.get_mut(&tag) {
                if value < t.enqueued {
                    t.rising = false;
                }
                t.enqueued = t.enqueued.max(value);
            } else if let Ok(event) = self.inner.new_event() {
                tags.insert(tag, TagState { event, enqueued: value, rising: true });
            }
        }
    }

    /// Runs a host segment on this thread (its queue has nothing in flight),
    /// then its signals and writes.
    fn host(&mut self, q: QueueId) -> bool {
        let inner = self.inner.clone();
        let s = self.pending[q].pop_front().expect("front exists");
        let Work::Host(run) = s.work else { unreachable!("host segment") };
        if s.forced && !s.waits.is_empty() {
            lock(&inner.stats).forced += 1;
        }
        let ok = run();
        lock(&inner.stats).host_runs += 1;
        inner.progressed();
        if !ok {
            inner.fault(format!("queue {q}: host work ({}) failed", s.label));
            inner.pending_done(1);
            return false;
        }
        for w in s.writes {
            (w.action)();
            if let Some((tag, value)) = w.tag {
                self.enqueued(&[], &[(tag, value)]);
                if let Some(t) = lock(&inner.tags).get(&tag)
                    && t.event.signaled_value() < value
                {
                    t.event.signal(value);
                }
            }
        }
        for (fence, value) in s.signals {
            match inner.fence(fence) {
                Ok((event, _)) => {
                    if event.signaled_value() < value {
                        event.signal(value);
                    }
                    self.enqueued(&[(fence, value)], &[]);
                    (inner.hooks.fence_reached)(fence, value);
                }
                Err(e) => inner.fault(format!("fence event: {e}")),
            }
        }
        inner.pending_done(1);
        true
    }
}
