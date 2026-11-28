//! Synthetic tests of the execution model. Every GPU workload is a
//! fixed-trip-count kernel; the spin kernel is calibrated upward in steps
//! from a tiny size, each step checked before the next, and never targets
//! more than about 1 ms per dispatch.

use super::*;
use crate::{Buffer, ComputePipeline};
use std::sync::atomic::AtomicU64;

const WORDS: usize = 4096;

const KERNELS: &str = "#include <metal_stdlib>
using namespace metal;
struct Args { uint dst; uint src; uint add; uint pad; };
kernel void add(device uint *b [[buffer(0)]], constant Args &a [[buffer(1)]]) { b[a.dst] = b[a.src] + a.add; }
kernel void ring_add(device uint *b [[buffer(0)]], device const Args &a [[buffer(2)]]) { b[a.dst] = b[a.src] + a.add; }
";

fn spin_source(iterations: u32) -> String {
    format!(
        "#include <metal_stdlib>
using namespace metal;
struct Args {{ uint dst; uint src; uint add; uint pad; }};
kernel void spin(device uint *o [[buffer(0)]], constant Args &a [[buffer(1)]], uint g [[thread_position_in_grid]]) {{
  uint v = a.src + g;
  for (uint i = 0; i < {iterations}u; ++i) {{ v = v * a.add + 1013904223u; }}
  o[a.dst + g] = v;
}}"
    )
}

fn lcg(seed: u32, mul: u32, iterations: u32) -> u32 {
    let mut v = seed;
    for _ in 0..iterations {
        v = v.wrapping_mul(mul).wrapping_add(1013904223);
    }
    v
}

const MUL: u32 = 1664525;
const SPIN_THREADS: u64 = 32;
/// Where calibration dispatches write (out of every test's way).
const CALIBRATION: u32 = 2048;

struct Gpu {
    device: Device,
    add: ComputePipeline,
    ring_add: ComputePipeline,
    buf: Buffer,
    cpu: *mut u32,
}

unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

fn args(dst: u32, src: u32, add: u32) -> Vec<u8> {
    [dst, src, add, 0].iter().flat_map(|v| v.to_le_bytes()).collect()
}

impl Gpu {
    fn new() -> Self {
        let device = Device::first().unwrap();
        let lib = device.library_msl(KERNELS).unwrap();
        let add = device.compute_pipeline_from_library(&lib, "add").unwrap();
        let ring_add = device.compute_pipeline_from_library(&lib, "ring_add").unwrap();
        let (buf, cpu) = device.shared_buffer_untracked_mapped(&vec![0u8; WORDS * 4]).unwrap();
        Self { device, add, ring_add, buf, cpu: cpu.cast() }
    }
    fn get(&self, i: usize) -> u32 {
        assert!(i < WORDS);
        // SAFETY: inside the buffer; read after the GPU work completed.
        unsafe { self.cpu.add(i).read_volatile() }
    }
    fn set(&self, i: usize, v: u32) {
        assert!(i < WORDS);
        // SAFETY: inside the buffer; no GPU work in flight.
        unsafe { self.cpu.add(i).write_volatile(v) }
    }
    fn words(&self, n: usize) -> Vec<u32> {
        (0..n).map(|i| self.get(i)).collect()
    }
    fn range(&self, i: u32) -> Range {
        Range::of(&self.buf, u64::from(i) * 4, 4)
    }
    /// `b[dst] = b[src] + add`, one thread.
    fn add(&self, dst: u32, src: u32, add: u32) -> ComputeItem {
        ComputeItem::Dispatch(Dispatch {
            pipeline: self.add.clone(),
            threads_per_group: [1; 3],
            groups: [1; 3],
            buffers: vec![(self.buf.clone(), 0, 0)],
            bytes: Some((args(dst, src, add), 1)),
            ring: None,
            patches: vec![],
            reads: vec![self.range(src)],
            writes: vec![self.range(dst)],
        })
    }
    /// The same with its arguments in the ring.
    fn ring_add(&self, dst: u32, src: u32, add: u32) -> ComputeItem {
        ComputeItem::Dispatch(Dispatch {
            pipeline: self.ring_add.clone(),
            threads_per_group: [1; 3],
            groups: [1; 3],
            buffers: vec![(self.buf.clone(), 0, 0)],
            bytes: None,
            ring: Some((args(dst, src, add), 2)),
            patches: vec![],
            reads: vec![self.range(src)],
            writes: vec![self.range(dst)],
        })
    }
    fn timeline(&self, cfg: Config) -> Timeline {
        Timeline::new(self.device.clone(), cfg, Hooks::default()).unwrap()
    }
}

/// A spin kernel whose one-group dispatch takes about the target (at most
/// ~1 ms), calibrated upward from 4096 iterations with a check at each step.
struct Spin {
    pso: ComputePipeline,
    iterations: u32,
    ns: u64,
}

impl Gpu {
    fn spin_once(&self, pso: &ComputePipeline) -> u64 {
        let queue = self.device.new_queue(1).unwrap();
        let cb = queue.command_buffer().unwrap();
        let a = args(CALIBRATION, 7, MUL);
        cb.compute(&[
            crate::Compute::Pipeline { pipeline: pso, threads_per_group: [SPIN_THREADS, 1, 1] },
            crate::Compute::Buffer { buffer: &self.buf, offset: 0, index: 0 },
            crate::Compute::Bytes { bytes: &a, index: 1 },
            crate::Compute::Dispatch { groups: [1, 1, 1] },
        ])
        .unwrap();
        cb.commit();
        cb.wait().unwrap();
        cb.gpu_time_ns()
    }

    fn spin(&self, target_ns: u64) -> Spin {
        assert!(target_ns <= 1_000_000, "spin dispatches stay at or under 1 ms");
        let mut iterations = 4096u32;
        loop {
            let pso = self.device.compute_pipeline_msl(&spin_source(iterations), "spin").unwrap();
            self.spin_once(&pso); // compile/warm
            let ns = self.spin_once(&pso);
            assert!(ns < 20_000_000, "calibration step of {iterations} iterations took {ns} ns");
            assert_eq!(self.get(CALIBRATION as usize + 1), lcg(8, MUL, iterations), "spin output");
            // Next step at most 4x, and only while predicted under the target.
            if ns * 4 > target_ns || iterations >= 1 << 22 {
                // Scale to the target from the measured point (at most 4x).
                let scale = (target_ns as f64 / ns.max(1) as f64).clamp(0.25, 4.0);
                let final_iterations = ((f64::from(iterations) * scale) as u32).max(1024);
                let pso = self.device.compute_pipeline_msl(&spin_source(final_iterations), "spin").unwrap();
                self.spin_once(&pso);
                let ns = self.spin_once(&pso);
                assert!(ns < 5_000_000, "final spin of {final_iterations} iterations took {ns} ns");
                println!("spin: {final_iterations} iterations ≈ {ns} ns per one-group dispatch");
                return Spin { pso, iterations: final_iterations, ns };
            }
            iterations *= 4;
        }
    }

    fn spin_item(&self, s: &Spin, dst: u32, seed: u32) -> ComputeItem {
        ComputeItem::Dispatch(Dispatch {
            pipeline: s.pso.clone(),
            threads_per_group: [SPIN_THREADS, 1, 1],
            groups: [1; 3],
            buffers: vec![(self.buf.clone(), 0, 0)],
            bytes: Some((args(dst, seed, MUL), 1)),
            ring: None,
            patches: vec![],
            reads: vec![],
            writes: vec![Range::of(&self.buf, u64::from(dst) * 4, SPIN_THREADS * 4)],
        })
    }
}

type Log = Arc<Mutex<Vec<(usize, u32)>>>;

fn logger(log: &Log, queue: usize, id: u32) -> impl FnOnce() + Send + 'static {
    let log = log.clone();
    move || lock(&log).push((queue, id))
}

fn cfg(sync: bool) -> Config {
    Config { sync, ..Config::default() }
}

const F0: FenceId = 0xf0;
const F1: FenceId = 0xf1;
const SECOND: Duration = Duration::from_secs(5);

fn fence_order(sync: bool) {
    let _pool = Pool::new();
    let g = Gpu::new();
    g.set(0, 1);
    let log: Log = Default::default();
    let t = g.timeline(cfg(sync));
    // The graphics queue is given its wait before the compute queue has
    // been given the signal: it must hold, then run after.
    t.submit(0, vec![Segment::new("consumer", Work::Compute(vec![g.add(2, 1, 10)])).wait_fence(F1, 1).write(None, logger(&log, 0, 2))]).unwrap();
    t.submit(1, vec![Segment::new("producer", Work::Compute(vec![g.add(1, 0, 1)])).signal(F1, 1).write(None, logger(&log, 1, 1))]).unwrap();
    assert!(t.drain(SECOND).unwrap(), "everything ran");
    assert_eq!(g.get(2), 12);
    assert_eq!(*lock(&log), [(1, 1), (0, 2)]);
    assert_eq!(t.fence_value(F1), 1);
    let s = t.stats();
    assert_eq!(s.cpu_held, 1, "{s:?}");
    assert_eq!(s.committed, [1, 1]);
    assert!(t.stopped().is_none());
}

#[test]
#[ignore = "requires Metal"]
fn two_queues_fence_order_async() {
    fence_order(false);
}

#[test]
#[ignore = "requires Metal"]
fn two_queues_fence_order_sync() {
    fence_order(true);
}

/// A fence signal with no bottom-of-pipe writes on its queue is encoded on
/// the GPU; the other queue's wait for it, given after the signal was
/// committed, is a GPU wait.
#[test]
#[ignore = "requires Metal"]
fn gpu_fence_wait_after_committed_signal() {
    let _pool = Pool::new();
    let g = Gpu::new();
    let spin = g.spin(1_000_000);
    // The producer spins (~2 ms) first so the consumer is encoded while it
    // runs. On a loaded machine the encoder can lose that race (the
    // producer done before the consumer is looked at: no wait needed at
    // all), so a few rounds, each checked for order, one needing a GPU wait.
    let mut waits = Vec::new();
    for _ in 0..5 {
        g.set(0, 5);
        g.set(1, 0);
        g.set(2, 0);
        let t = g.timeline(cfg(false));
        t.submit(1, vec![Segment::new("producer", Work::Compute(vec![g.spin_item(&spin, 64, 3), ComputeItem::Barrier, g.spin_item(&spin, 64, 4), ComputeItem::Barrier, g.add(1, 0, 1)])).signal(F1, 1)]).unwrap();
        t.submit(0, vec![Segment::new("consumer", Work::Compute(vec![g.add(2, 1, 10)])).wait_fence(F1, 1)]).unwrap();
        assert!(t.drain(SECOND).unwrap(), "everything ran");
        assert_eq!(g.get(2), 16);
        let s = t.stats();
        println!("stats: gpu_fence_waits={} cpu_held={} cpu_signals={}", s.gpu_fence_waits, s.cpu_held, s.cpu_signals);
        assert_eq!(s.cpu_signals, 0);
        assert!(t.stopped().is_none());
        waits.push(s.gpu_fence_waits);
        if s.gpu_fence_waits == 1 {
            return;
        }
    }
    panic!("no round needed a GPU wait: {waits:?}");
}

fn tag_order(sync: bool, ge: bool) -> Stats {
    let _pool = Pool::new();
    let g = Gpu::new();
    g.set(0, 1);
    let memory = Arc::new(AtomicU64::new(0));
    let log: Log = Default::default();
    let spin = g.spin(1_000_000);
    let t = g.timeline(cfg(sync));
    // Producer first (~2 ms) so the consumer sees its committed write.
    let mw = memory.clone();
    let l = logger(&log, 1, 1);
    t.submit(1, vec![Segment::new("producer", Work::Compute(vec![g.spin_item(&spin, 64, 3), ComputeItem::Barrier, g.spin_item(&spin, 64, 4), ComputeItem::Barrier, g.add(1, 0, 1)]))
        .write(Some((0x7a9, 5)), move || {
            mw.store(5, Ordering::Release);
            l();
        })]).unwrap();
    let m = memory.clone();
    t.submit(0, vec![Segment::new("consumer", Work::Compute(vec![g.add(2, 1, 10)]))
        .wait_tag(0x7a9, 5, ge, Arc::new(move || m.load(Ordering::Acquire) >= 5))
        .write(None, logger(&log, 0, 2))]).unwrap();
    assert!(t.drain(SECOND).unwrap(), "everything ran");
    assert_eq!(g.get(2), 12);
    assert_eq!(*lock(&log), [(1, 1), (0, 2)]);
    assert!(t.stopped().is_none());
    t.stats()
}

#[test]
#[ignore = "requires Metal"]
fn tag_wait_becomes_gpu_wait() {
    let s = tag_order(false, true);
    println!("stats: gpu_tag_waits={} cpu_held={}", s.gpu_tag_waits, s.cpu_held);
    assert_eq!(s.gpu_tag_waits, 1, "{s:?}");
}

#[test]
#[ignore = "requires Metal"]
fn tag_wait_falls_back_to_cpu() {
    // Not a ">=" wait: never on the GPU.
    let s = tag_order(false, false);
    assert_eq!(s.gpu_tag_waits, 0);
    assert_eq!(s.cpu_held, 1, "{s:?}");
    // And the sync path never puts waits on the GPU.
    let s = tag_order(true, true);
    assert_eq!(s.gpu_tag_waits + s.gpu_fence_waits, 0);
}

#[test]
#[ignore = "requires Metal"]
fn cpu_signal_releases_held_segment() {
    let _pool = Pool::new();
    let g = Gpu::new();
    g.set(0, 4);
    let t = g.timeline(cfg(false));
    t.submit(0, vec![Segment::new("waits", Work::Compute(vec![g.add(1, 0, 1)])).wait_fence(F0, 3)]).unwrap();
    assert!(!t.drain(SECOND).unwrap(), "held at a wait nothing in flight meets");
    assert_eq!(t.stats().committed[0], 0);
    t.cpu_signal(F0, 3).unwrap();
    assert!(t.drain(SECOND).unwrap(), "everything ran");
    assert_eq!(g.get(1), 5);
    assert_eq!(t.stats().cpu_held, 1);
}

#[test]
#[ignore = "requires Metal"]
fn force_runs_held_segment() {
    let _pool = Pool::new();
    let g = Gpu::new();
    g.set(0, 4);
    let t = g.timeline(cfg(false));
    t.submit(0, vec![Segment::new("waits", Work::Compute(vec![g.add(1, 0, 1)])).wait_fence(F0, 9)]).unwrap();
    t.force().unwrap();
    assert!(t.drain(SECOND).unwrap(), "everything ran");
    assert_eq!(g.get(1), 5);
    assert_eq!(t.stats().forced, 1);
}

/// Eight rounds chained across the two queues by fences and a tag, with
/// independent work in each round: (final memory, bottom-of-pipe writes in
/// the order they ran, fence values, stats).
fn chained_workload(sync: bool) -> (Vec<u32>, Vec<(usize, u32)>, [u64; 2], Stats) {
    let _pool = Pool::new();
    let g = Gpu::new();
    g.set(0, 3);
    let memory = Arc::new(AtomicU64::new(0));
    let log: Log = Default::default();
    let t = g.timeline(cfg(sync));
    for r in 0..8u32 {
        // Graphics: waits for compute's round, uses its result, signals.
        let m = memory.clone();
        let want = u64::from(r) + 1;
        let gfx = Segment::new("gfx", Work::Compute(vec![
            g.add(100 + r, 10 + r, 1000),
            g.add(200 + r, 0, r),
            ComputeItem::Barrier,
            g.add(300 + r, 100 + r, 1),
        ]))
        .wait_fence(F1, want)
        .wait_tag(0x7a9, want, true, Arc::new(move || m.load(Ordering::Acquire) >= want))
        .signal(F0, want)
        .write(None, logger(&log, 0, r));
        t.submit(0, vec![gfx]).unwrap();
        // Compute: waits for graphics' previous round.
        let src = if r == 0 { 0 } else { 300 + r - 1 };
        let mw = memory.clone();
        let l = logger(&log, 1, r);
        let mut comp = Segment::new("compute", Work::Compute(vec![g.add(10 + r, src, r + 1), g.add(400 + r, 0, 7)]))
            .signal(F1, want)
            .write(Some((0x7a9, want)), move || {
                mw.store(want, Ordering::Release);
                l();
            });
        if r > 0 {
            comp = comp.wait_fence(F0, u64::from(r));
        }
        t.submit(1, vec![comp]).unwrap();
    }
    assert!(t.drain(SECOND).unwrap(), "everything ran");
    assert!(t.stopped().is_none());
    let words = g.words(512);
    let fences = [t.fence_value(F0), t.fence_value(F1)];
    let log = lock(&log).clone();
    (words, log, fences, t.stats())
}

#[test]
#[ignore = "requires Metal"]
fn sync_and_async_agree() {
    let (sync_words, sync_log, sync_fences, sync_stats) = chained_workload(true);
    let (async_words, async_log, async_fences, async_stats) = chained_workload(false);
    assert_eq!(sync_words, async_words, "final memory");
    assert_eq!(sync_log, async_log, "bottom-of-pipe writes, in order");
    assert_eq!(sync_fences, async_fences);
    assert_eq!(sync_fences, [8, 8]);
    // Reference values.
    let mut prev = 3u32;
    for r in 0..8u32 {
        let c = prev + r + 1;
        assert_eq!(sync_words[10 + r as usize], c);
        assert_eq!(sync_words[300 + r as usize], c + 1001);
        assert_eq!(sync_words[400 + r as usize], 10);
        prev = c + 1001;
    }
    assert_eq!(sync_stats.gpu_fence_waits + sync_stats.gpu_tag_waits, 0);
    assert_eq!(sync_stats.max_in_flight, [1, 1]);
    println!(
        "async: committed={:?} gpu_fence_waits={} gpu_tag_waits={} cpu_held={} cpu_signals={} max_in_flight={:?}; sync: cpu_held={}",
        async_stats.committed, async_stats.gpu_fence_waits, async_stats.gpu_tag_waits, async_stats.cpu_held,
        async_stats.cpu_signals, async_stats.max_in_flight, sync_stats.cpu_held
    );
}

fn busy(d: Duration) {
    let start = Instant::now();
    while start.elapsed() < d {
        std::hint::spin_loop();
    }
}

/// The game thread does 1 ms of CPU work, then submits ~1 ms of GPU work,
/// sixteen times. Synchronous: the times add up. Asynchronous: they overlap.
#[test]
#[ignore = "requires Metal"]
fn cpu_and_gpu_overlap() {
    let _pool = Pool::new();
    let g = Gpu::new();
    let spin = g.spin(1_000_000);
    let run = |sync: bool| -> (Duration, Duration, u64) {
        let t = g.timeline(Config { sync, max_in_flight: 2, ..Config::default() });
        t.submit(0, vec![Segment::new("warm", Work::Compute(vec![g.spin_item(&spin, 0, 1)]))]).unwrap();
        assert!(t.drain(SECOND).unwrap(), "everything ran");
        let gpu_before = t.stats().gpu_ns[0];
        let start = Instant::now();
        let mut in_submit = Duration::ZERO;
        for i in 0..16u32 {
            busy(Duration::from_millis(1));
            let s = Instant::now();
            t.submit(0, vec![Segment::new("spin", Work::Compute(vec![g.spin_item(&spin, (i % 4) * 64, i)])).write(None, || {})]).unwrap();
            in_submit += s.elapsed();
        }
        assert!(t.drain(SECOND).unwrap(), "everything ran");
        let wall = start.elapsed();
        assert!(t.stopped().is_none());
        (wall, in_submit, t.stats().gpu_ns[0] - gpu_before)
    };
    let (sync_wall, sync_submit, sync_gpu) = run(true);
    let (async_wall, async_submit, async_gpu) = run(false);
    println!(
        "OVERLAP cpu+gpu: 16 x (1 ms CPU + {} ns GPU); sync wall {:?} (game thread in submit {:?}, gpu {} ns); async wall {:?} (in submit {:?}, gpu {} ns)",
        spin.ns, sync_wall, sync_submit, sync_gpu, async_wall, async_submit, async_gpu
    );
    assert!(async_wall.as_secs_f64() < 0.85 * sync_wall.as_secs_f64(), "no CPU/GPU overlap");
    assert!(async_submit * 4 < sync_submit, "the game thread still waits");
}

/// Both queues run ~2 ms of GPU work at the same time: their command
/// buffers' GPU intervals overlap.
#[test]
#[ignore = "requires Metal"]
fn two_queues_overlap() {
    let _pool = Pool::new();
    let g = Gpu::new();
    let spin = g.spin(1_000_000);
    let run = |sync: bool| -> (u32, Vec<i64>) {
        let t = g.timeline(cfg(sync));
        let mut overlapped = 0;
        let mut overlaps = Vec::new();
        for round in 0..5u32 {
            let work = |dst: u32| Work::Compute(vec![g.spin_item(&spin, dst, round), ComputeItem::Barrier, g.spin_item(&spin, dst, round + 1)]);
            t.submit(0, vec![Segment::new("gfx", work(0))]).unwrap();
            t.submit(1, vec![Segment::new("compute", work(64))]).unwrap();
            assert!(t.drain(SECOND).unwrap(), "everything ran");
            let s = t.stats();
            let last = |q: usize| s.intervals.iter().rev().find(|i| i.queue == q).cloned().unwrap();
            let (a, b) = (last(0), last(1));
            let overlap = a.gpu_end.min(b.gpu_end) as i64 - a.gpu_start.max(b.gpu_start) as i64;
            overlaps.push(overlap);
            if overlap > 0 {
                overlapped += 1;
            }
        }
        assert!(t.stopped().is_none());
        (overlapped, overlaps)
    };
    let (sync_n, sync_ns) = run(true);
    let (async_n, async_ns) = run(false);
    println!("OVERLAP two queues: 2 x {} ns per queue; async overlapped {async_n}/5 by {async_ns:?} ns; sync {sync_n}/5 ({sync_ns:?} ns)", spin.ns);
    assert_eq!(sync_n, 0, "the sync path runs one command buffer at a time");
    assert!(async_n >= 3, "queues did not overlap");
}

/// Sixteen independent one-group dispatches: in a concurrent encoder with
/// no barriers they overlap; with a barrier before each (serial) they do not.
#[test]
#[ignore = "requires Metal"]
fn concurrent_dispatches_overlap() {
    let _pool = Pool::new();
    let g = Gpu::new();
    let spin = g.spin(250_000);
    let run = |policy: BarrierPolicy| -> u64 {
        let t = g.timeline(Config { barriers: policy, ..cfg(false) });
        let items: Vec<_> = (0..16u32).map(|i| g.spin_item(&spin, 1024 + i * 32, i)).collect();
        t.submit(0, vec![Segment::new("warm", Work::Compute(items.clone()))]).unwrap();
        assert!(t.drain(SECOND).unwrap(), "everything ran");
        let before = t.stats().gpu_ns[0];
        t.submit(0, vec![Segment::new("dispatches", Work::Compute(items))]).unwrap();
        assert!(t.drain(SECOND).unwrap(), "everything ran");
        for i in 0..16u32 {
            assert_eq!(g.get((1024 + i * 32 + 1) as usize), lcg(i + 1, MUL, spin.iterations));
        }
        let s = t.stats();
        if policy == BarrierPolicy::Title {
            assert_eq!(s.barriers, 0);
            assert_eq!(s.barriers_elided, 30);
        } else {
            assert_eq!(s.barriers, 30);
        }
        s.gpu_ns[0] - before
    };
    let serial = run(BarrierPolicy::Serial);
    let concurrent = run(BarrierPolicy::Title);
    println!("OVERLAP dispatches: 16 x {} ns; serial {serial} ns, concurrent {concurrent} ns", spin.ns);
    assert!(concurrent * 2 < serial, "independent dispatches did not overlap");
}

/// Read after write, write after read and write after write, each across a
/// title barrier (title mode) or found from declared ranges (tracked mode).
#[test]
#[ignore = "requires Metal"]
fn hazards_across_barriers() {
    let _pool = Pool::new();
    let g = Gpu::new();
    for policy in [BarrierPolicy::Title, BarrierPolicy::Tracked] {
        g.set(0, 10);
        g.set(4, 20);
        g.set(5, 7);
        let t = g.timeline(Config { barriers: policy, ..cfg(false) });
        let title = policy == BarrierPolicy::Title;
        let bar = |v: &mut Vec<ComputeItem>| {
            if title {
                v.push(ComputeItem::Barrier);
            }
        };
        // RAW: b1 = b0 + 1; b2 = b1 + 1.
        let mut raw = vec![g.add(1, 0, 1)];
        bar(&mut raw);
        raw.push(g.add(2, 1, 1));
        // WAR: b6 = b5; b5 = b4 + 100.
        let mut war = vec![g.add(6, 5, 0)];
        bar(&mut war);
        war.push(g.add(5, 4, 100));
        // WAW: b8 = b4 + 1; b8 = b4 + 2.
        let mut waw = vec![g.add(8, 4, 1)];
        bar(&mut waw);
        waw.push(g.add(8, 4, 2));
        for items in [raw, war, waw] {
            t.submit(0, vec![Segment::new("hazard", Work::Compute(items)).write(None, || {})]).unwrap();
        }
        assert!(t.drain(SECOND).unwrap(), "everything ran");
        assert_eq!((g.get(1), g.get(2)), (11, 12), "{policy:?} RAW");
        assert_eq!((g.get(6), g.get(5)), (7, 120), "{policy:?} WAR");
        assert_eq!(g.get(8), 22, "{policy:?} WAW");
        let s = t.stats();
        assert_eq!(s.barriers, 3);
        assert_eq!(s.barriers_from_ranges, if title { 0 } else { 3 });
    }
}

/// Two compute segments packed into one command buffer: two encoders over
/// untracked memory, ordered by the queue's `MTLFence`.
#[test]
#[ignore = "requires Metal"]
fn encoders_ordered_by_fence() {
    let _pool = Pool::new();
    let g = Gpu::new();
    g.set(0, 1);
    let t = g.timeline(cfg(false));
    t.submit(0, vec![
        Segment::new("first", Work::Compute(vec![g.add(20, 0, 5)])),
        Segment::new("second", Work::Compute(vec![g.add(21, 20, 5)])).write(None, || {}),
    ])
    .unwrap();
    assert!(t.drain(SECOND).unwrap(), "everything ran");
    assert_eq!(g.get(21), 11);
    let s = t.stats();
    assert_eq!(s.committed[0], 1);
    assert_eq!(s.encoder_fences, 1);
}

/// Sixty-four command buffers, each with its arguments in a 512-byte ring
/// (two regions) with four allowed in flight: the ring is what holds them
/// back, it wraps 32 times, and no argument is overwritten while read.
#[test]
#[ignore = "requires Metal"]
fn ring_regions_come_back_on_completion() {
    let _pool = Pool::new();
    let g = Gpu::new();
    g.set(0, 1000);
    let t = g.timeline(Config { ring_bytes: 512, max_in_flight: 4, ..cfg(false) });
    let segs = (0..64u32).map(|i| Segment::new("ring", Work::Compute(vec![g.ring_add(600 + i, 0, i)])).write(None, || {})).collect();
    t.submit(0, segs).unwrap();
    assert!(t.drain(SECOND).unwrap(), "everything ran");
    for i in 0..64u32 {
        assert_eq!(g.get((600 + i) as usize), 1000 + i);
    }
    let s = t.stats();
    println!("ring: stalls={} high_water={:?} max_in_flight={:?}", s.ring_stalls, s.ring_high_water, s.max_in_flight);
    assert_eq!(s.ring_high_water[0], 512);
    assert!(t.stopped().is_none());
}

/// The in-flight budget holds the encoder, not the game thread.
#[test]
#[ignore = "requires Metal"]
fn in_flight_budget_blocks_the_encoder_only() {
    let _pool = Pool::new();
    let g = Gpu::new();
    let spin = g.spin(500_000);
    let t = g.timeline(Config { max_in_flight: 2, ..cfg(false) });
    let start = Instant::now();
    for i in 0..10u32 {
        t.submit(0, vec![Segment::new("spin", Work::Compute(vec![g.spin_item(&spin, 0, i)])).write(None, || {})]).unwrap();
    }
    let submitted = start.elapsed();
    assert!(t.drain(SECOND).unwrap(), "everything ran");
    let all = start.elapsed();
    let s = t.stats();
    println!("budget: 10 submits in {submitted:?}, all done in {all:?}; max_in_flight={:?} budget_stalls={}", s.max_in_flight, s.budget_stalls);
    assert_eq!(s.max_in_flight[0], 2);
    assert!(s.budget_stalls > 0);
    assert!(submitted * 2 < all);
}

// Fault injection.

#[test]
#[ignore = "requires Metal"]
fn fault_event_creation() {
    let _pool = Pool::new();
    let g = Gpu::new();
    let mut c = cfg(false);
    c.faults.fail_event = true;
    let t = g.timeline(c);
    t.submit(0, vec![Segment::new("signals", Work::Compute(vec![g.add(1, 0, 1)])).signal(F0, 1)]).unwrap();
    assert!(t.drain(SECOND).is_err());
    let why = t.stopped().unwrap();
    assert!(why.contains("injected event creation failure"), "{why}");
    assert_eq!(t.stats().committed[0], 0);
    assert!(t.submit(0, vec![Segment::new("after", Work::None)]).is_err());
}

#[test]
#[ignore = "requires Metal"]
fn fault_completion_thread_exits() {
    let _pool = Pool::new();
    let g = Gpu::new();
    let mut c = cfg(false);
    c.faults.completion_exits_after = Some((0, 1));
    let t = g.timeline(c);
    let log: Log = Default::default();
    t.submit(0, vec![Segment::new("one", Work::Compute(vec![g.add(1, 0, 1)])).write(None, logger(&log, 0, 1))]).unwrap();
    assert!(t.drain(SECOND).unwrap(), "everything ran");
    t.submit(0, vec![Segment::new("two", Work::Compute(vec![g.add(2, 0, 1)])).write(None, logger(&log, 0, 2))]).unwrap();
    assert!(t.drain(SECOND).is_err());
    let why = t.stopped().unwrap();
    assert!(why.contains("completion thread of queue 0"), "{why}");
    assert_eq!(*lock(&log), [(0, 1)], "the second command buffer's writes never ran");
    assert!(t.submit(0, vec![Segment::new("three", Work::None)]).is_err());
}

#[test]
#[ignore = "requires Metal"]
fn fault_ring_exhausted() {
    let _pool = Pool::new();
    let g = Gpu::new();
    let t = g.timeline(Config { ring_bytes: 256, ..cfg(false) });
    t.submit(0, vec![Segment::new("big", Work::Encode { ring_bytes: 1024, groups: 0, encode: Box::new(|_| Ok(())) })]).unwrap();
    assert!(t.drain(SECOND).is_err());
    let why = t.stopped().unwrap();
    assert!(why.contains("ring"), "{why}");
    assert_eq!(t.stats().committed[0], 0);
}

/// Measured GPU time over the fuse stops everything after that command
/// buffer: the next one, already submitted, is never committed.
#[test]
#[ignore = "requires Metal"]
fn fault_fuse_stops_after_one() {
    let _pool = Pool::new();
    let g = Gpu::new();
    let stops = Arc::new(AtomicU64::new(0));
    let s2 = stops.clone();
    let hooks = Hooks {
        stop: Box::new(move |_| {
            s2.fetch_add(1, Ordering::Relaxed);
        }),
        ..Hooks::default()
    };
    let t = Timeline::new(g.device.clone(), Config { fuse_ns: 1, max_in_flight: 1, ..cfg(false) }, hooks).unwrap();
    let log: Log = Default::default();
    t.submit(0, vec![
        Segment::new("one", Work::Compute(vec![g.add(1, 0, 1)])).write(None, logger(&log, 0, 1)),
        Segment::new("two", Work::Compute(vec![g.add(2, 0, 1)])).write(None, logger(&log, 0, 2)),
    ])
    .unwrap();
    assert!(t.drain(SECOND).is_err());
    let why = t.stopped().unwrap();
    assert!(why.contains("over the fuse"), "{why}");
    let s = t.stats();
    assert_eq!(s.committed[0], 1);
    assert_eq!(s.discarded, 1);
    assert!(lock(&log).is_empty(), "no write after a fused command buffer");
    assert_eq!(stops.load(Ordering::Relaxed), 1, "the outer fuse was told once");
}

/// The outer (process-wide) fuse holds: nothing more is committed.
#[test]
#[ignore = "requires Metal"]
fn fault_outer_stop() {
    let _pool = Pool::new();
    let g = Gpu::new();
    let tripped = Arc::new(AtomicBool::new(false));
    let t2 = tripped.clone();
    let hooks = Hooks { stopped: Box::new(move || t2.load(Ordering::Acquire)), ..Hooks::default() };
    let t = Timeline::new(g.device.clone(), cfg(false), hooks).unwrap();
    t.submit(0, vec![Segment::new("one", Work::Compute(vec![g.add(1, 0, 1)]))]).unwrap();
    assert!(t.drain(SECOND).unwrap(), "everything ran");
    tripped.store(true, Ordering::Release);
    t.submit(0, vec![Segment::new("two", Work::Compute(vec![g.add(2, 0, 1)]))]).unwrap();
    assert!(t.drain(SECOND).is_err());
    assert_eq!(t.stats().committed[0], 1);
    assert!(t.stopped().unwrap().contains("outer"));
}

/// A command buffer that never completes (it waits for an event nothing
/// signals; the GPU does no work meanwhile) is found by the completion
/// thread's timeout.
#[test]
#[ignore = "requires Metal"]
fn fault_hang_detected() {
    let _pool = Pool::new();
    let g = Gpu::new();
    let never = SharedEvent::new(&g.device).unwrap();
    let n2 = never.clone();
    let t = g.timeline(Config { hang_ms: 300, ..cfg(false) });
    let start = Instant::now();
    t.submit(0, vec![Segment::new("hangs", Work::Encode {
        ring_bytes: 0,
        groups: 0,
        encode: Box::new(move |e| {
            e.cb.encode_wait(&n2, 1);
            Ok(())
        }),
    })])
    .unwrap();
    assert!(t.drain(SECOND).is_err());
    let why = t.stopped().unwrap();
    assert!(why.contains("did not complete within 300 ms"), "{why}");
    assert!(start.elapsed() < Duration::from_secs(2));
    // Let the stuck command buffer finish.
    never.signal(1);
}

/// Host segments (the existing synchronous backend path) run on the encoder
/// thread, in queue order, after the queue's command buffers.
#[test]
#[ignore = "requires Metal"]
fn host_segments_run_in_order() {
    let _pool = Pool::new();
    let g: &'static Gpu = Box::leak(Box::new(Gpu::new()));
    g.set(0, 1);
    let t = g.timeline(cfg(false));
    let log: Log = Default::default();
    t.submit(0, vec![
        Segment::new("gpu", Work::Compute(vec![g.add(1, 0, 1)])).write(None, logger(&log, 0, 1)),
        Segment::new("host", Work::Host(Box::new(move || {
            // The command buffer before it has completed.
            let seen = g.get(1);
            g.set(3, seen + 100);
            true
        })))
        .write(None, logger(&log, 0, 2))
        .signal(F0, 1),
    ])
    .unwrap();
    assert!(t.drain(SECOND).unwrap(), "everything ran");
    assert_eq!(g.get(3), 102);
    assert_eq!(*lock(&log), [(0, 1), (0, 2)]);
    assert_eq!(t.fence_value(F0), 1);
    assert_eq!(t.stats().host_runs, 1);
}

/// A host segment whose wait could be a GPU wait (its producer is in
/// flight) waits on the CPU instead, and sees the producer's result.
#[test]
#[ignore = "requires Metal"]
fn host_segment_waits_on_the_cpu() {
    let _pool = Pool::new();
    let g: &'static Gpu = Box::leak(Box::new(Gpu::new()));
    g.set(0, 1);
    let spin = g.spin(1_000_000);
    let t = g.timeline(cfg(false));
    let memory = Arc::new(AtomicU64::new(0));
    let mw = memory.clone();
    t.submit(1, vec![Segment::new("producer", Work::Compute(vec![g.spin_item(&spin, 64, 3), ComputeItem::Barrier, g.spin_item(&spin, 64, 4), ComputeItem::Barrier, g.add(1, 0, 1)]))
        .write(Some((0x7a9, 5)), move || mw.store(5, Ordering::Release))]).unwrap();
    let m = memory.clone();
    t.submit(0, vec![Segment::new("host", Work::Host(Box::new(move || {
        g.set(3, g.get(1) + 100);
        true
    })))
    .wait_tag(0x7a9, 5, true, Arc::new(move || m.load(Ordering::Acquire) >= 5))]).unwrap();
    assert!(t.drain(SECOND).unwrap(), "everything ran");
    assert_eq!(g.get(3), 102);
    let s = t.stats();
    // Never a GPU wait for host work; held on the CPU if the producer was
    // still running when it was looked at (timing), and correct either way.
    assert_eq!(s.gpu_tag_waits, 0, "{s:?}");
    assert!(s.cpu_held <= 1, "{s:?}");
}

/// A producer that never completes (it waits on an event outside the
/// timeline) and whose fence another queue's committed command buffer waits
/// for on the GPU: when the hang is found, the timeline releases its events,
/// the waiting command buffer finishes, and every completion thread gets
/// past its command buffers within the timeout. Nothing waits forever.
fn stuck_producer(stop_from_outside: bool) {
    let _pool = Pool::new();
    let g = Gpu::new();
    g.set(0, 1);
    let never = SharedEvent::new(&g.device).unwrap();
    let n2 = never.clone();
    let hang_ms = if stop_from_outside { 3000 } else { 300 };
    let t = g.timeline(Config { hang_ms, ..cfg(false) });
    t.submit(1, vec![
        Segment::new("stuck producer", Work::Encode {
            ring_bytes: 0,
            groups: 0,
            encode: Box::new(move |e| {
                e.cb.encode_wait(&n2, 1);
                Ok(())
            }),
        })
        .signal(F1, 1),
        Segment::new("behind it", Work::Compute(vec![g.add(5, 0, 1)])),
    ])
    .unwrap();
    t.submit(0, vec![Segment::new("consumer", Work::Compute(vec![g.add(2, 0, 10)])).wait_fence(F1, 1)]).unwrap();
    // The consumer is committed with a GPU wait on the stuck producer's signal.
    let start = Instant::now();
    while t.stats().gpu_fence_waits == 0 {
        assert!(start.elapsed() < Duration::from_secs(2), "consumer never committed: {:?}", t.stats());
        std::thread::sleep(Duration::from_millis(5));
    }
    let start = Instant::now();
    if stop_from_outside {
        t.stop("test");
    }
    // The consumer's command buffer finishes (released, not timed out).
    while !t.stats().intervals.iter().any(|i| i.queue == 0) {
        assert!(start.elapsed() < Duration::from_secs(2), "consumer still waiting: {:?}", t.stats());
        std::thread::sleep(Duration::from_millis(5));
    }
    let released_after = start.elapsed();
    // Every completion thread got past its command buffers.
    while t.in_flight() != [0, 0] {
        assert!(start.elapsed() < Duration::from_millis(hang_ms * 3 + 1000), "completion threads stuck: {:?}", t.in_flight());
        std::thread::sleep(Duration::from_millis(5));
    }
    let s = t.stats();
    println!("stuck producer (outer stop {stop_from_outside}): consumer done {released_after:?} after; released={} stopped={:?}", s.released, t.stopped());
    assert_eq!(s.released, 1);
    assert!(t.stopped().is_some());
    if stop_from_outside {
        assert!(released_after < Duration::from_millis(1000), "released by the stop, not by a timeout");
    }
    assert!(t.drain(SECOND).is_err());
    // Dropping the timeline is bounded even with the producer still stuck.
    let start = Instant::now();
    drop(t);
    assert!(start.elapsed() < Duration::from_secs(2), "drop took {:?}", start.elapsed());
    never.signal(1);
}

#[test]
#[ignore = "requires Metal"]
fn fault_hang_releases_waiters() {
    stuck_producer(false);
}

#[test]
#[ignore = "requires Metal"]
fn fault_outer_stop_releases_waiters() {
    stuck_producer(true);
}

/// Thirty-two command buffers on one queue, each reading what the one
/// before it wrote (untracked memory): ordered by the queue's fence (no
/// chain event waits) and, for comparison, by the chain event.
#[test]
#[ignore = "requires Metal"]
fn fence_orders_command_buffers() {
    let _pool = Pool::new();
    let g = Gpu::new();
    for fence_chain in [true, false] {
        g.set(100, 5);
        let t = g.timeline(Config { fence_chain, max_in_flight: 8, max_segments_per_cb: 1, ..cfg(false) });
        let segs = (0..32u32).map(|i| Segment::new("link", Work::Compute(vec![g.add(101 + i, 100 + i, 1)]))).collect();
        t.submit(0, segs).unwrap();
        assert!(t.drain(SECOND).unwrap(), "everything ran");
        for i in 0..=32u32 {
            assert_eq!(g.get((100 + i) as usize), 5 + i, "fence_chain {fence_chain}: link {i}");
        }
        let s = t.stats();
        println!("fence_chain {fence_chain}: committed {:?} chain_waits_skipped {} max_in_flight {:?}", s.committed, s.chain_waits_skipped, s.max_in_flight);
        assert_eq!(s.committed[0], 32);
        assert_eq!(s.chain_waits_skipped, if fence_chain { 31 } else { 0 });
    }
}

/// `gpu_idle` returns once every command buffer committed before it has
/// finished on the GPU, and nothing is committed while its guard is held.
#[test]
#[ignore = "requires Metal"]
fn gpu_idle_waits_and_holds_commits() {
    let _pool = Pool::new();
    let g = Gpu::new();
    g.set(0, 5);
    let spin = g.spin(1_000_000);
    let t = g.timeline(cfg(false));
    t.submit(1, vec![Segment::new("producer", Work::Compute(vec![g.spin_item(&spin, 64, 3), ComputeItem::Barrier, g.add(1, 0, 1)]))]).unwrap();
    // Until the producer is committed (it may already be done).
    let start = Instant::now();
    while t.stats().committed[1] == 0 {
        assert!(start.elapsed() < SECOND, "producer never committed");
        std::thread::sleep(Duration::from_millis(1));
    }
    let idle = t.gpu_idle(SECOND);
    assert!(idle.idle());
    assert_eq!(g.get(1), 6, "the producer finished before gpu_idle returned");
    // Held: submitted work is not committed.
    t.submit(0, vec![Segment::new("held back", Work::Compute(vec![g.add(2, 1, 10)]))]).unwrap();
    std::thread::sleep(Duration::from_millis(30));
    assert_eq!(t.stats().committed[0], 0, "committed while the GPU was to stay idle");
    assert_eq!(g.get(2), 0);
    drop(idle);
    assert!(t.drain(SECOND).unwrap(), "everything ran");
    assert_eq!(g.get(2), 16);
    // Nothing in flight: idle at once.
    assert!(t.gpu_idle(SECOND).idle());
    assert!(t.stopped().is_none());
}

/// `gpu_idle` with a command buffer that never finishes: bounded, says the
/// GPU was not idle, and stops the timeline (its waiters are released).
#[test]
#[ignore = "requires Metal"]
fn fault_gpu_idle_bounded() {
    let _pool = Pool::new();
    let g = Gpu::new();
    let never = SharedEvent::new(&g.device).unwrap();
    let n2 = never.clone();
    let t = g.timeline(Config { hang_ms: 3000, ..cfg(false) });
    t.submit(1, vec![Segment::new("stuck", Work::Encode {
        ring_bytes: 0,
        groups: 0,
        encode: Box::new(move |e| {
            e.cb.encode_wait(&n2, 1);
            Ok(())
        }),
    })])
    .unwrap();
    let start = Instant::now();
    while t.stats().committed[1] == 0 {
        assert!(start.elapsed() < SECOND, "never committed");
        std::thread::sleep(Duration::from_millis(1));
    }
    let start = Instant::now();
    let idle = t.gpu_idle(Duration::from_millis(100));
    let took = start.elapsed();
    println!("gpu_idle on a stuck GPU: idle={} after {took:?} stopped={:?}", idle.idle(), t.stopped());
    assert!(!idle.idle());
    assert!(took < Duration::from_millis(1000), "gpu_idle took {took:?}");
    assert!(t.stopped().is_some());
    drop(idle);
    drop(t);
    never.signal(1);
}

/// Run V0 R3: a drain before a present was stopped after 5 s while the
/// encoder thread encoded one 4,772-op batch (2.8 s) with nothing in
/// flight. Encoding and host work longer than the drain's GPU timeout are
/// not a hang: the drain waits for them and everything runs.
#[test]
#[ignore = "requires Metal"]
fn drain_waits_for_a_slow_encoder() {
    let _pool = Pool::new();
    let g: &'static Gpu = Box::leak(Box::new(Gpu::new()));
    g.set(0, 1);
    let t = g.timeline(cfg(false));
    t.submit(0, vec![
        Segment::new("slow host work", Work::Host(Box::new(|| {
            std::thread::sleep(Duration::from_millis(600));
            true
        }))),
        Segment::new("slow encoding", Work::Encode {
            ring_bytes: 0,
            groups: 1,
            encode: Box::new(move |e| {
                std::thread::sleep(Duration::from_millis(600));
                let a = args(2, 0, 5);
                e.cb.compute(&[
                    crate::Compute::Pipeline { pipeline: &g.add, threads_per_group: [1; 3] },
                    crate::Compute::Buffer { buffer: &g.buf, offset: 0, index: 0 },
                    crate::Compute::Bytes { bytes: &a, index: 1 },
                    crate::Compute::Dispatch { groups: [1; 3] },
                ])
            }),
        }),
        Segment::new("after", Work::Compute(vec![g.add(1, 0, 1)])),
    ])
    .unwrap();
    let start = Instant::now();
    let drained = t.drain(Duration::from_millis(200));
    let took = start.elapsed();
    println!("drain behind 1.2 s of encoder work: {drained:?} after {took:?}");
    assert!(drained.unwrap(), "everything ran");
    assert!(took >= Duration::from_millis(1000), "the drain waited for the encoder ({took:?})");
    assert!(t.stopped().is_none());
    assert_eq!((g.get(1), g.get(2)), (2, 6));
}

/// A committed command buffer that never completes is still found by the
/// drain (here before the completion thread's own, longer, hang timeout).
#[test]
#[ignore = "requires Metal"]
fn fault_drain_finds_a_gpu_stall() {
    let _pool = Pool::new();
    let g = Gpu::new();
    let never = SharedEvent::new(&g.device).unwrap();
    let n2 = never.clone();
    let t = g.timeline(Config { hang_ms: 10_000, ..cfg(false) });
    t.submit(0, vec![Segment::new("stuck", Work::Encode {
        ring_bytes: 0,
        groups: 0,
        encode: Box::new(move |e| {
            e.cb.encode_wait(&n2, 1);
            Ok(())
        }),
    })])
    .unwrap();
    let start = Instant::now();
    let drained = t.drain(Duration::from_millis(300));
    let took = start.elapsed();
    println!("drain on a stuck GPU: {drained:?} after {took:?}");
    let e = drained.unwrap_err().to_string();
    assert!(e.contains("no GPU progress"), "{e}");
    assert!(took >= Duration::from_millis(300) && took < Duration::from_secs(2), "{took:?}");
    t.stop("test");
    never.signal(1);
    drop(t);
}

/// The backstop: a timeline that makes no progress at all (here host work
/// that does not return) ends a drain after `encoder_stall_ms`.
#[test]
#[ignore = "requires Metal"]
fn fault_drain_backstop() {
    let _pool = Pool::new();
    let g = Gpu::new();
    let t = g.timeline(Config { encoder_stall_ms: 300, ..cfg(false) });
    t.submit(0, vec![Segment::new("stuck host work", Work::Host(Box::new(|| {
        std::thread::sleep(Duration::from_millis(1200));
        true
    })))])
    .unwrap();
    let start = Instant::now();
    let drained = t.drain(Duration::from_millis(100));
    let took = start.elapsed();
    println!("drain behind host work that does not return: {drained:?} after {took:?}");
    let e = drained.unwrap_err().to_string();
    assert!(e.contains("made no progress"), "{e}");
    assert!(took >= Duration::from_millis(300) && took < Duration::from_millis(1100), "{took:?}");
    drop(t);
}
