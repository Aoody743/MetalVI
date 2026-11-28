//! The same compute ops through the old path (`MetalBackend::run`) and as
//! execution-timeline segments: same memory afterwards. One-thread kernels;
//! milliseconds of GPU work at most.

use super::*;
use crate::backend::metal::{MetalBackend, MetalCompute, MetalDispatchIndirect, diag};
use std::time::Duration;
use victory_metal::exec::{Config, Hooks, Timeline};

/// The ABI shape of MSC's compute shaders: the top-level argument buffer at
/// bind point 2, a one-view table in it naming the memory by address.
const MSL: &str = "#include <metal_stdlib>
using namespace metal;
struct Entry { device uint *va; ulong tex; ulong meta; };
struct Top { device const Entry *table; uint dst; uint src; uint add; uint pad; };
kernel void k(constant Top &t [[buffer(2)]]) { device uint *b = t.table->va; b[t.dst] = b[t.src] + t.add; }
";

const WORDS: usize = 64;

struct Setup {
    backend: MetalBackend,
    pso: Arc<MetalCompute>,
    mem: victory_metal::Buffer,
    cpu: *mut u32,
    records: victory_metal::Buffer,
}

impl Setup {
    fn new() -> Self {
        let backend = MetalBackend::new().unwrap();
        let pso = backend.device.compute_pipeline_msl(MSL, "k").unwrap();
        let pso = Arc::new(MetalCompute {
            // Warm (as after a first run): 100 µs per dispatch, so the
            // estimate window does not cut every dispatch into its own buffer.
            cost_ns: std::sync::atomic::AtomicU64::new(10_000),
            name: "exec compute test".into(),
            pso,
            threadgroup: [1, 1, 1],
            top_level: victory_metal::ir::TopLevel::new(Vec::new(), 24),
        });
        let (mem, cpu) = backend.device.shared_buffer_untracked_mapped(&[0u8; WORDS * 4]).unwrap();
        let records = backend.device.shared_buffer(&[2u32, 1, 1].map(u32::to_le_bytes).concat()).unwrap();
        backend.chunks().residency().add(&[mem.handle(), records.handle()]);
        backend.chunks().residency().commit();
        Self { backend, pso, mem, cpu: cpu.cast(), records }
    }

    fn reset(&self) {
        for i in 0..WORDS {
            // SAFETY: inside the buffer; no GPU work in flight.
            unsafe { self.cpu.add(i).write_volatile(if i == 0 { 7 } else { 0 }) };
        }
    }

    fn words(&self) -> Vec<u32> {
        // SAFETY: read after the GPU work completed.
        (0..WORDS).map(|i| unsafe { self.cpu.add(i).read_volatile() }).collect()
    }

    fn top(&self, dst: u32, src: u32, add: u32) -> (Vec<u8>, Vec<(u32, DescriptorEntry)>) {
        let mut top = vec![0u8; 8];
        for v in [dst, src, add, 0] {
            top.extend_from_slice(&v.to_le_bytes());
        }
        (top, vec![(0, DescriptorEntry::buffer(self.mem.gpu_address, (WORDS * 4) as u64))])
    }

    fn direct(&self, dst: u32, src: u32, add: u32, after_barrier: bool) -> Arc<MetalOp> {
        let (top, tables) = self.top(dst, src, add);
        Arc::new(MetalOp::Dispatch(MetalCmd {
            pso: self.pso.clone(),
            heap: None,
            sampler_heap: None,
            top,
            groups: [1, 1, 1],
            tables,
            loop_counts: None,
            after_barrier,
        }))
    }

    fn indirect(&self, dst: u32, src: u32, add: u32) -> Arc<MetalOp> {
        let (top, tables) = self.top(dst, src, add);
        Arc::new(MetalOp::DispatchIndirect(MetalDispatchIndirect {
            pso: self.pso.clone(),
            heap: None,
            sampler_heap: None,
            top,
            tables,
            records: self.records.clone(),
            records_offset: 0,
            stride: 12,
            max: 1,
            count: None,
            records_va: 0,
            count_va: None,
        }))
    }

    /// b1 = b0+1 | b2 = b1+1 (after a barrier) | b10 = b0+5, b11 = b0+6
    /// (no barrier: may overlap) | indirect b3 = b2+1 | b4 = b3+1 | b12 = b4+1.
    fn ops(&self) -> Vec<Arc<MetalOp>> {
        vec![
            self.direct(1, 0, 1, true),
            self.direct(2, 1, 1, true),
            self.direct(10, 0, 5, false),
            self.direct(11, 0, 6, false),
            self.indirect(3, 2, 1),
            self.direct(4, 3, 1, true),
            self.direct(12, 4, 1, true),
        ]
    }
}

#[test]
#[ignore = "requires Metal"]
fn timeline_compute_matches_old_path() {
    let _pool = victory_metal::Pool::new();
    let s = Setup::new();
    let ops = s.ops();
    assert!(compute_only(&ops));

    s.reset();
    assert!(s.backend.run(&ops), "old path failed");
    let old = s.words();
    let mut want = vec![0u32; WORDS];
    want[0] = 7;
    want[1] = 8;
    want[2] = 9;
    want[10] = 12;
    want[11] = 13;
    want[3] = 10;
    want[4] = 11;
    want[12] = 12;
    assert_eq!(old, want, "old path");

    s.reset();
    let t = Timeline::new(s.backend.device.clone(), Config { max_segments_per_cb: 1, ..Config::default() }, Hooks::default()).unwrap();
    for q in 0..2 {
        t.queue(q).add_residency_set(s.backend.chunks().residency());
    }
    let rings: Vec<_> = t.ring_buffers().iter().map(|b| b.handle()).collect();
    s.backend.chunks().residency().add(&rings);
    s.backend.chunks().residency().commit();
    diag::LAST_GPU_NS.store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
    let before = ENCODED.each_ref().map(|c| c.load(Ordering::Relaxed));
    let segs = segments(&ops);
    assert_eq!(segs.len(), 3, "4 dispatches | indirect | 2 dispatches");
    t.submit(0, segs).unwrap();
    assert!(t.drain(Duration::from_secs(5)).unwrap(), "everything ran");
    assert!(t.stopped().is_none());
    let new = s.words();
    assert_eq!(new, old, "timeline path");
    let after = ENCODED.each_ref().map(|c| c.load(Ordering::Relaxed));
    assert_eq!([after[0] - before[0], after[1] - before[1], after[2] - before[2]], [6, 1, 3]);
    assert_eq!(
        diag::LAST_GPU_NS.load(std::sync::atomic::Ordering::Relaxed),
        u64::MAX,
        "the old completion path (submit_checked) was not used"
    );
    let st = t.stats();
    println!(
        "timeline compute: committed={:?} barriers={} elided={} ring_high_water={:?}",
        st.committed, st.barriers, st.barriers_elided, st.ring_high_water
    );
    assert_eq!(st.committed[0], 3);
    // The title barrier before b2 = b1+1 and before b12 = b4+1; b10 and b11
    // run without one (the old path had a barrier between every dispatch).
    assert_eq!((st.barriers, st.barriers_elided), (2, 2));
    assert!(st.ring_high_water[0] > 0, "tables went through the timeline ring");
}

/// Queues on two timelines with the title's compute ops: the compute queue's
/// ops and the graphics queue's ops are encoded and committed without
/// waiting for each other, and their results are right.
#[test]
#[ignore = "requires Metal"]
fn timeline_compute_on_two_queues() {
    let _pool = victory_metal::Pool::new();
    let s = Setup::new();
    s.reset();
    let t = Timeline::new(s.backend.device.clone(), Config { max_segments_per_cb: 1, ..Config::default() }, Hooks::default()).unwrap();
    for q in 0..2 {
        t.queue(q).add_residency_set(s.backend.chunks().residency());
    }
    let rings: Vec<_> = t.ring_buffers().iter().map(|b| b.handle()).collect();
    s.backend.chunks().residency().add(&rings);
    s.backend.chunks().residency().commit();
    let gfx: Vec<_> = (0..8).map(|i| s.direct(20 + i, 0, i, true)).collect();
    let compute: Vec<_> = (0..8).map(|i| s.direct(40 + i, 0, 100 + i, true)).collect();
    t.submit(0, segments(&gfx)).unwrap();
    t.submit(1, segments(&compute)).unwrap();
    assert!(t.drain(Duration::from_secs(5)).unwrap(), "everything ran");
    let w = s.words();
    for i in 0..8u32 {
        assert_eq!(w[(20 + i) as usize], 7 + i);
        assert_eq!(w[(40 + i) as usize], 107 + i);
    }
    let st = t.stats();
    assert_eq!(st.committed, [2, 2]);
    assert!(t.stopped().is_none());
}

fn spin_pso(s: &Setup, iterations: u32) -> Arc<MetalCompute> {
    let src = format!(
        "#include <metal_stdlib>
using namespace metal;
struct Entry {{ device uint *va; ulong tex; ulong meta; }};
struct Top {{ device const Entry *table; uint dst; uint src; uint add; uint pad; }};
kernel void k(constant Top &t [[buffer(2)]], uint g [[thread_position_in_grid]]) {{
  device uint *b = t.table->va; uint v = t.src + g;
  for (uint i = 0; i < {iterations}u; ++i) {{ v = v * 1664525u + t.add; }}
  if (g == 0) {{ b[t.dst] = v; }}
}}"
    );
    Arc::new(MetalCompute {
        cost_ns: std::sync::atomic::AtomicU64::new(10_000),
        name: format!("spin {iterations}"),
        pso: s.backend.device.compute_pipeline_msl(&src, "k").unwrap(),
        threadgroup: [32, 1, 1],
        top_level: victory_metal::ir::TopLevel::new(Vec::new(), 24),
    })
}

fn spin_op(s: &Setup, pso: &Arc<MetalCompute>, dst: u32) -> Arc<MetalOp> {
    let (top, tables) = s.top(dst, dst, 1013904223);
    Arc::new(MetalOp::Dispatch(MetalCmd {
        pso: pso.clone(),
        heap: None,
        sampler_heap: None,
        top,
        groups: [1, 1, 1],
        tables,
        loop_counts: None,
        after_barrier: false,
    }))
}

/// Sixteen independent dispatches per queue, two queues: the old path runs
/// them one at a time (a barrier between each, a wait per command buffer, one
/// queue); the timeline runs those without a title barrier concurrently and
/// the two queues side by side.
#[test]
#[ignore = "requires Metal"]
fn timeline_compute_overlaps() {
    let _pool = victory_metal::Pool::new();
    let s = Setup::new();
    s.reset();
    // Calibrate: one dispatch must stay well under a millisecond or two.
    let pso = spin_pso(&s, 2500);
    for _ in 0..2 {
        assert!(s.backend.run(&[spin_op(&s, &pso, 5)]));
    }
    let one = diag::LAST_GPU_NS.load(std::sync::atomic::Ordering::Relaxed);
    assert!(one < 3_000_000, "one spin dispatch took {one} ns");
    let ops = |base: u32| -> Vec<Arc<MetalOp>> { (0..16).map(|i| spin_op(&s, &pso, base + i)).collect() };
    let (a, b) = (ops(20), ops(40));
    s.reset();
    let start = std::time::Instant::now();
    assert!(s.backend.run(&a));
    assert!(s.backend.run(&b));
    let old = start.elapsed();
    let old_words = s.words();

    s.reset();
    let t = Timeline::new(s.backend.device.clone(), Config { max_segments_per_cb: 1, ..Config::default() }, Hooks::default()).unwrap();
    for q in 0..2 {
        t.queue(q).add_residency_set(s.backend.chunks().residency());
    }
    let rings: Vec<_> = t.ring_buffers().iter().map(|r| r.handle()).collect();
    s.backend.chunks().residency().add(&rings);
    s.backend.chunks().residency().commit();
    // Warm the timeline's queues.
    t.submit(0, segments(&[spin_op(&s, &pso, 5)])).unwrap();
    t.submit(1, segments(&[spin_op(&s, &pso, 6)])).unwrap();
    assert!(t.drain(Duration::from_secs(5)).unwrap(), "everything ran");
    s.reset();
    let gpu_before = t.stats().gpu_ns.clone();
    let start = std::time::Instant::now();
    t.submit(0, segments(&a)).unwrap();
    t.submit(1, segments(&b)).unwrap();
    let submitted = start.elapsed();
    assert!(t.drain(Duration::from_secs(5)).unwrap(), "everything ran");
    let new = start.elapsed();
    assert_eq!(s.words(), old_words, "same results");
    let st = t.stats();
    let intervals: Vec<_> = st.intervals.iter().rev().take(8).cloned().collect();
    let overlap = |q: usize| intervals.iter().filter(|i| i.queue == q).map(|i| (i.gpu_start, i.gpu_end)).fold((u64::MAX, 0), |(s0, e0), (s1, e1)| (s0.min(s1), e0.max(e1)));
    let (g, c) = (overlap(0), overlap(1));
    let both = g.1.min(c.1) as i64 - g.0.max(c.0) as i64;
    println!(
        "OVERLAP title compute: 2 queues x 16 dispatches ({one} ns each alone): old path wall {old:?}; timeline wall {new:?} (submits {submitted:?}), gpu {:?} ns, queues overlapped {both} ns, barriers {} elided {}",
        [st.gpu_ns[0] - gpu_before[0], st.gpu_ns[1] - gpu_before[1]], st.barriers, st.barriers_elided
    );
    assert!(both > 0, "the two queues did not overlap");
    // GPU-side (wall clock depends on what else the machine runs): each
    // queue's sixteen dispatches took well under the sixteen run serially.
    for q in 0..2 {
        assert!((st.gpu_ns[q] - gpu_before[q]) * 2 < 16 * one, "queue {q}: dispatches did not overlap");
    }
}

/// Host cost per operation: 256 one-thread dispatches (a title barrier every
/// fourth), through the old path and as timeline segments. Prints the game
/// thread's time per op, the encoder thread's, and the wall time.
#[test]
#[ignore = "requires Metal"]
fn host_cost_per_op() {
    let _pool = victory_metal::Pool::new();
    let s = Setup::new();
    let ops: Vec<_> = (0..256u32).map(|i| s.direct(20 + (i % 40), 0, i, i % 4 == 0)).collect();
    // Warm both paths.
    assert!(s.backend.run(&ops[..8]));
    let n = ops.len() as f64;
    let start = std::time::Instant::now();
    assert!(s.backend.run(&ops));
    let old = start.elapsed().as_secs_f64() * 1e6 / n;
    for max_in_flight in [2, 8] {
    let t = Timeline::new(s.backend.device.clone(), Config { max_segments_per_cb: 1, max_in_flight, ..Config::default() }, Hooks::default()).unwrap();
    for q in 0..2 {
        t.queue(q).add_residency_set(s.backend.chunks().residency());
    }
    let rings: Vec<_> = t.ring_buffers().iter().map(|r| r.handle()).collect();
    s.backend.chunks().residency().add(&rings);
    s.backend.chunks().residency().commit();
    t.submit(0, segments(&ops[..8])).unwrap();
    assert!(t.drain(Duration::from_secs(5)).unwrap());

    let encode_before = t.stats().encode_ns;
    let gpu_before = t.stats().gpu_ns[0];
    let start = std::time::Instant::now();
    let segs = segments(&ops);
    let built = start.elapsed().as_secs_f64() * 1e6 / n;
    t.submit(0, segs).unwrap();
    let game = start.elapsed().as_secs_f64() * 1e6 / n;
    assert!(t.drain(Duration::from_secs(5)).unwrap());
    let wall = start.elapsed().as_secs_f64() * 1e6 / n;
    let st = t.stats();
    let encoder = (st.encode_ns - encode_before) as f64 / 1e3 / n;
    let gpu = (st.gpu_ns[0] - gpu_before) as f64 / 1e3 / n;
    println!(
        "HOST COST per op ({} build): old run() {old:.1} µs; timeline (in flight {max_in_flight}): game thread {game:.1} µs (segments {built:.1}), encoder {encoder:.1} µs, GPU busy {gpu:.1} µs, wall {wall:.1} µs; {} command buffers",
        if cfg!(debug_assertions) { "debug" } else { "release" },
        st.committed[0]
    );
    assert!(game < old / 4.0, "the game thread still pays the old cost");
    }
}

/// An indirect dispatch of 150 records (more than one bounded translation
/// takes) with a GPU count of 130: split into parts of at most 64 records,
/// every one of the 130 runs exactly once, none past the count.
#[test]
#[ignore = "requires Metal"]
fn indirect_dispatch_over_the_record_limit_is_split() {
    let _pool = victory_metal::Pool::new();
    let s = Setup::new();
    s.reset();
    let msl = "#include <metal_stdlib>
using namespace metal;
struct Entry { device atomic_uint *va; ulong tex; ulong meta; };
struct Top { device const Entry *table; uint dst; uint src; uint add; uint pad; };
kernel void k(constant Top &t [[buffer(2)]], uint3 g [[threadgroup_position_in_grid]]) {
  atomic_fetch_add_explicit(&t.table->va[t.dst], 1u, memory_order_relaxed);
  atomic_fetch_add_explicit(&t.table->va[t.dst + 1], g.x, memory_order_relaxed);
}";
    let pso = Arc::new(MetalCompute {
        cost_ns: std::sync::atomic::AtomicU64::new(10_000),
        name: "counting".into(),
        pso: s.backend.device.compute_pipeline_msl(msl, "k").unwrap(),
        threadgroup: [1, 1, 1],
        top_level: victory_metal::ir::TopLevel::new(Vec::new(), 24),
    });
    // Record r dispatches (1 + r % 3) groups.
    let records: Vec<u8> = (0..150u32).flat_map(|r| [1 + r % 3, 1, 1]).flat_map(u32::to_le_bytes).collect();
    let records = s.backend.device.shared_buffer(&records).unwrap();
    let count = s.backend.device.shared_buffer(&130u32.to_le_bytes()).unwrap();
    s.backend.chunks().residency().add(&[records.handle(), count.handle()]);
    s.backend.chunks().residency().commit();
    let (top, tables) = s.top(50, 0, 0);
    let op = Arc::new(MetalOp::DispatchIndirect(MetalDispatchIndirect {
        pso,
        heap: None,
        sampler_heap: None,
        top,
        tables,
        records,
        records_offset: 0,
        stride: 12,
        max: 150,
        count: Some((count, 0)),
        records_va: 0,
        count_va: None,
    }));
    let t = Timeline::new(s.backend.device.clone(), Config { max_segments_per_cb: 1, ..Config::default() }, Hooks::default()).unwrap();
    for q in 0..2 {
        t.queue(q).add_residency_set(s.backend.chunks().residency());
    }
    let rings: Vec<_> = t.ring_buffers().iter().map(|b| b.handle()).collect();
    s.backend.chunks().residency().add(&rings);
    s.backend.chunks().residency().commit();
    let segs = segments(&[op]);
    assert_eq!(segs.len(), 3, "150 records: 64 + 64 + 22");
    t.submit(0, segs).unwrap();
    assert!(t.drain(Duration::from_secs(5)).unwrap(), "everything ran");
    let w = s.words();
    let groups: u32 = (0..130u32).map(|r| 1 + r % 3).sum();
    assert_eq!(w[50], groups, "one add per group of the 130 live records");
    // Group x indices: record r adds 0 + 1 + ... + (gx - 1).
    let ids: u32 = (0..130u32).map(|r| (0..1 + r % 3).sum::<u32>()).sum();
    assert_eq!(w[51], ids);
    assert_eq!(t.stats().committed[0], 3);
}

mod draws {
    use super::super::*;
    use crate::backend::metal::{MetalBackend, diag};
    use crate::backend::metal_draw::{DrawKind, MetalClear, MetalDraw, MetalGraphics, MetalTexCopy, Targets};
    use crate::backend::metal_texture::MetalTexture;
    use std::time::Duration;
    use victory_metal::exec::{Config, Hooks, Timeline};
    use victory_metal::render::{DepthStencilDesc, RenderPipelineDesc};
    use victory_metal::sys::{self, WMTPixelFormat as P};
    use victory_metal::texture::TextureDescriptor;

    /// The depth-replay sequence of `metal_depth_replay_tests` (clears,
    /// copies, two depth-companion draws, a rectangle clear) on `backend`:
    /// the ops and the four textures read back afterwards.
    fn sequence(backend: &MetalBackend) -> (Vec<Arc<MetalOp>>, [Arc<MetalTexture>; 4]) {
        let make_texture = |key, format| {
            let texture = backend
                .device
                .new_texture(&TextureDescriptor {
                    pixel_format: P::WMTPixelFormatR32Float,
                    width: 4,
                    height: 4,
                    usage: sys::WMTTextureUsage(1 | 2 | 4 | 0x10),
                    options: sys::WMTResourceOptions(0x20),
                    ..Default::default()
                })
                .unwrap();
            // SAFETY: ResourceDesc is an integer-only C record.
            let mut desc: crate::resource::ResourceDesc = unsafe { std::mem::zeroed() };
            desc.width = 4;
            desc.height = 4;
            desc.dimension = 3;
            desc.format = format;
            desc.sample_count = 1;
            let t = Arc::new(MetalTexture { companion: Default::default(), texture, pixel: P::WMTPixelFormatR32Float, desc, mips: 1, slices: 1 });
            backend.textures.lock().unwrap().insert(key, Some(t.clone()));
            t
        };
        let original = make_texture(1, 39);
        let input = make_texture(2, 41);
        let output = make_texture(3, 41);
        let cleared = make_texture(4, 41);
        let input_attachment = backend.attachment(2 as crate::object::This, 41, 0, 0, false).unwrap();
        let attachment = backend.attachment(1 as crate::object::This, 40, 0, 0, true).unwrap();
        let lib = backend
            .device
            .library_msl("#include <metal_stdlib>\nusing namespace metal;\nvertex float4 vs(uint id [[vertex_id]]) { return float4(id == 1 ? 3 : -1, id == 2 ? 3 : -1, 0.5, 1); }\n")
            .unwrap();
        let desc = RenderPipelineDesc { depth_format: P::WMTPixelFormatDepth32Float.0, sample_count: 1, ..Default::default() };
        let pso = backend.device.render_pipeline_desc((&lib, "vs"), None, &desc).unwrap();
        let dss = backend.device.depth_stencil_state(&DepthStencilDesc { depth_write: true, depth_compare: 1, ..Default::default() }).unwrap();
        let pso = Arc::new(MetalGraphics {
            cost_ns: 1.into(),
            name: "stage 4 synthetic depth draw".into(),
            pso,
            dss,
            depth_write: true,
            raster: [0; 4],
            bias: [0.0; 3],
            stages: Vec::new(),
            vertex_slots: 0,
            mesh: false,
            mesh_threads: [0; 3],
            geometry: None,
            colors: [0; 8],
            depth: desc.depth_format,
            stencil: 0,
            samples: 1,
        });
        let mut targets = Targets { depth: Some(attachment.clone()), ..Default::default() };
        targets.finish();
        let draw = Arc::new(MetalOp::Draw(MetalDraw {
            trace_inputs: Vec::new(),
            pso,
            targets: Arc::new(targets),
            stages: Vec::new(),
            vertex: None,
            viewport: sys::WMTViewport { originX: 0.0, originY: 0.0, width: 4.0, height: 4.0, znear: 0.0, zfar: 1.0 },
            scissor: sys::WMTScissorRect { x: 0, y: 0, width: 4, height: 4 },
            blend: [0.0; 4],
            stencil_ref: 0,
            primitive: 3,
            kind: DrawKind::Vertices { count: 3, instances: 1, first: 0, first_instance: 0 },
        }));
        let copy = |src: &Arc<MetalTexture>, dst: &Arc<MetalTexture>| {
            Arc::new(MetalOp::Copy(MetalTexCopy {
                src: src.clone(),
                dst: dst.clone(),
                whole: true,
                src_slice: 0,
                src_level: 0,
                src_origin: [0; 3],
                size: [4, 4, 1],
                dst_slice: 0,
                dst_level: 0,
                dst_origin: [0; 3],
            }))
        };
        let clear = |target, color, depth, rects| {
            Arc::new(MetalOp::Clear(MetalClear { target, color, depth, stencil: None, has_stencil: false, rects }))
        };
        let ops = vec![
            clear(attachment.clone(), None, Some(0.125), Vec::new()),
            copy(&original, &cleared),
            clear(input_attachment, Some([0.75, 0.0, 0.0, 0.0]), None, Vec::new()),
            copy(&input, &original),
            draw.clone(),
            draw,
            copy(&original, &output),
            clear(attachment.clone(), None, Some(0.25), vec![victory_metal::clear_resolve::ClearRect { left: 0, top: 0, right: 2, bottom: 2 }]),
            copy(&original, &input),
        ];
        (ops, [output, original, input, cleared])
    }

    /// The four textures' 16 floats each, read back on `backend`'s queue.
    fn read(backend: &MetalBackend, textures: &[Arc<MetalTexture>; 4]) -> Vec<f32> {
        let (out, cpu) = backend.device.shared_buffer_mapped(&[0u8; 4096]).unwrap();
        let cb = backend.queue.command_buffer().unwrap();
        let mut blit = cb.blit_encoder().unwrap();
        for (k, t) in textures.iter().enumerate() {
            blit.copy_from_texture_to_buffer(&t.texture, 0, 0, [0; 3], [4, 4, 1], &out, (k * 1024) as u64, 256, 1024);
        }
        blit.finish();
        cb.commit();
        cb.wait().unwrap();
        let mut v = Vec::new();
        for k in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    // SAFETY: inside the 4 KiB buffer, after completion.
                    v.push(unsafe { cpu.add(k * 1024 + y * 256 + x * 4).cast::<f32>().read_unaligned() });
                }
            }
        }
        v
    }

    fn expected() -> Vec<f32> {
        let mut v = Vec::new();
        for k in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    v.push(match k {
                        3 => 0.125,
                        1 | 2 if x < 2 && y < 2 => 0.25,
                        _ => 0.5,
                    });
                }
            }
        }
        v
    }

    fn timeline(backend: &MetalBackend) -> Timeline {
        let t = Timeline::new(backend.device.clone(), Config { max_segments_per_cb: 1, ..Config::default() }, Hooks::default()).unwrap();
        for q in 0..2 {
            t.queue(q).add_residency_set(backend.chunks().residency());
        }
        let rings: Vec<_> = t.ring_buffers().iter().map(|r| r.handle()).collect();
        backend.chunks().residency().add(&rings);
        backend.chunks().residency().commit();
        t
    }

    /// Clears, copies, depth-companion draws and a rectangle clear encoded by
    /// run_batch into timeline command buffers give what the old path gives.
    #[test]
    #[ignore = "requires Metal"]
    fn timeline_draws_match_old_path() {
        let _pool = victory_metal::Pool::new();
        let old_backend = MetalBackend::new().unwrap();
        let (ops, textures) = sequence(&old_backend);
        assert!(old_backend.run(&ops), "old path failed");
        let old = read(&old_backend, &textures);
        assert_eq!(old, expected(), "old path");

        let backend: &'static MetalBackend = Box::leak(Box::new(MetalBackend::new().unwrap()));
        let (ops, textures) = sequence(backend);
        assert!(encodable(&ops) && !compute_only(&ops));
        let t = timeline(backend);
        diag::LAST_GPU_NS.store(u64::MAX, Ordering::Relaxed);
        let before = GENERAL.each_ref().map(|c| c.load(Ordering::Relaxed));
        let segs = general_segments(backend, &ops, None);
        t.submit(0, segs).unwrap();
        assert!(t.drain(Duration::from_secs(5)).unwrap(), "everything ran");
        assert!(t.stopped().is_none(), "{:?}", t.stopped());
        let new = read(backend, &textures);
        assert_eq!(new, old, "timeline path");
        assert_eq!(diag::LAST_GPU_NS.load(Ordering::Relaxed), u64::MAX, "the old completion path was not used");
        let after = GENERAL.each_ref().map(|c| c.load(Ordering::Relaxed));
        let s = t.stats();
        println!("timeline draws: {} ops in {} chunks, committed {:?}", after[0] - before[0], after[1] - before[1], s.committed);
        assert_eq!(after[0] - before[0], ops.len() as u64);
    }

    /// An indirect draw of 150 records (more than one bounded translation
    /// takes) with a GPU count of 130, on the timeline: drawn in parts of at
    /// most 64 records, every one of the 130 live records exactly once.
    #[test]
    #[ignore = "requires Metal"]
    fn indirect_draw_over_the_record_limit_is_split() {
        let _pool = victory_metal::Pool::new();
        let backend: &'static MetalBackend = Box::leak(Box::new(MetalBackend::new().unwrap()));
        let (seq, _) = sequence(backend);
        let MetalOp::Draw(depth) = &*seq[4] else { panic!("the sequence's draw") };
        // Every vertex invocation counts itself.
        let lib = backend
            .device
            .library_msl(
                "#include <metal_stdlib>\nusing namespace metal;\n\
                 vertex float4 vs(uint id [[vertex_id]], device atomic_uint *n [[buffer(0)]]) {\n\
                 atomic_fetch_add_explicit(n, 1u, memory_order_relaxed);\n\
                 return float4(id == 1 ? 3 : -1, id == 2 ? 3 : -1, 0.5, 1); }\n",
            )
            .unwrap();
        let desc = RenderPipelineDesc { depth_format: P::WMTPixelFormatDepth32Float.0, sample_count: 1, ..Default::default() };
        let pso = Arc::new(MetalGraphics {
            cost_ns: 1.into(),
            name: "stage 4 synthetic indirect draw".into(),
            pso: backend.device.render_pipeline_desc((&lib, "vs"), None, &desc).unwrap(),
            dss: backend.device.depth_stencil_state(&DepthStencilDesc { depth_write: false, depth_compare: 7, ..Default::default() }).unwrap(),
            depth_write: false,
            raster: [0; 4],
            bias: [0.0; 3],
            stages: Vec::new(),
            vertex_slots: 0,
            mesh: false,
            mesh_threads: [0; 3],
            geometry: None,
            colors: [0; 8],
            depth: desc.depth_format,
            stencil: 0,
            samples: 1,
        });
        // Record r: 3 vertices, 1 + r % 2 instances.
        let records: Vec<u8> = (0..150u32).flat_map(|r| [3, 1 + r % 2, 0, 0]).flat_map(u32::to_le_bytes).collect();
        let records = backend.device.shared_buffer(&records).unwrap();
        let count = backend.device.shared_buffer(&130u32.to_le_bytes()).unwrap();
        let (counter, cpu) = backend.device.shared_buffer_mapped(&[0u8; 16]).unwrap();
        backend.chunks().residency().add(&[records.handle(), count.handle(), counter.handle()]);
        backend.chunks().residency().commit();
        let draw = Arc::new(MetalOp::Draw(MetalDraw {
            trace_inputs: Vec::new(),
            pso,
            targets: depth.targets.clone(),
            stages: vec![crate::backend::metal_draw::StageBind { bind: 0, top: Vec::new(), tables: Vec::new(), heap: Some(counter), sampler_heap: None }],
            vertex: None,
            viewport: depth.viewport,
            scissor: depth.scissor,
            blend: [0.0; 4],
            stencil_ref: 0,
            primitive: 3,
            kind: DrawKind::Indirect { records, records_offset: 0, records_va: 0, stride: 16, max: 150, count: Some((count, 0)), count_va: None, index: None },
        }));
        let ops = vec![draw];
        assert!(encodable(&ops) && !compute_only(&ops));
        let t = timeline(backend);
        let before = DRAW_PARTS.load(Ordering::Relaxed);
        t.submit(0, general_segments(backend, &ops, None)).unwrap();
        assert!(t.drain(Duration::from_secs(5)).unwrap(), "everything ran");
        assert!(t.stopped().is_none(), "{:?}", t.stopped());
        // SAFETY: inside the 16-byte buffer, after completion.
        let n = unsafe { cpu.cast::<u32>().read_volatile() };
        let expected: u32 = (0..130u32).map(|r| 3 * (1 + r % 2)).sum();
        println!("indirect draw parts: {} vertex invocations {n} (expected {expected})", DRAW_PARTS.load(Ordering::Relaxed) - before);
        assert_eq!(DRAW_PARTS.load(Ordering::Relaxed) - before, 3, "150 records: 64 + 64 + 22");
        assert_eq!(n, expected, "each of the 130 live records drawn once");
    }

    /// Host cost per op for draws: 64 depth draws with clears between, old
    /// path vs timeline (game thread, encoder, wall).
    #[test]
    #[ignore = "requires Metal"]
    fn host_cost_per_draw() {
        let _pool = victory_metal::Pool::new();
        let backend: &'static MetalBackend = Box::leak(Box::new(MetalBackend::new().unwrap()));
        let (seq, _) = sequence(backend);
        // The draw and the full clear, repeated: 64 draws, 16 clears.
        let mut ops = Vec::new();
        for _ in 0..16 {
            ops.push(seq[0].clone());
            for _ in 0..4 {
                ops.push(seq[4].clone());
            }
        }
        assert!(backend.run(&ops[..5]));
        let n = ops.len() as f64;
        let start = std::time::Instant::now();
        assert!(backend.run(&ops));
        let old = start.elapsed().as_secs_f64() * 1e6 / n;
        let t = timeline(backend);
        t.submit(0, general_segments(backend, &ops[..5], None)).unwrap();
        assert!(t.drain(Duration::from_secs(5)).unwrap());
        let encode_before = t.stats().encode_ns;
        let start = std::time::Instant::now();
        let segs = general_segments(backend, &ops, None);
        t.submit(0, segs).unwrap();
        let game = start.elapsed().as_secs_f64() * 1e6 / n;
        assert!(t.drain(Duration::from_secs(5)).unwrap());
        let wall = start.elapsed().as_secs_f64() * 1e6 / n;
        let st = t.stats();
        let encoder = (st.encode_ns - encode_before) as f64 / 1e3 / n;
        println!(
            "HOST COST per draw op ({} build): old run() {old:.1} µs; timeline: game thread {game:.1} µs, encoder {encoder:.1} µs, wall {wall:.1} µs; {} command buffers",
            if cfg!(debug_assertions) { "debug" } else { "release" },
            st.committed[0]
        );
        assert!(t.stopped().is_none());
        assert!(game < old / 4.0);
        // Where run_batch's encoding goes (this thread, old path).
        crate::backend::metal::work::profile::start();
        assert!(backend.run(&ops));
        let phases = crate::backend::metal::work::profile::take();
        let total: u64 = phases.iter().map(|p| p.1).sum();
        for (phase, ns, k) in phases.iter().take(10) {
            println!("PROFILE {phase:?}: {:.1} µs/op ({:.0}%), entered {k}", *ns as f64 / 1e3 / n, *ns as f64 * 100.0 / total.max(1) as f64);
        }
    }
}
