//! Command-buffer cutting and the hang watch's host phases (2026-11-28
//! freezes): run on the Metal path under F85 Wine, with Metal API
//! validation and `VICTORY_METAL_PASS_DISPATCHES` > 1.
use super::*;
use victory_metal::ir::ResourceType;
use victory_metal::msc::RootParam;

fn dxil_dir() -> Option<std::path::PathBuf> {
    if let Some(p) = std::env::var_os("F85_DXIL_DIR") {
        return Some(std::path::PathBuf::from(p));
    }
    let default = std::path::Path::new(r"Z:\Users\aoody743\Documents\GTA VI\metal-backend\flash\F85\dxil");
    default.exists().then(|| default.to_path_buf())
}

/// Direct and indirect dispatches alternating in one batch (`x = x * 3 + k`
/// on title memory reached by GPU address): with buffers cut by dispatch
/// count they share command buffers, and must still run in order.
#[test]
fn direct_and_indirect_dispatches_sharing_buffers_run_in_order() {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
    let dxil = std::fs::read(dxil_dir().expect("F85 DXIL fixtures required").join("cs_order.dxil")).unwrap();
    let _pool = victory_metal::Pool::new();
    let backend = MetalBackend::new().unwrap();
    let params = [RootParam::Constants { register: 0, space: 0, num: 4 }, RootParam::Uav { register: 0, space: 1 }];
    let mem = unsafe { VirtualAlloc(std::ptr::null(), 8192, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
    assert!(!mem.is_null());
    // The first page: the value; the second: one indirect record (1, 1, 1).
    let out = unsafe { backend.device.wrap(mem, 8192) }.unwrap();
    backend.chunks().residency().add(&[out.handle()]);
    backend.chunks().residency().commit();
    unsafe { mem.cast::<u8>().add(4096).cast::<[u32; 3]>().write([1, 1, 1]) };
    let (pso, conv) = backend.device.compute_pipeline_dxil(&dxil, &params, &[]).unwrap();
    let mp = Arc::new(MetalCompute {
        cost_ns: Default::default(),
        name: "mixed order probe".into(),
        pso,
        threadgroup: conv.threadgroup.map(u64::from),
        top_level: conv.top_level.clone(),
    });
    let mut ops = Vec::new();
    let mut expected = 0u32;
    for k in 1..=40u32 {
        let mut top = conv.top_level.clone();
        assert!(top.set_constants(0, 0, &[k, 0, 0, 0]));
        assert!(top.set_root_descriptor(ResourceType::Uav, 1, 0, out.gpu_address));
        ops.push(Arc::new(if k % 2 == 0 {
            MetalOp::DispatchIndirect(MetalDispatchIndirect {
                pso: mp.clone(),
                heap: None,
                sampler_heap: None,
                top: top.bytes,
                tables: Vec::new(),
                records: out.clone(),
                records_offset: 4096,
                stride: 12,
                max: 1,
                count: None,
                records_va: mem as u64 + 4096,
                count_va: None,
            })
        } else {
            MetalOp::Dispatch(MetalCmd {
                pso: mp.clone(),
                heap: None,
                sampler_heap: None,
                top: top.bytes,
                groups: [1; 3],
                tables: Vec::new(),
                loop_counts: None,
                after_barrier: true,
            })
        }));
        expected = expected.wrapping_mul(3).wrapping_add(k);
    }
    unsafe { mem.cast::<u32>().write(0) };
    // Twice: the second batch runs with the solo times learnt from the first.
    for round in 0..2 {
        unsafe { mem.cast::<u32>().write(0) };
        let before = work::SPLITS.load(std::sync::atomic::Ordering::Relaxed);
        assert!(backend.run(&ops), "round {round}");
        let splits = work::SPLITS.load(std::sync::atomic::Ordering::Relaxed) - before;
        let got = unsafe { mem.cast::<u32>().read() };
        println!("round {round}: {splits} splits");
        assert_eq!(got, expected, "round {round}: dispatches ran out of order or were lost");
    }
    // Nothing left in a host phase once the batch is done.
    assert!(work::stuck_phase(work::LANE_RUN, 0).is_none());
}

/// The hang watch's host phases: a phase older than the limit is reported
/// once per start, idle is never reported.
#[test]
fn a_host_phase_older_than_the_limit_is_stuck() {
    work::phase(work::LANE_CAPTURE, "test phase");
    assert!(work::stuck_phase(work::LANE_CAPTURE, 60_000).is_none());
    std::thread::sleep(std::time::Duration::from_millis(30));
    let (phase, age, _) = work::stuck_phase(work::LANE_CAPTURE, 10).unwrap();
    assert_eq!(phase, "test phase");
    assert!(age >= 10);
    work::phase(work::LANE_CAPTURE, "");
    assert!(work::stuck_phase(work::LANE_CAPTURE, 0).is_none());
}

/// A dirty partly filled texture: `texture_for` refills it and returns, it
/// does not deadlock on the textures lock it still held (2026-11-28 freezes).
#[test]
fn texture_for_refills_a_dirty_partial_texture_without_deadlock() {
    let _pool = victory_metal::Pool::new();
    let backend: &'static MetalBackend = Box::leak(Box::new(MetalBackend::new().unwrap()));
    let texture = backend.device.new_texture(&victory_metal::texture::TextureDescriptor {
        pixel_format: victory_metal::sys::WMTPixelFormat::WMTPixelFormatR32Float,
        width: 4,
        height: 4,
        usage: victory_metal::sys::WMTTextureUsage(1 | 2 | 4 | 0x10),
        options: victory_metal::sys::WMTResourceOptions(0x20),
        ..Default::default()
    }).unwrap();
    // SAFETY: ResourceDesc is an integer-only C record.
    let mut rdesc: crate::resource::ResourceDesc = unsafe { std::mem::zeroed() };
    rdesc.width = 4;
    rdesc.height = 4;
    rdesc.dimension = 3;
    rdesc.sample_count = 1;
    const KEY: usize = 0x7e57_0001;
    backend.textures.lock().unwrap().insert(KEY, Some(Arc::new(crate::backend::metal_texture::MetalTexture {
        companion: Default::default(),
        texture,
        pixel: victory_metal::sys::WMTPixelFormat::WMTPixelFormatR32Float,
        desc: rdesc,
        mips: 1,
        slices: 1,
    })));
    let memory = Box::leak(vec![0u8; 4096].into_boxed_slice());
    MetalBackend::mark_partial_dirty_for_test(KEY, memory.as_ptr() as u64);
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _pool = victory_metal::Pool::new();
        let _ = tx.send(backend.texture_for(KEY as crate::object::This).is_some());
    });
    assert_eq!(rx.recv_timeout(std::time::Duration::from_secs(10)), Ok(true), "texture_for deadlocked on a dirty partial texture");
}

/// Title memory is wired only while no command buffer is in flight
/// (`IN_FLIGHT`): a span asked for while a long bounded compute buffer runs
/// waits for it to complete; with none in flight it does not wait.
#[test]
fn wiring_waits_for_the_command_buffer_in_flight() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
    let _pool = victory_metal::Pool::new();
    let backend: &'static MetalBackend = Box::leak(Box::new(MetalBackend::new().unwrap()));
    let alloc = |bytes: usize| {
        // SAFETY: a fresh allocation of our own, never freed (test).
        let mem = unsafe { VirtualAlloc(std::ptr::null(), bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        assert!(!mem.is_null());
        mem as u64
    };
    // Nothing in flight: a span is made at once.
    let idle = alloc(32 << 20);
    let t = Instant::now();
    assert!(backend.chunks().window_buffer(idle + 4096, 4096).is_some());
    let idle_ms = t.elapsed().as_millis();
    // A long compute buffer (bounded: a fixed loop, under the 1 s test fuse).
    let pipeline = backend.device.compute_pipeline_msl(
        "#include <metal_stdlib>\nkernel void spin(device uint *p [[buffer(0)]], uint id [[thread_position_in_grid]]) {\n uint a = id; for (uint i = 0; i < 6000000u; ++i) { a = a * 1664525u + 1013904223u; } p[id] = a; }",
        "spin",
    ).unwrap();
    let out = backend.device.shared_buffer(&[0u8; 256]).unwrap();
    let committed: &'static AtomicBool = Box::leak(Box::new(AtomicBool::new(false)));
    let busy = alloc(32 << 20);
    let runner = std::thread::spawn(move || {
        let _pool = victory_metal::Pool::new();
        let cb = backend.queue.command_buffer().unwrap();
        cb.compute(&[
            victory_metal::Compute::Pipeline { pipeline: &pipeline, threads_per_group: [64, 1, 1] },
            victory_metal::Compute::Buffer { buffer: &out, offset: 0, index: 0 },
            victory_metal::Compute::Dispatch { groups: [1, 1, 1] },
        ]).unwrap();
        committed.store(true, Ordering::SeqCst);
        submit_checked(&cb, &[], true, "long test buffer").unwrap();
        (Instant::now(), cb.gpu_time_ns())
    });
    while !committed.load(Ordering::SeqCst) {
        std::thread::yield_now();
    }
    // Let the runner take IN_FLIGHT and commit.
    std::thread::sleep(Duration::from_millis(5));
    let asked = Instant::now();
    assert!(backend.chunks().window_buffer(busy + 4096, 4096).is_some());
    let wired = Instant::now();
    let (done, gpu_ns) = runner.join().unwrap();
    let waited_ms = (wired - asked).as_millis();
    println!("idle span {idle_ms} ms; span during a {} ms buffer waited {waited_ms} ms", gpu_ns / 1_000_000);
    assert!(gpu_ns > 30_000_000, "the test buffer must run long enough to overlap: {gpu_ns} ns");
    assert!(idle_ms < 20, "no buffer in flight: no wait ({idle_ms} ms)");
    // The span was made only after the buffer completed (the runner returns
    // just after releasing IN_FLIGHT, so allow a little scheduling slack).
    assert!(wired + Duration::from_millis(5) >= done, "wired {:?} before the buffer completed", done - wired);
    assert!(u128::from(gpu_ns / 1_000_000) <= waited_ms + 10, "waited {waited_ms} ms for a {} ms buffer", gpu_ns / 1_000_000);
}
