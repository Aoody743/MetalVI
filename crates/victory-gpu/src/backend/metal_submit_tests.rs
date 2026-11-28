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
