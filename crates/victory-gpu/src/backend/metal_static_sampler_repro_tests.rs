use super::*;
use crate::backend::metal_draw::{Attachment, DrawKind, MetalClear, MetalDraw, MetalGraphics, STAGE_FRAGMENT, STAGE_VERTEX, StageBind, Targets};
use crate::backend::metal_texture::MetalTexture;
use victory_metal::ir::{DescriptorEntry, ResourceType};
use victory_metal::msc::{self, RootParam, Stage, StaticSampler};
use victory_metal::render::{DepthStencilDesc, RenderPipelineDesc};
use victory_metal::sampler::SamplerDescriptor;
use victory_metal::sys::{self, WMTPixelFormat as P, WMTResourceOptions as Storage};
use victory_metal::texture::{Texture, TextureDescriptor};

fn dxil_dir() -> Option<std::path::PathBuf> {
    if let Some(p) = std::env::var_os("F85_DXIL_DIR") {
        return Some(std::path::PathBuf::from(p));
    }
    let default = std::path::Path::new(r"Z:\Users\aoody743\Documents\GTA VI\metal-backend\flash\F85\dxil");
    default.exists().then(|| default.to_path_buf())
}

#[test]
fn concurrent_auxiliary_submissions_complete() {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
    let _pool = victory_metal::Pool::new();
    let backend = MetalBackend::new().unwrap();
    let mem = unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
    assert!(!mem.is_null());
    let output = unsafe { backend.device.wrap(mem, 4096) }.unwrap();
    let barrier = Arc::new(std::sync::Barrier::new(2));
    std::thread::scope(|scope| {
        for index in 0..2u64 {
            let queue = backend.queue.clone();
            let output = output.clone();
            let barrier = barrier.clone();
            scope.spawn(move || {
                let _pool = victory_metal::Pool::new();
                let cb = queue.command_buffer().unwrap();
                let mut blit = cb.blit_encoder().unwrap();
                blit.fill_buffer(&output, index * 16, 16, (index + 1) as u8);
                blit.finish();
                barrier.wait();
                super::submit_checked(&cb, &[], true, "concurrent auxiliary test").unwrap();
            });
        }
    });
    // SAFETY: both command buffers completed and the allocation remains live.
    let bytes = unsafe { std::slice::from_raw_parts(mem.cast::<u8>(), 32) };
    assert_eq!(&bytes[..16], &[1; 16]);
    assert_eq!(&bytes[16..], &[2; 16]);
    drop(output);
    unsafe { VirtualFree(mem, 0, MEM_RELEASE) };
}

#[test]
fn depth_stencil_copy_keeps_only_depth() {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
    let _pool = victory_metal::Pool::new();
    let backend = MetalBackend::new().unwrap();
    let tex = |format| backend.device.new_texture(&TextureDescriptor {
        pixel_format: format, width: 4, height: 1, usage: sys::WMTTextureUsage(1 | 4),
        options: Storage::WMTResourceStorageModePrivate, ..Default::default()
    }).unwrap();
    let src = tex(P::WMTPixelFormatDepth32Float_Stencil8);
    let dst = tex(P::WMTPixelFormatDepth32Float);
    let expected = [0.0f32, 0.125, 0.75, 1.0];
    let mut bytes = [0x9Du8; 256];
    for (i, f) in expected.iter().enumerate() { bytes[i*4..i*4+4].copy_from_slice(&f.to_le_bytes()); }
    let upload = backend.device.shared_buffer(&bytes).unwrap();
    let mem = unsafe { VirtualAlloc(std::ptr::null(), 16384, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
    assert!(!mem.is_null());
    let output = unsafe { backend.device.wrap(mem, 16384) }.unwrap();
    let cb = backend.queue.command_buffer().unwrap();
    let mut blit = cb.blit_encoder().unwrap();
    blit.copy_from_buffer_to_texture_with_options(&upload, 0, 256, 256, [4,1,1], &src, 0, 0, [0;3], victory_metal::blit::BLIT_OPTION_DEPTH);
    blit.finish();
    let mut scratch = None;
    crate::backend::metal_copy::encode_texture_to_texture(&cb,&src,0,0,[0;3],[4,1,1],&dst,0,0,[0;3],false,&backend.device,&mut scratch).unwrap();
    let mut blit = cb.blit_encoder().unwrap();
    blit.copy_from_texture_to_buffer(&dst,0,0,[0;3],[4,1,1],&output,0,256,256);
    blit.finish();
    cb.commit();
    cb.wait().unwrap();
    assert_eq!(unsafe { *mem.cast::<[f32;4]>() }, expected);
    drop(cb);
    drop(output);
    unsafe { VirtualFree(mem, 0, MEM_RELEASE) };
}

fn make_r8_texture(backend: &MetalBackend, key: usize, value: u8) -> Arc<MetalTexture> {
    let texture = backend
        .device
        .new_texture(&TextureDescriptor {
            pixel_format: P::WMTPixelFormatR8Unorm,
            width: 4,
            height: 4,
            usage: sys::WMTTextureUsage(1 | 0x10), // ShaderRead | PixelFormatView
            options: Storage::WMTResourceStorageModePrivate,
            ..Default::default()
        })
        .unwrap();
    let staging = backend.device.shared_buffer(&[value; 256 * 4]).unwrap();
    let cb = backend.queue.command_buffer().unwrap();
    let mut blit = cb.blit_encoder().unwrap();
    blit.copy_from_buffer_to_texture(&staging, 0, 256, 256 * 4, [4, 4, 1], &texture, 0, 0, [0, 0, 0]);
    blit.finish();
    cb.commit();
    cb.wait().unwrap();
    backend.chunks().residency().add(&[texture.handle()]);
    backend.chunks().residency().commit();

    // SAFETY: ResourceDesc is an integer-only C record.
    let mut desc: crate::resource::ResourceDesc = unsafe { std::mem::zeroed() };
    desc.dimension = 3; // TEXTURE2D
    desc.width = 4;
    desc.height = 4;
    desc.depth_or_array_size = 1;
    desc.mip_levels = 1;
    desc.format = 61; // DXGI_FORMAT_R8_UNORM
    desc.sample_count = 1;
    let tex = Arc::new(MetalTexture {
        texture,
        pixel: P::WMTPixelFormatR8Unorm,
        desc,
        mips: 1,
        slices: 1,
        companion: Default::default(),
        twin: None,
    });
    backend.textures.lock().unwrap().insert(key, Some(tex.clone()));
    tex
}

fn entry_bytes(entry: &DescriptorEntry) -> &[u8] {
    // SAFETY: three initialized u64s with no padding.
    unsafe { core::slice::from_raw_parts(core::ptr::from_ref(entry).cast(), size_of::<DescriptorEntry>()) }
}

#[test]
fn gpu_float_row_then_title_copy_preserves_order_and_guards() {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
    let _pool = victory_metal::Pool::new();
    let backend = MetalBackend::new().unwrap();
    let tex = backend.device.new_texture(&TextureDescriptor {
        pixel_format: P::WMTPixelFormatRGBA32Float, width: 5, height: 1,
        options: Storage::WMTResourceStorageModePrivate, ..Default::default()
    }).unwrap();
    let payload: [u8; 256] = std::array::from_fn(|i| i as u8);
    let upload = backend.device.shared_buffer(&payload).unwrap();
    let cb = backend.queue.command_buffer().unwrap();
    let mut blit = cb.blit_encoder().unwrap();
    blit.copy_from_buffer_to_texture(&upload, 0, 256, 256, [5, 1, 1], &tex, 0, 0, [0; 3]);
    blit.finish();
    cb.commit();
    cb.wait().unwrap();
    // A page-aligned allocation shared with Metal; compare both guards after completion.
    let mem = unsafe { VirtualAlloc(std::ptr::null(), 16384, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
    assert!(!mem.is_null());
    let output = unsafe { backend.device.wrap(mem, 16384) }.unwrap();
    let mut copied = [0xC2u8; 82];
    unsafe { std::ptr::write_bytes(mem, 0xA1, 16384) };
    assert!(backend.run(&[
        Arc::new(MetalOp::FloatRowReadback { src: tex, dst: output.clone(), offset: 4, width: 5 }),
        Arc::new(MetalOp::BufferCopy { dst: copied[1..].as_mut_ptr() as usize, src: mem as usize + 4, bytes: 80 }),
    ]));
    assert_eq!(&copied[1..81], &payload[..80]);
    assert_eq!([copied[0], copied[81]], [0xC2; 2]);
    assert_eq!(unsafe { [*mem.cast::<u8>(), *mem.cast::<u8>().add(84)] }, [0xA1; 2]);
    drop(output);
    unsafe { VirtualFree(mem, 0, MEM_RELEASE) };
}

fn readback_rgba8(backend: &MetalBackend, tex: &Texture) -> [u8; 4] {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
    let mem = unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
    assert!(!mem.is_null());
    let out = unsafe { backend.device.wrap(mem, 4096) }.unwrap();
    let cb = backend.queue.command_buffer().unwrap();
    let mut blit = cb.blit_encoder().unwrap();
    blit.copy_from_texture_to_buffer(tex, 0, 0, [0, 0, 0], [4, 4, 1], &out, 0, 256, 1024);
    blit.finish();
    cb.commit();
    cb.wait().unwrap();
    let px = unsafe { *mem.cast::<[u8; 4]>() };
    drop(cb);
    drop(out);
    unsafe { VirtualFree(mem, 0, MEM_RELEASE) };
    px
}

#[test]
fn static_sampler_r8_draw_and_compute_repro() {
    let Some(dir) = dxil_dir() else {
        println!("SKIP: F85_DXIL_DIR not found");
        return;
    };
    let _pool = victory_metal::Pool::new();
    let backend = MetalBackend::new().unwrap();

    let vs_dxil = std::fs::read(dir.join("vs_main.dxil")).unwrap();
    let ps_dxil = std::fs::read(dir.join("ps_static.dxil")).unwrap();
    let cs_static_dxil = std::fs::read(dir.join("cs_static.dxil")).unwrap();
    let cs_heap_dxil = std::fs::read(dir.join("cs_heap_sampler.dxil")).unwrap();

    // Bink 2 Y/U/V R8_UNORM SRV words from run-metal101 dossier:
    // [61, 4, 5768, 0, 0, 1, 0, 0, 0, 0]
    let _tex_y = make_r8_texture(&backend, 1001, 83);
    let _tex_u = make_r8_texture(&backend, 1002, 147);
    let _tex_v = make_r8_texture(&backend, 1003, 113);
    let _tex_200 = make_r8_texture(&backend, 1004, 200);
    let srv_words = [61u32, 4, 5768, 0, 0, 1, 0, 0, 0, 0];

    let e_y = backend.texture_entry(1001 as crate::object::This, Some(&srv_words), false).unwrap();
    let e_u = backend.texture_entry(1002 as crate::object::This, Some(&srv_words), false).unwrap();
    let e_v = backend.texture_entry(1003 as crate::object::This, Some(&srv_words), false).unwrap();
    let e_200 = backend.texture_entry(1004 as crate::object::This, Some(&srv_words), false).unwrap();

    // Descriptor heap (bind point 0): slots 5, 6, 7, 8
    let mut heap_bytes = vec![0u8; 16 * size_of::<DescriptorEntry>()];
    for (slot, e) in [(5usize, &e_200), (6, &e_y), (7, &e_u), (8, &e_v)] {
        let off = slot * size_of::<DescriptorEntry>();
        heap_bytes[off..off + size_of::<DescriptorEntry>()].copy_from_slice(entry_bytes(e));
    }
    let heap_buf = backend.device.shared_buffer(&heap_bytes).unwrap();
    backend.chunks().residency().add(&[heap_buf.handle()]);

    // Sampler heap (bind point 1): populated identically to draw.rs
    let heap_smp = backend
        .device
        .new_sampler_state(&SamplerDescriptor {
            support_argument_buffers: true,
            ..Default::default()
        })
        .unwrap();
    let sheap_buf = backend
        .device
        .shared_buffer(entry_bytes(&DescriptorEntry::sampler(heap_smp.gpu_resource_id(), 0.0)))
        .unwrap();
    backend.chunks().residency().add(&[sheap_buf.handle()]);

    // Static sampler s0 (D3D12_FILTER_MIN_MAG_MIP_POINT=0, CLAMP=3, NEVER=1)
    let static_sampler = StaticSampler {
        filter: 0,
        address_u: 3,
        address_v: 3,
        address_w: 3,
        mip_lod_bias: 0.0,
        max_anisotropy: 1,
        comparison_func: 1,
        border_color: 0,
        min_lod: 0.0,
        max_lod: 1000.0,
        shader_register: 0,
        register_space: 0,
    };

    let cs_heap_params = [
        RootParam::Constants { register: 0, space: 0, num: 4 },
        RootParam::Uav { register: 0, space: 1 },
    ];

    // Use VirtualAlloc+wrap for output buffer readback (same pattern as metal.rs)
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
    let out_mem = unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
    assert!(!out_mem.is_null());
    let out_wrap = unsafe { backend.device.wrap(out_mem, 4096) }.unwrap();
    backend.chunks().residency().add(&[out_wrap.handle()]);
    backend.chunks().residency().commit();

    let run_compute = |dxil: &[u8], static_smps: &[StaticSampler]| -> [f32; 4] {
        let (pso, mut conv) = backend
            .device
            .compute_pipeline_dxil(dxil, &cs_heap_params, static_smps)
            .unwrap();
        backend
            .bind_static_samplers(&mut conv.top_level, cs_heap_params.len(), static_smps)
            .unwrap();
        assert!(conv.top_level.set_constants(0, 0, &[5, 0, 0, 0]));
        assert!(conv.top_level.set_root_descriptor(ResourceType::Uav, 1, 0, out_wrap.gpu_address));
        unsafe { out_mem.cast::<[f32; 4]>().write([f32::NAN; 4]) };
        let mp = Arc::new(MetalCompute {
            cost_ns: Default::default(),
            name: String::new(),
            pso,
            threadgroup: conv.threadgroup.map(u64::from),
            top_level: conv.top_level.clone(),
        });
        assert!(backend.run(&[Arc::new(MetalOp::Dispatch(MetalCmd {
            pso: mp,
            heap: Some(heap_buf.clone()),
            sampler_heap: Some(sheap_buf.clone()),
            top: conv.top_level.bytes,
            groups: [1, 1, 1],
            tables: Vec::new(),
            loop_counts: None,
        }))]));
        unsafe { out_mem.cast::<[f32; 4]>().read() }
    };

    let got_cs_heap = run_compute(&cs_heap_dxil, &[]);
    let got_cs_static = run_compute(&cs_static_dxil, &[static_sampler]);
    println!("COMPUTE heap_sampler   = {got_cs_heap:?}");
    println!("COMPUTE static_sampler = {got_cs_static:?}");

    // 2. Render draw: VS + PS (ps_static) with static sampler s0, exact metal_draw.rs / metal.rs binding
    let vs = msc::convert(&vs_dxil, Stage::Vertex, &[], &[]).unwrap();
    let ps_params = [RootParam::Constants { register: 0, space: 0, num: 4 }];
    let ps = msc::convert(&ps_dxil, Stage::Fragment, &ps_params, &[static_sampler]).unwrap();
    println!("PS top_level layout = {:?}", ps.top_level.layout);

    let vs_lib = backend.device.library(&vs.metallib).unwrap();
    let ps_lib = backend.device.library(&ps.metallib).unwrap();
    let desc = RenderPipelineDesc {
        colors: {
            let mut c = [victory_metal::render::ColorDesc::default(); 8];
            c[0].pixel_format = P::WMTPixelFormatRGBA8Unorm.0;
            c[0].write_mask = 0xf;
            c
        },
        sample_count: 1,
        topology_class: 3, // Triangle
        ..Default::default()
    };
    let pso = backend
        .device
        .render_pipeline_desc((&vs_lib, &vs.entry), Some((&ps_lib, &ps.entry)), &desc)
        .unwrap();
    let dss = backend
        .device
        .depth_stencil_state(&DepthStencilDesc {
            depth_compare: 7, // Always
            depth_write: false,
            ..Default::default()
        })
        .unwrap();

    let mut top = ps.top_level.clone();
    assert!(top.set_constants(0, 0, &[5, 0, 0, 0]));
    let graphics = Arc::new(MetalGraphics {
        cost_ns: Default::default(),
        name: "F85 static sampler repro".into(),
        pso,
        dss,
        depth_write: false,
        raster: [0, 0, 0, 1],
        bias: [0.0; 3],
        stages: vec![(vs.top_level.clone(), STAGE_VERTEX), (top.clone(), STAGE_FRAGMENT)],
        vertex_slots: 0,
        mesh: false,
        mesh_threads: [1, 1, 1],
        geometry: None,
        colors: [P::WMTPixelFormatRGBA8Unorm.0, 0, 0, 0, 0, 0, 0, 0],
        depth: 0,
        stencil: 0,
        samples: 1,
    });

    let rt_tex = backend
        .device
        .new_texture(&TextureDescriptor {
            pixel_format: P::WMTPixelFormatRGBA8Unorm,
            width: 4,
            height: 4,
            usage: sys::WMTTextureUsage(1 | 4),
            options: Storage::WMTResourceStorageModePrivate,
            ..Default::default()
        })
        .unwrap();
    let att = Attachment {
        resource: 0,
        texture: rt_tex.clone(),
        pixel: P::WMTPixelFormatRGBA8Unorm.0,
        level: 0,
        slice: 0,
        width: 4,
        height: 4,
        samples: 1,
        companion: None,
        twin: None,
    };
    let mut targets = Targets::default();
    targets.colors[0] = Some(att.clone());
    targets.finish();
    let targets = Arc::new(targets);

    let draw_with = |base: &TopLevel, slots: [u32; 4]| -> [u8; 4] {
        let mut ps_top = base.clone();
        assert!(ps_top.set_constants(0, 0, &slots));
        let draw = MetalDraw {
            trace_inputs: Vec::new(),
            pso: graphics.clone(),
            targets: targets.clone(),
            stages: vec![
                StageBind {
                    bind: STAGE_VERTEX,
                    top: vs.top_level.bytes.clone(),
                    tables: Vec::new(),
                    heap: Some(heap_buf.clone()),
                    sampler_heap: Some(sheap_buf.clone()),
                },
                StageBind {
                    bind: STAGE_FRAGMENT,
                    top: ps_top.bytes,
                    tables: Vec::new(),
                    heap: Some(heap_buf.clone()),
                    sampler_heap: Some(sheap_buf.clone()),
                },
            ],
            vertex: None,
            viewport: sys::WMTViewport { originX: 0.0, originY: 0.0, width: 4.0, height: 4.0, znear: 0.0, zfar: 1.0 },
            scissor: sys::WMTScissorRect { x: 0, y: 0, width: 4, height: 4 },
            blend: [1.0; 4],
            stencil_ref: 0,
            primitive: 3, // Triangle
            kind: DrawKind::Vertices { count: 3, instances: 1, first: 0, first_instance: 0 },
        };
        let clear = MetalClear {
            target: att.clone(),
            color: Some([0.0, 0.0, 0.0, 0.0]),
            depth: None,
            stencil: None,
            has_stencil: false,
            rects: Vec::new(),
        };
        assert!(backend.run(&[Arc::new(MetalOp::Clear(clear)), Arc::new(MetalOp::Draw(draw))]));
        readback_rgba8(&backend, &rt_tex)
    };

    // Unbound baseline (reproduces the pre-fix bug: null static sampler -> 0)
    let got_unbound_r8 = draw_with(&ps.top_level, [5, 0, 0, 0]);
    let got_unbound_yuv = draw_with(&ps.top_level, [6, 7, 8, 1]);
    println!("PIXEL unbound static_sampler R8  = {got_unbound_r8:?}");
    println!("PIXEL unbound static_sampler YUV = {got_unbound_yuv:?}");
    assert_eq!(got_unbound_r8, [0, 0, 0, 0]);
    assert_eq!(got_unbound_yuv, [0, 135, 0, 255]);

    // With bind_static_samplers fix applied
    backend.bind_static_samplers(&mut top, ps_params.len(), &[static_sampler]).unwrap();
    let got_ps_r8 = draw_with(&top, [5, 0, 0, 0]);
    let got_ps_yuv = draw_with(&top, [6, 7, 8, 1]);
    println!("PIXEL static_sampler R8  = {got_ps_r8:?}");
    println!("PIXEL static_sampler YUV = {got_ps_yuv:?}");

    drop(out_wrap);
    unsafe { VirtualFree(out_mem, 0, MEM_RELEASE) };

    assert!((got_cs_heap[0] - 200.0 / 255.0).abs() < 0.01, "heap sampler compute failed: {got_cs_heap:?}");
    assert!((got_cs_static[0] - 200.0 / 255.0).abs() < 0.01, "static sampler compute returned zeros: {got_cs_static:?}");
    assert_eq!(got_ps_r8, [200, 0, 0, 255], "static sampler pixel R8 returned wrong color: {got_ps_r8:?}");
    assert_ne!(got_ps_yuv, [0, 135, 0, 255], "YUV draw still getting Y=U=V=0 -> (0,135,0): {got_ps_yuv:?}");
}

/// Compute split over several command buffers (`run_batch`, at most
/// `VICTORY_METAL_PASS_DISPATCHES` per pass) still runs in order: 37 dispatches each doing
/// `x = x * 3 + k` (k = 1..=37) give the CPU's value only in order.
#[test]
fn split_compute_runs_in_order() {
    run_order_probe(None);
}

#[test]
#[ignore = "trips process-wide fuse; run alone"]
fn exhausted_ring_stops_submission() { run_order_probe(Some("ring")); }

#[test]
#[ignore = "trips process-wide fuse; run alone"]
fn missing_event_stops_submission() { run_order_probe(Some("event")); }

#[test]
fn large_direct_dispatch_is_not_rejected() {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
    const GROUPS: u64 = 4097;
    const BYTES: usize = 32 * 1024;
    let _pool = victory_metal::Pool::new();
    let backend = MetalBackend::new().unwrap();
    let ptr = unsafe { VirtualAlloc(std::ptr::null(), BYTES, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
    assert!(!ptr.is_null());
    let dst = unsafe { backend.device.wrap(ptr, BYTES as u64) }.unwrap();
    let pipeline = backend.device.compute_pipeline_msl(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         kernel void mark(device uint *dst [[buffer(0)]], uint3 id [[threadgroup_position_in_grid]]) { dst[id.x] = id.x + 1; }",
        "mark",
    ).unwrap();
    let cb = backend.queue.command_buffer().unwrap();
    cb.compute(&[
        victory_metal::Compute::Pipeline { pipeline: &pipeline, threads_per_group: [1; 3] },
        victory_metal::Compute::Buffer { buffer: &dst, offset: 0, index: 0 },
        victory_metal::Compute::Dispatch { groups: [GROUPS, 1, 1] },
    ]).unwrap();
    assert_eq!(cb.encoded_work().groups, GROUPS);
    submit_checked(&cb, &[], true, "large direct test").unwrap();
    // SAFETY: the mapped buffer is live and its sole GPU writer has completed.
    let bytes = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), GROUPS as usize * 4) };
    assert!(bytes.chunks_exact(4).enumerate().all(|(i, value)| {
        u32::from_ne_bytes(value.try_into().unwrap()) == i as u32 + 1
    }));
    assert!(!backend.submissions_stopped());
    drop(cb);
    drop(dst);
    unsafe { VirtualFree(ptr, 0, MEM_RELEASE) };
}

#[test]
#[ignore = "requires Metal; process-local fuse"]
fn fused_capture_and_present() {
    use std::sync::atomic::Ordering;
    let _pool = victory_metal::Pool::new();
    let backend = MetalBackend::new().unwrap();
    super::fuse::stop();
    let before = super::STOPPED_ENDPOINTS.load(Ordering::Relaxed);
    backend.capture(std::ptr::null(), "fused");
    assert!(!backend.present_metal(std::ptr::null()));
    assert_eq!(super::STOPPED_ENDPOINTS.load(Ordering::Relaxed) - before, 2);
}

#[test]
fn over_window_estimate_runs_alone() { run_order_probe(Some("estimated-time")); }

#[test]
fn budget_boundary_preserves_order() { run_order_probe(Some("split-time")); }

#[test]
#[ignore = "trips process-wide fuse; run alone"]
fn measured_fuse_stops_batch_before_next_operation() { run_order_probe(Some("measured-time")); }

fn run_order_probe(fault: Option<&'static str>) {
    let dxil = std::fs::read(dxil_dir().expect("F85 DXIL fixtures required").join("cs_order.dxil")).unwrap();
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
    let _pool = victory_metal::Pool::new();
    let backend = MetalBackend::new().unwrap();
    let params = [
        RootParam::Constants { register: 0, space: 0, num: 4 },
        RootParam::Uav { register: 0, space: 1 },
    ];
    let mem = unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
    assert!(!mem.is_null());
    let out = unsafe { backend.device.wrap(mem, 4096) }.unwrap();
    backend.chunks().residency().add(&[out.handle()]);
    backend.chunks().residency().commit();
    let (pso, conv) = backend.device.compute_pipeline_dxil(&dxil, &params, &[]).unwrap();
    let mp = Arc::new(MetalCompute {
        cost_ns: Default::default(),
        name: "order probe".into(),
        pso,
        threadgroup: conv.threadgroup.map(u64::from),
        top_level: conv.top_level.clone(),
    });
    let mut ops = Vec::new();
    let mut expected = 0u32;
    for k in 1..=if fault.is_some() { 2 } else { 37u32 } {
        let mut top = conv.top_level.clone();
        assert!(top.set_constants(0, 0, &[k, 0, 0, 0]));
        assert!(top.set_root_descriptor(ResourceType::Uav, 1, 0, out.gpu_address));
        ops.push(Arc::new(MetalOp::Dispatch(MetalCmd {
            pso: mp.clone(),
            heap: None,
            sampler_heap: None,
            top: top.bytes,
            groups: [1; 3],
            tables: Vec::new(),
            loop_counts: None,
        })));
        expected = expected.wrapping_mul(3).wrapping_add(k);
    }
    unsafe { mem.cast::<u32>().write(0) };
    work::FAULT.set(fault);
    if fault == Some("estimated-time") {
        mp.cost_ns.store(work::WINDOW_NS + 1, std::sync::atomic::Ordering::Relaxed);
    }
    if fault == Some("measured-time") {
        super::fuse::TEST_ONE_NS.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(!backend.run(&ops));
        assert!(backend.submissions_stopped());
        assert_eq!(unsafe { mem.cast::<u32>().read() }, 1, "only the first dispatch may execute");
        return;
    }
    if fault == Some("split-time") {
        mp.cost_ns.store(30_000_000, std::sync::atomic::Ordering::Relaxed);
    } else if matches!(fault, Some("ring" | "event")) {
        assert!(!backend.run(&ops));
        assert_eq!(work::REJECTIONS.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(unsafe { mem.cast::<u32>().read() }, 0);
        assert!(backend.submissions_stopped());
        let cb = backend.queue.command_buffer().unwrap();
        assert!(submit_checked(&cb, &[], false, "after injected fault").is_err());
        assert_eq!(unsafe { sys::MTLCommandBuffer_status(cb.handle()) },
            sys::WMTCommandBufferStatus::WMTCommandBufferStatusNotEnqueued);
        return;
    }
    assert!(backend.run(&ops));
    if matches!(fault, Some("split-time" | "estimated-time")) {
        assert_eq!(work::SPLITS.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert!(!backend.submissions_stopped());
    }
    let got = unsafe { mem.cast::<u32>().read() };
    assert_eq!(got, expected, "dispatches ran out of order or were lost");
    assert_ne!(mp.cost_ns.load(std::sync::atomic::Ordering::Relaxed), 0, "completion worker did not publish pipeline timing");
}

/// The fuse (`VICTORY_METAL_FUSE_MS`): after a command buffer exceeds the
/// limit, later work in the same batch and subsequent submissions fail.
/// Run alone with `VICTORY_METAL_FUSE_MS=1` and
/// `VICTORY_METAL_PASS_DISPATCHES=1` (the settings are read once per process).
#[test]
fn fuse_leaves_out_a_slow_pipeline() {
    if std::env::var("VICTORY_METAL_FUSE_MS").as_deref() != Ok("1")
        || std::env::var("VICTORY_METAL_PASS_DISPATCHES").as_deref() != Ok("1")
    {
        println!("SKIP: run with VICTORY_METAL_FUSE_MS=1 and VICTORY_METAL_PASS_DISPATCHES=1");
        return;
    }
    let Some(dir) = dxil_dir() else { return };
    let Ok(dxil) = std::fs::read(dir.join("cs_slow.dxil")) else {
        println!("SKIP: cs_slow.dxil not built");
        return;
    };
    let Ok(order_dxil) = std::fs::read(dir.join("cs_order.dxil")) else {
        println!("SKIP: cs_order.dxil not built");
        return;
    };
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
    let _pool = victory_metal::Pool::new();
    let backend = MetalBackend::new().unwrap();
    let params = [
        RootParam::Constants { register: 0, space: 0, num: 4 },
        RootParam::Uav { register: 0, space: 1 },
    ];
    let mem = unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
    let out = unsafe { backend.device.wrap(mem, 4096) }.unwrap();
    backend.chunks().residency().add(&[out.handle()]);
    backend.chunks().residency().commit();
    let (pso, conv) = backend.device.compute_pipeline_dxil(&dxil, &params, &[]).unwrap();
    let mp = Arc::new(MetalCompute { cost_ns: Default::default(), name: "slow probe".into(), pso, threadgroup: conv.threadgroup.map(u64::from), top_level: conv.top_level.clone() });
    let mut top = conv.top_level.clone();
    assert!(top.set_constants(0, 0, &[7, 0, 0, 0]));
    assert!(top.set_root_descriptor(ResourceType::Uav, 1, 0, out.gpu_address));
    let op = || Arc::new(MetalOp::Dispatch(MetalCmd { pso: mp.clone(), heap: None, sampler_heap: None, top: top.bytes.clone(), groups: [1, 1, 1], tables: Vec::new(), loop_counts: None }));
    let (order_pso, order_conv) = backend.device.compute_pipeline_dxil(&order_dxil, &params, &[]).unwrap();
    let order_mp = Arc::new(MetalCompute {
        cost_ns: Default::default(),
        name: "after slow probe".into(),
        pso: order_pso,
        threadgroup: order_conv.threadgroup.map(u64::from),
        top_level: order_conv.top_level.clone(),
    });
    let mut order_top = order_conv.top_level.clone();
    assert!(order_top.set_constants(0, 0, &[17, 0, 0, 0]));
    assert!(order_top.set_root_descriptor(ResourceType::Uav, 1, 0, out.gpu_address + 4));
    let after = Arc::new(MetalOp::Dispatch(MetalCmd {
        pso: order_mp,
        heap: None,
        sampler_heap: None,
        top: order_top.bytes,
        groups: [1, 1, 1],
        tables: Vec::new(),
        loop_counts: None,
    }));
    unsafe { mem.cast::<u64>().write(0) };
    assert!(!backend.run(&[op(), after]));
    assert_ne!(unsafe { mem.cast::<u32>().read() }, 0, "the first run ran");
    assert_eq!(unsafe { mem.cast::<u32>().add(1).read() }, 0, "later work in the batch must not run");
    unsafe { mem.cast::<u32>().write(0) };
    assert!(!backend.run(&[op()]));
    assert_eq!(unsafe { mem.cast::<u32>().read() }, 0, "the fuse stopped subsequent submissions");
    let auxiliary = backend.queue.command_buffer().unwrap();
    assert!(super::submit_checked(&auxiliary, &[], true, "test auxiliary").is_err());
    // SAFETY: the command buffer is live and has not been submitted.
    let status = unsafe { victory_metal::sys::MTLCommandBuffer_status(auxiliary.handle()) };
    assert_eq!(status, victory_metal::sys::WMTCommandBufferStatus::WMTCommandBufferStatusNotEnqueued);
}
