use super::*;
use crate::msc::{self, Stage};
use crate::render::{ColorAttachment, ColorDesc, RenderPipelineDesc};
use crate::sys;
use crate::texture::{Texture, TextureDescriptor};
use crate::{Object, Pool};

const TRIANGLE: sys::WMTPrimitiveType = sys::WMTPrimitiveType::WMTPrimitiveTypeTriangle;
const RGBA: sys::WMTPixelFormat = sys::WMTPixelFormat::WMTPixelFormatRGBA8Unorm;
const W: u32 = 64;
const H: u32 = 32;

struct Shared(Buffer, *mut u8);

impl Shared {
    fn new<T>(device: &Device, data: &[T]) -> Self {
        let len = std::mem::size_of_val(data);
        let mut info = sys::WMTBufferInfo {
            length: len as u64,
            options: sys::WMTResourceOptions::WMTResourceStorageModeShared,
            memory: sys::WMTMemoryPointer {
                ptr: std::ptr::null_mut(),
            },
            gpu_address: 0,
        };
        // SAFETY: a live device, initialized info, and null memory requests a new allocation.
        let object =
            Object::adopt(unsafe { sys::MTLDevice_newBuffer(device.handle(), &mut info) }).unwrap();
        let ptr = info.memory.ptr.cast::<u8>();
        assert!(!ptr.is_null());
        // SAFETY: data is initialized and the fresh shared allocation is CPU-owned.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr().cast(), ptr, len) };
        Self(
            Buffer {
                object,
                gpu_address: info.gpu_address,
                len: len as u64,
            },
            ptr,
        )
    }

    fn words(&self) -> Vec<u32> {
        // SAFETY: used only after GPU completion with the shared buffer retained.
        unsafe {
            std::slice::from_raw_parts(self.1.cast::<u32>(), self.0.len as usize / 4).to_vec()
        }
    }
}

fn target(device: &Device) -> Texture {
    device
        .new_texture(&TextureDescriptor {
            pixel_format: RGBA,
            width: W,
            height: H,
            usage: sys::WMTTextureUsage::WMTTextureUsageRenderTarget,
            ..Default::default()
        })
        .unwrap()
}

// Fixed trip count, one lane per group. The dependent index never leaves the
// 16-word allocation. This measures a whole CB, not instruction latency.
fn calibration(iterations: u32, groups: u32) {
    assert!([1, 4, 16].contains(&iterations) && [1, 16, 256].contains(&groups));
    let _pool = Pool::new();
    let device = Device::first().unwrap();
    let queue = device.new_queue(1).unwrap();
    let table = Shared::new(&device, &std::array::from_fn::<_, 16, _>(|i| (i as u32 + 3) & 15));
    let dst = Shared::new(&device, &vec![u32::MAX; groups as usize]);
    let src = format!("#include <metal_stdlib>\nusing namespace metal;\n\
        kernel void chase(device const volatile uint *t [[buffer(0)]], device uint *o [[buffer(1)]], \
        uint g [[thread_position_in_grid]]) {{ uint v=g&15; for(uint i=0;i<{iterations};++i) \
        v=t[v&15]; o[g]=v; }}");
    let lib = device.library_msl(&src).unwrap();
    let pso = device.compute_pipeline_from_library(&lib, "chase").unwrap();
    let cb = queue.command_buffer().unwrap();
    cb.compute(&[
        Compute::Pipeline { pipeline: &pso, threads_per_group: [1; 3] },
        Compute::Buffer { buffer: &table.0, offset: 0, index: 0 },
        Compute::Buffer { buffer: &dst.0, offset: 0, index: 1 },
        Compute::Dispatch { groups: [groups.into(), 1, 1] },
    ]).unwrap();
    cb.commit();
    cb.wait().unwrap();
    let ns = cb.gpu_time_ns();
    println!("calibration device={} groups={groups} iterations={iterations} cb_ns={ns}", device.name());
    assert!(ns > 0 && ns < 50_000_000);
    assert_eq!(dst.words(), (0..groups).map(|g| (g + 3 * iterations) & 15).collect::<Vec<_>>());
}

#[test]
#[ignore = "requires Metal"]
fn calibration_one() { calibration(1, 1); }
#[test]
#[ignore = "requires Metal"]
fn calibration_four() { calibration(4, 1); }
#[test]
#[ignore = "requires Metal"]
fn calibration_sixteen() { calibration(16, 1); }
#[test]
#[ignore = "requires Metal"]
fn calibration_groups_16() { calibration(16, 16); }
#[test]
#[ignore = "requires Metal"]
fn calibration_groups_256() { calibration(16, 256); }

#[test]
#[ignore = "requires Metal"]
fn bounded_mesh_fragment_amplification() {
    let _pool = Pool::new();
    let dev = Device::first().unwrap();
    let queue = dev.new_queue(1).unwrap();
    let lib = dev.library_msl(include_str!("test.metal")).unwrap();
    let mut desc = RenderPipelineDesc::default();
    desc.colors[0] = ColorDesc { pixel_format: RGBA.0, write_mask: 15, ..Default::default() };
    desc.sample_count = 1;
    let pso = dev.mesh_render_pipeline_desc((&lib, "double_mesh"), Some((&lib, "count_fragments")), &desc).unwrap();
    let image = target(&dev);
    let hits = Shared::new(&dev, &[0u32; (W*H) as usize]);
    let cb = queue.command_buffer().unwrap();
    cb.render_pass(&[color(&image)], W, H, &[
        Render::Pipeline { pipeline: &pso },
        Render::FragmentBuffer { buffer: &hits.0, offset: 0, index: 0 },
        Render::DrawMesh { groups: [1;3], object_threads: [1;3], mesh_threads: [1;3] },
    ]).unwrap();
    assert_eq!(cb.encoded_work().groups, 1);
    cb.commit();
    cb.wait().unwrap();
    assert!(hits.words().iter().all(|&n| n == 2));
    println!("mesh groups=1 triangles=2 fragment_hits={} cb_ns={}", 2*W*H, cb.gpu_time_ns());
}

#[test]
#[ignore = "requires Metal"]
fn bounded_title_mesh_records() {
    let _pool = Pool::new();
    let device = Device::first().unwrap();
    let queue = device.new_queue(4).unwrap();
    let translator = IndirectTranslator::new(&device).unwrap();
    let words = [4u32,1,1, 65536,1,1, 1,4,1, 1,1,2, 65535,1,1];
    let src = Shared::new(&device, &words);
    let out = Shared::new(&device, &[0xFEu32;25]);
    let cb = queue.command_buffer().unwrap();
    let batch = translator.encode(&device, &cb, &IndirectSource { dispatch_segments: None,
        kind: IndirectKind::TitleMesh1D, records: (&src.0, 0), stride: 12, max_count: 5, count: None,
    }).unwrap().unwrap();
    let mut blit = cb.blit_encoder().unwrap();
    blit.copy_buffer(&batch.output, 0, &out.0, 0, 100);
    blit.finish();
    cb.commit();
    cb.wait().unwrap();
    assert_eq!(out.words(), [[4,1,1,0,0],[0;5],[0;5],[0;5],[65535,1,1,0,0]].as_flattened());
}

fn color(texture: &Texture) -> ColorAttachment<'_> {
    ColorAttachment {
        texture,
        load_action: sys::WMTLoadAction::WMTLoadActionClear,
        store_action: sys::WMTStoreAction::WMTStoreActionStore,
        clear: [0.0, 0.0, 1.0, 1.0],
    }
}

fn read_texture(texture: &Texture) -> Vec<[u8; 4]> {
    let mut pixels = vec![[0; 4]; (W * H) as usize];
    // SAFETY: called after GPU completion; pixels has room for the whole texture.
    unsafe {
        sys::MTLTexture_getBytes(
            texture.handle(),
            sys::WMTOrigin { x: 0, y: 0, z: 0 },
            sys::WMTSize {
                width: W as u64,
                height: H as u64,
                depth: 1,
            },
            0,
            0,
            sys::WMTMemoryPointer {
                ptr: pixels.as_mut_ptr().cast(),
            },
            (W * 4) as u64,
            (W * H * 4) as u64,
        )
    };
    pixels
}

#[test]
#[ignore = "requires Rosetta Wine, private victorymetal and optional F42_SHADER_DIR DXIL fixtures"]
fn wine_gpu_readback() {
    let _pool = Pool::new();
    let device = Device::first().unwrap();
    println!("Metal device: {}", device.name());
    let queue = device.new_queue(4).unwrap();
    let translator = IndirectTranslator::new(&device).unwrap();
    let library = device.library_msl(include_str!("test.metal")).unwrap();
    let draw_pso = device
        .render_pipeline(&library, "verify_vertex", &library, "solid", RGBA, 1)
        .unwrap();
    let count_pso = device
        .compute_pipeline_from_library(&library, "produce_count")
        .unwrap();
    let hits_pso = device
        .compute_pipeline_from_library(&library, "record_hits")
        .unwrap();
    let mut colors = [ColorDesc::default(); 8];
    colors[0] = ColorDesc {
        pixel_format: RGBA.0,
        write_mask: 15,
        ..Default::default()
    };
    let mesh_pso = device
        .mesh_render_pipeline_desc(
            (&library, "test_mesh"),
            Some((&library, "solid")),
            &RenderPipelineDesc {
                colors,
                depth_format: 0,
                stencil_format: 0,
                sample_count: 1,
                alpha_to_coverage: false,
                topology_class: 0,
            },
        )
        .unwrap();

    let requests = [
        0xfabc_abcd_u32,
        0xfabc_abcd,
        0xfabc_abcd,
        3,
        1,
        3,
        1,
        0xaabb_ccdd,
        0xaabb_ccdd,
        0,
        1,
        6,
        2,
        0xaabb_ccdd,
        0xaabb_ccdd,
        3,
        1,
        9,
        3,
        0xaabb_ccdd,
        0xaabb_ccdd,
    ];
    let indexed_request = [0xaabb_u32, 3, 1, 4, 0, 2, 99, 1, 1, (-5_i32) as u32, 4];
    let dispatch_requests = [99_u32, 2, 2, 1, 88, 2, 0, 1, 77, 3, 1, 1, 66, 1, 3, 1];
    let mesh_requests = [2_u32, 1, 1, 333, 0, 1, 1, 444, 3, 1, 1];
    let draw_src = Shared::new(&device, &requests);
    let count = Shared::new(&device, &[0xdeaf_u32, 3, 99, 2, 3]);
    let indexed_src = Shared::new(&device, &indexed_request);
    let indices = Shared::new(&device, &[0xcccc_u16, 0xcccc, 5, 6, 7]);
    let dispatch_src = Shared::new(&device, &dispatch_requests);
    let mesh_src = Shared::new(&device, &mesh_requests);
    let compute_hits = Shared::new(&device, &[0_u32; 9]);
    let mesh_hits = Shared::new(&device, &[0_u32; 3]);
    let readback = Shared::new(&device, &[u32::MAX; 15 * 5]);
    let params_readback = Shared::new(&device, &[u32::MAX; 8 * 5]);
    let zero = Shared::new(&device, &[0_u32]);
    let render_target = target(&device);
    let mesh_target = target(&device);

    let cb = queue.command_buffer().unwrap();
    let source = IndirectSource { dispatch_segments: None,
        kind: IndirectKind::Draw,
        records: (&draw_src.0, 12),
        stride: 24,
        max_count: 3,
        count: Some((&count.0, 4)),
    };
    assert!(
        translator
            .encode(
                &device,
                &cb,
                &IndirectSource { dispatch_segments: None,
                    max_count: 0,
                    ..source
                }
            )
            .unwrap()
            .is_none()
    );
    assert!(
        translator
            .encode(
                &device,
                &cb,
                &IndirectSource { dispatch_segments: None,
                    stride: 23,
                    ..source
                }
            )
            .is_err()
    );
    assert!(
        translator
            .encode(
                &device,
                &cb,
                &IndirectSource { dispatch_segments: None,
                    records: (&draw_src.0, 13),
                    ..source
                }
            )
            .is_err()
    );
    assert!(
        translator
            .encode(
                &device,
                &cb,
                &IndirectSource { dispatch_segments: None,
                    max_count: 4,
                    ..source
                }
            )
            .is_err()
    );
    assert!(
        translator
            .encode(
                &device,
                &cb,
                &IndirectSource { dispatch_segments: None,
                    count: Some((&count.0, 20)),
                    ..source
                }
            )
            .is_err()
    );

    // The CPU writes count=3, the GPU producer changes it to 2 before translation.
    cb.compute(&[
        Compute::Pipeline {
            pipeline: &count_pso,
            threads_per_group: [1, 1, 1],
        },
        Compute::Buffer {
            buffer: &count.0,
            offset: 0,
            index: 0,
        },
        Compute::Dispatch { groups: [1, 1, 1] },
    ])
    .unwrap();
    let draw = translator.encode(&device, &cb, &source).unwrap().unwrap();
    assert_eq!(draw.record_count(), 3);
    assert!(draw.record(3).is_none());
    let zero_draw = translator
        .encode(
            &device,
            &cb,
            &IndirectSource { dispatch_segments: None,
                max_count: 1,
                count: Some((&zero.0, 0)),
                ..source
            },
        )
        .unwrap()
        .unwrap();
    let indexed_source = IndirectSource { dispatch_segments: None,
        kind: IndirectKind::DrawIndexed {
            index_capacity: 4,
            index_type: 0,
        },
        records: (&indexed_src.0, 4),
        stride: 20,
        max_count: 2,
        count: None,
    };
    let indexed = translator
        .encode(&device, &cb, &indexed_source)
        .unwrap()
        .unwrap();
    let indexed_bounded = translator
        .encode(
            &device,
            &cb,
            &IndirectSource { dispatch_segments: None,
                count: Some((&count.0, 8)),
                ..indexed_source // min(2, 99)
            },
        )
        .unwrap()
        .unwrap();
    let dispatch = translator
        .encode(
            &device,
            &cb,
            &IndirectSource { dispatch_segments: None,
                kind: IndirectKind::Dispatch,
                records: (&dispatch_src.0, 4),
                stride: 16,
                max_count: 4,
                count: Some((&count.0, 16)),
            },
        )
        .unwrap()
        .unwrap();
    let mesh = translator
        .encode(
            &device,
            &cb,
            &IndirectSource { dispatch_segments: None,
                kind: IndirectKind::DrawMesh,
                records: (&mesh_src.0, 0),
                stride: 16,
                max_count: 3,
                count: Some((&count.0, 12)), // cuts the nonempty third mesh record
            },
        )
        .unwrap()
        .unwrap();

    let draw_steps = |pipeline| {
        let mut steps = vec![Render::Pipeline { pipeline }];
        for i in 0..draw.record_count() {
            let record = draw.record(i).unwrap();
            steps.extend(record.msc_vertex_bindings().unwrap());
            steps.push(Render::DrawIndirect {
                primitive_type: TRIANGLE,
                arguments: record.buffer,
                offset: record.offset,
            });
        }
        for i in 0..indexed.record_count() {
            let record = indexed.record(i).unwrap();
            steps.extend(record.msc_vertex_bindings().unwrap());
            steps.push(Render::DrawIndexedIndirect {
                primitive_type: TRIANGLE,
                index_type: 0,
                index_buffer: &indices.0,
                index_offset: 2,
                arguments: record.buffer,
                offset: record.offset,
            });
        }
        steps
    };
    cb.render_pass(&[color(&render_target)], W, H, &draw_steps(&draw_pso))
        .unwrap();

    let mut compute_steps = vec![
        Compute::Pipeline {
            pipeline: &hits_pso,
            threads_per_group: [1, 1, 1],
        },
        Compute::Buffer {
            buffer: &compute_hits.0,
            offset: 0,
            index: 0,
        },
    ];
    for i in 0..dispatch.record_count() {
        let record = dispatch.record(i).unwrap();
        assert!(record.msc_vertex_bindings().is_none());
        compute_steps.push(Compute::DispatchIndirect {
            arguments: record.buffer,
            offset: record.offset,
        });
    }
    cb.compute(&compute_steps).unwrap();

    let mut mesh_steps = vec![
        Render::Pipeline {
            pipeline: &mesh_pso,
        },
        Render::MeshBuffer {
            buffer: &mesh_hits.0,
            offset: 0,
            index: 0,
        },
    ];
    for i in 0..mesh.record_count() {
        let record = mesh.record(i).unwrap();
        mesh_steps.push(Render::DrawMeshIndirect {
            arguments: record.buffer,
            offset: record.offset,
            object_threads: [1, 1, 1],
            mesh_threads: [1, 1, 1],
        });
    }
    cb.render_pass(&[color(&mesh_target)], W, H, &mesh_steps)
        .unwrap();

    let mut blit = cb.blit_encoder().unwrap();
    let mut dst = 0;
    let mut params_dst = 0;
    for batch in [
        &draw,
        &indexed,
        &indexed_bounded,
        &zero_draw,
        &dispatch,
        &mesh,
    ] {
        let size = u64::from(batch.record_count()) * RECORD_BYTES;
        blit.copy_buffer(&batch.output, 0, &readback.0, dst, size);
        dst += size;
        if let Some(params) = &batch.draw_params {
            let size = u64::from(batch.record_count()) * DRAW_PARAMS_BYTES;
            assert_ne!(params.handle(), batch.output.handle());
            blit.copy_buffer(params, 0, &params_readback.0, params_dst, size);
            params_dst += size;
        }
    }
    assert_eq!(dst, readback.0.len);
    assert_eq!(params_dst, params_readback.0.len);
    blit.finish();
    cb.commit();
    cb.wait().unwrap();

    let expected = [
        [3, 1, 3, 1, 0],
        [0, 1, 6, 2, 0],
        [0; 5],
        [0; 5],
        [3, 1, 1, (-5_i32) as u32, 4],
        [0; 5],
        [3, 1, 1, (-5_i32) as u32, 4],
        [0; 5],
        [2, 2, 1, 0, 0],
        [2, 0, 1, 0, 0],
        [3, 1, 1, 0, 0],
        [0; 5],
        [2, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0; 5],
    ];
    assert_eq!(readback.words(), expected.as_flattened());
    assert_eq!(params_readback.words(), expected[..8].as_flattened());
    assert_eq!(draw_src.words(), requests);
    assert_eq!(indexed_src.words(), indexed_request);
    assert_eq!(dispatch_src.words(), dispatch_requests);
    assert_eq!(mesh_src.words(), mesh_requests);
    assert_eq!(count.words(), [0xdeaf, 2, 99, 2, 3]);
    assert_eq!(compute_hits.words(), [2, 2, 1, 0, 1, 1, 0, 0, 0]);
    assert_eq!(mesh_hits.words(), [1, 1, 0]);
    println!(
        "{} translated records, all {dst} argument and {params_dst} parameter bytes exact; compute hits {:?}; mesh hits {:?}",
        expected.len(),
        compute_hits.words(),
        mesh_hits.words()
    );

    let pixels = read_texture(&render_target);
    let mesh_pixels = read_texture(&mesh_target);
    let at = |p: &[[u8; 4]], x| p[16 * W as usize + x];
    let blue = [0, 0, 255, 255];
    let red = [255, 0, 0, 255];
    let green = [0, 255, 0, 255];
    assert_eq!(
        [
            at(&pixels, 8),
            at(&pixels, 24),
            at(&pixels, 40),
            at(&pixels, 56)
        ],
        [red, blue, blue, green]
    );
    assert_eq!(
        [
            at(&mesh_pixels, 16),
            at(&mesh_pixels, 32),
            at(&mesh_pixels, 48)
        ],
        [red, blue, red]
    );
    for (i, p) in pixels.iter().enumerate() {
        let x = (i % W as usize) as u32;
        assert!(
            *p == blue || (x < 16 && *p == red) || (x >= 48 && *p == green),
            "draw pixel {i}: {p:?}"
        );
    }
    println!("draw centers red/blue/blue/green; mesh centers red/blue/red");

    if let Some(dir) = std::env::var_os("F42_SHADER_DIR") {
        let dir = std::path::Path::new(&dir);
        let vs = msc::convert(
            &std::fs::read(dir.join("vs.dxil")).unwrap(),
            Stage::Vertex,
            &[],
            &[],
        )
        .unwrap();
        let ps = msc::convert(
            &std::fs::read(dir.join("ps.dxil")).unwrap(),
            Stage::Fragment,
            &[],
            &[],
        )
        .unwrap();
        let pso = device
            .render_pipeline(
                &device.library(&vs.metallib).unwrap(),
                &vs.entry,
                &device.library(&ps.metallib).unwrap(),
                &ps.entry,
                RGBA,
                1,
            )
            .unwrap();
        let rt_indirect = target(&device);
        let rt_direct = target(&device);
        let cb = queue.command_buffer().unwrap();
        let viewports = std::array::from_fn::<_, 4, _>(|i| sys::WMTViewport {
            originX: (i * 16) as f64,
            originY: 0.0,
            width: 16.0,
            height: H as f64,
            znear: 0.0,
            zfar: 1.0,
        });
        let mut indirect_steps = vec![Render::Pipeline { pipeline: &pso }];
        for i in 0..draw.record_count() {
            let record = draw.record(i).unwrap();
            indirect_steps.push(Render::Viewports {
                viewports: &viewports[i as usize..=i as usize],
            });
            indirect_steps.extend(record.msc_vertex_bindings().unwrap());
            indirect_steps.push(Render::DrawIndirect {
                primitive_type: TRIANGLE,
                arguments: record.buffer,
                offset: record.offset,
            });
        }
        let indexed_record = indexed.record(1).unwrap();
        indirect_steps.push(Render::Viewports {
            viewports: &viewports[3..],
        });
        indirect_steps.extend(indexed_record.msc_vertex_bindings().unwrap());
        indirect_steps.push(Render::DrawIndexedIndirect {
            primitive_type: TRIANGLE,
            index_type: 0,
            index_buffer: &indices.0,
            index_offset: 2,
            arguments: indexed_record.buffer,
            offset: indexed_record.offset,
        });
        cb.render_pass(&[color(&rt_indirect)], W, H, &indirect_steps)
            .unwrap();
        let direct_params = Shared::new(&device, &[3_u32, 1, 3, 1, 0, 3, 1, 4, (-5_i32) as u32, 4]);
        let direct_mode = Shared::new(&device, &[0_u32, 1]);
        let mut direct_steps = vec![
            Render::Pipeline { pipeline: &pso },
            Render::Viewports {
                viewports: &viewports[..1],
            },
            Render::VertexBuffer {
                buffer: &direct_params.0,
                offset: 0,
                index: 4,
            },
            Render::VertexBuffer {
                buffer: &direct_mode.0,
                offset: 0,
                index: 5,
            },
        ];
        direct_steps.push(Render::Draw {
            primitive_type: TRIANGLE,
            vertex_start: 3,
            vertex_count: 3,
            instance_count: 1,
            base_instance: 1,
        });
        direct_steps.push(Render::Viewports {
            viewports: &viewports[3..],
        });
        direct_steps.push(Render::VertexBuffer {
            buffer: &direct_params.0,
            offset: 20,
            index: 4,
        });
        direct_steps.push(Render::VertexBuffer {
            buffer: &direct_mode.0,
            offset: 4,
            index: 5,
        });
        direct_steps.push(Render::DrawIndexed {
            primitive_type: TRIANGLE,
            index_type: 0,
            index_count: 3,
            index_buffer: &indices.0,
            index_offset: 4,
            instance_count: 1,
            base_vertex: -5,
            base_instance: 4,
        });
        cb.render_pass(&[color(&rt_direct)], W, H, &direct_steps)
            .unwrap();
        cb.commit();
        cb.wait().unwrap();
        let actual = read_texture(&rt_indirect);
        let reference = read_texture(&rt_direct);
        assert_ne!(at(&reference, 8), blue);
        assert_ne!(at(&reference, 56), blue);
        assert_eq!([at(&actual, 24), at(&actual, 40)], [blue, blue]);
        let mismatched = actual
            .iter()
            .zip(&reference)
            .filter(|(a, b)| a != b)
            .count();
    assert_eq!(mismatched, 0, "MSC indirect and direct draw pixels differ");
    println!(
        "MSC indirect and direct draws matched all {} pixels; centers {:?}",
        actual.len(),
        [
            at(&actual, 8),
            at(&actual, 24),
            at(&actual, 40),
            at(&actual, 56)
        ]
    );
    }
}

#[test]
fn clamps_oversized_dispatch_groups() {
    let device = Device::first().unwrap();
    let queue = device.new_queue(1).unwrap();
    let translator = IndirectTranslator::new(&device).unwrap();
    // Record 0: normal; Record 1: X > 65535 -> clamped to 65535;
    // Record 2: product > 1 << 20 -> zeroed; Record 3: product == 1 << 20 -> preserved.
    let src = Shared::new(
        &device,
        &[
            4_u32, 3, 2,
            70_000, 1, 1,
            2048, 1024, 1,
            1024, 1024, 1,
        ],
    );
    let readback = Shared::new(&device, &[u32::MAX; 4 * 5]);
    let cb = queue.command_buffer().unwrap();
    let batch = translator
        .encode(
            &device,
            &cb,
            &IndirectSource { dispatch_segments: None,
                kind: IndirectKind::Dispatch,
                records: (&src.0, 0),
                stride: 12,
                max_count: 4,
                count: None,
            },
        )
        .unwrap()
        .unwrap();
    let mut blit = cb.blit_encoder().unwrap();
    blit.copy_buffer(&batch.output, 0, &readback.0, 0, 4 * RECORD_BYTES);
    blit.finish();
    cb.commit();
    cb.wait().unwrap();
    let expected = [
        4, 3, 2, 0, 0,
        65535, 1, 1, 0, 0,
        0, 0, 0, 0, 0,
        1024, 1024, 1, 0, 0,
    ];
    assert_eq!(readback.words(), &expected);
}

// Only translate two constant records and blit 40 bytes. Never execute the
// translated draws/dispatches, even if the translator itself regresses.
fn check_work_clamp<const N: usize>(kind: IndirectKind, input: [u32; N], expected: [u32; 10]) {
    let _pool = Pool::new();
    let device = Device::first().unwrap();
    let queue = device.new_queue(1).unwrap();
    let translator = IndirectTranslator::new(&device).unwrap();
    let counter = ClampCounter::new(&device).unwrap();
    let src = Shared::new(&device, &input);
    let dst = Shared::new(&device, &[u32::MAX; 10]);
    let cb = queue.command_buffer().unwrap();
    let source = IndirectSource { dispatch_segments: None, kind, records: (&src.0, 0), stride: (N * 2) as u32, max_count: 2, count: None };
    let batch = translator.encode_bounded(&device, &cb, &source, &counter).unwrap().unwrap();
    let mut blit = cb.blit_encoder().unwrap();
    blit.copy_buffer(&batch.output, 0, &dst.0, 0, dst.0.len);
    blit.finish();
    cb.commit();
    cb.wait().unwrap();
    assert_eq!(dst.words(), expected);
    assert_eq!(src.words(), input);
    assert_eq!(unsafe { counter.completed_count() }, 1);
}

#[test]
fn counted_dispatch_clamp() {
    check_work_clamp(IndirectKind::Dispatch, [9, 8, 2, 0, 2, 2, 1, 0], [9, 3, 1, 0, 0, 2, 2, 1, 0, 0]);
}

#[test]
fn trace_counted_dispatch_before_and_after_clamp() {
    let _pool = Pool::new();
    let device = Device::first().unwrap();
    let queue = device.new_queue(1).unwrap();
    let translator = IndirectTranslator::new(&device).unwrap();
    let counter = ClampCounter::new(&device).unwrap();
    let args = Shared::new(&device, &[9, 8, 2, 2, 2, 1]);
    let count = Shared::new(&device, &[1]);
    let cb = queue.command_buffer().unwrap();
    let source = IndirectSource { dispatch_segments: None,
        kind: IndirectKind::Dispatch,
        records: (&args.0, 0),
        stride: 12,
        max_count: 2,
        count: Some((&count.0, 0)),
    };
    let mut batch = translator.encode_bounded_with_trace(&device, &cb, &source, &counter, true).unwrap().unwrap();
    let trace = batch.take_trace().unwrap();
    cb.commit();
    cb.wait().unwrap();
    let records: Vec<_> = unsafe { trace.completed_records().collect() };
    assert_eq!(records.len(), 2);
    assert_eq!((records[0].raw_count, records[0].active), (1, true));
    assert_eq!(records[0].before, [9, 8, 2, 0, 0]);
    assert_eq!(records[0].after, [9, 3, 1, 0, 0]);
    assert_eq!((records[1].raw_count, records[1].active), (1, false));
    assert_eq!(records[1].before, [0; 5]);
    assert_eq!(records[1].after, [0; 5]);
    assert_eq!(unsafe { counter.completed_count() }, 1);
}

#[test]
fn counted_draw_clamp() {
    check_work_clamp(IndirectKind::Draw, [300, 4, 7, 9, 3, 1, 0, 2], [128, 4, 7, 9, 0, 3, 1, 0, 2, 0]);
}

#[test]
fn counted_index_clamp() {
    check_work_clamp(IndirectKind::DrawIndexed { index_capacity: 10, index_type: 1 },
        [18, 2, 3, u32::MAX, 4, 3, 1, 0, 0, 0], [7, 2, 3, u32::MAX, 4, 3, 1, 0, 0, 0]);
}

#[test]
fn counted_mesh_clamp() {
    check_work_clamp(IndirectKind::TitleMesh1D, [48, 1, 1, 4, 1, 1], [32, 1, 1, 0, 0, 4, 1, 1, 0, 0]);
}

/// A bounded rectangle record is cut into `RECT_SEGMENTS` draws of at most
/// 32 object groups, in vertex order; only what no segment covers is clamped.
#[test]
fn bounded_rectangles_are_segmented_in_vertex_order() {
    let _pool = Pool::new();
    let device = Device::first().unwrap();
    let queue = device.new_queue(1).unwrap();
    let translator = IndirectTranslator::new(&device).unwrap();
    let cfg = crate::msc::GeometryConfig { vertex_output_size: 32, max_input_primitives: 32, max_payload_size: 16256 };
    let stride = cfg.object_vertex_stride().unwrap();
    assert_eq!(stride, 255);
    let segs = RECT_SEGMENTS as usize;
    // Per segment with 4 instances: (32 / 4) groups * 255 vertices = 2040.
    let chunk = 2040u32;
    let cases: [([u32; 4], u32); 3] = [
        // Fits one segment: drawn whole, not clamped.
        ([300, 4, 7, 9], 0),
        // Needs 5 segments: drawn whole, not clamped.
        ([4 * chunk + 30, 4, 7, 9], 0),
        // More than all segments cover: the rest is clamped (counted once).
        ([RECT_SEGMENTS * chunk + 600, 4, 7, 9], 1),
    ];
    for (input, clamped) in cases {
        let counter = ClampCounter::new(&device).unwrap();
        let src = Shared::new(&device, &input);
        let out = Shared::new(&device, &vec![u32::MAX; segs * 5]);
        let params = Shared::new(&device, &vec![u32::MAX; segs * 12]);
        let cb = queue.command_buffer().unwrap();
        let source = IndirectSource { dispatch_segments: None, kind: IndirectKind::Rectangles(cfg), records: (&src.0, 0), stride: 16, max_count: 1, count: None };
        let batch = translator.encode_bounded(&device, &cb, &source, &counter).unwrap().unwrap();
        assert_eq!(batch.record_count(), RECT_SEGMENTS);
        let mut blit = cb.blit_encoder().unwrap();
        blit.copy_buffer(&batch.output, 0, &out.0, 0, segs as u64 * RECORD_BYTES);
        blit.copy_buffer(batch.draw_params.as_ref().unwrap(), 0, &params.0, 0, segs as u64 * 48);
        blit.finish();
        cb.commit();
        cb.wait().unwrap();
        let (out, params) = (out.words(), params.words());
        let total = input[0] - input[0] % 3;
        let mut drawn = 0u32;
        for seg in 0..segs {
            let first = seg as u32 * chunk;
            let n = if first < total { (total - first).min(chunk) } else { 0 };
            let groups = n.div_ceil(stride);
            let o = &out[seg * 5..seg * 5 + 5];
            let p = &params[seg * 12..seg * 12 + 4];
            if n == 0 {
                assert_eq!(o, [0; 5], "segment {seg} of {input:?}");
                assert_eq!(&p[..2], [0, 0], "segment {seg} of {input:?}");
                continue;
            }
            assert!(groups * input[1] <= 32, "segment {seg}: {groups} groups x {} instances", input[1]);
            assert_eq!(o, [groups, input[1], 1, 0, 0], "segment {seg} of {input:?}");
            // Vertex count, instances, first vertex, first instance.
            assert_eq!(p, [n, input[1], input[2] + first, input[3]], "segment {seg} of {input:?}");
            drawn += n;
        }
        assert_eq!(drawn, total.min(RECT_SEGMENTS * chunk));
        assert_eq!(unsafe { counter.completed_count() }, clamped, "{input:?}");
    }
}

#[test]
fn translates_and_clamps_indirect_rectangles() {
    let device = Device::first().unwrap();
    let queue = device.new_queue(1).unwrap();
    let translator = IndirectTranslator::new(&device).unwrap();
    let cfg = crate::msc::GeometryConfig {
        vertex_output_size: 32,
        max_input_primitives: 32,
        max_payload_size: 16256,
    };
    let stride = cfg.object_vertex_stride().unwrap();
    assert_eq!(stride, 255);
    let src = Shared::new(
        &device,
        &[
            9_u32, 2, 3, 1,
            2, 5, 0, 0,
            0x00ff_ffff, 100, 0, 0,
            65536 * stride, 1, 0, 0,
        ],
    );
    let out_readback = Shared::new(&device, &[u32::MAX; 4 * 5]);
    let params_readback = Shared::new(&device, &[u32::MAX; 4 * 12]);
    let cb = queue.command_buffer().unwrap();
    let batch = translator
        .encode(
            &device,
            &cb,
            &IndirectSource { dispatch_segments: None,
                kind: IndirectKind::Rectangles(cfg),
                records: (&src.0, 0),
                stride: 16,
                max_count: 4,
                count: None,
            },
        )
        .unwrap()
        .unwrap();
    let mut blit = cb.blit_encoder().unwrap();
    blit.copy_buffer(&batch.output, 0, &out_readback.0, 0, 4 * RECORD_BYTES);
    blit.copy_buffer(batch.draw_params.as_ref().unwrap(), 0, &params_readback.0, 0, 4 * 48);
    blit.finish();
    cb.commit();
    cb.wait().unwrap();

    let expected_out = [
        1, 2, 1, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    ];
    assert_eq!(out_readback.words(), &expected_out);

    let params = params_readback.words();
    assert_eq!(
        &params[0..12],
        &[
            9, 2, 3, 1, 0, 0,
            0x0303_0000,
            cfg.max_input_primitives | (stride << 16),
            cfg.max_input_primitives | (1 << 16),
            (stride / 3) | (3 << 16),
            0, 0,
        ]
    );
    for rec in 1..4 {
        assert_eq!(&params[rec * 12..rec * 12 + 2], &[0, 0]);
    }
}

/// A bounded dispatch of a program that adds a group base, cut in segments
/// with the base of each set as the host does, writes exactly what the
/// whole dispatch writes, also past 65,535 groups (no host axis takes more):
/// every group writes `group * 7 + thread + 1` at its own place.
#[test]
fn segmented_dispatch_writes_what_the_whole_dispatch_writes() {
    let _pool = Pool::new();
    let device = Device::first().unwrap();
    let queue = device.new_queue(4).unwrap();
    let translator = IndirectTranslator::new(&device).unwrap();
    const THREADS: u32 = 2;
    let pipeline = device.compute_pipeline_msl(
        "#include <metal_stdlib>\nusing namespace metal;\nkernel void k(device uint *out [[buffer(0)]], constant uint *base [[buffer(1)]],\n uint3 group [[threadgroup_position_in_grid]], uint3 t [[thread_position_in_threadgroup]]) {\n uint g = base[0] + group.x; out[g * 2 + t.x] = g * 7 + t.x + 1; }",
        "k",
    ).unwrap();
    let run = |total: u32, segments: Option<DispatchSegments>| -> (Vec<u32>, u32) {
        let out = Shared::new(&device, &vec![0u32; 80_000 * THREADS as usize]);
        let src = Shared::new(&device, &[total, 1, 1]);
        let counter = ClampCounter::new(&device).unwrap();
        let cb = queue.command_buffer().unwrap();
        let source = IndirectSource { dispatch_segments: segments, kind: IndirectKind::Dispatch, records: (&src.0, 0), stride: 12, max_count: 1, count: None };
        let batch = translator.encode_bounded(&device, &cb, &source, &counter).unwrap().unwrap();
        let bases: Vec<[u32; 1]> = (0..batch.record_count()).map(|r| [segments.map_or(0, |d| r * d.groups)]).collect();
        let base_bytes: Vec<Vec<u8>> = bases.iter().map(|b| b[0].to_le_bytes().to_vec()).collect();
        let mut steps = Vec::new();
        for r in 0..batch.record_count() {
            let rec = batch.record(r).unwrap();
            steps.push(Compute::Pipeline { pipeline: &pipeline, threads_per_group: [u64::from(THREADS), 1, 1] });
            steps.push(Compute::Buffer { buffer: &out.0, offset: 0, index: 0 });
            steps.push(Compute::Bytes { bytes: &base_bytes[r as usize], index: 1 });
            steps.push(Compute::DispatchIndirect { arguments: rec.buffer, offset: rec.offset });
        }
        cb.compute(&steps).unwrap();
        cb.commit();
        cb.wait().unwrap();
        (out.words(), unsafe { counter.completed_count() })
    };
    let expect = |total: u32| -> Vec<u32> {
        let mut v = vec![0u32; 80_000 * THREADS as usize];
        for g in 0..total {
            for t in 0..THREADS {
                v[(g * THREADS + t) as usize] = g * 7 + t + 1;
            }
        }
        v
    };
    let seg = |segments, groups| Some(DispatchSegments { segments, groups });
    // Within the segments: all groups, nothing clamped.
    let (got, clamped) = run(70_000, seg(4, 32_768));
    assert_eq!(clamped, 0);
    assert!(got == expect(70_000), "70,000 groups in 32,768-group segments");
    let (got, clamped) = run(1_000, seg(16, 64));
    assert_eq!(clamped, 0);
    assert!(got == expect(1_000), "1,000 groups in 64-group segments");
    // Past all segments: the rest clamped, counted once.
    let (got, clamped) = run(1_000, seg(4, 64));
    assert_eq!(clamped, 1);
    assert!(got == expect(256), "4 x 64 groups drawn");
    // Not segmented: the group budget, as before.
    let (got, clamped) = run(1_000, None);
    assert_eq!(clamped, 1);
    assert!(got == expect(GROUPS_PER_RECORD), "the group budget only");
}

/// Multi-row dispatches (y or z over 1) are not segmented: the first
/// segment gets the old clamp, the others nothing.
#[test]
fn multi_row_dispatches_keep_the_old_clamp_when_segmented() {
    let _pool = Pool::new();
    let device = Device::first().unwrap();
    let queue = device.new_queue(1).unwrap();
    let translator = IndirectTranslator::new(&device).unwrap();
    let counter = ClampCounter::new(&device).unwrap();
    let src = Shared::new(&device, &[9u32, 8, 2]);
    let out = Shared::new(&device, &[u32::MAX; 4 * 5]);
    let cb = queue.command_buffer().unwrap();
    let source = IndirectSource {
        dispatch_segments: Some(DispatchSegments { segments: 4, groups: 1024 }),
        kind: IndirectKind::Dispatch, records: (&src.0, 0), stride: 12, max_count: 1, count: None,
    };
    let batch = translator.encode_bounded(&device, &cb, &source, &counter).unwrap().unwrap();
    assert_eq!(batch.record_count(), 4);
    let mut blit = cb.blit_encoder().unwrap();
    blit.copy_buffer(&batch.output, 0, &out.0, 0, 4 * RECORD_BYTES);
    blit.finish();
    cb.commit();
    cb.wait().unwrap();
    let w = out.words();
    assert_eq!(&w[0..5], &[9, 3, 1, 0, 0], "the old clamp: 9 x 3 x 1 <= 32 groups");
    assert!(w[5..].iter().all(|&x| x == 0));
    assert_eq!(unsafe { counter.completed_count() }, 1);
}
