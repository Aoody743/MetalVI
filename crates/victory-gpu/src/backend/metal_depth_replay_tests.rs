use super::*;
use crate::backend::metal_draw::{
    DrawKind, MetalClear, MetalDraw, MetalGraphics, MetalTexCopy, Targets,
};
use crate::backend::metal_texture::MetalTexture;
use victory_metal::render::{DepthStencilDesc, RenderPipelineDesc};
use victory_metal::sys::{self, WMTPixelFormat as P};
use victory_metal::texture::TextureDescriptor;

#[test]
fn companion_draw_clear_and_copy_replay() {
    let _pool = victory_metal::Pool::new();
    let backend = MetalBackend::new().unwrap();
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
        let t = Arc::new(MetalTexture {
            companion: Default::default(),
            twin: None,
            texture,
            pixel: P::WMTPixelFormatR32Float,
            desc,
            mips: 1,
            slices: 1,
        });
        backend
            .textures
            .lock()
            .unwrap()
            .insert(key, Some(t.clone()));
        t
    };
    let original = make_texture(1, 39);
    let input = make_texture(2, 41);
    let output = make_texture(3, 41);
    let cleared = make_texture(4, 41);
    let input_attachment = backend
        .attachment(2 as crate::object::This, 41, 0, 0, false)
        .unwrap();
    assert!(original.companion.get().is_none());
    assert!(
        backend
            .attachment(1 as crate::object::This, 55, 0, 0, true)
            .is_none()
    );
    let attachment = backend
        .attachment(1 as crate::object::This, 40, 0, 0, true)
        .unwrap();
    assert!(attachment.companion.is_some());
    let same = backend
        .attachment(1 as crate::object::This, 40, 0, 0, true)
        .unwrap();
    assert_eq!(attachment.texture.handle(), same.texture.handle());

    let lib = backend
        .device
        .library_msl(
            r#"
#include <metal_stdlib>
using namespace metal;
vertex float4 vs(uint id [[vertex_id]]) {
    return float4(id == 1 ? 3 : -1, id == 2 ? 3 : -1, 0.5, 1);
}
"#,
        )
        .unwrap();
    let desc = RenderPipelineDesc {
        depth_format: P::WMTPixelFormatDepth32Float.0,
        sample_count: 1,
        ..Default::default()
    };
    let pso = backend
        .device
        .render_pipeline_desc((&lib, "vs"), None, &desc)
        .unwrap();
    let dss = backend
        .device
        .depth_stencil_state(&DepthStencilDesc {
            depth_write: true,
            depth_compare: 1,
            ..Default::default()
        })
        .unwrap();
    let pso = Arc::new(MetalGraphics {
        cost_ns: 1.into(),
        name: "F76 synthetic depth draw".into(),
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
    let mut targets = Targets {
        depth: Some(attachment.clone()),
        ..Default::default()
    };
    targets.finish();
    let draw = MetalDraw {
        trace_inputs: Vec::new(),
        pso,
        targets: Arc::new(targets),
        stages: Vec::new(),
        vertex: None,
        viewport: sys::WMTViewport {
            originX: 0.0,
            originY: 0.0,
            width: 4.0,
            height: 4.0,
            znear: 0.0,
            zfar: 1.0,
        },
        scissor: sys::WMTScissorRect {
            x: 0,
            y: 0,
            width: 4,
            height: 4,
        },
        blend: [0.0; 4],
        stencil_ref: 0,
        primitive: 3,
        kind: DrawKind::Vertices {
            count: 3,
            instances: 1,
            first: 0,
            first_instance: 0,
        },
    };
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
    let clear = |target, color, depth| {
        Arc::new(MetalOp::Clear(MetalClear {
            target,
            color,
            depth,
            stencil: None,
            has_stencil: false,
            rects: Vec::new(),
        }))
    };
    let splits = || super::work::SPLITS.load(std::sync::atomic::Ordering::Relaxed);
    let before = splits();
    let draw = Arc::new(MetalOp::Draw(draw));
    assert!(backend.run(&[
        clear(attachment.clone(), None, Some(0.125)),
        copy(&original, &cleared),
        clear(input_attachment, Some([0.75, 0.0, 0.0, 0.0]), None),
        copy(&input, &original),
        draw.clone(),
        draw,
        copy(&original, &output),
        Arc::new(MetalOp::Clear(MetalClear {
            target: attachment.clone(),
            color: None,
            depth: Some(0.25),
            stencil: None,
            has_stencil: false,
            rects: vec![victory_metal::clear_resolve::ClearRect {
                left: 0,
                top: 0,
                right: 2,
                bottom: 2
            }],
        })),
        copy(&original, &input),
    ]));
    if std::env::var("VICTORY_METAL_PASS_DISPATCHES").as_deref() == Ok("1") {
        assert_eq!(splits() - before, 8, "each of the nine operations needs its own buffer");
    }

    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    let mem = unsafe {
        VirtualAlloc(
            std::ptr::null(),
            4096,
            MEM_RESERVE | MEM_COMMIT,
            PAGE_READWRITE,
        )
    };
    assert!(!mem.is_null());
    let out = unsafe { backend.device.wrap(mem, 4096) }.unwrap();
    let cb = backend.queue.command_buffer().unwrap();
    let mut blit = cb.blit_encoder().unwrap();
    for (texture, offset) in [
        (&output.texture, 0),
        (&attachment.texture, 1024),
        (&input.texture, 2048),
        (&cleared.texture, 3072),
    ] {
        blit.copy_from_texture_to_buffer(texture, 0, 0, [0; 3], [4, 4, 1], &out, offset, 256, 1024);
    }
    blit.finish();
    cb.commit();
    cb.wait().unwrap();
    let data = unsafe { std::slice::from_raw_parts(mem.cast::<u8>(), 4096) };
    for base in [0, 1024, 2048, 3072] {
        for y in 0..4 {
            for x in 0..4 {
                let at = base + y * 256 + x * 4;
                let expected: f32 = match base {
                    3072 => 0.125,
                    1024 | 2048 if x < 2 && y < 2 => 0.25,
                    _ => 0.5,
                };
                assert_eq!(
                    &data[at..at + 4],
                    &expected.to_le_bytes(),
                    "buffer {base}, pixel {x},{y}"
                );
            }
        }
    }
    drop(cb);
    drop(out);
    unsafe { VirtualFree(mem, 0, MEM_RELEASE) };
}
