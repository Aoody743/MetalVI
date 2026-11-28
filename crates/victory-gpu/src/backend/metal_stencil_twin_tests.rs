//! A3: shaders write a depth-stencil texture's twins (material class into
//! the depth plane, a mask into the stencil plane); a draw with the depth
//! test "equal" at class × 0x37800080 and the stencil test "equal" covers
//! only the pixels of that class and mask.

use super::*;
use crate::backend::metal_depth::StencilTwin;
use crate::backend::metal_draw::{DrawKind, MetalClear, MetalDraw, MetalGraphics, Targets};
use crate::backend::metal_texture::{MetalTexture, make_texture_entry};
use victory_metal::render::{ColorDesc, DepthStencilDesc, RenderPipelineDesc, StencilFace};
use victory_metal::sys::{self, WMTPixelFormat as P};
use victory_metal::texture::{Texture, TextureDescriptor};

const SIZE: u32 = 8;
const STEP: u32 = 0x3780_0080;

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
// Class (x + y) % 4; mask 5 left of x = 4, 3 right.
kernel void classes16(texture2d<uint, access::write> plane [[texture(0)]],
                      texture2d<uint, access::write> stencil [[texture(1)]],
                      uint2 p [[thread_position_in_grid]]) {
    plane.write(uint4((p.x + p.y) % 4), p);
    stencil.write(uint4(p.x < 4 ? 5 : 3), p);
}
kernel void classes32(texture2d<float, access::write> plane [[texture(0)]],
                      texture2d<uint, access::write> stencil [[texture(1)]],
                      uint2 p [[thread_position_in_grid]]) {
    plane.write(float4(float((p.x + p.y) % 4) * as_type<float>(0x37800080u)), p);
    stencil.write(uint4(p.x < 4 ? 5 : 3), p);
}
// The title's rectangle: class 2.
vertex float4 rect(uint id [[vertex_id]]) {
    return float4(id == 1 ? 3 : -1, id == 2 ? 3 : -1, 2.0 * as_type<float>(0x37800080u), 1);
}
fragment float4 hit() { return float4(1, 0, 0, 1); }
"#;

fn texture(backend: &MetalBackend, pixel: P, usage: u64) -> Texture {
    backend
        .device
        .new_texture(&TextureDescriptor {
            pixel_format: pixel,
            width: SIZE,
            height: SIZE,
            usage: sys::WMTTextureUsage(usage as _),
            options: sys::WMTResourceOptions(0x20),
            ..Default::default()
        })
        .unwrap()
}

fn desc(format: u32, flags: u32) -> crate::resource::ResourceDesc {
    // SAFETY: ResourceDesc is an integer-only C record.
    let mut d: crate::resource::ResourceDesc = unsafe { std::mem::zeroed() };
    d.width = SIZE.into();
    d.height = SIZE;
    d.dimension = 3;
    d.format = format;
    d.flags = flags;
    d.sample_count = 1;
    d.mip_levels = 1;
    d.depth_or_array_size = 1;
    d
}

/// Reads level 0 of `t` (`bytes` a texel) back.
fn read(backend: &MetalBackend, t: &Texture, bytes: u32) -> Vec<u8> {
    let row = 256u32;
    let (buffer, ptr) = backend.device.shared_buffer_mapped(&vec![0; (row * SIZE) as usize]).unwrap();
    let cb = backend.queue.command_buffer().unwrap();
    let mut blit = cb.blit_encoder().unwrap();
    blit.copy_from_texture_to_buffer(t, 0, 0, [0; 3], [SIZE.into(), SIZE.into(), 1], &buffer, 0, row, row * SIZE);
    blit.finish();
    cb.commit();
    cb.wait().unwrap();
    // SAFETY: the shared buffer's contents, no GPU work pending.
    let all = unsafe { std::slice::from_raw_parts(ptr, (row * SIZE) as usize) };
    (0..SIZE as usize)
        .flat_map(|y| all[y * row as usize..][..(SIZE * bytes) as usize].to_vec())
        .collect()
}

fn twin_case(format: u32) {
    let _pool = victory_metal::Pool::new();
    let backend = MetalBackend::new().unwrap();
    let sixteen = (118..=120).contains(&format);
    let d = desc(format, 6);
    assert!(StencilTwin::wanted(&d, P::WMTPixelFormatDepth32Float_Stencil8));
    let depth = texture(&backend, P::WMTPixelFormatDepth32Float_Stencil8, 1 | 2 | 4 | 0x10);
    let twin = Arc::new(StencilTwin::new(&backend.device, &depth, &d, 1, 1).unwrap());
    let t = Arc::new(MetalTexture {
        texture: depth,
        pixel: P::WMTPixelFormatDepth32Float_Stencil8,
        desc: d,
        mips: 1,
        slices: 1,
        companion: Default::default(),
        twin: Some(twin.clone()),
    });
    backend.textures.lock().unwrap().insert(1, Some(t.clone()));
    let colour = Arc::new(MetalTexture {
        texture: texture(&backend, P::WMTPixelFormatR32Float, 1 | 4 | 0x10),
        pixel: P::WMTPixelFormatR32Float,
        desc: desc(41, 1),
        mips: 1,
        slices: 1,
        companion: Default::default(),
        twin: None,
    });
    backend.textures.lock().unwrap().insert(2, Some(colour.clone()));

    // Views: UAVs (format, dimension 2D, mip, plane) and SRVs (format,
    // dimension 2D, mapping, _, mips, levels, plane, min LOD) go to twins.
    let view = |w: [u32; 10], unordered: bool| {
        let (view, _) = make_texture_entry(&t, Some(&w), unordered).unwrap();
        Texture::from_object(view)
    };
    let plane_format = if sixteen { 57 } else { 41 };
    let plane = view([plane_format, 4, 0, 0, 0, 0, 0, 0, 0, 0], true);
    let stencil = view([62, 4, 0, 1, 0, 0, 0, 0, 0, 0], true);
    assert_eq!(plane.pixel_format(), if sixteen { P::WMTPixelFormatR16Uint } else { P::WMTPixelFormatR32Float });
    assert_eq!(stencil.pixel_format(), P::WMTPixelFormatR8Uint);
    let srv_depth = view([if sixteen { 46 } else { 21 }, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0], false);
    assert_eq!(srv_depth.pixel_format(), if sixteen { P::WMTPixelFormatR16Unorm } else { P::WMTPixelFormatR32Float });
    let srv_stencil = view([if sixteen { 47 } else { 22 }, 4, 0x1688, 0, 0, 1, 1, 0, 0, 0], false);
    assert_eq!(srv_stencil.pixel_format(), P::WMTPixelFormatR8Uint);

    let lib = backend.device.library_msl(SOURCE).unwrap();
    let kernel = backend
        .device
        .compute_pipeline_from_library(&lib, if sixteen { "classes16" } else { "classes32" })
        .unwrap();
    let cb = backend.queue.command_buffer().unwrap();
    cb.compute(&[
        victory_metal::Compute::Pipeline { pipeline: &kernel, threads_per_group: [8, 8, 1] },
        victory_metal::Compute::Texture { texture: &plane, index: 0 },
        victory_metal::Compute::Texture { texture: &stencil, index: 1 },
        victory_metal::Compute::DispatchThreads { threads: [SIZE.into(), SIZE.into(), 1] },
    ])
    .unwrap();
    cb.commit();
    cb.wait().unwrap();

    let ds = P::WMTPixelFormatDepth32Float_Stencil8.0;
    let mut pdesc = RenderPipelineDesc { depth_format: ds, stencil_format: ds, sample_count: 1, topology_class: 3, ..Default::default() };
    pdesc.colors[0] = ColorDesc { pixel_format: P::WMTPixelFormatR32Float.0, write_mask: 0xf, ..Default::default() };
    let pso = backend.device.render_pipeline_desc((&lib, "rect"), Some((&lib, "hit")), &pdesc).unwrap();
    let face = StencilFace { compare: 2, read_mask: 0xff, ..Default::default() };
    let dss = backend
        .device
        .depth_stencil_state(&DepthStencilDesc { depth_compare: 2, depth_write: false, stencil: true, front: face, back: face })
        .unwrap();
    let graphics = Arc::new(MetalGraphics {
        cost_ns: 1.into(),
        name: "A3 class rectangle".into(),
        pso,
        dss,
        depth_write: false,
        raster: [0; 4],
        bias: [0.0; 3],
        stages: Vec::new(),
        vertex_slots: 0,
        mesh: false,
        mesh_threads: [0; 3],
        geometry: None,
        colors: [P::WMTPixelFormatR32Float.0, 0, 0, 0, 0, 0, 0, 0],
        depth: ds,
        stencil: ds,
        samples: 1,
    });
    let depth_attachment = backend.attachment(1 as crate::object::This, 0, 0, 0, true).unwrap();
    assert!(depth_attachment.twin.is_some());
    assert_eq!(depth_attachment.texture.handle(), t.texture.handle());
    let colour_attachment = backend.attachment(2 as crate::object::This, 41, 0, 0, false).unwrap();
    let mut targets = Targets { depth: Some(depth_attachment.clone()), stencil: true, ..Default::default() };
    targets.colors[0] = Some(colour_attachment.clone());
    targets.finish();
    let draw = MetalDraw {
        trace_inputs: Vec::new(),
        pso: graphics,
        targets: Arc::new(targets),
        stages: Vec::new(),
        vertex: None,
        viewport: sys::WMTViewport { originX: 0.0, originY: 0.0, width: SIZE.into(), height: SIZE.into(), znear: 0.0, zfar: 1.0 },
        scissor: sys::WMTScissorRect { x: 0, y: 0, width: SIZE.into(), height: SIZE.into() },
        blend: [0.0; 4],
        stencil_ref: 5,
        primitive: 3,
        kind: DrawKind::Vertices { count: 3, instances: 1, first: 0, first_instance: 0 },
    };
    let clear = |target, color, depth, stencil| {
        Arc::new(MetalOp::Clear(MetalClear { target, color, depth, stencil, has_stencil: stencil.is_some(), rects: Vec::new() }))
    };
    assert!(backend.run(&[clear(colour_attachment, Some([0.0; 4]), None, None), Arc::new(MetalOp::Draw(draw))]));

    let hits = read(&backend, &colour.texture, 4);
    let classes = read(&backend, &twin.plane.texture, if sixteen { 2 } else { 4 });
    let masks = read(&backend, &twin.stencil.texture, 1);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let i = (y * SIZE + x) as usize;
            let class = (x + y) % 4;
            let expected = if class == 2 && x < 4 { 1.0f32 } else { 0.0 };
            assert_eq!(f32::from_le_bytes(hits[i * 4..][..4].try_into().unwrap()), expected, "format {format} pixel {x},{y}");
            // The pass gave the twins back what it found (round trip).
            if sixteen {
                assert_eq!(u16::from_le_bytes(classes[i * 2..][..2].try_into().unwrap()), class as u16, "class {x},{y}");
            } else {
                let bits = (class as f32 * f32::from_bits(STEP)).to_bits();
                assert_eq!(u32::from_le_bytes(classes[i * 4..][..4].try_into().unwrap()), bits, "class {x},{y}");
            }
            assert_eq!(masks[i], if x < 4 { 5 } else { 3 }, "mask {x},{y}");
        }
    }

    // A depth-stencil clear reaches the twins.
    let depth_value = 3.0 * f32::from_bits(STEP);
    assert!(backend.run(&[clear(depth_attachment, None, Some(depth_value), Some(7))]));
    let classes = read(&backend, &twin.plane.texture, if sixteen { 2 } else { 4 });
    let masks = read(&backend, &twin.stencil.texture, 1);
    for i in 0..(SIZE * SIZE) as usize {
        if sixteen {
            assert_eq!(u16::from_le_bytes(classes[i * 2..][..2].try_into().unwrap()), 3);
        } else {
            assert_eq!(u32::from_le_bytes(classes[i * 4..][..4].try_into().unwrap()), depth_value.to_bits());
        }
        assert_eq!(masks[i], 7);
    }
}

#[test]
fn stencil_twin_16_bit_classes_draw_equal() {
    twin_case(118);
}

#[test]
fn stencil_twin_32_bit_classes_draw_equal() {
    twin_case(19);
}
