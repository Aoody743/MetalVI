use super::*;
use sys::WMTPixelFormat as F;
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
};

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
vertex float4 vs(uint i [[vertex_id]]) {
    return float4(i == 1 ? 3 : -1, i == 2 ? 3 : -1, 0, 1);
}
fragment float4 draw_unorm(float4 p [[position]], uint sample [[sample_id]]) {
    return float4(float(16 + uint(p.x) * 8 + sample * 4) / 255.0,
                  float(40 + uint(p.y) * 8 + sample * 4) / 255.0, 0.0, 1.0);
}
fragment float4 draw_float(float4 p [[position]], uint sample [[sample_id]]) {
    return float4(float(uint(p.x) + uint(p.y) * 16 + sample * 4));
}
fragment uint4 draw_uint(float4 p [[position]], uint sample [[sample_id]]) {
    return uint4(1000 + uint(p.x) + uint(p.y) * 16 + sample * 400);
}
fragment int4 draw_sint(float4 p [[position]], uint sample [[sample_id]]) {
    return int4(-1000 + int(p.x) - int(p.y) * 16 + int(sample) * 400);
}
struct DepthOut { float depth [[depth(any)]]; };
fragment DepthOut draw_depth(float4 p [[position]], uint sample [[sample_id]]) {
    return { float(4 + uint(p.x) + uint(p.y) + sample * 2) / 32.0 };
}
"#;

const SENTINEL: [u8; 4] = [0x19; 4];
const SIDE: u32 = 8;
const PITCH: u32 = 256;
const IMAGE: u32 = SIDE * PITCH;

fn check(format: F, fragment: &str, expected: fn(u32, u32) -> [u8; 4]) {
    let _pool = Pool::new();
    let device = Device::first().unwrap();
    let queue = device.new_queue(4).unwrap();
    let cb = queue.command_buffer().unwrap();
    let helper = ClearResolve::new(&device).unwrap();
    let depth = format == F::WMTPixelFormatDepth32Float;
    let dst_format = if format == F::WMTPixelFormatR32Float {
        F::WMTPixelFormatR32Uint
    } else {
        format
    };
    let src = device
        .new_texture(&TextureDescriptor {
            pixel_format: format,
            width: SIDE,
            height: SIDE,
            sample_count: 4,
            type_: WMTTextureType::WMTTextureType2DMultisample,
            usage: sys::WMTTextureUsage(1 | 4 | 0x10),
            options: sys::WMTResourceOptions::WMTResourceStorageModePrivate,
            ..Default::default()
        })
        .unwrap();
    let dst = device
        .new_texture(&TextureDescriptor {
            pixel_format: dst_format,
            width: SIDE * 2,
            height: SIDE * 2,
            mipmap_level_count: 2,
            array_length: 3,
            type_: WMTTextureType::WMTTextureType2DArray,
            usage: sys::WMTTextureUsage(1 | 0x10 | if depth { 4 } else { 0 }),
            options: sys::WMTResourceOptions::WMTResourceStorageModePrivate,
            ..Default::default()
        })
        .unwrap();
    let lib = device.library_msl(SOURCE).unwrap();
    let mut pso_desc = RenderPipelineDesc {
        sample_count: 4,
        ..Default::default()
    };
    if depth {
        pso_desc.depth_format = format.0;
    } else {
        pso_desc.colors[0] = ColorDesc {
            pixel_format: format.0,
            write_mask: 0xf,
            ..Default::default()
        };
    }
    let pso = device
        .render_pipeline_desc((&lib, "vs"), Some((&lib, fragment)), &pso_desc)
        .unwrap();
    let mut pass = RenderPass {
        width: SIDE,
        height: SIDE,
        samples: 4,
        ..Default::default()
    };
    let mut steps = vec![Render::Pipeline { pipeline: &pso }];
    if depth {
        pass.depth = Some(PassDepth {
            texture: &src,
            level: 0,
            slice: 0,
            load: sys::WMTLoadAction::WMTLoadActionDontCare,
            store: sys::WMTStoreAction::WMTStoreActionStore,
            clear_depth: 1.0,
            clear_stencil: 0,
        });
        steps.push(Render::DepthStencil {
            state: &helper.resolve_depth_dss,
            stencil_ref: 0,
        });
    } else {
        pass.colors[0] = Some(PassColor {
            texture: &src,
            level: 0,
            slice: 0,
            load: sys::WMTLoadAction::WMTLoadActionDontCare,
            store: sys::WMTStoreAction::WMTStoreActionStore,
            clear: [0.0; 4],
        });
    }
    steps.push(Render::Draw {
        primitive_type: sys::WMTPrimitiveType::WMTPrimitiveTypeTriangle,
        vertex_start: 0,
        vertex_count: 3,
        instance_count: 1,
        base_instance: 0,
    });
    cb.render_pass_with(&pass, &steps).unwrap();

    let seed = device
        .shared_buffer(&[SENTINEL[0]; IMAGE as usize])
        .unwrap();
    let mut blit = cb.blit_encoder().unwrap();
    for slice in [1, 2] {
        blit.copy_from_buffer_to_texture(&seed, 0, PITCH, IMAGE, [8, 8, 1], &dst, slice, 1, [0; 3]);
    }
    blit.finish();
    let region = ResolveRegion {
        src_origin: [3, 2],
        dst_origin: [1, 4],
        size: [4, 3],
    };
    helper
        .resolve_region(&cb, &src, 0, &dst, 1, 0, format, None)
        .unwrap();
    // Resolving the same texture again must still see all of its samples.
    helper
        .resolve_region(&cb, &src, 0, &dst, 1, 1, format, Some(region))
        .unwrap();
    assert!(
        helper
            .resolve_region(&cb, &src, 0, &dst, 2, 1, format, None)
            .is_err()
    );
    assert!(
        helper
            .resolve_region(
                &cb,
                &src,
                0,
                &dst,
                1,
                1,
                format,
                Some(ResolveRegion {
                    dst_origin: [u32::MAX, 0],
                    ..region
                })
            )
            .is_err()
    );

    let len = (IMAGE * 3) as usize;
    // SAFETY: fresh committed pages, kept alive through the GPU wait and buffer drop.
    let mem = unsafe {
        VirtualAlloc(
            std::ptr::null(),
            8192,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_READWRITE,
        )
    };
    assert!(!mem.is_null());
    let out = unsafe { device.wrap(mem, 8192) }.unwrap();
    let mut blit = cb.blit_encoder().unwrap();
    for slice in 0..3 {
        blit.copy_from_texture_to_buffer(
            &dst,
            slice,
            1,
            [0; 3],
            [8, 8, 1],
            &out,
            u64::from(IMAGE * slice),
            PITCH,
            IMAGE,
        );
    }
    blit.finish();
    cb.commit();
    cb.wait().unwrap();
    let data = unsafe { std::slice::from_raw_parts(mem.cast::<u8>(), len) };
    for slice in 0..3 {
        for y in 0..SIDE {
            for x in 0..SIDE {
                let at = (slice * IMAGE + y * PITCH + x * 4) as usize;
                let value = match (slice, x, y) {
                    (0, _, _) => expected(x, y),
                    (1, 1..=4, 4..=6) => expected(x - 1 + 3, y - 4 + 2),
                    _ => SENTINEL,
                };
                assert_eq!(&data[at..at + 4], &value, "slice {slice}, texel {x},{y}");
            }
        }
    }
    drop(cb);
    drop(out);
    unsafe { VirtualFree(mem, 0, MEM_RELEASE) };
}

#[test]
fn hardware_whole_and_region_keep_msaa_source() {
    check(F::WMTPixelFormatBGRA8Unorm, "draw_unorm", |x, y| {
        [0, (46 + y * 8) as u8, (22 + x * 8) as u8, 255]
    });
}

#[test]
fn compute_averages_all_float_samples() {
    check(F::WMTPixelFormatR32Float, "draw_float", |x, y| {
        ((x + y * 16 + 6) as f32).to_le_bytes()
    });
}

#[test]
fn compute_selects_sample_zero_unsigned() {
    check(F::WMTPixelFormatR32Uint, "draw_uint", |x, y| {
        (1000 + x + y * 16).to_le_bytes()
    });
}

#[test]
fn compute_selects_sample_zero_signed() {
    check(F::WMTPixelFormatR32Sint, "draw_sint", |x, y| {
        (-1000 + x as i32 - y as i32 * 16).to_le_bytes()
    });
}

#[test]
fn fragment_depth_averages_all_samples() {
    check(F::WMTPixelFormatDepth32Float, "draw_depth", |x, y| {
        ((7 + x + y) as f32 / 32.0).to_le_bytes()
    });
}
