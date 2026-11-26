use super::*;
use victory_metal::Compute;
use victory_metal::render::{
    ColorDesc, DepthStencilDesc, PassColor, PassDepth, Render, RenderPass, RenderPipelineDesc,
};
use victory_metal::sys::{
    WMTLoadAction as Load, WMTPrimitiveType as Prim, WMTStoreAction as Store,
};

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
vertex float4 vs(uint id [[vertex_id]]) {
    return float4(id == 1 ? 3 : -1, id == 2 ? 3 : -1, 0.5, 1);
}
fragment float4 ps() { return float4(1, 0, 0, 1); }
kernel void seed_d16(texture2d_array<half, access::write> tex [[texture(0)]],
                     uint2 p [[thread_position_in_grid]]) {
    // Half +/-2 has storage bits 0x4000/0xc000, not numeric depth values.
    tex.write(half4(p.x % 2 == 0 ? 2.0h : -2.0h), p, 1, 1);
}
kernel void seed_d32(texture2d_array<uint, access::write> tex [[texture(0)]],
                     uint2 p [[thread_position_in_grid]]) {
    tex.write(uint4(as_type<uint>(p.x % 2 == 0 ? 0.25f : 0.75f)), p, 1, 1);
}
"#;

fn check_alias(colour_format: P, depth: P, low: u32, high: u32, mid: u32) {
    let _pool = victory_metal::Pool::new();
    let device = Device::first().unwrap();
    let queue = device.new_queue(4).unwrap();
    let cb = queue.command_buffer().unwrap();
    let mut desc = TextureDescriptor {
        pixel_format: colour_format,
        width: 8,
        height: 8,
        array_length: 2,
        type_: T::WMTTextureType2DArray,
        mipmap_level_count: 2,
        usage: WMTTextureUsage(1 | 2 | 0x10),
        options: WMTResourceOptions(0x20),
        ..Default::default()
    };
    let colour = MetalTexture {
        texture: device.new_texture(&desc).unwrap(),
        pixel: colour_format,
        // SAFETY: integer-only record; zero denotes a single sample to base_type.
        desc: unsafe { std::mem::zeroed() },
        mips: 2,
        slices: 2,
        companion: Default::default(),
    };
    let alias = DepthCompanion::new(&device, &colour, depth).unwrap();
    assert_eq!(
        (
            alias.texture.width(),
            alias.texture.height(),
            alias.texture.mipmap_level_count(),
            alias.texture.array_length()
        ),
        (8, 8, 2, 2)
    );
    desc.pixel_format = P::WMTPixelFormatRGBA8Unorm;
    desc.width = 4;
    desc.height = 4;
    desc.array_length = 1;
    desc.type_ = T::WMTTextureType2D;
    desc.mipmap_level_count = 1;
    desc.usage = WMTTextureUsage(4);
    let target = device.new_texture(&desc).unwrap();
    let lib = device.library_msl(SOURCE).unwrap();
    let mut pipeline_desc = RenderPipelineDesc {
        depth_format: depth.0,
        sample_count: 1,
        ..Default::default()
    };
    pipeline_desc.colors[0] = ColorDesc {
        pixel_format: desc.pixel_format.0,
        write_mask: 15,
        ..Default::default()
    };
    let pipeline = device
        .render_pipeline_desc((&lib, "vs"), Some((&lib, "ps")), &pipeline_desc)
        .unwrap();
    let state = device
        .depth_stencil_state(&DepthStencilDesc {
            depth_write: true,
            depth_compare: 1,
            ..Default::default()
        })
        .unwrap();
    let steps = [
        Render::Pipeline {
            pipeline: &pipeline,
        },
        Render::DepthStencil {
            state: &state,
            stencil_ref: 0,
        },
        Render::Draw {
            primitive_type: Prim::WMTPrimitiveTypeTriangle,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 1,
            base_instance: 0,
        },
    ];
    let mut pass = RenderPass {
        width: 4,
        height: 4,
        samples: 1,
        ..Default::default()
    };
    pass.colors[0] = Some(PassColor {
        texture: &target,
        level: 0,
        slice: 0,
        load: Load::WMTLoadActionClear,
        store: Store::WMTStoreActionStore,
        clear: [0.0; 4],
    });
    pass.depth = Some(PassDepth {
        texture: &alias.texture,
        level: 1,
        slice: 1,
        load: Load::WMTLoadActionLoad,
        store: Store::WMTStoreActionStore,
        clear_depth: 1.0,
        clear_stencil: 0,
    });

    let n = alias.bytes as usize;
    let mut raw = [0u8; 2048];
    for y in 0..4 {
        for x in 0..4 {
            let at = y * 256 + x * n;
            raw[at..at + n].copy_from_slice(&high.to_le_bytes()[..n]);
            raw[1024 + at..1024 + at + n].copy_from_slice(&low.to_le_bytes()[..n]);
        }
    }
    let upload = device.shared_buffer(&raw).unwrap();
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    let mem = unsafe {
        VirtualAlloc(
            std::ptr::null(),
            8192,
            MEM_RESERVE | MEM_COMMIT,
            PAGE_READWRITE,
        )
    };
    assert!(!mem.is_null());
    let out_buffer = unsafe { device.wrap(mem, 8192) }.unwrap();
    let out = &out_buffer;
    let seed = if n == 2 { "seed_d16" } else { "seed_d32" };
    let seed = device.compute_pipeline_from_library(&lib, seed).unwrap();
    cb.compute(&[
        Compute::Pipeline {
            pipeline: &seed,
            threads_per_group: [4, 4, 1],
        },
        Compute::Texture {
            texture: &colour.texture,
            index: 0,
        },
        Compute::Dispatch { groups: [1, 1, 1] },
    ])
    .unwrap();
    let mut blit = cb.blit_encoder().unwrap();
    blit.copy_from_buffer_to_texture(
        &upload,
        0,
        256,
        1024,
        [4, 4, 1],
        &colour.texture,
        0,
        1,
        [0; 3],
    );
    blit.finish();
    let mut sync = DepthSync::default();
    sync.load(&device, &cb, &alias, 1, 1).unwrap();
    assert_eq!(sync.synced.len(), 1);
    cb.render_pass_with(&pass, &steps).unwrap();
    sync.colour_may_change();
    sync.store(&device, &cb, &alias, 1, 1).unwrap();
    sync.load(&device, &cb, &alias, 1, 0).unwrap();
    let mut blit = cb.blit_encoder().unwrap();
    for (tex, slice, level, offset) in [
        (&alias.texture, 1, 1, 0),
        (&colour.texture, 1, 1, 1024),
        (&target, 0, 0, 2048),
        (&alias.texture, 0, 1, 3072),
        (&colour.texture, 0, 1, 4096),
    ] {
        blit.copy_from_texture_to_buffer(
            tex,
            slice,
            level,
            [0; 3],
            [4, 4, 1],
            &out,
            offset,
            256,
            1024,
        );
    }
    blit.copy_from_buffer_to_texture(
        &upload,
        1024,
        256,
        1024,
        [4, 4, 1],
        &colour.texture,
        1,
        1,
        [0; 3],
    );
    blit.finish();
    sync.colour_may_change();
    sync.load(&device, &cb, &alias, 1, 1).unwrap();
    let mut blit = cb.blit_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &alias.texture,
        1,
        1,
        [0; 3],
        [4, 4, 1],
        &out,
        5120,
        256,
        1024,
    );
    blit.finish();
    cb.commit();
    cb.wait().unwrap();
    // SAFETY: shared buffer is mapped, live and its GPU writes have completed.
    let out = unsafe { std::slice::from_raw_parts(mem.cast::<u8>(), 6144) };
    for y in 0..4 {
        for x in 0..4 {
            let at = y * 256 + x * n;
            let expected = if x % 2 == 0 { low } else { mid };
            assert_eq!(
                &out[at..at + n],
                &expected.to_le_bytes()[..n],
                "depth {x},{y}"
            );
            assert_eq!(
                &out[1024 + at..1024 + at + n],
                &expected.to_le_bytes()[..n],
                "colour {x},{y}"
            );
            assert_eq!(
                out[2048 + y * 256 + x * 4],
                if x % 2 == 0 { 0 } else { 255 },
                "depth test {x},{y}"
            );
            for base in [3072, 4096] {
                assert_eq!(
                    &out[base + at..base + at + n],
                    &high.to_le_bytes()[..n],
                    "other slice {x},{y}"
                );
            }
            assert_eq!(
                &out[5120 + at..5120 + at + n],
                &low.to_le_bytes()[..n],
                "reload {x},{y}"
            );
        }
    }
    drop(cb);
    drop(out_buffer);
    unsafe { VirtualFree(mem, 0, MEM_RELEASE) };
}

#[test]
fn d16_over_r16_float() {
    check_alias(
        P::WMTPixelFormatR16Float,
        P::WMTPixelFormatDepth16Unorm,
        0x4000,
        0xc000,
        0x8000,
    );
}

#[test]
fn d32_over_r32_uint() {
    check_alias(
        P::WMTPixelFormatR32Uint,
        P::WMTPixelFormatDepth32Float,
        0.25f32.to_bits(),
        0.75f32.to_bits(),
        0.5f32.to_bits(),
    );
}
