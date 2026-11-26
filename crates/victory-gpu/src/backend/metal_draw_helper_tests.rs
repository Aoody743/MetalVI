use super::*;
use crate::resource::ResourceDesc;
use victory_metal::Device;
use victory_metal::sys::WMTPixelFormat as Pixel;
use victory_metal::texture::TextureDescriptor;
use windows::Win32::Foundation::RECT;

#[test]
fn expands_console_topologies_from_index_values() {
    use ConsoleTopology::*;
    let source = [43, 11, 87, 23, 23, 7, 99];
    for (topology, count, expected) in [
        (TriangleFan, 4, &[43, 11, 87, 43, 87, 23][..]),
        (LineLoop, 3, &[43, 11, 11, 87, 87, 43]),
        (Quads, 7, &[43, 11, 87, 43, 87, 23]),
        (
            QuadStrip,
            7,
            &[43, 11, 23, 43, 23, 87, 87, 23, 7, 87, 7, 23],
        ),
    ] {
        assert_eq!(
            console_indices(topology, &source[..count]),
            expected,
            "{topology:?}"
        );
    }
    assert_eq!(console_indices(LineLoop, &source[..2]), [43, 11, 11, 43]);
    assert_eq!(
        console_indices(Quads, &[0, 1, 2, 3, 4, 5, 6, 7]),
        [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]
    );
    for (topology, minimum) in [(TriangleFan, 3), (LineLoop, 2), (Quads, 4), (QuadStrip, 4)] {
        for n in 0..minimum {
            assert!(
                console_indices(topology, &source[..n]).is_empty(),
                "{topology:?}: {n}"
            );
        }
    }
}

#[test]
fn decodes_all_rtv_and_dsv_dimensions() {
    use Some as S;
    // Distinct word values catch confusing DSV flags, mip and slice.
    for (depth, cases) in [
        (
            false,
            &[
                (0, None),
                (1, None),
                (2, S((2, 0))),
                (3, S((2, 7))),
                (4, S((2, 0))),
                (5, S((2, 7))),
                (6, S((0, 0))),
                (7, S((0, 2))),
                (8, None),
                (9, None),
            ][..],
        ),
        (
            true,
            &[
                (0, None),
                (1, S((7, 0))),
                (2, S((7, 9))),
                (3, S((7, 0))),
                (4, S((7, 9))),
                (5, S((0, 0))),
                (6, S((0, 7))),
                (7, None),
            ],
        ),
    ] {
        for &(dim, expected) in cases {
            assert_eq!(
                level_slice(&[28, dim, 2, 7, 9, 11], depth),
                expected,
                "depth={depth}, dim={dim}"
            );
        }
    }
}

#[test]
fn scissors_clip_each_edge_and_reject_empty_intersections() {
    let rect = |left, top, right, bottom| RECT {
        left,
        top,
        right,
        bottom,
    };
    let s = scissor(&rect(-5, -4, 20, 30), 12, 10).unwrap();
    assert_eq!((s.x, s.y, s.width, s.height), (0, 0, 12, 10));
    let s = scissor(&rect(2, 3, 9, 8), 12, 10).unwrap();
    assert_eq!((s.x, s.y, s.width, s.height), (2, 3, 7, 5));
    for r in [rect(-5, 1, -1, 2), rect(0, 12, 3, 13), rect(1, 2, 5, 2)] {
        assert!(scissor(&r, 12, 10).is_none());
    }
    assert!(scissor(&rect(0, 0, 5, 5), 0, 0).is_none());
}

#[test]
fn scissor_reversed_edges_are_empty() {
    let r = RECT {
        left: 6,
        top: 1,
        right: 4,
        bottom: 4,
    };
    assert!(scissor(&r, 12, 10).is_none());
    let r = RECT {
        left: 1,
        top: 6,
        right: 4,
        bottom: 4,
    };
    assert!(scissor(&r, 12, 10).is_none());
}

#[test]
fn packs_all_draw_variants_for_msc() {
    let device = Device::first().unwrap();
    let buffer = device.shared_buffer(&[0; 64]).unwrap();
    let vertices = DrawKind::Vertices {
        count: 6,
        instances: 3,
        first: 5,
        first_instance: 9,
    };
    assert_eq!(draw_params(&vertices), ([6, 3, 5, 9, 0], 0));
    for (index_type, offset, uniform) in [(0, 6, 1), (1, 12, 2)] {
        let indexed = DrawKind::Indexed {
            count: 6,
            instances: 3,
            base: -7,
            first_instance: 9,
            buffer: buffer.clone(),
            offset,
            index_type,
        };
        assert_eq!(
            draw_params(&indexed),
            ([6, 3, offset, 0xffff_fff9, 9], uniform)
        );
    }
    let generated = DrawKind::Generated {
        indices: vec![4, 2, 3],
        instances: 2,
        base: -1,
        first_instance: 5,
    };
    assert_eq!(draw_params(&generated), ([3, 2, 0, u32::MAX, 5], 2));
    assert_eq!(
        draw_params(&DrawKind::Mesh { groups: [4, 2, 1] }),
        ([0; 5], 0)
    );
    let geometry = DrawKind::Geometry {
        draw: victory_metal::msc::GeometryDraw {
            args: [1; 20],
            info: [2; 24],
            groups: [2, 1, 1],
            object_threads: [3, 1, 1],
            mesh_threads: [4, 1, 1],
        },
        _index: Some(buffer),
    };
    assert_eq!(draw_params(&geometry), ([0; 5], 0));
}

fn texture(device: &Device, format: Pixel, array: bool) -> Texture {
    device
        .new_texture(&TextureDescriptor {
            pixel_format: format,
            width: 8,
            height: 8,
            array_length: if array { 2 } else { 1 },
            type_: if array {
                sys::WMTTextureType::WMTTextureType2DArray
            } else {
                sys::WMTTextureType::WMTTextureType2D
            },
            mipmap_level_count: if array { 2 } else { 1 },
            usage: sys::WMTTextureUsage(4),
            options: sys::WMTResourceOptions(0x20),
            ..Default::default()
        })
        .unwrap()
}

fn attachment(device: &Device, format: Pixel, array: bool) -> Attachment {
    Attachment {
        resource: 0,
        texture: texture(device, format, array),
        pixel: format.0,
        level: 0,
        slice: 0,
        width: 8,
        height: 8,
        samples: 1,
        companion: None,
    }
}

#[test]
fn target_identity_includes_handle_level_slice_and_slot() {
    let device = Device::first().unwrap();
    let mut baseline = Targets::default();
    assert!(baseline.same(&Targets::default()));
    baseline.colors[0] = Some(attachment(&device, Pixel::WMTPixelFormatRGBA8Unorm, true));
    baseline.depth = Some(attachment(
        &device,
        Pixel::WMTPixelFormatDepth32Float,
        false,
    ));
    assert!(baseline.same(&baseline.clone()));
    let mut other = baseline.clone();
    other.colors[0].as_mut().unwrap().level = 1;
    assert!(!baseline.same(&other));
    let mut other = baseline.clone();
    other.colors[0].as_mut().unwrap().slice = 1;
    assert!(!baseline.same(&other));
    let mut other = baseline.clone();
    other.colors[0].as_mut().unwrap().texture =
        texture(&device, Pixel::WMTPixelFormatRGBA8Unorm, true);
    assert!(!baseline.same(&other));
    let mut other = baseline.clone();
    other.colors.swap(0, 1);
    assert!(!baseline.same(&other));
    let mut other = baseline.clone();
    other.depth = None;
    assert!(!baseline.same(&other));
}

#[test]
fn targets_require_exact_formats_and_sample_count() {
    let device = Device::first().unwrap();
    let color = Pixel::WMTPixelFormatRGBA8Unorm;
    let depth = Pixel::WMTPixelFormatDepth32Float_Stencil8;
    let lib = device
        .library_msl(
            r#"
        #include <metal_stdlib>
        vertex float4 f71_vs() { return float4(0, 0, 0, 1); }
        fragment float4 f71_fs() { return float4(1); }
    "#,
        )
        .unwrap();
    // fit reads cached metadata only; the real PSO/state are never drawn.
    let mut pipeline = MetalGraphics {
        name: String::new(),
        depth_write: true,
        pso: device
            .render_pipeline(&lib, "f71_vs", &lib, "f71_fs", color, 1)
            .unwrap(),
        dss: device
            .depth_stencil_state(&DepthStencilDesc::default())
            .unwrap(),
        raster: [0; 4],
        bias: [0.0; 3],
        stages: vec![],
        vertex_slots: 0,
        mesh: false,
        mesh_threads: [0; 3],
        geometry: None,
        colors: [color.0, 0, 0, 0, 0, 0, 0, 0],
        depth: depth.0,
        stencil: depth.0,
        samples: 1,
    };
    let mut targets = Targets {
        depth: Some(attachment(&device, depth, false)),
        stencil: true,
        ..Default::default()
    };
    targets.colors[0] = Some(attachment(&device, color, false));
    targets.finish();
    assert!(targets.fit(&pipeline));
    pipeline.samples = 0;
    assert!(targets.fit(&pipeline));
    pipeline.samples = 4;
    assert!(!targets.fit(&pipeline));
    pipeline.samples = 1;
    pipeline.colors[0] = Pixel::WMTPixelFormatBGRA8Unorm.0;
    assert!(!targets.fit(&pipeline));
    pipeline.colors[0] = color.0;
    pipeline.colors[7] = color.0;
    assert!(!targets.fit(&pipeline));
    pipeline.colors[7] = 0;
    pipeline.depth = Pixel::WMTPixelFormatDepth32Float.0;
    assert!(!targets.fit(&pipeline));
    pipeline.depth = depth.0;
    targets.stencil = false;
    assert!(!targets.fit(&pipeline));
    pipeline.stencil = 0;
    assert!(targets.fit(&pipeline));
}

#[test]
fn classifies_blend_factors_and_pixel_formats() {
    use super::super::metal::integer_format;
    use super::super::metal_texture::{depth_pixel, is_depth_pixel};
    assert_eq!(
        [14, 15, 16, 17, 18, 19].map(single_source),
        [14, 2, 3, 4, 5, 19]
    );
    for p in [
        Pixel::WMTPixelFormatR8Uint,
        Pixel::WMTPixelFormatR16Sint,
        Pixel::WMTPixelFormatRGBA8Uint,
        Pixel::WMTPixelFormatRGBA32Sint,
        Pixel::WMTPixelFormatRGB10A2Uint,
    ] {
        assert!(integer_format(p.0), "{}", p.0);
    }
    for p in [
        Pixel::WMTPixelFormatR8Unorm,
        Pixel::WMTPixelFormatRGBA8Unorm_sRGB,
        Pixel::WMTPixelFormatRGBA32Float,
        Pixel::WMTPixelFormatRGB10A2Unorm,
        Pixel::WMTPixelFormatDepth32Float,
    ] {
        assert!(!integer_format(p.0), "{}", p.0);
    }
    for (dxgi, expected) in [
        (55, Some(Pixel::WMTPixelFormatDepth16Unorm)),
        (40, Some(Pixel::WMTPixelFormatDepth32Float)),
        (20, Some(Pixel::WMTPixelFormatDepth32Float_Stencil8)),
        (45, Some(Pixel::WMTPixelFormatDepth32Float_Stencil8)),
        (44, Some(Pixel::WMTPixelFormatDepth32Float_Stencil8)),
        (39, Some(Pixel::WMTPixelFormatDepth32Float)),
        (53, Some(Pixel::WMTPixelFormatDepth16Unorm)),
        (61, None),
    ] {
        assert_eq!(depth_pixel(dxgi), expected, "DXGI {dxgi}");
    }
    for p in [250, 252, 253, 255, 260, 261, 262] {
        assert!(is_depth_pixel(p), "{p}");
    }
    for p in [0, 70, 249, 251, 254, 256, 259, 263] {
        assert!(!is_depth_pixel(p), "{p}");
    }
}

#[test]
fn depth_resource_views_use_only_compatible_formats() {
    use super::super::metal_texture::{MetalTexture, view_format};
    let device = Device::first().unwrap();
    let pixel = Pixel::WMTPixelFormatDepth32Float_Stencil8;
    // SAFETY: ResourceDesc is an integer-only C record.
    let mut desc: ResourceDesc = unsafe { std::mem::zeroed() };
    desc.format = 28; // Title format appears to be color; the Metal backing is depth.
    let mut t = MetalTexture {
        texture: texture(&device, pixel, false),
        pixel,
        desc,
        mips: 1,
        slices: 1,
        companion: Default::default(),
    };
    for wanted in [62, 47, 21] {
        assert_eq!(
            view_format(&t, wanted),
            Pixel::WMTPixelFormatX32_Stencil8,
            "DXGI {wanted}"
        );
    }
    for wanted in [0, 28, 41, 46] {
        assert_eq!(view_format(&t, wanted), pixel, "DXGI {wanted}");
    }
    t.desc.format = 19; // A declared typeless depth/stencil resource.
    assert_eq!(view_format(&t, 21), Pixel::WMTPixelFormatX32_Stencil8);
    let pixel = Pixel::WMTPixelFormatDepth32Float;
    let plain = MetalTexture {
        texture: texture(&device, pixel, false),
        pixel,
        ..t
    };
    assert_eq!(view_format(&plain, 62), pixel);
}
