use super::*;

fn transfer(linear: f64) -> u8 {
    let linear = linear.clamp(0.0, 1.0);
    let code = if linear <= 0.0031308 {
        12.92 * linear
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (255.0 * code).round() as u8
}

fn output_pixel(rgb: [f64; 3]) -> [u8; 4] {
    [transfer(rgb[2]), transfer(rgb[1]), transfer(rgb[0]), 255]
}

#[test]
fn typed_formats_have_explicit_transfer() {
    let defaults = PresentOptions::default();
    let linear = PresentOptions {
        unorm_is_linear: true,
        ..defaults
    };
    assert_eq!(
        source_mode(24, defaults).unwrap(),
        (Format::WMTPixelFormatRGB10A2Unorm, 0)
    );
    assert_eq!(
        source_mode(28, linear).unwrap(),
        (Format::WMTPixelFormatRGBA8Unorm, 1)
    );
    assert_eq!(
        source_mode(29, defaults).unwrap(),
        (Format::WMTPixelFormatRGBA8Unorm_sRGB, 1)
    );
    assert_eq!(
        source_mode(10, linear).unwrap(),
        (Format::WMTPixelFormatRGBA16Float, 2)
    );
    assert!(source_mode(9, defaults).is_err()); // Typeless is not a color encoding.
}

#[test]
#[ignore = "requires Rosetta Wine with victorymetal"]
fn wine_readback() {
    let _pool = Pool::new();
    let device = Device::first().unwrap();
    let queue = device.new_queue(4).unwrap();
    let mut converter = PresentConverter::new(&device).unwrap();
    let defaults = PresentOptions::default();

    // Read raw BGRA bytes to detect missing or double application of the OETF.
    let mut verify = |dxgi, row: &[u8], expected: &[[u8; 4]], options, tolerance, enlarge| {
        let src_width = expected.len() as u32;
        let input = device
            .new_texture(&crate::texture::TextureDescriptor {
                width: src_width,
                height: 2,
                pixel_format: source_mode(dxgi, options).unwrap().0,
                ..Default::default()
            })
            .unwrap();
        let mut bytes = row.repeat(2);
        // SAFETY: bytes are sized for exactly two tightly packed rows in the
        // chosen format; the shared source isn't in use by the GPU yet.
        unsafe {
            sys::MTLTexture_replaceRegion(
                input.handle(),
                sys::WMTOrigin { x: 0, y: 0, z: 0 },
                sys::WMTSize {
                    width: src_width.into(),
                    height: 2,
                    depth: 1,
                },
                0,
                0,
                sys::WMTMemoryPointer {
                    ptr: bytes.as_mut_ptr().cast(),
                },
                row.len() as u64,
                0,
            );
        }
        if enlarge {
            assert!(expected.windows(2).all(|pair| pair[0] == pair[1]));
        }
        let width = if enlarge { 2 * src_width } else { src_width };
        let height = if enlarge { 8 } else { 6 };
        for pixel_format in [
            Format::WMTPixelFormatBGRA8Unorm,
            Format::WMTPixelFormatBGRA8Unorm_sRGB,
        ] {
            let target = device
                .new_texture(&crate::texture::TextureDescriptor {
                    width,
                    height,
                    pixel_format,
                    usage: sys::WMTTextureUsage::WMTTextureUsageRenderTarget,
                    ..Default::default()
                })
                .unwrap();
            let cb = queue.command_buffer().unwrap();
            converter.draw(&cb, &input, dxgi, &target, options).unwrap();
            cb.commit();
            cb.wait().unwrap();

            let mut raw = vec![[0_u8; 4]; (width * height) as usize];
            // SAFETY: the GPU is finished and raw spans the shared BGRA image.
            unsafe {
                sys::MTLTexture_getBytes(
                    target.handle(),
                    sys::WMTOrigin { x: 0, y: 0, z: 0 },
                    sys::WMTSize {
                        width: width.into(),
                        height: height.into(),
                        depth: 1,
                    },
                    0,
                    0,
                    sys::WMTMemoryPointer {
                        ptr: raw.as_mut_ptr().cast(),
                    },
                    u64::from(width) * 4,
                    0,
                );
            }
            for (y, row) in raw.chunks_exact(width as usize).enumerate() {
                for (x, pixel) in row.iter().enumerate() {
                    if (2..(height as usize - 2)).contains(&y) {
                        let want = expected[if enlarge { x / 2 } else { x }];
                        for channel in 0..3 {
                            assert!(
                                pixel[channel].abs_diff(want[channel]) <= tolerance,
                                "DXGI {dxgi} -> Metal {pixel_format:?} ({x}, {y}): got {pixel:?}, want {want:?}",
                            );
                        }
                        assert_eq!(pixel[3], 255);
                    } else {
                        assert_eq!(*pixel, [0, 0, 0, 255], "bar at ({x}, {y})");
                    }
                }
            }
            println!(
                "DXGI {dxgi} -> Metal {}: {} pixels match (RGB tolerance {tolerance})",
                pixel_format.0,
                raw.len()
            );
        }
    };

    let rgba = [
        8, 64, 128, 17, 200, 220, 240, 37, 0, 10, 255, 0, 255, 0, 0, 128,
    ];
    let bgra = [
        128, 64, 8, 17, 240, 220, 200, 37, 255, 10, 0, 0, 0, 0, 255, 128,
    ];
    let coded = [
        [128, 64, 8, 255],
        [240, 220, 200, 255],
        [255, 10, 0, 255],
        [0, 0, 255, 255],
    ];
    for (dxgi, row) in [(28, &rgba), (29, &rgba), (87, &bgra), (91, &bgra)] {
        verify(dxgi, row, &coded, defaults, 1, false);
    }
    let linear = PresentOptions {
        unorm_is_linear: true,
        ..defaults
    };
    let expected = coded.map(|[b, g, r, _]| output_pixel([r, g, b].map(|v| f64::from(v) / 255.0)));
    verify(28, &rgba, &expected, linear, 1, false);
    verify(
        28,
        &[8, 64, 128, 17].repeat(4),
        &[coded[0]; 4],
        defaults,
        1,
        true,
    );

    let rgb10: [[u32; 3]; 4] = [[32, 512, 768], [1023, 512, 0], [0, 32, 1023], [685, 431, 8]];
    let packed = rgb10.map(|[r, g, b]| (r | (g << 10) | (b << 20) | (2 << 30)).to_le_bytes());
    let expected = rgb10
        .map(|[r, g, b]| [b, g, r].map(|v| (v as f64 * 255.0 / 1023.0).round() as u8))
        .map(|[b, g, r]| [b, g, r, 255]);
    verify(24, packed.as_flattened(), &expected, defaults, 1, false);
    let expected = rgb10.map(|rgb| output_pixel(rgb.map(|v| v as f64 / 1023.0)));
    verify(24, packed.as_flattened(), &expected, linear, 1, false);

    // FP16 pixels: 0.5 gray, paper white, highlights 2x and 4x paper white.
    let half = [0x3800_u16, 0x3c00, 0x4000, 0x4400].map(|x| [x, x, x, 0].map(u16::to_le_bytes));
    let expected = [0.375, 0.75, 0.9375, 0.975].map(|gray| output_pixel([gray; 3]));
    verify(
        10,
        half.as_flattened().as_flattened(),
        &expected,
        defaults,
        1,
        false,
    );
    let expected = [0.1875, 0.375, 0.75, 0.9375].map(|gray| output_pixel([gray; 3]));
    let paper_white = PresentOptions {
        scrgb_paper_white_nits: 160.0,
        ..defaults
    };
    verify(
        10,
        half.as_flattened().as_flattened(),
        &expected,
        paper_white,
        1,
        false,
    );
}

/// Series S presents two planes: the scene (black in menus) and the UI over it.
/// The overlay draw keeps the plane below where the UI is transparent and puts
/// the premultiplied UI over it elsewhere, the UI larger than the scene (both
/// fitted to the target).
#[test]
#[ignore = "requires Rosetta Wine with victorymetal"]
fn wine_overlay_plane_blends_over_the_scene() {
    let _pool = Pool::new();
    let device = Device::first().unwrap();
    let queue = device.new_queue(4).unwrap();
    let mut converter = PresentConverter::new(&device).unwrap();
    let defaults = PresentOptions::default();
    let texture = |width: u32, height: u32, mut bytes: Vec<u8>| {
        let t = device
            .new_texture(&crate::texture::TextureDescriptor {
                width,
                height,
                pixel_format: Format::WMTPixelFormatRGBA8Unorm,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(bytes.len(), (width * height * 4) as usize);
        // SAFETY: tightly packed RGBA8 rows of this texture; not in use by the GPU yet.
        unsafe {
            sys::MTLTexture_replaceRegion(
                t.handle(),
                sys::WMTOrigin { x: 0, y: 0, z: 0 },
                sys::WMTSize { width: width.into(), height: height.into(), depth: 1 },
                0,
                0,
                sys::WMTMemoryPointer { ptr: bytes.as_mut_ptr().cast() },
                u64::from(width) * 4,
                0,
            );
        }
        t
    };
    // Scene 4x4 dark red; UI 8x8: left half transparent, right half
    // premultiplied green at alpha 0.5 (code 128 -> 64 premultiplied).
    let scene = texture(4, 4, [100, 0, 0, 255].repeat(16));
    let row = [[0u8, 0, 0, 0].repeat(4), [0u8, 64, 0, 128].repeat(4)].concat();
    let ui = texture(8, 8, row.repeat(8));
    let target = device
        .new_texture(&crate::texture::TextureDescriptor {
            width: 8,
            height: 8,
            pixel_format: Format::WMTPixelFormatBGRA8Unorm,
            usage: sys::WMTTextureUsage::WMTTextureUsageRenderTarget,
            ..Default::default()
        })
        .unwrap();
    let cb = queue.command_buffer().unwrap();
    converter.draw(&cb, &scene, 28, &target, defaults).unwrap();
    converter.overlay(&cb, &ui, 28, &target, defaults).unwrap();
    cb.commit();
    cb.wait().unwrap();
    let mut raw = vec![[0_u8; 4]; 64];
    // SAFETY: the GPU is finished and raw spans the 8x8 BGRA target.
    unsafe {
        sys::MTLTexture_getBytes(
            target.handle(),
            sys::WMTOrigin { x: 0, y: 0, z: 0 },
            sys::WMTSize { width: 8, height: 8, depth: 1 },
            0,
            0,
            sys::WMTMemoryPointer { ptr: raw.as_mut_ptr().cast() },
            32,
            0,
        );
    }
    for (i, pixel) in raw.iter().enumerate() {
        let x = i % 8;
        // BGRA: the scene alone on the left, red * (1 - 0.5) + green 64 on the right.
        let want: [u8; 4] = if x < 4 { [0, 0, 100, 255] } else { [0, 64, 50, 255] };
        for c in 0..3 {
            assert!(pixel[c].abs_diff(want[c]) <= 1, "({x}, {}): got {pixel:?}, want {want:?}", i / 8);
        }
    }
    println!("overlay plane: 64 pixels match");
}
