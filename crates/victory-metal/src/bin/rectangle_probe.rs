#[cfg(not(windows))]
fn main() {}

#[cfg(windows)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::PathBuf;
    use victory_metal::msc::{self, Stage, VERTEX_TABLE_BIND_POINT, VERTEX_TABLE_SLOTS, VertexBufferEntry, VertexInput, vertex_table_bytes};
    use victory_metal::render::{ColorAttachment, ColorDesc, DepthStencilDesc, PassDepth, Render, RenderPass, RenderPipelineDesc};
    use victory_metal::sys::*;
    use victory_metal::texture::{Texture, TextureDescriptor};
    use victory_metal::{Buffer, Device, Pool, ResidencySet};

    const W: u32 = 64;
    const H: u32 = 64;
    const FORMAT: WMTPixelFormat = WMTPixelFormat::WMTPixelFormatBGRA8Unorm;

    fn readback_bytes(tex: &Texture, bytes_per_pixel: u32) -> Vec<u8> {
        let mut out = vec![0u8; (W * H * bytes_per_pixel) as usize];
        // SAFETY: out covers the tightly packed 64x64 slice after GPU wait.
        unsafe {
            MTLTexture_getBytes(
                tex.handle(),
                WMTOrigin { x: 0, y: 0, z: 0 },
                WMTSize { width: W.into(), height: H.into(), depth: 1 },
                0,
                0,
                WMTMemoryPointer { ptr: out.as_mut_ptr().cast() },
                u64::from(W * bytes_per_pixel),
                out.len() as u64,
            );
        }
        out
    }

    fn run_baseline(device: &Device, queue: &victory_metal::Queue, residency: &ResidencySet, tex: &Texture, vs: &[u8], gs: &[u8], ps: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        let v = msc::convert_geometry_vertex(vs, &[], &[], &[])?;
        let g = msc::convert_geometry(gs, &[], &[], v.vertex_output_size)?;
        let f = msc::convert(ps, Stage::Fragment, &[], &[])?;
        println!("Case 1 (baseline HLSL): {:?}, object={}, mesh={}", g.config, v.shader.shader.entry, g.shader.entry);
        let object_lib = device.library(&v.shader.shader.metallib)?;
        let stage_in = device.library(&v.shader.stage_in)?;
        let mesh_lib = device.library(&g.shader.metallib)?;
        let fragment_lib = device.library(&f.metallib)?;
        let mut desc = RenderPipelineDesc { topology_class: 3, sample_count: 1, ..Default::default() };
        desc.colors[0] = ColorDesc { pixel_format: FORMAT.0, write_mask: 15, ..Default::default() };
        let pipeline = device.geometry_render_pipeline_desc(
            (&object_lib, &v.shader.shader.entry),
            (&mesh_lib, &g.shader.entry),
            &stage_in,
            Some((&fragment_lib, &f.entry)),
            &desc,
            g.config,
        )?;
        let indices0 = device.shared_buffer(&[0u16, 1, 2, 3, 4, 5].map(u16::to_le_bytes).concat())?;
        let indices1 = device.shared_buffer(&[99u16, 0, 1, 2, 3, 4, 5].map(u16::to_le_bytes).concat())?;
        let indices32 = device.shared_buffer(&[99u32, 88, 0, 1, 2, 3, 4, 5].map(u32::to_le_bytes).concat())?;
        residency.add(&[indices0.handle(), indices1.handle(), indices32.handle()]);
        residency.commit();
        for (label, first, index) in [
            ("direct", 0, None),
            ("indexed_r16", 0, Some((0, indices0.gpu_address, 0))),
            ("indexed_r16_offset", 1, Some((0, indices1.gpu_address, 0))),
            ("indexed_r32_offset", 2, Some((1, indices32.gpu_address, 0))),
        ] {
            let draw = g.config.draw(6, 1, first, 0, index).expect("six valid vertices");
            assert_eq!(draw.groups, [1, 1, 1]);
            let mut upload = [0u8; 1280];
            upload[256..276].copy_from_slice(&draw.args);
            upload[512..536].copy_from_slice(&draw.info);
            let buffer = device.shared_buffer(&upload)?;
            residency.add(&[buffer.handle()]);
            residency.commit();
            let mut commands = vec![Render::Pipeline { pipeline: &pipeline }];
            for (idx, offset) in [(0, 0), (1, 0), (2, 0), (4, 256), (5, 512), (6, 768)] {
                commands.push(Render::ObjectBuffer { buffer: &buffer, offset, index: idx });
                if idx != 6 {
                    commands.push(Render::MeshBuffer { buffer: &buffer, offset, index: idx });
                }
            }
            commands.push(Render::DrawMesh { groups: draw.groups, object_threads: draw.object_threads, mesh_threads: draw.mesh_threads });
            let cb = queue.command_buffer()?;
            cb.render_pass(&[ColorAttachment {
                texture: tex, load_action: WMTLoadAction::WMTLoadActionClear,
                store_action: WMTStoreAction::WMTStoreActionStore, clear: [1., 0., 0., 1.],
            }], W, H, &commands)?;
            cb.commit();
            cb.wait()?;
            let image = readback_bytes(tex, 4);
            assert_eq!(&image[(18 * 64 + 22) * 4..][..4], &[0, 255, 0, 255], "{label}: synthesized TR");
            assert_eq!(&image[(45 * 64 + 54) * 4..][..4], &[0, 255, 0, 255], "{label}: synthesized BR");
            for y in 0..H {
                for x in 0..W {
                    let green = (16..48).contains(&y) && ((8..24).contains(&x) || (40..56).contains(&x));
                    let expected = if green { [0, 255, 0, 255] } else { [0, 0, 255, 255] };
                    assert_eq!(&image[((y * W + x) * 4) as usize..][..4], &expected, "{label} pixel ({x}, {y})");
                }
            }
            println!("  PASS {label}: 4096/4096 BGRA texels, including both synthesized corners");
        }
        Ok(())
    }

    fn bind_common<'a>(commands: &mut Vec<Render<'a>>, buffer: &'a Buffer, vertex_table_offset: Option<u64>) {
        for (idx, offset) in [(0, 0), (1, 0), (2, 0), (4, 256), (5, 512)] {
            commands.push(Render::ObjectBuffer { buffer, offset, index: idx });
            commands.push(Render::MeshBuffer { buffer, offset, index: idx });
        }
        commands.push(Render::ObjectBuffer {
            buffer,
            offset: vertex_table_offset.unwrap_or(768),
            index: VERTEX_TABLE_BIND_POINT,
        });
    }

    fn run_suite(dir: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let _pool = Pool::new();
        let device = Device::first()?;
        let queue = device.new_queue(8)?;
        let residency = ResidencySet::new(&device, 64)?;
        let tex = device.new_texture(&TextureDescriptor {
            pixel_format: FORMAT, width: W, height: H, depth: 1, array_length: 1,
            type_: WMTTextureType::WMTTextureType2D, mipmap_level_count: 1, sample_count: 1,
            usage: WMTTextureUsage::WMTTextureUsageRenderTarget,
            options: WMTResourceOptions::WMTResourceStorageModeShared,
        })?;
        residency.add(&[tex.handle()]);
        residency.commit();
        queue.add_residency_set(&residency);

        let read = |name: &str| std::fs::read(dir.join(name));
        // Case 1: baseline HLSL
        run_baseline(&device, &queue, &residency, &tex, &read("vs.dxil")?, &read("gs.dxil")?, &read("ps.dxil")?)?;

        // Case 2: real rect_list_gs + all 3 corner orderings + float varying + flat int tag
        {
            let v = msc::convert_geometry_vertex(&read("vs_a.dxil")?, &[], &[], &[])?;
            let g = msc::convert_geometry(&read("rect_gs_a.dxil")?, &[], &[], v.vertex_output_size)?;
            let f = msc::convert(&read("ps_a.dxil")?, Stage::Fragment, &[], &[])?;
            println!("Case 2 (real rect_list_gs, 3 corner orders + varyings): {:?}", g.config);
            let olib = device.library(&v.shader.shader.metallib)?;
            let sin = device.library(&v.shader.stage_in)?;
            let mlib = device.library(&g.shader.metallib)?;
            let flib = device.library(&f.metallib)?;
            let mut desc = RenderPipelineDesc { topology_class: 3, sample_count: 1, ..Default::default() };
            desc.colors[0] = ColorDesc { pixel_format: FORMAT.0, write_mask: 15, ..Default::default() };
            let pso = device.geometry_render_pipeline_desc(
                (&olib, &v.shader.shader.entry), (&mlib, &g.shader.entry), &sin, Some((&flib, &f.entry)), &desc, g.config,
            )?;
            let draw = g.config.draw(9, 1, 0, 0, None).unwrap();
            let mut upload = [0u8; 1280];
            upload[256..276].copy_from_slice(&draw.args);
            upload[512..536].copy_from_slice(&draw.info);
            let buf = device.shared_buffer(&upload)?;
            residency.add(&[buf.handle()]);
            residency.commit();
            let mut commands = vec![Render::Pipeline { pipeline: &pso }];
            bind_common(&mut commands, &buf, None);
            commands.push(Render::DrawMesh { groups: draw.groups, object_threads: draw.object_threads, mesh_threads: draw.mesh_threads });
            let cb = queue.command_buffer()?;
            cb.render_pass(&[ColorAttachment {
                texture: &tex, load_action: WMTLoadAction::WMTLoadActionClear,
                store_action: WMTStoreAction::WMTStoreActionStore, clear: [0., 0., 0., 1.],
            }], W, H, &commands)?;
            cb.commit();
            cb.wait()?;
            let image = readback_bytes(&tex, 4);
            let rects = [(4u32..20, 40u8), (24..40, 80), (44..60, 120)];
            for y in 0..H {
                for x in 0..W {
                    let p = &image[((y * W + x) * 4) as usize..][..4];
                    if (8..24).contains(&y)
                        && let Some((xr, tag)) = rects.iter().find(|(xr, _)| xr.contains(&x))
                    {
                        let u = (x as f32 + 0.5 - xr.start as f32) / 16.0;
                        let v_coord = (y as f32 + 0.5 - 8.0) / 16.0;
                        let er = (u * 255.0).round() as u8;
                        let eg = (v_coord * 255.0).round() as u8;
                        assert_eq!(p[0], *tag, "Case 2 flat tag mismatch at ({x}, {y})");
                        assert_eq!(p[3], 255, "Case 2 alpha at ({x}, {y})");
                        assert!(p[1].abs_diff(eg) <= 1 && p[2].abs_diff(er) <= 1, "Case 2 uv mismatch at ({x}, {y}): got {p:?}, expect [{tag}, {eg}, {er}, 255]");
                    } else {
                        assert_eq!(p, &[0, 0, 0, 255], "Case 2 bg mismatch at ({x}, {y})");
                    }
                }
            }
            println!("  PASS Case 2: 4096/4096 texels across corner-first/second/third with linear uv & flat tag");
        }

        // Case 3: real rect_list_gs + 2-slot IA stage-in + instancing + negative baseVertex
        {
            let inputs = [
                VertexInput { semantic: "POSITION", semantic_index: 0, format: 16, slot: 0, offset: 4, instance_step_rate: None },
                VertexInput { semantic: "TEXCOORD", semantic_index: 0, format: 16, slot: 0, offset: 12, instance_step_rate: None },
                VertexInput { semantic: "TEXCOORD", semantic_index: 1, format: 16, slot: 1, offset: 4, instance_step_rate: Some(1) },
                VertexInput { semantic: "TEXCOORD", semantic_index: 2, format: 2, slot: 1, offset: 12, instance_step_rate: Some(1) },
            ];
            let v = msc::convert_geometry_vertex(&read("vs_b.dxil")?, &[], &[], &inputs)?;
            let g = msc::convert_geometry(&read("rect_gs_b.dxil")?, &[], &[], v.vertex_output_size)?;
            let f = msc::convert(&read("ps_b.dxil")?, Stage::Fragment, &[], &[])?;
            println!("Case 3 (real rect_list_gs + 2-slot IA stage-in + instancing + negative baseVertex): {:?}", g.config);
            let olib = device.library(&v.shader.shader.metallib)?;
            let sin = device.library(&v.shader.stage_in)?;
            let mlib = device.library(&g.shader.metallib)?;
            let flib = device.library(&f.metallib)?;
            let mut desc = RenderPipelineDesc { topology_class: 3, sample_count: 1, ..Default::default() };
            desc.colors[0] = ColorDesc { pixel_format: FORMAT.0, write_mask: 15, ..Default::default() };
            let pso = device.geometry_render_pipeline_desc(
                (&olib, &v.shader.shader.entry), (&mlib, &g.shader.entry), &sin, Some((&flib, &f.entry)), &desc, g.config,
            )?;
            let slot0_words: [f32; 15] = [
                99.0, -0.75, 0.5, 0.0, 0.0, // TL
                99.0, -0.75, 0.0, 0.0, 1.0, // BL
                99.0, -0.25, 0.5, 1.0, 0.0, // TR
            ];
            let slot1_words: [f32; 21] = [
                99.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0, // instance 0 dummy
                99.0, 0.0, 0.0, 1.0, 0.0, 0.5, 1.0, // instance 1: px 8..24, py 16..32, tint (1,0,0.5)
                99.0, 1.0, -0.5, 0.0, 1.0, 0.75, 1.0, // instance 2: px 40..56, py 32..48, tint (0,1,0.75)
            ];
            let slot0 = device.shared_buffer(&slot0_words.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<_>>())?;
            let slot1 = device.shared_buffer(&slot1_words.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<_>>())?;
            let idx32 = device.shared_buffer(&[999u32, 5, 6, 7].map(u32::to_le_bytes).concat())?;
            residency.add(&[slot0.handle(), slot1.handle(), idx32.handle()]);
            residency.commit();
            let mut entries = [VertexBufferEntry::default(); VERTEX_TABLE_SLOTS];
            entries[0] = VertexBufferEntry { address: slot0.gpu_address, length: slot0.len as u32, stride: 20 };
            entries[1] = VertexBufferEntry { address: slot1.gpu_address, length: slot1.len as u32, stride: 28 };
            let vtable = vertex_table_bytes(&entries);

            for (label, first, index) in [
                ("ia_direct", 0u32, None),
                ("ia_r32_offset_neg_base", 1u32, Some((1u8, idx32.gpu_address, -5i32))),
            ] {
                let draw = g.config.draw(3, 2, first, 1, index).unwrap();
                assert_eq!(draw.groups, [1, 2, 1]);
                let mut upload = [0u8; 1536];
                upload[256..276].copy_from_slice(&draw.args);
                upload[512..536].copy_from_slice(&draw.info);
                upload[768..768 + vtable.len()].copy_from_slice(&vtable);
                let buf = device.shared_buffer(&upload)?;
                residency.add(&[buf.handle()]);
                residency.commit();
                let mut commands = vec![Render::Pipeline { pipeline: &pso }];
                bind_common(&mut commands, &buf, Some(768));
                commands.push(Render::DrawMesh { groups: draw.groups, object_threads: draw.object_threads, mesh_threads: draw.mesh_threads });
                let cb = queue.command_buffer()?;
                cb.render_pass(&[ColorAttachment {
                    texture: &tex, load_action: WMTLoadAction::WMTLoadActionClear,
                    store_action: WMTStoreAction::WMTStoreActionStore, clear: [0., 0., 0., 1.],
                }], W, H, &commands)?;
                cb.commit();
                cb.wait()?;
                let image = readback_bytes(&tex, 4);
                for y in 0..H {
                    for x in 0..W {
                        let p = &image[((y * W + x) * 4) as usize..][..4];
                        if (16..32).contains(&y) && (8..24).contains(&x) {
                            let u = (x as f32 + 0.5 - 8.0) / 16.0;
                            let er = (u * 255.0).round() as u8;
                            assert!(p[0].abs_diff(128) <= 1 && p[1] == 0 && p[2].abs_diff(er) <= 1 && p[3] == 255,
                                "{label} inst1 mismatch at ({x}, {y}): {p:?}");
                        } else if (32..48).contains(&y) && (40..56).contains(&x) {
                            let v_coord = (y as f32 + 0.5 - 32.0) / 16.0;
                            let eg = (v_coord * 255.0).round() as u8;
                            assert!(p[0].abs_diff(191) <= 1 && p[1].abs_diff(eg) <= 1 && p[2] == 0 && p[3] == 255,
                                "{label} inst2 mismatch at ({x}, {y}): {p:?}");
                        } else {
                            assert_eq!(p, &[0, 0, 0, 255], "{label} bg mismatch at ({x}, {y})");
                        }
                    }
                }
                println!("  PASS {label}: 4096/4096 texels (2-slot stage-in, 2 instances, firstInstance=1)");
            }
        }

        // Case 4: 96 rectangles (groups=[2,1,1], crossing mesh TG 32 and object TG 85)
        {
            let v = msc::convert_geometry_vertex(&read("vs_c.dxil")?, &[], &[], &[])?;
            let g = msc::convert_geometry(&read("rect_gs_c.dxil")?, &[], &[], v.vertex_output_size)?;
            let f = msc::convert(&read("ps_c.dxil")?, Stage::Fragment, &[], &[])?;
            println!("Case 4 (96 rects, multi-object-TG): {:?}", g.config);
            let olib = device.library(&v.shader.shader.metallib)?;
            let sin = device.library(&v.shader.stage_in)?;
            let mlib = device.library(&g.shader.metallib)?;
            let flib = device.library(&f.metallib)?;
            let mut desc = RenderPipelineDesc { topology_class: 3, sample_count: 1, ..Default::default() };
            desc.colors[0] = ColorDesc { pixel_format: FORMAT.0, write_mask: 15, ..Default::default() };
            let pso = device.geometry_render_pipeline_desc(
                (&olib, &v.shader.shader.entry), (&mlib, &g.shader.entry), &sin, Some((&flib, &f.entry)), &desc, g.config,
            )?;
            let draw = g.config.draw(288, 1, 0, 0, None).unwrap();
            assert_eq!(draw.groups, [2, 1, 1], "must cross object threadgroup boundary 85");
            let mut upload = [0u8; 1280];
            upload[256..276].copy_from_slice(&draw.args);
            upload[512..536].copy_from_slice(&draw.info);
            let buf = device.shared_buffer(&upload)?;
            residency.add(&[buf.handle()]);
            residency.commit();
            let mut commands = vec![Render::Pipeline { pipeline: &pso }];
            bind_common(&mut commands, &buf, None);
            commands.push(Render::DrawMesh { groups: draw.groups, object_threads: draw.object_threads, mesh_threads: draw.mesh_threads });
            let cb = queue.command_buffer()?;
            cb.render_pass(&[ColorAttachment {
                texture: &tex, load_action: WMTLoadAction::WMTLoadActionClear,
                store_action: WMTStoreAction::WMTStoreActionStore, clear: [0., 0., 0., 1.],
            }], W, H, &commands)?;
            cb.commit();
            cb.wait()?;
            let image = readback_bytes(&tex, 4);
            for y in 0..H {
                for x in 0..W {
                    let p = &image[((y * W + x) * 4) as usize..][..4];
                    if y < 48 {
                        let col = x / 8;
                        let row = y / 4;
                        let pid = row * 8 + col;
                        let eg = if pid >= 85 { 255u8 } else { 64u8 };
                        let er = (pid + 1) as u8;
                        assert!(p[0].abs_diff(128) <= 1 && p[1] == eg && p[2] == er && p[3] == 255,
                            "Case 4 cell ({x},{y}) rect={pid} mismatch: got {p:?}, expect [128, {eg}, {er}, 255]");
                    } else {
                        assert_eq!(p, &[0, 0, 0, 255], "Case 4 bg mismatch at ({x},{y})");
                    }
                }
            }
            println!("  PASS Case 4: 4096/4096 texels across 96 rects (groups=[2,1,1], TG0 rect 0..84 + TG1 rect 85..95)");
        }

        // Case 5: Depth-only pipeline (fragment = None) into Depth32Float
        {
            let v = msc::convert_geometry_vertex(&read("vs_a.dxil")?, &[], &[], &[])?;
            let g = msc::convert_geometry(&read("rect_gs_depth.dxil")?, &[], &[], v.vertex_output_size)?;
            println!("Case 5 (depth-only, fragment=None): {:?}", g.config);
            let olib = device.library(&v.shader.shader.metallib)?;
            let sin = device.library(&v.shader.stage_in)?;
            let mlib = device.library(&g.shader.metallib)?;
            let desc = RenderPipelineDesc {
                depth_format: WMTPixelFormat::WMTPixelFormatDepth32Float.0,
                topology_class: 3,
                sample_count: 1,
                ..Default::default()
            };
            let pso = device.geometry_render_pipeline_desc(
                (&olib, &v.shader.shader.entry), (&mlib, &g.shader.entry), &sin, None, &desc, g.config,
            )?;
            let dss = device.depth_stencil_state(&DepthStencilDesc {
                depth_compare: WMTCompareFunction::WMTCompareFunctionLess.0 as u8,
                depth_write: true,
                stencil: false,
                ..Default::default()
            })?;
            let dtex = device.new_texture(&TextureDescriptor {
                pixel_format: WMTPixelFormat::WMTPixelFormatDepth32Float, width: W, height: H, depth: 1, array_length: 1,
                type_: WMTTextureType::WMTTextureType2D, mipmap_level_count: 1, sample_count: 1,
                usage: WMTTextureUsage(WMTTextureUsage::WMTTextureUsageRenderTarget.0 | WMTTextureUsage::WMTTextureUsageShaderRead.0),
                options: WMTResourceOptions::WMTResourceStorageModeShared,
            })?;
            let draw = g.config.draw(9, 1, 0, 0, None).unwrap();
            let mut upload = [0u8; 1280];
            upload[256..276].copy_from_slice(&draw.args);
            upload[512..536].copy_from_slice(&draw.info);
            let buf = device.shared_buffer(&upload)?;
            residency.add(&[dtex.handle(), buf.handle()]);
            residency.commit();
            let mut commands = vec![
                Render::Pipeline { pipeline: &pso },
                Render::DepthStencil { state: &dss, stencil_ref: 0 },
            ];
            bind_common(&mut commands, &buf, None);
            commands.push(Render::DrawMesh { groups: draw.groups, object_threads: draw.object_threads, mesh_threads: draw.mesh_threads });
            let pass = RenderPass {
                depth: Some(PassDepth {
                    texture: &dtex, level: 0, slice: 0,
                    load: WMTLoadAction::WMTLoadActionClear,
                    store: WMTStoreAction::WMTStoreActionStore,
                    clear_depth: 1.0,
                    clear_stencil: 0,
                }),
                width: W,
                height: H,
                samples: 1,
                ..Default::default()
            };
            let cb = queue.command_buffer()?;
            cb.render_pass_with(&pass, &commands)?;
            cb.commit();
            cb.wait()?;
            let raw = readback_bytes(&dtex, 4);
            for y in 0..H {
                for x in 0..W {
                    let i = ((y * W + x) * 4) as usize;
                    let d = f32::from_le_bytes(raw[i..i + 4].try_into().unwrap());
                    let inside = (8..24).contains(&y) && ((4..20).contains(&x) || (24..40).contains(&x) || (44..60).contains(&x));
                    let expect = if inside { 0.5 } else { 1.0 };
                    assert!((d - expect).abs() < 1e-4, "Case 5 depth mismatch at ({x}, {y}): got {d}, expect {expect}");
                }
            }
            println!("  PASS Case 5: 4096/4096 Depth32Float texels (0.5 inside 3 rects, 1.0 background)");
        }
        println!("ALL 5 F37 SUITE CASES PASSED ON {}", device.name());
        Ok(())
    }

    let mut args = std::env::args_os().skip(1);
    let first = args.next().expect("usage: rectangle_probe VS GS PS | rectangle_probe --suite DIR");
    if first == "--suite" {
        let dir = PathBuf::from(args.next().expect("missing suite dir"));
        return run_suite(&dir);
    }
    let vs = std::fs::read(first)?;
    let gs = std::fs::read(args.next().expect("missing GS"))?;
    let ps = std::fs::read(args.next().expect("missing PS"))?;
    let _pool = Pool::new();
    let device = Device::first()?;
    let queue = device.new_queue(4)?;
    let residency = ResidencySet::new(&device, 16)?;
    let tex = device.new_texture(&TextureDescriptor {
        pixel_format: FORMAT, width: W, height: H, depth: 1, array_length: 1,
        type_: WMTTextureType::WMTTextureType2D, mipmap_level_count: 1, sample_count: 1,
        usage: WMTTextureUsage::WMTTextureUsageRenderTarget,
        options: WMTResourceOptions::WMTResourceStorageModeShared,
    })?;
    residency.add(&[tex.handle()]);
    residency.commit();
    queue.add_residency_set(&residency);
    run_baseline(&device, &queue, &residency, &tex, &vs, &gs, &ps)
}
