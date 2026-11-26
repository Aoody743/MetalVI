//! F35: draws two instances with an interleaved POSITION slot and a
//! separate per-instance TEXCOORD slot, using a synthesized stage-in.
//! Exercises direct, 16-bit indexed, and 32-bit indexed draws.

#[cfg(not(windows))]
fn main() {}

#[cfg(windows)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::ffi::c_void;
    use victory_metal::msc::{self, Stage, VertexBufferEntry, VertexInput};
    use victory_metal::render::{ColorAttachment, ColorDesc, Render, RenderPipelineDesc};
    use victory_metal::sys::*;
    use victory_metal::texture::TextureDescriptor;
    use victory_metal::{Device, Pool, ResidencySet};

    const W: u32 = 64;
    const H: u32 = 64;
    const FORMAT: WMTPixelFormat = WMTPixelFormat::WMTPixelFormatBGRA8Unorm;
    let vs = std::fs::read(std::env::args_os().nth(1).ok_or("missing vertex DXIL")?)?;
    let ps = std::fs::read(std::env::args_os().nth(2).ok_or("missing fragment DXIL")?)?;
    let _pool = Pool::new();
    let device = Device::first()?;
    let inputs = [
        VertexInput { semantic: "POSITION", semantic_index: 0, format: 16, slot: 0, offset: 4, instance_step_rate: None },
        VertexInput { semantic: "TEXCOORD", semantic_index: 0, format: 2, slot: 1, offset: 4, instance_step_rate: Some(1) },
    ];
    let vs = msc::convert_vertex_with_stage_in(&vs, &[], &[], &inputs)?;
    let ps = msc::convert(&ps, Stage::Fragment, &[], &[])?;
    let vlib = device.library(&vs.shader.metallib)?;
    let plib = device.library(&ps.metallib)?;
    let helper = device.library(&vs.stage_in)?;
    let mut desc = RenderPipelineDesc { topology_class: 3, sample_count: 1, ..Default::default() };
    desc.colors[0] = ColorDesc { pixel_format: FORMAT.0, write_mask: 15, ..Default::default() };
    let pso = device.render_pipeline_stage_in((&vlib, &vs.shader.entry), Some((&plib, &ps.entry)), &desc, &helper)?;

    let upload = |words: &[f32], stride: u32| -> Result<_, Box<dyn std::error::Error>> {
        let bytes: Vec<_> = words.iter().copied().flat_map(f32::to_le_bytes).collect();
        let buffer = device.shared_buffer(&bytes)?;
        let entry = VertexBufferEntry { address: buffer.gpu_address, length: bytes.len() as u32, stride };
        Ok((buffer, entry))
    };
    let (verts, positions) = upload(&[9., 5., 5., 9., 9., -0.3, -0.4, 9., 9., 0.3, -0.4, 9., 9., 0., 0.4, 9.], 16)?;
    let (instances, instance_data) = upload(&[9., 5., 5., 0., 0., 9., 9., -0.5, 0., 1., 0., 9., 9., 0.5, 0., 0., 1., 9.], 24)?;
    let mut entries = [VertexBufferEntry::default(); msc::VERTEX_TABLE_SLOTS];
    entries[0] = positions;
    entries[1] = instance_data;
    let table = msc::vertex_table_bytes(&entries);
    let r16 = device.shared_buffer(&[5u16, 5, 6, 7, 8].into_iter().flat_map(u16::to_le_bytes).collect::<Vec<_>>())?;
    let r32 = device.shared_buffer(&[5u32, 5, 6, 7, 8].into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>())?;
    let target = device.new_texture(&TextureDescriptor {
        pixel_format: FORMAT,
        width: W,
        height: H,
        depth: 1,
        array_length: 1,
        type_: WMTTextureType::WMTTextureType2D,
        mipmap_level_count: 1,
        sample_count: 1,
        usage: WMTTextureUsage::WMTTextureUsageRenderTarget,
        options: WMTResourceOptions::WMTResourceStorageModeShared,
    })?;
    let queue = device.new_queue(4)?;
    let residency = ResidencySet::new(&device, 5)?;
    residency.add(&[verts.handle(), instances.handle(), r16.handle(), r32.handle(), target.handle()]);
    residency.commit();
    queue.add_residency_set(&residency);
    let primitive_type = WMTPrimitiveType::WMTPrimitiveTypeTriangle;
    let mut image = vec![0u8; (W * H * 4) as usize];
    for (name, index) in [("direct", None), ("R16_UINT", Some((&r16, 4, 0u8))), ("R32_UINT", Some((&r32, 8, 1)))] {
        let (args, index_uniform, draw) = if let Some((index_buffer, index_offset, index_type)) = index {
            let base_vertex = -5i32;
            (
                [3, 2, index_offset as u32, base_vertex as u32, 1],
                u16::from(index_type) + 1,
                Render::DrawIndexed {
                    primitive_type,
                    index_type,
                    index_count: 3,
                    index_buffer,
                    index_offset,
                    instance_count: 2,
                    base_vertex,
                    base_instance: 1,
                },
            )
        } else {
            ([3, 2, 1, 1, 0], 0, Render::Draw {
                primitive_type,
                vertex_start: 1,
                vertex_count: 3,
                instance_count: 2,
                base_instance: 1,
            })
        };
        let draw_args: Vec<_> = args.into_iter().flat_map(u32::to_le_bytes).collect();
        let command = queue.command_buffer()?;
        command.render_pass(
            &[ColorAttachment {
                texture: &target,
                load_action: WMTLoadAction::WMTLoadActionClear,
                store_action: WMTStoreAction::WMTStoreActionStore,
                clear: [0., 0., 1., 1.],
            }],
            W,
            H,
            &[
                Render::Pipeline { pipeline: &pso },
                Render::VertexBytes { bytes: &table, index: msc::VERTEX_TABLE_BIND_POINT },
                Render::VertexBytes { bytes: &draw_args, index: msc::DRAW_ARGUMENTS_BIND_POINT },
                Render::VertexBytes { bytes: &index_uniform.to_le_bytes(), index: msc::DRAW_UNIFORMS_BIND_POINT },
                draw,
            ],
        )?;
        command.commit();
        command.wait()?;
        // SAFETY: callback fills the synchronous readback into the entire slice.
        unsafe {
            MTLTexture_getBytes(
                target.handle(),
                WMTOrigin { x: 0, y: 0, z: 0 },
                WMTSize { width: W.into(), height: H.into(), depth: 1 },
                0,
                0,
                WMTMemoryPointer { ptr: image.as_mut_ptr().cast::<c_void>() },
                u64::from(W * 4),
                image.len() as u64,
            )
        };
        for (x, y, want) in [(16, 32, [0, 0, 255, 255]), (48, 32, [0, 255, 0, 255]), (32, 32, [255, 0, 0, 255])] {
            let start = ((y * W + x) * 4) as usize;
            let actual = &image[start..start + 4];
            println!("{name} pixel ({x}, {y}): {actual:?}, expected {want:?}");
            if actual != want {
                return Err(format!("{name} stage-in pixel mismatch at ({x}, {y})").into());
            }
        }
    }
    println!("PASS: stage-in fetched both slots and advanced per-instance data on {}", device.name());
    Ok(())
}
