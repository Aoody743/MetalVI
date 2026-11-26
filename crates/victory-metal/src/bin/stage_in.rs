//! F35: draws two instances with an interleaved POSITION slot and a
//! separate per-instance TEXCOORD slot, using a synthesized stage-in.

#[cfg(not(windows))]
fn main() {}

#[cfg(windows)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::ffi::c_void;
    use victory_metal::msc::{self, Stage, VertexBufferEntry, VertexInput};
    use victory_metal::render::{ColorAttachment, ColorDesc, Render, RenderPipelineDesc};
    use victory_metal::sys::*;
    use victory_metal::texture::TextureDescriptor;
    use victory_metal::{Device, Object, Pool, ResidencySet};

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
        let n = std::mem::size_of_val(words);
        let mut info = WMTBufferInfo {
            length: n as u64,
            options: WMTResourceOptions::WMTResourceStorageModeShared,
            memory: WMTMemoryPointer { ptr: std::ptr::null_mut() },
            gpu_address: 0,
        };
        // SAFETY: the descriptor outlives the call; the allocation is owned.
        let buffer = Object::adopt(unsafe { MTLDevice_newBuffer(device.handle(), &mut info) }).ok_or("new buffer")?;
        if info.memory.ptr.is_null() {
            return Err("shared buffer has no CPU pointer".into());
        }
        // SAFETY: this shared allocation has exactly n bytes.
        unsafe { std::ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), info.memory.ptr.cast(), n) };
        Ok((buffer, VertexBufferEntry { address: info.gpu_address, length: n as u32, stride }))
    };
    let (verts, positions) = upload(&[9., -0.3, -0.4, 9., 9., 0.3, -0.4, 9., 9., 0., 0.4, 9.], 16)?;
    let (instances, instance_data) = upload(&[9., -0.5, 0., 1., 0., 9., 9., 0.5, 0., 0., 1., 9.], 24)?;
    let mut entries = [VertexBufferEntry::default(); msc::VERTEX_TABLE_SLOTS];
    entries[0] = positions;
    entries[1] = instance_data;
    let table = msc::vertex_table_bytes(&entries);
    let draw_args: Vec<_> = [3u32, 2, 0, 0, 0].into_iter().flat_map(u32::to_le_bytes).collect();
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
    let residency = ResidencySet::new(&device, 3)?;
    residency.add(&[verts.handle(), instances.handle(), target.handle()]);
    residency.commit();
    queue.add_residency_set(&residency);
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
            Render::VertexBytes { bytes: &[0; 2], index: msc::DRAW_UNIFORMS_BIND_POINT },
            Render::Draw {
                primitive_type: WMTPrimitiveType::WMTPrimitiveTypeTriangle,
                vertex_start: 0,
                vertex_count: 3,
                instance_count: 2,
                base_instance: 0,
            },
        ],
    )?;
    command.commit();
    command.wait()?;
    let mut image = vec![0u8; (W * H * 4) as usize];
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
        println!("pixel ({x}, {y}): {actual:?}, expected {want:?}");
        if actual != want {
            return Err(format!("stage-in pixel mismatch at ({x}, {y})").into());
        }
    }
    println!("PASS: stage-in fetched both slots and advanced per-instance data on {}", device.name());
    Ok(())
}
