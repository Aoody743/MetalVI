use super::*;
use crate::backend::metal::{MetalOp, backend};
use crate::backend::metal_draw::{Attachment, MetalClear, MetalTexCopy};
use crate::backend::metal_texture::MetalTexture;
use std::sync::Arc;
use victory_metal::sys::{
    WMTPixelFormat as F, WMTResourceOptions, WMTTextureType as T, WMTTextureUsage,
};
use victory_metal::texture::TextureDescriptor;
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
};

#[test]
#[ignore = "requires private Wine, Metal validation and VICTORY_METAL=compute VICTORY_METAL_DRAW=1"]
fn records_and_replays_clipped_resolve_before_copy() {
    let _pool = victory_metal::Pool::new();
    let m = backend().unwrap();
    let pixel = F::WMTPixelFormatR32Float;
    let make = |width, height, mips, slices, samples| {
        let ty = if samples > 1 {
            T::WMTTextureType2DMultisample
        } else if slices > 1 {
            T::WMTTextureType2DArray
        } else {
            T::WMTTextureType2D
        };
        let texture = m
            .device
            .new_texture(&TextureDescriptor {
                pixel_format: pixel,
                width,
                height,
                mipmap_level_count: mips,
                array_length: slices,
                sample_count: samples,
                type_: ty,
                usage: WMTTextureUsage(if samples > 1 { 5 } else { 1 }),
                options: WMTResourceOptions::WMTResourceStorageModePrivate,
                ..Default::default()
            })
            .unwrap();
        // SAFETY: ResourceDesc is an integer-only record.
        let mut desc: crate::resource::ResourceDesc = unsafe { std::mem::zeroed() };
        desc.width = width.into();
        desc.height = height;
        desc.dimension = 3;
        desc.format = 41;
        desc.flags = u32::from(samples > 1);
        desc.sample_count = samples;
        Arc::new(MetalTexture {
            texture,
            pixel,
            desc,
            mips,
            slices,
            companion: std::sync::OnceLock::new(),
            twin: None,
        })
    };
    let src = make(8, 8, 1, 1, 4);
    let dst = make(12, 8, 2, 2, 1);
    let out = make(6, 4, 1, 1, 1);
    // These synthetic resource handles are looked up in the texture cache;
    // the test never enters the host D3D12 path or dereferences the handles.
    let src_handle = Arc::as_ptr(&src) as This;
    let dst_handle = Arc::as_ptr(&dst) as This;
    {
        let mut textures = m.textures.lock().unwrap();
        textures.insert(src_handle as usize, Some(src.clone()));
        textures.insert(dst_handle as usize, Some(dst.clone()));
    }
    let mut list = Box::new(List {
        object: Object::new(&LIST_FUNCTIONS),
        kind: 0,
        put: [0; 5],
        reserved: [0; 64],
        graphics_root_headers: [0; 128],
        compute_root_headers: [0; 128],
        pipeline_shadow: [0; 8],
        newer: [0; 256],
        uploads: Default::default(),
        host: None,
        state: Default::default(),
        earlier: Vec::new(),
        gate: Vec::new(),
    });
    let this = (&raw mut *list) as This;
    list.state
        .get_mut()
        .unwrap()
        .metal
        .push(Arc::new(MetalOp::Clear(MetalClear {
            target: Attachment {
                resource: src_handle as usize,
                texture: src.texture.clone(),
                pixel: pixel.0,
                level: 0,
                slice: 0,
                width: 8,
                height: 8,
                samples: 4,
                companion: None,
                twin: None,
            },
            color: Some([0.375, 0.0, 0.0, 0.0]),
            depth: None,
            stencil: None,
            has_stencil: false,
            rects: Vec::new(),
        })));
    resolve(this, dst_handle, 3, src_handle, 1, 41); // Invalid source plane.
    resolve(this, dst_handle, 3, src_handle, 0, 41);
    let mut ops = metal_commands(this);
    assert_eq!(ops.len(), 2);
    let MetalOp::Resolve(op) = &*ops[1] else {
        panic!("resolve was not recorded")
    };
    assert_eq!(
        (op.dst_level, op.dst_slice, op.region.unwrap().size),
        (1, 1, [6, 4])
    );
    ops.push(Arc::new(MetalOp::Copy(MetalTexCopy {
        src: dst,
        dst: out.clone(),
        whole: false,
        src_slice: 1,
        src_level: 1,
        src_origin: [0; 3],
        size: [6, 4, 1],
        dst_slice: 0,
        dst_level: 0,
        dst_origin: [0; 3],
    })));
    assert!(m.run(&ops));

    // SAFETY: fresh committed pages outlive the submitted buffer.
    let mem = unsafe {
        VirtualAlloc(
            std::ptr::null(),
            4096,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_READWRITE,
        )
    };
    assert!(!mem.is_null());
    let read = unsafe { m.device.wrap(mem, 4096) }.unwrap();
    let cb = m.queue.command_buffer().unwrap();
    let mut blit = cb.blit_encoder().unwrap();
    blit.copy_from_texture_to_buffer(&out.texture, 0, 0, [0; 3], [6, 4, 1], &read, 0, 256, 1024);
    blit.finish();
    cb.commit();
    cb.wait().unwrap();
    let data = unsafe { std::slice::from_raw_parts(mem.cast::<u8>(), 1024) };
    for y in 0..4 {
        for x in 0..6 {
            let offset = y * 256 + x * 4;
            assert_eq!(&data[offset..offset + 4], &0.375f32.to_le_bytes());
        }
    }
    drop(cb);
    drop(read);
    unsafe { VirtualFree(mem, 0, MEM_RELEASE) };
    let mut textures = m.textures.lock().unwrap();
    textures.remove(&(src_handle as usize));
    textures.remove(&(dst_handle as usize));
}
