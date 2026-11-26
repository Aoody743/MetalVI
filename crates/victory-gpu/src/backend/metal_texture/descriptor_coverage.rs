use super::*;
use victory_metal::ir::ResourceType;
use victory_metal::msc::{self, RootParam};
use victory_metal::render::{PassColor, Render, RenderPass};
use victory_metal::sampler::SamplerDescriptor;
use victory_metal::sys::{WMTPixelFormat as P, WMTResourceOptions as Storage, WMTTextureType as T};
use victory_metal::{Compute, Device, Object, Pool, Queue, ResidencySet, Result};

const LEVELS: u32 = 4;
const SIDE: u32 = 8;
const FORMAT: P = P::WMTPixelFormatRGBA8Unorm;
const MAP: u32 = 0x1688;

fn rgba(mip: u32, slice: u32, x: u32, y: u32, z: u32) -> [u8; 4] {
    let v = (13 * mip + 31 * slice + 3 * x + 5 * y + 7 * z) % 224;
    [v as u8 + 1, v as u8 + 3, v as u8 + 5, v as u8 + 7]
}

fn norm(v: [u8; 4]) -> [f32; 4] {
    v.map(|n| f32::from(n) / 255.0)
}

fn make_texture(device: &Device, queue: &Queue, ty: T, slices: u32) -> Result<MetalTexture> {
    let one_d = matches!(ty, T::WMTTextureType1D | T::WMTTextureType1DArray);
    let three_d = ty == T::WMTTextureType3D;
    let ms = ty == T::WMTTextureType2DMultisample;
    let depth = if three_d { SIDE } else { 1 };
    let mips = if one_d || ms { 1 } else { LEVELS };
    let height = if one_d { 1 } else { SIDE };
    let samples = if ms { 4 } else { 1 };
    let texture = device.new_texture(&TextureDescriptor {
        pixel_format: FORMAT,
        type_: if one_d {
            if slices > 1 {
                T::WMTTextureType2DArray
            } else {
                T::WMTTextureType2D
            }
        } else {
            ty
        },
        mipmap_level_count: mips,
        sample_count: samples,
        width: SIDE,
        height,
        depth,
        array_length: slices,
        usage: WMTTextureUsage(1 | 16 | if ms { 4 } else { 2 }),
        options: Storage::WMTResourceStorageModePrivate,
    })?;
    let cb = queue.command_buffer()?;
    if ms {
        let source = r#"
#include <metal_stdlib>
using namespace metal;
vertex float4 v(uint n [[vertex_id]]) {
    return float4(n == 1 ? 3 : -1, n == 2 ? 3 : -1, 0, 1);
}
fragment float4 f(float4 p [[position]], uint s [[sample_id]]) {
    uint x = 32 + 20 * s + 3 * uint(p.x) + 5 * uint(p.y);
    return float4(x, x+1, x+2, x+3) / 255.0;
}
"#;
        let library = device.library_msl(source)?;
        let pipeline =
            device.render_pipeline(&library, "v", &library, "f", FORMAT, samples as u8)?;
        let mut pass = RenderPass {
            width: SIDE,
            height,
            samples: samples as u8,
            ..Default::default()
        };
        pass.colors[0] = Some(PassColor {
            texture: &texture,
            level: 0,
            slice: 0,
            load: sys::WMTLoadAction::WMTLoadActionClear,
            store: sys::WMTStoreAction::WMTStoreActionStore,
            clear: [0.0; 4],
        });
        let viewport = sys::WMTViewport {
            originX: 0.0,
            originY: 0.0,
            width: SIDE.into(),
            height: height.into(),
            znear: 0.0,
            zfar: 1.0,
        };
        cb.render_pass_with(
            &pass,
            &[
                Render::Pipeline {
                    pipeline: &pipeline,
                },
                Render::Viewports {
                    viewports: &[viewport],
                },
                Render::Draw {
                    primitive_type: sys::WMTPrimitiveType::WMTPrimitiveTypeTriangle,
                    vertex_start: 0,
                    vertex_count: 3,
                    instance_count: 1,
                    base_instance: 0,
                },
            ],
        )?;
    } else {
        let mut data = Vec::new();
        let mut copies = Vec::new();
        for slice in 0..slices {
            for mip in 0..mips {
                let (w, h, d) = (SIDE >> mip, (height >> mip).max(1), (depth >> mip).max(1));
                let off = data.len();
                let row = 256usize;
                let image = row * h as usize;
                data.resize(off + image * d as usize, 0);
                for z in 0..d {
                    for y in 0..h {
                        for x in 0..w {
                            let at = off + z as usize * image + y as usize * row + x as usize * 4;
                            data[at..at + 4].copy_from_slice(&rgba(mip, slice, x, y, z));
                        }
                    }
                }
                copies.push((
                    slice,
                    mip,
                    off as u64,
                    row as u32,
                    image as u32,
                    [w.into(), h.into(), d.into()],
                ));
            }
        }
        let buffer = device.shared_buffer(&data)?;
        let mut blit = cb.blit_encoder()?;
        for (slice, mip, off, row, image, size) in copies {
            blit.copy_from_buffer_to_texture(
                &buffer, off, row, image, size, &texture, slice, mip, [0; 3],
            );
        }
        blit.finish();
    }
    cb.commit();
    cb.wait()?;
    // SAFETY: ResourceDesc is an integer-only C record.
    let mut desc: ResourceDesc = unsafe { std::mem::zeroed() };
    desc.dimension = if one_d {
        2
    } else if three_d {
        4
    } else {
        3
    };
    desc.width = SIDE.into();
    desc.height = height;
    desc.depth_or_array_size = if three_d { depth as u16 } else { slices as u16 };
    desc.mip_levels = mips as u16;
    desc.format = 28;
    desc.sample_count = samples;
    desc.flags = if ms { 1 } else { 4 };
    Ok(MetalTexture {
        texture,
        desc,
        pixel: FORMAT,
        mips,
        slices,
    })
}

fn srv(ty: u32, mip: u32, count: u32, slice: u32, slices: u32, clamp: f32) -> [u32; 10] {
    let mut words = [28, ty, MAP, 0, mip, count, 0, 0, 0, 0];
    let lod = match ty {
        3 | 10 => {
            words[6] = slice;
            words[7] = slices;
            8
        }
        5 => {
            words[6] = slice;
            words[7] = slices;
            9
        }
        _ => 6,
    };
    words[lod] = clamp.to_bits();
    words
}

fn uav(ty: u32, mip: u32, start: u32, count: u32) -> [u32; 10] {
    [28, ty, mip, start, count, 0, 0, 0, 0, 0]
}

#[test]
#[ignore = "requires DXC-generated F59_DXIL and an isolated victorymetal Wine installation"]
fn gpu_descriptor_coverage() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let path = std::env::var_os("F59_DXIL").expect("path to descriptor_coverage.dxil");
    let filter = std::env::var("F59_CASE").unwrap_or_default();
    let dxil = std::fs::read(path)?;
    let _pool = Pool::new();
    let backend = super::super::metal::backend().expect("VICTORY_METAL=compute");
    let (device, queue) = (&backend.device, &backend.queue);
    let strict = std::env::var("F59_STRICT").is_ok_and(|v| v == "1");
    let residency = ResidencySet::new(&device, 64)?;
    queue.add_residency_set(&residency);
    let sampler = device.new_sampler_state(&SamplerDescriptor {
        mip_filter: sys::WMTSamplerMipFilter::WMTSamplerMipFilterNearest,
        support_argument_buffers: true,
        ..Default::default()
    })?;
    let sampler_buf = device.shared_buffer(entry_bytes(&DescriptorEntry::sampler(
        sampler.gpu_resource_id(),
        0.0,
    )))?;
    // A Metal-owned output buffer accessed through a root address, not a no-copy wrapper.
    let mut out = sys::WMTBufferInfo {
        length: 64,
        options: Storage::WMTResourceStorageModeShared,
        memory: sys::WMTMemoryPointer {
            ptr: core::ptr::null_mut(),
        },
        gpu_address: 0,
    };
    // SAFETY: newBuffer owns the allocation and populates the CPU/GPU addresses.
    let output =
        Object::adopt(unsafe { sys::MTLDevice_newBuffer(device.handle(), &mut out) }).unwrap();
    let output_mem = out.memory.ptr.cast::<[f32; 4]>();
    assert!(!output_mem.is_null());
    residency.add(&[output.handle()]);
    let params = [
        RootParam::Constants {
            register: 0,
            space: 0,
            num: 8,
        },
        RootParam::Uav {
            register: 0,
            space: 1,
        },
    ];
    let (pso, mut converted) = device.compute_pipeline_dxil(&dxil, &params, &[])?;
    let tg = converted.threadgroup.map(u64::from);
    let top = &mut converted.top_level;
    assert!(top.set_root_descriptor(ResourceType::Uav, 1, 0, out.gpu_address));
    let mut failures = Vec::new();
    let mut ran = 0;
    let mut known = 0;

    let mut probe = |name: &str,
                     texture: &MetalTexture,
                     words: Option<[u32; 10]>,
                     kind: u32,
                     args: [u32; 4],
                     expected: [f32; 4]|
     -> Result<()> {
        if !name.contains(&filter) {
            return Ok(());
        }
        println!("CASE {name}");
        ran += 1;
        let _pool = Pool::new();
        let unordered = (10..=14).contains(&kind);
        let words = words.as_ref().map(|w| &w[..]);
        let Some((view, entry)) = backend.make_entry_or_sink(texture, words, unordered) else {
            failures.push(format!("{name}: no descriptor"));
            return Ok(());
        };
        let mut heap = [0u8; 6 * size_of::<DescriptorEntry>()];
        heap[5 * size_of::<DescriptorEntry>()..].copy_from_slice(entry_bytes(&entry));
        let heap = device.shared_buffer(&heap)?;
        residency.add(&[texture.texture.handle()]);
        residency.commit();
        let [x, y, z, lod] = args;
        assert!(top.set_constants(0, 0, &[kind, 5, x, y, z, lod, 0, 0]));
        unsafe { output_mem.write([f32::NAN; 4]) };
        let cb = queue.command_buffer()?;
        cb.compute(&[
            Compute::Pipeline {
                pipeline: &pso,
                threads_per_group: tg,
            },
            Compute::Buffer {
                buffer: &heap,
                offset: 0,
                index: msc::DESCRIPTOR_HEAP_BIND_POINT,
            },
            Compute::Buffer {
                buffer: &sampler_buf,
                offset: 0,
                index: msc::SAMPLER_HEAP_BIND_POINT,
            },
            Compute::Bytes {
                bytes: &top.bytes,
                index: msc::ARGUMENT_BUFFER_BIND_POINT,
            },
            Compute::Dispatch { groups: [1, 1, 1] },
        ])?;
        cb.commit();
        let waited = cb.wait();
        residency.remove(&[texture.texture.handle()]);
        residency.commit();
        waited?;
        let actual = unsafe { output_mem.read() };
        if actual
            .iter()
            .zip(expected)
            .all(|(&a, e)| (a - e).abs() <= 0.0001)
        {
            println!("PASS {name}");
        } else if !strict
            && name.ends_with("clamp")
            && actual
                .iter()
                .zip(norm(rgba(u32::from(name.contains("rebased")), 0, 0, 0, 0)))
                .all(|(&a, e)| (a - e).abs() <= 0.0001)
        {
            known += 1;
            println!("KNOWN {name}: {actual:?} != {expected:?}");
        } else {
            failures.push(format!("{name}: {actual:?} != {expected:?}"));
        }
        drop(view);
        Ok(())
    };

    let d1 = make_texture(&device, &queue, T::WMTTextureType1D, 1)?;
    let a1 = make_texture(&device, &queue, T::WMTTextureType1DArray, 3)?;
    let d2 = make_texture(&device, &queue, T::WMTTextureType2D, 1)?;
    let a2 = make_texture(&device, &queue, T::WMTTextureType2DArray, 18)?;
    let d3 = make_texture(&device, &queue, T::WMTTextureType3D, 1)?;
    let ms = make_texture(&device, &queue, T::WMTTextureType2DMultisample, 1)?;

    probe(
        "srv 1d",
        &d1,
        Some(srv(2, 0, 1, 0, 0, 0.0)),
        0,
        [3, 0, 0, 0],
        norm(rgba(0, 0, 3, 0, 0)),
    )?;
    probe(
        "srv 1d array rebased",
        &a1,
        Some(srv(3, 0, 1, 1, 2, 0.0)),
        1,
        [3, 0, 1, 0],
        norm(rgba(0, 2, 3, 0, 0)),
    )?;
    probe(
        "srv 2d default all",
        &d2,
        None,
        2,
        [1, 2, 0, 1],
        norm(rgba(1, 0, 1, 2, 0)),
    )?;
    probe(
        "srv 2d rebased mips",
        &d2,
        Some(srv(4, 1, u32::MAX, 0, 0, 0.0)),
        2,
        [1, 1, 0, 1],
        norm(rgba(2, 0, 1, 1, 0)),
    )?;
    probe(
        "srv 2d array mip and slices",
        &a2,
        Some(srv(5, 1, 2, 4, 3, 0.0)),
        3,
        [1, 1, 1, 1],
        norm(rgba(2, 5, 1, 1, 0)),
    )?;
    probe(
        "srv 3d mip and z",
        &d3,
        Some(srv(8, 1, 2, 0, 0, 0.0)),
        4,
        [1, 1, 1, 1],
        norm(rgba(2, 0, 1, 1, 1)),
    )?;
    for face in 0..6 {
        probe(
            &format!("srv cube face {face}"),
            &a2,
            Some(srv(9, 3, 1, 0, 6, 0.0)),
            5,
            [face, 0, 0, 0],
            norm(rgba(3, face, 0, 0, 0)),
        )?;
        probe(
            &format!("srv cube array face {face}"),
            &a2,
            Some(srv(10, 3, 1, 6, 2, 0.0)),
            6,
            [face, 0, 1, 0],
            norm(rgba(3, 12 + face, 0, 0, 0)),
        )?;
    }
    for sample in 0..4 {
        let v = (32 + 20 * sample + 3 + 10) as u8;
        probe(
            &format!("srv 2dms sample {sample}"),
            &ms,
            Some(srv(6, 0, 1, 0, 0, 0.0)),
            7,
            [1, 2, sample, 0],
            norm([v, v + 1, v + 2, v + 3]),
        )?;
    }
    let mut swizzled = srv(4, 1, 1, 0, 0, 0.0);
    swizzled[2] = (1 << 12) | 2 | (0 << 3) | (5 << 6) | (4 << 9);
    let [r, _, b, _] = norm(rgba(1, 0, 1, 2, 0));
    probe(
        "srv swizzle B R 1 0",
        &d2,
        Some(swizzled),
        2,
        [1, 2, 0, 0],
        [b, r, 1.0, 0.0],
    )?;
    let mut uint = srv(4, 0, 1, 0, 0, 0.0);
    uint[0] = 30; // R8G8B8A8_UINT view of R8G8B8A8_UNORM.
    probe(
        "srv uint reinterpretation",
        &d2,
        Some(uint),
        8,
        [1, 2, 0, 0],
        rgba(0, 0, 1, 2, 0).map(f32::from),
    )?;
    probe(
        "srv sample level control",
        &d2,
        Some(srv(4, 0, 3, 0, 0, 0.0)),
        9,
        [0; 4],
        norm(rgba(0, 0, 0, 0, 0)),
    )?;
    probe(
        "srv sample level clamp",
        &d2,
        Some(srv(4, 0, 3, 0, 0, 1.0)),
        9,
        [0; 4],
        norm(rgba(1, 0, 0, 0, 0)),
    )?;
    probe(
        "srv sample level rebased clamp",
        &d2,
        Some(srv(4, 1, 3, 0, 0, 2.0)),
        9,
        [0; 4],
        norm(rgba(2, 0, 0, 0, 0)),
    )?;
    probe(
        "srv sample grad clamp",
        &d2,
        Some(srv(4, 0, 3, 0, 0, 1.0)),
        9,
        [0, 1, 0, 0],
        norm(rgba(1, 0, 0, 0, 0)),
    )?;
    probe(
        "srv sample grad rebased clamp",
        &d2,
        Some(srv(4, 1, 3, 0, 0, 2.0)),
        9,
        [0, 1, 0, 0],
        norm(rgba(2, 0, 0, 0, 0)),
    )?;
    probe(
        "srv sample implicit clamp",
        &d2,
        Some(srv(4, 0, 3, 0, 0, 1.0)),
        15,
        [0; 4],
        norm(rgba(1, 0, 0, 0, 0)),
    )?;

    probe(
        "uav 1d",
        &d1,
        Some(uav(2, 0, 0, 0)),
        10,
        [2, 0, 0, 0],
        norm(rgba(0, 0, 2, 0, 0)),
    )?;
    probe(
        "uav 1d array",
        &a1,
        Some(uav(3, 0, 1, 1)),
        11,
        [2, 0, 0, 0],
        norm(rgba(0, 1, 2, 0, 0)),
    )?;
    probe(
        "uav 2d",
        &d2,
        Some(uav(4, 2, 0, 0)),
        12,
        [1, 1, 0, 0],
        norm(rgba(2, 0, 1, 1, 0)),
    )?;
    probe(
        "uav 2d array",
        &a2,
        Some(uav(5, 1, 5, 2)),
        13,
        [2, 1, 1, 0],
        norm(rgba(1, 6, 2, 1, 0)),
    )?;
    probe(
        "uav 3d full",
        &d3,
        Some(uav(8, 1, 0, u32::MAX)),
        14,
        [1, 1, 0, 0],
        norm(rgba(1, 0, 1, 1, 0)),
    )?;
    assert!(make_texture_entry(&d3, Some(&uav(8, 1, 1, 2)), true).is_none());
    probe(
        "uav 3d partial w isolated",
        &d3,
        Some(uav(8, 1, 1, 2)),
        14,
        [0; 4],
        [0.0; 4],
    )?;
    probe(
        "uav 3d partial w preserves source",
        &d3,
        Some(srv(8, 1, 1, 0, 0, 0.0)),
        4,
        [0; 4],
        norm(rgba(1, 0, 0, 0, 0)),
    )?;
    drop(probe);
    residency.remove(&[output.handle()]);
    residency.commit();
    assert!(ran > 0, "filter did not match any case");
    for fail in &failures {
        println!("FAIL {fail}");
    }
    assert!(
        failures.is_empty(),
        "{} of {ran} descriptor probes mismatch",
        failures.len()
    );
    println!("{} verified, {known} known mismatches", ran - known);
    Ok(())
}

fn entry_bytes(entry: &DescriptorEntry) -> &[u8] {
    // SAFETY: three initialized u64s, with no padding.
    unsafe {
        core::slice::from_raw_parts(
            core::ptr::from_ref(entry).cast(),
            size_of::<DescriptorEntry>(),
        )
    }
}
