//! End-to-end offscreen render test (run in Wine with victorymetal installed):
//! two fragment shaders translated from DXIL through the Metal Shader
//! Converter, drawn by a fullscreen triangle built from SV_VertexID (no vertex
//! buffer), rendered to a 64×64 BGRA8 offscreen target and read back pixel by
//! pixel.
//!
//! Case 1: the fragment shader reads only root constants (b0, space0).
//! Case 2: the fragment shader samples a 4×4 texture from the directly indexed
//! descriptor heap (bind point 0) with a static sampler; the sampler heap
//! (bind point 1) is populated and bound for parity.
//!
//! usage: victory-metal-render <vs.dxil> <ps_const.dxil> <ps_sample.dxil>

#[cfg(windows)]
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => {
            println!("RESULT: ALL CHECKS PASSED");
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            println!("RESULT: FAILED: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(windows)]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use victory_metal::ir::DescriptorEntry;
    use victory_metal::msc::{self, RootParam, Stage, StaticSampler};
    use victory_metal::render::{ColorAttachment, Render, RenderPipeline};
    use victory_metal::sampler::SamplerDescriptor;
    use victory_metal::texture::TextureDescriptor;
    use victory_metal::{Buffer, Device, Pool, ResidencySet};
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};

    const W: u32 = 64;
    const H: u32 = 64;
    const ROW: u64 = W as u64 * 4;
    const BYTES: usize = (W * H * 4) as usize;
    const HEAP_BYTES: usize = 64 << 10;
    const TEX: u32 = 4; // source texture 4×4

    let vs_path = std::env::args().nth(1).ok_or("usage: victory-metal-render <vs.dxil> <ps_const.dxil> <ps_sample.dxil>")?;
    let ps_const_path = std::env::args().nth(2).ok_or("missing ps_const.dxil")?;
    let ps_sample_path = std::env::args().nth(3).ok_or("missing ps_sample.dxil")?;
    let vs_dxil = std::fs::read(&vs_path)?;
    let ps_const_dxil = std::fs::read(&ps_const_path)?;
    let ps_sample_dxil = std::fs::read(&ps_sample_path)?;

    let _pool = Pool::new();
    let device = Device::first()?;
    println!("device: {}", device.name());
    let queue = device.new_queue(4)?;

    // ---- convert the vertex shader once (both cases share it) ----
    let vs = msc::convert(&vs_dxil, Stage::Vertex, &[], &[])?;
    println!("VS: entry {:?} metallib {} bytes", vs.entry, vs.metallib.len());
    let _ = std::fs::write(r"C:\f19\vs.metallib", &vs.metallib);
    let vs_lib = device.library(&vs.metallib)?;

    // Readback helper: copies the whole 2D slice into a fresh allocation.
    let readback = |texture: &victory_metal::texture::Texture, w: u32, h: u32| -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let n = (w * h * 4) as usize;
        let row = w as u64 * 4;
        let mem = unsafe { VirtualAlloc(std::ptr::null(), n, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        if mem.is_null() {
            return Err("VirtualAlloc readback".into());
        }
        // SAFETY: committed n bytes; getBytes fills the whole slice.
        unsafe {
            victory_metal::sys::MTLTexture_getBytes(
                texture.handle(),
                victory_metal::sys::WMTOrigin { x: 0, y: 0, z: 0 },
                victory_metal::sys::WMTSize { width: w as u64, height: h as u64, depth: 1 },
                0,
                0,
                victory_metal::sys::WMTMemoryPointer { ptr: mem },
                row,
                n as u64,
            );
        }
        // SAFETY: committed; copy out before the allocation is dropped at
        // process exit (the slice lives only for this function).
        Ok(unsafe { std::slice::from_raw_parts(mem.cast::<u8>(), n) }.to_vec())
    };

    // Expected BGRA8 bytes from an RGBA float color (round to nearest unorm8).
    let expect_bgra = |rgba: [f32; 4]| -> [u8; 4] {
        let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        [byte(rgba[2]), byte(rgba[1]), byte(rgba[0]), byte(rgba[3])]
    };

    // Compare `got` against `want` pixel by pixel; report mismatched pixel count
    // (any channel off by more than one unorm step) and the max channel error.
    let compare = |got: &[u8], want: &[[u8; 4]], label: &str| -> Result<(), Box<dyn std::error::Error>> {
        let mut wrong = 0usize;
        let mut max_err = 0u8;
        for i in 0..BYTES / 4 {
            for c in 0..4 {
                let e = got[i * 4 + c].abs_diff(want[i][c]);
                max_err = max_err.max(e);
            }
            if got[i * 4..i * 4 + 4] != want[i] {
                wrong += 1;
            }
        }
        println!("{label}: {} / {} pixels differ, max channel error {max_err}/255", wrong, BYTES / 4);
        if wrong != 0 {
            let first = (0..BYTES / 4).find(|&i| got[i * 4..i * 4 + 4] != want[i]).unwrap();
            println!("  first mismatch pixel {first}: got {:?}, want {:?}", &got[first * 4..first * 4 + 4], want[first]);
            return Err(format!("{label}: {wrong} pixels wrong").into());
        }
        Ok(())
    };

    let rt_desc = TextureDescriptor {
        pixel_format: victory_metal::sys::WMTPixelFormat::WMTPixelFormatBGRA8Unorm,
        width: W,
        height: H,
        depth: 1,
        array_length: 1,
        type_: victory_metal::sys::WMTTextureType::WMTTextureType2D,
        mipmap_level_count: 1,
        sample_count: 1,
        usage: victory_metal::sys::WMTTextureUsage(
            victory_metal::sys::WMTTextureUsage::WMTTextureUsageShaderRead.0
                | victory_metal::sys::WMTTextureUsage::WMTTextureUsageRenderTarget.0,
        ),
        options: victory_metal::sys::WMTResourceOptions::WMTResourceStorageModeShared,
    };

    // ============ Case 1: root constants only ============
    println!("--- case 1: root constants ---");
    let ps1 = msc::convert(
        &ps_const_dxil,
        Stage::Fragment,
        &[RootParam::Constants { register: 0, space: 0, num: 4 }],
        &[],
    )?;
    println!("PS const: entry {:?} metallib {} bytes, top-level {:?}", ps1.entry, ps1.metallib.len(), ps1.top_level.layout);
    let _ = std::fs::write(r"C:\f19\ps_const.metallib", &ps1.metallib);
    let ps1_lib = device.library(&ps1.metallib)?;
    let pso1: RenderPipeline = device.render_pipeline(
        &vs_lib,
        &vs.entry,
        &ps1_lib,
        &ps1.entry,
        victory_metal::sys::WMTPixelFormat::WMTPixelFormatBGRA8Unorm,
        1,
    )?;
    println!("pipeline 1 created");

    let color = [0.86f32, 0.32, 0.55, 1.0];
    let want1: Vec<[u8; 4]> = std::iter::repeat(expect_bgra(color)).take(BYTES / 4).collect();

    let rt1 = device.new_texture(&rt_desc)?;
    let mut top1 = ps1.top_level.clone();
    let color_bits: [u32; 4] = [
        color[0].to_bits(),
        color[1].to_bits(),
        color[2].to_bits(),
        color[3].to_bits(),
    ];
    if !top1.set_constants(0, 0, &color_bits) {
        return Err("no root constants at b0 space0 in the layout".into());
    }

    let residency = ResidencySet::new(&device, 8)?;
    residency.add(&[rt1.handle()]);
    residency.commit();
    queue.add_residency_set(&residency);

    let t0 = std::time::Instant::now();
    let cb = queue.command_buffer()?;
    cb.render_pass(
        &[ColorAttachment {
            texture: &rt1,
            load_action: victory_metal::sys::WMTLoadAction::WMTLoadActionClear,
            store_action: victory_metal::sys::WMTStoreAction::WMTStoreActionStore,
            clear: [0.0, 0.0, 0.0, 0.0],
        }],
        W,
        H,
        &[
            Render::Pipeline { pipeline: &pso1 },
            Render::FragmentBytes { bytes: &top1.bytes, index: msc::ARGUMENT_BUFFER_BIND_POINT },
            Render::Draw {
                primitive_type: victory_metal::sys::WMTPrimitiveType::WMTPrimitiveTypeTriangle,
                vertex_start: 0,
                vertex_count: 3,
                instance_count: 1,
                base_instance: 0,
            },
        ],
    )?;
    cb.commit();
    cb.wait()?;
    let case1_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("case 1 GPU time: {case1_ms:.2} ms");

    let got1 = readback(&rt1, W, H)?;
    compare(&got1, &want1, "case 1")?;

    // ============ Case 2: texture from descriptor heap + sampler ============
    println!("--- case 2: heap texture + sampler ---");

    // Source 4×4 pattern: B = x*64, G = y*64, R = 128, A = 255.
    let mut pattern = vec![0u8; (TEX * TEX * 4) as usize];
    for y in 0..TEX {
        for x in 0..TEX {
            let b = (x * 64) as u8;
            let g = (y * 64) as u8;
            let r = 128u8;
            let a = 255u8;
            let at = ((y * TEX + x) * 4) as usize;
            pattern[at..at + 4].copy_from_slice(&[b, g, r, a]);
        }
    }
    let tex_desc = TextureDescriptor {
        pixel_format: victory_metal::sys::WMTPixelFormat::WMTPixelFormatBGRA8Unorm,
        width: TEX,
        height: TEX,
        depth: 1,
        array_length: 1,
        type_: victory_metal::sys::WMTTextureType::WMTTextureType2D,
        mipmap_level_count: 1,
        sample_count: 1,
        usage: victory_metal::sys::WMTTextureUsage::WMTTextureUsageShaderRead,
        options: victory_metal::sys::WMTResourceOptions::WMTResourceStorageModeShared,
    };
    let src_tex = device.new_texture(&tex_desc)?;

    // Upload the pattern through a staging buffer (buffer→texture blit).
    let mem_staging = unsafe { VirtualAlloc(std::ptr::null(), pattern.len(), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
    if mem_staging.is_null() {
        return Err("VirtualAlloc staging".into());
    }
    unsafe { std::slice::from_raw_parts_mut(mem_staging.cast::<u8>(), pattern.len()).copy_from_slice(&pattern) };
    // SAFETY: staging lives until the blit finishes.
    let staging = unsafe { device.wrap(mem_staging, pattern.len() as u64)? };
    let cb = queue.command_buffer()?;
    let mut blit = cb.blit_encoder()?;
    blit.copy_from_buffer_to_texture(
        &staging,
        0,
        TEX * 4,
        TEX * 4 * TEX,
        [TEX as u64, TEX as u64, 1],
        &src_tex,
        0,
        0,
        [0, 0, 0],
    );
    blit.finish();
    cb.commit();
    cb.wait()?;

    // Debug: verify the pattern reached the GPU texture.
    let src_back = readback(&src_tex, TEX, TEX)?;
    println!("  debug src texel(0,0)={:?} (3,3)={:?}", &src_back[0..4], &src_back[12..16]);

    // Descriptor heap: slot 0 holds the source texture entry; rest empty.
    let mem_heap = unsafe { VirtualAlloc(std::ptr::null(), HEAP_BYTES, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
    let mem_sheap = unsafe { VirtualAlloc(std::ptr::null(), HEAP_BYTES, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
    if mem_heap.is_null() || mem_sheap.is_null() {
        return Err("VirtualAlloc heap".into());
    }
    // SAFETY: committed; heap memory holds HEAP_BYTES/24 entries.
    let dheap = unsafe { std::slice::from_raw_parts_mut(mem_heap.cast::<DescriptorEntry>(), HEAP_BYTES / 24) };
    let sheap = unsafe { std::slice::from_raw_parts_mut(mem_sheap.cast::<DescriptorEntry>(), HEAP_BYTES / 24) };
    dheap.fill(DescriptorEntry::NULL);
    sheap.fill(DescriptorEntry::NULL);

    let sampler = device.new_sampler_state(&SamplerDescriptor {
        support_argument_buffers: true,
        ..Default::default()
    })?;
    let tex_id = src_tex.gpu_resource_id();
    let samp_id = sampler.gpu_resource_id();
    if tex_id == 0 || samp_id == 0 {
        return Err(format!("zero gpu resource id (texture {tex_id:#x}, sampler {samp_id:#x})").into());
    }
    dheap[0] = DescriptorEntry::texture(tex_id, 0.0, 0);
    sheap[0] = DescriptorEntry::sampler(samp_id, 0.0);

    // SAFETY: the allocations outlive the buffers and command buffers.
    let heap_buf: Buffer = unsafe { device.wrap(mem_heap, HEAP_BYTES as u64)? };
    let sheap_buf: Buffer = unsafe { device.wrap(mem_sheap, HEAP_BYTES as u64)? };

    // IR enum numbering (metal_irconverter/*.h): point filter 0, clamp 3,
    // always 8, transparent black 0.
    let samplers = [StaticSampler {
        filter: 0,
        address_u: 3,
        address_v: 3,
        address_w: 3,
        mip_lod_bias: 0.0,
        max_anisotropy: 1,
        comparison_func: 8,
        border_color: 0,
        min_lod: 0.0,
        max_lod: 1000.0,
        shader_register: 0,
        register_space: 0,
    }];
    let ps2 = msc::convert(
        &ps_sample_dxil,
        Stage::Fragment,
        &[RootParam::TableSrv { register: 0, space: 1 }],
        &samplers,
    )?;
    println!("PS sample: entry {:?} metallib {} bytes, top-level {:?}", ps2.entry, ps2.metallib.len(), ps2.top_level.layout);
    let _ = std::fs::write(r"C:\f19\ps_sample.metallib", &ps2.metallib);
    let ps2_lib = device.library(&ps2.metallib)?;
    let pso2: RenderPipeline = device.render_pipeline(
        &vs_lib,
        &vs.entry,
        &ps2_lib,
        &ps2.entry,
        victory_metal::sys::WMTPixelFormat::WMTPixelFormatBGRA8Unorm,
        1,
    )?;
    println!("pipeline 2 created");

    // Top-level: descriptor-table locations come in root-parameter order.
    // Location 0 is the SRV (resource heap slot 0); when a second location is
    // present it is the sampler (sampler heap slot 0). Table locations report
    // space/slot as UINT_MAX, so they are matched by order, not register.
    let mut top2 = ps2.top_level.clone();
    if top2.layout.is_empty() {
        return Err("no resource locations in the layout".into());
    }
    let srv_at = top2.layout[0].offset as usize;
    if srv_at + 8 > top2.bytes.len() {
        return Err("SRV location out of top-level bounds".into());
    }
    top2.bytes[srv_at..srv_at + 8].copy_from_slice(&heap_buf.gpu_address.to_le_bytes());
    if let Some(samp_loc) = top2.layout.get(1) {
        let at = samp_loc.offset as usize;
        if at + 8 <= top2.bytes.len() {
            top2.bytes[at..at + 8].copy_from_slice(&sheap_buf.gpu_address.to_le_bytes());
        }
    }
    println!("  debug tex_id={tex_id:#x} samp_id={samp_id:#x} heap={:#x} sheap={:#x}", heap_buf.gpu_address, sheap_buf.gpu_address);
    println!("  debug top2 bytes: {:?}", &top2.bytes[..top2.bytes.len().min(16)]);

    let rt2 = device.new_texture(&rt_desc)?;
    residency.add(&[src_tex.handle(), rt2.handle(), sampler.handle(), heap_buf.handle(), sheap_buf.handle()]);
    residency.commit();

    // Expected output: each 16×16 block copies one 4×4 texel.
    let mut want2 = vec![[0u8; 4]; BYTES / 4];
    for y in 0..H as usize {
        for x in 0..W as usize {
            let tx = (x / 16).min(TEX as usize - 1);
            let ty = (y / 16).min(TEX as usize - 1);
            want2[y * W as usize + x] = [
                (tx * 64) as u8,
                (ty * 64) as u8,
                128,
                255,
            ];
        }
    }

    let t1 = std::time::Instant::now();
    let cb = queue.command_buffer()?;
    cb.render_pass(
        &[ColorAttachment {
            texture: &rt2,
            load_action: victory_metal::sys::WMTLoadAction::WMTLoadActionClear,
            store_action: victory_metal::sys::WMTStoreAction::WMTStoreActionStore,
            clear: [0.0, 0.0, 0.0, 0.0],
        }],
        W,
        H,
        &[
            Render::Pipeline { pipeline: &pso2 },
            Render::FragmentBuffer { buffer: &heap_buf, offset: 0, index: msc::DESCRIPTOR_HEAP_BIND_POINT },
            Render::FragmentBuffer { buffer: &sheap_buf, offset: 0, index: msc::SAMPLER_HEAP_BIND_POINT },
            Render::FragmentBytes { bytes: &top2.bytes, index: msc::ARGUMENT_BUFFER_BIND_POINT },
            Render::Draw {
                primitive_type: victory_metal::sys::WMTPrimitiveType::WMTPrimitiveTypeTriangle,
                vertex_start: 0,
                vertex_count: 3,
                instance_count: 1,
                base_instance: 0,
            },
        ],
    )?;
    cb.commit();
    cb.wait()?;
    let case2_ms = t1.elapsed().as_secs_f64() * 1000.0;
    println!("case 2 GPU time: {case2_ms:.2} ms");

    let got2 = readback(&rt2, W, H)?;
    compare(&got2, &want2, "case 2")?;

    println!("timing: case 1 {case1_ms:.2} ms, case 2 {case2_ms:.2} ms");
    Ok(())
}

#[cfg(not(windows))]
fn main() {
    eprintln!("victory-metal-render runs on Windows (Wine) only");
}
