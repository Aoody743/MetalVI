//! Blit encoder smoke test (run in Wine with victorymetal installed):
//! buffer→buffer copy verified on the CPU after the GPU finishes, then
//! buffer→texture copy read back through `getBytes` and compared pixel by pixel.

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
    use victory_metal::texture::TextureDescriptor;
    use victory_metal::{Device, Pool};
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};

    const W: u32 = 64;
    const H: u32 = 64;
    const ROW: u32 = W * 4;
    const BYTES: usize = (W * H * 4) as usize;
    const WORDS: usize = BYTES / 4;

    let _pool = Pool::new();
    let device = Device::first()?;
    println!("device: {}", device.name());
    let queue = device.new_queue(4)?;

    // 1. copyBuffer: A (pattern) → B (zeroed), read back on the CPU.
    // SAFETY: two fresh allocations living until the end of run().
    let mem_a = unsafe { VirtualAlloc(std::ptr::null(), BYTES, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
    let mem_b = unsafe { VirtualAlloc(std::ptr::null(), BYTES, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
    if mem_a.is_null() || mem_b.is_null() {
        return Err("VirtualAlloc".into());
    }
    // SAFETY: both ranges are committed.
    let words_a = unsafe { std::slice::from_raw_parts_mut(mem_a.cast::<u32>(), WORDS) };
    let words_b = unsafe { std::slice::from_raw_parts_mut(mem_b.cast::<u32>(), WORDS) };
    for (i, w) in words_a.iter_mut().enumerate() {
        *w = (i as u32).wrapping_mul(2654435761);
    }
    words_b.fill(0);
    // SAFETY: allocations outlive the buffers and the command buffer.
    let buf_a = unsafe { device.wrap(mem_a, BYTES as u64)? };
    let buf_b = unsafe { device.wrap(mem_b, BYTES as u64)? };

    let cb = queue.command_buffer()?;
    let mut encoder = cb.blit_encoder()?;
    encoder.copy_buffer(&buf_a, 0, &buf_b, 0, BYTES as u64);
    encoder.finish();
    cb.commit();
    cb.wait()?;

    let bad = words_b.iter().zip(words_a.iter()).filter(|(b, a)| b != a).count();
    println!("copyBuffer: {bad} words differ");
    if bad != 0 {
        return Err("copyBuffer results wrong".into());
    }

    // 2. copyFromBufferToTexture: pattern buffer → 64×64 BGRA8 texture,
    //    read back with getBytes and compared.
    let desc = TextureDescriptor {
        pixel_format: victory_metal::sys::WMTPixelFormat::WMTPixelFormatBGRA8Unorm,
        width: W,
        height: H,
        depth: 1,
        array_length: 1,
        type_: victory_metal::sys::WMTTextureType::WMTTextureType2D,
        mipmap_level_count: 1,
        sample_count: 1,
        usage: victory_metal::sys::WMTTextureUsage::WMTTextureUsageShaderRead,
        options: victory_metal::sys::WMTResourceOptions::WMTResourceStorageModeShared,
    };
    let texture = device.new_texture(&desc)?;

    // SAFETY: a fresh committed allocation.
    let mem_c = unsafe { VirtualAlloc(std::ptr::null(), BYTES, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
    if mem_c.is_null() {
        return Err("VirtualAlloc C".into());
    }
    // SAFETY: committed; BGRA8 little-endian word: A R G B.
    let words_c = unsafe { std::slice::from_raw_parts_mut(mem_c.cast::<u32>(), WORDS) };
    for y in 0..H as usize {
        for x in 0..W as usize {
            let b = x as u8;
            let g = y as u8;
            let r = 0x80u8;
            let a = 0xFFu8;
            words_c[y * W as usize + x] = u32::from_be_bytes([a, r, g, b]);
        }
    }
    // SAFETY: allocation outlives the buffer and command buffer.
    let buf_c = unsafe { device.wrap(mem_c, BYTES as u64)? };

    let cb = queue.command_buffer()?;
    let mut encoder = cb.blit_encoder()?;
    encoder.copy_from_buffer_to_texture(
        &buf_c,
        0,
        ROW,
        BYTES as u32,
        [W as u64, H as u64, 1],
        &texture,
        0,
        0,
        [0, 0, 0],
    );
    encoder.finish();
    cb.commit();
    cb.wait()?;

    // SAFETY: a fresh committed allocation for the readback.
    let mem_d = unsafe { VirtualAlloc(std::ptr::null(), BYTES, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
    if mem_d.is_null() {
        return Err("VirtualAlloc D".into());
    }
    // SAFETY: committed; getBytes fills it for the whole 2D slice.
    unsafe {
        victory_metal::sys::MTLTexture_getBytes(
            texture.handle(),
            victory_metal::sys::WMTOrigin { x: 0, y: 0, z: 0 },
            victory_metal::sys::WMTSize { width: W as u64, height: H as u64, depth: 1 },
            0,
            0,
            victory_metal::sys::WMTMemoryPointer { ptr: mem_d },
            ROW as u64,
            BYTES as u64,
        );
    }
    let words_d = unsafe { std::slice::from_raw_parts(mem_d.cast::<u32>(), WORDS) };
    let bad = words_d.iter().zip(words_c.iter()).filter(|(d, c)| d != c).count();
    println!("buffer→texture: {bad} pixels differ");
    if bad != 0 {
        return Err("buffer→texture copy wrong".into());
    }
    Ok(())
}

#[cfg(not(windows))]
fn main() {
    eprintln!("victory-metal-blit runs on Windows (Wine) only");
}
