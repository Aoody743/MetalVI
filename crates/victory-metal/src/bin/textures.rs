//! Texture and sampler smoke test (run in Wine with victorymetal installed):
//! a 2D texture from a descriptor, a texture view over a wrapped buffer, and a
//! sampler state; dimensions/format read back through the safe accessors.

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
    use victory_metal::sampler::SamplerDescriptor;
    use victory_metal::texture::TextureDescriptor;
    use victory_metal::{Device, Pool};
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};

    const W: u32 = 64;
    const H: u32 = 64;
    const ROW: u64 = W as u64 * 4;
    const BYTES: usize = (W * H * 4) as usize;

    let _pool = Pool::new();
    let device = Device::first()?;
    println!("device: {}", device.name());

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
    println!(
        "descriptor texture: {}x{}x{} array {}, format {:?}",
        texture.width(),
        texture.height(),
        texture.depth(),
        texture.array_length(),
        texture.pixel_format()
    );
    if texture.width() != W as u64
        || texture.height() != H as u64
        || texture.depth() != 1
        || texture.array_length() != 1
    {
        return Err("descriptor texture dimensions wrong".into());
    }
    if texture.pixel_format()
        != victory_metal::sys::WMTPixelFormat::WMTPixelFormatBGRA8Unorm
    {
        return Err("descriptor texture pixel format wrong".into());
    }

    // A wrapped buffer with a texture view over it (offset 0, row stride).
    // SAFETY: a fresh allocation living until the end of run().
    let mem = unsafe { VirtualAlloc(std::ptr::null(), BYTES, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
    if mem.is_null() {
        return Err("VirtualAlloc".into());
    }
    // SAFETY: BYTES bytes just committed.
    let words = unsafe { std::slice::from_raw_parts_mut(mem.cast::<u32>(), BYTES / 4) };
    words.fill(0x4242_4242);
    // SAFETY: the allocation lives until after the view is dropped.
    let buffer = unsafe { device.wrap(mem, BYTES as u64)? };
    // SAFETY: the wrapped buffer holds BYTES bytes; the 64×64 BGRA view fits
    // and the buffer outlives the view.
    let view = unsafe { buffer.new_texture(&desc, 0, ROW)? };
    println!(
        "buffer view: {}x{} format {:?}",
        view.width(),
        view.height(),
        view.pixel_format()
    );
    if view.width() != W as u64 || view.height() != H as u64 {
        return Err("buffer-backed texture view dimensions wrong".into());
    }

    let sampler = device.new_sampler_state(&SamplerDescriptor::default())?;
    println!("sampler state: {:#x}", sampler.handle());
    Ok(())
}

#[cfg(not(windows))]
fn main() {
    eprintln!("victory-metal-textures runs on Windows (Wine) only");
}
