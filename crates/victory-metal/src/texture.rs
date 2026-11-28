//! MTLTexture wrappers: a texture allocated by the device from a descriptor,
//! or a texture view over an existing [`Buffer`]
//! (`newTextureWithDescriptor:offset:bytesPerRow:`).

use crate::sys;
use crate::{Buffer, Device, Error, Object, Pool, Result};
use sys::obj_handle_t;

/// A Metal texture (+1 reference released on drop; `Clone` retains).
#[derive(Clone, Debug)]
pub struct Texture(Object);

impl Texture {
    /// Its `gpuResourceID`, for descriptor entries.
    pub fn gpu_resource_id(&self) -> u64 {
        self.0.gpu_resource_id()
    }

    /// A texture from an owned object (a texture view, say).
    pub fn from_object(object: Object) -> Texture {
        Texture(object)
    }

    /// The texture as a plain object (to keep it alive with others).
    pub fn into_object(self) -> Object {
        self.0
    }

    pub fn handle(&self) -> obj_handle_t {
        self.0.handle()
    }

    pub fn width(&self) -> u64 {
        // SAFETY: a live texture.
        unsafe { sys::MTLTexture_width(self.handle()) }
    }

    pub fn height(&self) -> u64 {
        // SAFETY: a live texture.
        unsafe { sys::MTLTexture_height(self.handle()) }
    }

    pub fn depth(&self) -> u64 {
        // SAFETY: a live texture.
        unsafe { sys::MTLTexture_depth(self.handle()) }
    }

    pub fn pixel_format(&self) -> sys::WMTPixelFormat {
        // SAFETY: a live texture.
        unsafe { sys::MTLTexture_pixelFormat(self.handle()) }
    }

    pub fn array_length(&self) -> u64 {
        // SAFETY: a live texture.
        unsafe { sys::MTLTexture_arrayLength(self.handle()) }
    }

    pub fn mipmap_level_count(&self) -> u64 {
        // SAFETY: a live texture.
        unsafe { sys::MTLTexture_mipmapLevelCount(self.handle()) }
    }
}

/// Safe mirror of the bindgen `WMTTextureInfo`; the bitfield getters/setters
/// stay inside the conversion below.
#[derive(Clone, Debug)]
pub struct TextureDescriptor {
    pub pixel_format: sys::WMTPixelFormat,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub array_length: u32,
    pub type_: sys::WMTTextureType,
    pub mipmap_level_count: u32,
    pub sample_count: u32,
    pub usage: sys::WMTTextureUsage,
    pub options: sys::WMTResourceOptions,
}

impl Default for TextureDescriptor {
    fn default() -> Self {
        Self {
            pixel_format: sys::WMTPixelFormat::WMTPixelFormatR8Unorm,
            width: 1,
            height: 1,
            depth: 1,
            array_length: 1,
            type_: sys::WMTTextureType::WMTTextureType2D,
            mipmap_level_count: 1,
            sample_count: 1,
            usage: sys::WMTTextureUsage::WMTTextureUsageShaderRead,
            options: sys::WMTResourceOptions::WMTResourceStorageModeShared,
        }
    }
}

impl TextureDescriptor {
    /// Converts to the C ABI struct; the bitfield and padding start at zero.
    fn into_info(&self) -> sys::WMTTextureInfo {
        // SAFETY: all-zero is the default for the bitfield, padding and
        // reserved fields, as the C smoke test does.
        let mut info: sys::WMTTextureInfo = unsafe { std::mem::zeroed() };
        info.pixel_format = self.pixel_format;
        info.width = self.width;
        info.height = self.height;
        info.depth = self.depth;
        info.array_length = self.array_length;
        info.set_type(self.type_);
        info.set_mipmap_level_count(self.mipmap_level_count);
        info.set_sample_count(self.sample_count);
        info.set_usage(self.usage);
        info.options = self.options;
        info
    }
}

impl TextureDescriptor {
    /// Metal's size limits for a device-allocated texture (Apple GPUs:
    /// 16384 for 1D/2D, 2048 for 3D and array layers).
    pub fn check_limits(&self) -> Result<()> {
        use sys::WMTTextureType as T;
        let three_d = self.type_ == T::WMTTextureType3D;
        let side = if three_d { 2048 } else { 16384 };
        let bad = self.width == 0
            || self.width > side
            || self.height > side
            || (three_d && self.depth > 2048)
            || self.array_length > 2048
            || self.mipmap_level_count == 0
            || self.mipmap_level_count > 15
            || !matches!(self.sample_count, 1 | 2 | 4 | 8);
        if bad {
            return Err(Error(format!(
                "texture past Metal's limits: {}x{}x{} x{} layers, {} mips, {} samples, type {}",
                self.width, self.height, self.depth, self.array_length, self.mipmap_level_count, self.sample_count, self.type_.0
            )));
        }
        Ok(())
    }
}

impl Device {
    /// A new texture allocated by the device (`newTextureWithDescriptor:`).
    pub fn new_texture(&self, desc: &TextureDescriptor) -> Result<Texture> {
        // Metal aborts the process on a descriptor past its limits (a failed
        // assertion, not an error): refused here instead. A size read from a
        // title resource the title already freed reached it once (759916560
        // wide, a frame capture, stage-r-m3-fix-20261128-023844).
        desc.check_limits()?;
        let _pool = Pool::new();
        let mut info = desc.into_info();
        // SAFETY: `info` is valid for the call; `newTexture` returns owned (+1).
        let texture = Object::adopt(unsafe { sys::MTLDevice_newTexture(self.handle(), &mut info) })
            .ok_or_else(|| Error(format!("newTexture {}x{}", desc.width, desc.height)))?;
        Ok(Texture(texture))
    }
}

impl Buffer {
    /// A texture view over this buffer: `offset` bytes into the buffer and
    /// `bytes_per_row` stride per image row
    /// (`newTextureWithDescriptor:offset:bytesPerRow:`).
    ///
    /// # Safety
    /// The descriptor's dimensions and `bytes_per_row` must match the bytes
    /// available from `offset` in this buffer; the buffer lives at least as
    /// long as the returned texture and any command using it.
    pub unsafe fn new_texture(
        &self,
        desc: &TextureDescriptor,
        offset: u64,
        bytes_per_row: u64,
    ) -> Result<Texture> {
        let _pool = Pool::new();
        let mut info = desc.into_info();
        // SAFETY: `info`/offset/stride are valid for the call; `newTexture`
        // returns owned (+1).
        let texture = Object::adopt(unsafe {
            sys::MTLBuffer_newTexture(self.handle(), &mut info, offset, bytes_per_row)
        })
        .ok_or_else(|| Error(format!("newTexture over buffer offset {offset:#x}")))?;
        Ok(Texture(texture))
    }
}

#[cfg(test)]
mod limit_tests {
    use super::TextureDescriptor;
    use crate::sys::WMTTextureType as T;

    #[test]
    fn sizes_past_metals_limits_are_refused_before_metal_sees_them() {
        let ok = TextureDescriptor { width: 1920, height: 1080, type_: T::WMTTextureType2D, ..Default::default() };
        assert!(ok.check_limits().is_ok());
        for bad in [
            TextureDescriptor { width: 759_916_560, ..ok },
            TextureDescriptor { height: 16385, ..ok },
            TextureDescriptor { width: 0, ..ok },
            TextureDescriptor { type_: T::WMTTextureType3D, width: 4096, depth: 4, ..ok },
            TextureDescriptor { array_length: 4096, ..ok },
            TextureDescriptor { sample_count: 3, ..ok },
            TextureDescriptor { mipmap_level_count: 0, ..ok },
        ] {
            assert!(bad.check_limits().is_err(), "{bad:?}");
        }
    }
}
