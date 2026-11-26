//! Depth-only DSV aliases of single-channel colour textures.

use std::collections::HashMap;

use victory_metal::sys::{
    WMTPixelFormat as P, WMTResourceOptions, WMTTextureType as T, WMTTextureUsage,
};
use victory_metal::texture::{Texture, TextureDescriptor};
use victory_metal::{Buffer, CommandBuffer, Device, Error, Result};

use super::metal_texture::{MetalTexture, base_type};

#[derive(Clone)]
pub(crate) struct DepthCompanion {
    pub(crate) texture: Texture,
    colour: Texture,
    bytes: u32,
}

impl DepthCompanion {
    pub(crate) fn format(pixel: P) -> Option<P> {
        Some(match pixel {
            P::WMTPixelFormatR16Unorm
            | P::WMTPixelFormatR16Snorm
            | P::WMTPixelFormatR16Uint
            | P::WMTPixelFormatR16Sint
            | P::WMTPixelFormatR16Float => P::WMTPixelFormatDepth16Unorm,
            P::WMTPixelFormatR32Uint | P::WMTPixelFormatR32Sint | P::WMTPixelFormatR32Float => {
                P::WMTPixelFormatDepth32Float
            }
            _ => return None,
        })
    }

    pub(crate) fn new(device: &Device, colour: &MetalTexture, pixel: P) -> Result<Self> {
        let type_ = base_type(colour);
        if Self::format(colour.pixel) != Some(pixel)
            || !matches!(type_, T::WMTTextureType2D | T::WMTTextureType2DArray)
        {
            return Err(Error(
                "incompatible depth companion format or texture type".into(),
            ));
        }
        let width = colour.texture.width() as u32;
        let height = colour.texture.height() as u32;
        let texture = device.new_texture(&TextureDescriptor {
            pixel_format: pixel,
            width,
            height,
            depth: 1,
            array_length: colour.slices,
            type_,
            mipmap_level_count: colour.mips,
            sample_count: 1,
            usage: WMTTextureUsage(1 | 4), // shader read | render target
            options: WMTResourceOptions(0x20), // private, hazard tracked
        })?;
        let bytes = if pixel == P::WMTPixelFormatDepth16Unorm {
            2
        } else {
            4
        };
        Ok(Self {
            texture,
            colour: colour.texture.clone(),
            bytes,
        })
    }

    fn copy(
        &self,
        device: &Device,
        cb: &CommandBuffer,
        scratch: &mut Option<Buffer>,
        level: u16,
        slice: u16,
        to_depth: bool,
    ) -> Result<()> {
        let _pool = victory_metal::Pool::new();
        let (src, dst) = if to_depth {
            (&self.colour, &self.texture)
        } else {
            (&self.texture, &self.colour)
        };
        let width = (src.width() >> level).max(1);
        let height = (src.height() >> level).max(1);
        let row = (width as u32 * self.bytes).next_multiple_of(256);
        let pitch = row * height as u32;
        if scratch.as_ref().is_none_or(|b| b.len < u64::from(pitch)) {
            *scratch = Some(device.private_buffer(u64::from(pitch))?);
        }
        let staging = scratch.as_ref().unwrap();
        let size = [width, height, 1];
        let mut blit = cb.blit_encoder()?;
        // A DSV reinterprets the storage bits. Do not numerically convert
        // R16Float to D16_UNORM or R32Uint to D32_FLOAT.
        blit.copy_from_texture_to_buffer(
            src,
            slice.into(),
            level.into(),
            [0; 3],
            size,
            staging,
            0,
            row,
            pitch,
        );
        blit.copy_from_buffer_to_texture(
            staging,
            0,
            row,
            pitch,
            size,
            dst,
            slice.into(),
            level.into(),
            [0; 3],
        );
        blit.finish();
        Ok(())
    }
}

/// Replay-local: a failed or abandoned submission must not publish freshness.
/// Bindless shaders may write colour without a per-command resource list.
/// Advancing the epoch conservatively invalidates all cached depth contents.
#[derive(Default)]
pub(crate) struct DepthSync {
    epoch: u64,
    synced: HashMap<(u64, u16, u16), u64>,
    scratch: Option<Buffer>,
}

impl DepthSync {
    pub(crate) fn colour_may_change(&mut self) {
        self.epoch += 1;
    }

    pub(crate) fn load(
        &mut self,
        device: &Device,
        cb: &CommandBuffer,
        d: &DepthCompanion,
        level: u16,
        slice: u16,
    ) -> Result<()> {
        let key = (d.texture.handle(), level, slice);
        if self.synced.get(&key) != Some(&self.epoch) {
            d.copy(device, cb, &mut self.scratch, level, slice, true)?;
            self.preserve(d, level, slice);
        }
        Ok(())
    }

    pub(crate) fn store(
        &mut self,
        device: &Device,
        cb: &CommandBuffer,
        d: &DepthCompanion,
        level: u16,
        slice: u16,
    ) -> Result<()> {
        d.copy(device, cb, &mut self.scratch, level, slice, false)?;
        self.preserve(d, level, slice);
        Ok(())
    }

    /// The active DSV subresource cannot also be written through a colour view.
    pub(crate) fn preserve(&mut self, d: &DepthCompanion, level: u16, slice: u16) {
        self.synced
            .insert((d.texture.handle(), level, slice), self.epoch);
    }
}

#[cfg(test)]
#[path = "metal_depth_tests.rs"]
mod tests;
