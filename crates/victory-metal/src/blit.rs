//! Blit encoder: GPU-side buffer↔buffer, buffer↔texture, and texture↔texture copies.
//!
//! Like [`crate::CommandBuffer::compute`], the commands are built here and
//! handed to the unix side as a linked chain in one `encodeCommands` call.
//! The encoder itself is autoreleased and stays valid while the current
//! [`crate::Pool`] lives.

use core::ffi::c_void;

use crate::sys;
use crate::texture::Texture;
use crate::{Buffer, CommandBuffer, Error, Result};
use sys::obj_handle_t;

/// `MTLBlitOption` values from Metal `MTLBlitCommandEncoder.h`.
pub const BLIT_OPTION_NONE: u16 = 0;
pub const BLIT_OPTION_DEPTH: u16 = 1 << 0;
pub const BLIT_OPTION_STENCIL: u16 = 1 << 1;

/// A blit encoder for one command buffer; commands accumulate until
/// [`BlitEncoder::finish`] (also runs on drop).
pub struct BlitEncoder {
    encoder: obj_handle_t,
    commands: Vec<Box<dyn BlitCmd>>,
    finished: bool,
}

/// One winemetal blit command: every one starts like [`sys::wmtcmd_base`].
trait BlitCmd {
    fn as_base(&self) -> *const sys::wmtcmd_base;
    fn set_next(&mut self, next: *const sys::wmtcmd_base);
}

macro_rules! blit_cmd {
    ($($t:ty),*) => {$(
        impl BlitCmd for $t {
            fn as_base(&self) -> *const sys::wmtcmd_base {
                (self as *const Self).cast()
            }
            fn set_next(&mut self, next: *const sys::wmtcmd_base) {
                self.next = sys::WMTMemoryPointer { ptr: next as *mut c_void };
            }
        }
    )*};
}
blit_cmd!(
    sys::wmtcmd_blit_copy_from_buffer_to_buffer,
    sys::wmtcmd_blit_copy_from_buffer_to_texture,
    sys::wmtcmd_blit_copy_from_buffer_to_texture_withblitoption,
    sys::wmtcmd_blit_copy_from_texture_to_buffer,
    sys::wmtcmd_blit_copy_from_texture_to_buffer_withblitoption,
    sys::wmtcmd_blit_copy_from_texture_to_texture,
    sys::wmtcmd_blit_copy_texture,
    sys::wmtcmd_blit_fillbuffer
);

fn null_next() -> sys::WMTMemoryPointer {
    sys::WMTMemoryPointer { ptr: core::ptr::null_mut() }
}

impl CommandBuffer {
    /// A blit encoder for this command buffer (`blitCommandEncoder`).
    pub fn blit_encoder(&self) -> Result<BlitEncoder> {
        // SAFETY: a live command buffer; the encoder is autoreleased.
        let encoder = unsafe { sys::MTLCommandBuffer_blitCommandEncoder(self.handle()) };
        if encoder == 0 {
            return Err(Error("blitCommandEncoder".into()));
        }
        Ok(BlitEncoder { encoder, commands: Vec::new(), finished: false })
    }
}

impl BlitEncoder {
    /// `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:`
    pub fn copy_buffer(
        &mut self,
        src: &Buffer,
        src_offset: u64,
        dst: &Buffer,
        dst_offset: u64,
        length: u64,
    ) {
        self.commands.push(Box::new(sys::wmtcmd_blit_copy_from_buffer_to_buffer {
            type_: sys::WMTBlitCommandType::WMTBlitCommandCopyFromBufferToBuffer,
            reserved: [0; 3],
            next: null_next(),
            src: src.handle(),
            src_offset,
            dst: dst.handle(),
            dst_offset,
            copy_length: length,
        }));
    }

    /// `copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:
    /// sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:`
    pub fn copy_from_buffer_to_texture(
        &mut self,
        src: &Buffer,
        src_offset: u64,
        bytes_per_row: u32,
        bytes_per_image: u32,
        size: [u64; 3],
        dst: &Texture,
        slice: u32,
        level: u32,
        origin: [u64; 3],
    ) {
        self.commands.push(Box::new(sys::wmtcmd_blit_copy_from_buffer_to_texture {
            type_: sys::WMTBlitCommandType::WMTBlitCommandCopyFromBufferToTexture,
            reserved: [0; 3],
            next: null_next(),
            src: src.handle(),
            src_offset,
            bytes_per_row,
            bytes_per_image,
            size: sys::WMTSize { width: size[0], height: size[1], depth: size[2] },
            dst: dst.handle(),
            slice,
            level,
            origin: sys::WMTOrigin { x: origin[0], y: origin[1], z: origin[2] },
        }));
    }

    /// `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:
    /// toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:`
    pub fn copy_from_texture_to_buffer(
        &mut self,
        src: &Texture,
        slice: u32,
        level: u32,
        origin: [u64; 3],
        size: [u64; 3],
        dst: &Buffer,
        dst_offset: u64,
        bytes_per_row: u32,
        bytes_per_image: u32,
    ) {
        self.commands.push(Box::new(sys::wmtcmd_blit_copy_from_texture_to_buffer {
            type_: sys::WMTBlitCommandType::WMTBlitCommandCopyFromTextureToBuffer,
            reserved: [0; 3],
            next: null_next(),
            src: src.handle(),
            slice,
            level,
            origin: sys::WMTOrigin { x: origin[0], y: origin[1], z: origin[2] },
            size: sys::WMTSize { width: size[0], height: size[1], depth: size[2] },
            dst: dst.handle(),
            offset: dst_offset,
            bytes_per_row,
            bytes_per_image,
        }));
    }

    /// `copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:
    /// sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:options:`
    #[allow(clippy::too_many_arguments)]
    pub fn copy_from_buffer_to_texture_with_options(
        &mut self,
        src: &Buffer,
        src_offset: u64,
        bytes_per_row: u32,
        bytes_per_image: u32,
        size: [u64; 3],
        dst: &Texture,
        slice: u32,
        level: u16,
        origin: [u64; 3],
        options: u16,
    ) {
        self.commands.push(Box::new(sys::wmtcmd_blit_copy_from_buffer_to_texture_withblitoption {
            type_: sys::WMTBlitCommandType::WMTBlitCommandCopyFromBufferToTextureWithBlitOption,
            reserved: [0; 3],
            next: null_next(),
            src: src.handle(),
            src_offset,
            bytes_per_row,
            bytes_per_image,
            size: sys::WMTSize { width: size[0], height: size[1], depth: size[2] },
            dst: dst.handle(),
            slice,
            level,
            options,
            origin: sys::WMTOrigin { x: origin[0], y: origin[1], z: origin[2] },
        }));
    }

    /// `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:
    /// toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options:`
    #[allow(clippy::too_many_arguments)]
    pub fn copy_from_texture_to_buffer_with_options(
        &mut self,
        src: &Texture,
        slice: u32,
        level: u16,
        origin: [u64; 3],
        size: [u64; 3],
        dst: &Buffer,
        dst_offset: u64,
        bytes_per_row: u32,
        bytes_per_image: u32,
        options: u16,
    ) {
        self.commands.push(Box::new(sys::wmtcmd_blit_copy_from_texture_to_buffer_withblitoption {
            type_: sys::WMTBlitCommandType::WMTBlitCommandCopyFromTextureToBufferWithBlitOption,
            reserved: [0; 3],
            next: null_next(),
            src: src.handle(),
            slice,
            level,
            options,
            origin: sys::WMTOrigin { x: origin[0], y: origin[1], z: origin[2] },
            size: sys::WMTSize { width: size[0], height: size[1], depth: size[2] },
            dst: dst.handle(),
            offset: dst_offset,
            bytes_per_row,
            bytes_per_image,
        }));
    }

    /// `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:
    /// toTexture:destinationSlice:destinationLevel:destinationOrigin:`
    #[allow(clippy::too_many_arguments)]
    pub fn copy_from_texture_to_texture(
        &mut self,
        src: &Texture,
        src_slice: u32,
        src_level: u32,
        src_origin: [u64; 3],
        src_size: [u64; 3],
        dst: &Texture,
        dst_slice: u32,
        dst_level: u32,
        dst_origin: [u64; 3],
    ) {
        self.commands.push(Box::new(sys::wmtcmd_blit_copy_from_texture_to_texture {
            type_: sys::WMTBlitCommandType::WMTBlitCommandCopyFromTextureToTexture,
            reserved: [0; 3],
            next: null_next(),
            src: src.handle(),
            src_slice,
            src_level,
            src_origin: sys::WMTOrigin { x: src_origin[0], y: src_origin[1], z: src_origin[2] },
            src_size: sys::WMTSize { width: src_size[0], height: src_size[1], depth: src_size[2] },
            dst: dst.handle(),
            dst_slice,
            dst_level,
            dst_origin: sys::WMTOrigin { x: dst_origin[0], y: dst_origin[1], z: dst_origin[2] },
        }));
    }

    /// `copyFromTexture:toTexture:` (whole resource, identical dimensions & format).
    pub fn copy_texture(&mut self, src: &Texture, dst: &Texture) {
        self.commands.push(Box::new(sys::wmtcmd_blit_copy_texture {
            type_: sys::WMTBlitCommandType::WMTBlitCommandCopyTexture,
            reserved: [0; 3],
            next: null_next(),
            src: src.handle(),
            dst: dst.handle(),
        }));
    }

    /// `fillBuffer:range:value:`
    pub fn fill_buffer(&mut self, buffer: &Buffer, offset: u64, length: u64, value: u8) {
        self.commands.push(Box::new(sys::wmtcmd_blit_fillbuffer {
            type_: sys::WMTBlitCommandType::WMTBlitCommandFillBuffer,
            reserved: [0; 3],
            next: null_next(),
            buffer: buffer.handle(),
            offset,
            length,
            value,
        }));
    }

    /// Links the command chain, hands it to the encoder, and ends encoding.
    /// A second call is a no-op; [`Drop`] calls this if it was not finished.
    pub fn finish(&mut self) {
        if self.finished {
            return;
        }
        for i in 1..self.commands.len() {
            let next = self.commands[i].as_base();
            self.commands[i - 1].set_next(next);
        }
        if let Some(head) = self.commands.first() {
            // SAFETY: the chain is well formed and lives until the call returns.
            unsafe { sys::MTLBlitCommandEncoder_encodeCommands(self.encoder, head.as_base()) };
        }
        // SAFETY: a live encoder.
        unsafe { sys::MTLCommandEncoder_endEncoding(self.encoder) };
        self.finished = true;
    }
}

impl Drop for BlitEncoder {
    fn drop(&mut self) {
        self.finish();
    }
}
