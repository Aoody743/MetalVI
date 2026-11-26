//! Title views as Metal Shader Converter descriptor entries: what the Metal
//! backend writes into its view heap (bind point 0) for each slot of a
//! title heap. Buffers are title memory seen through [`Chunks`] (no copy);
//! textures and samplers come with the render path (P2) and give `None`
//! here for now, as do typed buffer views (they need a texture buffer view).

use victory_metal::chunks::Chunks;
use victory_metal::ir::DescriptorEntry;

use super::metal_views::{Window, placed_address, srv_window, uav_window};
use crate::descriptor::View;
use crate::object::This;

/// Why a view has no entry yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Missing {
    /// Textures, samplers: the render path (P2).
    NotYet,
    /// A typed buffer view: needs a texture buffer view of this title
    /// range in this DXGI format ([`super::metal::MetalBackend`] makes it).
    Typed { va: u64, bytes: u64, format: u32 },
    /// The memory is not wrapped (not mapped, outside the chunk range).
    Unmapped,
    /// A description we cannot read.
    Unreadable,
}

/// The entry for `view`.
pub(crate) fn entry(view: &View, chunks: &mut Chunks) -> Result<DescriptorEntry, Missing> {
    match view {
        View::Constant(address, size) => buffer(chunks, *address, Window { offset: 0, bytes: u64::from(*size), typed_format: None }),
        View::Shader(resource, desc) => resource_buffer(chunks, *resource, desc.as_ref().map(|d| srv_window(d))),
        View::Unordered(resource, _counter, desc) => resource_buffer(chunks, *resource, desc.as_ref().map(|d| uav_window(d))),
        View::Placed(unordered, _resource, desc) => {
            let window = if *unordered { uav_window(&desc[..10]) } else { srv_window(&desc[..10]) };
            let window = window.ok_or(Missing::Unreadable)?;
            buffer(chunks, placed_address(desc), window)
        }
        View::Sampler(_) | View::ConsoleSampler(_) => Err(Missing::NotYet),
    }
}

/// A view of a buffer resource: its memory is the title's (the address
/// the title sees is the pointer victory gave it).
fn resource_buffer(chunks: &mut Chunks, resource: usize, window: Option<Option<Window>>) -> Result<DescriptorEntry, Missing> {
    let Some((data, width)) = crate::resource::buffer_bytes(resource as This) else {
        return Err(Missing::NotYet); // a texture
    };
    let window = match window {
        // No description: the whole buffer, raw.
        None => Window { offset: 0, bytes: width, typed_format: None },
        Some(w) => w.ok_or(Missing::Unreadable)?,
    };
    buffer(chunks, data as u64, window)
}

fn buffer(chunks: &mut Chunks, base: u64, window: Window) -> Result<DescriptorEntry, Missing> {
    if let Some(format) = window.typed_format {
        return Err(Missing::Typed { va: base + window.offset, bytes: window.bytes, format });
    }
    if window.bytes == 0 {
        return Ok(DescriptorEntry::NULL);
    }
    let gpu = chunks.window(base + window.offset, window.bytes);
    if gpu == 0 {
        return Err(Missing::Unmapped);
    }
    Ok(DescriptorEntry::buffer(gpu, window.bytes))
}
