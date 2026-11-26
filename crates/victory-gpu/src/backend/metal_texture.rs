//! Title textures on Metal (TEXTURES.md "The Metal path", first version):
//!
//! - made lazily, the first time a view of them is bound;
//! - title-filled textures (no render target / depth / unordered flag)
//!   get their console memory detiled once on the CPU (victory-formats'
//!   `Layout::detile`) and blitted in; refreshing on change comes later;
//! - textures the GPU writes are Metal's alone (the truth), never written
//!   back into title memory, as on the D3D12 path;
//! - views are Metal texture views (format, mip range, slices, swizzle from
//!   the D3D12 description words) whose `gpuResourceID` goes in the entry.

use std::sync::Arc;

use victory_metal::ir::DescriptorEntry;
use victory_metal::sys::{self, WMTPixelFormat, WMTTextureSwizzle, WMTTextureSwizzleChannels, WMTTextureType, WMTTextureUsage};
use victory_metal::texture::{Texture, TextureDescriptor};

use super::metal::MetalBackend;
use super::metal_map::{format_props, map_pixel_format};
use crate::object::This;
use crate::resource::{ResourceDesc, texture_layout, texture_memory};

const RENDER_TARGET: u32 = 1;
const DEPTH: u32 = 2;
const UNORDERED: u32 = 4;
const TEXTURE_1D: u32 = 2;
const TEXTURE_3D: u32 = 4;

/// Logs why a texture or view could not be made (first 96 times).
fn fail(why: &'static str, d: Option<&ResourceDesc>, words: Option<&[u32]>) {
    // once per (reason, format, dimension, flags, view format, view dimension)
    static SEEN: std::sync::Mutex<Option<std::collections::HashSet<(&'static str, [u32; 5])>>> = std::sync::Mutex::new(None);
    let key = (
        why,
        [
            d.map_or(0, |d| d.format),
            d.map_or(0, |d| d.dimension),
            d.map_or(0, |d| d.flags),
            words.and_then(|w| w.first().copied()).unwrap_or(0),
            words.and_then(|w| w.get(1).copied()).unwrap_or(0),
        ],
    );
    {
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let seen = seen.get_or_insert_with(Default::default);
        if seen.len() >= 256 || !seen.insert(key) {
            return;
        }
    }
    match d {
        Some(d) => tracing::warn!(
            why,
            format = d.format,
            dimension = d.dimension,
            flags = d.flags,
            width = d.width,
            height = d.height,
            depth_or_array = d.depth_or_array_size,
            mips = d.mip_levels,
            samples = d.sample_count,
            words = ?words.map(|w| &w[..w.len().min(10)]),
            "Metal texture: left empty"
        ),
        None => tracing::warn!(why, words = ?words.map(|w| &w[..w.len().min(10)]), "Metal texture: left empty"),
    }
}

pub(crate) struct MetalTexture {
    pub(crate) texture: Texture,
    pub(crate) pixel: WMTPixelFormat,
    pub(crate) desc: ResourceDesc,
    /// Mip levels and array slices of the texture.
    pub(crate) mips: u32,
    pub(crate) slices: u32,
}

/// The Metal pixel format of a DXGI format (with the Apple GPU's depth
/// formats: no 24-bit depth).
pub(crate) fn pixel_format(dxgi: u32) -> Option<WMTPixelFormat> {
    let p = WMTPixelFormat(map_pixel_format(crate::resource::pc_format(dxgi) as i32)? as u32);
    Some(match p {
        WMTPixelFormat::WMTPixelFormatDepth24Unorm_Stencil8 => WMTPixelFormat::WMTPixelFormatDepth32Float_Stencil8,
        WMTPixelFormat::WMTPixelFormatX24_Stencil8 => WMTPixelFormat::WMTPixelFormatX32_Stencil8,
        p => p,
    })
}

impl MetalBackend {
    /// The Metal texture of title texture `r`, made (and for title-filled
    /// textures, filled) on first use.
    pub(crate) fn texture_for(&self, r: This) -> Option<Arc<MetalTexture>> {
        let key = r as usize;
        if let Some(t) = self.textures.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
            return t.clone();
        }
        let made = self.make_texture(r).map(Arc::new);
        self.textures.lock().unwrap_or_else(|e| e.into_inner()).insert(key, made.clone());
        made
    }

    fn make_texture(&self, r: This) -> Option<MetalTexture> {
        let Some((memory, d)) = texture_memory(r) else {
            fail("not a texture", None, None);
            return None;
        };
        let Some(pixel) = pixel_format(d.format) else {
            fail("no pixel format", Some(&d), None);
            return None;
        };
        let mips = u32::from(d.mip_levels.max(1));
        let (type_, depth, slices) = match d.dimension {
            TEXTURE_1D if d.depth_or_array_size > 1 => (WMTTextureType::WMTTextureType1DArray, 1, u32::from(d.depth_or_array_size)),
            TEXTURE_1D => (WMTTextureType::WMTTextureType1D, 1, 1),
            TEXTURE_3D => (WMTTextureType::WMTTextureType3D, u32::from(d.depth_or_array_size.max(1)), 1),
            _ if d.sample_count > 1 => (WMTTextureType::WMTTextureType2DMultisample, 1, 1),
            _ if d.depth_or_array_size > 1 => (WMTTextureType::WMTTextureType2DArray, 1, u32::from(d.depth_or_array_size)),
            _ => (WMTTextureType::WMTTextureType2D, 1, 1),
        };
        // shader read | shader write | render target | pixel format view
        let mut usage = 1 | 0x10;
        if d.flags & UNORDERED != 0 {
            usage |= 2;
        }
        if d.flags & (RENDER_TARGET | DEPTH) != 0 {
            usage |= 4;
        }
        let Ok(width) = u32::try_from(d.width) else {
            fail("width", Some(&d), None);
            return None;
        };
        let desc = TextureDescriptor {
            pixel_format: pixel,
            width,
            height: d.height.max(1),
            depth,
            array_length: slices,
            type_,
            mipmap_level_count: mips,
            sample_count: d.sample_count.max(1),
            usage: WMTTextureUsage(usage as _),
            options: sys::WMTResourceOptions(0x20), // storage mode private
        };
        let _pool = victory_metal::Pool::new();
        let texture = match self.device.new_texture(&desc) {
            Ok(t) => t,
            Err(e) => {
                tracing::debug!(error = %e, "newTexture");
                fail("newTexture failed", Some(&d), None);
                return None;
            }
        };
        // Shaders reach it through argument buffers (its gpuResourceID):
        // it must be resident, as title memory is (T4 §3.6).
        {
            let chunks = self.chunks();
            chunks.residency().add(&[texture.handle()]);
            chunks.residency().commit();
        }
        let t = MetalTexture { texture, pixel, desc: d, mips, slices };
        if d.flags & (RENDER_TARGET | DEPTH | UNORDERED) == 0 && super::metal::texture_fill_on() {
            self.fill(&t, memory);
        }
        Some(t)
    }

    /// Detiles a title-filled texture's console memory into it (all mips
    /// and slices of plane 0). Skipped (left undefined) when the layout is
    /// not addressable (XOR modes, MSAA) or the memory is not all mapped.
    fn fill(&self, t: &MetalTexture, memory: *const u8) {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        let Some(layout) = texture_layout(&t.desc) else { return };
        let Some(plane) = layout.planes.first() else { return };
        let layout = &plane.layout;
        if !layout.addressable() {
            return;
        }
        let Some(props) = format_props(crate::resource::pc_format(t.desc.format) as i32) else { return };
        let (bw, bh, bytes) = (props.block_width.max(1), props.block_height.max(1), props.block_size_bytes);
        let tiled_size = layout.size;
        if bytes == 0 || victory_metal::chunks::committed_extent(memory as u64, tiled_size) < tiled_size {
            return;
        }
        // SAFETY: committed, read only.
        let tiled = unsafe { std::slice::from_raw_parts(memory, tiled_size as usize) };
        let w = u64::from(t.desc.width as u32);
        let h = u64::from(t.desc.height.max(1));
        let d3 = if t.desc.dimension == TEXTURE_3D { u64::from(t.desc.depth_or_array_size.max(1)) } else { 1 };
        // Rows of every subresource, one after the other, in a staging buffer.
        let mut plan = Vec::new();
        let mut total = 0u64;
        for slice in 0..t.slices {
            for mip in 0..t.mips {
                let mw = (w >> mip).max(1).div_ceil(u64::from(bw));
                let mh = (h >> mip).max(1).div_ceil(u64::from(bh));
                let md = (d3 >> mip).max(1);
                let row = mw * u64::from(bytes);
                let image = row * mh;
                plan.push((slice, mip, total, row, image, md, [(w >> mip).max(1), (h >> mip).max(1), md]));
                total += (image * md).next_multiple_of(256);
            }
        }
        let size = total.next_multiple_of(64 << 10) as usize;
        // SAFETY: a fresh allocation, released below after the GPU is done.
        let staging = unsafe { VirtualAlloc(std::ptr::null(), size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        if staging.is_null() {
            return;
        }
        let _pool = victory_metal::Pool::new();
        // SAFETY: the staging memory lives until the command buffer is waited for.
        let Ok(buffer) = (unsafe { self.device.wrap(staging, size as u64) }) else {
            // SAFETY: ours.
            unsafe { VirtualFree(staging, 0, MEM_RELEASE) };
            return;
        };
        // The detile on the GPU, reading the title's memory where it is
        // (zero copy), for every subresource the kernels take (F10,
        // `metal_detile`); the CPU does the rest into the same staging.
        // On the GPU only where it pays (F16: under ~6 MB a command buffer's
        // fixed cost outweighs it, and linear layouts copy faster on the CPU).
        static MIN_BYTES: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let min_bytes = *MIN_BYTES.get_or_init(|| {
            std::env::var("VICTORY_METAL_GPU_DETILE_MIN_KB").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(4096) << 10
        });
        let worth_gpu = tiled_size >= min_bytes && layout.desc.swizzle.0 != 0;
        let window = if super::metal::gpu_detile_on() && worth_gpu {
            self.chunks().window_buffer(memory as u64, tiled_size)
        } else {
            None
        };
        let detiler = window.as_ref().and_then(|_| self.detiler());
        // Each subresource's kernel parameters, if the GPU takes it.
        let params: Vec<Option<super::metal_detile::Params>> = plan
            .iter()
            .map(|&(slice, mip, ..)| {
                (window.is_some() && detiler.is_some()).then(|| super::metal_detile::kernel_params(layout, mip, slice)).flatten()
            })
            .collect();
        let mut steps = Vec::new();
        if let (Some((src, src_offset)), Some(d)) = (&window, detiler) {
            for (&(_, _, at, ..), p) in plan.iter().zip(&params) {
                if let Some(p) = p {
                    d.encode(&mut steps, src, *src_offset, &buffer, at, p);
                }
            }
        }
        // SAFETY: committed above; the GPU writes other ranges of it.
        let linear = unsafe { std::slice::from_raw_parts_mut(staging.cast::<u8>(), size) };
        for (&(slice, mip, at, row, image, md, _), p) in plan.iter().zip(&params) {
            if p.is_some() {
                continue;
            }
            let end = (at + image * md) as usize;
            if layout.detile(tiled, mip, slice, &mut linear[at as usize..end], row as usize, image as usize).is_err() {
                drop(steps);
                drop(buffer);
                // SAFETY: ours, the GPU never saw it.
                unsafe { VirtualFree(staging, 0, MEM_RELEASE) };
                return;
            }
        }
        if let Ok(cb) = self.queue.command_buffer() {
            let detiled = steps.is_empty() || cb.compute(&steps).is_ok();
            if detiled && let Ok(mut blit) = cb.blit_encoder() {
                for &(slice, mip, at, row, image, _md, size3) in &plan {
                    blit.copy_from_buffer_to_texture(&buffer, at, row as u32, image as u32, size3, &t.texture, slice, mip, [0, 0, 0]);
                }
                blit.finish();
            }
            cb.commit();
            let _ = cb.wait();
        }
        drop(buffer);
        // SAFETY: the GPU is done with it.
        unsafe { VirtualFree(staging, 0, MEM_RELEASE) };
    }

    /// The entry of a texture view: `words` is the D3D12 SRV (or, with
    /// `unordered`, UAV) description, or `None` for the whole texture.
    pub(crate) fn texture_entry(&self, r: This, words: Option<&[u32]>, unordered: bool) -> Option<DescriptorEntry> {
        if r.is_null() {
            return self.null_texture(words, unordered);
        }
        let mut key_words = [u32::MAX; 10];
        if let Some(w) = words {
            for (k, v) in key_words.iter_mut().zip(w) {
                *k = *v;
            }
        }
        let key = (r as usize, key_words, unordered);
        if let Some((_, e)) = self.views.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
            return Some(*e);
        }
        let t = self.texture_for(r)?;
        let (format, type_, levels, slices, swizzle, min_lod) = match words {
            None => (t.pixel, None, (0, t.mips), (0, t.slices), identity(), 0.0),
            Some(w) => {
                let Some(v) = (if unordered { uav_view(w, &t) } else { srv_view(w, &t) }) else {
                    fail(if unordered { "UAV description" } else { "SRV description" }, Some(&t.desc), Some(w));
                    return None;
                };
                let format = view_format(&t, w[0]);
                (format, v.type_, v.levels, v.slices, v.swizzle, v.min_lod)
            }
        };
        let type_ = type_.unwrap_or(match t.desc.dimension {
            TEXTURE_1D if t.slices > 1 => WMTTextureType::WMTTextureType1DArray,
            TEXTURE_1D => WMTTextureType::WMTTextureType1D,
            TEXTURE_3D => WMTTextureType::WMTTextureType3D,
            _ if t.slices > 1 => WMTTextureType::WMTTextureType2DArray,
            _ => WMTTextureType::WMTTextureType2D,
        });
        let levels = (levels.0.min(t.mips - 1), levels.1.min(t.mips - levels.0.min(t.mips - 1)).max(1));
        let slices = (slices.0.min(t.slices - 1), slices.1.min(t.slices - slices.0.min(t.slices - 1)).max(1));
        let type_ = view_type(&t, type_, slices.1);
        let mut id = 0u64;
        let _pool = victory_metal::Pool::new();
        // SAFETY: a live texture; the view is kept below.
        let view = unsafe {
            sys::MTLTexture_newTextureView(
                t.texture.handle(),
                format,
                type_,
                levels.0 as u16,
                levels.1 as u16,
                slices.0 as u16,
                slices.1 as u16,
                swizzle,
                &mut id,
            )
        };
        let Some(view) = victory_metal::Object::adopt(view) else {
            fail("newTextureView failed", Some(&t.desc), words);
            return None;
        };
        if id == 0 {
            id = view.gpu_resource_id();
        }
        if id == 0 {
            fail("no gpuResourceID", Some(&t.desc), words);
            return None;
        }
        let entry = DescriptorEntry::texture(id, min_lod, 0);
        self.views.lock().unwrap_or_else(|e| e.into_inner()).insert(key, (view, entry));
        Some(entry)
    }
}

impl MetalBackend {
    /// A null view (D3D12 reads zeros, drops writes): a zeroed 1x1 texture
    /// of the view's type and format, one per (type, format), shared.
    /// Metal has no null texture in an argument buffer that is safe to read.
    fn null_texture(&self, words: Option<&[u32]>, unordered: bool) -> Option<DescriptorEntry> {
        use WMTTextureType as T;
        let dim = words.and_then(|w| w.get(1).copied()).unwrap_or(4);
        let (type_, slices) = match dim {
            2 => (T::WMTTextureType1D, 1),
            3 => (T::WMTTextureType1DArray, 1),
            5 => (T::WMTTextureType2DArray, 1),
            8 => (T::WMTTextureType3D, 1),
            9 if !unordered => (T::WMTTextureTypeCube, 6),
            10 if !unordered => (T::WMTTextureTypeCubeArray, 6),
            _ => (T::WMTTextureType2D, 1),
        };
        let format = words
            .and_then(|w| w.first().copied())
            .filter(|&f| f != 0)
            .filter(|&f| format_props(crate::resource::pc_format(f) as i32).is_some_and(|p| !p.depth_stencil && p.block_width <= 1))
            .and_then(pixel_format)
            .unwrap_or(WMTPixelFormat::WMTPixelFormatRGBA8Unorm);
        let key = (0usize, [type_.0 as u32, format.0, u32::MAX, 0, 0, 0, 0, 0, 0, 0], unordered);
        if let Some((_, e)) = self.views.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
            return Some(*e);
        }
        let desc = TextureDescriptor {
            pixel_format: format,
            width: 1,
            height: 1,
            depth: 1,
            array_length: 1, // one slice, or one cube (whose 6 faces are `slices`)
            type_,
            mipmap_level_count: 1,
            sample_count: 1,
            usage: WMTTextureUsage(1 | 2),
            options: sys::WMTResourceOptions(0x20),
        };
        let _pool = victory_metal::Pool::new();
        let texture = match self.device.new_texture(&desc) {
            Ok(t) => t,
            Err(_) => {
                fail("null texture", None, words);
                return None;
            }
        };
        {
            let chunks = self.chunks();
            chunks.residency().add(&[texture.handle()]);
            chunks.residency().commit();
        }
        // zeros from a fresh (zeroed) allocation, into every face
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        const SIZE: usize = 64 << 10;
        // SAFETY: a fresh allocation, released after the GPU is done.
        let staging = unsafe { VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        if !staging.is_null() {
            // SAFETY: lives until the blit is waited for.
            if let Ok(buffer) = unsafe { self.device.wrap(staging, SIZE as u64) }
                && let Ok(cb) = self.queue.command_buffer()
                && let Ok(mut blit) = cb.blit_encoder()
            {
                for slice in 0..slices {
                    blit.copy_from_buffer_to_texture(&buffer, 0, 256, 256, [1, 1, 1], &texture, slice, 0, [0, 0, 0]);
                }
                blit.finish();
                cb.commit();
                let _ = cb.wait();
            }
            // SAFETY: ours, the GPU is done.
            unsafe { VirtualFree(staging, 0, MEM_RELEASE) };
        }
        let id = texture.gpu_resource_id();
        if id == 0 {
            fail("null texture id", None, words);
            return None;
        }
        let entry = DescriptorEntry::texture(id, 0.0, 0);
        self.views.lock().unwrap_or_else(|e| e.into_inner()).insert(key, (texture.into_object(), entry));
        Some(entry)
    }
}

/// The pixel format of a view in DXGI format `wanted` of texture `t`, as
/// Metal allows it (a disallowed view is an assertion that kills the
/// thread, not an error):
/// - a depth texture: a stencil view for 8-bit integer / stencil formats,
///   else its own depth format (D3D12 reads depth as a colour format; a
///   depth view samples the same value in red);
/// - a colour texture: `wanted` only if its texel or block size is the
///   texture's (Metal's pixel format views), else its own format.
fn view_format(t: &MetalTexture, wanted: u32) -> WMTPixelFormat {
    if wanted == 0 {
        return t.pixel;
    }
    let own = format_props(crate::resource::pc_format(t.desc.format) as i32);
    let want = format_props(crate::resource::pc_format(wanted) as i32);
    let (Some(own), Some(want)) = (own, want) else { return t.pixel };
    if own.depth_stencil {
        // R8_UINT (62), X24_TYPELESS_G8_UINT (47), X32_TYPELESS_G8X24_UINT (21)
        let stencil = matches!(wanted, 62 | 47 | 21);
        return match (stencil, t.pixel) {
            (true, WMTPixelFormat::WMTPixelFormatDepth32Float_Stencil8) => WMTPixelFormat::WMTPixelFormatX32_Stencil8,
            _ => t.pixel,
        };
    }
    if want.depth_stencil
        || want.block_size_bytes != own.block_size_bytes
        || want.block_width != own.block_width
        || want.block_height != own.block_height
    {
        return t.pixel;
    }
    pixel_format(wanted).unwrap_or(t.pixel)
}

/// The view type `wanted` if Metal allows it on `t` (a disallowed view type
/// is an assertion, like a format), else the texture's own type.
fn view_type(t: &MetalTexture, wanted: WMTTextureType, slices: u32) -> WMTTextureType {
    use WMTTextureType as T;
    let base = match t.desc.dimension {
        TEXTURE_1D if t.slices > 1 => T::WMTTextureType1DArray,
        TEXTURE_1D => T::WMTTextureType1D,
        TEXTURE_3D => T::WMTTextureType3D,
        _ if t.desc.sample_count > 1 => T::WMTTextureType2DMultisample,
        _ if t.slices > 1 => T::WMTTextureType2DArray,
        _ => T::WMTTextureType2D,
    };
    let square = u64::from(t.desc.height) == t.desc.width;
    let ok = match (base, wanted) {
        (b, w) if b == w => true,
        (T::WMTTextureType1D | T::WMTTextureType1DArray, T::WMTTextureType1D | T::WMTTextureType1DArray) => true,
        (T::WMTTextureType2D | T::WMTTextureType2DArray, T::WMTTextureType2D | T::WMTTextureType2DArray) => true,
        (T::WMTTextureType2DArray, T::WMTTextureTypeCube) => square && slices == 6,
        (T::WMTTextureType2DArray, T::WMTTextureTypeCubeArray) => square && slices % 6 == 0 && slices > 0,
        _ => false,
    };
    if !ok {
        return base;
    }
    // A single-slice 2D view of an array, or an array view of one slice.
    match wanted {
        T::WMTTextureType2D if slices > 1 => T::WMTTextureType2DArray,
        w => w,
    }
}

fn identity() -> WMTTextureSwizzleChannels {
    WMTTextureSwizzleChannels {
        r: WMTTextureSwizzle(2),
        g: WMTTextureSwizzle(3),
        b: WMTTextureSwizzle(4),
        a: WMTTextureSwizzle(5),
    }
}

/// D3D12's 4-component mapping word (3 bits per channel: 0–3 a source
/// channel, 4 zero, 5 one) as Metal swizzle channels.
fn swizzle(mapping: u32) -> WMTTextureSwizzleChannels {
    let c = |i: u32| {
        WMTTextureSwizzle(match (mapping >> (3 * i)) & 7 {
            n @ 0..=3 => n as u8 + 2,
            4 => 0,
            _ => 1,
        })
    };
    WMTTextureSwizzleChannels { r: c(0), g: c(1), b: c(2), a: c(3) }
}

struct ViewShape {
    type_: Option<WMTTextureType>,
    levels: (u32, u32),
    slices: (u32, u32),
    swizzle: WMTTextureSwizzleChannels,
    min_lod: f32,
}

/// `D3D12_SHADER_RESOURCE_VIEW_DESC` words: format, dimension, mapping,
/// then (word 4) the dimension's fields.
fn srv_view(w: &[u32], t: &MetalTexture) -> Option<ViewShape> {
    if w.len() < 10 {
        return None;
    }
    let all = |n: u32, max: u32| if n == u32::MAX || n == 0 { max } else { n };
    let (type_, levels, slices, min_lod) = match w[1] {
        2 => (WMTTextureType::WMTTextureType1D, (w[4], all(w[5], t.mips)), (0, 1), f32::from_bits(w[6])),
        3 => (WMTTextureType::WMTTextureType1DArray, (w[4], all(w[5], t.mips)), (w[6], w[7]), f32::from_bits(w[8])),
        4 => (WMTTextureType::WMTTextureType2D, (w[4], all(w[5], t.mips)), (0, 1), f32::from_bits(w[7])),
        5 => (WMTTextureType::WMTTextureType2DArray, (w[4], all(w[5], t.mips)), (w[6], w[7]), f32::from_bits(w[9])),
        6 => (WMTTextureType::WMTTextureType2DMultisample, (0, 1), (0, 1), 0.0),
        8 => (WMTTextureType::WMTTextureType3D, (w[4], all(w[5], t.mips)), (0, 1), f32::from_bits(w[6])),
        9 => (WMTTextureType::WMTTextureTypeCube, (w[4], all(w[5], t.mips)), (0, 6), f32::from_bits(w[6])),
        10 => (WMTTextureType::WMTTextureTypeCubeArray, (w[4], all(w[5], t.mips)), (w[6], w[7] * 6), f32::from_bits(w[8])),
        _ => return None,
    };
    Some(ViewShape { type_: Some(type_), levels, slices, swizzle: swizzle(w[2]), min_lod })
}

/// `D3D12_UNORDERED_ACCESS_VIEW_DESC` words: format, dimension, then (word
/// 2) the dimension's fields; one mip.
fn uav_view(w: &[u32], _t: &MetalTexture) -> Option<ViewShape> {
    if w.len() < 6 {
        return None;
    }
    let (type_, mip, slices) = match w[1] {
        2 => (WMTTextureType::WMTTextureType1D, w[2], (0, 1)),
        3 => (WMTTextureType::WMTTextureType1DArray, w[2], (w[3], w[4])),
        4 => (WMTTextureType::WMTTextureType2D, w[2], (0, 1)),
        5 => (WMTTextureType::WMTTextureType2DArray, w[2], (w[3], w[4])),
        8 => (WMTTextureType::WMTTextureType3D, w[2], (0, 1)),
        _ => return None,
    };
    Some(ViewShape { type_: Some(type_), levels: (mip, 1), slices, swizzle: identity(), min_lod: 0.0 })
}
