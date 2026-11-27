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
    pub(crate) companion: std::sync::OnceLock<Option<super::metal_depth::DepthCompanion>>,
}

impl MetalTexture {
    pub(crate) fn depth_companion(&self, device: &victory_metal::Device, dsv: u32) -> Option<super::metal_depth::DepthCompanion> {
        use super::metal_depth::DepthCompanion;
        let pixel = DepthCompanion::format(self.pixel)?;
        if dsv != 0 && pixel_format(dsv) != Some(pixel) {
            return None;
        }
        self.companion.get_or_init(|| {
            let _pool = victory_metal::Pool::new();
            match DepthCompanion::new(device, self, pixel) {
                Ok(companion) => Some(companion),
                Err(error) => {
                    tracing::debug!(%error, format = self.desc.format, "Metal: depth companion unavailable");
                    None
                }
            }
        }).clone()
    }
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

#[derive(Clone, Copy)]
struct PartialEntry {
    mem: u64,
    size: u64,
    first_mip: u32,
    dirty: bool,
}

/// Title memory of title-filled Metal textures: address to resource.
static FILLED: std::sync::Mutex<std::collections::BTreeMap<usize, usize>> = std::sync::Mutex::new(std::collections::BTreeMap::new());
/// Title-filled textures that only had tail mips committed on first fill.
static PARTIAL_FILLED: std::sync::Mutex<std::collections::BTreeMap<usize, PartialEntry>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());
static PARTIAL_DIRTY: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static PARTIAL_HAS_REMAPS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static PARTIAL_ALL_DIRTY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static REMAP_RANGES: std::sync::Mutex<Vec<(u64, u64)>> = std::sync::Mutex::new(Vec::new());

pub(crate) fn tracks_memory(address: usize) -> bool {
    FILLED.lock().unwrap_or_else(|e| e.into_inner()).contains_key(&address)
}

pub(crate) fn forget_resource_textures(resource: usize) {
    FILLED.lock().unwrap_or_else(|e| e.into_inner()).retain(|_, &mut r| r != resource);
    if matches!(PARTIAL_FILLED.lock().unwrap_or_else(|e| e.into_inner()).remove(&resource), Some(e) if e.dirty) {
        PARTIAL_DIRTY.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}

/// Records a remapped title range in O(1) while the runtime holds `MAPPING.write()`.
/// Never touches `PARTIAL_FILLED`; [`drain_remap_partial`] applies the batch later.
pub(crate) fn note_remap_partial(at: u64, len: u64) {
    if len == 0 {
        return;
    }
    let end = at.saturating_add(len);
    {
        let mut ranges = REMAP_RANGES.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(last) = ranges.last_mut()
            && last.1 == at
        {
            last.1 = end;
        } else if ranges.len() < 65536 {
            ranges.push((at, end));
        } else {
            PARTIAL_ALL_DIRTY.store(true, std::sync::atomic::Ordering::Release);
            ranges.clear();
        }
    }
    PARTIAL_HAS_REMAPS.store(true, std::sync::atomic::Ordering::Release);
}

pub(crate) fn drain_remap_partial() {
    if !PARTIAL_HAS_REMAPS.swap(false, std::sync::atomic::Ordering::AcqRel) {
        return;
    }
    let all_dirty = PARTIAL_ALL_DIRTY.swap(false, std::sync::atomic::Ordering::AcqRel);
    let mut ranges = {
        let mut guard = REMAP_RANGES.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut *guard)
    };
    let mut partial = PARTIAL_FILLED.lock().unwrap_or_else(|e| e.into_inner());
    if partial.is_empty() {
        return;
    }
    if all_dirty {
        for v in partial.values_mut() {
            if !v.dirty {
                v.dirty = true;
                PARTIAL_DIRTY.fetch_add(1, std::sync::atomic::Ordering::Release);
            }
        }
        return;
    }
    if ranges.is_empty() {
        return;
    }
    ranges.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(ranges.len().min(4096));
    for (s, e) in ranges {
        if let Some(last) = merged.last_mut()
            && s <= last.1
        {
            last.1 = last.1.max(e);
        } else {
            merged.push((s, e));
        }
    }
    for v in partial.values_mut() {
        if v.dirty {
            continue;
        }
        let tex_end = v.mem.saturating_add(v.size);
        let idx = merged.partition_point(|&(s, _)| s < tex_end);
        if idx > 0 && merged[idx - 1].1 > v.mem {
            v.dirty = true;
            PARTIAL_DIRTY.fetch_add(1, std::sync::atomic::Ordering::Release);
        }
    }
}

const UPLOAD_ARENA_BYTES: u64 = 32 << 20;

struct UploadArena {
    buffer: Option<victory_metal::Buffer>,
    mem: *mut u8,
    capacity: u64,
    cursor: u64,
}

impl Drop for UploadArena {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
        // Release the Metal buffer wrapper before freeing the backing memory.
        self.buffer.take();
        if !self.mem.is_null() {
            // SAFETY: allocated by VirtualAlloc in UploadArena::new.
            unsafe { VirtualFree(self.mem.cast(), 0, MEM_RELEASE) };
            self.mem = std::ptr::null_mut();
        }
    }
}

struct PendingCopy {
    texture: victory_metal::texture::Texture,
    offset: u64,
    row: u32,
    image: u32,
    size: [u64; 3],
    slice: u32,
    mip: u32,
}

/// Reusable CPU-detile staging arena and queued texture blits flushed in one command buffer.
#[derive(Default)]
pub(crate) struct PendingUploads {
    arena: Option<UploadArena>,
    copies: Vec<PendingCopy>,
}

// SAFETY: only accessed under `MetalBackend::pending_uploads`.
unsafe impl Send for PendingUploads {}

impl PendingUploads {
    fn ensure_arena(&mut self, device: &victory_metal::Device) -> Option<&mut UploadArena> {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        if self.arena.is_none() {
            // SAFETY: fresh page-aligned 32 MiB upload arena freed on Drop.
            let mem = unsafe { VirtualAlloc(std::ptr::null(), UPLOAD_ARENA_BYTES as usize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
            if mem.is_null() {
                return None;
            }
            // SAFETY: kept alive inside `UploadArena` until `Drop`.
            let Ok(buffer) = (unsafe { device.wrap(mem, UPLOAD_ARENA_BYTES) }) else {
                unsafe { VirtualFree(mem, 0, MEM_RELEASE) };
                return None;
            };
            self.arena = Some(UploadArena {
                buffer: Some(buffer),
                mem: mem.cast(),
                capacity: UPLOAD_ARENA_BYTES,
                cursor: 0,
            });
        }
        self.arena.as_mut()
    }

    fn flush(&mut self, queue: &victory_metal::Queue) -> bool {
        if self.copies.is_empty() {
            return true;
        }
        let succeeded = if let Some(arena) = &mut self.arena
            && let Some(buf) = &arena.buffer
            && let Ok(cb) = queue.command_buffer()
            && let Ok(mut blit) = cb.blit_encoder()
        {
            for c in &self.copies {
                blit.copy_from_buffer_to_texture(buf, c.offset, c.row, c.image, c.size, &c.texture, c.slice, c.mip, [0, 0, 0]);
            }
            blit.finish();
            super::metal::submit_checked(&cb, &[], true, "texture uploads").is_ok()
        } else {
            false
        };
        if !succeeded {
            return false;
        }
        self.copies.clear();
        if let Some(arena) = &mut self.arena {
            arena.cursor = 0;
        }
        true
    }
}

impl MetalBackend {
    pub(crate) fn flush_uploads(&self) -> bool {
        if self.uploads_pending.load(std::sync::atomic::Ordering::Acquire) == 0 {
            return true;
        }
        let mut uploads = self.pending_uploads.lock().unwrap_or_else(|e| e.into_inner());
        if !uploads.flush(&self.queue) {
            super::metal::stop_submissions("texture upload could not complete");
            return false;
        }
        self.uploads_pending.store(0, std::sync::atomic::Ordering::Release);
        true
    }

    fn maybe_refill_partial(&self, key: usize) {
        if PARTIAL_DIRTY.load(std::sync::atomic::Ordering::Acquire) == 0 {
            return;
        }
        let entry = {
            let mut partial = PARTIAL_FILLED.lock().unwrap_or_else(|e| e.into_inner());
            partial.get_mut(&key).and_then(|v| {
                if v.dirty {
                    v.dirty = false;
                    PARTIAL_DIRTY.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
                    Some((v.mem, v.first_mip))
                } else {
                    None
                }
            })
        };
        let Some((mem, prev_first)) = entry else { return };
        if mem == 0 {
            return;
        }
        let Some(Some(t)) = self.textures.lock().unwrap_or_else(|e| e.into_inner()).get(&key).cloned() else { return };
        let first = self.fill(&t, mem as *const u8, prev_first);
        if first < prev_first {
            let mut partial = PARTIAL_FILLED.lock().unwrap_or_else(|e| e.into_inner());
            if first == 0 {
                if matches!(partial.remove(&key), Some(e) if e.dirty) {
                    PARTIAL_DIRTY.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
                }
            } else if let Some(v) = partial.get_mut(&key) {
                v.first_mip = first;
            }
        }
    }

    /// The Metal texture of title texture `r`, made (and for title-filled
    /// textures, filled) on first use.
    pub(crate) fn texture_for(&self, r: This) -> Option<Arc<MetalTexture>> {
        let key = r as usize;
        if let Some(t) = self.textures.lock().unwrap_or_else(|e| e.into_inner()).get(&key).cloned() {
            if t.is_some() {
                self.maybe_refill_partial(key);
            }
            return t;
        }
        let made = self.make_texture(r).map(Arc::new);
        self.textures.lock().unwrap_or_else(|e| e.into_inner()).entry(key).or_insert(made).clone()
    }

    fn make_texture(&self, r: This) -> Option<MetalTexture> {
        let Some((memory, d)) = texture_memory(r) else {
            fail("not a texture", None, None);
            return None;
        };
        let fmt = if d.flags & (RENDER_TARGET | UNORDERED) != 0 {
            crate::resource::target_format(crate::resource::pc_format(d.format))
        } else {
            d.format
        };
        let Some(pixel) = pixel_format(fmt) else {
            fail("no pixel format", Some(&d), None);
            return None;
        };
        // A texture the title may use as a depth target gets a depth format
        // (its typeless or colour format says only the size): Metal takes
        // only depth formats as depth attachments (a colour one there is a
        // GPU fault). Shaders still read it, depth sampling into red.
        // Not for one shaders also write: a depth view uses a companion.
        let pixel = if d.flags & DEPTH != 0 && d.flags & UNORDERED == 0 { depth_pixel(d.format).unwrap_or(pixel) } else { pixel };
        let mips = u32::from(d.mip_levels.max(1));
        // Metal Shader Converter lowers DXIL 1D textures to height-one 2D.
        let (type_, depth, slices) = match d.dimension {
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
            height: if d.dimension == TEXTURE_1D { 1 } else { d.height.max(1) },
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
        self.add_resident(&[texture.handle()]);
        let tiled_size = texture_layout(&d)
            .and_then(|l| l.planes.first().map(|p| p.layout.size))
            .unwrap_or(0);
        let t = MetalTexture {
            texture,
            pixel,
            desc: d,
            mips,
            slices,
            companion: Default::default(),
        };
        if d.flags & (RENDER_TARGET | DEPTH | UNORDERED) == 0 && super::metal::texture_fill_on() {
            let first = self.fill(&t, memory, t.mips);
            FILLED.lock().unwrap_or_else(|e| e.into_inner()).insert(memory as usize, r as usize);
            if first > 0 && tiled_size > 0 {
                PARTIAL_FILLED.lock().unwrap_or_else(|e| e.into_inner()).insert(
                    r as usize,
                    PartialEntry {
                        mem: memory as u64,
                        size: tiled_size,
                        first_mip: first,
                        dirty: false,
                    },
                );
            }
        }
        Some(t)
    }

    /// The title rewrote the memory of a title-filled texture (the D3D12
    /// path found it changed when a list sampled it): fills its Metal copy
    /// again. `VICTORY_METAL_REFILL=0` turns it off.
    pub(crate) fn refill_memory(&self, memory: *const u8) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var("VICTORY_METAL_REFILL").map_or(true, |v| v != "0")) {
            return;
        }
        let Some(r) = FILLED.lock().unwrap_or_else(|e| e.into_inner()).get(&(memory as usize)).copied() else { return };
        let Some(Some(t)) = self.textures.lock().unwrap_or_else(|e| e.into_inner()).get(&r).cloned() else { return };
        static COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n.is_power_of_two() {
            tracing::info!(refills = n, "Metal textures filled again from title memory");
        }
        let first = self.fill(&t, memory, t.mips);
        let mut partial = PARTIAL_FILLED.lock().unwrap_or_else(|e| e.into_inner());
        if first == 0 {
            if matches!(partial.remove(&r), Some(e) if e.dirty) {
                PARTIAL_DIRTY.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
            }
        } else if let Some(v) = partial.get_mut(&r) {
            v.first_mip = first;
        }
    }

    /// Detiles a title-filled texture's console memory into it (`first_mip..prev_first_mip`
    /// across slices of plane 0). Small CPU-detiled uploads batch into the shared upload arena;
    /// large/GPU-detiled uploads snapshot title memory out of `MAPPING.read()` first.
    /// Returns the finest mip filled (or `prev_first_mip` if nothing new was committed).
    fn fill(&self, t: &MetalTexture, memory: *const u8, prev_first_mip: u32) -> u32 {
        use std::cell::RefCell;
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        thread_local! {
            static TILE_SCRATCH: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
        }
        let Some(layout) = texture_layout(&t.desc) else { return prev_first_mip };
        let Some(plane) = layout.planes.first() else { return prev_first_mip };
        let layout = &plane.layout;
        if !layout.addressable() {
            return prev_first_mip;
        }
        let Some(props) = format_props(crate::resource::pc_format(t.desc.format) as i32) else { return prev_first_mip };
        let (bw, bh, bytes) = (props.block_width.max(1), props.block_height.max(1), props.block_size_bytes);
        let tiled_size = layout.size;
        if bytes == 0 || tiled_size == 0 || prev_first_mip == 0 {
            return prev_first_mip;
        }
        let (first_mip, read_bytes) = {
            let _reading = crate::watch::title_memory_reading_range(memory as usize, tiled_size as usize);
            let committed = victory_metal::chunks::committed_extent(memory as u64, tiled_size);
            let Some(first_mip) = (0..t.mips).find(|&m| layout.bytes_from(m) <= committed) else {
                return prev_first_mip;
            };
            if first_mip >= prev_first_mip {
                return prev_first_mip;
            }
            (first_mip, layout.bytes_from(first_mip))
        };
        let w = u64::from(t.desc.width as u32);
        let h = if t.desc.dimension == TEXTURE_1D { 1 } else { u64::from(t.desc.height.max(1)) };
        let d3 = if t.desc.dimension == TEXTURE_3D { u64::from(t.desc.depth_or_array_size.max(1)) } else { 1 };
        let mut plan = Vec::new();
        let mut total = 0u64;
        for slice in 0..t.slices {
            for mip in first_mip..prev_first_mip {
                let mw = (w >> mip).max(1).div_ceil(u64::from(bw));
                let mh = (h >> mip).max(1).div_ceil(u64::from(bh));
                let md = (d3 >> mip).max(1);
                let row = mw * u64::from(bytes);
                let image = row * mh;
                plan.push((slice, mip, total, row, image, md, [(w >> mip).max(1), (h >> mip).max(1), md]));
                total += (image * md).next_multiple_of(256);
            }
        }
        static MIN_BYTES: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let min_bytes = *MIN_BYTES.get_or_init(|| {
            std::env::var("VICTORY_METAL_GPU_DETILE_MIN_KB").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(4096) << 10
        });
        let worth_gpu = read_bytes >= min_bytes && layout.desc.swizzle.0 != 0 && super::metal::gpu_detile_on();
        // Fast batched path: snapshot title memory under the reading lock, then detile directly
        // into the reusable 32 MiB upload arena without VirtualAlloc or blocking on cb.wait().
        if !worth_gpu && total <= (16 << 20) {
            return TILE_SCRATCH.with(|scratch| {
                let mut scratch = scratch.borrow_mut();
                scratch.resize(read_bytes as usize, 0);
                {
                    let _reading = crate::watch::title_memory_reading_range(memory as usize, read_bytes as usize);
                    if victory_metal::chunks::committed_extent(memory as u64, read_bytes) < read_bytes {
                        return prev_first_mip;
                    }
                    // SAFETY: verified committed and guarded against remapping.
                    unsafe { std::ptr::copy_nonoverlapping(memory, scratch.as_mut_ptr(), read_bytes as usize) };
                }
                let need = total.next_multiple_of(256);
                let mut uploads = self.pending_uploads.lock().unwrap_or_else(|e| e.into_inner());
                if uploads.arena.as_ref().is_some_and(|a| a.cursor + need > a.capacity) {
                    if !uploads.flush(&self.queue) {
                        super::metal::stop_submissions("texture upload arena could not flush");
                        return prev_first_mip;
                    }
                    self.uploads_pending.store(0, std::sync::atomic::Ordering::Release);
                }
                let Some(arena) = uploads.ensure_arena(&self.device) else { return prev_first_mip };
                let base = arena.cursor;
                let dst_ptr = arena.mem;
                let mut batch_copies = Vec::with_capacity(plan.len());
                for &(slice, mip, at, row, image, md, size3) in &plan {
                    let offset = base + at;
                    let len = (image * md) as usize;
                    // SAFETY: `[offset .. offset + len)` is exclusively in `arena.mem` under `pending_uploads`.
                    let linear = unsafe { std::slice::from_raw_parts_mut(dst_ptr.add(offset as usize), len) };
                    if layout.detile(&scratch, mip, slice, linear, row as usize, image as usize).is_err() {
                        return prev_first_mip;
                    }
                    batch_copies.push(PendingCopy {
                        texture: t.texture.clone(),
                        offset,
                        row: row as u32,
                        image: image as u32,
                        size: size3,
                        slice,
                        mip,
                    });
                }
                arena.cursor = base + need;
                uploads.copies.extend(batch_copies);
                self.uploads_pending.fetch_add(1, std::sync::atomic::Ordering::Release);
                first_mip
            });
        }

        // Large / GPU-detile path: snapshot into page-aligned source outside MAPPING.read(),
        // then run GPU detile + blit in one command buffer completely unlocked from title memory.
        let src_size = (read_bytes as usize).next_multiple_of(64 << 10);
        // SAFETY: temporary snapshot buffer freed below.
        let src_mem = unsafe { VirtualAlloc(std::ptr::null(), src_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        if src_mem.is_null() {
            return prev_first_mip;
        }
        {
            let _reading = crate::watch::title_memory_reading_range(memory as usize, read_bytes as usize);
            if victory_metal::chunks::committed_extent(memory as u64, read_bytes) < read_bytes {
                // SAFETY: ours.
                unsafe { VirtualFree(src_mem, 0, MEM_RELEASE) };
                return prev_first_mip;
            }
            // SAFETY: verified committed and guarded against remapping.
            unsafe { std::ptr::copy_nonoverlapping(memory, src_mem.cast::<u8>(), read_bytes as usize) };
        }
        let size = total.next_multiple_of(64 << 10) as usize;
        // SAFETY: a fresh allocation, released below after the GPU is done.
        let staging = unsafe { VirtualAlloc(std::ptr::null(), size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        if staging.is_null() {
            unsafe { VirtualFree(src_mem, 0, MEM_RELEASE) };
            return prev_first_mip;
        }
        let _pool = victory_metal::Pool::new();
        // SAFETY: the snapshot and staging memory live until the command buffer is waited for.
        let Ok(src_buf) = (unsafe { self.device.wrap(src_mem, src_size as u64) }) else {
            unsafe { VirtualFree(staging, 0, MEM_RELEASE) };
            unsafe { VirtualFree(src_mem, 0, MEM_RELEASE) };
            return prev_first_mip;
        };
        let Ok(buffer) = (unsafe { self.device.wrap(staging, size as u64) }) else {
            drop(src_buf);
            unsafe { VirtualFree(staging, 0, MEM_RELEASE) };
            unsafe { VirtualFree(src_mem, 0, MEM_RELEASE) };
            return prev_first_mip;
        };
        let detiler = if worth_gpu { self.detiler() } else { None };
        let params: Vec<Option<super::metal_detile::Params>> = plan
            .iter()
            .map(|&(slice, mip, ..)| {
                detiler.and_then(|_| super::metal_detile::kernel_params(layout, mip, slice))
            })
            .collect();
        let mut steps = Vec::new();
        if let Some(d) = detiler {
            for (&(_, _, at, ..), p) in plan.iter().zip(&params) {
                if let Some(p) = p {
                    d.encode(&mut steps, &src_buf, 0, &buffer, at, p);
                }
            }
        }
        // SAFETY: committed above; snapshot covers `read_bytes`.
        let tiled = unsafe { std::slice::from_raw_parts(src_mem.cast::<u8>(), read_bytes as usize) };
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
                drop(src_buf);
                unsafe { VirtualFree(staging, 0, MEM_RELEASE) };
                unsafe { VirtualFree(src_mem, 0, MEM_RELEASE) };
                return prev_first_mip;
            }
        }
        let uploaded = if let Ok(cb) = self.queue.command_buffer() {
            let detiled = steps.is_empty() || cb.compute(&steps).is_ok();
            if detiled && let Ok(mut blit) = cb.blit_encoder() {
                for &(slice, mip, at, row, image, _md, size3) in &plan {
                    blit.copy_from_buffer_to_texture(&buffer, at, row as u32, image as u32, size3, &t.texture, slice, mip, [0, 0, 0]);
                }
                blit.finish();
                super::metal::submit_checked(&cb, &[], true, "texture detile and upload").is_ok()
            } else {
                false
            }
        } else {
            false
        };
        drop(buffer);
        drop(src_buf);
        // SAFETY: the GPU is done with both allocations.
        unsafe { VirtualFree(staging, 0, MEM_RELEASE) };
        unsafe { VirtualFree(src_mem, 0, MEM_RELEASE) };
        if uploaded {
            first_mip
        } else {
            super::metal::stop_submissions("texture detile or upload could not complete");
            prev_first_mip
        }
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
        if unordered {
            super::metal_draw::note_write(r as usize, 5);
        }
        self.maybe_refill_partial(r as usize);
        if let Some((_, e)) = self.views.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
            return Some(*e);
        }
        let t = self.texture_for(r)?;
        let (view, entry) = self.make_entry_or_sink(&t, words, unordered)?;
        self.views.lock().unwrap_or_else(|e| e.into_inner()).insert(key, (view, entry));
        Some(entry)
    }

    fn make_entry_or_sink(&self, t: &MetalTexture, words: Option<&[u32]>, unordered: bool) -> Option<(victory_metal::Object, DescriptorEntry)> {
        if let Some(w) = words.filter(|w| unordered && partial_3d_uav(w, t)) {
            return self.allocate_partial_3d_uav(t, w);
        }
        make_texture_entry(t, words, unordered)
    }

    /// Staging 3D texture for a partial-W-range 3D UAV: Metal 3D texture views
    /// can select a mip level but not a depth slice range, so we copy
    /// `[FirstWSlice .. FirstWSlice + WSize)` of `MipSlice` into a dedicated
    /// 3D texture of that depth. Writes stay isolated to the staging texture.
    fn allocate_partial_3d_uav(&self, t: &MetalTexture, w: &[u32]) -> Option<(victory_metal::Object, DescriptorEntry)> {
        let format = view_format(t, w[0]);
        let mip = w[2];
        if mip >= t.mips {
            fail("partial 3D UAV mip out of range", Some(&t.desc), Some(w));
            return self.allocate_null_texture(Some(w), WMTTextureType::WMTTextureType3D, 1, format);
        }
        let depth = u32::from(t.desc.depth_or_array_size).checked_shr(mip).unwrap_or(0).max(1);
        let first_w = w[3];
        if first_w >= depth || w[4] == 0 {
            fail("partial 3D UAV W range empty", Some(&t.desc), Some(w));
            return self.allocate_null_texture(Some(w), WMTTextureType::WMTTextureType3D, 1, format);
        }
        let w_size = if w[4] == u32::MAX { depth - first_w } else { w[4].min(depth - first_w) };
        let mw = (t.desc.width as u32 >> mip).max(1);
        let mh = (t.desc.height >> mip).max(1);
        let desc = TextureDescriptor {
            pixel_format: t.pixel,
            width: mw,
            height: mh,
            depth: w_size,
            array_length: 1,
            type_: WMTTextureType::WMTTextureType3D,
            mipmap_level_count: 1,
            sample_count: 1,
            usage: WMTTextureUsage(1 | 2 | 0x10),
            options: sys::WMTResourceOptions(0x20),
        };
        let _pool = victory_metal::Pool::new();
        let staging = match self.device.new_texture(&desc) {
            Ok(s) => s,
            Err(_) => {
                fail("partial 3D UAV staging texture", Some(&t.desc), Some(w));
                return None;
            }
        };
        let cb = self.queue.command_buffer().ok()?;
        let mut blit = cb.blit_encoder().ok()?;
        blit.copy_from_texture_to_texture(
            &t.texture,
            0,
            mip,
            [0, 0, u64::from(first_w)],
            [u64::from(mw), u64::from(mh), u64::from(w_size)],
            &staging,
            0,
            0,
            [0, 0, 0],
        );
        blit.finish();
        super::metal::submit_checked(&cb, &[], true, "partial texture upload").ok()?;
        self.add_resident(&[staging.handle()]);
        if format == t.pixel {
            let id = staging.gpu_resource_id();
            if id == 0 {
                fail("partial 3D UAV staging id", Some(&t.desc), Some(w));
                return None;
            }
            return Some((staging.into_object(), DescriptorEntry::texture(id, 0.0, 0)));
        }
        let mut id = 0u64;
        // SAFETY: `staging` is a live 3D texture with PixelFormatView usage; the view retains it.
        let view = unsafe {
            sys::MTLTexture_newTextureView(
                staging.handle(),
                format,
                WMTTextureType::WMTTextureType3D,
                0,
                1,
                0,
                1,
                identity(),
                &mut id,
            )
        };
        let Some(view) = victory_metal::Object::adopt(view) else {
            fail("partial 3D UAV staging view failed", Some(&t.desc), Some(w));
            return None;
        };
        if id == 0 {
            id = view.gpu_resource_id();
        }
        if id == 0 {
            fail("partial 3D UAV staging view id", Some(&t.desc), Some(w));
            return None;
        }
        Some((view, DescriptorEntry::texture(id, 0.0, 0)))
    }
}

fn make_texture_entry(t: &MetalTexture, words: Option<&[u32]>, unordered: bool) -> Option<(victory_metal::Object, DescriptorEntry)> {
    let (format, type_, levels, slices, swizzle, min_lod) = match words {
        None => (t.pixel, None, (0, t.mips), (0, t.slices), identity(), 0.0),
        Some(w) => {
            let Some(v) = (if unordered { uav_view(w, t) } else { srv_view(w, t) }) else {
                fail(if unordered { "UAV description" } else { "SRV description" }, Some(&t.desc), Some(w));
                return None;
            };
            let format = view_format(t, w[0]);
            (format, v.type_, v.levels, v.slices, v.swizzle, v.min_lod)
        }
    };
    let type_ = type_.unwrap_or_else(|| base_type(t));
    let levels = (levels.0.min(t.mips - 1), levels.1.min(t.mips - levels.0.min(t.mips - 1)).max(1));
    // D3D12 ResourceMinLODClamp is relative to resource mip 0; Metal views rebase mip 0 to levels.0.
    let min_lod = (min_lod - levels.0 as f32).max(0.0);
    let slices = (slices.0.min(t.slices - 1), slices.1.min(t.slices - slices.0.min(t.slices - 1)).max(1));
    let type_ = view_type(t, type_, slices.1);
    // No swizzle on depth formats (not something to risk an assertion on).
    let swizzle = if is_depth_pixel(format.0) { identity() } else { swizzle };
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
    Some((view, entry))
}

#[cfg(test)]
mod descriptor_coverage;

impl MetalBackend {
    /// A null view (D3D12 reads zeros, drops writes): a zeroed 1x1 texture
    /// of the view's type and format, one per (type, format), shared.
    /// Metal has no null texture in an argument buffer that is safe to read.
    fn null_texture(&self, words: Option<&[u32]>, unordered: bool) -> Option<DescriptorEntry> {
        let (type_, slices, format) = null_texture_shape(words, unordered);
        let key = (0usize, [type_.0 as u32, format.0, u32::MAX, 0, 0, 0, 0, 0, 0, 0], unordered);
        if let Some((_, e)) = self.views.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
            return Some(*e);
        }
        let (view, entry) = self.allocate_null_texture(words, type_, slices, format)?;
        self.views.lock().unwrap_or_else(|e| e.into_inner()).insert(key, (view, entry));
        Some(entry)
    }

    fn allocate_null_texture(
        &self,
        words: Option<&[u32]>,
        type_: WMTTextureType,
        slices: u32,
        format: WMTPixelFormat,
    ) -> Option<(victory_metal::Object, DescriptorEntry)> {
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
        let buffer = self.device.shared_buffer(&[0; 256]).ok()?;
        let cb = self.queue.command_buffer().ok()?;
        let mut blit = cb.blit_encoder().ok()?;
        for slice in 0..slices {
            blit.copy_from_buffer_to_texture(&buffer, 0, 256, 256, [1, 1, 1], &texture, slice, 0, [0, 0, 0]);
        }
        blit.finish();
        super::metal::submit_checked(&cb, &[], true, "null texture upload").ok()?;
        let id = texture.gpu_resource_id();
        if id == 0 {
            fail("null texture id", None, words);
            return None;
        }
        self.add_resident(&[texture.handle()]);
        let entry = DescriptorEntry::texture(id, 0.0, 0);
        Some((texture.into_object(), entry))
    }
}

fn null_texture_shape(words: Option<&[u32]>, unordered: bool) -> (WMTTextureType, u32, WMTPixelFormat) {
    use WMTTextureType as T;
    let dim = words.and_then(|w| w.get(1).copied()).unwrap_or(4);
    let (type_, slices) = match dim {
        3 | 5 => (T::WMTTextureType2DArray, 1),
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
    (type_, slices, format)
}

fn mtl_view_group(p: u32) -> Option<u8> {
    match p {
        10..=14 => Some(1),
        20..=34 => Some(2),
        53..=94 => Some(3),
        103..=115 => Some(4),
        123..=125 => Some(5),
        130 | 131 => Some(10),
        132 | 133 => Some(11),
        134 | 135 => Some(12),
        140 | 141 => Some(13),
        142 | 143 => Some(14),
        150 | 151 => Some(15),
        152 | 153 => Some(16),
        _ => None,
    }
}

/// The pixel format of a view in DXGI format `wanted` of texture `t`, as
/// Metal allows it (a disallowed view is an assertion that kills the
/// thread, not an error):
/// - a depth texture: a stencil view for 8-bit integer / stencil formats,
///   else its own depth format (D3D12 reads depth as a colour format; a
///   depth view samples the same value in red);
/// - a colour texture: `wanted` only if its Metal pixel format is
///   view-compatible with the backing texture's `t.pixel` (including
///   stand-in targets whose `t.pixel` differs from `t.desc.format`), else
///   `t.pixel`.
pub(crate) fn view_format(t: &MetalTexture, wanted: u32) -> WMTPixelFormat {
    if wanted == 0 {
        return t.pixel;
    }
    let own = format_props(crate::resource::pc_format(t.desc.format) as i32);
    // A depth texture by its Metal format (a typeless title format made a
    // depth one, `depth_pixel`): depth formats cast to nothing else.
    if own.is_some_and(|o| o.depth_stencil) || is_depth_pixel(t.pixel.0) {
        // R8_UINT (62), X24_TYPELESS_G8_UINT (47), X32_TYPELESS_G8X24_UINT (21)
        let stencil = matches!(wanted, 62 | 47 | 21);
        return match (stencil, t.pixel) {
            (true, WMTPixelFormat::WMTPixelFormatDepth32Float_Stencil8) => WMTPixelFormat::WMTPixelFormatX32_Stencil8,
            _ => t.pixel,
        };
    }
    let Some(candidate) = pixel_format(wanted) else {
        return t.pixel;
    };
    if candidate == t.pixel || is_depth_pixel(candidate.0) {
        return t.pixel;
    }
    match (mtl_view_group(t.pixel.0), mtl_view_group(candidate.0)) {
        (Some(a), Some(b)) if a == b => candidate,
        _ => t.pixel,
    }
}

/// The view type `wanted` if Metal allows it on `t` (a disallowed view type
/// is an assertion, like a format), else the texture's own type.
fn view_type(t: &MetalTexture, wanted: WMTTextureType, slices: u32) -> WMTTextureType {
    use WMTTextureType as T;
    let base = base_type(t);
    let square = u64::from(t.desc.height) == t.desc.width;
    let ok = match (base, wanted) {
        (b, w) if b == w => true,
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

pub(crate) fn identity() -> WMTTextureSwizzleChannels {
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
        2 => (WMTTextureType::WMTTextureType2D, (w[4], all(w[5], t.mips)), (0, 1), f32::from_bits(w[6])),
        3 => (WMTTextureType::WMTTextureType2DArray, (w[4], all(w[5], t.mips)), (w[6], w[7]), f32::from_bits(w[8])),
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
fn uav_view(w: &[u32], t: &MetalTexture) -> Option<ViewShape> {
    if w.len() < 6 {
        return None;
    }
    let (type_, mip, slices) = match w[1] {
        2 => (WMTTextureType::WMTTextureType2D, w[2], (0, 1)),
        3 => (WMTTextureType::WMTTextureType2DArray, w[2], (w[3], w[4])),
        4 => (WMTTextureType::WMTTextureType2D, w[2], (0, 1)),
        5 => (WMTTextureType::WMTTextureType2DArray, w[2], (w[3], w[4])),
        8 if !partial_3d_uav(w, t) => (WMTTextureType::WMTTextureType3D, w[2], (0, 1)),
        _ => return None,
    };
    Some(ViewShape { type_: Some(type_), levels: (mip, 1), slices, swizzle: identity(), min_lod: 0.0 })
}

// Metal 3D texture views can select mip levels, not a range of depth planes.
fn partial_3d_uav(w: &[u32], t: &MetalTexture) -> bool {
    if w.len() < 5 || w[1] != 8 || t.desc.dimension != TEXTURE_3D {
        return false;
    }
    let depth = u32::from(t.desc.depth_or_array_size).checked_shr(w[2]).unwrap_or(0).max(1);
    w[3] != 0 || (w[4] != depth && w[4] != u32::MAX)
}

/// The type of texture `t` itself (what an all-levels, all-slices view of
/// it has).
pub(crate) fn base_type(t: &MetalTexture) -> WMTTextureType {
    match t.desc.dimension {
        TEXTURE_3D => WMTTextureType::WMTTextureType3D,
        _ if t.desc.sample_count > 1 => WMTTextureType::WMTTextureType2DMultisample,
        _ if t.slices > 1 => WMTTextureType::WMTTextureType2DArray,
        _ => WMTTextureType::WMTTextureType2D,
    }
}

/// The Metal depth format for a texture of DXGI format `dxgi` used as a
/// depth target: its own if it is one, else by its size and planes (the
/// 24-bit depth families as Depth32Float_Stencil8, Apple GPUs having no
/// 24-bit depth).
pub(crate) fn depth_pixel(dxgi: u32) -> Option<WMTPixelFormat> {
    let pc = crate::resource::pc_format(dxgi);
    let props = format_props(pc as i32)?;
    if props.depth_stencil {
        return pixel_format(dxgi);
    }
    Some(match pc {
        // R24G8 and R32G8X24 families: depth with stencil
        44..=47 | 19..=22 => WMTPixelFormat::WMTPixelFormatDepth32Float_Stencil8,
        _ => match props.block_size_bytes {
            2 => WMTPixelFormat::WMTPixelFormatDepth16Unorm,
            4 => WMTPixelFormat::WMTPixelFormatDepth32Float,
            _ => return None,
        },
    })
}

/// Whether Metal pixel format `p` is a depth and/or stencil format.
pub(crate) fn is_depth_pixel(p: u32) -> bool {
    matches!(p, 250 | 252 | 253 | 255 | 260 | 261 | 262)
}
