//! The Metal backend's state (DESIGN.md "P1 implementation path", step 1):
//! the device and queue, title memory as GPU memory ([`Chunks`], sparse over
//! the whole address space: Metal Shader Converter's shaders get addresses
//! through descriptors and root parameters), and Metal pipelines for title
//! pipelines.
//!
//! Selected with `VICTORY_METAL` (`compute`: compute dispatches run on
//! Metal; meant for null-GPU runs while draws still go nowhere). Nothing
//! here runs unless it is set.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use victory_metal::chunks::Chunks;
use victory_metal::ir::{DescriptorEntry, TopLevel};
use victory_metal::msc::{RootParam, StaticSampler};
use victory_metal::{Buffer, ComputePipeline, Device, Queue};
use victory_shader::emit::{HostShader, ROOT_CBV_SPACE, USER_DATA_SPACE};

use crate::pipeline::host::HostPipeline;

/// What `VICTORY_METAL` asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Off,
    Compute,
}

pub(crate) fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_METAL").as_deref() {
        Ok("compute") => {
            tracing::warn!("VICTORY_METAL=compute: compute dispatches run on Metal (experimental)");
            Mode::Compute
        }
        Ok(other) if !other.is_empty() && other != "0" => {
            tracing::warn!(value = other, "VICTORY_METAL: unknown value, Metal stays off");
            Mode::Off
        }
        _ => Mode::Off,
    })
}

/// Whether title textures are made on Metal (`VICTORY_METAL_TEXTURES=0`:
/// texture slots stay empty, as before textures; `fill`: made but not
/// filled from title memory; `nofill` is the same).
pub(crate) fn textures_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_METAL_TEXTURES").map_or(true, |v| v != "0"))
}

/// Whether title-filled textures are detiled on the GPU
/// (`VICTORY_METAL_GPU_DETILE=0`: on the CPU, as before).
pub(crate) fn gpu_detile_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_METAL_GPU_DETILE").map_or(true, |v| v != "0"))
}

pub(crate) fn texture_fill_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_METAL_TEXTURES").map_or(true, |v| v != "nofill"))
}

/// Counts mapping changes. Recorded Metal lists can still contain old GPU
/// addresses, so the post-map notification cannot safely re-wrap memory.
pub(crate) fn title_memory_remapped(at: u64, len: u64) {
    static CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if n == 1 && std::env::var("VICTORY_METAL_REWRAP").is_ok_and(|v| v == "1") {
        tracing::warn!("VICTORY_METAL_REWRAP is disabled; pre-remap GPU quiescence is not wired up");
    }
    if n.is_power_of_two() {
        tracing::info!(calls = n, at = format_args!("{at:#x}"), len, "Metal: title memory remapped");
    }
}

impl MetalBackend {
    /// The no-copy wrappers of title memory.
    pub(crate) fn chunks(&self) -> std::sync::MutexGuard<'_, Chunks> {
        self.chunks.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Why dispatches did not run on Metal, counted for the log.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Drop {
    Ran,
    NoBackend,
    NoPipeline,
    MemoryDirectory,
    UserDataLayout,
    SlotMissing,
    SlotTyped,
    SlotUnmapped,
}

/// Empty slots by the kind of view (texture SRV, texture UAV, placed
/// texture, console sampler, D3D12 sampler refused, other).
static EMPTY: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

fn empty_kind(view: &crate::descriptor::View) -> usize {
    use crate::descriptor::View;
    match view {
        View::Shader(..) => 0,
        View::Unordered(..) => 1,
        View::Placed(..) => 2,
        View::ConsoleSampler(_) => 3,
        View::Sampler(_) => 4,
        View::Constant(..) => 5,
    }
}

static COUNTS: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

/// Counts one outcome; every 2^k dispatches (from 64Ki on) logs the totals.
pub(crate) fn count(what: Drop) {
    use std::sync::atomic::Ordering::Relaxed;
    COUNTS[what as usize].fetch_add(1, Relaxed);
    static ALL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = ALL.fetch_add(1, Relaxed) + 1;
    if n >= 1 << 16 && n.is_power_of_two() {
        let c: Vec<u64> = COUNTS.iter().map(|c| c.load(Relaxed)).collect();
        tracing::info!(
            dispatches = n,
            ran = c[0],
            no_backend = c[1],
            no_pipeline = c[2],
            memory_directory = c[3],
            user_data_layout = c[4],
            slot_missing = c[5],
            slot_typed = c[6],
            slot_unmapped = c[7],
            empty_texture_srv = EMPTY[0].load(Relaxed),
            empty_texture_uav = EMPTY[1].load(Relaxed),
            empty_placed = EMPTY[2].load(Relaxed),
            empty_console_sampler = EMPTY[3].load(Relaxed),
            empty_sampler_refused = EMPTY[4].load(Relaxed),
            empty_other = EMPTY[5].load(Relaxed),
            "Metal compute: outcomes so far (slot_* count dispatches that ran with an empty heap slot)"
        );
    }
}

/// A title pipeline as a Metal compute pipeline.
pub(crate) struct MetalCompute {
    pub(crate) pso: ComputePipeline,
    pub(crate) threadgroup: [u64; 3],
    /// An empty top-level argument buffer of its root signature.
    pub(crate) top_level: TopLevel,
}

/// One compute dispatch as recorded into a title list: everything resolved
/// at record time (root arguments are, as on the console), run when the
/// list is submitted ([`MetalBackend::run`]).
pub(crate) struct MetalCmd {
    pub(crate) pso: Arc<MetalCompute>,
    pub(crate) heap: Option<Buffer>,
    pub(crate) sampler_heap: Option<Buffer>,
    pub(crate) top: Vec<u8>,
    pub(crate) groups: [u64; 3],
    /// One-view tables: (offset in `top` of the table's address, its one
    /// entry), written into the submission's table ring when run.
    pub(crate) tables: Vec<(u32, DescriptorEntry)>,
}

/// What a title list records for Metal, in its order.
pub(crate) enum MetalOp {
    Dispatch(MetalCmd),
    Draw(super::metal_draw::MetalDraw),
    Clear(super::metal_draw::MetalClear),
    Copy(super::metal_draw::MetalTexCopy),
    Resolve(super::metal_draw::MetalResolve),
}

/// Memory for the one-view tables of a submission (resident, GPU reads it
/// in place); reused from the start by every submission, which is safe
/// because a submission is waited for before the next one runs.
struct Ring {
    buffer: Buffer,
    entries: *mut DescriptorEntry,
    capacity: usize,
}
// Only touched under the ring lock.
unsafe impl Send for Ring {}

pub(crate) struct MetalBackend {
    pub(crate) device: Device,
    pub(crate) queue: Queue,
    pub(crate) chunks: Mutex<Chunks>,
    /// By title pipeline object (its address); `None`: refused, not asked again.
    compute: Mutex<HashMap<usize, Option<Arc<MetalCompute>>>>,
    /// Metal view heaps (bind point 0), by title heap object.
    heaps: Mutex<HashMap<usize, MetalHeap>>,
    /// Texture buffer views for typed buffer views, by (title address,
    /// bytes, DXGI format): the entry to write (it keeps the view alive).
    typed: Mutex<HashMap<(u64, u64, u32), (victory_metal::texture::Texture, DescriptorEntry)>>,
    /// Title textures by resource object (`None`: could not be made).
    pub(crate) textures: Mutex<HashMap<usize, Option<Arc<super::metal_texture::MetalTexture>>>>,
    /// Texture views by (texture, view description words, unordered): the
    /// view (kept alive) and its entry.
    pub(crate) views: Mutex<HashMap<(usize, [u32; 10], bool), (victory_metal::Object, DescriptorEntry)>>,
    /// The one-view tables' memory; the lock also serializes submissions.
    ring: Mutex<Option<Ring>>,
    /// Sampler states by their D3D12 description words.
    samplers: Mutex<HashMap<[u32; 13], (victory_metal::sampler::SamplerState, DescriptorEntry)>>,
    /// The sampler states by their Metal description: D3D12 descriptions
    /// that differ only in what Metal does not keep (the LOD bias, which
    /// goes in the entry; a compare function with comparison off) share
    /// one. Apple7 samples wrong past 1024 distinct samplers (F8/F9).
    sampler_states: Mutex<HashMap<[u32; 11], victory_metal::sampler::SamplerState>>,
    /// The GPU detile kernels, compiled on first use (`None`: they failed).
    detiler: OnceLock<Option<super::metal_detile::Detiler>>,
    /// Graphics pipelines by (title pipeline object, rectangle emulation) (`None`: refused).
    pub(crate) graphics: Mutex<HashMap<(usize, bool), Option<Arc<super::metal_draw::MetalGraphics>>>>,
    /// Depth-stencil states by description.
    pub(crate) depth_stencils: Mutex<HashMap<victory_metal::render::DepthStencilDesc, victory_metal::render::DepthStencilState>>,
    /// Render target views in another pixel format, by (resource, format).
    pub(crate) target_views: Mutex<HashMap<(usize, u32), victory_metal::texture::Texture>>,
}

/// A title view heap as Metal Shader Converter descriptor entries, in
/// memory of our own the GPU reads in place. Slots are refreshed when a
/// shader that reads them is bound ([`MetalBackend::refresh_slot`]).
pub(crate) struct MetalHeap {
    pub(crate) buffer: Buffer,
    entries: *mut DescriptorEntry,
    count: u32,
}

// The entries are only written under the heaps lock.
unsafe impl Send for MetalHeap {}

static BACKEND: OnceLock<Option<MetalBackend>> = OnceLock::new();

#[cfg(feature = "diagnostics")]
pub(crate) fn stall_locks() -> impl Iterator<Item = (&'static str, usize)> {
    // A stall may occur while the backend is starting. Do not wait for it.
    BACKEND.get().and_then(Option::as_ref).into_iter().flat_map(|b| [
        ("metal.chunks", std::ptr::from_ref(&b.chunks) as usize),
        ("metal.typed", std::ptr::from_ref(&b.typed) as usize),
    ])
}

/// The backend, made on first use when [`mode`] is not `Off`.
pub(crate) fn backend() -> Option<&'static MetalBackend> {
    BACKEND
        .get_or_init(|| {
            if mode() == Mode::Off {
                return None;
            }
            match MetalBackend::new() {
                Ok(b) => {
                    tracing::info!(device = %b.device.name(), "Metal backend ready");
                    Some(b)
                }
                Err(e) => {
                    tracing::error!(error = %e, "Metal backend could not start (is victorymetal installed in this wine?)");
                    None
                }
            }
        })
        .as_ref()
}

impl MetalBackend {
    fn new() -> victory_metal::Result<Self> {
        if !victory_metal::sys::available() {
            return Err(victory_metal::Error("victorymetal.dll is not installed in this wine".into()));
        }
        let _pool = victory_metal::Pool::new();
        let device = Device::first()?;
        let chunks = Chunks::sparse(&device)?;
        let queue = device.new_queue(64)?;
        queue.add_residency_set(chunks.residency());
        Ok(Self {
            device,
            queue,
            chunks: Mutex::new(chunks),
            compute: Mutex::new(HashMap::new()),
            heaps: Mutex::new(HashMap::new()),
            typed: Mutex::new(HashMap::new()),
            samplers: Mutex::new(HashMap::new()),
            sampler_states: Mutex::new(HashMap::new()),
            detiler: OnceLock::new(),
            graphics: Mutex::new(HashMap::new()),
            depth_stencils: Mutex::new(HashMap::new()),
            target_views: Mutex::new(HashMap::new()),
            ring: Mutex::new(None),
            textures: Mutex::new(HashMap::new()),
            views: Mutex::new(HashMap::new()),
        })
    }

    /// The Metal pipeline for title compute pipeline `key` (its object's
    /// address), made from its host pipeline's translated shader on first
    /// use. `None` if Metal Shader Converter or Metal refused it.
    pub(crate) fn compute_pipeline(&self, key: usize, p: &HostPipeline) -> Option<Arc<MetalCompute>> {
        if let Some(known) = self.compute.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
            return known.clone();
        }
        let made = self.make_compute(p);
        if let Err(e) = &made {
            tracing::warn!(pipeline = %p.name, error = %e, "no Metal pipeline for a compute pipeline");
        }
        let made = made.ok().map(Arc::new);
        self.compute.lock().unwrap_or_else(|e| e.into_inner()).insert(key, made.clone());
        made
    }

    /// Runs what one submission recorded, in order, in one command buffer
    /// (dispatches in compute passes with a barrier between them, runs of
    /// draws to the same targets in one render pass each, clears as passes
    /// of their own) and waits for it: the title's fences may be signalled
    /// after this returns. False if it failed.
    pub(crate) fn run(&self, ops: &[Arc<MetalOp>]) -> bool {
        use super::metal_draw::{DrawKind, STAGE_VERTEX, draw_params, stage_buffer};
        use victory_metal::render::{PassColor, PassDepth, Render, RenderPass};
        use victory_metal::{Compute, msc, sys};
        if ops.is_empty() {
            return true;
        }
        let _pool = victory_metal::Pool::new();
        let mut ring = self.ring.lock().unwrap_or_else(|e| e.into_inner());
        // Ring space: the one-view tables' entries, then the bytes draws
        // bind (top levels, draw arguments, index type), 256-aligned.
        let entry = size_of::<DescriptorEntry>();
        let mut tables = 0usize;
        let mut blob = 0usize;
        for op in ops {
            match &**op {
                MetalOp::Dispatch(c) => tables += c.tables.len(),
                MetalOp::Draw(d) => {
                    for s in &d.stages {
                        tables += s.tables.len();
                        blob += s.top.len().next_multiple_of(256);
                    }
                    blob += if d.pso.geometry.is_some() { 1536 } else { 512 };
                    if let DrawKind::Generated { indices, .. } = &d.kind {
                        blob += (indices.len() * 4).next_multiple_of(256);
                    }
                }
                MetalOp::Clear(_) | MetalOp::Copy(_) | MetalOp::Resolve(_) => {}
            }
        }
        let need = tables + blob.div_ceil(entry) + 256 / entry + 1;
        if ring.as_ref().is_none_or(|r| r.capacity < need) {
            *ring = self.new_ring(need.max(1 << 16));
        }
        let Some(r) = ring.as_ref() else { return false };
        let base = r.entries.cast::<u8>();
        let mut next_entry = 0usize;
        let mut next_byte = (tables * entry).next_multiple_of(256);
        // Writes `top` with its tables' addresses: (bytes, ring offset if
        // put in the ring).
        let mut patch = |top: &[u8], tabs: &[(u32, DescriptorEntry)]| -> Vec<u8> {
            let mut top = top.to_vec();
            for &(at, e) in tabs {
                // SAFETY: next_entry < capacity (sized above); the GPU is
                // idle (the previous submission was waited for).
                unsafe { r.entries.add(next_entry).write(e) };
                let gpu = r.buffer.gpu_address + (next_entry * entry) as u64;
                let at = at as usize;
                if at + 8 <= top.len() {
                    top[at..at + 8].copy_from_slice(&gpu.to_le_bytes());
                }
                next_entry += 1;
            }
            top
        };
        let mut put = |bytes: &[u8]| -> u64 {
            let at = next_byte;
            // SAFETY: inside the ring (sized above).
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), base.add(at), bytes.len()) };
            next_byte = (at + bytes.len()).next_multiple_of(256);
            at as u64
        };
        // Per op: dispatch top bytes, or per draw its stages' top offsets,
        // draw-argument / uniforms-or-info offsets, and geometry vertex-table / zero-TLAB offsets.
        let mut dispatch_tops: Vec<Vec<u8>> = Vec::with_capacity(ops.len());
        let mut draw_offsets: Vec<(Vec<u64>, u64, u64, u64, u64)> = Vec::with_capacity(ops.len());
        for op in ops {
            match &**op {
                MetalOp::Dispatch(c) => {
                    dispatch_tops.push(patch(&c.top, &c.tables));
                    draw_offsets.push((Vec::new(), 0, 0, 0, 0));
                }
                MetalOp::Draw(d) => {
                    let tops = d.stages.iter().map(|s| { let t = patch(&s.top, &s.tables); put(&t) }).collect();
                    let (a, u, vt, zt) = match &d.kind {
                        DrawKind::Geometry { draw, .. } => {
                            let a = put(&draw.args);
                            let u = put(&draw.info);
                            let vt = if let Some(fetch) = &d.vertex {
                                put(&fetch.table)
                            } else {
                                put(&[0u8; msc::VERTEX_TABLE_SLOTS * 16])
                            };
                            let zt = put(&[0u8; 8]);
                            (a, u, vt, zt)
                        }
                        DrawKind::Generated { indices, .. } => {
                            // The indices in the ring; MSC's draw arguments
                            // take the index buffer's byte offset.
                            let bytes: Vec<u8> = indices.iter().flat_map(|v| v.to_le_bytes()).collect();
                            let io = put(&bytes);
                            let (mut args, index_type) = draw_params(&d.kind);
                            args[2] = io as u32;
                            let args: Vec<u8> = args.iter().flat_map(|v| v.to_le_bytes()).collect();
                            (put(&args), put(&index_type.to_le_bytes()), io, 0)
                        }
                        _ => {
                            let (args, index_type) = draw_params(&d.kind);
                            let args: Vec<u8> = args.iter().flat_map(|v| v.to_le_bytes()).collect();
                            (put(&args), put(&index_type.to_le_bytes()), 0, 0)
                        }
                    };
                    dispatch_tops.push(Vec::new());
                    draw_offsets.push((tops, a, u, vt, zt));
                }
                MetalOp::Clear(_) | MetalOp::Copy(_) | MetalOp::Resolve(_) => {
                    dispatch_tops.push(Vec::new());
                    draw_offsets.push((Vec::new(), 0, 0, 0, 0));
                }
            }
        }
        let Ok(mut cb) = self.queue.command_buffer() else { return false };
        // `VICTORY_METAL_SPLIT=1`: every pass in a command buffer of its own,
        // waited for, so a GPU fault names the pass that made it.
        static SPLIT: OnceLock<bool> = OnceLock::new();
        static CR: OnceLock<victory_metal::Result<victory_metal::clear_resolve::ClearResolve>> = OnceLock::new();
        let split = *SPLIT.get_or_init(|| std::env::var("VICTORY_METAL_SPLIT").is_ok_and(|v| v == "1"));
        let load = sys::WMTLoadAction::WMTLoadActionLoad;
        let store = sys::WMTStoreAction::WMTStoreActionStore;
        let mut depth_sync = super::metal_depth::DepthSync::default();
        let load_depth = |sync: &mut super::metal_depth::DepthSync, cb: &victory_metal::CommandBuffer, a: &super::metal_draw::Attachment| {
            a.companion.as_ref().is_none_or(|d| {
                encoded("depth companion load", sync.load(&self.device, cb, d, a.level, a.slice), || format!("format {} level {} slice {}", a.pixel, a.level, a.slice))
            })
        };
        let store_depth = |sync: &mut super::metal_depth::DepthSync, cb: &victory_metal::CommandBuffer, a: &super::metal_draw::Attachment| {
            a.companion.as_ref().is_none_or(|d| {
                encoded("depth companion store", sync.store(&self.device, cb, d, a.level, a.slice), || format!("format {} level {} slice {}", a.pixel, a.level, a.slice))
            })
        };
        let mut i = 0;
        let mut ok = true;
        while i < ops.len() && ok {
            let group_start = i;
            match &*ops[i] {
                MetalOp::Dispatch(_) => {
                    let mut steps = Vec::new();
                    while let Some(MetalOp::Dispatch(c)) = ops.get(i).map(|o| &**o) {
                        if !steps.is_empty() {
                            steps.push(Compute::Barrier);
                        }
                        steps.push(Compute::Pipeline { pipeline: &c.pso.pso, threads_per_group: c.pso.threadgroup });
                        if let Some(b) = &c.heap {
                            steps.push(Compute::Buffer { buffer: b, offset: 0, index: msc::DESCRIPTOR_HEAP_BIND_POINT });
                        }
                        if let Some(b) = &c.sampler_heap {
                            steps.push(Compute::Buffer { buffer: b, offset: 0, index: msc::SAMPLER_HEAP_BIND_POINT });
                        }
                        if !dispatch_tops[i].is_empty() {
                            steps.push(Compute::Bytes { bytes: &dispatch_tops[i], index: msc::ARGUMENT_BUFFER_BIND_POINT });
                        }
                        steps.push(Compute::Dispatch { groups: c.groups });
                        i += 1;
                    }
                    ok = encoded("compute pass", cb.compute(&steps), || String::new());
                    depth_sync.colour_may_change();
                }
                MetalOp::Copy(c) => {
                    if let Err(e) = super::metal_copy::encode_texture_to_texture(
                        &cb,
                        &c.src.texture,
                        c.src_slice,
                        c.src_level,
                        c.src_origin,
                        c.size,
                        &c.dst.texture,
                        c.dst_slice,
                        c.dst_level,
                        c.dst_origin,
                        c.whole,
                        None,
                    ) {
                        static SEEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                        if SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                            tracing::info!(error = %e, "Metal: a texture copy left out");
                        }
                    }
                    depth_sync.colour_may_change();
                    i += 1;
                }
                MetalOp::Resolve(r) => {
                    let result = match CR.get_or_init(|| victory_metal::clear_resolve::ClearResolve::new(&self.device)) {
                        Ok(cr) => cr.resolve_region(
                            &cb, &r.src, r.src_slice, &r.dst, r.dst_level, r.dst_slice, r.format, r.region,
                        ),
                        Err(e) => Err(e.clone()),
                    };
                    if let Err(error) = result {
                        static SEEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                        if SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                            tracing::warn!(%error, format = r.format.0, "Metal: a resolve left out");
                        }
                    }
                    i += 1;
                }
                MetalOp::Clear(c) if !c.rects.is_empty() => {
                    // Rectangles: F49's quad-per-rectangle clear.
                    if let Ok(cr) = CR.get_or_init(|| victory_metal::clear_resolve::ClearResolve::new(&self.device)) {
                        if !load_depth(&mut depth_sync, &cb, &c.target) {
                            return false;
                        }
                        let target = victory_metal::clear_resolve::ClearTarget {
                            texture: &c.target.texture,
                            pixel_format: sys::WMTPixelFormat(c.target.pixel),
                            level: c.target.level,
                            slice: c.target.slice,
                            samples: c.target.samples,
                        };
                        let value = match c.color {
                            Some(v) => victory_metal::clear_resolve::ClearValue::Color(v.map(|x| x as f32)),
                            None => victory_metal::clear_resolve::ClearValue::DepthStencil { depth: c.depth, stencil: c.stencil },
                        };
                        match cr.clear_rects(&cb, &target, &c.rects, value) {
                            Ok(()) => {
                                depth_sync.colour_may_change();
                                if c.depth.is_some() {
                                    ok = store_depth(&mut depth_sync, &cb, &c.target);
                                }
                            }
                            Err(e) => {
                                static SEEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                                if SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                                    tracing::info!(error = %e, "Metal: a rectangle clear left out");
                                }
                            }
                        }
                    }
                    i += 1;
                }
                MetalOp::Clear(c) => {
                    let clear = sys::WMTLoadAction::WMTLoadActionClear;
                    let mut pass = RenderPass { width: c.target.width, height: c.target.height, samples: c.target.samples, ..Default::default() };
                    if c.depth.is_none() && !load_depth(&mut depth_sync, &cb, &c.target) {
                        return false;
                    }
                    if let Some(color) = c.color {
                        pass.colors[0] = Some(PassColor { texture: &c.target.texture, level: c.target.level, slice: c.target.slice, load: clear, store, clear: color });
                    } else {
                        let d = PassDepth {
                            texture: &c.target.texture,
                            level: c.target.level,
                            slice: c.target.slice,
                            load,
                            store,
                            clear_depth: c.depth.unwrap_or(1.0),
                            clear_stencil: c.stencil.unwrap_or(0),
                        };
                        pass.depth = Some(PassDepth { load: if c.depth.is_some() { clear } else { load }, ..d });
                        if c.has_stencil {
                            pass.stencil = Some(PassDepth { load: if c.stencil.is_some() { clear } else { load }, ..d });
                        }
                    }
                    ok = encoded("clear pass", cb.render_pass_with(&pass, &[]), || format!("{}x{} format {} level {} slice {}", c.target.width, c.target.height, c.target.pixel, c.target.level, c.target.slice));
                    depth_sync.colour_may_change();
                    if ok && c.depth.is_some() {
                        ok = store_depth(&mut depth_sync, &cb, &c.target);
                    }
                    i += 1;
                }
                MetalOp::Draw(first) => {
                    let t = first.targets.clone();
                    if let Some(d) = &t.depth
                        && !load_depth(&mut depth_sync, &cb, d)
                    {
                        return false;
                    }
                    // Indirect records of this run of draws, translated on
                    // the GPU before its render pass opens (F42).
                    static TRANSLATOR: OnceLock<Option<victory_metal::indirect::IndirectTranslator>> = OnceLock::new();
                    let mut batches: Vec<Option<victory_metal::indirect::IndirectBatch>> = Vec::new();
                    {
                        let mut j = i;
                        while let Some(MetalOp::Draw(d)) = ops.get(j).map(|o| &**o) {
                            if !d.targets.same(&t) {
                                break;
                            }
                            let mut batch = None;
                            if let DrawKind::Indirect { records, records_offset, stride, max, count, index } = &d.kind
                                && let Some(tr) = TRANSLATOR.get_or_init(|| victory_metal::indirect::IndirectTranslator::new(&self.device).ok())
                            {
                                use victory_metal::indirect::{IndirectKind, IndirectSource};
                                let kind = match index {
                                    Some((_, _, ty, cap)) => IndirectKind::DrawIndexed { index_capacity: *cap, index_type: *ty },
                                    None => IndirectKind::Draw,
                                };
                                let source = IndirectSource {
                                    kind,
                                    records: (records, *records_offset),
                                    stride: *stride,
                                    max_count: *max,
                                    count: count.as_ref().map(|(b, o)| (b, *o)),
                                };
                                match tr.encode(&self.device, &cb, &source) {
                                    Ok(b) => batch = b,
                                    Err(e) => {
                                        static SEEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                                        if SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                                            tracing::info!(error = %e, "Metal: indirect records left out");
                                        }
                                    }
                                }
                            }
                            batches.push(batch);
                            j += 1;
                        }
                    }
                    let group_first = i;
                    let mut pass = RenderPass { width: t.width, height: t.height, samples: t.samples, ..Default::default() };
                    for (out, a) in pass.colors.iter_mut().zip(&t.colors) {
                        *out = a.as_ref().map(|a| PassColor { texture: &a.texture, level: a.level, slice: a.slice, load, store, clear: [0.0; 4] });
                    }
                    if let Some(d) = &t.depth {
                        let pd = PassDepth { texture: &d.texture, level: d.level, slice: d.slice, load, store, clear_depth: 1.0, clear_stencil: 0 };
                        pass.depth = Some(pd);
                        if t.stencil {
                            pass.stencil = Some(pd);
                        }
                    }
                    let mut steps: Vec<Render> = Vec::new();
                    let mut depth_written = false;
                    while let Some(MetalOp::Draw(d)) = ops.get(i).map(|o| &**o) {
                        if !d.targets.same(&t) {
                            break;
                        }
                        let g = &d.pso;
                        depth_written |= g.depth_write;
                        steps.push(Render::Pipeline { pipeline: &g.pso });
                        steps.push(Render::DepthStencil { state: &g.dss, stencil_ref: d.stencil_ref });
                        steps.push(Render::Rasterizer {
                            fill: g.raster[0],
                            cull: g.raster[1],
                            depth_clip: g.raster[2],
                            winding: g.raster[3],
                            depth_bias: g.bias[0],
                            slope_scale: g.bias[1],
                            clamp: g.bias[2],
                        });
                        steps.push(Render::BlendColorStencil { color: d.blend, stencil_ref: d.stencil_ref });
                        steps.push(Render::Viewports { viewports: std::slice::from_ref(&d.viewport) });
                        steps.push(Render::Scissors { rects: std::slice::from_ref(&d.scissor) });
                        let (tops, args, uniforms, vtable, zero_tlab) = &draw_offsets[i];
                        for (s, &top) in d.stages.iter().zip(tops) {
                            if let Some(b) = &s.heap {
                                steps.push(stage_buffer(s.bind, b, 0, msc::DESCRIPTOR_HEAP_BIND_POINT));
                            }
                            if let Some(b) = &s.sampler_heap {
                                steps.push(stage_buffer(s.bind, b, 0, msc::SAMPLER_HEAP_BIND_POINT));
                            }
                            if !s.top.is_empty() {
                                steps.push(stage_buffer(s.bind, &r.buffer, top, msc::ARGUMENT_BUFFER_BIND_POINT));
                            } else if s.bind == super::metal_draw::STAGE_OBJECT {
                                steps.push(Render::ObjectBuffer { buffer: &r.buffer, offset: *zero_tlab, index: msc::ARGUMENT_BUFFER_BIND_POINT });
                            }
                            if s.bind == STAGE_VERTEX {
                                // MSC's draw arguments and index type (IRRuntimeDrawPrimitives).
                                steps.push(stage_buffer(s.bind, &r.buffer, *args, msc::DRAW_ARGUMENTS_BIND_POINT));
                                steps.push(stage_buffer(s.bind, &r.buffer, *uniforms, msc::DRAW_UNIFORMS_BIND_POINT));
                                if let Some(fetch) = &d.vertex {
                                    steps.push(Render::VertexBytes { bytes: &fetch.table, index: msc::VERTEX_TABLE_BIND_POINT });
                                }
                            } else if s.bind == super::metal_draw::STAGE_OBJECT {
                                // MSC's geometry-emulation object/mesh ABI.
                                steps.push(Render::ObjectBuffer { buffer: &r.buffer, offset: *args, index: msc::DRAW_ARGUMENTS_BIND_POINT });
                                steps.push(Render::ObjectBuffer { buffer: &r.buffer, offset: *uniforms, index: msc::DRAW_UNIFORMS_BIND_POINT });
                                steps.push(Render::ObjectBuffer { buffer: &r.buffer, offset: *vtable, index: msc::VERTEX_TABLE_BIND_POINT });
                                steps.push(Render::MeshBuffer { buffer: &r.buffer, offset: *zero_tlab, index: msc::ARGUMENT_BUFFER_BIND_POINT });
                                steps.push(Render::MeshBuffer { buffer: &r.buffer, offset: *args, index: msc::DRAW_ARGUMENTS_BIND_POINT });
                                steps.push(Render::MeshBuffer { buffer: &r.buffer, offset: *uniforms, index: msc::DRAW_UNIFORMS_BIND_POINT });
                            }
                        }
                        let primitive = sys::WMTPrimitiveType(d.primitive);
                        steps.push(match &d.kind {
                            DrawKind::Vertices { count, instances, first, first_instance } => Render::Draw {
                                primitive_type: primitive,
                                vertex_start: u64::from(*first),
                                vertex_count: u64::from(*count),
                                instance_count: *instances,
                                base_instance: *first_instance,
                            },
                            DrawKind::Indexed { count, instances, base, first_instance, buffer, offset, index_type } => Render::DrawIndexed {
                                primitive_type: primitive,
                                index_type: *index_type,
                                index_count: u64::from(*count),
                                index_buffer: buffer,
                                index_offset: u64::from(*offset),
                                instance_count: *instances,
                                base_vertex: *base,
                                base_instance: *first_instance,
                            },
                            DrawKind::Mesh { groups } => Render::DrawMesh { groups: *groups, object_threads: [1, 1, 1], mesh_threads: g.mesh_threads },
                            DrawKind::Indirect { index, .. } => {
                                // One Metal indirect draw per record, with
                                // its MSC draw arguments.
                                if let Some(batch) = batches.get(i - group_first).and_then(Option::as_ref) {
                                    for r in 0..batch.record_count() {
                                        let Some(rec) = batch.record(r) else { continue };
                                        if let Some(bind) = rec.msc_vertex_bindings() {
                                            steps.extend(bind);
                                        }
                                        steps.push(match index {
                                            Some((ib, io, ty, _)) => Render::DrawIndexedIndirect {
                                                primitive_type: primitive,
                                                index_type: *ty,
                                                index_buffer: ib,
                                                index_offset: *io,
                                                arguments: rec.buffer,
                                                offset: rec.offset,
                                            },
                                            None => Render::DrawIndirect { primitive_type: primitive, arguments: rec.buffer, offset: rec.offset },
                                        });
                                    }
                                }
                                i += 1;
                                continue;
                            }
                            DrawKind::Generated { indices, instances, base, first_instance } => Render::DrawIndexed {
                                primitive_type: primitive,
                                index_type: 1,
                                index_count: indices.len() as u64,
                                index_buffer: &r.buffer,
                                index_offset: *vtable,
                                instance_count: *instances,
                                base_vertex: *base,
                                base_instance: *first_instance,
                            },
                            DrawKind::Geometry { draw, .. } => Render::DrawMesh {
                                groups: draw.groups,
                                object_threads: draw.object_threads,
                                mesh_threads: draw.mesh_threads,
                            },
                        });
                        i += 1;
                    }
                    ok = encoded("render pass", cb.render_pass_with(&pass, &steps), || {
                        let colors: Vec<u32> = t.colors.iter().map(|c| c.as_ref().map_or(0, |c| c.pixel)).collect();
                        format!("{}x{} samples {} colors {:?} depth {:?} stencil {}", t.width, t.height, t.samples, colors, t.depth.as_ref().map(|d| d.pixel), t.stencil)
                    });
                    depth_sync.colour_may_change();
                    if ok && let Some(a) = &t.depth {
                        if depth_written {
                            ok = store_depth(&mut depth_sync, &cb, a);
                        } else if let Some(d) = &a.companion {
                            depth_sync.preserve(d, a.level, a.slice);
                        }
                    }
                }
            }
            if split && ok {
                cb.commit();
                if let Err(e) = cb.wait() {
                    depth_sync.colour_may_change();
                    let what: Vec<String> = ops[group_start..i]
                        .iter()
                        .take(4)
                        .map(|o| match &**o {
                            MetalOp::Dispatch(_) => "dispatch".to_string(),
                            MetalOp::Draw(d) => format!("draw {}", d.pso.name),
                            MetalOp::Copy(c) => format!("copy {} → {}", c.src.pixel.0, c.dst.pixel.0),
                            MetalOp::Resolve(r) => format!("resolve {} mip {} slice {}", r.format.0, r.dst_level, r.dst_slice),
                            MetalOp::Clear(c) => {
                                let t = &c.target.texture;
                                format!(
                                    "clear {} format {} level {} slice {} ({}x{}, texture {}x{} mips {} layers {} samples {}){}",
                                    if c.color.is_some() { "color" } else { "depth" },
                                    c.target.pixel,
                                    c.target.level,
                                    c.target.slice,
                                    c.target.width,
                                    c.target.height,
                                    t.width(),
                                    t.height(),
                                    t.mipmap_level_count(),
                                    t.array_length(),
                                    c.target.samples,
                                    if c.has_stencil { " +stencil" } else { "" }
                                )
                            }
                        })
                        .collect();
                    // SAFETY: a live device.
                    let allocated = unsafe { sys::MTLDevice_currentAllocatedSize(self.device.handle()) } >> 20;
                    tracing::warn!(ops = i - group_start, first = ?what, allocated_mb = allocated, error = %e, "Metal: a pass failed on the GPU (split)");
                }
                let Ok(next) = self.queue.command_buffer() else { return false };
                cb = next;
            }
        }
        if !ok {
            return false;
        }
        // GPU-written ocean textures back into title memory (F44).
        if ops.iter().any(|o| matches!(**o, MetalOp::Dispatch(_))) {
            self.writeback_all(&cb);
        }
        cb.commit();
        match cb.wait() {
            Ok(()) => true,
            Err(e) => {
                let (mut dispatches, mut draws, mut clears) = (0, 0, 0);
                for op in ops {
                    match &**op {
                        MetalOp::Dispatch(_) => dispatches += 1,
                        MetalOp::Draw(_) => draws += 1,
                        MetalOp::Clear(_) | MetalOp::Copy(_) | MetalOp::Resolve(_) => clears += 1,
                    }
                }
                static SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                if SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                    tracing::warn!(dispatches, draws, clears, error = %e, "Metal: a submission failed on the GPU");
                }
                if draws + clears > 0 && !super::metal_draw::DRAWS_BROKEN.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    tracing::warn!("Metal: draws stop for this run (a submission with draws faulted); compute goes on");
                }
                false
            }
        }
    }

    /// Writes slot `index` of title heap `heap` (`count` slots) from the
    /// view at `handle`; returns the Metal heap buffer, and why the slot was
    /// left empty if it was.
    pub(crate) fn refresh_slot(
        &self,
        heap: usize,
        count: u32,
        index: u32,
        handle: u64,
    ) -> Option<(Buffer, Option<super::metal_heap::Missing>)> {
        let mut heaps = self.heaps.lock().unwrap_or_else(|e| e.into_inner());
        if !heaps.contains_key(&heap) {
            heaps.insert(heap, self.new_heap(count)?);
        }
        let h = heaps.get(&heap)?;
        if index >= h.count {
            return None;
        }
        let view = crate::descriptor::view(handle);
        let entry = match view.clone() {
            Some(crate::descriptor::View::Sampler(words)) => self.sampler_entry(&words).ok_or(super::metal_heap::Missing::NotYet),
            Some(crate::descriptor::View::ConsoleSampler(console)) => {
                // decoded to D3D12's description the way the D3D12 path does
                let d = crate::sampler::console(&console);
                let words = [
                    d.Filter.0 as u32,
                    d.AddressU.0 as u32,
                    d.AddressV.0 as u32,
                    d.AddressW.0 as u32,
                    d.MipLODBias.to_bits(),
                    d.MaxAnisotropy,
                    d.ComparisonFunc.0 as u32,
                    d.BorderColor[0].to_bits(),
                    d.BorderColor[1].to_bits(),
                    d.BorderColor[2].to_bits(),
                    d.BorderColor[3].to_bits(),
                    d.MinLOD.to_bits(),
                    d.MaxLOD.to_bits(),
                ];
                self.sampler_entry(&words).ok_or(super::metal_heap::Missing::NotYet)
            }
            Some(crate::descriptor::View::Shader(..) | crate::descriptor::View::Unordered(..) | crate::descriptor::View::Placed(false, ..))
                if !textures_on()
                    && !matches!(&view, Some(crate::descriptor::View::Placed(_, _, d)) if d[1] == 1)
                    && !matches!(&view, Some(crate::descriptor::View::Shader(r, _)) | Some(crate::descriptor::View::Unordered(r, _, _)) if crate::resource::buffer_bytes(*r as crate::object::This).is_some()) =>
            {
                Err(super::metal_heap::Missing::NotYet)
            }
            Some(crate::descriptor::View::Shader(r, desc)) if crate::resource::buffer_bytes(r as crate::object::This).is_none() => self
                .texture_entry(r as crate::object::This, desc.as_ref().map(|d| &d[..]), false)
                .ok_or(super::metal_heap::Missing::NotYet),
            Some(crate::descriptor::View::Unordered(r, _, desc)) if crate::resource::buffer_bytes(r as crate::object::This).is_none() => self
                .texture_entry(r as crate::object::This, desc.as_ref().map(|d| &d[..]), true)
                .ok_or(super::metal_heap::Missing::NotYet),
            Some(crate::descriptor::View::Placed(false, resource, desc)) if desc[1] != 1 => {
                // a placed texture view: one resource per (address, description)
                let address = super::metal_views::placed_address(&desc);
                crate::resource::texture_at(address, &resource)
                    .and_then(|t| self.texture_entry(t, Some(&desc[..10]), false))
                    .ok_or(super::metal_heap::Missing::NotYet)
            }
            Some(view) => {
                let e = {
                    let mut chunks = self.chunks();
                    super::metal_heap::entry(&view, &mut chunks)
                };
                match e {
                    Err(super::metal_heap::Missing::Typed { va, bytes, format }) => {
                        self.typed_entry(va, bytes, format).ok_or(super::metal_heap::Missing::Typed { va, bytes, format })
                    }
                    e => e,
                }
            }
            None => Ok(DescriptorEntry::NULL),
        };
        let missing = entry.err();
        if missing.is_some()
            && let Some(v) = &view
        {
            EMPTY[empty_kind(v)].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        // SAFETY: index < count, the entries' memory holds count entries.
        unsafe { h.entries.add(index as usize).write(entry.unwrap_or(DescriptorEntry::NULL)) };
        Some((h.buffer.clone(), missing))
    }

    /// The entry of a typed buffer view: a texture buffer view of the
    /// title range in the view's format, made once per (range, format).
    /// The view starts at an aligned offset; the rest goes in the entry's
    /// metadata as an offset in elements (at most 255, MSC's field).
    fn typed_entry(&self, va: u64, bytes: u64, format: u32) -> Option<DescriptorEntry> {
        use victory_metal::sys::{WMTPixelFormat, WMTTextureType, WMTTextureUsage};
        use victory_metal::texture::TextureDescriptor;
        let key = (va, bytes, format);
        if let Some((_, e)) = self.typed.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
            return Some(*e);
        }
        let pixel = super::metal_map::map_pixel_format(format as i32)?;
        let elem = u64::from(super::metal_map::format_props(format as i32)?.block_size_bytes);
        if elem == 0 {
            return None;
        }
        let (buffer, offset) = self.chunks().window_buffer(va, bytes)?;
        let pixel = WMTPixelFormat(pixel as u32);
        // SAFETY: a plain query.
        let align = unsafe { victory_metal::sys::MTLDevice_minimumLinearTextureAlignmentForPixelFormat(self.device.handle(), pixel) }.max(1);
        let start = offset / align * align;
        let skip = offset - start;
        if skip % elem != 0 || skip / elem > 255 {
            return None;
        }
        let width = (skip + bytes) / elem;
        let desc = TextureDescriptor {
            pixel_format: pixel,
            width: u32::try_from(width).ok()?,
            height: 1,
            depth: 1,
            array_length: 1,
            type_: WMTTextureType::WMTTextureTypeTextureBuffer,
            mipmap_level_count: 1,
            sample_count: 1,
            usage: WMTTextureUsage(1 | 2), // shader read | write
            ..Default::default()
        };
        let row = (width * elem).next_multiple_of(align);
        // SAFETY: the view lies inside the buffer (window_buffer covers
        // offset..offset+bytes); the buffer lives in the chunk manager and
        // the view keeps it alive.
        let view = unsafe { buffer.new_texture(&desc, start, row) }.ok()?;
        let id = view.gpu_resource_id();
        if id == 0 {
            return None;
        }
        let entry = DescriptorEntry::typed_buffer(buffer.gpu_address + offset, bytes, id, (skip / elem) as u32);
        self.typed.lock().unwrap_or_else(|e| e.into_inner()).insert(key, (view, entry));
        Some(entry)
    }

    /// The entry of a sampler from its D3D12_SAMPLER_DESC words (filter,
    /// address U/V/W, mip LOD bias, max anisotropy, comparison, border
    /// color ×4, min/max LOD): a sampler state usable from argument
    /// buffers, made once per description.
    /// The GPU detile kernels (compiled once).
    pub(crate) fn detiler(&self) -> Option<&super::metal_detile::Detiler> {
        self.detiler
            .get_or_init(|| match super::metal_detile::Detiler::new(&self.device) {
                Ok(d) => Some(d),
                Err(e) => {
                    tracing::warn!(error = %e, "Metal: the detile kernels did not compile; textures are detiled on the CPU");
                    None
                }
            })
            .as_ref()
    }

    fn sampler_entry(&self, w: &[u32; 13]) -> Option<DescriptorEntry> {
        use super::metal_map::{map_address_mode, map_border_color, map_compare_function, map_sampler_filter};
        use victory_metal::sampler::SamplerDescriptor;
        use victory_metal::sys::{WMTCompareFunction, WMTSamplerAddressMode, WMTSamplerBorderColor, WMTSamplerMinMagFilter, WMTSamplerMipFilter};
        if let Some((_, e)) = self.samplers.lock().unwrap_or_else(|e| e.into_inner()).get(w) {
            return Some(*e);
        }
        let f = map_sampler_filter(w[0] as i32)?;
        let addr = |v: u32| map_address_mode(v as i32).map(|m| WMTSamplerAddressMode(m as _));
        let lod_bias = f32::from_bits(w[4]);
        let border = map_border_color([f32::from_bits(w[7]), f32::from_bits(w[8]), f32::from_bits(w[9]), f32::from_bits(w[10])]);
        let compare = if f.compare { map_compare_function(w[6] as i32)? as u32 } else { 7 }; // MTLCompareFunctionAlways
        let desc = SamplerDescriptor {
            min_filter: WMTSamplerMinMagFilter(f.min_filter as _),
            mag_filter: WMTSamplerMinMagFilter(f.mag_filter as _),
            mip_filter: WMTSamplerMipFilter(f.mip_filter as _),
            s_address_mode: addr(w[1])?,
            t_address_mode: addr(w[2])?,
            r_address_mode: addr(w[3])?,
            border_color: WMTSamplerBorderColor(border.border_color as _),
            compare_function: WMTCompareFunction(compare as _),
            lod_min_clamp: f32::from_bits(w[11]).max(0.0),
            lod_max_clamp: f32::from_bits(w[12]).min(1000.0),
            max_anisotropy: if f.anisotropic { w[5].clamp(1, 16) } else { 1 },
            normalized_coords: true,
            lod_average: false,
            support_argument_buffers: true,
        };
        let key = [
            desc.min_filter.0 as u32,
            desc.mag_filter.0 as u32,
            desc.mip_filter.0 as u32,
            desc.s_address_mode.0 as u32,
            desc.t_address_mode.0 as u32,
            desc.r_address_mode.0 as u32,
            desc.border_color.0 as u32,
            desc.compare_function.0 as u32,
            desc.lod_min_clamp.to_bits(),
            desc.lod_max_clamp.to_bits(),
            desc.max_anisotropy,
        ];
        let state = {
            let mut states = self.sampler_states.lock().unwrap_or_else(|e| e.into_inner());
            match states.get(&key) {
                Some(s) => s.clone(),
                None => {
                    let s = self.device.new_sampler_state(&desc).ok()?;
                    states.insert(key, s.clone());
                    let n = states.len();
                    if n == 768 || n == 1024 || (n > 1024 && n.is_power_of_two()) {
                        tracing::warn!(distinct = n, "Metal: distinct sampler states (Apple7 samples wrong past 1024)");
                    }
                    s
                }
            }
        };
        let id = state.gpu_resource_id();
        if id == 0 {
            return None;
        }
        let entry = DescriptorEntry::sampler(id, lod_bias);
        self.samplers.lock().unwrap_or_else(|e| e.into_inner()).insert(*w, (state, entry));
        Some(entry)
    }

    fn new_ring(&self, capacity: usize) -> Option<Ring> {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        let bytes = (capacity * size_of::<DescriptorEntry>()).next_multiple_of(64 << 10);
        // SAFETY: a fresh allocation, kept for the run (a smaller old ring
        // is dropped with its buffer; nothing in flight uses it).
        let mem = unsafe { VirtualAlloc(std::ptr::null(), bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        if mem.is_null() {
            return None;
        }
        // SAFETY: as above.
        let buffer = unsafe { self.device.wrap(mem, bytes as u64) }.ok()?;
        let chunks = self.chunks();
        chunks.residency().add(&[buffer.handle()]);
        chunks.residency().commit();
        Some(Ring { buffer, entries: mem.cast(), capacity: bytes / size_of::<DescriptorEntry>() })
    }

    fn new_heap(&self, count: u32) -> Option<MetalHeap> {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        let bytes = (count as usize * size_of::<DescriptorEntry>()).next_multiple_of(64 << 10);
        // SAFETY: a fresh allocation (zero: every slot empty), never freed
        // (heaps live as long as the title's).
        let mem = unsafe { VirtualAlloc(std::ptr::null(), bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        if mem.is_null() {
            return None;
        }
        // SAFETY: as above.
        let buffer = unsafe { self.device.wrap(mem, bytes as u64) }.ok()?;
        let chunks = self.chunks();
        chunks.residency().add(&[buffer.handle()]);
        chunks.residency().commit();
        Some(MetalHeap { buffer, entries: mem.cast(), count })
    }

    fn make_compute(&self, p: &HostPipeline) -> Result<MetalCompute, String> {
        if !p.compute {
            return Err("not a compute pipeline".into());
        }
        let stage = p.stages.first().ok_or("no stage")?;
        let host = &stage.translated().host;
        let _pool = victory_metal::Pool::new();
        let params = root_params(host);
        let samplers = static_samplers(host);
        let path = msc_cache_path(&host.dxil, &params, &samplers);
        if let Some(path) = &path
            && let Ok(bytes) = crate::read_file(path)
            && let Some(converted) = victory_metal::msc::Converted::from_bytes(&bytes)
        {
            match self.device.compute_pipeline_converted(&converted) {
                Ok(pso) => {
                    return Ok(MetalCompute { pso, threadgroup: converted.threadgroup.map(u64::from), top_level: converted.top_level });
                }
                Err(e) => tracing::debug!(error = %e, "a cached Metal pipeline did not load; converting again"),
            }
        }
        let (pso, converted) = self.device.compute_pipeline_dxil(&host.dxil, &params, &samplers).map_err(|e| e.to_string())?;
        if let Some(path) = &path {
            let tmp = path.with_extension(format!("tmp{}", std::process::id()));
            if std::fs::write(&tmp, converted.to_bytes()).is_ok() {
                let _ = std::fs::rename(&tmp, path);
            }
        }
        Ok(MetalCompute { pso, threadgroup: converted.threadgroup.map(u64::from), top_level: converted.top_level })
    }
}


/// Where Metal Shader Converter's output for a shader and root signature is
/// kept between runs: `VICTORY_METAL_CACHE`, else `<VICTORY_CACHE_DIR>/metal`.
/// The name is a hash of the DXIL, the signature and the converter version.
pub(crate) fn msc_cache_path(
    dxil: &[u8],
    params: &[victory_metal::msc::RootParam],
    samplers: &[victory_metal::msc::StaticSampler],
) -> Option<std::path::PathBuf> {
    static DIR: OnceLock<Option<std::path::PathBuf>> = OnceLock::new();
    let dir = DIR.get_or_init(|| {
        let dir = std::env::var_os("VICTORY_METAL_CACHE")
            .filter(|v| !v.is_empty())
            .map(std::path::PathBuf::from)
            .or_else(|| std::env::var_os("VICTORY_CACHE_DIR").filter(|v| !v.is_empty()).map(|d| std::path::PathBuf::from(d).join("metal")))?;
        crate::make_dir(&dir).ok()?;
        tracing::info!(dir = %dir.display(), "Metal pipeline cache");
        Some(dir)
    });
    let dir = dir.as_ref()?;
    let mut h = blake3::Hasher::new();
    h.update(b"msc-4.0.3/bounds-check/v1\0");
    h.update(dxil);
    h.update(format!("{params:?}{samplers:?}").as_bytes());
    Some(dir.join(format!("{}.vmsc", &h.finalize().to_hex()[..32])))
}

/// One root parameter of a translated shader's Metal root signature, in
/// order: user data constants, root CBVs, then the title's root SRVs/UAVs
/// as one-view tables (bounded, as the D3D12 path's tables), victory's own
/// (strides, memory directory, faults) as root descriptors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RootSlot {
    Constants(u32),
    Cbv(u8),
    Srv(u8),
    Uav(u8),
    TableSrv(u8),
    TableUav(u8),
}

pub(crate) fn root_layout(t: &HostShader) -> Vec<RootSlot> {
    use victory_shader::emit::{SLOT_FAULTS, SLOT_MEMORY, SLOT_STRIDES};
    let mut out = Vec::new();
    let dwords = t.user_data_dwords();
    if dwords > 0 {
        out.push(RootSlot::Constants(dwords));
    }
    out.extend(t.root_cbvs.iter().map(|&s| RootSlot::Cbv(s)));
    out.extend(t.root_srvs.iter().map(|&s| if s == SLOT_STRIDES || s == SLOT_MEMORY { RootSlot::Srv(s) } else { RootSlot::TableSrv(s) }));
    out.extend(t.root_uavs.iter().map(|&s| if s == SLOT_FAULTS { RootSlot::Uav(s) } else { RootSlot::TableUav(s) }));
    out
}

/// The root signature of [`root_layout`] for Metal Shader Converter.
pub(crate) fn root_params(t: &HostShader) -> Vec<RootParam> {
    let space = ROOT_CBV_SPACE;
    root_layout(t)
        .into_iter()
        .map(|r| match r {
            RootSlot::Constants(num) => RootParam::Constants { register: 0, space: USER_DATA_SPACE, num },
            RootSlot::Cbv(s) => RootParam::Cbv { register: s.into(), space },
            RootSlot::Srv(s) => RootParam::Srv { register: s.into(), space },
            RootSlot::Uav(s) => RootParam::Uav { register: s.into(), space },
            RootSlot::TableSrv(s) => RootParam::TableSrv { register: s.into(), space },
            RootSlot::TableUav(s) => RootParam::TableUav { register: s.into(), space },
        })
        .collect()
}

/// The static samplers of a translated shader: `s<n>` in space 0.
pub(crate) fn static_samplers(t: &HostShader) -> Vec<StaticSampler> {
    t.static_samplers
        .iter()
        .enumerate()
        .map(|(i, s)| StaticSampler {
            filter: s.filter,
            address_u: s.address[0],
            address_v: s.address[1],
            address_w: s.address[2],
            mip_lod_bias: s.mip_lod_bias,
            max_anisotropy: s.max_anisotropy,
            comparison_func: s.comparison,
            border_color: s.border,
            min_lod: s.min_lod,
            max_lod: s.max_lod,
            shader_register: i as u32,
            register_space: 0,
        })
        .collect()
}

/// P2 census of vertex/pixel pipelines already accepted by the legacy host.
/// Each stage gets standard MSC conversion on a background thread; only
/// pipelines without a vertex input layout are tested for PSO creation.
/// This does not exercise the draw path's synthesized stage-in or mesh PSOs.
/// Off unless `VICTORY_METAL_CENSUS=1`.
pub(crate) fn census_graphics(
    shaders: Vec<std::sync::Arc<crate::pipeline::Shader>>,
    input: super::metal_render_map::PipelineStateInput,
) {
    static ON: OnceLock<bool> = OnceLock::new();
    if mode() != Mode::Compute || !*ON.get_or_init(|| std::env::var("VICTORY_METAL_CENSUS").is_ok_and(|v| v == "1")) {
        return;
    }
    type Job = (Vec<std::sync::Arc<crate::pipeline::Shader>>, super::metal_render_map::PipelineStateInput);
    static TX: OnceLock<Option<std::sync::Mutex<std::sync::mpsc::Sender<Job>>>> = OnceLock::new();
    let tx = TX.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("metal census".into())
            .spawn(move || {
                let mut c = Census::default();
                for (shaders, input) in rx {
                    c.one(&shaders, &input);
                }
            })
            .ok()?;
        Some(std::sync::Mutex::new(tx))
    });
    if let Some(tx) = tx {
        let _ = tx.lock().unwrap_or_else(|e| e.into_inner()).send((shaders, input));
    }
}

#[derive(Default)]
struct Census {
    pipelines: u64,
    made: u64,
    stage_ok: u64,
    stage_failed: u64,
    pipeline_failed: u64,
    with_inputs: u64,
    reasons: HashMap<String, u64>,
}

impl Census {
    fn one(&mut self, shaders: &[std::sync::Arc<crate::pipeline::Shader>], input: &super::metal_render_map::PipelineStateInput) {
        let Some(m) = backend() else { return };
        let _pool = victory_metal::Pool::new();
        self.pipelines += 1;
        let needs_stage_in = !input.elements.is_empty();
        if needs_stage_in {
            self.with_inputs += 1;
        }
        let mut vertex = None;
        let mut fragment = None;
        let mut failed = false;
        for s in shaders {
            let stage = match s.stage {
                "vertex" => victory_metal::msc::Stage::Vertex,
                "pixel" => victory_metal::msc::Stage::Fragment,
                other => {
                    self.stage_failed += 1;
                    failed = true;
                    self.reason(format!("{other} stage outside vertex/pixel census"));
                    continue;
                }
            };
            let Some(t) = s.translated() else {
                self.stage_failed += 1;
                failed = true;
                self.reason(format!("{} stage: no host translation", s.stage));
                continue;
            };
            match convert_cached(&t.host, stage).and_then(|c| m.device.library(&c.metallib).map(|l| (l, c.entry)).map_err(|e| e.to_string())) {
                Ok(lib) => {
                    self.stage_ok += 1;
                    if stage == victory_metal::msc::Stage::Vertex {
                        vertex = Some(lib);
                    } else {
                        fragment = Some(lib);
                    }
                }
                Err(e) => {
                    self.stage_failed += 1;
                    failed = true;
                    self.reason(format!("{} stage: {e}", s.stage));
                }
            }
        }
        if !needs_stage_in && !failed && let Some((vlib, vname)) = &vertex {
            use super::metal_render_map as map;
            let mapped = map::map_pipeline_state(input);
            let result = map::check_integer_blending(&mapped.color).and_then(|()| {
                let desc = render_desc(&mapped);
                m.device
                    .render_pipeline_desc((vlib, vname), fragment.as_ref().map(|(l, n)| (l, n.as_str())), &desc)
                    .map_err(|e| e.to_string())
            });
            match result {
                Ok(_) => self.made += 1,
                Err(e) => {
                    self.pipeline_failed += 1;
                    self.reason(format!("pipeline: {e}"));
                }
            }
        }
        if self.pipelines.is_power_of_two() || self.pipelines % 256 == 0 {
            let pipeline_attempts = self.made + self.pipeline_failed;
            tracing::info!(
                pipelines = self.pipelines,
                made = self.made,
                stages_converted = self.stage_ok,
                stages_failed = self.stage_failed,
                pipelines_refused = self.pipeline_failed,
                with_vertex_inputs = self.with_inputs,
                pipeline_attempts,
                pipelines_not_attempted = self.pipelines - pipeline_attempts,
                scope = "legacy-accepted vertex/pixel; no stage-in or mesh PSOs",
                "Metal census of graphics pipelines"
            );
        }
    }

    fn reason(&mut self, text: String) {
        let key: String = text.chars().take(160).collect();
        let n = self.reasons.entry(key.clone()).or_insert(0);
        *n += 1;
        if *n == 1 && self.reasons.len() <= 48 {
            tracing::warn!(reason = %key, "Metal census: a graphics pipeline did not make it (first of its kind)");
        }
    }
}

/// One stage through Metal Shader Converter, from the disk cache when there.
pub(crate) fn convert_cached(host: &HostShader, stage: victory_metal::msc::Stage) -> Result<victory_metal::msc::Converted, String> {
    let params = root_params(host);
    let samplers = static_samplers(host);
    let path = msc_cache_path(&host.dxil, &params, &samplers).map(|p| p.with_extension(format!("{stage:?}.vmsc")));
    if let Some(path) = &path
        && let Ok(bytes) = crate::read_file(path)
        && let Some(c) = victory_metal::msc::Converted::from_bytes(&bytes)
    {
        return Ok(c);
    }
    let c = victory_metal::msc::convert(&host.dxil, stage, &params, &samplers).map_err(|e| e.to_string())?;
    if let Some(path) = &path {
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        if std::fs::write(&tmp, c.to_bytes()).is_ok() {
            let _ = std::fs::rename(&tmp, path);
        }
    }
    Ok(c)
}

/// F7's map as victory-metal's render pipeline description.
pub(crate) fn render_desc(map: &super::metal_render_map::RenderPipelineMap) -> victory_metal::render::RenderPipelineDesc {
    let mut d = victory_metal::render::RenderPipelineDesc::default();
    for (out, c) in d.colors.iter_mut().zip(&map.color) {
        let Some(c) = c else { continue };
        out.pixel_format = c.pixel_format.map_or(0, |f| f as u64 as u32);
        // D3D12 ignores blending on integer targets; Metal aborts the
        // process on it (a descriptor validation assertion).
        out.blending = c.blending_enabled && !integer_format(out.pixel_format);
        out.src_rgb = c.source_rgb_factor.map_or(1, |f| f as u64 as u8);
        out.dst_rgb = c.destination_rgb_factor.map_or(0, |f| f as u64 as u8);
        out.op_rgb = c.rgb_operation.map_or(0, |f| f as u64 as u8);
        out.src_alpha = c.source_alpha_factor.map_or(1, |f| f as u64 as u8);
        out.dst_alpha = c.destination_alpha_factor.map_or(0, |f| f as u64 as u8);
        out.op_alpha = c.alpha_operation.map_or(0, |f| f as u64 as u8);
        out.write_mask = c.write_mask.0 as u8;
    }
    d.depth_format = map.depth_format.map_or(0, |f| f as u64 as u32);
    d.stencil_format = map.stencil_format.map_or(0, |f| f as u64 as u32);
    d.sample_count = map.raster_sample_count.clamp(1, 8) as u8;
    d.alpha_to_coverage = map.alpha_to_coverage;
    d.topology_class = map.input_primitive_topology.map_or(0, |t| t as u64 as u32);
    d
}

/// Integer (uint/sint) Metal pixel formats: never blendable.
pub(crate) fn integer_format(f: u32) -> bool {
    matches!(f, 13 | 14 | 23 | 24 | 33 | 34 | 53 | 54 | 63 | 64 | 73 | 74 | 91 | 103 | 104 | 113 | 114 | 123 | 124)
}

/// Whether a pass encoded; a failure (the whole submission is then dropped)
/// logged the first times with what the pass was.
fn encoded(what: &str, r: victory_metal::Result<()>, detail: impl FnOnce() -> String) -> bool {
    match r {
        Ok(()) => true,
        Err(e) => {
            static SEEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
                tracing::warn!(error = %e, detail = %detail(), "Metal: a {what} did not encode; the submission is dropped");
            }
            false
        }
    }
}

#[cfg(test)]
#[path = "metal_depth_replay_tests.rs"]
mod depth_replay_tests;
