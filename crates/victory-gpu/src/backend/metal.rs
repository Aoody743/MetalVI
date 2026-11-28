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

/// The runtime's answer to where a title range's physical pages are
/// mapped for good ([`victory_metal::chunks::Alias`]); set once at start.
static PHYSICAL_ALIAS: OnceLock<fn(u64, u64) -> Option<u64>> = OnceLock::new();

/// Registers [`PHYSICAL_ALIAS`] (xmem, before the first draw). It may take
/// the runtime's page-table lock: it must not call back into the GPU side.
pub fn set_physical_alias(f: fn(u64, u64) -> Option<u64>) {
    let _ = PHYSICAL_ALIAS.set(f);
}

fn physical_alias(va: u64, len: u64) -> Option<u64> {
    PHYSICAL_ALIAS.get().and_then(|f| f(va, len))
}

/// Held from a command buffer's commit to its completion, and by every
/// residency-set wiring and commit: wiring never runs while the GPU executes
/// work of this backend. A residency-set commit (wiring a 16 MiB span of a
/// persistent alias view, no remap near) stuck in the kernel
/// (`IOGPUResourceGroupUpdateResources`) and the game could not exit,
/// stage-r-budget8k2-20261128-115816; the four such runs all had long
/// multi-operation command buffers in flight. Lock order: the mapping gate,
/// then this; never the reverse.
pub(super) static IN_FLIGHT: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Held while title memory is wrapped and wired ([`victory_metal::chunks::Gate`]):
/// the readers' side of the runtime's mapping gate, which xmem holds as a
/// writer while it changes mappings. Each wiring is logged first (rate
/// limited), so a stall in the kernel names its range.
fn wiring_gate(start: u64, len: u64, aliased: bool) -> Box<dyn std::any::Any> {
    static LINES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = LINES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if n <= 2000 || n.is_power_of_two() {
        tracing::info!(n, at = format_args!("{start:#x}"), mb = len >> 20, kb = len >> 10, aliased, "Metal: wiring title memory");
    }
    struct Wiring(#[allow(dead_code)] crate::watch::Reading, #[allow(dead_code)] std::sync::MutexGuard<'static, ()>);
    impl std::ops::Drop for Wiring {
        fn drop(&mut self) {
            work::after_wait();
        }
    }
    let reading = crate::watch::title_memory_reading_range(start as usize, len as usize);
    work::here("waiting for the GPU to go idle (wiring)");
    let flight = IN_FLIGHT.lock().unwrap_or_else(|e| e.into_inner());
    work::here("wiring title memory");
    Box::new(Wiring(reading, flight))
}

/// Persistent views of physical-page sections the runtime made, to be wired
/// whole at the next safe point. On by default
/// (`VICTORY_METAL_PREWIRE_SECTIONS=0` turns it off): every stuck exit came
/// from wiring during play, and this leaves one wiring per section instead
/// of one per span. Up to `VICTORY_METAL_PREWIRE_CAP_MB` (default 32 GiB, the user's choice;
/// the kernel's wire limit here is 52 GiB, 17.7 GiB were wired at the last
/// freeze); sections past it get spans as before.
static PREWIRE: std::sync::Mutex<Vec<(u64, u64)>> = std::sync::Mutex::new(Vec::new());

fn prewire_sections() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_METAL_PREWIRE_SECTIONS").as_deref() != Ok("0"))
}

fn prewire_cap() -> u64 {
    static CAP: OnceLock<u64> = OnceLock::new();
    *CAP.get_or_init(|| std::env::var("VICTORY_METAL_PREWIRE_CAP_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(32768) << 20)
}

/// The runtime made a physical-page section with a never-remapped view at
/// `[at, at+len)` (xmem). Logged (count and total: how much wiring every
/// section whole would pin); queued for [`MetalBackend::prewire_pending`]
/// when `VICTORY_METAL_PREWIRE_SECTIONS=1`. Runs in the runtime's
/// allocation path: it only records.
pub fn title_memory_persistent(at: usize, len: usize) {
    static COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    let total = BYTES.fetch_add(len as u64, std::sync::atomic::Ordering::Relaxed) + len as u64;
    if n <= 64 || n.is_power_of_two() {
        tracing::info!(n, at = format_args!("{at:#x}"), mb = len >> 20, total_mb = total >> 20, "Metal: a persistent section view");
    }
    if prewire_sections() {
        PREWIRE.lock().unwrap_or_else(|e| e.into_inner()).push((at as u64, len as u64));
    }
}

/// A section's persistent view at `[at, at+len)` is about to be unmapped
/// (xmem frees the section). A prewired buffer over it would then show
/// unmapped memory: warned, counted (not handled yet). Only try-locks
/// `chunks` (the runtime's free path may hold its own locks).
pub fn title_memory_persistent_gone(at: usize, len: usize) {
    PREWIRE.lock().unwrap_or_else(|e| e.into_inner()).retain(|&(a, _)| a != at as u64);
    static GONE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let resident = BACKEND.get().and_then(|b| b.as_ref()).and_then(|m| m.chunks.try_lock().ok())
        .map(|c| c.is_span(at as u64, len as u64));
    if resident != Some(false) {
        let n = GONE.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        tracing::warn!(n, at = format_args!("{at:#x}"), mb = len >> 20, resident = ?resident,
            "Metal: a section freed while its prewired buffer may still be resident");
    }
}

impl MetalBackend {
    /// Wires the queued persistent views whole ([`title_memory_persistent`]),
    /// before a batch encodes: inside the mapping gate and [`IN_FLIGHT`]
    /// (the chunks gate), never while this backend's GPU work runs.
    fn prewire_pending(&self) {
        let pending = std::mem::take(&mut *PREWIRE.lock().unwrap_or_else(|e| e.into_inner()));
        if pending.is_empty() {
            return;
        }
        let mut chunks = self.chunks();
        for (at, len) in pending {
            if chunks.stats.prewired_bytes.saturating_add(len) > prewire_cap() {
                tracing::info!(at = format_args!("{at:#x}"), mb = len >> 20, prewired_mb = chunks.stats.prewired_bytes >> 20,
                    "Metal: a persistent section view left to spans (prewire cap)");
                continue;
            }
            let ok = chunks.prewire(at, len);
            tracing::info!(at = format_args!("{at:#x}"), mb = len >> 20, ok, prewired_mb = chunks.stats.prewired_bytes >> 20, "Metal: a persistent section view wired whole");
        }
    }
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
    super::metal_texture::note_remap_partial(at, len);
    // Counted only, and never waited for: the runtime calls this holding
    // its page table, which span lookups take under `chunks`.
    if let Some(Some(m)) = BACKEND.get()
        && let Ok(mut chunks) = m.chunks.try_lock()
    {
        chunks.note_remap(at, len);
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
    pub(crate) cost_ns: std::sync::atomic::AtomicU64,
    /// The title pipeline's name, as logged.
    pub(crate) name: String,
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
    pub(crate) loop_counts: Option<[(u64, Buffer, u64); 2]>,
}

pub(crate) fn want_loop_counts(name: &str) -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    name.contains("a1f6cdf8874960a9")
        && *ON.get_or_init(|| std::env::var_os("VICTORY_METAL_TRACE_LOOP_COUNTS").is_some_and(|s| s == "1"))
}

impl MetalBackend {
    pub(crate) fn a1f6_loop_counts(&self, handle: u64, base: u32) -> Option<[(u64, Buffer, u64); 2]> {
        use super::metal_views::{placed_address, srv_window};
        use crate::descriptor::{View, view};
        let (address, window) = match view(handle)? {
            View::Placed(false, resource, desc) if resource[0] == 1 =>
                (placed_address(&desc), srv_window(&desc[..10])?),
            View::Shader(resource, Some(desc)) =>
                (crate::resource::buffer_bytes(resource as crate::object::This)?.0 as u64, srv_window(&desc)?),
            _ => return None,
        };
        let mut chunks = self.chunks();
        let [first, second] = work::a1f6_count_offsets(base, window.bytes)?;
        let mut at = |offset| {
            let va = address.checked_add(window.offset)?.checked_add(offset)?;
            let (buffer, offset) = chunks.window_buffer(va, 4)?;
            Some((va, buffer, offset))
        };
        Some([at(first)?, at(second)?])
    }
}

/// One indirect compute dispatch as recorded into a title list.
pub(crate) struct MetalDispatchIndirect {
    pub(crate) pso: Arc<MetalCompute>,
    pub(crate) heap: Option<Buffer>,
    pub(crate) sampler_heap: Option<Buffer>,
    pub(crate) top: Vec<u8>,
    pub(crate) tables: Vec<(u32, DescriptorEntry)>,
    pub(crate) records: Buffer,
    pub(crate) records_offset: u64,
    pub(crate) stride: u32,
    pub(crate) max: u32,
    pub(crate) count: Option<(Buffer, u64)>,
    /// Title addresses of the records and the count (diagnostics).
    pub(crate) records_va: u64,
    pub(crate) count_va: Option<u64>,
    /// Where in `top` the program reads its first group's number: the
    /// dispatch is cut in segments ([`victory_metal::indirect::DispatchSegments`]).
    pub(crate) group_base_at: Option<usize>,
}

/// What a title list records for Metal, in its order.
pub(crate) enum MetalOp {
    Dispatch(MetalCmd),
    DispatchIndirect(MetalDispatchIndirect),
    Draw(super::metal_draw::MetalDraw),
    Clear(super::metal_draw::MetalClear),
    Copy(super::metal_draw::MetalTexCopy),
    FloatRowReadback { src: victory_metal::texture::Texture, dst: Buffer, offset: u64, width: u32 },
    BufferCopy { dst: usize, src: usize, bytes: usize },
    Resolve(super::metal_draw::MetalResolve),
}

/// Serialize every backend command buffer through completion and the fuse.
/// This cannot interrupt a command buffer which has already been submitted.
pub(crate) fn submit_checked(
    cb: &victory_metal::CommandBuffer,
    ops: &[Arc<MetalOp>],
    has_work: bool,
    label: &'static str,
) -> victory_metal::Result<()> {
    completion::submit(cb, ops, has_work, label)
}

fn complete_checked(
    cb: &victory_metal::CommandBuffer,
    ops: &[Arc<MetalOp>],
    has_work: bool,
    label: &'static str,
) -> victory_metal::Result<()> {
    let encoded = cb.encoded_work();
    let command = cb.handle();
    work::report_large(label, encoded);
    let estimated_ns = work::reservation(ops);
    work::here("waiting for residency changes");
    let flight = IN_FLIGHT.lock().unwrap_or_else(|e| e.into_inner());
    work::here("commit");
    if !fuse::commit_if_open(cb) {
        return Err(victory_metal::Error("Metal safety stop already tripped".into()));
    }
    let watch = inflight::watch(ops);
    work::here("waiting for a command buffer");
    let result = cb.wait();
    drop(flight);
    work::after_wait();
    #[cfg(test)]
    let result = result.and_then(|()| if completion::INJECT_TIMEOUT.swap(false, std::sync::atomic::Ordering::AcqRel) {
        Err(victory_metal::Error("injected system timeout after bounded GPU completion".into()))
    } else { Ok(()) });
    drop(watch);
    if let Err(e) = result {
        fuse::stop();
        diag::report(label, command as usize, || inflight::describe(ops));
        tracing::error!(%label, command, error = %e, ops = %inflight::describe(ops), ?encoded, "Metal safety stop: submission failed");
        return Err(e);
    }
    let gpu_ns = cb.gpu_time_ns();
    tracing::debug!(%label, command, ops = ops.len(), estimated_ns, gpu_ns, ?encoded, "Metal: command buffer completed");
    let limit_ns = fuse::limit_ns();
    if (has_work || !ops.is_empty() || encoded != Default::default()) && fuse::unsafe_timing(gpu_ns, limit_ns) {
        fuse::stop();
        diag::report(label, command as usize, || inflight::describe(ops));
        tracing::error!(%label, command, gpu_ns, limit_ns, ops = %inflight::describe(ops), ?encoded, "Metal safety stop: submission failed");
        return Err(victory_metal::Error("Metal safety stop: unsafe GPU duration".into()));
    }
    diag::report(label, command as usize, || inflight::describe(ops));
    #[cfg(test)]
    diag::LAST_GPU_NS.store(gpu_ns, std::sync::atomic::Ordering::Relaxed);
    work::observe(ops, gpu_ns);
    work::gpu_busy(gpu_ns, ops);
    Ok(())
}

pub(crate) fn stop_submissions(reason: &'static str) {
    fuse::stop();
    tracing::error!(%reason, "Metal safety stop: submission failed");
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
    /// Resident tables of `DescriptorEntry`s for MSC's static samplers, keyed
    /// by serialized entries so identical static sampler sets share one buffer.
    static_sampler_tables: Mutex<HashMap<Vec<u8>, Buffer>>,
    /// The GPU detile kernels, compiled on first use (`None`: they failed).
    detiler: OnceLock<Option<super::metal_detile::Detiler>>,
    /// Graphics pipelines by (title pipeline object, rectangle emulation) (`None`: refused).
    pub(crate) graphics: Mutex<HashMap<(usize, bool), Option<Arc<super::metal_draw::MetalGraphics>>>>,
    /// Depth-stencil states by description.
    pub(crate) depth_stencils: Mutex<HashMap<victory_metal::render::DepthStencilDesc, victory_metal::render::DepthStencilState>>,
    /// Render target views in another pixel format, by (resource, format).
    pub(crate) target_views: Mutex<HashMap<(usize, u32), victory_metal::texture::Texture>>,
    /// Per-threadgroup scratch buffer for spilled wave64-on-wave32 helpers (`u249, space1`).
    wave_scratch: OnceLock<Option<Buffer>>,
    /// Shader diagnostic counters (`u248, space1`, [`diag`]).
    pub(crate) diag: OnceLock<Option<Buffer>>,
    /// Whether allocations have been added to `chunks.residency()` since the last `commit()`.
    residency_dirty: std::sync::atomic::AtomicBool,
    /// Batched CPU-detiled texture uploads waiting for a single command buffer.
    pub(crate) pending_uploads: Mutex<super::metal_texture::PendingUploads>,
    /// Fast-path count of pending batched uploads.
    pub(crate) uploads_pending: std::sync::atomic::AtomicUsize,
}

/// A title view heap as Metal Shader Converter descriptor entries, in
/// memory of our own the GPU reads in place. Slots are refreshed when
/// written via [`note_slot_dirty`] / [`MetalBackend::sync_heap`] and on bind
/// via [`MetalBackend::refresh_slot`].
pub(crate) struct MetalHeap {
    pub(crate) buffer: Buffer,
    entries: *mut DescriptorEntry,
    count: u32,
    /// Entries the buffer holds: a view heap's `count` and then the run of
    /// title-memory views (`memory`, `COPIES + 1` slots).
    capacity: u32,
    strides: Option<(Buffer, *mut u32)>,
}

// The entries and stride table are only written under the heaps lock.
unsafe impl Send for MetalHeap {}

/// Title descriptor slots written since the last [`MetalBackend::sync_heap`]:
/// `heap -> (index -> handle)`. Always a leaf lock.
static DIRTY_SLOTS: Mutex<Option<HashMap<usize, HashMap<u32, u64>>>> = Mutex::new(None);

pub(crate) fn note_slot_dirty(heap: usize, index: u32, handle: u64) {
    if mode() == Mode::Off || handle == 0 {
        return;
    }
    DIRTY_SLOTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashMap::new)
        .entry(heap)
        .or_default()
        .insert(index, handle);
}

pub(crate) fn forget_heap(heap: usize) {
    if let Some(map) = DIRTY_SLOTS.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        map.remove(&heap);
    }
    if let Some(Some(b)) = BACKEND.get() {
        b.heaps.lock().unwrap_or_else(|e| e.into_inner()).remove(&heap);
    }
}

pub(crate) fn forget_resource(resource: usize) {
    super::metal_texture::forget_resource_textures(resource);
    let Some(Some(b)) = BACKEND.get() else { return };
    b.textures.lock().unwrap_or_else(|e| e.into_inner()).remove(&resource);
    b.views.lock().unwrap_or_else(|e| e.into_inner()).retain(|k, _| k.0 != resource);
    b.target_views.lock().unwrap_or_else(|e| e.into_inner()).retain(|k, _| k.0 != resource);
}

static BACKEND: OnceLock<Option<MetalBackend>> = OnceLock::new();

#[path = "metal_memory.rs"]
mod memory;

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
        let mut chunks = Chunks::sparse(&device)?;
        // `VICTORY_METAL_SPAN_CAP_MB` (default 6144): what span buffers over
        // title memory may hold before new ones are refused (their draws
        // drop). Unbounded, they wired tens of GB for the GPU and stalled
        // the window server in the kernel (2026-11-27, whole-machine hang).
        chunks.span_cap = std::env::var("VICTORY_METAL_SPAN_CAP_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(6144)
            << 20;
        chunks.alias = Some(physical_alias);
        chunks.gate = Some(wiring_gate);
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
            static_sampler_tables: Mutex::new(HashMap::new()),
            detiler: OnceLock::new(),
            graphics: Mutex::new(HashMap::new()),
            depth_stencils: Mutex::new(HashMap::new()),
            target_views: Mutex::new(HashMap::new()),
            wave_scratch: OnceLock::new(),
            diag: OnceLock::new(),
            residency_dirty: std::sync::atomic::AtomicBool::new(false),
            pending_uploads: Mutex::new(Default::default()),
            uploads_pending: std::sync::atomic::AtomicUsize::new(0),
            ring: Mutex::new(None),
            textures: Mutex::new(HashMap::new()),
            views: Mutex::new(HashMap::new()),
        })
    }

    pub(crate) fn add_resident(&self, handles: &[victory_metal::sys::obj_handle_t]) {
        if handles.is_empty() {
            return;
        }
        let chunks = self.chunks();
        chunks.residency().add(handles);
        self.residency_dirty.store(true, std::sync::atomic::Ordering::Release);
    }

    pub(crate) fn commit_residency(&self) {
        if self.residency_dirty.swap(false, std::sync::atomic::Ordering::AcqRel) {
            let chunks = self.chunks();
            // Wires what was added: never while xmem changes mappings.
            let _gate = crate::watch::title_memory_reading();
            let _flight = IN_FLIGHT.lock().unwrap_or_else(|e| e.into_inner());
            chunks.residency().commit();
        }
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
        static SUBMISSION: std::sync::Mutex<()> = std::sync::Mutex::new(());
        // No phase while waiting for the lock: the holder's own phases show
        // whether it moves (a waiter behind a long batch is not a hang).
        let _submission = SUBMISSION.lock().unwrap_or_else(|e| e.into_inner());
        let _lane = work::lane(work::LANE_RUN);
        work::here("run_batch");
        let ok = self.run_serial(ops);
        if !ok {
            fuse::stop();
            static REPORTED: OnceLock<()> = OnceLock::new();
            REPORTED.get_or_init(|| tracing::error!("Metal safety stop: submission failed"));
        }
        ok
    }

    pub(super) fn stopped_endpoint(&self, path: &'static str) -> bool {
        if !self.submissions_stopped() { return false; }
        let total = STOPPED_ENDPOINTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        tracing::warn!(path, total, "Metal: endpoint refused after fuse");
        true
    }

    pub(crate) fn submissions_stopped(&self) -> bool {
        fuse::tripped()
    }

    fn run_serial(&self, ops: &[Arc<MetalOp>]) -> bool {
        if super::metal_draw::trace_scene() {
            static ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let frame = crate::frame::current();
            let mut start = 0;
            for (i, op) in ops.iter().enumerate() {
                let MetalOp::Draw(d) = &**op else { continue };
                if !super::metal_draw::trace_pipeline(&d.pso.name) { continue; }
                let Some(t) = d.targets.colors.iter().flatten().find(|t| (t.width == 3840 || !d.trace_inputs.is_empty()) && t.level == 0 && t.slice == 0) else { continue };
                if !self.run_batch(&ops[start..i]) { return false; }
                let id = ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let name = format!("writer-{frame}-{id:04}-{:x}", t.resource);
                tracing::info!(%name, pipeline=%d.pso.name, "scene writer");
                self.capture(t.resource as crate::object::This, &format!("{name}-in"));
                for (k, &input) in d.trace_inputs.iter().enumerate() {
                    self.capture(input as crate::object::This, &format!("{name}-srv{k}-{input:x}"));
                }
                if !self.run_batch(&ops[i..=i]) { return false; }
                self.capture(t.resource as crate::object::This, &format!("{name}-out"));
                start = i + 1;
            }
            return self.run_batch(&ops[start..]);
        }
        self.run_batch(ops)
    }

    fn run_batch(&self, ops: &[Arc<MetalOp>]) -> bool {
        if fuse::tripped() {
            return false;
        }
        {
            static LOGGED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let st = self.chunks().stats;
            let last = LOGGED.load(std::sync::atomic::Ordering::Relaxed);
            if st.spans >= (last * 2).max(64) || st.spans >= last + 2048 || (st.capped > 0 && last != u64::MAX && st.spans == last) {
                LOGGED.store(if st.capped > 0 && st.spans == last { u64::MAX } else { st.spans }, std::sync::atomic::Ordering::Relaxed);
                let stale_uses = victory_metal::chunks::STALE_USES.load(std::sync::atomic::Ordering::Relaxed);
                tracing::info!(spans = st.spans, direct_spans = st.direct_spans, span_mb = st.span_bytes >> 20, capped = st.capped, refused = st.refused, remaps_under_spans = st.remaps_under_spans, stale_spans = st.stale_spans, stale_uses, "Metal: title memory span buffers");
            }
        }
        use super::metal_draw::{DrawKind, STAGE_VERTEX, draw_params, stage_buffer};
        use victory_metal::render::{PassColor, PassDepth, Render, RenderPass};
        use victory_metal::{Compute, msc, sys};
        fn describe_fault_ops(ops: &[Arc<MetalOp>]) -> Vec<String> {
            let peek_u32s = |va: u64, n: usize| -> Option<Vec<u32>> {
                if va == 0 || va % 4 != 0 || n == 0 {
                    return None;
                }
                let bytes = n * 4;
                let _reading = crate::watch::title_memory_reading_range(va as usize, bytes);
                if victory_metal::chunks::committed_extent(va, bytes as u64) < bytes as u64 {
                    return None;
                }
                let mut out = Vec::with_capacity(n);
                for k in 0..n {
                    // SAFETY: committed, guarded against remapping, and 4-byte aligned.
                    out.push(unsafe { (va as *const u32).add(k).read_volatile() });
                }
                Some(out)
            };
            let indirect_ops = ops
                .iter()
                .filter(|o| matches!(&***o, MetalOp::Draw(d) if matches!(d.kind, DrawKind::Indirect { .. })));
            let other_ops = ops
                .iter()
                .filter(|o| !matches!(&***o, MetalOp::Draw(d) if matches!(d.kind, DrawKind::Indirect { .. })));
            indirect_ops
                .chain(other_ops)
                .take(12)
                .map(|o| match &**o {
                    MetalOp::Dispatch(_) | MetalOp::DispatchIndirect(_) => "dispatch".to_string(),
                    MetalOp::BufferCopy { .. } => unreachable!("split before encoding"),
                    MetalOp::Draw(d) => match &d.kind {
                        DrawKind::Indirect { stride, max, index, records_va, count_va, .. } => format!(
                            "draw {} indirect(max={}, stride={}, indexed={}, rect={:?}, rec0={:?}, count={:?})",
                            d.pso.name,
                            max,
                            stride,
                            index.is_some(),
                            d.pso.geometry,
                            peek_u32s(*records_va, 4),
                            count_va.and_then(|c| peek_u32s(c, 1)),
                        ),
                        _ => format!("draw {}", d.pso.name),
                    },
                    MetalOp::Copy(c) => format!("copy {} → {}", c.src.pixel.0, c.dst.pixel.0),
                    MetalOp::FloatRowReadback { width, .. } => format!("readback {width} float pixels"),
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
                .collect()
        }
        if ops.is_empty() {
            return true;
        }
        // Title buffers are shared memory. Complete preceding GPU work before
        // the CPU copy, and publish its write before encoding later consumers.
        if ops.iter().any(|op| matches!(&**op, MetalOp::BufferCopy { .. })) {
            let mut start = 0;
            for (i, op) in ops.iter().enumerate() {
                if let MetalOp::BufferCopy { dst, src, bytes } = **op {
                    if !self.run_batch(&ops[start..i]) { return false; }
                    crate::host::copy_mapped(dst as *mut u8, src as *const u8, bytes, true);
                    crate::watch::wrote(dst, bytes);
                    start = i + 1;
                }
            }
            return self.run_batch(&ops[start..]);
        }
        static BATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let batch = BATCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        work::trace_ops(batch, ops);
        let started = std::time::Instant::now();
        self.prewire_pending();
        super::metal_texture::drain_remap_partial();
        if !self.flush_uploads() {
            return false;
        }
        let uploaded = started.elapsed();
        self.commit_residency();
        let prologue = started.elapsed();
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
                MetalOp::DispatchIndirect(c) => tables += c.tables.len(),
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
                MetalOp::BufferCopy { .. } | MetalOp::Clear(_) | MetalOp::Copy(_) | MetalOp::FloatRowReadback { .. } | MetalOp::Resolve(_) => {}
            }
        }
        let need = tables + blob.div_ceil(entry) + 256 / entry + 1;
        if ring.as_ref().is_none_or(|r| r.capacity < need) {
            *ring = self.new_ring(need.max(1 << 16));
            self.commit_residency();
        }
        let Some(r) = ring.as_ref() else {
            return work::reject("descriptor ring unavailable");
        };
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
                MetalOp::DispatchIndirect(c) => {
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
                        DrawKind::Indirect { .. } if d.pso.geometry.is_some() => {
                            let vt = if let Some(fetch) = &d.vertex {
                                put(&fetch.table)
                            } else {
                                put(&[0u8; msc::VERTEX_TABLE_SLOTS * 16])
                            };
                            let zero = put(&[0u8; 24]);
                            (zero, zero, vt, zero)
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
                MetalOp::BufferCopy { .. } | MetalOp::Clear(_) | MetalOp::Copy(_) | MetalOp::FloatRowReadback { .. } | MetalOp::Resolve(_) => {
                    dispatch_tops.push(Vec::new());
                    draw_offsets.push((Vec::new(), 0, 0, 0, 0));
                }
            }
        }
        let Ok(mut cb) = self.queue.command_buffer() else { return false };
        // Compute goes out a few dispatches per command buffer. Wait before
        // submitting the next buffer so a failure or fuse trip cannot leave
        // more of this batch queued. One command buffer of seconds of compute kept the window
        // server from the GPU until macOS's watchdog panicked the machine
        // (2026-11-27). `VICTORY_METAL_PASS_DISPATCHES` defaults to 4;
        // 1 isolates every operation, 0 leaves only the estimate-based cuts.
        static PASS_DISPATCHES: OnceLock<u32> = OnceLock::new();
        let pass_cap = *PASS_DISPATCHES.get_or_init(|| {
            std::env::var("VICTORY_METAL_PASS_DISPATCHES").ok().and_then(|v| v.parse().ok()).unwrap_or(4)
        });
        static SPLIT_EVENT: OnceLock<Option<victory_metal::events::SharedEvent>> = OnceLock::new();
        static SPLIT_VALUE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let split_event = SPLIT_EVENT.get_or_init(|| {
            #[cfg(test)]
            if work::FAULT.get() == Some("event") { return None; }
            victory_metal::events::SharedEvent::new(&self.device).ok()
        }).as_ref();
        if pass_cap > 0 && split_event.is_none() {
            return work::reject("split event unavailable");
        }
        let bounded = pass_cap > 0 && split_event.is_some();
        let pass_dispatches = if bounded { pass_cap } else { u32::MAX };
        let mut cb_start = 0usize;
        // Dispatches (direct and indirect) in the current command buffer: a
        // bounded buffer is cut once it holds `pass_dispatches`. Cutting after
        // every compute pass instead put each indirect dispatch (a pass of its
        // own) in a command buffer of its own, waited for: ~30,000 of them in
        // the loading stretch of stage-r-stall1-20261128-084750.
        let mut cb_dispatches = 0u32;
        // `VICTORY_METAL_SPLIT=1`: every pass in a command buffer of its own,
        // waited for, so a GPU fault names the pass that made it.
        // `VICTORY_METAL_SPLIT=rects`: split only batches that contain an
        // indirect rectangle draw (`geometry.is_some()`).
        static SPLIT: OnceLock<u8> = OnceLock::new();
        static CR: OnceLock<victory_metal::Result<victory_metal::clear_resolve::ClearResolve>> = OnceLock::new();
        static TRANSLATOR: OnceLock<Option<victory_metal::indirect::IndirectTranslator>> = OnceLock::new();
        let split_mode = *SPLIT.get_or_init(|| match std::env::var("VICTORY_METAL_SPLIT").as_deref() {
            Ok("1") => 1,
            Ok("rects") => 2,
            _ => 0,
        });
        let split = match split_mode {
            1 => true,
            2 => ops.iter().any(|o| matches!(&**o, MetalOp::Draw(d) if matches!(d.kind, DrawKind::Indirect { .. }) && d.pso.geometry.is_some())),
            _ => false,
        };
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
        // Twinned depth-stencil textures (`StencilTwin`): given their
        // twins before a pass or copy reads them, giving them what a pass
        // or copy wrote after.
        let load_twin = |sync: &mut super::metal_depth::DepthSync, cb: &victory_metal::CommandBuffer, t: Option<&std::sync::Arc<super::metal_depth::StencilTwin>>, level: u16, slice: u16| {
            t.is_none_or(|t| encoded("stencil twin load", sync.load_twin(&self.device, cb, t, level, slice), || format!("level {level} slice {slice}")))
        };
        let store_twin = |sync: &mut super::metal_depth::DepthSync, cb: &victory_metal::CommandBuffer, t: Option<&std::sync::Arc<super::metal_depth::StencilTwin>>, level: u16, slice: u16| {
            t.is_none_or(|t| encoded("stencil twin store", sync.store_twin(&self.device, cb, t, level, slice), || format!("level {level} slice {slice}")))
        };
        // Every subresource a copy touches.
        let copied = |t: &super::metal_texture::MetalTexture, whole: bool, level: u32, slice: u32| -> Vec<(u16, u16)> {
            if whole {
                (0..t.mips).flat_map(|l| (0..t.slices.max(1)).map(move |s| (l as u16, s as u16))).collect()
            } else {
                vec![(level as u16, slice as u16)]
            }
        };
        let mut i = 0;
        let mut planned_end = 0;
        let mut clamp_counter = None;
        let mut indirect_traces = Vec::new();
        let mut ok = true;
        let mut copy_scratch = None;
        let mut readback_scratch: Option<Buffer> = None;
        while i < ops.len() && ok {
            if i == cb_start {
                let max_ops = if pass_cap == 1 { 1 } else { usize::MAX };
                planned_end = i + work::prefix(&ops[i..], max_ops);
                if planned_end < ops.len() && split_event.is_none() {
                    return work::reject("budget split needs an ordering event");
                }
            }
            let group_start = i;
            let mut indirect_writes = None;
            match &*ops[i] {
                MetalOp::BufferCopy { .. } => unreachable!("split before encoding"),
                MetalOp::Dispatch(producer) => {
                    work::here("encode dispatches");
                    if pass_cap == 1 && let Some(counts) = &producer.loop_counts {
                        let sources = counts.each_ref().map(|(_, buffer, offset)| (buffer, *offset));
                        match work::snapshot_words::<1>(&self.device, &self.queue, &sources) {
                            Ok(values) => tracing::debug!(batch, op_index = i, pipeline = %producer.pso.name,
                                first_va = counts[0].0, second_va = counts[1].0, first = values[0][0],
                                second = values[1][0], "Metal: dispatch loop count inputs"),
                            Err(e) => {
                                tracing::error!(%e, batch, op_index = i, "Metal: loop count snapshot failed");
                                return work::reject("loop count snapshot");
                            }
                        }
                    }
                    if pass_cap == 1 && work::trace_indirect_writer(&producer.pso.name) {
                        let consumers: Vec<_> = ops.iter().enumerate().skip(i + 1).take(32).filter_map(|(index, op)| {
                            match &**op {
                                MetalOp::DispatchIndirect(c) if c.max != 0 && work::trace_indirect(&c.pso.name) => Some((index, c)),
                                _ => None,
                            }
                        }).take(8).collect();
                        if !consumers.is_empty() {
                            let sources: Vec<_> = consumers.iter().map(|(_, c)| (&c.records, c.records_offset)).collect();
                            let before = match work::snapshot_words::<3>(&self.device, &self.queue, &sources) {
                                Ok(args) => args,
                                Err(e) => {
                                    tracing::error!(%e, batch, producer = i, "Metal: indirect writer snapshot before failed");
                                    return work::reject("indirect writer snapshot");
                                }
                            };
                            indirect_writes = Some((consumers, sources, before));
                        }
                    }
                    let mut steps = Vec::new();
                    let mut in_pass = 0u32;
                    while let Some(MetalOp::Dispatch(c)) = ops.get(i).map(|o| &**o) {
                        // A pass of at most `pass_dispatches` (see `bounded`).
                        let cap = if work::cut_every_pass() { in_pass } else { cb_dispatches.saturating_add(in_pass) };
                        if i == planned_end || cap >= pass_dispatches {
                            break;
                        }
                        in_pass += 1;
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
                    cb_dispatches = cb_dispatches.saturating_add(in_pass);
                    ok = encoded("compute pass", cb.compute(&steps), || String::new());
                    depth_sync.colour_may_change();
                }
                MetalOp::DispatchIndirect(c) => {
                    work::here("encode indirect dispatch");
                    cb_dispatches = cb_dispatches.saturating_add(1);
                    if c.max > 256 {
                        static LARGE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                        if LARGE.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                            tracing::info!(max = c.max, stride = c.stride, kind = "dispatch", "Metal: large ExecuteIndirect max_count");
                        }
                    }
                    if let Some(tr) = TRANSLATOR.get_or_init(|| victory_metal::indirect::IndirectTranslator::new(&self.device).ok()) {
                        use victory_metal::indirect::{DispatchSegments, IndirectKind, IndirectSource};
                        // A program that adds a group base: the record is cut
                        // in segments, each with its own base in its top bytes.
                        let segments = c.group_base_at.filter(|_| work::dispatch_segments_on()).map(|_| DispatchSegments::DEFAULT);
                        let seg_tops: Vec<Vec<u8>> = match (segments, c.group_base_at) {
                            (Some(d), Some(at)) if at + 4 <= dispatch_tops[i].len() => (0..d.segments)
                                .map(|k| {
                                    let mut top = dispatch_tops[i].clone();
                                    top[at..at + 4].copy_from_slice(&(k * d.groups).to_le_bytes());
                                    top
                                })
                                .collect(),
                            _ => Vec::new(),
                        };
                        {
                            // Once per pipeline: whether its dispatches can be cut.
                            static SEEN: std::sync::Mutex<Option<std::collections::HashSet<String>>> = std::sync::Mutex::new(None);
                            let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
                            if seen.get_or_insert_with(Default::default).insert(c.pso.name.clone()) {
                                tracing::info!(pipeline = %c.pso.name, segmented = !seg_tops.is_empty(), group_base = c.group_base_at.is_some(),
                                    "Metal: an indirect dispatch pipeline (segments need a group base)");
                            }
                        }
                        let source = IndirectSource { dispatch_segments: segments.filter(|_| !seg_tops.is_empty()),
                            kind: IndirectKind::Dispatch,
                            records: (&c.records, c.records_offset),
                            stride: c.stride,
                            max_count: c.max,
                            count: c.count.as_ref().map(|(b, o)| (b, *o)),
                        };
                        if clamp_counter.is_none() {
                            clamp_counter = victory_metal::indirect::ClampCounter::new(&self.device).ok();
                        }
                        let Some(counter) = clamp_counter.as_ref() else { return work::reject("indirect counter unavailable") };
                        match tr.encode_bounded_with_trace(&self.device, &cb, &source, counter, work::trace_indirect(&c.pso.name)) {
                            Ok(Some(mut batch)) => {
                                if let Some(trace) = batch.take_trace() {
                                    indirect_traces.push((i, c.pso.name.clone(), trace));
                                }
                                let mut steps = Vec::new();
                                for rec_idx in 0..batch.record_count() {
                                    let Some(rec) = batch.record(rec_idx) else { continue };
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
                                    let top = if seg_tops.is_empty() { &dispatch_tops[i] } else { &seg_tops[(rec_idx as usize) % seg_tops.len()] };
                                    if !top.is_empty() {
                                        steps.push(Compute::Bytes { bytes: top, index: msc::ARGUMENT_BUFFER_BIND_POINT });
                                    }
                                    steps.push(Compute::DispatchIndirect { arguments: rec.buffer, offset: rec.offset });
                                }
                                if !steps.is_empty() {
                                    ok = encoded("indirect compute pass", cb.compute(&steps), || String::new());
                                    depth_sync.colour_may_change();
                                }
                            }
                            Ok(None) => {}
                            Err(e) => {
                                tracing::error!(error = %e, "Metal: indirect dispatch translation failed");
                                return work::reject("indirect dispatch translation");
                            }
                        }
                    } else { return work::reject("indirect translator unavailable"); }
                    i += 1;
                }
                MetalOp::FloatRowReadback { src, dst, offset, width } => {
                    work::here("encode readback");
                    let bytes = width * 16;
                    let pitch = bytes.next_multiple_of(256);
                    if readback_scratch.as_ref().is_none_or(|b| b.len < u64::from(pitch)) {
                        readback_scratch = self.device.private_buffer(u64::from(pitch)).ok();
                    }
                    if let Some(staging) = &readback_scratch && let Ok(mut blit) = cb.blit_encoder() {
                        blit.copy_from_texture_to_buffer(src, 0, 0, [0; 3], [u64::from(*width), 1, 1], staging, 0, pitch, pitch);
                        blit.copy_buffer(staging, 0, dst, *offset, u64::from(bytes));
                        blit.finish();
                    } else {
                        ok = false;
                    }
                    i += 1;
                }
                MetalOp::Copy(c) => {
                    work::here("encode texture copy");
                    for (level, slice) in copied(&c.src, c.whole, c.src_level, c.src_slice) {
                        if !load_twin(&mut depth_sync, &cb, c.src.twin.as_ref(), level, slice) {
                            return false;
                        }
                    }
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
                        &self.device,
                        &mut copy_scratch,
                    ) {
                        static SEEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                        if SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                            tracing::info!(src = c.src.pixel.0, dst = c.dst.pixel.0, size = ?c.size, error = %e, "Metal: a texture copy left out");
                        }
                    }
                    depth_sync.colour_may_change();
                    for (level, slice) in copied(&c.dst, c.whole, c.dst_level, c.dst_slice) {
                        ok &= store_twin(&mut depth_sync, &cb, c.dst.twin.as_ref(), level, slice);
                    }
                    i += 1;
                }
                MetalOp::Resolve(r) => {
                    work::here("encode resolve");
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
                    work::here("encode rectangle clear");
                    // Rectangles: F49's quad-per-rectangle clear.
                    if let Ok(cr) = CR.get_or_init(|| victory_metal::clear_resolve::ClearResolve::new(&self.device)) {
                        if !load_depth(&mut depth_sync, &cb, &c.target)
                            || !load_twin(&mut depth_sync, &cb, c.target.twin.as_ref(), c.target.level, c.target.slice)
                        {
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
                                if ok && (c.depth.is_some() || c.stencil.is_some()) {
                                    ok = store_twin(&mut depth_sync, &cb, c.target.twin.as_ref(), c.target.level, c.target.slice);
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
                    work::here("encode clear");
                    let clear = sys::WMTLoadAction::WMTLoadActionClear;
                    let mut pass = RenderPass { width: c.target.width, height: c.target.height, samples: c.target.samples, ..Default::default() };
                    if c.depth.is_none() && !load_depth(&mut depth_sync, &cb, &c.target) {
                        return false;
                    }
                    if !load_twin(&mut depth_sync, &cb, c.target.twin.as_ref(), c.target.level, c.target.slice) {
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
                    if ok && (c.depth.is_some() || c.stencil.is_some()) {
                        ok = store_twin(&mut depth_sync, &cb, c.target.twin.as_ref(), c.target.level, c.target.slice);
                    }
                    i += 1;
                }
                MetalOp::Draw(first) => {
                    work::here("encode draws");
                    let t = first.targets.clone();
                    if let Some(d) = &t.depth
                        && (!load_depth(&mut depth_sync, &cb, d) || !load_twin(&mut depth_sync, &cb, d.twin.as_ref(), d.level, d.slice))
                    {
                        return false;
                    }
                    // Indirect records of this run of draws, translated on
                    // the GPU before its render pass opens (F42).
                    let mut batches: Vec<Option<victory_metal::indirect::IndirectBatch>> = Vec::new();
                    {
                        let mut j = i;
                        while let Some(MetalOp::Draw(d)) = ops.get(j).map(|o| &**o) {
                            if j == planned_end || !d.targets.same(&t) {
                                break;
                            }
                            let mut batch = None;
                            if let DrawKind::Indirect { records, records_offset, stride, max, count, index, .. } = &d.kind {
                                let Some(tr) = TRANSLATOR.get_or_init(|| victory_metal::indirect::IndirectTranslator::new(&self.device).ok())
                                    else { return work::reject("indirect translator unavailable") };
                                if *max > 256 {
                                    static LARGE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                                    if LARGE.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                                        tracing::info!(max = *max, stride = *stride, kind = if index.is_some() { "indexed" } else if d.pso.mesh { "mesh" } else { "draw" }, "Metal: large ExecuteIndirect max_count");
                                    }
                                }
                                use victory_metal::indirect::{IndirectKind, IndirectSource};
                                if index.is_some() && (d.pso.mesh || d.pso.geometry.is_some()) {
                                    return work::reject("indexed indirect mesh ABI unsupported");
                                }
                                let kind = match index {
                                    Some((_, _, ty, cap)) => IndirectKind::DrawIndexed { index_capacity: *cap, index_type: *ty },
                                    None if let Some(cfg) = d.pso.geometry => IndirectKind::Rectangles(cfg),
                                    None if d.pso.mesh => IndirectKind::TitleMesh1D,
                                    None => IndirectKind::Draw,
                                };
                                let source = IndirectSource { dispatch_segments: None,
                                    kind,
                                    records: (records, *records_offset),
                                    stride: *stride,
                                    max_count: *max,
                                    count: count.as_ref().map(|(b, o)| (b, *o)),
                                };
                                if clamp_counter.is_none() {
                                    clamp_counter = victory_metal::indirect::ClampCounter::new(&self.device).ok();
                                }
                                let Some(counter) = clamp_counter.as_ref() else { return work::reject("indirect counter unavailable") };
                                match tr.encode_bounded_with_trace(&self.device, &cb, &source, counter, work::trace_indirect(&d.pso.name)) {
                                    Ok(mut b) => {
                                        if let Some(trace) = b.as_mut().and_then(|batch| batch.take_trace()) {
                                            indirect_traces.push((j, d.pso.name.clone(), trace));
                                        }
                                        batch = b;
                                    }
                                    Err(e) => {
                                        tracing::error!(error = %e, "Metal: indirect draw translation failed");
                                        return work::reject("indirect draw translation");
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
                        if i == planned_end || !d.targets.same(&t) {
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
                                        if g.mesh && g.geometry.is_none() {
                                            steps.push(Render::DrawMeshIndirect { arguments: rec.buffer, offset: rec.offset,
                                                object_threads: [1, 1, 1], mesh_threads: g.mesh_threads });
                                            continue;
                                        }
                                        if let Some(cfg) = g.geometry
                                            && let Some(template) = cfg.draw(3, 1, 0, 0, None)
                                            && let Some(bindings) = rec.rectangle_bindings()
                                        {
                                            steps.extend(bindings);
                                            steps.push(Render::DrawMeshIndirect {
                                                arguments: rec.buffer, offset: rec.offset,
                                                object_threads: template.object_threads, mesh_threads: template.mesh_threads,
                                            });
                                            continue;
                                        }
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
                        // Stencil writes are not told apart: a pass with
                        // a stencil attachment may have written it.
                        if let Some(twin) = &a.twin {
                            if ok && (depth_written || t.stencil) {
                                ok = store_twin(&mut depth_sync, &cb, Some(twin), a.level, a.slice);
                            } else {
                                depth_sync.preserve_twin(twin, a.level, a.slice);
                            }
                        }
                    }
                }
            }
            let cut = if pass_cap == 1 {
                Some("single operation")
            } else if i == planned_end {
                Some("estimated time")
            } else if split {
                Some("diagnostic pass")
            } else if bounded && (cb_dispatches >= pass_dispatches
                || work::cut_every_pass() && matches!(*ops[group_start], MetalOp::Dispatch(_) | MetalOp::DispatchIndirect(_)))
            {
                Some("compute pass")
            } else {
                None
            };
            if ok && i < ops.len() && let Some(reason) = cut {
                let Some(split_event) = split_event else {
                    return work::reject("split event unavailable");
                };
                // The next command buffer waits for this one to finish: the
                // queue starts them in order but does not order their memory
                // (views reach title memory by GPU address, untracked), so
                // without the event they overlapped and raced
                // (`split_compute_runs_in_order`).
                let value = SPLIT_VALUE.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                cb.encode_signal_event(split_event, value);
                tracing::debug!(batch, command = cb.handle(), first = cb_start, end = i, reason, "Metal: command buffer operations");
                work::here("bounded pass submit");
                let result = submit_checked(&cb, &ops[cb_start..i], true, "bounded pass");
                work::report(&cb, clamp_counter.take(), batch, &mut indirect_traces, &ops[cb_start..i]);
                if let Err(e) = result {
                    tracing::warn!(error = %e, "Metal: a bounded command buffer failed");
                    fuse::stop();
                    return false;
                }
                if let Some((consumers, sources, before)) = indirect_writes {
                    let after = match work::snapshot_words::<3>(&self.device, &self.queue, &sources) {
                        Ok(args) => args,
                        Err(e) => {
                            tracing::error!(%e, batch, producer = group_start, "Metal: indirect writer snapshot after failed");
                            return work::reject("indirect writer snapshot");
                        }
                    };
                    for (((consumer, c), before), after) in consumers.into_iter().zip(before).zip(after) {
                        tracing::debug!(batch, producer = group_start, consumer, records_va = c.records_va,
                            ?before, ?after, "Metal: indirect writer snapshot");
                    }
                }
                work::here("new command buffer");
                let Ok(next) = self.queue.command_buffer() else { return false };
                next.encode_wait_for_event(split_event, value);
                work::here("run_batch encode");
                let total = work::SPLITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                tracing::info!(total, reason, ops = i - cb_start, first = work::kind(&ops[cb_start]),
                    estimated_us = work::reservation(&ops[cb_start..i]) / 1000, next = work::kind(&ops[i]),
                    next_us = work::reservation(&ops[i..=i]) / 1000, "Metal: command buffer split");
                cb = next;
                cb_start = i;
                cb_dispatches = 0;
            }
        }
        if !ok {
            return false;
        }
        // GPU-written ocean textures back into title memory (F44).
        if ops.iter().any(|o| matches!(**o, MetalOp::Dispatch(_) | MetalOp::DispatchIndirect(_))) {
            work::here("writeback");
            if let Err(e) = self.writeback_all(&cb) {
                tracing::error!(error = %e, "Metal: writeback failed before commit");
                return work::reject("writeback unavailable");
            }
        }
        let encoded_at = started.elapsed();
        work::here("residency commit");
        self.commit_residency();
        work::here("main batch submit");
        let committed_at = started.elapsed();
        tracing::debug!(batch, command = cb.handle(), first = cb_start, end = ops.len(), reason = "batch end", "Metal: command buffer operations");
        let waited = submit_checked(&cb, &ops[cb_start..], cb_start < ops.len(), "main batch");
        work::report(&cb, clamp_counter.take(), batch, &mut indirect_traces, &ops[cb_start..]);
        let waited_at = started.elapsed();
        let res = match waited {
            Ok(()) => {
                // The final-draw dossier: the presented texture right after
                // the draw into it ran (before a later clear).
                let wanted = std::mem::take(&mut *super::metal_draw::CAPTURE_AFTER.lock().unwrap_or_else(|e| e.into_inner()));
                for (resource, name) in wanted {
                    self.capture(resource as crate::object::This, &name);
                }
                true
            }
            Err(e) => {
                let (mut dispatches, mut draws, mut clears) = (0, 0, 0);
                for op in ops {
                    match &**op {
                        MetalOp::Dispatch(_) | MetalOp::DispatchIndirect(_) => dispatches += 1,
                        MetalOp::BufferCopy { .. } => unreachable!("split before encoding"),
                        MetalOp::Draw(_) => draws += 1,
                        MetalOp::Clear(_) | MetalOp::Copy(_) | MetalOp::FloatRowReadback { .. } | MetalOp::Resolve(_) => clears += 1,
                    }
                }
                static SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                if SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                    let what = describe_fault_ops(ops);
                    // SAFETY: a live device.
                    let allocated = unsafe { sys::MTLDevice_currentAllocatedSize(self.device.handle()) } >> 20;
                    // SAFETY: a live device.
                    let recommended = unsafe { sys::MTLDevice_recommendedMaxWorkingSetSize(self.device.handle()) } >> 20;
                    tracing::warn!(dispatches, draws, clears, first = ?what, allocated_mb = allocated, recommended_mb = recommended, error = %e, "Metal: a submission failed on the GPU");
                }
                if draws + clears > 0 && !super::metal_draw::DRAWS_BROKEN.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    tracing::warn!("Metal: draws stop for this run (a submission with draws faulted); compute goes on");
                }
                false
            }
        };
        work::RUN_NS.fetch_add(started.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        let elapsed_ms = started.elapsed().as_millis() as u64;
        if elapsed_ms > 500 {
            let ms = |d: std::time::Duration| d.as_millis() as u64;
            tracing::warn!(
                ops = ops.len(),
                elapsed_ms,
                uploads_ms = ms(uploaded),
                residency_ms = ms(prologue - uploaded),
                encode_ms = ms(encoded_at - prologue),
                commit_ms = ms(committed_at - encoded_at),
                gpu_ms = ms(waited_at - committed_at),
                "Metal: slow run_batch"
            );
        }
        res
    }

    /// The heap entry for the view at `handle`, its stride table words (stride word, buffer
    /// view title address) and why the entry was left empty if it was.
    fn resolve_slot(&self, handle: u64) -> (DescriptorEntry, (u32, u64), Option<super::metal_heap::Missing>) {
        if handle == 0 {
            return (DescriptorEntry::NULL, (0, 0), None);
        }
        let view = crate::descriptor::view(handle);
        let stride_word = view.as_ref().map_or((0, 0), |v| {
            (crate::descriptor::host::view_stride_word(v), crate::descriptor::host::view_address_word(v))
        });
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
            Some(crate::descriptor::View::Shader(..) | crate::descriptor::View::Unordered(..) | crate::descriptor::View::Placed(..))
                if !textures_on()
                    && !matches!(&view, Some(crate::descriptor::View::Placed(_, resource, _)) if resource[0] == 1)
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
            Some(crate::descriptor::View::Placed(unordered, resource, desc)) if resource[0] != 1 => {
                // a placed texture view: one resource per (address, description)
                let address = super::metal_views::placed_address(&desc);
                crate::resource::texture_at(address, &resource)
                    .and_then(|t| self.texture_entry(t, Some(&desc[..10]), unordered))
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
        (entry.unwrap_or(DescriptorEntry::NULL), stride_word, missing)
    }

    fn ensure_heap(&self, heap: usize, count: u32, samplers: bool) -> bool {
        if self.heaps.lock().unwrap_or_else(|e| e.into_inner()).contains_key(&heap) {
            return true;
        }
        let Some(made) = self.new_heap(count, samplers) else {
            return false;
        };
        self.heaps.lock().unwrap_or_else(|e| e.into_inner()).entry(heap).or_insert(made);
        true
    }

    /// The run of title-memory view slots after the `count` slots of view
    /// heap `heap`: its null view (`first`) and `entries` from view
    /// `made + 1` on. The run's first slot.
    pub(crate) fn memory_run(&self, heap: usize, count: u32, first: bool, made: u32, entries: &[DescriptorEntry]) -> Option<u32> {
        if !self.ensure_heap(heap, count, false) {
            return None;
        }
        let heaps = self.heaps.lock().unwrap_or_else(|e| e.into_inner());
        let h = heaps.get(&heap)?;
        let base = h.count.max(victory_shader::emit::SLOT_STRIDES_COUNT);
        if base + 1 + made + entries.len() as u32 > h.capacity {
            return None;
        }
        // SAFETY: every slot written is below the capacity checked above.
        unsafe {
            if first {
                h.entries.add(base as usize).write(DescriptorEntry::NULL);
            }
            for (k, e) in entries.iter().enumerate() {
                h.entries.add((base + 1 + made) as usize + k).write(*e);
            }
        }
        Some(base)
    }

    /// Writes slot `index` of title heap `heap` (`count` slots) from the
    /// view at `handle`; returns the Metal heap buffer, stride table GPU address,
    /// and why the slot was left empty if it was.
    pub(crate) fn refresh_slot(
        &self,
        heap: usize,
        count: u32,
        samplers: bool,
        index: u32,
        handle: u64,
    ) -> Option<(Buffer, u64, Option<super::metal_heap::Missing>)> {
        let (entry, stride_word, missing) = self.resolve_slot(handle);
        if !self.ensure_heap(heap, count, samplers) {
            return None;
        }
        let heaps = self.heaps.lock().unwrap_or_else(|e| e.into_inner());
        let h = heaps.get(&heap)?;
        if index >= h.count {
            return None;
        }
        // SAFETY: index < count, the entries' memory holds >= count entries.
        unsafe { h.entries.add(index as usize).write(entry) };
        if let Some((_, strides_ptr)) = h.strides {
            // SAFETY: the strides table holds SLOT_STRIDES_DWORDS dwords (`new_heap`).
            unsafe { crate::descriptor::host::write_stride_slot(strides_ptr, index, stride_word.0, stride_word.1) };
        }
        let strides_gpu = h.strides.as_ref().map_or(0, |(b, _)| b.gpu_address);
        Some((h.buffer.clone(), strides_gpu, missing))
    }

    /// Mirrors all descriptor slots dirtied since the last sync for `heap`,
    /// returning `(heap_buffer, strides_gpu)`.
    pub(crate) fn sync_heap(&self, heap: usize, count: u32, samplers: bool) -> Option<(Buffer, u64)> {
        super::metal_texture::drain_remap_partial();
        let dirty: Vec<(u32, u64)> = DIRTY_SLOTS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_mut()
            .and_then(|m| m.remove(&heap))
            .map(|m| m.into_iter().collect())
            .unwrap_or_default();
        let started = (!dirty.is_empty()).then(std::time::Instant::now);
        let slots = dirty.len();
        let resolved: Vec<(u32, DescriptorEntry, (u32, u64))> = dirty
            .into_iter()
            .filter(|&(idx, _)| idx < count)
            .map(|(idx, handle)| {
                let (entry, stride_word, _) = self.resolve_slot(handle);
                (idx, entry, stride_word)
            })
            .collect();
        if !self.flush_uploads() {
            return None;
        }
        if !self.ensure_heap(heap, count, samplers) {
            return None;
        }
        let heaps = self.heaps.lock().unwrap_or_else(|e| e.into_inner());
        let h = heaps.get(&heap)?;
        for (idx, entry, stride_word) in resolved {
            if idx < h.count {
                // SAFETY: idx < h.count.
                unsafe { h.entries.add(idx as usize).write(entry) };
                if let Some((_, strides_ptr)) = h.strides {
                    // SAFETY: the strides table holds SLOT_STRIDES_DWORDS dwords (`new_heap`).
                    unsafe { crate::descriptor::host::write_stride_slot(strides_ptr, idx, stride_word.0, stride_word.1) };
                }
            }
        }
        let strides_gpu = h.strides.as_ref().map_or(0, |(b, _)| b.gpu_address);
        let out = Some((h.buffer.clone(), strides_gpu));
        if let Some(started) = started {
            let elapsed_ms = started.elapsed().as_millis() as u64;
            if elapsed_ms > 500 {
                tracing::warn!(slots, elapsed_ms, "Metal: slow sync_heap");
            }
        }
        out
    }

    /// Resident 32 MiB zero-initialized scratch buffer for spilled wave64-on-wave32 helpers.
    pub(crate) fn wave_scratch_gpu(&self) -> u64 {
        self.wave_scratch
            .get_or_init(|| {
                use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
                let bytes = 32usize << 20;
                // SAFETY: fresh page-aligned allocation kept for the run.
                let mem = unsafe { VirtualAlloc(std::ptr::null(), bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
                if mem.is_null() {
                    return None;
                }
                // SAFETY: lives for the run (never freed).
                let buffer = unsafe { self.device.wrap(mem, bytes as u64) }.ok()?;
                self.add_resident(&[buffer.handle()]);
                Some(buffer)
            })
            .as_ref()
            .map_or(0, |b| b.gpu_address)
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

    fn static_sampler_entry(&self, s: &StaticSampler) -> Option<DescriptorEntry> {
        let border: [f32; 4] = match s.border_color {
            0 => [0.0, 0.0, 0.0, 0.0],
            1 | 3 => [0.0, 0.0, 0.0, 1.0],
            _ => [1.0, 1.0, 1.0, 1.0],
        };
        let words = [
            s.filter,
            s.address_u,
            s.address_v,
            s.address_w,
            s.mip_lod_bias.to_bits(),
            s.max_anisotropy,
            s.comparison_func,
            border[0].to_bits(),
            border[1].to_bits(),
            border[2].to_bits(),
            border[3].to_bits(),
            s.min_lod.to_bits(),
            s.max_lod.to_bits(),
        ];
        self.sampler_entry(&words)
    }

    /// Binds MSC's static sampler table (the `Table` location at `num_params`
    /// in `top`) to a resident Metal buffer of `DescriptorEntry`s. Without it
    /// that pointer stays 0 in the top-level argument buffer and every sample
    /// through a static sampler reads zeros.
    pub(crate) fn bind_static_samplers(
        &self,
        top: &mut TopLevel,
        num_params: usize,
        samplers: &[StaticSampler],
    ) -> Result<(), String> {
        if samplers.is_empty() {
            return Ok(());
        }
        let mut bytes = Vec::with_capacity(samplers.len() * size_of::<DescriptorEntry>());
        for s in samplers {
            let e = self.static_sampler_entry(s).ok_or_else(|| format!("unsupported static sampler {s:?}"))?;
            bytes.extend_from_slice(&e.gpu_va.to_le_bytes());
            bytes.extend_from_slice(&e.texture_view_id.to_le_bytes());
            bytes.extend_from_slice(&e.metadata.to_le_bytes());
        }
        let gpu_address = {
            let mut tables = self.static_sampler_tables.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(buf) = tables.get(&bytes) {
                buf.gpu_address
            } else {
                let buf = self.device.shared_buffer(&bytes).map_err(|e| e.to_string())?;
                self.add_resident(&[buf.handle()]);
                let addr = buf.gpu_address;
                tables.insert(bytes, buf);
                addr
            }
        };
        if !top.set_static_samplers(num_params, gpu_address) {
            return Err(format!(
                "static sampler table missing from top-level layout (num_params={num_params}, layout={:?})",
                top.layout
            ));
        }
        Ok(())
    }

    fn new_ring(&self, capacity: usize) -> Option<Ring> {
        #[cfg(test)]
        if work::FAULT.get() == Some("ring") { return None; }
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
        self.add_resident(&[buffer.handle()]);
        Some(Ring { buffer, entries: mem.cast(), capacity: bytes / size_of::<DescriptorEntry>() })
    }

    fn new_heap(&self, count: u32, samplers: bool) -> Option<MetalHeap> {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        let capacity = if samplers {
            count
        } else {
            count.max(victory_shader::emit::SLOT_STRIDES_COUNT) + memory::COPIES + 1
        };
        let bytes = (capacity as usize * size_of::<DescriptorEntry>()).next_multiple_of(64 << 10);
        // SAFETY: a fresh allocation (zero: every slot empty), never freed
        // (heaps live as long as the title's).
        let mem = unsafe { VirtualAlloc(std::ptr::null(), bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        if mem.is_null() {
            return None;
        }
        // SAFETY: as above.
        let buffer = unsafe { self.device.wrap(mem, bytes as u64) }.ok()?;
        let strides = if samplers {
            None
        } else {
            let sbytes = (victory_shader::emit::SLOT_STRIDES_DWORDS as usize * 4).next_multiple_of(64 << 10);
            // SAFETY: fresh allocation for the SLOT_STRIDES table: strides, then view addresses
            // (zero-initialized).
            let smem = unsafe { VirtualAlloc(std::ptr::null(), sbytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
            if smem.is_null() {
                return None;
            }
            // SAFETY: as above.
            let sbuf = unsafe { self.device.wrap(smem, sbytes as u64) }.ok()?;
            Some((sbuf, smem.cast::<u32>()))
        };
        if let Some((sbuf, _)) = &strides {
            self.add_resident(&[buffer.handle(), sbuf.handle()]);
        } else {
            self.add_resident(&[buffer.handle()]);
        }
        Some(MetalHeap { buffer, entries: mem.cast(), count, capacity, strides })
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
            && let Some(mut converted) = victory_metal::msc::Converted::from_bytes(&bytes)
        {
            match self.device.compute_pipeline_converted(&converted) {
                Ok(pso) => {
                    self.bind_static_samplers(&mut converted.top_level, params.len(), &samplers)?;
                    return Ok(MetalCompute { cost_ns: Default::default(), name: p.name.clone(), pso, threadgroup: converted.threadgroup.map(u64::from), top_level: converted.top_level });
                }
                Err(e) => tracing::debug!(error = %e, "a cached Metal pipeline did not load; converting again"),
            }
        }
        let (pso, mut converted) = self.device.compute_pipeline_dxil(&host.dxil, &params, &samplers).map_err(|e| e.to_string())?;
        if let Some(path) = &path {
            let tmp = path.with_extension(format!("tmp{}", std::process::id()));
            if std::fs::write(&tmp, converted.to_bytes()).is_ok() {
                let _ = std::fs::rename(&tmp, path);
            }
        }
        self.bind_static_samplers(&mut converted.top_level, params.len(), &samplers)?;
        Ok(MetalCompute { cost_ns: Default::default(), name: p.name.clone(), pso, threadgroup: converted.threadgroup.map(u64::from), top_level: converted.top_level })
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
    use victory_shader::emit::{SLOT_DIAG, SLOT_FAULTS, SLOT_MEMORY, SLOT_STRIDES, SLOT_WAVE_SCRATCH};
    let mut out = Vec::new();
    let dwords = t.user_data_dwords();
    if dwords > 0 {
        out.push(RootSlot::Constants(dwords));
    }
    out.extend(t.root_cbvs.iter().map(|&s| RootSlot::Cbv(s)));
    out.extend(t.root_srvs.iter().map(|&s| if s == SLOT_STRIDES || s == SLOT_MEMORY { RootSlot::Srv(s) } else { RootSlot::TableSrv(s) }));
    out.extend(t.root_uavs.iter().map(|&s| if s == SLOT_FAULTS || s == SLOT_WAVE_SCRATCH || s == SLOT_DIAG { RootSlot::Uav(s) } else { RootSlot::TableUav(s) }));
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
        && !(stage == victory_metal::msc::Stage::Mesh && c.threadgroup == [0, 0, 0])
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

#[cfg(test)]
#[path = "metal_static_sampler_repro_tests.rs"]
mod static_sampler_repro_tests;

#[cfg(test)]
#[path = "metal_stencil_twin_tests.rs"]
mod stencil_twin_tests;

#[path = "metal_work.rs"]
pub(super) mod work;

#[path = "metal_completion.rs"]
mod completion;

#[path = "metal_diag.rs"]
pub(crate) mod diag;

#[cfg(test)]
#[path = "metal_half_sync_repro_tests.rs"]
mod half_sync_repro_tests;

#[cfg(test)]
#[path = "metal_append_idiom_repro_tests.rs"]
mod append_idiom_repro_tests;

#[cfg(test)]
#[path = "metal_submit_tests.rs"]
mod submit_tests;

#[cfg(test)]
#[path = "metal_msc_check_tests.rs"]
mod msc_check_tests;

/// Names what a command buffer that does not complete was running: a
/// GPU-side hang (a shader that never ends) blocks `cb.wait()` with no
/// error and no log, and the title stops presenting.
mod inflight {
    use super::MetalOp;
    use std::sync::{Arc, Mutex, Once};
    use std::time::{Duration, Instant};

    /// An indirect dispatch's arguments: name, records, stride, max, count.
    type Args = (String, u64, u32, u32, Option<u64>);
    /// (serial, started, what it holds, its first indirect dispatches) of
    /// the command buffer waited for.
    static NOW: Mutex<Option<(u64, Instant, String, Vec<Args>)>> = Mutex::new(None);

    /// Up to `n` dwords of title memory at `va`, when committed.
    fn peek(va: u64, n: usize) -> Option<Vec<u32>> {
        if va == 0 || va % 4 != 0 {
            return None;
        }
        let bytes = n * 4;
        let _reading = crate::watch::title_memory_reading_range(va as usize, bytes);
        if victory_metal::chunks::committed_extent(va, bytes as u64) < bytes as u64 {
            return None;
        }
        // SAFETY: committed, guarded against remapping, 4-byte aligned.
        Some((0..n).map(|k| unsafe { (va as *const u32).add(k).read_volatile() }).collect())
    }

    fn args_now(args: &[Args]) -> String {
        args.iter()
            .map(|(name, va, stride, max, count)| {
                let live = count.and_then(|c| peek(c, 1)).map(|v| v[0]);
                let shown = (*max).min(live.unwrap_or(*max)).min(4) as usize;
                let records: Vec<Option<Vec<u32>>> = (0..shown).map(|k| peek(va + k as u64 * u64::from(*stride), 3)).collect();
                format!("{name} max={max} count={live:?} records={records:?}")
            })
            .collect::<Vec<_>>()
            .join("; ")
    }
    static SERIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    pub(super) struct Watch;

    impl Drop for Watch {
        fn drop(&mut self) {
            *NOW.lock().unwrap_or_else(|e| e.into_inner()) = None;
        }
    }

    pub(super) fn describe(ops: &[Arc<MetalOp>]) -> String {
        let mut names: Vec<(String, u32)> = Vec::new();
        for op in ops {
            let name = match &**op {
                MetalOp::Dispatch(c) => format!("compute {}", c.pso.name),
                MetalOp::DispatchIndirect(c) => format!("indirect {}", c.pso.name),
                MetalOp::Draw(d) => d.pso.name.clone(),
                MetalOp::Clear(_) => "clear".into(),
                MetalOp::Copy(_) => "copy".into(),
                MetalOp::FloatRowReadback { .. } => "readback".into(),
                MetalOp::BufferCopy { .. } => "buffer copy".into(),
                MetalOp::Resolve(_) => "resolve".into(),
            };
            match names.last_mut() {
                Some((last, n)) if *last == name => *n += 1,
                _ => names.push((name, 1)),
            }
        }
        let mut out = names.iter().map(|(n, k)| format!("{n} x{k}")).collect::<Vec<_>>().join("; ");
        // The first draws' sizes: a cheap shader can still run long.
        use crate::backend::metal_draw::DrawKind;
        let draws: Vec<String> = ops
            .iter()
            .filter_map(|op| match &**op {
                MetalOp::Draw(d) => Some(d),
                _ => None,
            })
            .take(6)
            .map(|d| {
                let kind = match &d.kind {
                    DrawKind::Vertices { count, instances, .. } => format!("vertices {count}x{instances}"),
                    DrawKind::Indexed { count, instances, .. } => format!("indexed {count}x{instances}"),
                    DrawKind::Mesh { groups } => format!("mesh {groups:?}"),
                    DrawKind::Geometry { draw, .. } => format!("geometry {:?}", draw.groups),
                    DrawKind::Generated { indices, instances, .. } => format!("generated {}x{instances}", indices.len()),
                    DrawKind::Indirect { max, .. } => format!("indirect max {max}"),
                };
                format!("{kind} into {}x{}x{} prim {}", d.targets.width, d.targets.height, d.targets.samples, d.primitive)
            })
            .collect();
        if !draws.is_empty() {
            out.push_str(&format!(" draws={draws:?}"));
        }
        out
    }

    /// Marks `ops` in flight until the returned guard drops; a monitor
    /// thread logs them once if that takes over 8 s.
    pub(super) fn watch(ops: &[Arc<MetalOp>]) -> Watch {
        static MONITOR: Once = Once::new();
        MONITOR.call_once(|| {
            let _ = std::thread::Builder::new().name("metal-hang-watch".into()).spawn(|| {
                let mut told = 0;
                let mut stuck_told = [0u64; super::work::LANES];
                loop {
                    std::thread::sleep(Duration::from_secs(1));
                    // Host phases first: lock-free, so a thread stuck holding a
                    // lock (or in a Metal call) cannot keep this quiet.
                    for (lane, told_at) in stuck_told.iter_mut().enumerate() {
                        if let Some((phase, since_ms, at)) = super::work::stuck_phase(lane, 8_000)
                            && *told_at != at
                        {
                            *told_at = at;
                            tracing::warn!(lane, phase, since_ms, "Metal: stuck in a host phase (hang?)");
                        }
                    }
                    // Copy out and unlock before logging or reading title memory
                    // (`Watch::drop` takes this lock on the submitting thread).
                    let held = {
                        let now = NOW.lock().unwrap_or_else(|e| e.into_inner());
                        match &*now {
                            Some((serial, started, what, args)) if *serial != told && started.elapsed() > Duration::from_secs(8) =>
                                Some((*serial, started.elapsed().as_secs(), what.clone(), args.clone())),
                            _ => None,
                        }
                    };
                    if let Some((serial, elapsed_s, what, args)) = held {
                        told = serial;
                        tracing::warn!(elapsed_s, ops = %what, "Metal: a command buffer has not completed (GPU hang?)");
                        let args = args_now(&args);
                        tracing::warn!(elapsed_s, indirect = %args, "Metal: the hung command buffer's indirect arguments");
                    }
                }
            });
        });
        let serial = SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        let mut args: Vec<Args> = Vec::new();
        for op in ops {
            if let MetalOp::DispatchIndirect(c) = &**op
                && !args.iter().any(|a| a.0 == c.pso.name)
                && args.len() < 6
            {
                args.push((c.pso.name.clone(), c.records_va, c.stride, c.max, c.count_va));
            }
        }
        *NOW.lock().unwrap_or_else(|e| e.into_inner()) = Some((serial, Instant::now(), describe(ops), args));
        Watch
    }
}

/// Stop further `run` submissions after a command buffer exceeds
/// `VICTORY_METAL_FUSE_MS` (default 1500; 0: off). This is measured only
/// after completion and cannot interrupt an already running command buffer.
static STOPPED_ENDPOINTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
mod fuse {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Mutex, OnceLock};

    static TRIPPED: AtomicBool = AtomicBool::new(false);
    static COMMIT: Mutex<()> = Mutex::new(());

    pub(super) fn commit_if_open(cb: &victory_metal::CommandBuffer) -> bool {
        let _gate = COMMIT.lock().unwrap_or_else(|e| e.into_inner());
        if tripped() { return false; }
        cb.commit();
        true
    }

    pub(super) fn tripped() -> bool {
        TRIPPED.load(Ordering::Acquire)
    }

    pub(super) fn stop() {
        let _gate = COMMIT.lock().unwrap_or_else(|e| e.into_inner());
        TRIPPED.store(true, Ordering::Release);
    }

    pub(super) fn limit_ns() -> u64 {
        #[cfg(test)]
        if TEST_ONE_NS.load(Ordering::Relaxed) { return 1; }
        static MS: OnceLock<u64> = OnceLock::new();
        MS.get_or_init(|| std::env::var("VICTORY_METAL_FUSE_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(1500)).saturating_mul(1_000_000)
    }
    #[cfg(test)]
    pub(super) static TEST_ONE_NS: AtomicBool = AtomicBool::new(false);

    pub(super) fn unsafe_timing(ns: u64, limit: u64) -> bool {
        limit != 0 && (ns == 0 || ns > limit)
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn timing_rejects_missing_or_excessive_duration() {
            assert!(super::unsafe_timing(0, 100));
            assert!(super::unsafe_timing(101, 100));
            assert!(!super::unsafe_timing(100, 100));
            assert!(!super::unsafe_timing(0, 0));
        }
    }
}
