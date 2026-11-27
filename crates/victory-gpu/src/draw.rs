//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, topology, targets) is set on the host list as the
//! title sets it.

use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::resource::RootIndexedView;
use victory_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::View;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::Interface;

use crate::host::HostList;
use crate::object::This;
use crate::pipeline::host::HostPipeline;

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Vec<u32>),
    /// A root view's GPU address.
    View(u64),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VertexView {
    address: u64,
    size: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IndexView {
    address: u64,
    size: u32,
    format: u32,
}

/// Host states of a depth target, and of a texture shaders read.
const DEPTH_WRITE: u32 = 0x10;
const DEPTH_READ: u32 = 0x20;
const SHADER_RESOURCE: u32 = 0x40 | 0x80;

thread_local! {
    /// While a draw is prepared (`State::bind`, then its targets' states):
    /// the depth target the draw only tests (0: none, or one it writes),
    /// and whether a view its shaders read is of that same texture.
    static TESTED_DEPTH: std::cell::Cell<(usize, bool)> =
        const { std::cell::Cell::new((0, false)) };
    /// While a stage's tables are walked (`State::tables`): whether that
    /// stage's shader can store to buffers (`Translated::stores`).
    static STORES: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    /// While `State::early_textures` runs the texture pass: the
    /// milliseconds and the bytes that run may take, in place of the
    /// pass's own per frame.
    static PASS_BUDGET: std::cell::Cell<Option<(u64, u64)>> = const { std::cell::Cell::new(None) };
}

/// The milliseconds of every frame the pass that uploads textures picked
/// by index takes before a shader that picks by index has asked for it
/// (`VICTORY_EARLY_TEXTURES_MS`, 12; 0: never, as before), and the
/// milliseconds it may take just before a draw of the title's decoration
/// bake (`VICTORY_BAKE_TEXTURES_MS`, 1500; 0: nothing special).
///
/// The pass ran only from the first world frame on, where it met some
/// 24,000 slots at once and needed 60-80 frames for them. The title bakes
/// its characters' skin and decorations into textures of its own while it
/// loads, 159-1,165 frames before that (runs 156-174), with a pixel shader
/// that picks its sources by index: whatever was not on the card then was
/// baked as nothing, once, for the run. On the console a texture is there
/// when its bytes are.
fn early_budgets() -> (u64, u64) {
    static MS: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *MS.get_or_init(|| {
        let ms = |name: &str, default: u64| {
            std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(default)
        };
        (ms("VICTORY_EARLY_TEXTURES_MS", 12), ms("VICTORY_BAKE_TEXTURES_MS", 1500))
    })
}

/// The pixel shaders of the title's decoration bake (the title:
/// `PS_BakeDecorations_*`), by the first 16 hex digits of their hash as the
/// frame listings print it. They draw into a 256x256 corner of 2048x2048
/// targets and name no texture in a table.
const BAKE_PIXEL_SHADERS: [&str; 4] =
    ["c0a7fe77b1862e10", "0d1575a396dfe513", "b7cfd8dda91ebb6b", "280967ad158cee0b"];

/// Since the start: early runs of the texture pass, and those of them
/// made for a bake draw.
pub(crate) static EARLY_TEXTURE_PASSES: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Whether a buffer range a shader names through an unordered access view
/// is given to shaders as written (snapshot before the submission, compare
/// and write back after) only if that shader's code can store to buffers.
/// The title binds buffers its shaders only read through such views, and
/// every range declared costs a snapshot and a compare of its bytes: 3.97
/// GB a frame in 31 submissions, 0.49 s of snapshots and 0.28 s of
/// comparing (run 28). On by default; `VICTORY_WRITE_SKIP=0` declares
/// every such range, as before.
fn write_skip() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WRITE_SKIP").is_none_or(|v| v != "0"))
}

/// A run of indirect records clamped in one pass ([`windows`]): the host
/// copy they came from (its number), their kind and stride, the offset of
/// the first in that copy, how many, and the list's scratch buffer that
/// holds them clamped, record `i` at `i * step`.
#[derive(Clone)]
struct Window {
    copy: u64,
    kind: u32,
    stride: u32,
    /// How far apart the records are in the copy, and in the scratch
    /// buffer: what the title walks them by ([`windows`]).
    step: u32,
    base: u64,
    records: u32,
    scratch: ID3D12Resource,
}

/// The most records clamped in one pass ([`windows`]).
const WINDOW_MOST: u64 = 64;

/// Whether indirect draws that read record after record of one argument
/// buffer are clamped a window at a time (`VICTORY_INDIRECT_WINDOW=1`; off
/// by default).
///
/// The title issues its mesh draws one record per call, each from the next
/// slot of the same argument buffer: 3,460 a frame in gameplay, and before
/// each our clamp ran as a compute dispatch of its own between two waits,
/// so the card went from drawing to compute and back for every record. In
/// the timing run of the lit scene (run 79) a shadow draw of next to
/// nothing cost 14-16 microseconds, 4,126 records a frame.
///
/// With the switch, when a draw asks for a record a short step after the
/// one drawn before, the clamp takes that record and the ones that step
/// apart after it (16, then twice as many each time, [`WINDOW_MOST`] at
/// most, never past the buffer) into
/// the list's scratch buffer in one pass, and the draws that follow read
/// their record there with no pass of ours, until one of these: a compute
/// dispatch, a direct draw, an indirect call with a count buffer or more
/// than one record, a draw whose shaders can store to buffers, a record
/// outside the window, another host list.
///
/// The step is the one the title is seen to take, not the stride its
/// command signature names: it draws one record a call, for which the
/// stride means nothing, and the title names 12 bytes (the three
/// words of a mesh dispatch) for records it keeps 32 bytes apart (2,173 of
/// a frame's 3,349 mesh records in run 87's listed frame; 283 are 16
/// apart, which ends a window). Walked by the named stride the window
/// never opened: run 87, with the switch, still clamped 5,068 records a
/// frame one at a time.
///
/// What it costs the guard against garbage records: nothing against a
/// hang (every record is bounded as before and charged to the list's
/// budget as before), but the budget is charged when a window is clamped,
/// so also for records of the window's tail that are never drawn. Those
/// hold garbage in this title (slots of all ones: run 93 spent lists'
/// budgets on them and emptied the draws after), so in a window a record
/// that asks for more than can run is left empty and charges nothing
/// (`crate::indirect::bound_ahead`); a tail record that looks real is
/// charged what it asks, a few thousand of 2^27. And arguments the title's CPU side changes are seen as
/// before, at the next submission.
fn windows() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_INDIRECT_WINDOW").is_some_and(|v| v == "1"))
}

/// What one title list has set, and what its host list was last given.
#[derive(Default)]
pub(crate) struct State {
    /// Compute dispatches recorded for Metal (`VICTORY_METAL=compute`),
    /// run when the list is submitted.
    #[cfg(windows)]
    pub(crate) metal: Vec<std::sync::Arc<crate::backend::metal::MetalOp>>,
    /// A Metal dispatch was recorded since the list began or since its last
    /// barrier (or command that orders like one): the next dispatch may
    /// overlap it ([`crate::backend::metal::MetalCmd::after_barrier`]).
    #[cfg(windows)]
    pub(crate) metal_since_barrier: bool,
    /// The bound targets as Metal attachments, made at the first Metal draw
    /// after they were set (`None`: not made yet).
    #[cfg(windows)]
    metal_targets: Option<std::sync::Arc<crate::backend::metal_draw::Targets>>,
    pipeline: usize,
    /// View heap, sampler heap.
    heaps: [usize; 2],
    graphics: Vec<Root>,
    compute: Vec<Root>,
    vertex: [VertexView; 32],
    index: IndexView,
    /// Bound render target and depth resources, for their host states.
    targets: Vec<usize>,
    depth: usize,
    /// The host state the bound depth view needs ([`depth_state`]).
    depth_state: u32,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the title set that goes straight onto the host list, kept for
    /// the host list that continues this one after a cut ([`State::replay`]):
    /// whether a topology was set, the render target and depth handles, the
    /// viewports, the scissors, the blend factor and the stencil reference.
    topology_set: bool,
    target_handles: Vec<u64>,
    depth_handle: u64,
    targets_set: bool,
    viewports: Vec<D3D12_VIEWPORT>,
    scissors: Vec<RECT>,
    blend: Option<[f32; 4]>,
    stencil: Option<u32>,
    depth_bounds: Option<[f32; 2]>,
    /// The indirect records clamped last, if later draws may still read
    /// them ([`windows`]).
    window: Option<Window>,
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    host_heaps: [usize; 2],
    /// Tables made afresh in this list: (table, slot count, contents) → first
    /// dynamic slot.
    fresh_tables: std::collections::HashMap<(u64, u32, u64), u32>,
    /// The dynamic slots the list holds: title heap, first, count; given
    /// back when the list has run, is reset or released.
    dynamic: Vec<(usize, u32, u32)>,
    /// Root SRVs bound as one-view tables in this list: (title heap, title
    /// address; 0 for none) → dynamic slot, and the host copy the slot
    /// shows: later draws of the list bring that one up to date, whatever
    /// copy the address would find by then (`shadow.rs` `tails`).
    root_srv_slots: std::collections::HashMap<(usize, u64), RootSlot>,
    /// Root UAVs bound as one-view tables in this list, likewise.
    root_uav_slots: std::collections::HashMap<(usize, u64), RootSlot>,
    /// The list ran and its dynamic slots were given back.
    tables_gone: bool,
}

impl Drop for State {
    fn drop(&mut self) {
        for &(heap, first, n) in &self.dynamic {
            crate::descriptor::release_dynamic(heap, first, n);
        }
    }
}

impl State {
    /// The list's submission has run (submissions are waited for): the
    /// dynamic slots its tables were made in are given back, not kept
    /// until the title resets the list. A frame of the opening cutscene
    /// asks for 12,500 tables; held until the resets they ran the 131,072
    /// slots out, and up to 4,900 draws a frame went without their tables.
    /// shortcut: a list the title runs a second time without a reset then
    /// reads slots others may have taken (logged once).
    pub(crate) fn tables_done(&mut self) {
        if self.tables_gone && self.dynamic.is_empty() {
            return;
        }
        if self.tables_gone {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!("a list ran again after its descriptor tables were given back"));
        }
        for (heap, first, n) in self.dynamic.drain(..) {
            crate::descriptor::release_dynamic(heap, first, n);
        }
        self.fresh_tables.clear();
        self.root_srv_slots.clear();
        self.root_uav_slots.clear();
        self.tables_gone = true;
    }
}

/// An experiment on the title's scissor rectangles (`VICTORY_SCISSOR_TEST=full`):
/// they are all widened to the whole target. The title's room at
/// the start of the prologue lost its walls in five runs of seven, and in
/// those runs the player's body and the world outside were cut by one
/// rectangle around the window (run 85, frame 2046): one run with this on
/// says whether the walls are drawn and cut away by such a rectangle, or
/// not drawn at all. Off by default: a rectangle the title sets is part of
/// its picture.
fn scissor_test() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_SCISSOR_TEST").as_deref() == Ok("full"))
}

/// An experiment on mesh draws (`VICTORY_MESH_TEST`): 1 for `skip` (none
/// is issued), 2 for `scissor` (each runs its shaders but rasterises one
/// pixel), 0 otherwise. It tells a card hung by a mesh shader from one hung
/// by what the shader's triangles cover.
fn mesh_test() -> u8 {
    static ON: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_MESH_TEST").as_deref() {
        Ok("skip") => 1,
        Ok("scissor") => 2,
        _ => 0,
    })
}

/// `VICTORY_MESH_TEST=keep:<hex digits>[:<hex digits>]`: only the mesh
/// draws whose mesh shader's hash ends in one of the first digits (and
/// whose digit before it is one of the second, if given) are issued: a
/// search for the shaders that stop the card.
fn mesh_kept(pipeline: usize) -> bool {
    static KEEP: std::sync::OnceLock<Option<(String, String)>> = std::sync::OnceLock::new();
    let Some((last, before)) = KEEP.get_or_init(|| {
        let value = std::env::var("VICTORY_MESH_TEST").ok()?;
        let mut parts = value.strip_prefix("keep:")?.split(':');
        Some((parts.next()?.to_owned(), parts.next().unwrap_or("").to_owned()))
    }) else {
        return true;
    };
    let Some(p) = crate::pipeline::host_pipeline(pipeline as This) else {
        return true;
    };
    let Some(at) = p.name.find("mesh ") else {
        return true;
    };
    let hash = p.name[at + 5..].as_bytes();
    if hash.len() < 16 {
        return true;
    }
    last.as_bytes().contains(&hash[15]) && (before.is_empty() || before.as_bytes().contains(&hash[14]))
}

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
}

/// The draws traced since the last present (diagnostics builds, with
/// debug logging for this module): logged when the frame comes out black.
#[cfg(feature = "diagnostics")]
static FRAME: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The draws traced since the last call.
#[cfg(feature = "diagnostics")]
pub(crate) fn take_frame() -> Vec<String> {
    std::mem::take(&mut *FRAME.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// (no host pipeline) since the last call.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::Relaxed)
}
/// A root SRV's dynamic slot in a list, and the host copy it shows.
type RootSlot = (u32, Option<std::sync::Arc<crate::shadow::Shadow>>);

type HeapBufferRange = (std::sync::Arc<crate::shadow::Shadow>, u64, u64);
type HeapBufferSpans = crate::fx::Map<usize, HeapBufferRange>;

#[derive(Default)]
struct HeapBufferBucket {
    slots: crate::fx::Map<u32, HeapBufferRange>,
    spans: HeapBufferSpans,
}

#[derive(Default)]
pub(crate) struct HeapBufferCache {
    passes: usize,
    buckets: [HeapBufferBucket; crate::descriptor::host::BUFFER_BUCKETS],
}

fn cover_buffer_span(spans: &mut HeapBufferSpans, range: &HeapBufferRange) {
    let key = std::sync::Arc::as_ptr(&range.0) as usize;
    spans.entry(key).and_modify(|old| {
        old.1 = old.1.min(range.1);
        old.2 = old.2.max(range.2);
    }).or_insert_with(|| range.clone());
}

/// Buffer views of the heap whose slot showed another host buffer than
/// the copy the pass brought up to date for them, since the start
/// (diagnostics; for the memory line). It stays 0 while the pass and the
/// slots agree; with `VICTORY_VIEW_EXTENT=0` it counts the views that were
/// read stale.
#[cfg(feature = "diagnostics")]
pub(crate) static HEAP_VIEWS_ELSEWHERE: AtomicU64 = AtomicU64::new(0);
/// Passes over the view heap's buffer views for shaders that pick them by
/// index ([`State::heap_buffers`]), and the microseconds they took.
pub(crate) static HEAP_BUFFER_PASSES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_BUFFER_MICROS: AtomicU64 = AtomicU64::new(0);
/// Textures brought up to date for shaders that pick them by index
/// ([`State::heap_textures`]), the bytes of their memory, and the slots
/// still waiting.
pub(crate) static HEAP_TEXTURES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_TEXTURE_BYTES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_TEXTURES_WAITING: AtomicU64 = AtomicU64::new(0);
/// Passes that made no texture because the card had no room
/// (`crate::resource::room`), for the memory line.
pub(crate) static HEAP_TEXTURES_FULL: AtomicU64 = AtomicU64::new(0);

/// Looks at uploaded textures picked by index that found the memory as it
/// was, that found it changed (uploaded again), since the start; and the
/// slots that wait for such a look now.
pub(crate) static HEAP_TEXTURES_RECHECKED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Whether textures picked by index are looked at again after their upload
/// (`VICTORY_HEAP_RECHECK=0`: no).
fn heap_recheck() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_RECHECK").is_none_or(|v| v != "0"))
}

/// Times a placed texture's slot was passed over because its memory is not
/// mapped.
pub(crate) static HEAP_TEXTURES_AWAY: AtomicU64 = AtomicU64::new(0);
/// Slots the pass found up to date already: their texture is uploaded by
/// the draws that read it through a table (it has a staging buffer).
pub(crate) static HEAP_TEXTURES_TABLED: AtomicU64 = AtomicU64::new(0);
/// The queue of textures picked by index at the last pass
/// (`TextureQueue::census`): slots in it, due, waiting for their memory,
/// uploaded in part, and the frames the oldest due one has waited.
pub(crate) static HEAP_TEXTURES_QUEUE: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
/// Since the start: looks at a slot uploaded in part that found no more of
/// its memory mapped; uploads of textures not all mapped; and of those,
/// the ones whose first page was not there (asked by the first page alone,
/// as before, they counted as not there at all).
pub(crate) static HEAP_TEXTURES_PARTLY: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Marks the last traced draw as not run on the host.
#[cfg(feature = "diagnostics")]
fn dropped(pipeline: usize) {
    crate::pipeline::note_drop(pipeline as This);
    if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
        last.push_str(" DROPPED");
    }
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);
/// D3D12-era preparation the Metal record path still does (stage 4 is to
/// bring these to 0): heap syncs (per stage), slot refreshes, and root-view
/// bounds from the shadow copies.
static METAL_PREP: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// [`METAL_PREP`]: (heap syncs, slot refreshes, shadow bounds) so far.
pub(crate) fn metal_prep() -> [u64; 3] {
    METAL_PREP.each_ref().map(|c| c.load(Ordering::Relaxed))
}

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

/// The host address of a root view: its buffer's host copy, or for a
/// constant buffer not 256-byte aligned there (the console needs 16) a
/// copy in upload memory.
/// shortcut: a root view has no size; constant buffers are taken as 64 KiB
/// (cut at their buffer's end), other views as all that is left of their
/// buffer. (They were taken as 1 MiB: 1,452 of the 2,021 mesh vertex
/// streams of a cutscene frame reach further, and what lay past the first
/// megabyte was never brought up to date - meshes without bodies, exploded
/// triangles. `VICTORY_ROOT_VIEW_MB` sets a limit again.)
fn root_view(list: &HostList, address: u64, srv: bool) -> Option<u64> {
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let limit = *LIMIT.get_or_init(|| {
        std::env::var("VICTORY_ROOT_VIEW_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map_or(u64::MAX, |mb| mb << 20)
    });
    let shadow = crate::shadow::find(address, 1)?;
    let len = shadow
        .left(address)
        .min(if srv { limit } else { 1 << 16 });
    let gpu = shadow.gpu(address);
    if srv || gpu % 256 == 0 {
        shadow.refresh(list, address, len);
        return Some(gpu);
    }
    let (gpu, to) = list.scratch(len)?;
    list.defer_fill(to, address as *const u8, len as usize);
    Some(gpu)
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {
    let slot = slot as usize;
    if roots.len() <= slot {
        roots.resize(slot + 1, Root::None);
    }
    roots[slot] = root;
}

impl State {
    pub(crate) fn reset(&mut self, pipeline: This) {
        *self = State::default();
        self.pipeline = pipeline as usize;
    }

    /// The title's topology; console-only ones become host lists (see
    /// [`crate::primitive`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        self.topology_set = true;
        // Console lists the host lacks draw from expanded indices (F57).
        let host = match topology {
            RECTANGLES => 4,
            t => crate::primitive::ListExpansion::from_topology(t).map_or(t, crate::primitive::ListExpansion::host_topology),
        };
        // SAFETY: an open list.
        unsafe {
            list.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32))
        };
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: This) {
        self.pipeline = pipeline as usize;
    }

    pub(crate) fn set_heaps(&mut self, heaps: &[This]) {
        self.heaps = [0; 2];
        for &h in heaps {
            if !h.is_null() {
                let samplers = crate::descriptor::heap(h).samplers();
                self.heaps[usize::from(samplers)] = h as usize;
            }
        }
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        set(self.roots(compute), slot, Root::Table(handle));
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Vec::new()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let end = offset as usize + values.len();
            if c.len() < end {
                c.resize(end, 0);
            }
            c[offset as usize..end].copy_from_slice(values);
        }
    }

    pub(crate) fn set_vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = *v;
            }
        }
    }

    pub(crate) fn clear_vertex_buffers(&mut self, start: u32, count: u32) {
        for slot in self.vertex.iter_mut().skip(start as usize).take(count as usize) {
            *slot = VertexView::default();
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        self.index = view.copied().unwrap_or_default();
    }

    /// Render targets and depth, set on the host list at once.
    pub(crate) fn set_targets(&mut self, list: &HostList, handles: &[u64], depth: u64) {
        let Some(host) = crate::host::host() else {
            return;
        };
        #[cfg(windows)]
        {
            self.metal_targets = None;
        }
        self.target_handles.clear();
        self.target_handles.extend_from_slice(handles);
        self.depth_handle = depth;
        self.targets_set = true;
        self.targets.clear();
        let mut rtvs = Vec::new();
        for &h in handles {
            let Some(record) = crate::descriptor::rtv_record(h) else {
                once!(
                    handle = format_args!("{h:#x}"),
                    "render target without a recorded view"
                );
                continue;
            };
            let Some(texture) = crate::resource::host_resource(record.resource) else {
                // A typed null view (or a target without a host copy) still
                // takes its slot: left out, the next live target would move
                // into it and the pixel shader's outputs reach the wrong
                // targets.
                // shortcut: a view without a description or a format has no
                // null view to stand for it and is left out, as before.
                let null = record
                    .desc()
                    .filter(|d| d.Format.0 != 0 && d.ViewDimension.0 != 0);
                if let Some(desc) = null {
                    rtvs.push(host.rtv(None, Some(&desc)));
                }
                continue;
            };
            self.targets.push(record.resource as usize);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            rtvs.push(host.rtv(Some(texture), record.desc().as_ref()));
        }
        self.depth = 0;
        self.depth_state = DEPTH_WRITE;
        let dsv = crate::descriptor::dsv_record(depth).and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            self.depth = record.resource as usize;
            self.depth_state = depth_state(texture, record.depth_desc().as_ref());
            if crate::resource::depth_and_unordered(record.resource) {
                crate::resource::DEPTH_UNORDERED[1].fetch_add(1, Ordering::Relaxed);
            }
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            Some(host.dsv(texture, record.depth_desc().as_ref()))
        });
        // SAFETY: an open list; the views were just made.
        unsafe {
            list.list.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
    }

    /// Viewports, scissors, the blend factor and the stencil reference: set
    /// on the host list at once, and kept for [`State::replay`].
    pub(crate) fn set_viewports(&mut self, list: &HostList, viewports: &[D3D12_VIEWPORT]) {
        self.viewports.clear();
        self.viewports.extend_from_slice(viewports);
        crate::draw::viewports(list, viewports);
    }

    pub(crate) fn set_scissors(&mut self, list: &HostList, rects: &[RECT]) {
        self.scissors.clear();
        self.scissors.extend_from_slice(rects);
        if scissor_test() {
            // The experiment: every rectangle the title sets is widened to
            // the largest target there is, so nothing is cut by one. What
            // the listing shows (`op_text`) stays the title's own.
            let full = RECT {
                left: 0,
                top: 0,
                right: 16384,
                bottom: 16384,
            };
            crate::draw::scissors(list, &vec![full; rects.len()]);
            return;
        }
        crate::draw::scissors(list, rects);
    }

    pub(crate) fn set_blend_factor(&mut self, list: &HostList, factor: Option<&[f32; 4]>) {
        self.blend = factor.copied();
        crate::draw::blend_factor(list, factor);
    }

    pub(crate) fn set_stencil_ref(&mut self, list: &HostList, reference: u32) {
        self.stencil = Some(reference);
        crate::draw::stencil_ref(list, reference);
    }

    pub(crate) fn set_depth_bounds(&mut self, list: &HostList, min: f32, max: f32) {
        self.depth_bounds = Some([min, max]);
        crate::draw::depth_bounds(list, min, max);
    }

    /// `list` is a new host list that continues the title's list (it was
    /// cut at a wait for a memory value, `crate::list::cut`): a new host
    /// list has nothing set, so what the title set so far is set again.
    /// The pipeline, the root signatures and the heaps are set by the next
    /// draw or dispatch (`bind`, which is told here that the host list has
    /// none); root arguments, vertex and index buffers are set by every
    /// draw anyway. The dynamic descriptor slots and the tables made in
    /// them stay: the parts before the cut have not run yet, and they are
    /// given back when the title's list is reset.
    /// shortcut: the render target views are made again from the handles'
    /// descriptors as they are now, not as they were when the title set
    /// the targets (a descriptor rewritten in between names another view).
    pub(crate) fn replay(&mut self, list: &HostList) {
        self.host_pipeline = 0;
        self.host_roots = [0; 2];
        self.host_heaps = [0; 2];
        if self.topology_set {
            self.set_topology(list, self.topology);
        }
        if self.targets_set {
            let handles = std::mem::take(&mut self.target_handles);
            self.set_targets(list, &handles, self.depth_handle);
        }
        if !self.viewports.is_empty() {
            crate::draw::viewports(list, &self.viewports);
        }
        if !self.scissors.is_empty() {
            let rects = std::mem::take(&mut self.scissors);
            self.set_scissors(list, &rects);
        }
        if let Some(factor) = self.blend {
            crate::draw::blend_factor(list, Some(&factor));
        }
        if let Some(reference) = self.stencil {
            crate::draw::stencil_ref(list, reference);
        }
        if let Some([min, max]) = self.depth_bounds {
            crate::draw::depth_bounds(list, min, max);
        }
    }

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value.
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::Table(handle)), RootType::Table) => match self.heap_of(*handle) {
                Some(heap) => (handle - heap.start) as u32,
                None => {
                    once!(
                        handle = format_args!("{handle:#x}"),
                        "descriptor table outside the bound heaps"
                    );
                    0
                }
            },
            (Some(Root::Constants(c)), RootType::Constants) => {
                c.get(dword as usize).copied().unwrap_or(0)
            }
            // The host's own ([`Self::host_word`]): no root argument.
            (_, RootType::Other(_)) => 0,
            _ => {
                once!(
                    slot,
                    ?root,
                    "a shader reads a root argument the title did not set"
                );
                0
            }
        }
    }

    /// A user-data dword the host computes for a translated shader
    /// (`victory_shader::emit::USER_MEMORY_BASE`, `USER_VIEW_ADDRESS`).
    fn host_word(
        &self,
        compute: bool,
        kind: u32,
        slot: u8,
        dword: u32,
        memory: Option<&crate::memory::Ready>,
    ) -> u32 {
        use victory_shader::emit::{USER_GROUP_BASE, USER_MEMORY_BASE, USER_VIEW_ADDRESS, USER_VIEW_BIAS};
        // Set between the parts of a large dispatch (`group_base`).
        if kind == USER_GROUP_BASE || kind == victory_shader::emit::USER_DISPATCH_THREADS_X {
            return 0;
        }
        if kind == USER_MEMORY_BASE {
            return memory.map_or(0, |m| m.base);
        }
        if kind == USER_VIEW_BIAS {
            // The view begins on 16 bytes of the copy the address finds
            // (`root_uav`, `root_srv_table`).
            // shortcut: found again here, not taken from the list's slot.
            return match self.roots_ref(compute).get(usize::from(slot)) {
                Some(&Root::View(address)) => crate::shadow::find(address, 1).map_or(0, |s| {
                    // SAFETY: a live buffer.
                    let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                    ((s.gpu(address) - base) & 15) as u32
                }),
                _ => 0,
            };
        }
        if kind != USER_VIEW_ADDRESS && kind != USER_VIEW_ADDRESS + 1 {
            once!(kind, "a shader reads a host value this build does not know");
            return 0;
        }
        // The title address of the buffer view at that dword of the table.
        let address = match self.roots_ref(compute).get(usize::from(slot)) {
            Some(&Root::Table(table)) => crate::descriptor::view(table + u64::from(dword) * 4)
                .and_then(|view| crate::descriptor::host::buffer_range(&view))
                .map_or(0, |range| range.0),
            _ => 0,
        };
        if address == 0 {
            once!(slot, dword, "a shader takes the address of a buffer view that is not there: it reads 0");
        }
        if kind == USER_VIEW_ADDRESS {
            address as u32
        } else {
            (address >> 32) as u32
        }
    }

    /// For a shader that reads title memory by address
    /// ([`crate::memory`]): its directory, fault notes and view slots,
    /// with the pages of the root views the list has set made ready.
    fn memory_ready(&self, list: &HostList, compute: bool, writes: bool) -> Option<crate::memory::Ready> {
        let heap = match self.heaps[0] {
            0 => None,
            h => crate::descriptor::heap(h as This).host(),
        }?;
        let (mut addresses, mut constants) = (Vec::new(), Vec::new());
        for root in self.roots_ref(compute) {
            match root {
                Root::View(address) => addresses.push(*address),
                Root::Constants(values) => constants.extend_from_slice(values),
                _ => {}
            }
        }
        crate::memory::ready(list, heap, &addresses, &constants, writes)
    }

    /// The bound heap holding `handle`.
    fn heap_of(&self, handle: u64) -> Option<&'static crate::descriptor::Heap> {
        self.heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| h.index(handle).is_some())
    }

    /// Brings the host list up to date for a draw; false if the draw cannot
    /// run on the host.
    fn prepare(&mut self, list: &HostList, indexed: bool) -> bool {
        self.prepare_for(list, indexed, false)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    fn prepare_for(&mut self, list: &HostList, indexed: bool, mesh: bool) -> bool {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return false;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        if p.compute || p.mesh != mesh {
            once!(mesh, "a draw or mesh dispatch with another kind of pipeline");
            return false;
        }
        if p.rects && self.topology != RECTANGLES {
            // Its geometry stage would turn each triangle into a rectangle.
            once!("a rectangle pipeline drawn with another topology");
        }
        if !self.bind(list, p, false) {
            return false;
        }
        let l = &list.list;
        // Vertex buffers the input layout reads.
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        let mut count = 0;
        for (slot, (view, &v)) in views.iter_mut().zip(&self.vertex).enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            count = slot + 1;
            if v.address == 0 || v.size == 0 {
                continue;
            }
            let Some(shadow) = crate::shadow::find_any(v.address, u64::from(v.size)) else {
                continue;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            *view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                StrideInBytes: v.stride,
            };
        }
        // SAFETY: an open list; the views name live host buffers.
        unsafe {
            if count > 0 {
                l.IASetVertexBuffers(0, Some(&views[..count]));
            }
        }
        if indexed {
            let v = self.index;
            let Some(shadow) = crate::shadow::find_any(v.address, u64::from(v.size)) else {
                return false;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                Format: DXGI_FORMAT(v.format as i32),
            };
            // SAFETY: an open list; the view names a live host buffer.
            unsafe { l.IASetIndexBuffer(Some(&view)) };
        }
        const RENDER_TARGET: u32 = 0x4;
        for &t in &self.targets {
            crate::list::ensure_state(list, t as This, RENDER_TARGET);
        }
        if self.depth != 0 {
            // A depth texture shaders also write: what they wrote, first.
            list.twin_use(self.depth as This, false, true);
            // A view that writes neither plane keeps the texture readable:
            // the depth read state, together with the shader read states
            // if this draw's shaders read the same texture (the write state
            // put it out of the state those views need, and a move to it
            // and back for every such draw).
            let state = if self.depth_state == DEPTH_READ {
                let (tested, read) = TESTED_DEPTH.get();
                if read && tested == self.depth {
                    DEPTH_READ | SHADER_RESOURCE
                } else {
                    DEPTH_READ
                }
            } else {
                DEPTH_WRITE
            };
            crate::list::ensure_state(list, self.depth as This, state);
        }
        true
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    /// False if the draw or dispatch must not run: a root SRV it reads
    /// could not be given a view.
    pub(crate) fn bind(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> bool {
        let l = &list.list;
        // Textures picked by index, before anything asks for them.
        self.early_textures(list, p, compute);
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them.
        if self.host_heaps != self.heaps {
            let heaps: Vec<Option<ID3D12DescriptorHeap>> = self
                .heaps
                .iter()
                .filter(|&&h| h != 0)
                .filter_map(|&h| crate::descriptor::heap(h as This).host())
                .map(|h| Some(h.heap.clone()))
                .collect();
            heaps.iter().flatten().for_each(|h| list.keep(h));
            // SAFETY: an open list and shader-visible heaps.
            unsafe { l.SetDescriptorHeaps(&heaps) };
            self.host_heaps = self.heaps;
            self.host_roots = [0; 2];
        }
        // The depth target a draw only tests, for the views of it that the
        // draw's shaders read (`readable`).
        let tested = if !compute && self.depth_state == DEPTH_READ {
            self.depth
        } else {
            0
        };
        TESTED_DEPTH.set((tested, false));
        let root = p.root.as_raw() as usize;
        if self.host_roots[usize::from(compute)] != root {
            // SAFETY: an open list and a live root signature.
            unsafe {
                if compute {
                    l.SetComputeRootSignature(&p.root);
                } else {
                    l.SetGraphicsRootSignature(&p.root);
                }
            }
            self.host_roots[usize::from(compute)] = root;
        }
        // By host object: a title pipeline released and another made at its
        // address is another host pipeline (the list keeps the old one).
        if self.host_pipeline != p.pso.as_raw() as usize {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            self.host_pipeline = p.pso.as_raw() as usize;
            list.used_pipeline(&p.name);
            // The title may release the pipeline before it submits the list.
            list.keep(&p.pso);
            list.keep(&p.root);
        }
        for stage in &p.stages {
            let t = stage.translated();
            STORES.set(t.stores);
            let made = self.tables(list, &t.used, &t.host.root_indexed_views, compute);
            STORES.set(true);
            if made.iter().any(|m| m.1 == u32::MAX) {
                once!("a draw whose descriptor table found no free slots is dropped");
                return false;
            }
            // What the shader stores through root views and raw addresses is
            // copied back into title memory afterwards.
            let writes = (t.stores || !write_skip())
                && !crate::pipeline::builder_quarantined(self.pipeline as This);
            // Title memory by address: what the shader finds it by.
            let memory = if t.host.root_srvs.contains(&victory_shader::emit::SLOT_MEMORY) {
                let Some(ready) = self.memory_ready(list, compute, writes) else {
                    once!("a shader that reads title memory by address without a view heap to hold its views: the draw is dropped");
                    return false;
                };
                Some(ready)
            } else {
                None
            };
            if let Some((param, dwords)) = stage.user_data {
                let mut values = vec![0u32; dwords as usize];
                for u in &t.host.user_data {
                    if let RootType::Other(kind) = u.root {
                        values[u.index as usize] =
                            self.host_word(compute, kind, u.slot, u.dword, memory.as_ref());
                        continue;
                    }
                    let fresh = made
                        .iter()
                        .find(|m| m.0 == u.slot && u.root == RootType::Table);
                    values[u.index as usize] = match fresh {
                        Some(&(_, offset)) => offset,
                        None => self.user_value(compute, u.slot, u.root, u.dword),
                    };
                }
                // SAFETY: an open list; `values` covers the parameter.
                unsafe {
                    if compute {
                        l.SetComputeRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    } else {
                        l.SetGraphicsRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    }
                }
            }
            for (views, srv) in [(&stage.root_cbvs, false), (&stage.root_srvs, true)] {
                for &(slot, param) in views {
                    if srv && slot == victory_shader::emit::SLOT_MEMORY {
                        // Ours: the directory of title memory by address.
                        if let Some(m) = &memory {
                            // SAFETY: an open list; the directory lives as
                            // long as the process and the list holds it.
                            unsafe {
                                if compute {
                                    l.SetComputeRootShaderResourceView(param, m.directory);
                                } else {
                                    l.SetGraphicsRootShaderResourceView(param, m.directory);
                                }
                            }
                        }
                        continue;
                    }
                    if srv && slot == victory_shader::emit::SLOT_STRIDES {
                        // Ours, not the title's: the shader picks buffers
                        // from the view heap by an index it computes.
                        match self.heap_buffers(list) {
                            // SAFETY: an open list; the table lives as long
                            // as the heap the list holds.
                            Some(gpu) => unsafe {
                                if compute {
                                    l.SetComputeRootShaderResourceView(param, gpu);
                                } else {
                                    l.SetGraphicsRootShaderResourceView(param, gpu);
                                }
                            },
                            None => once!("a shader picks buffers from a view heap that is not set"),
                        }
                        continue;
                    }
                    let tabled = srv && crate::pipeline::host::bounded_root_srvs();
                    let Some(&Root::View(address)) = self.roots_ref(compute).get(usize::from(slot))
                    else {
                        once!(slot, "a shader reads a root view the title did not set");
                        // A table left unset is undefined: a null view.
                        if tabled && !self.root_srv_table(list, None, param, compute, false) {
                            return false;
                        }
                        continue;
                    };
                    // The shader adds the bytes a raw view begins early by.
                    let biased = srv
                        && t.host.user_data.iter().any(|u| {
                            u.slot == slot
                                && u.root == RootType::Other(victory_shader::emit::USER_VIEW_BIAS)
                        });
                    if tabled {
                        if !self.root_srv_table(list, Some(address), param, compute, biased) {
                            return false;
                        }
                        continue;
                    }
                    let Some(mut gpu) = root_view(list, address, srv) else {
                        continue;
                    };
                    if biased {
                        gpu -= u64::from(self.host_word(
                            compute,
                            victory_shader::emit::USER_VIEW_BIAS,
                            slot,
                            0,
                            None,
                        ));
                    }
                    // SAFETY: an open list; the address is in live memory.
                    unsafe {
                        match (compute, srv) {
                            (false, false) => l.SetGraphicsRootConstantBufferView(param, gpu),
                            (false, true) => l.SetGraphicsRootShaderResourceView(param, gpu),
                            (true, false) => l.SetComputeRootConstantBufferView(param, gpu),
                            (true, true) => l.SetComputeRootShaderResourceView(param, gpu),
                        }
                    }
                }
            }
            for &(slot, param) in &stage.root_uavs {
                if slot == victory_shader::emit::SLOT_DIAG {
                    // Ours: shader diagnostic counters (read only on the Metal path).
                    let gpu = d3d12_diag();
                    if gpu != 0 {
                        // SAFETY: an open list; the buffer lives as long as the process.
                        unsafe {
                            if compute {
                                l.SetComputeRootUnorderedAccessView(param, gpu);
                            } else {
                                l.SetGraphicsRootUnorderedAccessView(param, gpu);
                            }
                        }
                    }
                    continue;
                }
                if slot == victory_shader::emit::SLOT_FAULTS {
                    // Ours: where shaders note the pages they found empty.
                    if let Some(m) = &memory {
                        // SAFETY: an open list; the buffer lives as long as
                        // the process and the list holds it.
                        unsafe {
                            if compute {
                                l.SetComputeRootUnorderedAccessView(param, m.faults);
                            } else {
                                l.SetGraphicsRootUnorderedAccessView(param, m.faults);
                            }
                        }
                    }
                    continue;
                }
                let address = match self.roots_ref(compute).get(usize::from(slot)) {
                    Some(&Root::View(address)) => Some(address),
                    _ => {
                        once!(slot, "a shader writes through a root view the title did not set");
                        None
                    }
                };
                if !self.root_uav(list, address, param, compute, writes) {
                    return false;
                }
            }
        }
        // Depth textures it writes through unordered access: their colour
        // twins hold what they lack from here on.
        list.twins_written();
        true
    }

    /// Binds a root UAV: a raw unordered access view of the title buffer's
    /// host copy from `address` on, as a one-view table (a null view
    /// without an address, without a host copy shaders can write, or at an
    /// address not on 16 bytes), or as a root descriptor with
    /// `VICTORY_ROOT_SRV_DIRECT=1` (the dispatch is dropped then if there
    /// is nothing to bind). With `writes`, what the shader stored is copied
    /// back into title memory when the list has run. False if the draw
    /// must be dropped.
    /// shortcut: a root view has no size: the view, the refresh and the
    /// copy back cover all that is left of the title's buffer
    /// (`VICTORY_ROOT_UAV_MB` sets a limit for all three).
    fn root_uav(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
        writes: bool,
    ) -> bool {
        const UNORDERED: i32 = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0;
        static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let limit = *LIMIT.get_or_init(|| {
            std::env::var("VICTORY_ROOT_UAV_MB")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .map_or(u64::MAX, |mb| mb << 20)
        });
        let find = |a: u64| {
            let s = crate::shadow::find(a, 1)?;
            // SAFETY: a live buffer.
            let flags = unsafe { s.buffer.GetDesc() }.Flags;
            if flags.0 & UNORDERED == 0 {
                once!("a root UAV of a buffer whose host copy shaders cannot write");
                return None;
            }
            Some(s)
        };
        let ready = |a: u64, s: &std::sync::Arc<crate::shadow::Shadow>| {
            let len = s.left(a).min(limit);
            s.refresh(list, a, len);
            if writes {
                s.gpu_writes(list, a, len);
            }
        };
        if !crate::pipeline::host::bounded_root_srvs() {
            let Some((a, s)) = address.and_then(|a| Some((a, find(a)?))) else {
                once!("a root UAV with nothing to bind: the draw is dropped");
                return false;
            };
            ready(a, &s);
            // From the 16 bytes below: the shader adds the rest.
            // SAFETY: a live buffer.
            let base = unsafe { s.buffer.GetGPUVirtualAddress() };
            let gpu = base + ((s.gpu(a) - base) & !15);
            // SAFETY: an open list; the address is in a live buffer.
            unsafe {
                if compute {
                    list.list.SetComputeRootUnorderedAccessView(param, gpu);
                } else {
                    list.list.SetGraphicsRootUnorderedAccessView(param, gpu);
                }
            }
            return true;
        }
        let heap = match self.heaps[0] {
            0 => None,
            h => Some(crate::descriptor::heap(h as This)),
        };
        let Some((heap, host)) = heap.and_then(|h| Some((h, h.host()?))) else {
            once!("a root UAV without a bound view heap: the draw is dropped");
            return false;
        };
        let key = (std::ptr::from_ref(heap) as usize, address.unwrap_or(0));
        let (index, shadow) = match self.root_uav_slots.get(&key) {
            Some((index, shadow)) => (*index, address.zip(shadow.clone())),
            None => {
                let shadow = address.and_then(|a| find(a).map(|s| (a, s)));
                let Some(index) = host.dynamic(1) else {
                    return false;
                };
                self.dynamic.push((key.0, index, 1));
                if let Some((a, s)) = &shadow {
                    // SAFETY: a live buffer.
                    let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                    let offset = s.gpu(*a) - base;
                    // Raw views start on 16 bytes: from the 16 bytes
                    // below, and the shader adds the rest to its offsets
                    // (`victory_shader::emit::USER_VIEW_BIAS`). (An address
                    // not on 16 bytes got a null view at first: the
                    // builder's counters are at such addresses.)
                    let bias = offset & 15;
                    host.raw_uav_at(index, &s.buffer, offset - bias, s.left(*a).min(limit) + bias);
                }
                if shadow.is_none() {
                    host.null_raw_uav_at(index);
                }
                self.root_uav_slots
                    .insert(key, (index, shadow.as_ref().map(|(_, s)| s.clone())));
                (index, shadow)
            }
        };
        if let Some((a, s)) = &shadow {
            ready(*a, s);
        }
        let handle = host.gpu_handle(index);
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }

    /// Binds a root SRV as a one-view table
    /// ([`crate::pipeline::host::bounded_root_srvs`]): a raw view of the
    /// title buffer's host copy from `address` to the copy's end, or a null
    /// view (no address, no host copy, or an address not on 16 bytes). One
    /// dynamic slot per address per list, given back with the list's
    /// others. False if no slot could be had or no view heap is bound.
    fn root_srv_table(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
        biased: bool,
    ) -> bool {
        let heap = match self.heaps[0] {
            0 => None,
            h => Some(crate::descriptor::heap(h as This)),
        };
        let Some((heap, host)) = heap.and_then(|h| Some((h, h.host()?))) else {
            once!("a root SRV without a bound view heap: the draw is dropped");
            return false;
        };
        // A view that begins early is another view than one that does not
        // (title addresses stay below bit 63).
        let key = (
            std::ptr::from_ref(heap) as usize,
            address.unwrap_or(0) | u64::from(biased) << 63,
        );
        let (index, shadow) = match self.root_srv_slots.get(&key) {
            Some((index, shadow)) => (*index, address.zip(shadow.clone())),
            None => {
                let shadow = address.and_then(|a| crate::shadow::find(a, 1).map(|s| (a, s)));
                let Some(index) = host.dynamic(1) else {
                    return false;
                };
                self.dynamic.push((key.0, index, 1));
                self.root_srv_slots
                    .insert(key, (index, shadow.as_ref().map(|(_, s)| s.clone())));
                match &shadow {
                    Some((a, s)) => {
                        // SAFETY: a live buffer.
                        let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                        let offset = s.gpu(*a) - base;
                        if biased {
                            // From the 16 bytes below; the shader adds the
                            // rest (`victory_shader::emit::USER_VIEW_BIAS`).
                            let bias = offset & 15;
                            host.raw_at(index, &s.buffer, offset - bias, s.left(*a) + bias);
                        } else if offset.is_multiple_of(16) {
                            host.raw_at(index, &s.buffer, offset, s.left(*a));
                        } else {
                            // Raw views start on 16 bytes; one from the
                            // aligned start would shift every read.
                            once!(
                                address = format_args!("{a:#x}"),
                                "a root SRV not on 16 bytes reads zeros"
                            );
                            host.null_raw_at(index);
                        }
                    }
                    None => host.null_raw_at(index),
                }
                (index, shadow)
            }
        };
        // As for root descriptors (`root_view`): all that is left of the
        // buffer is brought up to date. The first 1 MiB only left what lay
        // past it stale (most mesh vertex streams reach further).
        if let Some((a, s)) = &shadow {
            s.refresh(list, *a, s.left(*a));
        }
        let handle = host.gpu_handle(index);
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }

    /// For a shader that picks buffers from the view heap by an index it
    /// computes (SM 6.6 dynamic resources): the GPU address of the heap's
    /// table of element strides, with every buffer view of the heap given
    /// its host slot and its memory brought up to date when `list` is
    /// submitted. Tables tell us which views a draw reads; nothing tells
    /// us which of the heap's a computed index names, so all of them are
    /// made ready, once per list.
    /// shortcut: the buffers are read, not written: what such a shader
    /// writes through a picked unordered access view stays in the host
    /// copy. A list refreshes them where it first runs such a shader;
    /// title writes recorded later in the same list are not seen.
    fn heap_buffers(&self, list: &HostList) -> Option<u64> {
        let heap = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())?;
        let host = heap.host()?;
        let gpu = host.strides_gpu()?;
        // One key per list, as textures have: no title resource is there.
        const ONCE: usize = usize::MAX - 0x5e;
        if !list.first_use(ONCE) {
            return Some(gpu);
        }
        // The list may outlive the title's heap object, which owns the table.
        if let Some(buffer) = host.strides_buffer() {
            list.keep(buffer);
        }
        // The pass itself once per frame: per list it ran up to 51 times a
        // frame over 81,000 views and 4,100 host copies, 18 s of every
        // frame (2026-11-24, run 11).
        // shortcut: what the title writes into such buffers between lists
        // of one frame is seen a frame late.
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        let frame = crate::frame::current();
        if FRAME.swap(frame, Ordering::Relaxed) == frame {
            return Some(gpu);
        }
        let started = std::time::Instant::now();
        // Check changed slots immediately and revalidate one bucket each
        // pass so stale views go away within 16 passes without a full scan.
        use crate::descriptor::host::{BUFFER_BUCKETS, buffer_bucket};
        let mut kept = host.buffer_cache.lock().unwrap_or_else(|e| e.into_inner());
        let cache = kept.get_or_insert_with(Default::default);
        let rolling = cache.passes.checked_sub(1).map(|pass| pass % BUFFER_BUCKETS);
        let mut slots = host.buffer_changes();
        slots.extend(host.buffer_slots(rolling));
        slots.sort_unstable();
        slots.dedup();
        cache.passes += 1;
        let mut dirty = [false; BUFFER_BUCKETS];
        // Whether the device was there before the pass: a view that removes
        // it is named below.
        static GONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let mut alive = !GONE.load(Ordering::Relaxed) && crate::descriptor::host::device_removed().is_none();
        for (n, &index) in slots.iter().enumerate() {
            let bucket = buffer_bucket(index);
            let ranges = &mut cache.buckets[bucket].slots;
            let view = heap.handle(index).and_then(|handle| host.mirror(index, handle, true));
            // Every 64 views while the device lives (a cheap query).
            if alive
                && n % 64 == 63
                && let Some(why) = crate::descriptor::host::device_removed()
            {
                alive = false;
                GONE.store(true, Ordering::Relaxed);
                let last: Vec<String> = slots[n - 63..=n]
                    .iter()
                    .filter_map(|&i| crate::descriptor::view(heap.handle(i)?).map(|v| format!("{i}: {v:?}")))
                    .map(|s| s.chars().take(200).collect())
                    .collect();
                tracing::error!(why, slot = index, "the host device went while buffer views of the heap were made; one of: {last:#?}");
                crate::host::log_removal_context();
            }
            // The copy the slot shows (`buffer_copy`), not the one the
            // view's range alone would find.
            let Some((shadow, address, len)) = view.as_ref().and_then(crate::descriptor::host::buffer_copy) else {
                dirty[bucket] |= ranges.remove(&index).is_some();
                continue;
            };
            #[cfg(feature = "diagnostics")]
            if host.shows(index, &shadow.buffer) == Some(false) {
                HEAP_VIEWS_ELSEWHERE.fetch_add(1, Ordering::Relaxed);
            }
            let end = address + len;
            if ranges.get(&index).is_none_or(|old| {
                !std::sync::Arc::ptr_eq(&old.0, &shadow) || old.1 != address || old.2 != end
            }) {
                ranges.insert(index, (shadow, address, end));
                dirty[bucket] = true;
            }
        }
        let mut views = 0;
        let mut spans = HeapBufferSpans::default();
        for (bucket, dirty) in cache.buckets.iter_mut().zip(dirty) {
            views += bucket.slots.len() as u32;
            if dirty {
                bucket.spans.clear();
                for range in bucket.slots.values() {
                    cover_buffer_span(&mut bucket.spans, range);
                }
            }
            for range in bucket.spans.values() {
                cover_buffer_span(&mut spans, range);
            }
        }
        drop(kept);
        let bytes: u64 = spans.values().map(|s| s.2 - s.1).sum();
        // Memory that is not write-watched is copied whole by every refresh.
        // It was once refreshed only every 120 frames (unmapped pages read
        // one by one cost 14 GB a frame in run 11, before they were
        // remembered): streamed geometry then drew from stale buffers, and
        // the garbage ended runs (the card stopped answering 14-90 frames
        // into the opening cutscene; with every frame it ran on). The pass
        // costs some 30 ms a frame this way.
        static WHOLE: std::sync::Mutex<Option<std::collections::HashMap<usize, u64>>> = std::sync::Mutex::new(None);
        let mut whole = WHOLE.lock().unwrap_or_else(|e| e.into_inner());
        let whole = whole.get_or_insert_with(Default::default);
        for (shadow, from, to) in spans.values() {
            if shadow.unwatched() {
                // By the copy's number: a copy made where one let go was
                // (`shadow::let_go_idle`) has never been refreshed.
                let key = shadow.id() as usize;
                // `VICTORY_UNWATCHED_FRAMES` sets the interval (1: every frame).
                static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
                let every = *EVERY.get_or_init(|| {
                    std::env::var("VICTORY_UNWATCHED_FRAMES")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(1)
                });
                match whole.get(&key) {
                    Some(&last) if frame < last + every => continue,
                    _ => {
                        whole.insert(key, frame);
                    }
                }
            }
            shadow.refresh(list, *from, to - from);
        }
        self.heap_textures(list, heap, host);
        HEAP_BUFFER_PASSES.fetch_add(1, Ordering::Relaxed);
        HEAP_BUFFER_MICROS.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        static LOGGED: AtomicU64 = AtomicU64::new(0);
        // The first pass and every growth of the set by half.
        let seen = LOGGED.load(Ordering::Relaxed);
        if u64::from(views) > seen + seen / 2 {
            LOGGED.store(u64::from(views), Ordering::Relaxed);
            tracing::info!(
                slots = slots.len(),
                views,
                copies = spans.len(),
                mb = bytes >> 20,
                ms = started.elapsed().as_millis(),
                "buffer views of the heap made ready for shaders that pick them by index"
            );
        }
        Some(gpu)
    }

    /// Runs the pass that uploads textures picked by index
    /// ([`State::heap_textures`]) where nothing has asked for it yet
    /// ([`early_budgets`]): once a frame for a few milliseconds, at the
    /// frame's first draw with a view heap set, and with a larger budget
    /// of its own before every draw whose pixel shader is one of the
    /// title's bake programs. The uploads are recorded into `list` ahead
    /// of the draw being bound, as the world's pass records them.
    fn early_textures(&self, list: &HostList, p: &HostPipeline, compute: bool) {
        let (early_ms, bake_ms) = early_budgets();
        if compute || (early_ms == 0 && bake_ms == 0) {
            return;
        }
        let bake = bake_ms > 0
            && p.stages.iter().any(|s| BAKE_PIXEL_SHADERS.contains(&s.shader.hash.as_str()));
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        // The milliseconds this frame's bake draws have taken so far: a
        // frame has several, and together they get `bake_ms`.
        static BAKED_MS: AtomicU64 = AtomicU64::new(0);
        let frame = crate::frame::current();
        let first = FRAME.swap(frame, Ordering::Relaxed) != frame;
        if first {
            BAKED_MS.store(0, Ordering::Relaxed);
        }
        let left = bake_ms.saturating_sub(BAKED_MS.load(Ordering::Relaxed));
        let budget = if bake && left > 0 {
            // What a bake may read must be there now: up to 2 GiB.
            (left, 2 << 30)
        } else if first && early_ms > 0 {
            (early_ms, 64 << 20)
        } else {
            return;
        };
        let Some(heap) = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())
        else {
            return;
        };
        let Some(host) = heap.host() else {
            return;
        };
        let bake = bake && left > 0;
        let n = EARLY_TEXTURE_PASSES[usize::from(bake)].fetch_add(1, Ordering::Relaxed) + 1;
        if bake && n.is_power_of_two() {
            let [slots, due, ..] = HEAP_TEXTURES_QUEUE.each_ref().map(|c| c.load(Ordering::Relaxed));
            tracing::info!(
                frame,
                bakes = n,
                queued_at_the_last_pass = slots,
                due_at_the_last_pass = due,
                uploaded_so_far = HEAP_TEXTURES.load(Ordering::Relaxed),
                "a draw of the title's decoration bake: textures picked by index are uploaded first (VICTORY_BAKE_TEXTURES_MS=0: not)"
            );
        }
        let began = std::time::Instant::now();
        PASS_BUDGET.set(Some(budget));
        self.heap_textures(list, heap, host);
        PASS_BUDGET.set(None);
        if bake {
            BAKED_MS.fetch_add(began.elapsed().as_millis() as u64, Ordering::Relaxed);
        }
    }

    /// The texture side of [`State::heap_buffers`]: textures the title fills
    /// itself (streamed pages, layout-library copies) whose views sit in the
    /// view heap are created (placed textures have no host texture until
    /// something reads them) and uploaded, as a table read would do, for
    /// shaders that pick them by a computed index. Within a budget of bytes
    /// per frame and in all; the rest waits for later lists and frames.
    /// The most recently written slots first (`TextureQueue::due`); slots
    /// draws read through tables are not in the queue.
    /// shortcut: uploaded once, and again only when the title rewrites the
    /// slot. A placed texture not all mapped yet is uploaded with what is
    /// there and looked at again (1, 2, 4, ... frames later, at most about
    /// a second): uploaded again when more of it is mapped. Pages mapped
    /// in place of others, and contents changed with no descriptor write,
    /// are missed (hook the page-mapping events for those).
    /// `VICTORY_HEAP_TEXTURES=0` turns it off.
    fn heap_textures(
        &self,
        list: &HostList,
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
    ) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_TEXTURES").is_none_or(|v| v != "0")) {
            return;
        }
        /// Bytes of texture memory staged per frame.
        // `VICTORY_TEXTURE_STAGE_MB` raises it: a backlog of tens of thousands of
        // textures (a scene's first minutes) then drains in fewer frames.
        static PER_FRAME_MB: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let per_frame: u64 = *PER_FRAME_MB.get_or_init(|| {
            std::env::var("VICTORY_TEXTURE_STAGE_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(192)
        }) << 20;
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        static SPENT: AtomicU64 = AtomicU64::new(0);
        let frame = crate::frame::current();
        if FRAME.swap(frame, Ordering::Relaxed) != frame {
            SPENT.store(0, Ordering::Relaxed);
        }
        // No more of them are made while the card has no room
        // (`resource::room`): by what is alive now, not by what was ever
        // staged (4 GB in all, after which later textures stayed empty for
        // the run). The ones nothing names any more are let go, which
        // makes room again: so the slots are still looked at (the pass
        // returned here, and a slot whose texture had been streamed out
        // kept its host copy for as long as the card was full).
        let mut room = crate::resource::room();
        let mut since_room_check = 0u64;
        if !room {
            HEAP_TEXTURES_FULL.fetch_add(1, Ordering::Relaxed);
            once!("textures picked by index: the card has no room for more; they wait");
        }
        static REUPLOAD_FULL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let reupload_full =
            *REUPLOAD_FULL.get_or_init(|| std::env::var_os("VICTORY_REUPLOAD_WHEN_FULL").is_none_or(|v| v != "0"));
        let mut refused = 0;
        // Slots whose texture's memory the runtime mapped, unmapped or
        // wrote since the last pass are due again.
        host.texture_slots_remapped(frame);
        host.texture_sweep(heap, frame);
        if heap_recheck() {
            let [same, changed, waiting] = host.texture_slots_rechecked(frame);
            HEAP_TEXTURES_RECHECKED[0].fetch_add(same, Ordering::Relaxed);
            HEAP_TEXTURES_RECHECKED[1].fetch_add(changed, Ordering::Relaxed);
            HEAP_TEXTURES_RECHECKED[2].store(waiting, Ordering::Relaxed);
        }
        let (due, census) = host.texture_slots_due(frame, 4096);
        HEAP_TEXTURES_WAITING.store(census[0], Ordering::Relaxed);
        for (count, now) in HEAP_TEXTURES_QUEUE.iter().zip(census) {
            count.store(now, Ordering::Relaxed);
        }
        // The frames in about four seconds, between 16 and 256: how long a
        // slot with nothing to upload waits at most for its next look. In
        // frames alone it was 120: four minutes at the opening cutscene's
        // half a frame per second, while the title streams by the clock.
        // (About a second before the runtime told of the memory it maps:
        // the looks are now only for what that does not tell.)
        static PASSED: AtomicU64 = AtomicU64::new(0);
        let now = crate::host::now_ms();
        let took = now.saturating_sub(PASSED.swap(now, Ordering::Relaxed)).max(1);
        let most = (4000 / took).clamp(16, 256);
        // What a table entry of a texture a shader samples says.
        let sampled = UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        };
        // Also within a time per pass: the pass runs on a thread that
        // records the frame, and making and staging 192 MB of textures took
        // 1.6-2.2 s of each of the world's first twenty frames (run 20).
        // The rest waits for the next frames, as it does past the bytes.
        static SPAN_MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let span = std::time::Duration::from_millis(*SPAN_MS.get_or_init(|| {
            std::env::var("VICTORY_TEXTURE_STAGE_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(250)
        }));
        // A backlog (a scene's first minutes: 22,000 slots due, 2-3 minutes of black
        // surfaces at 192 MB a frame) gets three times the bytes and the time until it
        // drains; `VICTORY_TEXTURE_CATCHUP=0` keeps the plain budget.
        static CATCHUP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let catchup = *CATCHUP.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_CATCHUP").is_none_or(|v| v != "0"));
        let (per_frame, span) = if catchup && due.len() >= 1024 { (per_frame * 3, span * 3) } else { (per_frame, span) };
        // A run made early (`State::early_textures`) has a budget of its
        // own: its bytes on top of what the frame has spent so far.
        // (Not `most`: that is the longest wait between looks, used below.)
        let (span, bytes_at_most) = match PASS_BUDGET.get() {
            Some((ms, bytes)) => {
                (std::time::Duration::from_millis(ms), SPENT.load(Ordering::Relaxed).saturating_add(bytes))
            }
            None => (span, per_frame),
        };
        let began = std::time::Instant::now();
        for (index, write) in due {
            if SPENT.load(Ordering::Relaxed) >= bytes_at_most || began.elapsed() > span {
                break;
            }
            // A placed texture whose memory is not mapped (not streamed in
            // yet, or streamed out again: the heap keeps tens of thousands
            // of such views) gets no host texture; looked at again soon (a
            // title writes a texture's view before its pages are mapped,
            // and the pass runs early in a frame). One of which some is
            // mapped (a texture streaming in: its smaller mips lie at the
            // start of its memory and arrive first) is uploaded with what
            // is there, and again when more is. (It was uploaded again
            // every 120 frames, changed or not; and asked by its first
            // page alone, one whose first page is the one missing counted
            // as not there.)
            let mut partly = None;
            if let Some(View::Placed(false, resource, desc)) = heap.handle(index).and_then(crate::descriptor::view) {
                let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                // Of the mips the view shows: what lies behind them is
                // another texture's.
                let len = crate::resource::placed_bytes_from(&resource, crate::resource::first_mip(&desc)).unwrap_or(1);
                let mapped = crate::texture::mapped_bytes(address as *const u8, len as usize);
                if mapped == 0 {
                    HEAP_TEXTURES_AWAY.fetch_add(1, Ordering::Relaxed);
                    // Streamed out: what the slot showed is let go.
                    host.texture_slot_gone(index);
                    host.texture_slot_later(index, write, frame, most);
                    continue;
                }
                if mapped < len {
                    if mapped <= host.texture_slot_mapped(index) {
                        HEAP_TEXTURES_PARTLY[0].fetch_add(1, Ordering::Relaxed);
                        host.texture_slot_later(index, write, frame, most);
                        continue;
                    }
                    if !crate::texture::committed(address as *const u8, 1) {
                        HEAP_TEXTURES_PARTLY[2].fetch_add(1, Ordering::Relaxed);
                    }
                    partly = Some(mapped);
                }
            }
            // A texture object the host renders to or writes holds the
            // host's own contents: nothing to bring up to date, and this
            // pass is no use of it (its host copy is made when something
            // uses it, and its slots then: `descriptor::make_waiting`).
            if let Some(View::Shader(object, _)) = heap.handle(index).and_then(crate::descriptor::view)
                && crate::resource::buffer_bytes(object as This).is_none()
                && !crate::resource::title_fills(object as This)
            {
                host.texture_slot_done(index, write, None);
                continue;
            }
            let handle = heap.handle(index);
            let view = if room {
                handle.and_then(|h| host.mirror(index, h, true))
            } else {
                // Leave new textures due, but allow a refresh of an
                // already mirrored view without mirroring a second time.
                if let Some(view) = handle.filter(|_| reupload_full).and_then(|h| host.existing_texture_view(index, h)) {
                    Some(view)
                } else {
                    refused += 1;
                    if refused > 256 {
                        break;
                    }
                    continue;
                }
            };
            let placed = view.as_ref().and_then(crate::descriptor::host::placed_texture);
            let resource = match placed.as_ref().or(view.as_ref()) {
                Some(&View::Shader(resource, _)) => resource as This,
                _ => {
                    host.texture_slot_done(index, write, None);
                    continue;
                }
            };
            let title_fills = crate::resource::host_resource(resource).is_some_and(crate::texture::title_written);
            if !title_fills {
                // a render target, a depth or unordered access texture: the host's own
                host.texture_slot_done(index, write, None);
                continue;
            }
            // A texture that has a staging buffer now is one draws read
            // through a table: they upload it, and check it at every
            // submission that samples it.
            let tabled = crate::texture::is_staged(resource);
            if tabled {
                crate::texture::upload_again(resource);
            }
            if let Some(view) = &view {
                readable(list, sampled.kind, view);
            }
            // A texture view the host refuses removes the device: name it.
            static GONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !GONE.load(Ordering::Relaxed)
                && let Some(why) = crate::descriptor::host::device_removed()
            {
                GONE.store(true, Ordering::Relaxed);
                let text: String = format!("{view:?}").chars().take(300).collect();
                tracing::error!(why, slot = index, "the host device went while textures picked by index were made ready; last: {text}");
                crate::host::log_removal_context();
                return;
            }
            let (bytes, whole) = crate::texture::staged(resource).unwrap_or((0, true));
            // Not all of it there: looked at again next frame, and from
            // then on as a slot with nothing new (a placed texture: until
            // more of it is mapped; a texture object at an address the
            // title mapped: again in 120 frames, as before).
            let again = match partly {
                Some(mapped) => Some((frame + 1, mapped)),
                None => (!whole).then_some((frame + 120, 0)),
            };
            if again.is_some() {
                HEAP_TEXTURES_PARTLY[1].fetch_add(1, Ordering::Relaxed);
            }
            if tabled {
                // Nothing was staged here, so nothing is counted or spent,
                // and the staging buffer stays: taking it away made the
                // next draw that reads the texture through a table stage,
                // detile and upload all of it once more.
                HEAP_TEXTURES_TABLED.fetch_add(1, Ordering::Relaxed);
                host.texture_slot_done(index, write, again);
                continue;
            }
            // Uploaded: the list holds the staging buffer until it has run;
            // a table read, or the next look at a texture not all mapped
            // yet, makes it anew. (Kept for those until now: every partly
            // streamed texture of the heap held a staging buffer of its full
            // size for good.)
            let memory = if again.is_none() && heap_recheck() {
                crate::texture::staged_memory(resource)
            } else {
                None
            };
            crate::texture::unstage(resource);
            SPENT.fetch_add(bytes, Ordering::Relaxed);
            since_room_check += bytes;
            if room && since_room_check >= 32 << 20 {
                since_room_check = 0;
                room = crate::resource::room();
            }
            HEAP_TEXTURE_BYTES.fetch_add(bytes, Ordering::Relaxed);
            HEAP_TEXTURES.fetch_add(1, Ordering::Relaxed);
            host.texture_slot_done(index, write, again);
            // Whole and uploaded: looked at again (`texture_slots_rechecked`).
            if let Some((at, len)) = memory {
                host.texture_slot_recheck(index, frame, (at, len), crate::texture::fingerprint(at, len));
            }
        }
    }

    /// For the frame listing, after a dropped dispatch's line: its group
    /// counts and the compute root arguments it was recorded with (root
    /// constants as hex dwords, root views with the title buffer they
    /// point into). A program that does not translate has no bindings to
    /// list; these say which memory it would have read and written (the
    /// ray-tracing structure builder passes its buffers this way).
    #[cfg(feature = "diagnostics")]
    fn list_dropped_roots(&self, groups: [u32; 3]) {
        crate::capture::op(0, || {
            use std::fmt::Write;
            let [x, y, z] = groups;
            let mut text = format!("  its roots ({x}x{y}x{z}):");
            for (slot, root) in self.compute.iter().enumerate() {
                match root {
                    Root::None => {}
                    Root::Table(a) => {
                        let _ = write!(text, " {slot}=table {a:#x}");
                    }
                    Root::Constants(v) => {
                        let _ = write!(text, " {slot}=constants {:x?}", &v[..v.len().min(48)]);
                    }
                    Root::View(a) => {
                        let _ = write!(text, " {slot}=view {a:#x}");
                        match crate::shadow::extent(*a) {
                            Some((start, len)) => {
                                let _ = write!(text, " in buffer {start:#x}+{len:#x}");
                            }
                            None => text.push_str(" in no buffer"),
                        }
                    }
                }
            }
            text
        });
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32, flags: u32) {
        // It may write the arguments a window holds clamped.
        self.window = None;
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        }
        // The ray-tracing structure builder's stages run all or none.
        if crate::pipeline::is_builder(self.pipeline as This) && !crate::pipeline::builder_runs() {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            {
                crate::pipeline::note_why("a stage of the ray-tracing structure builder, which is off".to_owned());
                self.list_dropped_roots([x, y, z]);
            }
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            {
                crate::pipeline::note_drop(self.pipeline as This);
                self.list_dropped_roots([x, y, z]);
            }
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            return;
        }
        let stage = p.stages[0].translated();
        let kind = victory_shader::emit::USER_DISPATCH_THREADS_X;
        let tail_word = compute_word(p, kind);
        let has_guard = stage.host.user_data.iter().any(|u| u.root == RootType::Other(kind));
        let metal = crate::backend::metal::mode() == crate::backend::metal::Mode::Compute;
        let Some(([x, y, z], limit)) = crate::dispatch_dims::plan(
            [x, y, z], stage.threads, flags, if metal { has_guard } else { tail_word.is_some() },
        ) else {
            once!(pipeline = %p.name, "unsupported DispatchX thread dimensions");
            DROPPED.fetch_add(1, Ordering::Relaxed);
            if crate::pipeline::is_builder(self.pipeline as This) {
                crate::pipeline::builder_off("a DispatchX thread tail could not be represented");
            }
            return;
        };
        if [x, y, z].contains(&0) {
            return;
        }
        if metal {
            if self.metal_dispatch(p, [x, y, z], limit) {
                DISPATCHED.fetch_add(1, Ordering::Relaxed);
            } else {
                DROPPED.fetch_add(1, Ordering::Relaxed);
            }
            return;
        }
        if !self.bind(list, p, true) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            // A stage of the structure builder that cannot be bound is a
            // dropped stage: the others stop too.
            if crate::pipeline::is_builder(self.pipeline as This) {
                crate::pipeline::builder_off("one of its dispatches could not be bound");
            }
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        if limit != 0 {
            let (param, word) = tail_word.expect("supported guarded dispatch");
            // SAFETY: bind set this pipeline's root signature on the open list.
            unsafe { list.list.SetComputeRoot32BitConstant(param, limit, word) };
        }
        match group_base(p) {
            Some((param, dword)) if x > crate::indirect::PART => {
                // More than the host runs in one dimension: in parts.
                // SAFETY: an open list with everything set; the parameter
                // and dword are the pipeline's.
                unsafe {
                    let mut base = 0;
                    while base < x {
                        list.list.SetComputeRoot32BitConstant(param, base, dword);
                        list.list.Dispatch((x - base).min(crate::indirect::PART), y, z);
                        base += crate::indirect::PART;
                    }
                    list.list.SetComputeRoot32BitConstant(param, 0, dword);
                }
            }
            // SAFETY: an open list with everything set.
            _ => unsafe { list.list.Dispatch(x, y, z) },
        }
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
    }

    /// What a draw or dispatch runs, for the device-removed report
    /// (diagnostics): the pipeline, then per stage the user data it gets
    /// (with the view at that heap index, if one is), the host addresses
    /// of its root views and the views its tables name.
    /// A compute dispatch on Metal (`VICTORY_METAL=compute`): the title's
    /// pipeline as a Metal pipeline, its user data and root views in the
    /// top-level argument buffer, the slots its shader reads refreshed in
    /// the Metal view heap, title memory reached in place. Recorded into
    /// the list ([`State::metal`]) and run at its submission. False if it
    /// cannot run.
    #[cfg(windows)]
    fn metal_dispatch(&mut self, p: &crate::pipeline::host::HostPipeline, groups: [u32; 3], limit: u32) -> bool {
        #[cfg(feature = "diagnostics")]
        if crate::backend::metal_draw::trace_scene() && crate::backend::metal_draw::trace_pipeline(&p.name) {
            tracing::info!(frame=crate::frame::current(), detail=%self.op_text(format!("dispatch {groups:?}"), true), "scene request");
        }
        use crate::backend::metal::backend;
        use crate::backend::metal::{Drop, count};
        let Some(m) = backend() else {
            count(Drop::NoBackend);
            return false;
        };
        let Some(mp) = m.compute_pipeline(self.pipeline, p) else {
            count(Drop::NoPipeline);
            return false;
        };
        let _pool = victory_metal::Pool::new();
        let bound = match self.metal_stage(m, p, &p.stages[0], true, mp.top_level.clone(), limit, [groups[0], groups[1]]) {
            Ok(b) => b,
            Err(d) => {
                count(d);
                return false;
            }
        };
        count(bound.outcome);
        let loop_counts = if crate::backend::metal::want_loop_counts(&p.name) {
            let counts = match self.roots_ref(true).as_slice() {
                [Root::Table(table), _, Root::Constants(values), ..] =>
                    table.checked_add(0x500)
                        .filter(|&handle| self.heap_of(handle).is_some())
                        .zip(values.get(1))
                        .and_then(|(handle, &base)| m.a1f6_loop_counts(handle, base)),
                _ => None,
            };
            if counts.is_none() {
                once!(pipeline = %p.name, "Metal: no mappable descriptor for dispatch loop count inputs");
            }
            counts
        } else { None };
        self.metal.push(std::sync::Arc::new(crate::backend::metal::MetalOp::Dispatch(crate::backend::metal::MetalCmd {
            pso: mp,
            heap: bound.heap,
            sampler_heap: bound.sampler_heap,
            top: bound.top,
            groups: groups.map(u64::from),
            tables: bound.tables,
            loop_counts,
            after_barrier: !std::mem::replace(&mut self.metal_since_barrier, true),
        })));
        true
    }

    /// An indirect compute dispatch on Metal, encoded via `IndirectTranslator`.
    #[cfg(windows)]
    fn metal_dispatch_indirect(&mut self, p: &crate::pipeline::host::HostPipeline, records: u64, stride: u32, max: u32, count: Option<u64>) -> bool {
        use crate::backend::metal::{Drop, MetalDispatchIndirect, MetalOp, backend, count as count_outcome};
        let Some(m) = backend() else {
            count_outcome(Drop::NoBackend);
            return false;
        };
        let Some(mp) = m.compute_pipeline(self.pipeline, p) else {
            count_outcome(Drop::NoPipeline);
            return false;
        };
        let bytes = u64::from(max.saturating_sub(1)) * u64::from(stride) + 12;
        let (rb, ro, count_buf) = {
            let mut chunks = m.chunks();
            let Some((rb, ro)) = chunks.window_buffer(records, bytes) else {
                count_outcome(Drop::SlotUnmapped);
                return false;
            };
            let cb = match count {
                Some(at) => match chunks.window_buffer(at, 4) {
                    Some(c) => Some(c),
                    None => {
                        count_outcome(Drop::SlotUnmapped);
                        return false;
                    }
                },
                None => None,
            };
            (rb, ro, cb)
        };
        let _pool = victory_metal::Pool::new();
        let bound = match self.metal_stage(m, p, &p.stages[0], true, mp.top_level.clone(), 0, [1, 1]) {
            Ok(b) => b,
            Err(d) => {
                count_outcome(d);
                return false;
            }
        };
        count_outcome(bound.outcome);
        self.metal.push(std::sync::Arc::new(MetalOp::DispatchIndirect(MetalDispatchIndirect {
            pso: mp,
            heap: bound.heap,
            sampler_heap: bound.sampler_heap,
            top: bound.top,
            tables: bound.tables,
            records: rb,
            records_offset: ro,
            stride,
            max,
            count: count_buf,
            records_va: records,
            count_va: count,
        })));
        true
    }

    /// A draw on Metal too (`VICTORY_METAL_DRAW=1`): recorded into the
    /// list as [`crate::backend::metal::MetalOp::Draw`], counted.
    #[cfg(windows)]
    /// The final-draw dossier's view of the heap slots the stages just bound
    /// read (index, handle, why it has no Metal entry), when capturing.
    #[cfg(windows)]
    fn note_slot(index: u32, handle: u64, missing: Option<crate::backend::metal_heap::Missing>) {
        if crate::backend::metal_draw::capture_every() == 0 && !crate::backend::metal_draw::trace_scene() {
            return;
        }
        SLOTS.with(|s| s.borrow_mut().push((index, handle, missing)));
    }

    /// Notes, for the captures, a draw at the bound colour targets
    /// (`kind`: an index into `metal_draw::WRITE_NAMES`).
    #[cfg(windows)]
    fn note_asked(&self, kind: usize) {
        if crate::backend::metal_draw::capture_every() > 0 {
            for &h in &self.target_handles {
                if let Some(r) = crate::descriptor::rtv_record(h) {
                    crate::backend::metal_draw::note_write(r.resource as usize, kind);
                }
            }
        }
    }

    fn metal_draw(&mut self, kind: crate::backend::metal_draw::DrawKindIn) {
        use crate::backend::metal_draw::{DrawDrop, DrawKind, DrawKindIn, MetalDraw, StageBind, count_draw};
        let Some(m) = crate::backend::metal::backend() else { return };
        self.note_asked(6);
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            self.note_asked(7);
            return;
        };
        #[cfg(feature = "diagnostics")]
        if crate::backend::metal_draw::trace_scene() && crate::backend::metal_draw::trace_pipeline(&p.name) {
            tracing::info!(frame=crate::frame::current(), topology=self.topology, detail=%self.op_text("draw".into(), false), "scene request");
        }
        // `VICTORY_METAL_RECTS=triangles` (experiment): rectangles drawn as
        // their first triangle, without MSC's geometry emulation.
        static RECTS_AS_TRIANGLES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let as_triangles = *RECTS_AS_TRIANGLES.get_or_init(|| std::env::var("VICTORY_METAL_RECTS").is_ok_and(|v| v == "triangles"));
        let rects = self.topology == RECTANGLES && !p.mesh && !matches!(kind, DrawKindIn::Mesh { .. }) && !as_triangles;
        let Some(g) = m.graphics_pipeline(self.pipeline, p, rects) else {
            count_draw(DrawDrop::NoPipeline);
            return;
        };
        let _pool = victory_metal::Pool::new();
        use crate::backend::metal_draw::ConsoleTopology;
        let console = match self.topology {
            TRIANGLE_FAN => Some(ConsoleTopology::TriangleFan),
            LINE_LOOP => Some(ConsoleTopology::LineLoop),
            QUADS => Some(ConsoleTopology::Quads),
            QUAD_STRIP => Some(ConsoleTopology::QuadStrip),
            _ => None,
        }
        .filter(|_| !g.mesh && g.geometry.is_none());
        let primitive = if g.mesh || g.geometry.is_some() {
            3
        } else if let Some(c) = console {
            c.primitive()
        } else if as_triangles && self.topology == RECTANGLES {
            3 // MTLPrimitiveTypeTriangle
        } else {
            match crate::backend::metal_map::map_topology(self.topology as i32) {
                Some(t) => t.primitive_type as u64 as u8,
                None => {
                    crate::backend::metal_draw::note_once(format!("topology unmapped: topology {} mesh {} geometry {}", self.topology, g.mesh, g.geometry.is_some()));
                    count_draw(DrawDrop::Topology);
                    return;
                }
            }
        };
        let targets = match &self.metal_targets {
            Some(t) => t.clone(),
            None => {
                let t = std::sync::Arc::new(self.make_metal_targets(m));
                self.metal_targets = Some(t.clone());
                t
            }
        };
        if targets.width == 0 || !targets.fit(&g) {
            if targets.width != 0 {
                crate::backend::metal_draw::note_mismatch(&targets, &g);
            } else {
                crate::backend::metal_draw::note_once(format!(
                    "no attachments: {} target handles, depth handle {}, {}",
                    self.target_handles.len(),
                    self.depth_handle != 0,
                    self.target_reasons(m)
                ));
            }
            count_draw(DrawDrop::Targets);
            return;
        }
        let vertex = if g.vertex_slots == 0 {
            None
        } else {
            use victory_metal::msc::{VERTEX_TABLE_SLOTS, VertexBufferEntry, vertex_table_bytes};
            let mut entries = [VertexBufferEntry::default(); VERTEX_TABLE_SLOTS];
            let mut buffers = Vec::new();
            let mut chunks = m.chunks();
            for (i, entry) in entries.iter_mut().enumerate() {
                if g.vertex_slots & (1u32 << i) == 0 {
                    continue;
                }
                let view = self.vertex[i];
                // An unbound slot reads zeros on D3D12: an empty entry. A
                // stride of 0 (every vertex the same element) is legal.
                if view.address == 0 || view.size == 0 {
                    continue;
                }
                // Wrapped on first use, as index buffers are (a resident
                // window alone refused every buffer not wrapped yet).
                let Some((buffer, offset)) = chunks.window_buffer(view.address, u64::from(view.size)) else {
                    crate::backend::metal_draw::note_once(format!(
                        "vertex fetch: slot {i} {:#x}+{:#x} stride {} not wrappable",
                        view.address, view.size, view.stride
                    ));
                    count_draw(DrawDrop::VertexFetch);
                    return;
                };
                *entry = VertexBufferEntry { address: buffer.gpu_address + offset, length: view.size, stride: view.stride };
                buffers.push(buffer);
            }
            Some(Box::new(crate::backend::metal_draw::VertexFetch { table: vertex_table_bytes(&entries), _buffers: buffers }))
        };
        SLOTS.with(|s| s.borrow_mut().clear());
        let mut stages = Vec::with_capacity(p.stages.len());
        for (stage, (top, bind)) in p.stages.iter().zip(&g.stages) {
            match self.metal_stage(m, p, stage, false, top.clone(), 0, [1, 1]) {
                Ok(b) => stages.push(StageBind { bind: *bind, top: b.top, tables: b.tables, heap: b.heap, sampler_heap: b.sampler_heap }),
                Err(_) => {
                    count_draw(DrawDrop::Bind);
                    return;
                }
            }
        }
        // Console topologies: list indices made here, from the draw's
        // vertex ids or from the title's index buffer (read in place).
        let kind = if let Some(c) = console {
            match self.console_draw(c, kind) {
                Some(k) => k,
                None => {
                    count_draw(DrawDrop::IndexBuffer);
                    return;
                }
            }
        } else if let Some(config) = g.geometry
            && !(matches!(kind, DrawKindIn::Indirect { indexed: false, .. }) && {
                static ENABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                *ENABLE.get_or_init(|| std::env::var("VICTORY_METAL_INDIRECT_RECTS").as_deref() == Ok("1"))
            })
        {
            match kind {
                DrawKindIn::Vertices { count, instances, first, first_instance } => {
                    let Some(draw) = config.draw(count, instances, first, first_instance, None) else {
                        if count >= 3 && instances != 0 {
                            crate::backend::metal_draw::note_once(format!("rectangle configuration refused: {}", p.name));
                        }
                        count_draw(DrawDrop::Topology);
                        return;
                    };
                    DrawKind::Geometry { draw, _index: None }
                }
                DrawKindIn::Indexed { count, instances, first, base, first_instance } => {
                    let v = self.index;
                    let (index_type, index_width) = match v.format {
                        57 => (0u8, 2u64), // R16_UINT
                        42 => (1, 4),      // R32_UINT
                        _ => {
                            count_draw(DrawDrop::IndexBuffer);
                            return;
                        }
                    };
                    let first_byte = u64::from(first) * index_width;
                    if v.address == 0 || v.address % index_width != 0 || first_byte + u64::from(count) * index_width > u64::from(v.size) {
                        count_draw(DrawDrop::IndexBuffer);
                        return;
                    }
                    let Some((buffer, view_offset)) = m.chunks().resident_window(v.address, u64::from(v.size)) else {
                        crate::backend::metal_draw::note_once(format!("index buffer (rectangles) not resident: {:#x}+{:#x}", v.address, v.size));
                        count_draw(DrawDrop::IndexBuffer);
                        return;
                    };
                    let address = buffer.gpu_address + view_offset;
                    let Some(draw) = config.draw(count, instances, first, first_instance, Some((index_type, address, base))) else {
                        if count >= 3 && instances != 0 {
                            crate::backend::metal_draw::note_once(format!("indexed rectangle configuration refused: {}", p.name));
                        }
                        count_draw(DrawDrop::Topology);
                        return;
                    };
                    DrawKind::Geometry { draw, _index: Some(buffer) }
                }
                DrawKindIn::Mesh { .. } | DrawKindIn::Indirect { .. } => {
                    crate::backend::metal_draw::note_once(format!("unsupported mesh/indirect rectangle: {}", p.name));
                    count_draw(DrawDrop::Topology);
                    return;
                }
            }
        } else {
            match kind {
                DrawKindIn::Vertices { count, instances, first, first_instance } => DrawKind::Vertices { count, instances, first, first_instance },
                DrawKindIn::Indexed { count, instances, first, base, first_instance } => {
                    let v = self.index;
                    let (index_type, index_width) = match v.format {
                        57 => (0, 2u64), // R16_UINT
                        42 => (1, 4),    // R32_UINT
                        _ => {
                            count_draw(DrawDrop::IndexBuffer);
                            return;
                        }
                    };
                    let first_byte = u64::from(first) * index_width;
                    if v.address == 0 || v.address % index_width != 0 || first_byte + u64::from(count) * index_width > u64::from(v.size) {
                        count_draw(DrawDrop::IndexBuffer);
                        return;
                    }
                    let Some((buffer, view_offset)) = m.chunks().window_buffer(v.address, u64::from(v.size)) else {
                        crate::backend::metal_draw::note_once(format!("index buffer not wrapped: {:#x}+{:#x}", v.address, v.size));
                        count_draw(DrawDrop::IndexBuffer);
                        return;
                    };
                    let Ok(offset) = u32::try_from(view_offset + first_byte) else {
                        crate::backend::metal_draw::note_once(format!("index buffer offset past 4 GiB: {view_offset:#x}+{first_byte:#x} in a span of {:#x}", buffer.len));
                        count_draw(DrawDrop::IndexBuffer);
                        return;
                    };
                    DrawKind::Indexed { count, instances, base, first_instance, buffer, offset, index_type }
                }
                DrawKindIn::Mesh { groups } => DrawKind::Mesh { groups: [u64::from(groups), 1, 1] },
                DrawKindIn::Indirect { indexed, records, stride, max, count } => {
                    // The records and count read in place by the GPU (zero
                    // copy), translated at replay (F42).
                    let words: u64 = if indexed { 5 } else if p.mesh { 3 } else { 4 };
                    let bytes = u64::from(max.saturating_sub(1)) * u64::from(stride) + words * 4;
                    let mut chunks = m.chunks();
                    let Some((rb, ro)) = chunks.window_buffer(records, bytes) else {
                        count_draw(DrawDrop::IndexBuffer);
                        return;
                    };
                    let count_buf = match count {
                        Some(at) => match chunks.window_buffer(at, 4) {
                            Some(c) => Some(c),
                            None => {
                                count_draw(DrawDrop::IndexBuffer);
                                return;
                            }
                        },
                        None => None,
                    };
                    let index = if indexed {
                        let v = self.index;
                        let (ty, width) = match v.format {
                            57 => (0u8, 2u32),
                            42 => (1, 4),
                            _ => {
                                count_draw(DrawDrop::IndexBuffer);
                                return;
                            }
                        };
                        let Some((ib, io)) = (v.address != 0 && v.size != 0).then(|| chunks.window_buffer(v.address, u64::from(v.size))).flatten() else {
                            count_draw(DrawDrop::IndexBuffer);
                            return;
                        };
                        Some((ib, io, ty, v.size / width))
                    } else {
                        None
                    };
                    DrawKind::Indirect { records: rb, records_offset: ro, records_va: records, stride, max, count: count_buf, count_va: count, index }
                }
            }
        };
        let full = victory_metal::sys::WMTViewport {
            originX: 0.0,
            originY: 0.0,
            width: f64::from(targets.width),
            height: f64::from(targets.height),
            znear: 0.0,
            zfar: 1.0,
        };
        let viewport = self.viewports.first().map_or(full, crate::backend::metal_draw::viewport);
        let scissor = match self.scissors.first() {
            Some(r) => match crate::backend::metal_draw::scissor(r, targets.width, targets.height) {
                Some(s) => s,
                None => {
                    count_draw(DrawDrop::Scissor);
                    return;
                }
            },
            None => victory_metal::sys::WMTScissorRect { x: 0, y: 0, width: u64::from(targets.width), height: u64::from(targets.height) },
        };
        count_draw(DrawDrop::Ran);
        crate::backend::metal_draw::note_targets(&targets);
        let op = std::sync::Arc::new(crate::backend::metal::MetalOp::Draw(MetalDraw {
            trace_inputs: if crate::backend::metal_draw::filtered_trace() && crate::backend::metal_draw::trace_pipeline(&g.name) {
                // Recording can precede the selected replay frame. Resolve the explicit SRVs now.
                p.stages.iter().flat_map(|stage| &stage.translated().used).filter_map(|u| {
                    if u.root != RootType::Table { return None; }
                    let Root::Table(base) = self.roots_ref(false).get(usize::from(u.slot))? else { return None };
                    match crate::descriptor::view(base.checked_add(u64::from(u.table_offset_dwords) * 4)?) {
                        Some(crate::descriptor::View::Shader(r, _)) if crate::resource::texture_memory(r as This).is_some() => Some(r),
                        _ => None,
                    }
                }).collect()
            } else { Vec::new() },
            pso: g,
            targets,
            stages,
            vertex,
            viewport,
            scissor,
            blend: self.blend.unwrap_or([1.0; 4]),
            stencil_ref: self.stencil.unwrap_or(0) as u8,
            primitive,
            kind,
        }));
        if let crate::backend::metal::MetalOp::Draw(d) = &*op
            && crate::backend::metal_draw::final_draw_dossier(d)
        {
            // What the draw into the presented texture reads, and those
            // textures as Metal has them right after it ran.
            let slots = SLOTS.with(|s| std::mem::take(&mut *s.borrow_mut()));
            let mut reads = Vec::new();
            let mut after = crate::backend::metal_draw::CAPTURE_AFTER.lock().unwrap_or_else(|e| e.into_inner());
            let n = after.last().map(|(_, name)| name.clone()).unwrap_or_default();
            for (k, (index, handle, missing)) in slots.into_iter().enumerate() {
                let view = crate::descriptor::view(handle);
                let what = match &view {
                    Some(crate::descriptor::View::Shader(r, w)) => {
                        let t = m.texture_for(*r as This);
                        after.push((*r, format!("{n}-read{k}")));
                        // As the draw will find it (what ran or was filled before).
                        m.capture(*r as This, &format!("{n}-read{k}-before"));
                        // Followed from now on: who draws into it is next.
                        crate::backend::metal_draw::note_presented(*r);
                        format!("srv res {r} words {w:?} metal {:?}", t.map(|t| (t.pixel.0, t.desc.width, t.desc.height, t.desc.flags)))
                    }
                    Some(crate::descriptor::View::Unordered(r, _, _)) => format!("uav res {r}"),
                    Some(crate::descriptor::View::Placed(u, d, _)) => format!("placed uav={u} desc {:?}", &d[..6]),
                    Some(crate::descriptor::View::Constant(a, z)) => format!("cbv {a:#x}+{z}"),
                    Some(v) => format!("{v:?}").chars().take(60).collect(),
                    None => "no view".into(),
                };
                reads.push(format!("slot {index}: {what} missing {missing:?}"));
            }
            drop(after);
            tracing::info!(reads = ?reads, "Metal final-draw dossier: what the draw reads from the heap");
        }
        self.metal.push(op);
    }

    /// A draw in console topology `c` as a [`DrawKind::Generated`]: its
    /// vertex ids (relative to the first vertex) or the title's indices
    /// expanded into list indices. `None` if the index buffer is unusable.
    #[cfg(windows)]
    fn console_draw(
        &self,
        c: crate::backend::metal_draw::ConsoleTopology,
        kind: crate::backend::metal_draw::DrawKindIn,
    ) -> Option<crate::backend::metal_draw::DrawKind> {
        use crate::backend::metal_draw::{DrawKind, DrawKindIn, console_indices};
        match kind {
            DrawKindIn::Vertices { count, instances, first, first_instance } => {
                let src: Vec<u32> = (0..count).collect();
                Some(DrawKind::Generated { indices: console_indices(c, &src), instances, base: first as i32, first_instance })
            }
            DrawKindIn::Indexed { count, instances, first, base, first_instance } => {
                let v = self.index;
                let width = match v.format {
                    57 => 2u64, // R16_UINT
                    42 => 4,    // R32_UINT
                    _ => return None,
                };
                let first_byte = u64::from(first) * width;
                if v.address == 0 || v.address % width != 0 || first_byte + u64::from(count) * width > u64::from(v.size) {
                    return None;
                }
                let at = (v.address + first_byte) as *const u8;
                // SAFETY: inside the title's index buffer view (checked
                // above); title memory is mapped in this process.
                let src: Vec<u32> = unsafe {
                    if width == 2 {
                        std::slice::from_raw_parts(at.cast::<u16>(), count as usize).iter().map(|&i| u32::from(i)).collect()
                    } else {
                        std::slice::from_raw_parts(at.cast::<u32>(), count as usize).to_vec()
                    }
                };
                Some(DrawKind::Generated { indices: console_indices(c, &src), instances, base, first_instance })
            }
            DrawKindIn::Mesh { .. } | DrawKindIn::Indirect { .. } => None,
        }
    }

    /// Why the bound targets gave no attachments (diagnostics).
    #[cfg(windows)]
    fn target_reasons(&self, m: &crate::backend::metal::MetalBackend) -> String {
        use crate::backend::metal_draw::level_slice;
        let mut out = Vec::new();
        for &h in &self.target_handles {
            out.push(match crate::descriptor::rtv_record(h) {
                None => "rtv: no record".to_string(),
                Some(r) => match r.target_words() {
                    None => "rtv: no words".to_string(),
                    Some(w) => match level_slice(&w, false) {
                        None => format!("rtv: dimension {}", w[1]),
                        Some((l, sl)) => match m.texture_for(r.resource) {
                            None => format!("rtv: no Metal texture (format {})", w[0]),
                            Some(t) => format!("rtv: format {} level {l} slice {sl} texture {} mips {} slices {}", w[0], t.pixel.0, t.mips, t.slices),
                        },
                    },
                },
            });
        }
        if self.depth_handle != 0 {
            out.push(match crate::descriptor::dsv_record(self.depth_handle) {
                None => "dsv: no record".to_string(),
                Some(r) => match r.depth_words() {
                    None => "dsv: no words".to_string(),
                    Some(w) => match m.texture_for(r.resource) {
                        None => format!("dsv: no Metal texture (format {})", w[0]),
                        Some(t) => format!("dsv: format {} dim {} texture {}", w[0], w[1], t.pixel.0),
                    },
                },
            });
        }
        out.join("; ")
    }

    /// The bound render targets and depth view as Metal attachments.
    #[cfg(windows)]
    fn make_metal_targets(&self, m: &crate::backend::metal::MetalBackend) -> crate::backend::metal_draw::Targets {
        use crate::backend::metal_draw::{Targets, level_slice};
        let mut t = Targets::default();
        for (slot, &h) in t.colors.iter_mut().zip(&self.target_handles) {
            let Some(record) = crate::descriptor::rtv_record(h) else { continue };
            let Some(words) = record.target_words() else { continue };
            let Some((level, slice)) = level_slice(&words, false) else { continue };
            *slot = m.attachment(record.resource, words[0], level, slice, false);
        }
        if let Some(record) = crate::descriptor::dsv_record(self.depth_handle)
            && let Some(words) = record.depth_words()
            && let Some((level, slice)) = level_slice(&words, true)
        {
            t.depth = m.attachment(record.resource, words[0], level, slice, true);
            // Depth32Float_Stencil8 (260), X32_Stencil8 (261): a stencil plane.
            t.stencil = t.depth.as_ref().is_some_and(|d| matches!(d.pixel, 260 | 261));
        }
        t.finish();
        t
    }

    /// One stage's Metal bindings (`compute`: the compute roots, else the
    /// graphics roots): its user data in a copy of `top`, its root views as
    /// one-view tables or root descriptors, the heap slots it reads
    /// refreshed. `limit`: the guarded dispatch's thread count.
    #[cfg(windows)]
    fn metal_stage(
        &self,
        m: &crate::backend::metal::MetalBackend,
        p: &crate::pipeline::host::HostPipeline,
        stage: &crate::pipeline::host::Stage,
        compute: bool,
        mut top: victory_metal::ir::TopLevel,
        limit: u32,
        groups_xy: [u32; 2],
    ) -> Result<MetalStage, crate::backend::metal::Drop> {
        use crate::backend::metal::Drop;
        use crate::backend::metal_heap::Missing;
        use victory_shader::emit::{SLOT_STRIDES, SLOT_WAVE_SCRATCH, USER_DATA_SPACE, USER_GROUP_BASE, USER_GROUP_COUNT_X, USER_GROUP_COUNT_Y};
        static SKIPPED_HOST_WORDS: AtomicU64 = AtomicU64::new(0);
        let t = stage.translated();
        let host = &t.host;

        // Title memory by address (`SLOT_MEMORY`): the views of what the
        // root arguments name, in the run after the view heap's slots.
        let memory = if host.root_srvs.contains(&victory_shader::emit::SLOT_MEMORY) {
            if self.heaps[0] == 0 {
                once!(pipeline = %p.name, "Metal: a shader reads title memory by address with no view heap bound: dropped");
                return Err(Drop::MemoryDirectory);
            }
            let (mut addresses, mut constants) = (Vec::new(), Vec::new());
            for root in self.roots_ref(compute) {
                match root {
                    Root::View(address) => addresses.push(*address),
                    Root::Constants(values) => constants.extend_from_slice(values),
                    _ => {}
                }
            }
            let heap = crate::descriptor::heap(self.heaps[0] as This);
            match m.memory_ready(self.heaps[0], heap.count(), &addresses, &constants) {
                Some(r) => Some(r),
                None => {
                    once!(pipeline = %p.name, "Metal: no room for title memory views: dropped");
                    return Err(Drop::MemoryDirectory);
                }
            }
        } else {
            None
        };

        // User data.
        let mut words = vec![0u32; host.user_data_dwords() as usize];
        for u in &host.user_data {
            let v = match u.root {
                RootType::Other(kind) if kind == victory_shader::emit::USER_MEMORY_BASE && memory.is_some() => {
                    memory.as_ref().map_or(0, |r| r.base)
                }
                RootType::Other(kind) if kind == USER_GROUP_BASE => 0, // Metal needs no split dispatches
                RootType::Other(kind) if kind == victory_shader::emit::USER_DISPATCH_THREADS_X => limit,
                RootType::Other(kind) if kind == USER_GROUP_COUNT_X => groups_xy[0],
                RootType::Other(kind) if kind == USER_GROUP_COUNT_Y => groups_xy[1],
                // Metal's one-view tables begin at the title's address itself.
                RootType::Other(kind) if kind == victory_shader::emit::USER_VIEW_BIAS => 0,
                RootType::Other(kind)
                    if kind == victory_shader::emit::USER_VIEW_ADDRESS || kind == victory_shader::emit::USER_VIEW_ADDRESS + 1 =>
                {
                    self.host_word(compute, kind, u.slot, u.dword, None)
                }
                RootType::Other(_) => {
                    // A host word this backend does not build
                    if SKIPPED_HOST_WORDS.fetch_add(1, Ordering::Relaxed).is_power_of_two() {
                        tracing::info!(pipeline = %p.name, "Metal: a shader needs the memory directory: skipped");
                    }
                    return Err(Drop::MemoryDirectory);
                }
                _ => self.user_value(compute, u.slot, u.root, u.dword),
            };
            if let Some(w) = words.get_mut(u.index as usize) {
                *w = v;
            }
        }
        if !words.is_empty() && !top.set_constants(USER_DATA_SPACE, 0, &words) {
            once!(pipeline = %p.name, "Metal: user data does not fit the top-level layout");
            return Err(Drop::UserDataLayout);
        }

        // Sync dirty descriptor slots on the bound view and sampler heaps first so
        // bindless reads and `SLOT_STRIDES` see up-to-date entries and strides.
        METAL_PREP[0].fetch_add(1, Ordering::Relaxed);
        let view_bound = (self.heaps[0] != 0)
            .then(|| {
                let heap = crate::descriptor::heap(self.heaps[0] as This);
                m.sync_heap(self.heaps[0], heap.count(), heap.samplers())
            })
            .flatten();
        let sampler_bound = (self.heaps[1] != 0)
            .then(|| {
                let heap = crate::descriptor::heap(self.heaps[1] as This);
                m.sync_heap(self.heaps[1], heap.count(), heap.samplers())
            })
            .flatten();
        let mut heap_buffer = view_bound.as_ref().map(|(b, _)| b.clone());
        let mut sampler_buffer = sampler_bound.as_ref().map(|(b, _)| b.clone());
        let strides_gpu = view_bound.as_ref().map_or(0, |(_, s)| *s);

        // Root views by the Metal root layout: victory's own as root
        // descriptors, the title's as one-view tables bounded by the title
        // buffer that holds them (zero copy: a stray write is corruption).
        let mut tables = Vec::new();
        // Our own buffers first: making one adds it to the residency set,
        // which takes the chunks lock held below.
        let wave_scratch_gpu = if host.root_uavs.contains(&SLOT_WAVE_SCRATCH) { m.wave_scratch_gpu() } else { 0 };
        let diag_gpu = if host.root_uavs.contains(&victory_shader::emit::SLOT_DIAG) { crate::backend::metal::diag::gpu(m) } else { 0 };
        {
            use crate::backend::metal::{RootSlot, root_layout};
            let mut chunks = m.chunks();
            for (i, r) in root_layout(host).into_iter().enumerate() {
                let slot = match r {
                    RootSlot::Constants(_) => continue,
                    RootSlot::Srv(s) if s == SLOT_STRIDES => {
                        top.set_u64(i, strides_gpu);
                        continue;
                    }
                    RootSlot::Uav(s) if s == SLOT_WAVE_SCRATCH => {
                        top.set_u64(i, wave_scratch_gpu);
                        continue;
                    }
                    RootSlot::Uav(s) if s == victory_shader::emit::SLOT_DIAG => {
                        top.set_u64(i, diag_gpu);
                        continue;
                    }
                    RootSlot::Srv(s) if s == victory_shader::emit::SLOT_MEMORY => {
                        top.set_u64(i, memory.as_ref().map_or(0, |r| r.directory));
                        continue;
                    }
                    RootSlot::Uav(s) if s == victory_shader::emit::SLOT_FAULTS => {
                        top.set_u64(i, memory.as_ref().map_or(0, |r| r.faults));
                        continue;
                    }
                    RootSlot::Cbv(s) | RootSlot::Srv(s) | RootSlot::Uav(s) | RootSlot::TableSrv(s) | RootSlot::TableUav(s) => s,
                };
                let va = match self.roots_ref(compute).get(usize::from(slot)) {
                    Some(Root::View(va)) => *va,
                    _ => 0,
                };
                let extent = if va == 0 {
                    0
                } else {
                    METAL_PREP[2].fetch_add(1, Ordering::Relaxed);
                    match crate::shadow::buffer_extent(va) {
                        Some((_, end)) => end - va,
                        None => victory_metal::chunks::committed_extent(va, 4 << 20),
                    }
                };
                let gpu = if extent == 0 { 0 } else { chunks.window(va, extent) };
                match r {
                    RootSlot::Cbv(_) if !compute && va != 0 && gpu == 0 => {
                        once!(pipeline = %p.name, slot, "Metal: graphics root constant buffer has no mapped GPU address");
                        return Err(Drop::SlotUnmapped);
                    }
                    RootSlot::TableSrv(_) | RootSlot::TableUav(_) => {
                        let entry = if gpu == 0 { victory_metal::ir::DescriptorEntry::NULL } else { victory_metal::ir::DescriptorEntry::buffer(gpu, extent) };
                        if let Some(at) = top.offset_of(i) {
                            tables.push((at, entry));
                        }
                    }
                    _ => {
                        top.set_u64(i, gpu);
                    }
                }
            }
        }

        // The heaps: the slots the shader reads, refreshed (view heap at
        // bind point 0, sampler heap at 1).
        let mut worst: Option<Missing> = None;
        for u in &t.used {
            if u.root != RootType::Table {
                continue;
            }
            let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else { continue };
            let handle = table + u64::from(u.table_offset_dwords) * 4;
            let Some(heap) = self.heap_of(handle) else { continue };
            let Some(index) = heap.index(handle) else { continue };
            let key = std::ptr::from_ref(heap) as usize;
            METAL_PREP[1].fetch_add(1, Ordering::Relaxed);
            if let Some((buffer, _, missing)) = m.refresh_slot(key, heap.count(), heap.samplers(), index, handle) {
                Self::note_slot(index, handle, missing);
                worst = worst.or(missing);
                if heap.samplers() {
                    sampler_buffer = Some(buffer);
                } else {
                    heap_buffer = Some(buffer);
                }
            }
        }
        // Views picked from the heap by a root constant (table base + byte
        // offset + constant << shift, 32-byte slots), as `tables` does.
        for index in &host.root_indexed_views {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(index.table_slot)) else { continue };
            let constant = self.user_value(compute, index.constant_slot, RootType::Constants, index.constant_dword);
            let slot = index.byte_offset.wrapping_add(constant << index.shift) / 32;
            let Some(handle) = table.checked_add(u64::from(slot) * 32) else { continue };
            let Some(heap) = self.heap_of(handle) else { continue };
            let Some(i) = heap.index(handle) else { continue };
            let key = std::ptr::from_ref(heap) as usize;
            METAL_PREP[1].fetch_add(1, Ordering::Relaxed);
            if let Some((buffer, _, missing)) = m.refresh_slot(key, heap.count(), heap.samplers(), i, handle) {
                Self::note_slot(i, handle, missing);
                worst = worst.or(missing);
                if heap.samplers() {
                    sampler_buffer = Some(buffer);
                } else {
                    heap_buffer = Some(buffer);
                }
            }
        }
        // Shaders that read no table still get the bound heaps.
        for (h, out) in [(self.heaps[0], &mut heap_buffer), (self.heaps[1], &mut sampler_buffer)] {
            if h != 0 && out.is_none() {
                let heap = crate::descriptor::heap(h as This);
                METAL_PREP[1].fetch_add(1, Ordering::Relaxed);
                *out = m.refresh_slot(h, heap.count(), heap.samplers(), 0, heap.handle(0).unwrap_or(0)).map(|(b, _, _)| b);
            }
        }
        let outcome = match worst {
            None => Drop::Ran,
            Some(Missing::Typed { .. }) => Drop::SlotTyped,
            Some(Missing::Unmapped) => Drop::SlotUnmapped,
            Some(_) => Drop::SlotMissing,
        };
        Ok(MetalStage { top: top.bytes, tables, heap: heap_buffer, sampler_heap: sampler_buffer, outcome })
    }

    #[cfg(feature = "diagnostics")]
    fn op_text(&self, kind: String, compute: bool) -> String {
        use std::fmt::Write;
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return kind;
        };
        let mut text = format!("{kind} {}", p.name);
        // What it draws into (a dispatch writes through its views below).
        if !compute {
            let described = |t: usize| {
                crate::resource::host_made(t as This).map(|r| {
                    // SAFETY: a live resource.
                    let d = unsafe { r.GetDesc() };
                    format!("{t:#x} {}x{} f{}", d.Width, d.Height, d.Format.0)
                })
            };
            let targets: Vec<String> = self.targets.iter().filter_map(|&t| described(t)).collect();
            let _ = write!(text, " targets {targets:?} depth {:?} state [{}]", described(self.depth), p.state);
            // The title's scissor and viewport at this draw (the first of
            // each): the title draws what it sees through a
            // portal (a window, from inside a room) under the portal's
            // rectangle, and the listing did not say which draws those are.
            if let Some(r) = self.scissors.first() {
                let _ = write!(text, " scissor [{} {} {} {}]", r.left, r.top, r.right, r.bottom);
            }
            if let Some(v) = self.viewports.first() {
                let _ = write!(
                    text,
                    " viewport [{} {} {} {} depth {}..{}]",
                    v.TopLeftX, v.TopLeftY, v.Width, v.Height, v.MinDepth, v.MaxDepth
                );
            }
        }
        let views = (self.heaps[0] != 0)
            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
            .flatten();
        for stage in &p.stages {
            let t = stage.translated();
            let _ = write!(text, " | {}:", stage.shader.hash);
            for u in &t.host.user_data {
                let v = match u.root {
                    RootType::Other(kind) => self.host_word(compute, kind, u.slot, u.dword, None),
                    _ => self.user_value(compute, u.slot, u.root, u.dword),
                };
                let _ = write!(text, " u{}={v:#x}", u.index);
                let index = if u.root == RootType::Table { v / 32 } else { v };
                if let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let _ = write!(text, " [heap {index}: {slot}]");
                }
            }
            for &(slot, _) in stage.root_cbvs.iter().chain(&stage.root_srvs).chain(&stage.root_uavs) {
                if let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) {
                    let host = crate::shadow::find(*address, 1)
                        .map(|s| (s.gpu(*address), s.left(*address)));
                    let _ = write!(text, " root{slot}={address:#x} host {host:#x?}");
                }
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        text
    }

    /// A mesh draw under `VICTORY_MESH_TEST=scissor` rasterises one pixel:
    /// the scissors before it (`on`), the title's after it.
    fn mesh_scissor(&self, list: &HostList, on: bool) {
        if mesh_test() != 2 {
            return;
        }
        if on {
            let one = RECT { left: 0, top: 0, right: 1, bottom: 1 };
            crate::draw::scissors(list, &vec![one; self.scissors.len().max(1)]);
        } else if !self.scissors.is_empty() {
            crate::draw::scissors(list, &self.scissors);
        }
    }

    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32) {
        self.window = None;
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        let ran = mesh_test() != 1
            && mesh_kept(self.pipeline)
            && self.prepare_for(list, false, true)
            && match list.list.cast::<ID3D12GraphicsCommandList6>() {
                Ok(l6) => {
                    #[cfg(windows)]
                    if crate::backend::metal_draw::draws_on() {
                        self.metal_draw(crate::backend::metal_draw::DrawKindIn::Mesh { groups: x });
                    }
                    #[cfg(feature = "diagnostics")]
                    list.note_op(|| self.op_text(format!("mesh {x}"), false));
                    self.mesh_scissor(list, true);
                    // SAFETY: an open list with everything set.
                    unsafe { l6.DispatchMesh(x, 1, 1) };
                    self.mesh_scissor(list, false);
                    DRAWN.fetch_add(1, Ordering::Relaxed);
                    true
                }
                Err(_) => false,
            };
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute {
            &self.compute
        } else {
            &self.graphics
        }
    }

    fn table_entry(&self, list: &HostList, u: &UsedResource, compute: bool) {
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        if let Some(view) = host.mirror(index, handle, true) {
            readable(list, u.kind, &view);
            // Up to date by this draw: not the texture pass's to make.
            if crate::descriptor::host::is_texture_view(&view) {
                host.texture_slot_tabled(index, true);
            }
        }
    }

    /// The descriptors a shader reads through tables. Tables holding a
    /// constant buffer are made afresh for the draw in dynamic slots, with
    /// the constant data copied to upload memory when the list is
    /// submitted: the host wants constant buffers 256-byte aligned, the
    /// console only 16. Returns those tables' new byte offsets by root
    /// parameter; the other tables' slots are made from the title's.
    ///
    /// Within a list, the same table holding the same descriptors is made
    /// once: its constant data is copied at submission either way, and the
    /// big tables the world binds would otherwise take hundreds of slots per
    /// draw, wrapping the dynamic slots onto lists not yet submitted.
    pub(crate) fn tables(
        &mut self,
        list: &HostList,
        used: &[UsedResource],
        indexed: &[RootIndexedView],
        compute: bool,
    ) -> Vec<(u8, u32)> {
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        let mut fresh: Vec<u8> = used
            .iter()
            .filter(tables)
            .filter(|u| u.kind == ResourceType::Cbv)
            .map(|u| u.slot)
            .collect();
        // Tables reaching past a host heap smaller than the title's (the
        // host's limit, 1,000,000 views on NVIDIA) are made afresh too: the
        // title's slots there have no host slot of their own.
        for u in used.iter().filter(tables) {
            if fresh.contains(&u.slot) {
                continue;
            }
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
                continue;
            };
            let end = table + u64::from(u.table_offset_dwords) * 4;
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let (Some(host), Some(index)) = (heap.host(), heap.index(end)) else {
                continue;
            };
            if index >= host.count() {
                fresh.push(u.slot);
            }
        }
        fresh.sort_unstable();
        fresh.dedup();
        for u in used.iter().filter(tables) {
            if !fresh.contains(&u.slot) {
                self.table_entry(list, u, compute);
            }
        }
        let mut made = Vec::new();
        for slot in fresh {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let entries: Vec<&UsedResource> = used
                .iter()
                .filter(tables)
                .filter(|u| u.slot == slot)
                .collect();
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            if heap.samplers() || (table - heap.start) % 32 != 0 {
                continue;
            }
            let Some(host) = heap.host() else {
                continue;
            };
            let mut selected: Vec<u32> = indexed
                .iter()
                .filter(|index| index.table_slot == slot)
                .map(|index| {
                    let constant = self.user_value(
                        compute,
                        index.constant_slot,
                        RootType::Constants,
                        index.constant_dword,
                    );
                    // The shader adds i32 byte offsets and shifts by 5.
                    index.byte_offset.wrapping_add(constant << index.shift) / 32
                })
                .collect();
            selected.sort_unstable();
            selected.dedup();
            let n = entries
                .iter()
                .map(|u| u.table_offset_dwords / 8)
                .chain(selected.last().copied())
                .max()
                .unwrap_or(0)
                + 1;
            let last = table.checked_add(u64::from(n - 1) * 32);
            if n > crate::descriptor::host::DYNAMIC
                || last.and_then(|handle| heap.index(handle)).is_none()
            {
                once!(slot, n, "fresh descriptor table exceeds the available span");
                continue;
            }
            // The table and what its entries hold (record words).
            let key = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                for k in 0..n {
                    let at = table + u64::from(k) * 32;
                    // SAFETY: a descriptor slot in title memory.
                    unsafe { (at as *const [u32; 3]).read_unaligned() }.hash(&mut h);
                }
                (table, n, h.finish())
            };
            if let Some(&first) = self.fresh_tables.get(&key) {
                for u in entries.iter().filter(|u| u.kind != ResourceType::Cbv) {
                    let handle = table + u64::from(u.table_offset_dwords) * 4;
                    if let Some(view) = crate::descriptor::view(handle) {
                        readable(list, u.kind, &view);
                    }
                }
                for k in selected {
                    if entries.iter().any(|u| u.table_offset_dwords / 8 == k) {
                        continue;
                    }
                    let handle = table + u64::from(k) * 32;
                    if let Some(view) = crate::descriptor::view(handle)
                        && let Some(kind) = view_kind(&view)
                    {
                        readable(list, kind, &view);
                    }
                }
                made.push((slot, first * 32));
                continue;
            }
            let Some(first) = host.dynamic(n) else {
                // No room: the draw must not run with the title's own
                // offsets (slots never made for it); `bind` drops it.
                made.push((slot, u32::MAX));
                continue;
            };
            self.fresh_tables.insert(key, first);
            self.dynamic
                .push((std::ptr::from_ref(heap) as usize, first, n));
            // The title's own slots of this table: what the draw reads of
            // them it brings up to date here, in the dynamic slots.
            let title_first = heap.index(table).filter(|first| first + n <= host.count());
            // The whole range: a shader may index past the entries its
            // used-resource list names (descriptor arrays).
            for k in 0..n {
                let index = first + k;
                let used = entries.iter().find(|u| u.table_offset_dwords / 8 == k);
                match crate::descriptor::view(table + u64::from(k) * 32) {
                    Some(View::Constant(address, size)) => {
                        let Some((gpu, to)) = list.scratch(u64::from(size)) else {
                            host.make_null(index);
                            continue;
                        };
                        list.defer_fill(to, address as *const u8, size as usize);
                        host.constant_at(index, gpu, size);
                    }
                    Some(view) => {
                        host.make_at(index, &view);
                        let kind = used.map(|u| u.kind).or_else(|| {
                            selected.binary_search(&k).ok().and_then(|_| view_kind(&view))
                        });
                        if let Some(kind) = kind {
                            readable(list, kind, &view);
                        }
                        // The title's slot is a table's: out of the queue
                        // of textures picked by index if the draw read it
                        // (its texture is up to date now), and made in
                        // place as the texture pass would have made it,
                        // for a shader that picks it by index all the same.
                        if let Some(title) = title_first
                            && crate::descriptor::host::is_texture_view(&view)
                        {
                            host.texture_slot_tabled(title + k, kind.is_some());
                            if kind.is_some() {
                                host.mirror(title + k, table + u64::from(k) * 32, true);
                            }
                        }
                    }
                    None => {
                        host.make_null(index);
                        if used.is_some() {
                            once!(slot, "a table entry without a view");
                        }
                    }
                }
            }
            made.push((slot, first * 32));
        }
        made
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn trace(&self, kind: &str, count: u32, first: u32) {
        static TRACED: AtomicU64 = AtomicU64::new(0);
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let listed = crate::capture::listed(crate::queue::presented() + 1);
        let logged = listed || TRACED.fetch_add(1, Ordering::Relaxed) < 64;
        let targets: Vec<String> = self
            .targets
            .iter()
            .filter_map(|&t| crate::resource::host_made(t as This))
            .map(|t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                format!("{}x{} f{}", d.Width, d.Height, d.Format.0)
            })
            .collect();
        let v = self.vertex[0];
        let line = format!(
            "{kind} {count} from {first} topology {} pipeline {:#x} targets {targets:?} vb {:#x}+{} stride {} reads {:?}",
            self.topology,
            self.pipeline,
            v.address,
            v.size,
            v.stride,
            self.reads(false)
        );
        if logged {
            tracing::debug!("draw {line}");
        }
        let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if frame.len() < 256 {
            frame.push(line);
        }
    }

    /// What a draw's shaders read, one line per used resource (diagnostics
    /// builds).
    #[cfg(feature = "diagnostics")]
    fn reads(&self, compute: bool) -> Vec<String> {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return vec!["no host pipeline".into()];
        };
        let mut out = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            for u in &t.used {
                let what = match (u.root, self.roots_ref(compute).get(usize::from(u.slot))) {
                    (RootType::Table, Some(Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        describe(crate::descriptor::view(handle).as_ref())
                    }
                    (root, _) => format!("{root:?}"),
                };
                out.push(format!(
                    "{} {:?} s{}+{}: {what}",
                    stage.shader.hash, u.kind, u.slot, u.table_offset_dwords
                ));
            }
        }
        out
    }

    pub(crate) fn draw(
        &mut self,
        list: &HostList,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        self.window = None;
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        let ready = self.prepare(list, false);
        // A pipeline with the rectangle geometry stage draws the title's
        // vertices as they are; any other gets its rectangles expanded.
        let rects = self.topology == RECTANGLES && !self.completes_rects();
        // Fans, line loops, quads and quad strips: drawn from expanded
        // indices here (F57), so the plain draw below is skipped.
        let expansion = crate::primitive::ListExpansion::from_topology(self.topology);
        let ran = ready
            && (!rects || self.rectangles(list, vertices, first))
            && expansion.is_none_or(|kind| {
                i32::try_from(first).is_ok_and(|base| self.host_console_list(list, kind, vertices, instances, base, first_instance, None))
            });
        #[cfg(windows)]
        if !ran && crate::backend::metal_draw::draws_on() {
            self.note_asked(8);
        }
        if ran {
            // Rectangles were expanded into their own vertex buffers.
            let (count, from) = if rects {
                (vertices / 3 * 6, 0)
            } else {
                (vertices, first)
            };
            #[cfg(windows)]
            if crate::backend::metal_draw::draws_on() {
                self.metal_draw(crate::backend::metal_draw::DrawKindIn::Vertices { count: vertices, instances, first, first_instance });
            }
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
            if expansion.is_none() {
                // SAFETY: an open list with everything set.
                unsafe {
                    list.list
                        .DrawInstanced(count, instances, from, first_instance)
                };
            }
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    /// A console list (fan, line loop, quads, quad strip) drawn on the host
    /// from expanded indices: generated for a plain draw, made from the
    /// title's indices for an indexed one (F57). False if it cannot be.
    #[allow(clippy::too_many_arguments)]
    fn host_console_list(
        &self,
        list: &HostList,
        expansion: crate::primitive::ListExpansion,
        count: u32,
        instances: u32,
        base: i32,
        first_instance: u32,
        indexed_first: Option<u32>,
    ) -> bool {
        let Some(expanded) = expansion.index_count(count) else {
            return false;
        };
        if expanded == 0 || instances == 0 {
            return true;
        }
        let view = if let Some(first) = indexed_first {
            let v = self.index;
            expansion.title_view(list, v.address, v.size, v.format, first, count)
        } else {
            expansion.generated_view(list, count)
        };
        let Some(view) = view else {
            return false;
        };
        // SAFETY: prepare set the pipeline and vertex buffers; the list keeps
        // the generated index upload until its submission is done.
        unsafe {
            list.list.IASetIndexBuffer(Some(&view));
            list.list.DrawIndexedInstanced(expanded, instances, 0, base, first_instance);
        }
        true
    }

    /// Whether the bound pipeline completes rectangles itself, in the
    /// geometry stage rectangle pipelines get (`HostPipeline::rects`).
    fn completes_rects(&self) -> bool {
        crate::pipeline::host_pipeline(self.pipeline as This).is_some_and(|p| p.rects)
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&self, list: &HostList, count: u32, first: u32) -> bool {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        let rects = u64::from(count / 3);
        for slot in 0..32 {
            let elements: Vec<(u32, u32)> = p
                .inputs
                .iter()
                .filter(|e| e.slot == slot && !e.per_instance)
                .map(|e| (e.offset, e.format))
                .collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            let stride = u64::from(v.stride);
            let from = v.address + u64::from(first) * stride;
            if v.address == 0
                || stride == 0
                || u64::from(first + count) * stride > u64::from(v.size)
                || crate::shadow::find(from, rects * 3 * stride).is_none()
            {
                once!(slot, "rectangle vertices outside their buffer");
                return false;
            }
            let Some((gpu, to)) = list.scratch(rects * 6 * stride) else {
                return false;
            };
            let (from, to, stride) = (from as usize, to as usize, stride as usize);
            let n = rects as usize;
            list.defer_host(move || {
                // The vertices are read in place, at submission: no change
                // of mapping under the read (see
                // `crate::watch::title_memory_reading`).
                let _mapped = crate::watch::title_memory_reading_range(from, n * 3 * stride);
                // SAFETY: the vertices lie in a title buffer (checked when
                // recorded) and the scratch memory is the list's.
                let (src, dst) = unsafe {
                    (
                        std::slice::from_raw_parts(from as *const u8, n * 3 * stride),
                        std::slice::from_raw_parts_mut(to as *mut u8, n * 6 * stride),
                    )
                };
                crate::primitive::expand_rects(src, dst, stride, &elements);
            });
            let view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: (rects * 6 * stride as u64) as u32,
                StrideInBytes: v.stride,
            };
            // SAFETY: an open list; the view names live scratch memory.
            unsafe { list.list.IASetVertexBuffers(slot, Some(&[view])) };
        }
        true
    }

    pub(crate) fn draw_indexed(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        self.window = None;
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        // Indexed rectangles are drawn by the rectangle geometry stage
        // alone (nothing expands them on the CPU).
        let lost = self.topology == RECTANGLES && !self.completes_rects();
        if lost {
            once!("indexed rectangles not drawn");
        }
        let expansion = crate::primitive::ListExpansion::from_topology(self.topology);
        let ran = !lost
            && self.prepare(list, expansion.is_none())
            && expansion.is_none_or(|kind| self.host_console_list(list, kind, indices, instances, base, first_instance, Some(first)));
        #[cfg(windows)]
        if !ran && crate::backend::metal_draw::draws_on() {
            self.note_asked(8);
        }
        if ran {
            #[cfg(windows)]
            if crate::backend::metal_draw::draws_on() {
                self.metal_draw(crate::backend::metal_draw::DrawKindIn::Indexed { count: indices, instances, first, base, first_instance });
            }
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
            if expansion.is_none() {
                // SAFETY: an open list with everything set.
                unsafe {
                    list.list
                        .DrawIndexedInstanced(indices, instances, first, base, first_instance)
                };
            }
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }
}

impl State {
    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. False if it cannot run on the host.
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
    ) -> bool {
        if max == 0 {
            return true;
        }
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        }
        if crate::pipeline::is_builder(self.pipeline as This) && !crate::pipeline::builder_runs() {
            return false;
        }
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| crate::pipeline::note_why(format!("indirect kind {kind}: {what}"));
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        };
        // Indirect rectangles (the title's tiled lighting, 256
        // indirect draws a frame, makes their corners from the vertex
        // number) are completed by the pipeline's geometry stage: the
        // arguments stay the title's, three vertices a rectangle. They
        // cannot be expanded on the CPU, which does not know their count.
        // shortcut: in a pipeline without that stage, a rectangle with no
        // per-vertex inputs is drawn as its first triangle (three vertices,
        // half of it), and one with vertex buffers is not drawn.
        let made_rects = !p.rects
            && kind == DRAW
            && self.topology == RECTANGLES
            && !p.inputs.iter().any(|e| !e.per_instance);
        if made_rects {
            once!("indirect rectangles without vertex inputs drawn as their first triangle");
        }
        let matches_kind = match kind {
            DRAW | INDEXED => {
                (self.topology != RECTANGLES || made_rects || p.rects) && !p.compute && !p.mesh
            }
            MESH => !p.compute && p.mesh,
            DISPATCH => p.compute,
            _ => false,
        };
        if !matches_kind {
            why(&format!(
                "pipeline mismatch (compute {}, mesh {}, topology {})",
                p.compute, p.mesh, self.topology
            ));
            return false;
        }
        if kind == MESH && (mesh_test() == 1 || !mesh_kept(self.pipeline)) {
            why("mesh draws are switched off (VICTORY_MESH_TEST=skip)");
            return false;
        }
        // Native mesh records contain only three group counts, not draw arguments.
        #[cfg(windows)]
        if kind == DISPATCH && crate::backend::metal::mode() != crate::backend::metal::Mode::Off {
            self.metal_dispatch_indirect(p, arguments, stride, max, count);
        } else if crate::backend::metal_draw::draws_on() && (kind == DRAW || kind == INDEXED || kind == MESH) && !made_rects {
            self.metal_draw(crate::backend::metal_draw::DrawKindIn::Indirect { indexed: kind == INDEXED, records: arguments, stride, max, count });
        }
        let len = u64::from(stride) * u64::from(max);
        let Some(args) = crate::shadow::find_any(arguments, len) else {
            why("arguments outside title buffers");
            return false;
        };
        let counter = match count {
            Some(at) => match crate::shadow::find_any(at, 4) {
                Some(s) => Some((s, at)),
                None => {
                    why("count outside title buffers");
                    return false;
                }
            },
            None => None,
        };
        let Some(signature) = matches!(kind, DRAW | INDEXED | DISPATCH | MESH)
            .then(|| host_signature(kind, stride))
            .flatten()
        else {
            why("no host signature");
            return false;
        };
        // A window of records clamped in one pass (`windows`): for single
        // records without a count buffer, drawn by shaders that store to no
        // buffer. Anything else ends the window: it uses the scratch buffer
        // or may write the arguments.
        const WINDOW_USE: usize = usize::MAX - 0x77;
        let offset = args.offset(arguments);
        // A dispatch that may have more groups than the host runs in one
        // dimension, by a program that takes the number of its first group.
        let parts = if kind == DISPATCH && max == 1 && count.is_none() {
            group_base(p).zip(host_signature(DISPATCH, crate::indirect::PART_BYTES as u32))
        } else {
            None
        };
        let windowed = windows()
            && kind != DISPATCH
            && max == 1
            && count.is_none()
            && !p.stages.iter().any(|stage| stage.translated().stores);
        // Another host list (a new one after a cut, or this one after its
        // reset) has another scratch buffer, or this one's with old bytes.
        if !windowed || list.first_use(WINDOW_USE) {
            self.window = None;
        }
        let same = |w: &Window| w.copy == args.id() && w.kind == kind && w.stride == stride;
        let hit = self
            .window
            .as_ref()
            .filter(|w| same(w) && offset >= w.base)
            .filter(|w| (offset - w.base) % u64::from(w.step) == 0 && (offset - w.base) / u64::from(w.step) < u64::from(w.records))
            .map(|w| (w.scratch.clone(), offset - w.base));
        let (bounded, at) = match hit {
            Some(hit) => hit,
            None => {
                // A record a short step after the last window's last one
                // (the same step as that window's, if it had one): the
                // title walks the buffer, so the ones that step apart
                // after it are clamped with it.
                let (step, records) = match &self.window {
                    Some(w) if windowed && same(w) => {
                        let last = w.base + u64::from(w.records - 1) * u64::from(w.step);
                        let step = offset.saturating_sub(last);
                        let walks = step >= u64::from(stride)
                            && step <= 256
                            && step % 4 == 0
                            && (w.records == 1 || step == u64::from(w.step));
                        if walks {
                            let fit = (args.left(arguments) / step).min(WINDOW_MOST) as u32;
                            (step as u32, (if w.records == 1 { 16 } else { w.records * 2 }).min(fit).max(1))
                        } else {
                            (stride, max)
                        }
                    }
                    _ => (stride, max),
                };
                args.refresh(list, arguments, len.max(u64::from(step) * u64::from(records)));
                if let Some((s, at)) = &counter {
                    s.refresh(list, *at, 4);
                }
                // Bounded first: the clamp binds a compute pipeline of its
                // own and copies clamped active records into a scratch
                // buffer at offset 0.
                let count_arg = counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at)));
                // More than the record asked for: a window clamped ahead,
                // whose unfilled slots must not spend the list's budget.
                let bounded = if parts.is_some() {
                    crate::indirect::bound_split(list, (&args.buffer, offset))
                } else if windowed && records > 1 {
                    crate::indirect::bound_ahead(list, (&args.buffer, offset), (kind, step, records))
                } else {
                    crate::indirect::bound(list, (&args.buffer, offset), count_arg, (kind, step, records))
                };
                let Some(bounded) = bounded else {
                    self.window = None;
                    why("not bounded (no clamp, or arguments not 4-byte aligned)");
                    return false;
                };
                self.host_pipeline = 0;
                self.host_roots[1] = 0;
                self.window = windowed.then(|| Window {
                    copy: args.id(),
                    kind,
                    stride,
                    step,
                    base: offset,
                    records,
                    scratch: bounded.clone(),
                });
                (bounded, 0)
            }
        };
        let ran = match kind {
            DRAW | INDEXED => self.prepare(list, kind == INDEXED),
            MESH => self.prepare_for(list, false, true),
            DISPATCH => self.bind(list, p, true),
            _ => false,
        };
        if !ran {
            crate::indirect::done(list, &bounded);
            why("prepare failed");
            return false;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(format!("indirect kind {kind} max {max} at {arguments:#x}"), kind == DISPATCH)
        });
        if kind == MESH {
            self.mesh_scissor(list, true);
        }
        // SAFETY: an open list with everything set; the scratch buffer and
        // host copies are kept alive by the list, and the signature is cached.
        unsafe {
            if let Some(((param, dword), signature)) = &parts {
                for part in 0..crate::indirect::SPLIT {
                    list.list.SetComputeRoot32BitConstant(*param, part * crate::indirect::PART, *dword);
                    let at = u64::from(part) * crate::indirect::PART_BYTES;
                    list.list.ExecuteIndirect(signature, 1, &bounded, at, None, 0);
                }
                list.list.SetComputeRoot32BitConstant(*param, 0, *dword);
            } else {
                list.list.ExecuteIndirect(
                    &signature,
                    max,
                    &bounded,
                    at,
                    counter.as_ref().map(|(s, _)| &s.buffer),
                    counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
                )
            }
        };
        if kind == MESH {
            self.mesh_scissor(list, false);
        }
        crate::indirect::done(list, &bounded);
        if kind == DISPATCH {
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
        } else {
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        true
    }
}

/// Where a compute pipeline's program takes the number of its first thread
/// group (`victory_shader::emit::USER_GROUP_BASE`): the root parameter of
/// its user data and the dword in it.
fn group_base(p: &crate::pipeline::host::HostPipeline) -> Option<(u32, u32)> {
    compute_word(p, victory_shader::emit::USER_GROUP_BASE)
}

fn compute_word(p: &crate::pipeline::host::HostPipeline, kind: u32) -> Option<(u32, u32)> {
    let stage = p.stages.first().filter(|_| p.compute)?;
    let (param, _) = stage.user_data?;
    let base = RootType::Other(kind);
    let word = stage.translated().host.user_data.iter().find(|u| u.root == base)?;
    Some((param, word.index))
}

/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once.
fn host_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> =
        std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe {
        crate::host::host()?.device.CreateCommandSignature(
            &desc,
            None::<&ID3D12RootSignature>,
            &mut signature,
        )
    };
    if let Err(e) = result {
        once!(error = %e, kind, stride, "host command signature refused");
        return None;
    }
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

/// Gets what a shader reads or writes through a view ready: buffer data
/// copied, textures moved to the state they need.
fn view_kind(view: &View) -> Option<ResourceType> {
    match view {
        View::Shader(..) | View::Placed(false, ..) => Some(ResourceType::Srv),
        View::Unordered(..) | View::Placed(true, ..) => Some(ResourceType::Uav),
        _ => None,
    }
}

/// The host state a depth view needs of its texture: the read state if the
/// view writes no plane the format has (the title's read-only flags), else
/// the write state.
/// shortcut: whole textures have one state, so a view that writes one of
/// two planes counts as writing.
fn depth_state(texture: &ID3D12Resource, desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>) -> u32 {
    // D16_UNORM and D32_FLOAT: depth formats without a stencil plane.
    const ONE_PLANE: [i32; 2] = [55, 40];
    let Some(desc) = desc else {
        return DEPTH_WRITE;
    };
    let format = if desc.Format.0 == 0 {
        // SAFETY: a live texture; a view without a format has its own.
        unsafe { texture.GetDesc() }.Format.0
    } else {
        desc.Format.0
    };
    let mut needed = D3D12_DSV_FLAG_READ_ONLY_DEPTH.0;
    if !ONE_PLANE.contains(&format) {
        needed |= D3D12_DSV_FLAG_READ_ONLY_STENCIL.0;
    }
    if desc.Flags.0 & needed == needed {
        DEPTH_READ
    } else {
        DEPTH_WRITE
    }
}

fn readable(list: &HostList, kind: ResourceType, view: &View) {
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    if let Some((shadow, address, len)) = crate::descriptor::host::buffer_copy(view) {
        shadow.refresh(list, address, len);
        if kind == ResourceType::Uav {
            // A shader that cannot store to buffers leaves the range as
            // the title has it: nothing to find afterwards.
            if STORES.get() || !write_skip() {
                shadow.gpu_writes(list, address, len);
            } else {
                #[cfg(feature = "diagnostics")]
                {
                    crate::copystat::add(crate::copystat::C::GpuWritesLeftOut, 1);
                    crate::copystat::add(crate::copystat::C::GpuWritesLeftOutBytes, len);
                }
            }
        }
    }
    match (view, kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            // A view the title placed, in another format, in the memory of
            // a texture the host writes: that texture's texels, copied on
            // the host (title memory does not have them).
            let aliased = crate::resource::alias_of(*resource as This)
                .is_some_and(|source| crate::texture::alias(list, *resource as This, source));
            if !aliased
                && let Some(t) = crate::resource::host_resource(*resource as This)
                && crate::texture::title_written(t)
            {
                // Of a texture the title placed, the mips the view shows.
                let first = match (&placed, view) {
                    (Some(_), View::Shader(_, Some(desc))) => crate::resource::first_mip(desc),
                    _ => 0,
                };
                crate::texture::upload_from(list, *resource as This, t, first);
            }
            // A depth texture shaders also write: what they wrote, first.
            list.twin_use(*resource as This, false, false);
            // The depth target the draw only tests is read in both ways at
            // once: both read states, which its preparation asks for too.
            let (tested, _) = TESTED_DEPTH.get();
            let state = if tested != 0 && tested == *resource {
                TESTED_DEPTH.set((tested, true));
                DEPTH_READ | SHADER_RESOURCE
            } else {
                SHADER_RESOURCE
            };
            crate::list::ensure_state(list, *resource as This, state);
            #[cfg(feature = "diagnostics")]
            if let Some(t) = crate::resource::host_resource(*resource as This) {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Flags.0 & 7 == 0 {
                    tracing::debug!(
                        resource = format_args!("{resource:#x}"),
                        ?d,
                        "a shader reads a texture the title fills"
                    );
                }
            }
        }
        (View::Unordered(resource, _, _), ResourceType::Uav) => {
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(*resource as This);
            // Unordered access to a depth texture is its colour twin's.
            if !list.twin_use(*resource as This, true, true) {
                crate::list::ensure_state(list, *resource as This, UNORDERED_ACCESS);
            }
        }
        _ => {}
    }
}

/// A view in a few words (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn describe(view: Option<&View>) -> String {
    let texture = |r: This| {
        crate::resource::host_made(r).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            format!("{}x{} f{} flags {:#x} of {r:?}", d.Width, d.Height, d.Format.0, d.Flags.0)
        })
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This)
            .unwrap_or_else(|| format!("buffer {:?}", crate::descriptor::host::buffer_range(v))),
        Some(v @ View::Unordered(r, ..)) => match texture(*r as This) {
            Some(t) => format!("uav texture {t}"),
            None => format!(
                "uav resource {r:#x} buffer {:?}",
                crate::descriptor::host::buffer_range(v)
            ),
        },
        Some(View::Placed(u, r, d)) => format!(
            "placed uav={u} dim {} {}x{} f{} layout {:#x} at {:#x}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32
        ),
        // The first four values, as a colour rectangle's constants are.
        Some(View::Constant(address, size))
            if *size >= 16 && crate::texture::committed(*address as *const u8, 16) =>
        {
            // SAFETY: 16 committed bytes of title memory.
            let v = unsafe { (*address as *const [f32; 4]).read_unaligned() };
            format!("Constant({address:#x}, {size}) {v:?}")
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

/// The console's own primitive topologies.
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

/// Fixed-function state, straight onto the host list.
pub(crate) fn viewports(list: &HostList, viewports: &[D3D12_VIEWPORT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetViewports(viewports) };
}

pub(crate) fn scissors(list: &HostList, rects: &[RECT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetScissorRects(rects) };
}

pub(crate) fn blend_factor(list: &HostList, factor: Option<&[f32; 4]>) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetBlendFactor(factor) };
}

pub(crate) fn stencil_ref(list: &HostList, reference: u32) {
    #[cfg(feature = "diagnostics")]
    crate::capture::op(std::ptr::from_ref(list) as usize, || format!("STENCIL REF {reference:#x}"));
    // SAFETY: an open list.
    unsafe { list.list.OMSetStencilRef(reference) };
}

pub(crate) fn depth_bounds(list: &HostList, min: f32, max: f32) {
    #[cfg(feature = "diagnostics")]
    crate::capture::op(std::ptr::from_ref(list) as usize, || {
        format!("DEPTH BOUNDS {min}..{max}")
    });
    if crate::host::host().is_some_and(|h| h.depth_bounds())
        && let Ok(l1) = list.list.cast::<ID3D12GraphicsCommandList1>()
    {
        // SAFETY: an open list on a device that supports depth bounds.
        unsafe { l1.OMSetDepthBounds(min, max) };
    }
}

/// One stage's Metal bindings ([`State::metal_stage`]).
#[cfg(windows)]
pub(crate) struct MetalStage {
    top: Vec<u8>,
    tables: Vec<(u32, victory_metal::ir::DescriptorEntry)>,
    heap: Option<victory_metal::Buffer>,
    sampler_heap: Option<victory_metal::Buffer>,
    outcome: crate::backend::metal::Drop,
}

#[cfg(windows)]
thread_local! {
    /// Heap slots read by the stages of the draw being recorded (the
    /// final-draw dossier).
    static SLOTS: std::cell::RefCell<Vec<(u32, u64, Option<crate::backend::metal_heap::Missing>)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The D3D12 path's buffer for `victory_shader::emit::SLOT_DIAG` (made once,
/// kept for the process): its GPU address, 0 without one.
#[cfg(windows)]
fn d3d12_diag() -> u64 {
    static GPU: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *GPU.get_or_init(|| {
        let Some(host) = crate::host::host() else { return 0 };
        let bytes = u64::from(victory_shader::emit::DIAG_DWORDS) * 4;
        let Some((buffer, data)) = host.shared_buffer(bytes) else { return 0 };
        // SAFETY: a fresh mapping of that many bytes.
        unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        std::mem::forget(buffer);
        gpu
    })
}
