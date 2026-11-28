//! DXIL emission (decision 0003, step 3): a console vertex, pixel, compute
//! or mesh shader as a PC DXIL container (SM 6.6) that host D3D12 accepts.
//! Compute shaders keep the console's thread-group size; one that depends
//! on the lane layout (cross-lane operations) declares the console's wave
//! size (`WaveSize`), so the host device must support it.
//!
//! What the host binds for a translated shader ([`HostShader`]):
//!
//! * **User data** at `b0, space 0` (32-bit root constants or a constant
//!   buffer, per stage): dword `index` holds a root parameter's value, as
//!   [`UserData`] lists. A root-constant dword is the constant. A
//!   descriptor table is the byte offset of its first descriptor in its
//!   heap: the descriptor index times 32 for CBV/SRV/UAV tables, times 16
//!   for sampler tables (the console's descriptor sizes; the code adds
//!   offsets in those units). The heap bases of dynamic resources are 0
//!   and not listed.
//! * **Descriptors** through SM 6.6 `ResourceDescriptorHeap[]` and
//!   `SamplerDescriptorHeap[]`, at the index the console address gives, so
//!   the host heaps keep the title's descriptor order. The root signature
//!   needs `CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED` and
//!   `SAMPLER_HEAP_DIRECTLY_INDEXED`.
//! * **Immediate samplers** as static samplers `s<n>, space 0`, `n` the
//!   position in [`HostShader::samplers`] (the console sampler words) and
//!   [`HostShader::static_samplers`] (the same decoded into a D3D12 static
//!   sampler description). Comparison sampling (added 2026-11-23): an
//!   immediate sampler it uses has a comparison filter there (bit 0x80);
//!   a heap sampler it uses must be a comparison sampler on the host.
//! * **Root CBVs and root SRVs** (raw buffers) as root descriptors
//!   `b<i>` / `t<i>` in [`ROOT_CBV_SPACE`], `i` the title's root parameter
//!   index ([`HostShader::root_cbvs`], [`HostShader::root_srvs`]); the
//!   address the title sets is the buffer's.
//! * **Vertex fetch** (added 2026-11-23). The title's mesh and
//!   vertex programs fetch vertices themselves with typed-format loads
//!   (`tbuffer_load_format_*`): the format (8_8_8_8 unorm, 16_16 snorm,
//!   2_10_10_10 unorm, 8_8_8_8 and 16_16_16_16 uint...) is in the
//!   instruction, so the translation reads raw bytes and converts them
//!   itself. Their descriptors are built around a root SRV (stride 1, no
//!   record limit): the host binds the root SRV `t<slot>` above with the
//!   address the title sets, nothing more. The vertex index, the stride
//!   and the attribute offset are the shader's own arithmetic on data in
//!   that buffer (per-stream headers), not descriptor fields.
//! * **Buffer views in the heap** are declared by how the shader reads
//!   them: raw (ByteAddressBuffer) for dword loads and stores, atomics and
//!   typed-format loads without an index; structured with the used-resource
//!   entry's stride when indexed; typed with four 32-bit float channels for
//!   `buffer_load_format_*` and `buffer_store_format_*`. The host's view at
//!   that index must be of that kind and start at the console descriptor's
//!   base address (raw: `R32_TYPELESS` with the raw flag), whatever kind the
//!   title's view description says. Typed-format loads from heap
//!   descriptors (1,024 world compute programs, ray-tracing material
//!   resolves) read raw views this way.
//!
//! * **Texture atomics** use a scalar integer UAV. The host binds R32_UINT
//!   for 32-bit operations and R32G32_UINT for 64-bit operations. The latter
//!   require SM 6.6 atomic support on typed and descriptor-heap resources.
//!
//! Entry points: [`translate`] (container bytes as the title passes them
//! at pipeline creation → [`HostShader`]), [`emit`] (the same with the
//! earlier stages already run). The root constants of each stage must
//! cover [`HostShader::user_data_dwords`]. Stability: these types and the
//! binding scheme are what pipeline creation builds on; changes are
//! announced.
//! * **Signatures** as the console container has them (same semantics,
//!   registers and masks), so the input layout and stage linking are the
//!   title's; pixel outputs are plain 32-bit values.
//! * **Mesh shaders** ([`HwStage::Ms`], `ms_6_6`): user data and
//!   descriptors as above (the root constants visible to the mesh stage);
//!   the pipeline is a mesh pipeline (pipeline-state stream, no input
//!   layout) with the title's pixel shader. The thread group is the
//!   console's in lanes (`[numthreads]` in the DXIL), the vertex and
//!   primitive maxima its own, triangles only, no wave size required. The
//!   console numbers groups in x only: the draw path passes the title's
//!   count as `DispatchMesh(x, 1, 1)`. Per-primitive outputs (`PSG1`) are
//!   the console's, without `SV_PrimitiveID` and `SV_CullPrimitive` (the
//!   console never exports them).
//! * **Rectangle lists** ([`rect_list_gs`], `gs_6_6`): the console draws a
//!   rectangle from three vertices; the host has no such topology. A
//!   pipeline the title makes for rectangles (topology type 19) gets a
//!   geometry shader of ours between the translated vertex and pixel
//!   shaders and draws triangle lists with the title's own vertex counts.
//!   It binds nothing.

mod bitcode;
mod container;
#[cfg(all(test, windows))]
mod cube_lod_tests;
#[cfg(all(test, windows))]
mod quad_edge_tests;
#[cfg(all(test, windows))]
mod diff_tests;
#[cfg(all(test, windows))]
mod intersect_tests;
#[cfg(all(windows, any(test, feature = "host")))]
pub mod host;
mod lower;
mod own_bits;
mod rect;
mod sampler;
mod shader;
mod threading;
#[cfg(windows)]
pub mod validator;
#[cfg(test)]
mod lds_start_tests;
#[cfg(test)]
mod shader_gaps_tests;
#[cfg(all(test, windows))]
mod valu_tests;
#[cfg(all(test, windows))]
mod waterfall_tests;
#[cfg(all(test, windows))]
mod wave64_tests;
mod wave;
mod water_tile_slots;

use crate::dxbc::{Container, FourCC};
use crate::native::{HwStage, NativeProgram, RootType};
use crate::{ShaderError, io, ir, resource, structure};

pub use bitcode::census;
pub use sampler::StaticSampler;
pub use lower::{
    RECT_PRIMITIVE_VARYING, host_attribute_at_vertex, host_has_image_atomic64, host_max_wave, rect_primitive_varying,
    set_host_attribute_at_vertex, set_host_f64, set_rect_primitive_varying, set_mesh_outputs_keep, mesh_outputs_keep_active,
    input_reads, set_mesh_output_reads, last_clip_alias, set_mesh_drop_clip, set_pixel_clip_inputs,
    set_host_image_atomic64, set_host_max_wave, host_has_f64,
};

/// The translation settings of the Metal backend (Metal Shader Converter, Apple GPUs), as
/// `victory_gpu::host` sets them for a run, `victory warm --metal` for the cache it fills
/// and the MSC checks for what they convert: waves of 32, no pixel-stage 64-bit image
/// atomics, no `attributeAtVertex`, and no `double` (Apple GPUs have none; MSC aborts the
/// whole process on a `WaveActiveSum` of doubles: "Unhandled LLVM type for
/// getMinMaxConstants").
pub fn set_metal_host() {
    set_host_max_wave(32);
    set_host_image_atomic64(false);
    set_host_attribute_at_vertex(false);
    set_host_f64(false);
}
pub use rect::rect_list_gs;
pub use shader::{
    ATTR_TABLE_ROWS, ATTR_TABLE_VERTEX_STRIDE, DIAG_DWORDS, DIAG_HALF_SYNC_TIMEOUTS, DIAG_LOOP_BUDGET,
    DIAG_LOOP_DWORDS, DIAG_LOOP_VALUES, DIAG_LOOPS, DIAG_LOOPS_MAX, DIAG_SITES, MEMORY_FAULTS, MEMORY_PAGES, ROOT_CBV_SPACE,
    SAMPLER_SPACE, SLOT_ATTR_SCRATCH, SLOT_ATTR_TABLE, SLOT_DIAG, SLOT_FAULTS, SLOT_MEMORY, SLOT_PRIM_IBO,
    SLOT_STRIDES, SLOT_STRIDES_COUNT, SLOT_STRIDES_DWORDS, STRIDES_VIEW_ADDRESSES, SLOT_WAVE_SCRATCH, USER_DATA_SPACE, USER_DISPATCH_THREADS_X,
    USER_GROUP_BASE, USER_GROUP_COUNT_X, USER_GROUP_COUNT_Y, USER_MEMORY_BASE, USER_VIEW_ADDRESS,
    USER_VIEW_BIAS, VIEW_INTEGER, VIEW_SIGNED, VIEW_STRIDE_BITS, WAVE_SCRATCH_GROUP_DWORDS,
    set_view_heap_limit, view_heap_limit,
};

/// The loop limit without an override: 2^20.
pub const DEFAULT_LOOP_LIMIT: u32 = lower::LOOP_LIMIT;

/// The iteration count after which a translated loop is left, per lane and
/// per entry into the loop: [`DEFAULT_LOOP_LIMIT`].
///
/// With the `diagnostics` feature, `VICTORY_SHADER_LOOP_LIMIT` (decimal or
/// `0x` hex, at least 1) sets it instead. A loop that hits the limit in
/// every lane of a full-screen pass still runs for seconds (2^20 times a
/// 4K screen is about 10^13 iterations); a much lower limit shows whether
/// such loops are what makes a frame slow. The value is part of the
/// translation, so the shader cache must key on it (`victory-gpu`'s does).
pub fn loop_limit() -> u32 {
    static LIMIT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        if cfg!(feature = "diagnostics") {
            parse_loop_limit(std::env::var("VICTORY_SHADER_LOOP_LIMIT").ok().as_deref())
        } else {
            lower::LOOP_LIMIT
        }
    })
}

/// Which of a translated program's group barriers wait for the whole
/// group (`GroupMemoryBarrierWithGroupSync`); the others keep their memory
/// fence and lose the wait. A wait that some thread of the group never
/// reaches is undefined on the host, and NVIDIA's wait is for every
/// thread: the group stands until Windows resets the card. (The console's
/// barrier is the wave's, whatever lanes are live, and a wave that ended
/// is not waited for.) `VICTORY_GROUP_SYNC`:
/// - unset or `2`: every barrier waits, as before.
/// - `1`: the title's `s_barrier` waits only in a block every lane runs,
///   in a program no lane can be cut out of at the iteration limit.
/// - `0`: no barrier waits, neither the title's nor the ones the
///   translation adds (wave64 programs in two host waves exchange through
///   them, so such programs compute wrong values). For one run that tells
///   whether a barrier is what stops the card: with it no run can stand in
///   a barrier.
///
/// Part of the translation: the shader cache keys on it (`victory-gpu`'s
/// does).
pub fn group_sync() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_GROUP_SYNC").ok().as_deref().map(str::trim) {
        Some("0") => 0,
        Some("1") => 1,
        _ => 2,
    })
}

/// The clean-up passes that run over a translated program before it is
/// written (`bitcode::opt`): a set of that module's pass bits. Every pass
/// is exact, so the switch is for telling a translator fault from a
/// clean-up fault, and for measuring what the clean-up is worth.
/// `VICTORY_SHADER_OPT`:
/// - unset: every pass.
/// - `0`: none (the module as the lowering made it).
/// - a number (decimal or `0x` hex): those passes; 1 dead code, 2 constants
///   and identities, 4 one instruction for one value, 8 what a branch taken
///   says about its condition, 16 one read for one place within a block,
///   32 a first-lane read of two host waves without the round that asks
///   which half has a lane (where every lane has one).
///
/// Part of the translation: the shader cache keys on it (`victory-gpu`'s
/// does).
pub fn shader_opt() -> u32 {
    static PASSES: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *PASSES.get_or_init(|| parse_shader_opt(std::env::var("VICTORY_SHADER_OPT").ok().as_deref()))
}

/// `VICTORY_SHADER_OPT`'s value; every pass when it is unset or
/// unreadable. Pure, for testing.
fn parse_shader_opt(value: Option<&str>) -> u32 {
    let parsed = value
        .map(str::trim)
        .and_then(|v| match v.strip_prefix("0x") {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => v.parse().ok(),
        });
    parsed.map_or(bitcode::opt::ALL, |n| n & bitcode::opt::ALL)
}

/// `VICTORY_SHADER_LOOP_LIMIT`'s value; the default when it is unset,
/// unreadable or 0. Pure, for testing.
fn parse_loop_limit(value: Option<&str>) -> u32 {
    let parsed = value
        .map(str::trim)
        .and_then(|v| match v.strip_prefix("0x") {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => v.parse().ok(),
        });
    match parsed {
        Some(n) if n > 0 => n,
        _ => lower::LOOP_LIMIT,
    }
}

/// A translated shader and what the host binds for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostShader {
    pub stage: HwStage,
    /// The DXIL container, hashed as D3D12 expects.
    pub dxil: Vec<u8>,
    /// The user-data dwords the shader reads.
    pub user_data: Vec<UserData>,
    /// Immediate samplers: the four sampler words of `s<n>`.
    pub samplers: Vec<[u32; 4]>,
    /// The same samplers decoded, `s<n>` at position n: the static sampler
    /// descriptions the host's root signature declares.
    pub static_samplers: Vec<StaticSampler>,
    /// Root CBVs the shader reads: root parameter indices, each bound as
    /// `b<index>` in [`ROOT_CBV_SPACE`].
    pub root_cbvs: Vec<u8>,
    /// Root SRVs (raw buffers) the shader reads: `t<index>` in
    /// [`ROOT_CBV_SPACE`].
    pub root_srvs: Vec<u8>,
    /// Root UAVs (raw buffers) the shader reads or writes: `u<index>` in
    /// [`ROOT_CBV_SPACE`].
    pub root_uavs: Vec<u8>,
    /// Heap descriptors that can be selected from root state at bind time.
    pub root_indexed_views: Vec<crate::resource::RootIndexedView>,
}

impl HostShader {
    /// Size of the user-data buffer the shader declares, in dwords (a
    /// multiple of 4; 0 when it reads none). Root constants must cover it.
    pub fn user_data_dwords(&self) -> u32 {
        let n = self
            .user_data
            .iter()
            .map(|u| u.index + 1)
            .max()
            .unwrap_or(0);
        n.next_multiple_of(4)
    }
}

/// One dword of user data and the root parameter that fills it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserData {
    /// Dword in the user-data buffer (the console's user-data index).
    pub index: u32,
    /// Root parameter index in the title's root signature.
    pub slot: u8,
    /// `Constants` or `Table` (see the module notes for the value).
    pub root: RootType,
    /// Dword within the parameter.
    pub dword: u32,
}

thread_local! {
    static LAST_WAVE: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
}

/// The wave size the last program emitted on this thread declared (`None`: none), for
/// translation surveys: after a successful [`translate`], the size of the shader it returned.
pub fn last_wave() -> Option<u32> {
    LAST_WAVE.get()
}

/// Brings a console program into the form this host translates, where one differs from the
/// program as the title has it: on a host of 32-lane waves the water light's tile lists take
/// a slot a lane ([`water_tile_slots`], upstream cd93552). [`translate`] does this itself; a
/// caller that parses the container and calls [`emit`] calls this first. Returns whether the
/// program changed.
pub fn host_form(c: &Container, native: &mut NativeProgram) -> bool {
    lower::host_max_wave() == 32
        && c.digest == water_tile_slots::CONTAINER
        && water_tile_slots::on()
        && water_tile_slots::rewrite(native)
}

/// Translates the vertex, pixel, compute or mesh program of a console
/// shader container.
pub fn translate(container: &[u8]) -> Result<HostShader, ShaderError> {
    let c = Container::parse(container)?;
    if c.part(FourCC::DXIL).is_none() && c.part(FourCC::ACCC).is_none() {
        return Err(ShaderError::Malformed("no DXIL part"));
    }
    // A DXIL part's programs, or an ACCC part's precompiled one (FSPX).
    let mut native = crate::container_programs(&c)?
        .into_iter()
        .find(|p| {
            matches!(
                p.stage,
                HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms
            )
        })
        .ok_or(ShaderError::Unsupported {
            offset: 0,
            what: "a container without a vertex, pixel, compute or mesh program",
        })?;
    host_form(&c, &mut native);
    let p = crate::spill::build(&native)?;
    // Structurisation first: tracing per-lane descriptors needs it (its
    // failure is reported after the earlier stages').
    let st = structure::structure(&p);
    let res = resource::track(&p, &native.layout, native.stage, st.as_ref().ok())?;
    let io = io::raise(&p, &native)?;
    let st = st?;
    emit(&c, &native, &p, &st, &io, &res)
}

/// Translates a compute program given as GFX10 code and layout (no
/// container: no signatures), as [`translate`] does: for tests and repros
/// that build programs by hand.
pub fn translate_native(native: &NativeProgram) -> Result<HostShader, ShaderError> {
    let c = Container { digest: [0; 16], version: (1, 0), parts: Vec::new() };
    let p = crate::spill::build(native)?;
    let st = structure::structure(&p);
    let res = resource::track(&p, &native.layout, native.stage, st.as_ref().ok())?;
    let io = io::raise(&p, native)?;
    let st = st?;
    emit(&c, native, &p, &st, &io, &res)
}

/// Emits a program whose earlier stages already ran.
pub fn emit(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
) -> Result<HostShader, ShaderError> {
    if std::env::var_os("VICTORY_DESC_WHY").is_some()
        && let crate::native::StageIo::Ps { db_shader_control, input_addr, .. } = &native.io
    {
        eprintln!("PS-IO db_shader_control={db_shader_control:#x} pops={} input_addr={input_addr:#x}", db_shader_control >> 19 & 1);
    }
    // `VICTORY_DUMP_CODE=<dir>`: each program's GFX10 code, by its hash,
    // for `llvm-mc --disassemble -triple=amdgcn -mcpu=gfx1030`.
    if let Some(dir) = std::env::var_os("VICTORY_DUMP_CODE") {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        native.code.hash(&mut h);
        let path = std::path::Path::new(&dir).join(format!("{:016x}.bin", h.finish()));
        let _ = std::fs::write(path, &native.code);
    }
    // (what an earlier program's lowering on this thread left behind)
    lower::take_needs();
    let original = emit_program(c, native, p, st, io, res);
    // A console wave64 compute program on a host of 32-lane waves that
    // still needs 64 lanes, or whose cross-lane code sits where not every
    // lane runs: once more with its early-outs left out
    // ([`ir::without_early_outs`]), which is what usually keeps the whole
    // body from being a block every lane runs.
    let stuck = match &original {
        Ok((_, wave)) => *wave == Some(64),
        Err(ShaderError::Unsupported { what, .. }) => lower::needs_every_lane(what),
        Err(_) => false,
    };
    // (a mask with lanes elsewhere is only helped by the exact forms: not by each half-wave
    // on its own)
    let exact_only = matches!(&original, Err(ShaderError::Unsupported { what, .. }) if *what == lower::MASK_ELSEWHERE);
    if stuck && native.stage == HwStage::Cs && native.wave64 && lower::host_max_wave() == 32 {
        let why = std::env::var_os("VICTORY_WAVE_WHY").is_some();
        let report = |what: &str, r: &Result<(HostShader, Option<u32>), ShaderError>| {
            if why {
                match r {
                    Ok((_, wave)) => eprintln!("WAVE64-RETRY {what}: wave {wave:?}"),
                    Err(e) => eprintln!("WAVE64-RETRY {what}: {e}"),
                }
            }
        };
        let done = |r: &Result<(HostShader, Option<u32>), ShaderError>| matches!(r, Ok((_, wave)) if *wave != Some(64));
        // Each host wave as a wave of its own (`lower::with_local_waves`),
        // for programs still stopped where not every lane runs.
        let local_on = std::env::var_os("VICTORY_WAVE_LOCAL").is_none_or(|v| v != "0");
        // Where the wave operations are the allocation idiom
        // ([`lower::wave_local`]) that is exact. Other programs are taken
        // the same way last of all, each half-wave giving its own results
        // (approximate: a tile-wide result becomes two half-tile ones);
        // `VICTORY_WAVE_LOCAL=idiom` refuses those instead.
        let strict = std::env::var_os("VICTORY_WAVE_LOCAL").is_some_and(|v| v == "idiom");
        let idiom = |p: &ir::Program, st: &structure::Structured| match lower::wave_local(p, st) {
            Ok(()) => true,
            Err((offset, what)) => {
                if why {
                    eprintln!("WAVE64-RETRY local waves: not the allocation idiom at {offset:#x}: {what}");
                }
                false
            }
        };
        // The program in two other forms: without its early-outs, and
        // without any branch that only skips code no lane is left for.
        let form = |p: Option<ir::Program>| {
            p.and_then(|p| {
                let st = structure::structure(&p).map_err(|e| if why { eprintln!("WAVE64-FORM structure: {e}") }).ok()?;
                let io = io::raise(&p, native).map_err(|e| if why { eprintln!("WAVE64-FORM io: {e}") }).ok()?;
                let res = resource::track(&p, &native.layout, native.stage, Some(&st))
                    .map_err(|e| if why { eprintln!("WAVE64-FORM resources: {e}") })
                    .ok()?;
                Some((p, st, io, res))
            })
        };
        let forms = [
            ("without early-outs", form(ir::without_early_outs(p, &never_taken(st)))),
            ("without skips", form(ir::without_skips(p, false))),
        ];
        for (name, f) in &forms {
            if let Some((p, st, io, res)) = f {
                let again = emit_program(c, native, p, st, io, res);
                report(name, &again);
                if done(&again) {
                    return again.map(|(shader, _)| shader);
                }
            }
        }
        // With per-console-wave bounded `half_sync` enabled (default on wave32 hosts),
        // widening per-wave branches around cross-lane/LDS code is safe across any
        // multiple of 64 threads without risking hardware barrier hangs.
        // (`VICTORY_WAVE_WIDE=0` or `VICTORY_WAVE_HALF_SYNC=0` disables it.)
        let half_sync = lower::half_sync_on();
        let wide_on = std::env::var_os("VICTORY_WAVE_WIDE")
            .map_or(half_sync, |v| v == "1");
        // exact first: each host wave on its own where that is exact (the idiom, every
        // form; no barriers), then the other exact forms; then approximate
        for exact in [true, false] {
            if !exact {
                // Before anything approximate, the exact forms that cost group barriers.
                // Every skip left out that the code after it cannot see
                // ([`ir::without_skips`], `deep`): most of the body is then code every
                // lane runs, where the two host waves can exchange.
                // (`VICTORY_WAVE_DEEP_SKIPS=0` and `VICTORY_WAVE_WIDE=0` turn the two off.)
                let deep_on = std::env::var_os("VICTORY_WAVE_DEEP_SKIPS").is_none_or(|v| v != "0");
                let deep = if deep_on { form(ir::without_skips(p, true)) } else { None };
                if let Some((p, st, io, res)) = &deep {
                    let again = emit_program(c, native, p, st, io, res);
                    report("without skips, scalars followed", &again);
                    if done(&again) {
                        return again.map(|(shader, _)| shader);
                    }
                }
                // Still stopped where not every lane runs: the per-lane branches around
                // those places taken as the console takes them, by the whole wave at once
                // ([`widen`]), at two group barriers per such branch.
                if wide_on {
                    let bases = [
                        deep.as_ref().map(|f| &f.0),
                        forms[1].1.as_ref().map(|f| &f.0),
                        forms[0].1.as_ref().map(|f| &f.0),
                        Some(p),
                    ];
                    if let Some(base) = bases.into_iter().flatten().next() {
                        let again = widen(c, native, base, why);
                        report("wave-wide branches", &again);
                        if done(&again) {
                            return again.map(|(shader, _)| shader);
                        }
                    }
                }
                if strict {
                    break;
                }
            }
            if !local_on || exact_only {
                continue;
            }
            for (name, f) in &forms {
                if let Some((p, st, io, res)) = f
                    && idiom(p, st) == exact
                {
                    let local = lower::with_local_waves(|| emit_program(c, native, p, st, io, res));
                    report(&format!("local waves ({}), {name}", if exact { "idiom" } else { "APPROXIMATE" }), &local);
                    if done(&local) {
                        return local.map(|(shader, _)| shader);
                    }
                }
            }
            if idiom(p, st) == exact {
                let local = lower::with_local_waves(|| emit_program(c, native, p, st, io, res));
                report(if exact { "local waves (idiom)" } else { "local waves (APPROXIMATE)" }, &local);
                if done(&local) {
                    return local.map(|(shader, _)| shader);
                }
            }
        }
        if local_on && !exact_only {
            // Last: every mask and lane read per host wave.
            if !strict {
                for (name, f) in &forms {
                    if let Some((p, st, io, res)) = f {
                        let local = lower::with_local_waves_any(|| emit_program(c, native, p, st, io, res));
                        report(&format!("local waves (APPROXIMATE, any mask), {name}"), &local);
                        if done(&local) {
                            return local.map(|(shader, _)| shader);
                        }
                    }
                }
                let local = lower::with_local_waves_any(|| emit_program(c, native, p, st, io, res));
                report("local waves (APPROXIMATE, any mask)", &local);
                if done(&local) {
                    return local.map(|(shader, _)| shader);
                }
            }
        }
    }
    // SCC of a mask operation joined with a scalar test behind a per-lane skip (the 32-lane
    // trace programs): right once the wave takes that skip together ([`widen`]; the
    // lowering has named the skipped blocks).
    let original = match original {
        Err(ShaderError::Unsupported { what, .. })
            if native.stage == HwStage::Cs
                && what == lower::LANE_MASK
                && std::env::var_os("VICTORY_WAVE_WIDE").is_none_or(|v| v != "0") =>
        {
            let again = widen(c, native, p, std::env::var_os("VICTORY_WAVE_WHY").is_some());
            match again {
                Ok(_) => again,
                Err(_) => emit_program(c, native, p, st, io, res),
            }
        }
        other => other,
    };
    let original = original.map(|(shader, _)| shader);
    // Retry unsupported programs only. Threading can separate an unknown
    // register value from its apparent use on a mutually exclusive path.
    if !matches!(original, Err(ShaderError::Unsupported { .. })) {
        return original;
    }
    let edges = threading::constant_edges(p, st);
    if edges.is_empty() {
        return original;
    }
    let p = ir::thread_edges(p, &edges)?;
    let st = structure::structure(&p)?;
    let io = io::raise(&p, native)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    emit_program(c, native, &p, &st, &io, &res).map(|(shader, _)| shader)
}

/// Per block: whether no lane reaching it takes its closing branch ([`ir::without_early_outs`]).
fn never_taken(st: &structure::Structured) -> Vec<bool> {
    st.exits
        .iter()
        .map(|e| matches!(e, structure::LaneExit::Branch { when, .. } if *when == structure::Bool::FALSE))
        .collect()
}

/// `p` (a console wave64 compute program on a host of 32-lane waves) with the per-lane
/// branches around its cross-lane code taken wave-wide ([`structure::structure_wide`]): the
/// lowering names the blocks where the two host waves must meet and could not
/// (`lower::take_needs`), the branches whose regions hold them are widened, and so on until
/// it translates or a branch cannot be widened. Only where a thread group is one console
/// wave: with several, the waves of a group would decide differently and a barrier inside
/// such a branch would wait for threads that never come.
fn widen(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    why: bool,
) -> Result<(HostShader, Option<u32>), ShaderError> {
    const NOT: ShaderError = ShaderError::Unsupported {
        offset: 0,
        what: "wave-wide branches in a thread group of several console waves",
    };
    match native.io {
        // (a program whose waves are the host's meets no one: no barrier, any group)
        crate::native::StageIo::Cs(_) if !(native.wave64 && lower::host_max_wave() == 32) => {}
        crate::native::StageIo::Cs(cs)
            if cs.threads.iter().product::<u32>() == 64
                || lower::half_sync_on() && cs.threads.iter().product::<u32>().is_multiple_of(64) => {}
        crate::native::StageIo::Cs(cs) => {
            if why {
                eprintln!("WAVE64-WIDE not tried: thread groups of {:?}", cs.threads);
            }
            return Err(NOT);
        }
        _ => return Err(NOT),
    }
    let mut wide = std::collections::HashSet::new();
    loop {
        let st = structure::structure_wide(p, &wide)?;
        let io = io::raise(p, native)?;
        let res = resource::track(p, &native.layout, native.stage, Some(&st))?;
        lower::take_needs();
        lower::take_widen();
        let r = emit_program(c, native, p, &st, &io, &res);
        let needs = lower::take_needs();
        let own = lower::take_widen();
        // (or translated for 64-lane waves still: shared memory read and written where not
        // every lane runs, whose blocks the lowering names too)
        let stuck = matches!(&r, Err(ShaderError::Unsupported { what, .. })
                if lower::needs_every_lane(what) || *what == lower::LANE_MASK && !needs.is_empty())
            || matches!(&r, Ok((_, Some(64)))) && !needs.is_empty();
        if !stuck {
            if why {
                let mut list: Vec<_> = wide.iter().map(|&b| format!("{b}@{:#x}", p.blocks[b].offset)).collect();
                list.sort();
                eprintln!("WAVE64-WIDE {} branches taken wave-wide: {}", wide.len(), list.join(" "));
            }
            return r;
        }
        let mut more: Vec<usize> = lower::branches_around(&st, &needs)
            .unwrap_or_default()
            .into_iter()
            .filter(|b| !wide.contains(b))
            .collect();
        // (loop exits the lowering wants taken by the whole wave, by their own block)
        for b in own {
            if !wide.contains(&b) && !more.contains(&b) {
                more.push(b);
            }
        }
        if more.is_empty() {
            if why {
                eprintln!("WAVE64-WIDE stopped with {} branches wave-wide: a branch around it cannot be", wide.len());
            }
            return r;
        }
        wide.extend(more);
    }
}

/// The translated program, and the wave size it requires of the host.
fn emit_program(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
) -> Result<(HostShader, Option<u32>), ShaderError> {
    let sig = |fourcc| -> Result<Vec<container::SigElement>, ShaderError> {
        match c.part(fourcc) {
            Some(part) => signature(part.data),
            None => Ok(Vec::new()),
        }
    };
    let pixel = native.stage == HwStage::Ps;
    let mut inputs = sig(FourCC::ISG1)?;
    let mut outputs = sig(FourCC::OSG1)?;
    // Interpolation as DXC declares it by default; pixel inputs that are
    // read get the mode their barycentrics say.
    let interpolated = if pixel { &mut inputs } else { &mut outputs };
    for e in interpolated {
        e.interp = match (e.kind, e.comp) {
            (3, _) => 4,
            // Constant, as DXC declares it (the AMD compiler crashed on a
            // front face without a mode).
            (13, _) => 1,
            // Until a read says which mode (`lower`).
            (28, _) => 2,
            (0 | 6 | 7, 9) => 2,
            (0, _) => 1,
            _ => 0,
        };
    }
    // (DexNom/VIctory 5f4c3df4.)
    if native.stage == HwStage::Vs
        && let Some(psv) = c.part(FourCC::PSV0)
    {
        vertex_output_interpolation(psv.data, &mut outputs)?;
    }
    // With a conservative depth output (SV_DepthLessEqual, GreaterEqual) DXIL
    // wants the position centroid (or per sample), as DXC declares it then.
    if pixel && outputs.iter().any(|e| matches!(e.kind, 18 | 19)) {
        for e in inputs.iter_mut().filter(|e| e.kind == 3) {
            e.interp = 5;
        }
    }
    // Per-primitive outputs are constant. The console never exports a
    // primitive id or cull flag a mesh shader writes (it culls with the
    // primitive export), so those rows are left out.
    let mut prims = sig(FourCC::PSG1)?;
    prims.retain(|e| !matches!(e.kind, 10 | 30));
    for e in &mut prims {
        e.interp = 1;
    }
    // A mesh group's id arrives multiplied by max(threads, vertices,
    // primitives) (the code divides by that); the thread count is in PSV0
    // from runtime info version 2 on.
    // shortcut: older containers fall back to the group's lane count, right
    // when the HLSL group fills its waves (the title's do).
    let mesh_stride = match &native.io {
        crate::native::StageIo::Ms(ms) => {
            let psv = c.part(FourCC::PSV0).map(|p| p.data);
            let threads = psv
                .filter(|d| crate::dxbc::u32_at(d, 0).is_ok_and(|n| n >= 48))
                .and_then(|d| crate::dxbc::u32_at(d, 4 + 36).ok());
            threads.map_or(ms.threads, |t| {
                t.max(ms.max_vertices).max(ms.max_primitives)
            })
        }
        _ => 0,
    };
    let l = lower::lower(&lower::Parts {
        native,
        p,
        st,
        io,
        res,
        inputs,
        outputs,
        prims,
        mesh_stride,
    })?;
    let wave_size = l.wave_size;
    LAST_WAVE.set(wave_size);
    let shader = HostShader {
        stage: native.stage,
        dxil: l.dxil,
        user_data: l.user_data,
        static_samplers: l
            .samplers
            .iter()
            .enumerate()
            .map(|(n, &w)| {
                let mut s = StaticSampler::from_words(w);
                if l.cmp_samplers >> n & 1 != 0 {
                    s.filter |= 0x80;
                }
                s
            })
            .collect(),
        samplers: l.samplers,
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        root_indexed_views: res.root_indexed_views.clone(),
    };
    Ok((shader, wave_size))
}

/// The title's vertex output interpolation lives in PSV0. A float packed
/// beside an integer can be explicitly constant, as can a float on its own.
/// OSG1 supplies the physical rows and columns used to match both tables.
fn vertex_output_interpolation(
    mut data: &[u8],
    outputs: &mut [container::SigElement],
) -> Result<(), ShaderError> {
    const TRUNCATED: ShaderError =
        ShaderError::Malformed("truncated vertex pipeline validation data");
    fn take<'a>(data: &mut &'a [u8], len: usize) -> Result<&'a [u8], ShaderError> {
        let (head, tail) = data.split_at_checked(len).ok_or(TRUNCATED)?;
        *data = tail;
        Ok(head)
    }
    fn size(data: &mut &[u8]) -> Result<usize, ShaderError> {
        Ok(u32::from_le_bytes(take(data, 4)?.try_into().unwrap()) as usize)
    }
    let info_size = size(&mut data)?;
    let info = take(&mut data, info_size)?;
    // PSVRuntimeInfo0 has no signature. PSVRuntimeInfo1 and later share
    // these stage and signature count fields.
    if info.len() < 36 || info[24] != 1 || info[29] == 0 {
        return Ok(());
    }
    let resources = size(&mut data)?;
    if resources != 0 {
        let stride = size(&mut data)?;
        take(&mut data, resources.checked_mul(stride).ok_or(TRUNCATED)?)?;
    }
    let strings = size(&mut data)?;
    take(&mut data, strings)?;
    let indices = size(&mut data)?;
    take(&mut data, indices.checked_mul(4).ok_or(TRUNCATED)?)?;
    let stride = size(&mut data)?;
    if stride < 16 {
        return Err(TRUNCATED);
    }
    let count = usize::from(info[28]) + usize::from(info[29]);
    let elements = take(&mut data, count.checked_mul(stride).ok_or(TRUNCATED)?)?;
    for record in elements.chunks_exact(stride).skip(usize::from(info[28])) {
        let packed = record[10];
        if packed & 0x40 == 0 {
            continue;
        }
        if record[13] > 7 {
            return Err(ShaderError::Malformed("invalid vertex output interpolation"));
        }
        let first = u32::from(record[9]);
        let rows = first..first + u32::from(record[8]);
        for output in outputs.iter_mut().filter(|e| {
            e.row.is_some_and(|r| rows.contains(&r))
                && e.col == ((packed >> 4) & 3)
                && e.cols == (packed & 15)
                && e.kind == record[11]
        }) {
            output.interp = record[13];
        }
    }
    Ok(())
}

/// The rows of an `ISG1`/`OSG1` part.
/// The (semantic name, index) of each input a host pixel shader declares
/// (its `ISG1`), user semantics only.
pub fn input_semantics(dxil: &[u8]) -> Option<Vec<(String, u32)>> {
    let c = Container::parse(dxil).ok()?;
    let part = c.part(FourCC::ISG1)?;
    let elements = signature(part.data).ok()?;
    Some(elements.into_iter().filter(|e| e.kind == 0).map(|e| (e.name, e.index)).collect())
}

/// Every element (name, index, kind) of a host shader's output signature
/// and, for a mesh shader, its primitive signature.
pub fn output_semantics(dxil: &[u8]) -> Option<Vec<(String, u32, u8)>> {
    let c = Container::parse(dxil).ok()?;
    let mut out = Vec::new();
    for fourcc in [FourCC::OSG1, FourCC::PSG1] {
        if let Some(part) = c.part(fourcc) {
            out.extend(signature(part.data).ok()?.into_iter().map(|e| (e.name, e.index, e.kind)));
        }
    }
    Some(out)
}

fn signature(part: &[u8]) -> Result<Vec<container::SigElement>, ShaderError> {
    use crate::dxbc::u32_at;
    let count = u32_at(part, 0)? as usize;
    let base = u32_at(part, 4)? as usize;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = base + 32 * i;
        let name_at = u32_at(part, at + 4)? as usize;
        let name = part
            .get(name_at..)
            .and_then(|s| s.split(|&b| b == 0).next())
            .ok_or(ShaderError::Malformed("signature name out of bounds"))?;
        let mask = *part
            .get(at + 24)
            .ok_or(ShaderError::Malformed("truncated"))?;
        if u32_at(part, at + 28)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a minimum-precision signature element",
            });
        }
        if u32_at(part, at)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a signature element of another stream",
            });
        }
        let kind = match u32_at(part, at + 12)? {
            0 => 0,
            1 => 3,
            2 => 6,
            3 => 7,
            4 => 4,
            5 => 5,
            6 => 1,
            7 => 10,
            8 => 2,
            9 => 13,
            10 => 12,
            64 => 16,
            65 => 17,
            66 => 14,
            67 => 19,
            68 => 18,
            69 => 20,
            70 => 15,
            23 => 28, // barycentrics
            25 => 30, // cull primitive
            _ => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: "a system value in the signature",
                });
            }
        };
        // 16-bit elements are declared 32-bit; reading or writing one
        // converts (16-bit values travel packed on the console).
        let raw = u32_at(part, at + 16)?;
        let comp = match (kind, raw) {
            // DXIL declares the front face a uint (the validator insists).
            (13, _) => 5,
            (_, 1 | 4) => 5,
            (_, 2 | 5) => 4,
            (_, 3 | 6) => 9,
            (_, c) => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: match c {
                        0 => "a signature element of unknown type",
                        _ => "a 64-bit signature element",
                    },
                });
            }
        };
        let register = u32_at(part, at + 20)?;
        out.push(container::SigElement {
            name: String::from_utf8_lossy(name).into_owned(),
            index: u32_at(part, at + 8)?,
            kind,
            comp,
            interp: 0,
            row: (register != u32::MAX).then_some(register),
            col: mask.trailing_zeros().min(3) as u8,
            cols: mask.count_ones().max(1) as u8,
            usage: 0,
            half: (4..=6).contains(&raw),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::bitcode::Attr;
    use super::lower::LOOP_LIMIT;
    use super::parse_loop_limit;

    #[test]
    fn the_clean_up_switch_parses_or_runs_every_pass() {
        let all = super::bitcode::opt::ALL;
        assert_eq!(parse_shader_opt(None), all);
        assert_eq!(parse_shader_opt(Some("0")), 0);
        assert_eq!(parse_shader_opt(Some(" 1 ")), 1);
        assert_eq!(parse_shader_opt(Some("0x1")), 1);
        assert_eq!(parse_shader_opt(Some("0xffffffff")), all);
        assert_eq!(parse_shader_opt(Some("on")), all);
    }

    #[test]
    fn the_loop_limit_switch_parses_or_falls_back() {
        assert_eq!(parse_loop_limit(None), LOOP_LIMIT);
        assert_eq!(parse_loop_limit(Some("4096")), 4096);
        assert_eq!(parse_loop_limit(Some(" 0x1000 ")), 4096);
        assert_eq!(parse_loop_limit(Some("1")), 1);
        for bad in ["", "0", "-5", "lots", "0x", "4294967296"] {
            assert_eq!(parse_loop_limit(Some(bad)), LOOP_LIMIT, "{bad:?}");
        }
    }
    use super::container::SigElement;
    use super::shader::Shader;
    use super::*;
    use crate::native::{ComputeIo, StageIo};

    /// A pixel shader writing SV_StencilRef (channel y of `exp mrtz`).
    /// DXC sets ShaderFlags.StencilRef (bit 11) for a pixel shader whose
    /// output signature has SV_StencilRef, and the feature bit
    /// D3D_SHADER_FEATURE_STENCIL_REF (0x200) in SFI0; the validator checks
    /// both against the module.
    #[test]
    fn a_stencil_export_declares_the_stencil_ref_feature() {
        let words = [
            0x7E00_02F2u32, // v_mov_b32 v0, 1.0
            0x7E02_0285,    // v_mov_b32 v1, 5
            0xF800_1883,
            0x0000_0100, // exp mrtz v0, v1, off, off done vm
            0xBF81_0000,
        ];
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let el = |name: &str, kind, comp| SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp,
            interp: 0,
            row: None,
            col: 0,
            cols: 1,
            usage: 1,
            half: false,
        };
        let osg1 =
            container::signature(&[el("SV_Depth", 17, 9), el("SV_StencilRef", 20, 5)], false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let out = emit(&c, &native, &p, &st, &io, &res).unwrap();
        // dxil.dll, where there is one, checks the flags against the module.
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut out.dxil.clone()).unwrap();
        }
        let host = Container::parse(&out.dxil).unwrap();
        let sfi0 = host.part(FourCC::SFI0).unwrap().data;
        let features = u64::from_le_bytes(sfi0.try_into().unwrap());
        assert_ne!(features & 0x200, 0, "SFI0 {features:#x}: no StencilRef");
    }

    /// A wave64 compute program that writes only EXEC's high dword
    /// (`s_mov_b32 exec_hi, 0`) before a vector instruction. The new mask was
    /// opaque and keyed by the unchanged low dword, so building its lane bit
    /// needed the mask's own value: lowering recursed until the stack
    /// overflowed, which takes the whole process down (found by
    /// random-input testing). Now that low dword, seen with two high
    /// dwords, has no mask, and the instruction fails like any other that
    /// cannot be expressed: its result is missing, and only a side effect
    /// that uses it would stop the translation.
    #[test]
    fn an_exec_high_write_does_not_overflow_the_stack() {
        // s_mov_b32 exec_hi, 0; v_mov_b32 v1, v0; s_endpgm
        let words = [0xBEFF_0380u32, 0x7E02_0300, 0xBF81_0000];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let container = crate::dxbc::build(&[]);
        let c = Container::parse(&container).unwrap();
        if let Err(e) = emit(&c, &native, &p, &st, &io, &res) {
            assert!(e.to_string().contains("its own value"), "{e}");
        }
    }

    /// A wave64 compute program: `body`, then a store of v1 to a UAV.
    fn stores_v1(body: &[u32]) -> Result<(), ShaderError> {
        use crate::native::RootParam;
        use crate::{ResourceType, RootType, UsedResource};
        // s_mov_b32 s1, 0; s_load_dwordx4 s[8:11], s[0:1], 0
        let mut words = vec![0xBE81_0380, 0xF408_0200, 0xFA00_0000];
        words.extend_from_slice(body);
        // v_lshlrev_b32 v3, 2, v0; buffer_store_dword v1, v3, s[8:11], 0 offen;
        // s_endpgm
        words.extend_from_slice(&[0x3406_0082, 0xE070_1000, 0x8002_0103, 0xBF81_0000]);
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 4,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 2,
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, None).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        let container = crate::dxbc::build(&[]);
        let c = Container::parse(&container).unwrap();
        emit(&c, &native, &p, &st, &io, &res).map(|_| ())
    }

    /// The high dword of a wave64 mask rewritten alone (`s_mov_b32 vcc_hi`
    /// or `exec_hi`): lanes 32-63 took the mask the low dword was first
    /// written with (the compare's, EXEC's on entry). Now the store refuses;
    /// without the write it translates.
    #[test]
    fn a_mask_whose_high_dword_is_rewritten_is_refused() {
        // v_cmp_eq_u32 vcc, 0, v0; v_mov_b32 v2, 11; v_cndmask_b32 v1, 0, v2, vcc
        let select = [0x7D84_0080, 0x7E04_028B, 0x0202_0480];
        // The same with v_add_co_ci_u32 v1, vcc, 0, v2, vcc.
        let carry = [0x7D84_0080, 0x7E04_028B, 0x5002_0480];
        // v_mov_b32 v1, 7
        let mov = [0x7E02_0287];
        for body in [&select[..], &carry, &mov] {
            stores_v1(body).unwrap();
        }
        // s_mov_b32 vcc_hi, 0 after the compare.
        let select_hi = [0x7D84_0080, 0xBEEB_0380, 0x7E04_028B, 0x0202_0480];
        let carry_hi = [0x7D84_0080, 0xBEEB_0380, 0x7E04_028B, 0x5002_0480];
        // s_mov_b32 exec_hi, 0; v_mov_b32 v1, 7, then the same after
        // s_mov_b64 s[2:3], exec.
        let exec_hi = [0xBEFF_0380, 0x7E02_0287];
        let exec_hi_saved = [0xBE82_047E, 0xBEFF_0380, 0x7E02_0287];
        for body in [&select_hi[..], &carry_hi, &exec_hi, &exec_hi_saved] {
            assert!(stores_v1(body).is_err(), "{body:x?}");
        }
    }

    /// `ds_bpermute_b32` stores nothing: each lane reads the lane its address
    /// names. It stopped as "LDS the program does not allocate" (sticky lips)
    /// or "an LDS operation" (the checkerboard shadow masks).
    #[test]
    fn a_backward_permute_reads_the_lane_its_address_names() {
        // v_mov_b32 v2, 11; ds_bpermute_b32 v1, v0, v2
        stores_v1(&[0x7E04_028B, 0xDACC_0000, 0x0100_0200]).unwrap();
    }

    /// A carry-in reading a join of compares after a divergent branch. A
    /// lane that skipped the inner compare has VCC clear there (VOPC clears
    /// inactive lanes), not the first compare's bit, so the join is no
    /// per-lane phi: lowering refuses it.
    #[test]
    fn a_carry_in_from_a_divergent_join_is_refused() {
        let body = [
            0x7E04_028B, // v_mov_b32 v2, 11
            0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
            0xBE82_376A, // s_andn1_saveexec_b64 s[2:3], vcc
            0xBF88_0001, // s_cbranch_execz 1
            0x7D84_0081, // v_cmp_eq_u32 vcc, 1, v0
            0x88FE_027E, // s_or_b64 exec, exec, s[2:3]
            0x5002_0480, // v_add_co_ci_u32 v1, vcc, 0, v2, vcc
        ];
        let e = stores_v1(&body).unwrap_err().to_string();
        assert!(e.contains("a carry-in from an untracked mask"), "{e}");
    }

    /// A pixel shader writing a constant colour.
    #[cfg(windows)]
    fn constant_color() -> Vec<u8> {
        let mut s = Shader::new(true);
        s.outputs.push(SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        });
        let b = s.entry;
        for (c, v) in [1.0f32, 0.5, 0.25, 1.0].into_iter().enumerate() {
            let args = [
                s.m.i32(0),
                s.m.i32(0),
                s.m.i8(c as u8),
                s.m.f32(v.to_bits()),
            ];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
        }
        s.m.terminate(b, super::bitcode::Term::Ret);
        s.finish()
    }

    #[test]
    #[cfg(windows)]
    fn stub_pipeline_is_created() {
        let Ok(host) = super::host::Host::new(false) else {
            eprintln!("no SM 6.6 device; skipped");
            return;
        };
        let ps = super::HostShader {
            stage: crate::HwStage::Ps,
            dxil: constant_color(),
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            root_uavs: Vec::new(),
            root_indexed_views: Vec::new(),
        };
        let file = |var: &str, stage| {
            std::env::var(var).ok().map(|p| super::HostShader {
                stage,
                dxil: std::fs::read(p).unwrap(),
                user_data: Vec::new(),
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                root_indexed_views: Vec::new(),
            })
        };
        let ps = file("VICTORY_REF_PS", crate::HwStage::Ps).unwrap_or(ps);
        let vs = file("VICTORY_REF_VS", crate::HwStage::Vs)
            .unwrap_or_else(|| super::host::stub_vs(&ps).unwrap());
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut vs.dxil.clone()).unwrap();
        }
        host.pipeline(&vs, &ps).unwrap();
        // The runtime checks the container hash: no hash or a wrong one is
        // refused. (Recent runtimes also take sixteen 0x01 bytes, a bypass
        // value; ours works with older ones too.)
        let mut wrong = ps.dxil[4..20].to_vec();
        wrong[0] ^= 1;
        for digest in [&[0u8; 16][..], &wrong] {
            let mut other = ps.clone();
            other.dxil[4..20].copy_from_slice(digest);
            assert!(
                host.pipeline(&vs, &other).is_err(),
                "digest {digest:02x?} accepted"
            );
        }
    }

    #[test]
    #[cfg(windows)]
    fn constant_color_validates() {
        let mut c = constant_color();
        let path = super::validator::Validator::default_path();
        let Ok(mut v) = super::validator::Validator::load(path.as_deref()) else {
            eprintln!("no dxil.dll; skipped");
            return;
        };
        let ours = c[4..20].to_vec();
        v.validate(&mut c).unwrap();
        assert_eq!(c[4..20], ours[..], "our hash differs from the validator's");
    }

    /// A pixel shader interpolating at sample positions (PERSP_SAMPLE
    /// barycentrics, `SPI_PS_INPUT_ADDR` bit 0) runs at sample frequency:
    /// DXC's PSV0 says so in PSInfo.SampleFrequency (runtime info byte 1)
    /// whenever an input is interpolated per sample.
    #[test]
    fn sample_interpolated_inputs_set_psv_sample_frequency() {
        let words = [
            0xC808_0000u32, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001,    // v_interp_p2_f32 v2, v1, attr0.x
            0xF800_180F,
            0x0202_0202, // exp mrt0, v2, v2, v2, v2 done vm
            0xBF81_0000,
        ];
        let psv_of = |input_addr: u32| {
            let native = super::NativeProgram {
                stage: super::HwStage::Ps,
                wave64: false,
                vgprs: 4,
                layout: vec![],
                used: vec![],
                io: crate::native::StageIo::Ps {
                    inputs: vec![crate::native::PsInput {
                        usage: crate::native::Usage::Generic,
                        row: 0,
                        mask: 0b1111,
                        attr: 0,
                        flat: false,
                        fp16: false,
                        pack2: false,
                        row2: 0,
                    }],
                    input_addr,
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let el = |name: &str, kind, usage| SigElement {
                name: name.into(),
                index: 0,
                kind,
                comp: 9,
                interp: 0,
                row: Some(0),
                col: 0,
                cols: 4,
                usage,
                half: false,
            };
            let isg1 = super::container::signature(&[el("TEXCOORD", 0, 0)], true);
            let osg1 = super::container::signature(&[el("SV_Target", 16, 15)], false);
            let bytes =
                crate::dxbc::build(&[(super::FourCC::ISG1, &isg1), (super::FourCC::OSG1, &osg1)]);
            let c = super::Container::parse(&bytes).unwrap();
            let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
            let st = crate::structure::structure(&p).unwrap();
            let res =
                crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = crate::io::raise(&p, &native).unwrap();
            let out = super::emit(&c, &native, &p, &st, &io, &res).unwrap();
            // dxil.dll, where there is one, checks PSV0 against the module.
            #[cfg(windows)]
            if let Ok(mut v) = super::validator::Validator::load(
                super::validator::Validator::default_path().as_deref(),
            ) {
                v.validate(&mut out.dxil.clone()).unwrap();
            }
            let host = super::Container::parse(&out.dxil).unwrap();
            host.part(super::FourCC::PSV0).unwrap().data.to_vec()
        };
        // Runtime info starts after its size; PSInfo is {DepthOutput, SampleFrequency}.
        let center = psv_of(1 << 1);
        assert_eq!(center[4 + 1], 0, "center interpolation is per pixel");
        let sample = psv_of(1 << 0);
        assert_eq!(sample[4 + 1], 1, "PSInfo.SampleFrequency");
    }

    /// A pixel shader reading channel z of an input whose element starts at
    /// w. Looking the channel up subtracted the element's first channel
    /// before checking the channel lies in it, so in a build with overflow
    /// checks (tests, debug) any read below an element that starts at z or
    /// w panicked, whichever row it was in (found by random-input testing).
    #[test]
    fn reading_a_channel_below_an_input_element_does_not_underflow() {
        // v_interp_mov_f32 v3, p0, attr0.z; s_endpgm
        let words = [0xC80E_0202u32, 0xBF81_0000];
        let native = super::NativeProgram {
            stage: super::HwStage::Ps,
            wave64: false,
            vgprs: 4,
            layout: vec![],
            used: vec![],
            io: crate::native::StageIo::Ps {
                inputs: vec![crate::native::PsInput {
                    usage: crate::native::Usage::Generic,
                    row: 0,
                    mask: 0b1000,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let w_only = SigElement {
            name: "TEXCOORD".into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 3,
            cols: 1,
            usage: 0,
            half: false,
        };
        let isg1 = super::container::signature(&[w_only], true);
        let bytes = crate::dxbc::build(&[(super::FourCC::ISG1, &isg1)]);
        let c = super::Container::parse(&bytes).unwrap();
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        // Translated or refused with a reason; it must not panic.
        let _ = super::emit(&c, &native, &p, &st, &io, &res);
    }

    #[test]
    fn valu_div_f32_sequence_executes_on_gpu() {
        let words: &[u32] = &[
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
            0xF408_0302,
            0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
            0x340E_0083, // v_lshlrev_b32 v7, 3, v0
            0xE034_1000,
            0x8002_0107, // buffer_load_dwordx2 v[1:2], v7, s[8:11], 0 offen
            0xD56D_6A03,
            0x0406_0502, // v_div_scale_f32 v3, vcc_lo, v2, v2, v1
            0xD56D_1008,
            0x0406_0501, // v_div_scale_f32 v8, s16, v1, v2, v1
            0x7E08_5503, // v_rcp_f32 v4, v3
            0xD54B_0005,
            0x23CA_0903, // v_fma_f32 v5, -v3, v4, 1.0
            0xD54B_0004,
            0x0412_0905, // v_fma_f32 v4, v5, v4, v4
            0x100A_0908, // v_mul_f32 v5, v8, v4
            0xD54B_0006,
            0x2422_0B03, // v_fma_f32 v6, -v3, v5, v8
            0xD56F_0004,
            0x0416_0906, // v_div_fmas_f32 v4, v6, v4, v5
            0xD55F_0001,
            0x0406_0504, // v_div_fixup_f32 v1, v4, v2, v1
            0x340E_0082, // v_lshlrev_b32 v7, 2, v0
            0xE070_1000,
            0x8003_0107, // buffer_store_dword v1, v7, s[12:15], 0 offen
            0xBF81_0000, // s_endpgm
        ];
        let native = super::NativeProgram {
            stage: super::HwStage::Cs,
            wave64: false,
            vgprs: 9,
            layout: vec![
                crate::RootParam {
                    root: crate::RootType::Constants,
                    table_offset_dwords: 0,
                    sgprs: 4,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                },
                crate::RootParam {
                    root: crate::RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 4,
                    slot: 1,
                },
            ],
            used: vec![
                crate::UsedResource {
                    kind: crate::ResourceType::Srv,
                    root: crate::RootType::Table,
                    table_offset_dwords: 0,
                    lower_bound: 0,
                    stride: 4,
                    dwords: 4,
                    user_data: 4,
                    slot: 1,
                },
                crate::UsedResource {
                    kind: crate::ResourceType::Uav,
                    root: crate::RootType::Table,
                    table_offset_dwords: 8,
                    lower_bound: 0,
                    stride: 4,
                    dwords: 4,
                    user_data: 4,
                    slot: 1,
                },
            ],
            io: crate::native::StageIo::Cs(crate::native::ComputeIo {
                rsrc2: 5 << 1,
                threads: [7, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = crate::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, None).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let l = super::lower::lower(&super::lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
        })
        .unwrap();
        let mut dxil = l.dxil;
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut dxil).unwrap();
        }
        #[cfg(windows)]
        if let Ok(host) = super::host::Host::new(false) {
            let cs = super::HostShader {
                stage: super::HwStage::Cs,
                dxil,
                user_data: l.user_data,
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: l.root_cbvs,
                root_srvs: l.root_srvs,
                root_uavs: l.root_uavs,
                root_indexed_views: Vec::new(),
            };
            let (root, pso) = host.compute_pipeline(&cs).unwrap();
            let inputs: Vec<u32> = [
                7.0f32, 3.0, -1.0, 1e-40, 1.0, 1e35, 1e-35, 1e-36, 0.0, 0.0, -1.0, 0.0, -1e35, 1e35,
            ]
            .into_iter()
            .map(f32::to_bits)
            .collect();
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &super::host::Dispatch {
                        roots: vec![(0, super::host::Root::Constants(vec![0, 0, 0, 0, 0]))],
                        views: &[
                            super::host::View::Buffer {
                                data: &inputs,
                                stride: super::host::Stride::Raw,
                            },
                            super::host::View::RawUav(64),
                        ],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap();
            let got: Vec<f32> = out[0][..7].iter().copied().map(f32::from_bits).collect();
            assert!((got[0] - (7.0f32 / 3.0)).abs() < 1e-6, "got {:?}", got);
            assert_eq!(out[0][1], (-1.0f32 / 1e-40).to_bits());
            assert!(
                (got[2] - 1e-35).abs() <= 1e-40,
                "lane 2 got {:?} ({:#x})",
                got[2],
                out[0][2]
            );
            assert!(
                (got[3] - 10.0).abs() < 1e-5,
                "lane 3 got {:?} ({:#x})",
                got[3],
                out[0][3]
            );
            assert_eq!(out[0][4], 0xFFC0_0000);
            assert_eq!(out[0][5], (-1.0f32 / 0.0).to_bits());
            assert!(
                (got[6] - (-1.0)).abs() < 1e-6,
                "lane 6 got {:?} ({:#x})",
                got[6],
                out[0][6]
            );
        }
    }
}

#[cfg(test)]
mod atomic_tests {
    use super::*;
    use crate::native::{ComputeIo, RootParam, StageIo};

    #[test]
    fn heap_texture_atomic_widths() {
        // A synthetic compute shader loads a 2D UAV from a root table,
        // compares or adds at (0, 0), then uses the returned register(s).
        for (opcode, mask, wide) in [
            (16u32, 3, false),
            (16, 15, true),
            (17, 1, false),
            (17, 3, true),
        ] {
            let words = [
                0xBE81_0380, // s_mov_b32 s1, 0 (table address high)
                0xF40C_0200,
                0xFA00_0000, // s_load_dwordx8 s[8:15], s[0:1], 0
                0x7E00_0280,
                0x7E02_0280, // v0, v1 = coordinates
                0x7E04_0281,
                0x7E06_0282,
                0x7E08_0283,
                0x7E0A_0284, // v2..v5 = 1..4
                0xF000_2008 | opcode << 18 | mask << 8,
                0x0002_0200,
                // Use the old value in a second atomic at the same location.
                0xF044_0008 | (if wide { 3 } else { 1 }) << 8,
                0x0002_0200,
                0xBF81_0000,
            ];
            let native = NativeProgram {
                stage: HwStage::Cs,
                wave64: false,
                vgprs: 6,
                layout: vec![RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                }],
                used: vec![],
                io: StageIo::Cs(ComputeIo {
                    rsrc2: 2,
                    threads: [32, 1, 1],
                }),
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let p = ir::build(&native.code, false, native.vgprs).unwrap();
            let bytes = crate::dxbc::build(&[]);
            let shader = emit(
                &Container::parse(&bytes).unwrap(),
                &native,
                &p,
                &structure::structure(&p).unwrap(),
                &io::raise(&p, &native).unwrap(),
                &resource::track(&p, &native.layout, native.stage, None).unwrap(),
            )
            .unwrap();
            let container = Container::parse(&shader.dxil).unwrap();
            let features = u64::from_le_bytes(
                container
                    .part(FourCC::SFI0)
                    .unwrap()
                    .data
                    .try_into()
                    .unwrap(),
            );
            let required = 0x1040_8000; // heap atomics, typed atomics, Int64Ops
            assert_eq!(features & required, if wide { required } else { 0 });
            #[cfg(windows)]
            if let Ok(mut v) =
                validator::Validator::load(validator::Validator::default_path().as_deref())
            {
                v.validate(&mut shader.dxil.clone()).unwrap();
            }
        }
    }

    fn dxil_has_symbol(shader: &HostShader, op: &str) -> bool {
        let c = Container::parse(&shader.dxil).unwrap();
        let part = c.part(FourCC::DXIL).unwrap();
        let bits: Vec<_> = part
            .data
            .iter()
            .flat_map(|&b| (0..8).map(move |k| (b >> k) & 1))
            .collect();
        let name: Vec<_> = op
            .bytes()
            .flat_map(|b| {
                let word = u16::from(b & 31) | 32 | (u16::from(b >> 5) << 6);
                (0..12).map(move |k| ((word >> k) & 1) as u8)
            })
            .collect();
        bits.windows(name.len()).any(|s| s == name)
    }

    fn sfi0(shader: &HostShader) -> u64 {
        let c = Container::parse(&shader.dxil).unwrap();
        u64::from_le_bytes(
            c.part(FourCC::SFI0)
                .unwrap()
                .data
                .try_into()
                .unwrap(),
        )
    }

    struct HostImageAtomic64Guard(Option<bool>);
    impl HostImageAtomic64Guard {
        fn set(v: bool) -> Self {
            let prev = super::lower::TEST_HOST_IMAGE_ATOMIC64.replace(Some(v));
            Self(prev)
        }
    }
    impl Drop for HostImageAtomic64Guard {
        fn drop(&mut self) {
            super::lower::TEST_HOST_IMAGE_ATOMIC64.set(self.0);
        }
    }

    fn synthetic_atomic64_program(stage: HwStage, opcode: u32, mask: u32, glc: bool) -> NativeProgram {
        let mimg0 = 0xF000_0008 | (opcode << 18) | (u32::from(glc) << 13) | (mask << 8);
        let words = [
            0xBE81_0380, // s_mov_b32 s1, 0 (table address high)
            0xF40C_0200,
            0xFA00_0000, // s_load_dwordx8 s[8:15], s[0:1], 0
            0x7E00_0280,
            0x7E02_0280, // v0, v1 = coordinates
            0x7E04_0281,
            0x7E06_0282,
            0x7E08_0283,
            0x7E0A_0284, // v2..v5 = 1..4
            mimg0,
            0x0002_0200, // vdata=v2, vaddr=v0, srsrc=s[8:15], dim=2d
            // Keep glc return registers live (export mrt0 for Ps, second atomic for Cs).
            if stage == HwStage::Ps { 0xF800_180F } else { 0xF044_0008 | (3 << 8) },
            if stage == HwStage::Ps { 0x0302_0302 } else { 0x0002_0200 },
            0xBF81_0000, // s_endpgm
        ];
        NativeProgram {
            stage,
            wave64: false,
            vgprs: 6,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![],
            io: match stage {
                HwStage::Ps => StageIo::Ps {
                    inputs: vec![],
                    input_addr: 0,
                    db_shader_control: 0,
                    patches: vec![],
                },
                HwStage::Cs => StageIo::Cs(ComputeIo {
                    rsrc2: 2,
                    threads: [32, 1, 1],
                }),
                _ => unreachable!(),
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        }
    }

    fn emit_synthetic(native: &NativeProgram) -> HostShader {
        let p = ir::build(&native.code, false, native.vgprs).unwrap();
        let bytes = if native.stage == HwStage::Ps {
            let target = container::SigElement {
                name: "SV_Target".into(),
                index: 0,
                kind: 16,
                comp: 9,
                interp: 0,
                row: Some(0),
                col: 0,
                cols: 4,
                usage: 15,
                half: false,
            };
            let osg1 = container::signature(&[target], false);
            crate::dxbc::build(&[(FourCC::OSG1, &osg1)])
        } else {
            crate::dxbc::build(&[])
        };
        emit(
            &Container::parse(&bytes).unwrap(),
            native,
            &p,
            &structure::structure(&p).unwrap(),
            &io::raise(&p, native).unwrap(),
            &resource::track(&p, &native.layout, native.stage, None).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn pixel_image_atomic64_rov_fallback() {
        const SFI0_TYPED_UAV_LOADS: u64 = 0x800;
        const SFI0_INT64: u64 = 0x8000;
        const SFI0_ATOMIC_TYPED64: u64 = 0x40_0000;
        const SFI0_RESOURCE_HEAP: u64 = 0x200_0000;
        const SFI0_ATOMIC_HEAP64: u64 = 0x1000_0000;

        for (opcode, mask, glc, expected_dxil_op) in [
            (0x17u32, 3, false, "dx.op.atomicBinOp.i64"), // image_atomic_umax (no glc)
            (0x17, 3, true, "dx.op.atomicBinOp.i64"),     // image_atomic_umax (glc)
            (0x11, 3, true, "dx.op.atomicBinOp.i64"),     // image_atomic_add (glc)
            (0x12, 3, true, "dx.op.atomicBinOp.i64"),     // image_atomic_sub (glc)
            (0x10, 15, true, "dx.op.atomicCompareExchange.i64"), // image_atomic_cmpswap
        ] {
            let native = synthetic_atomic64_program(HwStage::Ps, opcode, mask, glc);
            {
                let _g = HostImageAtomic64Guard::set(false);
                let shader = emit_synthetic(&native);
                let features = sfi0(&shader);
                assert_eq!(features & (SFI0_ATOMIC_TYPED64 | SFI0_ATOMIC_HEAP64), 0);
                assert_ne!(features & SFI0_TYPED_UAV_LOADS, 0);
                assert_ne!(features & SFI0_INT64, 0);
                assert_ne!(features & SFI0_RESOURCE_HEAP, 0);
                assert!(dxil_has_symbol(&shader, "dx.op.textureLoad.i32"));
                assert!(dxil_has_symbol(&shader, "dx.op.textureStore.i32"));
                assert!(!dxil_has_symbol(&shader, expected_dxil_op));
                #[cfg(windows)]
                if let Ok(mut v) =
                    validator::Validator::load(validator::Validator::default_path().as_deref())
                {
                    v.validate(&mut shader.dxil.clone()).unwrap();
                }
            }
            {
                let _g = HostImageAtomic64Guard::set(true);
                let shader = emit_synthetic(&native);
                let features = sfi0(&shader);
                assert_ne!(features & SFI0_ATOMIC_TYPED64, 0);
                assert_ne!(features & SFI0_ATOMIC_HEAP64, 0);
                assert!(dxil_has_symbol(&shader, expected_dxil_op));
            }
        }
    }

    #[test]
    fn compute_image_atomic64_uav_fallback_when_unsupported() {
        let _g = HostImageAtomic64Guard::set(false);
        let native = synthetic_atomic64_program(HwStage::Cs, 0x17, 3, true);
        let shader = emit_synthetic(&native);
        let features = sfi0(&shader);
        assert_eq!(features & (0x40_0000 | 0x1000_0000), 0);
        assert_ne!(features & 0x800, 0);
        assert_ne!(features & 0x8000, 0);
        assert!(dxil_has_symbol(&shader, "dx.op.textureLoad.i32"));
        assert!(dxil_has_symbol(&shader, "dx.op.textureStore.i32"));
        assert!(!dxil_has_symbol(&shader, "dx.op.atomicBinOp.i64"));

        let _g2 = HostImageAtomic64Guard::set(true);
        let native_ok = emit_synthetic(&native);
        assert!(dxil_has_symbol(&native_ok, "dx.op.atomicBinOp.i64"));
    }
}

#[cfg(test)]
mod wave64_lower_tests {
    use super::*;
    use crate::native::{ComputeIo, RootParam, StageIo, UsedResource, ResourceType};

    fn wave64_cs(body: &[u32], threads: u32, lds_blocks_64b: u32) -> NativeProgram {
        let mut words = vec![
            0xBE85_0380, // s_mov_b32 s5, 0
            0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
            125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
        ];
        words.extend_from_slice(body);
        words.push(0xBF81_0000); // s_endpgm
        NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 16,
            layout: vec![
                RootParam {
                    root: RootType::Constants,
                    table_offset_dwords: 0,
                    sgprs: 4,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                },
                RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 4,
                    slot: 1,
                },
            ],
            used: vec![UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 4,
                slot: 1,
            }],
            io: StageIo::Cs(ComputeIo {
                rsrc2: (5 << 1) | (lds_blocks_64b << 15),
                threads: [threads, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        }
    }

    fn psv_wave_range(shader: &HostShader) -> (u32, u32) {
        let parsed = Container::parse(&shader.dxil).unwrap();
        let psv = parsed.part(FourCC::PSV0).unwrap().data;
        let min = u32::from_le_bytes(psv[20..24].try_into().unwrap());
        let max = u32::from_le_bytes(psv[24..28].try_into().unwrap());
        (min, max)
    }

    fn emit_wave32(native: &NativeProgram, half_sync: bool) -> Result<HostShader, ShaderError> {
        let p = ir::build(&native.code, native.wave64, native.vgprs)?;
        let st = structure::structure(&p)?;
        let io = io::raise(&p, native)?;
        let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
        let empty = crate::dxbc::build(&[]);
        let c = Container::parse(&empty).unwrap();
        let was_w = lower::TEST_HOST_WAVE.replace(Some(32));
        let was_h = lower::TEST_HALF_SYNC.replace(Some(half_sync));
        let out = emit(&c, native, &p, &st, &io, &res);
        lower::TEST_HALF_SYNC.set(was_h);
        lower::TEST_HOST_WAVE.set(was_w);
        out
    }

    #[test]
    fn multi_wave_divergent_wave_or_lowers_to_wave32_with_half_sync() {
        let mut words = vec![
            0x7E0A_0281, // v_mov_b32 v5, 1
            0x2C0C_0081, // v_lshrrev_b32 v6, 1, v0
            0x340E_0B06, // v_lshlrev_b32 v7, v6, v5
            0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
            0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
            0xBF88_0017, // s_cbranch_execz +23
        ];
        words.extend_from_slice(&super::wave::WAVE_OR_V7_TO_S7);
        words.extend_from_slice(&[
            0x7E00_0207, // v_mov_b32 v0, s7
            0x38 << 26 | 0x1C << 18,
            0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
            0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        ]);
        let native = wave64_cs(&words, 128, 0);
        let shader = emit_wave32(&native, true).expect("emit with half_sync");
        assert_eq!(psv_wave_range(&shader), (32, 32));
    }

    #[test]
    fn lds_read_write_in_lane_branch_lowers_to_wave32() {
        let words = [
            0x7D04_0080, // v_cmp_eq_u32 vcc, 0, v0
            0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
            0xBF88_0008, // s_cbranch_execz +8 dwords -> s_mov_b64 exec, s[12:13]
            0x7E02_0280, // v_mov_b32 v1, 0
            0x7E04_02AA, // v_mov_b32 v2, 42
            0xD834_0000, 0x0000_0201, // ds_write_b32 v1, v2
            0xD8D8_0000, 0x0300_0001, // ds_read_b32 v3, v1
            0xD834_0000, 0x0000_0301, // ds_write_b32 v1, v3
            0xBEFE_040C, // s_mov_b64 exec, s[12:13]
            0x7E02_0280, // v_mov_b32 v1, 0
            0xD8D8_0000, 0x0000_0001, // ds_read_b32 v0, v1
            0x38 << 26 | 0x1C << 18,
            0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
        ];
        let native = wave64_cs(&words, 64, 2);
        let shader = emit_wave32(&native, true).expect("emit with half_sync");
        assert_eq!(psv_wave_range(&shader), (32, 32));
    }

    #[test]
    fn full_32k_lds_does_not_overflow_8192_dwords() {
        let mut words = vec![
            0x7E0A_0281, // v_mov_b32 v5, 1
            0x2C0C_0081, // v_lshrrev_b32 v6, 1, v0
            0x340E_0B06, // v_lshlrev_b32 v7, v6, v5
        ];
        words.extend_from_slice(&super::wave::WAVE_OR_V7_TO_S7);
        words.extend_from_slice(&[
            0x7E02_0280, // v_mov_b32 v1, 0
            0x7E04_0207, // v_mov_b32 v2, s7
            0xD834_0000, 0x0000_0201, // ds_write_b32 v1, v2
            0xD8D8_0000, 0x0000_0001, // ds_read_b32 v0, v1
            0x38 << 26 | 0x1C << 18,
            0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
        ]);
        let native = wave64_cs(&words, 1024, 64);
        let shader = emit_wave32(&native, true).expect("emit");
        assert_eq!(psv_wave_range(&shader), (32, 32));
        assert!(shader.root_uavs.contains(&SLOT_WAVE_SCRATCH));
        assert!(
            shader
                .user_data
                .iter()
                .any(|u| u.root == crate::native::RootType::Other(USER_GROUP_COUNT_X))
        );
        assert!(
            shader
                .user_data
                .iter()
                .any(|u| u.root == crate::native::RootType::Other(USER_GROUP_COUNT_Y))
        );
        // Verify @lds array length in the emitted DXIL bitcode does not exceed 8192 dwords.
        let parsed = Container::parse(&shader.dxil).unwrap();
        let bitcode = parsed.part(FourCC::DXIL).unwrap().data;
        assert!(!bitcode.is_empty());
    }
}

#[cfg(test)]
mod barycentric_tests {
    use super::container::SigElement;
    use super::*;
    use crate::native::{PsInput, StageIo, Usage};

    /// `SPI_PS_INPUT_ADDR` bits: PERSP_CENTER, LINEAR_CENTER.
    const PERSP_CENTER: u32 = 1 << 1;
    const LINEAR_CENTER: u32 = 1 << 5;

    /// `exp mrt0, v<r>, v<r>, v<r>, v<r> done vm`.
    fn exp_mrt0(r: u32) -> [u32; 2] {
        [0xF800_180F, r * 0x0101_0101]
    }

    /// VOP2 `v_add_f32 v<dst>, v<a>, v<b>`.
    fn v_add_f32(dst: u32, a: u32, b: u32) -> u32 {
        3 << 25 | dst << 17 | b << 9 | (256 + a)
    }

    fn element(name: &str, kind: u8, row: u32) -> SigElement {
        SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp: 9,
            interp: 0,
            row: Some(row),
            col: 0,
            cols: 4,
            usage: 0,
            half: false,
        }
    }

    /// Translates a pixel program with the input VGPRs `input_addr`
    /// enables and one float4 TEXCOORD0 attribute (attr 0, row 0).
    fn pixel(words: &[u32], input_addr: u32) -> Result<HostShader, ShaderError> {
        pixel_with_inputs(words, input_addr, &[element("TEXCOORD", 0, 0)])
    }

    fn pixel_with_inputs(words: &[u32], input_addr: u32, inputs: &[container::SigElement]) -> Result<HostShader, ShaderError> {
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 8,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![PsInput {
                    usage: Usage::Generic,
                    row: 0,
                    mask: 0b1111,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let mut target = element("SV_Target", 16, 0);
        target.usage = 15;
        let isg1 = container::signature(inputs, true);
        let osg1 = container::signature(&[target], false);
        let bytes = crate::dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
        let res = resource::track(&p, &native.layout, native.stage, None)?;
        let io = io::raise(&p, &native)?;
        let st = structure::structure(&p)?;
        emit(&c, &native, &p, &st, &io, &res)
    }

    #[test]
    fn an_unused_packed_input_matches_its_provoking_vertex_sibling() {
        let [e0, e1] = exp_mrt0(3);
        // v_interp_mov_f32 v3, p0, attr0.z followed by exp mrt0.
        let words = [0xC80E_0202, e0, e1, 0xBF81_0000];
        let xy = container::SigElement { cols: 2, ..element("TEXCOORD", 0, 0) };
        let zw = container::SigElement { col: 2, cols: 2, ..element("NORMAL", 0, 0) };
        let shader = pixel_with_inputs(&words, PERSP_CENTER, &[xy, zw]).unwrap();
        let c = Container::parse(&shader.dxil).unwrap();
        let psv = c.part(FourCC::PSV0).unwrap().data;
        for columns in [0x42, 0x62] {
            let record = [1, 0, columns, 0, 3, 1];
            assert!(psv.windows(record.len()).any(|w| w == record));
        }
        let inputs = c.part(FourCC::ISG1).unwrap().data;
        assert_eq!((inputs[8 + 25], inputs[8 + 32 + 25]), (0, 4));
        validate(&shader);
    }

    /// The translation's `SFI0` features and `ISG1` rows.
    fn parts(shader: &HostShader) -> (u64, Vec<SigElement>) {
        let c = Container::parse(&shader.dxil).unwrap();
        let sfi0 = c.part(FourCC::SFI0).unwrap().data.try_into().unwrap();
        (
            u64::from_le_bytes(sfi0),
            signature(c.part(FourCC::ISG1).unwrap().data).unwrap(),
        )
    }

    fn validate(shader: &HostShader) {
        #[cfg(windows)]
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            v.validate(&mut shader.dxil.clone()).unwrap();
        }
        let _ = shader;
    }

    #[test]
    fn pixel_first_lane_reads_select_the_helper_lanes_value() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0x7E00_0500, // v_readfirstlane_b32 s0, v0
            0x7E04_0200, // v_mov_b32 v2, s0
            e0, e1, 0xBF81_0000,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let c = Container::parse(&shader.dxil).unwrap();
        let part = c.part(FourCC::DXIL).unwrap();
        // The writer uses VBR6 for each ASCII operand in the symbol record.
        let bits: Vec<_> = part.data.iter().flat_map(|&b| (0..8).map(move |k| (b >> k) & 1)).collect();
        for op in ["dx.op.isHelperLane.i1", "dx.op.waveReadLaneFirst.i32"] {
            let name: Vec<_> = op.bytes().flat_map(|b| {
                let word = u16::from(b & 31) | 32 | (u16::from(b >> 5) << 6);
                (0..12).map(move |k| ((word >> k) & 1) as u8)
            }).collect();
            assert!(bits.windows(name.len()).any(|s| s == name), "{op}");
        }
        validate(&shader);
    }

    /// Custom interpolation reads I and J (v0, v1 with PERSP_CENTER) as
    /// plain floats: they are channels y and z of `SV_Barycentrics`, the
    /// weights of vertices 1 and 2.
    #[test]
    fn barycentrics_read_as_values_are_sv_barycentrics() {
        let [e0, e1] = exp_mrt0(2);
        let words = [v_add_f32(2, 0, 1), e0, e1, 0xBF81_0000];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0, "no Barycentrics feature");
        let bary: Vec<_> = inputs.iter().filter(|e| e.kind == 28).collect();
        assert_eq!(bary.len(), 1, "{inputs:?}");
        assert_eq!((bary[0].index, bary[0].col, bary[0].cols), (0, 0, 3));
        // Not packed: the validator wants packing location -1.
        assert_eq!(bary[0].row, None);
        // Only y and z (I and J) are read.
        let isg1 = Container::parse(&shader.dxil).unwrap();
        let isg1 = isg1.part(FourCC::ISG1).unwrap().data;
        let rw: Vec<u8> = (0..2).map(|k| isg1[8 + 32 * k + 25]).collect();
        assert!(rw.contains(&0b110), "{rw:?}");
        validate(&shader);
    }

    /// Two kinds of barycentrics read directly: one element per mode.
    #[test]
    fn two_barycentric_modes_get_two_elements() {
        // v0-v1 perspective i, j; v2-v3 linear i, j.
        let [e0, e1] = exp_mrt0(4);
        let words = [v_add_f32(4, 0, 3), e0, e1, 0xBF81_0000];
        let shader = pixel(&words, PERSP_CENTER | LINEAR_CENTER).unwrap();
        let (_, inputs) = parts(&shader);
        let mut indices: Vec<u32> = inputs
            .iter()
            .filter(|e| e.kind == 28)
            .map(|e| e.index)
            .collect();
        indices.sort();
        assert_eq!(indices, [0, 1], "{inputs:?}");
        validate(&shader);
    }

    /// I and J only fed to `v_interp_*` stay host interpolation: no
    /// barycentrics are declared.
    #[test]
    fn interpolation_alone_declares_no_barycentrics() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            e0,
            e1,
            0xBF81_0000,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_eq!(features & 0x2_0000, 0);
        assert!(inputs.iter().all(|e| e.kind != 28), "{inputs:?}");
        validate(&shader);
    }

    /// Both: the attribute is interpolated and J is used as a value.
    #[test]
    fn interpolation_and_a_barycentric_value_together() {
        let [e0, e1] = exp_mrt0(3);
        let words = [
            0xC808_0000,        // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001,        // v_interp_p2_f32 v2, v1, attr0.x
            v_add_f32(3, 2, 1), // v3 = attribute + j
            e0,
            e1,
            0xBF81_0000,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0);
        assert_eq!(inputs.iter().filter(|e| e.kind == 28).count(), 1);
        validate(&shader);
    }
    /// The `PSV0` interpolation mode of TEXCOORD0 (row 0, x-w, float).
    fn texcoord_mode(shader: &HostShader) -> u8 {
        let c = Container::parse(&shader.dxil).unwrap();
        let psv = c.part(FourCC::PSV0).unwrap().data;
        let row = [1, 0, 0x44, 0, 3];
        let at = psv.windows(5).position(|w| w == row).unwrap();
        psv[at + 5]
    }

    const S_ENDPGM: u32 = 0xBF81_0000;

    /// An interpolation whose i is computed (`v_mul_f32 v3, 0.5, v0`) reads
    /// the attribute's vertices: TEXCOORD0 becomes constant, read with
    /// `attributeAtVertex`; i and j are `SV_Barycentrics` values.
    #[test]
    fn computed_barycentrics_read_the_vertices() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0x1006_00F0, // v_mul_f32 v3, 0.5, v0
            0xC808_0003, // v_interp_p1_f32 v2, v3, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0, "no Barycentrics feature");
        assert_eq!(inputs.iter().filter(|e| e.kind == 28).count(), 1);
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
    }

    /// Weights from constants alone: the vertices, no `SV_Barycentrics`.
    #[test]
    fn constant_weights_need_no_barycentrics() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0x7E06_02F0, // v_mov_b32 v3, 0.5
            0xC808_0003, // v_interp_p1_f32 v2, v3, attr0.x
            0xC809_0003, // v_interp_p2_f32 v2, v3, attr0.x
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0, "no Barycentrics feature");
        assert!(inputs.iter().all(|e| e.kind != 28), "{inputs:?}");
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
    }

    /// The same attribute interpolated with the loaded i and j and with a
    /// computed j: both from its vertices, the first weighed with
    /// `SV_Barycentrics`.
    #[test]
    fn computed_and_loaded_barycentrics_on_one_attribute() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0xC810_0100, // v_interp_p1_f32 v4, v0, attr0.y
            0xC811_0101, // v_interp_p2_f32 v4, v1, attr0.y
            0x1006_02F0, // v_mul_f32 v3, 0.5, v1
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0003, // v_interp_p2_f32 v2, v3, attr0.x
            v_add_f32(2, 2, 4),
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (_, inputs) = parts(&shader);
        assert_eq!(inputs.iter().filter(|e| e.kind == 28).count(), 1);
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
    }

    /// P10 of an attribute the program interpolates is P1 - P0.
    #[test]
    fn p10_of_an_interpolated_attribute() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            0xC812_0000, // v_interp_mov_f32 v4, p10, attr0.x
            v_add_f32(2, 2, 4),
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
        // Alone, what the hardware holds is not known.
        let [e0, e1] = exp_mrt0(4);
        let words = [0xC812_0000, e0, e1, S_ENDPGM];
        assert!(pixel(&words, PERSP_CENTER).is_err());
    }

    fn with_attr_at_vertex<R>(supported: bool, f: impl FnOnce() -> R) -> R {
        let prev = lower::TEST_HOST_ATTR_AT_VERTEX.replace(Some(supported));
        let r = f();
        lower::TEST_HOST_ATTR_AT_VERTEX.set(prev);
        r
    }

    fn dxil_has_op(dxil: &[u8], op: &str) -> bool {
        let c = Container::parse(dxil).unwrap();
        let part = c.part(FourCC::DXIL).unwrap();
        let bits: Vec<_> = part
            .data
            .iter()
            .flat_map(|&b| (0..8).map(move |k| (b >> k) & 1))
            .collect();
        let name: Vec<_> = op
            .bytes()
            .flat_map(|b| {
                let word = u16::from(b & 31) | 32 | (u16::from(b >> 5) << 6);
                (0..12).map(move |k| ((word >> k) & 1) as u8)
            })
            .collect();
        bits.windows(name.len()).any(|s| s == name)
    }

    #[test]
    fn lowered_computed_barycentrics_uses_plan_b_tables() {
        with_attr_at_vertex(false, || {
            let [e0, e1] = exp_mrt0(2);
            let words = [
                0x1006_00F0, // v_mul_f32 v3, 0.5, v0
                0xC808_0003, // v_interp_p1_f32 v2, v3, attr0.x
                0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
                e0,
                e1,
                S_ENDPGM,
            ];
            let shader = pixel(&words, PERSP_CENTER).unwrap();
            let (features, inputs) = parts(&shader);
            assert_ne!(features & 0x2_0000, 0, "SV_Barycentrics still used for i/j");
            assert!(inputs.iter().any(|e| e.kind == 10 && e.name == "SV_PrimitiveID"));
            assert!(inputs.iter().any(|e| e.kind == 28));
            assert!(shader.root_srvs.contains(&SLOT_PRIM_IBO));
            assert!(shader.root_srvs.contains(&SLOT_ATTR_TABLE));
            assert!(!dxil_has_op(&shader.dxil, "dx.op.attributeAtVertex"));
            assert!(dxil_has_op(&shader.dxil, "dx.op.rawBufferLoad.f32"));
            assert_eq!(texcoord_mode(&shader), 1);
            validate(&shader);
        });
    }

    #[test]
    fn lowered_constant_weights_has_no_barycentrics_flag() {
        with_attr_at_vertex(false, || {
            let [e0, e1] = exp_mrt0(2);
            let words = [
                0x7E06_02F0, // v_mov_b32 v3, 0.5
                0xC808_0003, // v_interp_p1_f32 v2, v3, attr0.x
                0xC809_0003, // v_interp_p2_f32 v2, v3, attr0.x
                e0,
                e1,
                S_ENDPGM,
            ];
            let shader = pixel(&words, PERSP_CENTER).unwrap();
            let (features, inputs) = parts(&shader);
            assert_eq!(features & 0x2_0000, 0, "no Barycentrics needed with constant weights");
            assert!(inputs.iter().any(|e| e.kind == 10 && e.name == "SV_PrimitiveID"));
            assert!(inputs.iter().all(|e| e.kind != 28));
            assert!(shader.root_srvs.contains(&SLOT_PRIM_IBO));
            assert!(shader.root_srvs.contains(&SLOT_ATTR_TABLE));
            assert!(!dxil_has_op(&shader.dxil, "dx.op.attributeAtVertex"));
            validate(&shader);
        });
    }

    #[test]
    fn lowered_p10_and_p20_use_plan_b_tables() {
        with_attr_at_vertex(false, || {
            let [e0, e1] = exp_mrt0(2);
            let words = [
                0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
                0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
                0xC812_0000, // v_interp_mov_f32 v4, p10, attr0.x
                0xC812_0100, // v_interp_mov_f32 v5, p20, attr0.x
                v_add_f32(4, 4, 5),
                v_add_f32(2, 2, 4),
                e0,
                e1,
                S_ENDPGM,
            ];
            let shader = pixel(&words, PERSP_CENTER).unwrap();
            let (_, inputs) = parts(&shader);
            assert!(inputs.iter().any(|e| e.kind == 10));
            assert!(shader.root_srvs.contains(&SLOT_PRIM_IBO));
            assert!(shader.root_srvs.contains(&SLOT_ATTR_TABLE));
            assert!(!dxil_has_op(&shader.dxil, "dx.op.attributeAtVertex"));
            validate(&shader);
        });
    }

    #[test]
    fn lowered_vs_mirrors_param_outputs_to_scratch_uav() {
        use crate::native::VsOutput;
        with_attr_at_vertex(false, || {
            let words = [
                0x7E02_02F2, // v_mov_b32 v1, 1.0
                // exp pos0, v1, v1, v1, v1
                0xF800_00CF,
                0x0101_0101,
                // exp param0, v0, v1, v1, v1 done
                0xF800_0A0F,
                0x0101_0100,
                S_ENDPGM,
            ];
            let native = NativeProgram {
                stage: HwStage::Vs,
                wave64: false,
                vgprs: 4,
                layout: vec![],
                used: vec![],
                io: StageIo::Vs {
                    inputs: vec![],
                    outputs: vec![VsOutput {
                        usage: Usage::Generic,
                        row: 1,
                        param: 0,
                        fp16: false,
                        pack2: false,
                    }],
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let pos = SigElement { kind: 3, ..element("SV_Position", 3, 0) };
            let tex = element("TEXCOORD", 0, 1);
            let osg1 = container::signature(&[pos, tex], false);
            let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
            let c = Container::parse(&bytes).unwrap();
            let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
            let res = resource::track(&p, &native.layout, native.stage, None).unwrap();
            let io = io::raise(&p, &native).unwrap();
            let st = structure::structure(&p).unwrap();
            let shader = emit(&c, &native, &p, &st, &io, &res).unwrap();
            assert!(shader.root_uavs.contains(&SLOT_ATTR_SCRATCH));
            let (features, inputs) = parts(&shader);
            assert_ne!(features & 0x4, 0, "SFI0 UAVS_EVERY_STAGE flag");
            assert!(inputs.iter().any(|e| e.kind == 1 && e.name == "SV_VertexID"));
            assert!(dxil_has_op(&shader.dxil, "dx.op.rawBufferStore.i32"));
            validate(&shader);
        });
    }

    #[test]
    fn lowered_packed_row_uses_element_col_plus_channel() {
        with_attr_at_vertex(false, || {
            let [e0, e1] = exp_mrt0(2);
            // v_interp_mov_f32 v4, p10, attr0.w (chan 3 -> NORMAL.y at row 0, col 2 + 1 = 3)
            let words = [
                0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
                0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
                0xC812_0300 | (3 << 8), // v_interp_mov_f32 v4, p10, attr0.w? Wait: attr0.w = chan 3
                v_add_f32(2, 2, 4),
                e0,
                e1,
                S_ENDPGM,
            ];
            let xy = container::SigElement { cols: 2, ..element("TEXCOORD", 0, 0) };
            let zw = container::SigElement { col: 2, cols: 2, ..element("NORMAL", 0, 0) };
            let shader = pixel_with_inputs(&words, PERSP_CENTER, &[xy.clone(), zw.clone()]).unwrap();
            assert!(shader.root_srvs.contains(&SLOT_PRIM_IBO));
            assert!(shader.root_srvs.contains(&SLOT_ATTR_TABLE));
            assert!(!dxil_has_op(&shader.dxil, "dx.op.attributeAtVertex"));
            // Toggle back to true still emits attributeAtVertex.
            with_attr_at_vertex(true, || {
                let native_shader = pixel_with_inputs(&words, PERSP_CENTER, &[xy, zw]).unwrap();
                assert!(dxil_has_op(&native_shader.dxil, "dx.op.attributeAtVertex"));
            });
            validate(&shader);
        });
    }
}

/// `VICTORY_DUMP_IN=<container> VICTORY_DUMP_OUT=<file> cargo test -p victory-shader
/// --lib dump_translated -- --ignored`: the DXIL a container translates to on a host
/// of 32-lane waves (for `llvm-dis` after stripping the DXIL part header).
#[cfg(test)]
#[test]
#[ignore = "a tool: needs VICTORY_DUMP_IN / VICTORY_DUMP_OUT"]
fn dump_translated() {
    let input = std::env::var("VICTORY_DUMP_IN").expect("VICTORY_DUMP_IN");
    let out = std::env::var("VICTORY_DUMP_OUT").expect("VICTORY_DUMP_OUT");
    // `VICTORY_DUMP_METAL=1`: with the Metal backend's settings ([`set_metal_host`]).
    if std::env::var("VICTORY_DUMP_METAL").is_ok_and(|v| v == "1") {
        set_metal_host();
    }
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    bitcode::LLVM_DIS_LAYOUT.set(true);
    let shader = translate(&std::fs::read(input).unwrap());
    lower::TEST_HOST_WAVE.set(was);
    std::fs::write(out, shader.expect("translate").dxil).unwrap();
}

/// `VICTORY_DUMP_WORDS=<file of hex dwords> VICTORY_DUMP_OUT=<file> cargo test -p
/// victory-shader --lib dump_native -- --ignored`: [`dump_translated`] for a wave64
/// compute program of 64 threads given as its body (4 root constants in s0..s3, a raw
/// UAV's descriptor loaded into s[8:11]), the layout of victory-gpu's Metal repros.
#[cfg(test)]
#[test]
#[ignore = "a tool: needs VICTORY_DUMP_WORDS / VICTORY_DUMP_OUT"]
fn dump_native() {
    use crate::native::{ComputeIo, ResourceType, RootParam, RootType, StageIo, UsedResource};
    let text = std::fs::read_to_string(std::env::var("VICTORY_DUMP_WORDS").expect("VICTORY_DUMP_WORDS")).unwrap();
    let mut words = vec![0xBE85_0380, 0x3D << 26 | 0x02 << 18 | 8 << 6 | 2, 125 << 25, 0xBF8C_C07F];
    words.extend(text.split(|c: char| c.is_whitespace() || c == ',').filter(|w| !w.is_empty()).map(|w| {
        u32::from_str_radix(w.trim_start_matches("0x").replace('_', "").as_str(), 16).unwrap()
    }));
    words.push(0xBF81_0000);
    let root = |root, sgprs, user_data, slot| RootParam { root, table_offset_dwords: 0, sgprs, base_register: 0, user_data, slot };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 96,
        layout: vec![root(RootType::Constants, 4, 0, 0), root(RootType::Table, 1, 4, 1)],
        used: vec![UsedResource {
            kind: ResourceType::Uav, root: RootType::Table, table_offset_dwords: 0, lower_bound: 0,
            stride: 0, dwords: 4, user_data: 4, slot: 1,
        }],
        io: StageIo::Cs(ComputeIo { rsrc2: 5 << 1, threads: [64, 1, 1] }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    bitcode::LLVM_DIS_LAYOUT.set(true);
    let shader = translate_native(&native);
    lower::TEST_HOST_WAVE.set(was);
    std::fs::write(std::env::var("VICTORY_DUMP_OUT").unwrap(), shader.expect("translate").dxil).unwrap();
}
