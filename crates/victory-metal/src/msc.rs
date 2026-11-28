//! Translated shaders (PC DXIL from victory-shader) to Metal pipelines,
//! through Apple's Metal Shader Converter on the unix side
//! (`native/victorymetal/src/msc.h`).
//!
//! The root signature is the one victory builds per pipeline
//! (`victory-gpu/src/pipeline/host.rs`): 32-bit constants for user data and
//! root CBV/SRV/UAV descriptors, both heaps directly indexed. MSC turns it
//! into the top-level argument buffer layout (bind point 2); the view heap
//! goes at bind point 0 and the sampler heap at 1 (metal_irconverter_runtime.h).

use core::ffi::c_void;
use std::ffi::CString;

use crate::{ComputePipeline, Device, Error, Object, Pool, Result, error_text, sys};

/// Bind points MSC's shaders read (metal_irconverter_runtime.h).
pub const DESCRIPTOR_HEAP_BIND_POINT: u8 = 0;
pub const SAMPLER_HEAP_BIND_POINT: u8 = 1;
pub const ARGUMENT_BUFFER_BIND_POINT: u8 = 2;
pub const DRAW_ARGUMENTS_BIND_POINT: u8 = 4;
pub const DRAW_UNIFORMS_BIND_POINT: u8 = 5;
pub const VERTEX_TABLE_BIND_POINT: u8 = 6;
pub const VERTEX_TABLE_SLOTS: usize = 31;

/// IRShaderStage numbering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Stage {
    Vertex = 1,
    Fragment = 2,
    Mesh = 5,
    Amplification = 6,
    Geometry = 7,
    Compute = 8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootParam {
    /// `num` 32-bit values at `b<register>, space<space>`.
    Constants { register: u32, space: u32, num: u32 },
    Cbv { register: u32, space: u32 },
    Srv { register: u32, space: u32 },
    Uav { register: u32, space: u32 },
    /// A descriptor table of one SRV / UAV at the register: a root view
    /// given bounds (its entry carries the size).
    TableSrv { register: u32, space: u32 },
    TableUav { register: u32, space: u32 },
}

/// msc.h `struct msc_root_param`.
#[repr(C)]
#[derive(Clone, Copy)]
struct RawParam {
    kind: u32,
    shader_register: u32,
    space: u32,
    num_32bit_values: u32,
}

/// msc.h `struct msc_static_sampler` (D3D12_STATIC_SAMPLER_DESC's numbers).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct StaticSampler {
    pub filter: u32,
    pub address_u: u32,
    pub address_v: u32,
    pub address_w: u32,
    pub mip_lod_bias: f32,
    pub max_anisotropy: u32,
    pub comparison_func: u32,
    pub border_color: u32,
    pub min_lod: f32,
    pub max_lod: f32,
    pub shader_register: u32,
    pub register_space: u32,
}

/// msc.h `struct msc_compile_params`.
#[repr(C)]
struct CompileParams {
    dxil: *const c_void,
    dxil_len: u64,
    params: *const RawParam,
    num_params: u32,
    num_samplers: u32,
    samplers: *const StaticSampler,
    stage: u32,
    /// msc.h MSC_FLAG_* (an older unix side ignores it: bounds checks stay).
    flags: u32,
    metallib: *mut c_void,
    metallib_len: u64,
    threadgroup: [u32; 3],
    entry: [u8; 256],
    num_locations: u32,
    top_level_size: u32,
    locations: [RawLocation; 64],
    ret: i32,
    error: [u8; 1024],
}

/// D3D input element for a vertex shader with a separate Metal stage-in function.
#[derive(Clone, Copy, Debug)]
pub struct VertexInput<'a> {
    pub semantic: &'a str,
    pub semantic_index: u32,
    /// DXGI/IRFormat value.
    pub format: u32,
    pub slot: u32,
    /// Resolved byte offset, not D3D12_APPEND_ALIGNED_ELEMENT.
    pub offset: u32,
    /// `None` for per-vertex data or the D3D instance step rate.
    pub instance_step_rate: Option<u32>,
}

#[repr(C)]
struct RawInput {
    semantic: *const core::ffi::c_char,
    semantic_index: u32,
    format: u32,
    slot: u32,
    offset: u32,
    per_instance: u32,
    step_rate: u32,
}

#[repr(C)]
struct StageInParams {
    compile: *mut CompileParams,
    elements: *const RawInput,
    count: u32,
    metallib: *mut c_void,
    metallib_len: u64,
}

#[repr(C)]
struct GeometryParams {
    stage_in: StageInParams,
    vertex_output_size: u32,
    max_input_primitives: u32,
    max_payload_size: u32,
    gs_instances: u32,
}

/// Reflection needed to construct a triangle-list geometry emulation pipeline.
#[derive(Clone, Copy, Debug)]
pub struct GeometryConfig {
    pub vertex_output_size: u32,
    pub max_input_primitives: u32,
    pub max_payload_size: u32,
}

/// MSC's triangle-list object/mesh draw ABI (Metal 3 runtime helper).
#[derive(Clone, Copy, Debug)]
pub struct GeometryDraw {
    pub args: [u8; 20],
    pub info: [u8; 24],
    pub groups: [u64; 3],
    pub object_threads: [u64; 3],
    pub mesh_threads: [u64; 3],
}

impl GeometryConfig {
    pub(crate) fn object_vertex_stride(self) -> Option<u32> {
        if self.vertex_output_size == 0 || !(1..=256).contains(&self.max_input_primitives)
            || !(1..=16384).contains(&self.max_payload_size) {
            return None;
        }
        let primitives = ((16384 - 32) / self.vertex_output_size / 3)
            .min(1024 * self.max_input_primitives)
            .min(256 / 3);
        (primitives != 0).then_some(primitives * 3)
    }

    /// The index address points at the start of the index view; `first` is in elements.
    /// Returns None for an empty draw or invalid config.
    pub fn draw(
        self, count: u32, instances: u32, first: u32, first_instance: u32,
        index: Option<(u8, u64, i32)>,
    ) -> Option<GeometryDraw> {
        let stride = self.object_vertex_stride()?;
        let count = count / 3 * 3;
        if count == 0 || instances == 0 || instances > u16::MAX.into() {
            return None;
        }
        let (args_words, index_type, address) = match index {
            None => ([count, instances, first, first_instance, 0], 0u16, 0u64),
            Some((t @ 0..=1, address, base)) if address != 0 => (
                [count, instances, first, base as u32, first_instance], u16::from(t) + 1, address,
            ),
            _ => return None,
        };
        let mut args = [0; 20];
        for (out, value) in args.chunks_exact_mut(4).zip(args_words) {
            out.copy_from_slice(&value.to_le_bytes());
        }
        let mut info = [0; 24];
        info[..2].copy_from_slice(&index_type.to_le_bytes());
        info[2] = 3; // IRRuntimePrimitiveTypeTriangle
        info[3] = 3; // threadsPerPatch
        // gsInstanceCount is the GS `[instance(N)]` count (always 1 here,
        // checked in convert_geometry); draw instancing is already carried by
        // groups[1] = instances and args[1] = instances. Passing `instances`
        // here causes MSC's object shader to spawn `instances^2` mesh groups
        // and hit Metal Code=8 OOM.
        for (out, value) in info[4..16].chunks_exact_mut(2).zip([
            self.max_input_primitives, stride, self.max_input_primitives, 1, stride / 3, 3,
        ]) {
            out.copy_from_slice(&(value as u16).to_le_bytes());
        }
        info[16..].copy_from_slice(&address.to_le_bytes());
        Some(GeometryDraw {
            args, info,
            groups: [u64::from(count.div_ceil(stride)), u64::from(instances), 1],
            object_threads: [u64::from(stride), 1, 1],
            mesh_threads: [u64::from(self.max_input_primitives), 1, 1],
        })
    }
}

pub struct ConvertedGeometryVertex {
    pub shader: ConvertedVertex,
    pub vertex_output_size: u32,
}

pub struct ConvertedGeometry {
    pub shader: Converted,
    pub config: GeometryConfig,
}

/// One entry of MSC's IRRuntimeVertexBuffers table (vertex bind point 6).
#[derive(Clone, Copy, Debug, Default)]
pub struct VertexBufferEntry {
    pub address: u64,
    pub length: u32,
    pub stride: u32,
}

/// Serialize the GPU addresses and draw-time buffer sizes and strides.
pub fn vertex_table_bytes(entries: &[VertexBufferEntry; VERTEX_TABLE_SLOTS]) -> [u8; VERTEX_TABLE_SLOTS * 16] {
    let mut bytes = [0; VERTEX_TABLE_SLOTS * 16];
    for (out, entry) in bytes.chunks_exact_mut(16).zip(entries) {
        out[..8].copy_from_slice(&entry.address.to_le_bytes());
        out[8..12].copy_from_slice(&entry.length.to_le_bytes());
        out[12..].copy_from_slice(&entry.stride.to_le_bytes());
    }
    bytes
}

/// msc.h `struct msc_location`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawLocation {
    kind: u32,
    space: u32,
    slot: u32,
    offset: u32,
    size: u64,
}

/// `MSC_Compile` / `MSC_Free` from victorymetal.dll (loaded on first use).
#[allow(non_snake_case)]
unsafe fn MSC_Compile(params: *mut CompileParams) -> i32 {
    let f = sys::symbol(b"MSC_Compile\0");
    assert!(!f.is_null(), "victorymetal.dll has no MSC_Compile (rebuild native/victorymetal)");
    // SAFETY: msc.h's signature.
    unsafe { core::mem::transmute::<*const c_void, unsafe extern "C" fn(*mut CompileParams) -> i32>(f)(params) }
}

#[allow(non_snake_case)]
unsafe fn MSC_Free(ptr: *mut c_void) {
    let f = sys::symbol(b"MSC_Free\0");
    assert!(!f.is_null(), "victorymetal.dll has no MSC_Free");
    // SAFETY: msc.h's signature.
    unsafe { core::mem::transmute::<*const c_void, unsafe extern "C" fn(*mut c_void)>(f)(ptr) }
}

fn c_text(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// A metallib MSC made, with what its reflection says.
pub struct Converted {
    pub metallib: Vec<u8>,
    pub entry: String,
    pub threadgroup: [u32; 3],
    /// An empty top-level argument buffer of the root signature's layout.
    pub top_level: crate::ir::TopLevel,
}

impl Converted {
    /// For a pipeline cache: magic, threadgroup, entry, top-level layout,
    /// metallib.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.metallib.len() + 512);
        out.extend_from_slice(b"VMSC0001");
        for v in self.threadgroup {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&(self.entry.len() as u32).to_le_bytes());
        out.extend_from_slice(self.entry.as_bytes());
        out.extend_from_slice(&self.top_level.layout_bytes());
        out.extend_from_slice(&(self.metallib.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.metallib);
        out
    }

    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        let rest = b.strip_prefix(b"VMSC0001")?;
        let u32_at = |at: usize| rest.get(at..at + 4).map(|x| u32::from_le_bytes(x.try_into().unwrap()));
        let threadgroup = [u32_at(0)?, u32_at(4)?, u32_at(8)?];
        let n = u32_at(12)? as usize;
        let entry = String::from_utf8(rest.get(16..16 + n)?.to_vec()).ok()?;
        let at = 16 + n;
        let (top_level, used) = crate::ir::TopLevel::from_layout_bytes(rest.get(at..)?)?;
        let at = at + used;
        let len = u64::from_le_bytes(rest.get(at..at + 8)?.try_into().ok()?) as usize;
        let metallib = rest.get(at + 8..at + 8 + len)?.to_vec();
        (at + 8 + len == rest.len()).then_some(Self { metallib, entry, threadgroup, top_level })
    }
}

/// msc.h `MSC_FLAG_NO_BOUNDS_CHECK`.
const MSC_FLAG_NO_BOUNDS_CHECK: u32 = 1;

/// The hash a bounds-check allow-list names a shader by: BLAKE3 of the DXIL
/// container, lowercase hex (lines may give any prefix of 16+ digits).
pub fn dxil_hash(dxil: &[u8]) -> String {
    blake3::hash(dxil).to_hex().to_string()
}

/// Which pipelines are compiled without `IRCompatibilityFlagBoundsCheck`
/// (PLAN-NATIVE-METAL P7, G.2 "V level"). Bounds checks are what keeps a
/// stray write inside a view under zero copy, so the default keeps them for
/// every shader.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum BoundsPolicy {
    /// Every shader bounds checked (the default).
    #[default]
    All,
    /// `VICTORY_METAL_BOUNDS_CHECK=0`: none (measurement only).
    None,
    /// `VICTORY_METAL_BOUNDS_ALLOW=<file>`: shaders whose DXIL hash (a line
    /// of the file, `#` comments) is listed go without; all others keep them.
    AllowList(Vec<String>),
}

impl BoundsPolicy {
    /// From `VICTORY_METAL_BOUNDS_CHECK` and `VICTORY_METAL_BOUNDS_ALLOW`.
    pub fn from_env() -> Self {
        if std::env::var("VICTORY_METAL_BOUNDS_CHECK").is_ok_and(|v| v.trim() == "0") {
            return Self::None;
        }
        match std::env::var_os("VICTORY_METAL_BOUNDS_ALLOW").filter(|v| !v.is_empty()) {
            Some(path) => match std::fs::read_to_string(&path) {
                Ok(text) => Self::parse_list(&text),
                // An unreadable list keeps every check (fail safe).
                Err(_) => Self::All,
            },
            None => Self::All,
        }
    }

    /// An allow-list file's text: one hash (prefix) per line, `#` comments.
    pub fn parse_list(text: &str) -> Self {
        let list: Vec<String> = text
            .lines()
            .map(|l| l.split('#').next().unwrap_or("").trim().to_ascii_lowercase())
            .filter(|l| l.len() >= 16 && l.len() <= 64 && l.bytes().all(|b| b.is_ascii_hexdigit()))
            .collect();
        if list.is_empty() { Self::All } else { Self::AllowList(list) }
    }

    /// Whether `dxil` keeps its bounds checks under this policy.
    pub fn bounds_check(&self, dxil: &[u8]) -> bool {
        match self {
            Self::All => true,
            Self::None => false,
            Self::AllowList(list) => {
                let h = dxil_hash(dxil);
                !list.iter().any(|p| h.starts_with(p.as_str()))
            }
        }
    }
}

static BOUNDS: std::sync::RwLock<Option<BoundsPolicy>> = std::sync::RwLock::new(None);

/// The policy in force (read from the environment on first use).
pub fn bounds_policy() -> BoundsPolicy {
    let known = BOUNDS.read().unwrap_or_else(|e| e.into_inner()).clone();
    if let Some(p) = known {
        return p;
    }
    let mut slot = BOUNDS.write().unwrap_or_else(|e| e.into_inner());
    slot.get_or_insert_with(BoundsPolicy::from_env).clone()
}

/// Replaces the policy (tests; `None` re-reads the environment next time).
pub fn set_bounds_policy(policy: Option<BoundsPolicy>) {
    *BOUNDS.write().unwrap_or_else(|e| e.into_inner()) = policy;
}

/// Whether `dxil` is converted with bounds checks (see [`BoundsPolicy`]).
/// Callers that cache converted shaders must key on this.
pub fn bounds_check(dxil: &[u8]) -> bool {
    // The read guard ends with this statement: `bounds_policy` may take the write lock
    // (a guard held by a `match` scrutinee would live through its arms and deadlock).
    let known = BOUNDS.read().unwrap_or_else(|e| e.into_inner()).as_ref().map(|p| p.bounds_check(dxil));
    known.unwrap_or_else(|| bounds_policy().bounds_check(dxil))
}

/// Whether the installed victorymetal honours [`BoundsPolicy`] (an older
/// build ignores the flag and keeps every check).
pub fn bounds_policy_supported() -> bool {
    let f = sys::symbol(b"MSC_Features\0");
    if f.is_null() {
        return false;
    }
    // SAFETY: msc.h's signature.
    unsafe { core::mem::transmute::<*const c_void, unsafe extern "C" fn() -> u32>(f)() & 1 != 0 }
}

/// Converts one DXIL container (as victory-shader emits it) for `stage`.
pub fn convert(dxil: &[u8], stage: Stage, params: &[RootParam], samplers: &[StaticSampler]) -> Result<Converted> {
    convert_impl(dxil, stage, params, samplers, &[], false).map(|(shader, _, _)| shader)
}

/// The vertex shader and its matching, layout-specific stage-in metallib.
pub struct ConvertedVertex {
    pub shader: Converted,
    pub stage_in: Vec<u8>,
}

pub fn convert_vertex_with_stage_in(
    dxil: &[u8],
    params: &[RootParam],
    samplers: &[StaticSampler],
    inputs: &[VertexInput<'_>],
) -> Result<ConvertedVertex> {
    if inputs.is_empty() || inputs.len() > VERTEX_TABLE_SLOTS {
        return Err(Error("Metal stage-in needs 1 to 31 input elements".into()));
    }
    encode_vertex_inputs(inputs, |elements| {
        let (shader, stage_in, _) = convert_impl(dxil, Stage::Vertex, params, samplers, elements, false)?;
        Ok(ConvertedVertex { shader, stage_in })
    })
}

pub fn convert_geometry_vertex(
    dxil: &[u8], params: &[RootParam], samplers: &[StaticSampler], inputs: &[VertexInput<'_>],
) -> Result<ConvertedGeometryVertex> {
    encode_vertex_inputs(inputs, |elements| {
        let (shader, stage_in, reflection) = convert_impl(dxil, Stage::Vertex, params, samplers, elements, true)?;
        if reflection[0] == 0 || stage_in.is_empty() {
            return Err(Error("missing object-stage vertex reflection or stage-in".into()));
        }
        Ok(ConvertedGeometryVertex { shader: ConvertedVertex { shader, stage_in }, vertex_output_size: reflection[0] })
    })
}

pub fn convert_geometry(
    dxil: &[u8], params: &[RootParam], samplers: &[StaticSampler], vertex_output_size: u32,
) -> Result<ConvertedGeometry> {
    let (shader, _, [_, max_input_primitives, max_payload_size, gs_instances]) =
        convert_impl(dxil, Stage::Geometry, params, samplers, &[], true)?;
    let config = GeometryConfig { vertex_output_size, max_input_primitives, max_payload_size };
    if gs_instances != 1 || config.object_vertex_stride().is_none() {
        return Err(Error(format!("unsupported triangle geometry reflection {config:?}, GS instances={gs_instances}")));
    }
    Ok(ConvertedGeometry { shader, config })
}

fn encode_vertex_inputs<T>(inputs: &[VertexInput<'_>], use_elements: impl FnOnce(&[RawInput]) -> Result<T>) -> Result<T> {
    if inputs.len() > VERTEX_TABLE_SLOTS {
        return Err(Error("Metal stage-in supports up to 31 input elements".into()));
    }
    let names: Vec<_> = inputs
        .iter()
        .map(|input| CString::new(input.semantic).map_err(|_| Error("vertex semantic contains a NUL".into())))
        .collect::<Result<_>>()?;
    let elements: Vec<_> = inputs
        .iter()
        .zip(&names)
        .map(|(input, name)| {
            if input.slot as usize >= VERTEX_TABLE_SLOTS || input.offset == u32::MAX || name.as_bytes().is_empty() {
                return Err(Error("invalid Metal vertex input slot, offset or semantic".into()));
            }
            Ok(RawInput {
                semantic: name.as_ptr(),
                semantic_index: input.semantic_index,
                format: input.format,
                slot: input.slot,
                offset: input.offset,
                per_instance: u32::from(input.instance_step_rate.is_some()),
                step_rate: input.instance_step_rate.unwrap_or(0),
            })
        })
        .collect::<Result<_>>()?;
    use_elements(&elements)
}

/// The feature flags (`SFI0`) of a DXIL container, if it has them.
fn feature_flags(container: &[u8]) -> Option<u64> {
    let u32_at = |at: usize| container.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    if container.get(..4)? != b"DXBC" {
        return None;
    }
    (0..u32_at(28)? as usize).find_map(|i| {
        let at = u32_at(32 + 4 * i)? as usize;
        (container.get(at..at + 4)? == b"SFI0")
            .then(|| container.get(at + 8..at + 16).map(|b| u64::from_le_bytes(b.try_into().unwrap())))
            .flatten()
    })
}

/// Why DXIL is kept from Metal Shader Converter, which would end the whole
/// process on it (it aborts instead of returning an error), or `None`.
///
/// Doubles (`SFI0` bit 0): Apple GPUs have none, and MSC 4.0.3 aborts on a
/// wave sum of them ("Fatal: Unhandled LLVM type for getMinMaxConstants";
/// 6871d6e4…, 926c46ba…, a10179ab…, 5dc3b259…, 6f225b64… translated with
/// doubles on). The Metal host translates without them
/// (`victory_shader::emit::set_metal_host`): only a translation made with
/// other settings (a stale cache) gets here.
pub fn refused_before_conversion(dxil: &[u8]) -> Option<&'static str> {
    if feature_flags(dxil).is_some_and(|f| f & 1 != 0) {
        return Some("the shader uses doubles (MSC aborts the process on wave sums of them)");
    }
    None
}

fn convert_impl(
    dxil: &[u8],
    stage: Stage,
    params: &[RootParam],
    samplers: &[StaticSampler],
    inputs: &[RawInput],
    geometry: bool,
) -> Result<(Converted, Vec<u8>, [u32; 4])> {
    if let Some(why) = refused_before_conversion(dxil) {
        return Err(Error(format!("Metal Shader Converter not asked: {why}")));
    }
    let raw: Vec<RawParam> = params
        .iter()
        .map(|p| match *p {
            RootParam::Constants { register, space, num } => RawParam { kind: 0, shader_register: register, space, num_32bit_values: num },
            RootParam::Cbv { register, space } => RawParam { kind: 1, shader_register: register, space, num_32bit_values: 0 },
            RootParam::Srv { register, space } => RawParam { kind: 2, shader_register: register, space, num_32bit_values: 0 },
            RootParam::Uav { register, space } => RawParam { kind: 3, shader_register: register, space, num_32bit_values: 0 },
            RootParam::TableSrv { register, space } => RawParam { kind: 4, shader_register: register, space, num_32bit_values: 0 },
            RootParam::TableUav { register, space } => RawParam { kind: 5, shader_register: register, space, num_32bit_values: 0 },
        })
        .collect();
    let mut p = CompileParams {
        dxil: dxil.as_ptr().cast(),
        dxil_len: dxil.len() as u64,
        params: raw.as_ptr(),
        num_params: raw.len() as u32,
        num_samplers: samplers.len() as u32,
        samplers: samplers.as_ptr(),
        stage: stage as u32,
        flags: if bounds_check(dxil) { 0 } else { MSC_FLAG_NO_BOUNDS_CHECK },
        metallib: core::ptr::null_mut(),
        metallib_len: 0,
        threadgroup: [0; 3],
        entry: [0; 256],
        num_locations: 0,
        top_level_size: 0,
        locations: [RawLocation::default(); 64],
        ret: -1,
        error: [0; 1024],
    };
    let mut options = GeometryParams {
        stage_in: StageInParams {
            compile: &mut p, elements: inputs.as_ptr(), count: inputs.len() as u32,
            metallib: core::ptr::null_mut(), metallib_len: 0,
        },
        vertex_output_size: 0,
        max_input_primitives: 0,
        max_payload_size: 0,
        gs_instances: 0,
    };
    let ret = if geometry {
        let f = sys::symbol(b"MSC_CompileGeometry\0");
        if f.is_null() {
            return Err(Error("victorymetal.dll has no MSC_CompileGeometry (rebuild native/victorymetal)".into()));
        }
        // SAFETY: signature from msc.h; input pointers live through the call.
        unsafe {
            core::mem::transmute::<*const c_void, unsafe extern "C" fn(*mut GeometryParams) -> i32>(f)(&mut options)
        }
    } else if inputs.is_empty() {
        // SAFETY: every pointer is valid for the call; the unix side fills the outs.
        unsafe { MSC_Compile(&mut p) }
    } else {
        let f = sys::symbol(b"MSC_CompileStageIn\0");
        if f.is_null() {
            return Err(Error("victorymetal.dll has no MSC_CompileStageIn (rebuild native/victorymetal)".into()));
        }
        // SAFETY: signature from msc.h. The input layout and names outlive the call.
        unsafe {
            core::mem::transmute::<*const c_void, unsafe extern "C" fn(*mut StageInParams) -> i32>(f)(&mut options.stage_in)
        }
    };
    if ret != 0 || p.metallib.is_null() {
        return Err(Error(format!("Metal Shader Converter: {}", c_text(&p.error))));
    }
    let copy = |ptr: *mut c_void, len: u64| {
        if ptr.is_null() {
            return Vec::new();
        }
        // SAFETY: unix allocated and initialized len bytes; free with the same allocator.
        unsafe {
            let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), len as usize).to_vec();
            MSC_Free(ptr);
            bytes
        }
    };
    let metallib = copy(p.metallib, p.metallib_len);
    let helper = copy(options.stage_in.metallib, options.stage_in.metallib_len);
    let layout = p.locations[..p.num_locations as usize]
        .iter()
        .map(|l| crate::ir::Location {
            kind: crate::ir::ResourceType::from_raw(l.kind),
            space: l.space,
            slot: l.slot,
            offset: l.offset,
            size: l.size,
        })
        .collect();
    Ok((Converted {
        metallib,
        entry: c_text(&p.entry),
        threadgroup: p.threadgroup,
        top_level: crate::ir::TopLevel::new(layout, p.top_level_size),
    }, helper, [options.vertex_output_size, options.max_input_primitives, options.max_payload_size, options.gs_instances]))
}

impl Device {
    /// A library from metallib bytes.
    pub fn library(&self, metallib: &[u8]) -> Result<Object> {
        let _pool = Pool::new();
        // SAFETY: the bytes are valid for the call (the data object copies them).
        let data = Object::adopt(unsafe { sys::DispatchData_alloc_init(metallib.as_ptr() as u64, metallib.len() as u64) })
            .ok_or(Error("dispatch data".into()))?;
        let mut err = 0;
        // SAFETY: a live device and data object.
        Object::adopt(unsafe { sys::MTLDevice_newLibrary(self.handle(), data.handle(), &mut err) })
            .ok_or_else(|| error_text(err, "newLibraryWithData"))
    }

    /// A compute pipeline from a translated compute shader, with the root
    /// signature `params` (and static samplers). Returns the pipeline and
    /// what the conversion says (threadgroup size, top-level layout).
    pub fn compute_pipeline_dxil(
        &self,
        dxil: &[u8],
        params: &[RootParam],
        samplers: &[StaticSampler],
    ) -> Result<(ComputePipeline, Converted)> {
        let converted = convert(dxil, Stage::Compute, params, samplers)?;
        let pso = self.compute_pipeline_converted(&converted)?;
        Ok((pso, converted))
    }

    /// A compute pipeline from an earlier conversion (a cached one).
    pub fn compute_pipeline_converted(&self, converted: &Converted) -> Result<ComputePipeline> {
        let library = self.library(&converted.metallib)?;
        self.compute_pipeline_from_library(&library, &converted.entry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u16_at(bytes: &[u8], at: usize) -> u16 {
        u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap())
    }

    fn u32_at(bytes: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
    }

    fn u64_at(bytes: &[u8], at: usize) -> u64 {
        u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
    }

    #[test]
    fn doubles_are_kept_from_the_converter() {
        // A container with one part, SFI0 holding `flags`.
        let container = |flags: u64| {
            let mut c = b"DXBC".to_vec();
            c.resize(28, 0);
            c.extend(1u32.to_le_bytes());
            c.extend(36u32.to_le_bytes());
            c.extend(b"SFI0");
            c.extend(8u32.to_le_bytes());
            c.extend(flags.to_le_bytes());
            c
        };
        assert_eq!(feature_flags(&container(0x4001)), Some(0x4001));
        assert!(refused_before_conversion(&container(0x4001)).is_some());
        assert!(refused_before_conversion(&container(0x4000)).is_none());
        assert!(refused_before_conversion(b"not a container").is_none());
        let mut cut = container(1);
        cut.truncate(40);
        assert!(refused_before_conversion(&cut).is_none());
    }

    #[test]
    fn bounds_policy_lists() {
        let a = b"shader a";
        let b = b"shader b";
        let ha = dxil_hash(a);
        assert_eq!(BoundsPolicy::parse_list("# nothing\n\nxyz\n"), BoundsPolicy::All);
        let p = BoundsPolicy::parse_list(&format!("# verified\n{}  # by prefix\n", &ha[..20].to_uppercase()));
        assert!(!p.bounds_check(a));
        assert!(p.bounds_check(b));
        assert!(BoundsPolicy::All.bounds_check(a));
        assert!(!BoundsPolicy::None.bounds_check(b));
    }

    #[test]
    fn geometry_config_stride_and_validation() {
        // vertex_output_size=16 -> (16384-32)/16/3 = 340, capped to 256/3 = 85 prims -> 255 verts.
        let cfg = GeometryConfig { vertex_output_size: 16, max_input_primitives: 32, max_payload_size: 16256 };
        assert_eq!(cfg.object_vertex_stride(), Some(255));

        // Large vertex output (e.g. 256 bytes): (16384-32)/256/3 = 21 prims -> 63 verts.
        let big = GeometryConfig { vertex_output_size: 256, max_input_primitives: 32, max_payload_size: 16384 };
        assert_eq!(big.object_vertex_stride(), Some(63));

        // Invalid configs.
        assert_eq!(GeometryConfig { vertex_output_size: 0, max_input_primitives: 32, max_payload_size: 1024 }.object_vertex_stride(), None);
        assert_eq!(GeometryConfig { vertex_output_size: 16, max_input_primitives: 0, max_payload_size: 1024 }.object_vertex_stride(), None);
        assert_eq!(GeometryConfig { vertex_output_size: 16, max_input_primitives: 257, max_payload_size: 1024 }.object_vertex_stride(), None);
        assert_eq!(GeometryConfig { vertex_output_size: 16, max_input_primitives: 32, max_payload_size: 0 }.object_vertex_stride(), None);
        assert_eq!(GeometryConfig { vertex_output_size: 16, max_input_primitives: 32, max_payload_size: 16385 }.object_vertex_stride(), None);
        assert_eq!(GeometryConfig { vertex_output_size: 6000, max_input_primitives: 32, max_payload_size: 16384 }.object_vertex_stride(), None);
    }

    #[test]
    fn geometry_draw_abi_layout_direct_and_indexed() {
        let cfg = GeometryConfig { vertex_output_size: 16, max_input_primitives: 32, max_payload_size: 16256 };

        // Direct draw spanning 2 object threadgroups (288 > 255) and 3 instances, truncated from 290 to 288.
        let direct = cfg.draw(290, 3, 6, 2, None).unwrap();
        assert_eq!(direct.groups, [2, 3, 1]);
        assert_eq!(direct.object_threads, [255, 1, 1]);
        assert_eq!(direct.mesh_threads, [32, 1, 1]);
        assert_eq!([0, 4, 8, 12, 16].map(|o| u32_at(&direct.args, o)), [288, 3, 6, 2, 0]);
        assert_eq!(u16_at(&direct.info, 0), 0); // kIRNonIndexedDraw
        assert_eq!(direct.info[2], 3); // IRRuntimePrimitiveTypeTriangle
        assert_eq!(direct.info[3], 3); // threadsPerPatch
        assert_eq!(u16_at(&direct.info, 4), 32); // maxInputPrimitivesPerMeshThreadgroup
        assert_eq!(u16_at(&direct.info, 6), 255); // objectThreadgroupVertexStride
        assert_eq!(u16_at(&direct.info, 8), 32); // meshThreadgroupPrimitiveStride
        assert_eq!(u16_at(&direct.info, 10), 1); // gsInstanceCount
        assert_eq!(u16_at(&direct.info, 12), 85); // patchesPerObjectThreadgroup
        assert_eq!(u16_at(&direct.info, 14), 3); // inputControlPointsPerPatch
        assert_eq!(u64_at(&direct.info, 16), 0);

        // R16 indexed with negative baseVertex, R32 indexed.
        let r16 = cfg.draw(6, 1, 1, 0, Some((0, 0x1234_5678_9abc, -5))).unwrap();
        assert_eq!([0, 4, 8, 12, 16].map(|o| u32_at(&r16.args, o)), [6, 1, 1, (-5i32) as u32, 0]);
        assert_eq!(u16_at(&r16.info, 0), 1);
        assert_eq!(u64_at(&r16.info, 16), 0x1234_5678_9abc);

        let r32 = cfg.draw(3, 2, 4, 5, Some((1, 0xdead_beef, 10))).unwrap();
        assert_eq!([0, 4, 8, 12, 16].map(|o| u32_at(&r32.args, o)), [3, 2, 4, 10, 5]);
        assert_eq!(u16_at(&r32.info, 0), 2);
        assert_eq!(u64_at(&r32.info, 16), 0xdead_beef);

        // Rejections.
        assert!(cfg.draw(2, 1, 0, 0, None).is_none());
        assert!(cfg.draw(3, 0, 0, 0, None).is_none());
        assert!(cfg.draw(3, 1, 0, 0, Some((0, 0, 0))).is_none());
        assert!(cfg.draw(3, 1, 0, 0, Some((2, 0x1000, 0))).is_none());
    }
}
