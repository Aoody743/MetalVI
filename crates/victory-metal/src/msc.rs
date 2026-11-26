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

use crate::{ComputePipeline, Device, Error, Object, Pool, Result, error_text, sys};

/// Bind points MSC's shaders read (metal_irconverter_runtime.h).
pub const DESCRIPTOR_HEAP_BIND_POINT: u8 = 0;
pub const SAMPLER_HEAP_BIND_POINT: u8 = 1;
pub const ARGUMENT_BUFFER_BIND_POINT: u8 = 2;

/// IRShaderStage numbering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Stage {
    Vertex = 1,
    Fragment = 2,
    Mesh = 5,
    Amplification = 6,
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
    reserved: u32,
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

/// Converts one DXIL container (as victory-shader emits it) for `stage`.
pub fn convert(dxil: &[u8], stage: Stage, params: &[RootParam], samplers: &[StaticSampler]) -> Result<Converted> {
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
        reserved: 0,
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
    // SAFETY: every pointer is valid for the call; the unix side fills the outs.
    let ret = unsafe { MSC_Compile(&mut p) };
    if ret != 0 || p.metallib.is_null() {
        return Err(Error(format!("Metal Shader Converter: {}", c_text(&p.error))));
    }
    // SAFETY: the unix side malloc'd metallib_len bytes there.
    let metallib = unsafe { std::slice::from_raw_parts(p.metallib.cast::<u8>(), p.metallib_len as usize) }.to_vec();
    // SAFETY: given back to the allocator that made it.
    unsafe { MSC_Free(p.metallib) };
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
    Ok(Converted {
        metallib,
        entry: c_text(&p.entry),
        threadgroup: p.threadgroup,
        top_level: crate::ir::TopLevel::new(layout, p.top_level_size),
    })
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
