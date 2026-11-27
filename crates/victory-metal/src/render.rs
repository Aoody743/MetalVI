//! Render pipeline state, render-pass descriptor and draw-command encoding
//! through the winemetal unixlib (offscreen targets first).
//!
//! Like [`crate::CommandBuffer::compute`], the render commands are built here
//! as a linked chain of plain structs and handed to the unix side in one
//! `encodeCommands` call. The encoder itself is autoreleased and stays valid
//! while the current [`crate::Pool`] lives.
//!
//! Bind points follow the Metal Shader Converter runtime
//! (`metal_irconverter_runtime.h`, see [`crate::msc`]): descriptor heap at 0,
//! sampler heap at 1, top-level argument buffer at 2. On a render encoder the
//! heap buffers and top-level bytes can be bound to both shader stages.

use core::ffi::c_void;

use crate::sys;
use crate::texture::Texture;
use crate::{Buffer, CommandBuffer, Device, Error, Object, Pool, Result, error_text};
use sys::obj_handle_t;

/// A Metal render pipeline state (+1 reference released on drop; `Clone` retains).
#[derive(Clone, Debug)]
pub struct RenderPipeline(Object);

impl RenderPipeline {
    pub fn handle(&self) -> obj_handle_t {
        self.0.handle()
    }
}

/// One color attachment of a render pass.
#[derive(Clone, Copy, Debug)]
pub struct ColorAttachment<'a> {
    pub texture: &'a Texture,
    pub load_action: sys::WMTLoadAction,
    pub store_action: sys::WMTStoreAction,
    /// Clear color, RGBA in 0..1.
    pub clear: [f64; 4],
}

/// One step recorded into a render encoder.
pub enum Render<'a> {
    /// Select the pipeline state.
    Pipeline { pipeline: &'a RenderPipeline },
    /// `setVertexBuffer:offset:atIndex:` (heap buffers, vertex stage).
    VertexBuffer { buffer: &'a Buffer, offset: u64, index: u8 },
    /// `setVertexBytes:length:atIndex:`.
    VertexBytes { bytes: &'a [u8], index: u8 },
    /// `setFragmentBuffer:offset:atIndex:` (heap buffers, fragment stage).
    FragmentBuffer { buffer: &'a Buffer, offset: u64, index: u8 },
    /// `setFragmentBytes:length:atIndex:` (top-level argument buffer bytes).
    FragmentBytes { bytes: &'a [u8], index: u8 },
    /// `setFragmentTexture:atIndex:`.
    FragmentTexture { texture: &'a Texture, index: u8 },
    /// `setMeshBuffer:offset:atIndex:` / `setObjectBuffer:offset:atIndex:`.
    MeshBuffer { buffer: &'a Buffer, offset: u64, index: u8 },
    ObjectBuffer { buffer: &'a Buffer, offset: u64, index: u8 },
    /// `setViewports:count:` (up to 16).
    Viewports { viewports: &'a [sys::WMTViewport] },
    /// `setScissorRects:count:` (up to 16).
    Scissors { rects: &'a [sys::WMTScissorRect] },
    /// Fill mode, cull mode, depth clip mode, front-facing winding and depth
    /// bias, at once (Metal enum values).
    Rasterizer { fill: u8, cull: u8, depth_clip: u8, winding: u8, depth_bias: f32, slope_scale: f32, clamp: f32 },
    /// `setDepthStencilState:` + `setStencilReferenceValue:`.
    DepthStencil { state: &'a DepthStencilState, stencil_ref: u8 },
    /// `setBlendColorRed:green:blue:alpha:` + `setStencilReferenceValue:`.
    BlendColorStencil { color: [f32; 4], stencil_ref: u8 },
    /// `drawIndexedPrimitives:...` (index type 0: 16-bit, 1: 32-bit).
    DrawIndexed {
        primitive_type: sys::WMTPrimitiveType,
        index_type: u8,
        index_count: u64,
        index_buffer: &'a Buffer,
        index_offset: u64,
        instance_count: u32,
        base_vertex: i32,
        base_instance: u32,
    },
    /// `drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`.
    DrawMesh { groups: [u64; 3], object_threads: [u64; 3], mesh_threads: [u64; 3] },
    /// `drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:`.
    Draw {
        primitive_type: sys::WMTPrimitiveType,
        vertex_start: u64,
        vertex_count: u64,
        instance_count: u32,
        base_instance: u32,
    },
    /// Metal reads a 16-byte MTLDrawPrimitivesIndirectArguments at `offset`.
    DrawIndirect { primitive_type: sys::WMTPrimitiveType, arguments: &'a Buffer, offset: u64 },
    /// Metal reads a 20-byte MTLDrawIndexedPrimitivesIndirectArguments at `offset`.
    DrawIndexedIndirect {
        primitive_type: sys::WMTPrimitiveType,
        index_type: u8,
        index_buffer: &'a Buffer,
        index_offset: u64,
        arguments: &'a Buffer,
        offset: u64,
    },
    /// Metal reads three u32 threadgroup counts at `offset`.
    DrawMeshIndirect {
        arguments: &'a Buffer,
        offset: u64,
        object_threads: [u64; 3],
        mesh_threads: [u64; 3],
    },
}

/// One winemetal render command: every one starts like [`sys::wmtcmd_base`].
trait RenderLink {
    fn as_base(&self) -> *const sys::wmtcmd_base;
    fn set_next(&mut self, next: *const sys::wmtcmd_base);
}

macro_rules! render_link {
    ($($t:ty),*) => {$(
        impl RenderLink for $t {
            fn as_base(&self) -> *const sys::wmtcmd_base {
                (self as *const Self).cast()
            }
            fn set_next(&mut self, next: *const sys::wmtcmd_base) {
                self.next = sys::WMTMemoryPointer { ptr: next as *mut c_void };
            }
        }
    )*};
}
render_link!(
    sys::wmtcmd_render_setpso,
    sys::wmtcmd_render_setbuffer,
    sys::wmtcmd_render_setbytes,
    sys::wmtcmd_render_settexture,
    sys::wmtcmd_render_draw,
    sys::wmtcmd_render_setviewports,
    sys::wmtcmd_render_setscissorrects,
    sys::wmtcmd_render_setrasterizerstate,
    sys::wmtcmd_render_setdsso,
    sys::wmtcmd_render_setblendcolor,
    sys::wmtcmd_render_draw_indexed,
    sys::wmtcmd_render_draw_meshthreadgroups,
    sys::wmtcmd_render_draw_indirect,
    sys::wmtcmd_render_draw_indexed_indirect,
    sys::wmtcmd_render_draw_meshthreadgroups_indirect
);

/// A Metal depth-stencil state (+1 reference released on drop).
#[derive(Clone, Debug)]
pub struct DepthStencilState(Object);

impl DepthStencilState {
    pub fn handle(&self) -> obj_handle_t {
        self.0.handle()
    }
}

/// One stencil face (Metal enum values).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct StencilFace {
    pub compare: u8,
    pub stencil_fail: u8,
    pub depth_fail: u8,
    pub pass: u8,
    pub read_mask: u8,
    pub write_mask: u8,
}

/// A depth-stencil state description (Metal enum values).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DepthStencilDesc {
    /// MTLCompareFunction (7 = always when depth testing is off).
    pub depth_compare: u8,
    pub depth_write: bool,
    pub stencil: bool,
    pub front: StencilFace,
    pub back: StencilFace,
}

/// One color attachment of a [`RenderPass`].
#[derive(Clone, Copy, Debug)]
pub struct PassColor<'a> {
    pub texture: &'a Texture,
    pub level: u16,
    pub slice: u16,
    pub load: sys::WMTLoadAction,
    pub store: sys::WMTStoreAction,
    pub clear: [f64; 4],
}

/// The depth or stencil attachment of a [`RenderPass`].
#[derive(Clone, Copy, Debug)]
pub struct PassDepth<'a> {
    pub texture: &'a Texture,
    pub level: u16,
    pub slice: u16,
    pub load: sys::WMTLoadAction,
    pub store: sys::WMTStoreAction,
    /// Depth clear value (the stencil attachment uses `clear_stencil`).
    pub clear_depth: f32,
    pub clear_stencil: u8,
}

/// A whole render pass description.
#[derive(Clone, Copy, Debug, Default)]
pub struct RenderPass<'a> {
    pub colors: [Option<PassColor<'a>>; 8],
    pub depth: Option<PassDepth<'a>>,
    pub stencil: Option<PassDepth<'a>>,
    pub width: u32,
    pub height: u32,
    pub samples: u8,
    /// Layers rendered to (1 unless layered rendering).
    pub array_length: u8,
}

fn null_next() -> sys::WMTMemoryPointer {
    sys::WMTMemoryPointer { ptr: core::ptr::null_mut() }
}

impl Device {
    /// A render pipeline from a vertex function of `vertex_library` and a
    /// fragment function of `fragment_library` (the two stages are converted
    /// from separate DXIL containers, so they usually live in separate
    /// metallibs).
    ///
    /// `color_format` is the pixel format of color attachment 0; no blend,
    /// full write mask, triangle topology, single sample, no depth/stencil.
    pub fn render_pipeline(
        &self,
        vertex_library: &Object,
        vertex_function: &str,
        fragment_library: &Object,
        fragment_function: &str,
        color_format: sys::WMTPixelFormat,
        sample_count: u8,
    ) -> Result<RenderPipeline> {
        let _pool = Pool::new();
        let vname = std::ffi::CString::new(vertex_function).map_err(|_| Error("vertex name has a NUL".into()))?;
        let fname = std::ffi::CString::new(fragment_function).map_err(|_| Error("fragment name has a NUL".into()))?;
        // SAFETY: a live library; `newFunction` returns owned (+1).
        let vertex = Object::adopt(unsafe { sys::MTLLibrary_newFunction(vertex_library.handle(), vname.as_ptr()) })
            .ok_or_else(|| Error(format!("no vertex function {vertex_function:?} in the library")))?;
        let fragment = Object::adopt(unsafe { sys::MTLLibrary_newFunction(fragment_library.handle(), fname.as_ptr()) })
            .ok_or_else(|| Error(format!("no fragment function {fragment_function:?} in the library")))?;

        // SAFETY: all-zero is the default for the info struct (as the C smoke
        // test does); bitfields and reserved fields start at zero.
        let mut info: sys::WMTRenderPipelineInfo = unsafe { std::mem::zeroed() };
        info.colors[0].pixel_format = color_format;
        info.colors[0].rgb_blend_operation = sys::WMTBlendOperation::WMTBlendOperationAdd;
        info.colors[0].alpha_blend_operation = sys::WMTBlendOperation::WMTBlendOperationAdd;
        info.colors[0].src_rgb_blend_factor = sys::WMTBlendFactor::WMTBlendFactorOne;
        info.colors[0].dst_rgb_blend_factor = sys::WMTBlendFactor::WMTBlendFactorZero;
        info.colors[0].src_alpha_blend_factor = sys::WMTBlendFactor::WMTBlendFactorOne;
        info.colors[0].dst_alpha_blend_factor = sys::WMTBlendFactor::WMTBlendFactorZero;
        info.colors[0].write_mask = sys::WMTColorWriteMask::WMTColorWriteMaskAll.0;
        info.colors[0].blending_enabled = false;
        info.rasterization_enabled = true;
        info.raster_sample_count = sample_count;
        info.depth_pixel_format = sys::WMTPixelFormat::WMTPixelFormatInvalid;
        info.stencil_pixel_format = sys::WMTPixelFormat::WMTPixelFormatInvalid;
        info.vertex_function = vertex.handle();
        info.fragment_function = fragment.handle();
        info.input_primitive_topology = sys::WMTPrimitiveTopologyClass::WMTPrimitiveTopologyClassTriangle;
        // Metal rejects a zero tessellation factor even without tessellation.
        info.max_tessellation_factor = 64;

        let mut err = 0;
        // SAFETY: info is valid for the call; newRenderPipelineState returns owned (+1).
        let pso = Object::adopt(unsafe {
            sys::MTLDevice_newRenderPipelineState(self.handle(), &info, &mut err)
        })
        .ok_or_else(|| error_text(err, "newRenderPipelineState"))?;
        Ok(RenderPipeline(pso))
    }
}

/// One color attachment of a [`RenderPipelineDesc`] (Metal enum values).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ColorDesc {
    /// MTLPixelFormat; 0 = no attachment.
    pub pixel_format: u32,
    pub blending: bool,
    /// MTLBlendFactor / MTLBlendOperation values.
    pub src_rgb: u8,
    pub dst_rgb: u8,
    pub op_rgb: u8,
    pub src_alpha: u8,
    pub dst_alpha: u8,
    pub op_alpha: u8,
    /// MTLColorWriteMask bits.
    pub write_mask: u8,
}

/// A whole render pipeline description (Metal enum values throughout).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderPipelineDesc {
    pub colors: [ColorDesc; 8],
    /// MTLPixelFormat of the depth / stencil attachments; 0 = none.
    pub depth_format: u32,
    pub stencil_format: u32,
    pub sample_count: u8,
    pub alpha_to_coverage: bool,
    /// MTLPrimitiveTopologyClass; 0 = unspecified.
    pub topology_class: u32,
}

impl Device {
    /// A render pipeline from `desc`, a vertex function and, when given, a
    /// fragment function (depth-only pipelines have none).
    pub fn render_pipeline_desc(
        &self,
        vertex: (&Object, &str),
        fragment: Option<(&Object, &str)>,
        desc: &RenderPipelineDesc,
    ) -> Result<RenderPipeline> {
        self.render_pipeline_impl(vertex, fragment, desc, None)
    }

    /// A render pipeline whose vertex stage calls a synthesized stage-in function.
    pub fn render_pipeline_stage_in(
        &self,
        vertex: (&Object, &str),
        fragment: Option<(&Object, &str)>,
        desc: &RenderPipelineDesc,
        stage_in_library: &Object,
    ) -> Result<RenderPipeline> {
        self.render_pipeline_impl(vertex, fragment, desc, Some(stage_in_library))
    }

    fn render_pipeline_impl(
        &self,
        vertex: (&Object, &str),
        fragment: Option<(&Object, &str)>,
        desc: &RenderPipelineDesc,
        stage_in_library: Option<&Object>,
    ) -> Result<RenderPipeline> {
        let _pool = Pool::new();
        let function = |lib: &Object, name: &str| -> Result<Object> {
            let c = std::ffi::CString::new(name).map_err(|_| Error("function name has a NUL".into()))?;
            // SAFETY: a live library; `newFunction` returns owned (+1).
            Object::adopt(unsafe { sys::MTLLibrary_newFunction(lib.handle(), c.as_ptr()) })
                .ok_or_else(|| Error(format!("no function {name:?} in the library")))
        };
        let vertex = function(vertex.0, vertex.1)?;
        let fragment = match fragment {
            Some((lib, name)) => Some(function(lib, name)?),
            None => None,
        };
        // SAFETY: all-zero is the default for the info struct.
        let mut info: sys::WMTRenderPipelineInfo = unsafe { std::mem::zeroed() };
        for (out, c) in info.colors.iter_mut().zip(&desc.colors) {
            out.pixel_format = sys::WMTPixelFormat(c.pixel_format);
            out.blending_enabled = c.blending;
            out.src_rgb_blend_factor = sys::WMTBlendFactor(c.src_rgb);
            out.dst_rgb_blend_factor = sys::WMTBlendFactor(c.dst_rgb);
            out.rgb_blend_operation = sys::WMTBlendOperation(c.op_rgb);
            out.src_alpha_blend_factor = sys::WMTBlendFactor(c.src_alpha);
            out.dst_alpha_blend_factor = sys::WMTBlendFactor(c.dst_alpha);
            out.alpha_blend_operation = sys::WMTBlendOperation(c.op_alpha);
            out.write_mask = c.write_mask.into();
        }
        info.alpha_to_coverage_enabled = desc.alpha_to_coverage;
        info.rasterization_enabled = true;
        info.raster_sample_count = desc.sample_count.max(1);
        info.depth_pixel_format = sys::WMTPixelFormat(desc.depth_format);
        info.stencil_pixel_format = sys::WMTPixelFormat(desc.stencil_format);
        info.vertex_function = vertex.handle();
        info.fragment_function = fragment.as_ref().map_or(0, |f| f.handle());
        info.input_primitive_topology = sys::WMTPrimitiveTopologyClass(desc.topology_class);
        // Metal rejects a zero tessellation factor even without tessellation.
        info.max_tessellation_factor = 64;
        let mut err = 0;
        // SAFETY: info is valid for the call; newRenderPipelineState returns owned (+1).
        let pso = Object::adopt(unsafe {
            match stage_in_library {
                Some(lib) => sys::VM_NewRenderPipelineStageIn(self.handle(), &info, lib.handle(), &mut err),
                None => sys::MTLDevice_newRenderPipelineState(self.handle(), &info, &mut err),
            }
        })
            .ok_or_else(|| error_text(err, "newRenderPipelineState"))?;
        Ok(RenderPipeline(pso))
    }
}

impl Device {
    /// A mesh render pipeline from `desc`, a mesh function and, when given,
    /// a fragment function (no object stage).
    pub fn mesh_render_pipeline_desc(
        &self,
        mesh: (&Object, &str),
        fragment: Option<(&Object, &str)>,
        desc: &RenderPipelineDesc,
    ) -> Result<RenderPipeline> {
        let _pool = Pool::new();
        let function = |lib: &Object, name: &str| -> Result<Object> {
            let c = std::ffi::CString::new(name).map_err(|_| Error("function name has a NUL".into()))?;
            // SAFETY: a live library; `newFunction` returns owned (+1).
            Object::adopt(unsafe { sys::MTLLibrary_newFunction(lib.handle(), c.as_ptr()) })
                .ok_or_else(|| Error(format!("no function {name:?} in the library")))
        };
        let mesh = function(mesh.0, mesh.1)?;
        let fragment = match fragment {
            Some((lib, name)) => Some(function(lib, name)?),
            None => None,
        };
        let mut info = Self::mesh_pipeline_info(desc);
        info.mesh_function = mesh.handle();
        info.fragment_function = fragment.as_ref().map_or(0, |f| f.handle());
        let mut err = 0;
        // SAFETY: info is valid for the call; returns owned (+1).
        let pso = Object::adopt(unsafe { sys::MTLDevice_newMeshRenderPipelineState(self.handle(), &info, &mut err) })
            .ok_or_else(|| error_text(err, "newMeshRenderPipelineState"))?;
        Ok(RenderPipeline(pso))
    }

    /// An MSC vertex/geometry pair, linked as an object/mesh render pipeline.
    pub fn geometry_render_pipeline_desc(
        &self, vertex: (&Object, &str), mesh: (&Object, &str), stage_in: &Object,
        fragment: Option<(&Object, &str)>, desc: &RenderPipelineDesc, config: crate::msc::GeometryConfig,
    ) -> Result<RenderPipeline> {
        let _pool = Pool::new();
        let cstring = |name: &str| std::ffi::CString::new(name).map_err(|_| Error("function name has a NUL".into()));
        let function = |lib: &Object, name: &str| -> Result<Object> {
            let c = cstring(name)?;
            // SAFETY: library and string live through the call; returns owned (+1).
            Object::adopt(unsafe { sys::MTLLibrary_newFunction(lib.handle(), c.as_ptr()) })
                .ok_or_else(|| Error(format!("no function {name:?} in the library")))
        };
        let fragment = fragment.map(|(l, n)| function(l, n)).transpose()?;
        let vertex_name = cstring(vertex.1)?;
        let geometry_name = cstring(mesh.1)?;
        if config.object_vertex_stride().is_none() {
            return Err(Error("invalid MSC geometry configuration".into()));
        }
        #[repr(C)]
        struct Functions {
            vertex_library: obj_handle_t,
            geometry_library: obj_handle_t,
            stage_in_library: obj_handle_t,
            vertex_name: sys::WMTConstMemoryPointer,
            geometry_name: sys::WMTConstMemoryPointer,
            vertex_output_size: u32,
        }
        let functions = Functions {
            vertex_library: vertex.0.handle(), geometry_library: mesh.0.handle(), stage_in_library: stage_in.handle(),
            vertex_name: sys::WMTConstMemoryPointer { ptr: vertex_name.as_ptr().cast() },
            geometry_name: sys::WMTConstMemoryPointer { ptr: geometry_name.as_ptr().cast() },
            vertex_output_size: config.vertex_output_size,
        };
        let mut info = Self::mesh_pipeline_info(desc);
        info.fragment_function = fragment.as_ref().map_or(0, |f| f.handle());
        let stride_payload = config.object_vertex_stride().unwrap_or(0).saturating_mul(config.vertex_output_size);
        info.payload_memory_length = config.max_payload_size.max(stride_payload).saturating_add(32).min(16384) as u16;
        let raw = sys::symbol(b"VM_NewGeometryPipeline\0");
        if raw.is_null() {
            return Err(Error("victorymetal.dll has no VM_NewGeometryPipeline (rebuild native/victorymetal)".into()));
        }
        let mut err = 0;
        // SAFETY: the shared C structs and all pointed-to data live through the call.
        let pso = unsafe {
            let f: unsafe extern "C" fn(obj_handle_t, *const sys::WMTMeshRenderPipelineInfo, *const Functions, *mut obj_handle_t) -> obj_handle_t =
                core::mem::transmute(raw);
            f(self.handle(), &info, &functions, &mut err)
        };
        Object::adopt(pso).map(RenderPipeline).ok_or_else(|| error_text(err, "VM_NewGeometryPipeline"))
    }

    fn mesh_pipeline_info(desc: &RenderPipelineDesc) -> sys::WMTMeshRenderPipelineInfo {
        // SAFETY: all-zero is the default for the info struct.
        let mut info: sys::WMTMeshRenderPipelineInfo = unsafe { std::mem::zeroed() };
        for (out, c) in info.colors.iter_mut().zip(&desc.colors) {
            out.pixel_format = sys::WMTPixelFormat(c.pixel_format);
            out.blending_enabled = c.blending;
            out.src_rgb_blend_factor = sys::WMTBlendFactor(c.src_rgb);
            out.dst_rgb_blend_factor = sys::WMTBlendFactor(c.dst_rgb);
            out.rgb_blend_operation = sys::WMTBlendOperation(c.op_rgb);
            out.src_alpha_blend_factor = sys::WMTBlendFactor(c.src_alpha);
            out.dst_alpha_blend_factor = sys::WMTBlendFactor(c.dst_alpha);
            out.alpha_blend_operation = sys::WMTBlendOperation(c.op_alpha);
            out.write_mask = c.write_mask;
        }
        info.alpha_to_coverage_enabled = desc.alpha_to_coverage;
        info.rasterization_enabled = true;
        info.raster_sample_count = desc.sample_count.max(1);
        info.depth_pixel_format = sys::WMTPixelFormat(desc.depth_format);
        info.stencil_pixel_format = sys::WMTPixelFormat(desc.stencil_format);
        info
    }

    /// A depth-stencil state from `desc`.
    pub fn depth_stencil_state(&self, desc: &DepthStencilDesc) -> Result<DepthStencilState> {
        let face = |f: &StencilFace| sys::WMTStencilInfo {
            enabled: desc.stencil,
            depth_stencil_pass_op: sys::WMTStencilOperation(f.pass),
            stencil_fail_op: sys::WMTStencilOperation(f.stencil_fail),
            depth_fail_op: sys::WMTStencilOperation(f.depth_fail),
            stencil_compare_function: sys::WMTCompareFunction(f.compare),
            write_mask: f.write_mask,
            read_mask: f.read_mask,
        };
        let info = sys::WMTDepthStencilInfo {
            depth_compare_function: sys::WMTCompareFunction(desc.depth_compare),
            depth_write_enabled: desc.depth_write,
            front_stencil: face(&desc.front),
            back_stencil: face(&desc.back),
        };
        let _pool = Pool::new();
        // SAFETY: info valid for the call; returns owned (+1).
        Object::adopt(unsafe { sys::MTLDevice_newDepthStencilState(self.handle(), &info) })
            .map(DepthStencilState)
            .ok_or_else(|| Error("newDepthStencilState".into()))
    }
}

impl CommandBuffer {
    /// Opens a render pass for `attachments`, encodes `steps`, and ends the
    /// pass. The command buffer is not committed here.
    pub fn render_pass(
        &self,
        attachments: &[ColorAttachment<'_>],
        width: u32,
        height: u32,
        steps: &[Render<'_>],
    ) -> Result<()> {
        let mut pass = RenderPass { width, height, samples: 1, array_length: 0, ..Default::default() };
        for (slot, a) in pass.colors.iter_mut().zip(attachments) {
            *slot = Some(PassColor { texture: a.texture, level: 0, slice: 0, load: a.load_action, store: a.store_action, clear: a.clear });
        }
        self.render_pass_with(&pass, steps)
    }

    /// Opens the render pass `desc`, encodes `steps`, and ends the pass.
    /// The command buffer is not committed here.
    pub fn render_pass_with(&self, desc: &RenderPass<'_>, steps: &[Render<'_>]) -> Result<()> {
        {
            let mut w = self.1.lock().unwrap_or_else(|e| e.into_inner());
            let attachments = desc.colors.iter().flatten().count() + usize::from(desc.depth.is_some()) + usize::from(desc.stencil.is_some());
            let area = crate::volume([desc.width.into(), desc.height.into(), u64::from(desc.samples.max(1)) * u64::from(desc.array_length.max(1))]);
            w.attachment_samples = w.attachment_samples.saturating_add(area.saturating_mul(attachments.max(1) as u64));
            for step in steps {
                let inputs = match *step {
                    Render::Draw { vertex_count, instance_count, .. } => vertex_count.saturating_mul(instance_count.into()),
                    Render::DrawIndexed { index_count, instance_count, .. } => index_count.saturating_mul(instance_count.into()),
                    Render::DrawIndirect { .. } | Render::DrawIndexedIndirect { .. } => crate::indirect::INPUTS_PER_RECORD.into(),
                    Render::DrawMesh { groups, .. } => { w.groups = w.groups.saturating_add(crate::volume(groups)); continue; }
                    Render::DrawMeshIndirect { .. } => { w.groups = w.groups.saturating_add(crate::indirect::GROUPS_PER_RECORD.into()); continue; }
                    _ => continue,
                };
                w.draw_inputs = w.draw_inputs.saturating_add(inputs);
            }
        }
        let _pool = Pool::new();
        // SAFETY: all-zero pass descriptor; fields set below.
        let mut pass: sys::WMTRenderPassInfo = unsafe { std::mem::zeroed() };
        pass.default_raster_sample_count = desc.samples.max(1);
        pass.render_target_array_length = desc.array_length;
        pass.render_target_width = desc.width;
        pass.render_target_height = desc.height;
        for (out, a) in pass.colors.iter_mut().zip(&desc.colors) {
            let Some(a) = a else { continue };
            out.texture = a.texture.handle();
            out.load_action = a.load;
            out.store_action = a.store;
            out.level = a.level;
            out.slice = a.slice;
            out.clear_color = sys::WMTClearColor { r: a.clear[0], g: a.clear[1], b: a.clear[2], a: a.clear[3] };
        }
        if let Some(d) = &desc.depth {
            pass.depth.texture = d.texture.handle();
            pass.depth.load_action = d.load;
            pass.depth.store_action = d.store;
            pass.depth.level = d.level;
            pass.depth.slice = d.slice;
            pass.depth.clear_depth = d.clear_depth;
        }
        if let Some(d) = &desc.stencil {
            pass.stencil.texture = d.texture.handle();
            pass.stencil.load_action = d.load;
            pass.stencil.store_action = d.store;
            pass.stencil.level = d.level;
            pass.stencil.slice = d.slice;
            pass.stencil.clear_stencil = d.clear_stencil;
        }

        // SAFETY: a live command buffer; the encoder is autoreleased.
        let encoder = unsafe { sys::MTLCommandBuffer_renderCommandEncoder(self.handle(), &mut pass) };
        if encoder == 0 {
            return Err(Error("renderCommandEncoder".into()));
        }

        // Boxed so the commands' addresses do not move while linked.
        let mut chain: Vec<Box<dyn RenderLink>> = Vec::with_capacity(steps.len());
        for step in steps {
            chain.push(match *step {
                Render::Pipeline { pipeline } => Box::new(sys::wmtcmd_render_setpso {
                    type_: sys::WMTRenderCommandType::WMTRenderCommandSetPSO,
                    reserved: [0; 3],
                    next: null_next(),
                    pso: pipeline.handle(),
                }),
                Render::VertexBuffer { buffer, offset, index } => Box::new(sys::wmtcmd_render_setbuffer {
                    type_: sys::WMTRenderCommandType::WMTRenderCommandSetVertexBuffer,
                    reserved: [0; 3],
                    next: null_next(),
                    buffer: buffer.handle(),
                    offset,
                    index,
                }),
                Render::FragmentBuffer { buffer, offset, index } => Box::new(sys::wmtcmd_render_setbuffer {
                    type_: sys::WMTRenderCommandType::WMTRenderCommandSetFragmentBuffer,
                    reserved: [0; 3],
                    next: null_next(),
                    buffer: buffer.handle(),
                    offset,
                    index,
                }),
                Render::FragmentBytes { bytes, index } | Render::VertexBytes { bytes, index } => Box::new(sys::wmtcmd_render_setbytes {
                    type_: if matches!(step, Render::VertexBytes { .. }) {
                        sys::WMTRenderCommandType::WMTRenderCommandSetVertexBytes
                    } else {
                        sys::WMTRenderCommandType::WMTRenderCommandSetFragmentBytes
                    },
                    reserved: [0; 3],
                    next: null_next(),
                    bytes: sys::WMTMemoryPointer { ptr: bytes.as_ptr() as *mut c_void },
                    length: bytes.len() as u64,
                    index,
                }),
                Render::FragmentTexture { texture, index } => Box::new(sys::wmtcmd_render_settexture {
                    type_: sys::WMTRenderCommandType::WMTRenderCommandSetFragmentTexture,
                    reserved: [0; 3],
                    next: null_next(),
                    texture: texture.handle(),
                    index,
                }),
                Render::MeshBuffer { buffer, offset, index } => Box::new(sys::wmtcmd_render_setbuffer {
                    type_: sys::WMTRenderCommandType::WMTRenderCommandSetMeshBuffer,
                    reserved: [0; 3],
                    next: null_next(),
                    buffer: buffer.handle(),
                    offset,
                    index,
                }),
                Render::ObjectBuffer { buffer, offset, index } => Box::new(sys::wmtcmd_render_setbuffer {
                    type_: sys::WMTRenderCommandType::WMTRenderCommandSetObjectBuffer,
                    reserved: [0; 3],
                    next: null_next(),
                    buffer: buffer.handle(),
                    offset,
                    index,
                }),
                Render::Viewports { viewports } => Box::new(sys::wmtcmd_render_setviewports {
                    type_: sys::WMTRenderCommandType::WMTRenderCommandSetViewports,
                    reserved: [0; 3],
                    next: null_next(),
                    viewports: sys::WMTMemoryPointer { ptr: viewports.as_ptr() as *mut c_void },
                    viewport_count: viewports.len().min(16) as u8,
                }),
                Render::Scissors { rects } => Box::new(sys::wmtcmd_render_setscissorrects {
                    type_: sys::WMTRenderCommandType::WMTRenderCommandSetScissorRects,
                    reserved: [0; 3],
                    next: null_next(),
                    scissor_rects: sys::WMTMemoryPointer { ptr: rects.as_ptr() as *mut c_void },
                    rect_count: rects.len().min(16) as u8,
                }),
                Render::Rasterizer { fill, cull, depth_clip, winding, depth_bias, slope_scale, clamp } => {
                    Box::new(sys::wmtcmd_render_setrasterizerstate {
                        type_: sys::WMTRenderCommandType::WMTRenderCommandSetRasterizerState,
                        reserved: [0; 3],
                        next: null_next(),
                        fill_mode: sys::WMTTriangleFillMode(fill),
                        cull_mode: sys::WMTCullMode(cull),
                        depth_clip_mode: sys::WMTDepthClipMode(depth_clip),
                        winding: sys::WMTWinding(winding),
                        depth_bias,
                        scole_scale: slope_scale,
                        depth_bias_clamp: clamp,
                    })
                }
                Render::DepthStencil { state, stencil_ref } => Box::new(sys::wmtcmd_render_setdsso {
                    type_: sys::WMTRenderCommandType::WMTRenderCommandSetDSSO,
                    reserved: [0; 3],
                    next: null_next(),
                    dsso: state.handle(),
                    stencil_ref,
                }),
                Render::BlendColorStencil { color, stencil_ref } => Box::new(sys::wmtcmd_render_setblendcolor {
                    type_: sys::WMTRenderCommandType::WMTRenderCommandSetBlendFactorAndStencilRef,
                    reserved: [0; 3],
                    next: null_next(),
                    red: color[0],
                    green: color[1],
                    blue: color[2],
                    alpha: color[3],
                    stencil_ref,
                }),
                Render::DrawIndexed {
                    primitive_type,
                    index_type,
                    index_count,
                    index_buffer,
                    index_offset,
                    instance_count,
                    base_vertex,
                    base_instance,
                } => Box::new(sys::wmtcmd_render_draw_indexed {
                    type_: sys::WMTRenderCommandType::WMTRenderCommandDrawIndexed,
                    reserved: [0; 3],
                    next: null_next(),
                    primitive_type,
                    index_type: sys::WMTIndexType(index_type),
                    index_count,
                    index_buffer: index_buffer.handle(),
                    index_buffer_offset: index_offset,
                    instance_count,
                    base_vertex,
                    base_instance,
                }),
                Render::DrawMesh { groups, object_threads, mesh_threads } => {
                    let size = |v: [u64; 3]| sys::WMTSize { width: v[0], height: v[1], depth: v[2] };
                    Box::new(sys::wmtcmd_render_draw_meshthreadgroups {
                        type_: sys::WMTRenderCommandType::WMTRenderCommandDrawMeshThreadgroups,
                        reserved: [0; 3],
                        next: null_next(),
                        threadgroup_per_grid: size(groups),
                        object_threadgroup_size: size(object_threads),
                        mesh_threadgroup_size: size(mesh_threads),
                    })
                }
                Render::Draw { primitive_type, vertex_start, vertex_count, instance_count, base_instance } => {
                    Box::new(sys::wmtcmd_render_draw {
                        type_: sys::WMTRenderCommandType::WMTRenderCommandDraw,
                        reserved: [0; 3],
                        next: null_next(),
                        primitive_type,
                        vertex_start,
                        vertex_count,
                        instance_count,
                        base_instance,
                    })
                }
                Render::DrawIndirect { primitive_type, arguments, offset } => Box::new(sys::wmtcmd_render_draw_indirect {
                    type_: sys::WMTRenderCommandType::WMTRenderCommandDrawIndirect,
                    reserved: [0; 3],
                    next: null_next(),
                    primitive_type,
                    indirect_args_buffer: arguments.handle(),
                    indirect_args_offset: offset,
                }),
                Render::DrawIndexedIndirect { primitive_type, index_type, index_buffer, index_offset, arguments, offset } => {
                    Box::new(sys::wmtcmd_render_draw_indexed_indirect {
                        type_: sys::WMTRenderCommandType::WMTRenderCommandDrawIndexedIndirect,
                        reserved: [0; 3],
                        next: null_next(),
                        primitive_type,
                        index_type: sys::WMTIndexType(index_type),
                        index_buffer: index_buffer.handle(),
                        index_buffer_offset: index_offset,
                        indirect_args_buffer: arguments.handle(),
                        indirect_args_offset: offset,
                    })
                }
                Render::DrawMeshIndirect { arguments, offset, object_threads, mesh_threads } => {
                    let size = |v: [u64; 3]| sys::WMTSize { width: v[0], height: v[1], depth: v[2] };
                    Box::new(sys::wmtcmd_render_draw_meshthreadgroups_indirect {
                        type_: sys::WMTRenderCommandType::WMTRenderCommandDrawMeshThreadgroupsIndirect,
                        reserved: [0; 3],
                        next: null_next(),
                        indirect_args_buffer: arguments.handle(),
                        indirect_args_offset: offset,
                        object_threadgroup_size: size(object_threads),
                        mesh_threadgroup_size: size(mesh_threads),
                    })
                }
            });
        }
        for i in 1..chain.len() {
            let next = chain[i].as_base();
            chain[i - 1].set_next(next);
        }
        if let Some(head) = chain.first() {
            // SAFETY: the chain is well formed and lives until the call returns.
            unsafe { sys::MTLRenderCommandEncoder_encodeCommands(encoder, head.as_base()) };
        }
        // SAFETY: a live encoder.
        unsafe { sys::MTLCommandEncoder_endEncoding(encoder) };
        Ok(())
    }
}
