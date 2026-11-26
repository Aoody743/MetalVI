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
//! heap buffers are set on both the vertex and fragment stages; the top-level
//! bytes go to the fragment stage (the vertex stage has no `setVertexBytes`
//! command in winemetal).

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
    /// `setFragmentBuffer:offset:atIndex:` (heap buffers, fragment stage).
    FragmentBuffer { buffer: &'a Buffer, offset: u64, index: u8 },
    /// `setFragmentBytes:length:atIndex:` (top-level argument buffer bytes).
    FragmentBytes { bytes: &'a [u8], index: u8 },
    /// `setFragmentTexture:atIndex:`.
    FragmentTexture { texture: &'a Texture, index: u8 },
    /// `drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:`.
    Draw {
        primitive_type: sys::WMTPrimitiveType,
        vertex_start: u64,
        vertex_count: u64,
        instance_count: u32,
        base_instance: u32,
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
    sys::wmtcmd_render_draw
);

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
        let pso = Object::adopt(unsafe { sys::MTLDevice_newRenderPipelineState(self.handle(), &info, &mut err) })
            .ok_or_else(|| error_text(err, "newRenderPipelineState"))?;
        Ok(RenderPipeline(pso))
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
        let _pool = Pool::new();
        // SAFETY: all-zero pass descriptor; fields set below.
        let mut pass: sys::WMTRenderPassInfo = unsafe { std::mem::zeroed() };
        pass.default_raster_sample_count = 1;
        pass.render_target_width = width;
        pass.render_target_height = height;
        for (i, a) in attachments.iter().enumerate() {
            pass.colors[i].texture = a.texture.handle();
            pass.colors[i].load_action = a.load_action;
            pass.colors[i].store_action = a.store_action;
            pass.colors[i].clear_color = sys::WMTClearColor {
                r: a.clear[0],
                g: a.clear[1],
                b: a.clear[2],
                a: a.clear[3],
            };
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
                Render::FragmentBytes { bytes, index } => Box::new(sys::wmtcmd_render_setbytes {
                    type_: sys::WMTRenderCommandType::WMTRenderCommandSetFragmentBytes,
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
