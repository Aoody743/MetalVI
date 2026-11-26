//! F7: whole graphics-pipeline state → Metal render pipeline / depth-stencil /
//! rasterization description mapping.
//!
//! One level up from F1's [`super::metal_map`]: F1 translates individual
//! enums and pixel formats; this module turns an entire D3D12 graphics
//! pipeline description (blend, rasterizer, depth-stencil, render-target
//! formats, vertex input layout) into plain structs the eventual Metal
//! backend will feed to `MTLRenderPipelineDescriptor`,
//! `MTLDepthStencilDescriptor` and the encoder's rasterizer state.
//!
//! Pure Rust: no FFI, no Metal calls, no `windows` crate dependency. The
//! D3D12/DXGI side values are re-declared here as plain `i32`/`u32`
//! constants so the file compiles and runs natively on macOS under the F7
//! harness (`metal-backend/flash/F7/harness`).
//!
//! # Sources
//!
//! * D3D12 / DXGI numeric values: windows 0.62.2 generated bindings
//!   (`Win32::Graphics::Direct3D12`, `Win32::Graphics::Dxgi::Common`) and
//!   Microsoft public header `d3d12.h`.
//! * Metal side: macOS SDK Metal headers (`MTLVertexDescriptor.h`,
//!   `MTLRenderCommandEncoder.h`, `MTLRenderPipeline.h`, `MTLDepthStencil.h`),
//!   cited per enum in the comments below.

#![allow(dead_code)]
#![allow(non_camel_case_types)] // variant names follow Apple's Metal constants verbatim

use super::metal_map as f1;

// ===========================================================================
// D3D12 / DXGI side constants (plain values; windows 0.62.2 bindings / d3d12.h)
// ===========================================================================

/// `D3D12_APPEND_ALIGNED_ELEMENT` — d3d12.h: the element is appended at the
/// end of the previous element in the same input slot.
pub const APPEND_ALIGNED_ELEMENT: u32 = 0xffffffff;

/// `D3D12_INPUT_CLASSIFICATION` — Direct3D12 bindings.
pub mod d3d12_input_class {
    pub const PER_VERTEX_DATA: i32 = 0;
    pub const PER_INSTANCE_DATA: i32 = 1;
}

/// `D3D12_DEPTH_WRITE_MASK` — Direct3D12 bindings.
pub mod d3d12_depth_write_mask {
    pub const ZERO: i32 = 0;
    pub const ALL: i32 = 1;
}

/// `D3D12_LOGIC_OP` — Direct3D12 bindings. Metal has no fixed-function blend
/// logic op; values are kept so the pipeline can record/translate them later.
pub mod d3d12_logic_op {
    pub const CLEAR: i32 = 0;
    pub const SET: i32 = 1;
    pub const COPY: i32 = 2;
    pub const COPY_INVERTED: i32 = 3;
    pub const NOOP: i32 = 4;
    pub const INVERT: i32 = 5;
    pub const AND: i32 = 6;
    pub const NAND: i32 = 7;
    pub const OR: i32 = 8;
    pub const NOR: i32 = 9;
    pub const XOR: i32 = 10;
    pub const EQUIV: i32 = 11;
    pub const AND_REVERSE: i32 = 12;
    pub const AND_INVERTED: i32 = 13;
    pub const OR_REVERSE: i32 = 14;
    pub const OR_INVERTED: i32 = 15;
}

// ===========================================================================
// Metal-side enums for the pipeline state. Numeric values are copied verbatim
// from the macOS SDK headers; the source header and enum name are noted.
// ===========================================================================

/// `MTLVertexFormat` — MTLVertexDescriptor.h (Metal.framework/Headers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum MtlVertexFormat {
    Invalid = 0,
    UChar2 = 1,
    UChar3 = 2,
    UChar4 = 3,
    Char2 = 4,
    Char3 = 5,
    Char4 = 6,
    UChar2Normalized = 7,
    UChar3Normalized = 8,
    UChar4Normalized = 9,
    Char2Normalized = 10,
    Char3Normalized = 11,
    Char4Normalized = 12,
    UShort2 = 13,
    UShort3 = 14,
    UShort4 = 15,
    Short2 = 16,
    Short3 = 17,
    Short4 = 18,
    UShort2Normalized = 19,
    UShort3Normalized = 20,
    UShort4Normalized = 21,
    Short2Normalized = 22,
    Short3Normalized = 23,
    Short4Normalized = 24,
    Half2 = 25,
    Half3 = 26,
    Half4 = 27,
    Float = 28,
    Float2 = 29,
    Float3 = 30,
    Float4 = 31,
    Int = 32,
    Int2 = 33,
    Int3 = 34,
    Int4 = 35,
    UInt = 36,
    UInt2 = 37,
    UInt3 = 38,
    UInt4 = 39,
    Int1010102Normalized = 40,
    UInt1010102Normalized = 41,
    UChar4Normalized_BGRA = 42,
    UChar = 45,
    Char = 46,
    UCharNormalized = 47,
    CharNormalized = 48,
    UShort = 49,
    Short = 50,
    UShortNormalized = 51,
    ShortNormalized = 52,
    Half = 53,
    /// macOS 14.0 / iOS 17.0+.
    FloatRG11B10 = 54,
    /// macOS 14.0 / iOS 17.0+.
    FloatRGB9E5 = 55,
}

/// `MTLVertexStepFunction` — MTLVertexDescriptor.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum MtlVertexStepFunction {
    Constant = 0,
    PerVertex = 1,
    PerInstance = 2,
}

/// `MTLDepthClipMode` — MTLRenderCommandEncoder.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum MtlDepthClipMode {
    Clip = 0,
    Clamp = 1,
}

// ===========================================================================
// Input structs: plain mirrors of the D3D12 pipeline description fields.
// Numeric fields reuse the D3D12/DXGI raw values (no windows types).
// ===========================================================================

/// Per-render-target blend state, mirroring `D3D12_RENDER_TARGET_BLEND_DESC`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtBlend {
    pub blend_enable: bool,
    pub logic_op_enable: bool,
    /// `D3D12_BLEND` — source RGB factor.
    pub src_blend: i32,
    /// `D3D12_BLEND` — destination RGB factor.
    pub dest_blend: i32,
    /// `D3D12_BLEND_OP` — RGB operation.
    pub blend_op: i32,
    /// `D3D12_BLEND` — source alpha factor.
    pub src_blend_alpha: i32,
    /// `D3D12_BLEND` — destination alpha factor.
    pub dest_blend_alpha: i32,
    /// `D3D12_BLEND_OP` — alpha operation.
    pub blend_op_alpha: i32,
    /// `D3D12_LOGIC_OP` (only meaningful when `logic_op_enable`).
    pub logic_op: i32,
    /// `D3D12_COLOR_WRITE_ENABLE` bit mask.
    pub render_target_write_mask: u32,
}

/// Mirror of `D3D12_BLEND_DESC`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlendDesc {
    pub alpha_to_coverage: bool,
    pub independent_blend_enable: bool,
    pub render_targets: [RtBlend; 8],
}

/// Mirror of `D3D12_RASTERIZER_DESC`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RasterizerDesc {
    /// `D3D12_FILL_MODE`.
    pub fill_mode: i32,
    /// `D3D12_CULL_MODE`.
    pub cull_mode: i32,
    pub front_counter_clockwise: bool,
    pub depth_bias: i32,
    pub depth_bias_clamp: f32,
    pub slope_scaled_depth_bias: f32,
    pub depth_clip_enable: bool,
    pub multisample_enable: bool,
    pub antialiased_line_enable: bool,
    pub forced_sample_count: u32,
    pub conservative_raster_enable: bool,
}

/// Mirror of `D3D12_DEPTH_STENCILOP_DESC` (one stencil face).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StencilFaceDesc {
    /// `D3D12_STENCIL_OP` — fails stencil test.
    pub fail_op: i32,
    /// `D3D12_STENCIL_OP` — passes stencil test, fails depth test.
    pub depth_fail_op: i32,
    /// `D3D12_STENCIL_OP` — passes both tests.
    pub pass_op: i32,
    /// `D3D12_COMPARISON_FUNC`.
    pub func: i32,
}

/// Mirror of `D3D12_DEPTH_STENCIL_DESC`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthStencilDesc {
    pub depth_enable: bool,
    /// True = `D3D12_DEPTH_WRITE_MASK_ALL`, false = `ZERO`.
    pub depth_write_mask: bool,
    /// `D3D12_COMPARISON_FUNC`.
    pub depth_func: i32,
    pub stencil_enable: bool,
    pub stencil_read_mask: u8,
    pub stencil_write_mask: u8,
    pub front: StencilFaceDesc,
    pub back: StencilFaceDesc,
}

/// Mirror of `D3D12_INPUT_ELEMENT_DESC`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputElement {
    /// Semantic name as sent by the title (e.g. "POSITION", "TEXCOORD").
    pub semantic_name: String,
    pub semantic_index: u32,
    /// `DXGI_FORMAT`.
    pub format: i32,
    pub input_slot: u32,
    /// `AlignedByteOffset`; [`APPEND_ALIGNED_ELEMENT`] (0xffffffff) lets the
    /// descriptor compute the offset per slot.
    pub aligned_byte_offset: u32,
    /// `D3D12_INPUT_CLASSIFICATION`.
    pub slot_class: i32,
    /// `InstanceDataStepRate` (per-instance slots only).
    pub instance_step_rate: u32,
}

/// Full graphics-pipeline state handed to the mapper.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineStateInput {
    pub blend: BlendDesc,
    pub rasterizer: RasterizerDesc,
    pub depth_stencil: DepthStencilDesc,
    /// `DXGI_FORMAT` for render targets 0..7; unused slots carry 0.
    pub rtv_formats: [i32; 8],
    pub num_render_targets: u32,
    /// `DXGI_FORMAT` of the depth-stencil view (0 when none).
    pub dsv_format: i32,
    pub sample_count: u32,
    /// `D3D12_PRIMITIVE_TOPOLOGY_TYPE` (PSO level class).
    pub primitive_topology_type: i32,
    pub elements: Vec<InputElement>,
}

// ===========================================================================
// Output structs: plain Metal-side descriptions (no Metal objects).
// ===========================================================================

/// One render-target color attachment in the Metal pipeline state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorAttachmentMap {
    pub pixel_format: Option<f1::MtlPixelFormat>,
    pub blending_enabled: bool,
    pub source_rgb_factor: Option<f1::MtlBlendFactor>,
    pub destination_rgb_factor: Option<f1::MtlBlendFactor>,
    pub rgb_operation: Option<f1::MtlBlendOperation>,
    pub source_alpha_factor: Option<f1::MtlBlendFactor>,
    pub destination_alpha_factor: Option<f1::MtlBlendFactor>,
    pub alpha_operation: Option<f1::MtlBlendOperation>,
    pub write_mask: f1::MtlColorWriteMask,
}

/// One stencil face in the Metal depth-stencil state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StencilMap {
    pub compare: f1::MtlCompareFunction,
    pub stencil_fail: f1::MtlStencilOperation,
    pub depth_fail: f1::MtlStencilOperation,
    pub depth_pass: f1::MtlStencilOperation,
    pub read_mask: u8,
    pub write_mask: u8,
}

/// Metal depth-stencil pipeline state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthStencilMap {
    pub depth_compare: f1::MtlCompareFunction,
    pub depth_write: bool,
    pub front: StencilMap,
    pub back: StencilMap,
}

/// Depth-bias triple, mirroring the rasterizer's `DepthBias` /
/// `SlopeScaledDepthBias` / `DepthBiasClamp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthBiasMap {
    pub depth_bias: i32,
    pub slope_scale: f32,
    pub clamp: f32,
}

/// Metal rasterizer state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RasterMap {
    pub cull: Option<f1::MtlCullMode>,
    pub winding: f1::MtlWinding,
    pub fill: Option<f1::MtlTriangleFillMode>,
    pub depth_bias: DepthBiasMap,
    pub depth_clip_mode: MtlDepthClipMode,
}

/// One vertex attribute in the Metal vertex descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexAttributeMap {
    /// Attribute index (shader location). F7 assigns layout order; the real
    /// semantic → location mapping comes from shader reflection at PSO bind
    /// time (see REPORT open questions).
    pub index: u32,
    pub format: Option<MtlVertexFormat>,
    pub offset: u32,
    pub buffer_index: u32,
}

/// One vertex buffer layout in the Metal vertex descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexBufferLayoutMap {
    pub stride: u32,
    pub step_function: MtlVertexStepFunction,
    pub step_rate: u32,
}

/// Metal vertex descriptor: attributes + per-buffer layouts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexDescriptorMap {
    pub attributes: Vec<VertexAttributeMap>,
    /// Indexed by input slot; `None` when the slot has no element.
    pub layouts: Vec<Option<VertexBufferLayoutMap>>,
}

/// Translated render pipeline state.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderPipelineMap {
    pub color: [Option<ColorAttachmentMap>; 8],
    pub depth_format: Option<f1::MtlPixelFormat>,
    pub stencil_format: Option<f1::MtlPixelFormat>,
    pub raster_sample_count: u32,
    pub alpha_to_coverage: bool,
    pub input_primitive_topology: Option<f1::MtlPrimitiveTopologyClass>,
    pub vertex: VertexDescriptorMap,
    pub depth_stencil: DepthStencilMap,
    pub raster: RasterMap,
    /// Every lossy / unsupported / snapped case collected here; nothing is
    /// dropped silently.
    pub notes: Vec<&'static str>,
}

// ===========================================================================
// DXGI_FORMAT → MTLVertexFormat table.
//
// Values from the SDK header MTLVertexDescriptor.h (enum MTLVertexFormat).
// `size` is the element size in bytes, `align` the append-alignment the
// D3D12 runtime applies to APPEND_ALIGNED_ELEMENT offsets (1 / 2 / 4 / 8 by
// channel width). Formats with no Metal equivalent map to `None` and carry a
// reason in `note`.
// ===========================================================================

/// (DXGI_FORMAT, Metal vertex format, element size bytes, append alignment, note).
type VertexFormatRow = (i32, Option<MtlVertexFormat>, u32, u32, &'static str);

const VERTEX_FORMAT_TABLE: &[VertexFormatRow] = &[
    // --- 128 bit, four channels (32×4) -----------------------------------
    (f1::dxgi_format::R32G32B32A32_FLOAT, Some(MtlVertexFormat::Float4), 16, 16, ""),
    (f1::dxgi_format::R32G32B32A32_UINT, Some(MtlVertexFormat::UInt4), 16, 16, ""),
    (f1::dxgi_format::R32G32B32A32_SINT, Some(MtlVertexFormat::Int4), 16, 16, ""),
    // --- 96 bit, three channels (32×3; vertex Float3 exists even though the
    //     pixel format side has no 96-bit RGB32) ---------------------------
    (f1::dxgi_format::R32G32B32_FLOAT, Some(MtlVertexFormat::Float3), 12, 4, ""),
    (f1::dxgi_format::R32G32B32_UINT, Some(MtlVertexFormat::UInt3), 12, 4, ""),
    (f1::dxgi_format::R32G32B32_SINT, Some(MtlVertexFormat::Int3), 12, 4, ""),
    // --- 64 bit, four channels (16×4) -------------------------------------
    (f1::dxgi_format::R16G16B16A16_FLOAT, Some(MtlVertexFormat::Half4), 8, 8, ""),
    (f1::dxgi_format::R16G16B16A16_UNORM, Some(MtlVertexFormat::UShort4Normalized), 8, 8, ""),
    (f1::dxgi_format::R16G16B16A16_UINT, Some(MtlVertexFormat::UShort4), 8, 8, ""),
    (f1::dxgi_format::R16G16B16A16_SNORM, Some(MtlVertexFormat::Short4Normalized), 8, 8, ""),
    (f1::dxgi_format::R16G16B16A16_SINT, Some(MtlVertexFormat::Short4), 8, 8, ""),
    // --- 64 bit, two channels (32×2) --------------------------------------
    (f1::dxgi_format::R32G32_FLOAT, Some(MtlVertexFormat::Float2), 8, 8, ""),
    (f1::dxgi_format::R32G32_UINT, Some(MtlVertexFormat::UInt2), 8, 8, ""),
    (f1::dxgi_format::R32G32_SINT, Some(MtlVertexFormat::Int2), 8, 8, ""),
    // --- Packed 32 bit -----------------------------------------------------
    (f1::dxgi_format::R10G10B10A2_UNORM, Some(MtlVertexFormat::UInt1010102Normalized), 4, 4, ""),
    (f1::dxgi_format::R10G10B10A2_UINT, None, 4, 4,
     "Metal has only normalized 1010102 vertex formats; the UINT twin has no equivalent"),
    (f1::dxgi_format::R11G11B10_FLOAT, Some(MtlVertexFormat::FloatRG11B10), 4, 4,
     "FloatRG11B10 requires macOS 14.0 / iOS 17.0; verify runtime availability"),
    (f1::dxgi_format::R9G9B9E5_SHAREDEXP, Some(MtlVertexFormat::FloatRGB9E5), 4, 4,
     "FloatRGB9E5 requires macOS 14.0 / iOS 17.0; verify runtime availability"),
    // --- 32 bit, four channels (8×4) --------------------------------------
    (f1::dxgi_format::R8G8B8A8_UNORM, Some(MtlVertexFormat::UChar4Normalized), 4, 4, ""),
    (f1::dxgi_format::R8G8B8A8_UNORM_SRGB, None, 4, 4,
     "sRGB vertex data has no Metal vertex input format"),
    (f1::dxgi_format::R8G8B8A8_UINT, Some(MtlVertexFormat::UChar4), 4, 4, ""),
    (f1::dxgi_format::R8G8B8A8_SNORM, Some(MtlVertexFormat::Char4Normalized), 4, 4, ""),
    (f1::dxgi_format::R8G8B8A8_SINT, Some(MtlVertexFormat::Char4), 4, 4, ""),
    (f1::dxgi_format::B8G8R8A8_UNORM, Some(MtlVertexFormat::UChar4Normalized_BGRA), 4, 4, ""),
    (f1::dxgi_format::B8G8R8X8_UNORM, Some(MtlVertexFormat::UChar4Normalized_BGRA), 4, 4,
     "X8 channel discarded; mapped to BGRA normalized uchar4"),
    // --- 32 bit, two channels (16×2) --------------------------------------
    (f1::dxgi_format::R16G16_FLOAT, Some(MtlVertexFormat::Half2), 4, 4, ""),
    (f1::dxgi_format::R16G16_UNORM, Some(MtlVertexFormat::UShort2Normalized), 4, 4, ""),
    (f1::dxgi_format::R16G16_UINT, Some(MtlVertexFormat::UShort2), 4, 4, ""),
    (f1::dxgi_format::R16G16_SNORM, Some(MtlVertexFormat::Short2Normalized), 4, 4, ""),
    (f1::dxgi_format::R16G16_SINT, Some(MtlVertexFormat::Short2), 4, 4, ""),
    // --- 32 bit, single channel -------------------------------------------
    (f1::dxgi_format::R32_FLOAT, Some(MtlVertexFormat::Float), 4, 4, ""),
    (f1::dxgi_format::R32_UINT, Some(MtlVertexFormat::UInt), 4, 4, ""),
    (f1::dxgi_format::R32_SINT, Some(MtlVertexFormat::Int), 4, 4, ""),
    // --- 16 bit, two channels (8×2) ---------------------------------------
    (f1::dxgi_format::R8G8_UNORM, Some(MtlVertexFormat::UChar2Normalized), 2, 2, ""),
    (f1::dxgi_format::R8G8_UINT, Some(MtlVertexFormat::UChar2), 2, 2, ""),
    (f1::dxgi_format::R8G8_SNORM, Some(MtlVertexFormat::Char2Normalized), 2, 2, ""),
    (f1::dxgi_format::R8G8_SINT, Some(MtlVertexFormat::Char2), 2, 2, ""),
    // --- 16 bit, single channel -------------------------------------------
    (f1::dxgi_format::R16_FLOAT, Some(MtlVertexFormat::Half), 2, 2, ""),
    (f1::dxgi_format::R16_UNORM, Some(MtlVertexFormat::UShortNormalized), 2, 2, ""),
    (f1::dxgi_format::R16_UINT, Some(MtlVertexFormat::UShort), 2, 2, ""),
    (f1::dxgi_format::R16_SNORM, Some(MtlVertexFormat::ShortNormalized), 2, 2, ""),
    (f1::dxgi_format::R16_SINT, Some(MtlVertexFormat::Short), 2, 2, ""),
    // --- 8 bit, single channel --------------------------------------------
    (f1::dxgi_format::R8_UNORM, Some(MtlVertexFormat::UCharNormalized), 1, 1, ""),
    (f1::dxgi_format::R8_UINT, Some(MtlVertexFormat::UChar), 1, 1, ""),
    (f1::dxgi_format::R8_SNORM, Some(MtlVertexFormat::CharNormalized), 1, 1, ""),
    (f1::dxgi_format::R8_SINT, Some(MtlVertexFormat::Char), 1, 1, ""),
];

fn vertex_format_row(dxgi: i32) -> Option<&'static VertexFormatRow> {
    VERTEX_FORMAT_TABLE.iter().find(|r| r.0 == dxgi)
}

/// Translate a `DXGI_FORMAT` into an `MTLVertexFormat`; `None` when Metal has
/// no equivalent vertex format.
pub fn map_vertex_format(dxgi: i32) -> Option<MtlVertexFormat> {
    vertex_format_row(dxgi).and_then(|r| r.1)
}

// ===========================================================================
// Depth-stencil format split: one D3D12 DSV format → separate Metal depth and
// stencil pixel formats.
// ===========================================================================

/// Split a DSV `DXGI_FORMAT` into (depth format, stencil format, note).
///
/// * `D32_FLOAT_S8X24_UINT` → `Depth32Float_Stencil8` on both sides.
/// * `D24_UNORM_S8_UINT` → `Depth32Float_Stencil8` on both sides: Apple GPUs
///   have no 24-bit depth format, so the 24-bit depth is upconverted.
/// * `D16_UNORM` → `Depth16Unorm`, no stencil.
/// * `D32_FLOAT` → `Depth32Float`, no stencil.
/// * Stencil-only / depth-only views are split with the discarded channel
///   recorded in the note.
pub fn split_dsv_format(
    dxgi: i32,
) -> (Option<f1::MtlPixelFormat>, Option<f1::MtlPixelFormat>, Option<&'static str>) {
    use f1::dxgi_format as dx;
    match dxgi {
        dx::D32_FLOAT_S8X24_UINT | dx::R32G8X24_TYPELESS => (
            Some(f1::MtlPixelFormat::Depth32Float_Stencil8),
            Some(f1::MtlPixelFormat::Depth32Float_Stencil8),
            None,
        ),
        dx::D24_UNORM_S8_UINT | dx::R24G8_TYPELESS => (
            Some(f1::MtlPixelFormat::Depth32Float_Stencil8),
            Some(f1::MtlPixelFormat::Depth32Float_Stencil8),
            Some("24-bit depth+stencil upconverted to Depth32Float_Stencil8; Apple GPUs have no 24-bit depth format"),
        ),
        dx::D32_FLOAT => (Some(f1::MtlPixelFormat::Depth32Float), None, None),
        dx::D16_UNORM => (Some(f1::MtlPixelFormat::Depth16Unorm), None, None),
        dx::X24_TYPELESS_G8_UINT => (None, Some(f1::MtlPixelFormat::X24_Stencil8), None),
        dx::X32_TYPELESS_G8X24_UINT => (
            None,
            None,
            Some("stencil-only view of a 32-bit depth+stencil texture; Metal has no matching stencil-only format"),
        ),
        dx::R24_UNORM_X8_TYPELESS => (
            Some(f1::MtlPixelFormat::Depth32Float_Stencil8),
            None,
            Some("depth-only view of a 24/8 texture; stencil channel discarded; 24-bit depth upconverted"),
        ),
        dx::R32_FLOAT_X8X24_TYPELESS => (
            Some(f1::MtlPixelFormat::Depth32Float),
            None,
            Some("depth-only view of a 32/8 texture; stencil channel discarded"),
        ),
        0 => (None, None, None),
        _ => (None, None, Some("unrecognized depth-stencil DXGI format")),
    }
}

// ===========================================================================
// Depth-stencil state
// ===========================================================================

fn map_stencil_face(
    face: &StencilFaceDesc,
    read_mask: u8,
    write_mask: u8,
    notes: &mut Vec<&'static str>,
) -> StencilMap {
    let compare = f1::map_compare_function(face.func).unwrap_or_else(|| {
        notes.push("stencil compare function has no Metal equivalent; fell back to Always");
        f1::MtlCompareFunction::Always
    });
    let fail = f1::map_stencil_op(face.fail_op).unwrap_or_else(|| {
        notes.push("stencil fail operation has no Metal equivalent; fell back to Keep");
        f1::MtlStencilOperation::Keep
    });
    let depth_fail = f1::map_stencil_op(face.depth_fail_op).unwrap_or_else(|| {
        notes.push("stencil depth-fail operation has no Metal equivalent; fell back to Keep");
        f1::MtlStencilOperation::Keep
    });
    let pass = f1::map_stencil_op(face.pass_op).unwrap_or_else(|| {
        notes.push("stencil depth-pass operation has no Metal equivalent; fell back to Keep");
        f1::MtlStencilOperation::Keep
    });
    StencilMap {
        compare,
        stencil_fail: fail,
        depth_fail,
        depth_pass: pass,
        read_mask,
        write_mask,
    }
}

/// Translate a `D3D12_DEPTH_STENCIL_DESC` into the Metal depth-stencil state.
///
/// When depth is off, the depth compare falls back to `Always` and depth
/// writes are disabled. When stencil is off, both faces become
/// `Always`/`Keep`/`Keep`/`Keep` with masks `0xff`.
pub fn map_depth_stencil(desc: &DepthStencilDesc, notes: &mut Vec<&'static str>) -> DepthStencilMap {
    let depth_compare = if !desc.depth_enable {
        f1::MtlCompareFunction::Always
    } else {
        f1::map_compare_function(desc.depth_func).unwrap_or_else(|| {
            notes.push("depth compare function has no Metal equivalent; fell back to Always");
            f1::MtlCompareFunction::Always
        })
    };
    let depth_write = desc.depth_enable && desc.depth_write_mask;

    let stencil_off = StencilMap {
        compare: f1::MtlCompareFunction::Always,
        stencil_fail: f1::MtlStencilOperation::Keep,
        depth_fail: f1::MtlStencilOperation::Keep,
        depth_pass: f1::MtlStencilOperation::Keep,
        read_mask: 0xff,
        write_mask: 0xff,
    };
    let (front, back) = if !desc.stencil_enable {
        (stencil_off, stencil_off)
    } else {
        (
            map_stencil_face(&desc.front, desc.stencil_read_mask, desc.stencil_write_mask, notes),
            map_stencil_face(&desc.back, desc.stencil_read_mask, desc.stencil_write_mask, notes),
        )
    };

    DepthStencilMap { depth_compare, depth_write, front, back }
}

// ===========================================================================
// Rasterizer state
// ===========================================================================

/// Translate a `D3D12_RASTERIZER_DESC` into the Metal rasterizer state.
/// Unsupported rasterizer features (conservative rasterization, forced sample
/// count, antialiased line enable) are recorded in `notes` rather than
/// dropped silently.
pub fn map_rasterizer(desc: &RasterizerDesc, notes: &mut Vec<&'static str>) -> RasterMap {
    if desc.conservative_raster_enable {
        notes.push("conservative rasterization has no Metal equivalent; ignored");
    }
    if desc.forced_sample_count != 0 {
        notes.push("ForcedSampleCount is not expressible in a Metal render pipeline; ignored");
    }
    if desc.antialiased_line_enable {
        notes.push("antialiased line enable has no pipeline-level Metal flag; ignored");
    }

    let cull = f1::map_cull_mode(desc.cull_mode).or_else(|| {
        notes.push("D3D12 cull mode has no Metal equivalent; culling off");
        None
    });
    let fill = f1::map_fill_mode(desc.fill_mode).or_else(|| {
        notes.push("D3D12 fill mode has no Metal equivalent; filled");
        None
    });

    let depth_clip_mode = if desc.depth_clip_enable {
        MtlDepthClipMode::Clip
    } else {
        MtlDepthClipMode::Clamp
    };

    RasterMap {
        cull,
        winding: f1::map_winding(desc.front_counter_clockwise),
        fill,
        depth_bias: DepthBiasMap {
            depth_bias: desc.depth_bias,
            slope_scale: desc.slope_scaled_depth_bias,
            clamp: desc.depth_bias_clamp,
        },
        depth_clip_mode,
    }
}

// ===========================================================================
// Vertex input layout
// ===========================================================================

fn align_up(value: u32, align: u32) -> u32 {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

/// Translate the input element list into a Metal vertex descriptor.
///
/// `APPEND_ALIGNED_ELEMENT` offsets are accumulated **per input slot**: each
/// slot keeps its own running offset; the first element in a slot starts at 0.
/// The slot stride is the end of the slot's last element, rounded up to 4
/// bytes (D3D12 input-layout stride rule).
pub fn map_vertex_descriptor(
    elements: &[InputElement],
    notes: &mut Vec<&'static str>,
) -> VertexDescriptorMap {
    let mut attributes: Vec<VertexAttributeMap> = Vec::new();
    // Per-slot running values, indexed by input slot.
    let mut next_offset: Vec<u32> = Vec::new();
    let mut slot_end: Vec<u32> = Vec::new();
    let mut slot_class: Vec<i32> = Vec::new();
    let mut slot_step_rate: Vec<u32> = Vec::new();

    for (attr_index, el) in elements.iter().enumerate() {
        let row = vertex_format_row(el.format);
        let (format, size, align) = match row {
            Some(r) => (r.1, r.2, r.3),
            None => (None, 4u32, 4u32),
        };
        if format.is_none() {
            notes.push(
                "vertex DXGI_FORMAT has no Metal vertex format equivalent; attribute format is None and offset/stride use a conservative 4-byte estimate",
            );
            if let Some(r) = row {
                if !r.4.is_empty() {
                    notes.push(r.4);
                }
            }
        }

        let slot = el.input_slot as usize;
        if next_offset.len() <= slot {
            next_offset.resize(slot + 1, 0);
            slot_end.resize(slot + 1, 0);
            slot_class.resize(slot + 1, d3d12_input_class::PER_VERTEX_DATA);
            slot_step_rate.resize(slot + 1, 1);
        }

        let offset = if el.aligned_byte_offset == APPEND_ALIGNED_ELEMENT {
            align_up(next_offset[slot], align)
        } else {
            el.aligned_byte_offset
        };
        next_offset[slot] = offset + size;
        slot_end[slot] = slot_end[slot].max(offset + size);
        slot_class[slot] = el.slot_class;
        slot_step_rate[slot] = el.instance_step_rate;

        attributes.push(VertexAttributeMap {
            index: attr_index as u32,
            format,
            offset,
            buffer_index: el.input_slot,
        });
    }

    let layouts: Vec<Option<VertexBufferLayoutMap>> = (0..slot_end.len())
        .map(|slot| {
            if slot_end[slot] == 0 {
                return None;
            }
            let stride = align_up(slot_end[slot], 4);
            let (step_function, step_rate) = match slot_class[slot] {
                d3d12_input_class::PER_INSTANCE_DATA => {
                    // D3D12 step rate 0 on a per-instance slot is degenerate;
                    // Metal advances once per instance by default.
                    let rate = if slot_step_rate[slot] == 0 {
                        notes.push("per-instance slot InstanceDataStepRate=0; treated as step rate 1");
                        1
                    } else {
                        slot_step_rate[slot]
                    };
                    (MtlVertexStepFunction::PerInstance, rate)
                }
                _ => (MtlVertexStepFunction::PerVertex, 1),
            };
            Some(VertexBufferLayoutMap { stride, step_function, step_rate })
        })
        .collect();

    VertexDescriptorMap { attributes, layouts }
}

// ===========================================================================
// Color attachment / whole pipeline
// ===========================================================================

pub fn map_color_attachment(
    rt: &RtBlend,
    format: i32,
    notes: &mut Vec<&'static str>,
) -> ColorAttachmentMap {
    let pixel_format = f1::map_pixel_format(format);
    if pixel_format.is_none() {
        notes.push("render target DXGI_FORMAT has no Metal pixel format equivalent");
    }
    if rt.logic_op_enable {
        notes.push("fixed-function blend logic op is not supported by Metal; ignored");
    }

    let mut src_rgb = None;
    let mut dst_rgb = None;
    let mut op_rgb = None;
    let mut src_a = None;
    let mut dst_a = None;
    let mut op_a = None;

    if rt.blend_enable {
        src_rgb = f1::map_blend_factor(rt.src_blend);
        dst_rgb = f1::map_blend_factor(rt.dest_blend);
        op_rgb = f1::map_blend_op(rt.blend_op);
        src_a = f1::map_blend_factor(rt.src_blend_alpha);
        dst_a = f1::map_blend_factor(rt.dest_blend_alpha);
        op_a = f1::map_blend_op(rt.blend_op_alpha);
        if src_rgb.is_none() || dst_rgb.is_none() || op_rgb.is_none()
            || src_a.is_none() || dst_a.is_none() || op_a.is_none()
        {
            notes.push("blend factor or operation has no Metal equivalent; the unresolved blend field is None");
        }
    }

    ColorAttachmentMap {
        pixel_format,
        blending_enabled: rt.blend_enable,
        source_rgb_factor: src_rgb,
        destination_rgb_factor: dst_rgb,
        rgb_operation: op_rgb,
        source_alpha_factor: src_a,
        destination_alpha_factor: dst_a,
        alpha_operation: op_a,
        write_mask: f1::map_color_write_mask(rt.render_target_write_mask),
    }
}

/// Metal cannot blend with a signed or unsigned integer color attachment.
pub(crate) fn check_integer_blending(color: &[Option<ColorAttachmentMap>; 8]) -> Result<(), String> {
    use f1::MtlPixelFormat::*;
    for (slot, attachment) in color.iter().enumerate() {
        let Some(attachment) = attachment else { continue };
        let Some(format) = attachment.pixel_format else { continue };
        if attachment.blending_enabled
            && matches!(
                format,
                R8Uint | R8Sint | RG8Uint | RG8Sint | RGBA8Uint | RGBA8Sint
                    | R16Uint | R16Sint | RG16Uint | RG16Sint | RGBA16Uint | RGBA16Sint
                    | R32Uint | R32Sint | RG32Uint | RG32Sint | RGBA32Uint | RGBA32Sint
                    | RGB10A2Uint
            )
        {
            return Err(format!("render target {slot} uses integer Metal format {format:?} with blending enabled"));
        }
    }
    Ok(())
}

/// Translate the whole graphics-pipeline state into the Metal render pipeline
/// description. All lossy cases land in [`RenderPipelineMap::notes`].
pub fn map_pipeline_state(input: &PipelineStateInput) -> RenderPipelineMap {
    let mut notes: Vec<&'static str> = Vec::new();

    // Independent blend: when off, RT0's state applies to every render
    // target; flag any divergence instead of silently picking RT0.
    if !input.blend.independent_blend_enable {
        let rt0 = input.blend.render_targets[0];
        if input.blend.render_targets[1..].iter().any(|rt| *rt != rt0) {
            notes.push(
                "IndependentBlendEnable=false but render-target blend states differ; RT0 state copied to all render targets",
            );
        }
    }

    let mut color: [Option<ColorAttachmentMap>; 8] = [const { None }; 8];
    for (i, slot) in color.iter_mut().enumerate() {
        let rt = if input.blend.independent_blend_enable {
            &input.blend.render_targets[i]
        } else {
            &input.blend.render_targets[0]
        };
        if (i as u32) < input.num_render_targets {
            *slot = Some(map_color_attachment(rt, input.rtv_formats[i], &mut notes));
        }
    }

    let (depth_format, stencil_format, dsv_note) = split_dsv_format(input.dsv_format);
    if let Some(note) = dsv_note {
        notes.push(note);
    }

    let topology = f1::map_topology_class(input.primitive_topology_type);
    if topology.is_none() {
        notes.push("PSO primitive topology type has no Metal topology class equivalent");
    }

    let vertex = map_vertex_descriptor(&input.elements, &mut notes);
    let depth_stencil = map_depth_stencil(&input.depth_stencil, &mut notes);
    let raster = map_rasterizer(&input.rasterizer, &mut notes);

    RenderPipelineMap {
        color,
        depth_format,
        stencil_format,
        raster_sample_count: input.sample_count,
        alpha_to_coverage: input.blend.alpha_to_coverage,
        input_primitive_topology: topology,
        vertex,
        depth_stencil,
        raster,
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_blend_reports_the_offending_target() {
        use f1::{d3d12_blend, d3d12_blend_op, d3d12_color_write, dxgi_format};
        let mut rt = RtBlend {
            blend_enable: true,
            logic_op_enable: false,
            src_blend: d3d12_blend::ONE,
            dest_blend: d3d12_blend::ZERO,
            blend_op: d3d12_blend_op::ADD,
            src_blend_alpha: d3d12_blend::ONE,
            dest_blend_alpha: d3d12_blend::ZERO,
            blend_op_alpha: d3d12_blend_op::ADD,
            logic_op: d3d12_logic_op::NOOP,
            render_target_write_mask: d3d12_color_write::ALL,
        };
        let mut notes = Vec::new();
        let mut colors = [None; 8];
        colors[0] = Some(map_color_attachment(&rt, dxgi_format::R8G8B8A8_UNORM, &mut notes));
        colors[2] = Some(map_color_attachment(&rt, dxgi_format::R8_UINT, &mut notes));
        assert_eq!(
            check_integer_blending(&colors).unwrap_err(),
            "render target 2 uses integer Metal format R8Uint with blending enabled"
        );

        colors[2] = Some(map_color_attachment(&rt, dxgi_format::R32G32_SINT, &mut notes));
        assert_eq!(
            check_integer_blending(&colors).unwrap_err(),
            "render target 2 uses integer Metal format RG32Sint with blending enabled"
        );

        rt.blend_enable = false;
        colors[2] = Some(map_color_attachment(&rt, dxgi_format::R8_UINT, &mut notes));
        assert_eq!(check_integer_blending(&colors), Ok(()));
    }
}
