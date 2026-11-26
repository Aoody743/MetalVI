//! F1: pure D3D12/DXGI → Metal enum and format translation tables.
//!
//! This module contains **data and translation functions only**: no FFI, no
//! Metal calls, no `windows` crate dependency. The D3D12/DXGI side values are
//! re-declared here as plain constants so the file compiles and runs natively
//! on macOS under the F1 harness (`metal-backend/flash/F1/harness`).
//!
//! # Sources
//!
//! * D3D12 / DXGI numeric values: `windows` crate 0.62.2 generated bindings
//!   (`~/.cargo/registry/src/index.crates.io-*/windows-0.62.2/src/Windows/Win32/
//!   Graphics/Dxgi/Common/mod.rs` and `Direct3D12/mod.rs`, `Direct3D/mod.rs`);
//!   constants the bindings omit (the plain `D3D_PRIMITIVE_TOPOLOGY_*`
//!   list/strip values) are taken from Microsoft's public `d3dcommon.h`.
//! * Metal numeric values: macOS SDK Metal headers
//!   (`xcrun --show-sdk-path` → `Metal.framework/Headers/`), cited per enum in
//!   the comments below.
//!
//! The D3D12 side uses raw `i32`/`u32` values rather than the Windows newtype
//! structs, so this file has no Windows dependencies.

#![allow(dead_code)]
#![allow(non_camel_case_types)] // variant names follow Apple's Metal constants verbatim

// ===========================================================================
// Metal-side enums. Numeric values are copied verbatim from the macOS SDK
// headers; the source header and enum name are noted on each group.
// ===========================================================================

/// `MTLPixelFormat` — MTLPixelFormat.h (Metal.framework/Headers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum MtlPixelFormat {
    Invalid = 0,
    // Normal 8 bit
    A8Unorm = 1,
    R8Unorm = 10,
    R8Unorm_sRGB = 11,
    R8Snorm = 12,
    R8Uint = 13,
    R8Sint = 14,
    // Normal 16 bit
    R16Unorm = 20,
    R16Snorm = 22,
    R16Uint = 23,
    R16Sint = 24,
    R16Float = 25,
    RG8Unorm = 30,
    RG8Unorm_sRGB = 31,
    RG8Snorm = 32,
    RG8Uint = 33,
    RG8Sint = 34,
    // Packed 16 bit
    B5G6R5Unorm = 40,
    A1BGR5Unorm = 41,
    ABGR4Unorm = 42,
    BGR5A1Unorm = 43,
    // Normal 32 bit
    R32Uint = 53,
    R32Sint = 54,
    R32Float = 55,
    RG16Unorm = 60,
    RG16Snorm = 62,
    RG16Uint = 63,
    RG16Sint = 64,
    RG16Float = 65,
    RGBA8Unorm = 70,
    RGBA8Unorm_sRGB = 71,
    RGBA8Snorm = 72,
    RGBA8Uint = 73,
    RGBA8Sint = 74,
    BGRA8Unorm = 80,
    BGRA8Unorm_sRGB = 81,
    // Packed 32 bit
    RGB10A2Unorm = 90,
    RGB10A2Uint = 91,
    RG11B10Float = 92,
    RGB9E5Float = 93,
    BGR10A2Unorm = 94,
    // Normal 64 bit
    RG32Uint = 103,
    RG32Sint = 104,
    RG32Float = 105,
    RGBA16Unorm = 110,
    RGBA16Snorm = 112,
    RGBA16Uint = 113,
    RGBA16Sint = 114,
    RGBA16Float = 115,
    // Normal 128 bit
    RGBA32Uint = 123,
    RGBA32Sint = 124,
    RGBA32Float = 125,
    // Compressed: S3TC/DXT
    BC1_RGBA = 130,
    BC1_RGBA_sRGB = 131,
    BC2_RGBA = 132,
    BC2_RGBA_sRGB = 133,
    BC3_RGBA = 134,
    BC3_RGBA_sRGB = 135,
    // Compressed: RGTC
    BC4_RUnorm = 140,
    BC4_RSnorm = 141,
    BC5_RGUnorm = 142,
    BC5_RGSnorm = 143,
    // Compressed: BPTC
    BC6H_RGBFloat = 150,
    BC6H_RGBUfloat = 151,
    BC7_RGBAUnorm = 152,
    BC7_RGBAUnorm_sRGB = 153,
    // 422
    GBGR422 = 240,
    BGRG422 = 241,
    // Depth / stencil
    Depth16Unorm = 250,
    Depth32Float = 252,
    Stencil8 = 253,
    Depth24Unorm_Stencil8 = 255,
    Depth32Float_Stencil8 = 260,
    X32_Stencil8 = 261,
    X24_Stencil8 = 262,
}

/// `MTLBlendFactor` — MTLRenderPipeline.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum MtlBlendFactor {
    Zero = 0,
    One = 1,
    SourceColor = 2,
    OneMinusSourceColor = 3,
    SourceAlpha = 4,
    OneMinusSourceAlpha = 5,
    DestinationColor = 6,
    OneMinusDestinationColor = 7,
    DestinationAlpha = 8,
    OneMinusDestinationAlpha = 9,
    SourceAlphaSaturated = 10,
    BlendColor = 11,
    OneMinusBlendColor = 12,
    BlendAlpha = 13,
    OneMinusBlendAlpha = 14,
    Source1Color = 15,
    OneMinusSource1Color = 16,
    Source1Alpha = 17,
    OneMinusSource1Alpha = 18,
}

/// `MTLBlendOperation` — MTLRenderPipeline.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum MtlBlendOperation {
    Add = 0,
    Subtract = 1,
    ReverseSubtract = 2,
    Min = 3,
    Max = 4,
}

/// `MTLColorWriteMask` — MTLRenderPipeline.h (NS_OPTIONS).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct MtlColorWriteMask(pub u64);

impl MtlColorWriteMask {
    pub const NONE: Self = Self(0);
    pub const RED: Self = Self(0x1 << 3);
    pub const GREEN: Self = Self(0x1 << 2);
    pub const BLUE: Self = Self(0x1 << 1);
    pub const ALPHA: Self = Self(0x1 << 0);
    pub const ALL: Self = Self(0xf);
}

/// `MTLCompareFunction` — MTLDepthStencil.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum MtlCompareFunction {
    Never = 0,
    Less = 1,
    Equal = 2,
    LessEqual = 3,
    Greater = 4,
    NotEqual = 5,
    GreaterEqual = 6,
    Always = 7,
}

/// `MTLStencilOperation` — MTLDepthStencil.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum MtlStencilOperation {
    Keep = 0,
    Zero = 1,
    Replace = 2,
    IncrementClamp = 3,
    DecrementClamp = 4,
    Invert = 5,
    IncrementWrap = 6,
    DecrementWrap = 7,
}

/// `MTLCullMode` — MTLRenderCommandEncoder.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum MtlCullMode {
    None = 0,
    Front = 1,
    Back = 2,
}

/// `MTLWinding` — MTLRenderCommandEncoder.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum MtlWinding {
    Clockwise = 0,
    CounterClockwise = 1,
}

/// `MTLTriangleFillMode` — MTLRenderCommandEncoder.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum MtlTriangleFillMode {
    Fill = 0,
    Lines = 1,
}

/// `MTLPrimitiveType` — MTLRenderCommandEncoder.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum MtlPrimitiveType {
    Point = 0,
    Line = 1,
    LineStrip = 2,
    Triangle = 3,
    TriangleStrip = 4,
}

/// `MTLPrimitiveTopologyClass` — MTLRenderPipeline.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum MtlPrimitiveTopologyClass {
    Unspecified = 0,
    Point = 1,
    Line = 2,
    Triangle = 3,
}

/// `MTLIndexType` — MTLArgument.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum MtlIndexType {
    UInt16 = 0,
    UInt32 = 1,
}

/// `MTLSamplerMinMagFilter` — MTLSampler.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum MtlSamplerMinMagFilter {
    Nearest = 0,
    Linear = 1,
}

/// `MTLSamplerMipFilter` — MTLSampler.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum MtlSamplerMipFilter {
    NotMipmapped = 0,
    Nearest = 1,
    Linear = 2,
}

/// `MTLSamplerAddressMode` — MTLSampler.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum MtlSamplerAddressMode {
    ClampToEdge = 0,
    MirrorClampToEdge = 1,
    Repeat = 2,
    MirrorRepeat = 3,
    ClampToZero = 4,
    ClampToBorderColor = 5,
}

/// `MTLSamplerBorderColor` — MTLSampler.h. Metal exposes only three fixed
/// border colors; everything else is snapped to the nearest one and flagged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum MtlSamplerBorderColor {
    TransparentBlack = 0, // {0,0,0,0}
    OpaqueBlack = 1,      // {0,0,0,1}
    OpaqueWhite = 2,      // {1,1,1,1}
}

/// `MTLSamplerReductionMode` — MTLSampler.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum MtlSamplerReductionMode {
    WeightedAverage = 0,
    Minimum = 1,
    Maximum = 2,
}

/// `MTLTextureType` — MTLTexture.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum MtlTextureType {
    Type1D = 0,
    Type1DArray = 1,
    Type2D = 2,
    Type2DArray = 3,
    Type2DMultisample = 4,
    TypeCube = 5,
    TypeCubeArray = 6,
    Type3D = 7,
    Type2DMultisampleArray = 8,
    TypeTextureBuffer = 9,
}

/// `MTLTextureUsage` — MTLTexture.h (NS_OPTIONS).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct MtlTextureUsage(pub u64);

impl MtlTextureUsage {
    pub const UNKNOWN: Self = Self(0x0000);
    pub const SHADER_READ: Self = Self(0x0001);
    pub const SHADER_WRITE: Self = Self(0x0002);
    pub const RENDER_TARGET: Self = Self(0x0004);
    pub const PIXEL_FORMAT_VIEW: Self = Self(0x0010);
    pub const SHADER_ATOMIC: Self = Self(0x0020);
}

// ===========================================================================
// D3D12 / DXGI side constants (plain values; windows crate 0.62.2 bindings,
// Microsoft public headers where the bindings omit a constant).
// ===========================================================================

/// DXGI_FORMAT — windows-0.62.2 `Win32::Graphics::Dxgi::Common`.
pub mod dxgi_format {
    pub const UNKNOWN: i32 = 0;
    pub const R32G32B32A32_TYPELESS: i32 = 1;
    pub const R32G32B32A32_FLOAT: i32 = 2;
    pub const R32G32B32A32_UINT: i32 = 3;
    pub const R32G32B32A32_SINT: i32 = 4;
    pub const R32G32B32_TYPELESS: i32 = 5;
    pub const R32G32B32_FLOAT: i32 = 6;
    pub const R32G32B32_UINT: i32 = 7;
    pub const R32G32B32_SINT: i32 = 8;
    pub const R16G16B16A16_TYPELESS: i32 = 9;
    pub const R16G16B16A16_FLOAT: i32 = 10;
    pub const R16G16B16A16_UNORM: i32 = 11;
    pub const R16G16B16A16_UINT: i32 = 12;
    pub const R16G16B16A16_SNORM: i32 = 13;
    pub const R16G16B16A16_SINT: i32 = 14;
    pub const R32G32_TYPELESS: i32 = 15;
    pub const R32G32_FLOAT: i32 = 16;
    pub const R32G32_UINT: i32 = 17;
    pub const R32G32_SINT: i32 = 18;
    pub const R32G8X24_TYPELESS: i32 = 19;
    pub const D32_FLOAT_S8X24_UINT: i32 = 20;
    pub const R32_FLOAT_X8X24_TYPELESS: i32 = 21;
    pub const X32_TYPELESS_G8X24_UINT: i32 = 22;
    pub const R10G10B10A2_TYPELESS: i32 = 23;
    pub const R10G10B10A2_UNORM: i32 = 24;
    pub const R10G10B10A2_UINT: i32 = 25;
    pub const R11G11B10_FLOAT: i32 = 26;
    pub const R8G8B8A8_TYPELESS: i32 = 27;
    pub const R8G8B8A8_UNORM: i32 = 28;
    pub const R8G8B8A8_UNORM_SRGB: i32 = 29;
    pub const R8G8B8A8_UINT: i32 = 30;
    pub const R8G8B8A8_SNORM: i32 = 31;
    pub const R8G8B8A8_SINT: i32 = 32;
    pub const R16G16_TYPELESS: i32 = 33;
    pub const R16G16_FLOAT: i32 = 34;
    pub const R16G16_UNORM: i32 = 35;
    pub const R16G16_UINT: i32 = 36;
    pub const R16G16_SNORM: i32 = 37;
    pub const R16G16_SINT: i32 = 38;
    pub const R32_TYPELESS: i32 = 39;
    pub const D32_FLOAT: i32 = 40;
    pub const R32_FLOAT: i32 = 41;
    pub const R32_UINT: i32 = 42;
    pub const R32_SINT: i32 = 43;
    pub const R24G8_TYPELESS: i32 = 44;
    pub const D24_UNORM_S8_UINT: i32 = 45;
    pub const R24_UNORM_X8_TYPELESS: i32 = 46;
    pub const X24_TYPELESS_G8_UINT: i32 = 47;
    pub const R8G8_TYPELESS: i32 = 48;
    pub const R8G8_UNORM: i32 = 49;
    pub const R8G8_UINT: i32 = 50;
    pub const R8G8_SNORM: i32 = 51;
    pub const R8G8_SINT: i32 = 52;
    pub const R16_TYPELESS: i32 = 53;
    pub const R16_FLOAT: i32 = 54;
    pub const D16_UNORM: i32 = 55;
    pub const R16_UNORM: i32 = 56;
    pub const R16_UINT: i32 = 57;
    pub const R16_SNORM: i32 = 58;
    pub const R16_SINT: i32 = 59;
    pub const R8_TYPELESS: i32 = 60;
    pub const R8_UNORM: i32 = 61;
    pub const R8_UINT: i32 = 62;
    pub const R8_SNORM: i32 = 63;
    pub const R8_SINT: i32 = 64;
    pub const A8_UNORM: i32 = 65;
    pub const R1_UNORM: i32 = 66;
    pub const R9G9B9E5_SHAREDEXP: i32 = 67;
    pub const R8G8_B8G8_UNORM: i32 = 68;
    pub const G8R8_G8B8_UNORM: i32 = 69;
    pub const BC1_TYPELESS: i32 = 70;
    pub const BC1_UNORM: i32 = 71;
    pub const BC1_UNORM_SRGB: i32 = 72;
    pub const BC2_TYPELESS: i32 = 73;
    pub const BC2_UNORM: i32 = 74;
    pub const BC2_UNORM_SRGB: i32 = 75;
    pub const BC3_TYPELESS: i32 = 76;
    pub const BC3_UNORM: i32 = 77;
    pub const BC3_UNORM_SRGB: i32 = 78;
    pub const BC4_TYPELESS: i32 = 79;
    pub const BC4_UNORM: i32 = 80;
    pub const BC4_SNORM: i32 = 81;
    pub const BC5_TYPELESS: i32 = 82;
    pub const BC5_UNORM: i32 = 83;
    pub const BC5_SNORM: i32 = 84;
    pub const B5G6R5_UNORM: i32 = 85;
    pub const B5G5R5A1_UNORM: i32 = 86;
    pub const B8G8R8A8_UNORM: i32 = 87;
    pub const B8G8R8X8_UNORM: i32 = 88;
    pub const R10G10B10_XR_BIAS_A2_UNORM: i32 = 89;
    pub const B8G8R8A8_TYPELESS: i32 = 90;
    pub const B8G8R8A8_UNORM_SRGB: i32 = 91;
    pub const B8G8R8X8_TYPELESS: i32 = 92;
    pub const B8G8R8X8_UNORM_SRGB: i32 = 93;
    pub const BC6H_TYPELESS: i32 = 94;
    pub const BC6H_UF16: i32 = 95;
    pub const BC6H_SF16: i32 = 96;
    pub const BC7_TYPELESS: i32 = 97;
    pub const BC7_UNORM: i32 = 98;
    pub const BC7_UNORM_SRGB: i32 = 99;
    pub const AYUV: i32 = 100;
    pub const Y410: i32 = 101;
    pub const Y416: i32 = 102;
    pub const NV12: i32 = 103;
    pub const P010: i32 = 104;
    pub const P016: i32 = 105;
    pub const OPAQUE_420: i32 = 106; // upstream: DXGI_FORMAT_420_OPAQUE
    pub const YUY2: i32 = 107;
    pub const Y210: i32 = 108;
    pub const Y216: i32 = 109;
    pub const NV11: i32 = 110;
    pub const AI44: i32 = 111;
    pub const IA44: i32 = 112;
    pub const P8: i32 = 113;
    pub const A8P8: i32 = 114;
    pub const B4G4R4A4_UNORM: i32 = 115;
    pub const P208: i32 = 130;
    pub const V208: i32 = 131;
    pub const V408: i32 = 132;
    pub const A4B4G4R4_UNORM: i32 = 191;
    pub const SAMPLER_FEEDBACK_MIN_MIP_OPAQUE: i32 = 189;
    pub const SAMPLER_FEEDBACK_MIP_REGION_USED_OPAQUE: i32 = 190;
}

/// D3D12_BLEND — windows-0.62.2 `Win32::Graphics::Direct3D12`.
pub mod d3d12_blend {
    pub const ZERO: i32 = 1;
    pub const ONE: i32 = 2;
    pub const SRC_COLOR: i32 = 3;
    pub const INV_SRC_COLOR: i32 = 4;
    pub const SRC_ALPHA: i32 = 5;
    pub const INV_SRC_ALPHA: i32 = 6;
    pub const DEST_ALPHA: i32 = 7;
    pub const INV_DEST_ALPHA: i32 = 8;
    pub const DEST_COLOR: i32 = 9;
    pub const INV_DEST_COLOR: i32 = 10;
    pub const SRC_ALPHA_SAT: i32 = 11;
    pub const BLEND_FACTOR: i32 = 14;
    pub const INV_BLEND_FACTOR: i32 = 15;
    pub const SRC1_COLOR: i32 = 16;
    pub const INV_SRC1_COLOR: i32 = 17;
    pub const SRC1_ALPHA: i32 = 18;
    pub const INV_SRC1_ALPHA: i32 = 19;
}

/// D3D12_BLEND_OP — Direct3D12 bindings.
pub mod d3d12_blend_op {
    pub const ADD: i32 = 1;
    pub const SUBTRACT: i32 = 2;
    pub const REV_SUBTRACT: i32 = 3;
    pub const MIN: i32 = 4;
    pub const MAX: i32 = 5;
}

/// D3D12_COLOR_WRITE_ENABLE — Direct3D12 bindings (bit mask: R=1 G=2 B=4 A=8).
pub mod d3d12_color_write {
    pub const RED: u32 = 1;
    pub const GREEN: u32 = 2;
    pub const BLUE: u32 = 4;
    pub const ALPHA: u32 = 8;
    pub const ALL: u32 = 15;
}

/// D3D12_COMPARISON_FUNC — Direct3D12 bindings.
pub mod d3d12_compare {
    pub const NEVER: i32 = 1;
    pub const LESS: i32 = 2;
    pub const EQUAL: i32 = 3;
    pub const LESS_EQUAL: i32 = 4;
    pub const GREATER: i32 = 5;
    pub const NOT_EQUAL: i32 = 6;
    pub const GREATER_EQUAL: i32 = 7;
    pub const ALWAYS: i32 = 8;
}

/// D3D12_STENCIL_OP — Direct3D12 bindings.
pub mod d3d12_stencil_op {
    pub const KEEP: i32 = 1;
    pub const ZERO: i32 = 2;
    pub const REPLACE: i32 = 3;
    pub const INCR_SAT: i32 = 4;
    pub const DECR_SAT: i32 = 5;
    pub const INVERT: i32 = 6;
    pub const INCR: i32 = 7;
    pub const DECR: i32 = 8;
}

/// D3D12_CULL_MODE — Direct3D12 bindings.
pub mod d3d12_cull_mode {
    pub const NONE: i32 = 1;
    pub const FRONT: i32 = 2;
    pub const BACK: i32 = 3;
}

/// D3D12_FILL_MODE — Direct3D12 bindings.
pub mod d3d12_fill_mode {
    pub const SOLID: i32 = 3;
    pub const WIREFRAME: i32 = 2;
}

/// D3D_PRIMITIVE_TOPOLOGY — Microsoft public header `d3dcommon.h` (the plain
/// list/strip constants are not generated by the windows 0.62.2 bindings).
/// Patches (33..64) are intentionally omitted from the mapping table.
pub mod d3d_primitive_topology {
    pub const UNDEFINED: i32 = 0;
    pub const POINT_LIST: i32 = 1;
    pub const LINE_LIST: i32 = 2;
    pub const LINE_STRIP: i32 = 3;
    pub const TRIANGLE_LIST: i32 = 4;
    pub const TRIANGLE_STRIP: i32 = 5;
    pub const LINE_LIST_ADJ: i32 = 10;
    pub const LINE_STRIP_ADJ: i32 = 11;
    pub const TRIANGLE_LIST_ADJ: i32 = 12;
    pub const TRIANGLE_STRIP_ADJ: i32 = 13;
}

/// D3D12_PRIMITIVE_TOPOLOGY_TYPE — Direct3D12 bindings (PSO-level class).
pub mod d3d12_topology_type {
    pub const UNDEFINED: i32 = 0;
    pub const POINT: i32 = 1;
    pub const LINE: i32 = 2;
    pub const TRIANGLE: i32 = 3;
    pub const PATCH: i32 = 4;
}

/// D3D12_TEXTURE_ADDRESS_MODE — Direct3D12 bindings.
pub mod d3d12_address_mode {
    pub const WRAP: i32 = 1;
    pub const MIRROR: i32 = 2;
    pub const CLAMP: i32 = 3;
    pub const BORDER: i32 = 4;
    pub const MIRROR_ONCE: i32 = 5;
}

/// D3D12_FILTER bit layout — Direct3D12 bindings:
/// low 6 bits are the three 2-bit filter fields (mip: 0–1, mag: 2–3,
/// min: 4–5; 0 = point, 1 = linear, 3 = anisotropic); reduction type is
/// shifted by 7: 0 standard, 1 comparison, 2 minimum, 3 maximum.
pub mod d3d12_filter {
    pub const TYPE_MASK: i32 = 3;
    pub const TYPE_POINT: i32 = 0;
    pub const TYPE_LINEAR: i32 = 1;
    pub const TYPE_ANISOTROPIC: i32 = 3;
    pub const REDUCTION_MASK: i32 = 3;
    pub const REDUCTION_SHIFT: i32 = 7;
    pub const REDUCTION_STANDARD: i32 = 0;
    pub const REDUCTION_COMPARISON: i32 = 1;
    pub const REDUCTION_MINIMUM: i32 = 2;
    pub const REDUCTION_MAXIMUM: i32 = 3;
}

/// D3D12_RESOURCE_DIMENSION — Direct3D12 bindings.
pub mod d3d12_resource_dimension {
    pub const BUFFER: i32 = 1;
    pub const TEXTURE1D: i32 = 2;
    pub const TEXTURE2D: i32 = 3;
    pub const TEXTURE3D: i32 = 4;
}

/// D3D12_RESOURCE_FLAG_* bit values — Direct3D12 bindings.
pub mod d3d12_resource_flag {
    pub const NONE: i32 = 0;
    pub const ALLOW_RENDER_TARGET: i32 = 1 << 0;
    pub const ALLOW_DEPTH_STENCIL: i32 = 1 << 1;
    pub const ALLOW_UNORDERED_ACCESS: i32 = 1 << 2;
    pub const DENY_SHADER_RESOURCE: i32 = 1 << 3;
    pub const ALLOW_CROSS_ADAPTER: i32 = 1 << 4;
    pub const ALLOW_SIMULTANEOUS_ACCESS: i32 = 1 << 5;
    pub const VIDEO_DECODE_REFERENCE_ONLY: i32 = 1 << 6;
    pub const VIDEO_ENCODE_REFERENCE_ONLY: i32 = 1 << 7;
    pub const RAYTRACING_ACCELERATION_STRUCTURE: i32 = 1 << 8;
}

// ===========================================================================
// 1. DXGI_FORMAT → MTLPixelFormat + format properties
// ===========================================================================

/// Static properties of a pixel format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatProps {
    /// Bytes per compressed block (4×4 for BC) or per texel for uncompressed
    /// formats; 0 when the format has no memory layout Metal can represent.
    pub block_size_bytes: u32,
    /// Block width in texels (1 for uncompressed, 4 for BC).
    pub block_width: u32,
    /// Block height in texels (1 for uncompressed, 4 for BC).
    pub block_height: u32,
    /// True for BC1–BC7 compressed formats.
    pub compressed: bool,
    /// True for depth / stencil formats.
    pub depth_stencil: bool,
}

impl FormatProps {
    const fn uncompressed(bytes: u32) -> Self {
        Self { block_size_bytes: bytes, block_width: 1, block_height: 1, compressed: false, depth_stencil: false }
    }
    const fn depth_stencil(bytes: u32) -> Self {
        Self { block_size_bytes: bytes, block_width: 1, block_height: 1, compressed: false, depth_stencil: true }
    }
    const fn bc(block_bytes: u32) -> Self {
        Self { block_size_bytes: block_bytes, block_width: 4, block_height: 4, compressed: true, depth_stencil: false }
    }
    const fn none() -> Self {
        Self { block_size_bytes: 0, block_width: 0, block_height: 0, compressed: false, depth_stencil: false }
    }
}

/// One row of the format table: (DXGI_FORMAT, Metal format, properties, note).
/// A `None` Metal format means there is no Metal equivalent; the note records
/// the reason. Typeless formats are mapped to their most reasonable concrete
/// format; the note records the choice.
const FORMAT_TABLE: &[(i32, Option<MtlPixelFormat>, FormatProps, &str)] = &[
    // --- 128 bit, four channels -------------------------------------------
    (dxgi_format::R32G32B32A32_TYPELESS, Some(MtlPixelFormat::RGBA32Float), FormatProps::uncompressed(16),
     "typeless: concrete twin chosen as RGBA32Float (32-bit channels; reinterpreted by views)"),
    (dxgi_format::R32G32B32A32_FLOAT, Some(MtlPixelFormat::RGBA32Float), FormatProps::uncompressed(16), ""),
    (dxgi_format::R32G32B32A32_UINT, Some(MtlPixelFormat::RGBA32Uint), FormatProps::uncompressed(16), ""),
    (dxgi_format::R32G32B32A32_SINT, Some(MtlPixelFormat::RGBA32Sint), FormatProps::uncompressed(16), ""),
    (dxgi_format::R32G32B32_TYPELESS, None, FormatProps::none(), "96-bit three-channel; Metal has no 96-bit pixel format"),
    (dxgi_format::R32G32B32_FLOAT, None, FormatProps::none(), "96-bit RGB32; Metal has no 96-bit pixel format"),
    (dxgi_format::R32G32B32_UINT, None, FormatProps::none(), "96-bit RGB32; Metal has no 96-bit pixel format"),
    (dxgi_format::R32G32B32_SINT, None, FormatProps::none(), "96-bit RGB32; Metal has no 96-bit pixel format"),

    // --- 64 bit, four channels (16×4) --------------------------------------
    (dxgi_format::R16G16B16A16_TYPELESS, Some(MtlPixelFormat::RGBA16Float), FormatProps::uncompressed(8),
     "typeless: concrete twin chosen as RGBA16Float (console 16×4 textures are typically float)"),
    (dxgi_format::R16G16B16A16_FLOAT, Some(MtlPixelFormat::RGBA16Float), FormatProps::uncompressed(8), ""),
    (dxgi_format::R16G16B16A16_UNORM, Some(MtlPixelFormat::RGBA16Unorm), FormatProps::uncompressed(8), ""),
    (dxgi_format::R16G16B16A16_UINT, Some(MtlPixelFormat::RGBA16Uint), FormatProps::uncompressed(8), ""),
    (dxgi_format::R16G16B16A16_SNORM, Some(MtlPixelFormat::RGBA16Snorm), FormatProps::uncompressed(8), ""),
    (dxgi_format::R16G16B16A16_SINT, Some(MtlPixelFormat::RGBA16Sint), FormatProps::uncompressed(8), ""),

    // --- 64 bit, two channels (32×2) ---------------------------------------
    (dxgi_format::R32G32_TYPELESS, Some(MtlPixelFormat::RG32Float), FormatProps::uncompressed(8),
     "typeless: concrete twin chosen as RG32Float"),
    (dxgi_format::R32G32_FLOAT, Some(MtlPixelFormat::RG32Float), FormatProps::uncompressed(8), ""),
    (dxgi_format::R32G32_UINT, Some(MtlPixelFormat::RG32Uint), FormatProps::uncompressed(8), ""),
    (dxgi_format::R32G32_SINT, Some(MtlPixelFormat::RG32Sint), FormatProps::uncompressed(8), ""),

    // --- Depth/stencil, 32+8 ------------------------------------------------
    (dxgi_format::R32G8X24_TYPELESS, Some(MtlPixelFormat::Depth32Float_Stencil8), FormatProps::depth_stencil(8),
     "typeless twin of D32_FLOAT_S8X24_UINT"),
    (dxgi_format::D32_FLOAT_S8X24_UINT, Some(MtlPixelFormat::Depth32Float_Stencil8), FormatProps::depth_stencil(8), ""),
    (dxgi_format::R32_FLOAT_X8X24_TYPELESS, Some(MtlPixelFormat::X32_Stencil8), FormatProps::depth_stencil(8),
     "depth-only view: stencil channel discarded; maps to X32_Stencil8 paired format"),
    (dxgi_format::X32_TYPELESS_G8X24_UINT, Some(MtlPixelFormat::X32_Stencil8), FormatProps::depth_stencil(8),
     "stencil-only view of a 32-bit depth+stencil texture; Apple7 maps to X32_Stencil8 (F39)"),

    // --- Packed 32 bit ------------------------------------------------------
    (dxgi_format::R10G10B10A2_TYPELESS, Some(MtlPixelFormat::RGB10A2Unorm), FormatProps::uncompressed(4),
     "typeless: concrete twin chosen as RGB10A2Unorm"),
    (dxgi_format::R10G10B10A2_UNORM, Some(MtlPixelFormat::RGB10A2Unorm), FormatProps::uncompressed(4), ""),
    (dxgi_format::R10G10B10A2_UINT, Some(MtlPixelFormat::RGB10A2Uint), FormatProps::uncompressed(4), ""),
    (dxgi_format::R11G11B10_FLOAT, Some(MtlPixelFormat::RG11B10Float), FormatProps::uncompressed(4), ""),

    // --- 32 bit, four channels (8×4) ----------------------------------------
    (dxgi_format::R8G8B8A8_TYPELESS, Some(MtlPixelFormat::RGBA8Unorm), FormatProps::uncompressed(4),
     "typeless: concrete twin chosen as RGBA8Unorm"),
    (dxgi_format::R8G8B8A8_UNORM, Some(MtlPixelFormat::RGBA8Unorm), FormatProps::uncompressed(4), ""),
    (dxgi_format::R8G8B8A8_UNORM_SRGB, Some(MtlPixelFormat::RGBA8Unorm_sRGB), FormatProps::uncompressed(4), ""),
    (dxgi_format::R8G8B8A8_UINT, Some(MtlPixelFormat::RGBA8Uint), FormatProps::uncompressed(4), ""),
    (dxgi_format::R8G8B8A8_SNORM, Some(MtlPixelFormat::RGBA8Snorm), FormatProps::uncompressed(4), ""),
    (dxgi_format::R8G8B8A8_SINT, Some(MtlPixelFormat::RGBA8Sint), FormatProps::uncompressed(4), ""),

    // --- 32 bit, two channels (16×2) ---------------------------------------
    (dxgi_format::R16G16_TYPELESS, Some(MtlPixelFormat::RG16Float), FormatProps::uncompressed(4),
     "typeless: concrete twin chosen as RG16Float"),
    (dxgi_format::R16G16_FLOAT, Some(MtlPixelFormat::RG16Float), FormatProps::uncompressed(4), ""),
    (dxgi_format::R16G16_UNORM, Some(MtlPixelFormat::RG16Unorm), FormatProps::uncompressed(4), ""),
    (dxgi_format::R16G16_UINT, Some(MtlPixelFormat::RG16Uint), FormatProps::uncompressed(4), ""),
    (dxgi_format::R16G16_SNORM, Some(MtlPixelFormat::RG16Snorm), FormatProps::uncompressed(4), ""),
    (dxgi_format::R16G16_SINT, Some(MtlPixelFormat::RG16Sint), FormatProps::uncompressed(4), ""),

    // --- 32 bit, single channel / depth -------------------------------------
    (dxgi_format::R32_TYPELESS, Some(MtlPixelFormat::R32Float), FormatProps::uncompressed(4),
     "typeless: concrete twin chosen as R32Float"),
    (dxgi_format::D32_FLOAT, Some(MtlPixelFormat::Depth32Float), FormatProps::depth_stencil(4), ""),
    (dxgi_format::R32_FLOAT, Some(MtlPixelFormat::R32Float), FormatProps::uncompressed(4), ""),
    (dxgi_format::R32_UINT, Some(MtlPixelFormat::R32Uint), FormatProps::uncompressed(4), ""),
    (dxgi_format::R32_SINT, Some(MtlPixelFormat::R32Sint), FormatProps::uncompressed(4), ""),

    // --- Depth/stencil, 24+8 -------------------------------------------------
    (dxgi_format::R24G8_TYPELESS, Some(MtlPixelFormat::Depth32Float_Stencil8), FormatProps::depth_stencil(8),
     "typeless twin of D24_UNORM_S8_UINT; Depth24Unorm_Stencil8 unsupported on Apple7, upconvert to Depth32Float_Stencil8 (F39)"),
    (dxgi_format::D24_UNORM_S8_UINT, Some(MtlPixelFormat::Depth32Float_Stencil8), FormatProps::depth_stencil(8),
     "Depth24Unorm_Stencil8 invalid on Apple7; upconvert to Depth32Float_Stencil8 (F39)"),
    (dxgi_format::R24_UNORM_X8_TYPELESS, Some(MtlPixelFormat::Depth32Float_Stencil8), FormatProps::depth_stencil(8),
     "depth-only view of the upconverted Depth32Float_Stencil8 parent; stencil bits ignored (F39)"),
    (dxgi_format::X24_TYPELESS_G8_UINT, Some(MtlPixelFormat::X32_Stencil8), FormatProps::depth_stencil(8),
     "X24_Stencil8 invalid on Apple7; stencil-only view maps to X32_Stencil8 (F39)"),

    // --- 16 bit, two channels (8×2) ----------------------------------------
    (dxgi_format::R8G8_TYPELESS, Some(MtlPixelFormat::RG8Unorm), FormatProps::uncompressed(2),
     "typeless: concrete twin chosen as RG8Unorm"),
    (dxgi_format::R8G8_UNORM, Some(MtlPixelFormat::RG8Unorm), FormatProps::uncompressed(2), ""),
    (dxgi_format::R8G8_UINT, Some(MtlPixelFormat::RG8Uint), FormatProps::uncompressed(2), ""),
    (dxgi_format::R8G8_SNORM, Some(MtlPixelFormat::RG8Snorm), FormatProps::uncompressed(2), ""),
    (dxgi_format::R8G8_SINT, Some(MtlPixelFormat::RG8Sint), FormatProps::uncompressed(2), ""),

    // --- 16 bit, single channel --------------------------------------------
    (dxgi_format::R16_TYPELESS, Some(MtlPixelFormat::R16Float), FormatProps::uncompressed(2),
     "typeless: concrete twin chosen as R16Float"),
    (dxgi_format::R16_FLOAT, Some(MtlPixelFormat::R16Float), FormatProps::uncompressed(2), ""),
    (dxgi_format::D16_UNORM, Some(MtlPixelFormat::Depth16Unorm), FormatProps::depth_stencil(2), ""),
    (dxgi_format::R16_UNORM, Some(MtlPixelFormat::R16Unorm), FormatProps::uncompressed(2), ""),
    (dxgi_format::R16_UINT, Some(MtlPixelFormat::R16Uint), FormatProps::uncompressed(2), ""),
    (dxgi_format::R16_SNORM, Some(MtlPixelFormat::R16Snorm), FormatProps::uncompressed(2), ""),
    (dxgi_format::R16_SINT, Some(MtlPixelFormat::R16Sint), FormatProps::uncompressed(2), ""),

    // --- 8 bit, single channel ---------------------------------------------
    (dxgi_format::R8_TYPELESS, Some(MtlPixelFormat::R8Unorm), FormatProps::uncompressed(1),
     "typeless: concrete twin chosen as R8Unorm"),
    (dxgi_format::R8_UNORM, Some(MtlPixelFormat::R8Unorm), FormatProps::uncompressed(1), ""),
    (dxgi_format::R8_UINT, Some(MtlPixelFormat::R8Uint), FormatProps::uncompressed(1), ""),
    (dxgi_format::R8_SNORM, Some(MtlPixelFormat::R8Snorm), FormatProps::uncompressed(1), ""),
    (dxgi_format::R8_SINT, Some(MtlPixelFormat::R8Sint), FormatProps::uncompressed(1), ""),
    (dxgi_format::A8_UNORM, Some(MtlPixelFormat::A8Unorm), FormatProps::uncompressed(1), ""),
    (dxgi_format::R1_UNORM, None, FormatProps::none(), "1-bit per channel; Metal has no 1-bit format"),

    // --- Shared exponent / packed 16-bit ------------------------------------
    (dxgi_format::R9G9B9E5_SHAREDEXP, Some(MtlPixelFormat::RGB9E5Float), FormatProps::uncompressed(4), ""),
    (dxgi_format::R8G8_B8G8_UNORM, None, FormatProps::none(),
     "packed planar 422 (R8G8_B8G8); Metal 422 formats are UYVY/YUY2 layouts only"),
    (dxgi_format::G8R8_G8B8_UNORM, None, FormatProps::none(),
     "packed planar 422 (G8R8_G8B8); Metal 422 formats are UYVY/YUY2 layouts only"),

    // --- BC1 (S3TC DXT1): 8 bytes / 4×4 block ------------------------------
    (dxgi_format::BC1_TYPELESS, Some(MtlPixelFormat::BC1_RGBA), FormatProps::bc(8),
     "typeless: concrete twin chosen as BC1_RGBA"),
    (dxgi_format::BC1_UNORM, Some(MtlPixelFormat::BC1_RGBA), FormatProps::bc(8), ""),
    (dxgi_format::BC1_UNORM_SRGB, Some(MtlPixelFormat::BC1_RGBA_sRGB), FormatProps::bc(8), ""),
    // --- BC2 (DXT3): 16 bytes / 4×4 block ----------------------------------
    (dxgi_format::BC2_TYPELESS, Some(MtlPixelFormat::BC2_RGBA), FormatProps::bc(16),
     "typeless: concrete twin chosen as BC2_RGBA"),
    (dxgi_format::BC2_UNORM, Some(MtlPixelFormat::BC2_RGBA), FormatProps::bc(16), ""),
    (dxgi_format::BC2_UNORM_SRGB, Some(MtlPixelFormat::BC2_RGBA_sRGB), FormatProps::bc(16), ""),
    // --- BC3 (DXT5): 16 bytes / 4×4 block ----------------------------------
    (dxgi_format::BC3_TYPELESS, Some(MtlPixelFormat::BC3_RGBA), FormatProps::bc(16),
     "typeless: concrete twin chosen as BC3_RGBA"),
    (dxgi_format::BC3_UNORM, Some(MtlPixelFormat::BC3_RGBA), FormatProps::bc(16), ""),
    (dxgi_format::BC3_UNORM_SRGB, Some(MtlPixelFormat::BC3_RGBA_sRGB), FormatProps::bc(16), ""),
    // --- BC4 (RGTC R): 8 bytes / 4×4 block ----------------------------------
    (dxgi_format::BC4_TYPELESS, Some(MtlPixelFormat::BC4_RUnorm), FormatProps::bc(8),
     "typeless: concrete twin chosen as BC4_RUnorm"),
    (dxgi_format::BC4_UNORM, Some(MtlPixelFormat::BC4_RUnorm), FormatProps::bc(8), ""),
    (dxgi_format::BC4_SNORM, Some(MtlPixelFormat::BC4_RSnorm), FormatProps::bc(8), ""),
    // --- BC5 (RGTC RG): 16 bytes / 4×4 block --------------------------------
    (dxgi_format::BC5_TYPELESS, Some(MtlPixelFormat::BC5_RGUnorm), FormatProps::bc(16),
     "typeless: concrete twin chosen as BC5_RGUnorm"),
    (dxgi_format::BC5_UNORM, Some(MtlPixelFormat::BC5_RGUnorm), FormatProps::bc(16), ""),
    (dxgi_format::BC5_SNORM, Some(MtlPixelFormat::BC5_RGSnorm), FormatProps::bc(16), ""),

    // --- Packed 16-bit color ------------------------------------------------
    (dxgi_format::B5G6R5_UNORM, Some(MtlPixelFormat::B5G6R5Unorm), FormatProps::uncompressed(2), ""),
    (dxgi_format::B5G5R5A1_UNORM, Some(MtlPixelFormat::A1BGR5Unorm), FormatProps::uncompressed(2), ""),
    (dxgi_format::B4G4R4A4_UNORM, Some(MtlPixelFormat::ABGR4Unorm), FormatProps::uncompressed(2),
     "channel order reinterpreted as ABGR4Unorm; see open questions"),

    // --- BGRA 32-bit ---------------------------------------------------------
    (dxgi_format::B8G8R8A8_UNORM, Some(MtlPixelFormat::BGRA8Unorm), FormatProps::uncompressed(4), ""),
    (dxgi_format::B8G8R8X8_UNORM, Some(MtlPixelFormat::BGRA8Unorm), FormatProps::uncompressed(4),
     "X8 alpha is ignored; stored in the alpha channel"),
    (dxgi_format::R10G10B10_XR_BIAS_A2_UNORM, None, FormatProps::none(),
     "XR bias (extended range) format; Metal has no equivalent"),
    (dxgi_format::B8G8R8A8_TYPELESS, Some(MtlPixelFormat::BGRA8Unorm), FormatProps::uncompressed(4),
     "typeless: concrete twin chosen as BGRA8Unorm"),
    (dxgi_format::B8G8R8A8_UNORM_SRGB, Some(MtlPixelFormat::BGRA8Unorm_sRGB), FormatProps::uncompressed(4), ""),
    (dxgi_format::B8G8R8X8_TYPELESS, Some(MtlPixelFormat::BGRA8Unorm), FormatProps::uncompressed(4),
     "typeless: concrete twin chosen as BGRA8Unorm"),
    (dxgi_format::B8G8R8X8_UNORM_SRGB, Some(MtlPixelFormat::BGRA8Unorm_sRGB), FormatProps::uncompressed(4),
     "X8 alpha is ignored; stored in the alpha channel"),

    // --- BC6H: 16 bytes / 4×4 block ------------------------------------------
    (dxgi_format::BC6H_TYPELESS, Some(MtlPixelFormat::BC6H_RGBUfloat), FormatProps::bc(16),
     "typeless: concrete twin chosen as BC6H_RGBUfloat"),
    (dxgi_format::BC6H_UF16, Some(MtlPixelFormat::BC6H_RGBUfloat), FormatProps::bc(16), ""),
    (dxgi_format::BC6H_SF16, Some(MtlPixelFormat::BC6H_RGBFloat), FormatProps::bc(16), ""),
    // --- BC7: 16 bytes / 4×4 block ------------------------------------------
    (dxgi_format::BC7_TYPELESS, Some(MtlPixelFormat::BC7_RGBAUnorm), FormatProps::bc(16),
     "typeless: concrete twin chosen as BC7_RGBAUnorm"),
    (dxgi_format::BC7_UNORM, Some(MtlPixelFormat::BC7_RGBAUnorm), FormatProps::bc(16), ""),
    (dxgi_format::BC7_UNORM_SRGB, Some(MtlPixelFormat::BC7_RGBAUnorm_sRGB), FormatProps::bc(16), ""),

    // --- Packed / planar YUV: no Metal equivalent ----------------------------
    (dxgi_format::AYUV, None, FormatProps::none(), "packed YUV444; Metal has no YUV444 pixel format"),
    (dxgi_format::Y410, None, FormatProps::none(), "packed 10-bit YUV444; no Metal equivalent"),
    (dxgi_format::Y416, None, FormatProps::none(), "packed 16-bit YUV444; no Metal equivalent"),
    (dxgi_format::NV12, None, FormatProps::none(), "planar YUV 4:2:0; Metal has no planar NV12 texture format"),
    (dxgi_format::P010, None, FormatProps::none(), "planar 10-bit YUV 4:2:0; no Metal equivalent"),
    (dxgi_format::P016, None, FormatProps::none(), "planar 16-bit YUV 4:2:0; no Metal equivalent"),
    (dxgi_format::OPAQUE_420, None, FormatProps::none(), "opaque planar 4:2:0; no Metal equivalent"),
    (dxgi_format::YUY2, Some(MtlPixelFormat::GBGR422), FormatProps::uncompressed(4),
     "YUY2 (Y0,Cb,Y1,Cr) ≡ Metal GBGR422; 4 bytes per 2×1 texel group"),
    (dxgi_format::Y210, None, FormatProps::none(), "packed 10-bit YUV422; no Metal equivalent"),
    (dxgi_format::Y216, None, FormatProps::none(), "packed 16-bit YUV422; no Metal equivalent"),
    (dxgi_format::NV11, None, FormatProps::none(), "planar YUV 4:1:1; no Metal equivalent"),

    // --- Palette / nibble formats: no Metal equivalent -----------------------
    (dxgi_format::AI44, None, FormatProps::none(), "palettized nibble format; Metal has no palette formats"),
    (dxgi_format::IA44, None, FormatProps::none(), "palettized nibble format; Metal has no palette formats"),
    (dxgi_format::P8, None, FormatProps::none(), "8-bit indexed palette; Metal has no palette formats"),
    (dxgi_format::A8P8, None, FormatProps::none(), "8-bit indexed + alpha palette; Metal has no palette formats"),
    (dxgi_format::A4B4G4R4_UNORM, None, FormatProps::none(),
     "4444 RGBA order; Metal's only 4444 format is ABGR4 (order mismatch); see open questions"),

    // --- Miscellaneous / video feedback -------------------------------------
    (dxgi_format::P208, None, FormatProps::none(), "packed 8-bit YUV 4:2:2; no Metal equivalent"),
    (dxgi_format::V208, None, FormatProps::none(), "packed 8-bit YUV 4:4:4; no Metal equivalent"),
    (dxgi_format::V408, None, FormatProps::none(), "packed 8-bit YUV 4:4:4; no Metal equivalent"),
    (dxgi_format::SAMPLER_FEEDBACK_MIN_MIP_OPAQUE, None, FormatProps::none(),
     "sampler feedback opaque placeholder; not a renderable texture format"),
    (dxgi_format::SAMPLER_FEEDBACK_MIP_REGION_USED_OPAQUE, None, FormatProps::none(),
     "sampler feedback opaque placeholder; not a renderable texture format"),
];

/// Translate a `DXGI_FORMAT` into an `MTLPixelFormat`.
///
/// Returns `None` when Metal has no equivalent; the reason lives in the table
/// note (use [`format_note`] to read it).
pub fn map_pixel_format(dxgi: i32) -> Option<MtlPixelFormat> {
    FORMAT_TABLE.iter().find(|(f, _, _, _)| *f == dxgi).and_then(|(_, mtl, _, _)| *mtl)
}

/// Static properties for a `DXGI_FORMAT`. `None` when the format is unknown.
pub fn format_props(dxgi: i32) -> Option<FormatProps> {
    FORMAT_TABLE.iter().find(|(f, _, _, _)| *f == dxgi).map(|(_, _, props, _)| *props)
}

/// Free-text note for a `DXGI_FORMAT`: the typeless rationale, or the reason
/// there is no Metal equivalent. Empty string when the mapping is direct.
pub fn format_note(dxgi: i32) -> Option<&'static str> {
    FORMAT_TABLE.iter().find(|(f, _, _, _)| *f == dxgi).map(|(_, _, _, note)| *note)
}

// ===========================================================================
// 2. Blend / compare / stencil / rasterizer / topology / index
// ===========================================================================

/// Translate a `D3D12_BLEND` into an `MTLBlendFactor`.
pub fn map_blend_factor(d3d: i32) -> Option<MtlBlendFactor> {
    let mf = |v: MtlBlendFactor| Some(v);
    match d3d {
        d3d12_blend::ZERO => mf(MtlBlendFactor::Zero),
        d3d12_blend::ONE => mf(MtlBlendFactor::One),
        d3d12_blend::SRC_COLOR => mf(MtlBlendFactor::SourceColor),
        d3d12_blend::INV_SRC_COLOR => mf(MtlBlendFactor::OneMinusSourceColor),
        d3d12_blend::SRC_ALPHA => mf(MtlBlendFactor::SourceAlpha),
        d3d12_blend::INV_SRC_ALPHA => mf(MtlBlendFactor::OneMinusSourceAlpha),
        d3d12_blend::DEST_ALPHA => mf(MtlBlendFactor::DestinationAlpha),
        d3d12_blend::INV_DEST_ALPHA => mf(MtlBlendFactor::OneMinusDestinationAlpha),
        d3d12_blend::DEST_COLOR => mf(MtlBlendFactor::DestinationColor),
        d3d12_blend::INV_DEST_COLOR => mf(MtlBlendFactor::OneMinusDestinationColor),
        d3d12_blend::SRC_ALPHA_SAT => mf(MtlBlendFactor::SourceAlphaSaturated),
        d3d12_blend::BLEND_FACTOR => mf(MtlBlendFactor::BlendColor),
        d3d12_blend::INV_BLEND_FACTOR => mf(MtlBlendFactor::OneMinusBlendColor),
        d3d12_blend::SRC1_COLOR => mf(MtlBlendFactor::Source1Color),
        d3d12_blend::INV_SRC1_COLOR => mf(MtlBlendFactor::OneMinusSource1Color),
        d3d12_blend::SRC1_ALPHA => mf(MtlBlendFactor::Source1Alpha),
        d3d12_blend::INV_SRC1_ALPHA => mf(MtlBlendFactor::OneMinusSource1Alpha),
        _ => None,
    }
}

/// Translate a `D3D12_BLEND_OP` into an `MTLBlendOperation`.
pub fn map_blend_op(d3d: i32) -> Option<MtlBlendOperation> {
    match d3d {
        d3d12_blend_op::ADD => Some(MtlBlendOperation::Add),
        d3d12_blend_op::SUBTRACT => Some(MtlBlendOperation::Subtract),
        d3d12_blend_op::REV_SUBTRACT => Some(MtlBlendOperation::ReverseSubtract),
        d3d12_blend_op::MIN => Some(MtlBlendOperation::Min),
        d3d12_blend_op::MAX => Some(MtlBlendOperation::Max),
        _ => None,
    }
}

/// Translate the `D3D12_COLOR_WRITE_ENABLE` bit mask into an
/// `MTLColorWriteMask`. Note the channel bit order differs: D3D12 is
/// R=1 G=2 B=4 A=8, Metal is R=8 G=4 B=2 A=1.
pub fn map_color_write_mask(d3d: u32) -> MtlColorWriteMask {
    let mut out = MtlColorWriteMask::NONE;
    if d3d & d3d12_color_write::RED != 0 {
        out.0 |= MtlColorWriteMask::RED.0;
    }
    if d3d & d3d12_color_write::GREEN != 0 {
        out.0 |= MtlColorWriteMask::GREEN.0;
    }
    if d3d & d3d12_color_write::BLUE != 0 {
        out.0 |= MtlColorWriteMask::BLUE.0;
    }
    if d3d & d3d12_color_write::ALPHA != 0 {
        out.0 |= MtlColorWriteMask::ALPHA.0;
    }
    out
}

/// Translate a `D3D12_COMPARISON_FUNC` into an `MTLCompareFunction`.
pub fn map_compare_function(d3d: i32) -> Option<MtlCompareFunction> {
    match d3d {
        d3d12_compare::NEVER => Some(MtlCompareFunction::Never),
        d3d12_compare::LESS => Some(MtlCompareFunction::Less),
        d3d12_compare::EQUAL => Some(MtlCompareFunction::Equal),
        d3d12_compare::LESS_EQUAL => Some(MtlCompareFunction::LessEqual),
        d3d12_compare::GREATER => Some(MtlCompareFunction::Greater),
        d3d12_compare::NOT_EQUAL => Some(MtlCompareFunction::NotEqual),
        d3d12_compare::GREATER_EQUAL => Some(MtlCompareFunction::GreaterEqual),
        d3d12_compare::ALWAYS => Some(MtlCompareFunction::Always),
        _ => None,
    }
}

/// Translate a `D3D12_STENCIL_OP` into an `MTLStencilOperation`.
pub fn map_stencil_op(d3d: i32) -> Option<MtlStencilOperation> {
    match d3d {
        d3d12_stencil_op::KEEP => Some(MtlStencilOperation::Keep),
        d3d12_stencil_op::ZERO => Some(MtlStencilOperation::Zero),
        d3d12_stencil_op::REPLACE => Some(MtlStencilOperation::Replace),
        d3d12_stencil_op::INCR_SAT => Some(MtlStencilOperation::IncrementClamp),
        d3d12_stencil_op::DECR_SAT => Some(MtlStencilOperation::DecrementClamp),
        d3d12_stencil_op::INVERT => Some(MtlStencilOperation::Invert),
        d3d12_stencil_op::INCR => Some(MtlStencilOperation::IncrementWrap),
        d3d12_stencil_op::DECR => Some(MtlStencilOperation::DecrementWrap),
        _ => None,
    }
}

/// Translate a `D3D12_CULL_MODE` into an `MTLCullMode`.
pub fn map_cull_mode(d3d: i32) -> Option<MtlCullMode> {
    match d3d {
        d3d12_cull_mode::NONE => Some(MtlCullMode::None),
        d3d12_cull_mode::FRONT => Some(MtlCullMode::Front),
        d3d12_cull_mode::BACK => Some(MtlCullMode::Back),
        _ => None,
    }
}

/// Translate the rasterizer's `FrontCounterClockwise` flag into an
/// `MTLWinding`. D3D12's default (front face clockwise) maps to Metal's
/// clockwise winding.
pub fn map_winding(front_counter_clockwise: bool) -> MtlWinding {
    if front_counter_clockwise {
        MtlWinding::CounterClockwise
    } else {
        MtlWinding::Clockwise
    }
}

/// Translate a `D3D12_FILL_MODE` into an `MTLTriangleFillMode`.
pub fn map_fill_mode(d3d: i32) -> Option<MtlTriangleFillMode> {
    match d3d {
        d3d12_fill_mode::SOLID => Some(MtlTriangleFillMode::Fill),
        d3d12_fill_mode::WIREFRAME => Some(MtlTriangleFillMode::Lines),
        _ => None,
    }
}

/// Result of a topology translation: the per-draw primitive type plus the
/// pipeline-level topology class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TopologyMap {
    pub primitive_type: MtlPrimitiveType,
    pub topology_class: MtlPrimitiveTopologyClass,
}

/// Translate a `D3D_PRIMITIVE_TOPOLOGY` (the value set on the command list)
/// into an [`MTLPrimitiveType`] and its [`MTLPrimitiveTopologyClass`].
/// Adjacency topologies and tessellation patches have no Metal equivalent.
pub fn map_topology(d3d: i32) -> Option<TopologyMap> {
    match d3d {
        d3d_primitive_topology::POINT_LIST => Some(TopologyMap {
            primitive_type: MtlPrimitiveType::Point,
            topology_class: MtlPrimitiveTopologyClass::Point,
        }),
        d3d_primitive_topology::LINE_LIST => Some(TopologyMap {
            primitive_type: MtlPrimitiveType::Line,
            topology_class: MtlPrimitiveTopologyClass::Line,
        }),
        d3d_primitive_topology::LINE_STRIP => Some(TopologyMap {
            primitive_type: MtlPrimitiveType::LineStrip,
            topology_class: MtlPrimitiveTopologyClass::Line,
        }),
        d3d_primitive_topology::TRIANGLE_LIST => Some(TopologyMap {
            primitive_type: MtlPrimitiveType::Triangle,
            topology_class: MtlPrimitiveTopologyClass::Triangle,
        }),
        d3d_primitive_topology::TRIANGLE_STRIP => Some(TopologyMap {
            primitive_type: MtlPrimitiveType::TriangleStrip,
            topology_class: MtlPrimitiveTopologyClass::Triangle,
        }),
        // Adjacency topologies: Metal has no adjacency primitive type.
        d3d_primitive_topology::LINE_LIST_ADJ
        | d3d_primitive_topology::LINE_STRIP_ADJ
        | d3d_primitive_topology::TRIANGLE_LIST_ADJ
        | d3d_primitive_topology::TRIANGLE_STRIP_ADJ => None,
        _ => None, // UNDEFINED, patch lists (33..64), unknown values
    }
}

/// Translate a `D3D12_PRIMITIVE_TOPOLOGY_TYPE` (PSO level) into an
/// `MTLPrimitiveTopologyClass`. Patch topology has no Metal class.
pub fn map_topology_class(d3d: i32) -> Option<MtlPrimitiveTopologyClass> {
    match d3d {
        d3d12_topology_type::UNDEFINED => Some(MtlPrimitiveTopologyClass::Unspecified),
        d3d12_topology_type::POINT => Some(MtlPrimitiveTopologyClass::Point),
        d3d12_topology_type::LINE => Some(MtlPrimitiveTopologyClass::Line),
        d3d12_topology_type::TRIANGLE => Some(MtlPrimitiveTopologyClass::Triangle),
        d3d12_topology_type::PATCH => None,
        _ => None,
    }
}

/// Translate an index buffer format (a `DXGI_FORMAT`: R16_UINT / R32_UINT)
/// into an `MTLIndexType`.
pub fn map_index_format(dxgi: i32) -> Option<MtlIndexType> {
    match dxgi {
        dxgi_format::R16_UINT => Some(MtlIndexType::UInt16),
        dxgi_format::R32_UINT => Some(MtlIndexType::UInt32),
        _ => None,
    }
}

// ===========================================================================
// 3. Sampler
// ===========================================================================

/// Decoded sampler filter result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplerFilterMap {
    pub min_filter: MtlSamplerMinMagFilter,
    pub mag_filter: MtlSamplerMinMagFilter,
    pub mip_filter: MtlSamplerMipFilter,
    /// Standard / comparison / minimum / maximum reduction.
    pub reduction: MtlSamplerReductionMode,
    /// True when this is a comparison sampler (D3D12 reduction type 1).
    pub compare: bool,
    /// True when the filter is anisotropic; the caller supplies the requested
    /// maximum anisotropy separately.
    pub anisotropic: bool,
}

/// Anisotropic `D3D12_FILTER` values (min/mag/mip fields all carry the
/// anisotropic code 3), per `d3d12.h`.
const ANISOTROPIC_FILTERS: &[i32] = &[84, 85, 212, 213, 340, 341, 468, 469];

/// Decode a `D3D12_FILTER` into Metal sampler filter settings.
///
/// The bit layout (Direct3D12 bindings `D3D12_FILTER_TYPE_MASK`,
/// `D3D12_FILTER_REDUCTION_TYPE_SHIFT`): mip field bits 0–1, mag bits 2–3,
/// min bits 4–5; reduction type in bits 7–8.
pub fn map_sampler_filter(d3d: i32) -> Option<SamplerFilterMap> {
    let reduction_raw = (d3d >> d3d12_filter::REDUCTION_SHIFT) & d3d12_filter::REDUCTION_MASK;
    let reduction = match reduction_raw {
        d3d12_filter::REDUCTION_STANDARD => MtlSamplerReductionMode::WeightedAverage,
        d3d12_filter::REDUCTION_COMPARISON => MtlSamplerReductionMode::WeightedAverage,
        d3d12_filter::REDUCTION_MINIMUM => MtlSamplerReductionMode::Minimum,
        d3d12_filter::REDUCTION_MAXIMUM => MtlSamplerReductionMode::Maximum,
        _ => return None,
    };
    let compare = reduction_raw == d3d12_filter::REDUCTION_COMPARISON;
    let anisotropic = ANISOTROPIC_FILTERS.contains(&d3d);

    let field = |shift: u32| -> i32 { (d3d >> shift) & d3d12_filter::TYPE_MASK };
    let min = field(4);
    let mag = field(2);
    let mip = field(0);

    let mm = |v: i32| -> Option<MtlSamplerMinMagFilter> {
        match v {
            d3d12_filter::TYPE_POINT => Some(MtlSamplerMinMagFilter::Nearest),
            d3d12_filter::TYPE_LINEAR => Some(MtlSamplerMinMagFilter::Linear),
            // Anisotropic field code 3: Metal applies anisotropy on top of a
            // linear min/mag filter.
            d3d12_filter::TYPE_ANISOTROPIC => Some(MtlSamplerMinMagFilter::Linear),
            _ => None,
        }
    };
    let min_filter = mm(min)?;
    let mag_filter = mm(mag)?;
    let mip_filter = match mip {
        d3d12_filter::TYPE_POINT => MtlSamplerMipFilter::Nearest,
        d3d12_filter::TYPE_LINEAR => MtlSamplerMipFilter::Linear,
        d3d12_filter::TYPE_ANISOTROPIC => MtlSamplerMipFilter::Linear,
        _ => return None,
    };

    Some(SamplerFilterMap { min_filter, mag_filter, mip_filter, reduction, compare, anisotropic })
}

/// Translate a `D3D12_TEXTURE_ADDRESS_MODE` into an `MTLSamplerAddressMode`.
pub fn map_address_mode(d3d: i32) -> Option<MtlSamplerAddressMode> {
    match d3d {
        d3d12_address_mode::WRAP => Some(MtlSamplerAddressMode::Repeat),
        d3d12_address_mode::MIRROR => Some(MtlSamplerAddressMode::MirrorRepeat),
        d3d12_address_mode::CLAMP => Some(MtlSamplerAddressMode::ClampToEdge),
        d3d12_address_mode::BORDER => Some(MtlSamplerAddressMode::ClampToBorderColor),
        d3d12_address_mode::MIRROR_ONCE => Some(MtlSamplerAddressMode::MirrorClampToEdge),
        _ => None,
    }
}

/// Result of a border-color translation. `exact` is false when the D3D12
/// border color did not match one of Metal's three fixed values and was
/// snapped to the nearest one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BorderColorMap {
    pub border_color: MtlSamplerBorderColor,
    /// True when the source color was an exact match; false after snapping.
    pub exact: bool,
}

/// Translate a D3D12 sampler border color (`float[4]`: R, G, B, A) into an
/// `MTLSamplerBorderColor`. Metal only supports transparent black, opaque
/// black and opaque white; anything else snaps to the nearest fixed value.
pub fn map_border_color(rgba: [f32; 4]) -> BorderColorMap {
    let [r, g, b, a] = rgba;
    let is_zero = |x: f32| x == 0.0;
    let is_one = |x: f32| x == 1.0;

    // Exact matches against the three fixed colors first.
    if is_zero(r) && is_zero(g) && is_zero(b) && is_zero(a) {
        return BorderColorMap { border_color: MtlSamplerBorderColor::TransparentBlack, exact: true };
    }
    if is_zero(r) && is_zero(g) && is_zero(b) && is_one(a) {
        return BorderColorMap { border_color: MtlSamplerBorderColor::OpaqueBlack, exact: true };
    }
    if is_one(r) && is_one(g) && is_one(b) && is_one(a) {
        return BorderColorMap { border_color: MtlSamplerBorderColor::OpaqueWhite, exact: true };
    }

    // Non-fixed: snap to the nearest fixed color. RGB near zero → opaque
    // black when alpha is set, otherwise transparent black; bright RGB →
    // opaque white.
    let dark = |x: f32| x <= 0.5;
    let bright = |x: f32| x > 0.5;
    if dark(r) && dark(g) && dark(b) {
        let c = if dark(a) { MtlSamplerBorderColor::TransparentBlack } else { MtlSamplerBorderColor::OpaqueBlack };
        BorderColorMap { border_color: c, exact: false }
    } else if bright(r) && bright(g) && bright(b) {
        BorderColorMap { border_color: MtlSamplerBorderColor::OpaqueWhite, exact: false }
    } else if dark(a) {
        BorderColorMap { border_color: MtlSamplerBorderColor::TransparentBlack, exact: false }
    } else {
        BorderColorMap { border_color: MtlSamplerBorderColor::OpaqueBlack, exact: false }
    }
}

// ===========================================================================
// 4. Resource dimension + flags → texture type / usage
// ===========================================================================

/// Input description for the texture-type translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureDescInput {
    /// `D3D12_RESOURCE_DIMENSION_*`.
    pub dimension: i32,
    /// Array slice count (1 for a non-array texture).
    pub array_size: u32,
    /// True when the texture is a cube map (`D3D12_RESOURCE_MISC_TEXTURECUBE`).
    pub is_cube: bool,
    /// MSAA sample count (1 for a non-multisampled texture).
    pub sample_count: u32,
}

/// Translate the resource dimension / layout into an `MTLTextureType`.
/// Buffers are not textures and return `None` here (they map to MTLBuffer).
pub fn map_texture_type(desc: &TextureDescInput) -> Option<MtlTextureType> {
    match desc.dimension {
        d3d12_resource_dimension::BUFFER => None,
        d3d12_resource_dimension::TEXTURE1D => {
            if desc.array_size > 1 {
                Some(MtlTextureType::Type1DArray)
            } else {
                Some(MtlTextureType::Type1D)
            }
        }
        d3d12_resource_dimension::TEXTURE2D => {
            if desc.sample_count > 1 {
                if desc.array_size > 1 {
                    Some(MtlTextureType::Type2DMultisampleArray)
                } else {
                    Some(MtlTextureType::Type2DMultisample)
                }
            } else if desc.is_cube {
                if desc.array_size > 6 {
                    Some(MtlTextureType::TypeCubeArray)
                } else {
                    Some(MtlTextureType::TypeCube)
                }
            } else if desc.array_size > 1 {
                Some(MtlTextureType::Type2DArray)
            } else {
                Some(MtlTextureType::Type2D)
            }
        }
        d3d12_resource_dimension::TEXTURE3D => Some(MtlTextureType::Type3D),
        _ => None,
    }
}

/// Translate `D3D12_RESOURCE_FLAGS` into an `MTLTextureUsage` bit mask.
///
/// * `ALLOW_RENDER_TARGET` / `ALLOW_DEPTH_STENCIL` → `RenderTarget`
/// * `ALLOW_UNORDERED_ACCESS` → `ShaderWrite`
/// * everything else defaults to `ShaderRead`; `DENY_SHADER_RESOURCE` strips it
///
/// Flags with no Metal usage bit (cross-adapter, simultaneous access, video
/// decode/encode reference, raytracing AS) are ignored here and recorded by
/// the caller if needed.
pub fn map_texture_usage(resource_flags: i32) -> MtlTextureUsage {
    let mut usage = MtlTextureUsage::SHADER_READ;
    if resource_flags & d3d12_resource_flag::ALLOW_RENDER_TARGET != 0
        || resource_flags & d3d12_resource_flag::ALLOW_DEPTH_STENCIL != 0
    {
        usage.0 |= MtlTextureUsage::RENDER_TARGET.0;
    }
    if resource_flags & d3d12_resource_flag::ALLOW_UNORDERED_ACCESS != 0 {
        usage.0 |= MtlTextureUsage::SHADER_WRITE.0;
    }
    if resource_flags & d3d12_resource_flag::DENY_SHADER_RESOURCE != 0 {
        usage.0 &= !MtlTextureUsage::SHADER_READ.0;
    }
    usage
}
