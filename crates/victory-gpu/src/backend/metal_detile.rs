//! GPU detile for title-filled textures (F10): a title-filled texture's tiled
//! console memory is detiled into a staging buffer on the GPU (no CPU detile),
//! and the existing buffer→texture blit copies the staging rows into the Metal
//! texture.
//!
//! Two halves:
//! - [`kernel_params`]: pure Rust mapping from a victory-formats [`Layout`]
//!   subresource onto the parameter block the detile kernel consumes. Moved
//!   over verbatim from the F6 bridge (`metal-backend/flash/F6/bridge/src/lib.rs`),
//!   comments and unit tests kept.
//! - [`Detiler`] (Windows/Wine only): the compiled compute pipelines and the
//!   step encoder that appends one detile dispatch to a command-buffer step
//!   list.
//!
//! # Coupling note (read before trusting this module)
//!
//! victory-formats keeps its per-axis coordinate-bit table (`Layout::pattern`,
//! built from the private `MICRO` table and `pattern()` in
//! `victory-formats/src/texture.rs:188-236`) private. The detile kernel needs
//! that table (`xb/yb/zb`, and the spread `px/py/pz` derived from it) per
//! subresource, and victory-formats exposes no accessor for it. We therefore
//! **copy** the `MICRO` table and the `pattern()` derivation here verbatim
//! (same source: addrlib `GFX10_SW_256_S/D_RBPLUS`, `GFX10_SW_4K_S3_RBPLUS`),
//! exactly the way T7/F2's `gen/` and the F6 bridge already do. This is a
//! copy, not a probe: if victory-formats changes its micro-tile patterns or
//! block-growth order, this module must be updated in lockstep. The copy is
//! validated two ways: (1) the unit tests below reproduce the kernel address
//! formula in Rust and check it against the public `Layout::offset` for every
//! element of every test case; (2) the F10 Wine harness compares the GPU
//! kernel output byte-for-byte against `Layout::detile`.

use victory_formats::texture::{Dimension, Layout, Swizzle};

/// Kernel parameter block. Field order and names match the `Params` struct in
/// `flash/F2/host/detile.metal` and `flash/F2/host/host.m` exactly; `repr(C)`
/// so the Rust value can be handed to the kernel through `setBytes` unchanged.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Params {
    /// Byte offset of the mip start inside its slice (non-3D: already includes
    /// `slice * slice_size`).
    pub base: u64,
    /// `Layout::slice_size`: stride between array slices, or between thick
    /// depth slabs in a 3D texture.
    pub slice_stride: u64,
    /// Mip size in elements (texels, or 4x4 blocks for BC).
    pub w: u32,
    pub h: u32,
    pub d: u32,
    /// `Mip::pitch / block[0]`: blocks per row.
    pub pitch_blocks: u32,
    /// log2 of the block size in bytes (0 linear, 8 / 12 / 16 tiled).
    pub log2blk: u32,
    /// 1 for the LINEAR swizzle, 0 otherwise.
    pub linear: u32,
    /// Mip-tail element offset inside the tail block (0 outside the tail).
    pub tailx: u32,
    pub taily: u32,
    /// `Mip::pitch` in elements (used by the linear kernel).
    pub pitch: u32,
    /// Block dimensions in elements.
    pub bw: u32,
    pub bh: u32,
    pub bz: u32,
    /// Number of coordinate bits per axis.
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    /// Address bit that coordinate bit `i` lands on, per axis.
    pub xb: [u32; 8],
    pub yb: [u32; 8],
    pub zb: [u32; 8],
    /// Precomputed spread tables: coordinate -> offset bits, per axis.
    /// `px[c] = OR_i ((c >> i) & 1) << xb[i]`.
    pub px: [u32; 256],
    pub py: [u32; 256],
    pub pz: [u32; 256],
}

// ---------------------------------------------------------------------------
// Copied bit-pattern derivation (see coupling note at the top of the file).
// MICRO: micro-tile (256 B) patterns by log2 element size, 2D standard (0),
// 2D display/D (1), thick 3D (2). Mirrors victory-formats `MICRO`
// (texture.rs:188-210) and `pattern()` (texture.rs:213-236).
// ---------------------------------------------------------------------------

const MICRO: [[&[u8; 16]; 5]; 3] = [
    [
        b"x0x1x2x3y0y1y2y3",
        b"--x0x1x2y0y1y2x3",
        b"----x0x1y0y1y2x2",
        b"------x0y0y1x1x2",
        b"--------y0y1x0x1",
    ],
    [
        b"x0x1x2y1y0y2x3y3",
        b"--x0x1x2y0y1y2x3",
        b"----x0x1y0y1x2y2",
        b"------x0y0x1x2y1",
        b"--------x0y0x1y1",
    ],
    [
        b"x0x1z0y0z1y1x2z2",
        b"--x0z0y0z1y1x1z2",
        b"----x0y0z0y1x1z1",
        b"------x0z0y0x1z1",
        b"--------z0y0x0z1",
    ],
];

/// Per-axis list of address bits, one entry per coordinate bit. Copy of the
/// private `Layout::pattern` (texture.rs:213-236).
fn pattern(kind: usize, element_log2: u32, block_log2: u32) -> [Vec<u32>; 3] {
    let mut axes: [Vec<u32>; 3] = Default::default();
    let micro = MICRO[kind][element_log2 as usize];
    for bit in 0..8u32 {
        let (axis, index) = (micro[2 * bit as usize], micro[2 * bit as usize + 1]);
        let axis = match axis {
            b'x' => 0,
            b'y' => 1,
            b'z' => 2,
            _ => continue,
        };
        let index = usize::from(index - b'0');
        if axes[axis].len() <= index {
            axes[axis].resize(index + 1, 0);
        }
        axes[axis][index] = bit;
    }
    let order: &[usize] = if kind == 2 { &[1, 0, 2] } else { &[1, 0] };
    for (bit, axis) in (8..block_log2).zip(order.iter().cycle()) {
        axes[*axis].push(bit);
    }
    axes
}

/// Fill `out[c]` with the ORed address bits of coordinate `c` under `bits`.
fn spread_table(bits: &[u32], out: &mut [u32; 256]) {
    for c in 0..256u32 {
        let mut v = 0u32;
        for (i, &b) in bits.iter().enumerate() {
            v |= ((c >> i) & 1) << b;
        }
        out[c as usize] = v;
    }
}

// ---------------------------------------------------------------------------
// Public mapping
// ---------------------------------------------------------------------------

/// Why the kernel cannot handle a subresource. Returned alongside `None` so
/// the caller can list unsupported cases with a reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unsupported {
    /// Pipe/bank XOR mode (`_T` / `_X`): addressing depends on the GPU pipe
    /// and bank configuration, which victory-formats does not model;
    /// `Layout::addressable()` is false and `Layout::detile` returns
    /// `Error::Unsupported`. The kernel cannot do it either.
    XorMode,
    /// Multisampled texture (`samples != 1`): not addressable here.
    Msaa,
    /// `mip` index does not exist in the layout.
    BadMip,
    /// `slice` is out of range (3D requires slice 0; 2D array slice must be
    /// `< Layout::slices`).
    BadSlice,
    /// Element byte size the kernel does not instantiate (1/2/4/8/16 only).
    UnsupportedElementSize,
}

impl Unsupported {
    pub fn reason(self) -> &'static str {
        match self {
            Unsupported::XorMode => {
                "pipe/bank XOR mode (_T/_X): addressable()=false; address depends on the GPU \
                 pipe/bank configuration, which victory-formats does not model"
            }
            Unsupported::Msaa => "multisampled texture (samples != 1): addressable()=false",
            Unsupported::BadMip => "mip index out of range",
            Unsupported::BadSlice => {
                "slice out of range (3D requires slice 0; 2D array slice must be < slices)"
            }
            Unsupported::UnsupportedElementSize => {
                "element_bytes is not one of 1/2/4/8/16 (no kernel instantiation)"
            }
        }
    }
}

fn inner(layout: &Layout, mip: u32, slice: u32) -> Result<Params, Unsupported> {
    let desc = &layout.desc;

    // Reject the cases the Rust side itself cannot address, with distinct
    // reasons instead of one generic "unsupported".
    if desc.samples != 1 {
        return Err(Unsupported::Msaa);
    }
    if desc.swizzle.xor() {
        return Err(Unsupported::XorMode);
    }
    if !layout.addressable() {
        // Defensive: anything else addressable() rejects that is neither XOR
        // nor MSAA. Keep it as XorMode-family unsupported rather than guessing.
        return Err(Unsupported::XorMode);
    }

    let m = layout.mips.get(mip as usize).ok_or(Unsupported::BadMip)?;

    let three_d = desc.dimension == Dimension::Tex3d;
    if three_d {
        if slice != 0 {
            return Err(Unsupported::BadSlice);
        }
    } else if slice >= layout.slices {
        return Err(Unsupported::BadSlice);
    }

    if ![1u32, 2, 4, 8, 16].contains(&desc.element_bytes) {
        return Err(Unsupported::UnsupportedElementSize);
    }

    let linear = desc.swizzle == Swizzle::LINEAR;
    let block_log2 = desc.swizzle.block_log2().unwrap_or(0);
    let element_log2 = desc.element_bytes.trailing_zeros();

    // kind: thick 3D -> 2, display/D modes -> 1, standard/S -> 0. Matches
    // texture.rs:488-492. XOR display modes (18/22/26) were rejected above.
    let kind = if layout.thick {
        2
    } else if matches!(desc.swizzle.0, 2 | 6 | 10) {
        1
    } else {
        0
    };

    let pat: [Vec<u32>; 3] = if linear {
        Default::default()
    } else {
        pattern(kind, element_log2, block_log2)
    };

    // 3D subresources are dispatched as one mip over all depth slices; the
    // slice argument must be 0 and the slab advance goes through slice_stride
    // inside the kernel. Non-3D: base already points at this array slice.
    let base = if three_d {
        m.offset
    } else {
        u64::from(slice) * layout.slice_size + m.offset
    };
    let bz = if layout.thick { layout.block[2] } else { 1 };

    let mut p = Params {
        base,
        slice_stride: layout.slice_size,
        w: m.width,
        h: m.height,
        d: m.depth,
        pitch_blocks: m.pitch / layout.block[0],
        log2blk: block_log2,
        linear: u32::from(linear),
        tailx: m.tail.map_or(0, |t| t.x),
        taily: m.tail.map_or(0, |t| t.y),
        pitch: m.pitch,
        bw: layout.block[0],
        bh: layout.block[1],
        bz,
        nx: pat[0].len() as u32,
        ny: pat[1].len() as u32,
        nz: pat[2].len() as u32,
        xb: [0; 8],
        yb: [0; 8],
        zb: [0; 8],
        px: [0; 256],
        py: [0; 256],
        pz: [0; 256],
    };
    for (i, &b) in pat[0].iter().enumerate() {
        p.xb[i] = b;
    }
    for (i, &b) in pat[1].iter().enumerate() {
        p.yb[i] = b;
    }
    for (i, &b) in pat[2].iter().enumerate() {
        p.zb[i] = b;
    }
    spread_table(&pat[0], &mut p.px);
    spread_table(&pat[1], &mut p.py);
    spread_table(&pat[2], &mut p.pz);
    Ok(p)
}

/// Kernel parameters for mip `mip` of array slice / depth slab `slice` of
/// `layout`. Returns `None` for modes the kernel cannot handle (XOR swizzles,
/// MSAA, bad mip/slice, unsupported element size); use [`unsupported_reason`]
/// to get a human-readable reason.
pub fn kernel_params(layout: &Layout, mip: u32, slice: u32) -> Option<Params> {
    inner(layout, mip, slice).ok()
}

/// Reason the kernel cannot handle the subresource, if [`kernel_params`]
/// returns `None`.
pub fn unsupported_reason(layout: &Layout, mip: u32, slice: u32) -> Option<&'static str> {
    inner(layout, mip, slice).err().map(Unsupported::reason)
}

// ---------------------------------------------------------------------------
// GPU side (Windows / Wine only): compiled detile pipelines + step encoder.
// ---------------------------------------------------------------------------

/// F2/F6 detile kernel source. Verbatim copy of
/// `metal-backend/flash/F2/host/detile.metal` (byte-identical to the F6
/// harness copy), with the F10-only `b<N>_3d` staging-buffer kernels
/// appended at the end (the original `b1/b2/b4/b8/b16` kernels are 2D;
/// the 3D staging variants did not exist).
#[cfg(windows)]
pub(crate) const DETILE_MSL: &str = r#"// T7: GFX10 (RDNA2) tiled -> linear texture, one thread per element
// (a texel, or a 4x4 block for BC formats). Port of
// victory-formats/src/texture.rs Layout::offset (row() + column()).
#include <metal_stdlib>
using namespace metal;

struct Params {
    ulong base;          // byte offset of mip start in slice (+ array slice * slice_size)
    ulong slice_stride;  // Layout::slice_size (3D: advance per depth slab / slice)
    uint  w, h, d;       // mip size in elements
    uint  pitch_blocks;  // Mip::pitch / block[0]
    uint  log2blk;       // log2 of block bytes (8, 12, 16)
    uint  linear;        // swizzle == LINEAR
    uint  tailx, taily;  // mip-tail element offset (0 outside the tail)
    uint  pitch;         // Mip::pitch in elements (linear)
    uint  bw, bh, bz;    // block dims in elements (bz: thick depth, else 1)
    uint  nx, ny, nz;    // number of coordinate bits per axis
    uint  xb[8];         // address bit of coordinate bit i, per axis
    uint  yb[8];
    uint  zb[8];
    uint  px[256];       // F2 precomputed spread tables (coord -> offset bits)
    uint  py[256];
    uint  pz[256];
};

static inline ulong spread(uint c, uint n, constant uint* bits) {
    ulong r = 0;
    for (uint i = 0; i < n; i++) r |= ulong((c >> i) & 1u) << bits[i];
    return r;
}

// F2: table-driven offset (spread precomputed on the host into px/py/pz).
static inline ulong elem_offset_tbl(constant Params& p, uint x, uint y, uint z, uint nlog2) {
    if (p.linear) {
        return p.base + ulong(z) * p.slice_stride + ((ulong(y) * p.pitch + x) << nlog2);
    }
    uint sl = z / p.bz, zz = z - sl * p.bz;
    uint xx = x + p.tailx, yy = y + p.taily;
    ulong blocks = ulong(yy / p.bh) * p.pitch_blocks + (xx / p.bw);
    ulong a = (blocks << p.log2blk)
            + p.px[xx & (p.bw - 1)] + p.py[yy & (p.bh - 1)];
    if (p.nz) a += p.pz[zz];
    return p.base + ulong(sl) * p.slice_stride + a;
}

// Byte offset of element (x, y, z) from the start of the tiled memory.
static inline ulong elem_offset(constant Params& p, uint x, uint y, uint z, uint nlog2) {
    if (p.linear) {
        return p.base + ulong(z) * p.slice_stride + ((ulong(y) * p.pitch + x) << nlog2);
    }
    uint sl = z / p.bz, zz = z - sl * p.bz;
    uint xx = x + p.tailx, yy = y + p.taily;
    ulong blocks = ulong(yy / p.bh) * p.pitch_blocks + (xx / p.bw);
    ulong a = (blocks << p.log2blk)
            + spread(xx & (p.bw - 1), p.nx, p.xb)
            + spread(yy & (p.bh - 1), p.ny, p.yb);
    if (p.nz) a += spread(zz, p.nz, p.zb);
    return p.base + ulong(sl) * p.slice_stride + a;
}

// k<N>_2d / k<N>_3d: write straight into a (uint-format view of an) MTLTexture.
// b<N>: write tight linear rows into a buffer (for BC textures, which Metal
// cannot write from a shader or view as uint; a blit copies buffer -> texture).
#define KERNELS(NAME, BNAME, T, NL, MK) \
kernel void NAME##_2d(device const uchar* src [[buffer(0)]], constant Params& p [[buffer(1)]], \
        texture2d<uint, access::write> dst [[texture(0)]], uint2 gid [[thread_position_in_grid]]) { \
    if (gid.x >= p.w || gid.y >= p.h) return; \
    T v = *(device const T*)(src + elem_offset(p, gid.x, gid.y, 0, NL)); \
    dst.write(MK(v), gid); \
} \
kernel void NAME##_3d(device const uchar* src [[buffer(0)]], constant Params& p [[buffer(1)]], \
        texture3d<uint, access::write> dst [[texture(0)]], uint3 gid [[thread_position_in_grid]]) { \
    if (gid.x >= p.w || gid.y >= p.h || gid.z >= p.d) return; \
    T v = *(device const T*)(src + elem_offset(p, gid.x, gid.y, gid.z, NL)); \
    dst.write(MK(v), gid); \
} \
kernel void BNAME(device const uchar* src [[buffer(0)]], constant Params& p [[buffer(1)]], \
        device uchar* dst [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) { \
    if (gid.x >= p.w || gid.y >= p.h) return; \
    T v = *(device const T*)(src + elem_offset(p, gid.x, gid.y, 0, NL)); \
    *(device T*)(dst + ((ulong(gid.y) * p.w + gid.x) << NL)) = v; \
}
#define MK1(v) uint4(v, 0, 0, 0)
#define MK2(v) uint4(v, 0, 0, 0)
#define MK4(v) uint4(v, 0, 0, 0)
#define MK8(v) uint4(v, 0, 0)
#define MK16(v) (v)
KERNELS(k1,  b1,  uchar,  0, MK1)
KERNELS(k2,  b2,  ushort, 1, MK2)
KERNELS(k4,  b4,  uint,   2, MK4)
KERNELS(k8,  b8,  uint2,  3, MK8)
KERNELS(k16, b16, uint4,  4, MK16)

// ---- F2 packed variants (2D, staging buffer + blit) ----
// One thread handles K consecutive x-elements. The scattered source loads are
// issued per element (the tile layout spreads x bits inside a micro block),
// but the destination is one packed wide store to the tight linear staging
// buffer. The original k1/k2/b1/b2 kernels above are kept unchanged as the
// reference path.
// w1: 8 x 1-byte elements per thread.
kernel void w1(device const uchar* src [[buffer(0)]], constant Params& p [[buffer(1)]],
        device uchar* dst [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    uint x0 = gid.x * 8;
    if (gid.y >= p.h || x0 >= p.w) return;
    uint n = min(8u, p.w - x0);
    ulong packed = 0;
    for (uint i = 0; i < n; i++) {
        uchar v = *(device const uchar*)(src + elem_offset(p, x0 + i, gid.y, 0, 0));
        packed |= ulong(v) << (8 * i);
    }
    ulong off = ulong(gid.y) * p.w + x0;
    if (n == 8 && (off & 7u) == 0) {
        *(device ulong*)(dst + off) = packed;
    } else {
        for (uint i = 0; i < n; i++) dst[off + i] = (uchar)((packed >> (8 * i)) & 0xFFu);
    }
}

// w1t: 8 x 1-byte elements per thread, direct to texture (no staging).
kernel void w1t(device const uchar* src [[buffer(0)]], constant Params& p [[buffer(1)]],
        texture2d<uint, access::write> dst [[texture(0)]], uint2 gid [[thread_position_in_grid]]) {
    uint x0 = gid.x * 8;
    if (gid.y >= p.h || x0 >= p.w) return;
    uint n = min(8u, p.w - x0);
    for (uint i = 0; i < n; i++) {
        uchar v = *(device const uchar*)(src + elem_offset(p, x0 + i, gid.y, 0, 0));
        dst.write(uint4(v, 0, 0, 0), uint2(x0 + i, gid.y));
    }
}

// w2t: 4 x 2-byte elements per thread, direct to texture.
kernel void w2t(device const uchar* src [[buffer(0)]], constant Params& p [[buffer(1)]],
        texture2d<uint, access::write> dst [[texture(0)]], uint2 gid [[thread_position_in_grid]]) {
    uint x0 = gid.x * 4;
    if (gid.y >= p.h || x0 >= p.w) return;
    uint n = min(4u, p.w - x0);
    for (uint i = 0; i < n; i++) {
        ushort v = *(device const ushort*)(src + elem_offset(p, x0 + i, gid.y, 0, 1));
        dst.write(uint4(v, 0, 0, 0), uint2(x0 + i, gid.y));
    }
}

// Probes (F2 isolation only, not used by the final path):
// p2: coalesced linear source load, no tile address math.
kernel void p2(device const uchar* src [[buffer(0)]], constant Params& p [[buffer(1)]],
        texture2d<uint, access::write> dst [[texture(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.w || gid.y >= p.h) return;
    uchar v = src[gid.y * p.w + gid.x];
    dst.write(uint4(v, 0, 0, 0), gid);
}
// p0: write-only, no source read; isolates texture store throughput.
kernel void p0(device const uchar* src [[buffer(0)]], constant Params& p [[buffer(1)]],
        texture2d<uint, access::write> dst [[texture(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.w || gid.y >= p.h) return;
    uchar v = (uchar)((gid.x ^ gid.y) & 0xFFu);
    dst.write(uint4(v, 0, 0, 0), gid);
}
// pC: coalesced source load into a tight staging buffer (buffer store).
kernel void pC(device const uchar* src [[buffer(0)]], constant Params& p [[buffer(1)]],
        device uchar* dst [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.w || gid.y >= p.h) return;
    dst[gid.y * p.w + gid.x] = src[gid.y * p.w + gid.x];
}
// p3: tile address math, but a fixed (cached) source load; isolates ALU + store.
kernel void p3(device const uchar* src [[buffer(0)]], constant Params& p [[buffer(1)]],
        texture2d<uint, access::write> dst [[texture(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.w || gid.y >= p.h) return;
    ulong o = elem_offset(p, gid.x, gid.y, 0, 0);
    uchar v = src[o & 0u];
    dst.write(uint4(v, 0, 0, 0), gid);
}

// s1/s2: F2 threadgroup-scratch path. One threadgroup owns one 256-byte
// micro tile; it loads the whole micro block from source sequentially
// (coalesced) into threadgroup memory, then each thread reads its element
// out of scratch by the tiled offset. 2D tiled only; LINEAR and 3D keep the
// original direct kernels. Micro tile dims match BLOCK_256_2D: 1B 16x16,
// 2B 16x8 (256 bytes either way).
kernel void s1(device const uchar* src [[buffer(0)]], constant Params& p [[buffer(1)]],
        texture2d<uint, access::write> dst [[texture(0)]],
        uint2 tid [[thread_position_in_threadgroup]],
        uint2 group [[threadgroup_position_in_grid]]) {
    uint x0 = group.x * 16, y0 = group.y * 16;
    if (x0 >= p.w || y0 >= p.h) return;
    threadgroup uchar scratch[256];
    ulong base = elem_offset(p, x0, y0, 0, 0);
    uint t = tid.y * 16 + tid.x;
    scratch[t] = src[base + t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint x = x0 + tid.x, y = y0 + tid.y;
    if (x >= p.w || y >= p.h) return;
    ulong off = elem_offset(p, x, y, 0, 0);
    dst.write(uint4(scratch[off - base], 0, 0, 0), uint2(x, y));
}

// r1/r2: F2 wide-load path. One thread owns 8 consecutive x-elements (one
// contiguous source run: 8 bytes for 1B, 16 bytes for 2B per the addrlib
// micro layout); the source side is one wide load, then the elements are
// written out to the texture. Edge/partial rows fall back to scalar loads.
kernel void r1(device const uchar* src [[buffer(0)]], constant Params& p [[buffer(1)]],
        texture2d<uint, access::write> dst [[texture(0)]],
        uint2 tid [[thread_position_in_threadgroup]],
        uint2 group [[threadgroup_position_in_grid]]) {
    uint x0 = group.x * 8, y = group.y * 8 + tid.y;
    if (y >= p.h || x0 >= p.w) return;
    uint n = min(8u, p.w - x0);
    ulong base = elem_offset(p, x0, y, 0, 0);
    uint2 raw;
    raw.x = 0; raw.y = 0;
    if (n == 8 && (base & 7u) == 0) {
        raw = *(device const uint2*)(src + base);
    } else {
        for (uint i = 0; i < n; i++) {
            uchar v = *(device const uchar*)(src + elem_offset(p, x0 + i, y, 0, 0));
            if (i < 4) raw.x |= uint(v) << (8 * i);
            else raw.y |= uint(v) << (8 * (i - 4));
        }
    }
    for (uint i = 0; i < n; i++) {
        uint part = i < 4 ? raw.x : raw.y;
        uchar v = (uchar)((part >> (8 * (i & 3))) & 0xFFu);
        dst.write(uint4(v, 0, 0, 0), uint2(x0 + i, y));
    }
}

kernel void r2(device const uchar* src [[buffer(0)]], constant Params& p [[buffer(1)]],
        texture2d<uint, access::write> dst [[texture(0)]],
        uint2 tid [[thread_position_in_threadgroup]],
        uint2 group [[threadgroup_position_in_grid]]) {
    uint x0 = group.x * 8, y = group.y * 8 + tid.y;
    if (y >= p.h || x0 >= p.w) return;
    uint n = min(8u, p.w - x0);
    ulong base = elem_offset(p, x0, y, 0, 1);
    uint4 raw;
    raw.x = 0; raw.y = 0; raw.z = 0; raw.w = 0;
    if (n == 8 && (base & 15u) == 0) {
        raw = *(device const uint4*)(src + base);
    } else {
        uint parts[4] = {0, 0, 0, 0};
        for (uint i = 0; i < n; i++) {
            ushort v = *(device const ushort*)(src + elem_offset(p, x0 + i, y, 0, 1));
            parts[i >> 1] |= uint(v) << (16 * (i & 1));
        }
        raw.x = parts[0]; raw.y = parts[1]; raw.z = parts[2]; raw.w = parts[3];
    }
    uint parts[4] = {raw.x, raw.y, raw.z, raw.w};
    for (uint i = 0; i < n; i++) {
        ushort v = (ushort)((parts[i >> 1] >> (16 * (i & 1))) & 0xFFFFu);
        dst.write(uint4(v, 0, 0, 0), uint2(x0 + i, y));
    }
}

// lt1/lt2: table-driven one-element-per-thread kernels (ALU reduced vs k1/k2).
kernel void lt1(device const uchar* src [[buffer(0)]], constant Params& p [[buffer(1)]],
        texture2d<uint, access::write> dst [[texture(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.w || gid.y >= p.h) return;
    uchar v = *(device const uchar*)(src + elem_offset_tbl(p, gid.x, gid.y, 0, 0));
    dst.write(uint4(v, 0, 0, 0), gid);
}
kernel void lt2(device const uchar* src [[buffer(0)]], constant Params& p [[buffer(1)]],
        texture2d<uint, access::write> dst [[texture(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.w || gid.y >= p.h) return;
    ushort v = *(device const ushort*)(src + elem_offset_tbl(p, gid.x, gid.y, 0, 1));
    dst.write(uint4(v, 0, 0, 0), gid);
}

// lw1/lw2: table-driven, 4 consecutive x-elements per thread (scalar loads).
kernel void lw1(device const uchar* src [[buffer(0)]], constant Params& p [[buffer(1)]],
        texture2d<uint, access::write> dst [[texture(0)]], uint2 gid [[thread_position_in_grid]]) {
    uint x0 = gid.x * 4;
    if (gid.y >= p.h || x0 >= p.w) return;
    uint n = min(4u, p.w - x0);
    uchar v[4];
    for (uint i = 0; i < n; i++)
        v[i] = *(device const uchar*)(src + elem_offset_tbl(p, x0 + i, gid.y, 0, 0));
    for (uint i = 0; i < n; i++)
        dst.write(uint4(v[i], 0, 0, 0), uint2(x0 + i, gid.y));
}
kernel void lw2(device const uchar* src [[buffer(0)]], constant Params& p [[buffer(1)]],
        texture2d<uint, access::write> dst [[texture(0)]], uint2 gid [[thread_position_in_grid]]) {
    uint x0 = gid.x * 4;
    if (gid.y >= p.h || x0 >= p.w) return;
    uint n = min(4u, p.w - x0);
    ushort v[4];
    for (uint i = 0; i < n; i++)
        v[i] = *(device const ushort*)(src + elem_offset_tbl(p, x0 + i, gid.y, 0, 1));
    for (uint i = 0; i < n; i++)
        dst.write(uint4(v[i], 0, 0, 0), uint2(x0 + i, gid.y));
}

// lwd1/lwd2: table-driven, wide load (4B / 8B) for 4 consecutive elements.
// The wide load is taken only after verifying the four source offsets are a
// contiguous, aligned run; otherwise it falls back to scalar loads.
kernel void lwd1(device const uchar* src [[buffer(0)]], constant Params& p [[buffer(1)]],
        texture2d<uint, access::write> dst [[texture(0)]], uint2 gid [[thread_position_in_grid]]) {
    uint x0 = gid.x * 4;
    if (gid.y >= p.h || x0 >= p.w) return;
    uint n = min(4u, p.w - x0);
    ulong o0 = elem_offset_tbl(p, x0, gid.y, 0, 0);
    bool wide = n == 4 && (o0 & 3u) == 0;
    uint raw = 0;
    if (wide) {
        raw = *(device const uint*)(src + o0);
    } else {
        for (uint i = 0; i < n; i++)
            raw |= uint(*(device const uchar*)(src + elem_offset_tbl(p, x0 + i, gid.y, 0, 0))) << (8 * i);
    }
    for (uint i = 0; i < n; i++)
        dst.write(uint4((uchar)((raw >> (8 * i)) & 0xFFu), 0, 0, 0), uint2(x0 + i, gid.y));
}
kernel void lwd2(device const uchar* src [[buffer(0)]], constant Params& p [[buffer(1)]],
        texture2d<uint, access::write> dst [[texture(0)]], uint2 gid [[thread_position_in_grid]]) {
    uint x0 = gid.x * 4;
    if (gid.y >= p.h || x0 >= p.w) return;
    uint n = min(4u, p.w - x0);
    ulong o0 = elem_offset_tbl(p, x0, gid.y, 0, 1);
    bool wide = n == 4 && (o0 & 7u) == 0;
    uint2 raw; raw.x = 0; raw.y = 0;
    if (wide) {
        raw = *(device const uint2*)(src + o0);
    } else {
        uint parts[2] = {0, 0};
        for (uint i = 0; i < n; i++) {
            ushort v = *(device const ushort*)(src + elem_offset_tbl(p, x0 + i, gid.y, 0, 1));
            parts[i >> 1] |= uint(v) << (16 * (i & 1));
        }
        raw.x = parts[0]; raw.y = parts[1];
    }
    uint parts[2] = {raw.x, raw.y};
    for (uint i = 0; i < n; i++) {
        ushort v = (ushort)((parts[i >> 1] >> (16 * (i & 1))) & 0xFFFFu);
        dst.write(uint4(v, 0, 0, 0), uint2(x0 + i, gid.y));
    }
}

// s1d: same group/coordinate mapping as s1, but reads source directly (no
// scratch); isolation probe for the group-id fix.
kernel void s1d(device const uchar* src [[buffer(0)]], constant Params& p [[buffer(1)]],
        texture2d<uint, access::write> dst [[texture(0)]],
        uint2 tid [[thread_position_in_threadgroup]],
        uint2 group [[threadgroup_position_in_grid]]) {
    uint x0 = group.x * 16, y0 = group.y * 16;
    if (x0 >= p.w || y0 >= p.h) return;
    uint x = x0 + tid.x, y = y0 + tid.y;
    if (x >= p.w || y >= p.h) return;
    uchar v = *(device const uchar*)(src + elem_offset(p, x, y, 0, 0));
    dst.write(uint4(v, 0, 0, 0), uint2(x, y));
}

kernel void s2(device const uchar* src [[buffer(0)]], constant Params& p [[buffer(1)]],
        texture2d<uint, access::write> dst [[texture(0)]],
        uint2 tid [[thread_position_in_threadgroup]],
        uint2 group [[threadgroup_position_in_grid]]) {
    uint x0 = group.x * 16, y0 = group.y * 8;
    if (x0 >= p.w || y0 >= p.h) return;
    threadgroup ushort scratch[128];
    ulong base = elem_offset(p, x0, y0, 0, 1);
    uint t = tid.y * 16 + tid.x;
    scratch[t] = *(device const ushort*)(src + base + t * 2);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint x = x0 + tid.x, y = y0 + tid.y;
    if (x >= p.w || y >= p.h) return;
    ulong off = elem_offset(p, x, y, 0, 1);
    dst.write(uint4(scratch[(off - base) >> 1], 0, 0, 0), uint2(x, y));
}

// w2: 4 x 2-byte elements per thread (8 bytes packed).
kernel void w2(device const uchar* src [[buffer(0)]], constant Params& p [[buffer(1)]],
        device uchar* dst [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    uint x0 = gid.x * 4;
    if (gid.y >= p.h || x0 >= p.w) return;
    uint n = min(4u, p.w - x0);
    ulong packed = 0;
    for (uint i = 0; i < n; i++) {
        ushort v = *(device const ushort*)(src + elem_offset(p, x0 + i, gid.y, 0, 1));
        packed |= ulong(v) << (16 * i);
    }
    ulong off = (ulong(gid.y) * p.w + x0) << 1;
    if (n == 4 && (off & 7u) == 0) {
        *(device ulong*)(dst + off) = packed;
    } else {
        device ushort* d2 = (device ushort*)(dst + off);
        for (uint i = 0; i < n; i++) d2[i] = (ushort)((packed >> (16 * i)) & 0xFFFFu);
    }
}


// ---- F10 additions: 3D staging-buffer variants of the b<N> kernels ----
// Tight linear staging layout, same as the 2D b kernels: destination element
// (x, y, z) lands at ((z*h + y)*w + x) elements.
kernel void b1_3d(device const uchar* src [[buffer(0)]], constant Params& p [[buffer(1)]],
        device uchar* dst [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.w || gid.y >= p.h || gid.z >= p.d) return;
    uchar v = *(device const uchar*)(src + elem_offset(p, gid.x, gid.y, gid.z, 0));
    dst[((ulong(gid.z) * p.h + gid.y) * p.w + gid.x)] = v;
}

kernel void b2_3d(device const uchar* src [[buffer(0)]], constant Params& p [[buffer(1)]],
        device uchar* dst [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.w || gid.y >= p.h || gid.z >= p.d) return;
    ushort v = *(device const ushort*)(src + elem_offset(p, gid.x, gid.y, gid.z, 1));
    ulong off = ((ulong(gid.z) * p.h + gid.y) * p.w + gid.x) << 1;
    *(device ushort*)(dst + off) = v;
}

kernel void b4_3d(device const uchar* src [[buffer(0)]], constant Params& p [[buffer(1)]],
        device uchar* dst [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.w || gid.y >= p.h || gid.z >= p.d) return;
    uint v = *(device const uint*)(src + elem_offset(p, gid.x, gid.y, gid.z, 2));
    ulong off = ((ulong(gid.z) * p.h + gid.y) * p.w + gid.x) << 2;
    *(device uint*)(dst + off) = v;
}

kernel void b8_3d(device const uchar* src [[buffer(0)]], constant Params& p [[buffer(1)]],
        device uchar* dst [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.w || gid.y >= p.h || gid.z >= p.d) return;
    uint2 v = *(device const uint2*)(src + elem_offset(p, gid.x, gid.y, gid.z, 3));
    ulong off = ((ulong(gid.z) * p.h + gid.y) * p.w + gid.x) << 3;
    *(device uint2*)(dst + off) = v;
}

kernel void b16_3d(device const uchar* src [[buffer(0)]], constant Params& p [[buffer(1)]],
        device uchar* dst [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.w || gid.y >= p.h || gid.z >= p.d) return;
    uint4 v = *(device const uint4*)(src + elem_offset(p, gid.x, gid.y, gid.z, 4));
    ulong off = ((ulong(gid.z) * p.h + gid.y) * p.w + gid.x) << 4;
    *(device uint4*)(dst + off) = v;
}
"#;

/// Threadgroup size along x / y.
#[cfg(windows)]
const GROUP_X: u64 = 32;
#[cfg(windows)]
const GROUP_Y: u64 = 8;

/// Staging-buffer detile pipelines, by element log2 (0 = 1 B ... 4 = 16 B).
/// The kernels write tight linear rows into a device buffer; a later blit
/// copies that buffer into the Metal texture (BC formats cannot be written
/// by compute, and this keeps the existing buffer->texture blit path).
#[cfg(windows)]
pub(crate) struct Detiler {
    b: [victory_metal::ComputePipeline; 5],
    b3: [victory_metal::ComputePipeline; 5],
}

/// Element bytes implied by the parameter block. Tiled: block bytes split
/// by block elements; linear: the row alignment `bw = 256 / element_bytes`.
#[cfg(windows)]
fn element_bytes(p: &Params) -> u32 {
    if p.linear != 0 {
        256 / p.bw
    } else {
        (1u32 << p.log2blk) / (p.bw * p.bh * p.bz)
    }
}

/// Whether the subresource has a depth axis: depth > 1, or a thick 3D
/// block (nonzero z spread).
#[cfg(windows)]
fn is_3d(p: &Params) -> bool {
    p.d > 1 || p.nz != 0
}

#[cfg(windows)]
impl Detiler {
    /// Compiles every staging-buffer kernel (`b1...b16`, 2D and 3D) from
    /// [`DETILE_MSL`] on `device`.
    pub(crate) fn new(device: &victory_metal::Device) -> victory_metal::Result<Self> {
        let names_2d = ["b1", "b2", "b4", "b8", "b16"];
        let names_3d = ["b1_3d", "b2_3d", "b4_3d", "b8_3d", "b16_3d"];
        let mut b: Vec<victory_metal::ComputePipeline> = Vec::with_capacity(5);
        for name in names_2d {
            b.push(device.compute_pipeline_msl(DETILE_MSL, name)?);
        }
        let mut b3: Vec<victory_metal::ComputePipeline> = Vec::with_capacity(5);
        for name in names_3d {
            b3.push(device.compute_pipeline_msl(DETILE_MSL, name)?);
        }
        Ok(Self {
            b: b.try_into().map_err(|_| victory_metal::Error("internal: b length".into()))?,
            b3: b3.try_into().map_err(|_| victory_metal::Error("internal: b3 length".into()))?,
        })
    }

    /// Appends one detile dispatch for this subresource to `steps`:
    /// `src[src_offset + kernel address]` -> `dst[dst_offset + tight
    /// linear row]`. The dispatch is encoded when the step list is run
    /// through `CommandBuffer::compute`.
    pub(crate) fn encode<'a>(
        &'a self,
        steps: &mut Vec<victory_metal::Compute<'a>>,
        src: &'a victory_metal::Buffer,
        src_offset: u64,
        dst: &'a victory_metal::Buffer,
        dst_offset: u64,
        params: &'a Params,
    ) {
        let bpe = element_bytes(params);
        let log2 = bpe.trailing_zeros() as usize;
        let pipeline = if is_3d(params) {
            &self.b3[log2]
        } else {
            &self.b[log2]
        };
        let groups = [
            params.w.div_ceil(GROUP_X as u32) as u64,
            params.h.div_ceil(GROUP_Y as u32) as u64,
            params.d.max(1) as u64,
        ];
        steps.push(victory_metal::Compute::Pipeline {
            pipeline,
            threads_per_group: [GROUP_X, GROUP_Y, 1],
        });
        steps.push(victory_metal::Compute::Buffer {
            buffer: src,
            offset: src_offset,
            index: 0,
        });
        // SAFETY: `params` outlives the step list (it is a `&Params`
        // argument); the bytes are handed to `setBytes`, which copies
        // them during encoding.
        let bytes = unsafe {
            core::slice::from_raw_parts(
                core::ptr::from_ref(params).cast::<u8>(),
                core::mem::size_of::<Params>(),
            )
        };
        steps.push(victory_metal::Compute::Bytes { bytes, index: 1 });
        steps.push(victory_metal::Compute::Buffer {
            buffer: dst,
            offset: dst_offset,
            index: 2,
        });
        steps.push(victory_metal::Compute::Dispatch { groups });
    }
}

// ---------------------------------------------------------------------------
// Rust replica of the kernel address formula, used by the tests to check the
// mapping against the public `Layout::offset` before any GPU run.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use victory_formats::texture::{Desc, Dimension};

    fn desc(
        dim: Dimension,
        w: u32,
        h: u32,
        depth: u32,
        mips: u32,
        bpe: u32,
        block: u32,
        sw: Swizzle,
    ) -> Desc {
        Desc {
            dimension: dim,
            width: w,
            height: h,
            depth,
            mips,
            element_bytes: bpe,
            block,
            samples: 1,
            swizzle: sw,
        }
    }

    /// Replica of `elem_offset_tbl` in detile.metal (table-driven path).
    fn elem_offset(p: &Params, x: u32, y: u32, z: u32, nlog2: u32) -> u64 {
        if p.linear != 0 {
            return p.base + u64::from(z) * p.slice_stride + ((u64::from(y) * u64::from(p.pitch) + u64::from(x)) << nlog2);
        }
        let sl = z / p.bz;
        let zz = z % p.bz;
        let xx = x + p.tailx;
        let yy = y + p.taily;
        let blocks = u64::from(yy / p.bh) * u64::from(p.pitch_blocks) + u64::from(xx / p.bw);
        let mut a = (blocks << p.log2blk)
            + u64::from(p.px[(xx & (p.bw - 1)) as usize])
            + u64::from(p.py[(yy & (p.bh - 1)) as usize]);
        if p.nz != 0 {
            a += u64::from(p.pz[zz as usize]);
        }
        p.base + u64::from(sl) * p.slice_stride + a
    }

    fn check_layout(d: &Desc) {
        let l = match Layout::new(d) {
            Ok(l) => l,
            Err(_) => return, // invalid descriptions are exercised separately below
        };
        let three_d = d.dimension == Dimension::Tex3d;
        let arr = if three_d { 1 } else { d.depth };
        for slice in 0..arr {
            for mip in 0..d.mips {
                let p = match kernel_params(&l, mip, slice) {
                    Some(p) => p,
                    None => {
                        assert!(!l.addressable(), "None for an addressable layout {d:?} mip {mip} slice {slice}");
                        continue;
                    }
                };
                let m = &l.mips[mip as usize];
                assert_eq!(p.w, m.width, "w {d:?} mip {mip}");
                assert_eq!(p.h, m.height, "h {d:?} mip {mip}");
                assert_eq!(p.d, m.depth, "d {d:?} mip {mip}");
                assert_eq!(p.pitch, m.pitch, "pitch {d:?} mip {mip}");
                assert_eq!(p.bw, l.block[0]);
                assert_eq!(p.bh, l.block[1]);
                let bz = if l.thick { l.block[2] } else { 1 };
                assert_eq!(p.bz, bz);
                // Tables must match the bit arrays.
                for c in 0..256u32 {
                    let mut vx = 0u32;
                    for i in 0..p.nx as usize {
                        vx |= ((c >> i) & 1) << p.xb[i];
                    }
                    assert_eq!(p.px[c as usize], vx);
                    let mut vy = 0u32;
                    for i in 0..p.ny as usize {
                        vy |= ((c >> i) & 1) << p.yb[i];
                    }
                    assert_eq!(p.py[c as usize], vy);
                    let mut vz = 0u32;
                    for i in 0..p.nz as usize {
                        vz |= ((c >> i) & 1) << p.zb[i];
                    }
                    assert_eq!(p.pz[c as usize], vz);
                }
                // Every element address must match the public Layout::offset.
                for z in 0..m.depth {
                    for y in 0..m.height {
                        for x in 0..m.width {
                            let kernel_z = if three_d { z } else { 0 };
                            let expect = l.offset(mip, x, y, if three_d { z } else { slice }).unwrap();
                            let got = elem_offset(&p, x, y, kernel_z, d.element_bytes.trailing_zeros());
                            assert_eq!(got, expect, "addr {d:?} mip {mip} ({x},{y},{z}) slice {slice}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn mapping_matches_layout_offset() {
        let modes = [
            Swizzle::LINEAR,
            Swizzle::SW_256B_S,
            Swizzle::SW_256B_D,
            Swizzle::SW_4KB_S,
            Swizzle::SW_4KB_D,
            Swizzle::SW_64KB_S,
            Swizzle::SW_64KB_D,
        ];
        // Non-power-of-two sizes, full chains (mip tails included).
        for &m in &modes {
            for bpe in [1u32, 2, 4, 8, 16] {
                check_layout(&desc(Dimension::Tex2d, 300, 175, 1, 9, bpe, 1, m));
            }
        }
        // BC formats.
        for &m in &modes {
            for bpe in [8u32, 16] {
                check_layout(&desc(Dimension::Tex2d, 256, 256, 1, 9, bpe, 4, m));
                check_layout(&desc(Dimension::Tex2d, 1000, 750, 1, 10, bpe, 4, m));
            }
        }
        // Degenerate / odd sizes.
        for (w, h) in [(1u32, 1u32), (2, 3), (7, 5), (33, 65), (129, 1), (1, 130), (33, 17)] {
            check_layout(&desc(Dimension::Tex2d, w, h, 1, 1, 4, 1, Swizzle::SW_4KB_S));
            check_layout(&desc(Dimension::Tex2d, w, h, 1, 4, 16, 4, Swizzle::SW_4KB_S));
        }
        // Arrays with more than one slice.
        check_layout(&desc(Dimension::Tex2d, 256, 256, 6, 9, 4, 1, Swizzle::SW_4KB_S));
        check_layout(&desc(Dimension::Tex2d, 200, 100, 5, 8, 16, 4, Swizzle::SW_64KB_D));
        check_layout(&desc(Dimension::Tex2d, 100, 60, 3, 5, 2, 1, Swizzle::SW_4KB_D));
        // 3D: thick S modes and linear.
        for (m, bpe) in [
            (Swizzle::SW_4KB_S, 4u32),
            (Swizzle::SW_64KB_S, 4),
            (Swizzle::SW_64KB_S, 2),
            (Swizzle::SW_4KB_S, 8),
            (Swizzle::SW_64KB_S, 1),
            (Swizzle::LINEAR, 4),
        ] {
            check_layout(&desc(Dimension::Tex3d, 70, 50, 40, 6, bpe, 1, m));
        }
        check_layout(&desc(Dimension::Tex3d, 64, 32, 16, 7, 4, 1, Swizzle::SW_4KB_S));
        // 1D, linear.
        check_layout(&desc(Dimension::Tex1d, 100, 1, 1, 5, 4, 1, Swizzle::LINEAR));
    }

    #[test]
    fn unsupported_cases_return_none_with_reason() {
        // XOR / pipe-bank modes.
        for sw in [
            Swizzle::SW_64KB_S_T,
            Swizzle::SW_64KB_D_T,
            Swizzle::SW_4KB_S_X,
            Swizzle::SW_4KB_D_X,
            Swizzle::SW_4KB_R_X,
            Swizzle::SW_64KB_Z_X,
            Swizzle::SW_64KB_S_X,
            Swizzle::SW_64KB_D_X,
            Swizzle::SW_64KB_R_X,
        ] {
            let l = Layout::new(&desc(Dimension::Tex2d, 64, 64, 1, 1, 4, 1, sw)).unwrap();
            assert!(kernel_params(&l, 0, 0).is_none());
            assert!(unsupported_reason(&l, 0, 0).unwrap().contains("XOR"));
        }
        // MSAA.
        let msaa = Desc {
            samples: 2,
            ..desc(Dimension::Tex2d, 64, 64, 1, 1, 4, 1, Swizzle::SW_4KB_S)
        };
        let l = Layout::new(&msaa).unwrap();
        assert!(kernel_params(&l, 0, 0).is_none());
        assert!(unsupported_reason(&l, 0, 0).unwrap().contains("multisampled"));
        // Bad mip / slice.
        let l = Layout::new(&desc(Dimension::Tex2d, 64, 64, 1, 4, 4, 1, Swizzle::SW_4KB_S)).unwrap();
        assert!(kernel_params(&l, 99, 0).is_none());
        assert!(kernel_params(&l, 0, 3).is_none());
    }
}

// ---------------------------------------------------------------------------
// GPU-side Wine test (Windows only, `cargo test`). Stripped on native macOS.
// Synthetic random data only; no game / Wine resources are read.
// ---------------------------------------------------------------------------

#[cfg(all(test, windows))]
mod gpu_tests {
    use super::*;
    use victory_formats::texture::{Desc, Dimension, Layout};
    use victory_metal::{Device, Pool};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn fill(&mut self, b: &mut [u8]) {
            for c in b.chunks_mut(8) {
                let v = self.next().to_le_bytes();
                c.copy_from_slice(&v[..c.len()]);
            }
        }
    }

    struct Sub {
        mip: u32,
        slice: u32,
        off: usize,
        len: usize,
        row: usize,
        plane: usize,
        params: Params,
    }

    #[test]
    fn gpu_detile_matches_cpu() {
        let _pool = Pool::new();
        let device = Device::first().expect("Device::first");
        let queue = device.new_queue(4).expect("new_queue");
        let detiler = Detiler::new(&device).expect("Detiler::new");

        let cases = cases();
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let mut n_case = 0usize;
        let mut n_pass = 0usize;
        let mut n_sub = 0usize;
        let mut failures: Vec<String> = Vec::new();

        for desc in &cases {
            let label = format!(
                "{:?} {}x{}x{} m{} b{}{} sw{}",
                desc.dimension,
                desc.width,
                desc.height,
                desc.depth,
                desc.mips,
                desc.element_bytes,
                if desc.block == 4 { "bc" } else { "" },
                desc.swizzle.0,
            );
            let layout = match Layout::new(desc) {
                Ok(l) => l,
                Err(e) => {
                    println!("{label}: Layout::new refused ({e:?}), skipped");
                    continue;
                }
            };
            let bpe = desc.element_bytes as usize;
            let three_d = desc.dimension == Dimension::Tex3d;
            let arr = if three_d { 1 } else { desc.depth };

            // Source tiled memory: our own committed allocation, wrapped
            // no-copy.
            let src_len = layout.size as usize;
            let src_alloc = src_len.next_multiple_of(4096);
            let src_ptr = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    src_alloc,
                    MEM_RESERVE | MEM_COMMIT,
                    PAGE_READWRITE,
                )
            };
            assert!(!src_ptr.is_null(), "VirtualAlloc src {label}");
            // SAFETY: committed above; lives until after the buffers drop.
            let tiled = unsafe { std::slice::from_raw_parts_mut(src_ptr as *mut u8, src_len) };
            rng.fill(tiled);

            let mut subs: Vec<Sub> = Vec::new();
            let mut expect_len = 0usize;
            for slice in 0..arr {
                for mip in 0..desc.mips {
                    let m = &layout.mips[mip as usize];
                    let row = m.width as usize * bpe;
                    let plane = row * m.height as usize;
                    let len = plane * m.depth as usize;
                    let mut linear = vec![0u8; len];
                    rng.fill(&mut linear);
                    layout.tile(tiled, mip, slice, &linear, row, plane).expect("tile");
                    let params = kernel_params(&layout, mip, slice)
                        .unwrap_or_else(|| panic!("kernel_params None {label} mip {mip} slice {slice}"));
                    subs.push(Sub {
                        mip,
                        slice,
                        off: expect_len,
                        len,
                        row,
                        plane,
                        params,
                    });
                    expect_len += len;
                }
            }

            // CPU reference: Layout::detile into tight rows.
            let mut expect = vec![0u8; expect_len];
            for s in &subs {
                layout
                    .detile(tiled, s.mip, s.slice, &mut expect[s.off..s.off + s.len], s.row, s.plane)
                    .expect("detile");
            }

            // Staging allocation, wrapped no-copy.
            let dst_alloc = expect_len.next_multiple_of(4096);
            let dst_ptr = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    dst_alloc,
                    MEM_RESERVE | MEM_COMMIT,
                    PAGE_READWRITE,
                )
            };
            assert!(!dst_ptr.is_null(), "VirtualAlloc dst {label}");
            // SAFETY: committed above.
            let dst_slice = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, expect_len) };
            dst_slice.fill(0xCC);

            let src_buf = unsafe { device.wrap(src_ptr, src_alloc as u64) }.expect("wrap src");
            let dst_buf = unsafe { device.wrap(dst_ptr, dst_alloc as u64) }.expect("wrap dst");

            // One compute pass: every subresource dispatch, batched together.
            let mut steps = Vec::new();
            for s in &subs {
                detiler.encode(&mut steps, &src_buf, 0, &dst_buf, s.off as u64, &s.params);
            }
            let cb = queue.command_buffer().expect("command_buffer");
            cb.compute(&steps).expect("compute");
            cb.commit();
            cb.wait().expect("wait");

            let mut bad = 0usize;
            for s in &subs {
                for i in 0..s.len {
                    if dst_slice[s.off + i] != expect[s.off + i] {
                        bad += 1;
                    }
                }
            }
            n_case += 1;
            n_sub += subs.len();
            if bad == 0 {
                n_pass += 1;
                println!("PASS {label}: {} subresources, {} bytes", subs.len(), expect_len);
            } else {
                failures.push(format!("{label}: {bad} bytes differ"));
                println!("FAIL {label}: {bad} bytes differ");
            }

            drop(src_buf);
            drop(dst_buf);
            // SAFETY: ours; the command buffer has completed.
            unsafe {
                VirtualFree(src_ptr, 0, MEM_RELEASE);
                VirtualFree(dst_ptr, 0, MEM_RELEASE);
            }
        }

        println!("SUMMARY cases={n_case} pass={n_pass} subresources={n_sub}");
        assert!(n_case >= 20, "only {n_case} cases");
        assert!(failures.is_empty(), "failures: {failures:?}");
    }

    fn cases() -> Vec<Desc> {
        use victory_formats::texture::{Dimension, Swizzle};
        use Dimension::*;
        let mut v: Vec<Desc> = Vec::new();
        let mut push = |dim: Dimension, w: u32, h: u32, depth: u32, mips: u32, bpe: u32, block: u32, sw: Swizzle| {
            let max_side = w.max(h).max(depth).max(1);
            let max_mips = 32 - max_side.leading_zeros();
            v.push(Desc {
                dimension: dim,
                width: w,
                height: h,
                depth,
                mips: mips.min(max_mips),
                element_bytes: bpe,
                block,
                samples: 1,
                swizzle: sw,
            });
        };

        // 2D, every addressable mode, 1/2/4/8/16-byte elements, non-Po2.
        push(Tex2d, 300, 175, 1, 9, 1, 1, Swizzle::LINEAR);
        push(Tex2d, 300, 175, 1, 9, 4, 1, Swizzle::LINEAR);
        push(Tex2d, 200, 150, 1, 6, 1, 1, Swizzle::SW_256B_S);
        push(Tex2d, 200, 150, 1, 6, 2, 1, Swizzle::SW_256B_D);
        push(Tex2d, 300, 175, 1, 9, 4, 1, Swizzle::SW_4KB_S);
        push(Tex2d, 300, 175, 1, 9, 8, 1, Swizzle::SW_4KB_D);
        push(Tex2d, 300, 175, 1, 9, 16, 1, Swizzle::SW_64KB_S);
        push(Tex2d, 256, 256, 1, 9, 4, 1, Swizzle::SW_64KB_D);
        push(Tex2d, 129, 97, 1, 5, 2, 1, Swizzle::SW_256B_S);
        push(Tex2d, 33, 17, 1, 3, 1, 1, Swizzle::SW_4KB_S);
        push(Tex2d, 129, 1, 1, 1, 4, 1, Swizzle::SW_64KB_D);
        push(Tex2d, 1, 130, 1, 2, 2, 1, Swizzle::SW_4KB_S);
        // BC1 (8 B) / BC7 (16 B), 4x4 blocks.
        push(Tex2d, 1000, 750, 1, 10, 8, 4, Swizzle::SW_4KB_S);
        push(Tex2d, 256, 256, 1, 9, 16, 4, Swizzle::SW_64KB_D);
        push(Tex2d, 256, 256, 1, 9, 8, 4, Swizzle::LINEAR);
        push(Tex2d, 1000, 750, 1, 10, 16, 4, Swizzle::SW_256B_S);
        // 2D arrays (several slices).
        push(Tex2d, 256, 256, 6, 9, 4, 1, Swizzle::SW_4KB_S);
        push(Tex2d, 200, 100, 5, 8, 16, 4, Swizzle::SW_64KB_D);
        push(Tex2d, 100, 60, 3, 5, 2, 1, Swizzle::SW_4KB_D);
        push(Tex2d, 129, 97, 4, 6, 8, 1, Swizzle::SW_64KB_S);
        // 3D: thick S modes and linear, non-Po2 depth.
        push(Tex3d, 70, 50, 40, 6, 4, 1, Swizzle::SW_4KB_S);
        push(Tex3d, 64, 32, 16, 7, 4, 1, Swizzle::SW_4KB_S);
        push(Tex3d, 70, 50, 40, 6, 2, 1, Swizzle::SW_64KB_S);
        push(Tex3d, 70, 50, 40, 6, 8, 1, Swizzle::SW_4KB_S);
        push(Tex3d, 33, 21, 9, 4, 4, 1, Swizzle::LINEAR);
        // 1D, linear.
        push(Tex1d, 100, 1, 1, 5, 4, 1, Swizzle::LINEAR);
        // Small / degenerate / odd sizes, short chains.
        push(Tex2d, 7, 5, 1, 1, 1, 1, Swizzle::SW_4KB_S);
        push(Tex2d, 33, 65, 1, 2, 16, 1, Swizzle::SW_64KB_S);
        push(Tex2d, 120, 80, 1, 6, 4, 1, Swizzle::SW_4KB_D);
        push(Tex2d, 96, 64, 3, 5, 2, 1, Swizzle::SW_256B_D);
        v
    }
}
