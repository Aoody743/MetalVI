//! Console primitive types the host lacks. A rectangle list gives three
//! corners per rectangle, the second the one between the other two, and
//! the hardware adds the fourth at `v0 + v2 - v1` (the title's
//! full-screen rectangles come as top left, bottom left, bottom right); on
//! the host each rectangle becomes two triangles, with the fourth vertex
//! made from the other three on the CPU.
//!
//! shortcut: the fourth vertex is computed from vertex inputs, not from
//! vertex shader outputs as the hardware does; that is the same for vertex
//! shaders linear in their inputs (blits, 2D quads). Inputs in formats
//! [`components`] does not know keep the first vertex's value, and so do
//! components past the end of the vertex (an element that reaches into the
//! next one, which the layout allows).
//!
//! A pipeline the title makes for rectangles completes them on the GPU
//! instead, in a geometry stage (`victory_shader::emit::rect_list_gs`):
//! from the vertex shader's outputs, whichever vertex the corner is, for
//! direct, indexed and indirect draws alike. This expansion is what is
//! left for a pipeline that could not get that stage.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ListExpansion {
    Fan,
    Loop,
    Quads,
    QuadStrip,
}

impl ListExpansion {
    pub(crate) fn from_topology(topology: u32) -> Option<Self> {
        Some(match topology {
            6 => Self::Fan,
            18 => Self::Loop,
            19 => Self::Quads,
            20 => Self::QuadStrip,
            _ => return None,
        })
    }

    pub(crate) fn host_topology(self) -> u32 {
        if self == Self::Loop { 2 } else { 4 } // line list / triangle list
    }

    pub(crate) fn index_count(self, count: u32) -> Option<u32> {
        let indices = match self {
            Self::Fan => count.saturating_sub(2).checked_mul(3)?,
            Self::Loop if count >= 2 => count.checked_mul(2)?,
            Self::Loop => 0,
            Self::Quads => (count / 4).checked_mul(6)?,
            Self::QuadStrip => (count / 2).saturating_sub(1).checked_mul(6)?,
        };
        // A D3D12 index view's byte size is a UINT.
        indices.checked_mul(4)?;
        Some(indices)
    }

    fn emit(self, count: u32, mut index: impl FnMut(u32) -> u32, mut push: impl FnMut(u32)) {
        let mut primitive = |slots: &[u32]| {
            for &i in slots {
                push(index(i));
            }
        };
        match self {
            Self::Fan => {
                for i in 1..count.saturating_sub(1) {
                    primitive(&[0, i, i + 1]);
                }
            }
            Self::Loop if count >= 2 => {
                for i in 0..count - 1 {
                    primitive(&[i, i + 1]);
                }
                primitive(&[count - 1, 0]);
            }
            Self::Loop => {}
            Self::Quads => {
                for i in (0..count / 4).map(|i| i * 4) {
                    primitive(&[i, i + 1, i + 2, i, i + 2, i + 3]);
                }
            }
            Self::QuadStrip => {
                for i in (0..(count / 2).saturating_sub(1)).map(|i| i * 2) {
                    primitive(&[i, i + 1, i + 3, i, i + 3, i + 2]);
                }
            }
        }
    }
}

/// The part of the title index view consumed by a draw, in bytes.
fn index_range(format: u32, size: u32, first: u32, count: u32) -> Option<(u64, u64, usize)> {
    let width = match format {
        57 => 2, // R16_UINT
        42 => 4, // R32_UINT
        _ => return None,
    };
    let (offset, len) = (u64::from(first) * width, u64::from(count) * width);
    (offset + len <= u64::from(size)).then_some((offset, len, width as usize))
}

fn read_index(src: &[u8], width: usize, i: u32) -> u32 {
    let at = i as usize * width;
    if width == 2 {
        u32::from(u16::from_le_bytes(src[at..at + 2].try_into().unwrap()))
    } else {
        u32::from_le_bytes(src[at..at + 4].try_into().unwrap())
    }
}

#[cfg(windows)]
mod host_lists {
    use super::{ListExpansion, index_range, read_index};
    use crate::host::HostList;
    use windows::Win32::Graphics::{
        Direct3D12::D3D12_INDEX_BUFFER_VIEW, Dxgi::Common::DXGI_FORMAT_R32_UINT,
    };

    fn scratch(list: &HostList, count: u32) -> Option<(D3D12_INDEX_BUFFER_VIEW, usize)> {
        let bytes = count.checked_mul(4)?;
        let (gpu, cpu) = list.scratch(u64::from(bytes))?;
        let view = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: gpu,
            SizeInBytes: bytes,
            Format: DXGI_FORMAT_R32_UINT,
        };
        Some((view, cpu as usize))
    }

    impl ListExpansion {
        /// Non-indexed draws use relative indices, so `first` stays in BaseVertexLocation.
        pub(crate) fn generated_view(
            self,
            list: &HostList,
            count: u32,
        ) -> Option<D3D12_INDEX_BUFFER_VIEW> {
            let indices = self.index_count(count)?.max(1);
            let (view, dst) = scratch(list, indices)?;
            let mut out = dst as *mut u32;
            self.emit(
                count,
                |i| i,
                |index| {
                    // SAFETY: emit writes exactly index_count words into list-owned scratch.
                    unsafe {
                        out.write(index);
                        out = out.add(1);
                    }
                },
            );
            Some(view)
        }

        /// Expand CPU-visible indices at submission, after deferred title copies.
        /// This cannot see indices produced only in the GPU's separate host copy.
        pub(crate) fn title_view(
            self,
            list: &HostList,
            address: u64,
            size: u32,
            format: u32,
            first: u32,
            count: u32,
        ) -> Option<D3D12_INDEX_BUFFER_VIEW> {
            let indices = self.index_count(count)?;
            let (offset, len, width) = index_range(format, size, first, count)?;
            if address == 0 || !address.is_multiple_of(width as u64) {
                return None;
            }
            let src = usize::try_from(address.checked_add(offset)?).ok()?;
            let len = usize::try_from(len).ok()?;
            src.checked_add(len)?;
            let (view, dst) = scratch(list, indices.max(1))?;
            if indices != 0 {
                list.defer_host(move || {
                    let mut out = dst as *mut u32;
                    let mut bytes = Vec::new();
                    let read = bytes.try_reserve_exact(len).is_ok() && {
                        bytes.resize(len, 0);
                        let _mapping = crate::watch::title_memory_reading_range(src, len);
                        crate::host::read_into(src as *const u8, &mut bytes)
                    };
                    if !read {
                        // SAFETY: this is the list's own scratch. Zero indices make
                        // only degenerate primitives if the title unmapped the view.
                        unsafe { out.write_bytes(0, indices as usize) };
                        return;
                    }
                    self.emit(
                        count,
                        |i| read_index(&bytes, width, i),
                        |index| {
                            // SAFETY: exactly indices words in the list's scratch.
                            unsafe {
                                out.write(index);
                                out = out.add(1);
                            }
                        },
                    );
                });
            }
            Some(view)
        }
    }
}

#[cfg(all(test, windows))]
mod d3d12_test;

/// Expands `rects` rectangles of `stride`-byte vertices from `src` into six
/// vertices each in `dst`: `(v0, v1, v2)` and `(v0, v2, v3)`, so both
/// triangles turn the same way. `elements` are (byte offset, format).
pub(crate) fn expand_rects(src: &[u8], dst: &mut [u8], stride: usize, elements: &[(u32, u32)]) {
    for (rect, out) in src
        .chunks_exact(3 * stride)
        .zip(dst.chunks_exact_mut(6 * stride))
    {
        let v = |i: usize| &rect[i * stride..(i + 1) * stride];
        for (slot, i) in [0, 1, 2, 0, 2].into_iter().enumerate() {
            out[slot * stride..(slot + 1) * stride].copy_from_slice(v(i));
        }
        let fourth = &mut out[5 * stride..];
        fourth.copy_from_slice(v(0));
        for &(offset, format) in elements {
            let at = offset as usize;
            let Some((n, kind)) = components(format) else {
                continue;
            };
            // Only the components inside the vertex: the rest belong to the
            // next one.
            let inside = stride.saturating_sub(at) / kind.bytes();
            for c in 0..n.min(inside) {
                let get = |i: usize| kind.read(&v(i)[at..], c);
                let value = get(0) + get(2) - get(1);
                kind.write(&mut fourth[at..], c, value);
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Kind {
    F32,
    F16,
    Unorm8,
    Unorm16,
}

/// Component count and kind of the vertex formats rectangles use.
fn components(format: u32) -> Option<(usize, Kind)> {
    Some(match format {
        2 => (4, Kind::F32),
        6 => (3, Kind::F32),
        16 => (2, Kind::F32),
        41 => (1, Kind::F32),
        10 => (4, Kind::F16),
        34 => (2, Kind::F16),
        54 => (1, Kind::F16),
        11 => (4, Kind::Unorm16),
        35 => (2, Kind::Unorm16),
        28 | 29 | 87 | 91 => (4, Kind::Unorm8),
        _ => return None,
    })
}

impl Kind {
    fn bytes(self) -> usize {
        match self {
            Kind::F32 => 4,
            Kind::F16 | Kind::Unorm16 => 2,
            Kind::Unorm8 => 1,
        }
    }

    fn read(self, at: &[u8], c: usize) -> f32 {
        let b = &at[c * self.bytes()..];
        match self {
            Kind::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            Kind::F16 => half_to_f32(u16::from_le_bytes([b[0], b[1]])),
            Kind::Unorm16 => f32::from(u16::from_le_bytes([b[0], b[1]])) / 65535.0,
            Kind::Unorm8 => f32::from(b[0]) / 255.0,
        }
    }

    fn write(self, at: &mut [u8], c: usize, v: f32) {
        let n = self.bytes();
        let b = &mut at[c * n..(c + 1) * n];
        match self {
            Kind::F32 => b.copy_from_slice(&v.to_le_bytes()),
            Kind::F16 => b.copy_from_slice(&f32_to_half(v).to_le_bytes()),
            Kind::Unorm16 => {
                b.copy_from_slice(&((v.clamp(0.0, 1.0) * 65535.0).round() as u16).to_le_bytes())
            }
            Kind::Unorm8 => b[0] = (v.clamp(0.0, 1.0) * 255.0).round() as u8,
        }
    }
}

fn half_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match (exp, man) {
        (0, 0) => sign,
        // Subnormal: scale the mantissa.
        (0, m) => {
            let v = m as f32 * 2f32.powi(-24);
            return if sign != 0 { -v } else { v };
        }
        (31, m) => sign | 0x7f80_0000 | m << 13,
        (e, m) => sign | (e + 112) << 23 | m << 13,
    };
    f32::from_bits(bits)
}

fn f32_to_half(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let man = bits & 0x7f_ffff;
    if exp >= 31 {
        return sign | 0x7c00; // too large: infinity
    }
    if exp <= 0 {
        if exp < -10 {
            return sign; // below half the smallest subnormal: zero
        }
        // Half subnormals are `m * 2^-24`. An f32 in this range is still
        // normal (exponent field >= 1), so its significand has the implicit
        // leading 1; shift it into the half mantissa, rounding to even.
        let sig = (1u32 << 23) | man;
        let shift = (14 - exp) as u32;
        let halfway = 1u32 << (shift - 1);
        let q = sig >> shift;
        let r = sig & ((1u32 << shift) - 1);
        let m = q + u32::from(r > halfway || (r == halfway && (q & 1) != 0));
        return sign | m as u16;
    }
    // Rounded to nearest: a carry out of the mantissa is added into the
    // exponent (`2 - 2^-12` is 2.0); at most it reaches infinity, 0x7c00.
    let half = (exp as u32) << 10;
    sign | (half + ((man + 0x1000) >> 13)) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_expand_with_consistent_winding_and_drop_partial_primitives() {
        for (topology, n, expected) in [
            (6, 5, &[0, 1, 2, 0, 2, 3, 0, 3, 4][..]),
            (18, 3, &[0, 1, 1, 2, 2, 0]),
            (19, 9, &[0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]),
            (20, 7, &[0, 1, 3, 0, 3, 2, 2, 3, 5, 2, 5, 4]),
        ] {
            let expand = ListExpansion::from_topology(topology).unwrap();
            assert_eq!(expand.host_topology(), if topology == 18 { 2 } else { 4 });
            let mut actual = Vec::new();
            expand.emit(n, |i| i, |i| actual.push(i));
            assert_eq!(actual, expected);
            assert_eq!(expand.index_count(n), Some(expected.len() as u32));
            assert_eq!(expand.index_count(u32::MAX), None);
            assert_eq!(expand.index_count(1), Some(0));
        }
        assert_eq!(ListExpansion::from_topology(17), None);
    }

    #[test]
    fn indexed_lists_preserve_title_values_and_validate_view_ranges() {
        for format in [57, 42] {
            let indices = [0, 9, 8, 65535, 6, 5];
            let width = if format == 57 { 2 } else { 4 };
            let bytes: Vec<u8> = indices
                .iter()
                .flat_map(|x: &u32| x.to_le_bytes()[..width].to_vec())
                .collect();
            let size = bytes.len() as u32;
            let (offset, len, width) = index_range(format, size, 1, 4).unwrap();
            let src = &bytes[offset as usize..(offset + len) as usize];
            let mut out = Vec::new();
            ListExpansion::Quads.emit(4, |i| read_index(src, width, i), |i| out.push(i));
            assert_eq!(out, [9, 8, 65535, 9, 65535, 6]);
            assert!(index_range(format, size, 3, 4).is_none());
            assert!(index_range(format, size, u32::MAX, 1).is_none());
        }
        assert!(index_range(0, 1024, 0, 4).is_none());
        assert_eq!(read_index(&0xfedc_ba98u32.to_le_bytes(), 4, 0), 0xfedc_ba98);
    }

    /// A rectangle of float2 positions and RGBA8 colors, given as top
    /// left, bottom left, bottom right: the fourth corner completes the
    /// parallelogram at the top right.
    #[test]
    fn rectangles_get_their_fourth_corner() {
        let vertex = |x: f32, y: f32, r: u8| {
            let mut v = Vec::new();
            v.extend(x.to_le_bytes());
            v.extend(y.to_le_bytes());
            v.extend([r, 0, 0, 255]);
            v
        };
        let src: Vec<u8> = [
            vertex(-1.0, 1.0, 0),
            vertex(-1.0, -1.0, 100),
            vertex(1.0, -1.0, 150),
        ]
        .concat();
        let mut dst = vec![0u8; 6 * 12];
        expand_rects(&src, &mut dst, 12, &[(0, 16), (8, 28)]);
        assert_eq!(dst[..36], src[..]);
        assert_eq!(dst[36..48], src[..12]);
        assert_eq!(dst[48..60], src[24..36]);
        assert_eq!(dst[60..], vertex(1.0, 1.0, 50)[..]);
        assert_eq!(f32_to_half(1.5), 0x3e00);
        assert_eq!(half_to_f32(0x3e00), 1.5);
        assert_eq!(half_to_f32(0xc000), -2.0);
    }

    /// A rounding that carries out of the mantissa raises the exponent
    /// (the carry used to be or-ed into an odd exponent, halving the value).
    #[test]
    fn halves_round_up_into_the_next_power_of_two() {
        assert_eq!(f32_to_half(2.0 - 2f32.powi(-12)), 0x4000);
        assert_eq!(f32_to_half(0.5 - 2f32.powi(-13)), 0x3800);
        assert_eq!(f32_to_half(65520.0), 0x7c00, "past the largest half");
        // An R16_FLOAT element at 1.0, 2^-12 and 1.0: the fourth corner is
        // 2 - 2^-12, which rounds to 2.0.
        let src: Vec<u8> = [0x3c00_u16, 0x0c00, 0x3c00]
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect();
        let mut dst = vec![0u8; 6 * 2];
        expand_rects(&src, &mut dst, 2, &[(0, 54)]);
        assert_eq!(dst[10..], 0x4000_u16.to_le_bytes());
    }

    /// An element that reaches past the stride (float4 at 4 in 8-byte
    /// vertices, RGBA8 at the stride itself): the components inside get
    /// the fourth corner, the rest keep the first vertex's bytes. The old
    /// code indexed past the vertex and panicked.
    #[test]
    fn elements_past_the_stride_are_left_alone() {
        let vertex = |x: f32, y: f32| [x.to_le_bytes(), y.to_le_bytes()].concat();
        let src = [vertex(0.0, 2.0), vertex(1.0, 3.0), vertex(5.0, 7.0)].concat();
        let mut dst = vec![0u8; 6 * 8];
        expand_rects(&src, &mut dst, 8, &[(0, 16), (4, 2), (8, 28)]);
        assert_eq!(dst[..40], [&src[..], &src[..8], &src[16..]].concat()[..]);
        assert_eq!(dst[40..], vertex(4.0, 6.0)[..]);
    }

    /// `half_to_f32` reads subnormals, so an f16 rectangle output must be
    /// able to write them back instead of flushing every one to zero.
    #[test]
    fn half_subnormals_convert_in_both_directions() {
        assert_eq!(f32_to_half(2f32.powi(-24)), 0x0001);
        assert_eq!(f32_to_half(-2f32.powi(-24)), 0x8001);
        assert_eq!(f32_to_half(2f32.powi(-23)), 0x0002);
        assert_eq!(f32_to_half(1023.0 * 2f32.powi(-24)), 0x03ff);
        // Ties go to an even mantissa: 0/1 and 1/2.
        assert_eq!(f32_to_half(2f32.powi(-25)), 0x0000);
        assert_eq!(f32_to_half(3.0 * 2f32.powi(-25)), 0x0002);
        // Below half the smallest subnormal, and normal values, are unchanged.
        assert_eq!(f32_to_half(2f32.powi(-26)), 0x0000);
        assert_eq!(f32_to_half(1.5), 0x3e00);
        assert_eq!(half_to_f32(0x0001), 2f32.powi(-24));
    }
}
