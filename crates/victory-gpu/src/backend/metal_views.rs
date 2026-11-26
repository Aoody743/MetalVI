//! Buffer views as the Metal backend sees them: the byte window of title
//! memory a shader resource or unordered access view of a buffer covers,
//! read from the view description words victory keeps (descriptor.rs
//! `View`, which are the D3D12 view description structs as words).
//!
//! Pure functions (no Windows, no Metal): tested natively through
//! `metal-backend/tools/views-test`.

use super::metal_map::format_props;

/// `D3D12_SRV_DIMENSION_BUFFER` / `D3D12_UAV_DIMENSION_BUFFER`.
const DIMENSION_BUFFER: u32 = 1;
/// `D3D12_BUFFER_SRV_FLAG_RAW` / `D3D12_BUFFER_UAV_FLAG_RAW`.
const FLAG_RAW: u32 = 1;

/// The bytes of a buffer a view covers, relative to the buffer's start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    pub offset: u64,
    pub bytes: u64,
    /// A typed buffer view (format conversion on load/store): the DXGI
    /// format, which needs a texture buffer view on Metal. `None` for raw
    /// and structured buffers, which are plain memory.
    pub typed_format: Option<u32>,
}

/// Element size and typed-ness of a buffer view: structured (the stride),
/// raw (4-byte words) or typed (the format's size).
fn element(format: u32, stride: u32, flags: u32) -> Option<(u64, Option<u32>)> {
    if stride > 0 {
        return Some((u64::from(stride), None));
    }
    if flags & FLAG_RAW != 0 {
        return Some((4, None));
    }
    let props = format_props(format as i32)?;
    if props.compressed || props.block_size_bytes == 0 {
        return None;
    }
    Some((u64::from(props.block_size_bytes), Some(format)))
}

/// `D3D12_SHADER_RESOURCE_VIEW_DESC` as words: format, dimension,
/// component mapping, then (8-aligned) `Buffer { FirstElement: u64,
/// NumElements, StructureByteStride, Flags }`. `None` if it is not a
/// buffer view or its format has no element size.
pub fn srv_window(words: &[u32]) -> Option<Window> {
    if words.len() < 9 || words[1] != DIMENSION_BUFFER {
        return None;
    }
    let first = u64::from(words[4]) | u64::from(words[5]) << 32;
    let (size, typed_format) = element(words[0], words[7], words[8])?;
    Some(Window { offset: first * size, bytes: u64::from(words[6]) * size, typed_format })
}

/// `D3D12_UNORDERED_ACCESS_VIEW_DESC` as words: format, dimension, then
/// (8-aligned) `Buffer { FirstElement: u64, NumElements,
/// StructureByteStride, CounterOffsetInBytes: u64, Flags }`.
pub fn uav_window(words: &[u32]) -> Option<Window> {
    if words.len() < 9 || words[1] != DIMENSION_BUFFER {
        return None;
    }
    let first = u64::from(words[2]) | u64::from(words[3]) << 32;
    let (size, typed_format) = element(words[0], words[5], words[8])?;
    Some(Window { offset: first * size, bytes: u64::from(words[4]) * size, typed_format })
}

/// The title address of a view of placed memory: words 10 and 11 of the
/// view description (descriptor.rs `placed_of`).
pub fn placed_address(desc: &[u32; 16]) -> u64 {
    u64::from(desc[10]) | u64::from(desc[11]) << 32
}

#[cfg(test)]
mod tests {
    use super::*;

    // DXGI formats used below.
    const R32_TYPELESS: u32 = 39;
    const R32_UINT: u32 = 42;
    const R16G16B16A16_FLOAT: u32 = 10;
    const UNKNOWN: u32 = 0;
    const BC1_UNORM: u32 = 71;

    /// The union starts at word 4: three words, then 8-byte alignment.
    fn srv(format: u32, first: u64, n: u32, stride: u32, flags: u32) -> [u32; 10] {
        [format, DIMENSION_BUFFER, 0x1688, 0, first as u32, (first >> 32) as u32, n, stride, flags, 0]
    }

    fn uav(format: u32, first: u64, n: u32, stride: u32, flags: u32) -> [u32; 10] {
        [format, DIMENSION_BUFFER, first as u32, (first >> 32) as u32, n, stride, 0, 0, flags, 0]
    }

    #[test]
    fn structured_raw_and_typed_srvs() {
        // structured: 1000 elements of 48 bytes from element 10
        assert_eq!(
            srv_window(&srv(UNKNOWN, 10, 1000, 48, 0)),
            Some(Window { offset: 480, bytes: 48_000, typed_format: None })
        );
        // raw: 4-byte words
        assert_eq!(
            srv_window(&srv(R32_TYPELESS, 256, 64, 0, FLAG_RAW)),
            Some(Window { offset: 1024, bytes: 256, typed_format: None })
        );
        // typed: the format's size, and it is reported as typed
        assert_eq!(
            srv_window(&srv(R16G16B16A16_FLOAT, 2, 3, 0, 0)),
            Some(Window { offset: 16, bytes: 24, typed_format: Some(R16G16B16A16_FLOAT) })
        );
    }

    #[test]
    fn uav_layout_differs_from_srv() {
        assert_eq!(
            uav_window(&uav(R32_UINT, 4, 8, 0, 0)),
            Some(Window { offset: 16, bytes: 32, typed_format: Some(R32_UINT) })
        );
        assert_eq!(
            uav_window(&uav(UNKNOWN, 1 << 33, 2, 16, 0)),
            Some(Window { offset: 16 << 33, bytes: 32, typed_format: None })
        );
    }

    #[test]
    fn not_buffers_or_no_size() {
        let mut texture = srv(R32_UINT, 0, 1, 0, 0);
        texture[1] = 4; // TEXTURE2D
        assert_eq!(srv_window(&texture), None);
        // a typed view of a compressed format has no element size
        assert_eq!(srv_window(&srv(BC1_UNORM, 0, 1, 0, 0)), None);
        assert_eq!(uav_window(&[0; 4]), None);
    }

    #[test]
    fn placed_address_words() {
        let mut d = [0u32; 16];
        d[10] = 0x1234_5000;
        d[11] = 0x7;
        assert_eq!(placed_address(&d), 0x7_1234_5000);
    }
}
