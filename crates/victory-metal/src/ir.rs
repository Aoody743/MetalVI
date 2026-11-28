//! The data Metal Shader Converter's shaders read (metal_irconverter_runtime.h),
//! built without the runtime header: descriptor table entries for the view and
//! sampler heaps (bind points 0 and 1) and the top-level argument buffer
//! (bind point 2) whose layout the root signature gives.

/// `IRDescriptorTableEntry`: 24 bytes per heap slot.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DescriptorEntry {
    pub gpu_va: u64,
    pub texture_view_id: u64,
    pub metadata: u64,
}

const BUF_SIZE_MASK: u64 = 0xffff_ffff; // kIRBufSizeMask, at offset 0
const TEX_VIEW_OFFSET: u64 = 32; // kIRTexViewOffset
const TEX_VIEW_MASK: u64 = 0xff;
const TYPED_BUFFER_OFFSET: u64 = 63; // kIRTypedBufferOffset

impl DescriptorEntry {
    /// A buffer view (`IRDescriptorTableSetBuffer` with the metadata of
    /// `IRDescriptorTableGetBufferMetadata`): `size` bounds the shader's
    /// accesses; `typed` for typed buffers (then `texture_view_id` is the
    /// texture buffer view and `first_element` its offset in elements).
    pub fn buffer(gpu_va: u64, size: u64) -> Self {
        Self { gpu_va, texture_view_id: 0, metadata: size.min(BUF_SIZE_MASK) }
    }

    pub fn typed_buffer(gpu_va: u64, size: u64, texture_view_id: u64, first_element: u32) -> Self {
        Self {
            gpu_va,
            texture_view_id,
            metadata: size.min(BUF_SIZE_MASK)
                | (u64::from(first_element) & TEX_VIEW_MASK) << TEX_VIEW_OFFSET
                | 1 << TYPED_BUFFER_OFFSET,
        }
    }

    /// A texture (`IRDescriptorTableSetTexture`): its `gpuResourceID`.
    pub fn texture(resource_id: u64, min_lod_clamp: f32, metadata: u32) -> Self {
        Self {
            gpu_va: 0,
            texture_view_id: resource_id,
            metadata: u64::from(min_lod_clamp.to_bits()) | u64::from(metadata) << 32,
        }
    }

    /// A sampler (`IRDescriptorTableSetSampler`): its `gpuResourceID`.
    pub fn sampler(resource_id: u64, lod_bias: f32) -> Self {
        Self { gpu_va: resource_id, texture_view_id: 0, metadata: u64::from(lod_bias.to_bits()) }
    }

    /// An empty slot: reads give zeros (MSC bounds-checks by size 0).
    pub const NULL: Self = Self { gpu_va: 0, texture_view_id: 0, metadata: 0 };
}

/// `IRResourceType` numbering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceType {
    Table,
    Constant,
    Cbv,
    Srv,
    Uav,
    Sampler,
    Other(u32),
}

impl ResourceType {
    pub fn from_raw(v: u32) -> Self {
        match v {
            0 => Self::Table,
            1 => Self::Constant,
            2 => Self::Cbv,
            3 => Self::Srv,
            4 => Self::Uav,
            5 => Self::Sampler,
            v => Self::Other(v),
        }
    }

    pub fn to_raw(self) -> u32 {
        match self {
            Self::Table => 0,
            Self::Constant => 1,
            Self::Cbv => 2,
            Self::Srv => 3,
            Self::Uav => 4,
            Self::Sampler => 5,
            Self::Other(v) => v,
        }
    }
}

/// One root parameter's place in the top-level argument buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Location {
    pub kind: ResourceType,
    pub space: u32,
    pub slot: u32,
    pub offset: u32,
    pub size: u64,
}

/// The top-level argument buffer of one root signature: fill it per
/// dispatch or draw, then bind its bytes at bind point 2.
#[derive(Clone, Debug)]
pub struct TopLevel {
    pub layout: Vec<Location>,
    pub bytes: Vec<u8>,
}

impl TopLevel {
    pub fn new(layout: Vec<Location>, size: u32) -> Self {
        Self { layout, bytes: vec![0; (size as usize).next_multiple_of(8)] }
    }

    fn find(&self, kind: ResourceType, space: u32, slot: u32) -> Option<Location> {
        self.layout.iter().copied().find(|l| l.kind == kind && l.space == space && l.slot == slot)
    }

    /// The byte offset of the root constants at `b<slot>, space<space>`.
    pub fn constants_offset(&self, space: u32, slot: u32) -> Option<u32> {
        self.find(ResourceType::Constant, space, slot).map(|l| l.offset)
    }

    /// Root constants at `b<slot>, space<space>` (victory's user data:
    /// b0, space0). Returns false if the signature has none there or they
    /// do not fit.
    pub fn set_constants(&mut self, space: u32, slot: u32, values: &[u32]) -> bool {
        let Some(l) = self.find(ResourceType::Constant, space, slot) else { return false };
        let n = values.len() * 4;
        if n as u64 > l.size {
            return false;
        }
        let at = l.offset as usize;
        for (i, v) in values.iter().enumerate() {
            self.bytes[at + i * 4..at + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        true
    }

    /// The byte offset of root parameter `index` (locations come in root
    /// parameter order; static samplers' table, if any, after them).
    pub fn offset_of(&self, index: usize) -> Option<u32> {
        self.layout.get(index).map(|l| l.offset)
    }

    /// Writes a 64-bit value (a GPU address) at root parameter `index`.
    pub fn set_u64(&mut self, index: usize, value: u64) -> bool {
        let Some(at) = self.offset_of(index) else { return false };
        let at = at as usize;
        if at + 8 > self.bytes.len() {
            return false;
        }
        self.bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
        true
    }

    /// Static samplers' descriptor table: when a root signature declares
    /// static samplers, MSC appends one `Table` location at `layout[num_params]`
    /// (after the `0..num_params` root parameters, which may themselves
    /// include `Table` locations with `space`/`slot` `u32::MAX`).
    pub fn set_static_samplers(&mut self, num_params: usize, gpu_va: u64) -> bool {
        let Some(l) = self.layout.get(num_params) else { return false };
        if l.kind != ResourceType::Table || l.size < 8 {
            return false;
        }
        self.set_u64(num_params, gpu_va)
    }

    /// A root descriptor (CBV/SRV/UAV) at `<slot>, space<space>`: the GPU
    /// address it points at.
    pub fn set_root_descriptor(&mut self, kind: ResourceType, space: u32, slot: u32, gpu_va: u64) -> bool {
        let Some(l) = self.find(kind, space, slot) else { return false };
        let at = l.offset as usize;
        self.bytes[at..at + 8].copy_from_slice(&gpu_va.to_le_bytes());
        true
    }
}

impl TopLevel {
    /// The layout and size as bytes (for a pipeline cache).
    pub fn layout_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + self.layout.len() * 24);
        out.extend_from_slice(&(self.bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.layout.len() as u32).to_le_bytes());
        for l in &self.layout {
            for v in [l.kind.to_raw(), l.space, l.slot, l.offset] {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out.extend_from_slice(&l.size.to_le_bytes());
        }
        out
    }

    /// The inverse of [`TopLevel::layout_bytes`]: an empty top level and how
    /// many bytes it read.
    pub fn from_layout_bytes(b: &[u8]) -> Option<(Self, usize)> {
        let u32_at = |at: usize| b.get(at..at + 4).map(|x| u32::from_le_bytes(x.try_into().unwrap()));
        let size = u32_at(0)?;
        let n = u32_at(4)? as usize;
        if n > 4096 {
            return None;
        }
        let mut layout = Vec::with_capacity(n);
        let mut at = 8;
        for _ in 0..n {
            let kind = ResourceType::from_raw(u32_at(at)?);
            let (space, slot, offset) = (u32_at(at + 4)?, u32_at(at + 8)?, u32_at(at + 12)?);
            let size = u64::from_le_bytes(b.get(at + 16..at + 24)?.try_into().ok()?);
            layout.push(Location { kind, space, slot, offset, size });
            at += 24;
        }
        Some((Self::new(layout, size), at))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_is_24_bytes() {
        assert_eq!(size_of::<DescriptorEntry>(), 24);
    }

    #[test]
    fn buffer_metadata_matches_the_runtime_header() {
        // IRDescriptorTableGetBufferMetadata: size in bits 0..32, view offset
        // in 32..40, typed flag at 63.
        let e = DescriptorEntry::typed_buffer(0x1000, 0x1_0000_0010, 7, 3);
        assert_eq!(e.metadata & 0xffff_ffff, 0xffff_ffff);
        assert_eq!(e.metadata >> 32 & 0xff, 3);
        assert_eq!(e.metadata >> 63, 1);
        assert_eq!(DescriptorEntry::buffer(0x2000, 256).metadata, 256);
    }

    #[test]
    fn top_level_fills_by_location() {
        let layout = vec![
            Location { kind: ResourceType::Constant, space: 0, slot: 0, offset: 0, size: 16 },
            Location { kind: ResourceType::Uav, space: 1, slot: 0, offset: 16, size: 8 },
        ];
        let mut t = TopLevel::new(layout, 24);
        assert!(t.set_constants(0, 0, &[1, 2, 3, 4]));
        assert!(!t.set_constants(0, 0, &[0; 5]));
        assert!(t.set_root_descriptor(ResourceType::Uav, 1, 0, 0xAABB_CCDD_EEFF_0011));
        assert_eq!(&t.bytes[0..4], &1u32.to_le_bytes());
        assert_eq!(&t.bytes[16..24], &0xAABB_CCDD_EEFF_0011u64.to_le_bytes());
        assert!(!t.set_root_descriptor(ResourceType::Cbv, 1, 0, 1));
    }

    #[test]
    fn layout_round_trips() {
        let layout = vec![
            Location { kind: ResourceType::Constant, space: 0, slot: 0, offset: 0, size: 16 },
            Location { kind: ResourceType::Other(9), space: 1, slot: 3, offset: 16, size: 8 },
        ];
        let t = TopLevel::new(layout, 24);
        let b = t.layout_bytes();
        let (u, n) = TopLevel::from_layout_bytes(&b).unwrap();
        assert_eq!(n, b.len());
        assert_eq!(u.layout, t.layout);
        assert_eq!(u.bytes.len(), t.bytes.len());
    }

    #[test]
    fn static_sampler_table_follows_root_tables() {
        let layout = vec![
            Location { kind: ResourceType::Constant, space: 0, slot: 0, offset: 0, size: 16 },
            Location { kind: ResourceType::Table, space: u32::MAX, slot: u32::MAX, offset: 16, size: 8 },
            Location { kind: ResourceType::Table, space: u32::MAX, slot: u32::MAX, offset: 24, size: 8 },
        ];
        let mut t = TopLevel::new(layout, 32);
        assert!(t.set_u64(1, 0x1111_2222_3333_4444));
        assert!(t.set_static_samplers(2, 0xAAAA_BBBB_CCCC_DDDD));
        assert!(!t.set_static_samplers(3, 0x1));
        assert_eq!(&t.bytes[16..24], &0x1111_2222_3333_4444u64.to_le_bytes());
        assert_eq!(&t.bytes[24..32], &0xAAAA_BBBB_CCCC_DDDDu64.to_le_bytes());
    }
}
