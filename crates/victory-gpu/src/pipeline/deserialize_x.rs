//! Pure layout definitions and decoders for device method `0x218`
//! (`ID3D12Device::DeserializeComputePipelineStateX`).
//!
//! Kept free of Windows and crate-internal dependencies so unit tests can run
//! natively on macOS as well as on the Windows target.

use std::ffi::c_void;

/// Fixed byte size of `FSPX` section kind 0 (hardware compute state).
pub(crate) const DESERIALIZED_STATE_SIZE: usize = 0x1c;
/// Fixed byte size of `FSPX` section kind 1 (header/id block).
pub(crate) const DESERIALIZED_HEADER_SIZE: usize = 0x14;
/// Required alignment of `FSPX` section kind 6 (raw GFX10 ISA microcode).
pub(crate) const DESERIALIZED_CODE_ALIGN: usize = 0x100;
/// Size granule of `FSPX` section kind 6 microcode.
pub(crate) const DESERIALIZED_CODE_GRANULE: usize = 0x40;
/// Header size of `FSPX` section kind 7 (auxiliary/binding metadata).
pub(crate) const AUX_HEADER_BYTES: usize = 0x18;
/// Size of each metadata record following the kind 7 header.
pub(crate) const AUX_RECORD_BYTES: usize = 0x20;
/// Kind 7 record tag for root-layout entries.
pub(crate) const AUX_TAG_ROOT_LAYOUT: u32 = 0x41;
/// Kind 7 record tag for used-resource entries.
pub(crate) const AUX_TAG_USED_RESOURCE: u32 = 0x42;
/// Kind 7 record tag for shader-hash / metadata entries.
pub(crate) const AUX_TAG_HASH_META: u32 = 0x43;

/// Pointer-plus-length slice passed inside [`DeserializedComputeDesc`].
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RawBlob {
    pub(crate) data: *const u8,
    pub(crate) size: usize,
}

/// Device method `0x218` (`ID3D12Device::DeserializeComputePipelineStateX`)
/// description: a root signature plus the four precompiled `FSPX` compute
/// sections (kind 0 hardware state, kind 1 header, kind 6 256-byte-aligned
/// GFX10 ISA, kind 7 auxiliary/binding metadata). Total size: 72 bytes (`0x48`).
#[repr(C)]
pub(crate) struct DeserializedComputeDesc {
    pub(crate) root: *const c_void,
    /// `FSPX` section kind 0: fixed 28-byte (7-dword) compute hardware state.
    pub(crate) state: RawBlob,
    /// `FSPX` section kind 1: fixed 20-byte (5-dword) header/id block.
    pub(crate) header: RawBlob,
    /// `FSPX` section kind 6: 256-byte-aligned raw GFX10 ISA microcode.
    pub(crate) code: RawBlob,
    /// `FSPX` section kind 7: auxiliary/binding metadata (`0x18 + N * 0x20`).
    pub(crate) aux: RawBlob,
}
const _: () = assert!(size_of::<DeserializedComputeDesc>() == 0x48);

/// Decoded `FSPX` kind 0 compute state blob (28 bytes / 7 dwords).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ComputeStateHeader {
    pub(crate) words: [u32; 7],
    pub(crate) threads: [u32; 3],
    pub(crate) rsrc1: u32,
    pub(crate) rsrc2: u32,
}

impl ComputeStateHeader {
    /// VGPR count encoded in `COMPUTE_PGM_RSRC1` (granule of 4 VGPRs).
    pub(crate) fn vgprs(&self) -> u32 {
        ((self.rsrc1 & 0x3f) + 1) * 4
    }

    /// User-data SGPR count encoded in `COMPUTE_PGM_RSRC2` bits 1..=5.
    pub(crate) fn user_sgprs(&self) -> u8 {
        ((self.rsrc2 >> 1) & 0x1f) as u8
    }

    /// LDS bytes per thread group encoded in `COMPUTE_PGM_RSRC2` bits 15..=23
    /// (512-byte granules).
    pub(crate) fn lds_bytes(&self) -> u32 {
        ((self.rsrc2 >> 15) & 0x1ff) * 512
    }
}

/// Decodes a 28-byte `FSPX` kind 0 compute state blob into thread-group size
/// and hardware register settings. Returns `None` if `bytes.len() != 0x1c`.
pub(crate) fn parse_compute_state(bytes: &[u8]) -> Option<ComputeStateHeader> {
    if bytes.len() != DESERIALIZED_STATE_SIZE {
        return None;
    }
    let mut words = [0u32; 7];
    let (chunks, _) = bytes.as_chunks::<4>();
    for (w, chunk) in words.iter_mut().zip(chunks) {
        *w = u32::from_le_bytes(*chunk);
    }
    let threads = [words[3] & 0xffff, words[3] >> 16, words[4] & 0xffff];
    Some(ComputeStateHeader {
        words,
        threads,
        rsrc1: words[5],
        rsrc2: words[6],
    })
}

/// Summary of an `FSPX` kind 7 auxiliary/binding metadata blob (`0x18 + N * 0x20`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AuxMetadataSummary {
    pub(crate) version: u32,
    pub(crate) payload_size: u32,
    pub(crate) code_dwords: u32,
    pub(crate) record_count: usize,
    pub(crate) root_records: usize,
    pub(crate) resource_records: usize,
    pub(crate) hash_records: usize,
    pub(crate) other_records: usize,
}

/// Parses the header and 32-byte record tags of an `FSPX` kind 7 auxiliary
/// blob. Returns `None` if `bytes` does not match `0x18 + N * 0x20`.
pub(crate) fn parse_aux_summary(bytes: &[u8]) -> Option<AuxMetadataSummary> {
    if bytes.len() < AUX_HEADER_BYTES
        || !(bytes.len() - AUX_HEADER_BYTES).is_multiple_of(AUX_RECORD_BYTES)
    {
        return None;
    }
    let word = |at: usize| u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
    let version = word(0);
    let payload_size = word(4);
    let code_dwords = word(8);
    let records = &bytes[AUX_HEADER_BYTES..];
    let mut root_records = 0;
    let mut resource_records = 0;
    let mut hash_records = 0;
    let mut other_records = 0;
    let (recs, _) = records.as_chunks::<AUX_RECORD_BYTES>();
    for rec in recs {
        match u32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]]) {
            AUX_TAG_ROOT_LAYOUT => root_records += 1,
            AUX_TAG_USED_RESOURCE => resource_records += 1,
            AUX_TAG_HASH_META => hash_records += 1,
            _ => other_records += 1,
        }
    }
    Some(AuxMetadataSummary {
        version,
        payload_size,
        code_dwords,
        record_count: records.len() / AUX_RECORD_BYTES,
        root_records,
        resource_records,
        hash_records,
        other_records,
    })
}

/// Pure structural summary of a [`DeserializedComputeDesc`] without
/// dereferencing its inner blob pointers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeserializedComputeSummary {
    pub(crate) has_root: bool,
    pub(crate) state_size: usize,
    pub(crate) state_ok: bool,
    pub(crate) header_size: usize,
    pub(crate) header_ok: bool,
    pub(crate) code_size: usize,
    pub(crate) code_aligned: bool,
    pub(crate) aux_size: usize,
    pub(crate) aux_record_count: Option<usize>,
}

/// Summarizes a [`DeserializedComputeDesc`] from its inline fields alone.
pub(crate) fn summarize_deserialized_compute(
    desc: &DeserializedComputeDesc,
) -> DeserializedComputeSummary {
    let aux_size = desc.aux.size;
    let aux_record_count = (aux_size >= AUX_HEADER_BYTES
        && (aux_size - AUX_HEADER_BYTES).is_multiple_of(AUX_RECORD_BYTES))
    .then(|| (aux_size - AUX_HEADER_BYTES) / AUX_RECORD_BYTES);
    let code_aligned = !desc.code.data.is_null()
        && (desc.code.data as usize).is_multiple_of(DESERIALIZED_CODE_ALIGN)
        && desc.code.size.is_multiple_of(DESERIALIZED_CODE_GRANULE);
    DeserializedComputeSummary {
        has_root: !desc.root.is_null(),
        state_size: desc.state.size,
        state_ok: desc.state.size == DESERIALIZED_STATE_SIZE,
        header_size: desc.header.size,
        header_ok: desc.header.size == DESERIALIZED_HEADER_SIZE,
        code_size: desc.code.size,
        code_aligned,
        aux_size,
        aux_record_count,
    }
}

/// Zeroes `*out` when non-null and dereferences `desc` when non-null.
///
/// # Safety
/// Non-null `out` must point to writable pointer storage, and non-null `desc`
/// must point to a valid [`DeserializedComputeDesc`] for `'a`.
pub(crate) unsafe fn prepare_call<'a>(
    desc: *const DeserializedComputeDesc,
    out: *mut *const c_void,
) -> Option<&'a DeserializedComputeDesc> {
    if !out.is_null() {
        // SAFETY: caller guarantees non-null `out` is writable.
        unsafe { *out = std::ptr::null() };
    }
    // SAFETY: caller guarantees non-null `desc` points to a valid `DeserializedComputeDesc`.
    unsafe { desc.as_ref() }
}

#[cfg(test)]
mod tests {
    use std::mem::offset_of;

    use super::*;

    #[test]
    fn deserialized_compute_layout_and_node_stride() {
        assert_eq!(size_of::<DeserializedComputeDesc>(), 0x48);
        assert_eq!(offset_of!(DeserializedComputeDesc, root), 0x00);
        assert_eq!(offset_of!(DeserializedComputeDesc, state), 0x08);
        assert_eq!(offset_of!(DeserializedComputeDesc, header), 0x18);
        assert_eq!(offset_of!(DeserializedComputeDesc, code), 0x28);
        assert_eq!(offset_of!(DeserializedComputeDesc, aux), 0x38);

        // Simulate two consecutive 0x60-byte map nodes (`+0x00 next`, `+0x08 prev`,
        // `+0x10 key`, `+0x18..+0x60 DeserializedComputeDesc`) and verify that
        // reading 16 u64 words from node 0's descriptor lands on node 1's key,
        // root, and repeating 0x1c state size at w[14].
        #[repr(C)]
        struct SyntheticMapNode {
            next: u64,
            prev: u64,
            key: u64,
            desc: DeserializedComputeDesc,
        }
        assert_eq!(size_of::<SyntheticMapNode>(), 0x60);

        let dummy = [0u8; 8];
        let nodes = [
            SyntheticMapNode {
                next: 0x1000,
                prev: 0x2000,
                key: 0x1122_3344_5566_7788,
                desc: DeserializedComputeDesc {
                    root: 0x3000usize as *const c_void,
                    state: RawBlob {
                        data: dummy.as_ptr(),
                        size: DESERIALIZED_STATE_SIZE,
                    },
                    header: RawBlob {
                        data: dummy.as_ptr(),
                        size: DESERIALIZED_HEADER_SIZE,
                    },
                    code: RawBlob {
                        data: 0x4000usize as *const u8,
                        size: 0x180,
                    },
                    aux: RawBlob {
                        data: dummy.as_ptr(),
                        size: 0x118,
                    },
                },
            },
            SyntheticMapNode {
                next: 0x5000,
                prev: 0x6000,
                key: 0x99aa_bbcc_ddee_ff00,
                desc: DeserializedComputeDesc {
                    root: 0x7000usize as *const c_void,
                    state: RawBlob {
                        data: dummy.as_ptr(),
                        size: DESERIALIZED_STATE_SIZE,
                    },
                    header: RawBlob {
                        data: dummy.as_ptr(),
                        size: DESERIALIZED_HEADER_SIZE,
                    },
                    code: RawBlob {
                        data: 0x8000usize as *const u8,
                        size: 0x200,
                    },
                    aux: RawBlob {
                        data: dummy.as_ptr(),
                        size: 0x158,
                    },
                },
            },
        ];

        // SAFETY: `nodes` is a contiguous 2-element array (192 bytes); reading 16 u64s
        // (128 bytes) starting at `&nodes[0].desc` (offset 0x18) stays in bounds.
        let words =
            unsafe { std::slice::from_raw_parts((&raw const nodes[0].desc).cast::<u64>(), 16) };
        assert_eq!(words[0], 0x3000);
        assert_eq!(words[2], 0x1c);
        assert_eq!(words[4], 0x14);
        assert_eq!(words[6], 0x180);
        assert_eq!(words[8], 0x118);
        assert_eq!(words[11], 0x99aa_bbcc_ddee_ff00);
        assert_eq!(words[12], 0x7000);
        assert_eq!(words[14], 0x1c);
    }

    #[test]
    fn parse_compute_state_decodes_threads_and_rsrc() {
        let build = |d3: u32, d4: u32, rsrc1: u32, rsrc2: u32| {
            let mut bytes = [0u8; DESERIALIZED_STATE_SIZE];
            let words = [0x4005_cf00, 0x6c09_0102, 0, d3, d4, rsrc1, rsrc2];
            for (i, w) in words.iter().enumerate() {
                bytes[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
            }
            bytes
        };

        // (64, 1, 1), vgprs = (9 + 1) * 4 = 40, user_sgprs = 10, lds = 0.
        let b64 = build(0x0001_0040, 1, 0x402c_0349, 0x14);
        let s64 = parse_compute_state(&b64).unwrap();
        assert_eq!(s64.words[0], 0x4005_cf00);
        assert_eq!(s64.threads, [64, 1, 1]);
        assert_eq!(s64.vgprs(), 40);
        assert_eq!(s64.user_sgprs(), 10);
        assert_eq!(s64.lds_bytes(), 0);

        // (8, 8, 2), vgprs = (15 + 1) * 4 = 64, user_sgprs = 8, lds = 3 * 512 = 1536.
        let b882 = build((8 << 16) | 8, 2 | (0x10 << 16), 0x0f, (3 << 15) | (8 << 1));
        let s882 = parse_compute_state(&b882).unwrap();
        assert_eq!(s882.threads, [8, 8, 2]);
        assert_eq!(s882.vgprs(), 64);
        assert_eq!(s882.user_sgprs(), 8);
        assert_eq!(s882.lds_bytes(), 1536);

        assert!(parse_compute_state(&[]).is_none());
        assert!(parse_compute_state(&b64[..0x18]).is_none());
        assert!(parse_compute_state(&[0u8; 0x20]).is_none());
    }

    #[test]
    fn parse_aux_summary_counts_record_kinds() {
        let build_aux = |tags: &[u32]| {
            let mut bytes = vec![0u8; AUX_HEADER_BYTES + tags.len() * AUX_RECORD_BYTES];
            let payload_size = (bytes.len() - 0x14) as u32;
            bytes[0..4].copy_from_slice(&0x403u32.to_le_bytes());
            bytes[4..8].copy_from_slice(&payload_size.to_le_bytes());
            bytes[8..12].copy_from_slice(&0xc0u32.to_le_bytes());
            for (i, tag) in tags.iter().enumerate() {
                let at = AUX_HEADER_BYTES + i * AUX_RECORD_BYTES;
                bytes[at..at + 4].copy_from_slice(&tag.to_le_bytes());
            }
            bytes
        };

        let empty_aux = build_aux(&[]);
        let s0 = parse_aux_summary(&empty_aux).unwrap();
        assert_eq!(s0.version, 0x403);
        assert_eq!(s0.payload_size, 4);
        assert_eq!(s0.code_dwords, 0xc0);
        assert_eq!(s0.record_count, 0);

        let mixed = build_aux(&[0x41, 0x41, 0x41, 0x41, 0x42, 0x42, 0x42, 0x43, 0x43, 0x99]);
        let s1 = parse_aux_summary(&mixed).unwrap();
        assert_eq!(s1.record_count, 10);
        assert_eq!(s1.root_records, 4);
        assert_eq!(s1.resource_records, 3);
        assert_eq!(s1.hash_records, 2);
        assert_eq!(s1.other_records, 1);

        assert!(parse_aux_summary(&[0u8; 0x14]).is_none());
        assert!(parse_aux_summary(&[0u8; 0x20]).is_none());
    }

    #[test]
    fn summarize_checks_sizes_and_alignment() {
        #[repr(align(256))]
        struct Aligned([u8; 0x180]);
        let code = Aligned([0u8; 0x180]);

        let valid = DeserializedComputeDesc {
            root: 0x1000usize as *const c_void,
            state: RawBlob {
                data: std::ptr::null(),
                size: DESERIALIZED_STATE_SIZE,
            },
            header: RawBlob {
                data: std::ptr::null(),
                size: DESERIALIZED_HEADER_SIZE,
            },
            code: RawBlob {
                data: code.0.as_ptr(),
                size: code.0.len(),
            },
            aux: RawBlob {
                data: std::ptr::null(),
                size: 0x118,
            },
        };
        let s = summarize_deserialized_compute(&valid);
        assert!(s.has_root);
        assert!(s.state_ok);
        assert!(s.header_ok);
        assert!(s.code_aligned);
        assert_eq!(s.aux_record_count, Some(8));

        let bad = DeserializedComputeDesc {
            root: std::ptr::null(),
            state: RawBlob {
                data: std::ptr::null(),
                size: 0x10,
            },
            header: RawBlob {
                data: std::ptr::null(),
                size: 0x10,
            },
            code: RawBlob {
                data: (&raw const code.0[4]),
                size: 0x184,
            },
            aux: RawBlob {
                data: std::ptr::null(),
                size: 0x20,
            },
        };
        let sb = summarize_deserialized_compute(&bad);
        assert!(!sb.has_root);
        assert!(!sb.state_ok);
        assert!(!sb.header_ok);
        assert!(!sb.code_aligned);
        assert_eq!(sb.aux_record_count, None);
    }

    #[test]
    fn prepare_call_zeroes_out_and_handles_nulls() {
        let zero = RawBlob {
            data: std::ptr::null(),
            size: 0,
        };
        let desc = DeserializedComputeDesc {
            root: std::ptr::null(),
            state: zero,
            header: zero,
            code: zero,
            aux: zero,
        };

        let mut out: *const c_void = std::ptr::dangling();
        // SAFETY: `desc` and `out` are valid stack locals.
        assert!(unsafe { prepare_call(&desc, &mut out) }.is_some());
        assert!(out.is_null());

        let mut out: *const c_void = std::ptr::dangling();
        // SAFETY: `out` is a valid stack local; `desc` is null.
        assert!(unsafe { prepare_call(std::ptr::null(), &mut out) }.is_none());
        assert!(out.is_null());

        // SAFETY: `desc` is valid; `out` is null.
        assert!(unsafe { prepare_call(&desc, std::ptr::null_mut()) }.is_some());
    }
}
