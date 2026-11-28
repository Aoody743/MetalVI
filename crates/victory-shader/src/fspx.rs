//! Precompiled console compute programs (`FSPX`), as the AMD upscaler of
//! Series S (`amd_fidelityfx_upscaler_xbox.dll`, FSR4) ships them: a DXBC
//! container whose `ACCC` part holds the GFX10 code and its binding records
//! instead of a `DXIL` part. The same sections arrive one by one through
//! `DeserializeComputePipelineStateX` (device method 0x218, which victory
//! refuses so the title passes the whole container).
//!
//! ```text
//! "FSPX" u32 version (1) u32 sections u32 total size
//! sections × { u32 kind, u32 offset, u32 size, u32 0 }   (from 0x10)
//! kind 0  0x1c bytes: 7 dwords; [3] threads x | y << 16,
//!         [4] threads z | flags << 16 (bit 20: wave32),
//!         [5] COMPUTE_PGM_RSRC1, [6] COMPUTE_PGM_RSRC2
//! kind 1  0x14 bytes: an id and the constant 3
//! kind 6  GFX10 code, 256-byte aligned, padded with s_code_end
//! kind 7  header 0x18 bytes (u32 0x403, payload bytes, code dwords, 0, 0,
//!         0x41 records << 16 | 0x440), then 0x20-byte records:
//!         0x41: one descriptor the code reads, the 20 bytes of an FSSM
//!               "used" entry ([`UsedResource`]), then 8 bytes (-1, 0)
//!         0x42: the shape of each view (UAV or not, dimension); not needed
//!         0x43: the shader hash (as the container's HASH part)
//! ```
//!
//! What [`NativeProgram`] needs and FSPX does not hold:
//! * The root layout (FSSM keeps a table of every root parameter and table
//!   range and the user SGPRs holding it). It is made from the 0x41 records:
//!   each descriptor read names its root parameter (slot), the user SGPR
//!   that holds it, and its table offset and register, which is what an
//!   FSSM layout entry of a table range says. A table takes one SGPR, a
//!   root descriptor two, root constants their dwords, an immediate sampler
//!   none (as the title's FSSM layouts have them).
//! * The VGPR count: only `COMPUTE_PGM_RSRC1`'s granule field (4 VGPRs a
//!   granule on 64 lanes, 8 on 32), an upper bound, which is what the count
//!   is used for (the bound of indexed register access).
//!
//! The wave32 bit was checked three ways on the upscaler's 29 programs:
//! PSV0's expected lane count (0x20 or 0x40) where given, and the exec mask
//! instructions of the code (`s_andn2_b32 exec`, `s_and_saveexec_b32` for 32
//! lanes; their `_b64` forms for 64), calibrated on FSSM programs whose
//! wave32 flag is known.

use crate::ShaderError::{self, Malformed};
use crate::native::{ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource};

pub const MAGIC: &[u8; 4] = b"FSPX";
const VERSION: u32 = 1;
const KIND_STATE: u32 = 0;
const KIND_CODE: u32 = 6;
const KIND_BINDINGS: u32 = 7;
const STATE_BYTES: usize = 0x1c;
const BINDINGS_HEADER: usize = 0x18;
const RECORD_BYTES: usize = 0x20;
const TAG_USED: u32 = 0x41;
const FLAG_WAVE32: u32 = 1 << 20;
const MAX_SECTIONS: u32 = 16;

fn u32_at(data: &[u8], at: usize) -> Result<u32, ShaderError> {
    data.get(at..at + 4)
        .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
        .ok_or(Malformed("FSPX data out of bounds"))
}

/// Whether `part` (an `ACCC` part's data) is an FSPX program.
pub fn is_fspx(part: &[u8]) -> bool {
    part.get(0..4) == Some(MAGIC)
}

/// The sections of an FSPX blob, as (kind, bytes).
fn sections(part: &[u8]) -> Result<Vec<(u32, &[u8])>, ShaderError> {
    if !is_fspx(part) {
        return Err(Malformed("ACCC part lacks FSPX magic"));
    }
    if u32_at(part, 4)? != VERSION {
        return Err(Malformed("unknown FSPX version"));
    }
    let count = u32_at(part, 8)?;
    if count > MAX_SECTIONS {
        return Err(Malformed("implausible FSPX section count"));
    }
    (0..count as usize)
        .map(|i| {
            let at = 0x10 + 16 * i;
            let (kind, offset, size) = (u32_at(part, at)?, u32_at(part, at + 4)? as usize, u32_at(part, at + 8)? as usize);
            let bytes = part.get(offset..offset.checked_add(size).ok_or(Malformed("FSPX section out of bounds"))?)
                .ok_or(Malformed("FSPX section out of bounds"))?;
            Ok((kind, bytes))
        })
        .collect()
}

/// The 0x41 records of the bindings section: the descriptors the code reads.
fn used(bindings: &[u8]) -> Result<Vec<UsedResource>, ShaderError> {
    let records = bindings.get(BINDINGS_HEADER..).ok_or(Malformed("FSPX bindings header out of bounds"))?;
    if !records.len().is_multiple_of(RECORD_BYTES) {
        return Err(Malformed("FSPX bindings are not whole records"));
    }
    let mut out = Vec::new();
    for r in records.chunks_exact(RECORD_BYTES) {
        if u32_at(r, 0)? != TAG_USED {
            continue;
        }
        let e = &r[4..];
        out.push(UsedResource {
            kind: match e[0] {
                0 => ResourceType::Srv,
                1 => ResourceType::Sampler,
                2 => ResourceType::Cbv,
                3 => ResourceType::Uav,
                n => ResourceType::Other(n),
            },
            root: RootType::from_raw(e[1].into()),
            table_offset_dwords: u32_at(e, 4)?,
            lower_bound: u32_at(e, 8)?,
            stride: u32_at(e, 12)?,
            dwords: e[16],
            user_data: e[17],
            slot: e[18],
        });
    }
    Ok(out)
}

/// The root layout the used descriptors imply (see the module notes): one
/// entry per table range read, root descriptor or constants, immediate
/// sampler, in record order, without repeats.
fn layout(used: &[UsedResource]) -> Vec<RootParam> {
    let mut out: Vec<RootParam> = Vec::new();
    for u in used {
        let p = match u.root {
            RootType::Table => RootParam {
                root: RootType::Table,
                table_offset_dwords: u.table_offset_dwords,
                sgprs: 1,
                base_register: u.lower_bound,
                user_data: u.user_data,
                slot: u.slot,
            },
            RootType::Descriptor => RootParam {
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: u.lower_bound,
                user_data: u.user_data,
                slot: u.slot,
            },
            RootType::Constants => RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: u32::from(u.dwords).max(1),
                base_register: u.lower_bound,
                user_data: u.user_data,
                slot: u.slot,
            },
            root => RootParam {
                root,
                table_offset_dwords: 0,
                sgprs: 0,
                base_register: u.lower_bound,
                user_data: u.user_data,
                slot: u.slot,
            },
        };
        if !out.contains(&p) {
            out.push(p);
        }
    }
    out
}

/// The compute program of an FSPX blob (an `ACCC` part's data).
pub fn native_program(part: &[u8]) -> Result<NativeProgram, ShaderError> {
    let secs = sections(part)?;
    let find = |kind| secs.iter().find(|(k, _)| *k == kind).map(|(_, b)| *b);
    let state = find(KIND_STATE).ok_or(Malformed("FSPX without its compute state"))?;
    if state.len() != STATE_BYTES {
        return Err(Malformed("FSPX compute state of an unexpected size"));
    }
    let code = find(KIND_CODE).ok_or(Malformed("FSPX without code"))?;
    let bindings = find(KIND_BINDINGS).ok_or(Malformed("FSPX without bindings"))?;
    let w = |i: usize| u32_at(state, 4 * i);
    let threads = [w(3)? & 0xffff, w(3)? >> 16, w(4)? & 0xffff];
    if threads.contains(&0) || threads.iter().product::<u32>() > 1024 {
        return Err(Malformed("implausible FSPX thread-group size"));
    }
    let wave64 = w(4)? & FLAG_WAVE32 == 0;
    let rsrc1 = w(5)?;
    let rsrc2 = w(6)?;
    let vgprs = ((rsrc1 & 0x3f) + 1) * if wave64 { 4 } else { 8 };
    // The code as stored, without the s_code_end padding the header's
    // dword count leaves out.
    let code_dwords = u32_at(bindings, 8)? as usize;
    let code = code.get(..code_dwords * 4).unwrap_or(code);
    let used = used(bindings)?;
    Ok(NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: vgprs.min(256) as u16,
        layout: layout(&used),
        used,
        io: StageIo::Cs(ComputeIo { rsrc2, threads }),
        code: code.to_vec(),
    })
}

#[cfg(test)]
#[path = "fspx_tests.rs"]
pub(crate) mod tests;
