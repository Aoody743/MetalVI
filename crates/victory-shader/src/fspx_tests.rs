//! FSPX against the Series S upscaler's own containers, read at test time
//! from the game dump (`VICTORY_FSPX_DLL`, else the usual dump path); the
//! tests pass without checking anything when it is absent (the DLL is the
//! title's and is not copied into the repository).

use super::*;
use crate::{Container, FourCC};

/// The DXBC containers inside the upscaler DLL, in file order.
pub(crate) fn upscaler_containers() -> Option<Vec<Vec<u8>>> {
    let path = std::env::var_os("VICTORY_FSPX_DLL").map(std::path::PathBuf::from).unwrap_or_else(|| {
        "/Users/aoody743/Documents/GTA VI/runtime-env/game/amd_fidelityfx_upscaler_xbox.dll".into()
    });
    let data = std::fs::read(path).ok()?;
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(at) = data[i..].windows(4).position(|w| w == b"DXBC").map(|p| p + i) {
        let size = data.get(at + 24..at + 28).map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize);
        match size {
            Some(size) if size > 32 && at + size <= data.len() && Container::parse(&data[at..at + size]).is_ok() => {
                out.push(data[at..at + size].to_vec());
                i = at + size;
            }
            _ => i = at + 4,
        }
    }
    Some(out)
}

#[test]
fn upscaler_programs_parse() {
    let Some(all) = upscaler_containers() else {
        eprintln!("SKIP: no upscaler DLL");
        return;
    };
    assert_eq!(all.len(), 29, "the upscaler's containers");
    let mut wave32 = 0;
    for (k, bytes) in all.iter().enumerate() {
        let c = Container::parse(bytes).unwrap();
        assert!(c.part(FourCC::DXIL).is_none());
        let accc = c.part(FourCC::ACCC).expect("an ACCC part");
        let p = native_program(accc.data).unwrap_or_else(|e| panic!("#{k}: {e}"));
        assert_eq!(p.stage, HwStage::Cs);
        assert!(!p.code.is_empty() && p.code.len() % 4 == 0);
        assert!(!p.used.is_empty(), "#{k}: no descriptors");
        // Every used descriptor has a layout entry at its user SGPR.
        for u in &p.used {
            assert!(p.layout.iter().any(|l| l.slot == u.slot && l.user_data == u.user_data), "#{k}: {u:?}");
        }
        let StageIo::Cs(cs) = &p.io else { panic!("compute io") };
        // The user SGPRs hold every root parameter the layout places.
        for l in p.layout.iter().filter(|l| l.sgprs > 0) {
            assert!(u32::from(l.user_data) + l.sgprs <= u32::from(cs.user_sgprs()), "#{k}: {l:?} past {} user SGPRs", cs.user_sgprs());
        }
        // PSV0's expected lane count, where the container gives one.
        if let Some(psv) = c.part(FourCC::PSV0)
            && let Some(lanes) = psv.data.get(4 + 32..4 + 36).map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            && (lanes == 32 || lanes == 64)
        {
            assert_eq!(p.wave64, lanes == 64, "#{k}: PSV0 says {lanes} lanes");
        }
        wave32 += usize::from(!p.wave64);
    }
    println!("upscaler: {} programs, {wave32} of 32 lanes", all.len());
    assert_eq!(wave32, 24);
}

/// The record layout against one container decoded by hand (`#0`).
#[test]
fn first_upscaler_program_by_hand() {
    let Some(all) = upscaler_containers() else {
        eprintln!("SKIP: no upscaler DLL");
        return;
    };
    let c = Container::parse(&all[0]).unwrap();
    let p = native_program(c.part(FourCC::ACCC).unwrap().data).unwrap();
    assert!(p.wave64);
    assert_eq!(p.code.len(), 0xd00);
    let StageIo::Cs(cs) = p.io else { panic!() };
    assert_eq!(cs.threads, [64, 1, 1]);
    assert_eq!(cs.user_sgprs(), 8);
    assert_eq!(p.used.len(), 4);
    assert_eq!(p.used[0], UsedResource { kind: ResourceType::Uav, root: RootType::Table, table_offset_dwords: 0, lower_bound: 11, stride: 4, dwords: 8, user_data: 4, slot: 0 });
    assert_eq!(p.used[3], UsedResource { kind: ResourceType::Cbv, root: RootType::Descriptor, table_offset_dwords: 0, lower_bound: 0, stride: 0, dwords: 2, user_data: 6, slot: 2 });
    assert_eq!(p.layout[3], RootParam { root: RootType::Descriptor, table_offset_dwords: 0, sgprs: 2, base_register: 0, user_data: 6, slot: 2 });
    assert_eq!(p.layout.len(), 4);
}

#[test]
fn damaged_blobs_are_refused() {
    assert!(native_program(b"FSPX").is_err());
    assert!(native_program(b"XXXX\x01\0\0\0\0\0\0\0\0\0\0\0").is_err());
    let mut b = vec![0u8; 0x20];
    b[..4].copy_from_slice(MAGIC);
    b[4] = 1;
    b[8] = 1; // one section, pointing past the end
    b[0x14] = 0xf0;
    b[0x18] = 0x10;
    assert!(native_program(&b).is_err());
}
