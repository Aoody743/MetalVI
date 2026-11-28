//! `lower::lds_start` (upstream f299fba): a group's LDS starts as the program's first store of
//! constants leaves it, for programs whose thread 0 sets the LDS up behind an early-out.
//! Translation only; the Metal run is in victory-gpu's `metal_half_sync_repro_tests`.

use super::shader_gaps_tests::native;
use super::{lower, translate_native};
use crate::native::{NativeProgram, StageIo};

/// 64 threads, one console wave, 512 bytes of LDS; the threads from s0 up are live. Thread 0,
/// if it is live, sets the LDS up (dword 0 = 0, dword 1 = all ones); every live thread then
/// takes the minimum of dword 1 and its own index, reads it back and stores it to the UAV.
pub(crate) const NEAREST_LIVE_THREAD: [u32; 17] = [
    0x7D86_0000, // v_cmp_le_u32 vcc, s0, v0
    0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
    0x7E0A_0280, // v_mov_b32 v5, 0
    0x7E0C_02C1, // v_mov_b32 v6, -1
    0xBE8E_047E, // s_mov_b64 s[14:15], exec
    0x7DA4_0080, // v_cmpx_eq_u32 exec, 0, v0
    0xD838_0100,
    0x0006_0505, // ds_write2_b32 v5, v5, v6 offset1:1
    0xBEFE_040E, // s_mov_b64 exec, s[14:15]
    0xD81C_0004,
    0x0000_0005, // ds_min_u32 v5, v0 offset:4
    0xD8D8_0004,
    0x0700_0005, // ds_read_b32 v7, v5 offset:4
    0x7E00_0307, // v_mov_b32 v0, v7
    0x38 << 26 | 0x1C << 18,
    0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    0xBEFE_040C,          // s_mov_b64 exec, s[12:13]
];

fn with_lds(body: &[u32]) -> NativeProgram {
    let mut native = native(body, 64);
    if let StageIo::Cs(cs) = &mut native.io {
        cs.rsrc2 |= 1 << 15; // 512 bytes of LDS
    }
    native
}

fn start(words: &[u32]) -> Vec<(u32, u32)> {
    let native = with_lds(words);
    let p = crate::spill::build(&native).expect("ir");
    lower::lds_start(&p)
}

#[test]
fn the_set_up_behind_an_early_out_is_found() {
    assert_eq!(start(&NEAREST_LIVE_THREAD), [(0, 0), (1, u32::MAX)]);
}

/// Only a plain store of constants that stands before every other LDS access is a set-up.
#[test]
fn an_lds_start_is_a_first_store_of_constants_only() {
    // an atomic first: the program counts on what is there
    assert!(
        start(&[
            0x7E0A_0280,
            0x7E0C_02C1,
            0xD81C_0004,
            0x0000_0005,
            0xD838_0100,
            0x0006_0505
        ])
        .is_empty()
    );
    // a thread's own value: not a constant
    assert!(start(&[0x7E0A_0280, 0xD834_0000, 0x0000_0005]).is_empty());
    // one dword, at a byte offset
    assert_eq!(
        start(&[0x7E0A_0280, 0x7E0C_02C1, 0xD834_0008, 0x0000_0605]),
        [(2, u32::MAX)]
    );
    // no LDS access at all
    assert!(start(&[0x7E0A_0280]).is_empty());
}

/// On a 32-lane host the switch adds the two stores (and nothing else); off, the translation is
/// the one without them. The program's own LDS is 128 dwords: the stores stay below it and the
/// half-sync, exchange and `halves` dwords lie past it.
#[test]
fn on_a_32_lane_host_the_set_up_is_stored_at_the_groups_start() {
    let native = with_lds(&NEAREST_LIVE_THREAD);
    let shader_of = |on: bool| {
        let was = lower::TEST_LDS_START.replace(Some(on));
        let host = lower::TEST_HOST_WAVE.replace(Some(32));
        let r = translate_native(&native).expect("translates");
        let wave = super::last_wave();
        lower::TEST_HOST_WAVE.set(host);
        lower::TEST_LDS_START.set(was);
        assert_eq!(wave, Some(32));
        r
    };
    let (with, without) = (shader_of(true), shader_of(false));
    assert_ne!(
        with.dxil, without.dxil,
        "two stores more in the entry block"
    );
    assert!(start(&NEAREST_LIVE_THREAD).iter().all(|&(d, _)| d < 128));
}
