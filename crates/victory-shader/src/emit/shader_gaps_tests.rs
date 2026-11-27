//! Programs the title's shaders were stopped by, cut down to the construct that stopped them:
//! the skips no lane past an early-out takes (`2da4beb386509b77`), a lane mask parked in two
//! `ttmp` registers dword by dword (`405bfb31f54125ae`), and a buffer view read only for its
//! record count (`ceb5eb68cd99cb07`). Translation only (no GPU).

use super::{lower, never_taken, translate_native};
use crate::gfx10::Fields;
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ir, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
/// `buffer_store_dword v0, off, s[8:11], 0`
const STORE_V0: [u32; 2] = [0x38 << 26 | 0x1C << 18, 0x80 << 24 | 2 << 16];

/// A wave64 compute program of `threads` threads (thread index in v0) with 4 root constants in
/// s0..s3 and a raw UAV's descriptor loaded into s[8:11] from the table in s4, then `body`.
fn native(body: &[u32], threads: u32) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
    ];
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn execz_skips(p: &ir::Program) -> usize {
    p.blocks
        .iter()
        .filter(|b| {
            b.insts
                .last()
                .is_some_and(|i| i.name() == "s_cbranch_execz")
        })
        .count()
}

/// `if (t < s0) { if (any launched lane) { v0 = 1 } store v0 }`: the inner skip tests EXEC
/// ANDed with the launch mask, which no lane past the early-out finds empty.
fn guarded_by_launch_mask() -> NativeProgram {
    native(
        &[
            0xBEA8_047E, // s_mov_b64 s[40:41], exec
            0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
            0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
            0xBF88_0007, // s_cbranch_execz (to s_endpgm)
            0xBE94_2428, // s_and_saveexec_b64 s[20:21], s[40:41]
            0xBF88_0001, // s_cbranch_execz (past the move)
            0x7E00_0281, // v_mov_b32 v0, 1
            0xBEFE_0414, // s_mov_b64 exec, s[20:21]
            STORE_V0[0],
            STORE_V0[1],
            0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        ],
        64,
    )
}

#[test]
fn a_skip_no_lane_past_the_early_out_takes_goes_with_the_early_out() {
    let native = guarded_by_launch_mask();
    let p = crate::spill::build(&native).expect("ir");
    assert_eq!(execz_skips(&p), 2);
    let st = structure::structure(&p).expect("structure");
    let never = never_taken(&st);
    // the structure shows the inner skip is taken by no lane that reaches it
    assert_eq!(never.iter().filter(|&&n| n).count(), 1);
    // without that knowledge the lanes that would have left take it, each on its own
    let kept = ir::without_early_outs(&p, &vec![false; never.len()]).expect("early-out left out");
    assert_eq!(execz_skips(&kept), 1);
    let both = ir::without_early_outs(&p, &never).expect("early-out left out");
    assert_eq!(execz_skips(&both), 0);
    // and the program with them is code every lane runs
    let st = structure::structure(&both).expect("structure");
    assert!(
        st.exits
            .iter()
            .all(|e| !matches!(e, structure::LaneExit::Branch { .. }))
    );
}

#[test]
fn a_skip_that_lanes_take_stays() {
    // the same with the inner skip on a lane test: some lanes do take it
    let native = native(
        &[
            0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
            0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
            0xBF88_0008, // s_cbranch_execz (to s_endpgm)
            0x7D88_0001, // v_cmp_gt_u32 vcc, s1, v0
            0xBE94_246A, // s_and_saveexec_b64 s[20:21], vcc
            0xBF88_0001, // s_cbranch_execz (past the move)
            0x7E00_0281, // v_mov_b32 v0, 1
            0xBEFE_0414, // s_mov_b64 exec, s[20:21]
            STORE_V0[0],
            STORE_V0[1],
            0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        ],
        64,
    );
    let p = crate::spill::build(&native).expect("ir");
    let st = structure::structure(&p).expect("structure");
    assert!(never_taken(&st).iter().all(|&n| !n));
    assert_eq!(
        execz_skips(&ir::without_early_outs(&p, &never_taken(&st)).expect("early-out left out")),
        1
    );
}

/// `if (s0 == 0) { v0 = 1; store v0 }` with the condition held as a lane mask
/// (`s_cselect_b64 s[34:35], exec, 0`) that is parked in `ttmp3`/`ttmp4` dword by dword and
/// taken back the same way before `s_and_b64 vcc, exec, s[34:35]; s_cbranch_vccz` asks it.
/// `swap`: the two dwords parked the other way round.
fn parked_mask(swap: bool) -> NativeProgram {
    let (lo, hi) = if swap {
        (0xBEEF_0323, 0xBEF0_0322)
    } else {
        (0xBEEF_0322, 0xBEF0_0323)
    };
    native(
        &[
            0xBF06_8000, // s_cmp_eq_u32 s0, 0
            0x85A2_807E, // s_cselect_b64 s[34:35], exec, 0
            lo,          // s_mov_b32 ttmp3, s34 (s35 with `swap`)
            hi,          // s_mov_b32 ttmp4, s35 (s34 with `swap`)
            0xBEA2_0380, // s_mov_b32 s34, 0
            0xBEA3_0380, // s_mov_b32 s35, 0
            0xBEA2_036F, // s_mov_b32 s34, ttmp3
            0xBEA3_0370, // s_mov_b32 s35, ttmp4
            0x87EA_227E, // s_and_b64 vcc, exec, s[34:35]
            0xBF86_0003, // s_cbranch_vccz (past the store)
            0x7E00_0281, // v_mov_b32 v0, 1
            STORE_V0[0],
            STORE_V0[1],
        ],
        64,
    )
}

/// The lane masks of the pair `s_cselect_b64` writes and of the pair `s_and_b64` reads.
fn parked_and_restored(swap: bool) -> (Option<structure::Bool>, Option<structure::Bool>) {
    use crate::ir::Reg;
    let native = parked_mask(swap);
    let p = crate::spill::build(&native).expect("ir");
    let st = structure::structure(&p).expect("structure");
    let insts = || p.blocks.iter().flat_map(|b| &b.insts);
    let written = insts()
        .find(|i| i.name() == "s_cselect_b64")
        .and_then(|i| i.write(Reg::S(34)))
        .expect("s_cselect");
    let read = insts()
        .find(|i| i.name() == "s_and_b64")
        .and_then(|i| i.read(Reg::S(34)))
        .expect("s_and_b64");
    (
        st.masks.get(&written).copied(),
        st.masks.get(&read).copied(),
    )
}

#[test]
fn a_lane_mask_parked_dword_by_dword_is_the_mask_it_was() {
    let (written, read) = parked_and_restored(false);
    assert!(written.is_some());
    assert_eq!(written, read);
    translate_native(&parked_mask(false)).expect("the parked mask is the s_cselect's");
}

#[test]
fn halves_of_a_mask_parked_the_other_way_round_are_not_the_pair() {
    let (written, read) = parked_and_restored(true);
    assert!(written.is_some());
    assert_ne!(written, read);
}

#[test]
fn a_buffer_view_read_only_for_its_record_count_gets_the_hosts() {
    let mut native = native(
        &[
            0x3D << 26 | 0x02 << 18 | 16 << 6 | 2,
            125 << 25 | 0x20, // s_load_dwordx4 s[16:19], s[4:5], 0x20
            0xBF06_8012,      // s_cmp_eq_u32 s18, 0
            0xBF85_0003,      // s_cbranch_scc1 (past the store)
            0x7E00_0281,      // v_mov_b32 v0, 1
            STORE_V0[0],
            STORE_V0[1],
        ],
        64,
    );
    native.used.push(UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Table,
        table_offset_dwords: 8,
        lower_bound: 1,
        stride: 16,
        dwords: 4,
        user_data: 4,
        slot: 1,
    });
    let p = crate::spill::build(&native).expect("ir");
    assert!(
        p.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .any(|i| matches!(i.fields, Fields::Smem { .. }))
    );
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let r = translate_native(&native);
    lower::TEST_HOST_WAVE.set(was);
    r.expect("the record count from the host view");
}

/// A loop each lane leaves on its own (`a1bf269d5b19f025`'s counting loops): lane t runs the
/// body t + 1 times, the latch takes EXEC back to the lanes still counting.
fn counting_loop() -> NativeProgram {
    native(
        &[
            0x7E02_0280, // v_mov_b32 v1, 0
            0x4A04_0081, // v_add_nc_u32 v2, 1, v0
            0xBE98_047E, // s_mov_b64 s[24:25], exec
            0xBE94_047E, // s_mov_b64 s[20:21], exec
            0xBF82_0001, // s_branch (to the header)
            0x87FE_1416, // s_and_b64 exec, s[22:23], s[20:21]   (latch)
            0x4A02_0281, // v_add_nc_u32 v1, 1, v1                (header)
            0xBE96_047E, // s_mov_b64 s[22:23], exec
            0x7DA4_0501, // v_cmpx_eq_u32 exec, v1, v2
            0x8994_7E14, // s_andn2_b64 s[20:21], s[20:21], exec
            0xBF85_FFFA, // s_cbranch_scc1 (to the latch)
            0xBEFE_0418, // s_mov_b64 exec, s[24:25]
            0x7E00_0301, // v_mov_b32 v0, v1
            STORE_V0[0],
            STORE_V0[1],
        ],
        64,
    )
}

#[test]
fn the_body_of_a_loop_lanes_leave_on_their_own_is_not_code_every_lane_runs() {
    let native = counting_loop();
    let p = crate::spill::build(&native).expect("ir");
    let st = structure::structure(&p).expect("structure");
    let header = p
        .blocks
        .iter()
        .position(|b| b.insts.iter().any(|i| i.name() == "v_cmpx_eq_u32"))
        .expect("the loop");
    let branch = (0..p.blocks.len())
        .find(|&b| {
            p.blocks[b]
                .insts
                .last()
                .is_some_and(|i| i.name() == "s_cbranch_scc1")
        })
        .expect("the loop's branch");
    // the two halves of a console wave run the header a different number of times: meeting
    // there needs the loop taken wave-wide
    assert_eq!(lower::branches_around(&st, &[header]), Some(vec![branch]));
}
