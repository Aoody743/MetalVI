//! M3a: a console wave64 compute program run as two host waves of 32
//! (`half_sync`), translated by the real translator, converted by MSC and run
//! through `MetalBackend::run` as the game runs it. The program is the shape of
//! `2561af9577c47289`'s prologue and first reduction: lanes past a count leave
//! (`v_cmpx` + `s_cbranch_execz` to `s_endpgm`), then a whole-wave min over the
//! live lanes (`s_or_saveexec -1`, DPP, `v_permlanex16`, `v_readlane` 32 and 0).
//!
//! - Both halves present (64 threads a group): every count gives the console's
//!   answer and no half-wave sync times out, also when one half has no live lane.
//! - One half missing (the same pipeline run with 32 threads a group): every
//!   sync site times out once per group; the time is sites x 65,536 polls.

use super::*;
use victory_metal::ir::DescriptorEntry;
use victory_shader::native::{ComputeIo, HwStage, NativeProgram, ResourceType, RootParam as NativeRoot, RootType, StageIo, UsedResource};

/// The program body (see the module comment); s0 is the count of live lanes.
const BODY: &[u32] = &[
    0x3404_0082, // v_lshlrev_b32 v2, 2, v0
    0x7E02_0280, // v_mov_b32 v1, 0
    0x7DA8_0000, // v_cmpx_gt_u32 s0, v0
    0xBEC6_047E, // s_mov_b64 s[70:71], exec
    0xBF88_0019, // s_cbranch_execz END
    0x4C82_00BF, // v_sub_nc_u32 v65, 63, v0
    0xBE8C_25C1, // s_or_saveexec_b64 s[12:13], -1
    0x878E_0C46, // s_and_b64 s[14:15], s[70:71], s[12:13]
    0xD501_0003, 0x003A_82C1, // v_cndmask_b32_e64 v3, -1, v65, s[14:15]
    0x2606_06E9, 0xDE54_C103, // v_min_u32 v3, v3, v3 dpp8:[1,0,3,2,5,4,7,6]
    0x2606_06E9, 0x9360_1203, // v_min_u32 v3, v3, v3 dpp8:[2,2,0,0,6,6,4,4]
    0x2606_06E9, 0x0009_2403, // v_min_u32 v3, v3, v3 dpp8:[4,4,4,4,0,0,0,0]
    0x2606_06FA, 0xFF09_4003, // v_min_u32 v3, v3, v3 row_mirror bound_ctrl:1
    0xD778_0004, 0x0201_0103, // v_permlanex16_b32 v4, v3, 0, 0
    0x2606_0704, // v_min_u32 v3, v4, v3
    0xD760_0010, 0x0001_4103, // v_readlane_b32 s16, v3, 32
    0x2606_0610, // v_min_u32 v3, s16, v3
    0xD760_003E, 0x0001_0103, // v_readlane_b32 s62, v3, 0
    0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    0x7E02_023E, // v_mov_b32 v1, s62
    0xE070_1000, 0x8002_0102, // buffer_store_dword v1, v2, s[8:11], 0 offen
];

fn program() -> NativeProgram {
    program_of(BODY, true)
}

/// A loop that never meets its exit (s0 = 0): `s20` counts up until it equals s0. The
/// budget cuts it at 4,096 iterations; each thread stores what s20 reached.
const RUNAWAY: &[u32] = &[
    0x3404_0082, // v_lshlrev_b32 v2, 2, v0
    0xBE94_0380, // s_mov_b32 s20, 0
    0x8014_8114, // LOOP: s_add_u32 s20, s20, 1
    0xBF06_0014, // s_cmp_eq_u32 s20, s0
    0xBF84_FFFD, // s_cbranch_scc0 LOOP
    0x7E02_0214, // v_mov_b32 v1, s20
    0xE070_1000, 0x8002_0102, // buffer_store_dword v1, v2, s[8:11], 0 offen
];

fn program_of(body: &[u32], wave64: bool) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    ];
    words.extend_from_slice(body);
    words.push(0xBF81_0000); // END: s_endpgm
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 72,
        layout: vec![
            NativeRoot { root: RootType::Constants, table_offset_dwords: 0, sgprs: 4, base_register: 0, user_data: 0, slot: 0 },
            NativeRoot { root: RootType::Table, table_offset_dwords: 0, sgprs: 1, base_register: 0, user_data: 4, slot: 1 },
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
        io: StageIo::Cs(ComputeIo { rsrc2: 5 << 1 | u32::from(body == LDS_LOOP) << 15, threads: [64, 1, 1] }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

struct Run {
    out: Vec<u32>,
    timeouts: u64,
    sites: Vec<(u32, u32)>,
    ms: f64,
}

/// Runs the translated program over `groups` groups of `threads` host threads
/// with `live` live console lanes a group.
fn run(backend: &MetalBackend, shader: &victory_shader::emit::HostShader, threads: u64, groups: u64, live: u32) -> Run {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
    const BYTES: usize = 4096;
    let mem = unsafe { VirtualAlloc(std::ptr::null(), BYTES, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
    assert!(!mem.is_null());
    let out = unsafe { backend.device.wrap(mem, BYTES as u64) }.unwrap();
    // MSC's kernels take the view and sampler heaps even when they read none: null entries.
    let heap_mem = unsafe { VirtualAlloc(std::ptr::null(), 65536, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
    let heap = unsafe { backend.device.wrap(heap_mem, 65536) }.unwrap();
    // The program's UAV: a table at heap slot 7 (user data of kind Table), one raw view.
    const SLOT: u32 = 7;
    unsafe { heap_mem.cast::<DescriptorEntry>().add(SLOT as usize).write(DescriptorEntry::buffer(out.gpu_address, BYTES as u64)) };
    backend.chunks().residency().add(&[out.handle(), heap.handle()]);
    backend.chunks().residency().commit();
    let (pso, conv) = backend.device.compute_pipeline_dxil(&shader.dxil, &root_params(shader), &static_samplers(shader)).unwrap();
    assert_eq!(conv.threadgroup, [64, 1, 1]);
    let mp = Arc::new(MetalCompute {
        cost_ns: Default::default(),
        name: format!("half-sync repro ({threads} threads)"),
        pso,
        threadgroup: [threads, 1, 1],
        top_level: conv.top_level.clone(),
    });
    let mut top = conv.top_level.clone();
    let mut tables = Vec::new();
    for (i, r) in root_layout(shader).into_iter().enumerate() {
        match r {
            RootSlot::Constants(n) => {
                let mut words = vec![0u32; n as usize];
                for u in &shader.user_data {
                    match (u.root, u.slot, u.dword) {
                        (RootType::Constants, 0, 0) => words[u.index as usize] = live,
                        (RootType::Table, 1, 0) => words[u.index as usize] = SLOT * 32,
                        _ => {}
                    }
                }
                assert!(top.set_constants(victory_shader::emit::USER_DATA_SPACE, 0, &words));
            }
            RootSlot::TableUav(1) => tables.push((top.offset_of(i).unwrap(), DescriptorEntry::buffer(out.gpu_address, BYTES as u64))),
            RootSlot::Uav(s) if s == victory_shader::emit::SLOT_DIAG => assert!(top.set_u64(i, diag::gpu(backend))),
            other => panic!("unexpected root slot {other:?}"),
        }
    }
    let op = Arc::new(MetalOp::Dispatch(MetalCmd {
        pso: mp,
        heap: Some(heap.clone()),
        sampler_heap: Some(heap.clone()),
        top: top.bytes,
        groups: [groups, 1, 1],
        tables,
        loop_counts: None,
    }));
    let before = diag::HALF_SYNC_TIMEOUTS.load(std::sync::atomic::Ordering::Relaxed);
    diag::LAST.lock().unwrap().clear();
    let t = std::time::Instant::now();
    assert!(backend.run(&[op]), "submission failed");
    let _wall_ms = t.elapsed().as_secs_f64() * 1e3;
    let ms = diag::LAST_GPU_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
    let timeouts = diag::HALF_SYNC_TIMEOUTS.load(std::sync::atomic::Ordering::Relaxed) - before;
    let sites = diag::LAST.lock().unwrap().iter().copied().filter(|c| c.0 >= victory_shader::emit::DIAG_SITES).collect();
    let words = unsafe { std::slice::from_raw_parts(mem.cast::<u32>(), 64) }.to_vec();
    Run { out: words, timeouts, sites, ms }
}

#[test]
fn half_sync_timeouts_are_counted_and_absent_with_both_halves() {
    let _pool = victory_metal::Pool::new();
    victory_shader::emit::set_host_max_wave(32);
    let shader = victory_shader::emit::translate_native(&program()).expect("translate");
    assert!(shader.root_uavs.contains(&victory_shader::emit::SLOT_DIAG), "no half_sync was emitted: {:?}", shader.root_uavs);
    eprintln!("REPRO layout {:?} user_data {:?} views {:?}", root_layout(&shader), shader.user_data, shader.root_indexed_views);
    if let Some(dir) = std::env::var_os("REPRO_DXIL") {
        std::fs::write(std::path::Path::new(&dir).join("repro.dxil"), &shader.dxil).unwrap();
    }
    let backend = MetalBackend::new().unwrap();
    // Both halves: the console's answer for every count of live lanes, including
    // counts that leave the upper half (or both) with no live lane.
    for live in [0u32, 1, 10, 31, 32, 33, 40, 63, 64] {
        let r = run(&backend, &shader, 64, 1, live);
        let want = 64u32.saturating_sub(live);
        for (t, &got) in r.out.iter().enumerate() {
            let expect = if (t as u32) < live { want } else { 0 };
            assert_eq!(got, expect, "live {live}, lane {t}: {:?}", r.out);
        }
        assert_eq!(r.timeouts, 0, "live {live}: half-wave sync timed out at {:?}", r.sites);
        eprintln!("REPRO both-halves live={live:2}: correct, 0 timeouts, {:.3} ms", r.ms);
    }
}

#[test]
fn a_missing_half_times_out_at_every_other_site() {
    let _pool = victory_metal::Pool::new();
    victory_shader::emit::set_host_max_wave(32);
    let shader = victory_shader::emit::translate_native(&program()).expect("translate");
    let backend = MetalBackend::new().unwrap();
    let _warm = run(&backend, &shader, 64, 1, 40);
    let base = run(&backend, &shader, 64, 1, 40);
    assert_eq!(base.timeouts, 0);
    // One host wave a group: the other half never comes. The lone half's own
    // arrival at the next site completes that site's pair (target = even + 2),
    // so every other site times out: sites / 2 timeouts of 65,536 polls a group.
    for groups in [1u64, 1, 4, 64] {
        let r = run(&backend, &shader, 32, groups, 10);
        assert!(!r.sites.is_empty() && r.timeouts == groups * r.sites.len() as u64, "timeouts {} at {:?}", r.timeouts, r.sites);
        let polls = 65536.0 * r.sites.len() as f64;
        eprintln!(
            "REPRO one-half groups={groups:2}: {} timeouts at sites {:?}; GPU {:.3} ms (both halves {:.3} ms) = {:.1} ns a poll",
            r.timeouts, r.sites, r.ms, base.ms, (r.ms - base.ms) * 1e6 / polls
        );
    }
}

#[test]
fn a_loop_cut_by_the_budget_reports_its_header_and_counter() {
    use victory_shader::emit::{DIAG_LOOP_BUDGET, DIAG_LOOPS};
    let _pool = victory_metal::Pool::new();
    victory_shader::emit::set_host_max_wave(32);
    let shader = victory_shader::emit::translate_native(&program_of(RUNAWAY, false)).expect("translate");
    assert!(shader.root_uavs.contains(&victory_shader::emit::SLOT_DIAG));
    let backend = MetalBackend::new().unwrap();
    let r = run(&backend, &shader, 64, 1, 0);
    let last = diag::LAST.lock().unwrap().clone();
    eprintln!("REPRO runaway loop: stored {:?}.. diag {last:?}, GPU {:.3} ms", &r.out[..4], r.ms);
    let get = |d: u32| last.iter().find(|c| c.0 == d).map_or(0, |c| c.1);
    assert!(r.out.iter().all(|&v| v == 4096), "{:?}", r.out);
    assert_eq!(get(DIAG_LOOP_BUDGET), 64, "every thread is cut once");
    assert_eq!(get(DIAG_LOOPS), 64, "all at loop 0");
    // the header is the s_add after the 4-dword prologue and 2 body dwords
    assert_eq!(get(DIAG_LOOPS + 1), 6 * 4, "header offset");
    // s20 as the header joined it on the iteration cut
    assert_eq!(get(DIAG_LOOPS + 2), 4095, "s20 at the header when cut");
}

/// 2561af9577c47289's light-bit loop (0x3978..0x6124): take the first set bit
/// of the mask, clear it, leave once it was the last (`s67 == s80`). s21
/// counts iterations: the mask's population count on the console. (s12 for the
/// game's s10, which here holds the UAV descriptor.)
const BIT_LOOP: &[u32] = &[
    0x3404_0082, // v_lshlrev_b32 v2, 2, v0
    0xBE8C_0300, // s_mov_b32 s12, s0
    0xBE95_0380, // s_mov_b32 s21, 0
    0x7EA2_020C, // LOOP: v_mov_b32 v81, s12
    0x7E86_0551, // v_readfirstlane_b32 s67, v81
    0xBED3_1343, // s_ff1_i32_b32 s83, s67
    0x870C_9F53, // s_and_b32 s12, s83, 31
    0x8F50_0C81, // s_lshl_b32 s80, 1, s12
    0x8015_8115, // s_add_u32 s21, s21, 1
    0x890C_4350, // s_xor_b32 s12, s80, s67
    0xBF06_5043, // s_cmp_eq_u32 s67, s80
    0xBF84_FFF7, // s_cbranch_scc0 LOOP
    0x7E02_0215, // v_mov_b32 v1, s21
    0xE070_1000, 0x8002_0102, // buffer_store_dword v1, v2, s[8:11], 0 offen
];

#[test]
fn the_light_bit_loop_runs_once_a_set_bit() {
    let _pool = victory_metal::Pool::new();
    victory_shader::emit::set_host_max_wave(32);
    let shader = victory_shader::emit::translate_native(&program_of(BIT_LOOP, false)).expect("translate");
    let backend = MetalBackend::new().unwrap();
    let mut bad = Vec::new();
    for mask in [1u32, 0xF, 0x4000_0000, 0x8000_0000, 0x8000_0001, 0xC000_0000, u32::MAX] {
        let r = run(&backend, &shader, 64, 1, mask);
        eprintln!("REPRO bit loop mask {mask:#010x}: {} iterations (want {}), GPU {:.3} ms", r.out[0], mask.count_ones(), r.ms);
        if r.out[0] != mask.count_ones() {
            bad.push((mask, r.out[0]));
        }
    }
    assert!(bad.is_empty(), "iterations differ: {bad:x?}");
}

/// [`BIT_LOOP`] as 2561af9577c47289 runs it: a wave64 program as two host waves
/// that exchange (a lane read across halves elsewhere makes it so), so the
/// header's first-lane read goes through the halves' exchange, and inside the
/// loop a per-lane skip (`v_cmpx` + `s_cbranch_execz` to the latch) for the
/// lanes without the light's bit. Lane t holds bit t % 32.
const BIT_LOOP_64: &[u32] = &[
    0x3404_0082, // v_lshlrev_b32 v2, 2, v0
    0xD760_0014, 0x0001_4100, // v_readlane_b32 s20, v0, 32
    0xD51A_0050, 0x0001_0300, // v_lshlrev_b32 v80, v0, 1
    0xBE8C_0300, // s_mov_b32 s12, s0
    0xBE95_0380, // s_mov_b32 s21, 0
    0x7EA6_0280, // v_mov_b32 v83, 0
    0x7EA2_020C, // LOOP: v_mov_b32 v81, s12
    0xBEAE_047E, // s_mov_b64 s[46:47], exec
    0x7E86_0551, // v_readfirstlane_b32 s67, v81
    0xBED3_1343, // s_ff1_i32_b32 s83, s67
    0x870C_9F53, // s_and_b32 s12, s83, 31
    0x8F50_0C81, // s_lshl_b32 s80, 1, s12
    0x36A2_A050, // v_and_b32 v81, s80, v80
    0x7DAA_A280, // v_cmpx_ne_u32 0, v81
    0xBF88_0001, // s_cbranch_execz LATCH
    0x4AA6_A681, // v_add_nc_u32 v83, 1, v83
    0xBEFE_042E, // LATCH: s_mov_b64 exec, s[46:47]
    0x8015_8115, // s_add_u32 s21, s21, 1
    0x890C_4350, // s_xor_b32 s12, s80, s67
    0xBF06_5043, // s_cmp_eq_u32 s67, s80
    0xBF84_FFF1, // s_cbranch_scc0 LOOP
    0x7E02_0215, // v_mov_b32 v1, s21
    0xE070_1000, 0x8002_0102, // buffer_store_dword v1, v2, s[8:11], 0 offen
];

#[test]
fn the_light_bit_loop_runs_once_a_set_bit_in_two_halves() {
    let _pool = victory_metal::Pool::new();
    victory_shader::emit::set_host_max_wave(32);
    let shader = victory_shader::emit::translate_native(&program_of(BIT_LOOP_64, true)).expect("translate");
    assert!(shader.root_uavs.contains(&victory_shader::emit::SLOT_DIAG));
    let backend = MetalBackend::new().unwrap();
    let mut bad = Vec::new();
    for mask in [1u32, 0xF, 0x4000_0000, 0x8000_0000, 0x8000_0001, 0x8000_0100, 0xC000_0000, u32::MAX] {
        let r = run(&backend, &shader, 64, 1, mask);
        let last = diag::LAST.lock().unwrap().clone();
        eprintln!("REPRO bit loop (two halves) mask {mask:#010x}: lanes 0/32 counted {}/{} (want {}), timeouts {}, diag {last:x?}, GPU {:.3} ms",
            r.out[0], r.out[32], mask.count_ones(), r.timeouts, r.ms);
        if r.out[0] != mask.count_ones() || r.out[32] != mask.count_ones() {
            bad.push((mask, r.out[0], r.out[32]));
        }
    }
    assert!(bad.is_empty(), "iterations differ: {bad:x?}");
}

/// The cause of 2561af9577c47289's ~0.5 s dispatches: lanes past a count leave
/// (`execz` to `s_endpgm`), then a loop whose exit asks the lanes
/// (`s_cbranch_vccz` on a compare: some lane in EXEC equal) with a lane read
/// across halves inside. With no lane left the console ends at once; a host
/// wave that walked the body with EXEC clear (the early-out left out) never
/// met the exit and ran the loop to its budget.
const EARLY_OUT_LOOP: &[u32] = &[
    0x3404_0082, // v_lshlrev_b32 v2, 2, v0
    0x7E06_0285, // v_mov_b32 v3, 5
    0x7DA8_0000, // v_cmpx_gt_u32 s0, v0
    0xBF88_000A, // s_cbranch_execz END
    0xBE94_0380, // s_mov_b32 s20, 0
    0x8014_8114, // LOOP: s_add_u32 s20, s20, 1
    0xD760_0015, 0x0001_4103, // v_readlane_b32 s21, v3, 32
    0xD4C2_006A, 0x0000_2A14, // v_cmp_eq_u32 vcc, s20, s21
    0xBF86_FFFA, // s_cbranch_vccz LOOP
    0x7E02_0214, // v_mov_b32 v1, s20
    0xE070_1000, 0x8002_0102, // buffer_store_dword v1, v2, s[8:11], 0 offen
];

fn early_out_loop_case(live: &[u32]) -> Vec<(u32, u32, u64)> {
    use victory_shader::emit::DIAG_LOOP_BUDGET;
    let _pool = victory_metal::Pool::new();
    victory_shader::emit::set_host_max_wave(32);
    let shader = victory_shader::emit::translate_native(&program_of(EARLY_OUT_LOOP, true)).expect("translate");
    let backend = MetalBackend::new().unwrap();
    let mut bad = Vec::new();
    for &live in live {
        let r = run(&backend, &shader, 64, 2, live);
        let last = diag::LAST.lock().unwrap().clone();
        let cut = last.iter().find(|c| c.0 == DIAG_LOOP_BUDGET).map_or(0, |c| c.1);
        let ok = r.out.iter().enumerate().all(|(t, &v)| v == if (t as u32) < live { 5 } else { 0 });
        eprintln!("REPRO early-out over a loop, live {live:2}: {} stores right, {cut} threads cut by the budget, timeouts {}, GPU {:.3} ms",
            if ok { "all" } else { "NOT all" }, r.timeouts, r.ms);
        if !ok || cut != 0 || r.timeouts != 0 {
            bad.push((live, cut, r.timeouts));
        }
    }
    bad
}

/// A console wave with no lane left must not run the loop at all (before
/// `ir::loop_beyond` all 128 threads ran it to the budget, 9.4 ms here).
#[test]
fn an_early_out_over_a_loop_is_taken_by_the_whole_wave() {
    let bad = early_out_loop_case(&[0, 64]);
    assert!(bad.is_empty(), "(live, threads cut, timeouts): {bad:?}");
}

/// A wave with some lanes left: the lanes past the count walk the loop with
/// EXEC clear, and decided its `vccz` exit by their own bit (0 outside EXEC)
/// until the loop exits with half-wave meetings inside were taken by the whole
/// wave: they went round until the budget (108 threads cut).
#[test]
fn an_early_out_over_a_loop_with_some_lanes_left() {
    let bad = early_out_loop_case(&[10, 32, 40]);
    assert!(bad.is_empty(), "(live, threads cut, timeouts): {bad:?}");
}

/// a1bf269d5b19f025's loop at 0xe0: each lane walks its own range (lane t
/// t + 1 times), an LDS atomic add in the loop, the lanes done leave EXEC
/// (`v_cmpx`, `s_andn2`) and the wave goes round while any lane is left
/// (`s_cbranch_scc1`). Every lane then reads the sum, 2080, from LDS.
const LDS_LOOP: &[u32] = &[
    0x3404_0082, // v_lshlrev_b32 v2, 2, v0
    0xBE98_047E, // s_mov_b64 s[24:25], exec
    0x7E10_0280, // v_mov_b32 v8, 0
    0x7E14_0280, // v_mov_b32 v10, 0
    0xD834_0000, 0x0000_0A08, // ds_write_b32 v8, v10
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xBF8A_0000, // s_barrier
    0xBE84_047E, // s_mov_b64 s[4:5], exec
    0x7E0C_0280, // v_mov_b32 v6, 0
    0xD760_0015, 0x0001_4100, // v_readlane_b32 s21, v0, 32 (32: a lane read across halves,
    0x8095_A015, // s_sub_u32 s21, s21, 32              as a1bf's mbcnt sums are)
    0x4A26_0015, // v_add_nc_u32 v19, s21, v0
    0x4A26_2681, // v_add_nc_u32 v19, 1, v19
    0x7E12_0281, // v_mov_b32 v9, 1
    0xBF82_0001, // s_branch HDR
    0x87FE_0406, // LATCH: s_and_b64 exec, s[6:7], s[4:5]
    0x4A0C_0C81, // HDR: v_add_nc_u32 v6, 1, v6
    0xBE86_047E, // s_mov_b64 s[6:7], exec
    0xD800_0000, 0x0000_0908, // ds_add_u32 v8, v9
    0x7DA4_2706, // v_cmpx_eq_u32 v6, v19
    0x8A84_7E04, // s_andn2_b64 s[4:5], s[4:5], exec
    0xBF85_FFF8, // s_cbranch_scc1 LATCH
    0xBEFE_0418, // s_mov_b64 exec, s[24:25]
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xBF8A_0000, // s_barrier
    0xD8D8_0000, 0x0100_0008, // ds_read_b32 v1, v8
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE070_1000, 0x8002_0102, // buffer_store_dword v1, v2, s[8:11], 0 offen
];

/// a1bf269d5b19f025's counting loops (0xe0, 0x2a0, 0x350, 0x4d0): a loop each lane leaves on
/// its own (`v_cmpx_eq_u32` + `s_andn2_b64` of the lanes left + `s_cbranch_scc1` to a latch
/// that takes EXEC back to them) with an LDS add inside. Lane t goes round 4 times when
/// t < s0, else once; then every lane stores what LDS dword 0 summed up.
const COUNTING_LOOP: &[u32] = &[
    0x3404_0082, // v_lshlrev_b32 v2, 2, v0
    0x7E02_0280, // v_mov_b32 v1, 0
    0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
    0x7E0A_0284, // v_mov_b32 v5, 4
    0x0208_0A81, // v_cndmask_b32 v4, 1, v5, vcc
    0x7E0C_0281, // v_mov_b32 v6, 1
    0x7E0E_0280, // v_mov_b32 v7, 0
    0xD834_0000, 0x0000_0707, // ds_write_b32 v7, v7
    0xBE98_047E, // s_mov_b64 s[24:25], exec
    0xBE94_047E, // s_mov_b64 s[20:21], exec
    0xBF82_0001, // s_branch HEADER
    0x87FE_1416, // LATCH: s_and_b64 exec, s[22:23], s[20:21]
    0x4A02_0281, // HEADER: v_add_nc_u32 v1, 1, v1
    0xBE96_047E, // s_mov_b64 s[22:23], exec
    0xD800_0000, 0x0000_0607, // ds_add_u32 v7, v6
    0x7DA4_0901, // v_cmpx_eq_u32 exec, v1, v4
    0x8994_7E14, // s_andn2_b64 s[20:21], s[20:21], exec
    0xBF85_FFF8, // s_cbranch_scc1 LATCH
    0xBEFE_0418, // s_mov_b64 exec, s[24:25]
    0xD8D8_0000, 0x0100_0007, // ds_read_b32 v1, v7
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE070_1000, 0x8002_0102, // buffer_store_dword v1, v2, s[8:11], 0 offen
];

#[test]
fn a_per_lane_loop_with_lds_inside_runs_in_both_halves_together() {
    let _pool = victory_metal::Pool::new();
    victory_shader::emit::set_host_max_wave(32);
    let shader = victory_shader::emit::translate_native(&program_of(LDS_LOOP, true)).expect("translate");
    let backend = MetalBackend::new().unwrap();
    let r = run(&backend, &shader, 64, 2, 0);
    let last = diag::LAST.lock().unwrap().clone();
    eprintln!("REPRO per-lane LDS loop: lanes 0/31/32/63 read {}/{}/{}/{} (want 2080), timeouts {} {last:?}, GPU {:.3} ms",
        r.out[0], r.out[31], r.out[32], r.out[63], r.timeouts, r.ms);
    assert_eq!(r.timeouts, 0, "half-wave syncs timed out: {:?}", r.sites);
    assert!(r.out.iter().all(|&v| v == 2080), "{:?}", r.out);
}

fn counting_loop_case(backend: &MetalBackend, shader: &victory_shader::emit::HostShader) -> Vec<(u32, u32, u64)> {
    let mut bad = Vec::new();
    for live in [0u32, 10, 31, 32, 33, 40, 64] {
        let r = run(backend, shader, 64, 2, live);
        let want = live * 4 + (64 - live);
        let ok = r.out.iter().all(|&v| v == want);
        eprintln!("REPRO counting loop, live {live:2}: {} (want {want}, lanes 0/32/63 {}/{}/{}), timeouts {} at {:?}, GPU {:.3} ms",
            if ok { "right" } else { "WRONG" }, r.out[0], r.out[32], r.out[63], r.timeouts, r.sites, r.ms);
        if !ok || r.timeouts != 0 {
            bad.push((live, r.out[0], r.timeouts));
        }
    }
    bad
}

/// The halves of a console wave go round a different number of times when the lanes that
/// count longer are all in one half (s0 = 32): the half-wave sync before the LDS add then met
/// a different number of times, the halves fell out of step and one waited out the poll limit
/// (the one timeout of the M3a run). With the loop's exit taken by the whole wave (main's
/// b3813a9) both halves go round together.
/// `VICTORY_WAVE_LOOP_EXITS=0` translates it the old way, which must show the fault here.
#[test]
fn a_loop_lanes_leave_on_their_own_keeps_the_halves_in_step() {
    let _pool = victory_metal::Pool::new();
    victory_shader::emit::set_host_max_wave(32);
    let mut native = program_of(COUNTING_LOOP, true);
    native.io = StageIo::Cs(ComputeIo { rsrc2: 5 << 1 | 1 << 15, threads: [64, 1, 1] }); // 512 bytes of LDS
    let shader = victory_shader::emit::translate_native(&native).expect("translate");
    assert!(shader.root_uavs.contains(&victory_shader::emit::SLOT_DIAG), "no half_sync was emitted");
    // (the old translation, for the record: run alone, as the tests here are)
    unsafe { std::env::set_var("VICTORY_WAVE_LOOP_EXITS", "0") };
    let old = victory_shader::emit::translate_native(&native).expect("translate the old way");
    unsafe { std::env::remove_var("VICTORY_WAVE_LOOP_EXITS") };
    let backend = MetalBackend::new().unwrap();
    let bad_old = counting_loop_case(&backend, &old);
    eprintln!("REPRO counting loop, old translation: (live, lane 0, timeouts) wrong {bad_old:?}");
    let bad = counting_loop_case(&backend, &shader);
    assert!(bad.is_empty(), "(live, lane 0, timeouts): {bad:?}");
    assert!(!bad_old.is_empty(), "the old translation was expected to fall out of step");
}
