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
        io: StageIo::Cs(ComputeIo { rsrc2: 5 << 1, threads: [64, 1, 1] }),
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
