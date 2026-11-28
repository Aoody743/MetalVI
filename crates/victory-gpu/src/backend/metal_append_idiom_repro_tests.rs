//! The wave-aggregated append on a host of 32-lane waves, translated by the real translator,
//! converted by MSC and run through `MetalBackend::run`. 29 console wave64 programs append to
//! a list this way (`CS_Tiling_Generate_Opaque 2cc4c197ee29e291` and nine more in a 3D frame):
//! `s_ff1 exec` names the wave's first lane, that lane adds the wave's `s_bcnt1` count to the
//! list's counter (`buffer_atomic_add glc`), and every lane's slot is the returned value
//! (`v_readfirstlane`) plus its rank among the wave's lanes (`v_mbcnt_lo/hi exec`). Run as two
//! host waves, the first lane, the count and the rank all cross the halves (`exchange`).
//!
//! Each lane appends when its bit of s0 (lanes 0..31) / s1 (lanes 32..63) is set: groups where
//! both halves, one half or no lane append. Every slot must be given once (a hit counter per
//! slot), the counter must end at the number of appending lanes, every appending lane must own
//! one slot, and no half-wave sync may time out (`SLOT_DIAG`).
//!
//! Also here: `lower::lds_start` on Metal (thread 0 sets the LDS up behind an early-out).

use super::*;
use victory_metal::ir::DescriptorEntry;
use victory_shader::native::{ComputeIo, HwStage, NativeProgram, ResourceType, RootParam as NativeRoot, RootType, StageIo, UsedResource};

/// The idiom as `2cc4c197ee29e291` 0x530..0x5a0 has it (llvm-mc, gfx1030, wave64):
///
/// ```text
///   v_lshrrev_b32 v3, v0, s0 ; v_lshrrev_b32 v4, v0, s1 ; v_cmp_gt_u32 vcc, 32, v0
///   v_cndmask_b32 v3, v4, v3, vcc ; v_and_b32 v3, 1, v3 ; v_cmp_ne_u32 vcc, 0, v3
///   s_and_saveexec_b64 s[20:21], vcc ; s_cbranch_execz END
///   v_mbcnt_lo_u32_b32 v2, -1, 0 ; s_mov_b64 s[2:3], exec ; s_ff1_i32_b64 s12, exec
///   s_mov_b64 s[14:15], exec ; v_mbcnt_hi_u32_b32 v4, -1, v2 ; v_mov_b32 v2, 0
///   v_cmpx_eq_i32 s12, v4 ; s_cbranch_execz L1
///   s_bcnt1_i32_b64 s2, s[2:3] ; v_mov_b32 v2, s2 ; buffer_atomic_add v2, off, s[8:11], 0 glc
/// L1:
///   s_mov_b64 exec, s[14:15] ; s_waitcnt vmcnt(0) ; s_mov_b64 s[2:3], exec
///   v_readfirstlane_b32 s12, v2 ; v_mbcnt_lo_u32_b32 v4, s2, 0 ; v_mbcnt_hi_u32_b32 v4, s3, v4
///   v_add_nc_u32 v2, s12, v4                                   ; the slot
///   v_lshlrev_b32 v5, 2, v2 ; v_mov_b32 v7, 1
///   buffer_atomic_add v7, v5, s[8:11], 0 offen offset:64       ; hits[slot] += 1
///   v_add_nc_u32 v8, 1, v0
///   buffer_store_dword v8, v5, s[8:11], 0 offen offset:1088    ; owner[slot] = lane + 1
/// END:
/// ```
const APPEND: &[u32] = &[
    0xD516_0003, 0x0000_0100, 0xD516_0004, 0x0000_0300, 0x7D88_00A0, 0x0206_0704, 0x3606_0681, 0x7D8A_0680, 0xBE94_246A,
    0xBF88_001E, 0xD765_0002, 0x0001_00C1, 0xBE82_047E, 0xBE8C_147E, 0xBE8E_047E, 0xD766_0004, 0x0002_04C1, 0x7E04_0280,
    0x7D24_080C, 0xBF88_0004, 0xBE82_1002, 0x7E04_0202, 0xE0C8_4000, 0x8002_0200, 0xBEFE_040E, 0xBF8C_3F70, 0xBE82_047E,
    0x7E18_0502, 0xD765_0004, 0x0001_0002, 0xD766_0004, 0x0002_0803, 0x4A04_080C, 0x340A_0482, 0x7E0E_0281, 0xE0C8_1040,
    0x8002_0705, 0x4A10_0081, 0xE070_1440, 0x8002_0805,
];
/// The two appends of `water_shading:CS_TransmissionVolumetrics_GenerateTiles`
/// (`1aa247ed3653b616` 0x698..0x77c, the program upstream cd93552 rewrites) in the same form:
/// the appending lanes split by their bit of s2/s3 into an if (bit clear: the front list, counter
/// dword 0) and an else (bit set: the back list, counter dword 8), each arm its own
/// `s_ff1`/`s_bcnt1`/`buffer_atomic_add glc`/`v_readfirstlane`/`v_mbcnt exec_lo/exec_hi`, the
/// slot used after the join (hits at dword 16 + 512 x list + slot, owners 256 dwords on).
const WATER: &[u32] = &[
    0xD516_0003, 0x0000_0100, 0xD516_0004, 0x0000_0300, 0x7D88_00A0, 0x0206_0704, 0x3606_0681, 0x7D8A_0680, 0xBE94_246A,
    0xBF88_0042, 0xD516_0003, 0x0000_0500, 0xD516_0004, 0x0000_0700, 0x7D88_00A0, 0x0214_0704, 0x3614_1481, 0xBE80_047E,
    0x7DA4_1480, 0xBF88_0016, 0xD765_0005, 0x0001_00C1, 0xBE82_047E, 0xBE8E_147E, 0xBE8C_047E, 0xD766_0006, 0x0002_0AC1,
    0x7E0A_0280, 0x7D24_0C0E, 0xBF88_0004, 0xBE82_1002, 0x7E0A_0202, 0xE0C8_4000, 0x8002_0500, 0xBEFE_040C, 0xBF8C_3F70,
    0xD765_0006, 0x0001_007E, 0x7E04_0505, 0xD766_0005, 0x0002_0C7F, 0x4A0A_0A02, 0x8AFE_7E00, 0xBF88_0016, 0xD765_0005,
    0x0001_00C1, 0xBE82_047E, 0xBE8E_147E, 0xBE8C_047E, 0xD766_0006, 0x0002_0AC1, 0x7E0A_0280, 0x7D24_0C0E, 0xBF88_0004,
    0xBE82_1002, 0x7E0A_0202, 0xE0C8_4020, 0x8002_0500, 0xBEFE_040C, 0xBF8C_3F70, 0xD765_0006, 0x0001_007E, 0x7E06_0505,
    0xD766_0005, 0x0002_0C7F, 0x4A0A_0A03, 0xBEFE_0400, 0x340E_0A82, 0x3406_148B, 0x4A0E_0707, 0x7E10_0281, 0xE0C8_1040,
    0x8002_0807, 0x4A12_0081, 0xE070_1440, 0x8002_0907,
];

/// Dwords of the output buffer: the counter, the hits per slot and the owner per slot (a
/// second list's are 512 dwords on, its counter at dword 8).
const HITS: usize = 16;
const OWNERS: usize = 272;

/// 64 threads, one console wave, 512 bytes of LDS; the threads from s0 up are live. Thread 0,
/// if it is live, sets the LDS up (dword 0 = 0, dword 1 = all ones); every live thread takes the
/// minimum of dword 1 and its own index and stores it to dword 0 (upstream f299fba's test).
const NEAREST_LIVE_THREAD: &[u32] = &[
    0x7D86_0000, // v_cmp_le_u32 vcc, s0, v0
    0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
    0x7E0A_0280, // v_mov_b32 v5, 0
    0x7E0C_02C1, // v_mov_b32 v6, -1
    0xBE8E_047E, // s_mov_b64 s[14:15], exec
    0x7DA4_0080, // v_cmpx_eq_u32 exec, 0, v0
    0xD838_0100, 0x0006_0505, // ds_write2_b32 v5, v5, v6 offset1:1
    0xBEFE_040E, // s_mov_b64 exec, s[14:15]
    0xD81C_0004, 0x0000_0005, // ds_min_u32 v5, v0 offset:4
    0xD8D8_0004, 0x0700_0005, // ds_read_b32 v7, v5 offset:4
    0x7E00_0307, // v_mov_b32 v0, v7
    0x38 << 26 | 0x1C << 18, 0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    0xBEFE_040C, // s_mov_b64 exec, s[12:13]
];

fn program_of(body: &[u32], lds_bytes: u32) -> NativeProgram {
    program_shaped(body, lds_bytes, false)
}

/// With `square`, 8 x 8 threads as the title's tile programs have them (v0 = x, v1 = y): the
/// body's v0 becomes the flat thread id x + 8 y first.
fn program_shaped(body: &[u32], lds_bytes: u32, square: bool) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    ];
    if square {
        words.extend_from_slice(&[0xD746_0000, 0x0401_0701]); // v_lshl_add_u32 v0, v1, 3, v0
    }
    words.extend_from_slice(body);
    words.push(0xBF81_0000); // END: s_endpgm
    NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
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
        // LDS in rsrc2 bits 15..23, in 512-byte units
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1 | u32::from(square) << 11 | (lds_bytes / 512) << 15,
            threads: if square { [8, 8, 1] } else { [64, 1, 1] },
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

struct Run {
    out: Vec<u32>,
    timeouts: u64,
    sites: Vec<(u32, u32)>,
    ms: f64,
}

/// Runs the translated program over `groups` groups of 64 threads with root constants
/// `constants` (s0..s3) on a zeroed 8 KiB output buffer.
fn run(backend: &MetalBackend, shader: &victory_shader::emit::HostShader, groups: u64, constants: [u32; 4]) -> Run {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
    const BYTES: usize = 8192;
    let mem = unsafe { VirtualAlloc(std::ptr::null(), BYTES, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
    assert!(!mem.is_null());
    let out = unsafe { backend.device.wrap(mem, BYTES as u64) }.unwrap();
    let heap_mem = unsafe { VirtualAlloc(std::ptr::null(), 65536, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
    let heap = unsafe { backend.device.wrap(heap_mem, 65536) }.unwrap();
    const SLOT: u32 = 7;
    unsafe { heap_mem.cast::<DescriptorEntry>().add(SLOT as usize).write(DescriptorEntry::buffer(out.gpu_address, BYTES as u64)) };
    backend.chunks().residency().add(&[out.handle(), heap.handle()]);
    backend.chunks().residency().commit();
    let (pso, conv) = backend.device.compute_pipeline_dxil(&shader.dxil, &root_params(shader), &static_samplers(shader)).unwrap();
    assert_eq!(conv.threadgroup.iter().product::<u32>(), 64);
    let threadgroup = conv.threadgroup.map(u64::from);
    let mp = Arc::new(MetalCompute {
        cost_ns: Default::default(),
        name: "append-idiom repro".to_owned(),
        pso,
        threadgroup,
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
                        (RootType::Constants, 0, d @ 0..=3) => words[u.index as usize] = constants[d as usize],
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
    assert!(backend.run(&[op]), "submission failed");
    let ms = diag::LAST_GPU_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
    let timeouts = diag::HALF_SYNC_TIMEOUTS.load(std::sync::atomic::Ordering::Relaxed) - before;
    let sites = diag::LAST.lock().unwrap().iter().copied().filter(|c| c.0 >= victory_shader::emit::DIAG_SITES).collect();
    let words = unsafe { std::slice::from_raw_parts(mem.cast::<u32>(), BYTES / 4) }.to_vec();
    Run { out: words, timeouts, sites, ms }
}

/// Checks one list of a run: its counter (dword `counter`), its hits and owners (`list` x 512
/// dwords on), against the lanes of `mask` over `groups` groups.
fn check_list(r: &Run, groups: u64, mask: u64, counter: usize, list: usize, what: &str) {
    let per_group = mask.count_ones();
    let total = per_group * groups as u32;
    let count = r.out[counter];
    let hits = &r.out[HITS + 512 * list..HITS + 512 * list + 256];
    let owners = &r.out[OWNERS + 512 * list..OWNERS + 512 * list + 256];
    let used = hits.iter().filter(|&&h| h != 0).count() as u32;
    let duplicated = hits.iter().filter(|&&h| h > 1).count();
    let beyond = hits.iter().enumerate().filter(|&(s, &h)| h != 0 && s as u32 >= total).count();
    let mut per_lane = [0u32; 64];
    for (s, &o) in owners.iter().enumerate().take(total as usize) {
        if o != 0 && hits[s] != 0 {
            per_lane[(o - 1) as usize & 63] += 1;
        }
    }
    let wrong_lanes: Vec<usize> =
        (0..64).filter(|&l| per_lane[l] != if mask >> l & 1 == 1 { groups as u32 } else { 0 }).collect();
    assert!(
        count == total && used == total && duplicated == 0 && beyond == 0 && wrong_lanes.is_empty(),
        "{what} mask {mask:#018x} x{groups}: counter {count} (want {total}), slots used {used}, duplicated {duplicated}, \
         beyond the count {beyond}, lanes without exactly {groups} slot(s) {wrong_lanes:?}; hits {:?}",
        &hits[..(total as usize + 4).min(256)]
    );
}

fn check_append(r: &Run, groups: u64, mask: u64) {
    check_list(r, groups, mask, 0, 0, "append");
    assert_eq!(r.timeouts, 0, "mask {mask:#018x}: half-wave sync timed out at {:?}", r.sites);
}

/// Masks: both halves, only the lower, only the upper, none, sparse, lone lanes at the edges.
const MASKS: &[u64] = &[
    u64::MAX,
    0x0000_0000_FFFF_FFFF,
    0xFFFF_FFFF_0000_0000,
    0,
    0x8000_0001_0000_0000,
    0x0000_0000_8000_0000,
    0x8000_0000_0000_0000,
    0x0000_0001_8000_0000,
    0x5555_5555_AAAA_AAAA,
    0x0123_4567_89AB_CDEF,
    0xF000_0000_0000_000F,
];

#[test]
fn a_wave_aggregated_append_gives_every_lane_its_own_slot() {
    let _pool = victory_metal::Pool::new();
    victory_shader::emit::set_host_max_wave(32);
    let backend = MetalBackend::new().unwrap();
    for square in [false, true] {
    let shader = victory_shader::emit::translate_native(&program_shaped(APPEND, 0, square)).expect("translate");
    assert_eq!(victory_shader::emit::last_wave(), Some(32), "two host waves of 32");
    assert!(shader.root_uavs.contains(&victory_shader::emit::SLOT_DIAG), "no half_sync was emitted: {:?}", shader.root_uavs);
    for &mask in MASKS {
        for groups in [1u64, 4] {
            let r = run(&backend, &shader, groups, [mask as u32, (mask >> 32) as u32, 0, 0]);
            check_append(&r, groups, mask);
            eprintln!(
                "REPRO append 8x8={square} mask={mask:#018x} groups={groups}: {} slots, each once, 0 timeouts, {:.3} ms",
                r.out[0], r.ms
            );
        }
    }
    }
    eprintln!(
        "REPRO append switches: VICTORY_WATER_TILE_SLOTS={:?} (keyed to one container digest; not this program)",
        std::env::var("VICTORY_WATER_TILE_SLOTS").ok()
    );
}

/// The water program's form: two appends in the two arms of a lane branch. Every appending lane
/// gets one slot of its own list.
#[test]
fn the_water_tiles_two_appends_give_every_lane_its_own_slot() {
    let _pool = victory_metal::Pool::new();
    victory_shader::emit::set_host_max_wave(32);
    let backend = MetalBackend::new().unwrap();
    let mut bad = Vec::new();
    for square in [false, true] {
    let shader = victory_shader::emit::translate_native(&program_shaped(WATER, 0, square)).expect("translate");
    assert_eq!(victory_shader::emit::last_wave(), Some(32), "two host waves of 32");
    // (appending lanes, lanes of those that take the back list)
    let kinds: &[u64] = &[0, u64::MAX, 0x0000_0000_FFFF_FFFF, 0xFFFF_FFFF_0000_0000, 0xAAAA_AAAA_AAAA_AAAA, 0x0F0F_F0F0_3C3C_C3C3];
    for &mask in MASKS {
        for &kind in kinds {
            for groups in [1u64, 4] {
                let c = [mask as u32, (mask >> 32) as u32, kind as u32, (kind >> 32) as u32];
                let r = run(&backend, &shader, groups, c);
                let (front, back) = (mask & !kind, mask & kind);
                let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    check_list(&r, groups, front, 0, 0, "front");
                    check_list(&r, groups, back, 8, 1, "back");
                }));
                eprintln!(
                    "REPRO water 8x8={square} mask={mask:#018x} back={kind:#018x} groups={groups}: front {}/{} back {}/{} timeouts {} {} {:.3} ms",
                    r.out[0],
                    front.count_ones() * groups as u32,
                    r.out[8],
                    back.count_ones() * groups as u32,
                    r.timeouts,
                    if ok.is_ok() { "ok" } else { "WRONG" },
                    r.ms
                );
                if ok.is_err() || r.timeouts != 0 {
                    bad.push((square, mask, kind, groups));
                }
            }
        }
    }
    }
    assert!(bad.is_empty(), "wrong or timed out: {bad:x?}");
}

/// `lower::lds_start` on Metal: a group whose thread 0 is not live still starts its LDS as the
/// program's own set-up leaves it. With `VICTORY_LDS_START=0` the group reads whatever the
/// threadgroup memory held (the test then only prints what it got).
#[test]
fn a_groups_lds_starts_as_its_own_set_up_leaves_it_on_metal() {
    let _pool = victory_metal::Pool::new();
    victory_shader::emit::set_host_max_wave(32);
    let shader = victory_shader::emit::translate_native(&program_of(NEAREST_LIVE_THREAD, 512)).expect("translate");
    assert_eq!(victory_shader::emit::last_wave(), Some(32));
    let on = std::env::var_os("VICTORY_LDS_START").is_none_or(|v| v != "0");
    let backend = MetalBackend::new().unwrap();
    for s0 in [0u32, 1, 31, 32, 37, 63] {
        let r = run(&backend, &shader, 1, [s0, 0, 0, 0]);
        eprintln!("REPRO lds_start on={on} s0={s0:2}: got {} (want {s0}), {} timeouts, {:.3} ms", r.out[0], r.timeouts, r.ms);
        if on {
            assert_eq!(r.out[0], s0, "the threads below {s0} are not live");
            assert_eq!(r.timeouts, 0, "s0 {s0}: half-wave sync timed out at {:?}", r.sites);
        }
    }
}
