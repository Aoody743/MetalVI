//! Frame package tests: encode/decode round trips of the format and of the
//! address translation (CPU only), and a synthetic package recorded from
//! hand-built ops and replayed in a fresh backend at the same addresses
//! (Metal under F85 Wine).

use super::replay::{self, Options, Package};
use super::*;
use crate::backend::metal::{MetalCmd, MetalDispatchIndirect, submit_checked};
use victory_metal::ir::{DescriptorEntry, Location, ResourceType, TopLevel};

// --- CPU: the format -------------------------------------------------------

fn sample_refs() -> Vec<Ref> {
    vec![
        Ref::Null,
        Ref::Title { va: 0x1d42_0000, len: 4096 },
        Ref::Heap { id: 3, offset: 48 },
        Ref::HeapStrides { id: 3, offset: 8 },
        Ref::Texture { id: 9 },
        Ref::Sampler { id: 2 },
        Ref::Typed { id: 7 },
        Ref::Template,
        Ref::Special { kind: SPECIAL_DIAG, offset: 16 },
        Ref::RingTable,
        Ref::Unknown { raw: 0xdead_beef_0000 },
    ]
}

#[test]
fn refs_round_trip_in_16_bytes() {
    for r in sample_refs() {
        let mut w = Vec::new();
        r.put(&mut w);
        assert_eq!(w.len(), 16);
        let mut rd = Reader::new(&w);
        assert_eq!(Ref::get(&mut rd), Some(r));
        assert!(rd.done());
    }
    let mut rd = Reader::new(&[4u8; 16]);
    assert_eq!(Ref::get(&mut rd), None, "an unknown tag is refused");
}

fn sample_compute(pipeline: u32) -> PCompute {
    PCompute {
        pipeline,
        heap: Ref::Heap { id: 1, offset: 0 },
        sampler_heap: Ref::Null,
        top: PTop { raw: (0..32u8).collect(), refs: vec![(16, Ref::Title { va: 0x2000_0000, len: 64 }), (24, Ref::RingTable)] },
        tables: vec![(24, TEntry { gpu: Ref::Title { va: 0x2000_1000, len: 256 }, view: Ref::Null, metadata: 256 })],
    }
}

#[test]
fn ops_round_trip() {
    let ops = [
        POp::Dispatch { c: sample_compute(0), groups: [3, 2, 1], after_barrier: true, loop_counts: None },
        POp::Dispatch { c: sample_compute(1), groups: [1; 3], after_barrier: false, loop_counts: Some([0x10, 0x20]) },
        POp::DispatchIndirect {
            c: sample_compute(1),
            records: Ref::Title { va: 0x3000_0000, len: 12 },
            stride: 12,
            max: 1,
            count: Some(Ref::Title { va: 0x3000_0040, len: 4 }),
            records_va: 0x3000_0000,
            count_va: Some(0x3000_0040),
            group_base_at: Some(12),
        },
        POp::DispatchIndirect {
            c: sample_compute(2),
            records: Ref::Unknown { raw: 5 },
            stride: 16,
            max: 7,
            count: None,
            records_va: 0,
            count_va: None,
            group_base_at: None,
        },
        POp::BufferCopy { dst: 0x4000, src: 0x5000, bytes: 80 },
        POp::Unsupported { kind: KIND_DRAW },
    ];
    let mut file = OPS_MAGIC.to_vec();
    put_u32(&mut file, VERSION);
    put_run(&mut file, 0, &ops[..3]);
    put_run(&mut file, 1, &ops[3..]);
    let runs = get_runs(&file).unwrap();
    assert_eq!(runs.len(), 2);
    assert_eq!((runs[0].0, runs[1].0), (0, 1));
    assert_eq!(runs[0].1, ops[..3]);
    assert_eq!(runs[1].1, ops[3..]);
    // Another version is refused, a truncated file is an error.
    let mut other = file.clone();
    other[8] = 9;
    assert!(get_runs(&other).is_err());
    assert!(get_runs(&file[..file.len() - 3]).is_err());
}

#[test]
fn root_signatures_round_trip() {
    use victory_metal::msc::{RootParam, StaticSampler};
    let params = vec![
        RootParam::Constants { register: 0, space: 0, num: 4 },
        RootParam::Cbv { register: 1, space: 0 },
        RootParam::Srv { register: 2, space: 1 },
        RootParam::Uav { register: 248, space: 1 },
        RootParam::TableSrv { register: 3, space: 0 },
        RootParam::TableUav { register: 4, space: 0 },
    ];
    let samplers = vec![StaticSampler { filter: 21, address_u: 3, mip_lod_bias: -0.5, max_lod: 13.0, shader_register: 2, ..Default::default() }];
    let mut w = Vec::new();
    put_params(&mut w, &params);
    put_samplers(&mut w, &samplers);
    let mut r = Reader::new(&w);
    assert_eq!(get_params(&mut r).unwrap(), params);
    let back = get_samplers(&mut r).unwrap();
    assert!(r.done());
    assert_eq!(format!("{back:?}"), format!("{samplers:?}"));
}

// --- CPU: address translation ----------------------------------------------

/// A backend-like world: two title wrappers, a heap, its strides, the
/// diagnostic buffer, a sampler and a typed view.
fn world() -> AddressMap {
    let mut m = AddressMap {
        title: vec![(0x9_0000_0000, 1 << 20, 0x1d42_0000), (0x9_1000_0000, 16 << 20, 0x2_0000_0000)],
        own: vec![
            (0x8_0000_0000, 0x10000, Own::Heap(4)),
            (0x8_0001_0000, 0x10000, Own::Strides(4)),
            (0x8_0100_0000, 0x1000, Own::Special(SPECIAL_DIAG)),
        ],
        ..Default::default()
    };
    m.samplers.insert(0x5a5a, 6);
    m.typed.insert(0x7777, 2);
    m.textures.insert(0x1234, 11);
    m.sort();
    m
}

/// The inverse of [`world`] (what a replay in a process with the same GPU
/// addresses would answer).
struct Same;
impl Resolver for Same {
    fn address(&mut self, r: &Ref) -> u64 {
        match *r {
            Ref::Title { va, .. } if (0x1d42_0000..0x1d52_0000).contains(&va) => 0x9_0000_0000 + (va - 0x1d42_0000),
            Ref::Title { va, .. } => 0x9_1000_0000 + (va - 0x2_0000_0000),
            Ref::Heap { offset, .. } => 0x8_0000_0000 + offset,
            Ref::HeapStrides { offset, .. } => 0x8_0001_0000 + offset,
            Ref::Special { offset, .. } => 0x8_0100_0000 + offset,
            Ref::Sampler { .. } => 0x5a5a,
            Ref::Texture { .. } => 0x1234,
            _ => 0,
        }
    }
    fn typed(&mut self, _: u32) -> Option<DescriptorEntry> {
        Some(DescriptorEntry::typed_buffer(0x9_1000_0100, 64, 0x7777, 0))
    }
}

#[test]
fn addresses_translate_and_resolve_back() {
    let m = world();
    assert_eq!(m.address(0, None), Ref::Null);
    assert_eq!(m.address(0x9_0000_0010, Some(16)), Ref::Title { va: 0x1d42_0010, len: 16 });
    // A root view without bounds: up to 4 MiB, clipped to its wrapper.
    assert_eq!(m.address(0x9_0000_0010, None), Ref::Title { va: 0x1d42_0010, len: (1 << 20) - 16 });
    assert_eq!(m.address(0x9_1000_0000, None), Ref::Title { va: 0x2_0000_0000, len: 4 << 20 });
    assert_eq!(m.address(0x8_0000_0030, None), Ref::Heap { id: 4, offset: 0x30 });
    assert_eq!(m.address(0x8_0001_0008, None), Ref::HeapStrides { id: 4, offset: 8 });
    assert_eq!(m.address(0x8_0100_0004, None), Ref::Special { kind: SPECIAL_DIAG, offset: 4 });
    // Just past the first wrapper, and between buffers: unknown.
    assert_eq!(m.address(0x9_0010_0000, None), Ref::Unknown { raw: 0x9_0010_0000 });
    assert_eq!(m.address(0x7_0000_0000, None), Ref::Unknown { raw: 0x7_0000_0000 });

    let entries = [
        (DescriptorEntry::NULL, false),
        (DescriptorEntry::buffer(0x9_1000_0100, 64), false),
        (DescriptorEntry::typed_buffer(0x9_1000_0100, 64, 0x7777, 0), false),
        (DescriptorEntry::texture(0x1234, 0.0, 3), false),
        (DescriptorEntry::sampler(0x5a5a, 0.25), true),
    ];
    for (e, samplers) in entries {
        let t = m.entry(&e, samplers);
        assert!(![t.gpu, t.view].iter().any(|r| matches!(r, Ref::Unknown { .. })), "{e:?} → {t:?}");
        assert_eq!(resolve_entry(&mut Same, &t), e);
    }
    assert!(matches!(m.entry(&DescriptorEntry::texture(0x9999, 0.0, 0), false).view, Ref::Unknown { raw: 0x9999 }));
}

#[test]
fn top_level_decodes_by_layout_and_resolves_back() {
    let m = world();
    let layout = vec![
        Location { kind: ResourceType::Constant, space: 0, slot: 0, offset: 0, size: 16 },
        Location { kind: ResourceType::Cbv, space: 0, slot: 1, offset: 16, size: 8 },
        Location { kind: ResourceType::Table, space: u32::MAX, slot: u32::MAX, offset: 24, size: 8 },
        Location { kind: ResourceType::Srv, space: 1, slot: 0, offset: 32, size: 8 },
        Location { kind: ResourceType::Uav, space: 1, slot: 248, offset: 40, size: 8 },
        Location { kind: ResourceType::Uav, space: 1, slot: 3, offset: 48, size: 8 },
        // The static samplers' table: the pipeline's own value.
        Location { kind: ResourceType::Table, space: u32::MAX, slot: u32::MAX, offset: 56, size: 8 },
    ];
    let mut template = TopLevel::new(layout.clone(), 64);
    assert!(template.set_static_samplers(6, 0x6_0000_0000));
    let mut top = template.clone();
    assert!(top.set_constants(0, 0, &[1, 0x9_0000_0000u64 as u32, 3, 4]));
    assert!(top.set_u64(1, 0x9_0000_0100));
    assert!(top.set_u64(3, 0x8_0001_0000));
    assert!(top.set_u64(4, 0x8_0100_0000));
    let tables = [(24u32, DescriptorEntry::buffer(0x9_1000_0000, 512))];
    let p = m.top(&layout, &template.bytes, &top.bytes, &tables);
    assert_eq!(
        p.refs,
        vec![
            (16, Ref::Title { va: 0x1d42_0100, len: (1 << 20) - 0x100 }),
            (24, Ref::RingTable),
            (32, Ref::HeapStrides { id: 4, offset: 0 }),
            (40, Ref::Special { kind: SPECIAL_DIAG, offset: 0 }),
            (48, Ref::Null),
            (56, Ref::Template),
        ]
    );
    // Constants stay; address slots are zero in the stored bytes.
    assert_eq!(&p.raw[..16], &top.bytes[..16]);
    assert!(p.raw[16..].iter().all(|&b| b == 0));
    // Resolved against the same world and template: the same bytes (the
    // ring slot is written at run time, 0 before).
    assert_eq!(resolve_top(&mut Same, &p, &template.bytes), top.bytes);
}

// --- Metal: a synthetic package, recorded and replayed ----------------------

fn dxil_dir() -> Option<std::path::PathBuf> {
    if let Some(p) = std::env::var_os("F85_DXIL_DIR") {
        return Some(std::path::PathBuf::from(p));
    }
    let default = std::path::Path::new(r"Z:\Users\aoody743\Documents\GTA VI\metal-backend\flash\F85\dxil");
    default.exists().then(|| default.to_path_buf())
}

/// A kernel with Metal Shader Converter's binding model: view heap at
/// buffer 0, top level at 2 with root constants, a root UAV and a one-view
/// table; reads one heap slot and the table, writes the root UAV.
const HEAPMIX: &str = "#include <metal_stdlib>
using namespace metal;
struct Entry { device uint *va; ulong view; ulong meta; };
struct Top { uint k; uint idx; uint n; uint pad; device uint *out; const device Entry *table; };
kernel void heapmix(constant Top &top [[buffer(2)]], const device Entry *heap [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    device uint *src = heap[top.idx].va;
    device uint *t = top.table->va;
    if (tid < top.n) { top.out[tid] = src[tid] * top.k + t[tid] + tid; }
}";

fn heapmix_layout() -> TopLevel {
    TopLevel::new(
        vec![
            Location { kind: ResourceType::Constant, space: 0, slot: 0, offset: 0, size: 16 },
            Location { kind: ResourceType::Uav, space: 1, slot: 0, offset: 16, size: 8 },
            Location { kind: ResourceType::Table, space: u32::MAX, slot: u32::MAX, offset: 24, size: 8 },
        ],
        32,
    )
}

const PAGE_X: u64 = 0;
const PAGE_RECORD: u64 = 1;
const PAGE_IN: u64 = 2;
const PAGE_TABLE: u64 = 3;
const PAGE_OUT: u64 = 4;
const PAGE_COPY: u64 = 5;
const PAGE_OUT2: u64 = 6;
const TITLE_PAGES: u64 = 8;

/// Records two runs of hand-built compute ops (direct and indirect dispatches
/// of an MSC pipeline, a heap-and-table kernel, a buffer copy; the CPU
/// changes memory between the runs), frees everything, then replays the
/// package in a new backend with title memory back at its own addresses:
/// every page the GPU wrote must come out byte-identical.
#[test]
fn synthetic_compute_package_replays_byte_identical() {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
    let dxil = std::fs::read(dxil_dir().expect("F85 DXIL fixtures required").join("cs_order.dxil")).unwrap();
    let _pool = victory_metal::Pool::new();
    let dir = std::env::temp_dir().join(format!("t3-framepkg-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    // Title memory at an address the system picks, committed in the middle
    // of a reservation: spans over it (and root views, 4 MiB without a
    // known buffer) end at the uncommitted guard space instead of reaching
    // into the test's own allocations.
    let bytes = (TITLE_PAGES * PAGE) as usize;
    // SAFETY: a fresh reservation, released below before the replay.
    let reserved = unsafe { VirtualAlloc(std::ptr::null(), 32 << 20, MEM_RESERVE, PAGE_READWRITE) } as u64;
    assert_ne!(reserved, 0);
    // SAFETY: inside the reservation.
    let title = unsafe { VirtualAlloc((reserved + (16 << 20)) as *const core::ffi::c_void, bytes, MEM_COMMIT, PAGE_READWRITE) } as u64;
    assert_eq!(title, reserved + (16 << 20));
    let page = |p: u64| title + p * PAGE;
    let words = |p: u64| page(p) as *mut u32;
    // SAFETY: inside the allocation.
    unsafe {
        words(PAGE_RECORD).cast::<[u32; 3]>().write([1, 1, 1]);
        for i in 0..64 {
            words(PAGE_IN).add(i).write(1000 + i as u32 * 7);
            words(PAGE_TABLE).add(i).write(i as u32 * i as u32);
        }
    }

    let expected_x;
    let recorded: Vec<u8>;
    let stats;
    {
        let backend = MetalBackend::new().unwrap();
        // The order probe (MSC, from DXIL) and the heap kernel (MSL).
        let params = [victory_metal::msc::RootParam::Constants { register: 0, space: 0, num: 4 }, victory_metal::msc::RootParam::Uav { register: 0, space: 1 }];
        let (pso, conv) = backend.device.compute_pipeline_dxil(&dxil, &params, &[]).unwrap();
        register_source(&pso, Source { code: ShaderCode::Dxil { dxil: Arc::new(dxil.clone()), params: params.to_vec(), samplers: Vec::new() }, cache: None });
        let order = Arc::new(MetalCompute { cost_ns: Default::default(), name: "order probe".into(), pso, threadgroup: conv.threadgroup.map(u64::from), top_level: conv.top_level.clone() });
        let pso = backend.device.compute_pipeline_msl(HEAPMIX, "heapmix").unwrap();
        register_source(&pso, Source { code: ShaderCode::Msl { source: HEAPMIX.into(), function: "heapmix".into() }, cache: None });
        let mix = Arc::new(MetalCompute { cost_ns: Default::default(), name: "heapmix".into(), pso, threadgroup: [64, 1, 1], top_level: heapmix_layout() });

        // A view heap (slot 5: the input; slot 6: a typed view of it) and a
        // sampler heap (one sampler), as the backend keeps them.
        let window = |va: u64, len: u64| backend.chunks().window(va, len);
        let heap = backend.new_heap(16, false).unwrap();
        let sheap = backend.new_heap(4, true).unwrap();
        let in_entry = DescriptorEntry::buffer(window(page(PAGE_IN), 256), 256);
        let typed = backend.typed_entry(page(PAGE_IN), 256, 42).expect("R32_UINT typed view");
        let point = [0u32, 3, 3, 3, 0, 1, 1, 0, 0, 0, 0, 0, 1000f32.to_bits()];
        let sampler = backend.sampler_entry(&point).unwrap();
        // SAFETY: inside both heaps' entries (16 and 4).
        unsafe {
            heap.entries.add(5).write(in_entry);
            heap.entries.add(6).write(typed);
            sheap.entries.add(0).write(sampler);
        }
        let (heap_buf, sheap_buf) = (heap.buffer.clone(), sheap.buffer.clone());
        backend.heaps.lock().unwrap().insert(0x7e57_0a01, heap);
        backend.heaps.lock().unwrap().insert(0x7e57_0a02, sheap);

        let order_op = |k: u32, indirect: bool| {
            let mut top = conv.top_level.clone();
            assert!(top.set_constants(0, 0, &[k, 0, 0, 0]));
            assert!(top.set_root_descriptor(ResourceType::Uav, 1, 0, window(page(PAGE_X), 4)));
            Arc::new(if indirect {
                let (records, records_offset) = backend.chunks().window_buffer(page(PAGE_RECORD), 12).unwrap();
                MetalOp::DispatchIndirect(MetalDispatchIndirect {
                    pso: order.clone(), heap: None, sampler_heap: None, top: top.bytes, tables: Vec::new(),
                    records, records_offset, stride: 12, max: 1, count: None,
                    records_va: page(PAGE_RECORD), count_va: None, group_base_at: None,
                })
            } else {
                MetalOp::Dispatch(MetalCmd { pso: order.clone(), heap: None, sampler_heap: None, top: top.bytes, groups: [1; 3], tables: Vec::new(), loop_counts: None, after_barrier: true })
            })
        };
        let mix_op = |k: u32, out: u64| {
            let mut top = heapmix_layout();
            assert!(top.set_constants(0, 0, &[k, 5, 64, 0]));
            assert!(top.set_root_descriptor(ResourceType::Uav, 1, 0, window(page(out), 256)));
            Arc::new(MetalOp::Dispatch(MetalCmd {
                pso: mix.clone(), heap: Some(heap_buf.clone()), sampler_heap: Some(sheap_buf.clone()), top: top.bytes, groups: [1; 3],
                tables: vec![(24, DescriptorEntry::buffer(window(page(PAGE_TABLE), 256), 256))], loop_counts: None, after_barrier: true,
            }))
        };
        let run1 = vec![
            order_op(1, false),
            order_op(2, true),
            mix_op(3, PAGE_OUT),
            Arc::new(MetalOp::BufferCopy { dst: page(PAGE_COPY) as usize, src: page(PAGE_OUT) as usize, bytes: 256 }),
            order_op(4, false),
        ];
        let run2 = vec![mix_op(5, PAGE_OUT2), order_op(6, false)];

        let mut rec = Recorder::new(&dir, 7, 64 << 20).unwrap();
        let seen = rec.before(&backend, &run1);
        assert!(backend.run(&run1));
        rec.after(&seen);
        // The CPU between the runs: a new input, and x changed after the GPU wrote it.
        // SAFETY: inside the allocation; the GPU is idle.
        unsafe {
            for i in 0..64 {
                words(PAGE_IN).add(i).write(5 + i as u32);
            }
            *words(PAGE_X) ^= 0x55;
        }
        let seen = rec.before(&backend, &run2);
        assert!(backend.run(&run2));
        rec.after(&seen);
        stats = rec.finish(Some(&backend)).unwrap();
        println!("recorded: {stats:?}");

        let x1 = ((3u32 + 2) * 3 + 4) ^ 0x55; // x = x * 3 + k for k = 1, 2, 4 from 0
        expected_x = x1 * 3 + 6;
        // SAFETY: inside the allocation; the GPU is done.
        recorded = unsafe { std::slice::from_raw_parts(title as *const u8, bytes).to_vec() };
        assert_eq!(unsafe { *words(PAGE_X) }, expected_x, "the recorded runs themselves ran in order");
        for i in 0..64u32 {
            let out = unsafe { *words(PAGE_OUT).add(i as usize) };
            assert_eq!(out, (1000 + i * 7) * 3 + i * i + i);
            assert_eq!(unsafe { *words(PAGE_COPY).add(i as usize) }, out);
            assert_eq!(unsafe { *words(PAGE_OUT2).add(i as usize) }, (5 + i) * 5 + i * i + i);
        }
        drop((run1, run2));
        // The backend (and its spans over the title memory) goes first.
    }
    // SAFETY: nothing uses the allocation any more.
    assert_ne!(unsafe { VirtualFree(reserved as *mut core::ffi::c_void, 0, MEM_RELEASE) }, 0);

    // F1/F2: what the package holds.
    assert_eq!((stats.runs, stats.ops, stats.dispatches, stats.indirect, stats.buffer_copies), (2, 7, 5, 1, 1));
    assert_eq!(stats.unknown, 0, "unknown addresses: {:x?}", stats.unknown_samples);
    assert_eq!(stats.heap_unknown, 0, "unknown heap entries: {:x?}", stats.unknown_samples);
    assert_eq!(stats.pages, TITLE_PAGES, "only title pages are kept");
    assert_eq!(stats.pipelines, 2);
    assert_eq!(stats.pipelines_without_source, 0);
    assert!(stats.stopped.is_none());
    let pkg = Package::open(&dir).unwrap();
    assert_eq!(pkg.runs.len(), 2);
    assert_eq!(pkg.runs[0].1.len(), 5);
    let epochs = |p: u64| -> Vec<u32> { pkg.memory.iter().filter(|m| m.va == page(p)).map(|m| m.epoch).collect() };
    assert_eq!(epochs(PAGE_IN), vec![0, 1], "the CPU's new input is a second version");
    assert_eq!(epochs(PAGE_X), vec![0, 1], "x changed by the CPU after the GPU wrote it");
    assert_eq!(epochs(PAGE_OUT), vec![0], "the GPU's own writes are not stored again");
    let mut reference: Vec<u64> = pkg.reference.iter().map(|m| (m.va - title) / PAGE).collect();
    reference.sort_unstable();
    assert_eq!(reference, vec![PAGE_X, PAGE_OUT, PAGE_COPY, PAGE_OUT2]);
    assert_eq!(pkg.heaps.len(), 2);
    assert_eq!((pkg.samplers.len(), pkg.typed.len()), (1, 1));
    let manifest = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
    assert!(manifest.contains("\"unknown_addresses\": 0"), "{manifest}");

    // F3: replay at the same addresses in a new backend.
    let placed = replay::place(&pkg).unwrap_or_else(|e| panic!("cannot place the title memory: {e}"));
    println!("placed {placed:x?}");
    let backend = MetalBackend::new().unwrap();
    let report = replay::replay(&backend, &pkg, &Options::default()).unwrap();
    println!("replay: {report:?}");
    assert_eq!(report.failed_run, None);
    assert_eq!((report.runs, report.ops_run, report.pipelines_built), (2, 7, 2));
    assert!(report.pipeline_errors.is_empty());
    assert_eq!(report.layout_mismatches + report.heap_capacity_mismatches, 0);
    assert_eq!(report.skipped_unsupported + report.skipped_no_pipeline + report.skipped_memory_directory + report.skipped_unmapped, 0);
    assert_eq!(report.zeroed_unknown + report.zeroed_texture + report.unmapped_title, 0);
    let cmp = replay::compare_reference(&pkg).unwrap();
    println!("compare: {cmp:?}");
    assert_eq!(cmp.pages, 4);
    assert!(cmp.differing.is_empty(), "differing pages: {:x?}", cmp.differing);
    // And the whole title range as the recording left it.
    for p in [PAGE_X, PAGE_OUT, PAGE_COPY, PAGE_OUT2, PAGE_IN, PAGE_TABLE, PAGE_RECORD] {
        let at = (p * PAGE) as usize;
        // SAFETY: placed by the replay.
        let got = unsafe { std::slice::from_raw_parts(page(p) as *const u8, PAGE as usize) };
        assert!(got == &recorded[at..at + PAGE as usize], "page {p} differs from the recording");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A recorded package (`VICTORY_FRAMEPKG=<dir>`), replayed: compute only
/// (F3). `VICTORY_FRAMEPKG_RUNS=a..b` sends only those runs to the GPU.
#[test]
#[ignore = "needs a recorded package (VICTORY_FRAMEPKG)"]
fn replay_package() {
    let Some(dir) = std::env::var_os("VICTORY_FRAMEPKG") else {
        println!("SKIP: VICTORY_FRAMEPKG not set");
        return;
    };
    let runs = std::env::var("VICTORY_FRAMEPKG_RUNS").ok().and_then(|v| {
        let (a, b) = v.split_once("..")?;
        Some((a.parse().ok()?, b.parse().ok()?))
    });
    let _pool = victory_metal::Pool::new();
    let pkg = Package::open(std::path::Path::new(&dir)).unwrap();
    println!("package: {} runs, {} pipelines, {} memory pages, {} reference pages", pkg.runs.len(), pkg.pipelines.len(), pkg.memory.len(), pkg.reference.len());
    // The memory goes in before the backend exists: its own allocations
    // must not take title addresses.
    if let Err(e) = replay::place(&pkg) {
        panic!("{e}");
    }
    let backend = MetalBackend::new().unwrap();
    let report = replay::replay(&backend, &pkg, &Options { runs }).unwrap();
    println!("replay: {report:#?}");
    let cmp = replay::compare_reference(&pkg).unwrap();
    println!("compare: {} pages, {} identical, {} differ", cmp.pages, cmp.identical, cmp.differing.len());
    for (va, first, n) in cmp.differing.iter().take(32) {
        println!("  page {va:#x}: {n} bytes differ, first at {first:#x}");
    }
    assert_eq!(report.failed_run, None);
}

// --- F4: the draw-side ops ---------------------------------------------------

fn sample_attachment(id: u32, pixel: u32) -> draw::PAttachment {
    draw::PAttachment { tex: Ref::Tex { id, view: draw::VIEW_SELF }, pixel, level: 1, slice: 2, width: 64, height: 32, samples: 1, companion: id == 2, twin: false }
}

#[test]
fn draw_ops_round_trip() {
    use draw::{PClear, PCopy, PDraw, PDrawKind, PResolve, PStage, PTargets};
    let stage = PStage {
        bind: 1,
        top: PTop { raw: vec![1, 2, 3, 4, 0, 0, 0, 0], refs: vec![(4, Ref::Template)] },
        tables: vec![(8, TEntry { gpu: Ref::Title { va: 0x5000, len: 64 }, view: Ref::Null, metadata: 64 })],
        heap: Ref::Heap { id: 0, offset: 0 },
        sampler_heap: Ref::Heap { id: 1, offset: 0 },
    };
    let kinds = [
        PDrawKind::Vertices { count: 3, instances: 1, first: 0, first_instance: 0 },
        PDrawKind::Indexed { count: 6, instances: 2, base: -4, first_instance: 1, buffer: Ref::Title { va: 0x6000, len: 12 }, index_type: 0 },
        PDrawKind::Mesh { groups: [4, 1, 1] },
        PDrawKind::Generated { indices: vec![0, 1, 2, 0, 2, 3], instances: 1, base: 7, first_instance: 0 },
        PDrawKind::Indirect {
            records: Ref::Title { va: 0x7000, len: 40 },
            records_va: 0x7000,
            stride: 20,
            max: 2,
            count: Some(Ref::Title { va: 0x7100, len: 4 }),
            count_va: Some(0x7100),
            index: Some((Ref::Title { va: 0x8000, len: 1024 }, 1, 256)),
        },
    ];
    let mut ops: Vec<POp> = kinds
        .into_iter()
        .map(|kind| {
            let mut targets = PTargets { stencil: true, width: 64, height: 32, samples: 1, ..Default::default() };
            targets.colors[0] = Some(sample_attachment(0, 70));
            targets.colors[3] = Some(draw::PAttachment { tex: Ref::Tex { id: 1, view: draw::VIEW_TARGET | 71 }, ..sample_attachment(1, 71) });
            targets.depth = Some(draw::PAttachment { tex: Ref::Tex { id: 2, view: draw::VIEW_COMPANION }, ..sample_attachment(2, 252) });
            POp::Draw(Box::new(PDraw {
                pipeline: 3,
                targets,
                stages: vec![stage.clone()],
                vertex: Some(vec![(Ref::Title { va: 0x9000, len: 48 }, 48, 16), (Ref::Null, 0, 0)]),
                viewport: [0.0, 0.0, 64.0, 32.0, 0.0, 1.0],
                scissor: [0, 0, 64, 32],
                blend: [0.5, 0.25, 1.0, 0.0],
                stencil_ref: 3,
                primitive: 3,
                kind,
            }))
        })
        .collect();
    ops.push(POp::Clear(Box::new(PClear { target: sample_attachment(0, 70), color: Some([0.1, 0.2, 0.3, 1.0]), depth: None, stencil: None, has_stencil: false, rects: vec![[0, 0, 8, 8], [8, 8, 16, 16]] })));
    ops.push(POp::Clear(Box::new(PClear { target: sample_attachment(2, 260), color: None, depth: Some(1.0), stencil: Some(5), has_stencil: true, rects: Vec::new() })));
    ops.push(POp::Copy(Box::new(PCopy {
        src: Ref::Tex { id: 0, view: 0 },
        dst: Ref::Tex { id: 4, view: 0 },
        whole: false,
        src_slice: 1,
        src_level: 2,
        src_origin: [1, 2, 0],
        size: [8, 8, 1],
        dst_slice: 0,
        dst_level: 0,
        dst_origin: [4, 4, 0],
    })));
    ops.push(POp::Readback { src: Ref::Tex { id: 5, view: 0 }, dst: Ref::Title { va: 0xa000, len: 80 }, width: 5 });
    ops.push(POp::Resolve(Box::new(PResolve { src: Ref::Tex { id: 6, view: 0 }, src_slice: 0, dst: Ref::Tex { id: 0, view: 0 }, dst_level: 0, dst_slice: 1, format: 70, region: Some([0, 0, 2, 2, 8, 8]) })));
    ops.push(POp::Ambient { tex: Ref::Tex { id: 7, view: 0 }, mip: 1, slice: 0, size: [32, 16], colour: [0.1, 0.2, 0.3] });
    ops.push(POp::Unsupported { kind: KIND_DRAW });
    let mut file = OPS_MAGIC.to_vec();
    put_u32(&mut file, VERSION);
    put_run(&mut file, 0, &ops);
    let runs = get_runs(&file).unwrap();
    assert_eq!(runs[0].1, ops);
    // Their references are all seen (heap snapshots and texture lists use them).
    let refs: Vec<Ref> = ops.iter().flat_map(|o| o.refs()).collect();
    assert!(refs.contains(&Ref::Tex { id: 2, view: draw::VIEW_COMPANION }));
    assert!(refs.contains(&Ref::Title { va: 0x8000, len: 1024 }));
    assert!(refs.contains(&Ref::Title { va: 0x9000, len: 48 }));
    let (mut ids, mut written) = (std::collections::BTreeSet::new(), std::collections::BTreeSet::new());
    for op in &ops {
        draw::op_textures(op, &mut ids, &mut written);
    }
    assert_eq!(ids.into_iter().collect::<Vec<_>>(), vec![0, 1, 2, 4, 5, 6, 7]);
    assert_eq!(written.into_iter().collect::<Vec<_>>(), vec![0, 1, 2, 4, 7]);
}

/// A position-from-vertex-table vertex shader and a flat colour fragment
/// shader, with MSC's bindings (top level at 2, vertex table at 6).
const FLAT: &str = "#include <metal_stdlib>
using namespace metal;
struct VTop { float4 color; float z; float x0; float y0; float size; };
struct VEntry { const device float2 *pos; uint length; uint stride; };
struct VOut { float4 pos [[position]]; float4 color; };
vertex VOut vs_flat(uint vid [[vertex_id]], constant VTop &top [[buffer(2)]], constant VEntry *vt [[buffer(6)]]) {
    float2 p = vt[0].pos[vid] * top.size + float2(top.x0, top.y0);
    VOut o; o.pos = float4(p, top.z, 1.0); o.color = top.color; return o;
}
fragment float4 fs_flat(VOut in [[stage_in]]) { return in.color; }";

const K_RT: usize = 0x7e57_1001;
const K_DEPTH: usize = 0x7e57_1002;
const K_COPY: usize = 0x7e57_1003;
const K_SRC: usize = 0x7e57_1004;
const K_AMB: usize = 0x7e57_1005;

/// A title texture the way `make_texture` makes one (DXGI `format`, D3D12
/// `flags`), in the backend's table under `key`.
fn title_texture(backend: &MetalBackend, key: usize, format: u32, flags: u32, pixel: victory_metal::sys::WMTPixelFormat, size: u32) -> Arc<crate::backend::metal_texture::MetalTexture> {
    // SAFETY: an integer-only C record.
    let mut desc: crate::resource::ResourceDesc = unsafe { std::mem::zeroed() };
    desc.dimension = 3;
    desc.width = u64::from(size);
    desc.height = size;
    desc.depth_or_array_size = 1;
    desc.mip_levels = 1;
    desc.format = format;
    desc.sample_count = 1;
    desc.flags = flags;
    let mut usage = 1 | 0x10;
    if flags & 4 != 0 {
        usage |= 2;
    }
    if flags & 3 != 0 {
        usage |= 4;
    }
    let texture = backend
        .device
        .new_texture(&victory_metal::texture::TextureDescriptor {
            pixel_format: pixel,
            width: size,
            height: size,
            usage: victory_metal::sys::WMTTextureUsage(usage),
            options: victory_metal::sys::WMTResourceOptions(0x20),
            ..Default::default()
        })
        .unwrap();
    backend.add_resident(&[texture.handle()]);
    let t = Arc::new(crate::backend::metal_texture::MetalTexture { texture, pixel, desc, mips: 1, slices: 1, companion: Default::default(), twin: None });
    backend.textures.lock().unwrap().insert(key, Some(t.clone()));
    t
}

fn fill_r8(backend: &MetalBackend, t: &victory_metal::texture::Texture, value: u8) {
    let staging = backend.device.shared_buffer(&[value; 256 * 4]).unwrap();
    let cb = backend.queue.command_buffer().unwrap();
    let mut blit = cb.blit_encoder().unwrap();
    blit.copy_from_buffer_to_texture(&staging, 0, 256, 256 * 4, [4, 4, 1], t, 0, 0, [0, 0, 0]);
    blit.finish();
    submit_checked(&cb, &[], true, "test fill").unwrap();
}

/// F4: two runs of clears, draws (an MSC pipeline sampling a heap texture
/// view through a static sampler; a Metal one reading positions from title
/// memory through the vertex table, direct and indexed, depth tested), a
/// texture copy and the ambient stand-in, recorded, freed, and replayed in a
/// new backend: every texture the frame wrote must come out identical.
#[test]
fn synthetic_draw_package_replays_pixel_identical() {
    use crate::backend::metal_draw::{Attachment, DrawKind, MetalClear, MetalDraw, MetalGraphics, MetalTexCopy, STAGE_FRAGMENT, STAGE_VERTEX, StageBind, Targets, VertexFetch};
    use victory_metal::msc::{self, RootParam, Stage, StaticSampler};
    use victory_metal::render::{ColorDesc, DepthStencilDesc, RenderPipelineDesc};
    use victory_metal::sys::{WMTPixelFormat as P, WMTScissorRect, WMTViewport};
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
    let dir_dxil = dxil_dir().expect("F85 DXIL fixtures required");
    let _pool = victory_metal::Pool::new();
    let dir = std::env::temp_dir().join(format!("t3-framepkg-draw-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    // Title memory (positions, indices) in the middle of a reservation.
    // SAFETY: a fresh reservation, released before the replay.
    let reserved = unsafe { VirtualAlloc(std::ptr::null(), 32 << 20, MEM_RESERVE, PAGE_READWRITE) } as u64;
    assert_ne!(reserved, 0);
    let title = reserved + (16 << 20);
    // SAFETY: inside the reservation.
    assert_eq!(unsafe { VirtualAlloc(title as *const core::ffi::c_void, (2 * PAGE) as usize, MEM_COMMIT, PAGE_READWRITE) } as u64, title);
    let positions = title as *mut [f32; 2];
    let indices = (title + PAGE) as *mut u16;
    // SAFETY: inside the committed pages.
    unsafe {
        positions.cast::<[[f32; 2]; 3]>().write([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        indices.cast::<[u16; 3]>().write([2, 1, 0]);
    }
    let stats;
    {
        let backend = MetalBackend::new().unwrap();
        let rt = title_texture(&backend, K_RT, 28, 1, P::WMTPixelFormatRGBA8Unorm, 16);
        let depth = title_texture(&backend, K_DEPTH, 40, 2, P::WMTPixelFormatDepth32Float, 16);
        let copy = title_texture(&backend, K_COPY, 28, 1, P::WMTPixelFormatRGBA8Unorm, 16);
        let src = title_texture(&backend, K_SRC, 61, 0, P::WMTPixelFormatR8Unorm, 4);
        let amb = title_texture(&backend, K_AMB, 42, 4, P::WMTPixelFormatR32Uint, 16);
        fill_r8(&backend, &src.texture, 200);
        let attach = |t: &crate::backend::metal_texture::MetalTexture, key: usize| Attachment {
            resource: key,
            texture: t.texture.clone(),
            pixel: t.pixel.0,
            level: 0,
            slice: 0,
            width: 16,
            height: 16,
            samples: 1,
            companion: None,
            twin: None,
        };
        let mut targets = Targets::default();
        targets.colors[0] = Some(attach(&rt, K_RT));
        targets.depth = Some(attach(&depth, K_DEPTH));
        targets.finish();
        let targets = Arc::new(targets);
        let mut desc = RenderPipelineDesc { sample_count: 1, topology_class: 3, depth_format: P::WMTPixelFormatDepth32Float.0, ..Default::default() };
        desc.colors[0] = ColorDesc { pixel_format: P::WMTPixelFormatRGBA8Unorm.0, write_mask: 0xf, ..Default::default() };

        // The MSC pipeline: full-target triangle sampling heap slot 5.
        let vs = msc::convert(&std::fs::read(dir_dxil.join("vs_main.dxil")).unwrap(), Stage::Vertex, &[], &[]).unwrap();
        let ps_params = [RootParam::Constants { register: 0, space: 0, num: 4 }];
        let point = StaticSampler { filter: 0, address_u: 3, address_v: 3, address_w: 3, max_anisotropy: 1, comparison_func: 1, max_lod: 1000.0, ..Default::default() };
        let ps = msc::convert(&std::fs::read(dir_dxil.join("ps_static.dxil")).unwrap(), Stage::Fragment, &ps_params, &[point]).unwrap();
        let mut ps_top = ps.top_level.clone();
        backend.bind_static_samplers(&mut ps_top, ps_params.len(), &[point]).unwrap();
        let always = DepthStencilDesc { depth_compare: 7, depth_write: false, ..Default::default() };
        let sample_pso = {
            let (vl, pl) = (backend.device.library(&vs.metallib).unwrap(), backend.device.library(&ps.metallib).unwrap());
            backend.device.render_pipeline_desc((&vl, &vs.entry), Some((&pl, &ps.entry)), &desc).unwrap()
        };
        let sample = Arc::new(MetalGraphics {
            cost_ns: Default::default(),
            name: "sample heap view".into(),
            pso: sample_pso,
            dss: backend.device.depth_stencil_state(&always).unwrap(),
            depth_write: false,
            raster: [0, 0, 0, 1],
            bias: [0.0; 3],
            stages: vec![(vs.top_level.clone(), STAGE_VERTEX), (ps_top.clone(), STAGE_FRAGMENT)],
            vertex_slots: 0,
            mesh: false,
            mesh_threads: [1; 3],
            geometry: None,
            colors: [P::WMTPixelFormatRGBA8Unorm.0, 0, 0, 0, 0, 0, 0, 0],
            depth: P::WMTPixelFormatDepth32Float.0,
            stencil: 0,
            samples: 1,
        });
        draw::register_graphics(&sample, draw::GraphicsSource {
            vertex: Some((draw::Lib::Metallib(vs.metallib.clone()), vs.entry.clone())),
            fragment: Some((draw::Lib::Metallib(ps.metallib.clone()), ps.entry.clone())),
            mesh: None,
            stage_in: None,
            desc,
            dss: always,
            static_samplers: vec![(STAGE_FRAGMENT, ps_params.len() as u32, vec![point])],
        });
        // The flat pipeline: depth tested and written.
        let less = DepthStencilDesc { depth_compare: 1, depth_write: true, ..Default::default() };
        let flat_lib = backend.device.library_msl(FLAT).unwrap();
        let vtop = TopLevel::new(vec![Location { kind: ResourceType::Constant, space: 0, slot: 0, offset: 0, size: 32 }], 32);
        let flat = Arc::new(MetalGraphics {
            cost_ns: Default::default(),
            name: "flat".into(),
            pso: backend.device.render_pipeline_desc((&flat_lib, "vs_flat"), Some((&flat_lib, "fs_flat")), &desc).unwrap(),
            dss: backend.device.depth_stencil_state(&less).unwrap(),
            depth_write: true,
            raster: [0, 0, 0, 1],
            bias: [0.0; 3],
            stages: vec![(vtop.clone(), STAGE_VERTEX)],
            vertex_slots: 1,
            mesh: false,
            mesh_threads: [1; 3],
            geometry: None,
            colors: [P::WMTPixelFormatRGBA8Unorm.0, 0, 0, 0, 0, 0, 0, 0],
            depth: P::WMTPixelFormatDepth32Float.0,
            stencil: 0,
            samples: 1,
        });
        draw::register_graphics(&flat, draw::GraphicsSource {
            vertex: Some((draw::Lib::Msl(FLAT.into()), "vs_flat".into())),
            fragment: Some((draw::Lib::Msl(FLAT.into()), "fs_flat".into())),
            mesh: None,
            stage_in: None,
            desc,
            dss: less,
            static_samplers: Vec::new(),
        });

        // The heaps: slot 5 a view of the source texture; one sampler.
        let srv_words = [61u32, 4, 5768, 0, 0, 1, 0, 0, 0, 0];
        let view = backend.texture_entry(K_SRC as crate::object::This, Some(&srv_words), false).unwrap();
        let heap = backend.new_heap(16, false).unwrap();
        let sheap = backend.new_heap(4, true).unwrap();
        let sampler = backend.sampler_entry(&[0u32, 3, 3, 3, 0, 1, 1, 0, 0, 0, 0, 0, 1000f32.to_bits()]).unwrap();
        // SAFETY: inside the heaps' entries.
        unsafe {
            heap.entries.add(5).write(view);
            sheap.entries.add(0).write(sampler);
        }
        let (heap_buf, sheap_buf) = (heap.buffer.clone(), sheap.buffer.clone());
        backend.heaps.lock().unwrap().insert(0x7e57_0b01, heap);
        backend.heaps.lock().unwrap().insert(0x7e57_0b02, sheap);

        let viewport = WMTViewport { originX: 0.0, originY: 0.0, width: 16.0, height: 16.0, znear: 0.0, zfar: 1.0 };
        let scissor = WMTScissorRect { x: 0, y: 0, width: 16, height: 16 };
        let sample_draw = || {
            let mut top = ps_top.clone();
            assert!(top.set_constants(0, 0, &[5, 0, 0, 0]));
            let stage = |bind, top: Vec<u8>| StageBind { bind, top, tables: Vec::new(), heap: Some(heap_buf.clone()), sampler_heap: Some(sheap_buf.clone()) };
            Arc::new(MetalOp::Draw(MetalDraw {
                trace_inputs: Vec::new(),
                pso: sample.clone(),
                targets: targets.clone(),
                stages: vec![stage(STAGE_VERTEX, vs.top_level.bytes.clone()), stage(STAGE_FRAGMENT, top.bytes)],
                vertex: None,
                viewport,
                scissor,
                blend: [1.0; 4],
                stencil_ref: 0,
                primitive: 3,
                kind: DrawKind::Vertices { count: 3, instances: 1, first: 0, first_instance: 0 },
            }))
        };
        let flat_draw = |color: [f32; 4], z: f32, at: [f32; 2], size: f32, indexed: bool| {
            let mut top = vtop.clone();
            let words: Vec<u32> = color.iter().chain(&[z, at[0], at[1], size]).map(|f| f.to_bits()).collect();
            assert!(top.set_constants(0, 0, &words));
            let (vb, vo) = backend.chunks().window_buffer(title, 24).unwrap();
            let mut table = [0u8; msc::VERTEX_TABLE_SLOTS * 16];
            table[..8].copy_from_slice(&(vb.gpu_address + vo).to_le_bytes());
            table[8..12].copy_from_slice(&24u32.to_le_bytes());
            table[12..16].copy_from_slice(&8u32.to_le_bytes());
            let kind = if indexed {
                let (ib, io) = backend.chunks().window_buffer(title + PAGE, 6).unwrap();
                DrawKind::Indexed { count: 3, instances: 1, base: 0, first_instance: 0, buffer: ib, offset: io as u32, index_type: 0 }
            } else {
                DrawKind::Vertices { count: 3, instances: 1, first: 0, first_instance: 0 }
            };
            Arc::new(MetalOp::Draw(MetalDraw {
                trace_inputs: Vec::new(),
                pso: flat.clone(),
                targets: targets.clone(),
                stages: vec![StageBind { bind: STAGE_VERTEX, top: top.bytes, tables: Vec::new(), heap: None, sampler_heap: None }],
                vertex: Some(Box::new(VertexFetch { table, _buffers: vec![vb] })),
                viewport,
                scissor,
                blend: [1.0; 4],
                stencil_ref: 0,
                primitive: 3,
                kind,
            }))
        };
        let clear_colour = Arc::new(MetalOp::Clear(MetalClear { target: attach(&rt, K_RT), color: Some([0.1, 0.2, 0.3, 1.0]), depth: None, stencil: None, has_stencil: false, rects: Vec::new() }));
        let clear_depth = Arc::new(MetalOp::Clear(MetalClear { target: attach(&depth, K_DEPTH), color: None, depth: Some(1.0), stencil: None, has_stencil: false, rects: Vec::new() }));
        let run1 = vec![
            clear_colour,
            clear_depth,
            sample_draw(),
            flat_draw([1.0, 0.0, 0.0, 1.0], 0.5, [-1.0, -1.0], 1.5, false),
            flat_draw([0.0, 1.0, 0.0, 1.0], 0.7, [-0.5, -0.5], 1.5, true),
        ];
        let ambient = crate::backend::metal_ambient::MetalAmbient::new(&amb, None, 0.5).expect("R32Uint UAV ambient view");
        let run2 = vec![
            sample_draw(),
            flat_draw([0.0, 0.0, 1.0, 1.0], 0.2, [0.0, -1.0], 1.0, false),
            Arc::new(MetalOp::Copy(MetalTexCopy {
                src: rt.clone(),
                dst: copy.clone(),
                whole: true,
                src_slice: 0,
                src_level: 0,
                src_origin: [0; 3],
                size: [16, 16, 1],
                dst_slice: 0,
                dst_level: 0,
                dst_origin: [0; 3],
            })),
            Arc::new(MetalOp::Ambient(ambient)),
        ];
        let mut rec = Recorder::new(&dir, 9, 64 << 20).unwrap();
        let seen = rec.before(&backend, &run1);
        assert!(backend.run(&run1));
        rec.after(&seen);
        // The CPU between the runs: the title-filled source texture and a
        // vertex change.
        fill_r8(&backend, &src.texture, 90);
        // SAFETY: inside the committed page; the GPU is idle.
        unsafe { positions.add(1).write([0.5, 0.0]) };
        let seen = rec.before(&backend, &run2);
        assert!(backend.run(&run2));
        rec.after(&seen);
        stats = rec.finish(Some(&backend)).unwrap();
        println!("recorded: {stats:?}");
        drop((run1, run2));
    }
    // SAFETY: nothing uses the reservation any more.
    assert_ne!(unsafe { VirtualFree(reserved as *mut core::ffi::c_void, 0, MEM_RELEASE) }, 0);

    assert_eq!((stats.runs, stats.ops), (2, 9));
    assert_eq!(stats.kinds.get(&KIND_DRAW), Some(&5));
    assert_eq!(stats.kinds.get(&KIND_CLEAR), Some(&2));
    assert_eq!((stats.kinds.get(&KIND_COPY), stats.kinds.get(&KIND_AMBIENT)), (Some(&1), Some(&1)));
    assert!(stats.unsupported.is_empty());
    assert_eq!(stats.unknown + stats.heap_unknown, 0, "unknown: {:x?}", stats.unknown_samples);
    assert_eq!((stats.graphics_pipelines, stats.graphics_without_source), (2, 0));
    assert_eq!(stats.textures.textures, 5);
    assert_eq!(stats.textures.unread, 0);
    assert_eq!(stats.textures.versions, 6, "five textures, and the source again after the CPU changed it");
    assert_eq!(stats.textures.reference, 4, "target, depth, copy and ambient textures");
    assert!(stats.stopped.is_none());

    let pkg = Package::open(&dir).unwrap();
    // The reference target: run 2's full-target draw sampled the source as
    // the CPU left it (90), the blue triangle in front of it.
    {
        let rt_id = pkg.texture_defs.iter().find(|d| d.key == K_RT as u64).unwrap().id;
        let r = pkg.texture_reference.iter().find(|r| r.id == rt_id).unwrap();
        let bin = std::fs::read(dir.join("reference").join("textures.bin")).unwrap();
        let px: std::collections::BTreeSet<[u8; 4]> = bin[r.offset as usize..(r.offset + r.len) as usize].chunks_exact(4).map(|p| p.try_into().unwrap()).collect();
        println!("reference target colours: {px:?}");
        assert!(px.contains(&[90, 0, 0, 255]) && px.contains(&[0, 0, 255, 255]), "{px:?}");
    }
    replay::place(&pkg).unwrap_or_else(|e| panic!("cannot place the title memory: {e}"));
    let backend = MetalBackend::new().unwrap();
    let report = replay::replay(&backend, &pkg, &Options::default()).unwrap();
    println!("replay: {report:?}");
    assert_eq!(report.failed_run, None);
    assert_eq!((report.runs, report.ops_run), (2, 9));
    assert_eq!((report.graphics_built, report.textures_made, report.texture_uploads), (2, 5, 6));
    assert!(report.graphics_errors.is_empty() && report.texture_errors.is_empty(), "{report:?}");
    assert_eq!(report.skipped_unsupported + report.skipped_no_pipeline + report.skipped_memory_directory + report.skipped_unmapped + report.skipped_texture, 0);
    assert_eq!(report.zeroed_unknown + report.zeroed_texture + report.unmapped_title, 0);
    let cmp = replay::compare_textures(&backend, &pkg).unwrap();
    println!("textures: {cmp:?}");
    assert_eq!(cmp.pages, 4);
    assert!(cmp.differing.is_empty(), "textures differ (id, first byte, bytes): {:?}", cmp.differing);
    let _ = std::fs::remove_dir_all(&dir);
}
