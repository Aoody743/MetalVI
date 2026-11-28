//! P1 binding layer tests: the reverse index (CPU only), the shadow
//! comparison on synthetic compute dispatches (0 mismatches for the binding
//! kinds the layer supports, mutations caught), and `on` mode giving the
//! old path's GPU output without the D3D12-era record calls. Kernels of
//! one 16-thread group; no game.

use super::*;
use crate::backend::metal::{MetalCmd, MetalCompute, MetalOp};
use std::sync::{Arc, OnceLock};
use victory_metal::ir::{Location, ResourceType as IrType};
use victory_shader::emit::{HostShader, SLOT_STRIDES, USER_GROUP_COUNT_X, UserData};
use victory_shader::{HwStage, ResourceType, UsedResource};

static TEST_BACKEND: OnceLock<&'static MetalBackend> = OnceLock::new();

pub(super) fn backend() -> Option<&'static MetalBackend> {
    TEST_BACKEND.get().copied()
}

// ---------------------------------------------------------------------------
// Reverse index (no Metal).

fn bare_heap(count: u32) -> HeapState {
    HeapState { count, samplers: false, gpu: None, slots: vec![Slot::default(); count as usize], deps: HashMap::new(), pending: HashSet::new() }
}

fn clean(p: &mut Prep, key: Key, dep: Dep) {
    let h = p.heaps.get_mut(&key.0).unwrap();
    h.slots[key.1 as usize].state = State::Clean;
    p.set_dep(key, Some(dep));
}

fn state(p: &Prep, key: Key) -> State {
    p.heaps[&key.0].slots[key.1 as usize].state
}

#[test]
fn remaps_and_releases_make_only_their_slots_stale() {
    let mut p = Prep::default();
    let (h, other) = (0x1000usize, 0x2000usize);
    p.heaps.insert(h, bare_heap(8));
    p.heaps.insert(other, bare_heap(8));
    let mb = 1u64 << 20;
    clean(&mut p, (h, 0), Dep::Range { start: 0x40_0000_0000, end: 0x40_0000_0000 + mb, resource: 0x77 });
    clean(&mut p, (h, 1), Dep::Range { start: 0x40_0000_0000 + 8 * mb, end: 0x40_0000_0000 + 9 * mb, resource: 0 });
    clean(&mut p, (h, 2), Dep::Texture { resource: 0x88 });
    // Over 64 granules: on the list checked at every remap.
    clean(&mut p, (other, 3), Dep::Range { start: 0x50_0000_0000, end: 0x50_0000_0000 + 512 * mb, resource: 0 });
    clean(&mut p, (other, 4), Dep::Range { start: 0x60_0000_0000, end: 0x60_0000_1000, resource: 0x77 });
    assert!(p.huge.contains(&(other, 3)));

    // A remap over the first view's second half: it and the texture (any
    // remap, conservatively) go stale; the view 8 MiB on does not, nor
    // the views elsewhere.
    p.remapped(0x40_0000_0000 + mb / 2, mb);
    assert_eq!(state(&p, (h, 0)), State::Stale);
    assert_eq!(state(&p, (h, 1)), State::Clean);
    assert_eq!(state(&p, (h, 2)), State::Stale);
    assert_eq!(state(&p, (other, 3)), State::Clean);
    assert_eq!(state(&p, (other, 4)), State::Clean);
    // Inside the huge view.
    p.remapped(0x50_0000_0000 + 300 * mb, 4096);
    assert_eq!(state(&p, (other, 3)), State::Stale);
    // Releasing buffer 0x77: the views of it, in both heaps.
    clean(&mut p, (h, 0), Dep::Range { start: 0x40_0000_0000, end: 0x40_0000_0000 + mb, resource: 0x77 });
    p.resource_gone(0x77);
    assert_eq!(state(&p, (h, 0)), State::Stale);
    assert_eq!(state(&p, (other, 4)), State::Stale);
    assert_eq!(state(&p, (h, 1)), State::Clean);
    // A slot rewritten to another view leaves the old view's index.
    clean(&mut p, (h, 1), Dep::Range { start: 0x70_0000_0000, end: 0x70_0000_1000, resource: 0 });
    p.remapped(0x40_0000_0000 + 8 * mb, mb);
    assert_eq!(state(&p, (h, 1)), State::Clean);
    // A heap released: its slots leave every index.
    p.heap_gone(h);
    assert!(p.granules.values().all(|s| s.iter().all(|k| k.0 != h)));
    assert!(p.resources.values().all(|s| s.iter().all(|k| k.0 != h)));
    assert!(!p.textures.iter().any(|k| k.0 == h));
}

#[test]
fn title_buffers_by_address() {
    set_mode(Mode::Shadow);
    let start = 0x7b51_0000_0000u64;
    assert_eq!(buffer_extent(start + 8), None);
    buffer_made(0x91, start, 0x1000);
    buffer_made(0x92, start + 0x800, 0x100);
    // The one starting last at or below, if it holds the address.
    assert_eq!(buffer_extent(start + 0x810), Some((start + 0x800, start + 0x900)));
    assert_eq!(buffer_extent(start + 0x950), Some((start, start + 0x1000)));
    buffer_gone(0x92, start + 0x800);
    assert_eq!(buffer_extent(start + 0x810), Some((start, start + 0x1000)));
    buffer_gone(0x91, start);
    assert_eq!(buffer_extent(start + 8), None);
}

// ---------------------------------------------------------------------------
// Synthetic dispatches.

/// MSC's compute ABI: view heap at buffer 0, top level at 2 (user data;
/// root CBV; title root SRV as a one-view table; victory's stride table;
/// title root UAV as a one-view table). Reads every kind of binding the
/// layer makes and writes 16 words.
const MSL: &str = "#include <metal_stdlib>
using namespace metal;
struct Entry { device uint *va; ulong tex; ulong meta; };
struct Top { uint k; uint tbl; uint gx; uint n; device const uint *cbv; const device Entry *srv; device const uint *strides; const device Entry *uav; };
kernel void k(constant Top &t [[buffer(2)]], const device Entry *heap [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= 16) { return; }
    uint base = t.tbl / 32;
    device const uint *cb = heap[base].va;
    device const uint *raw = heap[base + 1].va;
    device const uint *structured = heap[base + 2].va;
    device const uint *picked = heap[base + t.n].va;
    device uint *out = t.uav->va;
    out[tid] = t.cbv[tid] + t.srv->va[tid] * t.k + cb[tid] + raw[tid] * 3u + structured[tid * 2u] * 5u + picked[tid] * 7u
        + t.gx * 11u + t.strides[2] * 13u;
}
";

const R32_UINT: u32 = 42;
/// Regions of the title memory, 64 KiB apart.
const CBV: u64 = 0;
const RAW: u64 = 1;
const STRUCTURED: u64 = 2;
const OUT_HEAP: u64 = 3;
const TYPED: u64 = 4;
const ROOT_CBV: u64 = 5;
const ROOT_SRV: u64 = 6;
const ROOT_UAV: u64 = 7;
const OTHER: u64 = 8;
const REGION: u64 = 64 << 10;
const REGIONS: u64 = 10;

struct Fixture {
    m: &'static MetalBackend,
    title: u64,
    views: This,
    samplers: This,
    t: Translated,
    top: TopLevel,
    pso: Arc<MetalCompute>,
}

fn region(f: &Fixture, r: u64) -> u64 {
    f.title + r * REGION
}

fn srv_words(address: u64, first: u64, n: u32, stride: u32, format: u32, raw: bool) -> [u32; 16] {
    let mut d = [0u32; 16];
    d[..10].copy_from_slice(&[format, 1, 0x1688, 0, first as u32, (first >> 32) as u32, n, stride, u32::from(raw), 0]);
    (d[10], d[11]) = (address as u32, (address >> 32) as u32);
    d
}

fn uav_words(address: u64, n: u32) -> [u32; 16] {
    let mut d = [0u32; 16];
    d[..10].copy_from_slice(&[R32_TYPELESS, 1, 0, 0, n, 0, 0, 0, 1, 0]);
    (d[10], d[11]) = (address as u32, (address >> 32) as u32);
    d
}
const R32_TYPELESS: u32 = 39;
const BUFFER_RESOURCE: [u32; 14] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

fn slot_handle(heap: This, index: u32) -> u64 {
    crate::descriptor::heap(heap).handle(index).unwrap()
}

impl Fixture {
    fn new() -> Self {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        // note_slot_dirty (the hook) runs with the Metal backend on.
        // SAFETY: set before anything reads it, one test per process.
        unsafe { std::env::set_var("VICTORY_METAL", "compute") };
        assert_eq!(super::super::mode(), super::super::Mode::Compute);
        let m: &'static MetalBackend = Box::leak(Box::new(MetalBackend::new().unwrap()));
        let _ = TEST_BACKEND.set(m);

        // Title memory committed in the middle of a reservation, so spans
        // over it end at the guard space.
        // SAFETY: fresh reservation, kept for the process.
        let reserved = unsafe { VirtualAlloc(std::ptr::null(), 32 << 20, MEM_RESERVE, PAGE_READWRITE) } as u64;
        assert_ne!(reserved, 0);
        // SAFETY: inside the reservation.
        let title = unsafe { VirtualAlloc((reserved + (16 << 20)) as *const core::ffi::c_void, (REGIONS * REGION) as usize, MEM_COMMIT, PAGE_READWRITE) } as u64;
        assert_eq!(title, reserved + (16 << 20));

        let host = HostShader {
            stage: HwStage::Cs,
            dxil: Vec::new(),
            user_data: vec![
                UserData { index: 0, slot: 0, root: RootType::Constants, dword: 0 },
                UserData { index: 1, slot: 1, root: RootType::Table, dword: 0 },
                UserData { index: 2, slot: 0, root: RootType::Other(USER_GROUP_COUNT_X), dword: 0 },
                UserData { index: 3, slot: 0, root: RootType::Constants, dword: 1 },
            ],
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: vec![2],
            root_srvs: vec![3, SLOT_STRIDES],
            root_uavs: vec![4],
            root_indexed_views: vec![victory_shader::resource::RootIndexedView { table_slot: 1, constant_slot: 0, constant_dword: 1, byte_offset: 0, shift: 5 }],
        };
        let used = |slot: u8, i: u32, kind: ResourceType| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords: 8 * i,
            lower_bound: i,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot,
        };
        let t = Translated {
            host,
            threads: Some([16, 1, 1]),
            used: vec![
                used(1, 0, ResourceType::Cbv),
                used(1, 1, ResourceType::Srv),
                used(1, 2, ResourceType::Srv),
                used(1, 3, ResourceType::Uav),
                used(1, 4, ResourceType::Srv),
                used(1, 5, ResourceType::Srv),
                used(5, 0, ResourceType::Sampler),
            ],
            stores: true,
        };
        let loc = |kind, space, slot, offset, size| Location { kind, space, slot, offset, size };
        let top = TopLevel::new(
            vec![
                loc(IrType::Constant, 0, 0, 0, 16),
                loc(IrType::Cbv, 1, 2, 16, 8),
                loc(IrType::Table, u32::MAX, u32::MAX, 24, 8),
                loc(IrType::Srv, 1, u32::from(SLOT_STRIDES), 32, 8),
                loc(IrType::Table, u32::MAX, u32::MAX, 40, 8),
            ],
            48,
        );
        assert_eq!(root_layout(&t.host).len(), 5);
        let pso = m.device.compute_pipeline_msl(MSL, "k").unwrap();
        let pso = Arc::new(MetalCompute { cost_ns: Default::default(), name: "prep test".into(), pso, threadgroup: [16, 1, 1], top_level: top.clone() });
        let f = Self { m, title, views: crate::descriptor::test_heap(16, false), samplers: crate::descriptor::test_heap(4, true), t, top, pso };
        // Root views name title buffers (what resource.rs tells both tables).
        for (k, r) in [ROOT_CBV, ROOT_SRV, ROOT_UAV, OTHER].into_iter().enumerate() {
            crate::shadow::register(0x5150_0000 + k, region(&f, r), REGION);
            buffer_made(0x5150_0000 + k, region(&f, r), REGION);
        }
        f.fill();
        f
    }

    fn words(&self, r: u64) -> *mut u32 {
        region(self, r) as *mut u32
    }

    fn fill(&self) {
        for r in 0..REGIONS {
            for i in 0..64u32 {
                // SAFETY: inside the committed title memory; no GPU work in flight.
                unsafe { self.words(r).add(i as usize).write_volatile((r as u32 + 1) * 1000 + i) };
            }
        }
    }

    /// The title's descriptor writes: view heap slots 0..=4, slot 5 left
    /// empty, one sampler.
    fn write_descriptors(&self) {
        let null = std::ptr::null();
        let cbv = region(self, CBV);
        crate::descriptor::create_cbv(null, &[cbv as u32, (cbv >> 32) as u32, 256, 0], slot_handle(self.views, 0));
        crate::descriptor::create_placed_srv(null, &BUFFER_RESOURCE, &srv_words(region(self, RAW), 0, 64, 0, R32_TYPELESS, true), slot_handle(self.views, 1));
        crate::descriptor::create_placed_srv(null, &BUFFER_RESOURCE, &srv_words(region(self, STRUCTURED), 0, 32, 8, 0, false), slot_handle(self.views, 2));
        crate::descriptor::create_placed_uav(null, &BUFFER_RESOURCE, &uav_words(region(self, OUT_HEAP), 64), slot_handle(self.views, 3));
        crate::descriptor::create_placed_srv(null, &BUFFER_RESOURCE, &srv_words(region(self, TYPED), 0, 64, 0, R32_UINT, false), slot_handle(self.views, 4));
        let sampler: [u32; 13] = [0x15, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0x7f7f_ffff];
        crate::descriptor::create_sampler(null, &sampler, slot_handle(self.samplers, 0));
    }

    fn roots(&self, n: u32) -> Vec<Root> {
        vec![
            Root::Constants(vec![3, n]),
            Root::Table(slot_handle(self.views, 0)),
            Root::View(region(self, ROOT_CBV)),
            Root::View(region(self, ROOT_SRV) + 16),
            Root::View(region(self, ROOT_UAV)),
            Root::Table(slot_handle(self.samplers, 0)),
        ]
    }

    fn bind(&self, prep: bool, n: u32) -> Result<MetalStage, Drop> {
        let heaps = [self.views as usize, self.samplers as usize];
        crate::draw::metal_bind_for_test(self.m, prep, &self.t, heaps, self.roots(n), self.top.clone(), 16, [1, 1])
    }

    fn op(&self, s: MetalStage) -> Arc<MetalOp> {
        Arc::new(MetalOp::Dispatch(MetalCmd {
            pso: self.pso.clone(),
            heap: s.heap,
            sampler_heap: s.sampler_heap,
            top: s.top,
            groups: [1, 1, 1],
            tables: s.tables,
            loop_counts: None,
            after_barrier: true,
        }))
    }

    fn out(&self) -> Vec<u32> {
        // SAFETY: read after the GPU work completed.
        (0..16).map(|i| unsafe { self.words(ROOT_UAV).add(i).read_volatile() }).collect()
    }
}

fn mismatch_list() -> Vec<(&'static str, u64)> {
    named(&CAT_NAMES, mismatches())
}

/// Shadow mode over a run of dispatches: the first binds the heaps (their
/// slots resolved at that bind), then the title rewrites descriptors over
/// wrapped memory (resolved at the write), memory is remapped (slots over
/// it resolved again at their next use), a slot is picked by another root
/// constant. The layer's bindings equal the record step's every time.
#[test]
#[ignore = "requires Metal"]
fn shadow_matches_the_record_step() {
    let _pool = victory_metal::Pool::new();
    set_mode(Mode::Shadow);
    set_eager(true);
    let f = Fixture::new();
    f.write_descriptors();
    let prep_before = crate::draw::metal_prep();
    for n in [2u32, 1] {
        let s = f.bind(true, n).expect("binds");
        assert_eq!(s.outcome, Drop::Ran);
    }
    assert_ne!(crate::draw::metal_prep(), prep_before, "shadow runs the record step too");
    // Rewritten over memory that is wrapped now: resolved at the write.
    let resolved = stats()[Stat::WriteResolved as usize];
    f.write_descriptors();
    let null = std::ptr::null();
    crate::descriptor::create_placed_srv(null, &BUFFER_RESOURCE, &srv_words(region(&f, RAW), 4, 32, 0, R32_TYPELESS, true), slot_handle(f.views, 1));
    assert!(stats()[Stat::WriteResolved as usize] >= resolved + 4, "{:?}", named(&STAT_NAMES, stats()));
    f.bind(true, 2).unwrap();
    // A remap over the raw view's memory and the typed one's.
    crate::backend::metal::title_memory_remapped(region(&f, RAW), REGION);
    crate::backend::metal::title_memory_remapped(region(&f, TYPED), 4096);
    f.bind(true, 4).unwrap();
    f.bind(true, 3).unwrap();
    let st = stats();
    println!("stats {:?}", named(&STAT_NAMES, st));
    println!("mismatches {:?}", mismatch_list());
    assert_eq!(st[Stat::Compared as usize], 5);
    assert!(st[Stat::FlushResolved as usize] >= 6, "the heaps' slots resolved at the first bind");
    assert!(st[Stat::Hits as usize] > 0);
    assert!(st[Stat::Invalidated as usize] >= 2, "remapped slots went stale");
    assert_eq!(real_mismatches(), 0, "{:?}", mismatch_list());
}

/// The comparison catches what the layer gets wrong: user data it
/// computes wrongly, and a descriptor write it never heard of (the slot
/// written without the hook: the record step resolves it again at use,
/// the layer keeps the entry it had).
#[test]
#[ignore = "requires Metal"]
fn shadow_catches_mutations() {
    let _pool = victory_metal::Pool::new();
    set_mode(Mode::Shadow);
    let f = Fixture::new();
    f.write_descriptors();
    f.bind(true, 2).unwrap();
    assert_eq!(real_mismatches(), 0, "{:?}", mismatch_list());

    fault_user_data(true);
    f.bind(true, 2).unwrap();
    fault_user_data(false);
    assert_eq!(mismatches()[Cat::UserData as usize], 1, "{:?}", mismatch_list());

    // Slot 1 takes slot 3's descriptor words behind the hook's back.
    // SAFETY: both are 32-byte slots of the heap.
    unsafe {
        let words = (slot_handle(f.views, 3) as *const [u32; 8]).read_unaligned();
        (slot_handle(f.views, 1) as *mut [u32; 8]).write_unaligned(words);
    }
    f.bind(true, 2).unwrap();
    println!("mismatches {:?}", mismatch_list());
    for s in lock(&SAMPLES_SEEN).iter() {
        println!("sample {s}");
    }
    assert!(mismatches()[Cat::HeapEntry as usize] >= 1, "{:?}", mismatch_list());
    assert_eq!(mismatches()[Cat::UserData as usize], 1);
}

/// `on`: the same dispatch bound by the layer gives the record step's GPU
/// output, and binding it calls none of the record step's D3D12-era prep
/// (`draw::metal_prep` unchanged); the record step's binding calls them.
#[test]
#[ignore = "requires Metal"]
fn on_mode_output_matches_the_record_step() {
    let _pool = victory_metal::Pool::new();
    // On from the start: the title's writes reach the layer (resolved at
    // its first bind); the first binding below is the record step's alone.
    set_mode(Mode::On);
    let f = Fixture::new();
    f.write_descriptors();

    let before = crate::draw::metal_prep();
    let old = f.bind(false, 2).expect("record step binds");
    let after = crate::draw::metal_prep();
    assert_eq!(old.outcome, Drop::Ran);
    let old_calls: u64 = (0..3).map(|k| after[k] - before[k]).sum();
    assert!(old_calls >= 7, "heap syncs + slot refreshes + root extents: {before:?} -> {after:?}");
    assert!(f.m.run(&[f.op(old)]), "old bindings ran");
    let want = f.out();

    f.fill();
    assert_ne!(f.out(), want);
    set_mode(Mode::On);
    let before = crate::draw::metal_prep();
    let new = f.bind(true, 2).expect("layer binds");
    assert_eq!(crate::draw::metal_prep(), before, "on mode calls no record-step prep");
    assert_eq!(new.outcome, Drop::Ran);
    assert!(f.m.run(&[f.op(new)]), "new bindings ran");
    let got = f.out();
    println!("old {want:?}\nnew {got:?}\nstats {:?}", named(&STAT_NAMES, stats()));
    assert_eq!(got, want);
    assert!(stats()[Stat::FlushResolved as usize] >= 6, "the title's writes resolved at the first bind");
    // The expected words, worked out on the CPU (region r word i = (r+1)*1000 + i).
    let w = |r: u64, i: u32| (r as u32 + 1) * 1000 + i;
    for i in 0..16u32 {
        let stride_word = 8; // structured, 8 bytes, not integer-typed
        let e = w(ROOT_CBV, i) + w(ROOT_SRV, i + 4) * 3 + w(CBV, i) + w(RAW, i) * 3 + w(STRUCTURED, i * 2) * 5 + w(STRUCTURED, i) * 7 + 11 + stride_word * 13;
        assert_eq!(got[i as usize], e, "word {i}");
    }
    // Again: everything clean, still no record-step prep.
    let before = crate::draw::metal_prep();
    let hits = stats()[Stat::Hits as usize];
    f.bind(true, 2).unwrap();
    assert_eq!(crate::draw::metal_prep(), before);
    assert!(stats()[Stat::Hits as usize] >= hits + 7);
}
