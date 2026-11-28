//! The ambient stand-in (Lead B): its settings and packing on the CPU, and
//! under F85 Wine with Metal API validation that its kernel writes the
//! right R9G9B9E5 words, only into its view, and in order between the
//! dispatches around it.
use std::sync::Arc;

use crate::backend::metal::{MetalBackend, MetalCmd, MetalCompute, MetalOp};
use crate::backend::metal_ambient::*;
use crate::backend::metal_texture::MetalTexture;
use victory_metal::sys::{WMTPixelFormat, WMTTextureType};
use victory_metal::sys::{WMTResourceOptions, WMTTextureUsage};
use victory_metal::texture::TextureDescriptor;

#[test]
fn settings() {
    assert_eq!(parse(""), Set::Off);
    assert_eq!(parse("0"), Set::Off);
    assert_eq!(parse("off"), Set::Off);
    // The estimate mode is not on the Metal path.
    assert_eq!(parse("0.8"), Set::Off);
    assert_eq!(parse("constant:0.3"), Set::Constant(0.3));
    assert_eq!(parse(" constant: 1.5 "), Set::Constant(1.5));
    assert_eq!(parse("constant:0"), Set::Off);
    assert_eq!(parse("constant:-1"), Set::Off);
    assert_eq!(parse("constant:nan"), Set::Off);
    assert_eq!(parse("constant:inf"), Set::Off);
}

#[test]
fn the_resolve_by_name() {
    assert!(is_probe_resolve(b"..CS_ScreenProbes_TemporalResolve_ReflectiveCaustics_34fb51b8.."));
    assert!(is_probe_resolve(b"CS_ScreenProbes_TemporalResolve_a911b49c_Wrapped"));
    assert!(!is_probe_resolve(b"CS_TemporalResolve"));
    assert!(!is_probe_resolve(b"CS_ScreenProbes_TraceRays_BVH"));
}

#[test]
fn packing() {
    assert_eq!(pack_rgb9e5([0.0; 3]), 0);
    // 1.0: exponent 16 (bias 15, 9-bit mantissa 256).
    assert_eq!(pack_rgb9e5([1.0, 0.0, 0.0]), 256 | 16 << 27);
    for level in [0.001f32, 0.05, 0.3, 1.0, 7.5, 300.0] {
        let c = TINT.map(|t| t * level);
        let back = unpack_rgb9e5(pack_rgb9e5(c));
        for (a, b) in c.iter().zip(back) {
            assert!((a - b).abs() <= a.max(c[0]) / 256.0, "level {level}: {c:?} -> {back:?}");
        }
    }
    // Clamped at RGB9E5's largest value, never wrapped.
    let top = unpack_rgb9e5(pack_rgb9e5([1e9; 3]));
    assert!(top[0] > 60000.0 && top[0] <= 65408.0, "{top:?}");
}

const SENTINEL: u32 = 0xA5A5_A5A5;
const W: u32 = 13;
const H: u32 = 7;

/// An RGB9E5 2D array the way the title's light image is made (UAV, view
/// casts), filled with `SENTINEL` in every level and slice.
fn light_texture(b: &MetalBackend, w: u32, h: u32, mips: u32, slices: u32) -> MetalTexture {
    let desc = TextureDescriptor {
        pixel_format: WMTPixelFormat::WMTPixelFormatRGB9E5Float,
        width: w,
        height: h,
        array_length: slices,
        type_: WMTTextureType::WMTTextureType2DArray,
        mipmap_level_count: mips,
        usage: WMTTextureUsage(1 | 2 | 0x10),
        options: WMTResourceOptions(0x20),
        ..Default::default()
    };
    let texture = b.device.new_texture(&desc).unwrap();
    b.add_resident(&[texture.handle()]);
    let fill = b.device.shared_buffer(&SENTINEL.to_le_bytes().repeat((w * h) as usize)).unwrap();
    let cb = b.queue.command_buffer().unwrap();
    let mut blit = cb.blit_encoder().unwrap();
    for level in 0..mips {
        let (lw, lh) = ((w >> level).max(1), (h >> level).max(1));
        for slice in 0..slices {
            blit.copy_from_buffer_to_texture(&fill, 0, lw * 4, lw * lh * 4, [lw.into(), lh.into(), 1], &texture, slice, level, [0; 3]);
        }
    }
    blit.finish();
    crate::backend::metal::submit_checked(&cb, &[], true, "ambient test fill").unwrap();
    // SAFETY: integer-only record; only the fields set below are read.
    let mut d: crate::resource::ResourceDesc = unsafe { std::mem::zeroed() };
    d.flags = 4; // unordered access
    d.sample_count = 1;
    MetalTexture { texture, pixel: desc.pixel_format, desc: d, mips, slices, companion: Default::default() }
}

/// Every level and slice of `t`, as words.
fn read_back(b: &MetalBackend, t: &MetalTexture) -> Vec<Vec<u32>> {
    let (w, h) = (t.texture.width() as u32, t.texture.height() as u32);
    let mut out = Vec::new();
    for level in 0..t.mips {
        let (lw, lh) = ((w >> level).max(1), (h >> level).max(1));
        for slice in 0..t.slices {
            let (buffer, ptr) = b.device.shared_buffer_mapped(&vec![0u8; (lw * lh * 4) as usize]).unwrap();
            let cb = b.queue.command_buffer().unwrap();
            let mut blit = cb.blit_encoder().unwrap();
            blit.copy_from_texture_to_buffer(&t.texture, slice, level, [0; 3], [lw.into(), lh.into(), 1], &buffer, 0, lw * 4, lw * lh * 4);
            blit.finish();
            crate::backend::metal::submit_checked(&cb, &[], true, "ambient test read").unwrap();
            // SAFETY: the copy completed; the buffer holds lw*lh words.
            out.push(unsafe { std::slice::from_raw_parts(ptr.cast::<u32>(), (lw * lh) as usize) }.to_vec());
        }
    }
    out
}

/// D3D12_UNORDERED_ACCESS_VIEW_DESC words of a one-slice 2D array view.
fn uav_words(mip: u32, slice: u32) -> [u32; 6] {
    [42, 5, mip, slice, 1, 0]
}

/// The kernel writes `pack(tint * level)` (as the CPU reference packs it)
/// into every texel of the view's level and slice, and nothing else.
#[test]
fn writes_rgb9e5_into_its_view_only() {
    let _pool = victory_metal::Pool::new();
    let b = MetalBackend::new().unwrap();
    for (mip, slice, level) in [(0u32, 1u32, 0.3f32), (1, 0, 1.0), (0, 0, 0.002), (1, 1, 70000.0)] {
        let t = light_texture(&b, W, H, 2, 2);
        b.commit_residency();
        let words = uav_words(mip, slice);
        let op = MetalAmbient::new(&t, Some(&words), level).expect("an R32Uint view of RGB9E5");
        assert_eq!(op.size, [(W >> mip).max(1), (H >> mip).max(1)]);
        assert!(b.run(&[Arc::new(MetalOp::Ambient(op))]));
        let want = pack_rgb9e5(TINT.map(|c| c * level));
        let got = read_back(&b, &t);
        for (k, sub) in got.iter().enumerate() {
            let (l, s) = (k as u32 / 2, k as u32 % 2);
            let expected = if (l, s) == (mip, slice) { want } else { SENTINEL };
            for (n, &v) in sub.iter().enumerate() {
                assert_eq!(v, expected, "level {level}: mip {l} slice {s} texel {n}: {v:#010x}, want {expected:#010x}");
            }
        }
        println!("mip {mip} slice {slice} level {level}: {want:#010x} in {} texels", got[(mip * 2 + slice) as usize].len());
    }
}

/// Views the stand-in does not take.
#[test]
fn refuses_other_views() {
    let _pool = victory_metal::Pool::new();
    let b = MetalBackend::new().unwrap();
    let t = light_texture(&b, W, H, 2, 2);
    assert!(MetalAmbient::new(&t, Some(&[42, 5, 0, 0, 2, 0]), 0.3).is_none(), "two slices");
    assert!(MetalAmbient::new(&t, Some(&[42, 5, 2, 0, 1, 0]), 0.3).is_none(), "mip out of range");
    assert!(MetalAmbient::new(&t, Some(&[42, 5, 0, 2, 1, 0]), 0.3).is_none(), "slice out of range");
    assert!(MetalAmbient::new(&t, Some(&[42, 8, 0, 0, 1, 0]), 0.3).is_none(), "3D view");
    let mut no_uav = light_texture(&b, W, H, 1, 1);
    no_uav.desc.flags = 0;
    assert!(MetalAmbient::new(&no_uav, None, 0.3).is_none(), "not unordered access");
}

/// Stand-ins for the title's programs around the resolve, reaching the
/// light image through the top-level argument buffer by its resource ID
/// (untracked, as title dispatches reach theirs): `before` writes garbage
/// over it, `after` copies it out.
const AROUND: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct Top {
    texture2d<uint, access::read_write> light;
    device uint* out;
    uint width;
    uint height;
    uint tag;
};
kernel void before(constant Top& top [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
    if (p.x >= top.width || p.y >= top.height) return;
    top.light.write(uint4(0xDEAD0000u + top.tag), p);
}
kernel void after(constant Top& top [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
    if (p.x >= top.width || p.y >= top.height) return;
    top.out[p.y * top.width + p.x] = top.light.read(p).x;
}
"#;

fn top(light: u64, out: u64, size: [u32; 2], tag: u32) -> Vec<u8> {
    let mut t = Vec::with_capacity(32);
    t.extend_from_slice(&light.to_le_bytes());
    t.extend_from_slice(&out.to_le_bytes());
    for v in [size[0], size[1], tag, 0] {
        t.extend_from_slice(&v.to_le_bytes());
    }
    t
}

/// Rounds of `before`, the stand-in and `after` in one batch (bounded
/// command buffers, ordinary splitting): every `after` sees exactly the
/// stand-in's words, never `before`'s garbage nor an older round's level.
#[test]
fn runs_between_the_dispatches_around_it() {
    let _pool = victory_metal::Pool::new();
    let b = MetalBackend::new().unwrap();
    let (w, h) = (256u32, 128u32);
    let t = light_texture(&b, w, h, 1, 1);
    let library = b.device.library_msl(AROUND).unwrap();
    let compute = |name: &str| {
        Arc::new(MetalCompute {
            cost_ns: Default::default(),
            name: format!("ambient order {name}"),
            pso: b.device.compute_pipeline_from_library(&library, name).unwrap(),
            threadgroup: [8, 8, 1],
            top_level: victory_metal::ir::TopLevel::new(Vec::new(), 32),
        })
    };
    let (before, after) = (compute("before"), compute("after"));
    const ROUNDS: usize = 6;
    let words = (w * h) as usize;
    let (out, out_ptr) = b.device.shared_buffer_mapped(&vec![0u8; words * 4 * ROUNDS]).unwrap();
    b.add_resident(&[out.handle()]);
    b.commit_residency();
    let mut ops: Vec<Arc<MetalOp>> = Vec::new();
    let mut keep = Vec::new();
    let mut want = Vec::new();
    for round in 0..ROUNDS {
        let level = 0.1 * (round + 1) as f32;
        let op = MetalAmbient::new(&t, Some(&uav_words(0, 0)), level).unwrap();
        // The title dispatches reach the image through a view of their own.
        let view = MetalAmbient::new(&t, Some(&uav_words(0, 0)), 1.0).unwrap().view;
        let id = view.gpu_resource_id();
        let groups = [u64::from(w.div_ceil(8)), u64::from(h.div_ceil(8)), 1];
        let dispatch = |pso: &Arc<MetalCompute>, out_at: u64, tag: u32| {
            Arc::new(MetalOp::Dispatch(MetalCmd {
                pso: pso.clone(),
                heap: None,
                sampler_heap: None,
                top: top(id, out_at, [w, h], tag),
                groups,
                tables: Vec::new(),
                loop_counts: None,
            }))
        };
        ops.push(dispatch(&before, 0, round as u32));
        ops.push(Arc::new(MetalOp::Ambient(op)));
        ops.push(dispatch(&after, out.gpu_address + (round * words * 4) as u64, 0));
        keep.push(view);
        want.push(pack_rgb9e5(TINT.map(|c| c * level)));
    }
    assert!(b.run(&ops));
    // SAFETY: run waited for every command buffer; ROUNDS * words words.
    let got = unsafe { std::slice::from_raw_parts(out_ptr.cast::<u32>(), words * ROUNDS) };
    for (round, &expected) in want.iter().enumerate() {
        let r = &got[round * words..(round + 1) * words];
        let wrong = r.iter().filter(|&&v| v != expected).count();
        assert_eq!(wrong, 0, "round {round}: {wrong} texels not {expected:#010x} (first {:#010x})", r[0]);
    }
    drop(keep);
    println!("{ROUNDS} rounds of before / ambient / after in order");
}
