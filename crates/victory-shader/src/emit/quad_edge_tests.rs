use super::{*, container::SigElement};
use crate::native::StageIo;

const TRIANGLE: u32 = 1 << 10 | 2 << 20;

/// A pixel program of `words` with no user data.
fn emit_native(words: &[u32], input_addr: u32) -> NativeProgram {
    NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 4,
        layout: vec![],
        used: vec![],
        io: StageIo::Ps {
            inputs: vec![],
            input_addr,
            db_shader_control: 0,
            patches: vec![],
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// A pixel program that takes its neighbour's position across its quad
/// (`ds_swizzle_b32`, as the console's compiler makes derivatives) and
/// exports the differences in x and y: 1 and -1 in every pixel of a
/// triangle, those at its edges too, whose neighbours are helpers.
#[test]
#[cfg(windows)]
fn quad_neighbours_at_a_triangles_edge_are_read() {
    use super::host::{Draw, Host, Root};
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_SHADER_VISIBILITY_MESH, D3D12_SHADER_VISIBILITY_PIXEL,
    };
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    // v2, v3: position x, y (input address: POS_X and POS_Y after the
    // two perspective barycentrics).
    let swizzle = |dst: u32, src: u32, pattern: u32| {
        [
            0xD800_0000 | 0x35 << 18 | 0x8000 | pattern,
            dst << 24 | src,
        ]
    };
    let mut words = vec![0xBEFC_0380u32]; // s_mov_b32 m0, 0
    words.extend(swizzle(4, 2, 0xB1)); // lanes 1 0 3 2
    words.extend(swizzle(5, 3, 0x4E)); // lanes 2 3 0 1
    words.extend(swizzle(6, 2, 0xA0)); // lanes 0 0 2 2
    words.extend([
        0x0808_0504,  // v_sub_f32 v4, v4, v2
        0x080A_0705,  // v_sub_f32 v5, v5, v3
        0x080C_0506,  // v_sub_f32 v6, v6, v2
        0xF800_180F,
        0x0302_0504, // exp mrt0 v4, v5, v2, v3 done vm
        0xBF81_0000,
    ]);
    let mut native = emit_native(&words, 1 << 1 | 1 << 8 | 1 << 9);
    native.vgprs = 8;
    let out = SigElement {
        name: "SV_Target".into(),
        index: 0,
        kind: 16,
        comp: 9,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 15,
        half: false,
    };
    let position = SigElement {
        name: "SV_Position".into(),
        index: 0,
        kind: 3,
        comp: 9,
        interp: 4,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 3,
        half: false,
    };
    let isg1 = container::signature(&[position], true);
    let osg1 = container::signature(&[out], false);
    let bytes = crate::dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
    let c = Container::parse(&bytes).unwrap();
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let ps = emit(&c, &native, &p, &st, &io, &res).unwrap();
    if let Some(path) = super::validator::Validator::default_path() {
        let mut v = super::validator::Validator::load(Some(&path)).unwrap();
        v.validate(&mut ps.dxil.clone()).unwrap();
    }
    let ms = triangle_mesh_to(TRIANGLE, 1.3).unwrap();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            continue;
        };
        let Ok(root) = host.root_signature_for(&[
            (&ms, D3D12_SHADER_VISIBILITY_MESH),
            (&ps, D3D12_SHADER_VISIBILITY_PIXEL),
        ]) else {
            continue;
        };
        let Ok(pso) = host.mesh_pipeline_with(&root, &ms.dxil, &ps.dxil) else {
            eprintln!("no mesh pipeline on {label}; skipped");
            continue;
        };
        let texels = host
            .draw(
                &root,
                &pso,
                &Draw {
                    roots: vec![(0, Root::Constants(vec![0])), (1, Root::Constants(vec![0]))],
                    textures: &[],
                    samplers: &[],
                    vertices: &[],
                    stride: 0,
                    size: (16, 16),
                    mesh_groups: 1,
                },
            )
            .unwrap_or_else(|e| panic!("{label}: draw: {e}"));
        let lit: Vec<(usize, &[f32; 4])> = texels.iter().enumerate().filter(|t| t.1[2] != 0.0).collect();
        assert!((20..240).contains(&lit.len()), "{label}: {} texels lit", lit.len());
        let bad: Vec<_> = lit
            .iter()
            .filter(|(n, t)| {
                let x = if n % 2 == 0 { 1.0 } else { -1.0 };
                let y = if n / 16 % 2 == 0 { 1.0 } else { -1.0 };
                (t[0], t[1]) != (x, y)
            })
            .collect();
        assert!(bad.is_empty(), "{label}: {} of {} texels: {:?}", bad.len(), lit.len(), bad.first());
    }
}

/// The triangle with its far corners at `reach` - 1.
fn triangle_mesh_to(prim: u32, reach: f32) -> Result<HostShader, ShaderError> {
    // (the second corner's y comes out as 2 x its lane bit: half of it)
    let [x, y] = [reach.to_bits(), (reach / 2.0).to_bits()];
    let words = [
        0xD765_0001u32, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0x0001_00C1,
        0x3604_0281, // v_and_b32 v2, 1, v1
        0x3606_0282, // v_and_b32 v3, 2, v1
        0x7E04_0D02, // v_cvt_f32_u32 v2, v2
        0x7E06_0D03, // v_cvt_f32_u32 v3, v3
        0x1008_04FF, // v_mul_f32 v4, reach, v2
        x,
        0x0608_08F3, // v_add_f32 v4, -1.0, v4
        0x100A_06FF, // v_mul_f32 v5, reach / 2, v3
        y,
        0x060A_0AF3, // v_add_f32 v5, -1.0, v5
        0x7E0C_02F0, // v_mov_b32 v6, 0.5
        0x7E0E_02F2, // v_mov_b32 v7, 1.0
        0x7E12_02FF, // v_mov_b32 v9, prim
        prim,
        0xF800_0941, // exp prim v9, off, off, off done
        0x0000_0009,
        0xF800_08CF, // exp pos0 v4, v5, v6, v7 done
        0x0706_0504,
        0xBF81_0000, // s_endpgm
    ];
    let native = NativeProgram {
        stage: HwStage::Ms,
        wave64: false,
        vgprs: 10,
        layout: vec![],
        used: vec![],
        io: StageIo::Ms(crate::native::MeshIo {
            outputs: vec![],
            clip_cntl: 0,
            threads: 32,
            max_vertices: 32,
            max_primitives: 32,
            topology: 2,
            lds_bytes: 0,
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let position = SigElement {
        name: "SV_Position".into(),
        index: 0,
        kind: 3,
        comp: 9,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0xF,
        half: false,
    };
    let osg1 = container::signature(&[position], false);
    let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
    let c = Container::parse(&bytes)?;
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    emit(&c, &native, &p, &st, &io, &res)
}

