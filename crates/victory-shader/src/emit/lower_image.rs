//! Image instructions, lowered (moved out of lower.rs to keep it under
//! the source size limit).

use super::*;

impl<'a> Lower<'a> {
    pub(super) fn image(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Image {
            data,
            addr,
            dmask,
            dim,
            unorm,
            a16,
            d16,
            tfe,
            lwe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        if *tfe || *lwe {
            return Err("an image access with tfe or lwe");
        }
        let name = inst.name();
        // A load or sample with no channel enabled returns the first.
        let dmask = if *dmask == 0 { &1 } else { dmask };
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let (kind, coords) = match dim {
            0 => (1, 1),
            1 => (2, 2),
            2 => (4, 3),
            // Sampled (face coordinates and face, see `cube_direction`) or
            // sized only.
            3 if name.starts_with("image_sample")
                || name.starts_with("image_gather")
                || name == "image_get_resinfo" =>
            {
                (5, 3)
            }
            4 => (6, 2),
            5 => (7, 3),
            _ => return Err("an image dimension (cube loads or MSAA)"),
        };
        // Address VGPRs in order.
        let mut regs = Vec::new();
        for a in addr {
            if let Operand::Vector { reg, dwords } = *a {
                regs.extend((0..dwords).map(|k| reg + k));
            }
        }
        // Full 32-bit address VGPR at `reg_idx` (offset, bias, z-compare).
        let addr_u32 = |l: &mut Self, reg_idx: usize| -> Result<V, Why> {
            let r = *regs.get(reg_idx).ok_or("an image address")?;
            let v = inst.read(Reg::V(r)).ok_or("an image address")?;
            l.i32_of(v)
        };
        // With a16, two 16-bit components per VGPR starting at `base_reg`, low half first.
        let addr_packed = |l: &mut Self, base_reg: usize, k: usize| -> Result<V, Why> {
            let x = addr_u32(l, base_reg + if *a16 { k / 2 } else { k })?;
            if !*a16 {
                return Ok(x);
            }
            let c = l.cur;
            let (sh, mask) = (l.s.m.i32(16 * (k as u32 % 2)), l.s.m.i32(0xFFFF));
            let x = l.s.m.bin(c, Bin::LShr, x, sh);
            Ok(l.s.m.bin(c, Bin::And, x, mask))
        };
        let addr_val = |l: &mut Self, k: usize| -> Result<V, Why> { addr_packed(l, 0, k) };
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked image")?;
        let store = name.starts_with("image_store");
        let uav = store
            || self
                .used_kind(src)
                .is_some_and(|u| u.kind == ResourceType::Uav);
        let props = (kind | if uav { 0x1000 } else { 0 }, 0x409);
        if let Some(op) = name.strip_prefix("image_atomic_") {
            // GLC returns the old value. DXIL implements subtraction by adding
            // the two's complement of the operand at the same width.
            let code = match op {
                "add" | "sub" => 0,
                "and" => 1,
                "or" => 2,
                "xor" => 3,
                "smin" => 4,
                "smax" => 5,
                "umin" => 6,
                "umax" => 7,
                "swap" => 8,
                "cmpswap" => u32::MAX,
                _ => return Err(name),
            };
            let Fields::Image { cache, .. } = &inst.fields else {
                unreachable!()
            };
            if *d16 || *a16 {
                return Err("a 16-bit image atomic");
            }
            // Compare-exchange consumes a replacement followed by a comparand.
            let compare = code == u32::MAX;
            let wide = match (compare, *dmask) {
                (false, 1) | (true, 3) => false,
                (false, 3) | (true, 15) => true,
                _ => return Err("an image atomic data mask"),
            };
            let iu = self.s.m.undef(self.s.t.i32);
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let Operand::Vector { reg, .. } = *data else {
                return Err("image atomic data");
            };
            let width = if wide { 2 } else { 1 };
            let vals = self.vgprs(inst, data, if compare { 2 * width } else { width })?;
            if wide && !host_has_image_atomic64() {
                self.s.flags |= flags::INT64 | flags::TYPED_UAV_LOADS;
                let original = self.join64(vals[0], vals[1]);
                let comparand = compare.then(|| self.join64(vals[2], vals[3]));
                // Lower 64-bit image atomics through an RG32Uint view on hosts without 64-bit
                // integer image atomics (Apple7 Metal): pixel uses ROV (`0x3000`) to serialize
                // overlapping fragments; compute lacks ROV and uses plain UAV (`0x1000`).
                let uav_kind = if self.s.pixel { kind | 0x3000 } else { kind | 0x1000 };
                let h = self.handle(site, src, false, (uav_kind, 0x205))?;
                let at = self.begin_guarded(guard);
                let cur = self.cur;
                let i32t = self.s.t.i32;
                let res_i32 = self.s.m.named_struct("dx.types.ResRet.i32", vec![i32t; 5]);
                let mut load_args = vec![h, iu];
                load_args.extend(&cs);
                load_args.extend([iu, iu, iu]);
                let loaded = self.s.op(
                    cur,
                    "textureLoad.i32",
                    Attr::ReadOnly,
                    res_i32,
                    66,
                    &load_args,
                );
                let old_lo = self.s.m.extract(cur, loaded, 0);
                let old_hi = self.s.m.extract(cur, loaded, 1);
                let old = self.join64(old_lo, old_hi);
                let (new_val, cond) = match op {
                    "add" => (self.s.m.bin(cur, Bin::Add, old, original), None),
                    "sub" => (self.s.m.bin(cur, Bin::Sub, old, original), None),
                    "and" => (self.s.m.bin(cur, Bin::And, old, original), None),
                    "or" => (self.s.m.bin(cur, Bin::Or, old, original), None),
                    "xor" => (self.s.m.bin(cur, Bin::Xor, old, original), None),
                    "swap" => (original, None),
                    "umax" => (original, Some(self.s.m.cmp(cur, pred::IUGT, original, old))),
                    "umin" => (original, Some(self.s.m.cmp(cur, pred::IULT, original, old))),
                    "smax" => (original, Some(self.s.m.cmp(cur, pred::SGT, original, old))),
                    "smin" => (original, Some(self.s.m.cmp(cur, pred::SLT, original, old))),
                    "cmpswap" => (
                        original,
                        Some(self.s.m.cmp(cur, pred::EQ, old, comparand.unwrap())),
                    ),
                    _ => return Err(name),
                };
                let emit_rov_store = |l: &mut Self, val: V| {
                    let (new_lo, new_hi) = l.split64(val);
                    let mut args = vec![h];
                    args.extend(&cs);
                    args.extend([new_lo, new_hi, new_lo, new_hi, l.s.m.i8(0xF)]);
                    let void = l.s.t.void;
                    l.s.op(l.cur, "textureStore.i32", Attr::NoUnwind, void, 67, &args);
                };
                if let Some(cond) = cond {
                    let (store_blk, done_blk) = (self.s.m.new_block(), self.s.m.new_block());
                    self.s.m.terminate(cur, Term::CondBr(cond, store_blk, done_blk));
                    self.cur = store_blk;
                    emit_rov_store(self, new_val);
                    self.s.m.terminate(self.cur, Term::Br(done_blk));
                    self.cur = done_blk;
                } else {
                    emit_rov_store(self, new_val);
                }
                if cache.glc {
                    let old = self.end_guarded(at, &[(old, original)])[0];
                    let (lo, hi) = self.split64(old);
                    self.write(inst, Reg::V(reg), lo, None);
                    self.write(inst, Reg::V(reg + 1), hi, None);
                } else {
                    self.end_guarded(at, &[]);
                }
                return Ok(());
            }
            let (ty, component) = if wide {
                self.s.flags |= flags::INT64 | flags::ATOMIC_TYPED64 | flags::ATOMIC_HEAP64;
                (self.s.m.int(64), 0x107) // one uint64_t in an R32G32_UINT view
            } else {
                (self.s.t.i32, 0x105)
            };
            let original = if wide {
                self.join64(vals[0], vals[1])
            } else {
                vals[0]
            };
            let value = if op == "sub" {
                let zero = self.s.m.int_const(ty, 0);
                self.s.m.bin(self.cur, Bin::Sub, zero, original)
            } else {
                original
            };
            let h = self.handle(site, src, false, (kind | 0x1000, component))?;
            let mut args = vec![h];
            let (intrinsic, opcode) = if compare {
                let comparand = if wide {
                    self.join64(vals[2], vals[3])
                } else {
                    vals[1]
                };
                args.extend(cs);
                args.extend([comparand, value]);
                let intrinsic = if wide {
                    "atomicCompareExchange.i64"
                } else {
                    "atomicCompareExchange.i32"
                };
                (intrinsic, 79)
            } else {
                args.push(self.s.m.i32(code));
                args.extend(cs);
                args.push(value);
                (
                    if wide {
                        "atomicBinOp.i64"
                    } else {
                        "atomicBinOp.i32"
                    },
                    78,
                )
            };
            let at = self.begin_guarded(guard);
            let r = self
                .s
                .op(self.cur, intrinsic, Attr::NoUnwind, ty, opcode, &args);
            if cache.glc {
                let old = self.end_guarded(at, &[(r, original)])[0];
                if wide {
                    let (lo, hi) = self.split64(old);
                    self.write(inst, Reg::V(reg), lo, None);
                    self.write(inst, Reg::V(reg + 1), hi, None);
                } else {
                    self.write(inst, Reg::V(reg), old, None);
                }
            } else {
                self.end_guarded(at, &[]);
            }
            return Ok(());
        }
        if store {
            if name != "image_store" {
                return Err("an image store to a mip level");
            }
            // shortcut: stores are taken to write all four channels of a
            // float format (the console converts by the view's format).
            if *dmask != 0xF {
                return Err("an image store of some channels");
            }
            let iu = self.s.m.undef(self.s.t.i32);
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let Operand::Vector { reg, .. } = *data else {
                return Err("image store data");
            };
            let mut vals = Vec::new();
            for k in 0..4u8 {
                // d16: two f16 channels per VGPR, low half first.
                let r = if *d16 { k / 2 } else { k };
                let v = inst.read(Reg::V(reg + r)).ok_or("image store data")?;
                vals.push(if *d16 {
                    let mut x = self.i32_of(v)?;
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(self.cur, Bin::LShr, x, k16);
                    }
                    let f32t = self.s.t.f32;
                    self.s
                        .op(self.cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x])
                } else {
                    self.float_of(v)?
                });
            }
            let h = self.handle(site, src, false, props)?;
            let at = self.begin_guarded(guard);
            let mut args = vec![h];
            args.extend(cs);
            args.extend(vals);
            args.push(self.s.m.i8(0xF));
            let void = self.s.t.void;
            self.s.op(
                self.cur,
                "textureStore.f32",
                Attr::NoUnwind,
                void,
                67,
                &args,
            );
            self.end_guarded(at, &[]);
            return Ok(());
        }
        let has_cl = name.ends_with("_cl");
        let (op, sample) = match name {
            "image_sample" | "image_sample_o" | "image_sample_cl" => (60, true),
            "image_sample_lz" | "image_sample_l" | "image_sample_lz_o" => (62, true),
            "image_sample_b" | "image_sample_b_cl" => (61, true),
            "image_sample_c_lz" => (65, true),
            "image_sample_d" | "image_sample_d_cl" => (63, true),
            "image_get_lod" if self.s.pixel => (81, true),
            // Implicit derivatives: pixel shaders only.
            "image_sample_c" if self.s.pixel => (64, true),
            "image_sample_c_cl" if self.s.pixel => (64, true),
            "image_gather4" | "image_gather4_lz" | "image_gather4_lz_o" | "image_gather4_l" | "image_gather4_l_o" => (73, true),
            "image_load" | "image_load_mip" => (66, false),
            "image_get_resinfo" => (72, false),
            _ => return Err(name),
        };
        if op == 72 {
            // Width, height, depth or layers, levels (per dmask) of the level
            // in the first address: DXIL's dimensions; a channel DXIL leaves
            // out reads 1 (UAVs: one level).
            let h = self.handle(site, src, false, props)?;
            let mip = if uav {
                self.s.m.undef(self.s.t.i32)
            } else {
                addr_val(self, 0)?
            };
            let c = self.cur;
            let i32t = self.s.t.i32;
            let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
            let d = self
                .s
                .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
            let one = self.s.m.i32(1);
            let present = match kind {
                1 => [true, false, false, true],
                2 | 5 | 6 => [true, true, false, true],
                _ => [true, true, true, true],
            };
            let Operand::Vector { reg, .. } = *data else {
                return Err("an image result");
            };
            for (k, ch) in (0..4u32).filter(|ch| dmask >> ch & 1 != 0).enumerate() {
                let x = if ch == 3 && uav || !present[ch as usize] {
                    one
                } else {
                    self.s.m.extract(c, d, ch)
                };
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        if uav && op != 66 {
            return Err("sampling a UAV");
        }
        let h = self.handle(site, src, false, props)?;
        let fu = self.s.m.undef(self.s.t.f32);
        let iu = self.s.m.undef(self.s.t.i32);
        // A gather at a named level, done as four texel reads (below).
        let mut gathered: Option<[V; 4]> = None;
        // (Sampler number test, handle) of the other immediate samplers a
        // join may pick (`choices` below).
        let mut choices: Vec<(V, V)> = Vec::new();
        let r = if sample {
            let ss = self
                .descriptors
                .get(&(site, true))
                .copied()
                .ok_or("an untracked sampler")?;
            // Comparison: the reference value is the first address after
            // the bias, and the sampler is a comparison sampler.
            let cmp = matches!(op, 64 | 65);
            if cmp && (*a16 || dmask.count_ones() != 1) {
                return Err("a comparison sample of 16-bit coordinates or several channels");
            }
            // Immediate samplers chosen by control flow (a branch picks a
            // point or a linear one, say): DXIL takes no select of handles,
            // so the read is made with each, and the join's sampler number
            // picks the result (below).
            let sh = match ss {
                Source::Phi { block, args } if Self::immediate_leaves(args) => {
                    let mut numbers = Vec::new();
                    let n = self.sampler_number_phi(*block, args, &mut numbers, 8)?;
                    numbers.sort_unstable();
                    numbers.dedup();
                    if cmp {
                        for &k in &numbers {
                            self.s.cmp_samplers |= 1 << k;
                        }
                    }
                    let c = self.cur;
                    let (&first, rest) = numbers.split_first().ok_or("a sampler join without samplers")?;
                    for &k in rest {
                        let hk = self.s.sampler_handle(c, k);
                        let kk = self.s.m.i32(k);
                        let is = self.s.m.cmp(c, pred::EQ, n, kk);
                        choices.push((is, hk));
                    }
                    self.s.sampler_handle(c, first)
                }
                _ => self.handle(site, ss, true, (14 | if cmp { 0x8000 } else { 0 }, 0))?,
            };
            // Texel offsets (`_o`) come first: x in bits 0-5, y in 8-13,
            // signed (programmable offsets: gathers only).
            let offset_word = name.ends_with("_o");
            let first = usize::from(offset_word)
                + usize::from(name.starts_with("image_sample_b"))
                + usize::from(cmp);
            // Coordinates the size divides (not the array slice); explicit
            // gradients (x then y) come before the coordinates.
            let spatial = coords - usize::from(kind == 6 || kind == 7);
            // A cube's gradients are the two of its face coordinates.
            let grads = if kind == 5 { 2 } else { spatial };
            let deriv_regs = if op == 63 {
                if *a16 {
                    grads.div_ceil(2)
                } else {
                    grads
                }
            } else {
                0
            };
            let coord_base = first + 2 * deriv_regs;
            let mut cs = vec![fu; 4];
            // With a16, sampling coordinates, gradients, LOD and clamp are f16.
            let to_float = |l: &mut Self, x: V| {
                if *a16 {
                    let f = l.s.t.f32;
                    l.s.op(l.cur, "legacyF16ToF32", Attr::ReadNone, f, 131, &[x])
                } else {
                    l.as_float(x)
                }
            };
            if kind == 5 && (*unorm || op == 81) {
                return Err("a cube sample with texel coordinates or a level query");
            }
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                let x = addr_packed(self, coord_base, k)?;
                *slot = to_float(self, x);
            }
            if kind == 5 {
                let dir = self.cube_direction(cs[0], cs[1], cs[2]);
                cs[..3].copy_from_slice(&dir);
            }
            if *unorm {
                // Texel coordinates: divided by the size of level 0.
                let c = self.cur;
                let i32t = self.s.t.i32;
                let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                let z = self.s.m.i32(0);
                let d = self
                    .s
                    .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, z]);
                for (k, slot) in cs.iter_mut().enumerate().take(spatial) {
                    let n = self.s.m.extract(c, d, k as u32);
                    let n = self.s.m.cast(c, Cast::UiToFp, n, self.s.t.f32);
                    *slot = self.s.m.bin(c, Bin::SDiv, *slot, n);
                }
            }
            if op == 81 && !choices.is_empty() {
                return Err("a level of detail with a sampler chosen by control flow");
            }
            if op == 81 {
                // The clamped then the unclamped level of detail.
                let Operand::Vector { reg, .. } = *data else {
                    return Err("an image result");
                };
                let f32t = self.s.t.f32;
                let c = self.cur;
                for (k, ch) in (0..2u32).filter(|ch| dmask >> ch & 1 != 0).enumerate() {
                    let clamped = self.s.m.bool(ch == 0);
                    let lod = self.s.op(
                        c,
                        "calculateLOD.f32",
                        Attr::ReadOnly,
                        f32t,
                        81,
                        &[h, sh, cs[0], cs[1], cs[2], clamped],
                    );
                    let x = self.as_int(lod);
                    self.write(inst, Reg::V(reg + k as u8), x, guard);
                }
                return Ok(());
            }
            let offsets = if kind == 5 {
                0
            } else {
                (coords - usize::from(kind == 6 || kind == 7)).min(3)
            };
            let zero = self.s.m.i32(0);
            let mut os = vec![iu; 3];
            for o in os.iter_mut().take(offsets) {
                *o = zero;
            }
            if offset_word && offsets > 0 {
                let w = addr_val(self, 0)?;
                let c = self.cur;
                for (k, at) in [(0, 26), (1, 18)].into_iter().take(offsets.min(2)) {
                    // Bits at..at+5 sign-extended: shift up, then down.
                    let left = self.s.m.i32(at);
                    let t = self.s.m.bin(c, Bin::Shl, w, left);
                    let k26 = self.s.m.i32(26);
                    os[k] = self.s.m.bin(c, Bin::AShr, t, k26);
                }
            }
            let mut args = vec![h, sh];
            args.extend(&cs);
            let clamp = if has_cl {
                let x = addr_packed(self, coord_base, coords)?;
                to_float(self, x)
            } else {
                fu
            };
            match op {
                // Outside a pixel shader there are no screen derivatives:
                // the validator refuses `sample` there ("features
                // incompatible with the shader stage": 730 compute programs,
                // among them the G-buffer resolves and the sky), and what a
                // driver does with it is its own affair.
                // shortcut: such a sample reads level 0 (a bias: that
                // level). The console takes derivatives from neighbouring
                // lanes there; compute them from the lanes if a resolve
                // shows the wrong mip.
                60 if !self.s.pixel => {
                    args.extend(&os);
                    args.push(self.s.m.f32(0));
                    self.s.op(
                        self.cur,
                        "sampleLevel.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        62,
                        &args,
                    )
                }
                60 => {
                    if offset_word {
                        // DXIL wants immediate offsets in -8..=7 here; the
                        // console's are 6-bit and may be computed. A
                        // constant word in range becomes immediates
                        // (upstream PR #634); anything else keeps the
                        // offsets computed above (upstream refuses those).
                        let w = addr_val(self, 0)?;
                        if let Some(w) = self.s.m.int_value(w) {
                            let w = w as u32;
                            let field = |k: usize| {
                                let x = (w >> (8 * k) & 0x3F) as i32;
                                if x >= 32 { x - 64 } else { x }
                            };
                            let n = offsets.min(2);
                            if (0..n).all(|k| (-8..=7).contains(&field(k))) {
                                for (k, o) in os.iter_mut().enumerate().take(n) {
                                    *o = self.s.m.i32(field(k) as u32);
                                }
                            }
                        }
                    }
                    args.extend(&os);
                    args.push(clamp);
                    self.s
                        .op(self.cur, "sample.f32", Attr::ReadOnly, self.s.t.res_ret, 60, &args)
                }
                61 => {
                    // With a16 the bias is an f16 in the low half of its
                    // word (the word is whole, the value is not).
                    let bias = addr_u32(self, usize::from(offset_word))?;
                    let c = self.cur;
                    let bias = if *a16 {
                        let low = self.s.m.i32(0xFFFF);
                        let bias = self.s.m.bin(c, Bin::And, bias, low);
                        to_float(self, bias)
                    } else {
                        self.as_float(bias)
                    };
                    args.extend(&os);
                    if self.s.pixel {
                        args.extend([bias, clamp]);
                        self.s.op(
                            c,
                            "sampleBias.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            61,
                            &args,
                        )
                    } else {
                        // As above: the bias is the level.
                        args.push(bias);
                        self.s.op(
                            c,
                            "sampleLevel.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            62,
                            &args,
                        )
                    }
                }
                63 if kind == 5 => {
                    // A cube with explicit gradients: the host wants them
                    // as changes of the direction, the console gives the
                    // changes of the face coordinates (the face is one
                    // unit wide). The level they select is taken from the
                    // longer of the two, in texels, and sampled directly.
                    // shortcut: no anisotropic filtering for these samples
                    // (the longer axis alone decides the level).
                    let mut len = [fu; 2];
                    for (axis, slot) in len.iter_mut().enumerate() {
                        let mut sum = self.s.m.f32(0);
                        for k in 0..2 {
                            let x = addr_packed(self, first + axis * deriv_regs, k)?;
                            let x = to_float(self, x);
                            let c = self.cur;
                            let sq = self.s.m.bin(c, Bin::Mul, x, x);
                            sum = self.s.m.bin(c, Bin::Add, sum, sq);
                        }
                        *slot = self.unary(24, sum); // square root
                    }
                    let longer = self.binary(35, len[0], len[1]); // maximum
                    let c = self.cur;
                    let i32t = self.s.t.i32;
                    let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                    let z = self.s.m.i32(0);
                    let d = self
                        .s
                        .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, z]);
                    let width = self.s.m.extract(c, d, 0);
                    let width = self.s.m.cast(c, Cast::UiToFp, width, self.s.t.f32);
                    let texels = self.s.m.bin(c, Bin::Mul, longer, width);
                    // DXIL's logarithm is the one to base 2 (it was taken for
                    // the natural one: every level came out 1.44 times too
                    // high; DexNom/VIctory 921d8d3b). A length of 0 gives
                    // minus infinity, which the sampler clamps to level 0.
                    // SampleLevel takes no clamp: the instruction's least
                    // level is applied here.
                    let lod = self.unary(23, texels);
                    let lod = if has_cl { self.binary(35, lod, clamp) } else { lod };
                    args.extend(&os);
                    args.push(lod);
                    self.s.op(
                        self.cur,
                        "sampleLevel.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        62,
                        &args,
                    )
                }
                63 => {
                    let mut g = vec![fu; 6];
                    for k in 0..spatial {
                        let x = addr_packed(self, first, k)?;
                        g[k] = to_float(self, x);
                        let y = addr_packed(self, first + deriv_regs, k)?;
                        g[3 + k] = to_float(self, y);
                    }
                    args.extend(&os);
                    args.extend(g);
                    args.push(clamp);
                    self.s.op(
                        self.cur,
                        "sampleGrad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        63,
                        &args,
                    )
                }
                64 | 65 => {
                    let reference = addr_u32(self, first - 1)?;
                    let reference = self.as_float(reference);
                    args.extend(&os);
                    args.push(reference);
                    let name = if op == 64 {
                        args.push(clamp);
                        "sampleCmp.f32"
                    } else {
                        "sampleCmpLevelZero.f32"
                    };
                    self.s
                        .op(self.cur, name, Attr::ReadOnly, self.s.t.res_ret, op, &args)
                }
                62 => {
                    let lod = if name == "image_sample_l" {
                        let x = addr_packed(self, coord_base, coords)?;
                        to_float(self, x)
                    } else {
                        self.s.m.f32(0)
                    };
                    args.extend(&os);
                    args.push(lod);
                    self.s.op(
                        self.cur,
                        "sampleLevel.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        62,
                        &args,
                    )
                }
                _ => {
                    if name == "image_gather4" {
                        return Err("a gather with implicit level");
                    }
                    if dmask.count_ones() != 1 {
                        return Err("a gather of several channels");
                    }
                    if matches!(name, "image_gather4_l" | "image_gather4_l_o")
                        && coords == 2
                        && spatial == 2
                    {
                        let lod = addr_packed(self, coord_base, coords)?;
                        let lod = to_float(self, lod);
                        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
                        let zero = self.s.m.f32(0);
                        let lod = self.binary(35, lod, zero); // maximum
                        let lod = self.unary(26, lod); // round to nearest
                        let c = self.cur;
                        let mip = self.s.m.cast(c, Cast::FpToUi, lod, i32t);
                        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                        let d = self
                            .s
                            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
                        let half = self.s.m.f32(0.5f32.to_bits());
                        let one = self.s.m.f32(1.0f32.to_bits());
                        let (mut lo, mut hi) = ([iu; 2], [iu; 2]);
                        for k in 0..2 {
                            let n = self.s.m.extract(c, d, k as u32);
                            let n = self.s.m.cast(c, Cast::UiToFp, n, f32t);
                            let last = self.s.m.bin(c, Bin::Sub, n, one);
                            let t = self.s.m.bin(c, Bin::Mul, cs[k], n);
                            let t = self.s.m.bin(c, Bin::Sub, t, half);
                            let first = self.unary(27, t); // round down
                            // The offset word's field (0 without one).
                            let by = self.s.m.cast(c, Cast::SiToFp, os[k], f32t);
                            let first = self.s.m.bin(c, Bin::Add, first, by);
                            let next = self.s.m.bin(c, Bin::Add, first, one);
                            for (slot, x) in [(&mut lo[k], first), (&mut hi[k], next)] {
                                let x = self.binary(35, x, zero); // maximum
                                let x = self.binary(36, x, last); // minimum
                                *slot = self.s.m.cast(c, Cast::FpToSi, x, i32t);
                            }
                        }
                        let ch = dmask.trailing_zeros();
                        let mut g = [fu; 4];
                        let mut read = None;
                        let texels = [(lo[0], hi[1]), (hi[0], hi[1]), (hi[0], lo[1]), (lo[0], lo[1])];
                        for (slot, (x, y)) in g.iter_mut().zip(texels) {
                            let r = self.s.op(
                                c,
                                "textureLoad.f32",
                                Attr::ReadOnly,
                                self.s.t.res_ret,
                                66,
                                &[h, mip, x, y, iu, iu, iu, iu],
                            );
                            *slot = self.s.m.extract(c, r, ch);
                            read = Some(r);
                        }
                        gathered = Some(g);
                        read.expect("four reads")
                    } else {
                        args.extend(&os[..2]);
                        args.push(self.s.m.i32(dmask.trailing_zeros()));
                        self.s.op(
                            self.cur,
                            "textureGather.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            73,
                            &args,
                        )
                    }
                }
            }
        } else {
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let mip = if name == "image_load_mip" {
                addr_val(self, coords)?
            } else if uav {
                // Our declarations have four channels.
                self.s.flags |= flags::TYPED_UAV_LOADS;
                iu
            } else {
                self.s.m.i32(0)
            };
            let mut args = vec![h, mip];
            args.extend(&cs);
            args.extend([iu, iu, iu]);
            self.s.op(
                self.cur,
                "textureLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                66,
                &args,
            )
        };
        let c = self.cur;
        let Operand::Vector { reg, .. } = *data else {
            return Err("an image result");
        };
        // The same read with each other sampler the join may pick: argument
        // 2 of the call is the sampler (after the opcode and the image).
        if !choices.is_empty() && gathered.is_some() {
            return Err("a gather with a sampler chosen by control flow");
        }
        let mut alts = Vec::new();
        for &(is, hk) in &choices {
            let again = self.s.m.call_with_arg(c, r, 2, hk).ok_or("a sampled read that is not one call")?;
            alts.push((is, again));
        }
        let channels: Vec<u32> = if op == 73 {
            vec![0, 1, 2, 3]
        } else {
            (0..4).filter(|ch| dmask >> ch & 1 != 0).collect()
        };
        let mut xs = Vec::new();
        for ch in channels {
            let x = match gathered {
                Some(g) => g[ch as usize],
                None => {
                    let mut x = self.s.m.extract(c, r, ch);
                    for &(is, again) in &alts {
                        let y = self.s.m.extract(c, again, ch);
                        x = self.s.m.select(c, is, y, x);
                    }
                    x
                }
            };
            xs.push(if *d16 {
                // shortcut: d16 results are taken as floats (rounded to
                // nearest even); integer formats would keep 16 bits.
                let i32t = self.s.t.i32;
                self.s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[x])
            } else {
                self.as_int(x)
            });
        }
        if *d16 {
            // Two channels per VGPR, low half first (an odd last high half 0).
            let k16 = self.s.m.i32(16);
            xs = xs
                .chunks(2)
                .map(|p| match *p {
                    [lo, hi] => {
                        let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                        self.s.m.bin(c, Bin::Or, lo, hi)
                    }
                    _ => p[0],
                })
                .collect();
        }
        for (k, x) in xs.into_iter().enumerate() {
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }
}
