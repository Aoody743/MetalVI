//! VALU operations of the Series S upscaler's code ([`crate::fspx`], made by the AMD
//! compiler) that the title's own shaders never use.
//!
//! Dot products of packed signed bytes (GFX10.3): `v_dot4c_i32_i8` (VOP2,
//! `D = D + Σ S0.i8[k] × S1.i8[k]` over the four bytes) and `v_dot4_i32_i8`
//! (VOP3P, the same with `S2` for `D`), 32-bit wrapping. The
//! Series S upscaler's network layers ([`crate::fspx`], FSR4) are made of
//! them (about 18,000 in its 13 largest programs), the weights in SGPRs and
//! the activations in VGPRs. Lowered to plain 32-bit integer operations:
//! each byte sign-extended by a shift pair, multiplied, summed.

use super::*;

impl Lower<'_> {
    /// `inst` lowered if it is one of these operations; `None` for anything else.
    pub(super) fn fspx_op(&mut self, inst: &Inst, name: &str, guard: Option<V>) -> Option<Result<(), Why>> {
        match name {
            "v_dot4c_i32_i8" | "v_dot4_i32_i8" => Some(self.dot4_i32_i8(inst, name == "v_dot4c_i32_i8", guard)),
            "v_pk_fmac_f16" => Some(self.pk_fmac_f16(inst, guard)),
            _ => None,
        }
    }

    /// `v_pk_fmac_f16` (VOP2): each half of the destination plus the product
    /// of the sources' same halves, `D.h = S0.h × S1.h + D.h`; as
    /// `v_pk_fma_f16` does it here, a fused multiply-add in f32. Registers
    /// only (the upscaler's 54 never take a constant).
    fn pk_fmac_f16(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let registers = v.src.iter().take(2).all(|o| matches!(o, Operand::Vector { .. } | Operand::Scalar { .. }));
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("an accumulator");
        };
        if !registers || v.src.len() < 2 || v.dpp.is_some() || v.sdwa.is_some() {
            return Err("a packed fmac operand");
        }
        let (a, b) = (self.src(inst, 0)?, self.src(inst, 1)?);
        let d = self.i32_of(inst.read(Reg::V(reg)).ok_or("an accumulator")?)?;
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let c = self.cur;
        let (k16, low) = (self.s.m.i32(16), self.s.m.i32(0xFFFF));
        let mut halves = Vec::new();
        for high in [false, true] {
            let half = |l: &mut Self, x: V| {
                let x = if high { l.s.m.bin(c, Bin::LShr, x, k16) } else { x };
                let x = l.s.m.bin(c, Bin::And, x, low);
                l.s.op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x])
            };
            let (x, y, z) = (half(self, a), half(self, b), half(self, d));
            let f = self.fmad(x, y, z);
            let h = self.s.op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
            halves.push(self.s.m.bin(c, Bin::And, h, low));
        }
        let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
        let r = self.s.m.bin(c, Bin::Or, halves[0], hi);
        self.write(inst, Reg::V(reg), r, guard);
        Ok(())
    }

    /// `c`: the accumulator is the destination (VOP2), else source 2.
    fn dot4_i32_i8(&mut self, inst: &Inst, c_form: bool, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        // shortcut: the clamp bit (saturate the sum) is refused, not done;
        // the upscaler's code never sets it.
        if v.clamp {
            return Err("a clamped dot product");
        }
        let (a, b) = (self.src(inst, 0)?, self.src(inst, 1)?);
        let acc = if c_form {
            let Some(Operand::Vector { reg, .. }) = v.dst else {
                return Err("an accumulator");
            };
            self.i32_of(inst.read(Reg::V(reg)).ok_or("an accumulator")?)?
        } else {
            self.src(inst, 2)?
        };
        let c = self.cur;
        let k24 = self.s.m.i32(24);
        let mut sum = acc;
        for k in 0..4 {
            let up = self.s.m.i32(24 - 8 * k);
            let byte = |l: &mut Self, x: V| {
                let x = l.s.m.bin(c, Bin::Shl, x, up);
                l.s.m.bin(c, Bin::AShr, x, k24)
            };
            let (x, y) = (byte(self, a), byte(self, b));
            let p = self.s.m.bin(c, Bin::Mul, x, y);
            sum = self.s.m.bin(c, Bin::Add, sum, p);
        }
        self.write_i(inst, sum, guard)
    }
}

#[cfg(test)]
mod tests {
    /// The upscaler's 29 programs (read from the game dump when present) on
    /// a host of 32-lane waves: all but four translate, and those four stop
    /// where they are known to.
    #[test]
    fn upscaler_programs_translate() {
        let Some(all) = crate::fspx::tests::upscaler_containers() else {
            eprintln!("SKIP: no upscaler DLL");
            return;
        };
        let results = std::thread::Builder::new()
            .stack_size(256 << 20)
            .spawn(move || {
                let host = super::super::TEST_HOST_WAVE.replace(Some(32));
                let r: Vec<Result<(), String>> = all.iter().map(|c| crate::emit::translate(c).map(|_| ()).map_err(|e| e.to_string())).collect();
                super::super::TEST_HOST_WAVE.set(host);
                r
            })
            .unwrap()
            .join()
            .unwrap();
        let known = [
            (6, "a lane mask exchanged where not every lane runs"),
            (8, "a lane mask exchanged where not every lane runs"),
            (9, "a lane mask exchanged where not every lane runs"),
            (28, "a root descriptor"),
        ];
        for (k, r) in results.iter().enumerate() {
            match known.iter().find(|(i, _)| *i == k) {
                Some((_, why)) => assert!(r.as_ref().is_err_and(|e| e.contains(why)), "#{k}: {r:?}"),
                None => assert!(r.is_ok(), "#{k}: {r:?}"),
            }
        }
    }
}
