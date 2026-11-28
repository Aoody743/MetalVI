//! Dot products of packed signed bytes (GFX10.3): `v_dot4c_i32_i8` (VOP2,
//! `D = D + Σ S0.i8[k] × S1.i8[k]` over the four bytes) and `v_dot4_i32_i8`
//! (VOP3P, the same with `S2` for `D`), 32-bit wrapping. The
//! Series S upscaler's network layers ([`crate::fspx`], FSR4) are made of
//! them (about 18,000 in its 13 largest programs), the weights in SGPRs and
//! the activations in VGPRs. Lowered to plain 32-bit integer operations:
//! each byte sign-extended by a shift pair, multiplied, summed.

use super::*;

impl Lower<'_> {
    /// `inst` if it is a byte dot product; `None` for anything else.
    pub(super) fn dot8(&mut self, inst: &Inst, name: &str, guard: Option<V>) -> Option<Result<(), Why>> {
        matches!(name, "v_dot4c_i32_i8" | "v_dot4_i32_i8").then(|| self.dot4_i32_i8(inst, name == "v_dot4c_i32_i8", guard))
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
