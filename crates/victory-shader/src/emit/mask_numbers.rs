//! A scalar operation on numbers whose result the structure also counts as a lane mask.
//!
//! The structure counts a value as a mask from how it is used anywhere, and a register the
//! compiler reuses for a number and for a saved EXEC can carry that over: in
//! `dbdbf36298727135` (`CS_TraceShadowRays_FullAdaptive`) `s35` holds a saved EXEC in one
//! place (`s_mov_b32 s35, exec_lo`, 0x8c90) and elsewhere an instance key built from two
//! numbers (`s_or_b32 s35, s32, s31`, 0x9c84; `s31 = s27 & 0x1fffff`,
//! `s32 = s29 & 0xffe00000`) that the code compares with a VGPR (`v_cmp_ne_u32 vcc_lo, s35,
//! v62`, 0xae74). The lowering left such a result without a number and then had to read it
//! as a ballot of lanes, which it refused ("a lane mask holding lanes elsewhere read as
//! data"). An operation that reads no mask, no EXEC, no VCC and no SCC computes a number:
//! it gets that number (where the result is used as a mask, its lanes come from the
//! structure as before). `VICTORY_MASK_NUMBERS=0`: refused as before.

use super::*;

impl Lower<'_> {
    /// Whether scalar operation `inst` computes a number: every value it reads is one
    /// ([`Self::number`]).
    pub(super) fn of_numbers(&self, inst: &Inst) -> bool {
        if std::env::var_os("VICTORY_MASK_NUMBERS").is_some_and(|v| v == "0") {
            return false;
        }
        let mut seen = Vec::new();
        !inst.reads.is_empty() && inst.reads.iter().all(|&(r, v)| self.number(r, v, &mut seen, 0))
    }

    /// Whether `v` (in `r`) is a number whatever the structure calls it: a scalar load, a lane
    /// read, an input other than EXEC, or a scalar operation or join of numbers (a lane mask
    /// joined with them left out). Not EXEC, VCC
    /// or SCC, a compare, `s_cselect` (of EXEC and 0, say) or a save of EXEC.
    fn number(&self, r: Reg, v: Value, seen: &mut Vec<Value>, depth: u32) -> bool {
        if !matches!(r, Reg::S(c) if !matches!(c, 106 | 107 | 126 | 127)) || depth > 64 {
            return false;
        }
        if seen.contains(&v) {
            return true; // (a loop of joins: as the rest of it says)
        }
        seen.push(v);
        let p = self.x.p;
        if let Some(&(b, i)) = self.phi_at.get(&v) {
            // A join of a number and a lane mask (the register reused, the mask dead on the
            // number's paths, as [`Lower::dead`] has it) is the number.
            let phi = &p.blocks[b].phis[i];
            let mut any = false;
            for &a in &phi.args {
                if self.x.st.masks.contains_key(&a) || self.x.st.highs.contains_key(&a) {
                    continue;
                }
                if !self.number(phi.reg, a, seen, depth + 1) {
                    return false;
                }
                any = true;
            }
            return any;
        }
        let def = p.blocks.iter().flat_map(|b| &b.insts).find(|i| i.writes.iter().any(|w| w.1 == v));
        let Some(inst) = def else {
            return p.inputs.iter().any(|i| i.1 == v);
        };
        let name = inst.name();
        if name.starts_with("s_load") || name.starts_with("s_buffer_load") || matches!(name, "v_readfirstlane_b32" | "v_readlane_b32") {
            return true;
        }
        matches!(inst.fields, Fields::Sop { .. })
            && !name.starts_with("s_cselect")
            && !name.contains("saveexec")
            && !name.contains("wrexec")
            && inst.write(Reg::S(126)).is_none()
            && inst.reads.iter().all(|&(r, a)| self.number(r, a, seen, depth + 1))
    }
}
