//! A group's LDS as the program's own first store leaves it (upstream f299fba,
//! `VICTORY_LDS_START`), for programs whose thread 0 sets the LDS up behind an early-out.

use super::*;

/// The start of a console program's LDS as its own first store leaves it: (dword, value) of
/// the plain store of constants at a constant address (`ds_write_b32`, `ds_write2_b32` of what
/// `v_mov_b32` moved there) that stands before every other LDS access of the program. Empty
/// without such a store.
///
/// The console does not clear a group's LDS: a group finds what an earlier wave left. A
/// program that sets its LDS up in one thread (`if (thread == 0) { nearest = ~0; farthest = 0; }`
/// behind its early-outs: the water's tiled light, `CS_TransmissionVolumetrics_Tiled`) does not
/// set it up in a group whose thread 0 is not live (a tile whose first pixel is not water), and
/// that group works with what was left: on the console an earlier tile's values, on a host
/// whatever its memory held, the translation's own exchange words among it. Stored at every
/// group's start ([`Lower::lds_start`]), the program's own values are a state the console may
/// be in, and the one the program means: the store changes nothing where thread 0 is live.
pub(in crate::emit) fn lds_start(p: &Program) -> Vec<(u32, u32)> {
    let lds = |i: &Inst| {
        matches!(i.fields, Fields::Ds { gds: false, .. })
            && !(i.name().starts_with("ds_swizzle") || i.name().contains("permute"))
    };
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut index = vec![usize::MAX; p.blocks.len()];
    for (n, &b) in order.iter().enumerate() {
        index[b] = n;
    }
    let with: Vec<usize> = order.iter().copied().filter(|&b| p.blocks[b].insts.iter().any(lds)).collect();
    let Some(&first) = with.first() else {
        return Vec::new();
    };
    // before every other access: its block is the first with one and dominates the others
    let idom = crate::ir::idoms(&p.blocks, &order, &index);
    let after_first = |mut b: usize| loop {
        if b == first {
            break true;
        }
        if idom[b] == b || idom[b] == usize::MAX {
            break false;
        }
        b = idom[b];
    };
    if !with.iter().all(|&b| after_first(b)) {
        return Vec::new();
    }
    let Some(inst) = p.blocks[first].insts.iter().find(|i| lds(i)) else {
        return Vec::new();
    };
    let Fields::Ds {
        addr: Some(Operand::Vector { reg: addr, .. }),
        data,
        offset0,
        offset1,
        ..
    } = &inst.fields
    else {
        return Vec::new();
    };
    // (byte offsets as [`Lower::ds`] takes them)
    let bytes: Vec<u32> = match inst.name() {
        "ds_write_b32" => vec![u32::from(*offset0) | u32::from(*offset1) << 8],
        "ds_write2_b32" => vec![4 * u32::from(*offset0), 4 * u32::from(*offset1)],
        _ => return Vec::new(),
    };
    // what a `v_mov_b32` of an inline constant or a literal put in the register
    let constant = |reg: u8| -> Option<u32> {
        let v = inst.read(Reg::V(reg))?;
        let def = p.blocks.iter().flat_map(|b| &b.insts).find(|i| i.writes.iter().any(|w| w.1 == v))?;
        let Fields::Valu(valu) = &def.fields else {
            return None;
        };
        if def.name() != "v_mov_b32" || valu.dpp.is_some() || valu.sdwa.is_some() {
            return None;
        }
        match *valu.src.first()? {
            Operand::Const(c) => crate::resource::inline(c),
            Operand::Literal(x) => Some(x),
            _ => None,
        }
    };
    let Some(base) = constant(*addr) else {
        return Vec::new();
    };
    if bytes.len() != data.len() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (byte, op) in bytes.iter().zip(data) {
        let Operand::Vector { reg, .. } = *op else {
            return Vec::new();
        };
        let (Some(value), at) = (constant(reg), base.wrapping_add(*byte)) else {
            return Vec::new();
        };
        if at % 4 != 0 || at / 4 >= 8192 {
            return Vec::new();
        }
        out.push((at / 4, value));
    }
    out
}

/// `VICTORY_LDS_START=0`: a group's LDS starts as the host's memory is, as before
/// ([`lds_start`]).
fn lds_start_on() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_LDS_START.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LDS_START").is_none_or(|v| v != "0"))
}

#[cfg(test)]
thread_local! {
    /// A test's choice for [`lds_start_on`], for its thread only.
    pub(in crate::emit) static TEST_LDS_START: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

impl Lower<'_> {
    /// A console wave run as two host waves, one console wave a group: every thread stores
    /// the program's own start of its LDS ([`lds_start`]) at the top of the entry block,
    /// before [`Self::init_half_sync`]'s group barrier, so every store lands before any thread
    /// runs the program's first LDS access. Only dwords of the program's own LDS
    /// (`cs.lds_bytes()`) are written: `half_sync_slot`, `exchange` and `halves` lie past them
    /// (or in `SLOT_WAVE_SCRATCH`). Not for local waves (no barrier orders the halves there).
    pub(super) fn lds_start(&mut self) {
        if !self.split_lds
            || self.half_sync_slot.is_none()
            || LOCAL_WAVES.get()
            || super::super::group_sync() == 0
            || std::mem::replace(&mut self.lds_started, true)
            || !lds_start_on()
        {
            return;
        }
        let StageIo::Cs(cs) = &self.x.native.io else {
            return;
        };
        if cs.threads.iter().product::<u32>() != 64 {
            return;
        }
        let own = cs.lds_bytes() / 4;
        let e = self.s.entry;
        let lds = self.s.lds();
        for (dword, value) in lds_start(self.x.p) {
            if dword >= own {
                continue;
            }
            let (k, x) = (self.s.m.i32(dword), self.s.m.i32(value));
            let ptr = self.s.m.element(e, lds, k);
            self.s.m.store(e, ptr, x);
        }
    }
}
