//! The lowering's diagnostic counters (`shader.rs` `SLOT_DIAG`): half-wave sync
//! timeouts and loop-budget cuts, written only when they happen.

use super::*;

impl Lower<'_> {
    /// Adds 1 to each of `dwords` of the diagnostic counters (`shader.rs` `SLOT_DIAG`) in `block`.
    pub(super) fn diag_bump(&mut self, block: usize, dwords: &[u32]) {
        let uav = self.s.root_uav_handle(super::super::shader::SLOT_DIAG);
        let (add, one) = (self.s.m.i32(0), self.s.m.i32(1));
        let iu = self.s.m.undef(self.s.t.i32);
        for &d in dwords {
            let byte = self.s.m.i32(d * 4);
            self.s.op(block, "atomicBinOp.i32", Attr::NoUnwind, self.s.t.i32, 78, &[uav, add, byte, iu, iu, one]);
        }
    }

    /// A compute loop's back edge from the current block, now terminated, that the loop
    /// budget cuts where `cut` holds: each thread cut counts it in `SLOT_DIAG`
    /// (in total and for the loop) and leaves there the loop's header offset and the
    /// scalar values its header joins this iteration (the last thread's), which say why the
    /// loop did not end.
    /// Only the cut path pays; the block's terminator moves to a block after the count.
    pub(super) fn count_cut(&mut self, cut: V, header: usize) {
        use super::super::shader::{DIAG_LOOP_BUDGET, DIAG_LOOP_DWORDS, DIAG_LOOP_VALUES, DIAG_LOOPS, DIAG_LOOPS_MAX};
        if self.s.threads.is_none() || self.s.mesh.is_some() || self.s.pixel {
            return;
        }
        let latch = self.cur;
        let (bump, after) = (self.s.m.new_block(), self.s.m.new_block());
        self.s.m.move_term(latch, after);
        // per thread, not per wave: a wave operation here would change what the loop
        // lowers to (waterfalls are kept free of them)
        self.s.m.terminate(latch, Term::CondBr(cut, bump, after));
        let mut loops: Vec<usize> = self.counters.keys().copied().collect();
        loops.sort_unstable();
        let k = loops.iter().position(|&h| h == header).unwrap_or(0) as u32;
        let base = DIAG_LOOPS + k.min(DIAG_LOOPS_MAX - 1) * DIAG_LOOP_DWORDS;
        self.diag_bump(bump, &[DIAG_LOOP_BUDGET, base]);
        let p = self.x.p;
        let offset = p.blocks[header].offset as u32;
        let mut words = vec![self.s.m.i32(offset)];
        let mut regs = Vec::new();
        for phi in &p.blocks[header].phis {
            if words.len() > DIAG_LOOP_VALUES as usize {
                break;
            }
            if let (Reg::S(r), Some(Val::I32(v))) = (phi.reg, self.vals.get(&phi.value)) {
                words.push(*v);
                regs.push(format!("s{r}"));
            }
        }
        let uav = self.s.root_uav_handle(super::super::shader::SLOT_DIAG);
        let (iu, mask, align) = (self.s.m.undef(self.s.t.i32), self.s.m.i8(1), self.s.m.i32(4));
        let void = self.s.t.void;
        for (n, w) in words.into_iter().enumerate() {
            let byte = self.s.m.i32((base + 1 + n as u32) * 4);
            self.s.op(bump, "rawBufferStore.i32", Attr::NoUnwind, void, 140, &[uav, byte, iu, w, iu, iu, iu, mask, align]);
        }
        if std::env::var_os("VICTORY_WAVE_WHY").is_some() {
            eprintln!("WAVE64-LOOP {k}: header IR block {header} at {offset:#x}, values [{}]", regs.join(", "));
        }
        self.s.m.terminate(bump, Term::Br(after));
        // whatever named the latch as a predecessor now names `after`
        for t in &mut self.tail {
            if *t == Some(latch) {
                *t = Some(after);
            }
        }
        for (_, incoming) in self.counters.values_mut() {
            for (_, from) in incoming.iter_mut() {
                if *from == latch {
                    *from = after;
                }
            }
        }
        for e in self.edges.values_mut() {
            if *e == latch {
                *e = after;
            }
        }
        self.cur = after;
    }

}
