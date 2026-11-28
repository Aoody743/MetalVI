//! The title address of a buffer view picked from the view heap by an index the code
//! computes, for console code that reads the address out of the descriptor
//! (`CS_TraceShadowRays_*`: 96ccb14ef981fd85, 002e93a6cc2aa355, 0e64f14fae060d04,
//! 07c93bfb77fecebb, dbdbf36298727135).
//!
//! Those programs load a buffer descriptor from the heap (`s_load_dwordx4 s[16:19], s[6:7],
//! s13`, the slot the ray's instance names), keep its base address (`s_and_b32 s17, s17,
//! 0xffff`; `s_lshr_b64 .., 8`, a first-lane read, `s_lshl_b64 .., 8`), read the structure's
//! header at +48 (`s_load_dword`) and give the address to `image_bvh_intersect_ray`. The
//! host keeps those addresses beside the strides ([`crate::emit::shader::STRIDES_VIEW_ADDRESSES`]),
//! as it gives a table's view address ([`crate::emit::shader::USER_VIEW_ADDRESS`]); the rest is the
//! title memory by address the ray-tracing programs already read.
//! `VICTORY_HEAP_VIEW_ADDRESS=0`: refused as before.

use super::*;

impl Lower<'_> {
    /// The title address (low, high dword) of the buffer view the heap descriptor load `site`
    /// names, from the host's table: `None` when the load is not from the view heap.
    pub(super) fn heap_view_address(&mut self, site: Site) -> Option<(V, V)> {
        if std::env::var_os("VICTORY_HEAP_VIEW_ADDRESS").is_some_and(|v| v == "0") {
            return None;
        }
        let a = self.scalar.get(&site).copied()?;
        if self.x.native.layout.get(a.param)?.root != RootType::ResourceHeap {
            return None;
        }
        let slot = self.heap_index(site, 0, false).ok()?;
        let c = self.cur;
        let i32t = self.s.t.i32;
        // inside the table whatever the index (as `slot_word`)
        let last = self.s.m.i32(crate::emit::shader::SLOT_STRIDES_COUNT - 1);
        let slot = self.s.op(c, "binary.i32", Attr::ReadNone, i32t, 40, &[slot, last]);
        self.s.m.tag(slot, Tag::Clamp);
        let three = self.s.m.i32(3);
        let byte = self.s.m.bin(c, Bin::Shl, slot, three);
        let base = self.s.m.i32(crate::emit::shader::STRIDES_VIEW_ADDRESSES * 4);
        let byte = self.s.m.bin(c, Bin::Add, byte, base);
        let h = self.s.root_srv_handle(crate::emit::shader::SLOT_STRIDES);
        let iu = self.s.m.undef(i32t);
        let (mask, align) = (self.s.m.i8(3), self.s.m.i32(4));
        let r = self.s.op(c, "rawBufferLoad.f32", Attr::ReadOnly, self.s.t.res_ret, 139, &[h, byte, iu, mask, align]);
        let lo = self.s.m.extract(c, r, 0);
        let hi = self.s.m.extract(c, r, 1);
        Some((self.as_int(lo), self.as_int(hi)))
    }
}
