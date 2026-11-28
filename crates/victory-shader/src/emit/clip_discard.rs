//! Clip distances done by the pixel shader, for a mesh pipeline still over Metal's 32 KiB
//! once its outputs are ordered and trimmed (`mesh_order`).
//!
//! Metal counts a mesh shader's clip distances twice per vertex: as floats before the
//! position (the hardware's, from `exp pos1`) and as the `SV_ClipDistance` element among
//! the outputs (the pixel shader's copy, from a parameter export of the same values:
//! `1c451759513efb4c` exports v21..v24 to both). For such a pipeline the mesh shader can
//! declare that element an arbitrary output, [`CLIP_VARYING`], which drops the
//! hardware's copy ([`set_mesh_drop_clip`]), and the pixel shader read it under that
//! name, interpolated with perspective, and discard where any component is below zero
//! ([`set_pixel_clip_inputs`]): that is where the hardware would have clipped (clip
//! distances are linear in clip space, as perspective-correct interpolation is).
//!
//! [`clip_alias`] checks the shape; the translation records it ([`last_clip_alias`]).

use super::*;

thread_local! {
    static CLIP_INPUTS: std::cell::RefCell<Option<Vec<(String, u32, u8)>>> = const { std::cell::RefCell::new(None) };
    static DROP_CLIP: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static LAST_ALIAS: std::cell::RefCell<Option<(String, u32, u8)>> = const { std::cell::RefCell::new(None) };
}

/// For pixel shaders translated on this thread: discard where any of the first `n`
/// components of input (name, index) is below zero. Returns the old setting.
pub fn set_pixel_clip_inputs(inputs: Option<Vec<(String, u32, u8)>>) -> Option<Vec<(String, u32, u8)>> {
    CLIP_INPUTS.with(|c| c.replace(inputs))
}

/// For mesh shaders translated on this thread with `set_mesh_outputs_keep`: leave the
/// clip distances out. Returns the old setting.
pub fn set_mesh_drop_clip(on: bool) -> bool {
    DROP_CLIP.with(|d| d.replace(on))
}

/// The arbitrary output the clip distances travel in when the pixel shader clips.
pub const CLIP_VARYING: &str = "VICTORYCLIP";

/// For a mesh shader asked to ([`set_mesh_drop_clip`]): its clip-distance elements as
/// arbitrary [`CLIP_VARYING`] outputs (the exports still find them by row; `pos1` then
/// finds no element and stores nothing).
pub(super) fn drop_clip(outputs: &mut [SigElement]) {
    if !DROP_CLIP.with(|d| d.get()) {
        return;
    }
    for e in outputs.iter_mut().filter(|e| e.kind == 6) {
        e.kind = 0;
        e.name = CLIP_VARYING.to_owned();
        e.interp = 2;
    }
}

/// The output (name, index, components) that carries the clip distances of the mesh
/// shader last translated on this thread with `set_mesh_outputs_keep`, if it has one.
pub fn last_clip_alias() -> Option<(String, u32, u8)> {
    LAST_ALIAS.with(|a| a.borrow().clone())
}

pub(super) fn note_clip_alias(alias: Option<(String, u32, u8)>) {
    LAST_ALIAS.with(|a| *a.borrow_mut() = alias);
}

/// The pixel input ([`CLIP_VARYING`], index, clip distances) a pixel shader reads the clip
/// distances in, if the program's `SV_ClipDistance` element gets the very values of its
/// clip distances (the same SSA values, `exp pos1` and the parameter export). None
/// with cull distances or a misc vector (they share `pos1`/`pos2`), or more than four.
/// `row` gives the output row of a parameter export (usage, row), as the lowering does.
pub(super) fn clip_alias(
    exports: &[crate::io::Export],
    outputs: &[SigElement],
    row: &dyn Fn(crate::native::Usage, u8) -> Option<u8>,
) -> Option<(String, u32, u8)> {
    if outputs.iter().any(|o| matches!(o.kind, 4 | 5 | 7)) {
        return None;
    }
    let n: u8 = outputs.iter().filter(|o| o.kind == 6).map(|o| o.cols).sum();
    if n == 0 || n > 4 {
        return None;
    }
    let pos1 = exports.iter().filter(|e| e.target == crate::io::Target::Position(1)).collect::<Vec<_>>();
    let [pos1] = pos1[..] else { return None };
    let clip = &pos1.values[..usize::from(n)];
    if clip.iter().any(Option::is_none) {
        return None;
    }
    exports.iter().find_map(|e| {
        let crate::io::Target::Param { usage, row: r } = e.target else { return None };
        if e.values[..usize::from(n)] != *clip {
            return None;
        }
        let r = row(usage, r)?;
        let o = outputs.iter().find(|o| o.kind == 6 && o.row == Some(u32::from(r)) && o.col == 0 && o.cols == n)?;
        Some((CLIP_VARYING.to_owned(), o.index, n))
    })
}

impl Lower<'_> {
    /// For a mesh shader translated with `set_mesh_outputs_keep`: records which output
    /// carries its clip distances ([`last_clip_alias`]).
    pub(super) fn note_clip_alias(&self) {
        if self.s.mesh.is_none() || !super::mesh_outputs_keep_active() {
            return;
        }
        let row = |usage: crate::native::Usage, row: u8| match usage {
            crate::native::Usage::Other(0) => self.unnamed_row(),
            _ => Some(row),
        };
        note_clip_alias(clip_alias(&self.x.io.exports, &self.s.outputs, &row));
    }

    /// In a pixel shader asked to ([`set_pixel_clip_inputs`]): discards where any listed
    /// input component is below zero, before anything else runs.
    pub(super) fn clip_discard(&mut self) -> Result<(), Why> {
        let Some(list) = CLIP_INPUTS.with(|c| c.borrow().clone()) else { return Ok(()) };
        if !self.s.pixel {
            return Ok(());
        }
        // The pixel shader's `SV_ClipDistance` input, read under the varying's name.
        for e in self.s.inputs.iter_mut().filter(|e| e.kind == 6) {
            e.kind = 0;
            e.name = CLIP_VARYING.to_owned();
            e.interp = 2;
        }
        let zero = self.s.m.f32(0);
        let mut kill = None;
        for (name, index, n) in list {
            let id = self
                .s
                .inputs
                .iter()
                .position(|e| e.kind == 0 && e.name.eq_ignore_ascii_case(&name) && e.index == index)
                .ok_or("a pixel shader without an SV_ClipDistance input")?;
            // Linear (2), centroid (3) or sample (5): perspective-correct, as clipping is.
            if !matches!(self.s.inputs[id].interp, 2 | 3 | 5) || self.s.inputs[id].comp != 9 {
                return Err("clip distances in an input not interpolated with perspective");
            }
            for k in 0..n.min(self.s.inputs[id].cols) {
                let Val::I32(bits) = self.load_any_input(id, k) else {
                    return Err("clip distances in an input that does not load");
                };
                let c = self.cur;
                let f = self.as_float(bits);
                // FCMP_OLT: below zero (a NaN distance is not clipped, as on the console).
                let below = self.s.m.cmp(c, 4, f, zero);
                kill = Some(match kill {
                    Some(k) => self.s.m.bin(c, Bin::Or, k, below),
                    None => below,
                });
            }
        }
        if let Some(k) = kill {
            let c = self.cur;
            self.s.op(c, "discard", Attr::NoUnwind, self.s.t.void, 82, &[k]);
        }
        Ok(())
    }
}
