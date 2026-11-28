//! The order of a mesh shader's vertex outputs, for Metal's 32 KiB.
//!
//! Metal lays one vertex's outputs out as the clip and cull distances (floats), the
//! position, then every output but the position in signature order (the distance
//! elements again among them), each at the next multiple of its own size (4 bytes a
//! component), and counts that times the declared vertex
//! maximum against 32 KiB (`victory-gpu` `pipeline::host::metal_mesh_bytes`, measured on
//! Apple's compiler). Signature order is the console's, so a `float3` after a `float4`
//! after a `float` wastes bytes: the title's 17 refused mesh shaders take 34,816 to
//! 38,400 bytes in that order, and 16 of them fit once whole size classes are placed
//! in the best order. System values keep their place in front; only the arbitrary
//! (TEXCOORD, COLOR) elements move.
//!
//! The list is reordered before the lowering takes element ids from it; exports find
//! their element by the console row and column, so the rows stay the console's until
//! the shader is finished, when [`renumber`] numbers them in the new order (the pixel
//! shader links by semantic). `VICTORY_MESH_OUTPUT_ORDER=0` keeps the signature order.

use crate::emit::container::SigElement;

/// DXIL semantic kinds of the position and the distances (see the module comment).
const POSITION: u8 = 3;
const CLIP: u8 = 6;
const CULL: u8 = 7;

/// The end of laying out blocks of `sizes` bytes from `start`, each at the next
/// multiple of its own size.
fn layout(start: u32, sizes: impl IntoIterator<Item = u32>) -> u32 {
    sizes.into_iter().filter(|&s| s > 0).fold(start, |end, s| end.next_multiple_of(s) + s)
}

/// The bytes Metal lays one vertex of `els` out in (see the module comment).
pub(super) fn vertex_bytes(els: &[SigElement]) -> u32 {
    let distances: u32 = els.iter().filter(|e| matches!(e.kind, CLIP | CULL)).map(|e| u32::from(e.cols)).sum();
    let head = layout(0, std::iter::repeat_n(4, distances as usize).chain([16]));
    layout(head, els.iter().filter(|e| e.kind != POSITION).map(|e| 4 * u32::from(e.cols)))
}

/// Puts the arbitrary outputs (after the system values, which keep their order) in the
/// order of size classes that lays them out smallest (each class keeps its signature order), if that is smaller than
/// the signature order. Returns the bytes a vertex takes before and after.
pub(super) fn tighten(els: &mut Vec<SigElement>) -> (u32, u32) {
    let before = vertex_bytes(els);
    if std::env::var_os("VICTORY_MESH_OUTPUT_ORDER").is_some_and(|v| v == "0") {
        return (before, before);
    }
    let original = els.clone();
    let (first, rest): (Vec<SigElement>, Vec<SigElement>) = std::mem::take(els).into_iter().partition(|e| e.kind != 0);
    let mut classes: Vec<u8> = rest.iter().map(|e| e.cols).collect();
    classes.sort_unstable_by(|a, b| b.cmp(a));
    classes.dedup();
    let ordered = |order: &[u8]| -> Vec<SigElement> {
        order.iter().flat_map(|&c| rest.iter().filter(move |e| e.cols == c).cloned()).collect()
    };
    let mut best: Option<(u32, Vec<u8>)> = None;
    permute(&mut classes.clone(), 0, &mut |order| {
        let mut trial = first.clone();
        trial.extend(ordered(order));
        let bytes = vertex_bytes(&trial);
        if best.as_ref().is_none_or(|b| bytes < b.0) {
            best = Some((bytes, order.to_vec()));
        }
    });
    match best {
        Some((bytes, order)) if bytes < before => {
            let mut out = first;
            out.extend(ordered(&order));
            *els = out;
            (before, bytes)
        }
        _ => {
            *els = original;
            (before, before)
        }
    }
}

thread_local! {
    static READS: std::cell::RefCell<Option<Vec<(String, u32, u8)>>> = const { std::cell::RefCell::new(None) };
}

/// For mesh programs translated on this thread with [`super::set_mesh_outputs_keep`]:
/// how many leading components of each user input (name, index) the pixel shader behind
/// reads ([`input_reads`]), so an output still over Metal's limit once ordered can leave
/// the rest out. Returns the old setting.
pub fn set_mesh_output_reads(reads: Option<Vec<(String, u32, u8)>>) -> Option<Vec<(String, u32, u8)>> {
    READS.with(|r| r.replace(reads))
}

/// For each user input a host pixel shader declares (its `ISG1`): (name, index, the
/// components up to the last one it reads, counted from the element's first).
pub fn input_reads(dxil: &[u8]) -> Option<Vec<(String, u32, u8)>> {
    use crate::dxbc::u32_at;
    let c = crate::dxbc::Container::parse(dxil).ok()?;
    let part = c.part(crate::dxbc::FourCC::ISG1)?.data;
    let (count, base) = (u32_at(part, 0).ok()? as usize, u32_at(part, 4).ok()? as usize);
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = base + 32 * i;
        if u32_at(part, at + 12).ok()? != 0 {
            continue; // a system value
        }
        let name_at = u32_at(part, at + 4).ok()? as usize;
        let name = part.get(name_at..)?.split(|&b| b == 0).next()?;
        let (mask, read) = (*part.get(at + 24)?, *part.get(at + 25)?);
        let first = mask.trailing_zeros().min(3);
        let n = 8 - ((read & mask) >> first).leading_zeros() as u8;
        out.push((String::from_utf8_lossy(name).into_owned(), u32_at(part, at + 8).ok()?, n));
    }
    Some(out)
}

/// The most bytes of outputs Metal takes for a mesh shader.
const METAL_LIMIT: u32 = 32 * 1024;

/// [`tighten`], and if `max_vertices` of the result are still over Metal's limit, the
/// components the pixel shader does not read past the last it does left out of each
/// arbitrary element ([`set_mesh_output_reads`]) and [`tighten`] again. Whether the
/// outputs changed order (their rows are then numbered again, [`renumber`]).
pub(super) fn fit(els: &mut Vec<SigElement>, max_vertices: u32) -> bool {
    let (was, mut now) = tighten(els);
    let reads = READS.with(|r| r.borrow().clone());
    // `VICTORY_MESH_OUTPUT_TRIM`: 0 never, `always` also for outputs that fit (a check).
    let trim = std::env::var("VICTORY_MESH_OUTPUT_TRIM").unwrap_or_default();
    if (now * max_vertices > METAL_LIMIT || trim == "always")
        && let Some(reads) = reads
        && trim != "0"
    {
        for e in els.iter_mut().filter(|e| e.kind == 0) {
            // An input the pixel shader declares and never reads keeps one component
            // (Metal links the two stages by name: the element stays).
            let n = reads.iter().find(|r| r.0.eq_ignore_ascii_case(&e.name) && r.1 == e.index).map(|r| r.2.max(1));
            if let Some(n) = n.filter(|&n| n < e.cols) {
                e.cols = n;
                e.usage &= (1 << n) - 1;
            }
        }
        now = tighten(els).1;
    }
    now < was
}

/// Numbers the rows of `els` in list order, one an element (columns as they were), for
/// a list [`tighten`] reordered: the signature's register order (`OSG1`) and its element
/// order (the metadata) then agree with the order chosen, whichever Metal lays out by.
/// Run once the exports have found their elements by the console's rows.
pub(in crate::emit) fn renumber(els: &mut [SigElement]) {
    let mut row = 0;
    for e in els.iter_mut().filter(|e| e.row.is_some()) {
        e.row = Some(row);
        row += 1;
    }
}

/// Calls `each` with every order of `items[k..]` after `items[..k]`.
fn permute(items: &mut [u8], k: usize, each: &mut dyn FnMut(&[u8])) {
    if k == items.len() {
        each(items);
        return;
    }
    for i in k..items.len() {
        items.swap(k, i);
        permute(items, k + 1, each);
        items.swap(k, i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn el(name: &str, index: u32, kind: u8, cols: u8, row: u32) -> SigElement {
        SigElement {
            name: name.into(),
            index,
            kind,
            comp: 9,
            interp: 2,
            row: Some(row),
            col: 0,
            cols,
            usage: (1 << cols) - 1,
            half: false,
        }
    }

    /// `a9184e594aae6932`'s vertex outputs (SV_Position, SV_ClipDistance and 19
    /// TEXCOORD/COLOR elements): 300 bytes a vertex in signature order (38,400 for 128
    /// vertices), 264 once ordered (33,792: still over; this one also needs the
    /// components its pixel shader does not read left out).
    #[test]
    fn the_title_mesh_shrinks_once_ordered() {
        let user = [4, 4, 4, 1, 2, 1, 4, 3, 1, 3, 3, 3, 3, 3, 3, 2, 4, 3, 3];
        let mut els = vec![el("SV_Position", 0, POSITION, 4, 0), el("SV_ClipDistance", 0, CLIP, 4, 1)];
        els.extend(user.iter().enumerate().map(|(i, &c)| el("TEXCOORD", i as u32, 0, c, 2 + i as u32)));
        let rows: Vec<_> = els.iter().map(|e| (e.name.clone(), e.index, e.row)).collect();
        let (before, after) = tighten(&mut els);
        assert_eq!(before, 300);
        assert_eq!(after, 264);
        assert_eq!(after * 128, 33_792); // still over: needs fewer components too
        assert_eq!(vertex_bytes(&els), after);
        // The same elements (rows and all), system values first.
        assert_eq!(els.len(), rows.len());
        for r in &rows {
            assert!(els.iter().any(|e| (e.name.clone(), e.index, e.row) == *r), "{r:?} lost");
        }
        assert!(els[0].kind == POSITION && els[1].kind == CLIP);
    }

    /// An order that is already the smallest stays as it is.
    #[test]
    fn a_tight_order_is_kept() {
        let mut els = vec![el("SV_Position", 0, POSITION, 4, 0), el("A", 0, 0, 4, 1), el("B", 0, 0, 2, 2), el("C", 0, 0, 1, 3)];
        let names = |e: &[SigElement]| e.iter().map(|e| e.name.clone()).collect::<Vec<_>>();
        let was = names(&els);
        let (before, after) = tighten(&mut els);
        assert_eq!((before, after), (44, 44));
        assert_eq!(names(&els), was);
    }

    #[test]
    fn layout_matches_metal() {
        // float, float4: 4, then 16 at 16 -> 32.
        assert_eq!(layout(0, [4, 16]), 32);
        // float3 after 16: at 24 (next multiple of 12) -> 36.
        assert_eq!(layout(16, [12]), 36);
    }
}
