//! Which Metal heap slots are already up to date, so a dispatch's record
//! step does not re-resolve them (stage 4: the D3D12-era per-dispatch heap
//! sync and slot refresh brought to 0 in the steady state).
//!
//! A slot is clean when it was resolved, with nothing missing, from the same
//! title descriptor (handle) at the current generation. It stops being clean
//! when the title writes that descriptor (`note_slot_dirty`), and every slot
//! stops being clean when anything an entry depends on may have changed:
//! title memory remapped (a view's home address follows the physical alias,
//! and partly remapped textures are refilled on resolve), a resource or a
//! heap released. Entries whose resolution was missing something are never
//! clean, so they are retried every time, as before.
//!
//! On only with `VICTORY_METAL_EXEC=1` or `VICTORY_METAL_SLOT_CACHE=1`
//! (`VICTORY_METAL_SLOT_CACHE=0` turns it off): without them the record path
//! behaves exactly as on main.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

static GENERATION: AtomicU64 = AtomicU64::new(1);
static CLEAN: Mutex<Option<HashMap<(usize, u32), (u64, u64)>>> = Mutex::new(None);
/// More than this many clean slots: start over (bounded memory).
const MOST: usize = 1 << 20;

pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_METAL_SLOT_CACHE").as_deref() {
        Ok("1") => true,
        Ok("0") => false,
        _ => std::env::var("VICTORY_METAL_EXEC").as_deref() == Ok("1"),
    })
}

/// The current generation (read before resolving, passed to [`mark`]).
pub(crate) fn generation() -> u64 {
    GENERATION.load(Ordering::Acquire)
}

/// Everything may have changed: no slot is clean any more.
pub(crate) fn stale() {
    GENERATION.fetch_add(1, Ordering::AcqRel);
}

/// The title wrote the descriptor of this slot.
pub(crate) fn dirty(heap: usize, index: u32) {
    if let Some(m) = CLEAN.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        m.remove(&(heap, index));
    }
}

pub(crate) fn is_clean(heap: usize, index: u32, handle: u64) -> bool {
    let g = generation();
    CLEAN.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|m| m.get(&(heap, index)).copied()) == Some((handle, g))
}

/// The slot was resolved from `handle`, nothing missing, at `generation`.
pub(crate) fn mark(heap: usize, index: u32, handle: u64, generation: u64) {
    let mut m = CLEAN.lock().unwrap_or_else(|e| e.into_inner());
    let m = m.get_or_insert_with(HashMap::new);
    if m.len() >= MOST {
        m.clear();
    }
    m.insert((heap, index), (handle, generation));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_stop_being_clean_on_writes_and_staleness() {
        let (heap, other) = (0x51_0000usize, 0x52_0000usize);
        let g = generation();
        mark(heap, 3, 0xaa, g);
        mark(other, 3, 0xbb, g);
        assert!(is_clean(heap, 3, 0xaa));
        // Another descriptor at the same slot is not the one resolved.
        assert!(!is_clean(heap, 3, 0xab));
        dirty(heap, 3);
        assert!(!is_clean(heap, 3, 0xaa));
        assert!(is_clean(other, 3, 0xbb));
        stale();
        assert!(!is_clean(other, 3, 0xbb));
        // Resolved before a staleness that came during the resolve: not clean.
        let before = generation();
        stale();
        mark(heap, 4, 0xcc, before);
        assert!(!is_clean(heap, 4, 0xcc));
    }
}
