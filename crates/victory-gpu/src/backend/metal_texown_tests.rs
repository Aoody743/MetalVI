//! P5 judge: each writer kind against a content hash of synthetic title
//! memory, the way `Staging::refresh` hashes it; and an untracked writer the
//! judge must flag as missed.

use super::*;

/// A fake title address for the synthetic memory.
const BASE: usize = 0x2_0000_0000;
const PAGE: usize = 64 << 10;

/// Title memory and a texture in it, checked as `refresh()` checks it: a
/// hash of its bytes against the last one.
struct World {
    model: Model,
    memory: Vec<u8>,
    frame: u64,
}

struct Tex {
    at: usize,
    len: usize,
    mark: Mark,
    hash: Option<blake3::Hash>,
}

impl World {
    fn new() -> Self {
        Self { model: Model::default(), memory: vec![0; 16 * PAGE], frame: 1 }
    }

    fn tex(&self, offset: usize, len: usize) -> Tex {
        Tex { at: BASE + offset, len, mark: Mark::default(), hash: None }
    }

    /// Writes `len` bytes of `value` at `offset` (no event: the caller notes
    /// one, or not).
    fn poke(&mut self, offset: usize, len: usize, value: u8) {
        self.memory[offset..offset + len].fill(value);
    }

    fn note(&mut self, kind: Kind, offset: usize, len: usize) {
        self.model.note(kind, BASE + offset, len, self.frame);
    }

    /// One check: the model's clock before the read, the hash, the judge.
    fn check(&mut self, t: &mut Tex) -> Outcome {
        let stamp = self.model.now(self.frame);
        let from = t.at - BASE;
        let hash = blake3::hash(&self.memory[from..from + t.len]);
        let changed = t.hash != Some(hash);
        t.hash = Some(hash);
        self.model.judge(&mut t.mark, t.at, t.len, Source::Compare, changed, stamp).0
    }

    /// Lets every live announcement run out: checks past their frames,
    /// the last of which must agree unchanged.
    fn settle(&mut self, t: &mut Tex) {
        self.frame += 8;
        let o = self.check(t);
        assert!(matches!(o, Outcome::Wasted | Outcome::AgreeUnchanged), "{o:?}");
        self.frame += 1;
        assert_eq!(self.check(t), Outcome::AgreeUnchanged);
    }

    fn missed(&self) -> u64 {
        self.model.counts.iter().map(|c| c[Outcome::Missed as usize]).sum()
    }
}

#[test]
fn a_texture_nothing_touches_agrees_unchanged() {
    let mut w = World::new();
    let mut t = w.tex(PAGE, 2 * PAGE);
    assert_eq!(w.check(&mut t), Outcome::First);
    for _ in 0..5 {
        w.frame += 1;
        assert_eq!(w.check(&mut t), Outcome::AgreeUnchanged);
    }
    // A write beside it is not its business.
    w.poke(4 * PAGE, 16, 9);
    w.note(Kind::CpuSeen, 4 * PAGE, 16);
    w.poke(0, PAGE, 3);
    w.note(Kind::CpuSeen, 0, PAGE);
    assert_eq!(w.check(&mut t), Outcome::AgreeUnchanged);
}

#[test]
fn a_cpu_write_the_watch_saw_agrees_changed() {
    let mut w = World::new();
    let mut t = w.tex(PAGE, 2 * PAGE);
    w.check(&mut t);
    w.poke(PAGE + 100, 8, 0x5a);
    w.note(Kind::CpuSeen, PAGE, PAGE);
    assert_eq!(w.check(&mut t), Outcome::AgreeChanged);
    assert_eq!(w.check(&mut t), Outcome::AgreeUnchanged);
    // The last byte of the texture.
    w.poke(3 * PAGE - 1, 1, 1);
    w.note(Kind::CpuSeen, 2 * PAGE, PAGE);
    assert_eq!(w.check(&mut t), Outcome::AgreeChanged);
    assert_eq!(w.model.blamed[0][Kind::CpuSeen as usize], 2);
    assert_eq!(w.missed(), 0);
}

#[test]
fn our_copies_and_bottom_of_pipe_writes_agree_changed() {
    let mut w = World::new();
    let mut t = w.tex(0, 4 * PAGE);
    w.check(&mut t);
    // A deferred copy at submission: noted after it landed.
    w.poke(10, 100, 7);
    w.note(Kind::HostCopy, 10, 100);
    assert_eq!(w.check(&mut t), Outcome::AgreeChanged);
    // A bottom-of-pipe value: noted just before the store.
    w.note(Kind::BottomOfPipe, 2 * PAGE, 8);
    w.poke(2 * PAGE, 8, 1);
    assert_eq!(w.check(&mut t), Outcome::AgreeChanged);
    w.settle(&mut t);
    assert_eq!(w.missed(), 0);
}

/// The kernel's and the GPU's writes are noted before they land: a check in
/// between finds nothing (a wasted refresh), the check after the write must
/// still count it, and once a check came after the live frames it is over.
#[test]
fn writes_announced_before_landing_are_never_missed() {
    for kind in [Kind::GpuWriteBack, Kind::GpuWrite, Kind::Kernel] {
        let mut w = World::new();
        let mut t = w.tex(PAGE, 2 * PAGE);
        w.check(&mut t);
        w.note(kind, PAGE, 2 * PAGE);
        // Checked while the GPU (or the disk) has not written yet.
        assert_eq!(w.check(&mut t), Outcome::Wasted, "{kind:?}");
        // It lands a frame later.
        w.frame += 1;
        w.poke(PAGE, 2 * PAGE, 0xee);
        assert_eq!(w.check(&mut t), Outcome::AgreeChanged, "{kind:?}");
        // Past the live frames: unchanged again (wasted while still live).
        w.frame += 8;
        let after = w.check(&mut t);
        assert!(matches!(after, Outcome::Wasted | Outcome::AgreeUnchanged), "{kind:?}: {after:?}");
        w.frame += 1;
        assert_eq!(w.check(&mut t), Outcome::AgreeUnchanged, "{kind:?}");
        assert_eq!(w.missed(), 0, "{kind:?}");
    }
}

/// An announced write still live when a later event covers the same bytes
/// keeps its live frames.
#[test]
fn a_newer_event_keeps_an_older_announcement_live() {
    let mut w = World::new();
    let mut t = w.tex(0, PAGE);
    w.check(&mut t);
    w.note(Kind::Kernel, 0, PAGE);
    w.note(Kind::CpuSeen, 0, PAGE);
    assert_eq!(w.check(&mut t), Outcome::Wasted);
    w.frame += 2;
    w.poke(0, 4, 1); // the overlapped read lands
    assert_eq!(w.check(&mut t), Outcome::AgreeChanged);
    assert_eq!(w.missed(), 0);
}

#[test]
fn a_remap_of_the_backing_pages_agrees_changed() {
    let mut w = World::new();
    let mut t = w.tex(2 * PAGE, 4 * PAGE);
    w.poke(2 * PAGE, 4 * PAGE, 0x11);
    w.check(&mut t);
    // xmem maps other physical pages under two of the texture's pages (the
    // runtime announces it after the mapping changed).
    w.poke(3 * PAGE, 2 * PAGE, 0x22);
    w.note(Kind::Remap, 3 * PAGE, 2 * PAGE);
    assert_eq!(w.check(&mut t), Outcome::AgreeChanged);
    // Unmapped: reads as zeros.
    w.poke(2 * PAGE, 4 * PAGE, 0);
    w.note(Kind::Remap, 0, 8 * PAGE);
    assert_eq!(w.check(&mut t), Outcome::AgreeChanged);
    assert_eq!(w.model.blamed[0][Kind::Remap as usize], 2);
    assert_eq!(w.missed(), 0);
}

#[test]
fn memory_given_to_another_resource_agrees_changed() {
    let mut w = World::new();
    let mut old = w.tex(PAGE, 2 * PAGE);
    w.poke(PAGE, 2 * PAGE, 0x33);
    w.check(&mut old);
    // The title releases the texture and places another resource there,
    // which a copy fills.
    w.note(Kind::Reuse, PAGE, 2 * PAGE);
    w.poke(PAGE, PAGE, 0x44);
    w.note(Kind::HostCopy, PAGE, PAGE);
    assert_eq!(w.check(&mut old), Outcome::AgreeChanged);
    // A texture made there anew starts with a fill of its own.
    let mut new = w.tex(PAGE, PAGE);
    assert_eq!(w.check(&mut new), Outcome::First);
    w.settle(&mut new);
    assert_eq!(w.missed(), 0);
}

/// A writer the model is not told of (the title's own stores where neither
/// the write watch nor the guards look): the hash finds the change, the
/// judge flags it as a missed writer.
#[test]
fn an_untracked_writer_is_flagged_as_missed() {
    let mut w = World::new();
    let mut t = w.tex(PAGE, 2 * PAGE);
    w.check(&mut t);
    w.frame += 1;
    w.poke(PAGE + 4096, 64, 0x99);
    assert_eq!(w.check(&mut t), Outcome::Missed);
    assert_eq!(w.model.counts[Source::Compare as usize][Outcome::Missed as usize], 1);
    assert_eq!(w.model.bytes[Source::Compare as usize][Outcome::Missed as usize], 2 * PAGE as u64);
    // Then nothing: agreement again.
    assert_eq!(w.check(&mut t), Outcome::AgreeUnchanged);
    // No events on the range to show in the sample.
    assert!(w.model.recent_over(t.at, t.len, 6).is_empty());
}

#[test]
fn rewriting_the_same_bytes_is_a_wasted_refresh() {
    let mut w = World::new();
    let mut t = w.tex(0, PAGE);
    w.poke(0, PAGE, 5);
    w.check(&mut t);
    w.poke(0, PAGE, 5);
    w.note(Kind::CpuSeen, 0, PAGE);
    assert_eq!(w.check(&mut t), Outcome::Wasted);
    assert_eq!(w.model.blamed[1][Kind::CpuSeen as usize], 1);
    let events = w.model.recent_over(t.at, t.len, 6);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].kind, Kind::CpuSeen);
}

/// The map keeps the newest event per stretch: a write in the middle of an
/// older one splits it, and what lies outside stays as it was.
#[test]
fn events_split_and_overwrite_stretches() {
    let mut m = Model::default();
    // A remap at frame 1: live through frame 2.
    m.note(Kind::Remap, 0, 10 * PAGE, 1);
    let mid = Stamp { seq: m.seq, frame: 3 };
    m.note(Kind::CpuSeen, 4 * PAGE, PAGE, 3);
    assert_eq!(m.changed_since(0, 4 * PAGE, mid), None);
    assert_eq!(m.changed_since(5 * PAGE, 5 * PAGE, mid), None);
    assert!(matches!(m.changed_since(4 * PAGE + 10, 1, mid), Some(Why::Event(Kind::CpuSeen, _))));
    // A check made at frame 1, while the remap was live, counts it even
    // though it came after it.
    assert!(matches!(m.changed_since(0, PAGE, Stamp { seq: mid.seq, frame: 1 }), Some(Why::Event(Kind::Remap, _))));
    // Stretches: [0,4) [4,5) [5,10).
    assert_eq!(m.map.len(), 3);
    // The write inside the live remap took on its live frame: a check at
    // frame 1 still counts the remap there.
    assert_eq!(m.map.get(&(4 * PAGE)).map(|e| (e.kind, e.live)), Some((Kind::CpuSeen, Some(2))));
    // A copy over two edges: [0,3) remap, [3,6) copy (the live frame of
    // what it covered kept), [6,10) remap.
    m.note(Kind::HostCopy, 3 * PAGE, 3 * PAGE, 5);
    let stretches: Vec<(usize, usize, Kind, Option<u64>)> = m.map.iter().map(|(&s, e)| (s / PAGE, e.end / PAGE, e.kind, e.live)).collect();
    assert_eq!(stretches, [(0, 3, Kind::Remap, Some(2)), (3, 6, Kind::HostCopy, Some(2)), (6, 10, Kind::Remap, Some(2))]);
    // Past a stretch's end, a newer event without live frames stays apart.
    m.note(Kind::HostCopy, 10 * PAGE, PAGE, 5);
    assert_eq!(m.map.range(10 * PAGE..).count(), 1);
    // Neighbours of one event with the same live frame are one stretch.
    m.note(Kind::CpuSeen, 20 * PAGE, PAGE, 5);
    m.note(Kind::CpuSeen, 22 * PAGE, PAGE, 5);
    m.note(Kind::GpuWrite, 19 * PAGE, 5 * PAGE, 5);
    assert_eq!(m.map.range(19 * PAGE..).count(), 1);
    // An event across all of it and beyond: live frames kept where they
    // were longer, else one stretch.
    m.note(Kind::Reuse, 0, usize::MAX, 9);
    assert!(m.map.values().all(|e| e.kind == Kind::Reuse));
    assert_eq!(m.changed_since(0, usize::MAX, Stamp { seq: m.seq, frame: 11 }), None);
}

/// Past its bound the map is emptied and every texture checked before
/// counts as changed (wasted, never missed).
#[test]
fn an_overflowing_map_counts_as_changed() {
    let mut w = World { model: Model::new(8), memory: vec![0; 64 * PAGE], frame: 1 };
    let mut t = w.tex(40 * PAGE, PAGE);
    w.check(&mut t);
    for k in 0..12 {
        w.note(Kind::CpuSeen, 2 * k * PAGE, PAGE);
    }
    assert!(w.model.map.len() <= 8);
    assert_eq!(w.model.changed_since(t.at, t.len, t.mark.stamp().unwrap()), Some(Why::Overflow));
    assert_eq!(w.check(&mut t), Outcome::Wasted);
    w.frame += 3;
    w.check(&mut t);
    w.frame += 1;
    assert_eq!(w.check(&mut t), Outcome::AgreeUnchanged);
}

/// The global hooks (the runtime's announcements, our own writes, the
/// write watch's pages) feed the judge `refresh()` calls, and an untracked
/// write shows as missed in its counters.
#[test]
fn the_global_hooks_feed_the_judge() {
    force_on();
    let mut memory = vec![0u8; 8 * PAGE];
    let at = memory.as_ptr() as usize;
    let len = memory.len();
    let hash = |m: &[u8]| blake3::hash(m);
    let mut mark = Mark::default();
    let mut last = hash(&memory);
    let check = |memory: &[u8], mark: &mut Mark, last: &mut blake3::Hash| {
        let stamp = stamp();
        let h = hash(memory);
        let changed = h != *last;
        *last = h;
        judge(mark, at, len, Source::Compare, changed, stamp, &"synthetic 8 pages");
    };
    let before = counts()[Source::Compare as usize];
    check(&memory, &mut mark, &mut last); // first
    // ReadFile into it.
    crate::watch::writing(at + 100, 50);
    memory[100..150].fill(1);
    check(&memory, &mut mark, &mut last);
    // A bottom-of-pipe value.
    crate::note_write(at + PAGE, 8, 1);
    memory[PAGE..PAGE + 8].fill(2);
    check(&memory, &mut mark, &mut last);
    // A copy at submission.
    memory[2 * PAGE..3 * PAGE].fill(3);
    crate::watch::wrote(at + 2 * PAGE, PAGE);
    check(&memory, &mut mark, &mut last);
    // A remap under part of it.
    memory[3 * PAGE..4 * PAGE].fill(0x77);
    crate::watch::title_memory_remapped(at + 3 * PAGE, PAGE);
    check(&memory, &mut mark, &mut last);
    // A store the write watch found.
    memory[5 * PAGE + 7] = 4;
    note(Kind::CpuSeen, at + 5 * PAGE, PAGE);
    check(&memory, &mut mark, &mut last);
    let mid = counts()[Source::Compare as usize];
    assert_eq!(mid[Outcome::First as usize] - before[Outcome::First as usize], 1);
    assert_eq!(mid[Outcome::Missed as usize] - before[Outcome::Missed as usize], 0);
    // Agreed changes, or wasted ones while announcements are still live;
    // never missed.
    assert!(mid[Outcome::AgreeChanged as usize] - before[Outcome::AgreeChanged as usize] >= 1);
    // Untracked: two frames' worth of live announcements may still count
    // it as changed, so the store comes after they cannot (a frame number
    // does not move in tests): a fresh mark taken past every live frame.
    let mut fresh = Mark(Some(Stamp { seq: stamp().seq, frame: u64::MAX / 2 }));
    let quiet = Stamp { seq: stamp().seq, frame: u64::MAX / 2 };
    memory[7 * PAGE] = 9;
    judge(&mut fresh, at, len, Source::Compare, true, quiet, &"synthetic 8 pages");
    let after = counts()[Source::Compare as usize];
    assert_eq!(after[Outcome::Missed as usize] - mid[Outcome::Missed as usize], 1);
    assert!(summary().contains("compare["));
    drop(memory);
}

/// The write watch's pages arrive as runs.
#[test]
fn written_pages_are_noted_as_runs() {
    force_on();
    let p = 0x7_1234_0000usize / PAGE;
    note_pages([p, p + 1, p + 2, p + 1, p + 5], PAGE);
    let events = with_model(|m| m.recent_over(p * PAGE, 6 * PAGE, 6));
    let runs: Vec<(usize, usize, Kind)> = events.iter().map(|e| (e.at / PAGE - p, e.len / PAGE, e.kind)).collect();
    assert_eq!(runs, [(5, 1, Kind::CpuSeen), (0, 3, Kind::CpuSeen)]);
}
