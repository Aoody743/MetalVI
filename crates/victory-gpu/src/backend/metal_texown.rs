//! P5: texture ownership, judge only (`VICTORY_METAL_TEXOWN=shadow`,
//! default off; P5-DESIGN.md).
//!
//! Today a title-filled texture is brought up to date by
//! [`crate::texture::Staging::refresh`], which asks the write watch or the
//! guards and otherwise compares (hashes) the whole memory at every
//! submission that samples it. The ownership model answers the same
//! question from events instead: every way the bytes behind a texture can
//! change is noted here as an event over an address range, and a texture
//! needs a new fill exactly when an event touched its range since its last
//! check.
//!
//! Writers and the events they leave ([`Kind`]):
//! - the title's CPU stores, as the write watch or the guards found them
//!   (the journal in `crate::watch`): [`Kind::CpuSeen`];
//! - our own copies into title memory at submission (D3D12 deferred copies,
//!   Metal buffer copies, `crate::watch::wrote`): [`Kind::HostCopy`];
//! - bottom-of-pipe value writes (`crate::note_write`): [`Kind::BottomOfPipe`];
//! - the kernel's writes the runtime announces (ReadFile, fread,
//!   DirectStorage, xmem physical writes; `crate::watch::writing`, which
//!   the D3D12 path's copy-back of shader writes calls too): [`Kind::Kernel`];
//! - the GPU writing title buffers: shaders through unordered views
//!   (`Shadow::gpu_writes`) and texture-to-buffer read-backs (`list.rs`),
//!   both noted at recording: [`Kind::GpuWrite`];
//! - the F44 write-back of GPU-written textures into tiled title memory:
//!   [`Kind::GpuWriteBack`];
//! - xmem remaps of the backing pages (`crate::watch::title_memory_remapped`):
//!   [`Kind::Remap`];
//! - the title giving the memory to another resource (a released texture,
//!   a new allocation there): [`Kind::Reuse`].
//!
//! Writes that are announced before they land (the kernel's, the GPU's)
//! stay "live" for a few frames: a check while one is live, or the first
//! check after a check made while it was live, still counts it. That turns
//! the race between announcement and landing into a wasted refresh instead
//! of a missed one.
//!
//! In shadow mode nothing changes: at each place `refresh()` decides
//! "changed" or "unchanged" ([`Source`]), [`judge`] asks the model too and
//! counts agreements and disagreements ([`Outcome`]); the first samples of
//! each disagreement are logged with the texture and the last events on its
//! range. `Outcome::Missed` (the model says unchanged, the check found a
//! change) is a writer the model does not know: hashing cannot be turned
//! off while those happen.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// Who changed the bytes of a range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    CpuSeen = 0,
    HostCopy,
    BottomOfPipe,
    Kernel,
    GpuWrite,
    GpuWriteBack,
    Remap,
    Reuse,
}

pub(crate) const KINDS: usize = 8;
const KIND_NAMES: [&str; KINDS] = ["cpu_seen", "host_copy", "bottom_of_pipe", "kernel", "gpu_write", "gpu_writeback", "remap", "reuse"];

impl Kind {
    /// Frames an event stays live after it was noted: writes announced
    /// before they land (the kernel's, possibly overlapped; the GPU's,
    /// which run when the submission does, frames later on the execution
    /// timeline; the GPU's unordered writes are noted when the list is
    /// recorded) or noted just after the mapping changed. `None`: noted
    /// after the write landed.
    fn live_frames(self) -> Option<u64> {
        match self {
            Kind::CpuSeen | Kind::HostCopy => None,
            Kind::BottomOfPipe | Kind::Remap | Kind::Reuse => Some(1),
            Kind::GpuWriteBack => Some(2),
            Kind::GpuWrite => Some(3),
            Kind::Kernel => Some(4),
        }
    }
}

/// Where `refresh()` decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Source {
    /// Write-watched memory: the kernel's dirty pages.
    WriteWatch = 0,
    /// View memory, guarded, no write seen and not its turn to be hashed.
    Guard,
    /// Partly mapped memory, hashed through the copy.
    PartialHash,
    /// Wholly mapped memory, hashed or compared byte for byte.
    Compare,
    /// Read through the kernel while title memory was being remapped.
    Unlocked,
}

pub(crate) const SOURCES: usize = 5;
const SOURCE_NAMES: [&str; SOURCES] = ["write_watch", "guard", "partial_hash", "compare", "unlocked"];

impl Source {
    /// Whether the verdict looked at the bytes (a hash or a compare): the
    /// truth a missed writer is measured against.
    pub(crate) fn content(self) -> bool {
        matches!(self, Source::PartialHash | Source::Compare | Source::Unlocked)
    }
}

/// What the model said against what the check found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    /// The texture's first check: nothing to compare.
    First = 0,
    AgreeChanged,
    AgreeUnchanged,
    /// The model says unchanged, the check found a change: a writer the
    /// model does not know (dangerous).
    Missed,
    /// The model says changed, the check found nothing (a refresh the model
    /// would waste).
    Wasted,
}

pub(crate) const OUTCOMES: usize = 5;
const OUTCOME_NAMES: [&str; OUTCOMES] = ["first", "agree_changed", "agree_unchanged", "missed", "wasted"];

/// The model's clock at a moment: the last event's number and the frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Stamp {
    pub(crate) seq: u64,
    pub(crate) frame: u64,
}

/// What a texture's last check saw (kept with the texture's other refresh
/// state). `None`: never checked.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Mark(Option<Stamp>);

impl Mark {
    pub(crate) fn stamp(&self) -> Option<Stamp> {
        self.0
    }
}

/// The newest event over a stretch of addresses.
#[derive(Clone, Copy, Debug)]
struct Entry {
    end: usize,
    seq: u64,
    kind: Kind,
    /// The last frame an announced write may still land in.
    live: Option<u64>,
}

/// A noted event, for the samples.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Event {
    pub(crate) seq: u64,
    pub(crate) frame: u64,
    pub(crate) kind: Kind,
    pub(crate) at: usize,
    pub(crate) len: usize,
}

/// Why the model says a range changed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Why {
    Event(Kind, u64),
    /// Events were dropped to bound the map (counted as changed).
    Overflow,
}

const RECENT: usize = 8192;
const SAMPLES_MISSED: u32 = 16;
const SAMPLES_WASTED: u32 = 8;

/// The ownership model: events over address ranges, newest first.
pub(crate) struct Model {
    seq: u64,
    /// Non-overlapping stretches, by start: the newest event over each
    /// (with the latest live frame of any event there).
    map: BTreeMap<usize, Entry>,
    /// Stretches kept at most; past it the map is emptied and `floor`
    /// stands for what it held.
    cap: usize,
    floor: Option<Stamp>,
    floor_live: Option<u64>,
    recent: VecDeque<Event>,
    pub(crate) counts: [[u64; OUTCOMES]; SOURCES],
    pub(crate) bytes: [[u64; OUTCOMES]; SOURCES],
    /// Of the agreed changes and the wasted refreshes, by the kind of the
    /// event the model blamed.
    pub(crate) blamed: [[u64; KINDS]; 2],
    pub(crate) events: [u64; KINDS],
    pub(crate) event_bytes: [u64; KINDS],
    samples: [[u32; 2]; SOURCES],
}

impl Default for Model {
    fn default() -> Self {
        Self::new(1 << 18)
    }
}

impl Model {
    pub(crate) fn new(cap: usize) -> Self {
        Self {
            seq: 0,
            map: BTreeMap::new(),
            cap: cap.max(4),
            floor: None,
            floor_live: None,
            recent: VecDeque::with_capacity(RECENT),
            counts: [[0; OUTCOMES]; SOURCES],
            bytes: [[0; OUTCOMES]; SOURCES],
            blamed: [[0; KINDS]; 2],
            events: [0; KINDS],
            event_bytes: [0; KINDS],
            samples: [[0; 2]; SOURCES],
        }
    }

    #[cfg(test)]
    pub(crate) fn now(&self, frame: u64) -> Stamp {
        Stamp { seq: self.seq, frame }
    }

    /// Notes that `kind` changed (or will change, see [`Kind::live_frames`])
    /// the bytes `[at, at + len)` in frame `frame`.
    pub(crate) fn note(&mut self, kind: Kind, at: usize, len: usize, frame: u64) {
        if len == 0 {
            return;
        }
        let end = at.saturating_add(len);
        self.seq += 1;
        let seq = self.seq;
        let live = kind.live_frames().map(|n| frame.saturating_add(n));
        self.events[kind as usize] += 1;
        self.event_bytes[kind as usize] = self.event_bytes[kind as usize].saturating_add((end - at) as u64);
        if self.recent.len() == RECENT {
            self.recent.pop_front();
        }
        self.recent.push_back(Event { seq, frame, kind, at, len: end - at });
        self.assign(at, end, seq, kind, live);
        if self.map.len() > self.cap {
            // Bounded: what the map held counts as changed for every
            // texture checked before now (wasted refreshes, never missed
            // ones).
            let live_max = self.map.values().filter_map(|e| e.live).max();
            self.floor = Some(Stamp { seq, frame });
            self.floor_live = live_max.max(self.floor_live);
            self.map.clear();
        }
    }

    /// Sets `[at, end)` to event `seq`, keeping the latest live frame of
    /// what it covers.
    fn assign(&mut self, at: usize, end: usize, seq: u64, kind: Kind, live: Option<u64>) {
        let mut overlapping: Vec<(usize, Entry)> = Vec::new();
        if let Some((&s, e)) = self.map.range(..at).next_back()
            && e.end > at
        {
            overlapping.push((s, *e));
        }
        overlapping.extend(self.map.range(at..end).map(|(&s, e)| (s, *e)));
        let mut cursor = at;
        for (s, e) in overlapping {
            self.map.remove(&s);
            // What lies outside the new event stays as it was.
            if s < at {
                self.map.insert(s, Entry { end: at, ..e });
            }
            if e.end > end {
                self.map.insert(end, e);
            }
            let (from, to) = (s.max(at), e.end.min(end));
            if cursor < from {
                self.map.insert(cursor, Entry { end: from, seq, kind, live });
            }
            self.map.insert(from, Entry { end: to, seq, kind, live: live.max(e.live) });
            cursor = to;
        }
        if cursor < end {
            self.map.insert(cursor, Entry { end, seq, kind, live });
        }
        // The event's pieces side by side with the same live frame are one
        // stretch.
        let keys: Vec<usize> = self.map.range(at..end).map(|(&s, _)| s).collect();
        let mut prev: Option<usize> = None;
        for k in keys {
            if let Some(p) = prev
                && let (Some(&pe), Some(&ke)) = (self.map.get(&p), self.map.get(&k))
                && pe.end == k
                && pe.seq == ke.seq
                && pe.live == ke.live
            {
                self.map.remove(&k);
                if let Some(e) = self.map.get_mut(&p) {
                    e.end = ke.end;
                }
                continue;
            }
            prev = Some(k);
        }
    }

    /// Whether an event touched `[at, at + len)` since a check at `since`:
    /// one noted after it, or one still live when it was made.
    pub(crate) fn changed_since(&self, at: usize, len: usize, since: Stamp) -> Option<Why> {
        let end = at.saturating_add(len);
        if let Some(floor) = self.floor
            && (since.seq < floor.seq || self.floor_live.is_some_and(|l| l >= since.frame))
        {
            return Some(Why::Overflow);
        }
        let counts = |e: &Entry| e.seq > since.seq || e.live.is_some_and(|l| l >= since.frame);
        let mut best: Option<&Entry> = None;
        let first = self.map.range(..at).next_back().filter(|(_, e)| e.end > at);
        for (_, e) in first.into_iter().chain(self.map.range(at..end)) {
            if counts(e) && best.is_none_or(|b| e.seq > b.seq) {
                best = Some(e);
            }
        }
        best.map(|e| Why::Event(e.kind, e.seq))
    }

    /// Judges one check of the texture at `[at, at + len)`: `changed` is
    /// what `refresh()` found, `stamp` the model's clock taken just before
    /// it read the bytes (after it asked the write watch). Moves `mark` on.
    pub(crate) fn judge(&mut self, mark: &mut Mark, at: usize, len: usize, source: Source, changed: bool, stamp: Stamp) -> (Outcome, Option<Why>) {
        let (outcome, why) = match mark.0 {
            None => (Outcome::First, None),
            Some(since) => {
                let why = self.changed_since(at, len, since);
                let outcome = match (why.is_some(), changed) {
                    (true, true) => Outcome::AgreeChanged,
                    (false, false) => Outcome::AgreeUnchanged,
                    (false, true) => Outcome::Missed,
                    (true, false) => Outcome::Wasted,
                };
                (outcome, why)
            }
        };
        mark.0 = Some(stamp);
        let (s, o) = (source as usize, outcome as usize);
        self.counts[s][o] += 1;
        self.bytes[s][o] = self.bytes[s][o].saturating_add(len as u64);
        if let Some(Why::Event(kind, _)) = why {
            match outcome {
                Outcome::AgreeChanged => self.blamed[0][kind as usize] += 1,
                Outcome::Wasted => self.blamed[1][kind as usize] += 1,
                _ => {}
            }
        }
        (outcome, why)
    }

    /// Whether to log a sample of `outcome` from `source` (the first few of
    /// each disagreement).
    fn sample(&mut self, source: Source, outcome: Outcome) -> bool {
        let (slot, most) = match outcome {
            Outcome::Missed => (0, SAMPLES_MISSED),
            Outcome::Wasted => (1, SAMPLES_WASTED),
            _ => return false,
        };
        let n = &mut self.samples[source as usize][slot];
        *n += 1;
        *n <= most
    }

    /// The last `most` events over `[at, at + len)`, newest first.
    pub(crate) fn recent_over(&self, at: usize, len: usize, most: usize) -> Vec<Event> {
        let end = at.saturating_add(len);
        self.recent.iter().rev().filter(|e| e.at < end && e.at.saturating_add(e.len) > at).take(most).copied().collect()
    }

    pub(crate) fn total(&self) -> u64 {
        self.counts.iter().flatten().sum()
    }

    /// One line of all counters.
    pub(crate) fn summary(&self) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        for (i, name) in SOURCE_NAMES.iter().enumerate() {
            if self.counts[i].iter().all(|&n| n == 0) {
                continue;
            }
            let _ = write!(s, "{name}[");
            for (o, outcome) in OUTCOME_NAMES.iter().enumerate() {
                let _ = write!(s, "{}{outcome}={} ({} MB)", if o == 0 { "" } else { " " }, self.counts[i][o], self.bytes[i][o] >> 20);
            }
            s.push_str("] ");
        }
        s.push_str("events[");
        for (k, name) in KIND_NAMES.iter().enumerate() {
            let _ = write!(s, "{}{name}={} ({} MB)", if k == 0 { "" } else { " " }, self.events[k], self.event_bytes[k] >> 20);
        }
        s.push_str("] blamed_changed[");
        for (k, name) in KIND_NAMES.iter().enumerate() {
            let _ = write!(s, "{}{name}={}", if k == 0 { "" } else { " " }, self.blamed[0][k]);
        }
        s.push_str("] blamed_wasted[");
        for (k, name) in KIND_NAMES.iter().enumerate() {
            let _ = write!(s, "{}{name}={}", if k == 0 { "" } else { " " }, self.blamed[1][k]);
        }
        let _ = write!(s, "] stretches={} overflowed={}", self.map.len(), self.floor.is_some());
        s
    }
}

/// `VICTORY_METAL_TEXOWN=shadow`: the judge runs beside `refresh()`.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    #[cfg(test)]
    if FORCED.load(Ordering::Relaxed) {
        return true;
    }
    *ON.get_or_init(|| {
        let on = std::env::var("VICTORY_METAL_TEXOWN").is_ok_and(|v| v == "shadow");
        if on {
            tracing::info!("texture ownership: shadow judge on (VICTORY_METAL_TEXOWN=shadow); refresh() still decides");
        }
        on
    })
}

#[cfg(test)]
static FORCED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Turns the global judge on for a test.
#[cfg(test)]
pub(crate) fn force_on() {
    FORCED.store(true, Ordering::Relaxed);
}

static MODEL: Mutex<Option<Model>> = Mutex::new(None);
/// The last event's number, readable without the lock ([`stamp`]).
static SEQ: AtomicU64 = AtomicU64::new(0);

fn with_model<R>(f: impl FnOnce(&mut Model) -> R) -> R {
    let mut m = MODEL.lock().unwrap_or_else(|e| e.into_inner());
    f(m.get_or_insert_with(Model::default))
}

fn frame() -> u64 {
    crate::frame::current()
}

/// Notes an event (nothing unless [`on`]).
pub(crate) fn note(kind: Kind, at: usize, len: usize) {
    if !on() || len == 0 {
        return;
    }
    let frame = frame();
    with_model(|m| {
        m.note(kind, at, len, frame);
        SEQ.store(m.seq, Ordering::Release);
    });
}

/// Notes the write watch's or the guards' written pages (page numbers of
/// `crate::watch::PAGE` bytes, in order) as runs.
pub(crate) fn note_pages(pages: impl IntoIterator<Item = usize>, page: usize) {
    if !on() {
        return;
    }
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for p in pages {
        match runs.last_mut() {
            Some(r) if r.1 == p => r.1 = p + 1,
            Some(r) if r.0 <= p && p < r.1 => {}
            _ => runs.push((p, p + 1)),
        }
    }
    if runs.is_empty() {
        return;
    }
    let frame = frame();
    with_model(|m| {
        for (a, b) in runs {
            m.note(Kind::CpuSeen, a * page, (b - a) * page, frame);
        }
        SEQ.store(m.seq, Ordering::Release);
    });
}

/// The model's clock now: taken by `refresh()` just before it reads a
/// texture's bytes (after asking the write watch, whose answers are events
/// themselves). Default (unused) when off.
pub(crate) fn stamp() -> Stamp {
    if !on() {
        return Stamp::default();
    }
    Stamp { seq: SEQ.load(Ordering::Acquire), frame: frame() }
}

/// [`Model::judge`] on the global model, logging the first samples of each
/// disagreement and the counters now and then.
pub(crate) fn judge(mark: &mut Mark, at: usize, len: usize, source: Source, changed: bool, stamp: Stamp, desc: &dyn std::fmt::Debug) {
    if !on() {
        return;
    }
    let before = mark.stamp();
    let (line, sample) = with_model(|m| {
        let (outcome, why) = m.judge(mark, at, len, source, changed, stamp);
        let sample = m.sample(source, outcome).then(|| (outcome, why, m.recent_over(at, len, 6)));
        let total = m.total();
        let line = (total.is_power_of_two() && total >= 64 || total % 65536 == 0).then(|| m.summary());
        (line, sample)
    });
    if let Some((outcome, why, events)) = sample {
        let events: Vec<String> = events
            .iter()
            .map(|e| format!("#{} f{} {} {:#x}+{:#x}", e.seq, e.frame, KIND_NAMES[e.kind as usize], e.at, e.len))
            .collect();
        let since = before.unwrap_or_default();
        match outcome {
            Outcome::Missed => tracing::warn!(
                source = SOURCE_NAMES[source as usize],
                content = source.content(),
                at = format_args!("{at:#x}"),
                len,
                ?desc,
                since_seq = since.seq,
                since_frame = since.frame,
                frame = stamp.frame,
                events = ?events,
                "texture ownership: MISSED WRITER (the model says unchanged, the check found a change)"
            ),
            _ => tracing::info!(
                source = SOURCE_NAMES[source as usize],
                at = format_args!("{at:#x}"),
                len,
                ?desc,
                why = ?why.map(|w| match w {
                    Why::Event(kind, seq) => format!("{} #{seq}", KIND_NAMES[kind as usize]),
                    Why::Overflow => "overflow".into(),
                }),
                since_seq = since.seq,
                since_frame = since.frame,
                frame = stamp.frame,
                events = ?events,
                "texture ownership: wasted refresh (the model says changed, the check found nothing)"
            ),
        }
    }
    if let Some(line) = line {
        tracing::info!("texture ownership (shadow): {line}");
    }
}

/// The global counters, one line (tests).
#[cfg(test)]
pub(crate) fn summary() -> String {
    with_model(|m| m.summary())
}

/// A copy of the global counters by source and outcome (tests).
#[cfg(test)]
pub(crate) fn counts() -> [[u64; OUTCOMES]; SOURCES] {
    with_model(|m| m.counts)
}

#[cfg(test)]
#[path = "metal_texown_tests.rs"]
mod tests;
