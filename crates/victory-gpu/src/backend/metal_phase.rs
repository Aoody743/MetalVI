//! Lock-free host phase tracking for the Metal hang watch.
use std::cell::Cell;
#[cfg(test)]
use std::cell::RefCell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

// Lane 0 is submission. Lane 1 captures and presents.
pub(super) const LANES: usize = 2;
pub(super) const LANE_RUN: usize = 0;
const LANE_CAPTURE: usize = 1;

// Keep phase IDs and their log strings in the same order.
macro_rules! phases {
    ($($phase:ident => $name:literal),+ $(,)?) => {
        #[derive(Clone, Copy)]
        pub(super) enum Phase { $($phase),+ }
        const PHASE_NAMES: &[&str] = &[$($name),+];
        #[cfg(test)]
        const ALL_PHASES: &[Phase] = &[$(Phase::$phase),+];
    };
}

phases! {
    Idle => "",
    WaitForGpuIdle => "waiting for the GPU to go idle (wiring)",
    WireTitleMemory => "wiring title memory",
    WaitForResidency => "waiting for residency changes",
    Commit => "commit",
    WaitForCommandBuffer => "waiting for a command buffer",
    RunBatch => "run_batch",
    EncodeDispatches => "encode dispatches",
    EncodeIndirectDispatch => "encode indirect dispatch",
    EncodeReadback => "encode readback",
    EncodeTextureCopy => "encode texture copy",
    EncodeResolve => "encode resolve",
    EncodeAmbient => "encode ambient",
    EncodeRectangleClear => "encode rectangle clear",
    EncodeClear => "encode clear",
    EncodeDraws => "encode draws",
    BoundedPassSubmit => "bounded pass submit",
    NewCommandBuffer => "new command buffer",
    RunBatchEncode => "run_batch encode",
    Writeback => "writeback",
    ResidencyCommit => "residency commit",
    MainBatchSubmit => "main batch submit",
    CaptureAlloc => "capture alloc",
    CaptureSubmit => "capture submit",
    CaptureWrite => "capture write",
}

const PHASE_BITS: u32 = 5;
const PHASE_MASK: u64 = (1 << PHASE_BITS) - 1;
const MAX_MS: u64 = u64::MAX >> PHASE_BITS; // Over 18 million years.
const _: () = assert!(PHASE_NAMES.len() <= (1 << PHASE_BITS) && Phase::Idle as u64 == 0);

// Each atomic holds the entire snapshot. The labels are immutable, so no
// separate memory needs to be synchronized with a publication.
struct HostPhase(AtomicU64);

impl HostPhase {
    const fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    fn publish(&self, phase: Phase, now_ms: impl Fn() -> u64) {
        // Resample after a competing write. An idle publication also keeps
        // its timestamp so a delayed writer cannot mistake it for the initial state.
        let _ = self.0.fetch_update(Relaxed, Relaxed, |previous| {
            let at = now_ms().max(previous >> PHASE_BITS);
            Some((at << PHASE_BITS) | phase as u64)
        });
    }

    fn stuck(&self, ms: u64, now_ms: impl FnOnce() -> u64) -> Option<(&'static str, u64, u64)> {
        let publication = self.0.load(Relaxed);
        let phase = (publication & PHASE_MASK) as usize;
        if phase == Phase::Idle as usize {
            return None;
        }
        let age = now_ms().saturating_sub(publication >> PHASE_BITS);
        (age > ms && self.0.load(Relaxed) == publication).then_some((
            PHASE_NAMES[phase],
            age,
            publication,
        ))
    }

    #[cfg(test)]
    fn is_idle(&self) -> bool {
        self.0.load(Relaxed) & PHASE_MASK == Phase::Idle as u64
    }
}

struct HostPhases {
    lanes: [HostPhase; LANES],
    epoch: OnceLock<Instant>,
}

impl HostPhases {
    const fn new() -> Self {
        Self {
            lanes: [const { HostPhase::new() }; LANES],
            epoch: OnceLock::new(),
        }
    }
}

static PHASES: HostPhases = HostPhases::new();

fn timestamp_ms(elapsed_ms: u128) -> u64 {
    elapsed_ms.min(u128::from(MAX_MS - 1)) as u64 + 1
}

fn clock_ms(epoch: &OnceLock<Instant>) -> u64 {
    timestamp_ms(epoch.get_or_init(Instant::now).elapsed().as_millis())
}

/// The phase that has exceeded the limit, its age, and its publication token.
/// A label republished in the same millisecond shares its token and age.
#[cfg(windows)]
pub(super) fn stuck_phase(lane: usize, ms: u64) -> Option<(&'static str, u64, u64)> {
    PHASES.lanes[lane].stuck(ms, || clock_ms(&PHASES.epoch))
}

thread_local! {
    static LANE: Cell<usize> = const { Cell::new(LANE_CAPTURE) };
}

#[cfg(test)]
thread_local! {
    static TEST_PHASES: RefCell<Option<HostPhases>> = const { RefCell::new(None) };
}

// Tests can give this thread fresh lanes and a separate clock epoch.
fn with_phases<T>(f: impl FnOnce(&HostPhases) -> T) -> T {
    #[cfg(test)]
    {
        TEST_PHASES.with_borrow(|phases| f(phases.as_ref().unwrap_or(&PHASES)))
    }
    #[cfg(not(test))]
    {
        f(&PHASES)
    }
}

fn publish(lane: usize, phase: Phase) {
    with_phases(|state| state.lanes[lane].publish(phase, || clock_ms(&state.epoch)));
}

#[cfg(test)]
fn with_test_phases<T>(test: impl FnOnce() -> T) -> T {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            TEST_PHASES.set(None);
        }
    }

    TEST_PHASES.with_borrow_mut(|phases| {
        assert!(phases.is_none());
        *phases = Some(HostPhases::new());
    });
    let _reset = Reset;
    test()
}

#[cfg(test)]
fn is_idle(lane: usize) -> bool {
    with_phases(|state| state.lanes[lane].is_idle())
}

/// This thread's lane until the guard drops, then idle and back to captures.
pub(super) fn lane(lane: usize) -> impl Drop {
    struct Lane(usize);
    impl Drop for Lane {
        fn drop(&mut self) {
            publish(self.0, Phase::Idle);
            LANE.set(LANE_CAPTURE);
        }
    }
    LANE.set(lane);
    Lane(lane)
}

/// Marks this thread's lane as in `phase`.
pub(super) fn here(phase: Phase) {
    publish(LANE.get(), phase);
}

/// After a wait or capture, return the capture lane to idle or the run lane to encoding.
pub(super) fn after_wait() {
    let lane = LANE.get();
    let phase = if lane == LANE_RUN {
        Phase::RunBatchEncode
    } else {
        Phase::Idle
    };
    publish(lane, phase);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::catch_unwind;
    use std::sync::{Barrier, mpsc};
    use std::thread;

    fn label(lane: usize) -> &'static str {
        with_phases(|state| PHASE_NAMES[(state.lanes[lane].0.load(Relaxed) & PHASE_MASK) as usize])
    }

    #[test]
    fn idle_and_strict_timeout() {
        let phase = HostPhase::new();
        assert_eq!(phase.stuck(8_000, || 50_000), None);
        phase.publish(Phase::Commit, || 100);
        assert_eq!(phase.stuck(8_000, || 99), None);
        assert_eq!(phase.stuck(8_000, || 8_099), None);
        assert_eq!(phase.stuck(8_000, || 8_100), None);
        let reported = phase.stuck(8_000, || 8_101).unwrap();
        assert_eq!((reported.0, reported.1), ("commit", 8_001));
        phase.publish(Phase::Idle, || 9_000);
        assert!(phase.is_idle());
        assert_eq!(phase.stuck(8_000, || 50_000), None);
    }

    #[test]
    fn warning_token_rearms_after_idle() {
        let phase = HostPhase::new();
        phase.publish(Phase::Commit, || 100);
        let (_, _, first) = phase.stuck(8_000, || 8_101).unwrap();
        assert_ne!(first, 0);
        assert_eq!(phase.stuck(8_000, || 8_200).unwrap().2, first);

        phase.publish(Phase::Idle, || 8_300);
        assert!(phase.is_idle());
        assert_eq!(phase.stuck(8_000, || 16_301), None);

        phase.publish(Phase::Commit, || 16_400);
        assert_eq!(phase.stuck(8_000, || 24_400), None);
        let (name, age, second) = phase.stuck(8_000, || 24_401).unwrap();
        assert_eq!((name, age), ("commit", 8_001));
        assert_ne!(second, 0);
        assert_ne!(second, first);
    }

    #[test]
    fn older_clock_sample_does_not_move_publication_backwards() {
        let phase = HostPhase::new();
        phase.publish(Phase::Commit, || 100);
        phase.publish(Phase::CaptureWrite, || 90);
        assert_eq!(phase.stuck(0, || 100), None);
        let (name, age, _) = phase.stuck(0, || 101).unwrap();
        assert_eq!((name, age), ("capture write", 1));
    }

    #[test]
    fn real_clock_initializes_once_and_never_moves_backwards() {
        let epoch = OnceLock::new();
        assert!(epoch.get().is_none());
        let first = clock_ms(&epoch);
        assert!((1..=MAX_MS).contains(&first));
        let start = *epoch.get().unwrap();

        let lower = timestamp_ms(start.elapsed().as_millis());
        let next = clock_ms(&epoch);
        let upper = timestamp_ms(start.elapsed().as_millis());
        assert!(next >= first);
        assert!((lower..=upper).contains(&next));
        assert_eq!(epoch.get(), Some(&start));
    }

    #[test]
    fn timestamp_conversion_saturates_without_wrapping() {
        assert_eq!(timestamp_ms(0), 1);
        assert_eq!(timestamp_ms(1), 2);
        let max = u128::from(MAX_MS);
        assert_eq!(timestamp_ms(max - 2), MAX_MS - 1);
        assert_eq!(timestamp_ms(max - 1), MAX_MS);
        assert_eq!(timestamp_ms(max), MAX_MS);
        assert_eq!(timestamp_ms(u128::MAX), MAX_MS);
    }

    #[test]
    fn rapid_transitions_reset_age_and_keep_the_latest_label() {
        let phase = HostPhase::new();
        for at in 1..=2_000 {
            phase.publish(Phase::Commit, || at);
            assert_eq!(phase.stuck(0, || at), None);
            phase.publish(Phase::WaitForCommandBuffer, || at);
            let (label, age, _) = phase.stuck(0, || at + 1).unwrap();
            assert_eq!((label, age), ("waiting for a command buffer", 1));
        }
    }

    #[test]
    fn same_tick_transitions_invalidate_the_previous_snapshot() {
        let phase = HostPhase::new();
        phase.publish(Phase::Commit, || 100);
        let (_, _, first) = phase.stuck(0, || 101).unwrap();
        assert_eq!(
            phase.stuck(0, || {
                phase.publish(Phase::CaptureWrite, || 100);
                101
            }),
            None
        );
        let (label, age, second) = phase.stuck(0, || 101).unwrap();
        assert_eq!((label, age), ("capture write", 1));
        assert_ne!(second, first);
        phase.publish(Phase::CaptureWrite, || 100);
        assert_eq!(phase.stuck(0, || 101).unwrap().2, second);
        phase.publish(Phase::CaptureWrite, || 101);
        assert_ne!(phase.stuck(0, || 102).unwrap().2, second);
        phase.publish(Phase::Idle, || 101);
        assert_eq!(phase.stuck(0, || 102), None);
    }

    #[test]
    fn phase_ids_and_large_timestamps_round_trip() {
        let phase = HostPhase::new();
        for (&id, &name) in ALL_PHASES.iter().zip(PHASE_NAMES).skip(1) {
            phase.publish(id, || MAX_MS - 1);
            let (label, age, _) = phase.stuck(0, || MAX_MS).unwrap();
            assert_eq!((label, age), (name, 1));
        }
    }

    #[test]
    fn delayed_writer_resamples_after_a_concurrent_clear() {
        let phase = HostPhase::new();
        let shared = &phase;
        let (sampled_tx, sampled_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        thread::scope(|scope| {
            scope.spawn(move || {
                let first_attempt = Cell::new(true);
                shared.publish(Phase::Commit, || {
                    if first_attempt.replace(false) {
                        sampled_tx.send(()).unwrap();
                        resume_rx.recv().unwrap();
                        10
                    } else {
                        30
                    }
                });
            });
            sampled_rx.recv().unwrap();
            phase.publish(Phase::CaptureAlloc, || 20);
            phase.publish(Phase::Idle, || 21);
            resume_tx.send(()).unwrap();
        });
        let (label, age, _) = phase.stuck(0, || 31).unwrap();
        assert_eq!((label, age), ("commit", 1));
    }

    #[test]
    fn concurrent_readers_only_see_coherent_publications() {
        const ROUNDS: u64 = 256;
        const OBSERVED_AT: u64 = ROUNDS + 1;
        let phase = &HostPhase::new();
        let barrier = &Barrier::new(4);
        let writers = [
            (Phase::Commit, Phase::RunBatch),
            (Phase::CaptureWrite, Phase::WaitForCommandBuffer),
        ];
        let matches = |label, at| match at % 4 {
            0 => false,
            2 => matches!(label, "commit" | "capture write"),
            _ => matches!(label, "run_batch" | "waiting for a command buffer"),
        };
        thread::scope(|scope| {
            for (even, odd) in writers {
                scope.spawn(move || {
                    for at in 1..=ROUNDS {
                        barrier.wait();
                        let next = match at % 4 {
                            0 => Phase::Idle,
                            2 => even,
                            _ => odd,
                        };
                        phase.publish(next, || at);
                        barrier.wait();
                    }
                });
            }
            for _ in 0..2 {
                scope.spawn(move || {
                    let mut coherent = true;
                    let mut reports = 0;
                    for at in 1..=ROUNDS {
                        barrier.wait();
                        for _ in 0..4 {
                            if let Some((label, age, _)) = phase.stuck(0, || OBSERVED_AT) {
                                let started = OBSERVED_AT - age;
                                coherent &=
                                    (started == at || started + 1 == at) && matches(label, started);
                            }
                        }
                        barrier.wait();
                        if let Some((label, age, _)) = phase.stuck(0, || OBSERVED_AT) {
                            reports += 1;
                            coherent &= OBSERVED_AT - age == at && matches(label, at);
                        } else {
                            coherent &= at.is_multiple_of(4) && phase.is_idle();
                        }
                    }
                    assert!(coherent);
                    assert_eq!(reports, ROUNDS - ROUNDS / 4);
                });
            }
        });
    }

    #[test]
    fn lane_guard_and_after_wait_keep_run_and_capture_separate() {
        with_test_phases(|| {
            here(Phase::CaptureAlloc);
            assert_eq!(label(LANE_CAPTURE), "capture alloc");
            assert!(is_idle(LANE_RUN));
            {
                let _run = lane(LANE_RUN);
                here(Phase::RunBatch);
                assert_eq!(label(LANE_RUN), "run_batch");
                after_wait();
                assert_eq!(label(LANE_RUN), "run_batch encode");
                assert_eq!(label(LANE_CAPTURE), "capture alloc");
            }
            assert!(is_idle(LANE_RUN), "run lane still active after guard drop");
            here(Phase::CaptureWrite);
            assert_eq!(label(LANE_CAPTURE), "capture write");
            after_wait();
            assert!(
                is_idle(LANE_CAPTURE),
                "capture lane still active after wait"
            );
        });
    }

    #[test]
    fn lane_guard_clears_run_and_restores_capture_during_unwind() {
        with_test_phases(|| {
            let panic = catch_unwind(|| {
                let _run = lane(LANE_RUN);
                here(Phase::RunBatch);
                assert_eq!(label(LANE_RUN), "run_batch");
                panic!("test unwind");
            })
            .unwrap_err();
            assert_eq!(panic.downcast_ref::<&str>(), Some(&"test unwind"));
            assert!(is_idle(LANE_RUN), "run lane still active after unwind");
            here(Phase::CaptureWrite);
            assert_eq!(label(LANE_CAPTURE), "capture write");
            assert!(is_idle(LANE_RUN));
        });
    }
}
