use super::{MetalOp, complete_checked, fuse, work};
use std::sync::{Arc, Condvar, Mutex, OnceLock, mpsc};
use victory_metal::{CommandBuffer, Error, Result};

#[cfg(test)]
pub(super) static INJECT_TIMEOUT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

struct Job {
    _permit: Permit<'static>,
    cb: CommandBuffer,
    ops: Vec<Arc<MetalOp>>,
    work: bool,
    label: &'static str,
    result: mpsc::Sender<Result<()>>,
}

struct Admission {
    count: u32,
    time: u64,
    used: Mutex<(u32, u64)>,
    changed: Condvar,
}

struct Permit<'a>(&'a Admission, u64);
static QUEUE: Admission = Admission::new(1, work::WINDOW_NS);

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        let mut used = self.0.used.lock().unwrap_or_else(|e| e.into_inner());
        used.0 -= 1;
        used.1 -= self.1;
        self.0.changed.notify_all();
    }
}

impl Admission {
    const fn new(count: u32, time: u64) -> Self {
        Self { count, time, used: Mutex::new((0, 0)), changed: Condvar::new() }
    }

    fn acquire(&self, ns: u64, mut on_wait: impl FnMut()) -> Result<Permit<'_>> {
        // An over-window buffer reserves the whole window and runs alone.
        let ns = ns.min(self.time);
        let mut used = self.used.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if fuse::tripped() { return Err(Error("Metal safety stop already tripped".into())); }
            if used.0 < self.count && ns <= self.time - used.1 {
                used.0 += 1;
                used.1 += ns;
                return Ok(Permit(self, ns));
            }
            on_wait();
            used = self.changed.wait(used).unwrap_or_else(|e| e.into_inner());
        }
    }
}

fn stopped(why: &str) -> Error {
    fuse::stop();
    tracing::error!(reason = why, "Metal safety stop: completion service unavailable");
    Error(why.into())
}

fn sender() -> Result<&'static mpsc::SyncSender<Job>> {
    static SENDER: OnceLock<std::result::Result<mpsc::SyncSender<Job>, String>> = OnceLock::new();
    SENDER.get_or_init(|| {
        // No backlog of committed work. All runtime submissions use this queue
        // until ring ownership and the caller's fence contract are asynchronous.
        let (tx, rx) = mpsc::sync_channel::<Job>(0);
        std::thread::Builder::new().name("metal-completion".into()).spawn(move || {
            for job in rx {
                let _pool = victory_metal::Pool::new();
                let result = complete_checked(&job.cb, &job.ops, job.work, job.label);
                let _ = job.result.send(result);
            }
        }).map_err(|e| e.to_string())?;
        Ok(tx)
    }).as_ref().map_err(|e| stopped(e))
}

pub(super) fn enqueue(cb: &CommandBuffer, ops: &[Arc<MetalOp>], work: bool, label: &'static str) -> Result<mpsc::Receiver<Result<()>>> {
    if fuse::tripped() {
        return Err(Error("Metal safety stop already tripped".into()));
    }
    let (result, ticket) = mpsc::channel();
    let permit = QUEUE.acquire(work::reservation(ops), || {})?;
    sender()?.send(Job { _permit: permit, cb: cb.clone(), ops: ops.to_vec(), work, label, result })
        .map_err(|_| stopped("completion worker disconnected"))?;
    Ok(ticket)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn admission_blocks_on_count_and_time() {
        for (capacity, first, next) in [(1, 1, 1), (2, 40, 11), (2, 51, 1), (2, 1, 51)] {
            let gate = Admission::new(capacity, 50);
            let held = gate.acquire(first, || unreachable!()).unwrap();
            std::thread::scope(|s| {
                let (tx, rx) = mpsc::channel();
                let gate = &gate;
                let handle = s.spawn(move || {
                    let _permit = gate.acquire(next, || tx.send("waiting").unwrap()).unwrap();
                    tx.send("admitted").unwrap();
                });
                assert_eq!(rx.recv_timeout(Duration::from_secs(3)).unwrap(), "waiting");
                assert!(rx.try_recv().is_err());
                drop(held);
                handle.join().unwrap();
                assert!(rx.try_iter().any(|s| s == "admitted"));
            });
        }
    }

    #[test]
    #[ignore = "trips process-wide fuse; run alone"]
    fn completion_error_refuses_next_commit() {
        failure_probe(true);
    }

    #[test]
    #[ignore = "trips process-wide fuse; run alone"]
    fn measured_duration_refuses_next_commit() {
        failure_probe(false);
    }

    fn failure_probe(inject_error: bool) {
        let _pool = victory_metal::Pool::new();
        let dev = victory_metal::Device::first().unwrap();
        let queue = dev.new_queue(1).unwrap();
        let dst = dev.shared_buffer(&[0; 16]).unwrap();
        let cb = queue.command_buffer().unwrap();
        let mut blit = cb.blit_encoder().unwrap();
        blit.fill_buffer(&dst, 0, 16, 42);
        blit.finish();
        assert_eq!(cb.encoded_work().blit_bytes, 16);
        INJECT_TIMEOUT.store(inject_error, std::sync::atomic::Ordering::Release);
        fuse::TEST_ONE_NS.store(!inject_error, std::sync::atomic::Ordering::Release);
        let result = enqueue(&cb, &[], true, "injected completion error").unwrap();
        let error = result.recv_timeout(Duration::from_secs(5)).unwrap().unwrap_err();
        assert!(error.0.contains(if inject_error { "injected system timeout" } else { "unsafe GPU duration" }));
        if !inject_error { assert!(cb.gpu_time_ns() > 1); }
        let next = queue.command_buffer().unwrap();
        assert!(submit(&next, &[], false, "after timeout").is_err());
        assert_eq!(unsafe { victory_metal::sys::MTLCommandBuffer_status(next.handle()) },
            victory_metal::sys::WMTCommandBufferStatus::WMTCommandBufferStatusNotEnqueued);
    }
}

pub(super) fn submit(cb: &CommandBuffer, ops: &[Arc<MetalOp>], work: bool, label: &'static str) -> Result<()> {
    static SYNC: OnceLock<bool> = OnceLock::new();
    if *SYNC.get_or_init(|| std::env::var("VICTORY_METAL_SYNC").as_deref() == Ok("1")) {
        let _permit = QUEUE.acquire(work::reservation(ops), || {})?;
        return complete_checked(cb, ops, work, label);
    }
    enqueue(cb, ops, work, label)?.recv().map_err(|_| stopped("completion worker lost result"))?
}
