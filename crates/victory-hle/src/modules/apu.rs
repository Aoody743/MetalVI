//! The console's audio decoder (`xapu.dll`): Opus packets in, float PCM out.
//!
//! A title connects once for one kind of processing, gets a block of memory
//! for its packets and output, and then queues per-stream commands
//! (activate, process, reset, deactivate) and dequeues one result per
//! command. Commands are decoded on a dedicated worker thread so the caller
//! is not blocked inside Opus decode; the order of results is the order of
//! commands.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::Layout;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use opus_pure::OpusDecoder;
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
use windows_sys::Win32::System::Threading::{CreateEventW, SetEvent};

use super::xasync::E_INVALIDARG;
use super::xgameruntime::{HRESULT, S_OK};
use crate::{HleModule, hle_export};

const E_NOTIMPL: HRESULT = 0x8000_4001_u32 as i32;
const E_OUTOFMEMORY: HRESULT = 0x8007_000E_u32 as i32;
const QUEUE_FULL: HRESULT = 0x8AC8_1000_u32 as i32;
/// No result yet, which titles poll for.
pub(crate) const PENDING_RESULTS: HRESULT = 0x8AC8_1001_u32 as i32;
const INVALID_COMMAND: HRESULT = 0x8AC8_1002_u32 as i32;
const INVALID_STREAM: HRESULT = 0x8AC8_1003_u32 as i32;
const ALREADY_ACTIVE: HRESULT = 0x8AC8_1005_u32 as i32;
const BAD_PACKET: HRESULT = 0x8AC8_1008_u32 as i32;

/// Processing kinds; only plain decoding is implemented.
const DECODE_OPUS: u32 = 1;

const ACTIVATE: u32 = 1;
const DEACTIVATE: u32 = 2;
const PROCESS: u32 = 3;
const NOTIFY: u32 = 4;
const RESET: u32 = 5;

/// Command option: signal the connection's event when the result is ready.
const SIGNAL: u32 = 1;
/// Connect options.
const DEINTERLEAVED_OUTPUT: u32 = 4;
const RESULT_EVENT: u32 = 8;

/// The longest Opus packet duration at 48 kHz (120 ms).
const MAX_FRAMES: usize = 5760;

#[repr(C)]
pub struct ConnectInput {
    process: u32,
    streams: u32,
    commands_per_stream: u32,
    memory: u32,
    frames_per_output: u32,
    options: u32,
}

#[repr(C)]
pub struct ConnectOutput {
    memory: *mut c_void,
    result_event: HANDLE,
}

/// What every command starts with, and what its result echoes.
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct CommandId {
    kind: u32,
    stream: u32,
    sequence: u32,
    options: u32,
    context: u64,
}

#[repr(C, packed)]
struct Activate {
    id: CommandId,
    channels: u32,
    init_length: u32,
    init: *const c_void,
    start_pitch: f32,
    processing_length: u32,
    processing: *mut c_void,
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct Process {
    id: CommandId,
    pitch_ramp: f32,
    target_pitch: f32,
    first_frame: u32,
    frame_count: u32,
    input_length: u32,
    input: *const u8,
    output_length: u32,
    output: *mut u8,
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct ApuResult {
    id: CommandId,
    state: u32,
    state_data: u64,
    usec: u64,
    error: HRESULT,
    frames: u32,
    output_length: u32,
    output: *mut u8,
    pitch: f32,
    input_length: u32,
    input: *const u8,
}

// SAFETY: the raw pointers are title addresses carried through, never
// dereferenced after the result is made.
unsafe impl Send for ApuResult {}

enum JobKind {
    Activate {
        channels: u32,
    },
    Deactivate,
    Reset,
    Notify,
    Process {
        first_frame: u32,
        frame_count: u32,
        input: *const u8,
        input_length: u32,
        packet: Vec<u8>,
        output: *mut u8,
        output_length: u32,
    },
}

struct DecodeJob {
    id: CommandId,
    kind: JobKind,
}

// SAFETY: `packet` owns a copy of the input bytes; `output` points into the
// connection's memory block which is kept alive until the worker joins.
unsafe impl Send for DecodeJob {}

pub struct Connection {
    inner: Arc<Inner>,
    worker: Option<JoinHandle<()>>,
}

struct Inner {
    memory: *mut u8,
    layout: Layout,
    event: HANDLE,
    options: u32,
    per_stream: u32,
    stream_count: usize,
    state: Mutex<State>,
    wake: Condvar,
}

// SAFETY: `memory` and `event` are freed only in `Drop for Inner` after
// `XApuDisconnect` joins the worker thread.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl Drop for Inner {
    fn drop(&mut self) {
        if !self.event.is_null() {
            // SAFETY: our event created in `XApuConnect`.
            unsafe { CloseHandle(self.event) };
        }
        // SAFETY: allocated in `XApuConnect` with this layout.
        unsafe { std::alloc::dealloc(self.memory, self.layout) };
    }
}

struct State {
    jobs: VecDeque<DecodeJob>,
    /// Results not yet dequeued, per stream.
    queued: Vec<u32>,
    results: VecDeque<ApuResult>,
    commands: u32,
    dequeued: u32,
    stop: bool,
}

fn lock(inner: &Inner) -> std::sync::MutexGuard<'_, State> {
    inner.state.lock().unwrap_or_else(|e| e.into_inner())
}

pub extern "system" fn XApuConnect(
    input: *const ConnectInput,
    output: *mut ConnectOutput,
    handle: *mut *mut Connection,
) -> HRESULT {
    if input.is_null() || output.is_null() || handle.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null.
    let input = unsafe { &*input };
    if !(1..=500).contains(&input.streams)
        || !(1..=16).contains(&input.commands_per_stream)
        || input.memory == 0
    {
        return E_INVALIDARG;
    }
    if input.process != DECODE_OPUS {
        // shortcut: rate conversion (decode-convert, PCM convert) is not
        // implemented; add a resampler if a title connects for it.
        tracing::warn!(
            process = input.process,
            "audio decoder: processing kind not implemented"
        );
        return E_NOTIMPL;
    }
    if input.frames_per_output != 0 {
        return E_INVALIDARG;
    }
    let Ok(layout) = Layout::from_size_align(input.memory as usize, 16) else {
        return E_INVALIDARG;
    };
    // SAFETY: non-zero size.
    let memory = unsafe { std::alloc::alloc_zeroed(layout) };
    if memory.is_null() {
        return E_OUTOFMEMORY;
    }
    let event = if input.options & RESULT_EVENT != 0 {
        // SAFETY: plain auto-reset event, no name.
        unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) }
    } else {
        std::ptr::null_mut()
    };
    let streams = input.streams as usize;
    tracing::info!(
        streams,
        memory = input.memory,
        options = input.options,
        "audio decoder connected"
    );
    let inner = Arc::new(Inner {
        memory,
        layout,
        event,
        options: input.options,
        per_stream: input.commands_per_stream,
        stream_count: streams,
        state: Mutex::new(State {
            jobs: VecDeque::new(),
            queued: vec![0; streams],
            results: VecDeque::new(),
            commands: 0,
            dequeued: 0,
            stop: false,
        }),
        wake: Condvar::new(),
    });
    let worker = std::thread::Builder::new()
        .name("victory APU Opus decoder".into())
        .spawn({
            let inner = Arc::clone(&inner);
            move || apu_worker(inner)
        })
        .ok();
    let connection = Box::new(Connection { inner, worker });
    // SAFETY: checked non-null.
    unsafe {
        output.write(ConnectOutput {
            memory: memory.cast(),
            result_event: event,
        });
        handle.write(Box::into_raw(connection));
    }
    S_OK
}

pub extern "system" fn XApuDisconnect(handle: *mut Connection) -> HRESULT {
    if handle.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: made by `XApuConnect`; disconnected once.
    let mut connection = unsafe { Box::from_raw(handle) };
    {
        let mut state = lock(&connection.inner);
        state.stop = true;
        connection.inner.wake.notify_all();
    }
    if let Some(worker) = connection.worker.take() {
        let _ = worker.join();
    }
    drop(connection);
    S_OK
}

fn apu_worker(inner: Arc<Inner>) {
    let mut decoders: Vec<Option<OpusDecoder>> = (0..inner.stream_count).map(|_| None).collect();
    let mut pcm = vec![0f32; MAX_FRAMES * 2];
    loop {
        let job = {
            let mut state = lock(&inner);
            while state.jobs.is_empty() && !state.stop {
                state = inner
                    .wake
                    .wait(state)
                    .unwrap_or_else(|e| e.into_inner());
            }
            if state.stop && state.jobs.is_empty() {
                return;
            }
            state.jobs.pop_front()
        };
        let Some(job) = job else { continue };
        let stream = job.id.stream as usize;
        let mut result = ApuResult {
            id: job.id,
            state: 0,
            state_data: 0,
            usec: 0,
            error: S_OK,
            frames: 0,
            output_length: 0,
            output: std::ptr::null_mut(),
            pitch: 1.0,
            input_length: 0,
            input: std::ptr::null(),
        };
        let decoder = &mut decoders[stream];
        match job.kind {
            JobKind::Activate { channels } => {
                tracing::debug!(stream, channels, "Opus stream activated");
                result.error = if decoder.is_some() {
                    ALREADY_ACTIVE
                } else {
                    match OpusDecoder::new(48_000, channels as usize) {
                        Ok(d) if (1..=2).contains(&channels) => {
                            *decoder = Some(d);
                            S_OK
                        }
                        _ => E_INVALIDARG,
                    }
                };
            }
            JobKind::Deactivate => *decoder = None,
            JobKind::Reset => {
                if let Some(d) = decoder {
                    let _ = d.reset_state();
                }
            }
            JobKind::Notify => {}
            JobKind::Process {
                first_frame,
                frame_count,
                input,
                input_length,
                packet,
                output,
                output_length,
            } => match decoder {
                Some(d) => process_packet(
                    d,
                    inner.options,
                    first_frame,
                    frame_count,
                    input,
                    input_length,
                    &packet,
                    output,
                    output_length,
                    &mut pcm,
                    &mut result,
                ),
                None => result.error = INVALID_STREAM,
            },
        }
        let signal = job.id.options & SIGNAL != 0 && !inner.event.is_null();
        {
            let mut state = lock(&inner);
            state.results.push_back(result);
        }
        if signal {
            // SAFETY: our event, kept alive until worker exits.
            unsafe { SetEvent(inner.event) };
        }
    }
}

pub extern "system" fn XApuEnqueueCommand(
    handle: *mut Connection,
    command: *const CommandId,
    total: *mut u32,
) -> HRESULT {
    if handle.is_null() || command.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: a connection from `XApuConnect`; every command starts with an id.
    let (connection, id) = unsafe { (&*handle, command.read_unaligned()) };
    let stream = id.stream as usize;
    let kind = match id.kind {
        ACTIVATE => {
            // SAFETY: an activate command is this struct.
            let channels = unsafe { command.cast::<Activate>().read_unaligned() }.channels;
            JobKind::Activate { channels }
        }
        DEACTIVATE => JobKind::Deactivate,
        RESET => JobKind::Reset,
        NOTIFY => JobKind::Notify,
        PROCESS => {
            // SAFETY: a process command is this struct.
            let p = unsafe { command.cast::<Process>().read_unaligned() };
            let packet = if !p.input.is_null() && p.input_length > 0 {
                // SAFETY: caller's packet bytes in connection memory.
                unsafe { std::slice::from_raw_parts(p.input, p.input_length as usize) }.to_vec()
            } else {
                Vec::new()
            };
            JobKind::Process {
                first_frame: p.first_frame,
                frame_count: p.frame_count,
                input: p.input,
                input_length: p.input_length,
                packet,
                output: p.output,
                output_length: p.output_length,
            }
        }
        _ => return INVALID_COMMAND,
    };
    let mut state = lock(&connection.inner);
    if stream >= connection.inner.stream_count {
        return INVALID_STREAM;
    }
    if state.queued[stream] >= connection.inner.per_stream {
        return QUEUE_FULL;
    }
    state.queued[stream] += 1;
    state.commands = state.commands.wrapping_add(1);
    state.jobs.push_back(DecodeJob { id, kind });
    if !total.is_null() {
        // SAFETY: optional out parameter.
        unsafe { total.write(state.commands) };
    }
    connection.inner.wake.notify_one();
    S_OK
}

/// Decodes one packet into the command's output buffer, keeping the frames
/// from `first_frame` on, at most `frame_count` of them.
#[allow(clippy::too_many_arguments)]
fn process_packet(
    decoder: &mut OpusDecoder,
    options: u32,
    first_frame: u32,
    frame_count: u32,
    input: *const u8,
    input_length: u32,
    packet: &[u8],
    output: *mut u8,
    output_length: u32,
    pcm: &mut [f32],
    result: &mut ApuResult,
) {
    let channels = decoder.channels();
    result.input = input;
    result.input_length = input_length;
    result.output = output;
    if input.is_null() || output.is_null() || input_length == 0 {
        result.error = E_INVALIDARG;
        return;
    }
    let pcm = &mut pcm[..MAX_FRAMES * channels];
    let decoded = match decoder.decode(packet, MAX_FRAMES, pcm) {
        Ok(n) => n,
        Err(_) => {
            result.error = BAD_PACKET;
            return;
        }
    };
    let first = (first_frame as usize).min(decoded);
    let frames = (decoded - first)
        .min(frame_count as usize)
        .min(output_length as usize / (4 * channels));
    let kept = &pcm[first * channels..(first + frames) * channels];
    // SAFETY: the title's output buffer holds `output_length` bytes.
    let out = unsafe { std::slice::from_raw_parts_mut(output.cast::<f32>(), frames * channels) };
    if options & DEINTERLEAVED_OUTPUT != 0 && channels == 2 {
        for (i, frame) in kept.as_chunks::<2>().0.iter().enumerate() {
            (out[i], out[frames + i]) = (frame[0], frame[1]);
        }
    } else {
        out.copy_from_slice(kept);
    }
    result.frames = frames as u32;
    result.output_length = (frames * channels * 4) as u32;
}

pub extern "system" fn XApuDequeueResult(handle: *mut Connection, out: *mut ApuResult) -> HRESULT {
    if handle.is_null() || out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: a connection from `XApuConnect`.
    let mut state = lock(&unsafe { &*handle }.inner);
    let Some(result) = state.results.pop_front() else {
        return PENDING_RESULTS;
    };
    let stream = result.id.stream as usize;
    state.queued[stream] -= 1;
    state.dequeued = state.dequeued.wrapping_add(1);
    // SAFETY: the title passes room for one result.
    unsafe { out.write_unaligned(result) };
    S_OK
}

pub extern "system" fn XApuGetResultCount(handle: *mut Connection, total: *mut u32) -> u32 {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: a connection from `XApuConnect`.
    let state = lock(&unsafe { &*handle }.inner);
    if !total.is_null() {
        // SAFETY: optional out parameter.
        unsafe { total.write(state.dequeued) };
    }
    state.results.len() as u32
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "apu",
        dlls: &["xapu.dll"],
        exports: vec![
            hle_export!(XApuConnect),
            hle_export!(XApuDisconnect),
            hle_export!(XApuEnqueueCommand),
            hle_export!(XApuDequeueResult),
            hle_export!(XApuGetResultCount),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opus_pure::{Application, OpusEncoder};

    fn id(kind: u32, stream: u32) -> CommandId {
        CommandId {
            kind,
            stream,
            sequence: 7,
            options: 0,
            context: 0x1234,
        }
    }

    #[test]
    fn decodes_opus_as_the_title_drives_it() {
        assert_eq!(size_of::<ApuResult>(), 0x50);
        // 256 mono/stereo streams, two commands each, as the title connects.
        let input = ConnectInput {
            process: DECODE_OPUS,
            streams: 256,
            commands_per_stream: 2,
            memory: 0x10_4000,
            frames_per_output: 0,
            options: 1,
        };
        let mut output = ConnectOutput {
            memory: std::ptr::null_mut(),
            result_event: std::ptr::null_mut(),
        };
        let mut handle = std::ptr::null_mut();
        assert_eq!(XApuConnect(&input, &mut output, &mut handle), S_OK);
        let base = output.memory.cast::<u8>();

        let activate = Activate {
            id: id(ACTIVATE, 3),
            channels: 2,
            init_length: 0,
            init: std::ptr::null(),
            start_pitch: 1.0,
            processing_length: 0,
            processing: std::ptr::null_mut(),
        };
        assert_eq!(
            XApuEnqueueCommand(handle, (&raw const activate).cast(), std::ptr::null_mut()),
            S_OK
        );

        // A 20 ms stereo packet of a 440 Hz tone.
        let tone: Vec<f32> = (0..960 * 2)
            .map(|i| ((i / 2) as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 0.5)
            .collect();
        let mut encoder = OpusEncoder::new(48_000, 2, Application::Audio).unwrap();
        let mut packet = [0u8; 1500];
        let n = encoder.encode(&tone, 960, &mut packet).unwrap();
        // SAFETY: inside the connection memory.
        unsafe { std::ptr::copy_nonoverlapping(packet.as_ptr(), base, n) };
        let process = Process {
            id: id(PROCESS, 3),
            pitch_ramp: 0.0,
            target_pitch: 1.0,
            first_frame: 0,
            frame_count: 960,
            input_length: n as u32,
            input: base,
            output_length: 0xf00 * 2,
            output: base.wrapping_add(0x14000),
        };
        let mut total = 0;
        assert_eq!(
            XApuEnqueueCommand(handle, (&raw const process).cast(), &mut total),
            S_OK
        );
        assert_eq!(total, 2);
        // Two commands queued for stream 3: its queue is full.
        assert_eq!(
            XApuEnqueueCommand(handle, (&raw const process).cast(), &mut total),
            QUEUE_FULL
        );
        let wait_start = std::time::Instant::now();
        while XApuGetResultCount(handle, std::ptr::null_mut()) < 2
            && wait_start.elapsed() < std::time::Duration::from_secs(2)
        {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(XApuGetResultCount(handle, std::ptr::null_mut()), 2);

        let mut result = std::mem::MaybeUninit::<ApuResult>::zeroed();
        assert_eq!(XApuDequeueResult(handle, result.as_mut_ptr()), S_OK);
        // SAFETY: written above.
        let first = unsafe { result.assume_init() };
        let (kind, error, context) = (first.id.kind, first.error, first.id.context);
        assert_eq!((kind, error, context), (ACTIVATE, S_OK, 0x1234));
        assert_eq!(XApuDequeueResult(handle, result.as_mut_ptr()), S_OK);
        // SAFETY: written above.
        let decoded = unsafe { result.assume_init() };
        let (kind, error, frames, length) = (
            decoded.id.kind,
            decoded.error,
            decoded.frames,
            decoded.output_length,
        );
        assert_eq!((kind, error, frames, length), (PROCESS, S_OK, 960, 960 * 8));
        // SAFETY: the decoder wrote 960 stereo frames there.
        let pcm =
            unsafe { std::slice::from_raw_parts(base.wrapping_add(0x14000).cast::<f32>(), 1920) };
        let energy: f32 = pcm.iter().map(|s| s * s).sum::<f32>() / 1920.0;
        assert!(energy > 0.01, "decoded a tone, energy {energy}");
        assert_eq!(
            XApuDequeueResult(handle, result.as_mut_ptr()),
            PENDING_RESULTS
        );
        let mut dequeued = 0;
        assert_eq!(XApuGetResultCount(handle, &mut dequeued), 0);
        assert_eq!(dequeued, 2);

        // A process on a stream that was never activated fails in its result.
        let mut stray = process;
        stray.id.stream = 4;
        assert_eq!(
            XApuEnqueueCommand(handle, (&raw const stray).cast(), &mut total),
            S_OK
        );
        let wait_start = std::time::Instant::now();
        while XApuGetResultCount(handle, std::ptr::null_mut()) < 1
            && wait_start.elapsed() < std::time::Duration::from_secs(2)
        {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(XApuDequeueResult(handle, result.as_mut_ptr()), S_OK);
        // SAFETY: written above.
        let error = unsafe { result.assume_init() }.error;
        assert_eq!(error, INVALID_STREAM);
        assert_eq!(XApuDisconnect(handle), S_OK);
    }
}
