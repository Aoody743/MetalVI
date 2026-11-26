//! Shared event smoke test (run in Wine with victorymetal installed):
//! create an event, signal it from the CPU, encode a GPU wait and signal on
//! a command buffer, and read the values back on both sides.

#[cfg(windows)]
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => {
            println!("RESULT: ALL CHECKS PASSED");
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            println!("RESULT: FAILED: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(windows)]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use victory_metal::events::SharedEvent;
    use victory_metal::{Device, Pool};

    let _pool = Pool::new();
    let device = Device::first()?;
    println!("device: {}", device.name());

    let event = SharedEvent::new(&device)?;
    if event.signaled_value() != 0 {
        return Err(format!("fresh event at {}", event.signaled_value()).into());
    }
    println!("fresh event at 0");

    event.signal(7);
    if event.signaled_value() != 7 {
        return Err(format!("CPU signal 7, got {}", event.signaled_value()).into());
    }
    println!("CPU signaled 7");

    let queue = device.new_queue(4)?;
    let cb = queue.command_buffer()?;
    // GPU waits for the CPU's 7, then signals 12 on the same event.
    cb.encode_wait_for_event(&event, 7);
    cb.encode_signal_event(&event, 12);
    cb.commit();
    cb.wait()?;

    if event.signaled_value() != 12 {
        return Err(format!("GPU signal 12, got {}", event.signaled_value()).into());
    }
    println!("GPU signaled 12");

    if !event.wait_until_signaled(12, 1000) {
        return Err("waitUntilSignaledValue(12) timed out".into());
    }
    println!("waitUntilSignaledValue(12) returned true");
    Ok(())
}

#[cfg(not(windows))]
fn main() {
    eprintln!("victory-metal-events runs on Windows (Wine) only");
}
