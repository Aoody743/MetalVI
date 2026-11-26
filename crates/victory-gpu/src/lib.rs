//! D3D12.X on host D3D12.
//!
//! D3D12.X records commands from code inlined into the title, which writes
//! packets into the command list while its put pointer is below a limit and
//! calls the list's method otherwise. Our lists keep the limits at zero, so
//! every command arrives here as a call, to be run on host D3D12. The
//! title's GPU addresses are CPU addresses (console memory is unified).
//!
//! [`device`] is the device object titles create, [`dxgi`] its adapter side,
//! [`resource`] heaps and resources, [`fence`] fences, [`queue`] command
//! queues, [`list`] command allocators and lists, [`descriptor`]
//! descriptor heaps, [`root`] root signatures, [`query`] query heaps,
//! [`signature`] command signatures, [`pipeline`] pipeline states,
//! [`frame`] frame pacing, [`object`] what all console graphics objects
//! share, [`console`] which console model the title sees. Planned:
//! `present` (swap chain).

#[cfg(all(windows, feature = "diagnostics"))]
mod capture;
pub(crate) mod backend;
pub mod console;
#[cfg(all(windows, feature = "diagnostics"))]
mod copystat;
pub mod descriptor;
pub mod device;
#[cfg(any(windows, test))]
mod dispatch_dims;
#[cfg(windows)]
mod draw;
pub mod dxgi;
pub mod fence;
pub mod frame;
#[cfg(windows)]
mod fx;
#[cfg(windows)]
mod host;
#[cfg(windows)]
mod indirect;
#[cfg(windows)]
mod inplace;
pub mod list;
pub mod object;
pub mod pipeline;
#[cfg(windows)]
mod primitive;
pub mod query;
pub mod queue;
pub mod resource;
pub mod root;
#[cfg(windows)]
mod sampler;
#[cfg(windows)]
pub mod shader_cache;
#[cfg(windows)]
mod memory;
#[cfg(windows)]
mod shadow;
pub mod signature;
#[cfg(windows)]
mod texture;
#[cfg(windows)]
mod watch;

#[cfg(windows)]
pub use host::on_failure;

#[cfg(all(windows, feature = "diagnostics"))]
pub fn present_count() -> u64 {
    queue::presented()
}

#[cfg(all(windows, feature = "diagnostics"))]
pub fn stall_locks() -> Vec<(&'static str, usize)> {
    let mut locks = vec![
        ("shadow.BUFFERS", shadow::buffers_lock_address()),
        ("watch.GUARDS", watch::guards_lock_address()),
    ];
    locks.extend(backend::metal::stall_locks());
    locks
}

/// Set by the runtime: notes our own writes into title memory (address,
/// bytes, kind: 1 a bottom-of-pipe write) for its crash reports
/// (`victory_loader::writelog`).
pub static WRITE_NOTE: std::sync::OnceLock<fn(usize, usize, u8)> = std::sync::OnceLock::new();

pub(crate) fn note_write(at: usize, len: usize, kind: u8) {
    if let Some(note) = WRITE_NOTE.get() {
        note(at, len, kind);
    }
}
#[cfg(windows)]
pub use inplace::{title_memory_freeing, title_memory_section, title_memory_view};
#[cfg(windows)]
pub use watch::{
    Remapping, fault_tracking, note_memory_changed, title_memory_reading, title_memory_reading_range,
    title_memory_remapped, title_memory_remapping, title_memory_remapping_range,
    writing as kernel_will_write,
};

/// A whole file, read without asking for its size or other metadata.
/// `std::fs::read` and `read_to_end` call `File::metadata()`, which Wine
/// answers with `NtQueryVolumeInformationFile`: an ioctl to its mount
/// manager whose completion runs as a system APC inside Wine's virtual
/// memory lock. Under Rosetta a thread was paused there by SIGUSR1 with the
/// lock held, and every other thread then waited on it for good: the game
/// froze right after a pipeline worker looked up a cached shader. From
/// VIctory for macOS (pragyaangaur/VIctory-macOS #72, GPL-3.0).
pub fn read_file(path: &std::path::Path) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut out = Vec::new();
    let mut chunk = vec![0u8; 1 << 16];
    loop {
        match file.read(&mut chunk) {
            Ok(0) => return Ok(out),
            Ok(n) => out.extend_from_slice(&chunk[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
}

/// Makes a folder without asking for an existing one's metadata (which
/// `create_dir_all` does, see [`read_file`]). VIctory for macOS #75.
pub fn make_dir(dir: &std::path::Path) -> std::io::Result<()> {
    match std::fs::create_dir(dir) {
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => std::fs::create_dir_all(dir),
        made => made,
    }
}

#[cfg(test)]
mod read_file_tests {
    #[test]
    fn whole_files_come_back_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        for len in [0usize, 1, 65_535, 65_536, 65_537, 300_000] {
            let path = dir.path().join(format!("f{len}"));
            let data: Vec<u8> = (0..len).map(|i| (i * 7) as u8).collect();
            std::fs::write(&path, &data).unwrap();
            assert_eq!(super::read_file(&path).unwrap(), data, "{len} bytes");
        }
        assert!(super::read_file(&dir.path().join("missing")).is_err());
        let sub = dir.path().join("a").join("b");
        super::make_dir(&sub).unwrap();
        super::make_dir(&sub).unwrap();
        assert!(sub.is_dir());
    }
}
