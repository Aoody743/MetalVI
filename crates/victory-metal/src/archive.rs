//! A pipeline binary archive (`MTLBinaryArchive`, PLAN-NATIVE-METAL P8 / G.1):
//! GPU binaries of pipelines made in one run are kept on disk and looked up
//! in the next, so a cold start does not compile every pipeline again.
//!
//! Every pipeline this crate makes goes through [`create`]: with an archive
//! installed ([`install`]) it is first asked for with
//! `MTLPipelineOptionFailOnBinaryArchiveMiss` (a hit: no compile); a miss
//! makes it again with the archive as the serialization target, which
//! compiles it and adds its functions. [`PipelineArchive::save`] writes the
//! archive (to a temporary file, then renamed over the old one).
//!
//! Lookups are keyed by Metal on the functions and the descriptor, so a
//! changed shader or state simply misses; an OS or GPU compiler update
//! misses too. Nothing here decides when to save: the caller does
//! ([`PipelineArchive::should_save`]).

use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use crate::sys::{self, obj_handle_t};
use crate::{Device, Error, Object, Pool, Result, error_text};

/// What a pipeline creation call is given for the archive: copied into the
/// `binary_archives_for_lookup` / `binary_archive_for_serialization` /
/// `fail_on_binary_archive_miss` fields of its info struct.
#[derive(Clone, Copy, Debug)]
pub struct Lookup {
    pub archives: *const obj_handle_t,
    pub count: u8,
    pub serialize: obj_handle_t,
    pub fail_on_miss: bool,
}

impl Lookup {
    /// No archive at all (what every call did before P8).
    pub const NONE: Lookup = Lookup { archives: std::ptr::null(), count: 0, serialize: 0, fail_on_miss: false };
}

/// Counters of one archive (this process).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Pipelines found in the archive (no compile).
    pub hits: u64,
    /// Pipelines compiled and added.
    pub misses: u64,
    /// Pipelines that failed even without the archive (real errors).
    pub failed: u64,
    /// `add*PipelineFunctions` failures the unix side counted (all archives).
    pub add_failures: u64,
    /// Successful saves, and pipelines added since the last one.
    pub saves: u64,
    pub unsaved: u64,
    /// Whether the archive was loaded from an existing file.
    pub loaded: bool,
}

pub struct PipelineArchive {
    object: Object,
    /// The file as this (Windows) process names it, and as Metal does.
    path: PathBuf,
    loaded: bool,
    hits: AtomicU64,
    misses: AtomicU64,
    failed: AtomicU64,
    saves: AtomicU64,
    unsaved: AtomicU64,
    last_save: Mutex<Instant>,
    /// Creations that may add hold it shared; a save holds it exclusively
    /// (Metal does not promise add and serialize may overlap).
    gate: RwLock<()>,
}

// SAFETY: the archive is an Objective-C object handle; Metal binary
// archives may be used from any thread (adds and saves are serialized by
// `gate`).
unsafe impl Send for PipelineArchive {}
unsafe impl Sync for PipelineArchive {}

/// The unix path Metal needs for a path this process uses: `Z:\a\b` (Wine's
/// root drive) or an already-unix `/a/b`. Other drives are not mapped here.
pub fn unix_path(path: &Path) -> Option<String> {
    let s = path.to_str()?;
    if s.starts_with('/') {
        return Some(s.to_owned());
    }
    let b = s.as_bytes();
    if b.len() >= 3 && b[0].eq_ignore_ascii_case(&b'z') && b[1] == b':' && (b[2] == b'\\' || b[2] == b'/') {
        return Some(s[2..].replace('\\', "/"));
    }
    None
}

/// The path this (Windows) process can open for a unix or Windows path.
fn windows_path(path: &Path) -> PathBuf {
    match path.to_str() {
        Some(s) if s.starts_with('/') => PathBuf::from(format!("Z:{}", s.replace('/', "\\"))),
        _ => path.to_path_buf(),
    }
}

fn c_path(path: &Path) -> Result<CString> {
    let unix = unix_path(path).ok_or_else(|| Error(format!("{}: not on Wine's Z: drive", path.display())))?;
    CString::new(unix).map_err(|_| Error("archive path has a NUL".into()))
}

fn new_archive(device: &Device, url: Option<&CString>) -> Result<Object> {
    let _pool = Pool::new();
    let mut err = 0;
    // SAFETY: a live device; url is NUL terminated or null (an empty archive).
    let handle = unsafe {
        sys::MTLDevice_newBinaryArchive(device.handle(), url.map_or(std::ptr::null(), |u| u.as_ptr()), &mut err)
    };
    match Object::adopt(handle) {
        Some(object) => {
            if err != 0 {
                // SAFETY: err_out objects come back retained.
                unsafe { sys::NSObject_release(err) };
            }
            Ok(object)
        }
        None => Err(error_text(err, "newBinaryArchive")),
    }
}

impl PipelineArchive {
    /// The archive at `path` (unix or `Z:\` path), loaded when the file is
    /// there and readable, otherwise empty. A file Metal will not load is
    /// moved aside to `<path>.bad`. The second value says why a file that
    /// was there did not load.
    pub fn open(device: &Device, path: &Path) -> Result<(Self, Option<String>)> {
        let path = windows_path(path);
        let mut why = None;
        let mut loaded = false;
        let object = if std::fs::metadata(&path).is_ok_and(|m| m.len() > 0) {
            match new_archive(device, Some(&c_path(&path)?)) {
                Ok(o) => {
                    loaded = true;
                    o
                }
                Err(e) => {
                    why = Some(e.to_string());
                    let _ = std::fs::rename(&path, path.with_extension("bad"));
                    new_archive(device, None)?
                }
            }
        } else {
            c_path(&path)?;
            new_archive(device, None)?
        };
        Ok((
            Self {
                object,
                path,
                loaded,
                hits: AtomicU64::new(0),
                misses: AtomicU64::new(0),
                failed: AtomicU64::new(0),
                saves: AtomicU64::new(0),
                unsaved: AtomicU64::new(0),
                last_save: Mutex::new(Instant::now()),
                gate: RwLock::new(()),
            },
            why,
        ))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn handle(&self) -> obj_handle_t {
        self.object.handle()
    }

    pub fn stats(&self) -> Stats {
        Stats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
            add_failures: add_failures(),
            saves: self.saves.load(Ordering::Relaxed),
            unsaved: self.unsaved.load(Ordering::Relaxed),
            loaded: self.loaded,
        }
    }

    /// Whether there is something new worth writing: `every` pipelines
    /// added, or any once `after` passed since the last save.
    pub fn should_save(&self, every: u64, after: Duration) -> bool {
        let n = self.unsaved.load(Ordering::Relaxed);
        n > 0 && (n >= every || self.last_save.lock().unwrap_or_else(|e| e.into_inner()).elapsed() >= after)
    }

    /// Writes the archive (temporary file, then renamed over `path`).
    /// Returns the bytes written.
    pub fn save(&self) -> Result<u64> {
        let _gate = self.gate.write().unwrap_or_else(|e| e.into_inner());
        let added = self.unsaved.swap(0, Ordering::AcqRel);
        *self.last_save.lock().unwrap_or_else(|e| e.into_inner()) = Instant::now();
        let tmp = self.path.with_extension(format!("tmp{}", std::process::id()));
        let _ = std::fs::remove_file(&tmp);
        let url = c_path(&tmp)?;
        let _pool = Pool::new();
        let mut err = 0;
        // SAFETY: a live archive; url is NUL terminated.
        unsafe { sys::MTLBinaryArchive_serialize(self.handle(), url.as_ptr(), &mut err) };
        if err != 0 {
            self.unsaved.fetch_add(added, Ordering::AcqRel);
            let _ = std::fs::remove_file(&tmp);
            return Err(error_text(err, "MTLBinaryArchive serialize"));
        }
        let bytes = std::fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);
        if let Err(e) = std::fs::rename(&tmp, &self.path) {
            self.unsaved.fetch_add(added, Ordering::AcqRel);
            let _ = std::fs::remove_file(&tmp);
            return Err(Error(format!("rename {} -> {}: {e}", tmp.display(), self.path.display())));
        }
        self.saves.fetch_add(1, Ordering::Relaxed);
        Ok(bytes)
    }

    /// Makes one pipeline through the archive (see the module docs).
    /// `make(lookup, &mut err)` runs the creation call with `lookup` copied
    /// into its info struct and returns the new (+1) pipeline handle or 0.
    pub fn create(&self, make: &dyn Fn(Lookup, &mut obj_handle_t) -> obj_handle_t, what: &str) -> Result<Object> {
        let archives = [self.handle()];
        let mut err = 0;
        let lookup = Lookup { archives: archives.as_ptr(), count: 1, serialize: 0, fail_on_miss: true };
        if let Some(pso) = Object::adopt(make(lookup, &mut err)) {
            release_error(err);
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(pso);
        }
        release_error(err);
        let mut err = 0;
        let pso = {
            let _gate = self.gate.read().unwrap_or_else(|e| e.into_inner());
            let add = Lookup { archives: archives.as_ptr(), count: 1, serialize: self.handle(), fail_on_miss: false };
            make(add, &mut err)
        };
        match Object::adopt(pso) {
            Some(pso) => {
                release_error(err);
                self.misses.fetch_add(1, Ordering::Relaxed);
                self.unsaved.fetch_add(1, Ordering::Relaxed);
                Ok(pso)
            }
            None => {
                self.failed.fetch_add(1, Ordering::Relaxed);
                Err(error_text(err, what))
            }
        }
    }
}

fn release_error(err: obj_handle_t) {
    if err != 0 {
        // SAFETY: err_out objects come back retained.
        unsafe { sys::NSObject_release(err) };
    }
}

/// How many archive adds the unix side saw fail (0 for an older build).
pub fn add_failures() -> u64 {
    let f = sys::symbol(b"VM_ArchiveAddFailures\0");
    if f.is_null() {
        return 0;
    }
    // SAFETY: msc.h's signature.
    unsafe { core::mem::transmute::<*const core::ffi::c_void, unsafe extern "C" fn() -> u64>(f)() }
}

static GLOBAL: RwLock<Option<Arc<PipelineArchive>>> = RwLock::new(None);

/// Makes `archive` the one every pipeline of this crate goes through
/// (`None`: none, as before P8).
pub fn install(archive: Option<Arc<PipelineArchive>>) {
    *GLOBAL.write().unwrap_or_else(|e| e.into_inner()) = archive;
}

/// The installed archive, if any.
pub fn installed() -> Option<Arc<PipelineArchive>> {
    GLOBAL.read().unwrap_or_else(|e| e.into_inner()).clone()
}

/// One pipeline creation through the installed archive, or straight
/// through without one.
pub(crate) fn create(make: &dyn Fn(Lookup, &mut obj_handle_t) -> obj_handle_t, what: &str) -> Result<Object> {
    match installed() {
        Some(archive) => archive.create(make, what),
        None => {
            let mut err = 0;
            let pso = make(Lookup::NONE, &mut err);
            match Object::adopt(pso) {
                Some(pso) => {
                    release_error(err);
                    Ok(pso)
                }
                None => Err(error_text(err, what)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void twice(device uint* b [[buffer(0)]], uint i [[thread_position_in_grid]]) { b[i] *= 2; }
kernel void thrice(device uint* b [[buffer(0)]], uint i [[thread_position_in_grid]]) { b[i] *= 3; }
kernel void square(device uint* b [[buffer(0)]], uint i [[thread_position_in_grid]]) { b[i] *= b[i]; }
"#;

    /// Run twice (two processes) with `VICTORY_ARCHIVE_TEST` = a `Z:\` path of a file that
    /// does not exist yet: the first run compiles three pipelines (misses) and saves; the
    /// second loads the file and finds all three (hits, no compile).
    #[test]
    #[ignore = "run twice under Wine with VICTORY_ARCHIVE_TEST=<Z: path>"]
    fn archive_hits_across_processes() {
        let Some(path) = std::env::var_os("VICTORY_ARCHIVE_TEST") else { return };
        let _pool = Pool::new();
        let device = Device::first().unwrap();
        let (archive, why) = PipelineArchive::open(&device, Path::new(&path)).unwrap();
        assert!(why.is_none(), "{why:?}");
        let loaded = archive.stats().loaded;
        install(Some(Arc::new(archive)));
        let library = device.library_msl(SOURCE).unwrap();
        let start = Instant::now();
        for f in ["twice", "thrice", "square"] {
            device.compute_pipeline_from_library(&library, f).unwrap();
        }
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let archive = installed().unwrap();
        let s = archive.stats();
        println!("ARCHIVE loaded={loaded} hits={} misses={} failed={} add_failures={} ms={ms:.1}", s.hits, s.misses, s.failed, s.add_failures);
        if loaded {
            assert_eq!((s.hits, s.misses, s.failed), (3, 0, 0));
        } else {
            assert_eq!((s.hits, s.misses, s.failed), (0, 3, 0));
            assert!(archive.should_save(3, Duration::from_secs(3600)));
            let bytes = archive.save().unwrap();
            println!("ARCHIVE saved {bytes} bytes");
            assert!(bytes > 0);
            assert!(!archive.should_save(1, Duration::ZERO));
        }
        install(None);
    }

    #[test]
    fn unix_paths() {
        assert_eq!(unix_path(Path::new(r"Z:\Users\a\c d\x.metalar")).as_deref(), Some("/Users/a/c d/x.metalar"));
        assert_eq!(unix_path(Path::new(r"z:/tmp/x")).as_deref(), Some("/tmp/x"));
        assert_eq!(unix_path(Path::new("/tmp/x")).as_deref(), Some("/tmp/x"));
        assert_eq!(unix_path(Path::new(r"C:\x")), None);
        assert_eq!(windows_path(Path::new("/tmp/a b/x")), PathBuf::from(r"Z:\tmp\a b\x"));
    }
}
