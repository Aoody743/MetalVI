//! Safe Rust side of victorymetal, MetalVI's Wine unixlib for Metal
//! (`native/victorymetal`, the Metal object layer of dxmt's winemetal).
//!
//! Every Metal object is an Objective-C pointer carried as a `u64` handle
//! ([`sys::obj_handle_t`]); the unix side runs the actual Metal call. The
//! PE and unix sides share one address space, so title memory can be handed
//! to Metal as it is ([`Device::wrap`], `newBufferWithBytesNoCopy`).
//!
//! Ownership follows Cocoa: `new*`/`Copy*` calls return an object we own
//! ([`Object::adopt`]); everything else is autoreleased and must be retained
//! to outlive the current [`Pool`] ([`Object::retain`]).

#![cfg(windows)]

pub mod blit;
pub mod chunks;
pub mod msc;
pub mod events;
pub mod ir;
pub mod sampler;
pub mod sys;
pub mod texture;

use core::ffi::c_void;
use std::fmt;

use sys::obj_handle_t;

/// A failure reported by Metal (its `NSError` description) or a null result.
#[derive(Debug, Clone)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// An owned reference to an Objective-C object (release on drop).
pub struct Object(obj_handle_t);

impl Object {
    /// Takes over a +1 reference (`new*`/`Copy*`); `None` for null.
    pub fn adopt(handle: obj_handle_t) -> Option<Self> {
        (handle != 0).then_some(Self(handle))
    }
    /// Retains an autoreleased object; `None` for null.
    pub fn retain(handle: obj_handle_t) -> Option<Self> {
        if handle == 0 {
            return None;
        }
        // SAFETY: a live object handle from the unix side.
        unsafe { sys::NSObject_retain(handle) };
        Some(Self(handle))
    }
    pub fn handle(&self) -> obj_handle_t {
        self.0
    }

    /// The `gpuResourceID` of a texture or (argument-buffer capable)
    /// sampler state, for descriptor entries; 0 for anything else.
    pub fn gpu_resource_id(&self) -> u64 {
        gpu_resource_id(self.0)
    }
}

/// `gpuResourceID` of a texture or sampler state handle (victorymetal's
/// `VM_GpuResourceID`); 0 for anything else.
pub fn gpu_resource_id(handle: obj_handle_t) -> u64 {
    let f = sys::symbol(b"VM_GpuResourceID\0");
    if f.is_null() || handle == 0 {
        return 0;
    }
    // SAFETY: msc.h's signature.
    unsafe { core::mem::transmute::<*const c_void, unsafe extern "C" fn(u64) -> u64>(f)(handle) }
}

impl Clone for Object {
    fn clone(&self) -> Self {
        // SAFETY: we hold a reference, so the object is alive.
        unsafe { sys::NSObject_retain(self.0) };
        Self(self.0)
    }
}

impl Drop for Object {
    fn drop(&mut self) {
        // SAFETY: releases the reference this value owns.
        unsafe { sys::NSObject_release(self.0) };
    }
}

impl fmt::Debug for Object {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Object({:#x})", self.0)
    }
}

/// An autorelease pool for the current thread: autoreleased objects made
/// while it lives are freed when it drops. Every thread that calls Metal
/// needs one around its work.
pub struct Pool(obj_handle_t);

impl Pool {
    pub fn new() -> Self {
        // SAFETY: no preconditions.
        Self(unsafe { sys::NSAutoreleasePool_alloc_init() })
    }
}

impl Default for Pool {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        // SAFETY: releasing the pool drains it.
        unsafe { sys::NSObject_release(self.0) };
    }
}

/// The text of an `NSString`.
pub fn text(handle: obj_handle_t) -> String {
    if handle == 0 {
        return String::new();
    }
    let mut buf = vec![0u8; 4096];
    // SAFETY: the buffer is writable for its length; 4 = NSUTF8StringEncoding.
    let ok = unsafe {
        sys::NSString_getCString(handle, buf.as_mut_ptr().cast(), buf.len() as u64, sys::WMTStringEncoding(4))
    };
    if ok == 0 {
        return String::from("<not a string>");
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

/// The description of an `NSError` returned through an `err_out` (owned:
/// released here).
pub(crate) fn error_text(err: obj_handle_t, what: &str) -> Error {
    if err == 0 {
        return Error(format!("{what}: failed"));
    }
    let _pool = Pool::new();
    // SAFETY: a live error; the description is autoreleased.
    let message = text(unsafe { sys::NSObject_description(err) });
    // SAFETY: err_out objects are returned retained (T1 §7.2.7).
    unsafe { sys::NSObject_release(err) };
    Error(format!("{what}: {message}"))
}

/// A Metal device.
#[derive(Clone, Debug)]
pub struct Device(Object);

impl Device {
    /// The first device (the only one on Apple Silicon).
    pub fn first() -> Result<Self> {
        let _pool = Pool::new();
        // SAFETY: Copy returns an owned array.
        let devices = Object::adopt(unsafe { sys::WMTCopyAllDevices() }).ok_or(Error("no device list".into()))?;
        // SAFETY: a live array.
        if unsafe { sys::NSArray_count(devices.handle()) } == 0 {
            return Err(Error("no Metal device".into()));
        }
        // SAFETY: index 0 exists; the element is borrowed from the array.
        let device = Object::retain(unsafe { sys::NSArray_object(devices.handle(), 0) }).ok_or(Error("device 0".into()))?;
        Ok(Self(device))
    }

    pub fn handle(&self) -> obj_handle_t {
        self.0.handle()
    }

    pub fn name(&self) -> String {
        let _pool = Pool::new();
        // SAFETY: a live device; the name is autoreleased.
        text(unsafe { sys::MTLDevice_name(self.handle()) })
    }

    /// Memory the caller already has, as a shared buffer without a copy
    /// (`newBufferWithBytesNoCopy`). The buffer keeps the physical pages
    /// mapped at `at` when it is made; after a remap, wrap again (T4).
    ///
    /// # Safety
    /// `at..at+len` must stay mapped for as long as the buffer or any
    /// command using it lives; `at` and `len` page aligned.
    pub unsafe fn wrap(&self, at: *mut c_void, len: u64) -> Result<Buffer> {
        let mut info = sys::WMTBufferInfo {
            length: len,
            options: sys::WMTResourceOptions::WMTResourceStorageModeShared,
            memory: sys::WMTMemoryPointer { ptr: at },
            gpu_address: 0,
        };
        // SAFETY: info is valid for the call; the caller guarantees the memory.
        let buffer = Object::adopt(unsafe { sys::MTLDevice_newBuffer(self.handle(), &mut info) })
            .ok_or_else(|| Error(format!("newBufferWithBytesNoCopy({at:p}, {len:#x}) refused")))?;
        Ok(Buffer { object: buffer, gpu_address: info.gpu_address, len })
    }

    /// A compute pipeline from MSL source (runtime compiled: tests and our
    /// own helper kernels; title shaders come as metallibs).
    pub fn compute_pipeline_msl(&self, source: &str, function: &str) -> Result<ComputePipeline> {
        let _pool = Pool::new();
        let mut err = 0;
        // SAFETY: source is valid for its length.
        let library = Object::adopt(unsafe {
            sys::MTLDevice_newLibraryWithSource(self.handle(), source.as_ptr().cast(), source.len() as u64, &mut err)
        })
        .ok_or_else(|| error_text(err, "newLibraryWithSource"))?;
        self.compute_pipeline_from_library(&library, function)
    }

    /// A compute pipeline from function `function` of `library`.
    pub fn compute_pipeline_from_library(&self, library: &Object, function: &str) -> Result<ComputePipeline> {
        let _pool = Pool::new();
        let name = std::ffi::CString::new(function).map_err(|_| Error("function name has a NUL".into()))?;
        // SAFETY: a live library; name is NUL terminated.
        let function = Object::adopt(unsafe { sys::MTLLibrary_newFunction(library.handle(), name.as_ptr()) })
            .ok_or_else(|| Error(format!("no function {function:?} in the library")))?;
        // SAFETY: all-zero is the default for the info struct (as the C smoke test does).
        let mut info: sys::WMTComputePipelineInfo = unsafe { std::mem::zeroed() };
        info.compute_function = function.handle();
        let mut err = 0;
        // SAFETY: info is valid for the call.
        let pso = Object::adopt(unsafe { sys::MTLDevice_newComputePipelineState(self.handle(), &info, &mut err) })
            .ok_or_else(|| error_text(err, "newComputePipelineState"))?;
        Ok(ComputePipeline(pso))
    }

    pub fn new_queue(&self, max_command_buffers: u64) -> Result<Queue> {
        // SAFETY: a live device.
        let queue = Object::adopt(unsafe { sys::MTLDevice_newCommandQueue(self.handle(), max_command_buffers) })
            .ok_or(Error("newCommandQueue".into()))?;
        Ok(Queue(queue))
    }
}

/// A buffer over memory we did not allocate (see [`Device::wrap`]).
#[derive(Clone, Debug)]
pub struct Buffer {
    object: Object,
    /// Where the GPU sees its first byte (not the CPU address).
    pub gpu_address: u64,
    pub len: u64,
}

impl Buffer {
    pub fn handle(&self) -> obj_handle_t {
        self.object.handle()
    }
}

#[derive(Clone, Debug)]
pub struct ComputePipeline(Object);

impl ComputePipeline {
    pub fn handle(&self) -> obj_handle_t {
        self.0.handle()
    }
}

/// A residency set: the allocations in it are resident for every command
/// buffer of the queues it is attached to, including memory shaders reach
/// only through a GPU address (T4 §3.6: needed, not an optimization).
#[derive(Clone, Debug)]
pub struct ResidencySet(Object);

impl ResidencySet {
    pub fn new(device: &Device, capacity: u64) -> Result<Self> {
        let mut err = 0;
        // SAFETY: a live device.
        let set = Object::adopt(unsafe { sys::MTLDevice_newResidencySet(device.handle(), capacity, &mut err) })
            .ok_or_else(|| error_text(err, "newResidencySet"))?;
        Ok(Self(set))
    }
    pub fn add(&self, handles: &[obj_handle_t]) {
        // SAFETY: live allocations, count matches.
        unsafe { sys::MTLResidencySet_addAllocations(self.0.handle(), handles.as_ptr(), handles.len() as u64) };
    }
    pub fn remove(&self, handles: &[obj_handle_t]) {
        // SAFETY: as above.
        unsafe { sys::MTLResidencySet_removeAllocations(self.0.handle(), handles.as_ptr(), handles.len() as u64) };
    }
    /// Makes additions and removals since the last commit take effect.
    pub fn commit(&self) {
        // SAFETY: a live set.
        unsafe { sys::MTLResidencySet_commit(self.0.handle()) };
    }
}

#[derive(Clone, Debug)]
pub struct Queue(Object);

impl Queue {
    /// Every command buffer of this queue keeps `set` resident.
    pub fn add_residency_set(&self, set: &ResidencySet) {
        // SAFETY: both live.
        unsafe { sys::MTLCommandQueue_addResidencySet(self.0.handle(), set.0.handle()) };
    }

    /// A command buffer (autoreleased by Metal, retained here).
    pub fn command_buffer(&self) -> Result<CommandBuffer> {
        let _pool = Pool::new();
        // SAFETY: a live queue.
        let cb = Object::retain(unsafe { sys::MTLCommandQueue_commandBuffer(self.0.handle()) })
            .ok_or(Error("commandBuffer".into()))?;
        Ok(CommandBuffer(cb))
    }
}

/// One step of a compute pass.
#[derive(Clone, Copy, Debug)]
pub enum Compute<'a> {
    Pipeline { pipeline: &'a ComputePipeline, threads_per_group: [u64; 3] },
    Buffer { buffer: &'a Buffer, offset: u64, index: u8 },
    Dispatch { groups: [u64; 3] },
    /// Bytes copied into the command (Metal's `setBytes`, at most 4 KiB):
    /// the top-level argument buffer goes this way.
    Bytes { bytes: &'a [u8], index: u8 },
    /// Writes of earlier dispatches to buffers are seen by later ones
    /// (Metal tracks no hazards for memory reached by address).
    Barrier,
}

pub struct CommandBuffer(Object);

impl CommandBuffer {
    pub fn handle(&self) -> obj_handle_t {
        self.0.handle()
    }

    /// Encodes a compute pass: the steps become winemetal's command chain
    /// (built here, read by the unix side during the call).
    pub fn compute(&self, steps: &[Compute<'_>]) -> Result<()> {
        let _pool = Pool::new();
        // SAFETY: a live command buffer; the encoder is autoreleased.
        let encoder = unsafe { sys::MTLCommandBuffer_computeCommandEncoder(self.handle(), false) };
        if encoder == 0 {
            return Err(Error("computeCommandEncoder".into()));
        }
        // Boxed so the commands' addresses do not move while linked.
        let mut chain: Vec<Box<dyn ChainLink>> = Vec::with_capacity(steps.len());
        for step in steps {
            chain.push(match *step {
                Compute::Pipeline { pipeline, threads_per_group: [w, h, d] } => Box::new(sys::wmtcmd_compute_setpso {
                    type_: sys::WMTComputeCommandType::WMTComputeCommandSetPSO,
                    reserved: [0; 3],
                    next: null_next(),
                    pso: pipeline.handle(),
                    threadgroup_size: sys::WMTSize { width: w, height: h, depth: d },
                }),
                Compute::Buffer { buffer, offset, index } => Box::new(sys::wmtcmd_compute_setbuffer {
                    type_: sys::WMTComputeCommandType::WMTComputeCommandSetBuffer,
                    reserved: [0; 3],
                    next: null_next(),
                    buffer: buffer.handle(),
                    offset,
                    index,
                }),
                Compute::Bytes { bytes, index } => Box::new(sys::wmtcmd_compute_setbytes {
                    type_: sys::WMTComputeCommandType::WMTComputeCommandSetBytes,
                    reserved: [0; 3],
                    next: null_next(),
                    bytes: sys::WMTMemoryPointer { ptr: bytes.as_ptr() as *mut c_void },
                    length: bytes.len() as u64,
                    index,
                }),
                Compute::Barrier => Box::new(sys::wmtcmd_compute_memory_barrier {
                    type_: sys::WMTComputeCommandType::WMTComputeCommandMemoryBarrier,
                    reserved: [0; 3],
                    next: null_next(),
                    scope: sys::WMTBarrierScope::WMTBarrierScopeBuffers,
                }),
                Compute::Dispatch { groups: [w, h, d] } => Box::new(sys::wmtcmd_compute_dispatch {
                    type_: sys::WMTComputeCommandType::WMTComputeCommandDispatch,
                    reserved: [0; 3],
                    next: null_next(),
                    size: sys::WMTSize { width: w, height: h, depth: d },
                }),
            });
        }
        for i in 1..chain.len() {
            let next = chain[i].as_base();
            chain[i - 1].set_next(next);
        }
        if let Some(head) = chain.first() {
            // SAFETY: the chain is well formed and lives until the call returns.
            unsafe { sys::MTLComputeCommandEncoder_encodeCommands(encoder, head.as_base()) };
        }
        // SAFETY: a live encoder.
        unsafe { sys::MTLCommandEncoder_endEncoding(encoder) };
        Ok(())
    }

    pub fn commit(&self) {
        // SAFETY: a live command buffer.
        unsafe { sys::MTLCommandBuffer_commit(self.handle()) };
    }

    /// Waits for the GPU; an error if the command buffer did not complete.
    pub fn wait(&self) -> Result<()> {
        let _pool = Pool::new();
        // SAFETY: a live, committed command buffer.
        unsafe { sys::MTLCommandBuffer_waitUntilCompleted(self.handle()) };
        // SAFETY: as above.
        let status = unsafe { sys::MTLCommandBuffer_status(self.handle()) };
        if status == sys::WMTCommandBufferStatus::WMTCommandBufferStatusCompleted {
            return Ok(());
        }
        // SAFETY: as above; the error and its description are autoreleased.
        let err = unsafe { sys::MTLCommandBuffer_error(self.handle()) };
        // SAFETY: a live error object when non-null.
        let description = if err == 0 { 0 } else { unsafe { sys::NSObject_description(err) } };
        Err(Error(format!("command buffer status {}: {}", status.0, text(description))))
    }
}

fn null_next() -> sys::WMTMemoryPointer {
    sys::WMTMemoryPointer { ptr: core::ptr::null_mut() }
}

/// A winemetal command struct: every one starts like [`sys::wmtcmd_base`].
trait ChainLink {
    fn as_base(&self) -> *const sys::wmtcmd_base;
    fn set_next(&mut self, next: *const sys::wmtcmd_base);
}

macro_rules! chain_link {
    ($($t:ty),*) => {$(
        impl ChainLink for $t {
            fn as_base(&self) -> *const sys::wmtcmd_base {
                (self as *const Self).cast()
            }
            fn set_next(&mut self, next: *const sys::wmtcmd_base) {
                self.next = sys::WMTMemoryPointer { ptr: next as *mut c_void };
            }
        }
    )*};
}
chain_link!(sys::wmtcmd_compute_setpso, sys::wmtcmd_compute_setbuffer, sys::wmtcmd_compute_dispatch, sys::wmtcmd_compute_setbytes, sys::wmtcmd_compute_memory_barrier);
pub mod render;
pub mod present;
