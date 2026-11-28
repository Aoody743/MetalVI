//! Shared events: CPU/GPU fences and command-buffer signal/wait encodings.
//!
//! A [`SharedEvent`] carries a monotonically increasing `u64` value. The CPU
//! signals it with [`SharedEvent::signal`]; a command buffer signals or waits
//! on it through [`CommandBuffer::encode_signal_event`] /
//! [`CommandBuffer::encode_wait_for_event`]. (DESIGN.md §5: fences →
//! MTLSharedEvent.)

use crate::sys;
use crate::{CommandBuffer, Device, Error, Object, Pool, Result};
use sys::obj_handle_t;

/// A Metal shared event (+1 reference released on drop; `Clone` retains).
#[derive(Clone, Debug)]
pub struct SharedEvent(Object);

impl SharedEvent {
    /// A new event, starting at value 0 (`newSharedEvent`).
    pub fn new(device: &Device) -> Result<Self> {
        let _pool = Pool::new();
        // SAFETY: `newSharedEvent` returns an owned (+1) object.
        let event = Object::adopt(unsafe { sys::MTLDevice_newSharedEvent(device.handle()) })
            .ok_or(Error("newSharedEvent".into()))?;
        Ok(Self(event))
    }

    pub fn handle(&self) -> obj_handle_t {
        self.0.handle()
    }

    /// Sets the event's value from the CPU (`setSignaledValue:`).
    pub fn signal(&self, value: u64) {
        // SAFETY: a live event.
        unsafe { sys::MTLSharedEvent_signalValue(self.0.handle(), value) };
    }

    /// The latest value the event has been signaled to (`signaledValue`).
    pub fn signaled_value(&self) -> u64 {
        // SAFETY: a live event.
        unsafe { sys::MTLSharedEvent_signaledValue(self.0.handle()) }
    }

    /// Blocks until the event reaches `value`; `timeout_ms` follows the unix
    /// side's `waitUntilSignaledValue:timeoutMS:` (0 = wait forever). Returns
    /// false on timeout.
    pub fn wait_until_signaled(&self, value: u64, timeout_ms: u64) -> bool {
        // SAFETY: a live event.
        unsafe { sys::MTLSharedEvent_waitUntilSignaledValue(self.0.handle(), value, timeout_ms) }
    }
}

impl CommandBuffer {
    /// The GPU signals `value` on `event` when this command buffer reaches the
    /// encoder (`encodeSignalEvent:value:`).
    pub fn encode_signal_event(&self, event: &SharedEvent, value: u64) {
        // SAFETY: live command buffer and event.
        unsafe { sys::MTLCommandBuffer_encodeSignalEvent(self.handle(), event.handle(), value) };
    }

    /// The GPU waits until `event` reaches `value` before continuing
    /// (`encodeWaitForEvent:value:`).
    pub fn encode_wait_for_event(&self, event: &SharedEvent, value: u64) {
        // SAFETY: live command buffer and event.
        unsafe { sys::MTLCommandBuffer_encodeWaitForEvent(self.handle(), event.handle(), value) };
    }
}

/// A GPU-only event (`newEvent`): orders command buffers of one queue (or
/// across queues) without CPU visibility (plan v2 D3). Its value is set by
/// [`CommandBuffer::encode_signal`] and waited on by
/// [`CommandBuffer::encode_wait`].
#[derive(Clone, Debug)]
pub struct Event(Object);

impl Event {
    pub fn new(device: &Device) -> Result<Self> {
        let _pool = Pool::new();
        // SAFETY: `newEvent` returns an owned (+1) object.
        let event = Object::adopt(unsafe { sys::MTLDevice_newEvent(device.handle()) })
            .ok_or(Error("newEvent".into()))?;
        Ok(Self(event))
    }

    pub fn handle(&self) -> obj_handle_t {
        self.0.handle()
    }
}

/// A fence between encoders (`newFence`): a compute encoder updates it at
/// its end, a later encoder waits for it (plan v2 D3).
#[derive(Clone, Debug)]
pub struct Fence(Object);

impl Fence {
    pub fn new(device: &Device) -> Result<Self> {
        let _pool = Pool::new();
        // SAFETY: `newFence` returns an owned (+1) object.
        let fence = Object::adopt(unsafe { sys::MTLDevice_newFence(device.handle()) })
            .ok_or(Error("newFence".into()))?;
        Ok(Self(fence))
    }

    pub fn handle(&self) -> obj_handle_t {
        self.0.handle()
    }
}

/// Something a command buffer can signal or wait on: a [`SharedEvent`] or
/// an [`Event`] (both are `MTLEvent`s on the Metal side).
pub trait AnyEvent {
    fn event_handle(&self) -> obj_handle_t;
}
impl AnyEvent for SharedEvent {
    fn event_handle(&self) -> obj_handle_t {
        self.handle()
    }
}
impl AnyEvent for Event {
    fn event_handle(&self) -> obj_handle_t {
        self.handle()
    }
}

impl CommandBuffer {
    /// `encodeSignalEvent:value:` for either kind of event.
    pub fn encode_signal(&self, event: &impl AnyEvent, value: u64) {
        // SAFETY: live command buffer and event.
        unsafe { sys::MTLCommandBuffer_encodeSignalEvent(self.handle(), event.event_handle(), value) };
    }

    /// `encodeWaitForEvent:value:` for either kind of event.
    pub fn encode_wait(&self, event: &impl AnyEvent, value: u64) {
        // SAFETY: live command buffer and event.
        unsafe { sys::MTLCommandBuffer_encodeWaitForEvent(self.handle(), event.event_handle(), value) };
    }
}
