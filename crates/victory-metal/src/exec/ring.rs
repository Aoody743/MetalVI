//! Per-queue ring memory for what one command buffer reads in place
//! (one-view descriptor tables, argument bytes, upload staging): plan v2 A6.
//!
//! Space is taken when a command buffer is encoded and given back when its
//! completion has been seen, in commit order (completion is in order per
//! queue, so a FIFO is enough). A request that does not fit waits for
//! completions; it never overwrites a region a command buffer in flight
//! still reads.

use std::collections::VecDeque;

use crate::{Buffer, Device, Result};

/// Alignment of every region (Metal's buffer-offset rule for constant data).
pub const ALIGN: usize = 256;

/// The FIFO bookkeeping, separate from the buffer so it is testable on the
/// CPU alone.
#[derive(Debug)]
pub struct Fifo {
    capacity: usize,
    head: usize,
    used: usize,
    /// Bytes each live region consumed (including padding skipped at the
    /// end of the buffer), oldest first.
    live: VecDeque<usize>,
    pub high_water: usize,
}

/// A region handed out by [`Fifo::take`]: give it back with
/// [`Fifo::give_back`] in the order taken.
#[derive(Debug, PartialEq, Eq)]
pub struct Token {
    pub offset: usize,
    pub len: usize,
    consumed: usize,
}

impl Fifo {
    pub fn new(capacity: usize) -> Self {
        Self { capacity: capacity / ALIGN * ALIGN, head: 0, used: 0, live: VecDeque::new(), high_water: 0 }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Whether a request of `len` bytes could ever fit.
    pub fn fits_ever(&self, len: usize) -> bool {
        len.next_multiple_of(ALIGN) <= self.capacity
    }

    /// A region of `len` bytes (0 is allowed and takes nothing), or `None`
    /// while the space is held by command buffers in flight.
    pub fn take(&mut self, len: usize) -> Option<Token> {
        let len = len.next_multiple_of(ALIGN);
        if len == 0 {
            return Some(Token { offset: 0, len: 0, consumed: 0 });
        }
        if len > self.capacity {
            return None;
        }
        // Free space is [head, tail) circularly, tail = head - used.
        let (offset, consumed) = if self.head + len <= self.capacity {
            (self.head, len)
        } else {
            // Skip the tail end of the buffer.
            (0, self.capacity - self.head + len)
        };
        if self.used + consumed > self.capacity {
            return None;
        }
        self.used += consumed;
        self.head = (offset + len) % self.capacity;
        self.live.push_back(consumed);
        self.high_water = self.high_water.max(self.used);
        Some(Token { offset, len, consumed })
    }

    /// Gives back the oldest live region (`token` must be it).
    pub fn give_back(&mut self, token: Token) {
        if token.consumed == 0 {
            return;
        }
        let oldest = self.live.pop_front();
        debug_assert_eq!(oldest, Some(token.consumed), "ring regions given back out of order");
        self.used -= token.consumed;
        if self.used == 0 {
            // Empty: start over so large requests find contiguous space.
            self.head = 0;
        }
    }

    pub fn used(&self) -> usize {
        self.used
    }
}

/// A shared Metal buffer managed by a [`Fifo`].
pub struct Ring {
    pub buffer: Buffer,
    cpu: *mut u8,
    pub fifo: Fifo,
}

// The CPU pointer is only written by the encoder thread for regions it owns
// until the region is given back.
unsafe impl Send for Ring {}

impl Ring {
    pub fn new(device: &Device, capacity: usize) -> Result<Self> {
        let capacity = capacity.max(ALIGN) / ALIGN * ALIGN;
        let (buffer, cpu) = device.shared_buffer_mapped(&vec![0u8; capacity])?;
        Ok(Self { buffer, cpu, fifo: Fifo::new(capacity) })
    }

    /// The CPU and GPU view of a region taken from this ring.
    pub fn slice(&self, token: &Token) -> RingSlice {
        RingSlice {
            buffer: self.buffer.clone(),
            // SAFETY: the region lies inside the buffer (Fifo invariant).
            cpu: unsafe { self.cpu.add(token.offset) },
            offset: token.offset as u64,
            len: token.len,
            next: 0,
        }
    }
}

/// One command buffer's part of its queue's ring. Writes go through
/// [`RingSlice::put`]; the GPU reads them at `buffer` + the returned offset.
pub struct RingSlice {
    pub buffer: Buffer,
    cpu: *mut u8,
    /// Byte offset of the slice in `buffer`.
    pub offset: u64,
    pub len: usize,
    next: usize,
}

impl RingSlice {
    /// An empty slice (no ring space was needed).
    pub fn empty(buffer: Buffer) -> Self {
        Self { buffer, cpu: std::ptr::null_mut(), offset: 0, len: 0, next: 0 }
    }

    /// Copies `bytes` into the slice: their offset in [`RingSlice::buffer`],
    /// or `None` if the slice is full (the caller sized it too small).
    pub fn put(&mut self, bytes: &[u8]) -> Option<u64> {
        let at = self.next;
        let end = at.checked_add(bytes.len())?;
        if end > self.len {
            return None;
        }
        // SAFETY: inside this slice, which no command buffer in flight reads.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.cpu.add(at), bytes.len()) };
        self.next = end.next_multiple_of(ALIGN).min(self.len);
        Some(self.offset + at as u64)
    }

    /// Reserves `len` bytes for the caller to write itself: their CPU
    /// address and their offset in [`RingSlice::buffer`].
    pub fn reserve(&mut self, len: usize) -> Option<(*mut u8, u64)> {
        let at = self.next;
        let end = at.checked_add(len)?;
        if end > self.len {
            return None;
        }
        self.next = end.next_multiple_of(ALIGN).min(self.len);
        // SAFETY: inside this slice.
        Some((unsafe { self.cpu.add(at) }, self.offset + at as u64))
    }

    /// Ring bytes `put` needs for these sizes (each aligned).
    pub fn need(sizes: impl IntoIterator<Item = usize>) -> usize {
        sizes.into_iter().map(|n| n.next_multiple_of(ALIGN)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fifo_wraps_and_frees_in_order() {
        let mut f = Fifo::new(1024);
        let a = f.take(300).unwrap(); // 512
        let b = f.take(256).unwrap(); // 256
        assert_eq!((a.offset, b.offset), (0, 512));
        // 256 left at the end; 512 does not fit until `a` is back.
        assert!(f.take(512).is_none());
        f.give_back(a);
        let c = f.take(512).unwrap();
        // Skips the 256 at the end, starts at 0.
        assert_eq!(c.offset, 0);
        assert_eq!(f.used(), 256 + 256 + 512);
        assert!(f.take(1).is_none());
        f.give_back(b);
        f.give_back(c);
        assert_eq!(f.used(), 0);
        assert!(f.take(1024).is_some());
    }

    #[test]
    fn fifo_never_hands_out_live_bytes() {
        // Random-ish sizes, freeing oldest when full: no two live regions overlap.
        let mut f = Fifo::new(4096);
        let mut live: VecDeque<Token> = VecDeque::new();
        let mut x = 12345u32;
        for _ in 0..2000 {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            let len = (x >> 20) as usize % 1500 + 1;
            let t = loop {
                if let Some(t) = f.take(len) { break t; }
                f.give_back(live.pop_front().expect("an empty ring fits every request"));
            };
            for o in &live {
                let disjoint = t.offset + t.len <= o.offset || o.offset + o.len <= t.offset;
                assert!(disjoint, "{t:?} overlaps {o:?}");
            }
            assert!(t.offset + t.len <= 4096);
            live.push_back(t);
        }
    }

    #[test]
    fn fifo_rejects_what_never_fits() {
        let mut f = Fifo::new(512);
        assert!(!f.fits_ever(513));
        assert!(f.take(513).is_none());
        assert_eq!(f.take(0).unwrap().len, 0);
    }
}
