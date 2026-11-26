//! Fast hashing for internal pointer, slot and range keys.

use std::hash::{BuildHasherDefault, Hasher};

pub(crate) type Map<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<Fx>>;
pub(crate) type Set<K> = std::collections::HashSet<K, BuildHasherDefault<Fx>>;

#[derive(Default)]
pub(crate) struct Fx(u64);

impl Hasher for Fx {
    fn finish(&self) -> u64 {
        // Move well-mixed bits into the bucket index for aligned addresses.
        self.0.rotate_left(26)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.write_u64(u64::from(byte));
        }
    }

    fn write_u32(&mut self, word: u32) {
        self.write_u64(u64::from(word));
    }

    fn write_u64(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x517c_c1b7_2722_0a95);
    }

    fn write_usize(&mut self, word: usize) {
        self.write_u64(word as u64);
    }
}
