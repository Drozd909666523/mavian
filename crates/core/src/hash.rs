//! Stable hash for the graph cache.
//!
//! Rust's built-in `Hash` does not promise the same numbers across compiler
//! versions, while the on-disk cache must survive a rebuild and another machine.
//! Hence our own hash: 128 bits, two independent splitmix64 mixes.

use crate::math::{Mat3, Vec3};
use std::fmt;

/// Fingerprint of data: same data — same fingerprint.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Fingerprint(pub u128);

impl fmt::Display for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

impl fmt::Debug for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Fingerprint({})", &format!("{self}")[..12])
    }
}

pub struct StableHasher {
    a: u64,
    b: u64,
    n: u64,
}

fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    x
}

impl Default for StableHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl StableHasher {
    pub fn new() -> Self {
        Self { a: 0x243F_6A88_85A3_08D3, b: 0x1319_8A2E_0370_7344, n: 0 }
    }

    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.n += 1;
        self.a = mix(self.a ^ v).wrapping_add(self.n);
        self.b = mix(self.b.rotate_left(17) ^ v ^ 0xA409_3822_299F_31D0);
        self
    }

    pub fn f64(&mut self, v: f64) -> &mut Self {
        // −0.0 and 0.0 are one number; all NaNs are one NaN
        let v = if v == 0.0 { 0.0 } else if v.is_nan() { f64::NAN } else { v };
        self.u64(v.to_bits())
    }

    pub fn bytes(&mut self, b: &[u8]) -> &mut Self {
        self.u64(b.len() as u64);
        for chunk in b.chunks(8) {
            let mut w = [0u8; 8];
            w[..chunk.len()].copy_from_slice(chunk);
            self.u64(u64::from_le_bytes(w));
        }
        self
    }

    pub fn add<T: StableHash + ?Sized>(&mut self, v: &T) -> &mut Self {
        v.stable_hash(self);
        self
    }

    pub fn finish(&self) -> Fingerprint {
        Fingerprint(((mix(self.a ^ self.n) as u128) << 64) | mix(self.b ^ self.a) as u128)
    }
}

/// Something with a stable fingerprint.
pub trait StableHash {
    fn stable_hash(&self, h: &mut StableHasher);

    fn fingerprint(&self) -> Fingerprint {
        let mut h = StableHasher::new();
        self.stable_hash(&mut h);
        h.finish()
    }
}

impl StableHash for f64 {
    fn stable_hash(&self, h: &mut StableHasher) {
        h.f64(*self);
    }
}

impl StableHash for u64 {
    fn stable_hash(&self, h: &mut StableHasher) {
        h.u64(*self);
    }
}

impl StableHash for u32 {
    fn stable_hash(&self, h: &mut StableHasher) {
        h.u64(*self as u64);
    }
}

impl StableHash for usize {
    fn stable_hash(&self, h: &mut StableHasher) {
        h.u64(*self as u64);
    }
}

impl StableHash for bool {
    fn stable_hash(&self, h: &mut StableHasher) {
        h.u64(*self as u64);
    }
}

impl StableHash for str {
    fn stable_hash(&self, h: &mut StableHasher) {
        h.bytes(self.as_bytes());
    }
}

impl StableHash for String {
    fn stable_hash(&self, h: &mut StableHasher) {
        h.bytes(self.as_bytes());
    }
}

impl StableHash for Vec3 {
    fn stable_hash(&self, h: &mut StableHasher) {
        h.f64(self.x).f64(self.y).f64(self.z);
    }
}

impl StableHash for Mat3 {
    fn stable_hash(&self, h: &mut StableHasher) {
        for row in &self.0 {
            for v in row {
                h.f64(*v);
            }
        }
    }
}

impl<T: StableHash> StableHash for [T] {
    fn stable_hash(&self, h: &mut StableHasher) {
        h.u64(self.len() as u64);
        for v in self {
            v.stable_hash(h);
        }
    }
}

impl<T: StableHash> StableHash for Vec<T> {
    fn stable_hash(&self, h: &mut StableHasher) {
        self.as_slice().stable_hash(h);
    }
}

impl<T: StableHash> StableHash for Option<T> {
    fn stable_hash(&self, h: &mut StableHasher) {
        match self {
            None => {
                h.u64(0);
            }
            Some(v) => {
                h.u64(1);
                v.stable_hash(h);
            }
        }
    }
}

impl<A: StableHash, B: StableHash> StableHash for (A, B) {
    fn stable_hash(&self, h: &mut StableHasher) {
        self.0.stable_hash(h);
        self.1.stable_hash(h);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_data_same_print_and_order_matters() {
        let a = (1.0f64, 2.0f64).fingerprint();
        assert_eq!(a, (1.0f64, 2.0f64).fingerprint());
        assert_ne!(a, (2.0f64, 1.0f64).fingerprint());
        assert_eq!(0.0f64.fingerprint(), (-0.0f64).fingerprint());
    }

    #[test]
    fn print_is_frozen() {
        // Fingerprints live in the on-disk cache: changing the algorithm must be deliberate.
        assert_eq!(format!("{}", "mavian".fingerprint()), FROZEN);
    }

    const FROZEN: &str = "881bbe5ebebd3424c3f6f9a0996517ea";
}
