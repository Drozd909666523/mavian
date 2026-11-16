//! Seeded noise. No state and no system random — only
//! a hash of the arguments. The same arguments always give the same number.
//!
//! Matches the prototype bit for bit: strings are reduced via CRC-32, numbers
//! via splitmix64 mixing.

/// Part of a seed: a number or a string.
pub enum Part<'a> {
    Int(u64),
    Str(&'a str),
}

impl<'a> From<&'a str> for Part<'a> {
    fn from(s: &'a str) -> Self {
        Part::Str(s)
    }
}

macro_rules! int_part {
    ($($t:ty),*) => {$(
        impl From<$t> for Part<'static> {
            fn from(v: $t) -> Self { Part::Int(v as i64 as u64) }
        }
    )*};
}
int_part!(i32, i64, isize, usize, u32);

impl From<u64> for Part<'static> {
    fn from(v: u64) -> Self {
        Part::Int(v)
    }
}

/// Seed from parts: `seed!(shot_seed, "shake", i)`.
#[macro_export]
macro_rules! seed {
    ($($p:expr),+ $(,)?) => {
        $crate::noise::seed_of(&[$($crate::noise::Part::from($p)),+])
    };
}

fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    x
}

/// CRC-32 (as zlib.crc32).
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in bytes {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { (c >> 1) ^ 0xEDB8_8320 } else { c >> 1 };
        }
    }
    !c
}

/// Reduce numbers and strings to one 64-bit seed.
pub fn seed_of(parts: &[Part]) -> u64 {
    let mut h = 0x9E37_79B9_7F4A_7C15u64;
    for p in parts {
        let v = match p {
            Part::Int(v) => *v,
            Part::Str(s) => crc32(s.as_bytes()) as u64,
        };
        h = mix64(h ^ v);
    }
    h
}

/// A number in [-1, 1) from a seed.
pub fn rand_of(seed: u64) -> f64 {
    (seed >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

/// Smooth step on [0, 1].
pub fn smooth(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Smooth noise along a line, values in [-1, 1].
pub fn noise1(seed: u64, x: f64) -> f64 {
    let i = x.floor();
    let f = smooth(x - i);
    let i = i as i64;
    let a = rand_of(seed!(seed, i));
    let b = rand_of(seed!(seed, i + 1));
    a + (b - a) * f
}

/// Smooth noise on a plane, values in [-1, 1].
pub fn noise2(seed: u64, x: f64, y: f64) -> f64 {
    let (ix, iy) = (x.floor(), y.floor());
    let (fx, fy) = (smooth(x - ix), smooth(y - iy));
    let (ix, iy) = (ix as i64, iy as i64);
    let a = rand_of(seed!(seed, ix, iy));
    let b = rand_of(seed!(seed, ix + 1, iy));
    let c = rand_of(seed!(seed, ix, iy + 1));
    let d = rand_of(seed!(seed, ix + 1, iy + 1));
    let top = a + (b - a) * fx;
    let bot = c + (d - c) * fx;
    top + (bot - top) * fy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_zlib() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn matches_prototype() {
        // numbers from engine/noise.py
        assert_eq!(seed!(5u64, "shake", 0), 17_659_630_863_215_915_171);
        assert_eq!(noise1(seed!(5u64, "shake-x"), 1.37), 0.298_693_891_850_790_87);
    }
}
