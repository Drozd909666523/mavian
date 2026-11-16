//! Portable maths: the same bits on every OS and CPU.
//!
//! The system maths library (glibc, musl, the MSVC runtime, Apple's libm)
//! may round `sin`, `pow`, `atan2` differently in the last bit, and one bit
//! in a pose or a pixel breaks byte-for-byte repeatability across machines.
//! These come from `libm` (a pure-Rust port of musl's), compiled into the
//! program. `sqrt`, `floor`, `round`, `abs` need nothing: IEEE 754 fixes them.
//! Clippy forbids the system versions (`clippy.toml`).

pub fn sin(x: f64) -> f64 {
    libm::sin(x)
}

pub fn cos(x: f64) -> f64 {
    libm::cos(x)
}

pub fn sin_cos(x: f64) -> (f64, f64) {
    libm::sincos(x)
}

pub fn tan(x: f64) -> f64 {
    libm::tan(x)
}

pub fn asin(x: f64) -> f64 {
    libm::asin(x)
}

pub fn acos(x: f64) -> f64 {
    libm::acos(x)
}

pub fn atan2(y: f64, x: f64) -> f64 {
    libm::atan2(y, x)
}

pub fn pow(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}

/// x to an integer power by repeated squaring — the same everywhere,
/// unlike the compiler intrinsic, which may call the system library.
pub fn powi(x: f64, n: i32) -> f64 {
    let mut base = if n < 0 { 1.0 / x } else { x };
    let mut k = n.unsigned_abs();
    let mut out = 1.0;
    while k > 0 {
        if k & 1 == 1 {
            out *= base;
        }
        base *= base;
        k >>= 1;
    }
    out
}

pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}

pub fn ln(x: f64) -> f64 {
    libm::log(x)
}

pub fn tanh(x: f64) -> f64 {
    libm::tanh(x)
}

pub fn hypot(x: f64, y: f64) -> f64 {
    libm::hypot(x, y)
}

pub fn sinf(x: f32) -> f32 {
    libm::sinf(x)
}

pub fn cosf(x: f32) -> f32 {
    libm::cosf(x)
}

pub fn atan2f(y: f32, x: f32) -> f32 {
    libm::atan2f(y, x)
}

pub fn hypotf(x: f32, y: f32) -> f32 {
    libm::hypotf(x, y)
}

pub fn powf(x: f32, y: f32) -> f32 {
    libm::powf(x, y)
}

/// f32 to an integer power by repeated squaring.
pub fn powif(x: f32, n: i32) -> f32 {
    powi(x as f64, n) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powi_matches_repeated_multiplication() {
        assert_eq!(powi(3.0, 4), 81.0);
        assert_eq!(powi(2.0, -2), 0.25);
        assert_eq!(powi(5.0, 0), 1.0);
    }
}
