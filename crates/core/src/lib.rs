//! The mavian core: what determinism rests on.
//!
//! - [`math`] — vectors and rotations (Y up, a character at rest faces +Z);
//! - [`noise`] — noise and randomness from a seed only, no state;
//! - [`time`] — keys, easing and three times (world, shot, drawing);
//! - [`hash`] — stable hash: the same data gives the same fingerprint
//!   in any run on any machine (the graph cache relies on it).
//!
//! Numbers are computed in the same order as in the Python prototype, so
//! frames match the MVP down to the last digits.

pub mod hash;
pub mod m;
pub mod math;
pub mod noise;
pub mod time;

pub use hash::{Fingerprint, StableHash, StableHasher};
pub use math::{Mat3, Vec3, mix, v3};
pub use time::{Blend, Clock, ClockKind, Ease, Keys, Timeline, Track};
