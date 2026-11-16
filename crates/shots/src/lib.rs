//! Shots ported from the MVP. For now a shot is described in Rust; the Python
//! description layer (PyO3) is the next step, and it will build the same
//! structures.

pub mod three_cams;

use mavian_scene::Shot;

/// Shot constructor.
pub type Make = fn() -> Shot;

/// All shots: name → constructor.
pub fn all() -> Vec<(&'static str, Make)> {
    vec![("three_cams", three_cams::make)]
}

pub fn by_name(name: &str) -> Option<Shot> {
    all().into_iter().find(|(n, _)| *n == name).map(|(_, f)| f())
}
