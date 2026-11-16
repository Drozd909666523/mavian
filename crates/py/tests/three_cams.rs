//! The Python description builds the same shot as the Rust one.
//!
//! `shots/three_cams.py` is the MVP shot file with only the import changed;
//! its poses must match the Rust `three_cams` exactly.

use mavian_check::poses_digest;
use std::path::Path;

#[test]
fn python_shot_matches_rust_shot() {
    let file = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shots/three_cams.py");
    let mut shots = mavian_py::load_file(&file).unwrap();
    let (_, py) = shots.remove(0);
    let rs = mavian_shots::by_name("three_cams").unwrap();
    assert_eq!(poses_digest(&py, &py.timeline()), poses_digest(&rs, &rs.timeline()));
    assert_eq!(py.timeline().world, rs.timeline().world);
}

#[test]
fn description_errors_point_at_the_line() {
    let dir = std::env::temp_dir().join(format!("mavian-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("bad.py");
    std::fs::write(&file, "from mavian import *\n\na = Actor(STICKMAN, poses=keys({0.0: 'stand', 1.0: 'dance'}))\nshot = Shot(actors=[a])\n").unwrap();
    let Err(err) = mavian_py::load_file(&file) else { panic!("a bad pose must fail") };
    assert!(err.contains("line 3") && err.contains("no pose 'dance'"), "{err}");
    std::fs::remove_dir_all(&dir).unwrap();
}
