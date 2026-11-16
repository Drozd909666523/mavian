//! Port regression: three_cams poses are pinned by a fingerprint.
//!
//! The port was checked against the MVP byte for byte (time, drawing, cameras, poses of all 318
//! frames — `mavian dump` against the MVP's `pose_signature`). After that
//! three things were changed on purpose: squash no longer moves planted
//! feet, pelvis velocity no longer jumps where segments meet, and the jump is
//! planned from intent (`jump`: a ballistic flight from the push's speed).
//! Any other edit that changes poses must change this fingerprint too —
//! on purpose.

use mavian_check::{Status, feet, poses_digest};

#[test]
fn poses_are_frozen() {
    let shot = mavian_shots::by_name("three_cams").unwrap();
    shot.validate().unwrap();
    assert_eq!(poses_digest(&shot, &shot.timeline()), FROZEN);
}

#[test]
fn planted_feet_do_not_slide() {
    let shot = mavian_shots::by_name("three_cams").unwrap();
    let r = feet(&shot, &shot.timeline());
    assert_eq!(r.status, Status::Ok, "{r}");
}

const FROZEN: &str = "35a9128e5a88ca9446add001b9e07782";
