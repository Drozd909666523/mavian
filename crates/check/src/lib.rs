//! Built-in checks: what the MVP measured by hand, the core checks itself.
//!
//! - planted feet do not slide;
//! - the camera does not jerk within a camera setup;
//! - changing the seed changes the drawing but not the poses.
//!
//! Repeatability (two independent runs give the same bytes) is checked by
//! `mavian check`: it needs to launch separate processes.

use mavian_core::m;
use mavian_core::{StableHasher, Timeline, Vec3};
use mavian_scene::Shot;
use std::fmt;

/// a foot point is on the ground below this height, world units
const ON_GROUND: f64 = 0.01;
/// planted-foot slide per drawing above which we warn (1 cm
/// for a 1.7 tall stick figure)
const SLIDE_WARN: f64 = 0.01;
/// camera jerk — jump in angular speed per frame, degrees per frame
const JERK_WARN: f64 = 1.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Ok,
    Warn,
    Fail,
}

#[derive(Clone, Debug)]
pub struct Report {
    pub status: Status,
    pub text: String,
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let tag = match self.status {
            Status::Ok => "ok",
            Status::Warn => "warning",
            Status::Fail => "ERROR",
        };
        write!(f, "[{tag}] {}", self.text)
    }
}

fn report(ok: bool, warn_only: bool, text: String) -> Report {
    let status = if ok {
        Status::Ok
    } else if warn_only {
        Status::Warn
    } else {
        Status::Fail
    };
    Report { status, text }
}

/// Fingerprint of the poses of all frames (without boil or shake).
pub fn poses_digest(shot: &Shot, tl: &Timeline) -> String {
    let mut h = StableHasher::new();
    for n in 0..tl.len() {
        h.add(shot.pose_signature(tl, n).as_str());
    }
    h.finish().to_string()
}

/// Feet do not slide: points of a planted foot touching the ground stay put
/// between neighbouring drawings. The figure is measured as seen in the frame
/// (after squash and stretch).
pub fn feet(shot: &Shot, tl: &Timeline) -> Report {
    let mut drawn: Vec<usize> = (0..tl.len()).map(|n| shot.drawn_frame(tl, n)).collect();
    drawn.dedup();
    let (mut worst, mut at) = (0.0f64, 0usize);
    for (i, actor) in shot.actors.iter().enumerate() {
        for pair in drawn.windows(2) {
            let (w0, w1) = (tl.world[pair[0]], tl.world[pair[1]]);
            let side = actor.planted_foot(w0);
            if side.is_none() || side != actor.planted_foot(w1) {
                continue;
            }
            let foot = actor.model.rig.foot(side.unwrap()).expect("foot");
            let (a, b) = (shot.posed(i, w0).0, shot.posed(i, w1).0);
            for j in [foot.heel, foot.toe] {
                let (p, q) = (a.joints[j], b.joints[j]);
                if p.y < ON_GROUND && q.y < ON_GROUND {
                    let slide = Vec3 { y: 0.0, ..(q - p) }.length();
                    if slide > worst {
                        (worst, at) = (slide, pair[1]);
                    }
                }
            }
        }
    }
    report(
        worst <= SLIDE_WARN,
        true,
        format!("planted foot slide: up to {:.1} mm per drawing (frame {at}), threshold {:.0} mm", worst * 1000.0, SLIDE_WARN * 1000.0),
    )
}

/// The camera does not jerk within a setup: angular speed changes smoothly.
/// Shake from hits is excluded — it jerks the camera on purpose.
pub fn camera(shot: &Shot, tl: &Timeline) -> Report {
    let mut prev: Option<(usize, Vec3, Option<f64>)> = None;
    let (mut worst, mut at) = (0.0f64, 0usize);
    for n in 0..tl.len() {
        let clock = tl.clock(n);
        let cut = shot.camera.pick(&clock).0;
        let fwd = shot.camera.frame(&clock, shot.size, Vec3::ZERO, shot).fwd;
        prev = match prev {
            Some((c, f0, speed0)) if c == cut => {
                let speed = m::acos(f0.dot(fwd).clamp(-1.0, 1.0)).to_degrees();
                if let Some(s0) = speed0 {
                    let jerk = (speed - s0).abs();
                    if jerk > worst {
                        (worst, at) = (jerk, n);
                    }
                }
                Some((cut, fwd, Some(speed)))
            }
            _ => Some((cut, fwd, None)),
        };
    }
    report(worst <= JERK_WARN, true, format!("camera jerk within a setup: up to {worst:.2}°/frame per frame (frame {at}), threshold {JERK_WARN}"))
}

/// Changing the seed changes the drawing (boil, shake) but not the poses.
pub fn seed_independence(shot: &Shot, other: &Shot) -> Report {
    let (a, b) = (poses_digest(shot, &shot.timeline()), poses_digest(other, &other.timeline()));
    report(a == b, false, format!("another seed — same poses: {}… and {}…", &a[..16], &b[..16]))
}

/// Contact residuals measured on the final poses, including deformations.
/// Intentional overreach is a warning; a failed support is a hard failure.
pub fn contacts(shot: &Shot, tl: &Timeline) -> Report {
    use mavian_scene::{ContactStatus, Priority};
    let (mut count, mut worst, mut at, mut support_failure) = (0usize, 0.0f64, 0usize, false);
    let mut last = None;
    for n in 0..tl.len() {
        let drawn = shot.drawn_frame(tl, n);
        if last == Some(drawn) {
            continue;
        }
        last = Some(drawn);
        for (_, r) in shot.contact_reports(tl.world[drawn]) {
            if matches!(r.status, ContactStatus::Unreachable | ContactStatus::Conflict | ContactStatus::Invalid | ContactStatus::Obstructed) {
                count += 1;
                if r.error > worst || !r.error.is_finite() {
                    worst = r.error;
                    at = n;
                }
                support_failure |= r.priority == Priority::Support;
            }
        }
    }
    Report {
        status: if count == 0 {
            Status::Ok
        } else if support_failure {
            Status::Fail
        } else {
            Status::Warn
        },
        text: format!("contacts: {count} unresolved samples, maximum residual {:.1} mm (frame {at})", worst * 1000.0),
    }
}

/// Acquisition decisions are independent of the frame grid; a failed attempt
/// remains visible even after its reach constraint has faded out.
pub fn grabs(shot: &Shot) -> Report {
    let failed: Vec<_> = shot.grab_reports.iter().filter(|r| !r.attached()).collect();
    let details = failed.iter().map(|r| format!("{} at {:.3}s: {:?} ({:.1} mm)", shot.props[r.prop].name, r.at, r.status, r.error * 1000.0))
        .collect::<Vec<_>>().join("; ");
    Report { status: if failed.is_empty() { Status::Ok } else { Status::Warn },
        text: format!("grabs: {} attached, {} rejected{}{}", shot.grab_reports.len() - failed.len(), failed.len(),
            if details.is_empty() { "" } else { "; " }, details) }
}


/// Placement decisions persist independently of the display frame grid.
pub fn placements(shot: &Shot) -> Report {
    let failed: Vec<_> = shot.placement_reports.iter().filter(|r| !r.placed()).collect();
    let details = failed
        .iter()
        .map(|r| {
            format!("{} on {} at {:.3}s: {:?} ({:.1} mm)", shot.props[r.prop].name, shot.props[r.support].name, r.at, r.status, r.error * 1000.0)
        })
        .collect::<Vec<_>>()
        .join("; ");
    Report {
        status: if failed.is_empty() { Status::Ok } else { Status::Warn },
        text: format!(
            "placements: {} placed, {} rejected{}{}",
            shot.placement_reports.len() - failed.len(),
            failed.len(),
            if details.is_empty() { "" } else { "; " },
            details
        ),
    }
}
