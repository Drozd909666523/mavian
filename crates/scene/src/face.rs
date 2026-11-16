//! The face on a ball head: the first try of the "trust dial" (a port of the
//! MVP's `engine/face.py`).
//!
//! Eyes and mouth can be placed two ways:
//!
//! - honestly (trust = 1) — as drawings on the surface of the ball: they are
//!   projected, shrink at the edge and go over the horizon;
//! - by rules (trust = 0) — in the screen plane, from the head's turn angle:
//!   features slide to the edge slower than by geometry, barely narrow and
//!   keep their height.
//!
//! Values in between blend the two drawings point by point.

use crate::camera::Camera;
use mavian_core::m;
use mavian_core::noise::smooth;
use mavian_core::{Mat3, Vec3, mix};
use mavian_rig::Face;

/// features of the face never go further than this fraction of the radius
const FACE_EDGE: f64 = 0.9;

/// The face of one head in screen pixels.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FaceShapes {
    /// filled eye outlines (closed)
    pub eyes: Vec<Vec<(f64, f64)>>,
    /// the mouth: an open line
    pub mouth: Vec<(f64, f64)>,
}

fn fade(angle: f64, (lo, hi): (f64, f64)) -> f64 {
    smooth((hi - angle.abs()) / (hi - lo))
}

fn wrap(a: f64) -> f64 {
    (a + 180.0).rem_euclid(360.0) - 180.0
}

fn area(pts: &[(f64, f64)]) -> f64 {
    let n = pts.len();
    let s: f64 = (0..n)
        .map(|i| {
            let ((x0, y0), (x1, y1)) = (pts[i], pts[(i + 1) % n]);
            x0 * y1 - x1 * y0
        })
        .sum();
    s.abs() * 0.5
}

/// A feature entirely inside the head's outline: (centre, size factor).
///
/// If the feature pokes out, it moves towards the centre as a whole, keeping
/// its shape — instead of being pressed to the edge point by point (then it
/// flattens and looks turned). If it does not fit even at the centre, it
/// shrinks.
fn fit(cx: f64, cy: f64, offsets: &[(f64, f64)]) -> (f64, f64, f64) {
    let reach = |k: f64| offsets.iter().map(|&(ox, oy)| m::hypot(cx * k + ox, cy * k + oy)).fold(0.0, f64::max);
    if reach(1.0) <= FACE_EDGE {
        return (cx, cy, 1.0);
    }
    if reach(0.0) > FACE_EDGE {
        return (0.0, 0.0, FACE_EDGE / reach(0.0));
    }
    let (mut lo, mut hi) = (0.0, 1.0);
    for _ in 0..20 {
        let mid = (lo + hi) / 2.0;
        if reach(mid) <= FACE_EDGE {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (cx * lo, cy * lo, 1.0)
}

/// Eyes and mouth of a head with centre `center`, orientation `rot` (columns:
/// left, up, forward) and world radius `radius`. `trust` — 0 by rules, 1 an
/// honest projection; `blink` — 0 open, 1 closed.
pub fn face(cam: &Camera, center: Vec3, rot: &Mat3, radius: f64, fc: &Face, trust: f64, blink: f64) -> Option<FaceShapes> {
    let (ax_l, ax_u, ax_f) = (rot.col(0), rot.col(1), rot.col(2));
    let c = cam.project(center)?;
    let r = radius * c.scale;
    let rad = f64::to_radians;

    // where the camera looks at the head from, in the head's own frame
    let view = (cam.pos - center).norm();
    let theta = m::atan2(view.dot(ax_l), view.dot(ax_f)).to_degrees();
    let phi = m::asin(view.dot(ax_u).clamp(-1.0, 1.0)).to_degrees();
    // by rules pitch acts with saturation (see Face::pitch_limit)
    let phi = fc.pitch_limit * m::tanh(phi / fc.pitch_limit);

    // the face's axes on screen: "up" — where the crown points, "sideways" — square to it
    let (ux, uy) = (ax_u.dot(cam.right), -ax_u.dot(cam.up));
    let un = m::hypot(ux, uy);
    let (ux, uy) = if un > 1e-6 { (ux / un, uy / un) } else { (0.0, -1.0) };
    let (sx, sy) = (-uy, ux);

    // a point on the ball → screen offset and facing (plus — towards the camera)
    let honest = |az: f64, el: f64| -> (f64, f64, f64) {
        let (az, el) = (rad(az), rad(el));
        let mut d = ax_f * (m::cos(el) * m::cos(az)) + ax_l * (m::cos(el) * m::sin(az)) + ax_u * m::sin(el);
        let facing = d.dot(view);
        if facing < 0.0 {
            d = (d - view * facing).norm(); // pressed to the edge
        }
        match cam.project(center + d * radius) {
            Some(q) => (q.x - c.x, q.y - c.y, facing),
            None => (0.0, 0.0, -1.0),
        }
    };
    // a point in the face's axes (in head radii) → screen offset
    let ruled = |x: f64, y: f64| (r * (x * sx + y * ux), r * (x * sy + y * uy));
    let place = |(x, y): (f64, f64)| (c.x + x, c.y + y);

    let trust = trust.clamp(0.0, 1.0);
    let open_eye = 1.0 - 0.9 * blink.clamp(0.0, 1.0); // a blink flattens the eye
    let sin_t2 = m::powi(m::sin(rad(theta)), 2);
    let pull_eye = 1.0 - fc.eye_pull * sin_t2;
    let pull_mouth = 1.0 - fc.mouth_pull * sin_t2;
    let mut out = FaceShapes::default();

    // --- eyes ---
    for side in [1.0, -1.0] {
        let az0 = side * fc.eye_az;
        let psi = wrap(az0 - theta);
        let vis = fade(psi, fc.eye_hide);
        let xc = pull_eye * m::sin(rad(psi)) * m::cos(rad(fc.eye_el));
        let yc = m::sin(rad(fc.eye_el - fc.pitch_gain * phi));
        // the eye narrows only turning away from the view: while it turns to
        // the camera (the near eye at the start of a turn) it does not grow
        let away = psi.abs().max(fc.eye_az).min(90.0);
        let half_w = m::sin(rad(fc.eye_rw)) * vis * (0.45 + 0.55 * m::cos(rad(away)));
        // turning away, the eye shrinks
        let mut half_h = m::sin(rad(fc.eye_rh)) * vis.sqrt();
        // from below or above the eye is seen at an angle and flattens in
        // height — otherwise it looks like a sticker on a round head
        half_h *= mix(1.0, m::cos(rad(fc.eye_el - phi)).max(0.0), fc.pitch_gain);
        half_h *= open_eye;
        let (_, _, center_facing) = honest(az0, fc.eye_el);
        let n = fc.eye_n;
        let ring: Vec<(f64, f64)> = (0..n)
            .map(|k| {
                let a = std::f64::consts::TAU * k as f64 / n as f64;
                (half_w * m::cos(a), half_h * m::sin(a))
            })
            .collect();
        let (xc, yc, k_fit) = fit(xc, yc, &ring);
        let pts: Vec<(f64, f64)> = (0..n)
            .map(|k| {
                let a = std::f64::consts::TAU * k as f64 / n as f64;
                let (rx, ry) = ruled(xc + ring[k].0 * k_fit, yc + ring[k].1 * k_fit);
                let (hx, hy, _) = if center_facing > 0.0 {
                    honest(az0 + fc.eye_rw * m::cos(a) / m::cos(rad(fc.eye_el)), fc.eye_el + fc.eye_rh * open_eye * m::sin(a))
                } else {
                    honest(az0, fc.eye_el) // over the horizon: pulled into a point
                };
                (mix(rx, hx, trust), mix(ry, hy, trust))
            })
            .collect();
        if area(&pts) > 0.004 * r * r {
            out.eyes.push(pts.into_iter().map(place).collect());
        }
    }

    // --- mouth ---
    let vis = fade(theta, fc.mouth_hide);
    let xm = pull_mouth * m::sin(rad(-theta)) * m::cos(rad(fc.mouth_el));
    let ym = m::sin(rad(fc.mouth_el - fc.pitch_gain * phi));
    let half_m = m::sin(rad(fc.mouth_half)) * vis * (0.35 + 0.65 * m::cos(rad(theta)).max(0.0));
    let lift = m::sin(rad(fc.smile)) * (vis * 3.0).min(1.0);
    let n = fc.mouth_n.max(2);
    let ts: Vec<f64> = (0..n).map(|k| 2.0 * k as f64 / (n - 1) as f64 - 1.0).collect();
    let curve: Vec<(f64, f64)> = ts.iter().map(|&t| (half_m * t, lift * t * t)).collect();
    let (xm, ym, k_fit) = fit(xm, ym, &curve);
    let mut pts = Vec::new();
    for (k, &t) in ts.iter().enumerate() {
        let (rx, ry) = ruled(xm + curve[k].0 * k_fit, ym + curve[k].1 * k_fit);
        let (hx, hy, facing) = honest(fc.mouth_half * t, fc.mouth_el + fc.smile * t * t);
        if facing < 0.0 && (trust > 0.999 || vis <= 0.0) {
            continue; // an honest mouth over the horizon
        }
        pts.push((mix(rx, hx, trust), mix(ry, hy, trust)));
    }
    if pts.len() > 1 {
        let (a, b) = (pts[0], pts[pts.len() - 1]);
        if m::hypot(b.0 - a.0, b.1 - a.1) > 0.03 * r {
            out.mouth = pts.into_iter().map(place).collect();
        }
    }
    Some(out)
}
