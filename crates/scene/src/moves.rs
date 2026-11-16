//! Ready-made moves: sets of pose keys to insert into a track.

use crate::actor::{G, PoseKey, contact, pelvis_height, pk};
use mavian_core::Ease;
use mavian_rig::{Model, Pose, Side};

/// Walk keys for `Keys::extend`.
///
/// At `start` — the contact pose with the `lead` leg forward, then `steps`
/// steps of `step` seconds. The supporting foot stays on the ground and the pelvis
/// follows it, so feet do not slide. `stop` — the pose the character settles into
/// after the last step (the back leg comes up to the front one) over `stop_time`.
/// `prefix` — another pose set for the same walk (`push_contact_l`, …).
pub fn walk(start: f64, steps: usize, step: f64, lead: Side, stop: Option<&str>, stop_time: f64, prefix: &str) -> Vec<(f64, PoseKey, Ease)> {
    let name = |kind: &str, side: Side| format!("{prefix}{kind}_{}", side.suffix());
    let mut lead = lead;
    let mut out = vec![(start, pk(&name("contact", lead)).plant(lead.other()), Ease::Linear)];
    let mut t = start;
    for _ in 0..steps {
        // passing: weight on the front leg, the back leg swings forward
        out.push((t + step / 2.0, pk(&name("pass", lead)).plant(lead), Ease::Linear));
        // contact: the other leg is now in front, weight still on the previous one
        out.push((t + step, pk(&name("contact", lead.other())).plant(lead), Ease::Linear));
        lead = lead.other();
        t += step;
    }
    if let Some(stop) = stop {
        out.push((t + stop_time, pk(stop).plant(lead), Ease::Out));
    }
    out
}

/// When a walk started with the same parameters ends.
pub fn walk_end(start: f64, steps: usize, step: f64, stop_time: f64) -> f64 {
    start + steps as f64 * step + stop_time
}

/// The poses a jump goes through.
#[derive(Clone, Debug)]
pub struct JumpPoses {
    /// wind-up, held before the push
    pub crouch: String,
    /// the end of the push: legs extended, the moment of take-off
    pub launch: String,
    /// in the air, at the top
    pub air: String,
    /// touching the ground
    pub reach: String,
    /// the landing crouch
    pub land: String,
}

impl Default for JumpPoses {
    fn default() -> Self {
        JumpPoses { crouch: "crouch".into(), launch: "leap_launch".into(), air: "leap_air".into(), reach: "leap_reach".into(), land: "land".into() }
    }
}

/// A jump planned from intent: keys and the moments they give.
#[derive(Clone, Debug)]
pub struct Jump {
    pub keys: Vec<(f64, PoseKey, Ease)>,
    /// the push starts (the crouch begins to unfold)
    pub push: f64,
    /// take-off
    pub launch: f64,
    /// the top of the flight
    pub apex: f64,
    /// touching the ground
    pub landing: f64,
    /// the landing crouch is reached
    pub end: f64,
    /// how high the pelvis rises over its take-off height
    pub height: f64,
    /// take-off speed: forward and up, m/s
    pub speed: (f64, f64),
    /// how far along the push (crouch → launch) the body leaves the ground
    pub takeoff: f64,
}

/// Pelvis position relative to the planted foot: (forward, height).
fn pelvis_from_foot(model: &Model, scale: f64, pose: &Pose, side: Side) -> (f64, f64) {
    let (pts, _) = model.rig.fk(pose, 0.0, scale);
    let c = contact(model, scale, &pts, side, (0.0, 1.0));
    (pts[model.root].z - c.1, pelvis_height(model, pose, scale))
}

/// A jump from intent: take off at `at` from the `side` foot, rise `height`
/// (the pelvis over its take-off height) and land `distance` further on (from
/// the take-off foot to the landing foot).
///
/// Nothing else is set by hand. The two numbers give the take-off angle;
/// the body leaves the ground at the point of the push (the crouch unfolding
/// towards the launch pose) where the pelvis moves at that angle — a pose's
/// geometry may carry the pelvis down at its very end (a leg already behind
/// the body), and a jump does not take off from there. The push's duration
/// gives the speed for the height; the flight's time follows from gravity;
/// the landing crouch lasts as long as it takes to absorb the forward speed.
/// The pelvis keeps its velocity through take-off, flight and landing (the
/// flight itself is `Move::Fly`: ballistic, from the speed at take-off).
pub fn jump(model: &Model, scale: f64, at: f64, distance: f64, height: f64, side: Side, poses: &JumpPoses) -> Result<Jump, String> {
    let pose = |n: &str| model.pose(n).map_err(|e| e.0);
    let (crouch, launch, reach, land) = (pose(&poses.crouch)?, pose(&poses.launch)?, pose(&poses.reach)?, pose(&poses.land)?);
    pose(&poses.air)?;
    if model.rig.foot(side).is_none() {
        return Err(format!("model {} has no foot to jump from", model.name));
    }
    if height <= 0.0 || distance <= 0.0 {
        return Err("jump: distance and height must be positive".into());
    }
    let g = G * scale;
    let e = 1e-5;
    let path = |u: f64| pelvis_from_foot(model, scale, &crouch.blend(launch, u), side);
    // the landing: reach → land. A foot reaching forward makes the pose
    // vault the pelvis up over it first; after an impact the pelvis does not
    // rise, so the foot touches down where the crouch stops lifting it. The
    // fall's vertical speed ends in the impact (squash, shake); the forward
    // speed carries on into the crouch
    let fall = |u: f64| pelvis_from_foot(model, scale, &reach.blend(land, u), side);
    let rising = |u: f64| fall(u + e).1 > fall(u).1;
    let ul = if !rising(0.0) {
        0.0
    } else {
        let (mut a, mut b) = (0.0, 0.95);
        if rising(b) {
            return Err(format!("jump: the landing from '{}' to '{}' never lowers the pelvis", poses.reach, poses.land));
        }
        for _ in 0..40 {
            let mid = 0.5 * (a + b);
            if rising(mid) {
                a = mid;
            } else {
                b = mid;
            }
        }
        b
    };
    let touch = fall(ul);
    let vy = (2.0 * g * height).sqrt();
    // taking off at blend u of the push: the push lasts tau, accelerating
    // (ease in); returns (distance covered, tau, flight time, vx)
    let plan = |u: f64| -> Option<(f64, f64, f64, f64)> {
        let q = |s: f64| path(u * Ease::In.apply(s));
        let (q1, q0) = (q(1.0), q(1.0 - e));
        let dq = ((q1.0 - q0.0) / e, (q1.1 - q0.1) / e);
        if dq.0 <= 0.0 || dq.1 <= 0.0 {
            return None;
        }
        let tau = dq.1 / vy;
        let vx = dq.0 / tau;
        let disc = vy * vy + 2.0 * g * (q1.1 - touch.1);
        if disc <= 0.0 {
            return None;
        }
        let t = (vy + disc.sqrt()) / g;
        Some((vx * t - touch.0 + q1.0, tau, t, vx))
    };
    // the later the take-off along the push, the flatter it is and the
    // further it goes for this height: find where it covers the distance
    let us: Vec<f64> = (1..=200).map(|k| k as f64 / 200.0).collect();
    let ok: Vec<(f64, f64)> = us.iter().filter_map(|&u| plan(u).map(|p| (u, p.0))).collect();
    let Some(&(_, best)) = ok.iter().max_by(|a, b| a.1.total_cmp(&b.1)) else {
        return Err(format!("jump: the push from '{}' to '{}' never carries the pelvis forward and up", poses.crouch, poses.launch));
    };
    let least = ok.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    if distance > best || distance < least {
        return Err(format!(
            "jump: rising {height} m, this push lands from {least:.2} to {best:.2} m away, not {distance} m; \
             change the height or the distance"
        ));
    }
    let (mut a, mut b) = (0.0, 0.0);
    for w in ok.windows(2) {
        if (w[0].1 - distance) * (w[1].1 - distance) <= 0.0 {
            (a, b) = (w[0].0, w[1].0);
            break;
        }
    }
    let side_of = |u: f64| plan(u).map_or(f64::NAN, |p| p.0 - distance);
    let lo_sign = side_of(a) < 0.0;
    for _ in 0..50 {
        let mid = 0.5 * (a + b);
        if (side_of(mid) < 0.0) == lo_sign {
            a = mid;
        } else {
            b = mid;
        }
    }
    let u = 0.5 * (a + b);
    let (_, tau, flight, vx) = plan(u).expect("found inside the valid range");
    // the landing crouch: from the touchdown on to land, decelerating (ease
    // out), as long as it takes to carry the flight's forward speed into it
    let landing = |s: f64| fall(ul + (1.0 - ul) * Ease::Out.apply(s));
    let (l0, l1) = (landing(0.0), landing(e));
    let dl = (l1.0 - l0.0) / e;
    if dl <= 0.0 {
        return Err(format!("jump: the landing from '{}' to '{}' must carry the pelvis forward", poses.reach, poses.land));
    }
    if landing(0.0).1 < 0.0 {
        return Err("jump: the landing pose is below the ground".into());
    }
    let absorb = dl / vx;
    let t_land = at + flight;
    let apex = at + vy / g;
    let keys = vec![
        (at - tau, pk(&poses.crouch).plant(side), Ease::Linear),
        (at, pk(&poses.crouch).toward(&poses.launch, u).plant(side), Ease::In),
        (apex, pk(&poses.air).fly(), Ease::Out),
        (t_land, pk(&poses.reach).toward(&poses.land, ul).fly(), Ease::In),
        (t_land + absorb, pk(&poses.land).plant(side), Ease::Out),
    ];
    Ok(Jump { keys, push: at - tau, launch: at, apex, landing: t_land, end: t_land + absorb, height, speed: (vx, vy), takeoff: u })
}
