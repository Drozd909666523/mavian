//! Coupled wrist/held-volume clearance and actor-body collision queries.
use crate::{ContactReport, ContactStatus, Motion, Priority, Shape, Shot};
use mavian_core::{Mat3, Vec3};

pub(crate) fn binding(m: &Motion) -> Option<(usize, usize, Vec3, Option<f64>)> {
    match m {
        Motion::Held { actor, hand, offset, .. } => Some((*actor, *hand, *offset, None)),
        Motion::Above { held, floor } => binding(held).map(|(a, h, offset, old)| (a, h, offset, Some(old.map_or(*floor, |y| y.max(*floor))))),
        _ => None,
    }
}

/// Horizontal separating-axis test. Boxes use their actual yawed footprint;
/// a sphere uses its projected circular support radius.
pub(crate) fn footprint(a: &Shape, ap: Vec3, ar: Mat3, b: &Shape, bp: Vec3, br: Mat3) -> bool {
    if let Shape::Sphere { r } = *a {
        let delta = br.transpose() * (ap - bp);
        return match *b {
            Shape::Sphere { r: other } => delta.x * delta.x + delta.z * delta.z < (r + other) * (r + other),
            Shape::Box { sx, sz, .. } => {
                let x = (delta.x.abs() - sx * 0.5).max(0.0);
                let z = (delta.z.abs() - sz * 0.5).max(0.0);
                x * x + z * z < r * r
            }
        };
    }
    if matches!(b, Shape::Sphere { .. }) {
        return footprint(b, bp, br, a, ap, ar);
    }
    let extent = |shape: &Shape, rot: Mat3, axis: Vec3| match *shape {
        Shape::Box { sx, sz, .. } => 0.5 * (sx * rot.col(0).dot(axis).abs() + sz * rot.col(2).dot(axis).abs()),
        Shape::Sphere { r } => r,
    };
    let delta = ap - bp;
    [ar.col(0), ar.col(2), br.col(0), br.col(2)].into_iter().all(|axis| delta.dot(axis).abs() < extent(a, ar, axis) + extent(b, br, axis) - 1e-8)
}

impl Shot {
    /// Required upward wrist correction. The prop stays rigidly attached:
    /// the solver moves the wrist, rather than clamping only the prop anchor.
    pub(crate) fn held_lift(&self, actor: usize, hand: usize, w: f64, fig: &mavian_rig::Figure) -> f64 {
        let obstacles = self.arm_obstacles(actor, hand, w, fig);
        let heading = Mat3::rot_y(self.actors[actor].facing.at(w));
        let mut lift = 0.0f64;
        for (prop, p) in self.props.iter().enumerate() {
            let Some((a, h, offset, floor)) = p.segment(w).and_then(binding) else { continue };
            if a != actor || h != hand {
                continue;
            }
            let anchor = fig.joints[hand] + heading * offset;
            let rot = obstacles[prop].rot;
            if let Some(floor) = floor {
                lift = lift.max(floor - anchor.y);
            }
            if self.ground.is_some() {
                lift = lift.max(-anchor.y);
            }
            for obstacle in &obstacles {
                if obstacle.prop == prop {
                    continue;
                }
                // A binding to another hand is not a supporting platform.
                if self.props[obstacle.prop].segment(w).and_then(binding).is_some() {
                    continue;
                }
                let top = obstacle.pos.y + obstacle.shape.top();
                if anchor.y >= obstacle.pos.y - 0.001
                    && anchor.y + p.shape.top() > obstacle.pos.y + 0.001
                    && footprint(&p.shape, anchor, rot, &obstacle.shape, obstacle.pos, obstacle.rot)
                {
                    lift = lift.max(top - anchor.y);
                }
            }
        }
        lift.max(0.0)
    }

    pub fn held_prop_reports(&self, w: f64) -> Vec<(usize, ContactReport)> {
        let mut reports = Vec::new();
        for (actor, a) in self.actors.iter().enumerate() {
            let fig = self.posed(actor, w).0;
            for name in ["hand_l", "hand_r"] {
                let Ok(hand) = a.model.rig.joint(name) else { continue };
                let error = self.held_lift(actor, hand, w, &fig);
                if error <= 0.001 {
                    continue;
                }
                let Ok(chain) = crate::actions::chain(a, hand) else { continue };
                let actual = fig.joints[hand];
                let goal = actual + Vec3::UP * error;
                reports.push((
                    actor,
                    ContactReport {
                        name: format!("volume:{name}"),
                        chain,
                        priority: Priority::Contact,
                        weight: 1.0,
                        target: goal,
                        goal,
                        actual,
                        error,
                        reach_error: 0.0,
                        status: ContactStatus::Obstructed,
                    },
                ));
            }
        }
        reports
    }
}

#[derive(Clone)]
pub(crate) enum BodySolid {
    Torso { a: Vec3, axis: Vec3, length: f64, side: Vec3, front: Vec3, rx: (f64, f64), rz: (f64, f64), cap: (f64, f64) },
    Ball { center: Vec3, radius: f64 },
}
impl BodySolid {
    /// The torso uses the same trimmed oval and rounded caps as frame geometry.
    /// Expanding its radii gives a smooth collision proxy for the arm capsule.
    pub(crate) fn distance(&self, point: Vec3, radius: f64) -> (f64, Vec3) {
        match *self {
            Self::Ball { center, radius: body } => {
                let delta = point - center;
                (delta.length() - body - radius, if delta.length() > 1e-10 { delta.norm() } else { Vec3::UP })
            }
            Self::Torso { a, axis, length, side, front, rx, rz, cap } => {
                let delta = point - a;
                let along = delta.dot(axis);
                let t = (along / length).clamp(0.0, 1.0);
                let x = rx.0 + (rx.1 - rx.0) * t + radius;
                let z = rz.0 + (rz.1 - rz.0) * t + radius;
                let y = if along < 0.0 { cap.0 + radius } else { cap.1 + radius };
                let beyond = if along < 0.0 {
                    along
                } else if along > length {
                    along - length
                } else {
                    0.0
                };
                let (sx, sz, sy) = (delta.dot(side) / x, delta.dot(front) / z, beyond / y.max(1e-8));
                let k = (sx * sx + sz * sz + sy * sy).sqrt();
                let gradient = side * (sx / x) + front * (sz / z) + axis * (sy / y.max(1e-8));
                let normal = if gradient.length() > 1e-10 { gradient.norm() } else { front };
                ((k - 1.0) * x.min(z).min(if beyond != 0.0 { y } else { f64::INFINITY }), normal)
            }
        }
    }
    pub(crate) fn capsule(&self, a: Vec3, ra: f64, b: Vec3, rb: f64) -> (f64, f64, Vec3) {
        let at = |u: f64| self.distance(a.lerp(b, u), ra + (rb - ra) * u);
        let (mut lo, mut hi) = (0.0, 1.0);
        for _ in 0..24 {
            let (x, y) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
            if at(x).0 < at(y).0 {
                hi = y;
            } else {
                lo = x;
            }
        }
        let mut u = (lo + hi) * 0.5;
        for end in [0.0, 1.0] {
            if at(end).0 < at(u).0 {
                u = end;
            }
        }
        let (d, n) = at(u);
        (d, u, n)
    }
}

pub(crate) fn body_solids(actor: &crate::Actor, fig: &mavian_rig::Figure) -> Vec<(String, BodySolid)> {
    use mavian_rig::Part;
    let mut solids = Vec::new();
    let sc = actor.scale;
    for part in &actor.model.parts {
        match *part {
            Part::Body { ref name, a, b, rx, rz, trim, cap } => {
                let axis = (fig.joints[b] - fig.joints[a]).norm();
                let start = fig.joints[a] + axis * (trim.0 * sc);
                let end = fig.joints[b] - axis * (trim.1 * sc);
                let side = (fig.rots[a].col(0) - axis * fig.rots[a].col(0).dot(axis)).norm();
                let front = axis.cross(side).norm();
                solids.push((
                    name.clone(),
                    BodySolid::Torso {
                        a: start,
                        axis,
                        length: (end - start).length().max(1e-8),
                        side,
                        front,
                        rx: (rx.0 * sc, rx.1 * sc),
                        rz: (rz.0 * sc, rz.1 * sc),
                        cap: (cap.0 * sc, cap.1 * sc),
                    },
                ));
            }
            Part::Head { joint, radius, .. } => solids.push(("head".into(), BodySolid::Ball { center: fig.joints[joint], radius: radius * sc })),
            _ => (),
        }
    }
    solids
}

pub(crate) fn body_arm(solid: &BodySolid, fig: &mavian_rig::Figure, chain: [usize; 3], radii: [f64; 3]) -> (f64, Vec3, Vec3) {
    let mut result = (f64::INFINITY, Vec3::ZERO, Vec3::ZERO);
    for k in 0..2 {
        let (a, b) = (fig.joints[chain[k]], fig.joints[chain[k + 1]]);
        // Proximal shoulder tissue belongs to the torso/arm attachment union.
        // Check the distal upper arm and the whole forearm/hand.
        let cut = if k == 0 { 0.6 } else { 0.0 };
        let start = a.lerp(b, cut);
        let radius = radii[k] + (radii[k + 1] - radii[k]) * cut;
        let (d, u, normal) = solid.capsule(start, radius, b, radii[k + 1]);
        if d < result.0 {
            result = (d, start.lerp(b, u), normal);
        }
    }
    result
}

impl Shot {
    pub fn self_collision_reports(&self, w: f64) -> Vec<(usize, ContactReport)> {
        let mut reports = Vec::new();
        for (actor, a) in self.actors.iter().enumerate() {
            let fig = self.posed(actor, w).0;
            let solids = body_solids(a, &fig);
            for name in ["hand_l", "hand_r"] {
                let Ok(hand) = a.model.rig.joint(name) else { continue };
                let Ok(chain) = crate::actions::chain(a, hand) else { continue };
                let radii = chain.map(|j| crate::obstacle::joint_radius(a, j, self.style.line));
                for (part, solid) in &solids {
                    let (d, actual, normal) = body_arm(solid, &fig, chain, radii);
                    if d >= -0.001 {
                        continue;
                    }
                    let goal = actual + normal * -d;
                    reports.push((
                        actor,
                        ContactReport {
                            name: format!("self:{part}:{name}"),
                            chain,
                            priority: Priority::Contact,
                            weight: 1.0,
                            target: goal,
                            goal,
                            actual,
                            error: -d,
                            reach_error: 0.0,
                            status: ContactStatus::Obstructed,
                        },
                    ));
                }
            }
        }
        reports
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Actor, Prop};

    #[test]
    fn yawed_box_and_sphere_footprints_do_not_invent_corner_support() {
        let long = Shape::Box { sx: 0.1, sy: 0.1, sz: 1.0 };
        let small = Shape::Box { sx: 0.1, sy: 1.0, sz: 0.1 };
        let offset = Vec3 { x: 0.45, y: 0.0, z: 0.0 };
        assert!(!footprint(&long, Vec3::ZERO, Mat3::IDENT, &small, offset, Mat3::IDENT));
        assert!(footprint(&long, Vec3::ZERO, Mat3::rot_y(90.0), &small, offset, Mat3::IDENT));
        let ball = Shape::Sphere { r: 0.1 };
        let cube = Shape::Box { sx: 0.2, sy: 1.0, sz: 0.2 };
        let corner = Vec3 { x: 0.18, y: 0.0, z: 0.18 };
        assert!(!footprint(&ball, corner, Mat3::IDENT, &cube, Vec3::ZERO, Mat3::IDENT));
        for angle in [0.0, 37.0, 123.0] {
            let r = Mat3::rot_y(angle);
            let origin = Vec3 { x: 3.0, y: 0.0, z: -2.0 };
            assert!(!footprint(&ball, origin + r * corner, r, &cube, origin, r));
        }
    }

    #[test]
    fn torso_proxy_uses_model_dimensions_and_follows_heading_and_scale() {
        for scale in [0.5, 1.0, 2.0] {
            for heading in [0.0, 73.0, 180.0] {
                let actor = Actor::new(mavian_models::mannequin::mannequin()).scale(scale).facing(heading);
                let fig = actor.figure(0.0);
                let solids = body_solids(&actor, &fig);
                let BodySolid::Torso { a, axis, length, front, .. } = solids[0].1 else { panic!() };
                let center = a + axis * (length * 0.5);
                assert!(solids[0].1.distance(center, 0.0).0 < -0.09 * scale);
                assert!(solids[0].1.distance(center + front * (0.3 * scale), 0.0).0 > 0.15 * scale);
            }
        }
    }

    #[test]
    fn pose_lock_keeps_a_penetrating_binding_and_reports_the_held_volume() {
        let mut actor = Actor::new(mavian_models::mannequin::mannequin());
        actor.pose_lock = true;
        let hand = actor.model.rig.joint("hand_r").unwrap();
        let raw = actor.figure(0.5);
        let pos = Vec3 { x: -0.3, y: 0.9, z: 0.35 };
        let mut cup = Prop::new("cup", Shape::Box { sx: 0.13, sy: 0.17, sz: 0.13 }, pos);
        cup.motion.push((0.0, Motion::Held { actor: 0, hand, offset: pos - raw.joints[hand], yaw: 0.0 }));
        let support = Prop::new("support", Shape::Box { sx: 0.4, sy: 1.0, sz: 0.4 }, Vec3 { y: 0.0, ..pos });
        let shot = Shot { actors: vec![actor], props: vec![cup, support], ..Shot::default() };
        assert_eq!(shot.posed(0, 0.5).0, raw);
        let reports = shot.held_prop_reports(0.5);
        assert_eq!(reports.len(), 1);
        assert!((reports[0].1.error - 0.1).abs() < 1e-8);
        assert_eq!(reports[0].1.status, ContactStatus::Obstructed);
    }

    #[test]
    fn a_locked_hand_inside_the_torso_is_reported_without_changing_the_pose() {
        let mut model = (*mavian_models::mannequin::mannequin()).clone();
        let inside = model.rig.pose().base(model.pose("stand").unwrap()).set("arm_r", (0, -30)).set("elbow_r", 0).build().unwrap();
        model.add_pose("inside", inside).unwrap();
        let mut actor = Actor::new(std::sync::Arc::new(model)).poses(crate::Poses::Fixed("inside".into()));
        actor.pose_lock = true;
        let raw = actor.figure(0.5);
        let shot = Shot { actors: vec![actor], ..Shot::default() };
        assert_eq!(shot.posed(0, 0.5).0, raw);
        let reports = shot.self_collision_reports(0.5);
        assert!(reports.iter().any(|(_, r)| r.name == "self:torso:hand_r" && r.error > 0.05));
    }
}

/// A compiled, world-time arm route. Playback never integrates frame history.
#[derive(Clone)]
pub(crate) struct ArmGuide {
    pub chain: [usize; 3],
    pub points: Vec<ArmPoint>,
}
#[derive(Clone, Copy)]
pub(crate) struct ArmPoint {
    pub at: f64,
    pub wrist: Vec3,
    pub pole: Vec3,
    pub active: bool,
}
impl ArmGuide {
    pub fn at(&self, w: f64) -> Option<ArmPoint> {
        let first = *self.points.first()?;
        let index = self.points.partition_point(|p| p.at <= w);
        if index == 0 {
            return Some(first);
        }
        let left = self.points[index - 1];
        let Some(right) = self.points.get(index) else { return Some(left) };
        let u = ((w - left.at) / (right.at - left.at)).clamp(0.0, 1.0);
        Some(ArmPoint {
            at: w,
            wrist: left.wrist.lerp(right.wrist, u),
            pole: left.pole.lerp(right.pole, u).norm(),
            active: left.active || right.active,
        })
    }
}
