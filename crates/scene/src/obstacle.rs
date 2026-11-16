//! Oriented prop distance queries and limb clearance diagnostics.
use crate::{Actor, Motion, Shape, Shot};
use mavian_core::{Mat3, Vec3};

#[derive(Clone, Debug)]
pub struct Obstacle {
    pub prop: usize,
    pub shape: Shape,
    pub pos: Vec3,
    pub rot: Mat3,
}
impl Obstacle {
    /// Signed surface distance and outward world normal.
    pub fn distance(&self, p: Vec3) -> (f64, Vec3) {
        let p = self.rot.transpose() * (p - self.pos);
        let (d, n) = match self.shape {
            Shape::Sphere { r } => {
                let delta = p - Vec3::UP * r;
                let len = delta.length();
                (len - r, if len > 1e-12 { delta * (1.0 / len) } else { Vec3::UP })
            }
            Shape::Box { sx, sy, sz } => {
                let p = p - Vec3::UP * (sy * 0.5);
                let h = Vec3 { x: sx * 0.5, y: sy * 0.5, z: sz * 0.5 };
                let q = Vec3 { x: p.x.abs() - h.x, y: p.y.abs() - h.y, z: p.z.abs() - h.z };
                let outside = Vec3 { x: q.x.max(0.0), y: q.y.max(0.0), z: q.z.max(0.0) };
                let len = outside.length();
                let sign = |x: f64| if x < 0.0 { -1.0 } else { 1.0 };
                let normal = if len > 1e-12 {
                    Vec3 { x: outside.x * sign(p.x), y: outside.y * sign(p.y), z: outside.z * sign(p.z) } * (1.0 / len)
                } else if q.x >= q.y && q.x >= q.z {
                    Vec3 { x: sign(p.x), y: 0.0, z: 0.0 }
                } else if q.y >= q.z {
                    Vec3 { x: 0.0, y: sign(p.y), z: 0.0 }
                } else {
                    Vec3 { x: 0.0, y: 0.0, z: sign(p.z) }
                };
                (len + q.x.max(q.y).max(q.z).min(0.0), normal)
            }
        };
        (d, self.rot * n)
    }
    /// Minimum clearance of a tapered capsule. Distance to these convex
    /// shapes is convex along a segment, so interval minimization also catches
    /// intersections between the endpoints of a bone.
    pub fn capsule(&self, a: Vec3, ra: f64, b: Vec3, rb: f64) -> (f64, f64, Vec3) {
        let at = |u: f64| self.distance(a.lerp(b, u)).0 - (ra + (rb - ra) * u);
        let (mut lo, mut hi) = (0.0, 1.0);
        for _ in 0..24 {
            let x = lo + (hi - lo) / 3.0;
            let y = hi - (hi - lo) / 3.0;
            if at(x) < at(y) {
                hi = y;
            } else {
                lo = x;
            }
        }
        let mut u = (lo + hi) * 0.5;
        for e in [0.0, 1.0] {
            if at(e) < at(u) {
                u = e;
            }
        }
        (at(u), u, self.distance(a.lerp(b, u)).1)
    }
}

pub(crate) fn joint_radius(actor: &Actor, joint: usize, line: f64) -> f64 {
    use mavian_rig::Part;
    actor
        .model
        .parts
        .iter()
        .filter_map(|p| match p {
            Part::Ball { joint: j, radius } | Part::Head { joint: j, radius, .. } if *j == joint => Some(*radius),
            Part::Limb { chain, radii, .. } => chain.iter().position(|&j| j == joint).map(|i| radii[i]),
            Part::Stick { a, b, w0, w1, .. } if *a == joint || *b == joint => Some(line * 0.5 * if *a == joint { *w0 } else { *w1 }),
            _ => None,
        })
        .fold(0.0, f64::max)
        * actor.scale
}

#[derive(Clone, Debug, PartialEq)]
pub struct ObstacleReport {
    pub actor: usize,
    pub prop: usize,
    pub chain: [usize; 3],
    pub bone: [usize; 2],
    /// Negative means penetration; positive means separation.
    pub clearance: f64,
    pub point: Vec3,
    pub normal: Vec3,
}
impl Shot {
    pub fn obstacle_reports(&self, w: f64) -> Vec<ObstacleReport> {
        let mut out = Vec::new();
        for (i, a) in self.actors.iter().enumerate() {
            let fig = self.posed(i, w).0;
            for side in ["l", "r"] {
                let Ok(hand) = a.model.rig.joint(&format!("hand_{side}")) else { continue };
                let Ok(chain) = crate::actions::chain(a, hand) else { continue };
                for mut o in self.arm_obstacles(i, hand, w, &fig) {
                    (o.pos, o.rot) = self.prop_rest(o.prop, w);
                    for [start, end] in [[chain[0], chain[1]], [chain[1], chain[2]]] {
                        let (clearance, u, normal) = o.capsule(
                            fig.joints[start],
                            joint_radius(a, start, self.style.line),
                            fig.joints[end],
                            joint_radius(a, end, self.style.line),
                        );
                        if clearance < -0.001 {
                            out.push(ObstacleReport {
                                actor: i,
                                prop: o.prop,
                                chain,
                                bone: [start, end],
                                clearance,
                                point: fig.joints[start].lerp(fig.joints[end], u),
                                normal,
                            });
                        }
                    }
                }
            }
        }
        out
    }
    pub(crate) fn arm_obstacles(&self, actor: usize, _hand: usize, w: f64, fig: &mavian_rig::Figure) -> Vec<Obstacle> {
        self.props
            .iter()
            .enumerate()
            .map(|(i, p)| {
                fn attached(m: &Motion, actor: usize, fig: &mavian_rig::Figure, heading: f64) -> Option<(Vec3, Mat3)> {
                    match m {
                        Motion::Held { actor: a, hand, offset, yaw } if *a == actor => {
                            Some((fig.joints[*hand] + Mat3::rot_y(heading) * *offset, Mat3::rot_y(heading + yaw)))
                        }
                        Motion::Above { held, floor } => attached(held, actor, fig, heading).map(|(p, r)| (Vec3 { y: p.y.max(*floor), ..p }, r)),
                        _ => None,
                    }
                }
                let (pos, rot) =
                    p.segment(w).and_then(|m| attached(m, actor, fig, self.actors[actor].facing.at(w))).unwrap_or_else(|| p.rest(w, &self.world()));
                Obstacle { prop: i, shape: p.shape.clone(), pos, rot }
            })
            .collect()
    }
}

fn clearance(obstacles: &[Obstacle], joints: &[Vec3], chain: [usize; 3], radii: [f64; 3]) -> f64 {
    let mut result = f64::INFINITY;
    for o in obstacles {
        // Endpoints reject distant props before the capsule minimizer runs.
        let center = joints[chain[1]];
        if o.distance(center).0
            > (joints[chain[0]] - center).length().max((joints[chain[2]] - center).length()) + radii.into_iter().fold(0.0, f64::max) + 0.01
        {
            continue;
        }
        for k in 0..2 {
            result = result.min(o.capsule(joints[chain[k]], radii[k], joints[chain[k + 1]], radii[k + 1]).0);
        }
    }
    result
}

impl Shot {
    /// Pose-space avoidance: preserve the shoulder and both bone lengths.
    /// Prefer changing the bend plane; move the endpoint only when necessary.
    pub(crate) fn avoid_props(&self, index: usize, w: f64, figure: &mavian_rig::Figure) -> mavian_rig::Figure {
        if self.props.is_empty() {
            return figure.clone();
        }
        let actor = &self.actors[index];
        let mut out = figure.clone();
        for side in ["l", "r"] {
            let Ok(hand) = actor.model.rig.joint(&format!("hand_{side}")) else { continue };
            let Ok(chain) = crate::actions::chain(actor, hand) else { continue };
            fn held_by(m: &Motion, actor: usize, hand: usize) -> bool {
                match m {
                    Motion::Held { actor: a, hand: h, .. } => *a == actor && *h == hand,
                    Motion::Above { held, .. } => held_by(held, actor, hand),
                    _ => false,
                }
            }
            let holding = self.props.iter().any(|p| p.segment(w).is_some_and(|m| held_by(m, index, hand)));
            // Holding a prop does not by itself authorize a new bend plane.
            // Preserve the solved/authored arm unless clearance requires a change.
            let obstacles = self.arm_obstacles(index, hand, w, &out);
            let radii = chain.map(|j| joint_radius(actor, j, self.style.line));
            if clearance(&obstacles, &out.joints, chain, radii) >= -0.0001 {
                continue;
            }
            let original = out.clone();
            let root = out.joints[chain[0]];
            let original_goal = out.joints[hand];
            let outward = if holding {
                actor.model.rig.parent(chain[0]).map(|parent| {
                    original.rots[parent] * Vec3 { x: actor.model.rig.joints()[chain[0]].side, y: 0.0, z: 0.0 }
                }).unwrap_or(Vec3::ZERO)
            } else { Vec3::ZERO };
            let cost = |candidate: &mavian_rig::Figure| {
                let hand_delta = candidate.joints[hand] - original_goal;
                let elbow_delta = candidate.joints[chain[1]] - original.joints[chain[1]];
                // Prefer the outside route when a held prop requires clearance.
                let medial = (root - candidate.joints[chain[1]]).dot(outward).max(0.0);
                hand_delta.dot(hand_delta) * 8.0 + elbow_delta.dot(elbow_delta) + medial * medial * 8.0
            };
            let mut goal = original_goal;
            let mut best = out.clone();
            let mut best_clear = clearance(&obstacles, &best.joints, chain, radii);
            let mut best_cost = f64::INFINITY;
            for _ in 0..12 {
                for o in &obstacles {
                    let (d, n) = o.distance(goal);
                    if d < radii[2] + 0.002 {
                        goal = goal + n * (radii[2] + 0.002 - d);
                    }
                }
                let direction = (goal - root).norm();
                let mut bend = original.joints[chain[1]] - root;
                bend = (bend - direction * bend.dot(direction)).norm();
                if bend.length() < 1e-8 {
                    bend = direction.cross(Vec3::UP).norm();
                }
                if bend.length() < 1e-8 {
                    bend = Vec3 { x: 1.0, y: 0.0, z: 0.0 };
                }
                let side = direction.cross(bend).norm();
                let mut chosen_angle = 0.0;
                for k in 0..32 {
                    // Try the original bend first, then both directions around
                    // the analytic elbow circle, including the opposite bend.
                    let angle = if k % 2 == 0 { k as f64 * 0.5 } else { -((k + 1) as f64) * 0.5 } * std::f64::consts::TAU / 32.0;
                    let pole = bend * mavian_core::m::cos(angle) + side * mavian_core::m::sin(angle);
                    let mut candidate = original.clone();
                    crate::contact::place(&mut candidate.joints, &original, chain, goal, Some(pole));
                    let free = clearance(&obstacles, &candidate.joints, chain, radii);
                    let cost = cost(&candidate);
                    if (free >= -0.0001 && (best_clear < -0.0001 || cost < best_cost)) || (best_clear < -0.0001 && free > best_clear) {
                        best = candidate;
                        best_clear = free;
                        best_cost = cost;
                        chosen_angle = angle;
                    }
                }
                if best_clear >= -0.0001 {
                    // Refine the angular search so the elbow does not step
                    // between the coarse samples as the target moves.
                    for level in 1..=9 {
                        let delta = std::f64::consts::TAU / 32.0 * mavian_core::m::powi(0.5, level);
                        for angle in [chosen_angle - delta, chosen_angle + delta] {
                            let pole = bend * mavian_core::m::cos(angle) + side * mavian_core::m::sin(angle);
                            let mut candidate = original.clone();
                            crate::contact::place(&mut candidate.joints, &original, chain, goal, Some(pole));
                            let free = clearance(&obstacles, &candidate.joints, chain, radii);
                            let cost = cost(&candidate);
                            if free >= -0.0001 && cost < best_cost {
                                best = candidate;
                                best_clear = free;
                                best_cost = cost;
                                chosen_angle = angle;
                            }
                        }
                    }
                    break;
                }
                // A projected endpoint can be clamped by reach limits. Use the
                // actual closest bone point to steer the next feasible target.
                let mut push = Vec3::ZERO;
                for o in &obstacles {
                    for k in 0..2 {
                        let (d, _, n) = o.capsule(best.joints[chain[k]], radii[k], best.joints[chain[k + 1]], radii[k + 1]);
                        if d < 0.002 {
                            push = push + n * (0.002 - d);
                        }
                    }
                }
                goal = best.joints[hand] + push;
            }
            out = best;
        }
        out
    }
}
