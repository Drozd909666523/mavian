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

pub(crate) fn clearance(obstacles: &[Obstacle], joints: &[Vec3], chain: [usize; 3], radii: [f64; 3]) -> f64 {
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
        for guide in &actor.clearance_guides {
            if let Some(point) = guide.at(w) && point.active {
                let parent = actor.model.rig.parent(guide.chain[0]).unwrap_or(guide.chain[0]);
                let reference = out.clone();
                let pole = out.rots[parent] * point.pole;
                crate::contact::place(&mut out.joints, &reference, guide.chain, point.wrist, Some(pole));
            }
        }
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
            let lift = self.held_lift(index, hand, w, &out);
            if lift > 1e-8 {
                let reference = out.clone();
                let target = out.joints[hand] + Vec3::UP * lift;
                let pole = out.joints[chain[1]] - out.joints[chain[0]];
                crate::contact::place(&mut out.joints, &reference, chain, target, Some(pole));
            }
            // Holding a prop does not by itself authorize a new bend plane.
            // Preserve the solved/authored arm unless clearance requires a change.
            let obstacles = self.arm_obstacles(index, hand, w, &out);
            let radii = chain.map(|j| joint_radius(actor, j, self.style.line));
            let action_owned = holding || actor.auto_contacts.iter().any(|c| c.chain == chain && c.weight_at(w) > 0.0);
            let bodies = if action_owned { crate::clearance::body_solids(actor, &out) } else { Vec::new() };
            let candidate_obstacles = |candidate: &mavian_rig::Figure| {
                if holding { self.arm_obstacles(index, hand, w, candidate) } else { obstacles.clone() }
            };
            let free_arm = |candidate: &mavian_rig::Figure| {
                let props = candidate_obstacles(candidate);
                let free = bodies.iter().fold(clearance(&props, &candidate.joints, chain, radii), |free, (_, body)| {
                    free.min(crate::clearance::body_arm(body, candidate, chain, radii).0)
                });
                let lift = if holding { self.held_lift(index, hand, w, candidate) } else { 0.0 };
                if lift > 1e-8 { free.min(-lift) } else { free }
            };

            if free_arm(&out) >= -0.0001 {
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
            let guide = actor.clearance_guides.iter().find(|g| g.chain == chain).and_then(|g| g.at(w)).filter(|p| p.active);
            let required = if guide.is_some() { 0.002 } else { -0.0001 };
            let preserve_branch = guide.is_some() && !actor.auto_contacts.iter().any(|c| c.chain == chain && c.weight_at(w) >= 1.0 - 1e-8)
                && !actor.contacts.iter().any(|c| c.priority == crate::Priority::Support && c.weight_at(w) > 0.0 && c.chain.contains(&hand));
            let preferred_elbow = guide
                .map(|direction| {
                    let parent = actor.model.rig.parent(chain[0]).unwrap_or(chain[0]);
                    root + (original.rots[parent] * direction.pole).norm() * (original.joints[chain[1]] - root).length()
                }).unwrap_or(original.joints[chain[1]]);
            let cost = |candidate: &mavian_rig::Figure| {
                let hand_delta = candidate.joints[hand] - original_goal;
                let elbow_delta = candidate.joints[chain[1]] - preferred_elbow;
                // Prefer the outside route when a held prop requires clearance.
                let medial = (root - candidate.joints[chain[1]]).dot(outward).max(0.0);
                hand_delta.dot(hand_delta) * 8.0 + elbow_delta.dot(elbow_delta) + medial * medial * 8.0
            };
            let mut goal = original_goal;
            let mut best = out.clone();
            let mut best_clear = free_arm(&best);
            let mut best_cost = f64::INFINITY;
            for _ in 0..12 {
                for o in &obstacles {
                    if self.props[o.prop].segment(w).is_some_and(|m| held_by(m, index, hand)) { continue; }
                    let (d, n) = o.distance(goal);
                    if d < radii[2] + 0.002 {
                        goal = goal + n * (radii[2] + 0.002 - d);
                    }
                }
                for (_, body) in &bodies {
                    let (d, n) = body.distance(goal, radii[2]);
                    if d < 0.002 { goal = goal + n * (0.002 - d); }
                }
                let direction = (goal - root).norm();
                let mut bend = preferred_elbow - root;
                bend = (bend - direction * bend.dot(direction)).norm();
                if bend.length() < 1e-8 {
                    bend = direction.cross(Vec3::UP).norm();
                }
                if bend.length() < 1e-8 {
                    bend = Vec3 { x: 1.0, y: 0.0, z: 0.0 };
                }
                let side = direction.cross(bend).norm();
                let mut chosen_angle = 0.0;
                let local_angles = (1..=7).flat_map(|level| {
                    let angle = std::f64::consts::TAU / 32.0 * mavian_core::m::powi(0.5, level);
                    [angle, -angle]
                });
                let coarse_angles = (0..32).map(|k| {
                    (if k % 2 == 0 { k as f64 * 0.5 } else { -((k + 1) as f64) * 0.5 }) * std::f64::consts::TAU / 32.0
                });
                // Resolve narrow clearance intervals beside the preceding bend
                // before accepting a distant branch from the coarse circle.
                for angle in local_angles.chain(coarse_angles) {
                    let pole = bend * mavian_core::m::cos(angle) + side * mavian_core::m::sin(angle);
                    let mut candidate = original.clone();
                    crate::contact::place(&mut candidate.joints, &original, chain, goal, Some(pole));
                    let free = free_arm(&candidate);
                    let cost = cost(&candidate);
                    if (free >= required && (best_clear < required || cost < best_cost)) || (best_clear < required && free > best_clear) {
                        best = candidate;
                        best_clear = free;
                        best_cost = cost;
                        chosen_angle = angle;
                    }
                }
                if best_clear >= required {
                    // Refine the angular search so the elbow does not step
                    // between the coarse samples as the target moves.
                    for level in 1..=9 {
                        let delta = std::f64::consts::TAU / 32.0 * mavian_core::m::powi(0.5, level);
                        for angle in [chosen_angle - delta, chosen_angle + delta] {
                            let pole = bend * mavian_core::m::cos(angle) + side * mavian_core::m::sin(angle);
                            let mut candidate = original.clone();
                            crate::contact::place(&mut candidate.joints, &original, chain, goal, Some(pole));
                            let free = free_arm(&candidate);
                            let cost = cost(&candidate);
                            if free >= required && cost < best_cost {
                                best = candidate;
                                best_clear = free;
                                best_cost = cost;
                                chosen_angle = angle;
                            }
                        }
                    }
                    if !preserve_branch || (best.joints[chain[1]] - preferred_elbow).length() < 0.04 * actor.scale {
                        break;
                    }
                }
                // A projected endpoint can be clamped by reach limits. Use the
                // actual closest bone point to steer the next feasible target.
                let mut steering = best.clone();
                if preserve_branch {
                    crate::contact::place(&mut steering.joints, &original, chain, goal, Some(bend));
                }
                let mut push = Vec3::ZERO;
                for o in &candidate_obstacles(&steering) {
                    for k in 0..2 {
                        let (d, _, n) = o.capsule(steering.joints[chain[k]], radii[k], steering.joints[chain[k + 1]], radii[k + 1]);
                        if d < 0.002 {
                            // Moving this wrist also moves its own prop. Its
                            // separation gradient has the opposite sign.
                            let coupled = self.props[o.prop].segment(w).is_some_and(|m| held_by(m, index, hand));
                            push = push + n * ((0.002 - d) * if coupled { -1.0 } else { 1.0 });
                        }
                    }
                }
                for (_, body) in &bodies {
                    let (d, _, n) = crate::clearance::body_arm(body, &steering, chain, radii);
                    if d < 0.002 { push = push + n * (0.002 - d); }
                }
                push = push + Vec3::UP * self.held_lift(index, hand, w, &steering);
                goal = steering.joints[hand] + push;
            }

            out = best;
        }
        out
    }
}
