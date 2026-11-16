//! Bounded upper-body assistance for contacts, evaluated independently at each time.
use crate::{Actor, Contact, Priority, Shot, contact::finite, obstacle::Obstacle};
use mavian_core::{Mat3, Vec3};
use mavian_rig::{Figure, Part};

struct Goal {
    root: usize,
    point: Vec3,
    min: f64,
    max: f64,
    tolerance: f64,
}
impl Goal {
    fn deficit(&self, root: Vec3) -> f64 {
        let d = (self.point - root).length();
        (d - d.clamp(self.min, self.max)).abs()
    }
}

impl Shot {
    pub(crate) fn adapt_body(&self, index: usize, w: f64, fig: &Figure, reference: &Figure, contacts: &[(Contact, Vec3)]) -> Figure {
        let actor = &self.actors[index];
        if actor.pose_lock || actor.body_reach <= 0.0 {
            return fig.clone();
        }
        let obstacles = self.arm_obstacles(index, 0, w, fig);
        assist(actor, w, fig, reference, contacts, &obstacles)
    }
}

pub(crate) fn assist(actor: &Actor, w: f64, fig: &Figure, reference: &Figure, contacts: &[(Contact, Vec3)], obstacles: &[Obstacle]) -> Figure {
    let rig = &actor.model.rig;
    let Ok(spine) = rig.joint("spine") else { return fig.clone() };
    if actor.pose_lock || actor.body_reach <= 0.0 {
        return fig.clone();
    }
    let upper: Vec<_> = (0..fig.joints.len())
        .map(|j| {
            let mut at = Some(j);
            while let Some(j) = at {
                if j == spine {
                    return true;
                }
                at = rig.parent(j);
            }
            false
        })
        .collect();
    // An explicitly pinned upper-body chain takes precedence over assistance.
    if contacts.iter().any(|(c, _)| c.priority == Priority::Support && c.chain.iter().any(|&j| upper[j])) {
        return fig.clone();
    }
    let mut ordered: Vec<_> = contacts.iter().filter(|(c, _)| c.priority == Priority::Contact && upper[c.chain[0]]).collect();
    ordered.sort_by(|(a, _), (b, _)| a.name.cmp(&b.name).then(a.chain.cmp(&b.chain)));
    // Reserve a small bend before the arm locks straight; this also avoids
    // the two-bone singularity during contact ramps.
    let goals: Vec<_> = ordered
        .iter()
        .filter_map(|(c, target)| {
            let point = c.goal(reference, *target, w);
            if !finite(point) {
                return None;
            }
            let [a, b, d] = c.chain;
            let l1 = (reference.joints[b] - reference.joints[a]).length();
            let l2 = (reference.joints[d] - reference.joints[b]).length();
            Some(Goal { root: a, point, min: (l1 - l2).abs(), max: (l1 + l2 - 0.015 * actor.scale).max((l1 - l2).abs()), tolerance: c.tolerance })
        })
        .collect();
    if goals.iter().all(|g| g.deficit(fig.joints[g.root]) <= g.tolerance * 0.25) {
        return fig.clone();
    }
    let limit = actor.body_reach * ordered.iter().map(|(c, _)| c.weight_at(w)).fold(0.0, f64::max);
    let pivot = fig.joints[spine];
    let facing = Mat3::rot_y(actor.facing.at(w));
    let rotation = |angles: [f64; 3]| facing * Mat3::rot_y(angles[2]) * Mat3::rot_z(angles[1]) * Mat3::rot_x(angles[0]) * facing.transpose();
    let transform = |angles| {
        let r = rotation(angles);
        let mut out = fig.clone();
        for (j, &moves) in upper.iter().enumerate() {
            if moves {
                out.joints[j] = pivot + r * (fig.joints[j] - pivot);
                out.rots[j] = r * fig.rots[j];
            }
        }
        // Counter-rotate the neck to retain the authored gaze and neck length.
        if let Ok(head) = rig.joint("head")
            && let Some(neck) = rig.parent(head)
            && upper[neck]
        {
            out.joints[head] = out.joints[neck] + fig.joints[head] - fig.joints[neck];
            out.rots[head] = fig.rots[head];
        }
        out
    };
    // Conservative torso capsules and head spheres prevent assistance from
    // introducing or deepening an existing body/prop intersection.
    let bodies: Vec<_> = actor
        .model
        .parts
        .iter()
        .filter_map(|part| match part {
            Part::Body { a, b, rx, rz, .. } if upper[*b] => Some((*a, *b, rx.0.max(rz.0) * actor.scale, rx.1.max(rz.1) * actor.scale)),
            Part::Head { joint, radius, .. } if upper[*joint] => Some((*joint, *joint, radius * actor.scale, radius * actor.scale)),
            _ => None,
        })
        .collect();
    let clearance = |f: &Figure, o: &Obstacle, (a, b, ra, rb): (usize, usize, f64, f64)| o.capsule(f.joints[a], ra, f.joints[b], rb).0;
    let guards: Vec<_> = obstacles.iter().flat_map(|o| bodies.iter().map(move |&body| (o, body, clearance(fig, o, body).min(0.0)))).collect();
    let residuals = |angles: [f64; 3]| {
        let r = rotation(angles);
        goals.iter().map(|g| (g.deficit(pivot + r * (fig.joints[g.root] - pivot)) - g.tolerance * 0.25).max(0.0)).collect::<Vec<_>>()
    };
    let regularizer = 1e-7 * actor.scale * actor.scale;
    let cost = |angles: [f64; 3]| residuals(angles).iter().map(|r| r * r).sum::<f64>() + angles.iter().map(|a| a * a).sum::<f64>() * regularizer;
    let mut angles = [0.0; 3];
    let mut best = cost(angles);
    // Regularized Gauss-Newton seeks a small correction from the
    // authored pose. Solve all axes together to avoid frame-dependent
    // switches between equivalent tilt/turn combinations.
    for _ in 0..64 {
        let residual = residuals(angles);
        let derivatives: Vec<_> = (0..3)
            .map(|axis| {
                let mut plus = angles;
                let mut minus = angles;
                plus[axis] += 0.01;
                minus[axis] -= 0.01;
                residuals(plus).iter().zip(residuals(minus)).map(|(a, b)| (a - b) / 0.02).collect::<Vec<_>>()
            })
            .collect();
        let mut normal = [[0.0; 4]; 3];
        for i in 0..3 {
            for j in 0..3 {
                normal[i][j] = derivatives[i].iter().zip(&derivatives[j]).map(|(a, b)| a * b).sum::<f64>();
            }
            normal[i][i] += regularizer;
            normal[i][3] = -derivatives[i].iter().zip(&residual).map(|(a, b)| a * b).sum::<f64>() - regularizer * angles[i];
        }
        for i in 0..3 {
            let divisor = normal[i][i];
            for value in normal[i].iter_mut().skip(i) {
                *value /= divisor;
            }
            for j in 0..3 {
                if i == j {
                    continue;
                }
                let factor = normal[j][i];
                for k in i..4 {
                    normal[j][k] -= factor * normal[i][k];
                }
            }
        }
        let mut accepted = false;
        for level in 0..20 {
            let step = mavian_core::m::powi(0.5, level);
            let mut candidate = std::array::from_fn::<_, 3, _>(|i| angles[i] + normal[i][3] * step);
            let tilt = (candidate[0] * candidate[0] + candidate[1] * candidate[1]).sqrt();
            if tilt > limit {
                candidate[0] *= limit / tilt;
                candidate[1] *= limit / tilt;
            }
            candidate[2] = candidate[2].clamp(-limit * (2.0 / 3.0), limit * (2.0 / 3.0));
            let value = cost(candidate);
            if value >= best - 1e-16 * actor.scale * actor.scale {
                continue;
            }
            let f = transform(candidate);
            if guards.iter().any(|&(o, body, floor)| clearance(&f, o, body) < floor - 1e-6) {
                continue;
            }
            angles = candidate;
            best = value;
            accepted = true;
            break;
        }
        if !accepted {
            break;
        }
    }
    transform(angles)
}
