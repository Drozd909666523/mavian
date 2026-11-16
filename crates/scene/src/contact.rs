//! Timed, prioritized limb contacts and their residuals. Targets are sampled
//! from world time; the solver has no state carried between frames.
use crate::{Target, props::World};
use mavian_core::{Track, Vec3};
use mavian_rig::Figure;
use std::collections::BTreeSet;

/// Pose changes yield to contacts; contacts yield to fixed supports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Pose,
    Contact,
    Support,
}

#[derive(Clone, Debug)]
pub struct Contact {
    pub name: String,
    pub chain: [usize; 3],
    pub target: Target,
    /// Active interval [from, to); weight controls approach and withdrawal.
    pub from: f64,
    pub to: f64,
    pub weight: Track<f64>,
    pub priority: Priority,
    pub tolerance: f64,
    pub pole: Option<Vec3>,
    /// Target-relative waypoint for a curved approach and withdrawal.
    pub approach: Option<Vec3>,
    /// Action-generated return route; authored approach offsets remain symmetric.
    pub(crate) withdrawal: Option<(f64, Option<Vec3>)>,
    /// An attached toe: preserve this vector from the ankle after solving.
    pub follower: Option<(usize, Vec3)>,
}
impl Contact {
    pub fn new(name: &str, chain: [usize; 3], target: Target, from: f64, to: f64) -> Self {
        Self {
            name: name.into(),
            chain,
            target,
            from,
            to,
            weight: Track::Const(1.0),
            priority: Priority::Contact,
            tolerance: 0.002,
            pole: None,
            approach: None,
            withdrawal: None,
            follower: None,
        }
    }
    pub fn weight_at(&self, w: f64) -> f64 {
        if w < self.from || w >= self.to { 0.0 } else { self.weight.at(w).clamp(0.0, 1.0) }
    }
    pub(crate) fn goal(&self, reference: &Figure, target: Vec3, w: f64) -> Vec3 {
        let weight = self.weight_at(w);
        let offset = match self.withdrawal {
            Some((at, offset)) if w > at => offset,
            _ => self.approach,
        };
        let start = reference.joints[self.chain[2]];
        if let Some(offset) = offset {
            start.lerp(target + offset, weight).lerp((target + offset).lerp(target, weight), weight)
        } else {
            start.lerp(target, weight)
        }
    }
    pub fn validate(&self, rig: &mavian_rig::Rig) -> Result<(), String> {
        if self.name.is_empty() || !self.from.is_finite() || !self.to.is_finite() || self.from >= self.to {
            return Err("contact needs a name and a finite, increasing active interval".into());
        }
        if !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err(format!("contact '{}': tolerance must be positive and finite", self.name));
        }
        let weights: Vec<_> = match &self.weight {
            Track::Const(v) => vec![*v],
            Track::Keys(k) => k.items().iter().map(|(_, v, _)| *v).collect(),
        };
        if weights.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v)) {
            return Err(format!("contact '{}': weights must be finite and in [0, 1]", self.name));
        }
        let [a, b, c] = self.chain;
        if [a, b, c].iter().any(|&j| j >= rig.joints().len()) || rig.parent(b) != Some(a) || rig.parent(c) != Some(b) {
            return Err(format!("contact '{}': expected three consecutive joints", self.name));
        }
        if let Some((j, v)) = self.follower
            && (j >= rig.joints().len() || rig.parent(j) != Some(c) || !finite(v))
        {
            return Err(format!("contact '{}': invalid follower", self.name));
        }
        if self.approach.is_some_and(|p| !finite(p)) {
            return Err(format!("contact '{}': approach must be finite", self.name));
        }
        if self.pole.is_some_and(|p| !finite(p)) {
            return Err(format!("contact '{}': pole must be finite", self.name));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContactStatus {
    Satisfied,
    Blending,
    Unreachable,
    Conflict,
    Invalid,
    Obstructed,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ContactReport {
    pub name: String,
    pub chain: [usize; 3],
    pub priority: Priority,
    pub weight: f64,
    pub target: Vec3,
    /// Target after blending from the authored pose.
    pub goal: Vec3,
    pub actual: Vec3,
    pub error: f64,
    /// Minimum remaining distance imposed by this chain's length limits.
    pub reach_error: f64,
    pub status: ContactStatus,
}

pub fn finite(p: Vec3) -> bool {
    p.x.is_finite() && p.y.is_finite() && p.z.is_finite()
}

/// Carry the authored bend along the shortest swing of the limb axis.
/// Projecting the upper bone onto the new endpoint axis can reverse the bend
/// when a nearby contact pulls the hand behind its authored elbow.
fn carried_bend(reference: &Figure, chain: [usize; 3], direction: Vec3, fallback: Option<Vec3>) -> Vec3 {
    let [a, b, c] = chain;
    let original = (reference.joints[c] - reference.joints[a]).norm();
    let upper = reference.joints[b] - reference.joints[a];
    let bend = upper - original * upper.dot(original);
    if original.length() < 1e-8 || direction.length() < 1e-8 || bend.length() < 1e-8 {
        return fallback.unwrap_or(upper);
    }
    let bend = bend.norm();
    let axis = original.cross(direction);
    let cosine = original.dot(direction).clamp(-1.0, 1.0);
    if cosine < -1.0 + 1e-8 {
        // For a half turn, use the authored bend as the rotation axis.
        return bend;
    }
    bend + axis.cross(bend) + axis.cross(axis.cross(bend)) * (1.0 / (1.0 + cosine))
}

/// Two-bone solve. Blending happens at the endpoint, never at the joints:
/// interpolation of joint positions would shorten both bones during a ramp.
pub(crate) fn place(joints: &mut [Vec3], reference: &Figure, chain: [usize; 3], target: Vec3, pole: Option<Vec3>) -> f64 {
    let [a, b, c] = chain;
    let (root, elbow, end) = (joints[a], joints[b], joints[c]);
    let (l1, l2) = ((reference.joints[b] - reference.joints[a]).length(), (reference.joints[c] - reference.joints[b]).length());
    let delta = target - root;
    let requested = delta.length();
    let min = (l1 - l2).abs();
    let max = l1 + l2;
    let distance = requested.clamp(min, max);
    let error = (requested - distance).abs();
    if l1 < 1e-10 || l2 < 1e-10 {
        return requested;
    }
    let direction = if requested > 1e-10 {
        delta * (1.0 / requested)
    } else if (end - root).length() > 1e-10 {
        (end - root).norm()
    } else {
        Vec3::UP
    };
    let mut bend = pole.unwrap_or(elbow - root);
    bend = bend - direction * bend.dot(direction);
    if bend.length() < 1e-8 {
        let axis = if direction.y.abs() < 0.9 { Vec3::UP } else { Vec3 { x: 1.0, y: 0.0, z: 0.0 } };
        bend = direction.cross(axis);
    }
    if distance < 1e-10 {
        joints[b] = root + bend.norm() * l1;
        joints[c] = root;
        return error;
    }
    let along = (l1 * l1 - l2 * l2 + distance * distance) / (2.0 * distance);
    let height = (l1 * l1 - along * along).max(0.0).sqrt();
    joints[b] = root + direction * along + bend.norm() * height;
    joints[c] = root + direction * distance;
    error
}

/// Solve resolved targets in descending priority. Equal priorities have a
/// stable name/chain/target order; input vector order never breaks a tie.
/// The root and bone lengths are hard constraints, so unreachable contacts
/// report their residual rather than moving a support or stretching a limb.
pub fn solve_resolved(fig: &Figure, contacts: impl IntoIterator<Item = (Contact, Vec3)>, w: f64) -> (Figure, Vec<ContactReport>) {
    solve_resolved_from(fig, fig, contacts, w)
}

pub fn solve_resolved_from(
    fig: &Figure,
    reference: &Figure,
    contacts: impl IntoIterator<Item = (Contact, Vec3)>,
    w: f64,
) -> (Figure, Vec<ContactReport>) {
    solve_protected(fig, reference, contacts, w, BTreeSet::new())
}

fn solve_protected(
    fig: &Figure,
    reference: &Figure,
    contacts: impl IntoIterator<Item = (Contact, Vec3)>,
    w: f64,
    mut protected: BTreeSet<usize>,
) -> (Figure, Vec<ContactReport>) {
    let mut active: Vec<_> = contacts
        .into_iter()
        .filter_map(|(c, t)| {
            let weight = c.weight_at(w);
            (weight > 0.0).then_some((c, t, weight))
        })
        .collect();
    active.sort_by(|(a, ta, wa), (b, tb, wb)| {
        b.priority
            .cmp(&a.priority)
            .then(a.name.cmp(&b.name))
            .then(a.chain.cmp(&b.chain))
            .then(ta.x.total_cmp(&tb.x))
            .then(ta.y.total_cmp(&tb.y))
            .then(ta.z.total_cmp(&tb.z))
            .then(wb.total_cmp(wa))
            .then_with(|| format!("{a:?}").cmp(&format!("{b:?}")))
    });
    let mut out = fig.clone();
    let mut reports = Vec::new();
    for (contact, target, weight) in active {
        let [_, b, c] = contact.chain;
        let goal = contact.goal(reference, target, w);
        let mut status = ContactStatus::Satisfied;
        let mut reach_error = 0.0;
        if !finite(target) || !finite(goal) || !(goal - out.joints[contact.chain[0]]).length().is_finite() {
            status = ContactStatus::Invalid;
        } else if [Some(b), Some(c), contact.follower.map(|f| f.0)].into_iter().flatten().any(|j| protected.contains(&j)) {
            // Redundant constraints at the same anchor need not be rejected.
            if (out.joints[c] - goal).length() > contact.tolerance {
                status = ContactStatus::Conflict;
            }
        } else {
            let direction = (goal - out.joints[contact.chain[0]]).norm();
            let authored = carried_bend(reference, contact.chain, direction, contact.pole);
            let pole = contact.pole.map(|p| {
                let from = (authored - direction * authored.dot(direction)).norm();
                let to = (p - direction * p.dot(direction)).norm();
                if from.length() < 1e-8 || to.length() < 1e-8 {
                    return p;
                }
                let angle = mavian_core::m::atan2(direction.dot(from.cross(to)), from.dot(to)) * weight;
                from * mavian_core::m::cos(angle) + direction.cross(from) * mavian_core::m::sin(angle)
            });
            // A previous reach may have fully straightened this chain. Keep
            // its authored bend instead of choosing a new plane from roundoff.
            let pole = pole.or(Some(authored));
            reach_error = place(&mut out.joints, reference, contact.chain, goal, pole);
            if let Some((follower, offset)) = contact.follower {
                out.joints[follower] = out.joints[c] + offset;
            }
            if reach_error > contact.tolerance {
                status = ContactStatus::Unreachable;
            } else if weight < 1.0 - 1e-8 {
                status = ContactStatus::Blending;
            }
            protected.extend(contact.chain);
            if let Some((follower, _)) = contact.follower {
                protected.insert(follower);
            }
        }
        reports.push(ContactReport {
            name: contact.name,
            chain: contact.chain,
            priority: contact.priority,
            weight,
            target,
            goal,
            actual: out.joints[c],
            error: (out.joints[c] - goal).length(),
            reach_error,
            status,
        });
    }
    // Measure the result after all constraints, rather than each intermediate.
    for r in &mut reports {
        r.actual = out.joints[r.chain[2]];
        r.error = (r.actual - r.goal).length();
    }
    (out, reports)
}

pub fn solve(fig: &Figure, contacts: &[Contact], w: f64, world: &World) -> (Figure, Vec<ContactReport>) {
    let valid = validate_dependencies(w, world).is_ok();
    let targets = contacts.iter().filter(|c| c.weight_at(w) > 0.0).map(|c| {
        let target = if valid { c.target.at(w, world) } else { Vec3 { x: f64::NAN, y: 0.0, z: 0.0 } };
        (c.clone(), target)
    });
    solve_resolved(fig, targets, w)
}

impl crate::Shot {
    /// The same contact solution used by rendering, with explicit residuals.
    pub fn contact_reports(&self, w: f64) -> Vec<(usize, ContactReport)> {
        let mut reports: Vec<_> = self
            .actors
            .iter()
            .enumerate()
            .flat_map(|(i, _)| {
                let (fig, _) = self.posed(i, w);
                let (base, m) = self.deformed(i, w);
                self.solve_actor_contacts(i, w, &base, m).1.into_iter().map(move |mut r| {
                    r.actual = fig.joints[r.chain[2]];
                    r.error = (r.actual - r.goal).length();
                    if r.error > 0.002 && matches!(r.status, ContactStatus::Satisfied | ContactStatus::Blending) {
                        r.status = ContactStatus::Obstructed;
                    }
                    (i, r)
                })
            })
            .collect();
        for (i, p) in self.props.iter().enumerate() {
            fn held(m: &crate::Motion) -> Option<(usize, usize, Vec3)> {
                match m {
                    crate::Motion::Held { actor, hand, offset, .. } => Some((*actor, *hand, *offset)),
                    crate::Motion::Above { held: m, .. } => held(m),
                    _ => None,
                }
            }
            if let Some(m) = p.segment(w)
                && let Some((actor, hand, offset)) = held(m)
            {
                let a = &self.actors[actor];
                let heading = a.facing.at(w);
                let actual = self.posed(actor, w).0.joints[hand];
                let target = self.prop_rest(i, w).0 - mavian_core::Mat3::rot_y(heading) * offset;
                let error = (actual - target).length();
                if let Ok(chain) = crate::actions::chain(a, hand) {
                    reports.push((
                        actor,
                        ContactReport {
                            name: format!("hold:{}", p.name),
                            chain,
                            priority: Priority::Contact,
                            weight: 1.0,
                            target,
                            goal: target,
                            actual,
                            error,
                            reach_error: 0.0,
                            status: if error <= 0.002 { ContactStatus::Satisfied } else { ContactStatus::Conflict },
                        },
                    ));
                }
            }
        }
        for r in self.obstacle_reports(w) {
            let goal = r.point + r.normal * (-r.clearance);
            reports.push((
                r.actor,
                ContactReport {
                    name: format!("collision:{}:{}", self.props[r.prop].name, self.actors[r.actor].model.rig.joints()[r.bone[1]].name),
                    chain: r.chain,
                    priority: Priority::Support,
                    weight: 1.0,
                    target: goal,
                    goal,
                    actual: r.point,
                    error: -r.clearance,
                    reach_error: 0.0,
                    status: ContactStatus::Obstructed,
                },
            ));
        }
        reports
    }

    pub(crate) fn solve_actor_contacts(&self, index: usize, w: f64, fig: &Figure, m: Option<mavian_core::Mat3>) -> (Figure, Vec<ContactReport>) {
        let actor = &self.actors[index];
        let mut contacts: Vec<_> = actor.contacts.iter().chain(&actor.auto_contacts).filter(|c| c.weight_at(w) > 0.0).cloned().collect();
        if contacts.is_empty() {
            return (fig.clone(), Vec::new());
        }
        let raw = actor.figure(w);
        // Supports are immutable world anchors at this sample. They precede
        // hand contacts and keep their toes attached during the final solve.
        if let Some(side) = actor.planted_foot(w)
            && let Some(foot) = actor.model.rig.foot(side)
            && let Ok(chain) = crate::actions::chain(actor, foot.heel)
        {
            let mut support = Contact::new("support:planted", chain, Target::Point(raw.joints[foot.heel]), w - 1.0, w + 1.0);
            support.priority = Priority::Support;
            support.follower = Some((foot.toe, raw.joints[foot.toe] - raw.joints[foot.heel]));
            contacts.push(support);
        }
        let reference = if actor.pose_lock {
            fig.clone()
        } else {
            let legacy = crate::ik::reach_all(&raw, &actor.reaches, &actor.avoids, w, &self.world());
            legacy.stretched_about(m, actor.stretch_pivot(&actor.figure_ik(w, &self.world())))
        };
        let resolved: Vec<_> = contacts
            .into_iter()
            .map(|c| {
                let target = self.contact_target(&c.target, w);
                (c, target)
            })
            .collect();
        if actor.pose_lock {
            return solve_protected(fig, &reference, resolved, w, (0..fig.joints.len()).collect());
        }
        let adapted = self.adapt_body(index, w, fig, &reference, &resolved);
        solve_resolved_from(&adapted, &reference, resolved, w)
    }

    fn contact_target(&self, target: &Target, w: f64) -> Vec3 {
        match target {
            Target::OnProp { prop, local } => {
                let (p, r) = self.prop_rest(*prop, w);
                p + r * *local
            }
            Target::Offset(target, offset) => self.contact_target(target, w) + *offset,
            _ => target.at(w, &self.world()),
        }
    }

    /// Props attached to a hand follow its final contact-constrained pose,
    /// including squash/stretch. Other motions keep their authored semantics.
    pub fn prop_rest(&self, index: usize, w: f64) -> (Vec3, mavian_core::Mat3) {
        fn motion(shot: &crate::Shot, m: &crate::Motion, w: f64) -> (Vec3, mavian_core::Mat3) {
            use crate::Motion;
            use mavian_core::Mat3;
            match m {
                Motion::Held { actor, hand, offset, yaw } => {
                    let a = &shot.actors[*actor];
                    let heading = a.facing.at(w);
                    let p = shot.posed(*actor, w).0.joints[*hand];
                    (p + Mat3::rot_y(heading) * *offset, Mat3::rot_y(heading + yaw))
                }
                Motion::Above { held, floor } => {
                    let (p, r) = motion(shot, held, w);
                    (Vec3 { y: p.y.max(*floor), ..p }, r)
                }
                Motion::Ride { support, offset, yaw, drop, t0 } => {
                    let (p, r) = shot.prop_rest(*support, w);
                    let fall = (drop - 0.5 * crate::actor::G * (w - t0).max(0.0) * (w - t0).max(0.0)).max(0.0);
                    (p + r * (*offset + Vec3::UP * fall), r * Mat3::rot_y(*yaw))
                }
                _ => m.at(w, &shot.world()),
            }
        }
        let p = &self.props[index];
        p.segment(w).map_or((p.at, mavian_core::Mat3::rot_y(p.turn)), |m| motion(self, m, w))
    }
}

/// Validate references before evaluating targets; report cycles instead of
/// recursively re-entering an actor's memoized pose computation.
pub fn validate_target(target: &Target, world: &World) -> Result<(), String> {
    let actor_joint = |actor: usize, joint: usize| {
        let a = world.actors.get(actor).ok_or_else(|| format!("contact target actor {actor} does not exist"))?;
        if joint >= a.model.rig.joints().len() { Err(format!("contact target joint {joint} does not exist on actor {actor}")) } else { Ok(()) }
    };
    match target {
        Target::Point(p) if !finite(*p) => Err("contact target must be finite".into()),
        Target::Descend { at, over, down, hover } if !finite(*at) || !over.is_finite() || !down.is_finite() || !hover.is_finite() => {
            Err("contact descend target must be finite".into())
        }
        Target::OnProp { prop, local } => {
            if *prop >= world.props.len() {
                Err(format!("contact target prop {prop} does not exist"))
            } else if !finite(*local) {
                Err("contact prop offset must be finite".into())
            } else {
                Ok(())
            }
        }
        Target::Aim { target, joint, attacker, .. } => {
            actor_joint(*target, *joint)?;
            actor_joint(*attacker, 0)
        }
        Target::Block { defender, head, .. } => actor_joint(*defender, *head),
        Target::BlockContact { defender, hand, .. } => actor_joint(*defender, *hand),
        Target::Offset(t, offset) => {
            if !finite(*offset) {
                return Err("contact offset must be finite".into());
            }
            validate_target(t, world)
        }
        _ => Ok(()),
    }
}

fn prop_dependencies(prop: usize, w: f64, world: &World, seen: &mut BTreeSet<usize>, out: &mut Vec<usize>) -> Result<(), String> {
    if !seen.insert(prop) {
        return Err(format!("cyclic prop support at prop {prop}"));
    }
    fn motion(m: &crate::Motion, w: f64, world: &World, seen: &mut BTreeSet<usize>, out: &mut Vec<usize>) -> Result<(), String> {
        match m {
            crate::Motion::Held { actor, .. } => out.push(*actor),
            crate::Motion::Above { held, .. } => motion(held, w, world, seen, out)?,
            crate::Motion::Ride { support, .. } => prop_dependencies(*support, w, world, seen, out)?,
            _ => {}
        }
        Ok(())
    }
    let p = world.props.get(prop).ok_or_else(|| format!("prop {prop} does not exist"))?;
    if let Some(m) = p.segment(w) {
        motion(m, w, world, seen, out)?;
    }
    seen.remove(&prop);
    Ok(())
}
fn dependencies(target: &Target, w: f64, world: &World) -> Result<Vec<usize>, String> {
    validate_target(target, world)?;
    let mut result = Vec::new();
    match target {
        Target::OnProp { prop, .. } => prop_dependencies(*prop, w, world, &mut BTreeSet::new(), &mut result)?,
        Target::BlockContact { defender, .. } => result.push(*defender),
        Target::Offset(t, _) => result = dependencies(t, w, world)?,
        _ => {}
    }
    Ok(result)
}
pub fn validate_dependencies(w: f64, world: &World) -> Result<(), String> {
    fn visit(i: usize, w: f64, world: &World, visiting: &mut BTreeSet<usize>, done: &mut BTreeSet<usize>) -> Result<(), String> {
        if done.contains(&i) {
            return Ok(());
        }
        if !visiting.insert(i) {
            return Err(format!("cyclic contact dependency at actor {i}, world time {w:.6}"));
        }
        let actor = world.actors.get(i).ok_or_else(|| format!("actor {i} does not exist"))?;
        for c in actor.contacts.iter().chain(&actor.auto_contacts).filter(|c| c.weight_at(w) > 0.0) {
            for dep in dependencies(&c.target, w, world)? {
                visit(dep, w, world, visiting, done)?;
            }
        }
        for r in actor.reaches.iter().filter(|r| r.weight.at(w) > 0.0) {
            for dep in dependencies(&r.target, w, world)? {
                visit(dep, w, world, visiting, done)?;
            }
        }
        visiting.remove(&i);
        done.insert(i);
        Ok(())
    }
    let mut done = BTreeSet::new();
    for i in 0..world.actors.len() {
        visit(i, w, world, &mut BTreeSet::new(), &mut done)?;
    }
    Ok(())
}
