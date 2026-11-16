//! Final-pose, contact-gated transfer from a hand to a named support.
use crate::{Action, Contact, ContactStatus, Motion, Onto, Shape, Shot, Target, actions, clearance::binding};
use mavian_core::{Ease, Keys, Mat3, Vec3};

const TOLERANCE: f64 = 0.002;
const PENETRATION: f64 = 0.001;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlacementStatus {
    Placed,
    NotHeld,
    Unreachable,
    Obstructed,
    OutsideSupport,
    Unsupported,
    Conflict,
    PoseLocked,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlacementReport {
    pub actor: usize,
    pub hand: usize,
    pub prop: usize,
    pub support: usize,
    pub at: f64,
    pub status: PlacementStatus,
    pub target: Vec3,
    pub actual: Vec3,
    pub error: f64,
    pub tolerance: f64,
    /// Signed bottom-to-support-top distance in metres.
    pub support_gap: f64,
    /// Signed horizontal containment margin; negative means overhang.
    pub edge_margin: f64,
    pub rotation: Mat3,
}
impl PlacementReport {
    pub fn placed(&self) -> bool {
        self.status == PlacementStatus::Placed
    }
}

/// Projected containment, including the item's yaw and radius. This does not
/// assert dynamic stability or friction on a curved support.
fn edge_margin(item: &Shape, pos: Vec3, rot: Mat3, support: &Shape, at: Vec3, turn: Mat3) -> f64 {
    let p = turn.transpose() * (pos - at);
    let r = turn.transpose() * rot;
    match *support {
        Shape::Box { sx, sz, .. } => {
            let (x, z) = match *item {
                Shape::Box { sx, sz, .. } => {
                    (0.5 * (sx * r.col(0).x.abs() + sz * r.col(2).x.abs()), 0.5 * (sx * r.col(0).z.abs() + sz * r.col(2).z.abs()))
                }
                Shape::Sphere { r } => (r, r),
            };
            (sx * 0.5 - p.x.abs() - x).min(sz * 0.5 - p.z.abs() - z)
        }
        Shape::Sphere { r: radius } => {
            let extent = match *item {
                Shape::Sphere { r } => (p.x * p.x + p.z * p.z).sqrt() + r,
                Shape::Box { sx, sz, .. } => [-1.0, 1.0]
                    .into_iter()
                    .flat_map(|x| [-1.0, 1.0].map(move |z| (x, z)))
                    .map(|(x, z)| {
                        let c = p + r * Vec3 { x: x * sx * 0.5, y: 0.0, z: z * sz * 0.5 };
                        (c.x * c.x + c.z * c.z).sqrt()
                    })
                    .fold(0.0, f64::max),
            };
            radius - extent
        }
    }
}

fn intersects(a: &Shape, ap: Vec3, ar: Mat3, b: &Shape, bp: Vec3, br: Mat3) -> bool {
    let depth = (ap.y + a.top()).min(bp.y + b.top()) - ap.y.max(bp.y);
    depth > PENETRATION && crate::clearance::footprint(a, ap, ar, b, bp, br)
}

pub(crate) fn held_obstructed(shot: &Shot, actor: usize, hand: usize, w: f64, fig: &mavian_rig::Figure) -> bool {
    let obstacles = shot.arm_obstacles(actor, hand, w, fig);
    for (i, p) in shot.props.iter().enumerate() {
        let Some((a, h, offset, _)) = p.segment(w).and_then(binding) else { continue };
        if (a, h) != (actor, hand) {
            continue;
        }
        let pos = fig.joints[hand] + Mat3::rot_y(shot.actors[actor].facing.at(w)) * offset;
        if obstacles.iter().any(|o| o.prop != i && intersects(&p.shape, pos, obstacles[i].rot, &o.shape, o.pos, o.rot)) {
            return true;
        }
    }
    false
}

fn evaluate(shot: &Shot, actor: usize, hand: usize, prop: usize, support: usize, t: f64, target: Vec3) -> PlacementReport {
    let a = &shot.actors[actor];
    let chain = actions::chain(a, hand).expect("validated placement chain");
    let (actual, rotation) = shot.prop_rest(prop, t);
    let (sp, sr) = shot.prop_rest(support, t);
    let support_gap = actual.y - sp.y - shot.props[support].shape.top();
    let margin = edge_margin(&shot.props[prop].shape, actual, rotation, &shot.props[support].shape, sp, sr);
    let error = (actual - target).length();
    let fig = shot.posed(actor, t).0;
    let (base, m) = shot.deformed(actor, t);
    let reports = shot.solve_actor_contacts(actor, t, &base, m).1;
    let contact = reports.iter().find(|r| r.chain == chain && r.name == format!("release:{}", shot.props[prop].name));
    let requested_margin = edge_margin(&shot.props[prop].shape, target, rotation, &shot.props[support].shape, sp, sr);
    let target_blocked = shot.props.iter().enumerate().any(|(i, p)| {
        if i == prop || i == support {
            return false;
        }
        let (pos, rot) = shot.prop_rest(i, t);
        intersects(&shot.props[prop].shape, target, rotation, &p.shape, pos, rot)
    });
    let obstructed = shot.obstacle_reports(t).iter().any(|r| r.actor == actor && r.chain == chain)
        || shot.self_collision_reports(t).iter().any(|(i, r)| *i == actor && r.chain == chain)
        || held_obstructed(shot, actor, hand, t, &fig);
    let status = if requested_margin < -TOLERANCE {
        PlacementStatus::OutsideSupport
    } else if error > TOLERANCE && a.pose_lock {
        PlacementStatus::PoseLocked
    } else if contact.is_some_and(|c| c.status == ContactStatus::Conflict) {
        PlacementStatus::Conflict
    } else if error > TOLERANCE && contact.is_some_and(|c| c.reach_error > TOLERANCE) {
        PlacementStatus::Unreachable
    } else if obstructed || target_blocked || support_gap < -PENETRATION {
        PlacementStatus::Obstructed
    } else if margin < -TOLERANCE {
        PlacementStatus::OutsideSupport
    } else if support_gap > TOLERANCE {
        PlacementStatus::Unsupported
    } else if error > TOLERANCE {
        PlacementStatus::Obstructed
    } else {
        PlacementStatus::Placed
    };
    PlacementReport {
        actor,
        hand,
        prop,
        support,
        at: t,
        status,
        target,
        actual,
        error,
        tolerance: TOLERANCE,
        support_gap,
        edge_margin: margin,
        rotation,
    }
}

fn forget(shot: &mut Shot) {
    for a in &mut shot.actors {
        a.forget();
    }
}

pub(crate) fn release(shot: &mut Shot, action: &Action) -> Result<(), String> {
    let Action::Release { actor, hand, prop, at: t, onto: Onto::Prop(support), reach, settle } = *action else { unreachable!() };
    if !t.is_finite() || !reach.is_finite() || reach <= 0.0 || !settle.is_finite() || settle <= 0.0 {
        return Err("placement needs finite time and positive reach/settle".into());
    }
    let a = shot.actors.get(actor).ok_or_else(|| "placement actor does not exist".to_string())?;
    if hand >= a.model.rig.joints().len() {
        return Err("placement hand does not exist".into());
    }
    let chain = actions::chain(a, hand)?;
    if prop >= shot.props.len() || support >= shot.props.len() || prop == support {
        return Err("placement needs distinct existing item and support".into());
    }
    let Some(Motion::Held { actor: owner, hand: grip, offset, .. }) = shot.props[prop].segment(t).cloned() else {
        let (actual, rotation) = shot.prop_rest(prop, t);
        shot.placement_reports.push(PlacementReport {
            actor,
            hand,
            prop,
            support,
            at: t,
            status: PlacementStatus::NotHeld,
            target: actual,
            actual,
            error: 0.0,
            tolerance: TOLERANCE,
            support_gap: 0.0,
            edge_margin: 0.0,
            rotation,
        });
        return Ok(());
    };
    if (owner, grip) != (actor, hand) {
        return Err("placement item is held by a different hand".into());
    }
    let acquired = shot.props[prop].motion.iter().rev().find(|(at, _)| *at <= t + 1e-9).unwrap().0;
    if t - reach < acquired - 1e-9 {
        return Err(format!("release at {t}: placement reach begins before acquisition"));
    }
    let start = shot.deformed(actor, t - reach).0.joints[hand];
    let end = shot.deformed(actor, t + settle).0.joints[hand];
    let heading = Mat3::rot_y(shot.actors[actor].facing.at(t));
    let desired = shot.actors[actor].figure(t).joints[chain[0]] + heading * offset;
    let (_, rot) = shot.prop_rest(prop, t);
    // near_spot supplies the historical footprint-aware target. Transform its
    // support-local point through the final support pose, not preliminary IK.
    let preliminary = shot.props[support].rest(t, &shot.world());
    let place = actions::near_spot(&shot.props[support], &shot.props[prop], rot, t, desired, &shot.world());
    let (sp, sr) = shot.prop_rest(support, t);
    let target = sp + sr * (preliminary.1.transpose() * (place - preliminary.0));
    let goal = target - heading * offset;
    let over = t - reach * 0.4;
    let mut c = Contact::new(
        &format!("release:{}", shot.props[prop].name),
        chain,
        Target::Descend { at: goal, over, down: t, hover: 0.07 },
        t - reach,
        t + settle,
    );
    c.weight = Keys::new(Ease::InOut).key(t - reach, 0.0).key(over, 1.0).key(t, 1.0).key(t + settle, 0.0).into();
    c.approach_from = Some(start);
    c.return_to = Some((t, end));
    let side = if shot.actors[actor].model.rig.joints()[hand].name.ends_with("_r") { -1.0 } else { 1.0 };
    c.pole = Some(heading * Vec3 { x: 0.4 * side, y: -0.5, z: -0.6 });
    c.approach = Some(
        crate::ik::flat_dir(goal, shot.actors[actor].figure(t).joints[chain[0]]) * 0.25 * shot.actors[actor].scale
            + Vec3::UP * (0.12 * shot.actors[actor].scale),
    );
    c.withdrawal = Some((t, Some(Vec3 { y: 0.0, ..end - goal })));
    shot.actors[actor].auto_contacts.push(c);
    forget(shot);
    crate::contact::validate_dependencies(t, &shot.world())?;
    let report = evaluate(shot, actor, hand, prop, support, t, target);
    if report.placed() {
        // Preserve the verified final transform, including sub-tolerance gaps.
        let (sp, sr) = shot.prop_rest(support, t);
        shot.props[prop].motion.push((
            t,
            Motion::Ride { support, offset: sr.transpose() * (report.actual - sp), yaw: report.rotation.heading() - sr.heading(), drop: 0.0, t0: t },
        ));
        if let Some(c) = shot.props[prop].contacts.iter_mut().rev().find(|c| c.2 == actor && c.3 == hand && c.1.is_infinite()) {
            c.1 = t + settle;
        }
        forget(shot);
        crate::contact::validate_dependencies(t, &shot.world())?;
    }
    // On refusal the original held segment remains; no floor clamp or ride.
    shot.placement_reports.push(report);
    Ok(())
}

pub(crate) fn plan_rejections(shot: &mut Shot) {
    for r in shot.placement_reports.clone().into_iter().filter(|r| !r.placed() && r.status != PlacementStatus::NotHeld) {
        if shot.actors[r.actor].pose_lock {
            continue;
        }
        let index = shot.actors[r.actor].auto_contacts.iter().position(|c| {
            c.chain[2] == r.hand && c.name == format!("release:{}", shot.props[r.prop].name) && c.withdrawal.is_some_and(|(t, _)| t == r.at)
        });
        if let Some(index) = index {
            // The held item returns along the same approach side; an empty-
            // hand withdrawal waypoint is inappropriate after refusal.
            let c = &mut shot.actors[r.actor].auto_contacts[index];
            c.withdrawal = Some((r.at, c.approach));
            crate::grip::limit_attempt(shot, r.actor, index, r.at);
        }
    }
}

pub(crate) fn verify(shot: &mut Shot) -> Result<(), String> {
    for index in 0..shot.placement_reports.len() {
        let r = shot.placement_reports[index].clone();
        if !r.placed() {
            let (actual, rotation) = shot.prop_rest(r.prop, r.at);
            shot.placement_reports[index].actual = actual;
            shot.placement_reports[index].rotation = rotation;
            shot.placement_reports[index].error = (actual - r.target).length();
            let (sp, sr) = shot.prop_rest(r.support, r.at);
            shot.placement_reports[index].support_gap = actual.y - sp.y - shot.props[r.support].shape.top();
            shot.placement_reports[index].edge_margin =
                edge_margin(&shot.props[r.prop].shape, actual, rotation, &shot.props[r.support].shape, sp, sr);
            continue;
        }
        let split = shot.props[r.prop].motion.iter().position(|(t, _)| *t >= r.at - 1e-9).unwrap();
        let later = shot.props[r.prop].motion.split_off(split);
        forget(shot);
        let final_report = evaluate(shot, r.actor, r.hand, r.prop, r.support, r.at, r.target);
        shot.props[r.prop].motion.extend(later);
        forget(shot);
        if !final_report.placed() || (final_report.actual - r.actual).length() > 1e-6 || final_report.rotation != r.rotation {
            return Err(format!(
                "placement at {}: later constraints change verified transfer of '{}'; separate overlapping contact intervals",
                r.at, shot.props[r.prop].name
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn support_containment_checks_the_whole_rotated_item() {
        let item = Shape::Box { sx: 0.18, sy: 0.1, sz: 0.08 };
        let support = Shape::Box { sx: 0.2, sy: 0.1, sz: 0.12 };
        let origin = Vec3 { x: 1.3, y: 1.0, z: -0.7 };
        for heading in [0.0, 37.0, 163.0] {
            let r = Mat3::rot_y(heading);
            assert!((edge_margin(&item, origin, r, &support, origin, r) - 0.01).abs() < 1e-8);
            assert!((edge_margin(&item, origin, r * Mat3::rot_y(90.0), &support, origin, r) + 0.03).abs() < 1e-8);
        }
        let ball = Shape::Sphere { r: 0.08 };
        let round = Shape::Sphere { r: 0.2 };
        assert!((edge_margin(&ball, Vec3 { x: 0.15, y: 0.0, z: 0.0 }, Mat3::IDENT, &round, Vec3::ZERO, Mat3::IDENT) + 0.03).abs() < 1e-8);
    }

    #[test]
    fn a_clear_hand_does_not_hide_a_collision_with_its_held_item() {
        let actor = crate::Actor::new(mavian_models::mannequin::mannequin());
        let hand = actor.model.rig.joint("hand_r").unwrap();
        let fig = actor.figure(0.0);
        let wrist = fig.joints[hand];
        let offset = Vec3 { x: 0.15, y: -0.1, z: 0.0 };
        let mut item = crate::Prop::new("item", Shape::Box { sx: 0.13, sy: 0.17, sz: 0.13 }, wrist + offset);
        item.motion.push((0.0, Motion::Held { actor: 0, hand, offset, yaw: 0.0 }));
        let barrier = crate::Prop::new("barrier", Shape::Box { sx: 0.02, sy: 0.17, sz: 0.13 }, wrist + Vec3 { x: 0.2, y: -0.1, z: 0.0 });
        let mut shot = Shot { actors: vec![actor], props: vec![item, barrier], ..Shot::default() };
        assert!(shot.arm_obstacles(0, hand, 0.0, &fig)[1].distance(wrist).0 > 0.15);
        assert!(held_obstructed(&shot, 0, hand, 0.0, &fig));
        shot.props[1].at.x += 0.1;
        assert!(!held_obstructed(&shot, 0, hand, 0.0, &fig));
    }
}
