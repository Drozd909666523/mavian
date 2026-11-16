//! Contact-gated acquisition, compiled once and sampled without frame history.
use crate::{Action, Contact, ContactStatus, Motion, Shot, Target, actions};
use mavian_core::{Mat3, Vec3};

const TOLERANCE: f64 = 0.002;
const PENETRATION: f64 = 0.001;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrabStatus {
    Attached,
    Unreachable,
    Obstructed,
    Conflict,
    PoseLocked,
    HandBusy,
    PropBusy,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GrabReport {
    pub actor: usize,
    pub hand: usize,
    pub prop: usize,
    pub at: f64,
    pub status: GrabStatus,
    pub target: Vec3,
    pub actual: Vec3,
    pub error: f64,
    /// Metres; checked on the final world-space hand, before drawing effects.
    pub tolerance: f64,
    pub prop_position: Vec3,
    pub prop_rotation: Mat3,
}
impl GrabReport {
    pub fn attached(&self) -> bool {
        self.status == GrabStatus::Attached
    }
}

fn owner(m: &Motion) -> Option<(usize, usize)> {
    match m {
        Motion::Held { actor, hand, .. } => Some((*actor, *hand)),
        Motion::Above { held, .. } => owner(held),
        _ => None,
    }
}

fn forget(shot: &mut Shot) {
    for a in &mut shot.actors {
        a.forget();
    }
}

fn obstructed(shot: &Shot, actor: usize, hand: usize, w: f64, fig: &mavian_rig::Figure) -> bool {
    let chain = actions::chain(&shot.actors[actor], hand).expect("validated grab chain");
    shot.arm_obstacles(actor, hand, w, fig).into_iter().any(|mut o| {
        // Own held props already use the supplied figure. Resolve other props
        // through the same final transforms that rendering uses.
        if shot.props[o.prop].segment(w).and_then(owner).is_none_or(|(a, _)| a != actor) {
            (o.pos, o.rot) = shot.prop_rest(o.prop, w);
        }
        (0..2).any(|k| {
            let (a, b) = (chain[k], chain[k + 1]);
            o.capsule(
                fig.joints[a],
                crate::obstacle::joint_radius(&shot.actors[actor], a, shot.style.line),
                fig.joints[b],
                crate::obstacle::joint_radius(&shot.actors[actor], b, shot.style.line),
            )
            .0 < -PENETRATION
        })
    })
}

/// Match the grasp envelope to the model's hand and drawing clearance.
/// The unit mannequin retains its existing 65 mm target offset.
pub(crate) fn surface(actor: &crate::Actor, hand: usize, shape: &crate::Shape, normal: Vec3, height: f64, line: f64) -> Vec3 {
    let radius = crate::obstacle::joint_radius(actor, hand, line);
    let envelope = if radius > 0.0 { radius + 0.013 * actor.scale } else { actions::HAND_R * actor.scale };
    actions::surface(shape, normal, height) + normal * (envelope - actions::HAND_R)
}

pub(crate) fn grab(shot: &mut Shot, action: &Action) -> Result<(), String> {
    let Action::Grab { actor, hand, prop, at: t, reach, settle, height, .. } = *action else { unreachable!() };
    if !t.is_finite()
        || !reach.is_finite()
        || reach <= 0.0
        || !settle.is_finite()
        || settle <= 0.0
        || !height.is_finite()
        || !(0.0..=1.0).contains(&height)
    {
        return Err("grab needs finite time, positive reach/settle and height in [0, 1]".into());
    }
    let a = shot.actors.get(actor).ok_or_else(|| format!("grab: actor {actor} does not exist"))?;
    if hand >= a.model.rig.joints().len() {
        return Err(format!("grab: hand {hand} does not exist"));
    }
    if prop >= shot.props.len() {
        return Err(format!("grab: prop {prop} does not exist"));
    }
    let ch = actions::chain(a, hand)?;
    // Resolve the unbound prop before introducing a dependency on this hand.
    let (pos, rot) = shot.prop_rest(prop, t);
    let (n, _) = actions::facing_side(pos, &rot, actions::root_at(a, t));
    let point = pos + rot * surface(a, hand, &shot.props[prop].shape, n, height, shot.style.line);
    let busy = if shot.props[prop].segment(t).and_then(owner).is_some() {
        Some(GrabStatus::PropBusy)
    } else if shot.props.iter().any(|p| p.segment(t).and_then(owner) == Some((actor, hand))) {
        Some(GrabStatus::HandBusy)
    } else {
        None
    };
    if let Some(status) = busy {
        let actual = shot.posed(actor, t).0.joints[hand];
        shot.grab_reports.push(GrabReport {
            actor,
            hand,
            prop,
            at: t,
            status,
            target: point,
            actual,
            error: (actual - point).length(),
            tolerance: TOLERANCE,
            prop_position: pos,
            prop_rotation: rot,
        });
        return Ok(());
    }
    let a = &mut shot.actors[actor];
    let mut contact = Contact::new(&format!("grab:{}", shot.props[prop].name), ch, Target::Point(point), t - reach, t + settle);
    contact.weight = actions::window(t, t, reach, settle);
    let side = if a.model.rig.joints()[hand].name.ends_with("_r") { 1.0 } else { -1.0 };
    contact.pole = Some(Mat3::rot_y(a.facing.at(t)) * Vec3 { x: 0.4 * side, y: -0.5, z: -0.6 });
    contact.approach = Some(crate::ik::flat_dir(pos, actions::root_at(a, t)) * 0.20 * a.scale + Vec3::UP * (0.08 * a.scale));
    contact.withdrawal = Some((t, contact.approach));
    a.auto_contacts.push(contact.clone());
    // Invalid target dependencies must be rejected before evaluating a final pose.
    crate::contact::validate_dependencies(t, &shot.world())?;
    forget(shot);
    let fig = shot.posed(actor, t).0;
    let actual = fig.joints[hand];
    let error = (actual - point).length();
    let (base, m) = shot.deformed(actor, t);
    let reports = shot.solve_actor_contacts(actor, t, &base, m).1;
    let report = reports.iter().find(|r| r.name == contact.name && r.chain == ch);
    let status = if obstructed(shot, actor, hand, t, &fig) {
        GrabStatus::Obstructed
    } else if error.is_finite() && error <= TOLERANCE {
        GrabStatus::Attached
    } else if shot.actors[actor].pose_lock {
        GrabStatus::PoseLocked
    } else if report.is_some_and(|r| r.status == ContactStatus::Conflict) {
        GrabStatus::Conflict
    } else if report.is_some_and(|r| r.reach_error > TOLERANCE) {
        GrabStatus::Unreachable
    } else {
        GrabStatus::Obstructed
    };
    let mut report =
        GrabReport { actor, hand, prop, at: t, status, target: point, actual, error, tolerance: TOLERANCE, prop_position: pos, prop_rotation: rot };
    if report.attached() {
        let heading = shot.actors[actor].facing.at(t);
        // Preserve the prop transform at acquisition, including the accepted
        // sub-tolerance residual; never compute the offset from preliminary IK.
        let offset = Mat3::rot_y(-heading) * (pos - actual);
        shot.props[prop].motion.push((t, Motion::Held { actor, hand, offset, yaw: rot.heading() - heading }));
        forget(shot);
        if let Err(error) = crate::contact::validate_dependencies(t, &shot.world()) {
            shot.props[prop].motion.pop();
            forget(shot);
            return Err(error);
        }
        let after = shot.prop_rest(prop, t);
        let held_fig = shot.posed(actor, t).0;
        if (after.0 - pos).length() > 1e-6 || obstructed(shot, actor, hand, t, &held_fig) {
            shot.props[prop].motion.pop();
            report.status = GrabStatus::Obstructed;
            forget(shot);
        } else {
            shot.props[prop].contacts.push((t - reach, f64::INFINITY, actor, hand));
        }
    }
    shot.grab_reports.push(report);
    Ok(())
}

/// Later actions may overlap an earlier acquisition's contact interval. Do
/// not accept a compiled scene if they invalidate an already verified binding.
pub(crate) fn verify(shot: &mut Shot) -> Result<(), String> {
    for r in shot.grab_reports.clone().into_iter().filter(GrabReport::attached) {
        let split = shot.props[r.prop].motion.iter().position(|(t, _)| *t >= r.at - 1e-9).unwrap();
        let later = shot.props[r.prop].motion.split_off(split);
        forget(shot);
        let (pos, rot) = shot.prop_rest(r.prop, r.at);
        let fig = shot.posed(r.actor, r.at).0;
        let changed = (fig.joints[r.hand] - r.actual).length() > 1e-6
            || (pos - r.prop_position).length() > 1e-6
            || rot != r.prop_rotation
            || obstructed(shot, r.actor, r.hand, r.at, &fig);
        shot.props[r.prop].motion.extend(later);
        forget(shot);
        if changed {
            return Err(format!(
                "grab at {}: later actions change the verified contact for '{}'; separate overlapping contact intervals",
                r.at, shot.props[r.prop].name
            ));
        }
    }
    Ok(())
}
