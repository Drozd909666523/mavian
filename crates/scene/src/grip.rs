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
    let start = shot.deformed(actor, t - reach).0.joints[hand];
    let end = shot.deformed(actor, t + settle).0.joints[hand];
    let a = &mut shot.actors[actor];
    let mut contact = Contact::new(&format!("grab:{}", shot.props[prop].name), ch, Target::Point(point), t - reach, t + settle);
    contact.weight = actions::window(t, t, reach, settle);
    contact.approach_from = Some(start);
    let side = if a.model.rig.joints()[hand].name.ends_with("_r") { -1.0 } else { 1.0 };
    contact.pole = Some(Mat3::rot_y(a.facing.at(t)) * Vec3 { x: 0.4 * side, y: -0.5, z: -0.6 });
    contact.approach = Some(crate::ik::flat_dir(pos, actions::root_at(a, t)) * 0.20 * a.scale + Vec3::UP * (0.08 * a.scale));
    contact.withdrawal = Some((t, contact.approach));
    contact.return_to = Some((t, end));
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

/// Refused acquisition does not authorize pushing an arm through its target.
/// Find a clear prefix of the requested reach, then use the same eased window
/// to return. Author tracks and successful acquisition contacts are untouched.
pub(crate) fn plan_rejections(shot: &mut Shot) {
    let rejected: Vec<_> = shot.grab_reports.iter().filter(|r| r.status == GrabStatus::Obstructed).cloned().collect();
    for r in rejected {
        let actor = &shot.actors[r.actor];
        if actor.pose_lock {
            continue;
        }
        let Some(index) = actor.auto_contacts.iter().position(|c| {
            c.name == format!("grab:{}", shot.props[r.prop].name) && c.chain[2] == r.hand && c.withdrawal.is_some_and(|(at, _)| at == r.at)
        }) else {
            continue;
        };
        limit_attempt(shot, r.actor, index, r.at);
    }
}

pub(crate) fn limit_attempt(shot: &mut Shot, actor: usize, index: usize, peak: f64) {
    let original = shot.actors[actor].auto_contacts[index].clone();
    let mut reach = original.clone();
    // A refused reach keeps the authored bend instead of rotating towards
    // the grip pole of an object that the hand will not acquire.
    reach.pole = None;
    reach.pole_turn = None;
    let set_scale = |shot: &mut Shot, amount| {
        let mut contact = reach.clone();
        contact.rejection = Some((peak, amount));
        shot.actors[actor].auto_contacts[index] = contact;
        forget(shot);
    };
    set_scale(shot, 0.0);
    if rejection_is_clear(shot, actor, original.chain, original.from, original.to) {
        let (mut low, mut high) = (0.0, 1.0);
        // Test growing prefixes rather than assuming disconnected feasible
        // intervals form one monotone search space.
        for n in 1..=16 {
            let amount = n as f64 / 16.0;
            set_scale(shot, amount);
            if rejection_is_clear(shot, actor, original.chain, original.from, original.to) {
                low = amount;
            } else {
                high = amount;
                break;
            }
        }
        for _ in 0..8 {
            if low == 1.0 {
                break;
            }
            let mid = (low + high) * 0.5;
            set_scale(shot, mid);
            if rejection_is_clear(shot, actor, original.chain, original.from, original.to) {
                low = mid;
            } else {
                high = mid;
            }
        }
        set_scale(shot, low);
    }
}

fn rejection_is_clear(shot: &Shot, actor: usize, chain: [usize; 3], from: f64, to: f64) -> bool {
    use crate::{
        clearance::{body_arm, body_solids},
        obstacle::{clearance, joint_radius},
    };
    let a = &shot.actors[actor];
    let radii = chain.map(|j| joint_radius(a, j, shot.style.line));
    let margin = 0.01 * a.scale;
    let samples = ((to - from) * 480.0).ceil().max(1.0) as usize;
    let mut previous: Option<(f64, mavian_rig::Figure)> = None;
    for n in 0..=samples {
        let w = from + (to - from) * n as f64 / samples as f64;
        let (base, m) = shot.deformed(actor, w);
        let (fig,reports) = shot.solve_actor_contacts(actor, w, &base, m);
        if reports.iter().any(|r| r.chain==chain && r.reach_error>1e-6) {return false;}
        // Do not run a refused attempt into the fully straight IK singularity.
        // Retain up to 2 mm of extension reserve, without demanding more bend
        // than the author's own (possibly straight) pose provides.
        let raw=a.figure(w);
        let reference=raw.stretched_about(m,a.stretch_pivot(&raw));
        let extension=|f:&mavian_rig::Figure| (f.joints[chain[1]]-f.joints[chain[0]]).length()
            +(f.joints[chain[2]]-f.joints[chain[1]]).length()-(f.joints[chain[2]]-f.joints[chain[0]]).length();
        if extension(&fig)+1e-8 < extension(&reference).clamp(0.0,0.002*a.scale) {return false;}
        let holding=shot.props.iter().any(|p| p.segment(w).and_then(crate::clearance::binding).is_some_and(|(i,h,_,_)| i==actor && h==chain[2]));
        let body_margin=if holding { 0.002*a.scale } else { -PENETRATION };
        if shot.held_lift(actor, chain[2], w, &fig) > 1e-8
            || crate::placement::held_obstructed(shot, actor, chain[2], w, &fig)
            || clearance(&shot.arm_obstacles(actor, chain[2], w, &fig), &fig.joints, chain, radii) < margin
            || body_solids(a, &fig).iter().any(|(_, body)| body_arm(body, &fig, chain, radii).0 < body_margin)
        {
            return false;
        }
        if let Some((old_w, old)) = previous {
            let root = fig.joints[chain[0]] - old.joints[chain[0]];
            if [chain[1], chain[2]].into_iter().any(|j| (fig.joints[j] - old.joints[j] - root).length() > 2.0 * a.scale * (w - old_w)) {
                return false;
            }
            let mid = (w + old_w) * 0.5;
            let mut swept = fig.clone();
            for j in chain {
                swept.joints[j] = old.joints[j].lerp(fig.joints[j], 0.5);
            }
            if clearance(&shot.arm_obstacles(actor, chain[2], mid, &swept), &swept.joints, chain, radii) < margin {
                return false;
            }
        }
        previous = Some((w, fig));
    }
    true
}

/// Reports describe the visible stopped hand, while retaining the reason that
/// acquisition failed and the original requested grasp anchor.
pub(crate) fn refresh_rejections(shot: &mut Shot) {
    for index in 0..shot.grab_reports.len() {
        let r = &shot.grab_reports[index];
        if r.attached() {
            continue;
        }
        let actual = shot.posed(r.actor, r.at).0.joints[r.hand];
        let error = (actual - r.target).length();
        shot.grab_reports[index].actual = actual;
        shot.grab_reports[index].error = error;
    }
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
