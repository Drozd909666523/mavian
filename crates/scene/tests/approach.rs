use mavian_core::{Track, Vec3};
use mavian_scene::{Action, Actor, Approach, Contact, ContactStatus, Priority, Prop, Shape, Shot, StepStatus, Target};

fn scene(z: f64, distance: Option<f64>) -> Shot {
    let actor = Actor::new(mavian_models::mannequin::mannequin());
    let hands = [actor.model.rig.joint("hand_l").unwrap(), actor.model.rig.joint("hand_r").unwrap()];
    let mut shot = Shot {
        actors: vec![actor],
        props: vec![Prop::new("box", Shape::Box { sx: 0.6, sy: 1.4, sz: 0.4 }, Vec3 { x: 0., y: 0., z })],
        actions: vec![Action::Push {
            actor: 0,
            prop: 0,
            start: 1.8,
            end: 2.6,
            hands,
            reach: 0.35,
            settle: 0.35,
            height: 0.85,
            spread: 0.17,
            step: distance.map(|max_distance| Approach { max_distance, duration: 0.9 }),
        }],
        ..Shot::default()
    };
    shot.compile().unwrap();
    shot
}

#[test]
fn preparation_moves_forward_and_back_without_sliding_supports_or_stretching_bones() {
    for (z, sign) in [(0.40, -1.0), (1.0, 1.0)] {
        let shot = scene(z, Some(0.35));
        let actor = &shot.actors[0];
        let report = &actor.step_reports[0];
        eprintln!("step at z={z}: {report:?}");
        assert_eq!(report.status, StepStatus::Applied);
        assert!(report.shift.z * sign > 0.05);
        assert!(report.shift.length() <= 0.35 + 1e-10);
        let start = actor.figure(0.0);
        let mut previous = None;
        let mut highest_foot = 0.0f64;
        for frame in 0..=768 {
            let w = frame as f64 / 240.0;
            let fig = actor.figure(w);
            let posed = shot.posed(0, w).0;
            for (prop, p) in shot.props.iter().enumerate() {
                let (pos, rot) = shot.prop_rest(prop, w);
                let obstacle = mavian_scene::obstacle::Obstacle { prop, shape: p.shape.clone(), pos, rot };
                let check = |a: usize, b: usize, ra: f64, rb: f64| {
                    let clearance = obstacle.capsule(posed.joints[a], ra, posed.joints[b], rb).0;
                    assert!(clearance >= -0.001, "body enters {} at {w}, bone {a}->{b}: {clearance}", p.name);
                };
                for part in &actor.model.parts {
                    match part {
                        mavian_rig::Part::Body { a, b, rx, rz, .. } => check(*a, *b, rx.0.max(rz.0), rx.1.max(rz.1)),
                        mavian_rig::Part::Head { joint, radius, .. } | mavian_rig::Part::Ball { joint, radius } => {
                            check(*joint, *joint, *radius, *radius)
                        }
                        mavian_rig::Part::Limb { chain, radii, .. } => {
                            for i in 0..chain.len() - 1 {
                                check(chain[i], chain[i + 1], radii[i], radii[i + 1]);
                            }
                        }
                        _ => (),
                    }
                }
            }
            for j in 0..fig.joints.len() {
                if let Some(parent) = actor.model.rig.parent(j) {
                    let before = (start.joints[j] - start.joints[parent]).length();
                    let after = (fig.joints[j] - fig.joints[parent]).length();
                    assert!((before - after).abs() < 1e-8, "bone {} stretches at {w}", actor.model.rig.joints()[j].name);
                }
            }
            for side in [mavian_rig::Side::L, mavian_rig::Side::R] {
                let foot = actor.model.rig.foot(side).unwrap();
                highest_foot = highest_foot.max(fig.joints[foot.heel].y);
                assert!(fig.joints[foot.heel].y >= -1e-8 && fig.joints[foot.toe].y >= -1e-8);
            }
            let support = actor.planted_foot(w);
            if let Some((old_w, old, old_support)) = previous {
                let old: std::sync::Arc<mavian_rig::Figure> = old;
                for (a, b) in fig.joints.iter().zip(&old.joints) {
                    assert!((*a - *b).length() < 0.025, "pose jumps at {old_w}->{w}: {:?}", *a - *b);
                }
                if let Some(side) = support
                    && old_support == support
                {
                    let foot = actor.model.rig.foot(side).unwrap();
                    assert!((fig.joints[foot.heel] - old.joints[foot.heel]).length() < 1e-8, "support heel slides at {w}");
                    assert!((fig.joints[foot.toe] - old.joints[foot.toe]).length() < 1e-8, "support toe slides at {w}");
                }
            }
            previous = Some((w, fig, support));
        }
        assert!(highest_foot > 0.01, "the step must lift a foot rather than translate the actor");
        assert_eq!(actor.figure(report.from).joints, start.joints);
        let finished = actor.figure(report.to);
        for (a, b) in finished.joints.iter().zip(&start.joints) {
            assert!((*a - *b - report.shift).length() < 1e-8, "the preparatory step must finish before reaching");
        }
        for (_, r) in shot.contact_reports(2.0).into_iter().filter(|(_, r)| r.name.starts_with("push:")) {
            assert_eq!(r.status, ContactStatus::Satisfied, "{r:?}");
            assert!(r.error < 1e-8);
        }
    }
}

#[test]
fn disabling_preparation_and_pose_lock_keep_the_authored_path() {
    let off = scene(1.0, None);
    assert!(off.actors[0].step_reports.is_empty());
    assert_eq!(off.actors[0].figure(0.).joints, off.actors[0].figure(2.).joints);
    let mut locked = scene(1.0, Some(0.35));
    locked.actors[0].pose_lock = true;
    locked.compile().unwrap();
    assert_eq!(locked.actors[0].step_reports[0].status, StepStatus::PoseLocked);
    assert_eq!(locked.actors[0].figure(0.).joints, locked.actors[0].figure(2.).joints);
    assert_eq!(locked.posed(0, 2.).0, locked.actors[0].figure(2.));
}

#[test]
fn grabbing_after_preparation_reaches_the_surface_without_an_attachment_jump() {
    let mut shot = scene(1.0, Some(0.35));
    shot.props[0] = Prop::new("cup", Shape::Box { sx: 0.18, sy: 0.2, sz: 0.18 }, Vec3 { x: 0., y: 1., z: 0.9 });
    shot.props.push(Prop::new("pedestal", Shape::Box { sx: 0.35, sy: 1., sz: 0.35 }, Vec3 { x: 0., y: 0., z: 0.9 }));
    let hand = shot.actors[0].model.rig.joint("hand_r").unwrap();
    shot.actions = vec![Action::Grab {
        actor: 0,
        hand,
        prop: 0,
        at: 1.8,
        reach: 0.35,
        settle: 0.35,
        height: 0.6,
        step: Some(Approach { max_distance: 0.35, duration: 0.9 }),
    }];
    shot.compile().unwrap();
    assert_eq!(shot.actors[0].step_reports[0].status, StepStatus::Applied);
    assert!((shot.prop_rest(0, 1.8).0 - shot.props[0].at).length() < 1e-8);
    let before = shot.prop_rest(0, 1.8 - 1e-6).0;
    let after = shot.prop_rest(0, 1.8 + 1e-6).0;
    assert!((before - after).length() < 1e-4);
    for (_, r) in shot.contact_reports(1.8) {
        assert_eq!(r.status, ContactStatus::Satisfied, "{r:?}");
        assert!(r.error < 1e-8);
    }
    let chain = mavian_scene::actions::chain(&shot.actors[0], hand).unwrap();
    let mut previous: Option<std::sync::Arc<mavian_rig::Figure>> = None;
    for n in 0..=720 {
        let w = n as f64 / 240.0;
        let fig = shot.posed(0, w).0;
        assert!(shot.obstacle_reports(w).is_empty(), "prop intersection at {w}");
        assert!(shot.held_prop_reports(w).is_empty(), "held volume intersects support at {w}");
        if w >= 1.45 { assert!(shot.self_collision_reports(w).is_empty(), "arm enters body at {w}"); }
        if let Some(old) = previous {
            for j in [chain[1], hand] {
                assert!((fig.joints[j] - old.joints[j]).length() < 0.04, "arm route jumps at {w}");
            }
        }
        previous = Some(fig);
    }
}

#[test]
fn blocked_path_and_distance_limit_leave_explicit_residuals() {
    let mut blocked = scene(1.0, Some(0.35));
    blocked.props.push(Prop::new("barrier", Shape::Box { sx: 0.65, sy: 0.18, sz: 0.04 }, Vec3 { x: 0., y: 0., z: 0.213 }));
    blocked.compile().unwrap();
    assert_eq!(blocked.actors[0].step_reports[0].status, StepStatus::Blocked);
    assert_eq!(blocked.actors[0].step_reports[0].shift, Vec3::ZERO);
    let limited = scene(1.3, Some(0.12));
    let report = &limited.actors[0].step_reports[0];
    assert_eq!(report.status, StepStatus::Limited);
    assert!(report.shift.length() <= 0.12 + 1e-10);
    assert!(report.error_after > 0.35 && report.error_after < report.error_before);
    assert!(limited.contact_reports(2.).iter().any(|(_, r)| r.status == ContactStatus::Unreachable));
}

#[test]
fn authored_motion_and_supports_prevent_automatic_preparation() {
    let mut moving = scene(1.0, Some(0.35));
    moving.actors[0].path = mavian_core::Keys::new(mavian_core::Ease::Linear).key(0., (0., 0.)).key(3., (0.3, 0.)).into();
    moving.compile().unwrap();
    assert_eq!(moving.actors[0].step_reports[0].status, StepStatus::AuthoredMotion);
    let mut pinned = scene(1.0, Some(0.35));
    let actor = &mut pinned.actors[0];
    let foot = actor.model.rig.foot(mavian_rig::Side::L).unwrap();
    let mut pin =
        Contact::new("foot", mavian_scene::actions::chain(actor, foot.heel).unwrap(), Target::Point(actor.figure(0.).joints[foot.heel]), 0., 3.);
    pin.priority = Priority::Support;
    pin.weight = Track::Const(1.0);
    actor.contacts.push(pin);
    pinned.compile().unwrap();
    assert_eq!(pinned.actors[0].step_reports[0].status, StepStatus::ContactBusy);
    let mut future = scene(1.0, Some(0.35));
    let hand = future.actors[0].model.rig.joint("hand_r").unwrap();
    future.actors.push(Actor::new(mavian_models::mannequin::mannequin()).path((2.0, 0.0)));
    let mut strike = mavian_scene::Strike::new(0, hand, 1, 2.1);
    strike.windup = 0.9;
    strike.miss = true;
    future.actions.push(Action::Strike(strike));
    future.compile().unwrap();
    assert_eq!(future.actors[0].step_reports[0].status, StepStatus::ContactBusy, "a later action already reserves its windup");
    let mut earlier_contact = scene(1.0, Some(0.35));
    earlier_contact.actions.push(Action::Grab {
        actor: 0,
        hand,
        prop: 0,
        at: 3.2,
        reach: 2.0,
        settle: 0.35,
        height: 0.6,
        step: Some(Approach { max_distance: 0.35, duration: 0.6 }),
    });
    earlier_contact.compile().unwrap();
    assert!(
        earlier_contact.actors[0].step_reports.iter().all(|r| r.status == StepStatus::ContactBusy),
        "a later action must not insert a step before an earlier compiled contact"
    );
}

#[test]
fn the_completed_offset_survives_a_later_authored_flight() {
    use mavian_core::{Ease, Keys};
    use mavian_scene::pk;
    let poses = || Keys::new(Ease::InOut).key(0., pk("stand")).key(1.5, pk("stand")).key(2.0, pk("leap_launch")).key(2.5, pk("leap_air").fly());
    let mut stepped = scene(1.0, Some(0.35));
    stepped.actors[0].poses = poses().into();
    stepped.compile().unwrap();
    let delta = stepped.actors[0].step_reports[0].shift;
    assert!(delta.length() > 0.1);
    let mut authored = scene(1.0, None);
    authored.actors[0].poses = poses().into();
    authored.compile().unwrap();
    for w in [1.6, 1.99, 2.01, 2.2, 2.4] {
        let a = stepped.actors[0].figure(w);
        let b = authored.actors[0].figure(w);
        for (a, b) in a.joints.iter().zip(&b.joints) {
            assert!((*a - *b - delta).length() < 1e-8, "step offset disappears during authored flight at {w}");
        }
    }
}

#[test]
fn plans_are_repeatable_and_invalid_step_options_are_rejected() {
    let mut shot = scene(1., Some(0.35));
    let reports = shot.actors[0].step_reports.clone();
    let samples: Vec<_> = [0.2, 0.6, 0.9, 1.2, 1.45, 2.0].into_iter().map(|w| (w, shot.posed(0, w).0)).collect();
    shot.compile().unwrap();
    assert_eq!(reports, shot.actors[0].step_reports);
    for (w, fig) in samples.into_iter().rev() {
        assert_eq!(shot.posed(0, w).0, fig);
    }
    if let Action::Push { step, .. } = &mut shot.actions[0] {
        *step = Some(Approach { max_distance: f64::NAN, duration: 0.9 });
    }
    assert!(shot.compile().unwrap_err().0.contains("positive finite"));
    let mut early = scene(1.0, Some(0.35));
    if let Action::Push { start, .. } = &mut early.actions[0] {
        *start = 1.0;
    }
    early.compile().unwrap();
    assert_eq!(early.actors[0].step_reports[0].status, StepStatus::NoTime);
    let close = scene(0.68, Some(0.35));
    assert_eq!(close.actors[0].step_reports[0].status, StepStatus::Unneeded);
    let mut assisted = scene(0.95, Some(0.35));
    assisted.actors[0].body_reach = 30.;
    assisted.compile().unwrap();
    assert_eq!(assisted.actors[0].step_reports[0].status, StepStatus::Unneeded);
    for angle in [90., 180., -90.] {
        let mut rotated = scene(1., Some(0.35));
        let rotation = mavian_core::Mat3::rot_y(angle);
        rotated.actors[0].facing = Track::Const(angle);
        rotated.props[0].at = rotation * Vec3 { x: 0., y: 0., z: 1. };
        rotated.props[0].turn = angle;
        rotated.compile().unwrap();
        assert_eq!(rotated.actors[0].step_reports[0].status, StepStatus::Applied);
        let local = rotation.transpose() * rotated.actors[0].step_reports[0].shift;
        assert!(local.z > 0.1 && local.x.abs() < 1e-8);
    }
}
