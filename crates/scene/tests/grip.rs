use mavian_core::{Ease, Keys, Mat3, Vec3};
use mavian_scene::{Action, Actor, Contact, GrabStatus, Motion, Onto, Priority, Prop, Shape, Shot, Target};

fn scene(z: f64) -> Shot {
    let actor = Actor::new(mavian_models::mannequin::mannequin());
    let hand = actor.model.rig.joint("hand_r").unwrap();
    Shot {
        actors: vec![actor],
        props: vec![Prop::new("cup", Shape::Box { sx: 0.13, sy: 0.17, sz: 0.13 }, Vec3 { x: -0.30, y: 1.0, z })],
        actions: vec![Action::Grab { actor: 0, hand, prop: 0, at: 1.0, reach: 0.4, settle: 0.35, height: 0.6, step: None }],
        ..Shot::default()
    }
}
fn compile(s: &mut Shot) {
    s.compile().unwrap();
    s.validate().unwrap();
    eprintln!("{:?}", s.grab_reports);
}

#[test]
fn valid_grip_attaches_at_authored_time_without_translation_or_rotation_jump() {
    for turn in [0.0, 37.0, -95.0] {
        let mut s = scene(0.35);
        s.props[0].turn = turn;
        compile(&mut s);
        let r = &s.grab_reports[0];
        assert_eq!(r.status, GrabStatus::Attached);
        assert!(r.error <= r.tolerance);
        assert!(s.props[0].segment(1.0 - 1e-6).is_none());
        assert!(matches!(s.props[0].segment(1.0), Some(Motion::Held { .. })));
        let (before, before_rot) = s.prop_rest(0, 1.0 - 1e-6);
        let (at, at_rot) = s.prop_rest(0, 1.0);
        let after = s.prop_rest(0, 1.0 + 1e-6).0;
        assert!((before - at).length() < 1e-9);
        assert!((before - after).length() < 1e-4);
        assert_eq!(before_rot, at_rot);
    }
}

#[test]
fn unreachable_grip_and_its_release_leave_the_prop_at_rest() {
    let mut s = scene(2.0);
    let hand = s.actors[0].model.rig.joint("hand_r").unwrap();
    s.actions.push(Action::Release { actor: 0, hand, prop: 0, at: 2.0, onto: Onto::Nothing, reach: 0.6, settle: 0.35 });
    compile(&mut s);
    assert_eq!(s.grab_reports[0].status, GrabStatus::Unreachable);
    assert!(s.grab_reports[0].error > 0.5);
    assert!(s.props[0].motion.is_empty() && s.props[0].contacts.is_empty());
    for w in [0.0, 0.9, 1.0, 1.2, 2.0, 3.0] {
        assert_eq!(s.prop_rest(0, w).0, s.props[0].at);
    }
}

#[test]
fn pose_lock_refuses_a_grip_that_the_authored_hand_does_not_touch() {
    let mut s = scene(0.35);
    s.actors[0].pose_lock = true;
    compile(&mut s);
    assert_eq!(s.grab_reports[0].status, GrabStatus::PoseLocked);
    assert!(s.props[0].motion.is_empty());
    assert_eq!(s.posed(0, 1.0).0, s.actors[0].figure(1.0));
}

#[test]
fn arm_penetration_refuses_acquisition_even_if_the_goal_is_reachable() {
    let mut s = scene(0.35);
    let actor = &s.actors[0];
    let shoulder = actor.model.rig.joint("arm_r").unwrap();
    let pos = actor.figure(1.0).joints[shoulder] - Vec3::UP * 0.25;
    s.props.push(Prop::new("barrier", Shape::Box { sx: 0.6, sy: 0.5, sz: 0.6 }, pos));
    compile(&mut s);
    assert_eq!(s.grab_reports[0].status, GrabStatus::Obstructed);
    assert!(s.props[0].motion.is_empty());
}

#[test]
fn fixed_support_on_the_same_hand_prevents_a_conflicting_acquisition() {
    let mut s = scene(0.35);
    let a = &mut s.actors[0];
    let hand = a.model.rig.joint("hand_r").unwrap();
    let chain = mavian_scene::actions::chain(a, hand).unwrap();
    let mut pin = Contact::new("pin-hand", chain, Target::Point(a.figure(1.0).joints[hand]), 0.0, 3.0);
    pin.priority = Priority::Support;
    a.contacts.push(pin);
    compile(&mut s);
    assert_eq!(s.grab_reports[0].status, GrabStatus::Conflict);
    assert!(s.props[0].motion.is_empty());
}

#[test]
fn an_occupied_hand_or_prop_cannot_be_acquired_again() {
    for same_prop in [true, false] {
        let mut s = scene(0.35);
        let mut second = s.actions[0].clone();
        if let Action::Grab { at, prop, .. } = &mut second {
            *at = 1.8;
            if !same_prop {
                let mut p = s.props[0].clone();
                p.name = "second-cup".into();
                p.at.x += 0.3;
                s.props.push(p);
                *prop = 1;
            }
        }
        s.actions.push(second);
        compile(&mut s);
        assert_eq!(s.grab_reports[0].status, GrabStatus::Attached);
        assert_eq!(s.grab_reports[1].status, if same_prop { GrabStatus::PropBusy } else { GrabStatus::HandBusy });
        assert_eq!(s.props[0].motion.len(), 1);
        if !same_prop { assert!(s.props[1].motion.is_empty()); }
    }
}

#[test]
fn held_offset_uses_the_final_deformed_hand_and_sampling_has_no_event_history() {
    let mut s = scene(0.35);
    s.actors[0].squash = 0.3;
    s.actors[0].path = Keys::new(Ease::Linear).key(0.0, (0.0, 0.0)).key(2.5, (0.2, 0.0)).into();
    compile(&mut s);
    assert_eq!(s.grab_reports[0].status, GrabStatus::Attached);
    assert!(s.posed(0, 1.0).1.is_some());
    let Motion::Held { hand, offset, .. } = s.props[0].segment(1.0).unwrap() else { panic!() };
    let hand = *hand;
    let offset = *offset;
    let mut samples = Vec::new();
    for w in [0.5, 1.0, 1.1, 1.5, 2.4] {
        let fig = s.posed(0, w).0;
        let prop = s.prop_rest(0, w);
        if w >= 1.0 {
            assert!((prop.0 - fig.joints[hand] - Mat3::rot_y(s.actors[0].facing.at(w)) * offset).length() < 1e-9);
        }
        samples.push((w, fig, prop));
    }
    assert!((s.prop_rest(0, 1.0).0 - s.props[0].at).length() < 1e-9);
    let reports = s.grab_reports.clone();
    s.compile().unwrap();
    assert_eq!(reports, s.grab_reports);
    for (w, fig, prop) in samples.into_iter().rev() {
        assert_eq!(fig, s.posed(0, w).0);
        assert_eq!(prop, s.prop_rest(0, w));
    }
}

#[test]
fn later_overlapping_contacts_cannot_invalidate_a_verified_acquisition_silently() {
    let mut s = scene(0.35);
    let hand = s.actors[0].model.rig.joint("hand_r").unwrap();
    let mut other = Actor::new(mavian_models::mannequin::mannequin());
    other.path = (0.0, 4.0).into();
    s.actors.push(other);
    let mut strike = mavian_scene::Strike::new(0, hand, 1, 1.1);
    strike.windup = 0.3;
    s.actions.push(Action::Strike(strike));
    let err = s.compile().unwrap_err();
    assert!(err.0.contains("later actions change the verified contact"), "{err}");
}

#[test]
fn hand_envelope_scales_with_the_actor_and_prop_dimensions() {
    for scale in [0.5, 1.0, 2.0] {
        let mut s = scene(0.35);
        s.actors[0].scale = scale;
        s.props[0].at = s.props[0].at * scale;
        s.props[0].shape = Shape::Box { sx: 0.13 * scale, sy: 0.17 * scale, sz: 0.13 * scale };
        compile(&mut s);
        let r = &s.grab_reports[0];
        assert_eq!(r.status, GrabStatus::Attached);
        let (pos, rot) = s.prop_rest(0, 1.0);
        let obstacle = mavian_scene::obstacle::Obstacle { prop: 0, shape: s.props[0].shape.clone(), pos, rot };
        assert!((obstacle.distance(r.actual).0 - 0.065 * scale).abs() < 1e-8);
    }
}

#[test]
fn a_locked_hand_already_touching_the_grip_can_acquire_without_pose_correction() {
    let mut s = scene(0.35);
    let a = &mut s.actors[0];
    a.pose_lock = true;
    let hand = a.model.rig.joint("hand_r").unwrap();
    let point = a.figure(1.0).joints[hand];
    s.props[0].at = point + Vec3 { x: -0.13, y: -0.102, z: 0.0 };
    compile(&mut s);
    assert_eq!(s.grab_reports[0].status, GrabStatus::Attached);
    assert_eq!(s.posed(0, 1.0).0, s.actors[0].figure(1.0));
}

#[test]
fn acquisition_cannot_create_a_self_held_target_cycle() {
    let mut s = scene(0.35);
    let a = &mut s.actors[0];
    let hand = a.model.rig.joint("hand_r").unwrap();
    let chain = mavian_scene::actions::chain(a, hand).unwrap();
    a.contacts.push(Contact::new("surface", chain, Target::OnProp { prop: 0, local: Vec3 { x: 0.0, y: 0.102, z: -0.13 } }, 0.0, 3.0));
    let err = s.compile().unwrap_err();
    assert!(err.0.contains("cyclic contact dependency"), "{err}");
    assert!(s.props[0].motion.is_empty());
}

#[test]
fn placement_cannot_start_its_held_motion_before_acquisition() {
    let mut s = scene(0.35);
    let hand = s.actors[0].model.rig.joint("hand_r").unwrap();
    s.actions.push(Action::Release { actor: 0, hand, prop: 0, at: 1.1, onto: Onto::Point(s.props[0].at), reach: 0.4, settle: 0.35 });
    assert!(s.compile().unwrap_err().0.contains("begins before acquisition"));
}
