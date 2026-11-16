use mavian_core::{Ease, Keys, Mat3, Vec3};
use mavian_scene::{Action, Actor, Motion, Onto, PlacementStatus, Prop, Shape, Shot, pk};

fn scene(mode: &str) -> Shot {
    let mut model = (*mavian_models::mannequin::mannequin()).clone();
    let carry = model.rig.pose().base(model.pose("stand").unwrap()).set("arm_r", (65, 6)).set("elbow_r", 100).build().unwrap();
    model.add_pose("carry", carry).unwrap();
    let mut poses = Keys::new(Ease::InOut).key(0.0, pk("stand")).key(1.0, pk("stand")).key(1.45, pk("carry")).key(1.8, pk("carry"));
    if mode == "valid" {
        poses = poses.key(2.75, pk("stand"));
    }
    let actor = Actor::new(std::sync::Arc::new(model)).poses(mavian_scene::Poses::Keys(poses));
    let hand = actor.model.rig.joint("hand_r").unwrap();
    let mut props = vec![
        Prop::new("cup", Shape::Box { sx: 0.13, sy: 0.17, sz: 0.13 }, Vec3 { x: -0.30, y: 1.0, z: 0.35 }),
        Prop::new("pedestal", Shape::Box { sx: 0.4, sy: 1.0, sz: 0.4 }, Vec3 { x: -0.30, y: 0.0, z: 0.35 }),
    ];
    let support = match mode {
        "high" => {
            props.push(Prop::new("high", Shape::Box { sx: 0.4, sy: 0.1, sz: 0.4 }, Vec3 { x: -0.14, y: 1.9, z: 0.6 }));
            2
        }
        "edge" => {
            props.push(Prop::new("narrow", Shape::Box { sx: 0.08, sy: 0.15, sz: 0.2 }, Vec3 { x: -0.14, y: 1.0, z: 0.35 }));
            2
        }
        "blocked" => {
            props.push(Prop::new("barrier", Shape::Box { sx: 0.10, sy: 0.20, sz: 0.04 }, Vec3 { x: -0.16, y: 1.0, z: 0.31 }));
            1
        }
        _ => 1,
    };
    Shot {
        actors: vec![actor],
        props,
        actions: vec![
            Action::Grab { actor: 0, hand, prop: 0, at: 1.0, reach: 0.4, settle: 0.35, height: 0.6, step: None },
            Action::Release { actor: 0, hand, prop: 0, at: 2.4, onto: Onto::Prop(support), reach: 0.6, settle: 0.35 },
        ],
        ..Shot::default()
    }
}

#[test]
fn named_placement_requires_reach_support_fit_and_clearance() {
    for (mode, expected) in [
        ("valid", PlacementStatus::Placed),
        ("high", PlacementStatus::Unreachable),
        ("blocked", PlacementStatus::Obstructed),
        ("edge", PlacementStatus::OutsideSupport),
    ] {
        let mut s = scene(mode);
        s.compile().unwrap_or_else(|e| panic!("{mode}: {e}"));
        s.validate().unwrap();
        eprintln!("{mode}: {:?}", s.placement_reports);
        assert!(s.grab_reports[0].attached(), "{mode} initial acquisition failed: {:?}", s.grab_reports);
        assert_eq!(s.placement_reports.len(), 1);
        assert_eq!(s.placement_reports[0].status, expected, "{mode}");
        let r = &s.placement_reports[0];
        if r.placed() {
            assert!(matches!(s.props[0].segment(2.4), Some(Motion::Ride { .. })));
            assert!(r.error <= r.tolerance);
            assert!(r.support_gap.abs() <= r.tolerance);
            assert!(r.edge_margin >= -r.tolerance);
            let before = s.prop_rest(0, 2.4 - 1e-6);
            let after = s.prop_rest(0, 2.4 + 1e-6);
            assert!((before.0 - after.0).length() < 1e-4);
            assert_eq!(before.1, after.1);
        } else {
            assert!(matches!(s.props[0].segment(2.9), Some(Motion::Held { .. })), "{mode} lost the binding");
            assert_eq!(s.prop_rest(0, r.at).0, r.actual);
        }
    }
}

#[test]
fn rejected_placements_keep_the_grip_and_have_a_continuous_clear_return() {
    for mode in ["high", "blocked", "edge"] {
        let mut s = scene(mode);
        s.compile().unwrap();
        let Motion::Held { hand, offset, .. } = s.props[0].segment(1.5).unwrap() else { panic!() };
        let (hand, offset) = (*hand, *offset);
        let chain = mavian_scene::actions::chain(&s.actors[0], hand).unwrap();
        let mut previous: Option<std::sync::Arc<mavian_rig::Figure>> = None;
        let mut snapshots = Vec::new();
        let mut worst = 0.0f64;
        let mut returned = (0.0f64, 0.0f64);
        let mut velocity = [Vec3::ZERO; 2];
        for n in 0..=1440 {
            let w = n as f64 / 480.0;
            let fig = s.posed(0, w).0;
            if w >= 1.0 {
                assert!(
                    (s.prop_rest(0, w).0 - fig.joints[hand] - Mat3::rot_y(s.actors[0].facing.at(w)) * offset).length() < 1e-8,
                    "{mode} grip changed at {w}"
                );
                assert!(s.held_prop_reports(w).is_empty(), "{mode} volume clearance fails at {w}");
                let at = s.prop_rest(0, w).0;
                let Shape::Box { sx, sy, sz } = s.props[0].shape else { panic!() };
                for i in 1..s.props.len() {
                    let other = s.prop_rest(i, w).0;
                    let Shape::Box { sx: ox, sy: oy, sz: oz } = s.props[i].shape else { panic!() };
                    let dx = (at.x + sx * 0.5).min(other.x + ox * 0.5) - (at.x - sx * 0.5).max(other.x - ox * 0.5);
                    let dy = (at.y + sy).min(other.y + oy) - at.y.max(other.y);
                    let dz = (at.z + sz * 0.5).min(other.z + oz * 0.5) - (at.z - sz * 0.5).max(other.z - oz * 0.5);
                    assert!(dx.min(dy).min(dz) <= 0.001, "{mode} held volume enters {} at {w}", s.props[i].name);
                }
            }
            assert!(s.obstacle_reports(w).is_empty(), "{mode} arm enters a prop at {w}");
            assert!(s.self_collision_reports(w).is_empty(), "{mode} arm enters body at {w}");
            if let Some(old) = previous {
                for j in [chain[1], hand] {
                    worst = worst.max((fig.joints[j] - old.joints[j]).length());
                }
            }
            if let Some((old, _)) = snapshots.last() {
                let old: &std::sync::Arc<mavian_rig::Figure> = old;
                for (k, j) in [chain[1], hand].into_iter().enumerate() {
                    let delta = fig.joints[j] - old.joints[j];
                    if w >= 1.8 {
                        returned.0 = returned.0.max(delta.length());
                        returned.1 = returned.1.max((delta - velocity[k]).length());
                    }
                    velocity[k] = delta;
                }
            }
            previous = Some(fig.clone());
            snapshots.push((fig, s.prop_rest(0, w)));
        }
        eprintln!("{mode} max displacement at 480Hz: {worst:.9}m");
        assert!(worst < 0.02, "{mode} discontinuous return: {worst}");
        eprintln!("{mode} placement/return max step and second difference: {returned:?}");
        assert!(returned.0 < 0.005 && returned.1 < 0.0005, "{mode} abrupt placement/return: {returned:?}");
        let reports = s.placement_reports.clone();
        s.compile().unwrap();
        assert_eq!(reports, s.placement_reports);
        for n in (0..=1440).rev() {
            assert_eq!(snapshots[n], (s.posed(0, n as f64 / 480.0).0, s.prop_rest(0, n as f64 / 480.0)));
        }
    }
}

#[test]
fn placement_preserves_yawed_scaled_final_transforms() {
    for scale in [0.5, 1.0, 2.0] {
        for heading in [0.0, 73.0, 180.0] {
            let mut s = scene("valid");
            let rot = Mat3::rot_y(heading);
            let origin = Vec3 { x: 1.3, y: 0.0, z: -0.8 };
            s.actors[0].scale = scale;
            s.actors[0].facing = heading.into();
            s.actors[0].path = (origin.x, origin.z).into();
            for p in &mut s.props {
                p.at = origin + rot * (p.at * scale);
                p.turn = heading;
                if let Shape::Box { sx, sy, sz } = &mut p.shape {
                    *sx *= scale;
                    *sy *= scale;
                    *sz *= scale;
                }
            }
            s.compile().unwrap();
            let r = &s.placement_reports[0];
            assert!(r.placed(), "{r:?}");
            assert!(r.error < 1e-8);
            assert!(r.support_gap.abs() < 1e-8);
            let before = s.prop_rest(0, 2.4 - 1e-6);
            let after = s.prop_rest(0, 2.4 + 1e-6);
            assert!((before.0 - after.0).length() < 1e-4);
            assert_eq!(before.1, after.1);
        }
    }
}

#[test]
fn locked_placement_reports_failure_without_changing_the_pose_or_binding() {
    let mut actor = Actor::new(mavian_models::mannequin::mannequin());
    actor.pose_lock = true;
    let hand = actor.model.rig.joint("hand_r").unwrap();
    let raw = actor.figure(1.0);
    let cup = Prop::new("cup", Shape::Box { sx: 0.13, sy: 0.17, sz: 0.13 }, raw.joints[hand] + Vec3 { x: -0.13, y: -0.102, z: 0.0 });
    let support = Prop::new("support", Shape::Box { sx: 0.3, sy: 0.1, sz: 0.3 }, Vec3 { x: -0.5, y: 0.7, z: 0.4 });
    let mut s = Shot {
        actors: vec![actor],
        props: vec![cup, support],
        actions: vec![
            Action::Grab { actor: 0, hand, prop: 0, at: 1.0, reach: 0.4, settle: 0.35, height: 0.6, step: None },
            Action::Release { actor: 0, hand, prop: 0, at: 2.4, onto: Onto::Prop(1), reach: 0.6, settle: 0.35 },
        ],
        ..Shot::default()
    };
    s.compile().unwrap();
    assert!(s.grab_reports[0].attached());
    assert_eq!(s.placement_reports[0].status, PlacementStatus::PoseLocked);
    for w in [1.0, 1.8, 2.0, 2.4, 2.7, 3.0] {
        assert_eq!(s.posed(0, w).0, s.actors[0].figure(w));
        assert!(matches!(s.props[0].segment(w), Some(Motion::Held { .. })));
    }
}

#[test]
fn missed_acquisition_does_not_create_a_placement_and_explicit_drop_still_falls() {
    let mut missed = scene("valid");
    missed.props[0].at.z = 3.0;
    missed.compile().unwrap();
    assert_eq!(missed.placement_reports[0].status, PlacementStatus::NotHeld);
    assert!(missed.props[0].motion.is_empty());
    // An intentional unsupported drop has no platform below the item.
    let mut dropped = scene("drop");
    dropped.props.truncate(1);
    if let Action::Release { onto, .. } = &mut dropped.actions[1] {
        *onto = Onto::Nothing;
    }
    dropped.compile().unwrap();
    assert!(dropped.grab_reports[0].attached());
    assert!(dropped.placement_reports.is_empty());
    assert!(matches!(dropped.props[0].segment(2.4), Some(Motion::Fall { .. })));
    assert!(dropped.prop_rest(0, 2.8).0.y < dropped.prop_rest(0, 2.4).0.y);
}

#[test]
fn accepted_placement_rides_a_support_that_moves_later() {
    let mut s = scene("valid");
    let mut mover = Actor::new(mavian_models::mannequin::mannequin());
    mover.path = Keys::new(Ease::Linear).key(0.0, (4.0, 0.0)).key(2.8, (4.0, 0.0)).key(3.5, (4.3, 0.0)).into();
    let hands = [mover.model.rig.joint("hand_l").unwrap(), mover.model.rig.joint("hand_r").unwrap()];
    s.actors.push(mover);
    s.length = 3.5;
    s.actions.push(Action::Push { actor: 1, prop: 1, start: 2.8, end: 3.5, hands, reach: 0.2, settle: 0.2, height: 0.8, spread: 0.1, step: None });
    s.compile().unwrap();
    assert!(s.placement_reports[0].placed());
    let at = s.prop_rest(0, 2.4).0;
    let platform = s.prop_rest(1, 2.4).0;
    for w in [2.4, 2.8, 3.0, 3.5] {
        assert!((s.prop_rest(0, w).0 - s.prop_rest(1, w).0 - at + platform).length() < 1e-8);
    }
    assert!(s.prop_rest(0, 3.5).0.x - at.x > 0.29);
}

#[test]
fn a_refused_placement_can_be_followed_by_a_successful_explicit_retry() {
    let mut s = scene("high");
    s.length = 4.0;
    s.actors[0].poses = mavian_scene::Poses::Keys(
        Keys::new(Ease::InOut).key(0.0, pk("stand")).key(1.0, pk("stand")).key(1.45, pk("carry")).key(3.0, pk("carry")).key(3.85, pk("stand")),
    );
    let hand = s.actors[0].model.rig.joint("hand_r").unwrap();
    s.actions.push(Action::Release { actor: 0, hand, prop: 0, at: 3.5, onto: Onto::Prop(1), reach: 0.5, settle: 0.35 });
    s.compile().unwrap();
    assert_eq!(s.placement_reports.len(), 2);
    assert_eq!(s.placement_reports[0].status, PlacementStatus::Unreachable);
    assert!(s.placement_reports[1].placed());
    assert!(matches!(s.props[0].segment(2.9), Some(Motion::Held { .. })));
    assert!(matches!(s.props[0].segment(3.6), Some(Motion::Ride { support: 1, .. })));
}

#[test]
fn placement_uses_the_final_hand_under_nonidentity_body_deformation() {
    let mut s = scene("valid");
    s.actors[0].squash = 0.3;
    s.actors[0].layers.push(mavian_scene::actor::Layer {
        angles: Vec::new(),
        shift: (0.0, 0.0),
        lift: 0.3,
        weight: Keys::new(Ease::Linear).key(0.0, 0.0).key(2.1, 0.0).key(2.2, 1.0).key(2.25, 1.0).key(2.39, 0.0).into(),
    });
    s.compile().unwrap();
    assert!(s.grab_reports[0].attached());
    let r = &s.placement_reports[0];
    assert!(r.placed(), "{r:?}");
    assert_ne!(s.posed(0, r.at).1.unwrap(), Mat3::IDENT);
    let Motion::Held { hand, offset, .. } = s.props[0].segment(r.at - 1e-6).unwrap() else { panic!() };
    let hand = *hand;
    let offset = *offset;
    assert!((s.posed(0, r.at).0.joints[hand] + offset - r.actual).length() < 1e-8);
    assert!((s.prop_rest(0, r.at - 1e-6).0 - s.prop_rest(0, r.at + 1e-6).0).length() < 1e-4);
}
