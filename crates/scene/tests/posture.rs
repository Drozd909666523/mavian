use mavian_core::{Mat3, Track, Vec3};
use mavian_scene::{Actor, Contact, ContactStatus, Priority, Prop, Shape, Shot, Target};

fn scene(distance: f64, direction: Vec3, scale: f64, heading: f64) -> Shot {
    let mut actor = Actor::new(mavian_models::mannequin::mannequin()).scale(scale).facing(heading);
    actor.body_reach = 30.0;
    let hand = actor.model.rig.joint("hand_r").unwrap();
    let chain = mavian_scene::actions::chain(&actor, hand).unwrap();
    let fig = actor.figure(0.0);
    let target = fig.joints[chain[0]] + Mat3::rot_y(heading) * direction * (distance * scale);
    actor.contacts.push(Contact::new("reach", chain, Target::Point(target), 0.0, 1.0));
    Shot { actors: vec![actor], ..Shot::default() }
}
fn check_lengths_and_feet(s: &Shot, w: f64) {
    let a = &s.actors[0];
    let raw = a.figure(w);
    let posed = s.posed(0, w).0;
    for j in 0..raw.joints.len() {
        if let Some(parent) = a.model.rig.parent(j) {
            let before = (raw.joints[j] - raw.joints[parent]).length();
            let after = (posed.joints[j] - posed.joints[parent]).length();
            assert!((before - after).abs() < 1e-9, "bone {} changed length: {before} -> {after}", a.model.rig.joints()[j].name);
        }
    }
    for name in ["pelvis", "leg_l", "leg_r", "knee_l", "knee_r", "foot_l", "foot_r", "toe_l", "toe_r"] {
        let j = a.model.rig.joint(name).unwrap();
        assert_eq!(raw.joints[j], posed.joints[j], "assistance moved {name}");
    }
}

#[test]
fn default_contacts_preserve_authored_body_and_report_insufficient_reach() {
    let mut actor = Actor::new(mavian_models::mannequin::mannequin()).poses(mavian_scene::Poses::Fixed("crouch".into()));
    let raw = actor.figure(0.5);
    let hand = actor.model.rig.joint("hand_r").unwrap();
    let chain = mavian_scene::actions::chain(&actor, hand).unwrap();
    let target = raw.joints[chain[0]] + Vec3 { x: 0., y: 0., z: 0.63 };
    actor.contacts.push(Contact::new("too-far-without-permission", chain, Target::Point(target), 0., 1.));
    let s = Shot { actors: vec![actor], ..Shot::default() };
    let posed = s.posed(0, 0.5).0;
    for j in 0..raw.joints.len() {
        if j != chain[1] && j != chain[2] {
            assert_eq!(posed.joints[j], raw.joints[j], "contact changed an authored joint without permission");
        }
    }
    assert_eq!(posed.rots, raw.rots);
    let reports = s.contact_reports(0.5);
    assert_eq!(reports[0].1.status, ContactStatus::Unreachable);
    assert!(reports[0].1.error > 0.09);
    check_lengths_and_feet(&s, 0.5);
}

#[test]
fn locked_pose_keeps_animation_and_deformation_while_reporting_conflicts_and_collisions() {
    use mavian_core::{Ease, Keys};
    let actor = || {
        Actor::new(mavian_models::mannequin::mannequin())
            .poses(Keys::new(Ease::InOut).key(0., mavian_scene::pk("stand")).key(1., mavian_scene::pk("crouch")))
            .path(Keys::new(Ease::Linear).key(0., (0., 0.)).key(1., (2.5, 0.)))
            .squash(0.2)
    };
    let baseline = Shot { actors: vec![actor()], ..Shot::default() };
    let mut locked = actor();
    locked.pose_lock = true;
    locked.body_reach = 30.0;
    let hand = locked.model.rig.joint("hand_r").unwrap();
    let chain = mavian_scene::actions::chain(&locked, hand).unwrap();
    locked.contacts.push(Contact::new("incompatible", chain, Target::Point(Vec3 { x: 0., y: 2., z: 2. }), 0., 1.));
    let wall = Prop::new("blocking-wall", Shape::Box { sx: 8., sy: 3., sz: 4. }, Vec3 { x: 1., y: 0., z: 0. });
    let shot = Shot { actors: vec![locked], props: vec![wall], ..Shot::default() };
    shot.validate().unwrap();
    let mut samples = Vec::new();
    for w in [0.1, 0.4, 0.7, 0.9] {
        let (expected, deformation) = baseline.posed(0, w);
        assert_ne!(deformation.unwrap(), Mat3::IDENT, "the fixture must deform the skeleton");
        let actual = shot.posed(0, w).0;
        assert_eq!(actual, expected, "automatic corrections changed the locked pose at {w}");
        assert_eq!(shot.actors[0].figure_ik(w, &shot.world()), shot.actors[0].figure(w));
        let reports = shot.contact_reports(w);
        let contact = &reports.iter().find(|(_, r)| r.name == "incompatible").unwrap().1;
        assert_eq!(contact.status, ContactStatus::Conflict);
        assert_eq!(contact.actual, actual.joints[hand]);
        assert!(contact.error > 0.1);
        assert!(reports.iter().any(|(_, r)| r.name.starts_with("collision:") && r.status == ContactStatus::Obstructed));
        samples.push((w, actual));
    }
    assert_ne!(samples[0].1, samples[3].1, "locking must preserve the animated pose, not freeze the actor in time");
    for (w, expected) in samples.into_iter().rev() {
        assert_eq!(shot.posed(0, w).0, expected);
    }
}

#[test]
fn nearby_targets_keep_the_authored_torso_exactly() {
    let s = scene(0.40, Vec3 { x: 0., y: 0., z: 1. }, 1.0, 0.0);
    let a = &s.actors[0];
    let f = s.posed(0, 0.5).0;
    for name in ["spine", "chest", "head", "arm_l", "arm_r"] {
        let j = a.model.rig.joint(name).unwrap();
        assert_eq!(f.joints[j], a.figure(0.5).joints[j]);
    }
    assert!(s.contact_reports(0.5)[0].1.error < 1e-9);
    check_lengths_and_feet(&s, 0.5);
}

#[test]
fn body_assistance_reaches_forward_sideways_and_back_at_different_scales() {
    for scale in [0.6, 1.0, 1.7] {
        for heading in [0., 75., 180.] {
            for direction in [Vec3 { x: 0., y: 0., z: 1. }, Vec3 { x: 0., y: 0., z: -1. }, Vec3 { x: 1., y: 0., z: 0. }] {
                let s = scene(0.63, direction, scale, heading);
                let report = &s.contact_reports(0.5)[0].1;
                assert!(report.error < 0.002, "scale {scale}, heading {heading}, direction {direction:?}: {report:?}");
                assert_eq!(report.status, ContactStatus::Satisfied);
                check_lengths_and_feet(&s, 0.5);
            }
        }
    }
}

#[test]
fn both_hands_share_body_assistance_independent_of_contact_order() {
    let mut s = scene(0.63, Vec3 { x: 0., y: 0., z: 1. }, 1., 0.);
    let a = &mut s.actors[0];
    let hand = a.model.rig.joint("hand_l").unwrap();
    let chain = mavian_scene::actions::chain(a, hand).unwrap();
    let target = a.figure(0.5).joints[chain[0]] + Vec3 { x: 0., y: 0., z: 0.63 };
    a.contacts.push(Contact::new("other-hand", chain, Target::Point(target), 0., 1.));
    let before = s.posed(0, 0.5).0;
    for (_, r) in s.contact_reports(0.5) {
        assert_eq!(r.status, ContactStatus::Satisfied);
        assert!(r.error < 1e-8);
    }
    check_lengths_and_feet(&s, 0.5);
    s.actors[0].contacts.reverse();
    s.actors[0].forget();
    assert_eq!(before, s.posed(0, 0.5).0);
}

#[test]
fn excessive_distance_and_disabled_assistance_keep_explicit_residuals() {
    let mut s = scene(1.4, Vec3 { x: 0., y: 0., z: 1. }, 1., 0.);
    let report = &s.contact_reports(0.5)[0].1;
    assert_eq!(report.status, ContactStatus::Unreachable);
    assert!(report.error > 0.5);
    let rig = &s.actors[0].model.rig;
    let spine = rig.joint("spine").unwrap();
    let chest = rig.joint("chest").unwrap();
    let base = s.actors[0].figure(0.5);
    let posed = s.posed(0, 0.5).0;
    let angle = mavian_core::m::acos((base.joints[chest] - base.joints[spine]).norm().dot((posed.joints[chest] - posed.joints[spine]).norm()));
    assert!(angle.to_degrees() <= 30.0 + 1e-8);
    check_lengths_and_feet(&s, 0.5);
    s.actors[0].body_reach = 0.0;
    let disabled = s.posed(0, 0.5).0;
    assert_eq!(disabled.joints[chest], base.joints[chest]);
    assert!(s.contact_reports(0.5)[0].1.error > report.error);
}

#[test]
fn contact_ramps_and_reverse_sampling_have_no_body_pop() {
    let mut s = scene(0.65, Vec3 { x: 0., y: 0., z: 1. }, 1., 0.);
    s.actors[0].contacts[0].weight = mavian_core::Keys::new(mavian_core::Ease::InOut).key(0.0, 0.0).key(0.5, 1.0).key(1.0, 0.0).into();
    let mut samples = Vec::new();
    for i in 0..=240 {
        let w = i as f64 / 240.0;
        let f = s.posed(0, w).0;
        if let Some((_, prev)) = samples.last() {
            let prev: &std::sync::Arc<mavian_rig::Figure> = prev;
            for (j, (a, b)) in prev.joints.iter().zip(&f.joints).enumerate() {
                assert!((*a - *b).length() < 0.025, "joint {} jumps at {w}: {:?}", s.actors[0].model.rig.joints()[j].name, *b - *a);
            }
        }
        check_lengths_and_feet(&s, w);
        samples.push((w, f));
    }
    for (w, f) in samples.into_iter().rev() {
        assert_eq!(s.posed(0, w).0, f);
    }
}

#[test]
fn body_assistance_respects_props_and_explicit_upper_body_supports() {
    let mut s = scene(0.9, Vec3 { x: 0., y: 0., z: 1. }, 1., 0.);
    s.props.push(Prop::new("wall", Shape::Box { sx: 4., sy: 3., sz: 0.1 }, Vec3 { x: 0., y: 0., z: 0.45 }));
    let f = s.posed(0, 0.5).0;
    let obstacle = mavian_scene::obstacle::Obstacle { prop: 0, shape: s.props[0].shape.clone(), pos: s.props[0].at, rot: Mat3::IDENT };
    for part in &s.actors[0].model.parts {
        match part {
            mavian_rig::Part::Body { a, b, rx, rz, .. } => {
                assert!(obstacle.capsule(f.joints[*a], rx.0.max(rz.0), f.joints[*b], rx.1.max(rz.1)).0 >= -1e-6);
            }
            mavian_rig::Part::Head { joint, radius, .. } => assert!(obstacle.distance(f.joints[*joint]).0 >= radius - 1e-6),
            _ => (),
        }
    }
    assert!(s.contact_reports(0.5).iter().any(|(_, r)| r.name == "reach" && r.error > 0.01));
    s.props.clear();
    let mut pin = s.actors[0].contacts[0].clone();
    pin.name = "pinned-hand".into();
    pin.target = Target::Point(s.actors[0].figure(0.5).joints[pin.chain[2]]);
    pin.priority = Priority::Support;
    pin.weight = Track::Const(1.0);
    s.actors[0].contacts.push(pin);
    let chest = s.actors[0].model.rig.joint("chest").unwrap();
    assert_eq!(s.posed(0, 0.5).0.joints[chest], s.actors[0].figure(0.5).joints[chest]);
}
