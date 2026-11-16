use mavian_core::{Mat3, Track, Vec3};
use mavian_rig::Figure;
use mavian_scene::{Contact, ContactStatus, Priority, Target, contact::solve_resolved};
fn v(x: f64, y: f64, z: f64) -> Vec3 {
    Vec3 { x, y, z }
}
fn figure() -> Figure {
    Figure { joints: vec![v(0., 0., 0.), v(1., 0., 0.), v(1., 1., 0.)], rots: vec![Mat3::IDENT; 3], pivot: Vec3::ZERO, lift: 0.0 }
}
fn lengths(f: &Figure) -> [f64; 2] {
    [(f.joints[1] - f.joints[0]).length(), (f.joints[2] - f.joints[1]).length()]
}
#[test]
fn ramps_preserve_bone_lengths_and_stop_exactly_at_release() {
    let base = figure();
    let target = v(0.5, 0.5, 1.0);
    for weight in [0.0, 0.1, 0.25, 0.5, 0.9, 1.0] {
        let mut c = Contact::new("grip", [0, 1, 2], Target::Point(target), 0., 1.);
        c.weight = Track::Const(weight);
        let (f, r) = solve_resolved(&base, [(c.clone(), target)], 0.5);
        for (a, b) in lengths(&base).into_iter().zip(lengths(&f)) {
            assert!((a - b).abs() < 1e-10);
        }
        assert!((f.joints[2] - base.joints[2].lerp(target, weight)).length() < 1e-10);
        assert_eq!(solve_resolved(&base, [(c, target)], 1.).0, base);
        if weight > 0.0 {
            assert!(r[0].error < 1e-10);
        }
    }
}
#[test]
fn pulling_a_hand_toward_the_body_does_not_reverse_its_elbow() {
    let mut base = figure();
    base.joints = vec![Vec3::ZERO, v(0.0, -0.20, 0.20), v(0.0, -0.20, 0.46)];
    let target = v(0.0, -0.43, 0.15);
    for angle in [0.0, 75.0, 180.0] {
        let rotation = Mat3::rot_y(angle);
        let mut reference = base.clone();
        reference.joints.iter_mut().for_each(|p| *p = rotation * *p);
        for weight in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
            for pole in [None, Some(rotation * v(0.6, -1.0, 0.0))] {
                let target = rotation * target;
                let mut c = Contact::new("close-push", [0, 1, 2], Target::Point(target), 0.0, 1.0);
                c.weight = Track::Const(weight);
                c.pole = pole;
                let (f, reports) = solve_resolved(&reference, [(c, target)], 0.5);
                let direction = (f.joints[2] - f.joints[0]).norm();
                let upper = f.joints[1] - f.joints[0];
                let bend = rotation.transpose() * (upper - direction * upper.dot(direction)).norm();
                assert!(bend.z < 0.0, "authored bend reversed: angle {angle}, weight {weight}, pole {pole:?}: {bend:?}");
                assert_eq!(f.joints[0], reference.joints[0]);
                for (before, after) in lengths(&reference).into_iter().zip(lengths(&f)) {
                    assert!((before - after).abs() < 1e-10);
                }
                if weight > 0.0 {
                    assert!(reports[0].error < 1e-10);
                }
            }
        }
    }
}

#[test]
fn unreachable_target_has_an_explicit_residual_without_stretching() {
    let base = figure();
    let target = v(3., 0., 0.);
    let c = Contact::new("too-far", [0, 1, 2], Target::Point(target), 0., 1.);
    let (f, r) = solve_resolved(&base, [(c, target)], 0.5);
    assert_eq!(r[0].status, ContactStatus::Unreachable);
    assert!((r[0].error - 1.).abs() < 1e-10);
    assert!((r[0].reach_error - 1.).abs() < 1e-10);
    assert_eq!(f.joints[0], base.joints[0]);
    assert_eq!(lengths(&f), [1., 1.]);
}
#[test]
fn support_wins_over_a_conflicting_contact_independent_of_input_order() {
    let base = figure();
    let support = base.joints[2];
    let target = v(0., 0., 2.);
    let mut a = Contact::new("pin", [0, 1, 2], Target::Point(support), 0., 1.);
    a.priority = Priority::Support;
    let b = Contact::new("reach", [0, 1, 2], Target::Point(target), 0., 1.);
    let x = solve_resolved(&base, [(a.clone(), support), (b.clone(), target)], 0.5);
    let y = solve_resolved(&base, [(b, target), (a, support)], 0.5);
    assert_eq!(x, y);
    assert_eq!(x.0.joints[2], support);
    assert_eq!(x.1[0].status, ContactStatus::Satisfied);
    assert_eq!(x.1[1].status, ContactStatus::Conflict);
}
#[test]
fn folded_and_too_close_targets_remain_finite_and_keep_lengths() {
    for target in [Vec3::ZERO, v(1e-12, 0., 0.), v(0., 2., 0.)] {
        let c = Contact::new("fold", [0, 1, 2], Target::Point(target), 0., 1.);
        let (f, _) = solve_resolved(&figure(), [(c, target)], 0.5);
        assert!(f.joints.iter().copied().all(mavian_scene::contact::finite));
        for x in lengths(&f) {
            assert!((x - 1.).abs() < 1e-8);
        }
    }
    let mut base = figure();
    base.joints[2] = v(1., 0.5, 0.);
    let c = Contact::new("inner-limit", [0, 1, 2], Target::Point(Vec3::ZERO), 0., 1.);
    let (f, r) = solve_resolved(&base, [(c, Vec3::ZERO)], 0.5);
    assert_eq!(r[0].status, ContactStatus::Unreachable);
    assert!((r[0].error - 0.5).abs() < 1e-8);
    for (a, b) in lengths(&f).into_iter().zip([1., 0.5]) {
        assert!((a - b).abs() < 1e-10);
    }
}
#[test]
fn self_held_target_is_rejected_instead_of_recursing() {
    use mavian_scene::{Actor, Motion, Prop, Shape, Shot};
    let mut a = Actor::new(mavian_models::mannequin::mannequin());
    let hand = a.model.rig.joint("hand_r").unwrap();
    let chain = mavian_scene::actions::chain(&a, hand).unwrap();
    a.contacts.push(Contact::new("cycle", chain, Target::OnProp { prop: 0, local: Vec3::ZERO }, 0., 1.));
    let mut p = Prop::new("cup", Shape::Sphere { r: 0.1 }, Vec3::ZERO);
    p.motion.push((0.0, Motion::Held { actor: 0, hand, offset: Vec3::ZERO, yaw: 0.0 }));
    let s = Shot { actors: vec![a], props: vec![p], ..Shot::default() };
    assert!(s.validate().unwrap_err().0.contains("cyclic contact"));
    let f = s.actors[0].figure_ik(0.5, &s.world());
    assert!(f.joints.iter().copied().all(mavian_scene::contact::finite));
}

#[test]
fn prop_local_contact_tracks_rotation_and_motion_after_squash() {
    use mavian_core::{Ease, Keys};
    use mavian_scene::{Actor, Motion, Prop, Shape, Shot};
    let mut a = Actor::new(mavian_models::mannequin::mannequin()).squash(0.2);
    let hand = a.model.rig.joint("hand_r").unwrap();
    let chain = mavian_scene::actions::chain(&a, hand).unwrap();
    let hp = a.figure(0.0).joints[hand];
    let mut p = Prop::new("moving", Shape::Box { sx: 0.2, sy: 0.2, sz: 0.2 }, hp);
    p.motion.push((0.0, Motion::Slide { actor: 0, offset: hp, y: hp.y, rot: Mat3::rot_y(90.0) }));
    a.path = Keys::new(Ease::Linear).key(0.0, (0.0, 0.0)).key(1.0, (2.5, 0.0)).into();
    a.forget();
    let local = v(0.0, 0.1, 0.17);
    a.contacts.push(Contact::new("surface", chain, Target::OnProp { prop: 0, local }, 0.0, 1.0));
    let s = Shot { actors: vec![a], props: vec![p], ..Shot::default() };
    s.validate().unwrap();
    assert_ne!(s.posed(0, 0.4).1.unwrap(), Mat3::IDENT, "the fixture must actually deform the skeleton");
    let mut samples = Vec::new();
    for w in [0.1, 0.4, 0.7, 0.9] {
        let f = s.posed(0, w).0;
        let (p, r) = s.prop_rest(0, w);
        let target = p + r * local;
        assert!((f.joints[hand] - target).length() < 1e-8);
        let reports = s.contact_reports(w);
        assert_eq!(reports[0].1.status, ContactStatus::Satisfied);
        samples.push((w, f));
    }
    assert!((samples[0].1.joints[hand] - samples[3].1.joints[hand]).length() > 0.05);
    for (w, f) in samples.into_iter().rev() {
        assert_eq!(f, s.posed(0, w).0);
    }
}

#[test]
fn malformed_contacts_fail_before_rendering() {
    use mavian_scene::{Actor, Shot};
    let mut a = Actor::new(mavian_models::mannequin::mannequin());
    let hand = a.model.rig.joint("hand_r").unwrap();
    let chain = mavian_scene::actions::chain(&a, hand).unwrap();
    let mut c = Contact::new("bad", chain, Target::Point(Vec3::ZERO), 0., 1.);
    c.weight = Track::Const(f64::NAN);
    a.contacts.push(c.clone());
    let mut s = Shot { actors: vec![a], ..Shot::default() };
    assert!(s.validate().unwrap_err().0.contains("weights"));
    c.weight = Track::Const(1.0);
    c.target = Target::OnProp { prop: 99, local: Vec3::ZERO };
    s.actors[0].contacts = vec![c];
    assert!(s.validate().unwrap_err().0.contains("prop 99"));
}

#[test]
fn reachable_support_stays_fixed_while_the_body_moves() {
    use mavian_core::{Ease, Keys};
    use mavian_scene::{Actor, Shot};
    let mut actor = Actor::new(mavian_models::mannequin::mannequin()).poses(mavian_scene::Poses::Fixed("crouch".into()));
    let foot = actor.model.rig.foot(mavian_rig::Side::L).unwrap();
    let chain = mavian_scene::actions::chain(&actor, foot.heel).unwrap();
    let start = actor.figure(0.0);
    let anchor = start.joints[foot.heel];
    let mut pin = Contact::new("foot-pin", chain, Target::Point(anchor), 0.0, 1.0);
    pin.priority = Priority::Support;
    pin.follower = Some((foot.toe, start.joints[foot.toe] - anchor));
    actor.contacts.push(pin);
    actor.path = Keys::new(Ease::Linear).key(0.0, (0.0, 0.0)).key(1.0, (0.1, 0.0)).into();
    actor.forget();
    let shot = Shot { actors: vec![actor], ..Shot::default() };
    shot.validate().unwrap();
    for w in [0.1, 0.3, 0.6, 0.9] {
        let fig = shot.posed(0, w).0;
        assert!((fig.joints[foot.heel] - anchor).length() < 1e-9);
        assert!((fig.joints[foot.toe] - start.joints[foot.toe]).length() < 1e-9);
        assert_eq!(shot.contact_reports(w)[0].1.status, ContactStatus::Satisfied);
    }
}
