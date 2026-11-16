use mavian_core::Vec3;
use mavian_frame::Layers;
use mavian_scene::{Action, Shot};
use std::path::Path;
fn load(name: &str) -> Shot {
    mavian_py::load_file(&Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../shots/{name}.py"))).unwrap().remove(0).1
}
#[test]
fn interact_cup_covers_hand_during_grab_carry_and_release() {
    let mut shot = load("interact");
    shot.size = (640, 360);
    let tl = shot.timeline();
    let cpu = mavian_cpu::Cpu::with_threads(2);
    for n in [65, 69, 70, 71, 75, 100, 135, 139, 140, 141, 145] {
        let f = shot.frame(&tl, n);
        let cup = shot.props.iter().find(|p| p.name == "cup").unwrap();
        let (pos, _) = cup.rest(f.w, &shot.world());
        let q = f.cam.project(pos + Vec3::UP * (cup.shape.top() * 0.5)).unwrap();
        let px = cpu.render(&shot, &f, Layers { paint: true, lines: true, ids: false });
        // The centre of this original shot's front face must be cup-coloured.
        // The old two-pass implementation drew the hand's ring through it.
        let (x, y) = (q.x.round() as usize, q.y.round() as usize);
        for yy in y - 5..=y + 5 {
            for xx in x - 4..=x + 4 {
                let p = &px[(yy * 640 + xx) * 4..][..3];
                assert!(p[0] > 180 && p[1] > 170 && p[2] > 150, "interact frame {n}: hand/line inside cup at {xx},{yy}: {p:?}");
            }
        }
    }
}
#[test]
fn fight_recoil_keeps_support_fixed_and_lifts_the_other_foot() {
    let shot = load("fight");
    for action in &shot.actions {
        let Action::Strike(s) = action else { continue };
        if s.miss {
            continue;
        }
        let actor = &shot.actors[s.target];
        let rig = &actor.model.rig;
        let duration = (0.32 + 0.2 * s.force * if s.blocked.is_some() { 0.35 } else { 1.0 } / actor.scale).clamp(0.32, 0.55);
        let mut previous = None;
        let mut lifted = false;
        for i in 1..100 {
            let t = s.at + s.hold + duration * i as f64 / 100.0;
            if s.force > 0.55 {
                assert!(actor.planted_foot(t).is_none());
                let fig = actor.figure(t);
                for side in [mavian_rig::Side::L, mavian_rig::Side::R] {
                    let foot = rig.foot(side).unwrap();
                    assert!(fig.joints[foot.heel].y > 0.0 && fig.joints[foot.toe].y > 0.0, "knockback must fly above the floor");
                }
                lifted = true;
                continue;
            }
            let side = actor.planted_foot(t).expect("recoil support");
            let foot = rig.foot(side).unwrap();
            let fig = actor.figure(t);
            let p = fig.joints[foot.heel];
            let toe = fig.joints[foot.toe];
            let rest = actor.figure(s.at - 1e-5);
            for side in [mavian_rig::Side::L, mavian_rig::Side::R] {
                let f = rig.foot(side).unwrap();
                let knee = rig.parent(f.heel).unwrap();
                let hip = rig.parent(knee).unwrap();
                for (a, b) in [(hip, knee), (knee, f.heel)] {
                    assert!(
                        ((fig.joints[a] - fig.joints[b]).length() - (rest.joints[a] - rest.joints[b]).length()).abs() < 1e-5,
                        "leg length changed"
                    );
                }
            }
            if let Some((old_side, old_p, old_toe)) = previous
                && old_side == side
            {
                assert!((p - old_p).length() < 0.002, "strike {}: support slides at {t}: {:?}", s.at, p - old_p);
                assert!((toe - old_toe).length() < 0.002, "toe slides at {t}");
            }
            previous = Some((side, p, toe));
            for side in [mavian_rig::Side::L, mavian_rig::Side::R] {
                let f = rig.foot(side).unwrap();
                let p = fig.joints[f.heel];
                let q = fig.joints[f.toe];
                assert!(p.y >= -0.003 && q.y >= -0.003, "foot under floor at {t}: {p:?} {q:?}");
                lifted |= p.y.min(q.y) > 0.02;
            }
        }
        assert!(lifted, "strike {} has no recovery step", s.at);
        for t in [s.at + s.hold, s.at + s.hold + duration] {
            let a = actor.figure(t - 1e-5);
            let b = actor.figure(t + 1e-5);
            for (a, b) in a.joints.iter().zip(&b.joints) {
                assert!((*a - *b).length() < 0.003, "pose pops at {t}: {:?}", *b - *a);
            }
        }
    }
}

#[test]
fn fight_recompile_and_reverse_sampling_are_repeatable() {
    let mut shot = load("fight");
    let times = [1.51, 1.7, 2.8, 2.95, 3.6, 3.8, 4.1];
    let before: Vec<_> = times.iter().map(|&t| shot.actors.iter().map(|a| a.figure(t).joints.clone()).collect::<Vec<_>>()).collect();
    shot.compile().unwrap();
    for (i, &t) in times.iter().enumerate().rev() {
        let after: Vec<_> = shot.actors.iter().map(|a| a.figure(t).joints.clone()).collect();
        assert_eq!(before[i], after);
    }
}

#[test]
fn fight_drawn_strikes_make_contact_and_heads_stay_separate() {
    let shot = load("fight");
    for action in &shot.actions {
        let Action::Strike(s) = action else { continue };
        let a = &shot.actors[s.attacker];
        let d = &shot.actors[s.target];
        let ha = a.model.rig.joint("head").unwrap();
        let hd = d.model.rig.joint("head").unwrap();
        for i in 0..=100 {
            let w = s.at - s.windup + (s.windup + s.hold) * i as f64 / 100.0;
            let af = shot.posed(s.attacker, w).0;
            let df = shot.posed(s.target, w).0;
            let dist = (af.joints[ha] - df.joints[hd]).length();
            let head_axis = af.joints[ha] - af.joints[a.model.rig.parent(ha).unwrap()];
            let neck_axis = a.figure(w).joints[ha] - a.figure(w).joints[a.model.rig.parent(ha).unwrap()];
            assert!(head_axis.length() >= neck_axis.length() * 0.65, "neck collapsed");
            assert!(dist >= 0.439, "strike {} heads overlap at {w}: {dist}", s.at);
        }
        if s.miss {
            continue;
        }
        for i in 0..=12 {
            let w = s.at + s.hold * i as f64 / 12.0;
            let af = shot.posed(s.attacker, w).0;
            let df = shot.posed(s.target, w).0;
            let toward = mavian_scene::ik::flat_dir(df.joints[d.model.root], af.joints[a.model.root]);
            let goal = if let Some(b) = s.blocked {
                df.joints[d.model.rig.parent(b).unwrap()].lerp(df.joints[b], 0.55) + toward * 0.10
            } else {
                let j = d.model.rig.joint(if s.aim == "chin" { "head" } else { &s.aim }).unwrap();
                let offset = if s.aim == "chin" {
                    (Vec3 { x: 0.0, y: -0.17, z: 0.0 }
                        + mavian_scene::ik::flat_dir(df.joints[j], af.joints[mavian_scene::actions::chain(a, s.limb).unwrap()[0]]) * 0.09)
                        .norm()
                        * mavian_scene::ik::CHIN_REACH
                } else {
                    toward * if s.aim == "head" { 0.25 } else { 0.116 }
                };
                df.joints[j] + offset
            };
            let error = (af.joints[s.limb] - goal).length();
            assert!(error < 0.025, "strike {} drawn contact misses at {w}: {error}", s.at);
        }
    }
}

#[test]
fn contact_fixture_reports_reachable_unreachable_and_pinned_support() {
    use mavian_scene::ContactStatus;
    let s = load("contacts");
    for (w, name, status) in [
        (0.5, "reach-point", ContactStatus::Satisfied),
        (1.5, "touch-prop", ContactStatus::Satisfied),
        (2.5, "body-assisted", ContactStatus::Satisfied),
        (3.5, "unreachable", ContactStatus::Unreachable),
    ] {
        let reports = s.contact_reports(w);
        let r = &reports.iter().find(|(_, r)| r.name == name).unwrap().1;
        assert_eq!(r.status, status, "{r:?}");
        if status == ContactStatus::Unreachable {
            assert!(r.reach_error > 0.5);
        }
        let support = &reports.iter().find(|(_, r)| r.name == "left-foot").unwrap().1;
        assert!(support.error < 1e-9);
        assert_eq!(support.status, ContactStatus::Satisfied);
    }
    assert!(s.contact_reports(1.0).iter().all(|(_, r)| r.name != "reach-point"));
    let chest = s.actors[0].model.rig.joint("chest").unwrap();
    assert!((s.posed(0, 2.5).0.joints[chest] - s.actors[0].figure(2.5).joints[chest]).length() > 0.08);
    for frame in 480..720 {
        let w = frame as f64 / 240.0;
        assert!(s.obstacle_reports(w).is_empty(), "body-assisted fixture enters a prop at {w}");
        for (_, r) in s.contact_reports(w).into_iter().filter(|(_, r)| r.name == "body-assisted") {
            assert!(r.error < 0.002, "assisted approach loses contact at {w}: {r:?}");
        }
    }
    assert!(s.contact_reports(4.0).is_empty());
}

#[test]
fn approach_fixture_checks_opt_in_retreat_obstacles_limits_and_pose_lock() {
    use mavian_scene::{ContactStatus, StepStatus};
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shots/approach.py");
    for (name, mut shot) in mavian_py::load_file(&path).unwrap() {
        let expected = [StepStatus::Applied, StepStatus::Applied, StepStatus::Blocked, StepStatus::Limited];
        for (index, actor) in shot.actors.iter().enumerate() {
            match name.as_str() {
                "base" => {
                    assert_eq!(actor.step_reports[0].status, expected[index]);
                    assert!(actor.step_reports[0].shift.length() <= actor.step_reports[0].max_distance + 1e-10);
                    if index < 2 {
                        let errors = shot.contact_reports(2.0);
                        for (_, report) in errors.iter().filter(|(i, r)| *i == index && r.name.starts_with("push:")) {
                            assert_eq!(report.status, ContactStatus::Satisfied, "{report:?}");
                        }
                    }
                }
                "off" => assert!(actor.step_reports.is_empty()),
                "locked" => {
                    assert_eq!(actor.step_reports[0].status, StepStatus::PoseLocked);
                    assert_eq!(shot.posed(index, 2.).0.joints, actor.figure(0.).joints);
                }
                _ => panic!("unexpected approach fixture variant {name}"),
            }
            if name != "base" {
                assert_eq!(actor.figure(0.).joints, actor.figure(2.).joints);
            }
        }
        if name == "base" {
            assert!(shot.actors[0].step_reports[0].shift.z < -0.1);
            assert!(shot.actors[1].step_reports[0].shift.z > 0.1);
        }
        let before: Vec<_> = (0..shot.actors.len()).map(|i| (shot.actors[i].step_reports.clone(), shot.posed(i, 1.).0)).collect();
        shot.compile().unwrap();
        for (i, (reports, figure)) in before.into_iter().enumerate().rev() {
            assert_eq!(reports, shot.actors[i].step_reports);
            assert_eq!(figure, shot.posed(i, 1.).0);
        }
    }
}

#[test]
fn python_contact_permissions_distinguish_default_body_assistance_and_pose_lock() {
    use mavian_scene::ContactStatus;
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shots/contacts.py");
    let variants = mavian_py::load_file(&path).unwrap();
    for (name, shot) in variants {
        let actor = &shot.actors[0];
        let chest = actor.model.rig.joint("chest").unwrap();
        let raw = actor.figure(2.5);
        let actual = shot.posed(0, 2.5).0;
        let reports = shot.contact_reports(2.5);
        let contact = &reports.iter().find(|(_, r)| r.name == "body-assisted").unwrap().1;
        match name.as_str() {
            "base" => {
                assert!((actual.joints[chest] - raw.joints[chest]).length() > 0.08);
                assert_eq!(contact.status, ContactStatus::Satisfied);
            }
            "limbs" => {
                assert_eq!(actual.joints[chest], raw.joints[chest]);
                assert_eq!(contact.status, ContactStatus::Unreachable);
                assert!(contact.error > 0.09);
            }
            "locked" => {
                assert_eq!(actual.joints, raw.joints);
                assert_eq!(actual.rots, raw.rots);
                assert_eq!(contact.status, ContactStatus::Conflict);
                assert!(contact.error > 0.1);
            }
            _ => panic!("unexpected contact fixture variant {name}"),
        }
    }
}

#[test]
fn interact_puts_the_cup_down_without_an_unnecessary_torso_lean() {
    let s = load("interact");
    let actor = &s.actors[0];
    let chest = actor.model.rig.joint("chest").unwrap();
    let spine = actor.model.rig.joint("spine").unwrap();
    for w in [5.2, 5.4, 5.6, 5.8] {
        let authored = actor.figure(w);
        let posed = s.posed(0, w).0;
        assert!((authored.joints[chest].z - authored.joints[spine].z).abs() < 1e-9, "put-down still scripts a lean at {w}");
        assert_eq!(posed.joints[chest], authored.joints[chest], "nearby cup placement should not move the torso at {w}");
    }
}

#[test]
fn interact_uses_contacts_for_props_and_keeps_the_held_cup_attached() {
    use mavian_scene::ContactStatus;
    let mut s = load("interact");
    assert!(s.actors[0].reaches.is_empty(), "prop actions must use the shared contact solver");
    assert_eq!(s.actors[0].auto_contacts.len(), 4);
    for w in [2.9, 3.2, 4.0, 5.2, 5.7] {
        let reports = s.contact_reports(w);
        let held = &reports.iter().find(|(_, r)| r.name == "hold:cup").unwrap().1;
        assert!(held.error < 1e-8);
        assert_eq!(held.status, ContactStatus::Satisfied);
    }
    for w in [6.8, 7.3, 8.2, 9.0] {
        let reports = s.contact_reports(w);
        for (_, r) in reports.iter().filter(|(_, r)| r.name.starts_with("push:")) {
            assert_eq!(r.status, ContactStatus::Satisfied, "{r:?}");
            assert!(r.error < 1e-8);
        }
    }
    let before = s.contact_reports(7.0);
    s.compile().unwrap();
    assert_eq!(before, s.contact_reports(7.0));
}

#[test]
fn interact_arms_avoid_props_through_the_entire_transition() {
    let shot = load("interact");
    let actor = &shot.actors[0];
    let hands = [actor.model.rig.joint("hand_l").unwrap(), actor.model.rig.joint("hand_r").unwrap()];
    let chains = hands.map(|hand| mavian_scene::actions::chain(actor, hand).unwrap());
    let mut previous = None;
    let mut maximum = 0.0f64;
    for n in 0..=(shot.length * 240.0) as usize {
        let w = n as f64 / 240.0;
        let fig = shot.posed(0, w).0;
        assert!(shot.obstacle_reports(w).is_empty(), "penetration at {w}: {:?}", shot.obstacle_reports(w));
        let raw = actor.figure(w);
        for chain in chains {
            for [a, b] in [[chain[0], chain[1]], [chain[1], chain[2]]] {
                assert!(
                    ((fig.joints[b] - fig.joints[a]).length() - (raw.joints[b] - raw.joints[a]).length()).abs() < 1e-7,
                    "bone length changed at {w}"
                );
            }
        }
        if let Some(old) = previous {
            let old: std::sync::Arc<mavian_rig::Figure> = old;
            if n > 0 {
                let mid = w - 1.0 / 480.0;
                for (prop, p) in shot.props.iter().enumerate() {
                    let (pos, rot) = shot.prop_rest(prop, mid);
                    let obstacle = mavian_scene::obstacle::Obstacle { prop, shape: p.shape.clone(), pos, rot };
                    for chain in chains {
                        let radii = [0.062, 0.050, 0.052];
                        for k in 0..2 {
                            let start = old.joints[chain[k]].lerp(fig.joints[chain[k]], 0.5);
                            let end = old.joints[chain[k + 1]].lerp(fig.joints[chain[k + 1]], 0.5);
                            let gap = obstacle.capsule(start, radii[k], end, radii[k + 1]).0;
                            assert!(
                                gap >= -0.001,
                                "swept limb enters {} between samples at {mid}: {gap}; chain {chain:?}; old {:?}; new {:?}",
                                p.name,
                                chain.map(|j| old.joints[j]),
                                chain.map(|j| fig.joints[j])
                            );
                        }
                    }
                }
            }
            for chain in chains {
                let jump = (fig.joints[chain[1]] - old.joints[chain[1]]).length().max((fig.joints[chain[2]] - old.joints[chain[2]]).length());
                if jump > 0.03 {
                    eprintln!(
                        "jump {jump} at {w}: {} elbow {:?}->{:?} hand {:?}->{:?}",
                        actor.model.rig.joints()[chain[2]].name,
                        old.joints[chain[1]],
                        fig.joints[chain[1]],
                        old.joints[chain[2]],
                        fig.joints[chain[2]]
                    );
                }
                maximum = maximum.max(jump);
            }
        }
        previous = Some(fig);
    }
    eprintln!("maximum limb displacement per 1/240 second: {maximum:.6}m");
    assert!(maximum < 0.04, "the avoidance path jumps between samples: {maximum}");
}

#[test]
fn prop_avoidance_is_repeatable_after_recompile_and_reverse_sampling() {
    let mut shot = load("interact");
    let times = [2.35, 2.7, 3.075, 5.8, 5.95, 6.433333333, 9.9, 10.05];
    let before: Vec<_> = times.iter().map(|&w| shot.posed(0, w).0).collect();
    shot.compile().unwrap();
    for (i, &w) in times.iter().enumerate().rev() {
        assert_eq!(before[i], shot.posed(0, w).0);
    }
}

#[test]
fn interact_put_down_uses_the_clear_direct_approach() {
    let shot = load("interact");
    let actor = &shot.actors[0];
    let contact = actor.auto_contacts.iter().find(|c| c.name == "release:cup").unwrap();
    eprintln!("selected release route: {contact:?}");
    assert!(contact.approach.is_none(), "a clear put-down approach must not add a retreat waypoint");
    for w in [5.21, 5.25, 5.30, 5.35, 5.40] {
        let actual = shot.contact_reports(w).into_iter().find(|(_, r)| r.name == contact.name).unwrap().1;
        let start = actor.figure(w).joints[contact.chain[2]];
        let target = contact.target.at(w, &shot.world());
        let delta = target - start;
        let perpendicular = (actual.goal - start).cross(delta).length() / delta.length().max(1e-9);
        assert!(perpendicular < 1e-9, "put-down takes an unnecessary detour at {w}: {perpendicular}");
    }
}

#[test]
fn interact_push_transitions_keep_elbows_on_the_authored_side() {
    let shot = load("interact");
    let actor = &shot.actors[0];
    let mut worst_forward = f64::NEG_INFINITY;
    let mut maximum_bend_change = 0.0f64;
    let (start, end, reach, settle) = shot
        .actions
        .iter()
        .find_map(|action| if let Action::Push { start, end, reach, settle, .. } = action { Some((*start, *end, *reach, *settle)) } else { None })
        .unwrap();
    for (start, end) in [(start - reach, start), (end, end + settle)] {
        let samples = ((end - start) * 480.0).ceil() as usize;
        let mut previous = [None; 2];
        for sample in 0..=samples {
            let w = start + (end - start) * sample as f64 / samples as f64;
            let figure = shot.posed(0, w).0;
            let heading = mavian_core::Mat3::rot_y(actor.facing.at(w));
            for (side, hand) in ["hand_l", "hand_r"].into_iter().enumerate() {
                let chain = mavian_scene::actions::chain(actor, actor.model.rig.joint(hand).unwrap()).unwrap();
                let [a, b, c] = chain;
                let direction = (figure.joints[c] - figure.joints[a]).norm();
                let upper = figure.joints[b] - figure.joints[a];
                let bend = (upper - direction * upper.dot(direction)).norm();
                let local = heading.transpose() * bend;
                worst_forward = worst_forward.max(local.z);
                assert!(local.z < -0.25, "{hand} twists forward at {w}: {local:?}");
                if let Some(old) = previous[side] {
                    maximum_bend_change = maximum_bend_change.max((bend - old).length());
                }
                previous[side] = Some(bend);
            }
        }
    }
    eprintln!("push transitions: maximum forward bend {worst_forward:.6}, maximum bend change at 480 Hz {maximum_bend_change:.6}");
    assert!(maximum_bend_change < 0.05, "elbow bend changes discontinuously: {maximum_bend_change}");
}

#[test]
fn original_interact_acquires_the_stationary_cup_before_carrying_it() {
    use mavian_scene::{GrabStatus, Motion};
    let mut s = load("interact");
    assert_eq!(s.grab_reports.len(), 1);
    let r = s.grab_reports[0].clone();
    assert_eq!(r.status, GrabStatus::Attached);
    assert!(r.error < 1e-8);
    assert!((s.posed(r.actor, r.at).0.joints[r.hand] - r.target).length() < 1e-8);
    let before = s.prop_rest(r.prop, r.at - 1e-6);
    let at = s.prop_rest(r.prop, r.at);
    let after = s.prop_rest(r.prop, r.at + 1e-6);
    assert!((before.0 - at.0).length() < 1e-8);
    assert!((before.0 - after.0).length() < 1e-4);
    assert_eq!(before.1, at.1);
    assert!(matches!(s.props[r.prop].segment(r.at), Some(Motion::Held { .. })));
    let saved = s.grab_reports.clone();
    s.compile().unwrap();
    assert_eq!(s.grab_reports, saved);
}

#[test]
fn grasp_fixture_attaches_only_the_clear_reachable_unlocked_attempt() {
    use mavian_scene::{GrabStatus, Motion};
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shots/grasp.py");
    for (name, mut s) in mavian_py::load_file(&path).unwrap() {
        assert_eq!(s.grab_reports.len(), 1);
        let r = &s.grab_reports[0];
        let expected = match name.as_str() {
            "base" => GrabStatus::Attached,
            "far" => GrabStatus::Unreachable,
            "locked" => GrabStatus::PoseLocked,
            "blocked" => GrabStatus::Obstructed,
            _ => panic!("unexpected variant {name}"),
        };
        assert_eq!(r.status, expected, "{name}: {r:?}");
        let prop = r.prop;
        assert_eq!(matches!(s.props[prop].segment(1.6), Some(Motion::Held { .. })), name == "base");
        if name != "base" {
            assert!(s.props[prop].motion.is_empty());
            for w in [1.0, 1.6, 2.4, 2.9] { assert_eq!(s.prop_rest(prop, w).0, s.props[prop].at); }
        }
        let saved = s.grab_reports.clone();
        let frames: Vec<_> = (0..72).map(|n| (s.posed(0, n as f64 / 24.0).0, s.prop_rest(prop, n as f64 / 24.0))).collect();
        s.compile().unwrap();
        assert_eq!(s.grab_reports, saved);
        for (n, (fig, pose)) in frames.into_iter().enumerate().rev() {
            assert_eq!(fig, s.posed(0, n as f64 / 24.0).0);
            assert_eq!(pose, s.prop_rest(prop, n as f64 / 24.0));
        }
    }
}

#[test]
fn grasp_carry_preserves_the_clear_authored_elbow_and_has_no_holding_branch_flip() {
    let mut s = load("grasp");
    let actor = &s.actors[0];
    let hand = actor.model.rig.joint("hand_r").unwrap();
    let chain = mavian_scene::actions::chain(actor, hand).unwrap();
    let mut worst = 0.0f64;
    for n in 348..=432 {
        let w = n as f64 / 240.0;
        let raw = actor.figure(w);
        let actual = s.posed(0, w).0;
        let drift = (actual.joints[chain[1]] - raw.joints[chain[1]]).length();
        worst = worst.max(drift);
        assert!(drift < 1e-8, "clear carrying elbow was overridden at {w}: authored {:?}, actual {:?}, drift {drift}",
            raw.joints[chain[1]], actual.joints[chain[1]]);
        assert!((actual.joints[hand] - raw.joints[hand]).length() < 1e-8);
    }
    eprintln!("grasp clear-carry maximum authored elbow displacement: {worst:.9} m");
    let samples: Vec<_> = (0..=720).map(|n| s.posed(0, n as f64 / 240.0).0).collect();
    let mut largest = 0.0f64;
    // Acquisition, lifting and carrying while the prop is bound to this hand.
    for n in 240..576 {
        let pair = &samples[n..=n + 1];
        for j in chain {
            largest = largest.max((pair[1].joints[j] - pair[0].joints[j]).length());
        }
    }
    eprintln!("grasp maximum holding arm displacement per 1/240 s: {largest:.9} m");
    assert!(largest < 0.04, "holding arm switches clearance branches: {largest}");
    // Post-release avoidance is measured separately; it has no holding-side bias.
    let mut after_release = (0.0f64, 0.0f64);
    for n in 576..720 {
        let delta = (samples[n + 1].joints[chain[1]] - samples[n].joints[chain[1]]).length();
        if delta > after_release.0 { after_release = (delta, (n + 1) as f64 / 240.0); }
    }
    eprintln!("grasp post-release maximum elbow displacement: {:.9} m at {:.9} s (separate remaining limit)", after_release.0, after_release.1);
    s.compile().unwrap();
    for n in (0..=720).rev() {
        assert_eq!(samples[n], s.posed(0, n as f64 / 240.0).0);
    }
}
