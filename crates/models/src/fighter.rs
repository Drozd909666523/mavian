//! The fighter: a silhouette stick figure with poses for a fight (a port of
//! the MVP's `characters/fighter.py`).
//!
//! As in old flash fights: a one-colour silhouette — a filled head without a
//! face, thick limbs of one width, no halos (overlaps merge into one
//! silhouette) and no feet. The colour comes from the actor
//! (`Actor::color`). The skeleton is the stick figure's, the poses its own.

use crate::stickman::stickman;
use mavian_rig::{Model, ModelError, Part, Pose, PoseBuilder};
use std::sync::{Arc, OnceLock};

/// limb width, fraction of the style's line
const THICK: f64 = 2.9;
const HEAD_R: f64 = 0.19;

fn build() -> Result<Model, ModelError> {
    let base = stickman();
    let rig = (*base.rig).clone();
    let bones = [("spine", "pelvis", "chest"), ("neck", "chest", "head")];
    let names: Vec<[String; 3]> = ["l", "r"]
        .iter()
        .flat_map(|t| {
            [
                ["clav", "chest", "arm"],
                ["uarm", "arm", "elbow"],
                ["farm", "elbow", "hand"],
                ["hipb", "pelvis", "leg"],
                ["thigh", "leg", "knee"],
                ["shin", "knee", "foot"],
            ]
            .map(|[n, a, b]| {
                let side = |j: &str| if j == "chest" || j == "pelvis" { j.to_string() } else { format!("{j}_{t}") };
                [format!("{n}_{t}"), side(a), side(b)]
            })
        })
        .collect();
    let mut parts = Vec::new();
    // width barely depends on depth (gamma = 1 — as in life), no halos
    for (n, a, b) in bones.iter().map(|(n, a, b)| (n.to_string(), a.to_string(), b.to_string())).chain(names.into_iter().map(|[n, a, b]| (n, a, b))) {
        let mut p = Part::stick(&rig, &n, &a, &b, THICK, THICK)?.no_halo().gamma(1.0);
        if n.starts_with("farm") || n.starts_with("shin") {
            p = p.free();
        }
        parts.push(p);
    }
    parts.push(Part::head(&rig, "head", HEAD_R)?.filled());
    let mut m = Model::new("fighter", rig, parts);
    for (name, pose) in base.poses() {
        m.add_pose(name, pose.clone())?;
    }
    let rig = Arc::clone(&m.rig);
    let stand = base.pose("stand")?.clone();
    let p = || rig.pose().base(&stand);
    // guard arms: the front (left) slightly forward, the rear (right) at the chin
    fn guard(b: PoseBuilder<'_>) -> PoseBuilder<'_> {
        b.set("arm_l", (55, 12)).set("elbow_l", 105).set("arm_r", (36, 14)).set("elbow_r", 126)
    }
    // in a step the arms shift a little: the front one forward-down, the rear tighter — the guard lives
    fn step(b: PoseBuilder<'_>) -> PoseBuilder<'_> {
        b.set("arm_l", (62, 10)).set("elbow_l", 96).set("arm_r", (40, 16)).set("elbow_r", 130)
    }
    let poses: Vec<(&str, Pose)> = vec![
        // an orthodox stance: left foot forward, knees bent, fists at the face
        ("guard", guard(p().set("spine", 8).set("neck", -8).set("leg_l", (16, 7)).set("knee_l", 14).set("foot_l", -6)
            .set("leg_r", (-14, 8)).set("knee_r", 18).set("foot_r", 2)).build()?),
        // a shuffle step: the front foot out, the stance wider and lower
        ("guard_step", step(p().set("spine", 11).set("neck", -10).set("leg_l", (30, 7)).set("knee_l", 20).set("foot_l", -10)
            .set("leg_r", (-26, 8)).set("knee_r", 12).set("foot_r", 8)).build()?),
        // a slip: down and aside, a wide crouch, the head leaves the line of the punch
        ("slip", guard(p().set("spine", (14, 20, 0)).set("neck", (-6, 12, 0))
            .set("leg_l", (52, 22)).set("knee_l", 98).set("foot_l", -38).set("leg_r", (28, 24)).set("knee_r", 82).set("foot_r", -28)).build()?),
        // a lunge out of the slip: a step forward, the rear leg straightens —
        // the body goes up and forward, the head stays aside (as in the slip),
        // past the opponent's front arm
        ("lunge", guard(p().set("spine", (12, -14, 0)).set("neck", (-10, -10, 0))
            .set("leg_l", (38, 8)).set("knee_l", 34).set("foot_l", -14)
            .set("leg_r", (-30, 8)).set("knee_r", 6).set("foot_r", 14)).build()?),
        // a kick's wind-up: the right knee up (IK brings the foot to the target)
        ("kick_r", guard(p().set("spine", -8).set("neck", 6).set("leg_l", (4, 9)).set("knee_l", 12).set("foot_l", -4)
            .set("leg_r", (70, 22)).set("knee_r", 70)).build()?),
        // knocked down: sits on the ground, leaning on the hands
        ("sit", p().set("leg", (86, 9)).set("knee", 6).set("foot", 12).set("spine", -28).set("neck", 14)
            .set("arm", (-38, 24)).set("elbow", 8).build()?),
        // victory: arms up
        ("victory", p().set("arm", (170, -28)).set("elbow", 14).set("leg", (0, 6)).build()?),
    ];
    for (n, pose) in poses {
        m.add_pose(n, pose)?;
    }
    m.rest = base.rest.clone();
    m.root = base.root;
    m.trails = base.trails.clone();
    m.shadow = base.shadow;
    Ok(m)
}

/// The fighter: built once and shared between shots.
pub fn fighter() -> Arc<Model> {
    static M: OnceLock<Arc<Model>> = OnceLock::new();
    Arc::clone(M.get_or_init(|| Arc::new(build().expect("fighter model"))))
}
