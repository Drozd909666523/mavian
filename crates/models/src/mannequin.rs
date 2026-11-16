//! Mannequin: the stick figure's skeleton and poses, but a volumetric body
//! (a port of `characters/mannequin.py`).

use crate::stickman::stickman;
use mavian_rig::{Model, ModelError, Part};
use std::sync::{Arc, OnceLock};

const HEAD_R: f64 = 0.19;
/// head outline — the same as volumes
const LINE: f64 = 0.6;
const TORSO: &str = "torso";

fn build() -> Result<Model, ModelError> {
    let base = stickman();
    // The body is wider than a stick: the stick figure's shoulder is 0.16 from
    // the centre, and the mannequin's arm would go through the torso.
    // Proportions are adjusted for the body, the poses stay the same.
    let rig = base.rig.with_offsets(&[
        ("arm", (0.205, -0.05, 0.0)), // shoulders wider than the torso
        ("leg", (0.095, 0.0, 0.0)),   // legs slightly apart
        ("head", (0.0, 0.34, 0.0)),   // head higher: the stick figure barely has a neck
    ])?;
    let mut parts = vec![
        // the bottom cap ends just below the hips: with the MVP's cap (the half
        // width, 0.15) the torso hung 15 cm below the hip joints, the legs
        // came out of its sides, and in a crouch it showed as a belly between
        // the thighs
        Part::body(&rig, TORSO, "spine", "chest", (0.15, 0.185), (0.10, 0.11), (0.0, 0.11))?.cap((0.06, 0.185)),
        Part::limb(&rig, "neck", &["chest", "head"], &[0.055, 0.052], Some(TORSO))?,
    ];
    for t in ["l", "r"] {
        let n = |s: &str| format!("{s}_{t}");
        parts.extend([
            // the arm grows from the torso: no seam at the shoulder
            Part::limb(&rig, &n("arm"), &[&n("arm"), &n("elbow"), &n("hand")], &[0.062, 0.050, 0.040], Some(TORSO))?,
            Part::ball(&rig, &n("hand"), 0.052)?,
            // the leg with the shoe is one chain: no seam at the ankle
            Part::limb(&rig, &n("leg"), &[&n("leg"), &n("knee"), &n("foot"), &n("toe")], &[0.085, 0.064, 0.050, 0.042], Some(TORSO))?,
        ]);
    }
    parts.push(Part::head(&rig, "head", HEAD_R)?.line(LINE));
    let mut m = Model::new("mannequin", rig, parts);
    for (name, pose) in base.poses() {
        m.add_pose(name, pose.clone())?;
    }
    m.rest = base.rest.clone();
    m.root = base.root;
    m.trails = base.trails.clone();
    m.shadow = 0.30;
    // Overlap rules on top of geometry. From below, the neck and torso are
    // honestly nearer than the chin and would lie over the face; a forearm
    // coming at the camera lies over the hand. That is not how it is drawn.
    m.over = [("head", "neck"), ("head", TORSO), ("hand_l", "arm_l"), ("hand_r", "arm_r")]
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
    Ok(m)
}

/// The mannequin: built once and shared between shots.
pub fn mannequin() -> Arc<Model> {
    static M: OnceLock<Arc<Model>> = OnceLock::new();
    Arc::clone(M.get_or_init(|| Arc::new(build().expect("mannequin model"))))
}

#[cfg(test)]
mod tests {
    #[test]
    fn builds_with_stick_figure_poses() {
        let m = super::mannequin();
        assert!(m.has_pose("leap_air") && m.part("torso").is_some());
    }
}
