//! Base model: the stick figure (a port of `engine/models/stickman.py`).

use mavian_rig::{Joint, Model, ModelError, Part, Rig, Side};
use std::sync::{Arc, OnceLock};

/// head radius
pub const HEAD_R: f64 = 0.20;

fn rig() -> Rig {
    let mut joints = vec![
        Joint::root("pelvis"),
        Joint::new("spine", "pelvis", (0.0, 0.0, 0.0)),
        Joint::new("chest", "spine", (0.0, 0.48, 0.0)),
        Joint::new("neck", "chest", (0.0, 0.0, 0.0)),
        Joint::new("head", "neck", (0.0, 0.27, 0.0)),
    ];
    for (t, s) in [("l", 1.0), ("r", -1.0)] {
        let j = |name: &str, parent: &str, off: (f64, f64, f64), pitch: f64| {
            Joint::new(&format!("{name}_{t}"), parent, off).pitch(pitch).side(s)
        };
        joints.extend([
            j("arm", "chest", (0.16 * s, -0.02, 0.0), -1.0),
            j("elbow", &format!("arm_{t}"), (0.0, -0.27, 0.0), -1.0),
            j("hand", &format!("elbow_{t}"), (0.0, -0.26, 0.0), -1.0),
            j("leg", "pelvis", (0.085 * s, 0.0, 0.0), -1.0),
            j("knee", &format!("leg_{t}"), (0.0, -0.42, 0.0), 1.0),
            j("foot", &format!("knee_{t}"), (0.0, -0.40, 0.0), 1.0),
            j("toe", &format!("foot_{t}"), (0.0, 0.0, 0.15), 1.0),
        ]);
    }
    Rig::new(joints, &[(Side::L, "foot_l", "toe_l", 0.15), (Side::R, "foot_r", "toe_r", 0.15)])
        .expect("stick figure skeleton")
}

fn parts(rig: &Rig) -> Result<Vec<Part>, ModelError> {
    // name, from joint, to joint, width at start and end (fractions of line width)
    let mut parts = vec![
        Part::stick(rig, "spine", "pelvis", "chest", 1.15, 1.10)?,
        // the neck ends at the head's surface, not its centre
        Part::stick(rig, "neck", "chest", "head", 0.95, 0.90)?.trim(HEAD_R * 0.9),
    ];
    for t in ["l", "r"] {
        let n = |s: &str| format!("{s}_{t}");
        parts.extend([
            Part::stick(rig, &n("clav"), "chest", &n("arm"), 1.05, 1.00)?,
            Part::stick(rig, &n("uarm"), &n("arm"), &n("elbow"), 1.00, 0.90)?,
            // the hand is a dot slightly bigger than the line
            Part::stick(rig, &n("farm"), &n("elbow"), &n("hand"), 0.90, 0.75)?.free().dot(),
            Part::stick(rig, &n("hipb"), "pelvis", &n("leg"), 1.10, 1.10)?,
            Part::stick(rig, &n("thigh"), &n("leg"), &n("knee"), 1.12, 0.98)?,
            Part::stick(rig, &n("shin"), &n("knee"), &n("foot"), 0.98, 0.84)?,
            Part::stick(rig, &n("foot"), &n("foot"), &n("toe"), 0.84, 0.72)?.free(),
        ]);
    }
    parts.push(Part::head(rig, "head", HEAD_R)?);
    Ok(parts)
}

fn build() -> Result<Model, ModelError> {
    let rig = rig();
    let parts = parts(&rig)?;
    let mut m = Model::new("stickman", rig, parts);
    let rig = Arc::clone(&m.rig);
    let p = || rig.pose();

    let stand = p().set("arm", (0, 9)).set("elbow", 10).set("leg", (0, 4)).build()?;
    m.add_pose("stand", stand.clone())?;
    // wind-up before a jump: crouch, torso forward, arms back
    m.add_pose("crouch", p().set("spine", 40).set("neck", -20).set("leg", (70, 7)).set("knee", 110)
        .set("foot", -40).set("arm", (-50, 12)).set("elbow", 25).build()?)?;
    // take-off: body extended, toes still on the ground, arms go forward and up
    m.add_pose("launch", p().set("spine", -6).set("neck", -6).set("leg", (-8, 3)).set("knee", 4)
        .set("foot", 45).set("arm", (150, -15)).set("elbow", 12).build()?)?;
    // top point: knees tucked, arms up
    m.add_pose("jump", p().lift(0.8).set("spine", 12).set("neck", -6).set("leg", (62, 10)).set("knee", 105)
        .set("foot", 25).set("arm", (170, -30)).set("elbow", 15).build()?)?;
    // before touchdown: legs reach for the ground, arms out to the sides
    m.add_pose("fall", p().set("spine", 8).set("leg", (22, 6)).set("knee", 22).set("foot", -8)
        .set("arm", (175, -70)).set("elbow", 20).build()?)?;
    // landing: deep crouch
    m.add_pose("land", p().set("spine", 46).set("neck", -26).set("leg", (85, 12)).set("knee", 135)
        .set("foot", -50).set("arm", (35, 38)).set("elbow", 35).build()?)?;

    // wave: like stand, but the right arm is raised and the elbow bends
    m.add_pose("wave_a", p().base(&stand).set("arm_r", (0, 105, 90)).set("elbow_r", 45).build()?)?;
    m.add_pose("wave_b", p().base(&stand).set("arm_r", (0, 105, 90)).set("elbow_r", 95).build()?)?;

    // --- walking: _l means "left leg forward" or "weight on the left" ---
    // contact: the left heel touches the ground, the right foot behind on its toe
    let contact_l = p().set("spine", 5).set("chest", (0, 0, 7)).set("head", (0, 0, -7))
        .set("leg_l", (21, 2)).set("knee_l", 4).set("foot_l", 4)
        .set("leg_r", (-18, 2)).set("knee_r", 22).set("foot_r", -14)
        .set("arm_l", (-20, 6)).set("elbow_l", 12).set("arm_r", (24, 6)).set("elbow_r", 24).build()?;
    // passing: the left leg supports, the right is bent and swings forward
    let pass_l = p().set("spine", 4)
        .set("leg_l", (0, 2)).set("knee_l", 3).set("foot_l", -3)
        .set("leg_r", (24, 2)).set("knee_r", 62).set("foot_r", -25)
        .set("arm_l", (3, 6)).set("elbow_l", 14).set("arm_r", (3, 6)).set("elbow_r", 14).build()?;
    m.add_pose("contact_r", rig.mirror(&contact_l))?;
    m.add_pose("pass_r", rig.mirror(&pass_l))?;
    m.add_pose("contact_l", contact_l)?;
    m.add_pose("pass_l", pass_l)?;

    // --- peering into the camera ---
    let peer = p().set("spine", 20).set("neck", -12).set("arm", (-16, 8)).set("elbow", 22).set("leg", (0, 4)).build()?;
    m.add_pose("peer_l", p().base(&peer).set("head", (0, 8, 24)).build()?)?;   // head to the left
    m.add_pose("peer_r", p().base(&peer).set("head", (0, -8, -24)).build()?)?; // head to the right
    m.add_pose("peer", peer)?;

    // --- long jump ---
    // take-off: the whole body leans forward, toes still on the ground
    m.add_pose("leap_launch", p().set("pelvis", 22).set("neck", -14).set("leg", (-12, 3)).set("knee", 6)
        .set("foot", 40).set("arm", (140, -12)).set("elbow", 15).build()?)?;
    // flight: knees tucked
    m.add_pose("leap_air", p().lift(0.95).set("pelvis", 14).set("spine", 14).set("neck", -10).set("leg", (72, 10))
        .set("knee", 110).set("foot", 20).set("arm", (120, -25)).set("elbow", 15).build()?)?;
    // before touchdown: legs reach forward
    m.add_pose("leap_reach", p().set("pelvis", -6).set("spine", 26).set("neck", -8).set("leg", (56, 6))
        .set("knee", 24).set("foot", -14).set("arm", (62, 20)).set("elbow", 20).build()?)?;
    // looking back over the left shoulder
    m.add_pose("look_back", p().base(&stand).set("chest", (0, 0, 30)).set("head", (0, 6, 72)).build()?)?;

    m.rest = "stand".into();
    m.root = rig.joint("pelvis")?;
    m.trails = ["hand_l", "hand_r", "toe_l", "toe_r"].iter().map(|n| rig.joint(n)).collect::<Result<_, _>>()?;
    m.shadow = 0.27;
    Ok(m)
}

/// Stick figure: the model is built once and shared between shots.
pub fn stickman() -> Arc<Model> {
    static M: OnceLock<Arc<Model>> = OnceLock::new();
    Arc::clone(M.get_or_init(|| Arc::new(build().expect("stick figure model"))))
}

#[cfg(test)]
mod tests {
    #[test]
    fn builds_with_all_poses() {
        let m = super::stickman();
        for p in ["stand", "contact_l", "contact_r", "pass_l", "pass_r", "peer_l", "leap_air", "look_back"] {
            assert!(m.has_pose(p), "{p}");
        }
    }
}
