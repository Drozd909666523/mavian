//! Joints, poses, forward kinematics.
//!
//! A pose is angles (pitch, roll, twist) in degrees for each joint:
//!
//! - pitch — plus: the bone's end goes forward (each bone has its own "forward",
//!   set by `pitch_sign`: for a knee plus means bending);
//! - roll — plus: a limb moves away from the body, central bones
//!   lean left;
//! - twist — rotation about the bone's own axis.
//!
//! Paired joints end in `_l` and `_r`. The same numbers on the left and
//! right give a mirrored pose, and a pose may set `arm` for both at once.

use mavian_core::{Mat3, Vec3, mix, v3};
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RigError(pub String);

impl fmt::Display for RigError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RigError {}

/// Body side.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    L,
    R,
}

impl Side {
    pub fn other(self) -> Side {
        match self {
            Side::L => Side::R,
            Side::R => Side::L,
        }
    }

    pub fn suffix(self) -> &'static str {
        match self {
            Side::L => "l",
            Side::R => "r",
        }
    }
}

/// A joint in a skeleton description: the parent is referenced by name.
#[derive(Clone, Debug)]
pub struct Joint {
    pub name: String,
    pub parent: Option<String>,
    /// where the joint is in the parent's frame
    pub offset: Vec3,
    /// which way is "forward" for this bone
    pub pitch_sign: f64,
    /// +1 left, −1 right, 0 centre
    pub side: f64,
}

impl Joint {
    pub fn root(name: &str) -> Joint {
        Joint { name: name.into(), parent: None, offset: Vec3::ZERO, pitch_sign: 1.0, side: 0.0 }
    }

    pub fn new(name: &str, parent: &str, offset: (f64, f64, f64)) -> Joint {
        Joint { name: name.into(), parent: Some(parent.into()), offset: offset.into(), pitch_sign: 1.0, side: 0.0 }
    }

    pub fn pitch(mut self, sign: f64) -> Joint {
        self.pitch_sign = sign;
        self
    }

    pub fn side(mut self, side: f64) -> Joint {
        self.side = side;
        self
    }
}

/// A foot to stand on: heel (the joint) and toe.
#[derive(Clone, Copy, Debug)]
pub struct Foot {
    pub heel: usize,
    pub toe: usize,
    /// foot length — from heel to toe along the ground
    pub length: f64,
}

/// Skeleton: joints (parents before children) and what it stands on.
#[derive(Clone, Debug)]
pub struct Rig {
    joints: Vec<Joint>,
    parents: Vec<Option<usize>>,
    index: HashMap<String, usize>,
    feet: Vec<(Side, Foot)>,
    /// joints of which the lowest stands on the ground (at height lift)
    ground: Vec<usize>,
}

impl Rig {
    /// `feet` — (side, heel, toe, foot length).
    pub fn new(joints: Vec<Joint>, feet: &[(Side, &str, &str, f64)]) -> Result<Rig, RigError> {
        let mut index = HashMap::new();
        let mut parents = Vec::new();
        for (i, j) in joints.iter().enumerate() {
            let parent = match &j.parent {
                None => None,
                Some(p) => Some(*index.get(p).ok_or_else(|| {
                    RigError(format!("joint {}: parent {p} must come earlier", j.name))
                })?),
            };
            if index.insert(j.name.clone(), i).is_some() {
                return Err(RigError(format!("joint {} appears twice", j.name)));
            }
            parents.push(parent);
        }
        let mut rig = Rig { joints, parents, index, feet: Vec::new(), ground: Vec::new() };
        for &(side, heel, toe, length) in feet {
            let foot = Foot { heel: rig.joint(heel)?, toe: rig.joint(toe)?, length };
            rig.feet.push((side, foot));
            rig.ground.extend([foot.heel, foot.toe]);
        }
        Ok(rig)
    }

    pub fn joints(&self) -> &[Joint] {
        &self.joints
    }

    pub fn len(&self) -> usize {
        self.joints.len()
    }

    /// The parent joint, if any.
    pub fn parent(&self, j: usize) -> Option<usize> {
        self.parents[j]
    }

    pub fn is_empty(&self) -> bool {
        self.joints.is_empty()
    }

    /// Joint index by name.
    pub fn joint(&self, name: &str) -> Result<usize, RigError> {
        self.index.get(name).copied().ok_or_else(|| RigError(format!("no such joint: {name}")))
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn foot(&self, side: Side) -> Option<Foot> {
        self.feet.iter().find(|(s, _)| *s == side).map(|(_, f)| *f)
    }

    /// The same skeleton with other proportions: new joint offsets.
    ///
    /// `("arm", (0.21, -0.05, 0.0))` sets both sides (x is mirrored for the
    /// right one), `("arm_l", ...)` one. Poses stay valid: they are angles.
    pub fn with_offsets(&self, offsets: &[(&str, (f64, f64, f64))]) -> Result<Rig, RigError> {
        let mut rig = self.clone();
        for &(name, (x, y, z)) in offsets {
            if let Some(&i) = self.index.get(name) {
                rig.joints[i].offset = Vec3 { x, y, z };
            } else if self.has(&format!("{name}_l")) {
                rig.joints[self.joint(&format!("{name}_l"))?].offset = Vec3 { x, y, z };
                rig.joints[self.joint(&format!("{name}_r"))?].offset = Vec3 { x: -x, y, z };
            } else {
                return Err(RigError(format!("no such joint: {name}")));
            }
        }
        Ok(rig)
    }

    // --- poses ---------------------------------------------------------------

    /// Build a pose: `rig.pose().set("arm", (0, 9)).set("elbow", 10).build()`.
    pub fn pose(&self) -> PoseBuilder<'_> {
        PoseBuilder { rig: self, base: None, lift: None, both: Vec::new(), single: Vec::new() }
    }

    /// Mirrored pose: left and right swap.
    pub fn mirror(&self, p: &Pose) -> Pose {
        let mut angles = vec![Vec3::ZERO; self.len()];
        for (i, j) in self.joints.iter().enumerate() {
            let a = p.angles[i];
            let swap = |from: &str, to: &str| {
                j.name.strip_suffix(from).and_then(|stem| self.index.get(&format!("{stem}{to}")))
            };
            if let Some(&k) = swap("_l", "_r").or_else(|| swap("_r", "_l")) {
                angles[k] = a;
            } else {
                angles[i] = v3(a.x, -a.y, -a.z);
            }
        }
        Pose { lift: p.lift, angles }
    }

    // --- kinematics ----------------------------------------------------------

    /// Forward kinematics with the root at the origin → (points, rotations).
    pub fn fk(&self, p: &Pose, facing: f64, scale: f64) -> (Vec<Vec3>, Vec<Mat3>) {
        let mut pts = Vec::with_capacity(self.len());
        let mut rots: Vec<Mat3> = Vec::with_capacity(self.len());
        for (i, j) in self.joints.iter().enumerate() {
            let local = joint_rot(j, p.angles[i]);
            match self.parents[i] {
                None => {
                    pts.push(Vec3::ZERO);
                    rots.push(Mat3::rot_y(facing) * local);
                }
                Some(pi) => {
                    let pr = rots[pi];
                    pts.push(pts[pi] + pr * (j.offset * scale));
                    rots.push(pr * local);
                }
            }
        }
        (pts, rots)
    }

    /// Put the figure into the world: the root's height is chosen so the lowest
    /// support point is at height lift — crouching lowers the pelvis by itself and
    /// the feet do not sink.
    pub fn place(&self, pts: Vec<Vec3>, rots: Vec<Mat3>, xz: (f64, f64), lift: f64) -> Figure {
        let low = if self.ground.is_empty() {
            0.0
        } else {
            self.ground.iter().map(|&n| pts[n].y).fold(f64::INFINITY, f64::min)
        };
        let root = v3(xz.0, lift - low, xz.1);
        let joints = pts.into_iter().map(|q| q + root).collect();
        Figure { joints, rots, pivot: v3(root.x, lift, root.z), lift }
    }
}

fn joint_rot(j: &Joint, a: Vec3) -> Mat3 {
    let (pitch, roll, twist) = (a.x, a.y, a.z);
    let mirror = if j.side != 0.0 { j.side } else { 1.0 };
    let roll_sign = if j.side != 0.0 { j.side } else { -1.0 };
    Mat3::rot_z(roll_sign * roll) * (Mat3::rot_x(j.pitch_sign * pitch) * Mat3::rot_y(mirror * twist))
}

/// Joint angles: a number (pitch), a pair or a triple.
pub trait Angles {
    fn angles(self) -> Vec3;
}

impl Angles for Vec3 {
    fn angles(self) -> Vec3 {
        self
    }
}

impl Angles for f64 {
    fn angles(self) -> Vec3 {
        v3(self, 0.0, 0.0)
    }
}

impl Angles for i32 {
    fn angles(self) -> Vec3 {
        v3(self as f64, 0.0, 0.0)
    }
}

impl Angles for (i32, i32) {
    fn angles(self) -> Vec3 {
        v3(self.0 as f64, self.1 as f64, 0.0)
    }
}

impl Angles for (i32, i32, i32) {
    fn angles(self) -> Vec3 {
        v3(self.0 as f64, self.1 as f64, self.2 as f64)
    }
}

impl Angles for (f64, f64) {
    fn angles(self) -> Vec3 {
        v3(self.0, self.1, 0.0)
    }
}

impl Angles for (f64, f64, f64) {
    fn angles(self) -> Vec3 {
        v3(self.0, self.1, self.2)
    }
}

/// Pose: height above the ground and the angles of each joint (by skeleton joint index).
#[derive(Clone, Debug, PartialEq)]
pub struct Pose {
    pub lift: f64,
    pub angles: Vec<Vec3>,
}

impl Pose {
    pub fn blend(&self, b: &Pose, u: f64) -> Pose {
        if u <= 0.0 {
            return self.clone();
        }
        Pose {
            lift: mix(self.lift, b.lift, u),
            angles: self.angles.iter().zip(&b.angles).map(|(x, y)| x.lerp(*y, u)).collect(),
        }
    }
}

/// Pose builder. Everything not named comes from `base` ("like stand, but
/// right arm raised"), and without base it is zero.
pub struct PoseBuilder<'a> {
    rig: &'a Rig,
    base: Option<&'a Pose>,
    lift: Option<f64>,
    both: Vec<(String, Vec3)>,
    single: Vec<(String, Vec3)>,
}

impl<'a> PoseBuilder<'a> {
    pub fn base(mut self, base: &'a Pose) -> Self {
        self.base = Some(base);
        self
    }

    /// Height of the lowest point above the ground (default — as in base, else 0).
    pub fn lift(mut self, lift: f64) -> Self {
        self.lift = Some(lift);
        self
    }

    /// `arm` — both arms at once, `arm_l` — one.
    pub fn set(mut self, name: &str, a: impl Angles) -> Self {
        let a = a.angles();
        if self.rig.has(name) {
            self.single.push((name.into(), a));
        } else {
            self.both.push((name.into(), a));
        }
        self
    }

    pub fn build(self) -> Result<Pose, RigError> {
        let rig = self.rig;
        let mut angles = match self.base {
            Some(b) => b.angles.clone(),
            None => vec![Vec3::ZERO; rig.len()],
        };
        let lift = self.lift.or(self.base.map(|b| b.lift)).unwrap_or(0.0);
        for (name, a) in &self.both {
            let (l, r) = (format!("{name}_l"), format!("{name}_r"));
            if !rig.has(&l) {
                return Err(RigError(format!("no such joint: {name}")));
            }
            angles[rig.joint(&l)?] = *a;
            angles[rig.joint(&r)?] = *a;
        }
        for (name, a) in &self.single {
            angles[rig.joint(name)?] = *a;
        }
        Ok(Pose { lift, angles })
    }
}

/// A figure in the world: joint points and rotations.
#[derive(Clone, Debug, PartialEq)]
pub struct Figure {
    pub joints: Vec<Vec3>,
    pub rots: Vec<Mat3>,
    /// point at support level under the root
    pub pivot: Vec3,
    pub lift: f64,
}

impl Figure {
    /// The figure after squash and stretch by matrix m around point `pv`.
    pub fn stretched_about(&self, m: Option<Mat3>, pv: Vec3) -> Figure {
        let Some(m) = m else { return self.clone() };
        Figure {
            joints: self.joints.iter().map(|&p| pv + m * (p - pv)).collect(),
            rots: self.rots.clone(),
            pivot: pv,
            lift: self.lift,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leg() -> Rig {
        Rig::new(
            vec![
                Joint::root("pelvis"),
                Joint::new("leg_l", "pelvis", (0.1, 0.0, 0.0)).pitch(-1.0).side(1.0),
                Joint::new("foot_l", "leg_l", (0.0, -0.8, 0.0)).side(1.0),
                Joint::new("leg_r", "pelvis", (-0.1, 0.0, 0.0)).pitch(-1.0).side(-1.0),
                Joint::new("foot_r", "leg_r", (0.0, -0.8, 0.0)).side(-1.0),
            ],
            &[(Side::L, "foot_l", "foot_l", 0.1), (Side::R, "foot_r", "foot_r", 0.1)],
        )
        .unwrap()
    }

    #[test]
    fn parent_must_come_first() {
        let err = Rig::new(vec![Joint::new("knee", "leg", (0.0, 0.0, 0.0))], &[]).unwrap_err();
        assert!(err.0.contains("parent leg"));
    }

    #[test]
    fn both_sides_and_unknown_joint() {
        let rig = leg();
        let p = rig.pose().set("leg", (20, 5)).build().unwrap();
        assert_eq!(p.angles[1], p.angles[3]);
        assert!(rig.pose().set("arm", 1).build().is_err());
    }

    #[test]
    fn mirror_swaps_sides() {
        let rig = leg();
        let p = rig.pose().set("leg_l", 30).build().unwrap();
        let m = rig.mirror(&p);
        assert_eq!(m.angles[3], v3(30.0, 0.0, 0.0));
        assert_eq!(m.angles[1], Vec3::ZERO);
    }

    #[test]
    fn place_stands_on_ground() {
        let rig = leg();
        let p = rig.pose().build().unwrap();
        let (pts, rots) = rig.fk(&p, 0.0, 1.0);
        let fig = rig.place(pts, rots, (1.0, 2.0), 0.0);
        assert!((fig.joints[2].y).abs() < 1e-12);
        assert!((fig.joints[0].y - 0.8).abs() < 1e-12);
    }
}
