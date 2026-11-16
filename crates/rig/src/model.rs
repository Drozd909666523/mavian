//! Character model: skeleton, the parts it is drawn from, and its poses.
//!
//! A model describes "what it is", without time. What it does in a shot belongs to the actor.

use crate::rig::{Pose, Rig, RigError};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ModelError(pub String);

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ModelError {}

impl From<RigError> for ModelError {
    fn from(e: RigError) -> Self {
        ModelError(e.0)
    }
}

/// Features of a face on a ball head and the rules for drawing them
/// (`engine/face.py` of the MVP). Angles are in degrees.
#[derive(Clone, Debug, PartialEq)]
pub struct Face {
    /// where an eye is on the ball: sideways from the "nose"
    pub eye_az: f64,
    /// and up
    pub eye_el: f64,
    /// half width and half height of an eye
    pub eye_rw: f64,
    pub eye_rh: f64,
    /// the mouth: below the "nose"
    pub mouth_el: f64,
    pub mouth_half: f64,
    /// how much the mouth's corners are above its middle
    pub smile: f64,
    pub eye_n: usize,
    pub mouth_n: usize,
    // rules for trust = 0
    /// how much features are pulled to the centre in profile
    pub eye_pull: f64,
    pub mouth_pull: f64,
    /// at which angles from the view a feature fades out
    pub eye_hide: (f64, f64),
    pub mouth_hide: (f64, f64),
    /// trust in geometry for pitch: 0 — features do not move, 1 — as on the
    /// ball. Turning left-right goes by rules, but an honest projection from
    /// below or above reads well
    pub pitch_gain: f64,
    /// but not beyond this angle: close from below (fisheye) the eyes would
    /// slide onto the crown — "rolled the eyes"
    pub pitch_limit: f64,
}

impl Default for Face {
    fn default() -> Face {
        Face {
            eye_az: 24.0,
            eye_el: 7.0,
            eye_rw: 8.5,
            eye_rh: 14.5,
            mouth_el: -33.0,
            mouth_half: 16.0,
            smile: 7.0,
            eye_n: 24,
            mouth_n: 13,
            eye_pull: 0.30,
            mouth_pull: 0.12,
            eye_hide: (50.0, 100.0),
            mouth_hide: (100.0, 122.0),
            pitch_gain: 0.9,
            pitch_limit: 28.0,
        }
    }
}

/// A part the character is drawn from. This is data: how the part looks
/// in the frame is decided by the drawing stage.
#[derive(Clone, Debug)]
pub enum Part {
    /// A stick bone from joint a to joint b.
    Stick {
        name: String,
        a: usize,
        b: usize,
        /// width at the start and end, fractions of the style's line width
        w0: f64,
        w1: f64,
        /// free end: there is no joint at it
        free: bool,
        /// dot at the end (a hand)
        dot: bool,
        /// cut the end by this much (for the neck — up to the head's surface)
        trim: f64,
        /// a light halo outside the line (separates overlaps); without it
        /// overlapping bones merge into one silhouette
        halo: bool,
        /// how width depends on depth; none — the style's
        gamma: Option<f64>,
    },
    /// A ball head on a joint; `line` — outline width, fraction of the style's
    /// line; `face` — where the eyes and mouth are on the ball.
    /// `filled` — a solid disc of the character's colour, without a face
    /// (the silhouette stick figure).
    Head { joint: usize, radius: f64, line: f64, face: Face, filled: bool },
    /// A chain of capsules along joints (arm, leg, neck); `radii` — radius at
    /// each joint. `root` — the part it grows from: no seam there.
    Limb { name: String, chain: Vec<usize>, radii: Vec<f64>, root: Option<String> },
    /// A torso: a capsule with an oval section from joint a to b. `rx` — half
    /// width (sideways), `rz` — half depth (front-back) at each end; `trim`
    /// shortens it at the ends; `cap` — how far the rounded cap reaches past
    /// each end (by default the half width, as in the MVP). Orientation comes
    /// from joint a.
    Body { name: String, a: usize, b: usize, rx: (f64, f64), rz: (f64, f64), trim: (f64, f64), cap: (f64, f64) },
    /// A ball on a joint: a hand, a hinge.
    Ball { joint: usize, radius: f64 },
}

impl Part {
    pub fn stick(rig: &Rig, name: &str, a: &str, b: &str, w0: f64, w1: f64) -> Result<Part, RigError> {
        Ok(Part::Stick { name: name.into(), a: rig.joint(a)?, b: rig.joint(b)?, w0, w1, free: false, dot: false, trim: 0.0, halo: true, gamma: None })
    }

    pub fn head(rig: &Rig, joint: &str, radius: f64) -> Result<Part, RigError> {
        Ok(Part::Head { joint: rig.joint(joint)?, radius, line: 0.95, face: Face::default(), filled: false })
    }

    pub fn limb(rig: &Rig, name: &str, chain: &[&str], radii: &[f64], root: Option<&str>) -> Result<Part, RigError> {
        if chain.len() < 2 || chain.len() != radii.len() {
            return Err(RigError(format!("limb {name}: needs 2+ joints and one radius per joint")));
        }
        let chain = chain.iter().map(|j| rig.joint(j)).collect::<Result<_, _>>()?;
        Ok(Part::Limb { name: name.into(), chain, radii: radii.to_vec(), root: root.map(String::from) })
    }

    pub fn body(rig: &Rig, name: &str, a: &str, b: &str, rx: (f64, f64), rz: (f64, f64), trim: (f64, f64)) -> Result<Part, RigError> {
        Ok(Part::Body { name: name.into(), a: rig.joint(a)?, b: rig.joint(b)?, rx, rz, trim, cap: rx })
    }

    /// How far a torso's caps reach past its ends.
    pub fn cap(mut self, ends: (f64, f64)) -> Part {
        if let Part::Body { cap, .. } = &mut self {
            *cap = ends;
        }
        self
    }

    pub fn ball(rig: &Rig, joint: &str, radius: f64) -> Result<Part, RigError> {
        Ok(Part::Ball { joint: rig.joint(joint)?, radius })
    }

    /// Outline width of a head, fraction of the style's line.
    pub fn line(mut self, w: f64) -> Part {
        if let Part::Head { line, .. } = &mut self {
            *line = w;
        }
        self
    }

    /// A stick without a halo.
    pub fn no_halo(mut self) -> Part {
        if let Part::Stick { halo, .. } = &mut self {
            *halo = false;
        }
        self
    }

    /// A stick's own dependence of width on depth.
    pub fn gamma(mut self, g: f64) -> Part {
        if let Part::Stick { gamma, .. } = &mut self {
            *gamma = Some(g);
        }
        self
    }

    /// A head as a solid disc of the character's colour, without a face.
    pub fn filled(mut self) -> Part {
        if let Part::Head { filled, .. } = &mut self {
            *filled = true;
        }
        self
    }

    pub fn free(mut self) -> Part {
        if let Part::Stick { free, .. } = &mut self {
            *free = true;
        }
        self
    }

    pub fn dot(mut self) -> Part {
        if let Part::Stick { dot, .. } = &mut self {
            *dot = true;
        }
        self
    }

    pub fn trim(mut self, by: f64) -> Part {
        if let Part::Stick { trim, .. } = &mut self {
            *trim = by;
        }
        self
    }

    pub fn name<'a>(&'a self, rig: &'a Rig) -> &'a str {
        match self {
            Part::Stick { name, .. } => name,
            Part::Limb { name, .. } | Part::Body { name, .. } => name,
            Part::Head { joint, .. } | Part::Ball { joint, .. } => &rig.joints()[*joint].name,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    pub name: String,
    pub rig: Arc<Rig>,
    pub parts: Vec<Part>,
    poses: Vec<(String, Pose)>,
    pose_index: HashMap<String, usize>,
    /// default pose
    pub rest: String,
    /// root joint: path, shadow, squash, line width
    pub root: usize,
    /// joints that leave speed trails
    pub trails: Vec<usize>,
    /// shadow blob radius (0 — no shadow)
    pub shadow: f64,
    /// overlap rules: (part, part under it) — the first is always on top of the
    /// second, however the geometry lies ("head over neck")
    pub over: Vec<(String, String)>,
}

impl Model {
    pub fn new(name: &str, rig: Rig, parts: Vec<Part>) -> Model {
        Model {
            name: name.into(),
            rig: Arc::new(rig),
            parts,
            poses: Vec::new(),
            pose_index: HashMap::new(),
            rest: String::new(),
            root: 0,
            trails: Vec::new(),
            shadow: 0.0,
            over: Vec::new(),
        }
    }

    /// Add a pose; the first one added is the default until rest is set.
    pub fn add_pose(&mut self, name: &str, pose: Pose) -> Result<(), ModelError> {
        if pose.angles.len() != self.rig.len() {
            return Err(ModelError(format!("model {}: pose '{name}' is not from this skeleton", self.name)));
        }
        if self.rest.is_empty() {
            self.rest = name.into();
        }
        match self.pose_index.get(name) {
            Some(&i) => self.poses[i].1 = pose,
            None => {
                self.pose_index.insert(name.into(), self.poses.len());
                self.poses.push((name.into(), pose));
            }
        }
        Ok(())
    }

    pub fn pose(&self, name: &str) -> Result<&Pose, ModelError> {
        self.pose_index.get(name).map(|&i| &self.poses[i].1).ok_or_else(|| {
            let known: Vec<&str> = self.poses.iter().map(|(n, _)| n.as_str()).collect();
            ModelError(format!("model {} has no pose '{name}'; available: {}", self.name, known.join(", ")))
        })
    }

    /// All poses in the order they were added.
    pub fn poses(&self) -> impl Iterator<Item = (&str, &Pose)> {
        self.poses.iter().map(|(n, p)| (n.as_str(), p))
    }

    /// Index of a part by name.
    pub fn part(&self, name: &str) -> Option<usize> {
        self.parts.iter().position(|p| p.name(&self.rig) == name)
    }

    pub fn has_pose(&self, name: &str) -> bool {
        self.pose_index.contains_key(name)
    }
}
