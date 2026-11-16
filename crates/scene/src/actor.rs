//! A character over time: pose, foot planting, squash and stretch.
//!
//! Walking without sliding: while a foot is planted it is fixed in the world and the
//! pelvis is computed from it. Heel on the ground — support on the heel; heel raised —
//! support on the toe, which stays where it stood.

use mavian_core::m;
use mavian_core::noise::smooth;
use mavian_core::{Ease, Keys, Mat3, Track, Vec3, mix};
type FigureMemo = mavian_graph::Memo<u64, Arc<Figure>>;
use mavian_rig::{Figure, Model, ModelError, Pose, Side};
use std::sync::{Arc, OnceLock};

/// step for estimating velocity, world seconds
const H: f64 = 1.0 / 48.0;
/// below this speed the figure does not stretch (ordinary walking)
const V_MIN: f64 = 1.5;
/// speed at which stretch is full, units per second
const V_REF: f64 = 5.0;
/// how far back we look for the loss of speed on hitting the ground
const LAG: f64 = 0.10;
/// loss of falling speed at which squash begins
const LOST_MIN: f64 = 0.5;
/// heel height above the toe at which support moves to the toe
const FLAT: f64 = 0.03;
/// step for the pelvis velocity at segment boundaries, world seconds
const VEL_STEP: f64 = 1e-3;
/// above this height a foot point no longer anchors the figure under squash
const GROUND_BAND: f64 = 0.05;
/// speed lines: points along a trail
pub const TRAIL_STEPS: usize = 6;
/// joint speed (m/s) from which a speed line appears; full at twice that
pub const TRAIL_FROM: f64 = 2.0;
/// gravity, m/s² (world units are metres; scaled with the actor)
pub const G: f64 = 9.81;
/// a flight may land this far (fraction of the actor's scale) from the pose's
/// pelvis height at the landing key; beyond that the landing key is mistimed
const LANDING_TOL: f64 = 0.03;

/// How the pelvis moves on the way to a key when no foot is planted.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Move {
    #[default]
    Stay,
    /// forward along the facing by this much
    Forward(f64),
    /// offset (dx, dz)
    Shift(f64, f64),
    /// in the air: the pelvis flies ballistically with the speed it took off
    /// with — horizontally straight on, vertically under gravity — and the
    /// poses only shape the body in flight
    Fly,
}

/// A correction of a pose by code: angles added to a joint over a span of
/// world time (not frame numbers — an MVP lesson: a patch by frames drifted
/// when the timing changed). It fades in and out over `ramp` seconds, so
/// the pose does not jump.
#[derive(Clone, Debug, PartialEq)]
pub struct Patch {
    pub from: f64,
    pub to: f64,
    pub joint: usize,
    /// pitch, roll, twist, degrees
    pub add: Vec3,
    pub ramp: f64,
}

impl Patch {
    /// How much of the patch applies at time w, 0..1.
    pub fn weight(&self, w: f64) -> f64 {
        if w < self.from || w > self.to {
            return 0.0;
        }
        if self.ramp <= 0.0 {
            return 1.0;
        }
        smooth((w - self.from) / self.ramp).min(smooth((self.to - w) / self.ramp))
    }
}

/// An addition over the pose in time: joint angles, a shift on the ground
/// and a lift (a hop). A weight of 1 left after an event keeps the shift for
/// good — so a knockback does not come back.
#[derive(Clone, Debug)]
pub struct Layer {
    /// (joint, pitch-roll-twist at full weight)
    pub angles: Vec<(usize, Vec3)>,
    /// (x, z) at full weight
    pub shift: (f64, f64),
    pub lift: f64,
    pub weight: Track<f64>,
}

/// Pose key: pose name and how the pelvis moves on the way to it.
///
/// `plant` — this foot stands on the ground, the pelvis follows it (no sliding);
/// `mv` — pelvis offset over the segment when no foot is planted.
#[derive(Clone, Debug, PartialEq)]
pub struct PoseKey {
    pub name: String,
    pub plant: Option<Side>,
    pub mv: Move,
    /// part of the way to another pose: (its name, fraction) — the moment a
    /// jump leaves the ground lies inside the push, not at its last pose
    pub toward: Option<(String, f64)>,
}

/// Pose key: `pk("crouch").plant(Side::L)`.
pub fn pk(name: &str) -> PoseKey {
    PoseKey { name: name.into(), plant: None, mv: Move::Stay, toward: None }
}

impl PoseKey {
    pub fn plant(mut self, side: Side) -> Self {
        self.plant = Some(side);
        self
    }

    pub fn forward(mut self, by: f64) -> Self {
        self.mv = Move::Forward(by);
        self
    }

    pub fn shift(mut self, dx: f64, dz: f64) -> Self {
        self.mv = Move::Shift(dx, dz);
        self
    }

    /// This pose part of the way (`by`, 0..1) to pose `other`.
    pub fn toward(mut self, other: &str, by: f64) -> Self {
        self.toward = Some((other.into(), by));
        self
    }

    /// In the air on the way to this key (see `Move::Fly`).
    pub fn fly(mut self) -> Self {
        self.mv = Move::Fly;
        self
    }

    fn moves(&self) -> bool {
        match self.mv {
            Move::Stay => false,
            Move::Forward(d) => d != 0.0,
            Move::Shift(dx, dz) => dx != 0.0 || dz != 0.0,
            Move::Fly => true,
        }
    }
}

impl From<&str> for PoseKey {
    fn from(name: &str) -> Self {
        pk(name)
    }
}

/// What poses an actor has.
#[derive(Clone, Debug, Default)]
pub enum Poses {
    /// the model's default pose
    #[default]
    Rest,
    Fixed(String),
    Keys(Keys<PoseKey>),
}

impl From<Keys<PoseKey>> for Poses {
    fn from(k: Keys<PoseKey>) -> Self {
        Poses::Keys(k)
    }
}

#[derive(Clone, Debug)]
enum Seg {
    Plant {
        side: Side,
        at: (f64, f64),
    },
    /// `v0`, `v1` — pelvis velocity at the start and end, so it does not jump at the boundaries
    Move {
        shift: (f64, f64),
        v0: (f64, f64),
        v1: (f64, f64),
    },
    /// a ballistic flight that took off at `t` from `at` (pelvis xz) at
    /// pelvis height `y0`, with horizontal velocity `v` and vertical `vy`
    Fly {
        t: f64,
        at: (f64, f64),
        y0: f64,
        v: (f64, f64),
        vy: f64,
    },
    Stay,
}

/// Layout of pelvis motion over the segments between pose keys.
#[derive(Clone, Debug)]
struct Plan {
    segs: Vec<(f64, f64, Seg, (f64, f64))>,
    first: (f64, f64),
    last: (f64, f64),
}

/// A character in a shot: a model and what it does over time.
pub struct Actor {
    pub model: Arc<Model>,
    pub poses: Poses,
    /// where it stands: (x, z); a track only if the poses have no planting
    pub path: Track<(f64, f64)>,
    /// facing, degrees
    pub facing: Track<f64>,
    /// squash and stretch strength (0 — off)
    pub squash: f64,
    /// Maximum extra torso tilt/turn for hand contacts, degrees; zero disables it.
    pub body_reach: f64,
    /// Preserve the sampled pose against IK and prop-avoidance corrections.
    /// Pose animation, paths, layers and authored squash/stretch still apply.
    pub pose_lock: bool,
    /// 0 — eyes open, 1 — closed
    pub blink: Track<f64>,
    /// draw the face
    pub face: bool,
    /// 0 — the face by rules, 1 — an honest projection
    pub face_trust: f64,
    pub scale: f64,
    /// own perspective: 1 — honest, more — parts nearer than the root grow
    /// and farther ones shrink faster than by geometry
    pub perspective: f64,
    /// speed lines: how long the trail behind fast joints is, world seconds (0 — none)
    pub trails: f64,
    /// corrections of the pose by code
    pub patches: Vec<Patch>,
    /// additions over the pose set by hand
    pub layers: Vec<Layer>,
    /// additions made by actions (a strike's turn, lunge, knockback)
    pub auto_layers: Vec<Layer>,
    pub(crate) recoils: Vec<crate::recoil::Recoil>,
    pub(crate) approach_steps: Vec<crate::approach::Step>,
    /// Compiled decisions for explicitly requested preparatory steps.
    pub step_reports: Vec<crate::StepReport>,
    /// targets for the hands; actions add them (grab, put down, push)
    pub reaches: Vec<crate::ik::Reach>,
    /// Authored contacts survive action recompilation.
    pub contacts: Vec<crate::Contact>,
    /// Contacts generated by grab, release and push.
    pub auto_contacts: Vec<crate::Contact>,
    /// constraints after the targets: arms do not enter another's head
    pub avoids: Vec<crate::ik::Avoid>,
    /// character colour; none — the style's ink colour
    pub color: Option<String>,
    plan: OnceLock<Option<Plan>>,
    figures: FigureMemo,
    /// figures with hands at their targets (they depend on the shot's world)
    reached: FigureMemo,
}

impl Actor {
    pub fn new(model: Arc<Model>) -> Actor {
        Actor {
            model,
            poses: Poses::Rest,
            path: Track::Const((0.0, 0.0)),
            facing: Track::Const(0.0),
            squash: 0.0,
            body_reach: 0.0,
            pose_lock: false,
            blink: Track::Const(0.0),
            face: true,
            face_trust: 0.0,
            scale: 1.0,
            perspective: 1.0,
            trails: 0.0,
            patches: Vec::new(),
            layers: Vec::new(),
            auto_layers: Vec::new(),
            recoils: Vec::new(),
            approach_steps: Vec::new(),
            step_reports: Vec::new(),
            reaches: Vec::new(),
            contacts: Vec::new(),
            auto_contacts: Vec::new(),
            avoids: Vec::new(),
            color: None,
            plan: OnceLock::new(),
            figures: FigureMemo::default(),
            reached: FigureMemo::default(),
        }
    }

    pub fn poses(mut self, p: impl Into<Poses>) -> Self {
        self.poses = p.into();
        self.forget();
        self
    }

    pub fn path(mut self, p: impl Into<Track<(f64, f64)>>) -> Self {
        self.path = p.into();
        self.forget();
        self
    }

    pub fn facing(mut self, f: impl Into<Track<f64>>) -> Self {
        self.facing = f.into();
        self.forget();
        self
    }

    pub fn squash(mut self, k: f64) -> Self {
        self.squash = k;
        self
    }

    pub fn blink(mut self, b: impl Into<Track<f64>>) -> Self {
        self.blink = b.into();
        self
    }

    pub fn face(mut self, on: bool) -> Self {
        self.face = on;
        self
    }

    pub fn face_trust(mut self, t: f64) -> Self {
        self.face_trust = t;
        self
    }

    pub fn scale(mut self, s: f64) -> Self {
        self.scale = s;
        self.forget();
        self
    }

    pub fn perspective(mut self, k: f64) -> Self {
        self.perspective = k;
        self
    }

    pub fn trails(mut self, seconds: f64) -> Self {
        self.trails = seconds;
        self
    }

    pub fn patch(mut self, p: Patch) -> Self {
        self.patches.push(p);
        self.forget();
        self
    }

    pub fn color(mut self, c: &str) -> Self {
        self.color = Some(c.into());
        self
    }

    /// Drop caches (after the description changes).
    pub fn forget(&mut self) {
        self.plan = OnceLock::new();
        self.figures = FigureMemo::default();
        self.reached = FigureMemo::default();
    }

    /// The figure with the hands at their targets and constraints applied,
    /// in the shot's world (a target may be on a prop). Cached like `figure`.
    pub fn figure_ik(&self, w: f64, world: &crate::props::World) -> Arc<Figure> {
        if self.pose_lock || (self.reaches.is_empty() && self.avoids.is_empty() && self.contacts.is_empty() && self.auto_contacts.is_empty()) {
            return self.figure(w);
        }
        self.reached.get_or(w.to_bits(), || {
            let base = crate::ik::reach_all(&self.figure(w), &self.reaches, &self.avoids, w, world);
            let contacts: Vec<_> = self.contacts.iter().chain(&self.auto_contacts).cloned().collect();
            Arc::new(crate::contact::solve(&base, &contacts, w, world).0)
        })
    }

    /// Check the description: the model has all poses, planting does not conflict with the path.
    pub fn validate(&self) -> Result<(), ModelError> {
        if !self.body_reach.is_finite() || !(0.0..=60.0).contains(&self.body_reach) {
            return Err(ModelError("body_reach must be finite and in [0, 60] degrees".into()));
        }
        let contacts: Vec<_> = self.contacts.iter().chain(&self.auto_contacts).collect();
        for (i, c) in contacts.iter().enumerate() {
            c.validate(&self.model.rig).map_err(ModelError)?;
            if contacts[..i].iter().any(|other| other.name == c.name && other.from < c.to && c.from < other.to) {
                return Err(ModelError(format!("overlapping contacts share the name '{}'", c.name)));
            }
        }
        match &self.poses {
            Poses::Rest => {
                self.model.pose(&self.model.rest)?;
            }
            Poses::Fixed(n) => {
                self.model.pose(n)?;
            }
            Poses::Keys(k) => {
                for (_, key, _) in k.items() {
                    self.model.pose(&key.name)?;
                    if let Some((other, _)) = &key.toward {
                        self.model.pose(other)?;
                    }
                    if key.plant.is_some() && self.model.rig.foot(key.plant.unwrap()).is_none() {
                        return Err(ModelError(format!("model {} has no foot to plant", self.model.name)));
                    }
                }
                let planted = k.items().iter().any(|(_, v, _)| v.plant.is_some() || v.moves());
                if planted && matches!(self.path, Track::Keys(_)) {
                    return Err(ModelError("foot planting and a path track do not work together: give path as a starting point (x, z)".into()));
                }
            }
        }
        // a flight comes down when gravity says, not when a key says
        if let Some((t, t_land, err, real)) = self.mistimed_landings().into_iter().next() {
            let when = if real.is_finite() { format!("; it comes down at {real:.3} s") } else { String::new() };
            return Err(ModelError(format!(
                "the flight from {t:.3} s does not land at its key at {t_land:.3} s: the pelvis is {:.0} cm {} the landing pose{when} \
                 (move the landing key, or plan the jump with jump())",
                err.abs() * 100.0,
                if err > 0.0 { "above" } else { "below" },
            )));
        }
        Ok(())
    }

    fn pose_of(&self, name: &str) -> &Pose {
        self.model.pose(name).expect("poses are checked in Shot::validate")
    }

    /// Pose at world time w (a blend of neighbouring keys).
    pub fn pose_at(&self, w: f64) -> Pose {
        let mut pose = match &self.poses {
            Poses::Rest => self.pose_of(&self.model.rest).clone(),
            Poses::Fixed(n) => self.pose_of(n).clone(),
            Poses::Keys(k) => {
                let (a, b, u) = k.span(w);
                self.key_pose(a).blend(&self.key_pose(b), u)
            }
        };
        for p in &self.patches {
            let k = p.weight(w);
            if k > 0.0 {
                pose.angles[p.joint] = pose.angles[p.joint] + p.add * k;
            }
        }
        pose
    }

    /// The pose a key stands for.
    fn key_pose(&self, k: &PoseKey) -> Pose {
        match &k.toward {
            Some((other, by)) => self.pose_of(&k.name).blend(self.pose_of(other), *by),
            None => self.pose_of(&k.name).clone(),
        }
    }

    fn fk_key(&self, k: &PoseKey, w: f64) -> Vec<Vec3> {
        self.model.rig.fk(&self.key_pose(k), self.facing.at(w), self.scale).0
    }

    /// Foot support point relative to the root, (x, z).
    fn contact(&self, joints: &[Vec3], side: Side, heading: (f64, f64)) -> (f64, f64) {
        contact(&self.model, self.scale, joints, side, heading)
    }

    fn plan(&self) -> Option<&Plan> {
        self.plan.get_or_init(|| self.build_plan()).as_ref()
    }

    fn build_plan(&self) -> Option<Plan> {
        let Poses::Keys(keys) = &self.poses else { return None };
        let items = keys.items();
        if !items.iter().any(|(_, v, _)| v.plant.is_some() || v.moves()) {
            return None;
        }
        let Track::Const(start) = self.path else { return None };
        let mut root = start;
        let mut plant: Option<(Side, (f64, f64))> = None;
        let mut segs = Vec::new();
        for pair in items.windows(2) {
            let ((t0, v0, _), (t1, k, _)) = (&pair[0], &pair[1]);
            let (t0, t1) = (*t0, *t1);
            if let Some(side) = k.plant {
                if plant.is_none_or(|(s, _)| s != side) {
                    let c = self.contact(&self.fk_key(v0, t0), side, dir(self.facing.at(t0)));
                    plant = Some((side, (root.0 + c.0, root.1 + c.1)));
                }
                let at = plant.unwrap().1;
                segs.push((t0, t1, Seg::Plant { side, at }, root));
                let c = self.contact(&self.fk_key(k, t1), side, dir(self.facing.at(t1)));
                root = (at.0 - c.0, at.1 - c.1);
            } else {
                plant = None;
                if k.mv == Move::Fly {
                    // one flight across consecutive keys in the air
                    let fly = match segs.last() {
                        Some((_, _, f @ Seg::Fly { .. }, _)) => f.clone(),
                        prev => {
                            let v = prev.map_or((0.0, 0.0), |p| self.edge_velocity(p, true));
                            let d = VEL_STEP.min(prev.map_or(VEL_STEP, |p| (p.1 - p.0) / 4.0));
                            let (y0, y1, y2) = (self.pelvis_height(t0), self.pelvis_height(t0 - d), self.pelvis_height(t0 - 2.0 * d));
                            Seg::Fly { t: t0, at: root, y0, v, vy: (3.0 * y0 - 4.0 * y1 + y2) / (2.0 * d) }
                        }
                    };
                    let Seg::Fly { t, at, v, .. } = fly else { unreachable!() };
                    segs.push((t0, t1, fly, root));
                    root = (at.0 + v.0 * (t1 - t), at.1 + v.1 * (t1 - t));
                } else if k.moves() {
                    let shift = match k.mv {
                        Move::Forward(d) => {
                            let h = dir(self.facing.at(t1));
                            (h.0 * d, h.1 * d)
                        }
                        Move::Shift(dx, dz) => (dx, dz),
                        Move::Stay | Move::Fly => unreachable!(),
                    };
                    segs.push((t0, t1, Seg::Move { shift, v0: (0.0, 0.0), v1: (0.0, 0.0) }, root));
                    root = (root.0 + shift.0, root.1 + shift.1);
                } else {
                    segs.push((t0, t1, Seg::Stay, root));
                }
            }
        }
        self.join_velocities(&mut segs);
        Some(Plan { segs, first: start, last: root })
    }

    /// Pelvis position in a planted segment: the foot stays, the pelvis follows the pose.
    fn plant_root(&self, side: Side, at: (f64, f64), w: f64) -> (f64, f64) {
        let joints = self.model.rig.fk(&self.pose_at(w), self.facing.at(w), self.scale).0;
        let c = self.contact(&joints, side, dir(self.facing.at(w)));
        (at.0 - c.0, at.1 - c.1)
    }

    /// Pelvis velocity of a segment at its start (`end = false`) or end.
    ///
    /// A planted segment's velocity comes from the pose (one-sided difference
    /// inside the segment); standing still is zero; between two moves it is
    /// their combined average (as Catmull-Rom).
    fn edge_velocity(&self, seg: &(f64, f64, Seg, (f64, f64)), end: bool) -> (f64, f64) {
        let (t0, t1, kind, _) = seg;
        match kind {
            Seg::Stay => (0.0, 0.0),
            Seg::Move { shift, .. } => (shift.0 / (t1 - t0), shift.1 / (t1 - t0)),
            Seg::Fly { v, .. } => *v,
            Seg::Plant { side, at } => {
                // one-sided, second order: exact for the quadratic of an
                // eased key, so a flight takes off with the push's true speed
                let d = VEL_STEP.min((t1 - t0) / 4.0);
                let k = if end { -1.0 } else { 1.0 };
                let edge = if end { *t1 } else { *t0 };
                let f = |i: f64| self.plant_root(*side, *at, edge + k * i * d);
                let (f0, f1, f2) = (f(0.0), f(1.0), f(2.0));
                (k * (-3.0 * f0.0 + 4.0 * f1.0 - f2.0) / (2.0 * d), k * (-3.0 * f0.1 + 4.0 * f1.1 - f2.1) / (2.0 * d))
            }
        }
    }

    /// Pelvis velocity must not jump where segments meet (MVP lesson: a take-off
    /// at 10.6 m/s went straight into a flight at a constant 3.6 m/s — a jerk).
    /// Moves keep their endpoints (so plant points do not change) and take the
    /// neighbours' velocity at their ends; before the first and after the last
    /// segment the pelvis stands.
    fn join_velocities(&self, segs: &mut [(f64, f64, Seg, (f64, f64))]) {
        let mut tangents = Vec::with_capacity(segs.len());
        for i in 0..segs.len() {
            if !matches!(segs[i].2, Seg::Move { .. }) {
                tangents.push(None);
                continue;
            }
            let side = |j: Option<usize>, end: bool| -> (f64, f64) {
                let Some(j) = j else { return (0.0, 0.0) };
                let other = &segs[j];
                match other.2 {
                    // two moves in a row: average over both, weighted by time
                    Seg::Move { shift, .. } => {
                        let (a, b) = (&segs[i], other);
                        let s = match a.2 {
                            Seg::Move { shift: sa, .. } => (sa.0 + shift.0, sa.1 + shift.1),
                            _ => unreachable!(),
                        };
                        let dt = (a.1 - a.0) + (b.1 - b.0);
                        (s.0 / dt, s.1 / dt)
                    }
                    _ => self.edge_velocity(other, end),
                }
            };
            let v0 = side(i.checked_sub(1), true);
            let v1 = side((i + 1 < segs.len()).then_some(i + 1), false);
            tangents.push(Some((v0, v1)));
        }
        for (seg, t) in segs.iter_mut().zip(tangents) {
            if let (Seg::Move { v0, v1, .. }, Some((a, b))) = (&mut seg.2, t) {
                (*v0, *v1) = (a, b);
            }
        }
    }

    /// Where the pelvis is on the ground plane at time w (`joints` — FK of the pose at that time).
    fn root(&self, w: f64, joints: &[Vec3]) -> (f64, f64) {
        let Some(plan) = self.plan() else { return self.path.at(w) };
        if plan.segs.is_empty() || w < plan.segs[0].0 {
            return plan.first;
        }
        for (t0, t1, seg, root0) in &plan.segs {
            if *t0 <= w && w < *t1 {
                return match seg {
                    Seg::Plant { side, at } => {
                        let c = self.contact(joints, *side, dir(self.facing.at(w)));
                        (at.0 - c.0, at.1 - c.1)
                    }
                    Seg::Move { shift, v0, v1 } => {
                        // cubic Hermite: ends at the given offset, velocities match the neighbours
                        let dt = t1 - t0;
                        let u = (w - t0) / dt;
                        let (u2, u3) = (u * u, u * u * u);
                        let (h10, h01, h11) = (u3 - 2.0 * u2 + u, -2.0 * u3 + 3.0 * u2, u3 - u2);
                        (root0.0 + h10 * v0.0 * dt + h01 * shift.0 + h11 * v1.0 * dt, root0.1 + h10 * v0.1 * dt + h01 * shift.1 + h11 * v1.1 * dt)
                    }
                    Seg::Fly { t, at, v, .. } => (at.0 + v.0 * (w - t), at.1 + v.1 * (w - t)),
                    Seg::Stay => *root0,
                };
            }
        }
        plan.last
    }

    /// The plan segment at time w.
    fn segment(&self, w: f64) -> Option<&Seg> {
        let plan = self.plan()?;
        plan.segs.iter().find(|(t0, t1, _, _)| *t0 <= w && w < *t1).map(|(_, _, s, _)| s)
    }

    /// Pelvis height at time w with the pose standing on the ground (lift
    /// from the pose).
    fn pelvis_height(&self, w: f64) -> f64 {
        pelvis_height(&self.model, &self.pose_at(w), self.scale)
    }

    /// Flights that do not come down at their landing key: (take-off time,
    /// landing key time, pelvis height error at the key, when the pelvis
    /// really reaches the landing pose's height).
    pub fn mistimed_landings(&self) -> Vec<(f64, f64, f64, f64)> {
        let Some(plan) = self.plan() else { return Vec::new() };
        let mut out = Vec::new();
        for pair in plan.segs.windows(2) {
            let ((_, t_land, a, _), (_, _, b, _)) = (&pair[0], &pair[1]);
            let (Seg::Fly { t, y0, vy, .. }, false) = (a, matches!(b, Seg::Fly { .. })) else { continue };
            let g = G * self.scale;
            let tau = t_land - t;
            let y = y0 + vy * tau - 0.5 * g * tau * tau;
            let want = self.pelvis_height(*t_land);
            if (y - want).abs() > LANDING_TOL * self.scale {
                let disc = vy * vy + 2.0 * g * (y0 - want);
                let real = if disc >= 0.0 { t + (vy + disc.sqrt()) / g } else { f64::NAN };
                out.push((*t, *t_land, y - want, real));
            }
        }
        out
    }

    /// Which foot is planted at time w.
    pub fn planted_foot(&self, w: f64) -> Option<Side> {
        if !self.pose_lock
            && let Some(step) = self.approach_steps.iter().find(|s| s.active(w))
        {
            return Some(step.support(w));
        }
        if let Some(r) = self.recoils.iter().rev().find(|r| r.active(w)) {
            return r.support(w);
        }
        let plan = self.plan()?;
        plan.segs.iter().find(|(t0, t1, _, _)| *t0 <= w && w < *t1).and_then(|(_, _, s, _)| match s {
            Seg::Plant { side, .. } => Some(*side),
            _ => None,
        })
    }

    /// Figure at world time w — before squash and stretch. Cached:
    /// it is asked for many times per frame (camera, squash, speed trails).
    pub fn figure(&self, w: f64) -> Arc<Figure> {
        self.figures.get_or(w.to_bits(), || Arc::new(self.compute_figure(w)))
    }

    fn compute_figure(&self, w: f64) -> Figure {
        let rig = &self.model.rig;
        let mut p = self.pose_at(w);
        // layers over the pose: angles, a hop, a shift on the ground
        let (mut dx, mut dz) = (0.0, 0.0);
        for l in self.layers.iter().chain(&self.auto_layers) {
            let k = l.weight.at(w);
            if k == 0.0 {
                continue;
            }
            for (j, a) in &l.angles {
                p.angles[*j] = p.angles[*j] + *a * k;
            }
            p.lift += l.lift * k;
            (dx, dz) = (dx + l.shift.0 * k, dz + l.shift.1 * k);
        }
        let (joints, rots) = rig.fk(&p, self.facing.at(w), self.scale);
        let root = self.root(w, &joints);
        let root = (root.0 + dx, root.1 + dz);
        if let Some(Seg::Fly { t, y0, vy, .. }) = self.segment(w) {
            // the pelvis at its ballistic height; the feet go where the pose puts them
            let tau = w - t;
            let y = y0 + vy * tau - 0.5 * G * self.scale * tau * tau;
            let on_ground = rig.place(joints.clone(), rots.clone(), root, 0.0).joints[self.model.root].y;
            let mut fig = rig.place(joints, rots, root, y - on_ground);
            if !self.pose_lock {
                for step in &self.approach_steps {
                    step.apply(&mut fig, w);
                }
            }
            return fig;
        }
        let mut fig = rig.place(joints, rots, root, p.lift);
        if let Some(r) = self.recoils.iter().rev().find(|r| r.active(w)) {
            r.apply(&mut fig, w);
        }
        if !self.pose_lock {
            for step in &self.approach_steps {
                step.apply(&mut fig, w);
            }
        }
        fig
    }

    /// Where a character's joint is at time w; without a name — the model's root joint.
    pub fn where_(&self, w: f64, joint: Option<&str>) -> Vec3 {
        let i = match joint {
            Some(n) => self.model.rig.joint(n).expect("joint"),
            None => self.model.root,
        };
        self.figure(w).joints[i]
    }

    /// Squash and stretch from the motion of the root joint.
    ///
    /// Stretch goes along the velocity, stronger the faster it is. Squash is
    /// vertical, when a fall stops abruptly. Volume is preserved.
    pub fn stretch(&self, w: f64) -> Option<Mat3> {
        let k = self.squash;
        if k <= 0.0 {
            return None;
        }
        let root = self.model.root;
        let velocity = |t: f64| {
            let a = self.figure(t - H).joints[root];
            let b = self.figure(t + H).joints[root];
            (b - a) * (1.0 / (2.0 * H))
        };
        let v = velocity(w);
        let speed = v.length();
        let lost = f64::max(0.0, -velocity(w - LAG).y - f64::max(-v.y, 0.0));
        let squash = 1.0 - k * f64::min(f64::max(lost - LOST_MIN, 0.0) / (V_REF / 2.0 - LOST_MIN), 1.0);
        let mut m = Mat3::scale_along(Vec3::UP, squash, 1.0 / squash.sqrt());
        if speed > V_MIN {
            let stretch = 1.0 + k * f64::min((speed - V_MIN) / (V_REF - V_MIN), 1.0);
            m = m * Mat3::scale_along(v * (1.0 / speed), stretch, 1.0 / stretch.sqrt());
        }
        Some(m)
    }
}

impl Actor {
    /// Which point to squash the figure around: the feet standing on the ground.
    ///
    /// Squashing around the point under the pelvis moves planted feet (in the MVP on
    /// landing — up to 15 cm per frame). Each foot point pulls the centre towards itself
    /// the more, the closer it is to the ground; in the air the centre is under the pelvis. The weight
    /// changes smoothly with foot height, so the centre does not jump.
    pub fn stretch_pivot(&self, fig: &Figure) -> Vec3 {
        let rig = &self.model.rig;
        let (mut sum, mut total, mut most) = (Vec3::ZERO, 0.0, 0.0f64);
        for side in [Side::L, Side::R] {
            let Some(f) = rig.foot(side) else { continue };
            for j in [f.heel, f.toe] {
                let p = fig.joints[j];
                let g = 1.0 - smooth(p.y / (GROUND_BAND * self.scale));
                sum = sum + Vec3 { y: 0.0, ..p } * g;
                total += g;
                most = most.max(g);
            }
        }
        if most <= 0.0 {
            return fig.pivot;
        }
        let feet = sum * (1.0 / total);
        let p = fig.pivot.lerp(Vec3 { y: fig.pivot.y, ..feet }, most);
        Vec3 { y: fig.pivot.y, ..p }
    }

    /// Foot points on the ground are not squashed: diagonal squash (along the velocity)
    /// moves the toe relative to the heel and the foot slides. A point at the ground stays
    /// where it was, a raised one squashes with the body; in between — smoothly.
    pub fn pin_feet(&self, raw: &Figure, out: &mut Figure) {
        let rig = &self.model.rig;
        for side in [Side::L, Side::R] {
            let Some(f) = rig.foot(side) else { continue };
            for j in [f.heel, f.toe] {
                let g = 1.0 - smooth(raw.joints[j].y / (GROUND_BAND * self.scale));
                out.joints[j] = out.joints[j].lerp(raw.joints[j], g);
            }
        }
    }
}

/// Where the character faces on the ground plane: (x, z).
/// Foot support point relative to the root, (x, z). Heel on the ground —
/// the heel; heel raised — the toe, which stays where it stood.
pub fn contact(model: &Model, scale: f64, joints: &[Vec3], side: Side, heading: (f64, f64)) -> (f64, f64) {
    let f = model.rig.foot(side).expect("foot is checked");
    let (ankle, toe) = (joints[f.heel], joints[f.toe]);
    let on_toe = smooth((ankle.y - toe.y) / (FLAT * scale));
    (mix(ankle.x, toe.x - f.length * scale * heading.0, on_toe), mix(ankle.z, toe.z - f.length * scale * heading.1, on_toe))
}

/// Pelvis height of a pose standing on the ground (with the pose's lift).
pub fn pelvis_height(model: &Model, pose: &Pose, scale: f64) -> f64 {
    let (pts, rots) = model.rig.fk(pose, 0.0, scale);
    model.rig.place(pts, rots, (0.0, 0.0), pose.lift).joints[model.root].y
}

fn dir(deg: f64) -> (f64, f64) {
    let a = deg.to_radians();
    (m::sin(a), m::cos(a))
}

/// Pose keys with easing: a shorthand for tracks.
pub fn pose_keys() -> Keys<PoseKey> {
    Keys::new(Ease::InOut)
}
