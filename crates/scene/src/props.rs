//! Props: a box or a ball, their motion in time (a port of the MVP's
//! `engine/props.py`).
//!
//! A prop is a shape and a motion. The motion is a list of segments (rests,
//! held, slides with a character, falls to a support, rides on another prop);
//! actions (`actions.rs`) make it when the shot is compiled. The position at
//! any moment is a pure function of world time — in the world of the shot:
//! a held prop follows a hand, and a hand may reach for a prop.
//!
//! A prop's anchor is the middle of its bottom: `at` is where it stands.

use crate::actor::{Actor, G};
use mavian_core::{Mat3, Vec3};

#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    Box { sx: f64, sy: f64, sz: f64 },
    Sphere { r: f64 },
}

impl Shape {
    pub fn top(&self) -> f64 {
        match self {
            Shape::Box { sy, .. } => *sy,
            Shape::Sphere { r } => 2.0 * r,
        }
    }
}

/// How a prop moves from some moment on.
#[derive(Clone, Debug)]
pub enum Motion {
    Rest {
        pos: Vec3,
        rot: Mat3,
    },
    /// falls from a height to a support, then lies there
    Fall {
        pos: Vec3,
        rot: Mat3,
        floor: f64,
        t0: f64,
    },
    /// in a hand: follows it, but stays upright (turns with the arm only
    /// about the vertical) — as a cup or a box is held
    Held {
        actor: usize,
        hand: usize,
        offset: Vec3,
        yaw: f64,
    },
    /// while a hand puts it down on a support it does not go through it
    Above {
        held: Box<Motion>,
        floor: f64,
    },
    /// slides on the ground with a character (pushed)
    Slide {
        actor: usize,
        offset: Vec3,
        y: f64,
        rot: Mat3,
    },
    /// stands on another prop and rides with it; put above its top, it
    /// first falls onto it
    Ride {
        support: usize,
        offset: Vec3,
        yaw: f64,
        drop: f64,
        t0: f64,
    },
}

/// The shot's world, as motions and targets see it.
pub struct World<'a> {
    pub actors: &'a [Actor],
    pub props: &'a [Prop],
}

impl Motion {
    pub fn at(&self, w: f64, world: &World) -> (Vec3, Mat3) {
        match self {
            Motion::Rest { pos, rot } => (*pos, *rot),
            Motion::Fall { pos, rot, floor, t0 } => {
                let y = floor.max(pos.y - 0.5 * G * (w - t0) * (w - t0));
                (Vec3 { y, ..*pos }, *rot)
            }
            Motion::Held { actor, hand, offset, yaw } => {
                let a = &world.actors[*actor];
                let p = a.figure_ik(w, world).joints[*hand];
                let heading = a.facing.at(w);
                (p + Mat3::rot_y(heading) * *offset, Mat3::rot_y(heading + yaw))
            }
            Motion::Above { held, floor } => {
                let (p, r) = held.at(w, world);
                (Vec3 { y: p.y.max(*floor), ..p }, r)
            }
            Motion::Slide { actor, offset, y, rot } => {
                let a = &world.actors[*actor];
                let root = a.figure(w).joints[a.model.root];
                (Vec3 { x: root.x + offset.x, y: *y, z: root.z + offset.z }, *rot)
            }
            Motion::Ride { support, offset, yaw, drop, t0 } => {
                let (sp, sr) = world.props[*support].rest(w, world);
                let fall = (drop - 0.5 * G * mavian_core::m::powi((w - t0).max(0.0), 2)).max(0.0);
                (sp + sr * (*offset + Vec3::UP * fall), sr * Mat3::rot_y(*yaw))
            }
        }
    }
}

/// A prop in the scene.
#[derive(Clone, Debug)]
pub struct Prop {
    pub name: String,
    pub shape: Shape,
    /// where it stands at first: the middle of its bottom
    pub at: Vec3,
    /// turn about the vertical, degrees
    pub turn: f64,
    /// fill; none — the paper colour
    pub fill: Option<String>,
    pub shadow: bool,
    pub boil: bool,
    pub contacts: Vec<(f64, f64, usize, usize)>,
    /// (from when, segment) — made by actions
    pub motion: Vec<(f64, Motion)>,
}

impl Prop {
    pub fn new(name: &str, shape: Shape, at: Vec3) -> Prop {
        Prop { name: name.into(), shape, at, turn: 0.0, fill: None, shadow: true, boil: false, contacts: Vec::new(), motion: Vec::new() }
    }

    /// The segment in force at time w.
    pub fn segment(&self, w: f64) -> Option<&Motion> {
        self.motion.iter().rev().find(|(t0, _)| *t0 <= w + 1e-9).map(|(_, m)| m)
    }

    /// Where it is at time w: (anchor, rotation).
    pub fn rest(&self, w: f64, world: &World) -> (Vec3, Mat3) {
        match self.segment(w) {
            Some(m) => m.at(w, world),
            None => (self.at, Mat3::rot_y(self.turn)),
        }
    }

    pub fn top_point(&self, w: f64, world: &World) -> Vec3 {
        let (p, _) = self.rest(w, world);
        p + Vec3::UP * self.shape.top()
    }
}

/// A frame of line segments (axes, a wire cube): drawn, not solid.
#[derive(Clone, Debug)]
pub struct Wire {
    pub segments: Vec<(Vec3, Vec3)>,
    pub color: String,
    /// line width, px
    pub width: f64,
    /// Translation anchor; segments are local when attached.
    pub at: Option<crate::Point>,
}

/// A wire cube.
pub fn cube(center: Vec3, size: f64, color: &str) -> Wire {
    let h = size / 2.0;
    let c: Vec<Vec3> = (0..8)
        .map(|i| {
            let s = |bit: usize| if i & bit != 0 { 1.0 } else { -1.0 };
            center + Vec3 { x: s(4) * h, y: s(2) * h, z: s(1) * h }
        })
        .collect();
    let mut segments = Vec::new();
    for a in 0..8usize {
        for b in a + 1..8 {
            if (a ^ b).count_ones() == 1 {
                segments.push((c[a], c[b]));
            }
        }
    }
    Wire { segments, color: color.into(), width: 2.0, at: None }
}

/// Coordinate axes: X red, Y green (up), Z blue.
pub fn axes(length: f64) -> Vec<Wire> {
    let o = Vec3::ZERO;
    [
        ("#c0392b", Vec3 { x: length, y: 0.0, z: 0.0 }),
        ("#27ae60", Vec3 { x: 0.0, y: length, z: 0.0 }),
        ("#2c6fbb", Vec3 { x: 0.0, y: 0.0, z: length }),
    ]
    .into_iter()
    .map(|(c, e)| Wire { segments: vec![(o, e)], color: c.into(), width: 3.0, at: None })
    .collect()
}
