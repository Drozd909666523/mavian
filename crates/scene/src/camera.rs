//! Camera: placement, projection, cuts, following, shake.
//!
//! A camera is a function of time: each camera kind builds a [`Camera`] for
//! one frame from the frame's clocks and the scene (where the characters are now).

use mavian_core::m;
use mavian_core::noise::noise1;
use mavian_core::{Clock, Keys, StableHash, StableHasher, Track, Vec3, seed};

/// What a camera can ask the scene.
pub trait Scene {
    /// Where joint of character `actor` is at world time `w` (as seen in the frame).
    fn joint(&self, actor: usize, joint: &str, w: f64) -> Vec3;
}

/// Camera for one frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Camera {
    pub pos: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub fwd: Vec3,
    /// focal length in pixels
    pub f: f64,
    pub cx: f64,
    pub cy: f64,
    pub fisheye: bool,
    pub near: f64,
}

/// Point on screen: x, y, depth, pixels per world unit at this point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Projected {
    pub x: f64,
    pub y: f64,
    pub depth: f64,
    pub scale: f64,
}

impl Camera {
    /// World point → screen; `None` if it is not visible.
    pub fn project(&self, p: Vec3) -> Option<Projected> {
        let d = p - self.pos;
        let (x, y, z) = (d.dot(self.right), d.dot(self.up), d.dot(self.fwd));
        if self.fisheye {
            // Equidistant fisheye: distance from the frame centre
            // is proportional to the angle from the camera axis.
            let rad = m::hypot(x, y);
            let dist = (rad * rad + z * z).sqrt();
            if dist < self.near {
                return None;
            }
            let theta = m::atan2(rad, z);
            // Drop everything far beyond the frame edge: otherwise lines behind
            // the camera stretch across the screen as false arcs.
            if self.f * theta > 1.15 * m::hypot(self.cx, self.cy) {
                return None;
            }
            let k = if rad > 1e-9 { self.f * theta / rad } else { self.f / dist };
            return Some(Projected { x: self.cx + x * k, y: self.cy - y * k, depth: dist, scale: self.f / dist });
        }
        if z < self.near {
            return None;
        }
        let k = self.f / z;
        Some(Projected { x: self.cx + x * k, y: self.cy - y * k, depth: z, scale: k })
    }

    /// Screen point and depth → world point: the inverse of `project`
    /// (depth is the distance along the axis, or from the camera for fisheye).
    pub fn unproject(&self, x: f64, y: f64, depth: f64) -> Vec3 {
        let (dx, dy) = (x - self.cx, self.cy - y);
        if self.fisheye {
            let r = m::hypot(dx, dy);
            if r < 1e-12 {
                return self.pos + self.fwd * depth;
            }
            let theta = r / self.f;
            let dir = self.fwd * m::cos(theta) + (self.right * dx + self.up * dy) * (m::sin(theta) / r);
            return self.pos + dir * depth;
        }
        self.pos + self.right * (dx * depth / self.f) + self.up * (dy * depth / self.f) + self.fwd * depth
    }
}

impl StableHash for Camera {
    fn stable_hash(&self, h: &mut StableHasher) {
        h.add(&self.pos).add(&self.right).add(&self.up).add(&self.fwd);
        h.f64(self.f).f64(self.cx).f64(self.cy).add(&self.fisheye).f64(self.near);
    }
}

/// A point that follows a character's joint.
#[derive(Clone, Debug)]
pub struct Follow {
    pub actor: usize,
    pub joint: String,
    pub offset: Vec3,
    /// lag, world seconds
    pub lag: f64,
    /// smoothing: average position over a window of this many seconds — the camera
    /// glides instead of bobbing with every step
    pub smooth: f64,
}

impl Follow {
    pub fn new(actor: usize, joint: &str) -> Follow {
        Follow { actor, joint: joint.into(), offset: Vec3::ZERO, lag: 0.0, smooth: 0.0 }
    }

    pub fn offset(mut self, o: (f64, f64, f64)) -> Self {
        self.offset = o.into();
        self
    }

    pub fn lag(mut self, s: f64) -> Self {
        self.lag = s;
        self
    }

    pub fn smooth(mut self, s: f64) -> Self {
        self.smooth = s;
        self
    }
}

/// A point between joints of several characters (the middle of a fight).
#[derive(Clone, Debug)]
pub struct Between {
    pub actors: Vec<usize>,
    pub joint: String,
    pub offset: Track<Vec3>,
    pub smooth: f64,
}

const SMOOTH_N: usize = 9;

impl Between {
    fn at(&self, scene: &dyn Scene, w: f64) -> Vec3 {
        let off = self.offset.at(w);
        let mid = |t: f64| {
            let sum = self.actors.iter().fold(Vec3::ZERO, |s, &a| s + scene.joint(a, &self.joint, t));
            sum * (1.0 / self.actors.len() as f64)
        };
        if self.smooth > 0.0 {
            let n = SMOOTH_N as f64;
            let sum = (0..SMOOTH_N).fold(Vec3::ZERO, |s, k| s + mid(w + self.smooth * (k as f64 / (n - 1.0) - 0.5)));
            return sum * (1.0 / n) + off;
        }
        mid(w) + off
    }
}

/// A point for a camera: a place (value or track), a follow, or a midpoint.
#[derive(Clone, Debug)]
pub enum Point {
    Fixed(Track<Vec3>),
    Follow(Follow),
    Between(Between),
}

impl Point {
    pub fn resolve(&self, clock: &Clock, scene: &dyn Scene) -> Vec3 {
        match self {
            Point::Fixed(t) => t.value(clock),
            Point::Between(b) => b.at(scene, clock.world),
            Point::Follow(f) => {
                let w = clock.world - f.lag;
                if f.smooth > 0.0 {
                    let n = SMOOTH_N as f64;
                    let at = |k: usize| scene.joint(f.actor, &f.joint, w + f.smooth * (k as f64 / (n - 1.0) - 0.5));
                    let mut p = at(0) * (1.0 / n);
                    for k in 1..SMOOTH_N {
                        p = p + at(k) * (1.0 / n);
                    }
                    return p + f.offset;
                }
                scene.joint(f.actor, &f.joint, w) + f.offset
            }
        }
    }

    pub(crate) fn follows(&self) -> Vec<(usize, &str)> {
        match self {
            Point::Fixed(_) => vec![],
            Point::Follow(f) => vec![(f.actor, f.joint.as_str())],
            Point::Between(b) => b.actors.iter().map(|&a| (a, b.joint.as_str())).collect(),
        }
    }
}

impl From<(f64, f64, f64)> for Point {
    fn from(v: (f64, f64, f64)) -> Self {
        Point::Fixed(Track::Const(v.into()))
    }
}

impl From<Vec3> for Point {
    fn from(v: Vec3) -> Self {
        Point::Fixed(Track::Const(v))
    }
}

impl From<Keys<Vec3>> for Point {
    fn from(k: Keys<Vec3>) -> Self {
        Point::Fixed(Track::Keys(k))
    }
}

impl From<Follow> for Point {
    fn from(f: Follow) -> Self {
        Point::Follow(f)
    }
}

impl From<Between> for Point {
    fn from(b: Between) -> Self {
        Point::Between(b)
    }
}

fn lens(fov: f64, h: f64, fisheye: bool) -> f64 {
    let half = fov.to_radians() / 2.0;
    if fisheye { (h / 2.0) / half } else { (h / 2.0) / m::tan(half) }
}

/// Camera on a circle around the target. angle: 0 — camera in front of the character,
/// plus — moves towards the character's left side.
#[derive(Clone, Debug)]
pub struct Orbit {
    pub radius: Track<f64>,
    pub height: Track<f64>,
    pub angle: Track<f64>,
    pub target: Point,
    /// vertical field of view, degrees
    pub fov: Track<f64>,
    pub fisheye: bool,
}

impl Default for Orbit {
    fn default() -> Self {
        Orbit { radius: 6.0.into(), height: 1.5.into(), angle: 0.0.into(), target: (0.0, 1.1, 0.0).into(), fov: 32.0.into(), fisheye: false }
    }
}

/// Camera at `pos` looking at `target`.
///
/// `turn` — where the camera body faces around the vertical (0 — towards +Z, 180 —
/// towards −Z). It only sets the horizon tilt; the camera still looks at the target.
/// Needed when the target passes right over the camera: a plain camera flips
/// abruptly there, while with turn the flip is spread over time.
#[derive(Clone, Debug)]
pub struct Watch {
    pub pos: Point,
    pub target: Point,
    pub fov: Track<f64>,
    pub fisheye: bool,
    pub turn: Option<Track<f64>>,
    /// horizon tilt, degrees (Dutch angle)
    pub roll: Track<f64>,
}

impl Watch {
    pub fn new(pos: impl Into<Point>, target: impl Into<Point>) -> Watch {
        Watch { pos: pos.into(), target: target.into(), fov: 32.0.into(), fisheye: false, turn: None, roll: 0.0.into() }
    }

    pub fn fov(mut self, fov: impl Into<Track<f64>>) -> Self {
        self.fov = fov.into();
        self
    }

    pub fn fisheye(mut self) -> Self {
        self.fisheye = true;
        self
    }

    pub fn turn(mut self, t: impl Into<Track<f64>>) -> Self {
        self.turn = Some(t.into());
        self
    }

    pub fn roll(mut self, r: impl Into<Track<f64>>) -> Self {
        self.roll = r.into();
        self
    }
}

/// Any shot camera.
#[derive(Clone, Debug)]
pub enum CameraRig {
    Orbit(Orbit),
    Watch(Watch),
    /// cuts: (world time, camera); one action, cameras take turns
    Cuts(Vec<(f64, CameraRig)>),
}

impl Default for CameraRig {
    fn default() -> Self {
        CameraRig::Orbit(Orbit::default())
    }
}

impl From<Watch> for CameraRig {
    fn from(w: Watch) -> Self {
        CameraRig::Watch(w)
    }
}

impl From<Orbit> for CameraRig {
    fn from(o: Orbit) -> Self {
        CameraRig::Orbit(o)
    }
}

impl CameraRig {
    /// Cuts from (world time, camera) pairs.
    pub fn cuts(table: impl IntoIterator<Item = (f64, CameraRig)>) -> CameraRig {
        let mut items: Vec<_> = table.into_iter().collect();
        items.sort_by(|a, b| a.0.total_cmp(&b.0));
        assert!(!items.is_empty(), "no cameras");
        CameraRig::Cuts(items)
    }

    /// Which camera is shooting at this moment (for cuts — the camera index).
    pub fn pick(&self, clock: &Clock) -> (usize, &CameraRig) {
        match self {
            CameraRig::Cuts(items) => {
                let mut pick = (0, &items[0].1);
                for (i, (t, c)) in items.iter().enumerate() {
                    if clock.world >= t - 1e-9 {
                        pick = (i, c);
                    }
                }
                let (i, c) = pick;
                (i, c)
            }
            other => (0, other),
        }
    }

    pub fn frame(&self, clock: &Clock, size: (u32, u32), shift: Vec3, scene: &dyn Scene) -> Camera {
        let (w, h) = (size.0 as f64, size.1 as f64);
        match self {
            CameraRig::Cuts(_) => self.pick(clock).1.frame(clock, size, shift, scene),
            CameraRig::Orbit(o) => {
                let ang = o.angle.value(clock).to_radians();
                let rad = o.radius.value(clock);
                let tgt = o.target.resolve(clock, scene);
                let pos = Vec3 { x: tgt.x + rad * m::sin(ang), y: o.height.value(clock), z: tgt.z + rad * m::cos(ang) };
                let (pos, tgt) = (pos + shift, tgt + shift);
                let fwd = (tgt - pos).norm();
                let right = fwd.cross(Vec3::UP).norm();
                let up = right.cross(fwd);
                let f = lens(o.fov.value(clock), h, o.fisheye);
                Camera { pos, right, up, fwd, f, cx: w / 2.0, cy: h / 2.0, fisheye: o.fisheye, near: 0.05 }
            }
            CameraRig::Watch(c) => {
                let pos = c.pos.resolve(clock, scene) + shift;
                let tgt = c.target.resolve(clock, scene) + shift;
                let fwd = (tgt - pos).norm();
                let level = fwd.cross(Vec3::UP);
                let mut reference = match &c.turn {
                    None => level,
                    Some(turn) => {
                        let a = turn.value(clock).to_radians();
                        // the camera body's "right"
                        let side = Vec3 { x: -m::cos(a), y: 0.0, z: m::sin(a) };
                        let r = side - fwd * side.dot(fwd);
                        if r.length() < 1e-6 { level } else { r }
                    }
                };
                if reference.length() < 1e-6 {
                    reference = Vec3 { x: 1.0, y: 0.0, z: 0.0 };
                }
                let mut right = reference.norm();
                let mut up = right.cross(fwd);
                let roll = c.roll.value(clock);
                if roll != 0.0 {
                    let (s, co) = m::sin_cos(roll.to_radians());
                    (right, up) = (right * co + up * s, up * co - right * s);
                }
                let f = lens(c.fov.value(clock), h, c.fisheye);
                Camera { pos, right, up, fwd, f, cx: w / 2.0, cy: h / 2.0, fisheye: c.fisheye, near: 0.05 }
            }
        }
    }

    /// All followed points (for checking the description).
    pub(crate) fn follows(&self) -> Vec<(usize, &str)> {
        match self {
            CameraRig::Orbit(o) => o.target.follows(),
            CameraRig::Watch(w) => [w.pos.follows(), w.target.follows()].concat(),
            CameraRig::Cuts(items) => items.iter().flat_map(|(_, c)| c.follows()).collect(),
        }
    }
}

/// Camera shake from a hit.
#[derive(Clone, Debug)]
pub struct Shake {
    /// moment of the hit in world time
    pub at: f64,
    /// amplitude in world units
    pub strength: f64,
    /// shot seconds for the shake to decay by a factor of e
    pub decay: f64,
    pub freq: f64,
}

impl Shake {
    pub fn new(at: f64, strength: f64, decay: f64) -> Shake {
        Shake { at, strength, decay, freq: 16.0 }
    }

    /// Camera offset (right, up) tau shot seconds after the hit.
    pub fn offset(&self, tau: f64, seed: u64) -> (f64, f64) {
        if tau < 0.0 {
            return (0.0, 0.0);
        }
        let amp = self.strength * m::exp(-tau / self.decay);
        if amp < 1e-4 {
            return (0.0, 0.0);
        }
        (amp * noise1(seed!(seed, "shake-x"), tau * self.freq), amp * noise1(seed!(seed, "shake-y"), tau * self.freq))
    }
}
