//! Keys, easing and three times.
//!
//! - World time — how much has passed inside the scene. Poses and paths live in it.
//! - Shot time — what the viewer sees: frame number divided by fps.
//! - Drawing time — which frames the character is redrawn on ("on twos").
//!
//! Shot time and world time are linked by `time_warp` — the speed of the world.

use crate::math::{Vec3, mix};

/// Easing into a key from the previous one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Ease {
    Linear,
    /// accelerate
    In,
    /// decelerate
    Out,
    /// accelerate and decelerate
    #[default]
    InOut,
    /// hold the previous value right up to the key
    Hold,
}

impl Ease {
    pub fn apply(self, u: f64) -> f64 {
        match self {
            Ease::Linear => u,
            Ease::In => u * u,
            Ease::Out => 1.0 - (1.0 - u) * (1.0 - u),
            Ease::InOut => u * u * (3.0 - 2.0 * u),
            Ease::Hold => 0.0,
        }
    }
}

/// Which clock a track is read by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ClockKind {
    #[default]
    World,
    Shot,
    Progress,
}

/// The three clock readings for one frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Clock {
    /// world seconds
    pub world: f64,
    /// shot seconds
    pub shot: f64,
    /// fraction of the shot, 0..1
    pub progress: f64,
}

impl Clock {
    pub fn read(&self, kind: ClockKind) -> f64 {
        match kind {
            ClockKind::World => self.world,
            ClockKind::Shot => self.shot,
            ClockKind::Progress => self.progress,
        }
    }
}

/// Something that can be blended between keys.
pub trait Blend: Clone {
    fn blend(&self, b: &Self, u: f64) -> Self;
}

impl Blend for f64 {
    fn blend(&self, b: &f64, u: f64) -> f64 {
        mix(*self, *b, u)
    }
}

impl Blend for Vec3 {
    fn blend(&self, b: &Vec3, u: f64) -> Vec3 {
        self.lerp(*b, u)
    }
}

impl Blend for (f64, f64) {
    fn blend(&self, b: &(f64, f64), u: f64) -> (f64, f64) {
        (mix(self.0, b.0, u), mix(self.1, b.1, u))
    }
}

/// A key track. Before the first key and after the last the value holds.
#[derive(Clone, Debug)]
pub struct Keys<T> {
    items: Vec<(f64, T, Ease)>,
    ease: Ease,
    pub clock: ClockKind,
}

impl<T: Clone> Keys<T> {
    /// Empty track; `ease` — easing for keys that do not specify one.
    pub fn new(ease: Ease) -> Self {
        Self { items: Vec::new(), ease, clock: ClockKind::World }
    }

    /// Track from (time, value) pairs with the default easing.
    pub fn of(table: impl IntoIterator<Item = (f64, T)>) -> Self {
        let mut k = Self::new(Ease::InOut);
        for (t, v) in table {
            k = k.key(t, v);
        }
        k
    }

    pub fn ease(mut self, ease: Ease) -> Self {
        for it in &mut self.items {
            if it.2 == self.ease {
                it.2 = ease;
            }
        }
        self.ease = ease;
        self
    }

    pub fn clock(mut self, clock: ClockKind) -> Self {
        self.clock = clock;
        self
    }

    pub fn key(self, t: f64, v: T) -> Self {
        let e = self.ease;
        self.eased(t, v, e)
    }

    /// Key with its own easing. A key at the same time replaces the old one.
    pub fn eased(mut self, t: f64, v: T, e: Ease) -> Self {
        match self.items.binary_search_by(|it| it.0.total_cmp(&t)) {
            Ok(i) => self.items[i] = (t, v, e),
            Err(i) => self.items.insert(i, (t, v, e)),
        }
        self
    }

    /// Add ready-made keys, e.g. walk steps: `.extend(walk(...))`.
    pub fn extend(mut self, items: impl IntoIterator<Item = (f64, T, Ease)>) -> Self {
        for (t, v, e) in items {
            self = self.eased(t, v, e);
        }
        self
    }

    pub fn items(&self) -> &[(f64, T, Ease)] {
        &self.items
    }

    /// Indices of the two neighbouring keys and the raw (uneased) fraction between them.
    pub fn locate(&self, t: f64) -> (usize, usize, f64) {
        let items = &self.items;
        assert!(!items.is_empty(), "empty key track");
        let last = items.len() - 1;
        if t <= items[0].0 {
            return (0, 0, 0.0);
        }
        if t >= items[last].0 {
            return (last, last, 0.0);
        }
        // first key strictly after t
        let j = items.partition_point(|it| it.0 <= t);
        let (t0, t1) = (items[j - 1].0, items[j].0);
        (j - 1, j, (t - t0) / (t1 - t0))
    }

    /// The two neighbouring values and the transition fraction (already eased).
    pub fn span(&self, t: f64) -> (&T, &T, f64) {
        let (i, j, raw) = self.locate(t);
        (&self.items[i].1, &self.items[j].1, self.items[j].2.apply(raw))
    }
}

impl<T: Blend> Keys<T> {
    pub fn at(&self, t: f64) -> T {
        let (a, b, u) = self.span(t);
        a.blend(b, u)
    }
}

/// A constant value or a key track.
#[derive(Clone, Debug)]
pub enum Track<T> {
    Const(T),
    Keys(Keys<T>),
}

impl<T: Blend> Track<T> {
    /// Value at world time `w` (the track's clock is ignored).
    pub fn at(&self, w: f64) -> T {
        match self {
            Track::Const(v) => v.clone(),
            Track::Keys(k) => k.at(w),
        }
    }

    /// Value by the track's own clock.
    pub fn value(&self, clock: &Clock) -> T {
        match self {
            Track::Const(v) => v.clone(),
            Track::Keys(k) => k.at(clock.read(k.clock)),
        }
    }
}

impl<T> From<Keys<T>> for Track<T> {
    fn from(k: Keys<T>) -> Self {
        Track::Keys(k)
    }
}

impl From<f64> for Track<f64> {
    fn from(v: f64) -> Self {
        Track::Const(v)
    }
}

impl From<Vec3> for Track<Vec3> {
    fn from(v: Vec3) -> Self {
        Track::Const(v)
    }
}

impl From<(f64, f64, f64)> for Track<Vec3> {
    fn from(v: (f64, f64, f64)) -> Self {
        Track::Const(v.into())
    }
}

impl From<(f64, f64)> for Track<(f64, f64)> {
    fn from(v: (f64, f64)) -> Self {
        Track::Const(v)
    }
}

/// The "frame → world time" table for the whole shot.
#[derive(Clone, Debug)]
pub struct Timeline {
    pub fps: u32,
    pub world: Vec<f64>,
}

impl Timeline {
    pub const MAX_FRAMES: usize = 20_000;
    const SUB: usize = 8;

    /// `speed(w)` — world speed at world time w.
    pub fn new(fps: u32, length: f64, speed: impl Fn(f64) -> f64) -> Self {
        let dt = 1.0 / fps as f64;
        let mut world = Vec::new();
        let mut w = 0.0;
        while w < length - 1e-9 && world.len() < Self::MAX_FRAMES {
            world.push(w);
            for _ in 0..Self::SUB {
                w += speed(w).max(0.01) * dt / Self::SUB as f64;
            }
        }
        if world.is_empty() {
            world.push(0.0);
        }
        Self { fps, world }
    }

    pub fn len(&self) -> usize {
        self.world.len()
    }

    pub fn is_empty(&self) -> bool {
        self.world.is_empty()
    }

    pub fn clock(&self, n: usize) -> Clock {
        let last = (self.world.len() - 1).max(1);
        Clock { world: self.world[n], shot: n as f64 / self.fps as f64, progress: n as f64 / last as f64 }
    }

    /// At which shot time world time reaches w.
    pub fn shot_time_of(&self, w: f64) -> f64 {
        let world = &self.world;
        for n in 1..world.len() {
            if world[n] >= w {
                let (a, b) = (world[n - 1], world[n]);
                let u = if b > a { (w - a) / (b - a) } else { 0.0 };
                return (n as f64 - 1.0 + u) / self.fps as f64;
            }
        }
        world.len() as f64 / self.fps as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_hold_before_first_and_after_last() {
        let k = Keys::of([(1.0, 10.0), (2.0, 20.0)]).ease(Ease::Linear);
        assert_eq!(k.at(0.0), 10.0);
        assert_eq!(k.at(1.5), 15.0);
        assert_eq!(k.at(3.0), 20.0);
    }

    #[test]
    fn hold_ease_keeps_previous() {
        let k = Keys::of([(0.0, 2.0)]).eased(1.0, 1.0, Ease::Hold);
        assert_eq!(k.at(0.99), 2.0);
        assert_eq!(k.at(1.0), 1.0);
    }

    #[test]
    fn timeline_slow_motion_stretches_world() {
        let tl = Timeline::new(24, 1.0, |_| 0.5);
        assert_eq!(tl.len(), 48);
    }
}
