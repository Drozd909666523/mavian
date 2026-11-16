//! Effects: fire (a port of the MVP's `engine/effects.py`).
//!
//! Fire in anime is not a simulation but stylized tongues of flame: a few
//! layers of coloured shapes swayed by seeded noise. Everything is a pure
//! function of world time, without state between frames:
//!
//! - the flame's shape is noise of (time, seed, tongue number);
//! - flying-off tongues and sparks live in "slots": a slot has its own
//!   period; the cycle number floor((t - phase) / period) seeds this birth,
//!   the remainder is the age — so any particle can be computed from scratch;
//! - the wind is a key track that bends tongues, flying tongues and sparks.
//!
//! The flame is a picture facing the camera (a billboard) standing at a
//! point of the world; it is laid over the frame (it burns in front of what
//! it stands on), and its glow lights the ground under the props.

use crate::camera::Camera;
use mavian_core::noise::{noise1, rand_of, smooth};
use mavian_core::{Track, Vec3, m, seed};
use std::fmt::Write;

const TONGUES: usize = 5;
/// (share of width, share of height, colour, seed shift)
const LAYERS: [(f64, f64, &str, i64); 3] = [(1.00, 1.00, "#e2462b", 0), (0.74, 0.78, "#f28c28", 1), (0.44, 0.52, "#ffd34e", 2)];

/// A campfire: stylized flame, flying-off tongues, sparks, a glow on the ground.
#[derive(Clone, Debug)]
pub struct Fire {
    /// the flame's base in the world
    pub at: Vec3,
    pub height: f64,
    /// width at the base
    pub width: f64,
    pub seed: u64,
    /// sway speed, noise cycles per second
    pub rate: f64,
    /// + — to the right on screen; 1 — tongues bend by the flame's width
    pub wind: Track<f64>,
    /// slots for flying-off tongues
    pub puffs: usize,
    /// slots for sparks
    pub sparks: usize,
    pub glow: String,
    /// Depth lift towards the camera, in world units.
    pub over: f64,
}

impl Fire {
    pub fn new(at: Vec3) -> Fire {
        Fire { at, height: 0.9, width: 0.5, seed: 1, rate: 2.2, wind: Track::Const(0.0), puffs: 6, sparks: 14, over: 1.0, glow: "#ff9a3c".into() }
    }

    fn rand<'a>(&self, kind: &'a str, j: usize, rest: Vec<mavian_core::noise::Part<'a>>) -> f64 {
        let mut parts = vec![mavian_core::noise::Part::from(self.seed), kind.into(), (j as i64).into()];
        parts.extend(rest);
        rand_of(mavian_core::noise::seed_of(&parts))
    }

    /// The glow on the ground: (centre, radius, brightness 0..1) — it
    /// flickers with the flame.
    pub fn glow_at(&self, w: f64) -> (Vec3, f64, f64) {
        let k = 0.75 + 0.25 * noise1(seed!(self.seed, "glow"), w * self.rate * 1.5);
        (self.at, self.height * 1.6 * (0.92 + 0.08 * k), k)
    }

    /// The flame, flying tongues and sparks as SVG elements (screen px).
    pub fn svg(&self, cam: &Camera, w: f64) -> String {
        let Some(base) = cam.project(self.at) else { return String::new() };
        let (bx, by, s) = (base.x, base.y, base.scale);
        let wind = self.wind.at(w);
        let mut out = String::new();
        for (fw, fh, color, shift) in LAYERS {
            let d = self.flame(bx, by, s, w, fw, fh, seed!(self.seed, "layer", shift), wind);
            let _ = writeln!(out, r#"<path d="{d}" fill="{color}"/>"#);
        }
        self.puffs_svg(bx, by, s, w, wind, &mut out);
        self.sparks_svg(bx, by, s, w, wind, &mut out);
        out
    }

    /// A flame layer's outline: the bottom a half ellipse, the top sharp
    /// tongues and valleys.
    #[allow(clippy::too_many_arguments)]
    fn flame(&self, bx: f64, by: f64, s: f64, w: f64, fw: f64, fh: f64, seed: u64, wind: f64) -> String {
        let (ww, hh) = (self.width * fw * s, self.height * fh * s);
        // tongues: lower at the edges, higher in the middle; height and sway — noise
        let mut tips = Vec::new();
        for i in 0..TONGUES {
            let u = (i as f64 + 0.5) / TONGUES as f64;
            let mid = 1.0 - (2.0 * u - 1.0) * (2.0 * u - 1.0);
            let h = hh * (0.45 + 0.55 * mid) * (0.72 + 0.28 * noise1(seed!(seed, "h", i as i64), w * self.rate + i as f64 * 1.7));
            let sway = 0.16 * ww * noise1(seed!(seed, "s", i as i64), w * self.rate * 0.8 + i as f64);
            let r = h / hh.max(1e-6);
            let lean = wind * self.width * s * r * r;
            tips.push((bx + (u - 0.5) * ww + sway + lean, by - h));
        }
        let mut valleys = Vec::new();
        for i in 0..TONGUES - 1 {
            let u = (i as f64 + 1.0) / TONGUES as f64;
            let h = hh * (0.28 + 0.12 * noise1(seed!(seed, "v", i as i64), w * self.rate * 1.3 + i as f64));
            let r = h / hh.max(1e-6);
            let lean = wind * self.width * s * r * r * 0.6;
            valleys.push((bx + (u - 0.5) * ww + lean, by - h));
        }
        // around: bottom left → up the left edge → tongues and valleys → the right edge → bottom
        let (left, right) = ((bx - ww * 0.5, by - hh * 0.03), (bx + ww * 0.5, by - hh * 0.03));
        let mut d = format!("M{:.1} {:.1}", left.0, left.1);
        let mut prev = left;
        let mut seq = Vec::new();
        for (i, tip) in tips.iter().enumerate() {
            seq.push((true, *tip));
            if i < valleys.len() {
                seq.push((false, valleys[i]));
            }
        }
        for (tip, p) in seq {
            // to a tip — a curve bending outwards; into a valley — smoothly
            let out = if p.0 > bx { 1.0 } else { -1.0 };
            let cx = (prev.0 + p.0) / 2.0 + if tip { 0.08 * ww } else { 0.0 } * out;
            let cy = (prev.1 + p.1) / 2.0 + if tip { 0.12 * hh } else { -0.02 * hh };
            let _ = write!(d, " Q{cx:.1} {cy:.1} {:.1} {:.1}", p.0, p.1);
            prev = p;
        }
        let (cx, cy) = ((prev.0 + right.0) / 2.0 + 0.06 * ww, (prev.1 + right.1) / 2.0 + 0.1 * hh);
        let _ = write!(d, " Q{cx:.1} {cy:.1} {:.1} {:.1}", right.0, right.1);
        // the bottom — a half ellipse
        let _ = write!(d, " A{:.1} {:.1} 0 0 1 {:.1} {:.1} Z", ww * 0.5, hh * 0.05, left.0, left.1);
        d
    }

    /// Slot j's birth: (cycle number, age, life), or None if it is empty now.
    fn slot(&self, kind: &str, j: usize, w: f64, life_min: f64, life_max: f64) -> Option<(i64, f64, f64)> {
        let period = 0.5 + 0.9 * (self.rand(kind, j, vec!["p".into()]) * 0.5 + 0.5);
        let phase = period * (self.rand(kind, j, vec!["f".into()]) * 0.5 + 0.5);
        let cycle = ((w - phase) / period).floor();
        let age = (w - phase) - cycle * period;
        let cycle = cycle as i64;
        let life = life_min + (life_max - life_min) * (self.rand(kind, j, vec![cycle.into(), "l".into()]) * 0.5 + 0.5);
        (age <= life).then_some((cycle, age, life))
    }

    /// Flying-off tongues: they break off at the top, fly up and melt.
    fn puffs_svg(&self, bx: f64, by: f64, s: f64, w: f64, wind: f64, out: &mut String) {
        for j in 0..self.puffs {
            let Some((cycle, age, life)) = self.slot("puff", j, w, 0.35, 0.6) else { continue };
            let k = age / life;
            let x0 = self.rand("puff", j, vec![cycle.into(), "x".into()]) * 0.35 * self.width;
            let rise = self.height * (0.55 + 0.6 * k);
            let drift = wind * self.width * 1.4 * k + 0.08 * noise1(seed!(self.seed, j as i64, cycle), age * 3.0);
            let (x, y) = (bx + (x0 + drift) * s, by - rise * s);
            let size = self.height * 0.085 * m::pow(1.0 - k, 0.7) * s;
            if size < 0.6 {
                continue;
            }
            let color = if k < 0.5 { LAYERS[1].2 } else { LAYERS[0].2 };
            let _ = writeln!(out, r#"<path d="{}" fill="{color}"/>"#, drop(x, y, size, wind));
        }
    }

    /// Sparks: they fly up out of the flame, the wind and noise carry them, they fade.
    fn sparks_svg(&self, bx: f64, by: f64, s: f64, w: f64, wind: f64, out: &mut String) {
        for j in 0..self.sparks {
            let Some((cycle, age, life)) = self.slot("spark", j, w, 0.8, 1.5) else { continue };
            let k = age / life;
            let x0 = self.rand("spark", j, vec![cycle.into(), "x".into()]) * 0.3 * self.width;
            let vy = 0.6 + 0.6 * (self.rand("spark", j, vec![cycle.into(), "v".into()]) * 0.5 + 0.5);
            let sw = seed!(self.seed, "sw", j as i64, cycle);
            let x = x0 + 0.12 * noise1(sw, age * 2.5) + wind * 0.5 * age * age;
            let y = self.height * 0.5 + vy * age - 0.15 * age * age;
            // the tail — where the spark was a moment ago (the direction of flight)
            let a2 = (age - 0.06).max(0.0);
            let px = x0 + 0.12 * noise1(sw, a2 * 2.5) + wind * 0.5 * a2 * a2;
            let py = self.height * 0.5 + vy * a2 - 0.15 * a2 * a2;
            let r = (0.012 * s * (1.0 - k)).max(0.9);
            let fade = 1.0 - smooth((k - 0.6) / 0.4);
            let _ = writeln!(
                out,
                r##"<path d="M{:.1} {:.1} L{:.1} {:.1}" stroke="#ffd34e" stroke-width="{:.1}" stroke-linecap="round" opacity="{fade:.2}"/>"##,
                bx + px * s,
                by - py * s,
                bx + x * s,
                by - y * s,
                r * 1.4
            );
        }
    }
}

/// A tongue drop: a sharp tip up (with wind — aside), a round bottom.
fn drop(x: f64, y: f64, size: f64, wind: f64) -> String {
    let tip = (x + wind * size * 0.8, y - size * 1.6);
    format!(
        "M{:.1} {:.1} Q{:.1} {:.1} {:.1} {:.1} A{:.1} {:.1} 0 0 1 {:.1} {:.1} Q{:.1} {:.1} {:.1} {:.1}Z",
        tip.0,
        tip.1,
        x + size * 0.9,
        y - size * 0.4,
        x + size * 0.55,
        y + size * 0.2,
        size * 0.55,
        size * 0.55,
        x - size * 0.55,
        y + size * 0.2,
        x - size * 0.9,
        y - size * 0.4,
        tip.0,
        tip.1
    )
}
