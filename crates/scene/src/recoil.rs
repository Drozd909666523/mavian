//! A recovery step after a strike: move one foot, transfer support, follow
//! with the other. Anchors are world points, so sampling order is irrelevant.
use crate::Actor;
use mavian_core::{Vec3, noise::smooth};
use mavian_rig::{Figure, Side};

#[derive(Clone, Debug)]
pub(crate) struct FootStep {
    chain: [usize; 3],
    toe: usize,
    from: [Vec3; 2],
    to: [Vec3; 2],
}
#[derive(Clone, Debug)]
pub(crate) struct Recoil {
    pub start: f64,
    pub duration: f64,
    first: usize,
    pub airborne: bool,
    height: f64,
    gravity: f64,
    feet: [FootStep; 2],
}
impl Recoil {
    pub fn new(actor: &Actor, start: f64, duration: f64, delta: Vec3) -> Option<Self> {
        let rig = &actor.model.rig;
        let from = actor.figure(start);
        let to = actor.figure(start + duration);
        // An authored flight retains its trajectory and pose.
        if from.lift > 0.02 * actor.scale {
            return None;
        }
        let foot = |side| {
            let f = rig.foot(side)?;
            let knee = rig.parent(f.heel)?;
            let hip = rig.parent(knee)?;
            Some(FootStep {
                chain: [hip, knee, f.heel],
                toe: f.toe,
                from: [from.joints[f.heel], from.joints[f.toe]],
                to: [to.joints[f.heel] + delta, to.joints[f.toe] + delta],
            })
        };
        let feet = [foot(Side::L)?, foot(Side::R)?];
        // The foot furthest in the push direction catches the body first.
        let first = if feet[0].from[0].dot(delta) >= feet[1].from[0].dot(delta) { 0 } else { 1 };
        Some(Self {
            start,
            duration,
            first,
            gravity: crate::actor::G * actor.scale,
            airborne: delta.length() > 0.55 * actor.scale,
            height: (delta.length() * 0.3).clamp(0.025 * actor.scale, 0.13 * actor.scale),
            feet,
        })
    }
    pub fn active(&self, w: f64) -> bool {
        self.start <= w && w < self.start + self.duration
    }
    pub fn support(&self, w: f64) -> Option<Side> {
        if self.airborne {
            return None;
        }
        let i = if (w - self.start) / self.duration < 0.5 { 1 - self.first } else { self.first };
        Some(if i == 0 { Side::L } else { Side::R })
    }
    pub fn apply(&self, fig: &mut Figure, w: f64) {
        let phase = ((w - self.start) / self.duration).clamp(0.0, 1.0);
        if self.airborne {
            // A large impact is a flight, not an impossible one-metre planted
            // step. The authored landing pose remains in control.
            let h = 0.5 * self.gravity * self.duration * self.duration * phase * (1.0 - phase);
            for p in &mut fig.joints {
                p.y += h;
            }
            fig.lift += h;
            fig.pivot.y += h;
            return;
        }
        let goals: Vec<_> = self
            .feet
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let u = (phase * 2.0 - if i == self.first { 0.0 } else { 1.0 }).clamp(0.0, 1.0);
                let lift = 16.0 * u * u * (1.0 - u) * (1.0 - u) * self.height;
                [f.from[0].lerp(f.to[0], smooth(u)) + Vec3::UP * lift, f.from[1].lerp(f.to[1], smooth(u)) + Vec3::UP * lift]
            })
            .collect();
        // Bend the supporting knee rather than stretching a leg or dragging
        // its anchor when the pelvis has moved away from a straight leg.
        let mut lower = 0.0f64;
        for (f, g) in self.feet.iter().zip(&goals) {
            let [hip, knee, heel] = f.chain;
            let h = fig.joints[hip];
            let length = (fig.joints[knee] - h).length() + (fig.joints[heel] - fig.joints[knee]).length();
            let dx = h.x - g[0].x;
            let dz = h.z - g[0].z;
            let max_y = (length * length * 0.998001 - dx * dx - dz * dz).max(0.0).sqrt();
            lower = lower.max(h.y - g[0].y - max_y);
        }
        for p in &mut fig.joints {
            p.y -= lower;
        }
        for (f, g) in self.feet.iter().zip(&goals) {
            crate::ik::solve(&mut fig.joints, f.chain, g[0], 1.0, None);
            // Preserve the authored foot direction and its length; toe follows
            // the ankle correction, never the untranslated pose.
            fig.joints[f.toe] = fig.joints[f.chain[2]] + (g[1] - g[0]);
        }
    }
}
