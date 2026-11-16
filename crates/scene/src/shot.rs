//! Shot: the scene description and the state of one frame.
//!
//! A frame is a pure function of (shot description, frame number). Nothing is kept
//! between frames, so any frame can be computed alone and again;
//! caches only speed things up and never affect the result.

use crate::actions::Action;
use crate::actor::Actor;
use crate::camera::{Camera, CameraRig, Scene, Shake};
use crate::props::{Prop, Wire, World};
use crate::style::Style;
use mavian_core::{Clock, Mat3, Timeline, Track, Vec3, seed};
use mavian_rig::{Figure, ModelError};
use std::fmt;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ShotError(pub String);

impl fmt::Display for ShotError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ShotError {}

impl From<ModelError> for ShotError {
    fn from(e: ModelError) -> Self {
        ShotError(e.0)
    }
}

/// Ground: grid and shadows.
#[derive(Clone, Debug)]
pub struct Ground {
    pub size: f64,
    pub step: f64,
    /// grid line width, px
    pub width: f64,
    /// the background is redrawn less often than characters
    pub draw_on: u32,
    pub shadow: bool,
    /// grid centre, (x, z)
    pub center: (f64, f64),
    /// grid boil, fraction of Style.boil (0 — still background)
    pub boil: f64,
    /// draw the grid at all (without it — clean background, shadows only)
    pub grid: bool,
}

impl Default for Ground {
    fn default() -> Self {
        Ground { size: 4.0, step: 1.0, width: 1.4, draw_on: 3, shadow: true, center: (0.0, 0.0), boil: 0.5, grid: true }
    }
}

/// Text in the frame at a point of the world.
#[derive(Clone, Debug)]
pub struct Label {
    pub text: String,
    pub at: Vec3,
    /// font size, px
    pub size: f64,
}

impl Label {
    /// The label as an SVG `<text>` element, or None if its point is not seen.
    pub fn svg(&self, cam: &Camera, ink: &str) -> Option<String> {
        let r = cam.project(self.at)?;
        let text = self.text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");
        Some(format!(
            r#"<text x="{:.2}" y="{:.2}" font-family="DejaVu Sans, sans-serif" font-size="{}" text-anchor="middle" fill="{ink}" stroke="none">{text}</text>"#,
            r.x, r.y, self.size
        ))
    }
}

/// Shot description.
pub struct Shot {
    pub name: String,
    pub camera: CameraRig,
    pub actors: Vec<Actor>,
    pub fps: u32,
    /// length in world time, seconds
    pub length: f64,
    pub seed: u64,
    pub size: (u32, u32),
    /// characters are redrawn every this many frames
    pub draw_on: Track<f64>,
    /// world speed
    pub time_warp: Track<f64>,
    pub shakes: Vec<Shake>,
    pub ground: Option<Ground>,
    pub style: Style,
    pub labels: Vec<Label>,
    pub props: Vec<Prop>,
    pub wires: Vec<Wire>,
    /// effects: drawn over the frame, their glow on the ground
    pub fires: Vec<crate::effects::Fire>,
    /// intent: grab, put down, push, strike — `compile` turns them into hand
    /// targets, layers, prop motion and the below
    pub actions: Vec<Action>,
    /// Deterministic acquisition decisions, including failed attempts.
    pub grab_reports: Vec<crate::GrabReport>,
    /// made by strikes: hit-stops (when, how long in shot seconds), impact
    /// frames (when), camera shakes
    pub stops: Vec<(f64, f64)>,
    pub flashes: Vec<f64>,
    pub auto_shakes: Vec<Shake>,
}

/// the world's speed in a hit-stop
pub const HIT_SPEED: f64 = 0.04;
/// how many frames an impact frame lasts (colours inverted)
pub const FLASH: f64 = 2.0;

impl Default for Shot {
    fn default() -> Self {
        Shot {
            name: "shot".into(),
            camera: CameraRig::default(),
            actors: Vec::new(),
            fps: 24,
            length: 3.0,
            seed: 1,
            size: (1280, 720),
            draw_on: 2.0.into(),
            time_warp: 1.0.into(),
            shakes: Vec::new(),
            ground: None,
            style: Style::default(),
            labels: Vec::new(),
            props: Vec::new(),
            wires: Vec::new(),
            fires: Vec::new(),
            actions: Vec::new(),
            grab_reports: Vec::new(),
            stops: Vec::new(),
            flashes: Vec::new(),
            auto_shakes: Vec::new(),
        }
    }
}

/// Everything known about frame n: clocks, camera, figures.
pub struct FrameState {
    pub n: usize,
    /// frame in which the characters were last drawn (drawing time)
    pub drawn: usize,
    /// world time of this drawing
    pub w: f64,
    pub clock: Clock,
    pub cam: Camera,
    /// camera index (for cuts)
    pub cut: usize,
    /// actor figures after squash and stretch
    pub figures: Vec<Arc<Figure>>,
    /// an impact frame: paper and ink swap
    pub flash: bool,
}

impl Shot {
    /// Check the description before the first frame: poses, planting, followed joints.
    pub fn validate(&self) -> Result<(), ShotError> {
        for a in &self.actors {
            a.validate()?;
            for c in a.contacts.iter().chain(&a.auto_contacts) {
                crate::contact::validate_target(&c.target, &self.world()).map_err(ShotError)?;
            }
        }
        let mut edges = vec![0.0, self.length];
        for a in &self.actors {
            for c in a.contacts.iter().chain(&a.auto_contacts) {
                edges.extend([c.from, c.to]);
                if let mavian_core::Track::Keys(k) = &c.weight {
                    edges.extend(k.items().iter().map(|(w, _, _)| *w));
                }
            }
        }
        for p in &self.props {
            edges.extend(p.motion.iter().map(|m| m.0));
        }
        edges.sort_by(f64::total_cmp);
        edges.dedup();
        for pair in edges.windows(2) {
            crate::contact::validate_dependencies((pair[0] + pair[1]) * 0.5, &self.world()).map_err(ShotError)?;
        }
        for (i, joint) in self.camera.follows() {
            let a = self.actors.get(i).ok_or_else(|| ShotError(format!("camera follows actor {i}, which does not exist")))?;
            a.model.rig.joint(joint).map_err(|e| ShotError(format!("camera: {e}")))?;
        }
        for wire in &self.wires {
            if !wire.width.is_finite() || wire.width <= 0.0 {
                return Err(ShotError("wire width must be positive and finite".into()));
            }
            let follows: Vec<(usize, &str)> = match &wire.at {
                Some(crate::Point::Follow(p)) => vec![(p.actor, &p.joint)],
                Some(crate::Point::Between(p)) => p.actors.iter().map(|&i| (i, p.joint.as_str())).collect(),
                _ => Vec::new(),
            };
            for (i, joint) in follows {
                let a = self.actors.get(i).ok_or_else(|| ShotError(format!("wire follows actor {i}, which does not exist")))?;
                a.model.rig.joint(joint).map_err(|e| ShotError(format!("wire: {e}")))?;
            }
        }
        if !self.actions.is_empty()
            && self.props.iter().all(|p| p.motion.is_empty())
            && self.actors.iter().all(|a| a.reaches.is_empty() && a.auto_contacts.is_empty())
        {
            return Err(ShotError("the shot has actions but was not compiled: call Shot::compile".into()));
        }
        Ok(())
    }

    pub fn timeline(&self) -> Timeline {
        Timeline::new(self.fps, self.length, |w| {
            // hit-stops: the world almost stops for `stop` shot seconds
            let mut k = self.time_warp.at(w);
            for &(t, stop) in &self.stops {
                if t <= w && w < t + stop * HIT_SPEED {
                    k *= HIT_SPEED;
                }
            }
            k
        })
    }

    /// An impact frame: the first `FLASH` frames after a strong hit.
    pub fn is_flash(&self, tl: &Timeline, n: usize) -> bool {
        self.flashes.iter().any(|&t| {
            let first = tl.shot_time_of(t) * self.fps as f64;
            first - 1e-6 <= n as f64 && (n as f64) < first + FLASH
        })
    }

    /// Drawing time: the frame in which the character was last drawn.
    pub fn drawn_frame(&self, tl: &Timeline, n: usize) -> usize {
        let step = self.draw_on.at(tl.world[n]).round_ties_even().max(1.0) as usize;
        n - n % step
    }

    /// The shot's world: what hand targets and prop motion see.
    pub fn world(&self) -> World<'_> {
        World { actors: &self.actors, props: &self.props }
    }

    /// Turn the actions into hand targets and prop motion (again from
    /// scratch, so it can be called after any change of the description).
    pub fn compile(&mut self) -> Result<(), ShotError> {
        let out = crate::actions::compile_shot(self).map_err(ShotError)?;
        (self.stops, self.flashes, self.auto_shakes) = out;
        Ok(())
    }

    /// Actor figure with squash and stretch, and the matrix itself.
    pub(crate) fn deformed(&self, index: usize, w: f64) -> (Arc<Figure>, Option<Mat3>) {
        let actor = &self.actors[index];
        // Velocity sampling must not anticipate a future knockback and deform
        // the target away from a limb while their contact is still held.
        let stretch_time = self
            .actions
            .iter()
            .find_map(|a| match a {
                Action::Strike(s) if s.target == index && !s.miss && w >= s.at && w <= s.at + s.hold + 0.12 => {
                    let blend = mavian_core::noise::smooth((w - s.at - s.hold) / 0.12);
                    Some(s.at + (w - s.at) * blend)
                }
                _ => None,
            })
            .unwrap_or(w);
        let m = actor.stretch(stretch_time);
        let fig = actor.figure_ik(w, &self.world());
        match m {
            None => (fig, None),
            Some(m) => {
                let mut out = fig.stretched_about(Some(m), actor.stretch_pivot(&fig));
                actor.pin_feet(&fig, &mut out);
                (Arc::new(out), Some(m))
            }
        }
    }

    /// Contact constraints run after squash/stretch: the rendered limb must
    /// reach the rendered opponent, and rendered bodies must remain separate.
    pub fn posed(&self, index: usize, w: f64) -> (Arc<Figure>, Option<Mat3>) {
        let (fig, m) = self.deformed(index, w);
        if self.actors[index].pose_lock {
            return (fig, m);
        }
        if !self
            .actions
            .iter()
            .any(|a| matches!(a,crate::Action::Strike(s) if s.attacker==index && w>=s.at-s.windup-0.12 && w<=s.at+s.hold+s.recover))
        {
            let actor = &self.actors[index];
            if actor.contacts.iter().chain(&actor.auto_contacts).any(|c| c.weight_at(w) > 0.0) {
                let solved = self.solve_actor_contacts(index, w, &fig, m).0;
                return (Arc::new(self.avoid_props(index, w, &solved)), m);
            }
            if !self.props.is_empty() {
                return (Arc::new(self.avoid_props(index, w, &fig)), m);
            }
            return (fig, m);
        }
        let mut out = (*fig).clone();
        for action in &self.actions {
            let crate::Action::Strike(s) = action else { continue };
            if s.attacker != index || w < s.at - s.windup - 0.12 || w > s.at + s.hold + s.recover {
                continue;
            }
            let defender = &self.actors[s.target];
            let actor = &self.actors[index];
            let target = self.deformed(s.target, w).0;
            let joint = |a: &Actor, name: &str| a.model.rig.joint(name).ok();
            let (Some(ha), Some(hd), Some(ca), Some(cd)) =
                (joint(actor, "head"), joint(defender, "head"), joint(actor, "chest"), joint(defender, "chest"))
            else {
                continue;
            };
            let away = crate::ik::flat_dir(target.joints[defender.model.root], out.joints[actor.model.root]);
            if s.aim == "chin"
                && let Some(neck) = actor.model.rig.parent(ha)
            {
                // Keep the uppercut's head behind its shoulder instead of
                // leading with the forehead into the opponent's guard.
                let up = mavian_core::noise::smooth((w - (s.at - s.windup)) / s.windup.max(1e-6));
                let down = 1.0 - mavian_core::noise::smooth((w - s.at - s.hold) / s.recover.max(1e-6));
                let heading = actor.facing.at(w).to_radians();
                let back = Vec3 { x: -mavian_core::m::sin(heading), y: 0.0, z: -mavian_core::m::cos(heading) };
                let axis = out.joints[ha] - out.joints[neck];
                out.joints[ha] = out.joints[neck] + (axis + back * (0.15 * actor.scale * up * down)).norm() * axis.length();
            }
            let gap = |shift: f64| {
                let offset = away * shift;
                let head = (out.joints[ha] + offset - target.joints[hd]).dot(away) - 0.48 * actor.scale.max(defender.scale);
                let body = (out.joints[ca] + offset - target.joints[cd]).length() - 0.32 * actor.scale.max(defender.scale);
                head.min(body)
            };
            if gap(0.0) < 0.0 {
                let (mut lo, mut hi) = (0.0, 2.0 * actor.scale);
                for _ in 0..24 {
                    let mid = (lo + hi) * 0.5;
                    if gap(mid) < 0.0 {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                for p in &mut out.joints {
                    *p = *p + away * hi;
                }
                out.pivot = out.pivot + away * hi;
            }
            if s.miss {
                continue;
            }
            let Ok(chain) = crate::actions::chain(actor, s.limb) else { continue };
            let Some(reach) = actor.reaches.iter().find(|r| r.chain == chain && r.weight.at(w) > 0.0) else { continue };
            let toward = crate::ik::flat_dir(target.joints[defender.model.root], out.joints[actor.model.root]);
            let goal = if let Some(hand) = s.blocked {
                target.joints[defender.model.rig.parent(hand).expect("blocking arm")].lerp(target.joints[hand], 0.55) + toward * 0.10
            } else {
                let aim = if s.aim == "chin" { "head" } else { &s.aim };
                let Some(j) = joint(defender, aim) else { continue };
                let offset = if s.aim == "chin" {
                    (Vec3 { x: 0.0, y: -0.17, z: 0.0 } + crate::ik::flat_dir(target.joints[j], out.joints[chain[0]]) * 0.09).norm()
                        * crate::ik::CHIN_REACH
                } else {
                    toward * {
                        // Stick silhouettes have a thinner torso than the
                        // mannequin: a fixed 0.18 target leaves a visible gap.
                        let radius = |a: &Actor, j: usize| {
                            a.model
                                .parts
                                .iter()
                                .filter_map(|p| match p {
                                    mavian_rig::Part::Stick { a, b, w0, w1, .. } if *a == j || *b == j => {
                                        Some(0.5 * self.style.line * if *a == j { *w0 } else { *w1 })
                                    }
                                    _ => None,
                                })
                                .reduce(f64::max)
                                .map(|r| r * a.scale)
                        };
                        match (radius(defender, j), radius(actor, s.limb)) {
                            (Some(body), Some(limb)) => body + limb,
                            _ => crate::actions::surface_of(&s.aim),
                        }
                    }
                };
                target.joints[j] + offset
            };
            if s.blocked.is_some() {
                // A jab into a guard needs room to extend, rather than folding
                // the elbow against the attacker's face at touching distance.
                let length = (out.joints[chain[1]] - out.joints[chain[0]]).length() + (out.joints[chain[2]] - out.joints[chain[1]]).length();
                let want = 0.85 * length;
                let delta = out.joints[chain[0]] - goal;
                if delta.length() < want {
                    let along = delta.dot(away);
                    let perpendicular = (delta.dot(delta) - along * along).max(0.0);
                    let shift = ((want * want - perpendicular).max(0.0).sqrt() - along).max(0.0) * reach.weight.at(w);
                    for p in &mut out.joints {
                        *p = *p + away * shift;
                    }
                    out.pivot = out.pivot + away * shift;
                }
            }
            if actor.model.rig.joints()[s.limb].name.starts_with("foot") {
                // Let the hip finish the kick's approach while the target is
                // held, instead of asking IK to lengthen a straight leg.
                let hip = out.joints[chain[0]];
                let length = (out.joints[chain[1]] - hip).length() + (out.joints[chain[2]] - out.joints[chain[1]]).length();
                let to = goal - hip;
                let horizontal = (to.x * to.x + to.z * to.z).sqrt();
                let allowed = (length * length * 0.998001 - to.y * to.y).max(0.0).sqrt();
                let shift = (horizontal - allowed).clamp(0.0, 0.12 * actor.scale) * reach.weight.at(w);
                let forward = crate::ik::flat_dir(hip, goal);
                for p in &mut out.joints {
                    *p = *p + forward * shift;
                }
                out.pivot = out.pivot + forward * shift;
            }
            crate::ik::solve(&mut out.joints, chain, goal, reach.weight.at(w), reach.pole);
        }
        let out = self.solve_actor_contacts(index, w, &out, m).0;
        (Arc::new(self.avoid_props(index, w, &out)), m)
    }

    /// The figure as drawn by camera `cam`: squash and stretch, then the
    /// actor's own perspective.
    pub fn drawn_figure(&self, index: usize, w: f64, cam: &Camera) -> Arc<Figure> {
        let fig = self.posed(index, w).0;
        let actor = &self.actors[index];
        if actor.perspective == 1.0 {
            return fig;
        }
        Arc::new(own_perspective(&fig, cam, fig.joints[actor.model.root], actor.perspective))
    }

    pub fn camera_for(&self, tl: &Timeline, n: usize) -> (usize, Camera) {
        let clock = tl.clock(n);
        let cut = self.camera.pick(&clock).0;
        let mut cam = self.camera.frame(&clock, self.size, Vec3::ZERO, self);
        let (mut dx, mut dy) = (0.0, 0.0);
        for (i, sh) in self.shakes.iter().chain(&self.auto_shakes).enumerate() {
            let (ox, oy) = sh.offset(clock.shot - tl.shot_time_of(sh.at), seed!(self.seed, "shake", i));
            (dx, dy) = (dx + ox, dy + oy);
        }
        if dx != 0.0 || dy != 0.0 {
            let shift = cam.right * dx + cam.up * dy;
            cam = self.camera.frame(&clock, self.size, shift, self);
        }
        (cut, cam)
    }

    pub fn frame(&self, tl: &Timeline, n: usize) -> FrameState {
        let drawn = self.drawn_frame(tl, n);
        let w = tl.world[drawn];
        let (cut, cam) = self.camera_for(tl, n);
        let figures = (0..self.actors.len()).map(|i| self.drawn_figure(i, w, &cam)).collect();
        FrameState { n, drawn, w, clock: tl.clock(n), cam, cut, figures, flash: self.is_flash(tl, n) }
    }

    /// Joint positions in frame n — without boil or shake (for checks and
    /// comparison with the prototype). Same format as `pose_signature` in the MVP.
    pub fn pose_signature(&self, tl: &Timeline, n: usize) -> String {
        let drawn = self.drawn_frame(tl, n);
        let mut out = Vec::new();
        for (i, actor) in self.actors.iter().enumerate() {
            let (fig, _) = self.posed(i, tl.world[drawn]);
            let mut named: Vec<(&str, Vec3)> = actor.model.rig.joints().iter().map(|j| j.name.as_str()).zip(fig.joints.iter().copied()).collect();
            named.sort_by(|a, b| a.0.cmp(b.0));
            out.extend(named.into_iter().map(|(n, p)| format!("{n}:{:.5},{:.5},{:.5}", p.x, p.y, p.z)));
        }
        out.join(";")
    }
}

/// Own perspective (`Actor::perspective`), as a change of the figure in the
/// world: each joint goes where the MVP's projector drew it — away from or
/// towards the pivot on screen by `depth / exaggerated depth` — and to the
/// exaggerated depth, so parts there are as much bigger or smaller and the
/// renderer needs to know nothing about it. k = 1 changes nothing.
pub fn own_perspective(fig: &Figure, cam: &Camera, pivot: Vec3, k: f64) -> Figure {
    let Some(pv) = cam.project(pivot) else { return fig.clone() };
    let mut out = fig.clone();
    for p in &mut out.joints {
        let Some(r) = cam.project(*p) else { continue };
        let z2 = (pv.depth + k * (r.depth - pv.depth)).max(0.25 * pv.depth);
        let f = r.depth / z2;
        *p = cam.unproject(pv.x + (r.x - pv.x) * f, pv.y + (r.y - pv.y) * f, z2);
    }
    out
}

impl Scene for Shot {
    fn joint(&self, actor: usize, joint: &str, w: f64) -> Vec3 {
        let a = &self.actors[actor];
        let i = a.model.rig.joint(joint).expect("followed joints are checked in validate");
        self.posed(actor, w).0.joints[i]
    }
}
