//! Geometry of a frame: every part of every actor as triangles in the world,
//! with a part ID per vertex, plus the per-part drawing table.
//!
//! Sticks are tubes whose radius follows the MVP's line-width rules
//! (`line_px`), so on screen they are exactly as thick as in the sketch.
//! Volumes (limbs, torso, balls, head) are true 3D bodies.

use mavian_core::m;
use mavian_core::{Mat3, Vec3};
use mavian_rig::Part;
use mavian_core::noise::smooth;
use mavian_scene::{Camera, FaceShapes, FrameState, Shape, Shot, Style, TRAIL_FROM, TRAIL_STEPS, face};

/// pixels per world unit in an "ordinary" wide shot (as in the MVP)
pub const SCALE0: f64 = 220.0;
/// volume outline width, fraction of the style's line (as in the MVP)
const VOLUME_LINE: f64 = 0.6;
/// how volume outline width depends on depth (as in the MVP)
const VOLUME_GAMMA: f64 = 1.0;
/// parts per actor in the ID space
pub const PARTS_PER_ACTOR: u32 = 64;
const SEGMENTS: usize = 28;
/// a seam with the root part is hidden within this many root radii of the
/// root joint; further on, a limb against its root part is outlined
const ROOT_ZONE: f64 = 1.5;
/// a prop's outline, fraction of the style's line (as the character's volumes)
const PROP_LINE: f64 = 0.6;
/// a box's edges inside its silhouette are thinner than the outline
const EDGE_LINE: f64 = 0.55;
/// a box's faces: corners around and the outward normal (prop frame); the
/// corner number is 4·(x>0) + 2·(y>0) + (z>0)
const BOX_FACES: [([usize; 4], [f64; 3]); 6] = [
    ([0, 1, 3, 2], [-1.0, 0.0, 0.0]),
    ([4, 6, 7, 5], [1.0, 0.0, 0.0]),
    ([0, 2, 6, 4], [0.0, 0.0, -1.0]),
    ([1, 5, 7, 3], [0.0, 0.0, 1.0]),
    ([0, 4, 5, 1], [0.0, -1.0, 0.0]),
    ([2, 3, 7, 6], [0.0, 1.0, 0.0]),
];

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub id: u32,
}

/// How a part is drawn, by part ID (index 0 is the background).
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PartStyle {
    pub fill: [f32; 4],
    pub line: [f32; 4],
    /// line width in px at the actor's reference scale (0 — no outline)
    pub base: f32,
    /// pixels per world unit at the actor's root point
    pub sref: f32,
    /// how width depends on depth
    pub gamma: f32,
    /// parts of one "skin" share a group: no seam between them where they meet
    pub group: u32,
    /// overlap rules: bit k set — this part is always over part k of its actor
    pub over_lo: u32,
    pub over_hi: u32,
    /// how far (world units) a rule lifts this part over the parts it is
    /// over: a rule bends depth a little, it does not override it — from
    /// below, the torso honestly hides the head
    pub bias: f32,
    /// 1 — the line is a halo: drawn only outside the part, over what is behind
    pub halo: u32,
    /// for a part growing from another: its root joint (xyz) and the radius
    /// of the zone around it where the seam with the root part is hidden (w)
    pub root: [f32; 4],
    /// the part it grows from (0 — none): seams are hidden only between a
    /// part and its parent, or two parts growing from one parent
    pub parent: u32,
    /// Cross-object overlap: actor first ID and its low/high part masks.
    pub _pad: [u32; 3],
    /// a stick's other end (xyz; with `root` it gives the stick's direction);
    /// its halo fades towards it within w, as in the MVP (0 — a free end):
    /// otherwise the neck's halo bites the head's outline
    pub tip: [f32; 4],
}

/// A shadow blob on the ground: (x, z, radius).
pub type Shadow = [f32; 6];

/// A silhouette curve of a part: points on its surface in the world, and how
/// its line is drawn.
pub struct Curve {
    pub points: Vec<Vec3>,
    /// the solid the curve lies on, and the model part it belongs to
    solid: usize,
    part: usize,
    pub closed: bool,
    /// the outline of a ball: its scraps between two tubes are beads
    pub ball: bool,
    /// the outline of a tube: short pieces of it are real lines (a far neck)
    pub tube: bool,
    pub id: u32,
    /// line width in px at the actor's reference scale
    pub base: f64,
    pub sref: f64,
    pub gamma: f64,
    pub color: [f32; 4],
    /// width factor per point (empty — 1): a speed line thins to its tail
    pub taper: Vec<f64>,
}

/// A face drawn on a head: eyes filled, the mouth a line; both only where
/// the head itself is visible.
pub struct FaceDraw {
    /// the head's part ID
    pub id: u32,
    pub color: [f32; 4],
    pub shapes: FaceShapes,
    /// mouth line width, px
    pub mouth_w: f64,
}

/// A stick as a line, for vector output: its axis, world radii at the ends,
/// and its halo.
#[derive(Clone, Debug)]
pub struct StickLine {
    pub id: u32,
    pub a: Vec3,
    pub b: Vec3,
    pub ra: f64,
    pub rb: f64,
    pub color: [f32; 4],
    /// halo: paper colour and width past the stick's edge, px at the
    /// reference scale; it fades within `ramp` (world) of the root joint and
    /// of a joined end
    pub halo_color: [f32; 4],
    pub halo: f64,
    pub ramp: f64,
    pub free: bool,
}

pub struct FrameMesh {
    /// silhouettes of volumes: the line stage draws them where visible
    pub curves: Vec<Curve>,
    pub vertices: Vec<Vertex>,
    /// parts on top by an overlap rule: drawn in a second pass that wins
    /// over the parts they are over, whatever the depth
    pub top: Vec<Vertex>,
    /// Contact props resolve after actor overlap rules (hands are already on top).
    pub contacts: Vec<Vertex>,
    pub parts: Vec<PartStyle>,
    pub shadows: Vec<Shadow>,
    pub faces: Vec<FaceDraw>,
    pub sticks: Vec<StickLine>,
}

pub fn color(hex: &str) -> [f32; 4] {
    let h = hex.trim_start_matches('#');
    let v = |i: usize| u8::from_str_radix(h.get(i..i + 2).unwrap_or("00"), 16).unwrap_or(0) as f32 / 255.0;
    [v(0), v(2), v(4), 1.0]
}

/// Line width scale at a point, as `line_px` in the MVP.
fn line_px(s: f64, sref: f64, st: &Style, gamma: f64) -> f64 {
    if sref <= 1e-9 {
        return s;
    }
    let near = m::pow(s / sref, gamma).clamp(0.45, 1.9);
    m::pow(sref, st.zoom) * m::pow(SCALE0, 1.0 - st.zoom) * near
}

/// A body a figure is made of, to test whether a point is inside it.
#[derive(Clone, Copy)]
enum Solid {
    Sphere(Vec3, f64),
    Cone(Vec3, f64, Vec3, f64),
    /// a torso: from a along axis for len, side and front axes, half sizes
    /// at both ends, caps
    Body { a: Vec3, ax: Vec3, len: f64, side: Vec3, front: Vec3, rx: (f64, f64), rz: (f64, f64), cap: (f64, f64) },
}

impl Solid {
    /// Is p inside, by more than `eps`?
    fn inside(&self, p: Vec3, eps: f64) -> bool {
        match *self {
            Solid::Sphere(c, r) => (p - c).length() < r - eps,
            Solid::Cone(a, ra, b, rb) => {
                // an open cone: its ends are closed by the joint balls, not
                // by caps of its own
                let ab = b - a;
                let t = (p - a).dot(ab) / ab.dot(ab).max(1e-12);
                (0.0..=1.0).contains(&t) && (p - a.lerp(b, t)).length() < ra + (rb - ra) * t - eps
            }
            Solid::Body { a, ax, len, side, front, rx, rz, cap } => {
                let d = p - a;
                let along = d.dot(ax);
                let t = (along / len).clamp(0.0, 1.0);
                let sx = rx.0 + (rx.1 - rx.0) * t;
                let sz = rz.0 + (rz.1 - rz.0) * t;
                let beyond = if along < 0.0 { -along / cap.0 } else if along > len { (along - len) / cap.1 } else { 0.0 };
                let (x, z) = (d.dot(side) / sx, d.dot(front) / sz);
                let k = 1.0 - eps / sx.min(sz);
                x * x + z * z + beyond * beyond < k * k
            }
        }
    }
}

struct Builder<'a> {
    cam: &'a Camera,
    out: Vec<Vertex>,
    curves: Vec<Curve>,
    /// the solids of the current actor, with the model part of each
    solids: Vec<(usize, Solid)>,
    part: usize,
    /// the line style of the part being built (curves take it)
    line: Option<(f64, f64, f64, [f32; 4])>,
}

/// spacing of curve points on screen, px at output size
const CURVE_STEP: f64 = 2.0;

impl Builder<'_> {
    fn tri(&mut self, id: u32, a: Vec3, b: Vec3, c: Vec3) {
        // a fisheye bends straight lines: drop triangles the lens cannot see
        if self.cam.fisheye && [a, b, c].iter().any(|p| self.cam.project(*p).is_none()) {
            return;
        }
        for p in [a, b, c] {
            self.out.push(Vertex { pos: [p.x as f32, p.y as f32, p.z as f32], id });
        }
    }

    fn quad(&mut self, id: u32, a: Vec3, b: Vec3, c: Vec3, d: Vec3) {
        self.tri(id, a, b, c);
        self.tri(id, a, c, d);
    }

    /// Rings of points joined into a band of quads.
    fn rings(&mut self, id: u32, rings: &[Vec<Vec3>]) {
        for w in rings.windows(2) {
            let n = w[0].len();
            for k in 0..n {
                let k1 = (k + 1) % n;
                self.quad(id, w[0][k], w[0][k1], w[1][k1], w[1][k]);
            }
        }
    }

    fn sphere(&mut self, id: u32, c: Vec3, r: f64) {
        self.solids.push((self.part, Solid::Sphere(c, r)));
        self.sphere_outline(id, c, r);
        let lat = SEGMENTS / 2;
        let rings: Vec<Vec<Vec3>> = (0..=lat)
            .map(|i| {
                let t = std::f64::consts::PI * i as f64 / lat as f64;
                (0..SEGMENTS)
                    .map(|k| {
                        let a = std::f64::consts::TAU * k as f64 / SEGMENTS as f64;
                        c + Vec3 { x: r * m::sin(t) * m::cos(a), y: r * m::cos(t), z: r * m::sin(t) * m::sin(a) }
                    })
                    .collect()
            })
            .collect();
        self.rings(id, &rings);
    }

    /// A tube between two spheres (a capsule segment with tapered radius).
    fn capsule(&mut self, id: u32, a: Vec3, ra: f64, b: Vec3, rb: f64) {
        self.sphere(id, a, ra);
        self.tube(id, a, ra, b, rb);
    }

    /// A capsule without the sphere at its start: the previous segment of the
    /// chain already has it. Two equal spheres with different IDs would fight
    /// for every pixel, and the ID would flicker along the edge.
    fn tube(&mut self, id: u32, a: Vec3, ra: f64, b: Vec3, rb: f64) {
        self.solids.push((self.part, Solid::Cone(a, ra, b, rb)));
        self.tube_outline(id, a, ra, b, rb);
        self.sphere(id, b, rb);
        let axis = b - a;
        let len = axis.length();
        if len < 1e-9 {
            return;
        }
        let d = axis * (1.0 / len);
        let (u, v) = basis(d);
        let ring = |c: Vec3, r: f64| -> Vec<Vec3> {
            (0..SEGMENTS)
                .map(|k| {
                    let t = std::f64::consts::TAU * k as f64 / SEGMENTS as f64;
                    c + u * (r * m::cos(t)) + v * (r * m::sin(t))
                })
                .collect()
        };
        let steps = 6;
        let rings: Vec<Vec<Vec3>> = (0..=steps)
            .map(|i| {
                let t = i as f64 / steps as f64;
                ring(a.lerp(b, t), ra + (rb - ra) * t)
            })
            .collect();
        self.rings(id, &rings);
    }

    /// A torso: oval section, half-ellipsoid caps (as the MVP's `Body`).
    #[allow(clippy::too_many_arguments)]
    fn body(&mut self, id: u32, a: Vec3, b: Vec3, rot: Mat3, rx: (f64, f64), rz: (f64, f64), cap: (f64, f64), sc: f64) {
        let axis = b - a;
        let ln = axis.length().max(1e-9);
        let ax = axis * (1.0 / ln);
        // side and front of joint a, made square to the axis
        let side = (rot.col(0) - ax * rot.col(0).dot(ax)).norm();
        let front = ax.cross(side).norm() * if ax.cross(side).dot(rot.col(2)) < 0.0 { -1.0 } else { 1.0 };
        let (cap0, cap1) = (cap.0 * sc, cap.1 * sc);
        let ring_at = |s: f64, segs: usize| -> Vec<Vec3> {
            let t = (s / ln).clamp(0.0, 1.0);
            let (sx, sz) = ((rx.0 + (rx.1 - rx.0) * t) * sc, (rz.0 + (rz.1 - rz.0) * t) * sc);
            let beyond = if s < 0.0 { -s / cap0 } else if s > ln { (s - ln) / cap1 } else { 0.0 };
            let k = (1.0 - beyond * beyond).max(0.0).sqrt();
            let c = a + ax * s;
            (0..segs)
                .map(|j| {
                    let t = std::f64::consts::TAU * j as f64 / segs as f64;
                    c + side * (sx * k * m::cos(t)) + front * (sz * k * m::sin(t))
                })
                .collect()
        };
        self.solids.push((
            self.part,
            Solid::Body { a, ax, len: ln, side, front, rx: (rx.0 * sc, rx.1 * sc), rz: (rz.0 * sc, rz.1 * sc), cap: (cap0, cap1) },
        ));
        // a finer grid for the silhouette than for the fill
        let fine = 48;
        let grid: Vec<Vec<Vec3>> = (0..=fine).map(|i| ring_at(-cap0 + (ln + cap0 + cap1) * i as f64 / fine as f64, 96)).collect();
        self.grid_outline(id, &grid);
        let n = 18;
        let rings: Vec<Vec<Vec3>> = (0..=n)
            .map(|i| {
                let s = -cap0 + (ln + cap0 + cap1) * i as f64 / n as f64;
                let t = (s / ln).clamp(0.0, 1.0);
                let (sx, sz) = ((rx.0 + (rx.1 - rx.0) * t) * sc, (rz.0 + (rz.1 - rz.0) * t) * sc);
                let beyond = if s < 0.0 { -s / cap0 } else if s > ln { (s - ln) / cap1 } else { 0.0 };
                let k = (1.0 - beyond * beyond).max(0.0).sqrt();
                let c = a + ax * s;
                (0..SEGMENTS)
                    .map(|j| {
                        let t = std::f64::consts::TAU * j as f64 / SEGMENTS as f64;
                        c + side * (sx * k * m::cos(t)) + front * (sz * k * m::sin(t))
                    })
                    .collect()
            })
            .collect();
        self.rings(id, &rings);
    }
}

impl Builder<'_> {
    fn curve(&mut self, id: u32, points: Vec<Vec3>, closed: bool) {
        let Some((base, sref, gamma, color)) = self.line else { return };
        if points.len() >= 2 {
            let (solid, part) = (self.solids.len() - 1, self.part);
            let ball = matches!(self.solids.last(), Some((_, Solid::Sphere(..))));
            let tube = matches!(self.solids.last(), Some((_, Solid::Cone(..))));
            self.curves.push(Curve { points, solid, part, closed, ball, tube, id, base, sref, gamma, color, taper: Vec::new() });
        }
    }

    /// Screen size of a length at a point, px.
    fn px(&self, p: Vec3, len: f64) -> f64 {
        self.cam.project(p).map_or(0.0, |r| len * r.scale)
    }

    /// The exact silhouette of a sphere: the circle where rays from the
    /// camera touch it.
    fn sphere_outline(&mut self, id: u32, c: Vec3, r: f64) {
        let to = c - self.cam.pos;
        let d = to.length();
        if d <= r * 1.001 {
            return;
        }
        let v = to * (1.0 / d);
        let center = c - v * (r * r / d);
        let rr = r * (1.0 - r * r / (d * d)).sqrt();
        let (u, w) = basis(v);
        let n = ((std::f64::consts::TAU * self.px(c, rr)) / CURVE_STEP).clamp(24.0, 720.0) as usize;
        let pts = (0..n)
            .map(|k| {
                let t = std::f64::consts::TAU * k as f64 / n as f64;
                center + u * (rr * m::cos(t)) + w * (rr * m::sin(t))
            })
            .collect();
        self.curve(id, pts, true);
    }

    /// The two side lines of a tapered tube (the end spheres outline themselves).
    fn tube_outline(&mut self, id: u32, a: Vec3, ra: f64, b: Vec3, rb: f64) {
        let axis = b - a;
        let len = axis.length();
        if len < 1e-9 {
            return;
        }
        let d = axis * (1.0 / len);
        let n = (self.px(a.lerp(b, 0.5), len) / CURVE_STEP).clamp(2.0, 400.0) as usize;
        let (mut left, mut right) = (Vec::new(), Vec::new());
        for i in 0..=n {
            let t = i as f64 / n as f64;
            let c = a.lerp(b, t);
            let r = ra + (rb - ra) * t;
            let v = (c - self.cam.pos).norm();
            let side = d.cross(v);
            if side.length() < 1e-6 {
                continue; // looking straight along the tube: no sides here
            }
            // the mesh tube is a plain cone: its ring at t touches the view
            // ray at the side, on the surface (a lean along the axis, as for a
            // sphere-swept cone, put the line off the surface and the joint
            // ball poked out as a barb)
            let side = side.norm();
            left.push(c + side * r);
            right.push(c - side * r);
        }
        self.curve(id, left, false);
        self.curve(id, right, false);
    }

    /// Silhouette of a surface given as a grid of rings: where it turns from
    /// facing the camera to facing away (marching squares on the grid).
    fn grid_outline(&mut self, id: u32, grid: &[Vec<Vec3>]) {
        let (rows, cols) = (grid.len(), grid[0].len());
        let at = |i: usize, j: usize| grid[i][j % cols];
        // facing: normal from the neighbours, dotted with the view direction
        let f: Vec<Vec<f64>> = (0..rows)
            .map(|i| {
                (0..cols)
                    .map(|j| {
                        let du = at(i, j + 1) - at(i, j + cols - 1);
                        let dv = at((i + 1).min(rows - 1), j) - at(i.saturating_sub(1), j);
                        du.cross(dv).dot(at(i, j) - self.cam.pos)
                    })
                    .collect()
            })
            .collect();
        // crossing point on a grid edge, keyed so neighbouring cells share it
        let cross = |a: (usize, usize), b: (usize, usize)| -> (Vec3, (usize, usize, usize, usize)) {
            let (fa, fb) = (f[a.0][a.1 % cols], f[b.0][b.1 % cols]);
            let t = fa / (fa - fb);
            let key = if a <= b { (a.0, a.1 % cols, b.0, b.1 % cols) } else { (b.0, b.1 % cols, a.0, a.1 % cols) };
            (at(a.0, a.1).lerp(at(b.0, b.1), t), key)
        };
        type Key = (usize, usize, usize, usize);
        let mut segs: Vec<((Vec3, Key), (Vec3, Key))> = Vec::new();
        for i in 0..rows - 1 {
            for j in 0..cols {
                let c = [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)];
                let mut hits = Vec::new();
                for k in 0..4 {
                    let (a, b) = (c[k], c[(k + 1) % 4]);
                    if (f[a.0][a.1 % cols] > 0.0) != (f[b.0][b.1 % cols] > 0.0) {
                        hits.push(cross(a, b));
                    }
                }
                if hits.len() == 2 {
                    segs.push((hits[0], hits[1]));
                } else if hits.len() == 4 {
                    segs.push((hits[0], hits[1]));
                    segs.push((hits[2], hits[3]));
                }
            }
        }
        // chain segments into polylines through shared crossing points
        let mut used = vec![false; segs.len()];
        let mut by_key: std::collections::HashMap<Key, Vec<usize>> = std::collections::HashMap::new();
        for (k, (a, b)) in segs.iter().enumerate() {
            by_key.entry(a.1).or_default().push(k);
            by_key.entry(b.1).or_default().push(k);
        }
        for start in 0..segs.len() {
            if used[start] {
                continue;
            }
            used[start] = true;
            let mut line = std::collections::VecDeque::from([segs[start].0, segs[start].1]);
            for forward in [true, false] {
                loop {
                    let end = if forward { line.back() } else { line.front() }.expect("two points").1;
                    let next = by_key.get(&end).and_then(|v| v.iter().copied().find(|&k| !used[k]));
                    let Some(k) = next else { break };
                    used[k] = true;
                    let (a, b) = segs[k];
                    let p = if a.1 == end { b } else { a };
                    if forward { line.push_back(p) } else { line.push_front(p) }
                }
            }
            let closed = line.front().map(|p| p.1) == line.back().map(|p| p.1);
            let mut pts: Vec<Vec3> = line.into_iter().map(|p| p.0).collect();
            if closed {
                pts.pop();
            }
            self.curve(id, pts, closed);
        }
    }
}

/// Two unit vectors square to d and to each other.
fn basis(d: Vec3) -> (Vec3, Vec3) {
    let helper = if d.y.abs() < 0.9 { Vec3::UP } else { Vec3 { x: 1.0, y: 0.0, z: 0.0 } };
    let u = d.cross(helper).norm();
    (u, d.cross(u).norm())
}

/// halo around sticks: it fades in this far from the joint the stick starts at
/// (as `HALO_RAMP` in the MVP)
const HALO_RAMP: f64 = 0.10;

pub fn build(shot: &Shot, f: &FrameState) -> FrameMesh {
    let st = &crate::frame_style(shot, f);
    let cam = &f.cam;
    let mut b = Builder { cam, out: Vec::new(), curves: Vec::new(), solids: Vec::new(), part: 0, line: None };
    let mut top = Vec::new();
    let mut contacts = Vec::new();
    let mut parts = vec![PartStyle::default(); 1 + shot.actors.len() * PARTS_PER_ACTOR as usize];
    let mut shadows = Vec::new();
    let mut faces = Vec::new();
    let mut sticks = Vec::new();
    let (paper, ink_default) = (color(&st.paper), color(&st.ink));
    for (i, actor) in shot.actors.iter().enumerate() {
        let model = &actor.model;
        let fig = &f.figures[i];
        let sc = actor.scale;
        // an impact frame draws everyone in the (swapped) ink
        let ink = if f.flash { ink_default } else { actor.color.as_deref().map(color).unwrap_or(ink_default) };
        let root = fig.joints[model.root];
        if shot.ground.as_ref().is_some_and(|g| g.shadow) && model.shadow > 0.0 {
            let r = model.shadow * sc * (1.0 - 0.5 * fig.lift.clamp(0.0, 1.0));
            shadows.push([root.x as f32, root.z as f32, r as f32, 0.0, 0.0, 0.0]);
        }
        let head = model.parts.iter().find_map(|p| match p {
            Part::Head { joint, .. } => Some(*joint),
            _ => None,
        });
        let sref = cam.project(root).or_else(|| head.and_then(|h| cam.project(fig.joints[h]))).map_or(SCALE0, |r| r.scale);
        let base = |w: f64| (st.line * w * sc * m::pow(sref, st.zoom) * m::pow(SCALE0, 1.0 - st.zoom)) as f32;
        // world radius of a stick at a point, so its screen width follows `line_px`
        let stick_r = |p: Vec3, w: f64, gamma: f64| -> f64 {
            match cam.project(p) {
                Some(r) => 0.5 * st.line * sc * w * line_px(r.scale, sref, st, gamma) / r.scale,
                None => 0.5 * st.line * sc * w,
            }
        };
        // IDs: one per part, one per segment of a limb (so a limb hiding
        // itself — a thigh in front of its shin — gets a line)
        let first = part_id(i, 0);
        let mut ids: Vec<Vec<u32>> = Vec::new();
        let mut next = first;
        for part in &model.parts {
            // a limb: a tube per segment and a ball per joint, each its own ID,
            // so a ball inside its own tube is hidden like any other part
            let n = match part {
                Part::Limb { chain, .. } => 2 * chain.len() - 1,
                _ => 1,
            };
            ids.push((next..next + n as u32).collect());
            next += n as u32;
        }
        assert!(next - first < PARTS_PER_ACTOR, "too many parts in model {}", model.name);
        // overlap rules as bit masks over the actor's IDs
        let mut over = vec![0u64; model.parts.len()];
        for (a, below) in &model.over {
            if let (Some(a), Some(below)) = (model.part(a), model.part(below)) {
                for id in &ids[below] {
                    over[a] |= 1 << (id - first);
                }
            }
        }
        let group_of = |name: &str| model.part(name).map_or(0, |k| ids[k][0]);
        let stick_group = first + PARTS_PER_ACTOR - 1;
        let at = |p: Vec3, r: f64| [p.x as f32, p.y as f32, p.z as f32, r as f32];
        // a stick grows from the stick that ends where it starts; at the root
        // joint (no stick ends there), from the first stick starting there
        let stick_parent = |k: usize| -> u32 {
            let Part::Stick { a, .. } = &model.parts[k] else { return 0 };
            let ends = model.parts.iter().position(|p| matches!(p, Part::Stick { b, .. } if b == a));
            let starts = model.parts[..k].iter().position(|p| matches!(p, Part::Stick { a: a2, .. } if a2 == a));
            ends.or(starts).map_or(0, |j| ids[j][0])
        };
        let first_curve = b.curves.len();
        b.solids.clear();
        for (k, part) in model.parts.iter().enumerate() {
            b.part = k;
            let start = b.out.len();
            // volumes are outlined by curves; the pixel line path is for halos
            let line_w = match part {
                Part::Head { line, .. } => *line,
                _ => VOLUME_LINE,
            };
            let gamma = if matches!(part, Part::Head { .. }) { st.depth_gamma } else { VOLUME_GAMMA };
            b.line = match part {
                Part::Stick { .. } => None,
                _ => Some((base(line_w) as f64, sref, gamma, ink)),
            };
            let volume = |_line: f64, group: u32| PartStyle {
                fill: paper,
                line: ink,
                base: 0.0,
                sref: sref as f32,
                gamma: VOLUME_GAMMA as f32,
                group,
                ..Default::default()
            };
            let mut styles: Vec<(u32, PartStyle)> = Vec::new();
            let id = ids[k][0];
            match part {
                Part::Stick { a, b: bj, w0, w1, trim, dot, free, halo, gamma, .. } => {
                    let g = gamma.unwrap_or(st.depth_gamma);
                    let (pa, mut pb) = (fig.joints[*a], fig.joints[*bj]);
                    if *trim > 0.0 {
                        let d = pb - pa;
                        let ln = d.length();
                        let cut = trim * sc;
                        if ln <= cut {
                            continue;
                        }
                        pb = pa + d * ((ln - cut) / ln);
                    }
                    let (ra, mut rb) = (stick_r(pa, *w0, g), stick_r(pb, *w1, g));
                    if *dot {
                        rb *= 1.9; // the hand: a dot slightly bigger than the line
                    }
                    // the stick ending at this joint already has its sphere here
                    let joined = model.parts.iter().any(|p| matches!(p, Part::Stick { b, .. } if b == a));
                    if joined { b.tube(id, pa, ra, pb, rb) } else { b.capsule(id, pa, ra, pb, rb) }
                    sticks.push(StickLine {
                        id,
                        a: pa,
                        b: pb,
                        ra,
                        rb,
                        color: ink,
                        halo_color: paper,
                        halo: if *halo { st.halo } else { 0.0 },
                        ramp: HALO_RAMP * sc,
                        free: *free,
                    });
                    // sticks are ink; a paper halo outside a nearer stick breaks
                    // the farther one, so overlaps read (as in the MVP)
                    styles.push((
                        id,
                        PartStyle {
                            fill: ink,
                            line: paper,
                            base: if *halo { (2.0 * st.halo) as f32 } else { 0.0 },
                            sref: sref as f32,
                            gamma: 0.0,
                            group: stick_group,
                            halo: *halo as u32,
                            root: at(pa, HALO_RAMP * sc),
                            parent: stick_parent(k),
                            tip: at(pb, if *free { 0.0 } else { HALO_RAMP * sc }),
                            ..Default::default()
                        },
                    ));
                }
                Part::Head { joint, radius, line, face: fc, filled } => {
                    let (center, rad) = (fig.joints[*joint], radius * sc);
                    if *filled {
                        // a solid disc of the character's colour: no outline, no face
                        b.line = None;
                        b.sphere(id, center, rad);
                        styles.push((id, PartStyle { fill: ink, line: ink, sref: sref as f32, ..Default::default() }));
                        for (id, mut style) in styles {
                            (style.over_lo, style.over_hi) = (over[k] as u32, (over[k] >> 32) as u32);
                            parts[id as usize] = style;
                        }
                        continue;
                    }
                    b.sphere(id, center, rad);
                    if actor.face
                        && let Some(c) = cam.project(center)
                        && let Some(shapes) = face(cam, center, &fig.rots[*joint], rad, fc, actor.face_trust, actor.blink.at(f.w))
                    {
                        // the mouth is half the head's outline, as in the MVP
                        let near = m::pow(c.scale / sref, st.depth_gamma).clamp(0.45, 1.9);
                        faces.push(FaceDraw { id, color: ink, shapes, mouth_w: 0.5 * base(*line) as f64 * near });
                    }
                    styles.push((id, PartStyle { gamma: st.depth_gamma as f32, bias: (0.5 * radius * sc) as f32, ..volume(*line, 0) }));
                }
                Part::Limb { chain, radii, root, .. } => {
                    // segments of one limb are one skin with each other and
                    // with the part the limb grows from
                    let group = root.as_deref().map_or(ids[k][0], group_of);
                    let segs = chain.len() - 1;
                    for (j, (&joint, &r)) in chain.iter().zip(radii).enumerate() {
                        let ball = ids[k][segs + j];
                        b.sphere(ball, fig.joints[joint], r * sc);
                        styles.push((ball, volume(VOLUME_LINE, group)));
                    }
                    for s in 1..chain.len() {
                        let seg = ids[k][s - 1];
                        let (pa, ra, pb, rb) = (fig.joints[chain[s - 1]], radii[s - 1] * sc, fig.joints[chain[s]], radii[s] * sc);
                        b.tube(seg, pa, ra, pb, rb);
                        let mut v = volume(VOLUME_LINE, group);
                        v.parent = if s > 1 { ids[k][s - 2] } else { root.as_deref().map_or(0, group_of) };
                        // the seam is hidden around the joint the segment starts at
                        if s > 1 || root.is_some() {
                            v.root = at(fig.joints[chain[s - 1]], ROOT_ZONE * radii[s - 1] * sc);
                        }
                        styles.push((seg, v));
                    }
                }
                Part::Body { name, a, b: bj, rx, rz, trim, cap } => {
                    let (mut pa, mut pb) = (fig.joints[*a], fig.joints[*bj]);
                    let ln = (pb - pa).length();
                    if ln > 1e-9 {
                        (pa, pb) = (pa.lerp(pb, trim.0 * sc / ln), pa.lerp(pb, 1.0 - trim.1 * sc / ln));
                    }
                    b.body(id, pa, pb, fig.rots[*a], *rx, *rz, *cap, sc);
                    styles.push((id, volume(VOLUME_LINE, group_of(name))));
                }
                Part::Ball { joint, radius } => {
                    b.sphere(id, fig.joints[*joint], radius * sc);
                    styles.push((id, PartStyle { bias: (radius * sc) as f32, ..volume(VOLUME_LINE, 0) }));
                }
            }
            for (id, mut style) in styles {
                (style.over_lo, style.over_hi) = (over[k] as u32, (over[k] >> 32) as u32);
                parts[id as usize] = style;
            }
            if over[k] != 0 {
                top.extend(b.out.drain(start..));
            }
        }
        let mine = b.curves.split_off(first_curve);
        let kept = union_outline(mine, &b.solids, &over, &ids, first, sc, cam, st);
        b.curves.extend(kept);
        // speed lines: where fast joints were a moment ago, thinning to the
        // tail; seen or hidden like any line (after the outline union, which
        // would drop the trail inside the body it starts from)
        if actor.trails > 0.0 {
            let dt = actor.trails / TRAIL_STEPS as f64;
            let past: Vec<_> = (0..=TRAIL_STEPS).map(|k| shot.drawn_figure(i, f.w - dt * k as f64, cam)).collect();
            for &j in &model.trails {
                let speed = (past[0].joints[j] - past[1].joints[j]).length() / dt;
                let gain = ((speed - TRAIL_FROM) / TRAIL_FROM).min(1.0);
                if gain <= 0.0 {
                    continue;
                }
                // the part the joint belongs to: its trail is not hidden by it
                let owner = model.parts.iter().enumerate().rev().find_map(|(k, p)| match p {
                    Part::Stick { b: e, .. } if *e == j => Some(ids[k][0]),
                    Part::Ball { joint, .. } | Part::Head { joint, .. } if *joint == j => Some(ids[k][0]),
                    Part::Limb { chain, .. } => chain.iter().position(|&c| c == j).map(|q| ids[k][chain.len() - 1 + q]),
                    _ => None,
                });
                b.curves.push(Curve {
                    points: past.iter().map(|p| p.joints[j]).collect(),
                    solid: usize::MAX,
                    part: usize::MAX,
                    closed: false,
                    ball: false,
                    tube: true,
                    id: owner.unwrap_or(first),
                    base: base(0.55 * gain) as f64,
                    sref,
                    gamma: st.depth_gamma,
                    color: ink,
                    taper: (0..=TRAIL_STEPS).map(|k| 1.0 - k as f64 / TRAIL_STEPS as f64).collect(),
                });
            }
        }
    }
    // props and wires after the characters, each with its own ID
    let first_prop = 1 + shot.actors.len() as u32 * PARTS_PER_ACTOR;
    parts.resize(first_prop as usize + shot.props.len() + shot.wires.len(), PartStyle::default());
    b.solids.clear();
    b.part = usize::MAX;
    for (k, prop) in shot.props.iter().enumerate() {
        let id = first_prop + k as u32;
        let (pos, rot) = shot.prop_rest(k,f.w);
        let center = pos + Vec3::UP * (prop.shape.top() / 2.0);
        // a prop's line has the same width all over it, by its own distance
        let s = cam.project(center).map_or(SCALE0, |r| r.scale);
        let width = PROP_LINE * st.line * line_px(s, s, st, 0.0);
        let fill = prop.fill.as_deref().map(color).unwrap_or(paper);
        parts[id as usize] = PartStyle { fill, line: ink_default, sref: s as f32, ..Default::default() };
        let held = prop.contacts.iter().find(|c| c.0 <= f.w && f.w <= c.1).map(|c| (c.2,c.3)).or_else(|| match prop.segment(f.w) {
            Some(mavian_scene::Motion::Held { actor, hand, .. }) => Some((*actor,*hand)),
            Some(mavian_scene::Motion::Above { held, .. }) => match held.as_ref() { mavian_scene::Motion::Held { actor, hand, .. } => Some((*actor,*hand)), _ => None },
            _ => None,
        });
        if let Some((actor, hand)) = held {
            let model = &shot.actors[actor].model;
            let mut mask = 0u64;
            let mut slot = 0;
            for p in &model.parts {
                let count = match p { Part::Limb { chain, .. } => 2*chain.len()-1, _ => 1 };
                let touches = match p { Part::Stick { b, .. } => *b == hand || model.parts.iter().any(|q| matches!(q, Part::Stick { a, b: end, .. } if *end == hand && a == b)), Part::Ball { joint, .. } => *joint == hand, Part::Limb { chain, .. } => chain.contains(&hand), _ => false };
                if touches { for k in slot..slot+count { mask |= 1u64 << k; } }
                slot += count;
            }
            parts[id as usize]._pad = [part_id(actor,0), mask as u32, (mask>>32) as u32];
            parts[id as usize].bias = 1000.0;
        }
        let prop_start = b.out.len();
        let lift = pos.y.clamp(0.0, 1.0);
        let k_shadow = 1.0 - 0.5 * lift;
        match prop.shape {
            Shape::Sphere { r } => {
                b.line = Some((width, s, 0.0, ink_default));
                b.sphere(id, pos + Vec3::UP * r, r);
                if prop.shadow {
                    shadows.push([pos.x as f32, pos.z as f32, (r * 1.1 * k_shadow) as f32, 0.0, 0.0, 0.0]);
                }
            }
            Shape::Box { sx, sy, sz } => {
                let corner = |i: usize| {
                    let x = if i & 4 != 0 { sx / 2.0 } else { -sx / 2.0 };
                    let y = if i & 2 != 0 { sy } else { 0.0 };
                    let z = if i & 1 != 0 { sz / 2.0 } else { -sz / 2.0 };
                    pos + rot * Vec3 { x, y, z }
                };
                let c: Vec<Vec3> = (0..8).map(corner).collect();
                // how squarely each face looks at the camera (> 0 — seen)
                let facing: Vec<f64> = BOX_FACES
                    .iter()
                    .map(|(q, n)| {
                        let mid = (c[q[0]] + c[q[1]] + c[q[2]] + c[q[3]]) * 0.25;
                        (rot * Vec3 { x: n[0], y: n[1], z: n[2] }).dot((cam.pos - mid).norm())
                    })
                    .collect();
                for (q, _) in &BOX_FACES {
                    b.quad(id, c[q[0]], c[q[1]], c[q[2]], c[q[3]]);
                }
                // edges: on the silhouette (a seen face and a hidden one) —
                // the outline; between two seen faces — thinner, fading as one
                // of them turns edge-on (else a double line at the silhouette)
                let mut edges: Vec<((usize, usize), Vec<f64>)> = Vec::new();
                for (fi, (q, _)) in BOX_FACES.iter().enumerate() {
                    for e in 0..4 {
                        let (a, z) = (q[e], q[(e + 1) % 4]);
                        let key = (a.min(z), a.max(z));
                        match edges.iter_mut().find(|(k, _)| *k == key) {
                            Some((_, f)) => f.push(facing[fi]),
                            None => edges.push((key, vec![facing[fi]])),
                        }
                    }
                }
                for ((a, z), f) in edges {
                    let seen: Vec<f64> = f.iter().copied().filter(|v| *v > 1e-9).collect();
                    let w = match seen.len() {
                        1 => width,
                        2 => width * EDGE_LINE / PROP_LINE * smooth((seen[0].min(seen[1]) - 0.03) / 0.22),
                        _ => 0.0,
                    };
                    if w <= 0.0 {
                        continue;
                    }
                    let n = ((b.px(c[a], (c[z] - c[a]).length()) / CURVE_STEP).ceil() as usize).clamp(2, 400);
                    let points = (0..=n).map(|i| c[a].lerp(c[z], i as f64 / n as f64)).collect();
                    b.curves.push(Curve {
                        points,
                        solid: usize::MAX,
                        part: usize::MAX,
                        closed: false,
                        ball: false,
                        tube: true,
                        id,
                        base: w,
                        sref: s,
                        gamma: 0.0,
                        color: ink_default,
                        taper: Vec::new(),
                    });
                }
                if prop.shadow {
                    let axis = rot.col(0);
                    shadows.push([pos.x as f32, pos.z as f32, (sx * 0.54 * k_shadow) as f32, (sz * 0.54 * k_shadow) as f32, axis.x as f32, axis.z as f32]);
                }
            }
        }
        if held.is_some() { contacts.extend(b.out.drain(prop_start..)); }
    }
    for (k, wire) in shot.wires.iter().enumerate() {
        let id = first_prop + (shot.props.len() + k) as u32;
        let offset = wire.at.as_ref().map_or(Vec3::ZERO, |p| p.resolve(&f.clock, shot));
        for &(a, z) in &wire.segments {
            let (a,z) = (a + offset, z + offset);
            let n = ((b.px(a, (z - a).length()) / CURVE_STEP).ceil() as usize).clamp(2, 400);
            b.curves.push(Curve {
                points: (0..=n).map(|i| a.lerp(z, i as f64 / n as f64)).collect(),
                solid: usize::MAX,
                part: usize::MAX,
                closed: false,
                ball: false,
                tube: true,
                id,
                base: wire.width,
                sref: 1.0,
                gamma: 0.0,
                color: color(&wire.color),
                taper: Vec::new(),
            });
        }
    }
    let drawing = crate::Drawing::new(shot, f);
    let warp = |p: Vec3, id: u32| {
        let active = id < first_prop || shot.props.get((id-first_prop) as usize).is_some_and(|p| p.boil);
        if active { drawing.world(cam, p) } else { p }
    };
    for v in b.out.iter_mut().chain(top.iter_mut()).chain(contacts.iter_mut()) {
        let mut p = Vec3 { x:v.pos[0] as f64,y:v.pos[1] as f64,z:v.pos[2] as f64 };
        if let Some(s) = sticks.iter().find(|s| s.id == v.id) {
            let ab = s.b-s.a;
            let u = ((p-s.a).dot(ab)/ab.dot(ab).max(1e-12)).clamp(0.0,1.0);
            let axis = s.a.lerp(s.b,u);
            p = axis + (p-axis)*drawing.pressure(s.id,u*ab.length());
        }
        p = warp(p,v.id);
        v.pos = [p.x as f32,p.y as f32,p.z as f32];
    }
    for c in &mut b.curves { for p in &mut c.points { *p = warp(*p,c.id); } }
    for face in &mut faces {
        for p in face.shapes.mouth.iter_mut().chain(face.shapes.eyes.iter_mut().flatten()) { *p = drawing.at(p.0,p.1); }
    }
    FrameMesh { curves: b.curves, vertices: b.out, top, contacts, parts, shadows, faces, sticks }
}

/// Keep only the outline of the union of an actor's solids: a silhouette point
/// inside another solid of the figure is not its edge (a joint ball inside
/// its tube, an arm's side inside the torso). Overlap rules count: a hand is
/// over the arm, so its outline stays whole inside the arm.
#[allow(clippy::too_many_arguments)]
fn union_outline(curves: Vec<Curve>, solids: &[(usize, Solid)], over: &[u64], ids: &[Vec<u32>], first: u32, sc: f64, cam: &Camera, _st: &Style) -> Vec<Curve> {
    // outline points of each solid, to tell a duplicate from a real outline
    let mut outline_of: Vec<Vec<Vec3>> = vec![Vec::new(); solids.len()];
    for c in &curves {
        if !c.ball {
            outline_of[c.solid].extend(c.points.iter().copied());
        }
    }
    let mut out = Vec::new();
    for c in curves {
        let eps = 0.002 * sc;
        // a ball point lying on another solid's surface where that solid's
        // own outline already runs is a duplicate of that line (a knee ball as
        // wide as the tube ends: its arc runs along the tube's side and shows
        // as a barb); where the solid draws no line there (looking along the
        // tube), the ball's outline is the real edge and stays
        let duplicate = |p: Vec3| {
            if !c.ball {
                return false;
            }
            let near = cam.project(p).map_or(0.01 * sc, |r| 2.5 * CURVE_STEP / r.scale) + 0.003 * sc;
            // only within one part (a knee ball and the tubes of its own leg):
            // a head is not a duplicate of the neck it sits on
            solids.iter().enumerate().any(|(k, (part, solid))| {
                k != c.solid
                    && *part == c.part
                    && solid.inside(p, -0.003 * sc)
                    && outline_of[k].iter().any(|q| (*q - p).length() < near)
            })
        };
        let hides = |p: Vec3| {
            solids.iter().enumerate().any(|(k, (part, solid))| {
                if k == c.solid {
                    return false;
                }
                // a part over another by a rule is not hidden inside it
                let lifted = ids[*part].iter().any(|id| over[c.part] & (1 << (id - first)) != 0);
                !lifted && solid.inside(p, eps)
            })
        };
        let keep: Vec<bool> = c.points.iter().map(|&p| !hides(p) && !duplicate(p)).collect();
        if keep.iter().all(|&k| k) {
            out.push(c);
            continue;
        }
        // split into runs of kept points; a closed curve starts after a gap.
        // Each run is carried to the exact point where the curve goes into
        // the other solid (found by halving): points are a step apart, and
        // two outlines meeting there would otherwise both stop short of it
        let gone = |p: Vec3| hides(p) || duplicate(p);
        let edge = |kept: Vec3, lost: Vec3| -> Vec3 {
            let (mut a, mut b) = (kept, lost);
            for _ in 0..20 {
                let m = a.lerp(b, 0.5);
                if gone(m) { b = m } else { a = m }
            }
            a
        };
        let n = c.points.len();
        let start = if c.closed { keep.iter().position(|&k| !k).unwrap_or(0) } else { 0 };
        let mut run: Vec<Vec3> = Vec::new();
        let flush = |run: &mut Vec<Vec3>, out: &mut Vec<Curve>| {
            if run.len() >= 2 {
                out.push(Curve { points: std::mem::take(run), closed: false, ..c.clone_style() });
            }
            run.clear();
        };
        let mut prev: Option<usize> = None;
        for i in 0..n {
            let j = (start + i) % n;
            if keep[j] {
                if run.is_empty()
                    && let Some(pj) = prev.filter(|&pj| !keep[pj])
                {
                    run.push(edge(c.points[j], c.points[pj]));
                }
                run.push(c.points[j]);
            } else {
                if let Some(pj) = prev.filter(|&pj| keep[pj]) {
                    run.push(edge(c.points[pj], c.points[j]));
                }
                flush(&mut run, &mut out);
            }
            prev = Some(j);
        }
        // a closed curve: the last run continues into the point it started after
        if c.closed && !run.is_empty() && !keep[start] {
            run.push(edge(*run.last().expect("not empty"), c.points[start]));
        }
        flush(&mut run, &mut out);
    }
    out
}

impl Curve {
    fn clone_style(&self) -> Curve {
        Curve {
            points: Vec::new(),
            solid: self.solid,
            part: self.part,
            closed: self.closed,
            ball: self.ball,
            tube: self.tube,
            id: self.id,
            base: self.base,
            sref: self.sref,
            gamma: self.gamma,
            color: self.color,
            taper: self.taper.clone(),
        }
    }
}

/// The first ID of actor `i` plus `slot` (0 is the background).
fn part_id(actor: usize, slot: usize) -> u32 {
    1 + actor as u32 * PARTS_PER_ACTOR + slot as u32
}
