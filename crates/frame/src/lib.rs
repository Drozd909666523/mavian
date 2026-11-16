//! A frame ready to draw, shared by the GPU renderer and the CPU reference:
//! geometry (`mesh`), uniforms, and the CPU side of the line stage (strokes
//! from curve visibility, faces). Both renderers take the same input and run
//! the same stages (05-architecture, §5): the CPU one checks the GPU.

mod mesh;

pub use mesh::{Curve, FaceDraw, FrameMesh, StickLine, PARTS_PER_ACTOR, PartStyle, SCALE0, Vertex, build, color};

use mavian_core::m;
use mavian_core::Vec3;
use mavian_scene::{Camera, FrameState, Shot};

/// Supersampling factor per axis.
pub const SS: u32 = 2;
pub const NEAR: f32 = 0.05;
pub const FAR: f32 = 200.0;
/// depth difference (world units) under which parts of one skin meet in a seam
pub const SEAM_TOL: f32 = 0.01;
/// jump flood steps (supersampled px): lines up to ~127 px wide
pub const STEPS: [i32; 7] = [64, 32, 16, 8, 4, 2, 1];
pub const MAX_SHADOWS: usize = 16;

/// Which layers to draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layers {
    pub paint: bool,
    pub lines: bool,
    /// debug: fill every part with its own colour
    pub ids: bool,
}

/// Something that draws frames: the GPU or the CPU reference.
pub trait Renderer {
    /// Draw frame `f` of `shot` → RGBA pixels, `size.0 × size.1 × 4` bytes.
    fn render(&mut self, shot: &Shot, f: &FrameState, layers: Layers) -> Vec<u8>;
    /// What draws: the frame cache keys on it, different devices may differ
    /// in the last bits of a pixel.
    fn device(&self) -> String;
    /// Reuse this frame's geometry for effect clipping. The SVG contains fire only.
    fn render_with_depth(&mut self, shot: &Shot, f: &FrameState, layers: Layers) -> RenderedFrame;
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CamU {
    pub pos: [f32; 4],
    pub right: [f32; 4],
    pub up: [f32; 4],
    pub fwd: [f32; 4],
    pub a: [f32; 4],
    pub b: [f32; 4],
    pub c: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneU {
    pub paper: [f32; 4],
    pub ground: [f32; 4],
    pub grid: [f32; 4],
    pub center: [f32; 4],
    pub layers: [f32; 4],
    pub shadows: [[f32; 4]; MAX_SHADOWS],
    pub shadow_axes: [[f32; 4]; MAX_SHADOWS],
    pub glow: [f32; 4],
    pub glow_color: [f32; 4],
}

/// The style a frame is drawn in: on an impact frame paper and ink swap
/// (and the ground goes ink), as in the MVP.
pub fn frame_style(shot: &Shot, f: &FrameState) -> mavian_scene::Style {
    let mut st = shot.style.clone();
    if f.flash {
        std::mem::swap(&mut st.paper, &mut st.ink);
        st.ground = st.paper.clone();
    }
    st
}

/// Camera and scene parameters of a frame, as the shader sees them.
pub fn uniforms(shot: &Shot, f: &FrameState, mesh: &FrameMesh, layers: Layers) -> (CamU, SceneU) {
    let size = shot.size;
    let c = &f.cam;
    let ss = SS as f32;
    let cutoff = 1.15 * m::hypot(c.cx, c.cy) as f32 * ss;
    let cam = CamU {
        pos: v4(c.pos),
        right: v4(c.right),
        up: v4(c.up),
        fwd: v4(c.fwd),
        a: [c.f as f32 * ss, c.cx as f32 * ss, c.cy as f32 * ss, if c.fisheye { 1.0 } else { 0.0 }],
        b: [NEAR, FAR, size.0 as f32 * ss, size.1 as f32 * ss],
        c: [c.f as f32, ss, cutoff, SEAM_TOL],
    };
    let st = &frame_style(shot, f);
    let mut scene = SceneU {
        paper: color(&st.paper),
        ground: color(&st.ground),
        grid: [0.0; 4],
        center: [0.0; 4],
        layers: [layers.paint as u32 as f32, layers.lines as u32 as f32, layers.ids as u32 as f32, 0.0],
        shadows: [[0.0; 4]; MAX_SHADOWS],
        shadow_axes: [[0.0; 4]; MAX_SHADOWS],
        glow: [0.0; 4],
        glow_color: [0.0; 4],
    };
    // the first fire's glow lights the ground
    if let Some(fire) = shot.fires.first() {
        let (c, r, k) = fire.glow_at(f.w);
        scene.glow = [c.x as f32, c.z as f32, r as f32, k as f32];
        scene.glow_color = color(&fire.glow);
    }
    if let Some(g) = &shot.ground {
        scene.grid = [g.size as f32, g.step as f32, g.width as f32, if g.grid { 1.0 } else { 0.0 }];
        let n = mesh.shadows.len().min(MAX_SHADOWS);
        scene.center = [g.center.0 as f32, g.center.1 as f32, n as f32, 1.0];
        for (k, s) in mesh.shadows.iter().take(n).enumerate() {
            scene.shadows[k] = [s[0], s[1], s[2], s[3]];
            scene.shadow_axes[k] = [s[4], s[5], 0.0, 0.0];
        }
    }
    (cam, scene)
}

/// A piece of a stroke: from a to b (supersampled px), radii at the ends.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Segment {
    pub a: [f32; 2],
    pub b: [f32; 2],
    pub r: [f32; 2],
    pub color: [f32; 4],
    pub depth: [f32; 2],
    pub id: u32,
    /// bit 0: end a is hidden, bit 1: end b is hidden — the stroke is cut by
    /// occluders only on the way into them; bit 2: drawn on the surface of
    /// part `id` — only where that part is visible (the face)
    pub hidden: u32,
}

pub const ON_SURFACE: u32 = 4;

/// A vertex of a filled shape on a part's surface (an eye), supersampled px.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FillVertex {
    pub pos: [f32; 2],
    pub color: [f32; 4],
    pub id: u32,
}

/// Faces: the mouth as a stroke, eyes as fills with a thin stroke along
/// their edge for anti-aliasing. All of it only on the visible head.
pub fn faces(faces: &[FaceDraw]) -> (Vec<Segment>, Vec<FillVertex>) {
    let ss = SS as f32;
    let (mut segs, mut fills) = (Vec::new(), Vec::new());
    let px = |(x, y): (f64, f64)| [x as f32 * ss, y as f32 * ss];
    for f in faces {
        let mut line = |a: [f32; 2], b: [f32; 2], r: f32| {
            segs.push(Segment { a, b, r: [r, r], color: f.color, depth: [0.0; 2], id: f.id, hidden: ON_SURFACE });
        };
        let r = 0.5 * f.mouth_w as f32 * ss;
        for w in f.shapes.mouth.windows(2) {
            line(px(w[0]), px(w[1]), r);
        }
        for eye in &f.shapes.eyes {
            let n = eye.len();
            let (cx, cy) = eye.iter().fold((0.0, 0.0), |(x, y), p| (x + p.0 / n as f64, y + p.1 / n as f64));
            let c = px((cx, cy));
            for i in 0..n {
                let (a, b) = (px(eye[i]), px(eye[(i + 1) % n]));
                for p in [c, a, b] {
                    fills.push(FillVertex { pos: p, color: f.color, id: f.id });
                }
                line(a, b, 0.5);
            }
        }
    }
    (segs, fills)
}

/// One point of a curve's stroke: position (supersampled px), radius,
/// depth; None — behind the camera.
pub type StrokePoint = Option<([f32; 2], f32, f32)>;

/// The points of curve `c` as a stroke, or None if the curve is a scrap not
/// worth drawing. `v` — visibility per point. Width follows depth
/// (`line_px` of the MVP); a hidden point next to a visible one keeps its
/// width: the stroke is cut exactly at the occluder's edge (pixel by pixel
/// in the raster, by bisection in vectors).
pub fn stroke_points(c: &Curve, v: &[f32], cam: &Camera) -> Option<Vec<StrokePoint>> {
    let ss = SS as f64;
    let n = c.points.len();
    let at = |i: isize| -> f32 {
        let i = if c.closed { i.rem_euclid(n as isize) } else { i.clamp(0, n as isize - 1) };
        v[i as usize]
    };
    let pts: Vec<StrokePoint> = (0..n)
        .map(|i| {
            let r = cam.project(c.points[i])?;
            let k = at(i as isize).max(at(i as isize - 1)).max(at(i as isize + 1));
            let near = m::pow(r.scale / c.sref, c.gamma).clamp(0.45, 1.9);
            let w = c.base * near * ss * c.taper.get(i).copied().unwrap_or(1.0);
            Some(([(r.x * ss) as f32, (r.y * ss) as f32], (0.5 * w) as f32 * k, r.depth as f32))
        })
        .collect();
    let last = if c.closed { n } else { n - 1 };
    // a scrap of a ball's or the torso's outline shorter than its own
    // width is a bead or noise at a cap's pole, not a line; a short tube
    // (a neck far away) is a real line and stays
    if !c.tube {
        let (mut len, mut width) = (0.0f32, 0.0f32);
        for i in 0..last {
            if let (Some((a, ra, _)), Some((b, rb, _))) = (pts[i], pts[(i + 1) % n]) {
                len += ((b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1])).sqrt();
                width = width.max(2.0 * ra.max(rb));
            }
        }
        // closed: a loop smaller than a few widths is noise at a cap's pole
        if len < if c.closed { 4.0 } else { 1.5 } * width {
            return None;
        }
    }
    Some(pts)
}

/// Strokes along the visible parts of curves (see `stroke_points`).
pub fn strokes(curves: &[Curve], vis: &[f32], cam: &Camera) -> Vec<Segment> {
    let mut out = Vec::new();
    let mut off = 0;
    for c in curves {
        let n = c.points.len();
        let v = &vis[off..off + n];
        off += n;
        let Some(pts) = stroke_points(c, v, cam) else { continue };
        let last = if c.closed { n } else { n - 1 };
        for i in 0..last {
            let (Some((a, ra, da)), Some((b, rb, db))) = (pts[i], pts[(i + 1) % n]) else { continue };
            if ra <= 0.0 && rb <= 0.0 {
                continue;
            }
            let j = (i + 1) % n;
            let hidden = (v[i] == 0.0) as u32 | (((v[j] == 0.0) as u32) << 1);
            out.push(Segment { a, b, r: [ra, rb], color: c.color, depth: [da, db], id: c.id, hidden });
        }
    }
    out
}

pub fn v4(v: Vec3) -> [f32; 4] {
    [v.x as f32, v.y as f32, v.z as f32, 0.0]
}

/// Deterministic drawing-space deformation shared by raster and vectors.
pub struct Drawing { sx: u64, sy: u64, seed: u64, amp: f64, scale: f64, pressure: f64 }
impl Drawing {
    pub fn new(shot: &Shot, f: &FrameState) -> Self {
        use mavian_core::seed;
        Self { sx: seed!(shot.seed,"boil-x",f.drawn), sy: seed!(shot.seed,"boil-y",f.drawn), seed: seed!(shot.seed,"pressure",f.drawn), amp: shot.style.boil, scale: shot.style.boil_scale.max(0.001), pressure: shot.style.pressure }
    }
    pub fn at(&self, x:f64,y:f64)->(f64,f64) {
        use mavian_core::noise::noise2;
        (x+self.amp*noise2(self.sx,x/self.scale,y/self.scale),y+self.amp*noise2(self.sy,x/self.scale,y/self.scale))
    }
    pub fn world(&self, cam:&Camera,p:Vec3)->Vec3 {
        cam.project(p).map_or(p, |q| { let (x,y)=self.at(q.x,q.y); cam.unproject(x,y,q.depth) })
    }
    pub fn pressure(&self,id:u32,d:f64)->f64 {
        (1.0+self.pressure*mavian_core::noise::noise1(mavian_core::seed!(self.seed,id),d/0.18)).max(0.05)
    }
}

/// Fire billboards clipped against the renderer's completed geometry buffers.
pub fn fires_svg(shot: &Shot, f: &FrameState, depth_at: impl Fn(u32,u32)->f32) -> String {
    use std::fmt::Write;
        let mut out = String::new();
        for (i,fire) in shot.fires.iter().enumerate() {
            let Some(base) = f.cam.project(fire.at) else { continue };
            let depth = (base.depth-fire.over).max(f.cam.near) as f32;
            let mut path = String::new();
            // Supersampled clip retains the renderer's edge antialiasing.
            for y in 0..shot.size.1*SS {
                let mut start = None;
                for x in 0..=shot.size.0*SS {
                    let visible = x < shot.size.0*SS && depth_at(x,y) >= depth;
                    if visible && start.is_none() { start = Some(x); }
                    if !visible && let Some(a) = start.take() {
                        let ss = SS as f64;
                        let _ = write!(path,"M{} {}h{}v{}h-{}Z",a as f64/ss,y as f64/ss,(x-a) as f64/ss,1.0/ss,(x-a) as f64/ss);
                    }
                }
            }
            let _ = write!(out,r#"<defs><clipPath id="fire-{i}"><path d="{path}"/></clipPath></defs><g clip-path="url(#fire-{i})">{}</g>"#,fire.svg(&f.cam,f.w));
        }
        out
}

/// Pixels and optional effect depth from the same geometry pass. SVG conversion
/// can run on a worker while the renderer moves on to the next frame.
pub struct RenderedFrame {
    pub rgba: Vec<u8>,
    pub fire_depth: Option<Vec<f32>>,
}
impl RenderedFrame {
    pub fn fires_svg(&self, shot: &Shot, f: &FrameState) -> String {
        self.fire_depth.as_ref().map_or_else(String::new,|depth|fires_svg(shot,f,|x,y|depth[(y*shot.size.0*SS+x) as usize]))
    }
}
