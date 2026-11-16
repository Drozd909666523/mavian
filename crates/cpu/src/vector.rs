//! The line stage as vectors: the frame's lines as SVG (05-architecture, §7:
//! "line art stays vector").
//!
//! The same lines as the raster, from the same data and rules: curve
//! visibility from the buffers, widths and scraps by `stroke_points`, the
//! face only on the visible head, stick halos fading at joined ends and over
//! sticks running along them. What the raster cuts pixel by pixel, vectors
//! cut by bisection: a visible run of a curve ends exactly where the curve
//! goes behind an occluder (to a fraction of a supersampled pixel).
//!
//! A line of changing width is a filled outline: offsets on both sides with
//! round caps. Sticks overlap by depth, far first, each over its paper halo
//! (as in the MVP's sketch); curves and the face are cut by visibility and
//! need no order.

use mavian_core::m;
use super::*;
use mavian_core::Vec3;
use mavian_frame::{Curve, StickLine, stroke_points};
use std::fmt::Write;

/// bisection steps for the end of a visible run: 1/4096 of the gap between
/// two curve points (~2 px), far below a pixel
const BISECT: usize = 12;

/// A point of a line in output px: x, y, radius.
type P = (f32, f32, f32);

/// A point of a stick on screen: x, y (output px), radius, depth.
type SP = (f32, f32, f32, f32);

fn hex(c: [f32; 4]) -> String {
    let u = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", u(c[0]), u(c[1]), u(c[2]))
}

/// The filled outline of a line of changing width: offsets on both sides
/// and round caps (open), or a ring (closed).
fn outline(pts: &[P], closed: bool) -> String {
    let mut q: Vec<P> = Vec::with_capacity(pts.len());
    for &p in pts {
        if q.last().is_none_or(|l: &P| m::hypotf(l.0 - p.0, l.1 - p.1) > 1e-3) {
            q.push(p);
        }
    }
    if closed && q.len() > 2 && m::hypotf(q[0].0 - q[q.len() - 1].0, q[0].1 - q[q.len() - 1].1) <= 1e-3 {
        q.pop();
    }
    let n = q.len();
    let mut d = String::new();
    if n == 0 {
        return d;
    }
    if n == 1 || (!closed && n == 2 && m::hypotf(q[0].0 - q[1].0, q[0].1 - q[1].1) < 1e-2) {
        let (x, y, r) = q[0];
        let _ = write!(d, "M{:.2} {:.2}m{:.2} 0a{r:.2} {r:.2} 0 1 0 {:.2} 0a{r:.2} {r:.2} 0 1 0 {:.2} 0Z", x, y, -r, 2.0 * r, -2.0 * r);
        return d;
    }
    // unit normal at each point, from the neighbours
    let normal = |i: usize| -> (f32, f32) {
        let (a, b) = if closed {
            (q[(i + n - 1) % n], q[(i + 1) % n])
        } else {
            (q[i.saturating_sub(1)], q[(i + 1).min(n - 1)])
        };
        let (tx, ty) = (b.0 - a.0, b.1 - a.1);
        let l = m::hypotf(tx, ty).max(1e-9);
        (-ty / l, tx / l)
    };
    let side = |i: usize, s: f32| {
        let (nx, ny) = normal(i);
        (q[i].0 + nx * q[i].2 * s, q[i].1 + ny * q[i].2 * s)
    };
    let pt = |d: &mut String, cmd: char, p: (f32, f32)| {
        let _ = write!(d, "{cmd}{:.2} {:.2}", p.0, p.1);
    };
    if closed {
        pt(&mut d, 'M', side(0, 1.0));
        for i in 1..n {
            pt(&mut d, 'L', side(i, 1.0));
        }
        d.push('Z');
        pt(&mut d, 'M', side(n - 1, -1.0));
        for i in (0..n - 1).rev() {
            pt(&mut d, 'L', side(i, -1.0));
        }
        d.push('Z');
        return d;
    }
    pt(&mut d, 'M', side(0, 1.0));
    for i in 1..n {
        pt(&mut d, 'L', side(i, 1.0));
    }
    let r = q[n - 1].2;
    let e = side(n - 1, -1.0);
    let _ = write!(d, "A{r:.2} {r:.2} 0 0 0 {:.2} {:.2}", e.0, e.1);
    for i in (0..n - 1).rev() {
        pt(&mut d, 'L', side(i, -1.0));
    }
    let r = q[0].2;
    let s = side(0, 1.0);
    let _ = write!(d, "A{r:.2} {r:.2} 0 0 0 {:.2} {:.2}Z", s.0, s.1);
    d
}

/// Maximal runs of `true` in `vis`; for a closed line a run may wrap around.
/// Each run: (first, last) indices, inclusive; (0, n-1) and `whole` when
/// everything is visible.
fn runs(vis: &[bool], closed: bool) -> (Vec<(usize, usize)>, bool) {
    let n = vis.len();
    if n == 0 {
        return (Vec::new(), false);
    }
    if vis.iter().all(|&v| v) {
        return (vec![(0, n - 1)], closed);
    }
    let mut out = Vec::new();
    // a closed line starts after a hidden point, so no run straddles the start
    let start = if closed { vis.iter().position(|&v| !v).map_or(0, |h| h + 1) } else { 0 };
    let mut i = 0;
    while i < n {
        if !vis[(start + i) % n] {
            i += 1;
            continue;
        }
        let first = (start + i) % n;
        while i + 1 < n && vis[(start + i + 1) % n] {
            i += 1;
        }
        out.push((first, (start + i) % n));
        i += 1;
    }
    (out, false)
}

impl Ctx<'_> {
    /// The point between a hidden `a` and a visible `b` of part `id` where
    /// the line goes behind an occluder: bisection along the 3D segment.
    fn edge3(&self, a: Vec3, b: Vec3, id: u32, seen: impl Fn(&Ctx, Vec3, u32) -> bool) -> Vec3 {
        let (mut lo, mut hi) = (a, b);
        for _ in 0..BISECT {
            let mid = lo.lerp(hi, 0.5);
            if seen(self, mid, id) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        hi
    }

    /// Is a screen point (output px) on the visible surface of part `id`?
    fn on_part(&self, x: f64, y: f64, id: u32) -> bool {
        let ss = SS as f64;
        let q = ((x * ss).floor() as i32, (y * ss).floor() as i32);
        self.inside(q) && self.id_at(q) == id
    }

    /// The same between two screen points (output px), for the face.
    fn edge2(&self, a: (f64, f64), b: (f64, f64), id: u32) -> (f64, f64) {
        let (mut lo, mut hi) = (a, b);
        for _ in 0..BISECT {
            let mid = ((lo.0 + hi.0) / 2.0, (lo.1 + hi.1) / 2.0);
            if self.on_part(mid.0, mid.1, id) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        hi
    }

    /// Is a point of a stick hidden by a part that is not a stick? Sticks
    /// overlap each other by drawing order, with halos.
    fn stick_seen(&self, p: Vec3, id: u32) -> bool {
        let s = self.project(V3 { x: p.x as f32, y: p.y as f32, z: p.z as f32 });
        if s[3] == 0.0 {
            return false;
        }
        let q = (s[0].floor() as i32, s[1].floor() as i32);
        if !self.inside(q) {
            return false;
        }
        let b = self.id_at(q);
        if b == 0 || self.parts[b as usize].halo != 0 {
            return true;
        }
        !self.occludes(b, self.depth_at(q), id, s[2])
    }
}

/// Visible runs of a curve as outlines.
fn curve_paths(cx: &Ctx, c: &Curve, cam: &mavian_scene::Camera, out: &mut Vec<String>) {
    let v: Vec<f32> = c.points.iter().map(|p| cx.visible(*p, c.id) as u32 as f32).collect();
    let Some(pts) = stroke_points(c, &v, cam) else { return };
    let ss = SS as f32;
    let n = pts.len();
    let vis: Vec<bool> = (0..n).map(|i| v[i] > 0.0 && pts[i].is_some()).collect();
    let (rs, whole) = runs(&vis, c.closed);
    let at = |i: usize| -> P {
        let (p, r, _) = pts[i].expect("visible points are projected");
        (p[0] / ss, p[1] / ss, r / ss)
    };
    let edge = |hidden: usize, seen: usize| -> Option<P> {
        let e = cx.edge3(c.points[hidden], c.points[seen], c.id, |cx, p, id| cx.visible(p, id));
        let r = cam.project(e)?;
        Some((r.x as f32, r.y as f32, at(seen).2))
    };
    for (first, last) in rs {
        let mut line = Vec::new();
        let prev = if first > 0 { Some(first - 1) } else if c.closed { Some(n - 1) } else { None };
        if !whole && let Some(h) = prev.filter(|&h| pts[h].is_some()) {
            line.extend(edge(h, first));
        }
        let mut i = first;
        loop {
            line.push(at(i));
            if i == last {
                break;
            }
            i = (i + 1) % n;
        }
        let next = if last + 1 < n { Some(last + 1) } else if c.closed { Some(0) } else { None };
        if !whole && let Some(h) = next.filter(|&h| pts[h].is_some()) {
            line.extend(edge(h, last));
        }
        out.push(format!(r#"<path d="{}" fill="{}"/>"#, outline(&line, whole), hex(c.color)));
    }
}

/// The face: the mouth's visible runs and the eyes cut to the visible head.
fn face_paths(cx: &Ctx, f: &mavian_frame::FaceDraw, out: &mut Vec<String>) {
    let col = hex(f.color);
    let mouth = &f.shapes.mouth;
    let r = (0.5 * f.mouth_w) as f32;
    let vis: Vec<bool> = mouth.iter().map(|p| cx.on_part(p.0, p.1, f.id)).collect();
    let (rs, _) = runs(&vis, false);
    for (first, last) in rs {
        let mut line: Vec<P> = Vec::new();
        let p = |q: (f64, f64)| (q.0 as f32, q.1 as f32, r);
        if first > 0 {
            line.push(p(cx.edge2(mouth[first - 1], mouth[first], f.id)));
        }
        line.extend(mouth[first..=last].iter().map(|&q| p(q)));
        if last + 1 < mouth.len() {
            line.push(p(cx.edge2(mouth[last + 1], mouth[last], f.id)));
        }
        out.push(format!(r#"<path d="{}" fill="{col}"/>"#, outline(&line, false)));
    }
    for eye in &f.shapes.eyes {
        let n = eye.len();
        let vis: Vec<bool> = eye.iter().map(|p| cx.on_part(p.0, p.1, f.id)).collect();
        if !vis.iter().any(|&v| v) {
            continue;
        }
        // the part of the eye on the visible head; where an occluder cuts it,
        // a chord between the two cut points
        let mut poly = Vec::new();
        for i in 0..n {
            let j = (i + 1) % n;
            if vis[i] {
                poly.push(eye[i]);
            }
            if vis[i] != vis[j] {
                let (h, s) = if vis[i] { (eye[j], eye[i]) } else { (eye[i], eye[j]) };
                poly.push(cx.edge2(h, s, f.id));
            }
        }
        let mut d = String::new();
        for (k, p) in poly.iter().enumerate() {
            let _ = write!(d, "{}{:.2} {:.2}", if k == 0 { 'M' } else { 'L' }, p.0, p.1);
        }
        d.push('Z');
        out.push(format!(r#"<path d="{d}" fill="{col}"/>"#));
    }
}

/// A stick on screen: points along its axis (output px, radius), depth.
struct StickView {
    pts: Vec<(Vec3, Option<SP>)>,
    depth: f32,
    a2: (f32, f32),
    b2: (f32, f32),
    r: f32,
}

fn stick_view(s: &StickLine, cam: &mavian_scene::Camera, drawing: &mavian_frame::Drawing) -> StickView {
    let (pa, pb) = (cam.project(s.a), cam.project(s.b));
    let len = match (pa, pb) {
        (Some(a), Some(b)) => m::hypot(a.x - b.x, a.y - b.y),
        _ => 50.0,
    };
    let k = ((len / 2.0).ceil() as usize).max(2);
    let pts: Vec<_> = (0..=k)
        .map(|i| {
            let u = i as f64 / k as f64;
            let p = s.a.lerp(s.b, u);
            let r = (s.ra + (s.rb - s.ra) * u)*drawing.pressure(s.id,u*(s.b-s.a).length());
            let p = drawing.world(cam,p);
            (p, cam.project(p).map(|q| (q.x as f32, q.y as f32, (r * q.scale) as f32, q.depth as f32)))
        })
        .collect();
    let seen: Vec<f32> = pts.iter().filter_map(|p| p.1.map(|q| q.3)).collect();
    let depth = if seen.is_empty() { f32::MAX } else { seen.iter().sum::<f32>() / seen.len() as f32 };
    let xy = |p: Option<mavian_scene::Projected>| p.map_or((0.0, 0.0), |q| (q.x as f32, q.y as f32));
    let r = pts.iter().filter_map(|p| p.1.map(|q| q.2)).fold(0.0f32, f32::max);
    StickView { pts, depth, a2: xy(pa), b2: xy(pb), r }
}

fn seg_dist(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let (abx, aby) = (b.0 - a.0, b.1 - a.1);
    let t = (((p.0 - a.0) * abx + (p.1 - a.1) * aby) / (abx * abx + aby * aby).max(1e-9)).clamp(0.0, 1.0);
    m::hypotf(p.0 - a.0 - abx * t, p.1 - a.1 - aby * t)
}

fn dir(a: (f32, f32), b: (f32, f32)) -> Option<(f32, f32)> {
    let (x, y) = (b.0 - a.0, b.1 - a.1);
    let l = m::hypotf(x, y);
    (l > 1e-3).then(|| (x / l, y / l))
}

/// Sticks far first, each over its halo; visible where no volume hides them.
fn stick_paths(cx: &Ctx, sticks: &[StickLine], cam: &mavian_scene::Camera, drawing: &mavian_frame::Drawing, out: &mut Vec<String>) {
    let views: Vec<StickView> = sticks.iter().map(|s| stick_view(s, cam, drawing)).collect();
    let mut order: Vec<usize> = (0..sticks.len()).collect();
    order.sort_by(|&a, &b| views[b].depth.total_cmp(&views[a].depth));
    for &i in &order {
        let (s, v) = (&sticks[i], &views[i]);
        let vis: Vec<bool> = v.pts.iter().map(|(p, q)| q.is_some() && cx.stick_seen(*p, s.id)).collect();
        let (rs, _) = runs(&vis, false);
        let my_dir = dir(v.a2, v.b2);
        // halo radius at a point: the line plus the halo, fading at joined
        // ends and over farther sticks running along this one
        let halo = |p: Vec3, q: SP| -> f32 {
            let mut k = smoothstep(0.0, s.ramp as f32, (p - s.a).length() as f32);
            if !s.free {
                k *= smoothstep(0.0, s.ramp as f32, (p - s.b).length() as f32);
            }
            for (j, o) in views.iter().enumerate() {
                if j == i || o.depth <= v.depth {
                    continue;
                }
                if seg_dist((q.0, q.1), o.a2, o.b2) > q.2 + s.halo as f32 + o.r {
                    continue;
                }
                if let (Some(da), Some(db)) = (my_dir, dir(o.a2, o.b2)) {
                    k = k.min(smoothstep(0.966, 0.866, (da.0 * db.0 + da.1 * db.1).abs()));
                }
            }
            // at a joint the halo is narrower than the line (as in the MVP):
            // otherwise its soft edge shows as a light seam around the joint
            q.2 - 0.75 + (s.halo as f32 + 0.75) * k
        };
        for (first, last) in rs {
            let mut line: Vec<(Vec3, SP)> = Vec::new();
            let edge = |h: usize, seen: usize, line: &mut Vec<(Vec3, SP)>| {
                let e = cx.edge3(v.pts[h].0, v.pts[seen].0, s.id, |cx, p, id| cx.stick_seen(p, id));
                if let (Some(q), Some(sq)) = (cam.project(e), v.pts[seen].1) {
                    line.push((e, (q.x as f32, q.y as f32, sq.2, q.depth as f32)));
                }
            };
            if first > 0 {
                edge(first - 1, first, &mut line);
            }
            line.extend(v.pts[first..=last].iter().map(|(p, q)| (*p, q.expect("visible points are projected"))));
            if last + 1 < v.pts.len() {
                edge(last + 1, last, &mut line);
            }
            let halo_line: Vec<P> = line.iter().map(|&(p, q)| (q.0, q.1, halo(p, q))).collect();
            let ink: Vec<P> = line.iter().map(|&(_, q)| (q.0, q.1, q.2)).collect();
            out.push(format!(r#"<path d="{}" fill="{}"/>"#, outline(&halo_line, false), hex(s.halo_color)));
            out.push(format!(r#"<path d="{}" fill="{}"/>"#, outline(&ink, false), hex(s.color)));
        }
    }
}

impl Cpu {
    /// The frame's lines as SVG: volume outlines, the face, sticks with their
    /// halos — on a transparent page, `background` adds the paper.
    pub fn lines_svg(&self, shot: &Shot, f: &FrameState, background: bool) -> String {
        let mesh = build(shot, f);
        let cx = self.geometry(shot, f, &mesh, Layers { paint: true, lines: true, ids: false });
        let mut items = Vec::new();
        for c in &mesh.curves {
            curve_paths(&cx, c, &f.cam, &mut items);
        }
        for face in &mesh.faces {
            face_paths(&cx, face, &mut items);
        }
        stick_paths(&cx, &mesh.sticks, &f.cam, &mavian_frame::Drawing::new(shot,f), &mut items);
        let (w, h) = shot.size;
        let mut out = format!(r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">"#);
        out.push('\n');
        if background {
            let _ = writeln!(out, r#"<rect width="{w}" height="{h}" fill="{}"/>"#, hex(cx.scene.paper));
        }
        for s in items {
            out.push_str(&s);
            out.push('\n');
        }
        for l in &shot.labels {
            if let Some(t) = l.svg(&f.cam, &shot.style.ink) {
                out.push_str(&t);
                out.push('\n');
            }
        }
        out.push_str("</svg>\n");
        out
    }
}
