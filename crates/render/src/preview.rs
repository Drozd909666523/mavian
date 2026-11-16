//! SVG frame preview (a port of the MVP's stick drawing).
//!
//! A line is a chain of short segments, each with its own width; width
//! depends on depth. Under each line is a paper-coloured "halo": a nearer line
//! slightly breaks a farther one. Faces, props, attached wires and depth-clipped
//! fire are included in the sketch.

use mavian_core::m;
use mavian_core::noise::{noise1, noise2, smooth};
use mavian_core::{Fingerprint, StableHasher, Vec3, seed};
use mavian_rig::Part;
use mavian_scene::{Camera, FrameState, Ground, Projected, Shot, Style, TRAIL_STEPS};
use std::fmt::Write;

/// Drawing version: bump on any edit that changes the picture, otherwise
/// the cache will serve old frames.
pub const VERSION: &str = "preview-9";

/// bone subdivision step, world units
const STEP: f64 = 0.05;
/// distance from a joint at which the halo reaches full width
const HALO_RAMP: f64 = 0.10;
/// pixels per world unit in an "ordinary" wide shot
const SCALE0: f64 = 220.0;

/// Screen-space offset field. A new seed is a new "drawing".
struct Boil {
    amp: f64,
    scale: f64,
    sx: u64,
    sy: u64,
}

impl Boil {
    fn new(seed: u64, index: usize, amp: f64, scale: f64) -> Boil {
        Boil { amp, scale, sx: seed!(seed, "boil-x", index), sy: seed!(seed, "boil-y", index) }
    }

    fn at(&self, x: f64, y: f64) -> (f64, f64) {
        if self.amp <= 0.0 {
            return (x, y);
        }
        let (u, v) = (x / self.scale, y / self.scale);
        (x + self.amp * noise2(self.sx, u, v), y + self.amp * noise2(self.sy, u, v))
    }
}

/// How many pixels one unit of line width takes.
///
/// `sref` — scale at the figure's root point, `s` — at this point. A close-up
/// does not make the line as many times thicker (`zoom`), and near parts are thicker
/// than far ones more than geometry says (`depth_gamma`).
fn line_px(s: f64, sref: f64, st: &Style) -> f64 {
    if sref <= 1e-9 {
        return s;
    }
    let near = m::pow(s / sref, st.depth_gamma).clamp(0.45, 1.9);
    m::pow(sref, st.zoom) * m::pow(SCALE0, 1.0 - st.zoom) * near
}

fn path_d(pts: &[(f64, f64)], closed: bool) -> String {
    let mut d = String::from("M");
    for (i, (x, y)) in pts.iter().enumerate() {
        if i > 0 {
            d.push_str(" L");
        }
        let _ = write!(d, "{x:.2} {y:.2}");
    }
    if closed {
        d.push('Z');
    }
    d
}

/// Variable-width line: neighbouring segments of equal width share one path.
fn stroke(pts: &[(f64, f64)], widths: &[f64], color: &str) -> String {
    let mut out = String::new();
    let mut run: Vec<(f64, f64)> = Vec::new();
    let mut run_w = f64::NAN;
    let flush = |run: &[(f64, f64)], w: f64, out: &mut String| {
        if run.len() > 1 && w > 0.0 {
            let _ = write!(out, r#"<path d="{}" stroke="{color}" stroke-width="{w}"/>"#, path_d(run, false));
        }
    };
    for i in 0..pts.len().saturating_sub(1) {
        let w = ((widths[i] + widths[i + 1]) * 0.5 * 10.0).round() / 10.0;
        if w != run_w {
            flush(&run, run_w, &mut out);
            run = vec![pts[i]];
            run_w = w;
        }
        run.push(pts[i + 1]);
    }
    flush(&run, run_w, &mut out);
    out
}

fn ground_svg(g: &Ground, cam: &Camera, st: &Style, boil: &Boil, out: &mut Vec<String>) {
    let n = (g.size / g.step).round() as i64;
    let cuts = 8.max((g.size * 6.0) as i64);
    let (cx, cz) = g.center;
    let span: Vec<f64> = (0..=cuts).map(|k| -g.size + 2.0 * g.size * k as f64 / cuts as f64).collect();
    for i in -n..=n {
        let c = i as f64 * g.step;
        let lines = [
            span.iter().map(|&t| Vec3 { x: cx + c, y: 0.0, z: cz + t }).collect::<Vec<_>>(),
            span.iter().map(|&t| Vec3 { x: cx + t, y: 0.0, z: cz + c }).collect(),
        ];
        for line in lines {
            for run in runs(&line, cam, boil) {
                out.push(format!(r#"<path d="{}" stroke="{}" stroke-width="{}"/>"#, path_d(&run, false), st.ground, g.width));
            }
        }
    }
}

/// Polyline projection; invisible points break it into pieces.
fn runs(pts: &[Vec3], cam: &Camera, boil: &Boil) -> Vec<Vec<(f64, f64)>> {
    let mut out = Vec::new();
    let mut run = Vec::new();
    for &p in pts {
        match cam.project(p) {
            None => {
                if run.len() > 1 {
                    out.push(std::mem::take(&mut run));
                }
                run.clear();
            }
            Some(r) => run.push(boil.at(r.x, r.y)),
        }
    }
    if run.len() > 1 {
        out.push(run);
    }
    out
}

/// Shadow blob under the figure: the higher the jump, the smaller it is.
fn shadow_svg(center: Vec3, radius: f64, lift: f64, cam: &Camera, boil: &Boil) -> Option<String> {
    let r = radius * (1.0 - 0.5 * lift.clamp(0.0, 1.0));
    let ring: Vec<Vec3> = (0..20)
        .map(|k| {
            let a = std::f64::consts::TAU * k as f64 / 20.0;
            Vec3 { x: center.x + r * m::cos(a), y: 0.0, z: center.z + r * m::sin(a) }
        })
        .collect();
    let rs = runs(&ring, cam, boil);
    let first = rs.first().filter(|r| r.len() == 20)?;
    Some(format!(r##"<path d="{}" fill="#000" fill-opacity="0.10" stroke="none"/>"##, path_d(first, true)))
}

/// Parts of one character: (depth, svg).
fn actor_items(shot: &Shot, f: &FrameState, index: usize, under: &mut Vec<String>, items: &mut Vec<(f64, String)>) {
    let (st, cam) = (&shot.style, &f.cam);
    let actor = &shot.actors[index];
    let model = &actor.model;
    let fig = &f.figures[index];
    let boil = Boil::new(shot.seed, f.drawn, st.boil, st.boil_scale);
    let root = fig.joints[model.root];
    if shot.ground.as_ref().is_some_and(|g| g.shadow) && model.shadow > 0.0 {
        under.extend(shadow_svg(root, model.shadow * actor.scale, fig.lift, cam, &boil));
    }
    let head = model.parts.iter().find_map(|p| match p {
        Part::Head { joint, .. } => Some(*joint),
        _ => None,
    });
    let sref = cam.project(root).or_else(|| head.and_then(|h| cam.project(fig.joints[h]))).map_or(SCALE0, |r| r.scale);
    let ink = actor.color.as_deref().unwrap_or(&st.ink);
    let tag = seed!(shot.seed, "actor", index, f.drawn);
    for part in &model.parts {
        let first_item = items.len();
        match part {
            Part::Stick { name, a, b, w0, w1, free, dot, trim, .. } => {
                let (a, mut b) = (fig.joints[*a], fig.joints[*b]);
                if *trim > 0.0 {
                    let d = b - a;
                    let ln = d.length();
                    let cut = trim * actor.scale;
                    if ln <= cut {
                        continue;
                    }
                    b = a + d * ((ln - cut) / ln);
                }
                let ln = (b - a).length();
                let n = 2.max((ln / (STEP * actor.scale)).ceil() as usize);
                let press = seed!(tag, name.as_str());
                let (mut pts, mut widths, mut halos, mut depth) = (vec![], vec![], vec![], 0.0);
                for i in 0..=n {
                    let u = i as f64 / n as f64;
                    let Some(Projected { x, y, depth: z, scale: s }) = cam.project(a.lerp(b, u)) else { continue };
                    let mut w = st.line * actor.scale * (w0 + (w1 - w0) * u) * line_px(s, sref, st);
                    w *= 1.0 + st.pressure * noise1(press, u * ln / 0.18);
                    let ramp = HALO_RAMP * actor.scale;
                    let mut hw = smooth(u * ln / ramp);
                    if !free {
                        hw *= smooth((1.0 - u) * ln / ramp);
                    }
                    pts.push(boil.at(x, y));
                    widths.push(w);
                    // at a joint the halo is narrower than the line itself, otherwise a light fringe remains at its edge
                    halos.push(w - 1.5 + (2.0 * st.halo + 1.5) * hw);
                    depth += z;
                }
                if pts.len() < 2 {
                    continue;
                }
                let mut svg = stroke(&pts, &halos, &st.paper) + &stroke(&pts, &widths, ink);
                if *dot {
                    let ((x, y), r) = (pts[pts.len() - 1], widths[widths.len() - 1] * 0.95);
                    let _ = write!(svg, r#"<circle cx="{x:.2}" cy="{y:.2}" r="{r:.2}" fill="{ink}"/>"#);
                }
                items.push((depth / pts.len() as f64, svg));
            }
            Part::Head { joint, radius, line, filled, face: fc } => {
                let Some(r) = cam.project(fig.joints[*joint]) else { continue };
                let (x, y) = boil.at(r.x, r.y);
                let rad = radius * actor.scale * r.scale;
                let w = st.line * line * actor.scale * line_px(r.scale, sref, st);
                let fill = if *filled { ink } else { &st.paper };
                let mut face_svg = String::new();
                if !filled
                    && actor.face
                    && let Some(shapes) = mavian_scene::face(
                        cam,
                        fig.joints[*joint],
                        &fig.rots[*joint],
                        radius * actor.scale,
                        fc,
                        actor.face_trust,
                        actor.blink.at(f.w),
                    )
                {
                    for eye in shapes.eyes {
                        let pts: Vec<_> = eye.iter().map(|p| boil.at(p.0, p.1)).collect();
                        let _ = write!(face_svg, r#"<path d="{}" fill="{ink}" stroke="none"/>"#, path_d(&pts, true));
                    }
                    let mouth: Vec<_> = shapes.mouth.iter().map(|p| boil.at(p.0, p.1)).collect();
                    let _ = write!(face_svg, r#"<path d="{}" stroke="{ink}" stroke-width="{}"/>"#, path_d(&mouth, false), w * 0.5);
                }
                items.push((
                    r.depth,
                    format!(
                        r#"<circle cx="{x:.2}" cy="{y:.2}" r="{rad:.2}" fill="{}" stroke="{}" stroke-width="{:.1}"/><circle cx="{x:.2}" cy="{y:.2}" r="{rad:.2}" fill="{}" stroke="{ink}" stroke-width="{w:.1}"/>{face_svg}"#,
                        st.paper,
                        st.paper,
                        w + 2.0 * st.halo,
                        fill
                    ),
                ));
            }
            Part::Limb { chain, radii, .. } => {
                let segs: Vec<_> = (1..chain.len())
                    .map(|i| (fig.joints[chain[i - 1]], radii[i - 1] * actor.scale, fig.joints[chain[i]], radii[i] * actor.scale))
                    .collect();
                volume_items(cam, &boil, &segs, ink, items);
            }
            Part::Body { a, b, rx, .. } => {
                let segs = [(fig.joints[*a], rx.0 * actor.scale, fig.joints[*b], rx.1 * actor.scale)];
                volume_items(cam, &boil, &segs, ink, items);
            }
            Part::Ball { joint, radius } => {
                let p = fig.joints[*joint];
                volume_items(cam, &boil, &[(p, radius * actor.scale, p, radius * actor.scale)], ink, items);
            }
        }
        for p in &shot.props {
            if let Some(c) = p.contacts.iter().find(|c| c.2 == index && c.0 <= f.w && f.w <= c.1) {
                let hand = c.3;
                let touches = match part {
                    Part::Stick { b, .. } => {
                        *b == hand || model.parts.iter().any(|q| matches!(q,Part::Stick { a,b:end,.. } if *end == hand && a == b))
                    }
                    Part::Limb { chain, .. } => chain.contains(&hand),
                    Part::Ball { joint, .. } => *joint == hand,
                    _ => false,
                };
                if touches && let Some(q) = cam.project(p.rest(f.w, &shot.world()).0 + Vec3::UP * (p.shape.top() * 0.5)) {
                    for item in &mut items[first_item..] {
                        item.0 = item.0.max(q.depth + 0.001);
                    }
                }
            }
        }
    }
}

/// Volumes in the sketch: a plain ink silhouette per segment, no outline.
/// Outlining volumes is the line stage's job (05-architecture, §7).
fn volume_items(cam: &Camera, boil: &Boil, segs: &[(Vec3, f64, Vec3, f64)], ink: &str, items: &mut Vec<(f64, String)>) {
    for &(a, ra, b, rb) in segs {
        let (Some(pa), Some(pb)) = (cam.project(a), cam.project(b)) else { continue };
        let ((xa, ya), (xb, yb)) = (boil.at(pa.x, pa.y), boil.at(pb.x, pb.y));
        let w = ra * pa.scale + rb * pb.scale;
        items.push((
            (pa.depth + pb.depth) / 2.0,
            format!(r#"<path d="M{xa:.2} {ya:.2} L{xb:.2} {yb:.2}" stroke="{ink}" stroke-width="{w:.1}" opacity="0.85"/>"#),
        ));
    }
}

/// Labels: text at points of the world, as SVG elements (empty if none).
pub fn labels_svg(shot: &Shot, f: &FrameState) -> String {
    shot.labels.iter().filter_map(|l| l.svg(&f.cam, &shot.style.ink)).map(|t| t + "\n").collect()
}

/// What is laid over a frame: effects (fire) and labels, on a transparent
/// page; None if the shot has neither.
pub fn overlay_layer(shot: &Shot, f: &FrameState) -> Option<String> {
    overlay_with_fires(shot, f, &mavian_cpu::Cpu::new().fires_svg(shot, f))
}

pub fn overlay_with_fires(shot: &Shot, f: &FrameState, fires: &str) -> Option<String> {
    if shot.labels.is_empty() && shot.fires.is_empty() {
        return None;
    }
    let (w, h) = shot.size;
    Some(format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\">\n{fires}{}</svg>\n",
        labels_svg(shot, f)
    ))
}

/// The whole frame as SVG text.
pub fn svg(shot: &Shot, f: &FrameState) -> String {
    svg_layer(shot, f, true)
}

/// The frame as SVG; without `background` — only the characters on a
/// transparent page, to lay the sketch over other layers.
pub fn svg_layer(shot: &Shot, f: &FrameState, background: bool) -> String {
    svg_frame(shot, f, background, None)
}

pub fn svg_with_fires(shot: &Shot, f: &FrameState, background: bool, fires: &str) -> String {
    svg_frame(shot, f, background, Some(fires))
}

fn svg_frame(shot: &Shot, f: &FrameState, background: bool, fires: Option<&str>) -> String {
    let st = &shot.style;
    let mut under = Vec::new();
    if let Some(g) = shot.ground.as_ref().filter(|g| g.grid && background) {
        let boil = Boil::new(seed!(shot.seed, "ground"), f.n / g.draw_on.max(1) as usize, st.boil * g.boil, st.boil_scale);
        ground_svg(g, &f.cam, st, &boil, &mut under);
    }
    let mut items = Vec::new();
    for i in 0..shot.actors.len() {
        actor_items(shot, f, i, &mut under, &mut items);
    }
    let mesh = mavian_frame::build(shot, f);
    let first_prop = 1 + shot.actors.len() as u32 * mavian_frame::PARTS_PER_ACTOR;
    let mut prop_items = vec![String::new(); shot.props.len()];
    for tri in mesh.vertices.as_chunks::<3>().0.iter().chain(mesh.top.as_chunks::<3>().0).chain(mesh.contacts.as_chunks::<3>().0) {
        if tri[0].id < first_prop {
            continue;
        }
        let pts: Vec<_> = tri.iter().filter_map(|v| f.cam.project(Vec3 { x: v.pos[0] as f64, y: v.pos[1] as f64, z: v.pos[2] as f64 })).collect();
        if pts.len() != 3 {
            continue;
        }
        let xy: Vec<_> = pts.iter().map(|p| (p.x, p.y)).collect();
        let p = &shot.props[(tri[0].id - first_prop) as usize];
        let fill = p.fill.as_deref().unwrap_or(&st.paper);
        prop_items[(tri[0].id - first_prop) as usize]
            .push_str(&format!(r#"<path d="{}" fill="{fill}" stroke="{fill}" stroke-width="0.5"/>"#, path_d(&xy, true)));
    }
    for c in &mesh.curves {
        if c.id < first_prop {
            continue;
        }
        let pts: Vec<_> = c.points.iter().filter_map(|p| f.cam.project(*p)).collect();
        if pts.len() < 2 {
            continue;
        }
        let xy: Vec<_> = pts.iter().map(|p| (p.x, p.y)).collect();
        let col = c.color.map(|v| (v * 255.0).round() as u8);
        let ink = format!("#{:02x}{:02x}{:02x}", col[0], col[1], col[2]);
        let element = format!(r#"<path d="{}" stroke="{ink}" stroke-width="{}"/>"#, path_d(&xy, c.closed), c.base);
        if let Some(item) = prop_items.get_mut((c.id - first_prop) as usize) {
            item.push_str(&element);
        } else {
            items.push((pts.iter().map(|p| p.depth).sum::<f64>() / pts.len() as f64, element));
        }
    }
    for (k, (p, svg)) in shot.props.iter().zip(prop_items).enumerate() {
        let (pos, _) = shot.prop_rest(k, f.w);
        if let Some(q) = f.cam.project(pos + Vec3::UP * (p.shape.top() * 0.5)) {
            items.push((q.depth, svg));
        }
    }
    if background {
        let actor_shadows = shot.actors.iter().filter(|a| shot.ground.as_ref().is_some_and(|g| g.shadow) && a.model.shadow > 0.0).count();
        for s in mesh.shadows.iter().skip(actor_shadows) {
            let pts: Vec<_> = if s[3] > 0.0 {
                [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
                    .iter()
                    .map(|(x, z)| Vec3 {
                        x: s[0] as f64 + x * s[2] as f64 * s[4] as f64 - z * s[3] as f64 * s[5] as f64,
                        y: 0.0,
                        z: s[1] as f64 + x * s[2] as f64 * s[5] as f64 + z * s[3] as f64 * s[4] as f64,
                    })
                    .collect()
            } else {
                (0..24)
                    .map(|i| {
                        let a = std::f64::consts::TAU * i as f64 / 24.0;
                        Vec3 { x: s[0] as f64 + s[2] as f64 * m::cos(a), y: 0.0, z: s[1] as f64 + s[2] as f64 * m::sin(a) }
                    })
                    .collect()
            };
            let xy: Vec<_> = pts.iter().filter_map(|p| f.cam.project(*p).map(|q| (q.x, q.y))).collect();
            if xy.len() == pts.len() {
                under.push(format!(r##"<path d="{}" fill="#000" fill-opacity="0.1" stroke="none"/>"##, path_d(&xy, true)));
            }
        }
    }
    if !background {
        under.clear(); // shadows belong to the ground
    }
    // far first, near on top; equal depth keeps part order
    items.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (w, h) = shot.size;
    let page = if background { format!("<rect width=\"{w}\" height=\"{h}\" fill=\"{}\"/>\n", st.paper) } else { String::new() };
    let mut out = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\">\n{page}<g fill=\"none\" stroke-linecap=\"round\" stroke-linejoin=\"round\">\n"
    );
    for s in under.iter().chain(items.iter().map(|(_, s)| s)) {
        out.push_str(s);
        out.push('\n');
    }
    out.push_str("</g>\n");
    match fires {
        Some(svg) => out.push_str(svg),
        None => out.push_str(&mavian_cpu::Cpu::new().fires_svg_from_mesh(shot, f, &mesh)),
    }
    out.push_str(&labels_svg(shot, f));
    out.push_str("</svg>\n");
    out
}

/// Fingerprint of all inputs of a frame's picture. A frame with the same fingerprint
/// need not be drawn again.
pub fn frame_key(shot: &Shot, f: &FrameState) -> Fingerprint {
    let mut h = StableHasher::new();
    h.add(VERSION).add(&shot.size.0).add(&shot.size.1).add(&shot.seed);
    let st = &shot.style;
    h.add(&st.paper).add(&st.ink).add(&st.ground);
    for v in [st.line, st.depth_gamma, st.zoom, st.halo, st.boil, st.boil_scale, st.pressure] {
        h.f64(v);
    }
    match &shot.ground {
        None => {
            h.u64(0);
        }
        Some(g) => {
            h.u64(1).f64(g.size).f64(g.step).f64(g.width).add(&g.draw_on).add(&g.shadow);
            h.f64(g.center.0).f64(g.center.1).f64(g.boil).add(&g.grid);
            h.add(&(f.n / g.draw_on.max(1) as usize)); // which background drawing
        }
    }
    h.add(&f.drawn).add(&f.cam).add(&f.flash);
    for (i, (actor, fig)) in shot.actors.iter().zip(&f.figures).enumerate() {
        h.add(&actor.model.name).f64(actor.scale).add(&actor.color.clone().unwrap_or_default());
        h.add(&fig.joints).f64(fig.lift);
        // the face: head turn, blink and how it is drawn
        for r in &fig.rots {
            for row in r.0 {
                for v in row {
                    h.f64(v);
                }
            }
        }
        h.add(&actor.face).f64(actor.face_trust).f64(actor.blink.at(f.w));
        // speed lines come from where the joints were a moment ago
        h.f64(actor.trails);
        if actor.trails > 0.0 {
            for k in 1..=TRAIL_STEPS {
                h.add(&shot.drawn_figure(i, f.w - actor.trails * k as f64 / TRAIL_STEPS as f64, &f.cam).joints);
            }
        }
    }
    for l in &shot.labels {
        h.add(&l.text).add(&l.at).f64(l.size);
    }
    // props where they are now, and wires
    for (i, p) in shot.props.iter().enumerate() {
        let (pos, rot) = shot.prop_rest(i, f.w);
        h.add(&p.boil)
            .add(&format!("{:?}", p.contacts))
            .add(&p.name)
            .add(&format!("{:?}", p.shape))
            .add(&p.fill.clone().unwrap_or_default())
            .add(&p.shadow)
            .add(&pos);
        for row in rot.0 {
            for v in row {
                h.f64(v);
            }
        }
    }
    if !shot.fires.is_empty() {
        h.f64(f.w);
        for fire in &shot.fires {
            h.add(&format!("{fire:?}"));
        }
    }
    for wire in &shot.wires {
        h.add(&wire.color).f64(wire.width);
        h.add(&wire.at.as_ref().map(|p| p.resolve(&f.clock, shot)));
        for (a, b) in &wire.segments {
            h.add(a).add(b);
        }
    }
    h.finish()
}
