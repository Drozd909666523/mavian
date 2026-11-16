//! The CPU reference renderer (05-architecture, §5): the GPU's stages in plain
//! Rust, pass by pass, in f32 like the shader — to check the GPU and to draw
//! where there is none.
//!
//! The input is the same (`mavian_frame`: geometry, uniforms, strokes and
//! faces), and every function below mirrors one in `shader.wgsl` under the
//! same name. Rows are drawn in parallel bands; each pixel depends only on
//! the input and on earlier passes, and triangles are taken in submission
//! order within a band, so the result does not depend on the thread count.
//!
//! What may differ from a GPU: rasterization rules at triangle edges (GPUs
//! snap vertices to a sub-pixel grid), the last bits of `pow`, `atan2`,
//! `sqrt` and the half-float rounding of the stroke buffer. `mavian compare`
//! measures that.

use mavian_core::m;
use mavian_frame::{
    FrameMesh, CamU, FillVertex, Layers, PartStyle, Renderer, SS, STEPS, SceneU, Segment, Vertex, build, faces, strokes, uniforms,
};
use mavian_scene::{FrameState, Shot};

/// Version of the CPU drawing: bump with the GPU's on any edit that changes the picture.
pub const VERSION: &str = "cpu-9";

mod vector;

const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct V2 {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct V3 {
    x: f32,
    y: f32,
    z: f32,
}

fn v2(x: f32, y: f32) -> V2 {
    V2 { x, y }
}

fn v3(a: [f32; 4]) -> V3 {
    V3 { x: a[0], y: a[1], z: a[2] }
}

impl V2 {
    fn dot(self, o: V2) -> f32 {
        self.x * o.x + self.y * o.y
    }
    fn len(self) -> f32 {
        self.dot(self).sqrt()
    }
    fn sub(self, o: V2) -> V2 {
        v2(self.x - o.x, self.y - o.y)
    }
    fn add(self, o: V2) -> V2 {
        v2(self.x + o.x, self.y + o.y)
    }
    fn mul(self, k: f32) -> V2 {
        v2(self.x * k, self.y * k)
    }
}

impl V3 {
    fn dot(self, o: V3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    fn len(self) -> f32 {
        self.dot(self).sqrt()
    }
    fn sub(self, o: V3) -> V3 {
        V3 { x: self.x - o.x, y: self.y - o.y, z: self.z - o.z }
    }
    fn add(self, o: V3) -> V3 {
        V3 { x: self.x + o.x, y: self.y + o.y, z: self.z + o.z }
    }
    fn mul(self, k: f32) -> V3 {
        V3 { x: self.x * k, y: self.y * k, z: self.z * k }
    }
    fn norm(self) -> V3 {
        self.mul(1.0 / self.len())
    }
}

fn distance(a: V3, b: V3) -> f32 {
    a.sub(b).len()
}

/// WGSL `smoothstep`, also with e0 > e1.
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

/// Round to a half float and back: the stroke buffer is Rgba16Float on the GPU.
fn half(v: f32) -> f32 {
    if v == 0.0 || !v.is_finite() {
        return v;
    }
    let bits = v.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    if exp < -14 {
        // subnormal half: a fixed step of 2^-24
        let step = m::powif(2f32, -24);
        return (v / step).round_ties_even() * step;
    }
    // keep 10 mantissa bits, round to nearest even
    let drop = 13;
    let mask = (1u32 << drop) - 1;
    let rest = bits & mask;
    let mut base = bits & !mask;
    let halfway = 1u32 << (drop - 1);
    if rest > halfway || (rest == halfway && (base >> drop) & 1 == 1) {
        base += 1 << drop;
    }
    f32::from_bits(base).min(65504.0)
}

/// The frame's buffers, supersampled.
struct Bufs {
    w: i32,
    h: i32,
    gid: Vec<u32>,
    gdepth: Vec<f32>,
    gid_top: Vec<u32>,
    gdepth_top: Vec<f32>,
    seeds: Vec<u32>,
    strength: Vec<f32>,
    strokes: Vec<[f32; 4]>,
}

/// The shader's world: uniforms, part table, buffers.
struct Ctx<'a> {
    cam: CamU,
    scene: SceneU,
    parts: &'a [PartStyle],
    b: Bufs,
}

/// Run `f(y, row)` over the rows of `buf` (width `w`) in parallel bands.
fn rows<T: Send>(buf: &mut [T], w: usize, threads: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let h = buf.len() / w;
    let band = h.div_ceil(threads.max(1)).max(1);
    std::thread::scope(|s| {
        for (k, chunk) in buf.chunks_mut(band * w).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (i, row) in chunk.chunks_mut(w).enumerate() {
                    f(k * band + i, row);
                }
            });
        }
    });
}

/// Like `rows`, for two buffers of one size.
fn rows2<A: Send, B: Send>(a: &mut [A], b: &mut [B], w: usize, threads: usize, f: impl Fn(usize, &mut [A], &mut [B]) + Sync) {
    let h = a.len() / w;
    let band = h.div_ceil(threads.max(1)).max(1);
    std::thread::scope(|s| {
        for (k, (ca, cb)) in a.chunks_mut(band * w).zip(b.chunks_mut(band * w)).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (i, (ra, rb)) in ca.chunks_mut(w).zip(cb.chunks_mut(w)).enumerate() {
                    f(k * band + i, ra, rb);
                }
            });
        }
    });
}

// --- geometry ------------------------------------------------------------------

/// A vertex after `vs_geo`: clip position and the depth varying.
#[derive(Clone, Copy)]
struct Clip {
    pos: [f32; 4],
    depth: f32,
}

fn vs_geo(cam: &CamU, p: [f32; 3]) -> Clip {
    let d = V3 { x: p[0], y: p[1], z: p[2] }.sub(v3(cam.pos));
    let x = d.dot(v3(cam.right));
    let y = d.dot(v3(cam.up));
    let z = d.dot(v3(cam.fwd));
    let [f, cx, cy, fish] = cam.a;
    let [near, far, w, h] = cam.b;
    if fish > 0.5 {
        let rad = v2(x, y).len();
        let dist = V3 { x, y, z }.len();
        let theta = m::atan2f(rad, z);
        let mut k = f / dist.max(1e-6);
        if rad > 1e-9 {
            k = f * theta / rad;
        }
        let sx = cx + x * k;
        let sy = cy - y * k;
        Clip { pos: [sx / w * 2.0 - 1.0, 1.0 - sy / h * 2.0, (dist / far).clamp(0.0, 1.0), 1.0], depth: dist }
    } else {
        Clip {
            pos: [
                (2.0 * cx / w - 1.0) * z + 2.0 * f * x / w,
                (1.0 - 2.0 * cy / h) * z + 2.0 * f * y / h,
                far * (z - near) / (far - near),
                z,
            ],
            depth: z,
        }
    }
}

/// Clip a polygon to 0 ≤ z ≤ w (the near and far planes), as the hardware does.
fn clip_poly(poly: Vec<Clip>) -> Vec<Clip> {
    let mut out = poly;
    for plane in [0, 1] {
        let dist = |c: &Clip| if plane == 0 { c.pos[2] } else { c.pos[3] - c.pos[2] };
        let input = std::mem::take(&mut out);
        let n = input.len();
        for i in 0..n {
            let (a, b) = (input[i], input[(i + 1) % n]);
            let (da, db) = (dist(&a), dist(&b));
            if da >= 0.0 {
                out.push(a);
            }
            if (da >= 0.0) != (db >= 0.0) {
                let t = da / (da - db);
                let mut pos = [0.0; 4];
                for (k, v) in pos.iter_mut().enumerate() {
                    *v = mix(a.pos[k], b.pos[k], t);
                }
                out.push(Clip { pos, depth: mix(a.depth, b.depth, t) });
            }
        }
    }
    out
}

/// A triangle in screen space, ready to rasterize.
#[derive(Clone, Copy)]
struct Tri {
    id: u32,
    p: [V2; 3],
    /// depth test value (z / w), linear on screen
    z: [f32; 3],
    /// 1 / w and depth / w: the depth varying is perspective-correct
    iw: [f32; 3],
    dw: [f32; 3],
    min_y: f32,
    max_y: f32,
}

fn setup(cam: &CamU, verts: &[Vertex]) -> Vec<Tri> {
    let (w, h) = (cam.b[2], cam.b[3]);
    let mut out = Vec::with_capacity(verts.len() / 3);
    for t in verts.as_chunks::<3>().0 {
        let poly: Vec<Clip> = t.iter().map(|v| vs_geo(cam, v.pos)).collect();
        let poly = clip_poly(poly);
        if poly.len() < 3 {
            continue;
        }
        let screen: Vec<(V2, f32, f32, f32)> = poly
            .iter()
            .map(|c| {
                let iw = 1.0 / c.pos[3];
                let (nx, ny, nz) = (c.pos[0] * iw, c.pos[1] * iw, c.pos[2] * iw);
                (v2((nx + 1.0) * 0.5 * w, (1.0 - ny) * 0.5 * h), nz, iw, c.depth * iw)
            })
            .collect();
        for k in 1..screen.len() - 1 {
            let (a, b, c) = (screen[0], screen[k], screen[k + 1]);
            let p = [a.0, b.0, c.0];
            out.push(Tri {
                id: t[0].id,
                p,
                z: [a.1, b.1, c.1],
                iw: [a.2, b.2, c.2],
                dw: [a.3, b.3, c.3],
                min_y: p[0].y.min(p[1].y).min(p[2].y),
                max_y: p[0].y.max(p[1].y).max(p[2].y),
            });
        }
    }
    out
}

/// Is the edge a→b a top or left edge (the D3D/Vulkan fill rule) for a
/// triangle of the given winding?
fn top_left(a: V2, b: V2, ccw: bool) -> bool {
    let (dx, dy) = if ccw { (b.x - a.x, b.y - a.y) } else { (a.x - b.x, a.y - b.y) };
    (dy == 0.0 && dx < 0.0) || dy > 0.0
}

/// Pixels of row `y` covered by triangle `t`: calls `f(x, z, depth)` per
/// covered pixel centre.
fn raster_row(t: &Tri, y: usize, w: usize, mut f: impl FnMut(usize, f32, f32)) {
    let py = y as f32 + 0.5;
    if py < t.min_y || py > t.max_y {
        return;
    }
    let [a, b, c] = t.p;
    let area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    if area == 0.0 || !area.is_finite() {
        return;
    }
    let ccw = area > 0.0;
    let edge = |p: V2, q: V2, x: f32| (q.x - p.x) * (py - p.y) - (q.y - p.y) * (x - p.x);
    let min_x = a.x.min(b.x).min(c.x).floor().max(0.0) as usize;
    let max_x = (a.x.max(b.x).max(c.x).ceil().max(0.0) as usize).min(w);
    let edges = [(b, c), (c, a), (a, b)];
    let tl = edges.map(|(p, q)| top_left(p, q, ccw));
    for x in min_x..max_x {
        let px = x as f32 + 0.5;
        let mut l = [0.0f32; 3];
        let mut inside = true;
        for k in 0..3 {
            let (p, q) = edges[k];
            let e = edge(p, q, px) * area.signum();
            if e < 0.0 || (e == 0.0 && !tl[k]) {
                inside = false;
                break;
            }
            l[k] = e / area.abs();
        }
        if !inside {
            continue;
        }
        let z = l[0] * t.z[0] + l[1] * t.z[1] + l[2] * t.z[2];
        let iw = l[0] * t.iw[0] + l[1] * t.iw[1] + l[2] * t.iw[2];
        let dw = l[0] * t.dw[0] + l[1] * t.dw[1] + l[2] * t.dw[2];
        f(x, z, dw / iw);
    }
}

impl Ctx<'_> {
    fn size(&self) -> (i32, i32) {
        (self.b.w, self.b.h)
    }

    fn inside(&self, p: (i32, i32)) -> bool {
        let (w, h) = self.size();
        p.0 >= 0 && p.1 >= 0 && p.0 < w && p.1 < h
    }

    fn at(&self, p: (i32, i32)) -> usize {
        (p.1 * self.b.w + p.0) as usize
    }

    fn over(&self, a: u32, b: u32) -> bool {
        over(self.parts, a, b)
    }

    fn in_front(&self, a: u32, da: f32, b: u32, db: f32) -> bool {
        in_front(self.parts, a, da, b, db)
    }

    fn id_at(&self, p: (i32, i32)) -> u32 {
        let i = self.at(p);
        let t = self.b.gid_top[i];
        if t != 0 { t } else { self.b.gid[i] }
    }

    fn depth_at(&self, p: (i32, i32)) -> f32 {
        let i = self.at(p);
        if self.b.gid_top[i] != 0 { self.b.gdepth_top[i] } else { self.b.gdepth[i] }
    }

    fn ray(&self, px: V2) -> V3 {
        let c = &self.cam;
        let f = c.a[0];
        let dx = px.x - c.a[1];
        let dy = c.a[2] - px.y;
        if c.a[3] > 0.5 {
            let r = v2(dx, dy).len();
            let theta = r / f;
            if r < 1e-6 {
                return v3(c.fwd);
            }
            return v3(c.fwd).mul(m::cosf(theta)).add(v3(c.right).mul(dx).add(v3(c.up).mul(dy)).mul(m::sinf(theta) / r));
        }
        v3(c.fwd).add(v3(c.right).mul(dx / f)).add(v3(c.up).mul(dy / f)).norm()
    }

    fn world_at(&self, p: (i32, i32), depth: f32) -> V3 {
        let px = v2(p.0 as f32 + 0.5, p.1 as f32 + 0.5);
        let c = &self.cam;
        if c.a[3] > 0.5 {
            return v3(c.pos).add(self.ray(px).mul(depth));
        }
        let f = c.a[0];
        let dir = v3(c.fwd).add(v3(c.right).mul((px.x - c.a[1]) / f)).add(v3(c.up).mul((c.a[2] - px.y) / f));
        v3(c.pos).add(dir.mul(depth))
    }

    fn edge_strength(&self, p: (i32, i32), a: u32, da: f32, b: u32, db: f32) -> f32 {
        if a == b {
            return 0.0;
        }
        if a == 0 || b == 0 {
            return 1.0;
        }
        let parts = self.parts;
        let ga = parts[a as usize].group;
        if ga == 0 || ga != parts[b as usize].group || self.over(a, b) || self.over(b, a) {
            return 1.0;
        }
        let pa = parts[a as usize].parent;
        let pb = parts[b as usize].parent;
        let siblings = pa != 0 && pa == pb && distance(v3(parts[a as usize].root), v3(parts[b as usize].root)) < 1e-3;
        if pa != b && pb != a && !siblings {
            return 1.0;
        }
        let w = self.world_at(p, da);
        let mut away = 1.0f32;
        let mut zone = 0.0f32;
        for root in [parts[a as usize].root, parts[b as usize].root] {
            if root[3] > 0.0 {
                away = away.min(smoothstep(root[3], 1.5 * root[3], distance(w, v3(root))));
                zone = zone.max(root[3]);
            }
        }
        if away >= 1.0 {
            return 1.0;
        }
        if parts[a as usize].halo != 0 && parts[b as usize].halo != 0 {
            return away;
        }
        let tol = self.cam.c[3] + 4.0 * da.min(db) / self.cam.a[0];
        let gap = smoothstep(tol, tol + zone / 3.0, (da - db).abs());
        gap.max(away)
    }

    fn project(&self, p: V3) -> [f32; 4] {
        let c = &self.cam;
        let d = p.sub(v3(c.pos));
        let x = d.dot(v3(c.right));
        let y = d.dot(v3(c.up));
        let z = d.dot(v3(c.fwd));
        let f = c.a[0];
        if c.a[3] > 0.5 {
            let rad = v2(x, y).len();
            let dist = V3 { x, y, z }.len();
            if dist < c.b[0] {
                return [0.0; 4];
            }
            let theta = m::atan2f(rad, z);
            let mut k = f / dist;
            if rad > 1e-9 {
                k = f * theta / rad;
            }
            return [c.a[1] + x * k, c.a[2] - y * k, dist, 1.0];
        }
        if z < c.b[0] {
            return [0.0; 4];
        }
        [c.a[1] + x * f / z, c.a[2] - y * f / z, z, 1.0]
    }

    fn occludes(&self, b: u32, db: f32, own: u32, z: f32) -> bool {
        if b == 0 || b == own {
            return false;
        }
        let tol = self.cam.c[3] + 0.004 * z;
        if self.over(own, b) {
            return db < z - self.parts[own as usize].bias - tol;
        }
        if self.over(b, own) {
            return db - self.parts[b as usize].bias < z;
        }
        db < z - tol
    }

    /// `cs_visible`: is a curve point of part `id` seen in the frame?
    fn visible(&self, p: mavian_core::Vec3, id: u32) -> bool {
        let s = self.project(V3 { x: p.x as f32, y: p.y as f32, z: p.z as f32 });
        if s[3] == 0.0 {
            return false;
        }
        let q = (s[0].floor() as i32, s[1].floor() as i32);
        !(!self.inside(q) || self.occludes(self.id_at(q), self.depth_at(q), id, s[2]))
    }

    fn strength_near(&self, sp: (i32, i32)) -> f32 {
        let mut sum = 0.0f32;
        let mut n = 0.0f32;
        for dy in -3..=3 {
            for dx in -3..=3 {
                let q = (sp.0 + dx, sp.1 + dy);
                if !self.inside(q) {
                    continue;
                }
                let v = self.b.strength[self.at(q)];
                if v > 0.0 {
                    sum += v;
                    n += 1.0;
                }
            }
        }
        sum / n.max(1.0)
    }

    fn line_width(&self, id: u32, depth: f32) -> f32 {
        let pt = &self.parts[id as usize];
        let s = self.cam.c[0] / depth.max(1e-6);
        pt.base * m::powf(s / pt.sref, pt.gamma).clamp(0.45, 1.9) * self.cam.c[1]
    }

    fn stick_dir(&self, id: u32) -> V2 {
        let a = self.project(v3(self.parts[id as usize].root));
        let b = self.project(v3(self.parts[id as usize].tip));
        let d = v2(b[0] - a[0], b[1] - a[1]);
        if a[3] == 0.0 || b[3] == 0.0 || d.len() < 1e-3 {
            return v2(0.0, 0.0);
        }
        d.mul(1.0 / d.len())
    }

    fn crossing(&self, a: u32, b: u32) -> f32 {
        let da = self.stick_dir(a);
        let db = self.stick_dir(b);
        if da.dot(da) == 0.0 || db.dot(db) == 0.0 {
            return 1.0;
        }
        smoothstep(0.966, 0.866, da.dot(db).abs())
    }

    fn ground_hit(&self, px: V2) -> Option<V3> {
        let d = self.ray(px);
        if d.y > -1e-6 {
            return None;
        }
        let t = -self.cam.pos[1] / d.y;
        Some(v3(self.cam.pos).add(d.mul(t)))
    }

    fn background(&self, p: (i32, i32)) -> V3 {
        let sc = &self.scene;
        let mut col = v3(sc.paper);
        if sc.center[3] < 0.5 {
            return col;
        }
        let px = v2(p.0 as f32 + 0.5, p.1 as f32 + 0.5);
        let Some(g) = self.ground_hit(px) else { return col };
        if sc.grid[3] > 0.5 {
            let gx = self.ground_hit(px.add(v2(1.0, 0.0)));
            let gy = self.ground_hit(px.add(v2(0.0, 1.0)));
            let rel = v2(g.x - sc.center[0], g.z - sc.center[1]);
            let size = sc.grid[0];
            let stp = sc.grid[1];
            if let (Some(gx), Some(gy)) = (gx, gy)
                && rel.x.abs() <= size
                && rel.y.abs() <= size
            {
                let jx = v2(gx.x - g.x, gy.x - g.x);
                let jz = v2(gx.z - g.z, gy.z - g.z);
                let ux = rel.x / stp;
                let uz = rel.y / stp;
                let dist_x = (ux - ux.round()).abs() * stp / jx.len().max(1e-9);
                let dist_z = (uz - uz.round()).abs() * stp / jz.len().max(1e-9);
                let half = 0.5 * sc.grid[2] * self.cam.c[1];
                let fade_x = smoothstep(2.0, 6.0, stp / jx.len().max(1e-9));
                let fade_z = smoothstep(2.0, 6.0, stp / jz.len().max(1e-9));
                let cov = ((half + 0.5 - dist_x).clamp(0.0, 1.0) * fade_x).max((half + 0.5 - dist_z).clamp(0.0, 1.0) * fade_z);
                let gr = v3(sc.ground);
                col = V3 { x: mix(col.x, gr.x, cov), y: mix(col.y, gr.y, cov), z: mix(col.z, gr.z, cov) };
            }
        }
        let n = sc.center[2] as i32;
        for k in 0..n as usize {
            let s = sc.shadows[k];
            let (dx, dz) = (g.x - s[0], g.z - s[1]);
            let a = sc.shadow_axes[k];
            let hit = if s[3] > 0.0 { (dx*a[0]+dz*a[1]).abs() < s[2] && (-dx*a[1]+dz*a[0]).abs() < s[3] } else { v2(dx,dz).len() < s[2] };
            if hit {
                col = col.mul(0.9);
            }
        }
        if sc.glow[3] > 0.0 {
            let a = 0.55 * sc.glow[3] * (1.0 - v2(g.x, g.z).sub(v2(sc.glow[0], sc.glow[1])).len() / sc.glow[2]).max(0.0);
            let gc = v3(sc.glow_color);
            col = V3 { x: mix(col.x, gc.x, a), y: mix(col.y, gc.y, a), z: mix(col.z, gc.z, a) };
        }
        col
    }

    fn shade(&self, p: (i32, i32)) -> V3 {
        let a = self.id_at(p);
        let sc = &self.scene;
        let mut col = v3(sc.paper);
        if a == 0 {
            col = self.background(p);
        } else if sc.layers[2] > 0.5 {
            let h = a.wrapping_mul(2654435761);
            col = V3 { x: ((h >> 8) & 255) as f32, y: ((h >> 16) & 255) as f32, z: ((h >> 24) & 255) as f32 }
                .mul(1.0 / 255.0 * 0.6)
                .add(V3 { x: 0.35, y: 0.35, z: 0.35 });
        } else if sc.layers[0] > 0.5 {
            col = v3(self.parts[a as usize].fill);
        }
        if sc.layers[1] > 0.5 {
            let l = self.b.strokes[self.at(p)];
            col = col.mul(1.0 - l[3]).add(V3 { x: l[0], y: l[1], z: l[2] });
            let s = self.b.seeds[self.at(p)];
            if s != NONE {
                let sp = unpack(s);
                let owner = self.id_at(sp);
                let po = &self.parts[owner as usize];
                if po.halo == 0 || owner != a {
                    let ds = self.depth_at(sp);
                    let mut w = self.line_width(owner, ds) * self.strength_near(sp);
                    if po.halo != 0 {
                        let ws = self.world_at(sp, ds);
                        if po.root[3] > 0.0 {
                            w *= smoothstep(0.0, po.root[3], distance(ws, v3(po.root)));
                        }
                        if po.tip[3] > 0.0 {
                            w *= smoothstep(0.0, po.tip[3], distance(ws, v3(po.tip)));
                        }
                        if a != 0 && self.parts[a as usize].halo != 0 {
                            w *= self.crossing(owner, a);
                        }
                    }
                    let d = v2((p.0 - sp.0) as f32, (p.1 - sp.1) as f32).len();
                    let mut cov = (0.5 * w + 0.5 - d).clamp(0.0, 1.0);
                    if po.halo != 0 && a != 0 {
                        cov *= self.edge_strength(p, owner, ds, a, self.depth_at(p));
                    }
                    let lc = v3(po.line);
                    col = V3 { x: mix(col.x, lc.x, cov), y: mix(col.y, lc.y, cov), z: mix(col.z, lc.z, cov) };
                }
            }
        }
        col
    }
}

fn over(parts: &[PartStyle], a: u32, b: u32) -> bool {
    if a == 0 || b == 0 {
        return false;
    }
    let rule = parts[a as usize]._pad;
    if rule[0] != 0 && b >= rule[0] && b < rule[0] + 64 {
        let k = b - rule[0];
        return if k < 32 { rule[1] & (1 << k) != 0 } else { rule[2] & (1 << (k-32)) != 0 };
    }
    if (a - 1) / 64 != (b - 1) / 64 {
        return false;
    }
    let k = (b - 1) % 64;
    let pa = &parts[a as usize];
    if k < 32 { pa.over_lo & (1 << k) != 0 } else { pa.over_hi & (1 << (k - 32)) != 0 }
}

fn in_front(parts: &[PartStyle], a: u32, da: f32, b: u32, db: f32) -> bool {
    if b == 0 {
        return true;
    }
    if over(parts, a, b) {
        return da - parts[a as usize].bias < db;
    }
    if over(parts, b, a) {
        return da < db - parts[b as usize].bias;
    }
    da < db
}

fn pack(p: (i32, i32)) -> u32 {
    p.0 as u32 | ((p.1 as u32) << 16)
}

fn unpack(s: u32) -> (i32, i32) {
    ((s & 0xffff) as i32, (s >> 16) as i32)
}

/// The CPU reference renderer.
pub struct Cpu {
    threads: usize,
}

impl Default for Cpu {
    fn default() -> Self {
        Cpu::new()
    }
}

impl Cpu {
    pub fn new() -> Cpu {
        Cpu { threads: std::thread::available_parallelism().map_or(4, |n| n.get()) }
    }

    /// With a fixed number of threads (the picture does not depend on it).
    pub fn with_threads(threads: usize) -> Cpu {
        Cpu { threads: threads.max(1) }
    }

    /// The geometry passes: part ID and depth buffers, with the parts on top
    /// by a rule against them.
    fn geometry<'a>(&self, shot: &Shot, f: &FrameState, mesh: &'a FrameMesh, layers: Layers) -> Ctx<'a> {
        let (cam, scene) = uniforms(shot, f, mesh, layers);
        let (w, h) = ((shot.size.0 * SS) as usize, (shot.size.1 * SS) as usize);
        let n = w * h;
        let threads = self.threads;
        let parts = &mesh.parts[..];
        // 1. geometry, then the parts on top by a rule against it
        let tris = setup(&cam, &mesh.vertices);
        let top = setup(&cam, &mesh.top);
        let mut gid = vec![0u32; n];
        let mut gdepth = vec![1e9f32; n];
        rows2(&mut gid, &mut gdepth, w, threads, |y, ids, depths| {
            let mut zbuf = vec![1.0f32; w];
            for t in &tris {
                raster_row(t, y, w, |x, z, d| {
                    if z < zbuf[x] {
                        zbuf[x] = z;
                        ids[x] = t.id;
                        depths[x] = d;
                    }
                });
            }
        });
        let mut gid_top = vec![0u32; n];
        let mut gdepth_top = vec![1e9f32; n];
        {
            let (gid, gdepth) = (&gid, &gdepth);
            rows2(&mut gid_top, &mut gdepth_top, w, threads, |y, ids, depths| {
                let mut zbuf = vec![1.0f32; w];
                for t in &top {
                    raster_row(t, y, w, |x, z, d| {
                        let i = y * w + x;
                        if !in_front(parts, t.id, d, gid[i], gdepth[i]) {
                            return; // discard: no depth write
                        }
                        if z < zbuf[x] {
                            zbuf[x] = z;
                            ids[x] = t.id;
                            depths[x] = d;
                        }
                    });
                }
            });
        }
        // Contact props must compare against the resolved actor, including hands
        // from its overlap pass. Preserve the nearest unrelated occluder.
        let contact_tris = setup(&cam,&mesh.contacts);
        if !contact_tris.is_empty() {
        rows2(&mut gid_top,&mut gdepth_top,w,threads,|y,ids,depths| {
            let prior_ids=ids.to_vec();let prior_depths=depths.to_vec();
            let mut zbuf=vec![1.0f32;w];
            for t in &contact_tris {
                raster_row(t,y,w,|x,z,d| {
                    let i=y*w+x;
                    let (id,depth)=if prior_ids[x]!=0 { (prior_ids[x],prior_depths[x]) } else { (gid[i],gdepth[i]) };
                    if in_front(parts,t.id,d,id,depth) && z<zbuf[x] {
                        zbuf[x]=z;ids[x]=t.id;depths[x]=d;
                    }
                });
            }
        });
        }
        Ctx {
            cam,
            scene,
            parts,
            b: Bufs {
                w: w as i32,
                h: h as i32,
                gid,
                gdepth,
                gid_top,
                gdepth_top,
                seeds: Vec::new(),
                strength: Vec::new(),
                strokes: vec![[0.0; 4]; n],
            },
        }
    }

    pub fn render(&self, shot: &Shot, f: &FrameState, layers: Layers) -> Vec<u8> {
        self.render_frame(shot,f,layers,false).rgba
    }

    pub fn render_with_effects(&self, shot: &Shot, f: &FrameState, layers: Layers) -> (Vec<u8>, String) {
        let frame=self.render_with_depth(shot,f,layers);
        let svg=frame.fires_svg(shot,f);
        (frame.rgba,svg)
    }

    pub fn render_with_depth(&self, shot: &Shot, f: &FrameState, layers: Layers) -> mavian_frame::RenderedFrame {
        self.render_frame(shot,f,layers,true)
    }

    fn render_frame(&self, shot: &Shot, f: &FrameState, layers: Layers, effects: bool) -> mavian_frame::RenderedFrame {
        let mesh = build(shot, f);
        let (w, h) = ((shot.size.0 * SS) as usize, (shot.size.1 * SS) as usize);
        let n = w * h;
        let threads = self.threads;
        // 1. geometry, then the parts on top by a rule against it
        let mut cx = self.geometry(shot, f, &mesh, layers);

        // 2. edges: line seeds of parts with an outline (stick halos)
        let mut seeds = vec![NONE; n];
        let mut strength = vec![0.0f32; n];
        {
            let c = &cx;
            rows2(&mut seeds, &mut strength, w, threads, |y, srow, strow| {
                for x in 0..w {
                    let p = (x as i32, y as i32);
                    let a = c.id_at(p);
                    if a == 0 || c.parts[a as usize].base <= 0.0 {
                        continue;
                    }
                    let da = c.depth_at(p);
                    let mut s = 0.0f32;
                    for o in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                        let q = (p.0 + o.0, p.1 + o.1);
                        if !c.inside(q) {
                            continue;
                        }
                        let b = c.id_at(q);
                        let db = c.depth_at(q);
                        if c.in_front(a, da, b, db) {
                            s = s.max(c.edge_strength(p, a, da, b, db));
                        }
                    }
                    if s > 0.0 {
                        srow[x] = pack(p);
                        strow[x] = s;
                    }
                }
            });
        }
        cx.b.strength = strength;

        // 3. jump flood
        let (wi, hi) = (w as i32, h as i32);
        for &step in &STEPS {
            let prev = seeds;
            let mut next = vec![NONE; n];
            rows(&mut next, w, threads, |y, row| {
                for (x, out) in row.iter_mut().enumerate() {
                    let p = (x as i32, y as i32);
                    let mut best = NONE;
                    let mut bestd = 1e30f32;
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let q = (p.0 + dx * step, p.1 + dy * step);
                            if q.0 < 0 || q.1 < 0 || q.0 >= wi || q.1 >= hi {
                                continue;
                            }
                            let s = prev[(q.1 * wi + q.0) as usize];
                            if s == NONE {
                                continue;
                            }
                            let u = unpack(s);
                            let d = v2((u.0 - p.0) as f32, (u.1 - p.1) as f32);
                            let dd = d.dot(d);
                            if dd < bestd {
                                bestd = dd;
                                best = s;
                            }
                        }
                    }
                    *out = best;
                }
            });
            seeds = next;
        }
        cx.b.seeds = seeds;

        // 4. curve visibility, then strokes and faces (shared with the GPU)
        let mut segments = Vec::new();
        let mut fills: Vec<FillVertex> = Vec::new();
        if layers.lines {
            let vis: Vec<f32> = mesh.curves.iter().flat_map(|c| c.points.iter().map(|p| cx.visible(*p, c.id) as u32 as f32)).collect();
            let (s, fl) = faces(&mesh.faces);
            segments = s;
            fills = fl;
            segments.extend(strokes(&mesh.curves, &vis, &f.cam));
        }

        // 5. strokes and fills into the stroke buffer (max blend)
        let mut buf = std::mem::take(&mut cx.b.strokes);
        {
            let c = &cx;
            let fill_tris: Vec<(V2, V2, V2, [f32; 4], u32)> = fills
                .as_chunks::<3>().0.iter()
                .map(|t| (v2(t[0].pos[0], t[0].pos[1]), v2(t[1].pos[0], t[1].pos[1]), v2(t[2].pos[0], t[2].pos[1]), t[0].color, t[0].id))
                .collect();
            rows(&mut buf, w, threads, |y, row| {
                for s in &segments {
                    stroke_row(c, s, y, row);
                }
                for &(a, b, cc, color, id) in &fill_tris {
                    let t = Tri {
                        id,
                        p: [a, b, cc],
                        z: [0.0; 3],
                        iw: [1.0; 3],
                        dw: [0.0; 3],
                        min_y: a.y.min(b.y).min(cc.y),
                        max_y: a.y.max(b.y).max(cc.y),
                    };
                    raster_row(&t, y, w, |x, _, _| {
                        if c.id_at((x as i32, y as i32)) != id {
                            return;
                        }
                        let px = &mut row[x];
                        for (k, v) in [color[0], color[1], color[2], 1.0].into_iter().enumerate() {
                            px[k] = px[k].max(half(v));
                        }
                    });
                }
            });
        }
        cx.b.strokes = buf;

        // 6. compose: the average of the supersampled pixels
        let (ow, oh) = (shot.size.0 as usize, shot.size.1 as usize);
        let mut out = vec![0u8; ow * oh * 4];
        let ss = SS as i32;
        {
            let c = &cx;
            rows(&mut out, ow * 4, threads, |y, row| {
                for x in 0..ow {
                    let mut acc = V3::default();
                    for sy in 0..ss {
                        for sx in 0..ss {
                            acc = acc.add(c.shade((x as i32 * ss + sx, y as i32 * ss + sy)));
                        }
                    }
                    let col = acc.mul(1.0 / (ss * ss) as f32);
                    let u = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                    row[x * 4..x * 4 + 4].copy_from_slice(&[u(col.x), u(col.y), u(col.z), 255]);
                }
            });
        }
        let _ = oh;
        let fire_depth=(effects && !shot.fires.is_empty()).then(|| {
            (0..h).flat_map(|y| (0..w).map(move |x|(x,y))).map(|(x,y)|cx.depth_at((x as i32,y as i32))).collect()
        });
        mavian_frame::RenderedFrame { rgba:out,fire_depth }
    }
}

/// `fs_stroke` over one row: coverage of a tapered capsule, max-blended.
fn stroke_row(c: &Ctx, s: &Segment, y: usize, row: &mut [[f32; 4]]) {
    let a = v2(s.a[0], s.a[1]);
    let b = v2(s.b[0], s.b[1]);
    let pad = s.r[0].max(s.r[1]) + 1.5;
    let py = y as f32 + 0.5;
    if py < a.y.min(b.y) - pad || py > a.y.max(b.y) + pad {
        return;
    }
    let ab = b.sub(a);
    let len = ab.len();
    let dir = if len > 1e-6 { ab.mul(1.0 / len) } else { v2(1.0, 0.0) };
    let nrm = v2(-dir.y, dir.x);
    // the quad of vs_stroke: pixels outside it are not shaded at all
    let q0 = a.sub(dir.mul(pad));
    let span = len + 2.0 * pad;
    let x0 = (a.x.min(b.x) - pad).floor().max(0.0) as usize;
    let x1 = ((a.x.max(b.x) + pad).ceil().max(0.0) as usize).min(row.len());
    let abab = ab.dot(ab).max(1e-9);
    for (x, px_out) in row.iter_mut().enumerate().take(x1).skip(x0) {
        let px = v2(x as f32 + 0.5, py);
        let rel = px.sub(q0);
        let (u, v) = (rel.dot(dir), rel.dot(nrm));
        if u < 0.0 || u > span || v.abs() > pad {
            continue;
        }
        let raw = px.sub(a).dot(ab) / abab;
        let t = raw.clamp(0.0, 1.0);
        let r = mix(s.r[0], s.r[1], t);
        let d = px.sub(a.add(ab.mul(t))).len() - r;
        let cov = (0.5 - d).clamp(0.0, 1.0) * (r * 2.0).min(1.0);
        if cov <= 0.0 {
            continue;
        }
        let p = (x as i32, y as i32);
        if s.hidden & 4 != 0 && (!c.inside(p) || c.id_at(p) != s.id) {
            continue;
        }
        let into_hidden = (s.hidden & 1 != 0 && raw < 0.5) || (s.hidden & 2 != 0 && raw > 0.5);
        if into_hidden && c.inside(p) && c.occludes(c.id_at(p), c.depth_at(p), s.id, mix(s.depth[0], s.depth[1], t)) {
            continue;
        }
        let v = [s.color[0] * cov, s.color[1] * cov, s.color[2] * cov, cov];
        for k in 0..4 {
            px_out[k] = px_out[k].max(half(v[k]));
        }
    }
}

impl Renderer for Cpu {
    fn render(&mut self, shot: &Shot, f: &FrameState, layers: Layers) -> Vec<u8> {
        Cpu::render(self, shot, f, layers)
    }

    fn render_with_depth(&mut self, shot: &Shot, f: &FrameState, layers: Layers) -> mavian_frame::RenderedFrame {
        Cpu::render_with_depth(self,shot,f,layers)
    }
    fn device(&self) -> String {
        VERSION.to_string()
    }
}

impl Cpu {
    /// Clip each billboard against the same depth/ID buffers as the line stage.
    pub fn fires_svg(&self, shot: &Shot, f: &FrameState) -> String {
        if shot.fires.is_empty() { return String::new(); }
        let mesh = build(shot,f);
        self.fires_svg_from_mesh(shot,f,&mesh)
    }

    /// Sketch rendering already has a mesh; use it for the effect mask too.
    pub fn fires_svg_from_mesh(&self, shot: &Shot, f: &FrameState, mesh: &FrameMesh) -> String {
        if shot.fires.is_empty() { return String::new(); }
        let cx = self.geometry(shot,f,mesh,Layers { paint:true,lines:false,ids:false });
        mavian_frame::fires_svg(shot,f,|x,y| cx.depth_at((x as i32,y as i32)))
    }
}
