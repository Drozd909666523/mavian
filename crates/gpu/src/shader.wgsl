// mavian GPU frame: geometry → buffers (part ID, depth) → edges → jump flood → compose.
//
// Lines come from the buffers, not from each object: a line is drawn where the
// part on one side differs from the other, except between parts of one "skin"
// (an arm growing out of the torso) where the depth is continuous — there is
// no seam. The nearer part owns the line and sets its width and colour.

struct Cam {
    pos: vec4f,
    right: vec4f,
    up: vec4f,
    fwd: vec4f,
    a: vec4f,   // f (supersampled px), cx, cy, fisheye (0/1)
    b: vec4f,   // near, far, width, height (supersampled px)
    c: vec4f,   // f at output size, supersampling factor, fisheye cutoff, seam tolerance
}

struct Part {
    fill: vec4f,
    line: vec4f,
    base: f32,
    sref: f32,
    gamma: f32,
    group: u32,
    over_lo: u32,
    over_hi: u32,
    bias: f32,
    halo: u32,
    root: vec4f,
    parent: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
    tip: vec4f,
}

struct Scene {
    paper: vec4f,
    ground: vec4f,
    grid: vec4f,     // size, step, line width (output px), grid on (0/1)
    center: vec4f,   // grid centre x, z, shadow count, ground on (0/1)
    layers: vec4f,   // paint (0/1), lines (0/1)
    shadows: array<vec4f, 16>,   // x, z, radius
    shadow_axes: array<vec4f, 16>,
    glow: vec4f,       // a fire's glow on the ground: x, z, radius, brightness (0 — none)
    glow_color: vec4f,
}

const NONE: u32 = 0xffffffffu;

@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> parts: array<Part>;

// --- geometry ------------------------------------------------------------------

struct GeoOut {
    @builtin(position) pos: vec4f,
    @location(0) @interpolate(flat) id: u32,
    @location(1) depth: f32,
}

@vertex
fn vs_geo(@location(0) p: vec3f, @location(1) id: u32) -> GeoOut {
    let d = p - cam.pos.xyz;
    let x = dot(d, cam.right.xyz);
    let y = dot(d, cam.up.xyz);
    let z = dot(d, cam.fwd.xyz);
    let f = cam.a.x;
    let cx = cam.a.y;
    let cy = cam.a.z;
    let near = cam.b.x;
    let far = cam.b.y;
    let w = cam.b.z;
    let h = cam.b.w;
    var o: GeoOut;
    o.id = id;
    if (cam.a.w > 0.5) {
        // equidistant fisheye, as Camera::project; affine interpolation is fine
        // for the small triangles the mesh is made of
        let rad = length(vec2f(x, y));
        let dist = length(vec3f(x, y, z));
        let theta = atan2(rad, z);
        var k = f / max(dist, 1e-6);
        if (rad > 1e-9) {
            k = f * theta / rad;
        }
        let sx = cx + x * k;
        let sy = cy - y * k;
        o.pos = vec4f(sx / w * 2.0 - 1.0, 1.0 - sy / h * 2.0, clamp(dist / far, 0.0, 1.0), 1.0);
        o.depth = dist;
    } else {
        // pinhole in clip space: the hardware clips at the near plane
        o.pos = vec4f(
            (2.0 * cx / w - 1.0) * z + 2.0 * f * x / w,
            (1.0 - 2.0 * cy / h) * z + 2.0 * f * y / h,
            far * (z - near) / (far - near),
            z,
        );
        o.depth = z;
    }
    return o;
}

struct GeoTargets {
    @location(0) id: u32,
    @location(1) depth: f32,
}

@fragment
fn fs_geo(i: GeoOut) -> GeoTargets {
    return GeoTargets(i.id, i.depth);
}

// --- shared --------------------------------------------------------------------

// first pass: all parts except those on top by a rule
@group(0) @binding(2) var gid: texture_2d<u32>;
@group(0) @binding(3) var gdepth: texture_2d<f32>;
// second pass: parts on top by a rule (0 where none won)
@group(0) @binding(9) var gid_top: texture_2d<u32>;
@group(0) @binding(10) var gdepth_top: texture_2d<f32>;

/// Is part a over part b by an overlap rule (same actor)?
fn over(a: u32, b: u32) -> bool {
    if (a == 0u || b == 0u) {
        return false;
    }
    let rule = parts[a];
    if (rule.pad1 != 0u && b >= rule.pad1 && b < rule.pad1 + 64u) {
        let k = b - rule.pad1;
        if (k < 32u) { return (rule.pad2 & (1u << k)) != 0u; }
        return (rule.pad3 & (1u << (k - 32u))) != 0u;
    }
    let pa = (a - 1u) / 64u;
    if (pa != (b - 1u) / 64u) {
        return false;
    }
    let k = (b - 1u) % 64u;
    if (k < 32u) {
        return (parts[a].over_lo & (1u << k)) != 0u;
    }
    return (parts[a].over_hi & (1u << (k - 32u))) != 0u;
}

/// Is part a (at depth da) in front of b (at db), with overlap rules: a rule
/// lifts a part by its bias over the parts it is over — a nudge, not an
/// override, so a body that is clearly in front still hides it.
fn in_front(a: u32, da: f32, b: u32, db: f32) -> bool {
    if (b == 0u) {
        return true;
    }
    if (over(a, b)) {
        return da - parts[a].bias < db;
    }
    if (over(b, a)) {
        return da < db - parts[b].bias;
    }
    return da < db;
}

/// The second geometry pass: parts on top by a rule, against the first pass.
@fragment
fn fs_geo_top(i: GeoOut) -> GeoTargets {
    let p = vec2i(i.pos.xy);
    let b = textureLoad(gid, p, 0).x;
    let db = textureLoad(gdepth, p, 0).x;
    if (!in_front(i.id, i.depth, b, db)) {
        discard;
    }
    return GeoTargets(i.id, i.depth);
}

@group(0) @binding(16) var gid_contacts: texture_2d<u32>;
@group(0) @binding(17) var gdepth_contacts: texture_2d<f32>;

@fragment
fn fs_geo_contacts(i: GeoOut) -> GeoTargets {
    let p=vec2i(i.pos.xy);
    let top=textureLoad(gid_top,p,0).x;
    let b=select(textureLoad(gid,p,0).x,top,top!=0u);
    let db=select(textureLoad(gdepth,p,0).x,textureLoad(gdepth_top,p,0).x,top!=0u);
    if (!in_front(i.id,i.depth,b,db)) { discard; }
    return GeoTargets(i.id,i.depth);
}

/// Part ID and depth after all geometry passes.
fn id_at(p: vec2i) -> u32 {
    let contact=textureLoad(gid_contacts,p,0).x;
    if (contact!=0u) { return contact; }
    let t = textureLoad(gid_top, p, 0).x;
    if (t != 0u) {
        return t;
    }
    return textureLoad(gid, p, 0).x;
}

fn depth_at(p: vec2i) -> f32 {
    if (textureLoad(gid_contacts,p,0).x!=0u) { return textureLoad(gdepth_contacts,p,0).x; }
    if (textureLoad(gid_top, p, 0).x != 0u) {
        return textureLoad(gdepth_top, p, 0).x;
    }
    return textureLoad(gdepth, p, 0).x;
}

struct FsOut {
    @builtin(position) pos: vec4f,
}

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> FsOut {
    let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return FsOut(vec4f(uv * 2.0 - 1.0, 0.0, 1.0));
}

fn size() -> vec2i {
    return vec2i(i32(cam.b.z), i32(cam.b.w));
}

fn inside(p: vec2i) -> bool {
    let s = size();
    return p.x >= 0 && p.y >= 0 && p.x < s.x && p.y < s.y;
}

fn pack(p: vec2i) -> u32 {
    return u32(p.x) | (u32(p.y) << 16u);
}

fn unpack(s: u32) -> vec2i {
    return vec2i(i32(s & 0xffffu), i32(s >> 16u));
}

/// The world point seen at a supersampled pixel at a depth.
fn world_at(p: vec2i, depth: f32) -> vec3f {
    let px = vec2f(p) + 0.5;
    if (cam.a.w > 0.5) {
        return cam.pos.xyz + ray(px) * depth;
    }
    let f = cam.a.x;
    let dir = cam.fwd.xyz + cam.right.xyz * ((px.x - cam.a.y) / f) + cam.up.xyz * ((cam.a.z - px.y) / f);
    return cam.pos.xyz + dir * depth;
}

/// How strong the line between neighbouring pixels of parts a and b is, 0..1.
///
/// A limb and the part it grows from (an arm and the torso) meet without a
/// seam — but only near the root joint, as in the MVP. There the line fades
/// with the depth gap, so it comes in smoothly as the arm leaves the torso.
/// Away from the root a limb against its root part is always outlined: a
/// thigh in front of the belly in a crouch.
fn edge_strength(p: vec2i, a: u32, da: f32, b: u32, db: f32) -> f32 {
    if (a == b) {
        return 0.0;
    }
    if (a == 0u || b == 0u) {
        return 1.0;
    }
    let ga = parts[a].group;
    if (ga == 0u || ga != parts[b].group || over(a, b) || over(b, a)) {
        return 1.0;
    }
    // only a part and what it grows from, or two parts growing from one place
    // (a forearm against the torso is outlined all the way, even at the elbow)
    let pa = parts[a].parent;
    let pb = parts[b].parent;
    // two parts grow "from one place" only from one joint (two collarbones
    // from the chest), not from one part (the left and right thigh)
    let siblings = pa != 0u && pa == pb && distance(parts[a].root.xyz, parts[b].root.xyz) < 1e-3;
    if (pa != b && pb != a && !siblings) {
        return 1.0;
    }
    // near the root of either part: the joint where one grows from the
    // other (the knee for a thigh and its shin, the hip for a thigh and the torso)
    let w = world_at(p, da);
    var away = 1.0;
    var zone = 0.0;
    for (var k = 0; k < 2; k++) {
        var root = parts[a].root;
        if (k == 1) {
            root = parts[b].root;
        }
        if (root.w > 0.0) {
            away = min(away, smoothstep(root.w, 1.5 * root.w, distance(w, root.xyz)));
            zone = max(zone, root.w);
        }
    }
    if (away >= 1.0) {
        return 1.0;
    }
    // thin sticks: the depth gap at a joint is just their thickness — only
    // the zone around the joint decides
    if (parts[a].halo != 0u && parts[b].halo != 0u) {
        return away;
    }
    // the line comes in as the gap grows by half the limb's radius (the zone
    // is 1.5 radii): a smooth start near, a full line far from the camera
    let tol = cam.c.w + 4.0 * min(da, db) / cam.a.x;
    let gap = smoothstep(tol, tol + zone / 3.0, abs(da - db));
    return max(gap, away);
}

/// Does a own the line against b: the one in front (with overlap rules).
fn owns(a: u32, da: f32, b: u32, db: f32) -> bool {
    return in_front(a, da, b, db);
}

// --- edges ---------------------------------------------------------------------

struct EdgeOut {
    @location(0) seed: u32,
    @location(1) strength: f32,
}

/// A pixel becomes a line seed if it owns an edge and its part has an outline.
@fragment
fn fs_edge(i: FsOut) -> EdgeOut {
    let p = vec2i(i.pos.xy);
    let a = id_at(p);
    if (a == 0u || parts[a].base <= 0.0) {
        return EdgeOut(NONE, 0.0);
    }
    let da = depth_at(p);
    let offsets = array<vec2i, 4>(vec2i(1, 0), vec2i(-1, 0), vec2i(0, 1), vec2i(0, -1));
    var strength = 0.0;
    for (var k = 0; k < 4; k++) {
        let q = p + offsets[k];
        if (!inside(q)) {
            continue;
        }
        let b = id_at(q);
        let db = depth_at(q);
        if (owns(a, da, b, db)) {
            strength = max(strength, edge_strength(p, a, da, b, db));
        }
    }
    if (strength <= 0.0) {
        return EdgeOut(NONE, 0.0);
    }
    return EdgeOut(pack(p), strength);
}

// --- jump flood ----------------------------------------------------------------

@group(0) @binding(4) var seeds: texture_2d<u32>;
@group(0) @binding(5) var<uniform> step: vec4i;

/// Every pixel learns its nearest line seed in log2(width) passes.
@fragment
fn fs_jfa(i: FsOut) -> @location(0) u32 {
    let p = vec2i(i.pos.xy);
    var best = NONE;
    var bestd = 1e30;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let q = p + vec2i(dx, dy) * step.x;
            if (!inside(q)) {
                continue;
            }
            let s = textureLoad(seeds, q, 0).x;
            if (s == NONE) {
                continue;
            }
            let d = vec2f(unpack(s) - p);
            let dd = dot(d, d);
            if (dd < bestd) {
                bestd = dd;
                best = s;
            }
        }
    }
    return best;
}

// --- compose -------------------------------------------------------------------

@group(0) @binding(6) var<uniform> scene: Scene;
@group(0) @binding(11) var strength: texture_2d<f32>;
@group(0) @binding(14) var strokes: texture_2d<f32>;

/// Line strength near a seed: the average over the seeds around it. A single
/// pixel's depth gap is noisy, and fading line ends came out ragged.
fn strength_near(sp: vec2i) -> f32 {
    var sum = 0.0;
    var n = 0.0;
    for (var dy = -3; dy <= 3; dy++) {
        for (var dx = -3; dx <= 3; dx++) {
            let q = sp + vec2i(dx, dy);
            if (!inside(q)) {
                continue;
            }
            let v = textureLoad(strength, q, 0).x;
            if (v > 0.0) {
                sum += v;
                n += 1.0;
            }
        }
    }
    return sum / max(n, 1.0);
}

/// Line width in supersampled px for a part at a depth (`line_px` of the MVP).
fn line_width(id: u32, depth: f32) -> f32 {
    let pt = parts[id];
    let s = cam.c.x / max(depth, 1e-6);
    return pt.base * clamp(pow(s / pt.sref, pt.gamma), 0.45, 1.9) * cam.c.y;
}

/// Screen direction of a stick (from its root to its other end).
fn stick_dir(id: u32) -> vec2f {
    let a = project(parts[id].root.xyz);
    let b = project(parts[id].tip.xyz);
    let d = b.xy - a.xy;
    if (a.w == 0.0 || b.w == 0.0 || length(d) < 1e-3) {
        return vec2f(0.0);
    }
    return normalize(d);
}

/// How much two sticks cross on screen: 0 when nearly parallel (under 15°),
/// 1 from 30°; smooth in between, so a halo comes in without a jump.
fn crossing(a: u32, b: u32) -> f32 {
    let da = stick_dir(a);
    let db = stick_dir(b);
    if (dot(da, da) == 0.0 || dot(db, db) == 0.0) {
        return 1.0;
    }
    return smoothstep(0.966, 0.866, abs(dot(da, db)));
}

/// Ray from the camera through a supersampled pixel position.
fn ray(px: vec2f) -> vec3f {
    let f = cam.a.x;
    let dx = px.x - cam.a.y;
    let dy = cam.a.z - px.y;
    if (cam.a.w > 0.5) {
        let r = length(vec2f(dx, dy));
        let theta = r / f;
        if (r < 1e-6) {
            return cam.fwd.xyz;
        }
        return cam.fwd.xyz * cos(theta) + (cam.right.xyz * dx + cam.up.xyz * dy) * (sin(theta) / r);
    }
    return normalize(cam.fwd.xyz + cam.right.xyz * (dx / f) + cam.up.xyz * (dy / f));
}

/// Where a ray meets the ground (y = 0), or w = 0 if it does not.
fn ground_hit(px: vec2f) -> vec4f {
    let d = ray(px);
    if (d.y > -1e-6) {
        return vec4f(0.0);
    }
    let t = -cam.pos.y / d.y;
    return vec4f(cam.pos.xyz + d * t, 1.0);
}

fn background(p: vec2i) -> vec3f {
    var col = scene.paper.rgb;
    if (scene.center.w < 0.5) {
        return col;
    }
    let px = vec2f(p) + 0.5;
    let g = ground_hit(px);
    if (g.w == 0.0) {
        return col;
    }
    if (scene.grid.w > 0.5) {
        let gx = ground_hit(px + vec2f(1.0, 0.0));
        let gy = ground_hit(px + vec2f(0.0, 1.0));
        let rel = g.xz - scene.center.xy;
        let size = scene.grid.x;
        let stp = scene.grid.y;
        if (gx.w > 0.0 && gy.w > 0.0 && abs(rel.x) <= size && abs(rel.y) <= size) {
            // grid line distance in px, from how the ground point moves per pixel
            let jx = vec2f(gx.x - g.x, gy.x - g.x);
            let jz = vec2f(gx.z - g.z, gy.z - g.z);
            let ux = rel.x / stp;
            let uz = rel.y / stp;
            let dist_x = abs(ux - round(ux)) * stp / max(length(jx), 1e-9);
            let dist_z = abs(uz - round(uz)) * stp / max(length(jz), 1e-9);
            let half = 0.5 * scene.grid.z * cam.c.y;
            // lines closer than a few px blend into tone instead of moiré
            let fade_x = smoothstep(2.0, 6.0, stp / max(length(jx), 1e-9));
            let fade_z = smoothstep(2.0, 6.0, stp / max(length(jz), 1e-9));
            let cov = max(clamp(half + 0.5 - dist_x, 0.0, 1.0) * fade_x, clamp(half + 0.5 - dist_z, 0.0, 1.0) * fade_z);
            col = mix(col, scene.ground.rgb, cov);
        }
    }
    let n = i32(scene.center.z);
    for (var k = 0; k < n; k++) {
        let s = scene.shadows[k];
        let d = g.xz - s.xy;
        let axis = scene.shadow_axes[k].xy;
        let hit = select(length(d) < s.z, abs(dot(d, axis)) < s.z && abs(dot(d, vec2f(-axis.y, axis.x))) < s.w, s.w > 0.0);
        if (hit) {
            col = mix(col, vec3f(0.0), 0.10);
        }
    }
    // a fire's glow: a soft spot, brightest at the centre (the MVP's radial gradient)
    if (scene.glow.w > 0.0) {
        let a = 0.55 * scene.glow.w * max(1.0 - distance(g.xz, scene.glow.xy) / scene.glow.z, 0.0);
        col = mix(col, scene.glow_color.rgb, a);
    }
    return col;
}

fn shade(p: vec2i) -> vec3f {
    let a = id_at(p);
    var col = scene.paper.rgb;
    if (a == 0u) {
        col = background(p);
    } else if (scene.layers.z > 0.5) {
        // debug: every part in its own colour, to see which part is where
        let h = a * 2654435761u;
        col = vec3f(f32((h >> 8u) & 255u), f32((h >> 16u) & 255u), f32((h >> 24u) & 255u)) / 255.0 * 0.6 + 0.35;
    } else if (scene.layers.x > 0.5) {
        col = parts[a].fill.rgb;
    }
    if (scene.layers.y > 0.5) {
        // lines from curves (premultiplied, coverage in alpha)
        let l = textureLoad(strokes, p, 0);
        col = col * (1.0 - l.a) + l.rgb;
        // halos of sticks, from the buffers
        let s = textureLoad(seeds, p, 0).x;
        if (s != NONE) {
            let sp = unpack(s);
            let owner = id_at(sp);
            // a halo lies only outside its part, over what is behind it — and
            // not over a stick joined to it at a shared joint (it would bite
            // a notch into the joint), by the same rule as seams
            if (parts[owner].halo == 0u || owner != a) {
                let ds = depth_at(sp);
                var w = line_width(owner, ds) * strength_near(sp);
                // the halo narrows into both joined ends of its stick, as in
                // the MVP (the neck into the head, a thigh into the hip),
                // whatever part lies next to it there
                if (parts[owner].halo != 0u) {
                    let ws = world_at(sp, ds);
                    let root = parts[owner].root;
                    let tip = parts[owner].tip;
                    if (root.w > 0.0) {
                        w *= smoothstep(0.0, root.w, distance(ws, root.xyz));
                    }
                    if (tip.w > 0.0) {
                        w *= smoothstep(0.0, tip.w, distance(ws, tip.xyz));
                    }
                    // a halo shows where sticks cross; over a stick running
                    // almost along it (an arm swinging past the body) it would
                    // cut the far one into flickering slivers
                    if (a != 0u && parts[a].halo != 0u) {
                        w *= crossing(owner, a);
                    }
                }
                let d = distance(vec2f(p), vec2f(sp));
                var cov = clamp(0.5 * w + 0.5 - d, 0.0, 1.0);
                if (parts[owner].halo != 0u && a != 0u) {
                    cov *= edge_strength(p, owner, ds, a, depth_at(p));
                }
                col = mix(col, parts[owner].line.rgb, cov);
            }
        }
    }
    return col;
}

/// One output pixel: the average of its supersampled pixels.
@fragment
fn fs_compose(i: FsOut) -> @location(0) vec4f {
    let ss = i32(cam.c.y);
    let o = vec2i(i.pos.xy) * ss;
    var acc = vec3f(0.0);
    for (var y = 0; y < ss; y++) {
        for (var x = 0; x < ss; x++) {
            acc += shade(o + vec2i(x, y));
        }
    }
    return vec4f(acc / f32(ss * ss), 1.0);
}

// --- line curves -----------------------------------------------------------------
//
// Lines from geometry (05-architecture, §7): the silhouette of every part is a
// curve; each point of it is tested against the buffers. A point is hidden if
// another part is in front of it there (or is over it by a rule). Seams need
// no rules: the arm's silhouette goes into the torso and the torso hides it;
// the torso's silhouette at the shoulder is hidden by the arm.

struct CurvePoint {
    pos: vec4f,
    id: u32,
    p1: u32,
    p2: u32,
    p3: u32,
}

@group(0) @binding(12) var<storage, read> cpoints: array<CurvePoint>;
@group(0) @binding(13) var<storage, read_write> cvis: array<f32>;

/// World point → supersampled screen position and depth (w = 0: not seen).
fn project(p: vec3f) -> vec4f {
    let d = p - cam.pos.xyz;
    let x = dot(d, cam.right.xyz);
    let y = dot(d, cam.up.xyz);
    let z = dot(d, cam.fwd.xyz);
    let f = cam.a.x;
    if (cam.a.w > 0.5) {
        let rad = length(vec2f(x, y));
        let dist = length(vec3f(x, y, z));
        if (dist < cam.b.x) {
            return vec4f(0.0);
        }
        let theta = atan2(rad, z);
        var k = f / dist;
        if (rad > 1e-9) {
            k = f * theta / rad;
        }
        return vec4f(cam.a.y + x * k, cam.a.z - y * k, dist, 1.0);
    }
    if (z < cam.b.x) {
        return vec4f(0.0);
    }
    return vec4f(cam.a.y + x * f / z, cam.a.z - y * f / z, z, 1.0);
}

/// Does the part b seen at depth db hide a curve point of part `own` at
/// depth z? The same overlap rules as the fill (a nudge by the bias, not an
/// override): from below the torso hides the head.
fn occludes(b: u32, db: f32, own: u32, z: f32) -> bool {
    if (b == 0u || b == own) {
        return false;
    }
    let tol = cam.c.w + 0.004 * z;
    if (over(own, b)) {
        return db < z - parts[own].bias - tol;
    }
    if (over(b, own)) {
        return db - parts[b].bias < z;
    }
    return db < z - tol;
}

@compute @workgroup_size(64)
fn cs_visible(@builtin(global_invocation_id) g: vec3u) {
    let i = g.x;
    if (i >= arrayLength(&cpoints)) {
        return;
    }
    let pt = cpoints[i];
    let s = project(pt.pos.xyz);
    if (s.w == 0.0) {
        cvis[i] = 0.0;
        return;
    }
    let p = vec2i(floor(s.xy));
    let tol = cam.c.w + 0.004 * s.z;
    // the same test as the stroke pass clips by, at the point's own pixel,
    // so the two agree and lines neither stop short nor poke out
    var hidden = !inside(p);
    if (!hidden && occludes(id_at(p), depth_at(p), pt.id, s.z)) {
        hidden = true;
    }
    cvis[i] = select(1.0, 0.0, hidden);
}

// --- strokes ---------------------------------------------------------------------

struct StrokeIn {
    @location(0) a: vec2f,
    @location(1) b: vec2f,
    @location(2) r: vec2f,
    @location(3) color: vec4f,
    @location(4) depth: vec2f,
    @location(5) id: u32,
    @location(6) hidden: u32,
}

struct StrokeOut {
    @builtin(position) pos: vec4f,
    @location(0) px: vec2f,
    @location(1) @interpolate(flat) a: vec2f,
    @location(2) @interpolate(flat) b: vec2f,
    @location(3) @interpolate(flat) r: vec2f,
    @location(4) @interpolate(flat) color: vec4f,
    @location(5) @interpolate(flat) depth: vec2f,
    @location(6) @interpolate(flat) id: u32,
    @location(7) @interpolate(flat) hidden: u32,
}

/// One piece of a stroke: a quad around the segment a→b with radii r.
@vertex
fn vs_stroke(@builtin(vertex_index) v: u32, s: StrokeIn) -> StrokeOut {
    let corners = array<vec2f, 6>(vec2f(0.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
                                  vec2f(0.0, -1.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0));
    let c = corners[v];
    let ab = s.b - s.a;
    let len = length(ab);
    var dir = vec2f(1.0, 0.0);
    if (len > 1e-6) {
        dir = ab / len;
    }
    let n = vec2f(-dir.y, dir.x);
    let pad = max(s.r.x, s.r.y) + 1.5;
    let px = mix(s.a - dir * pad, s.b + dir * pad, c.x) + n * pad * c.y;
    var o: StrokeOut;
    o.pos = vec4f(px.x / cam.b.z * 2.0 - 1.0, 1.0 - px.y / cam.b.w * 2.0, 0.0, 1.0);
    o.px = px;
    o.a = s.a;
    o.b = s.b;
    o.r = s.r;
    o.color = s.color;
    o.depth = s.depth;
    o.id = s.id;
    o.hidden = s.hidden;
    return o;
}

/// Coverage of a segment with a radius changing along it (a tapered capsule).
@fragment
fn fs_stroke(i: StrokeOut) -> @location(0) vec4f {
    let ab = i.b - i.a;
    let t = clamp(dot(i.px - i.a, ab) / max(dot(ab, ab), 1e-9), 0.0, 1.0);
    let r = mix(i.r.x, i.r.y, t);
    let d = length(i.px - (i.a + ab * t)) - r;
    let cov = clamp(0.5 - d, 0.0, 1.0) * min(r * 2.0, 1.0);
    // a line stops exactly at the edge of a part in front of it: a stroke
    // ending at an occluder does not poke its round end over it. Only on the
    // half going into a hidden end — along the line a joint ball slightly in
    // front of a tube would bite notches out of it
    let raw = dot(i.px - i.a, ab) / max(dot(ab, ab), 1e-9);
    let into_hidden = ((i.hidden & 1u) != 0u && raw < 0.5) || ((i.hidden & 2u) != 0u && raw > 0.5);
    let p = vec2i(floor(i.px));
    // a line on a part's surface (the face) exists only where the part is seen
    if ((i.hidden & 4u) != 0u && (!inside(p) || id_at(p) != i.id)) {
        discard;
    }
    if (into_hidden && inside(p)) {
        if (occludes(id_at(p), depth_at(p), i.id, mix(i.depth.x, i.depth.y, t))) {
            discard;
        }
    }
    return vec4f(i.color.rgb * cov, cov);
}

// --- fills on a surface -------------------------------------------------------------

struct FillIn {
    @location(0) p: vec2f,
    @location(1) color: vec4f,
    @location(2) id: u32,
}

struct FillOut {
    @builtin(position) pos: vec4f,
    @location(0) @interpolate(flat) color: vec4f,
    @location(1) @interpolate(flat) id: u32,
}

@vertex
fn vs_fill(s: FillIn) -> FillOut {
    var o: FillOut;
    o.pos = vec4f(s.p.x / cam.b.z * 2.0 - 1.0, 1.0 - s.p.y / cam.b.w * 2.0, 0.0, 1.0);
    o.color = s.color;
    o.id = s.id;
    return o;
}

/// A filled shape drawn on part `id` (an eye on the head): only where that
/// part is the visible one.
@fragment
fn fs_fill(i: FillOut) -> @location(0) vec4f {
    let p = vec2i(floor(i.pos.xy));
    if (id_at(p) != i.id) {
        discard;
    }
    return vec4f(i.color.rgb, 1.0);
}

@group(0) @binding(15) var<storage, read_write> effect_depth: array<f32>;
@compute @workgroup_size(8,8)
fn cs_effect_depth(@builtin(global_invocation_id) p: vec3u) {
    let size = vec2u(cam.b.zw);
    if (p.x >= size.x || p.y >= size.y) { return; }
    effect_depth[p.y*size.x+p.x] = depth_at(vec2i(p.xy));
}
