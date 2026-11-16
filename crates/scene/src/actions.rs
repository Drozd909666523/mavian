//! Actions: grab, put down, push, strike (a port of the MVP's
//! `engine/actions.py`).
//!
//! An action is the animator's intent: "at T take the cup with the right
//! hand". When the shot is compiled it becomes two things:
//!
//! - a target for the hand (`ik::Reach`): the hand reaches smoothly for the
//!   prop and back;
//! - a motion segment of the prop: lies → in the hand → falls to a support,
//!   or slides with the character.
//!
//! Everything follows from the description, without state: where a prop is
//! at T is known from the earlier actions, so only their order in time matters.

use crate::actor::Actor;
use crate::ik::{Avoid, Reach, Target};
use crate::props::{Motion, Prop, Shape, World};
use mavian_core::{Ease, Keys, Mat3, Track, Vec3, m};

/// how far short of the surface the hand stops (its radius and outline)
pub const HAND_R: f64 = 0.065;
/// how much higher than the spot the hand carries a thing before lowering it
const HOVER: f64 = 0.07;
/// which share of the putting-down motion is the vertical lowering
const DESCEND: f64 = 0.4;
/// a hand puts a thing at the near edge of a support's top, this far in
const EDGE_MARGIN: f64 = 0.11;

/// Where a released thing goes.
#[derive(Clone, Debug)]
pub enum Onto {
    /// let go: it falls to the ground
    Nothing,
    Point(Vec3),
    /// on the top of a prop, at the edge nearest the character
    Prop(usize),
}

#[derive(Clone, Debug)]
pub enum Action {
    /// take `prop` with `hand` at `at`; the hand starts reaching `reach`
    /// before, returns to the pose (with the prop) over `settle`, holds the
    /// side at `height` of it
    Grab { actor: usize, hand: usize, prop: usize, at: f64, reach: f64, settle: f64, height: f64, step: Option<crate::Approach> },
    /// put down what the hand holds at `at`
    Release { actor: usize, hand: usize, prop: usize, at: f64, onto: Onto, reach: f64, settle: f64 },
    /// push `prop` from `start` to `end`: it slides with the character, the
    /// palms against its near side at `height`, `spread` apart from its middle
    Push {
        actor: usize,
        prop: usize,
        start: f64,
        end: f64,
        hands: [usize; 2],
        reach: f64,
        settle: f64,
        height: f64,
        spread: f64,
        step: Option<crate::Approach>,
    },
    /// a strike with `limb` ("hand_r", "foot_l") at `at` into `aim` of
    /// `target` ("head", "chest", "chin"); see `Strike`
    Strike(Strike),
}

/// A strike: the hit gives by itself a knockback and a push of the one hit
/// (layers over their pose), a hit-stop, a camera shake and, if `flash`, an
/// impact frame. `blocked` — the defender's hand that blocks (the strike goes
/// into it, the reaction is weak); `miss` — the strike goes where the target
/// was when it began (the opponent ducked), no reaction.
#[derive(Clone, Debug)]
pub struct Strike {
    pub attacker: usize,
    pub limb: usize,
    pub target: usize,
    pub at: f64,
    pub aim: String,
    /// the limb starts for the target this long before `at`
    pub windup: f64,
    /// it stays at the target this long
    pub hold: f64,
    /// it comes back to the pose over this long
    pub recover: f64,
    pub blocked: Option<usize>,
    pub miss: bool,
    /// how far the hit pushes (world units)
    pub force: f64,
    /// hit-stop, shot seconds
    pub stop: f64,
    pub shake: f64,
    pub flash: bool,
}

impl Strike {
    pub fn new(attacker: usize, limb: usize, target: usize, at: f64) -> Strike {
        Strike {
            attacker,
            limb,
            target,
            at,
            aim: "head".into(),
            windup: 0.14,
            hold: 0.06,
            recover: 0.3,
            blocked: None,
            miss: false,
            force: 0.3,
            stop: 0.1,
            shake: 0.06,
            flash: false,
        }
    }
}

/// half a limb's thickness: a fist or a foot lands with its centre this far
/// off the target's surface, else the bone enters it
const LIMB_HALF: f64 = 0.06;
/// strikes reach this share of the limb's length (not dead straight)
const LUNGE_REACH: f64 = 0.92;
/// a miss passes this far above the centre of the dodging head
/// (head radius 0.19 + arm thickness + a margin)
const MISS_CLEAR: f64 = 0.36;
/// fighters' head centres no closer than this (radii 0.2 + a margin)
const HEAD_GAP: f64 = 0.44;
/// a head's centre from the opponent's arm bone no closer than this
const ARM_GAP: f64 = 0.27;

/// Where a strike lands: centre of the fist or foot on the target's surface plus half a limb.
pub(crate) fn surface_of(aim: &str) -> f64 {
    match aim {
        "head" => 0.19 + LIMB_HALF,
        "chest" | "pelvis" => 0.12 + LIMB_HALF,
        _ => 0.12,
    }
}

fn weighted(table: &[(f64, f64, Ease)]) -> Track<f64> {
    let mut k = Keys::new(Ease::InOut);
    for &(t, v, e) in table {
        k = k.eased(t, v, e);
    }
    k.into()
}

impl Action {
    pub fn time(&self) -> f64 {
        match self {
            Action::Grab { at, .. } | Action::Release { at, .. } => *at,
            Action::Push { start, .. } => *start,
            Action::Strike(s) => s.at,
        }
    }
}

/// Shoulder, elbow, hand — by the hand joint's parents.
pub fn chain(actor: &Actor, hand: usize) -> Result<[usize; 3], String> {
    let rig = &actor.model.rig;
    let elbow = rig.parent(hand).ok_or("a hand needs an elbow")?;
    let shoulder = rig.parent(elbow).ok_or("an elbow needs a shoulder")?;
    Ok([shoulder, elbow, hand])
}

pub(crate) fn root_at(actor: &Actor, w: f64) -> Vec3 {
    actor.figure(w).joints[actor.model.root]
}

/// The side of a prop facing a point → (normal, tangent) in the prop's frame.
pub(crate) fn facing_side(pos: Vec3, rot: &Mat3, toward: Vec3) -> (Vec3, Vec3) {
    let d = rot.transpose() * (toward - pos);
    let n =
        if d.x.abs() > d.z.abs() { Vec3 { x: d.x.signum(), y: 0.0, z: 0.0 } } else { Vec3 { x: 0.0, y: 0.0, z: if d.z > 0.0 { 1.0 } else { -1.0 } } };
    (n, Vec3 { x: -n.z, y: 0.0, z: n.x })
}

/// A point on the side with normal n (prop frame), at `height` of it.
pub(crate) fn surface(shape: &Shape, n: Vec3, height: f64) -> Vec3 {
    match shape {
        Shape::Sphere { r } => Vec3::UP * *r + n * (r + HAND_R),
        Shape::Box { sx, sy, sz } => {
            let half = if n.x != 0.0 { sx / 2.0 } else { sz / 2.0 };
            Vec3 { x: n.x * (half + HAND_R), y: sy * height, z: n.z * (half + HAND_R) }
        }
    }
}

/// The target's weight: up to 1 over `reach` before t0, held to t1, down over `settle`.
pub(crate) fn window(t0: f64, t1: f64, reach: f64, settle: f64) -> Track<f64> {
    Keys::new(Ease::InOut).key(t0 - reach, 0.0).key(t0, 1.0).key(t1, 1.0).key(t1 + settle, 0.0).into()
}

/// The spot on a support's top nearest the character (in from the edge):
/// "put it on the crate" — at the near edge, where the hand reaches.
fn near_spot(support: &Prop, t: f64, toward: Vec3, world: &World) -> Vec3 {
    let (pos, rot) = support.rest(t, world);
    match support.shape {
        Shape::Box { sx, sy, sz } => {
            let d = rot.transpose() * (toward - pos);
            let (hx, hz) = ((sx / 2.0 - EDGE_MARGIN).max(0.0), (sz / 2.0 - EDGE_MARGIN).max(0.0));
            pos + rot * Vec3 { x: d.x.clamp(-hx, hx), y: sy, z: d.z.clamp(-hz, hz) }
        }
        Shape::Sphere { .. } => support.top_point(t, world),
    }
}

/// Compile the shot's actions: clear what an earlier compile made, then
/// apply them in time order. Arms avoid other characters' heads, for all pairs.
/// What strikes add to the shot: hit-stops, impact frames, shakes.
pub type ShotEffects = (Vec<(f64, f64)>, Vec<f64>, Vec<crate::camera::Shake>);

/// Compile standalone arrays with the default drawing style. Prefer
/// `Shot::compile` when the scene changes line width or uses strike actions.
pub fn compile(actors: &mut [Actor], props: &mut [Prop], actions: &[Action]) -> Result<ShotEffects, String> {
    // Move actors rather than cloning their world-dependent pose caches.
    let owned = actors
        .iter_mut()
        .map(|a| {
            let empty = Actor::new(a.model.clone());
            std::mem::replace(a, empty)
        })
        .collect();
    let mut shot = crate::Shot { actors: owned, props: props.to_vec(), actions: actions.to_vec(), ..crate::Shot::default() };
    let result = compile_shot(&mut shot);
    for (a, compiled) in actors.iter_mut().zip(shot.actors) {
        *a = compiled;
    }
    props.clone_from_slice(&shot.props);
    result
}

pub(crate) fn compile_shot(shot: &mut crate::Shot) -> Result<ShotEffects, String> {
    let mut fx: ShotEffects = (Vec::new(), Vec::new(), Vec::new());
    shot.grab_reports.clear();
    let (actors, props) = (&mut shot.actors, &mut shot.props);
    for a in actors.iter_mut() {
        a.reaches.clear();
        a.auto_contacts.clear();
        a.avoids.clear();
        a.auto_layers.clear();
        a.recoils.clear();
        a.approach_steps.clear();
        a.step_reports.clear();
        a.forget();
    }
    for p in props.iter_mut() {
        p.motion.clear();
        p.contacts.clear();
    }
    // Acquisition must see the same actor-avoidance constraints as rendering.
    if actors.len() > 1 {
        for i in 0..actors.len() {
            let chains: Vec<[usize; 3]> =
                ["hand_l", "hand_r"].iter().filter_map(|h| actors[i].model.rig.joint(h).ok()).filter_map(|h| chain(&actors[i], h).ok()).collect();
            for other in 0..actors.len() {
                if other == i {
                    continue;
                }
                if let Ok(head) = actors[other].model.rig.joint("head") {
                    actors[i].avoids.push(Avoid { other, head, chains: chains.clone(), gap: 0.255 });
                }
            }
            actors[i].forget();
        }
    }
    let actions = shot.actions.clone();
    let mut order: Vec<_> = actions.iter().collect();
    order.sort_by(|a, b| a.time().total_cmp(&b.time()));
    for action in order {
        crate::approach::prepare(&mut shot.actors, &shot.props, action, &actions, shot.style.line)?;
        if let Action::Strike(s) = action {
            strike(&mut shot.actors, &shot.props, s, &mut fx)?;
        } else {
            apply(shot, action)?;
        }
        for a in &mut shot.actors {
            a.forget();
        }
    }
    plan_contact_routes(&mut shot.actors, &shot.props, shot.style.line);
    crate::grip::verify(shot)?;
    Ok(fx)
}

/// Keep the direct hand path unless its sampled swept sphere hits a prop.
/// The two phases are planned separately and stored, so route selection cannot
/// switch back and forth as the shot is evaluated in arbitrary frame order.
fn plan_contact_routes(actors: &mut [Actor], props: &[Prop], line: f64) {
    for actor in 0..actors.len() {
        for index in 0..actors[actor].auto_contacts.len() {
            let c = actors[actor].auto_contacts[index].clone();
            let Some((split, _)) = c.withdrawal else { continue };
            actors[actor].auto_contacts[index].approach = None;
            actors[actor].auto_contacts[index].withdrawal = Some((split, None));
            for a in actors.iter_mut() {
                a.forget();
            }
            let world = World { actors, props };
            let hand = c.chain[2];
            let radius = crate::obstacle::joint_radius(&actors[actor], hand, line);
            let blocked = |from: f64, to: f64| {
                let samples = (((to - from).max(0.0) * 120.0).ceil() as usize).max(1);
                let mut previous = None;
                for n in 0..=samples {
                    let w = from + (to - from) * n as f64 / samples as f64;
                    let point = actors[actor].figure_ik(w, &world).joints[hand];
                    let poses: Vec<_> = props.iter().map(|p| p.rest(w, &world)).collect();
                    for (prop, p) in props.iter().enumerate() {
                        let (pos, rot) = poses[prop];
                        let obstacle = crate::obstacle::Obstacle { prop, shape: p.shape.clone(), pos, rot };
                        if obstacle.distance(point).0 < radius - 0.001 {
                            return true;
                        }
                        if let Some((old_point, ref old_poses)) = previous {
                            let old_poses: &Vec<(Vec3, Mat3)> = old_poses;
                            let (old_pos, old_rot) = old_poses[prop];
                            // Sweep relative to the prop. A carried object
                            // must not hit a hand just because both translated.
                            let start = pos + rot * (old_rot.transpose() * (old_point - old_pos));
                            if obstacle.capsule(start, radius, point, radius).0 < -0.001 {
                                return true;
                            }
                        }
                    }
                    previous = Some((point, poses));
                }
                false
            };
            let inbound = blocked(c.from, split);
            let outbound = blocked(split, c.to);
            actors[actor].auto_contacts[index].approach = if inbound { c.approach } else { None };
            actors[actor].auto_contacts[index].withdrawal = Some((split, if outbound { c.approach } else { None }));
            for a in actors.iter_mut() {
                a.forget();
            }
        }
    }
}

/// How much further the target is than the limb reaches at `at` (≤ 0 — it
/// reaches): to put an opponent at striking distance instead of picking it
/// by number — then the strike does not lunge and nobody slides without steps.
pub fn reach_gap(attacker: &Actor, limb: usize, target: &Actor, at: f64, aim: &str) -> Result<f64, String> {
    let ch = chain(attacker, limb)?;
    let js = attacker.figure(at);
    let length = (js.joints[ch[1]] - js.joints[ch[0]]).length() + (js.joints[ch[2]] - js.joints[ch[1]]).length();
    let joint = target.model.rig.joint(if aim == "chin" { "head" } else { aim }).map_err(|e| e.0)?;
    let p = target.figure(at).joints[joint];
    let p = p + crate::ik::flat_dir(p, js.joints[attacker.model.root]) * surface_of(aim);
    Ok((p - js.joints[ch[0]]).length() - LUNGE_REACH * length)
}

fn seg_dist(p: Vec3, a: Vec3, b: Vec3) -> f64 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.dot(ab).max(1e-9)).clamp(0.0, 1.0);
    (p - (a + ab * t)).length()
}

/// Shorten a lunge so the attacker's head enters neither the opponent's
/// head nor their arms (the same "do not enter" as with a prop, between characters).
fn no_clash(actors: &[Actor], props: &[Prop], a: usize, d: usize, t: f64, toward: Vec3, step: f64) -> f64 {
    let head = |i: usize| actors[i].model.rig.joint("head").ok();
    let (Some(ha_j), Some(hd_j)) = (head(a), head(d)) else { return step };
    let ha = actors[a].figure(t).joints[ha_j];
    let world = World { actors, props };
    let jd = actors[d].figure_ik(t, &world);
    let hd = jd.joints[hd_j];
    let rig = &actors[d].model.rig;
    let mut bones = Vec::new();
    for s in ["l", "r"] {
        for (x, y) in [(format!("arm_{s}"), format!("elbow_{s}")), (format!("elbow_{s}"), format!("hand_{s}"))] {
            if let (Ok(x), Ok(y)) = (rig.joint(&x), rig.joint(&y)) {
                bones.push((jd.joints[x], jd.joints[y]));
            }
        }
    }
    let gap = |k: f64| {
        let p = Vec3 { x: ha.x + toward.x * k, y: ha.y, z: ha.z + toward.z * k };
        let mut g = (p - hd).length() - HEAD_GAP;
        for &(b0, b1) in &bones {
            g = g.min(seg_dist(p, b0, b1) - ARM_GAP);
        }
        g + HEAD_GAP
    };
    if gap(step) >= HEAD_GAP {
        return step;
    }
    if gap(0.0) < HEAD_GAP {
        return 0.0;
    }
    let (mut lo, mut hi) = (0.0, step);
    for _ in 0..20 {
        let mid = (lo + hi) / 2.0;
        if gap(mid) >= HEAD_GAP {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// `arm=` in a layer — both sides (as in a pose).
fn both(actor: &Actor, angles: &[(&str, Vec3)]) -> Vec<(usize, Vec3)> {
    let rig = &actor.model.rig;
    let mut out = Vec::new();
    for &(name, a) in angles {
        match rig.joint(name) {
            Ok(j) => out.push((j, a)),
            Err(_) => {
                for s in ["l", "r"] {
                    if let Ok(j) = rig.joint(&format!("{name}_{s}")) {
                        out.push((j, a));
                    }
                }
            }
        }
    }
    out
}

fn strike(actors: &mut [Actor], props: &[Prop], s: &Strike, fx: &mut ShotEffects) -> Result<(), String> {
    use crate::actor::Layer;
    use crate::ik::flat_dir;
    let (a, d, t) = (s.attacker, s.target, s.at);
    let limb_name = actors[a].model.rig.joints()[s.limb].name.clone();
    let hand = limb_name.starts_with("hand");
    let aim_joint = actors[d].model.rig.joint(if s.aim == "chin" { "head" } else { &s.aim }).map_err(|e| e.0)?;
    let aim = Target::Aim { target: d, joint: aim_joint, attacker: a, surface: surface_of(&s.aim), chin: s.aim == "chin" };
    let root = |i: usize, w: f64| actors[i].figure(w).joints[actors[i].model.root];
    let toward = flat_dir(root(a, t), root(d, t));
    // the attacker's body: a fist — a turn into the punch, a foot — leaning
    // back (before the lunge: it must see the strike's whole pose)
    let body = weighted(&[
        (t - s.windup - 0.05, 0.0, Ease::InOut),
        (t, 1.0, Ease::InOut),
        (t + s.hold, 1.0, Ease::InOut),
        (t + s.hold + s.recover, 0.0, Ease::InOut),
    ]);
    let angles: Vec<(&str, Vec3)> = if hand {
        let side = if limb_name.ends_with("_r") { 1.0 } else { -1.0 };
        vec![("spine", Vec3 { x: 8.0, y: 0.0, z: 22.0 * side })]
    } else {
        vec![("spine", Vec3 { x: -22.0, y: 0.0, z: 0.0 }), ("neck", Vec3 { x: 12.0, y: 0.0, z: 0.0 })]
    };
    let angles = both(&actors[a], &angles);
    actors[a].auto_layers.push(Layer { angles, shift: (0.0, 0.0), lift: 0.0, weight: body });
    actors[a].forget();

    // The target is further than the limb reaches — in the wind-up lean
    // forward by exactly what is missing: the strike closes the distance, not
    // a step picked by hand.
    let ch = chain(&actors[a], s.limb)?;
    let js = actors[a].figure(t);
    let length = (js.joints[ch[1]] - js.joints[ch[0]]).length() + (js.joints[ch[2]] - js.joints[ch[1]]).length();
    let start = js.joints[ch[0]];
    let goal = aim.at(t, &World { actors, props });
    let goal = if let Some(bh) = s.blocked {
        let head = actors[d].model.rig.joint("head").map_err(|e| e.0)?;
        let block = Target::Block { defender: d, head, toward };
        let mut fig = (*actors[d].figure(t)).clone();
        crate::ik::solve(&mut fig.joints, chain(&actors[d], bh)?, block.at(t, &World { actors, props }), 1.0, None);
        fig.joints[actors[d].model.rig.parent(bh).expect("blocking arm")].lerp(fig.joints[bh], 0.55) - toward * 0.10
    } else {
        goal
    };
    let gap = (goal - start).length() - LUNGE_REACH * length;
    if gap > 0.0 && !s.miss {
        let fnorm = (m::powi(goal.x - start.x, 2) + m::powi(goal.z - start.z, 2)).sqrt();
        let fnorm = if fnorm == 0.0 { 1.0 } else { fnorm };
        // how far to go on the ground for the distance to the target to be right
        let dy = goal.y - start.y;
        let want = (m::powi(LUNGE_REACH * length, 2) - dy * dy).max(0.0).sqrt();
        let step = (fnorm - want).max(0.0);
        // but not into the opponent: heads no closer than the radii with a margin
        let step = no_clash(actors, props, a, d, t, toward, step);
        // a fist — shifting the weight: leaned in and back; a foot — stays (a step into the kick)
        let mut lunge = vec![(t - s.windup - 0.12, 0.0, Ease::InOut), (t, 1.0, Ease::Out)];
        if hand {
            lunge.push((t + s.hold + s.recover, 0.0, Ease::InOut));
        } else {
            // with a hop — fastest in the middle, when the foot is highest
            lunge[1] = (t, 1.0, Ease::InOut);
        }
        actors[a].auto_layers.push(Layer { angles: Vec::new(), shift: (toward.x * step, toward.z * step), lift: 0.0, weight: weighted(&lunge) });
        if !hand && step > 0.04 {
            // a kick — a hop: the standing foot in the air while the body
            // carries forward, else it slides on the ground without a step
            let t0 = t - s.windup - 0.12;
            let hop = weighted(&[(t0, 0.0, Ease::InOut), ((t0 + t) / 2.0, 1.0, Ease::InOut), (t, 0.0, Ease::InOut)]);
            actors[a].auto_layers.push(Layer { angles: Vec::new(), shift: (0.0, 0.0), lift: (0.3 * step).min(0.14), weight: hop });
        }
        actors[a].forget();
    }
    let mut target = aim.clone();
    if s.miss {
        // where the target was when the strike began, and a little further
        // on by inertia, but above the dodging head: the arm passes by, not through it
        let world = World { actors, props };
        let fixed = aim.at(t - s.windup, &world) + toward * 0.12;
        let head_j = actors[d].model.rig.joint("head").map_err(|e| e.0)?;
        let head = actors[d].figure(t).joints[head_j];
        target = Target::Point(Vec3 { y: fixed.y.max(head.y + MISS_CLEAR), ..fixed });
    } else if let Some(bh) = s.blocked {
        // a block: the defender's hand in front of the face, towards the punch; the strike into it
        let head = actors[d].model.rig.joint("head").map_err(|e| e.0)?;
        let block = Target::Block { defender: d, head, toward };
        let dch = chain(&actors[d], bh)?;
        actors[d].reaches.push(Reach { chain: dch, target: block.clone(), weight: window(t, t + s.hold, s.windup * 1.6, s.recover), pole: None });
        actors[d].forget();
        target = Target::BlockContact { defender: d, hand: bh, offset: toward * -0.10 };
    }
    let weight =
        weighted(&[(t - s.windup, 0.0, Ease::InOut), (t, 1.0, Ease::In), (t + s.hold, 1.0, Ease::InOut), (t + s.hold + s.recover, 0.0, Ease::InOut)]);
    actors[a].reaches.push(Reach { chain: ch, target, weight, pole: None });
    actors[a].forget();
    if s.miss {
        return Ok(());
    }
    let k = if s.blocked.is_some() { 0.35 } else { 1.0 };
    // the one hit is thrown back and pushed away from the attacker
    let release = t + s.hold;
    let snap = weighted(&[(release, 0.0, Ease::InOut), (release + 0.05, 1.0, Ease::Out), (release + 0.55, 0.0, Ease::InOut)]);
    let hit = both(
        &actors[d],
        &[
            ("spine", Vec3 { x: -24.0 * k, y: 0.0, z: 0.0 }),
            ("neck", Vec3 { x: -20.0 * k, y: 0.0, z: 0.0 }),
            ("arm", Vec3 { x: -25.0 * k, y: 20.0 * k, z: 0.0 }),
        ],
    );
    actors[d].auto_layers.push(Layer { angles: hit, shift: (0.0, 0.0), lift: 0.0, weight: snap });
    actors[d].forget();
    let delta = toward * (s.force * k);
    let duration = (0.32 + 0.2 * delta.length() / actors[d].scale).clamp(0.32, 0.55);
    let mut airborne = false;
    if delta.length() > 1e-6
        && let Some(step) = crate::recoil::Recoil::new(&actors[d], release, duration, delta)
    {
        airborne = step.airborne;
        actors[d].recoils.push(step);
    }
    let away = weighted(&[(release, 0.0, Ease::InOut), (release + duration, 1.0, if airborne { Ease::Linear } else { Ease::InOut })]);
    actors[d].auto_layers.push(Layer { angles: Vec::new(), shift: (delta.x, delta.z), lift: 0.0, weight: away });
    actors[d].forget();
    fx.0.push((t, s.stop * k));
    if s.flash && s.blocked.is_none() {
        fx.1.push(t);
    }
    fx.2.push(crate::camera::Shake::new(t, s.shake * k, 0.2));
    Ok(())
}

fn apply(shot: &mut crate::Shot, action: &Action) -> Result<(), String> {
    if matches!(action, Action::Grab { .. }) {
        return crate::grip::grab(shot, action);
    }
    let (actors, props) = (&mut shot.actors, &mut shot.props);
    match *action {
        Action::Grab { .. } => unreachable!("grabs are contact-gated"),
        Action::Release { actor, hand, prop, at: t, ref onto, reach, settle } => {
            let world = World { actors, props };
            let Some(held @ Motion::Held { actor: owner, hand: grip, offset, .. }) = props[prop].segment(t).cloned() else {
                // A missed acquisition leaves its later release without a binding.
                if shot.grab_reports.iter().rev().find(|r| r.prop == prop && r.actor == actor && r.hand == hand && r.at <= t).is_some_and(|r| !r.attached()) {
                    return Ok(());
                }
                return Err(format!("release at {t}: prop '{}' is not in a hand", props[prop].name));
            };
            if owner != actor || grip != hand {
                return Err(format!("release at {t}: prop '{}' is held by a different hand", props[prop].name));
            }
            if !matches!(onto, Onto::Nothing)
                && let Some((acquired, _)) = props[prop].motion.iter().rev().find(|(from, _)| *from <= t + 1e-9)
                && t - reach < *acquired - 1e-9
            {
                return Err(format!("release at {t}: placement reach for '{}' begins before acquisition", props[prop].name));
            }
            let ch = chain(&actors[actor], hand)?;
            let mut floor = 0.0;
            let place = match onto {
                Onto::Nothing => None,
                Onto::Point(p) => Some(*p),
                Onto::Prop(s) => Some(near_spot(&props[*s], t, actors[actor].figure(t).joints[ch[0]], &world)),
            };
            let mut new_reach = None;
            if let Some(place) = place {
                floor = place.y;
                // the hand goes where the thing will stand exactly on the spot
                let heading = actors[actor].facing.at(t);
                let target = place - Mat3::rot_y(heading) * offset;
                // put down in two phases: over the spot, then straight down —
                // lowered on the way it would touch the support early and slide
                let over = t - reach * (1.0 - DESCEND);
                let weight = Keys::new(Ease::InOut).key(t - reach, 0.0).key(over, 1.0).key(t, 1.0).key(t + settle, 0.0).into();
                new_reach = Some(Reach { chain: ch, target: Target::Descend { at: target, over, down: t, hover: HOVER }, weight, pole: None });
            }
            if let Some(r) = new_reach {
                let mut contact = crate::Contact::new(&format!("release:{}", props[prop].name), r.chain, r.target, t - reach, t + settle);
                contact.weight = r.weight;
                let side = if actors[actor].model.rig.joints()[hand].name.ends_with("_r") { -1.0 } else { 1.0 };
                contact.pole = Some(Mat3::rot_y(actors[actor].facing.at(t)) * Vec3 { x: 0.4 * side, y: -0.5, z: -0.6 });
                let goal = contact.target.at(t, &World { actors, props });
                contact.approach = Some(
                    crate::ik::flat_dir(goal, root_at(&actors[actor], t)) * 0.25 * actors[actor].scale + Vec3::UP * (0.12 * actors[actor].scale),
                );
                contact.withdrawal = Some((t, contact.approach));
                actors[actor].auto_contacts.push(contact);
                actors[actor].forget();
                props[prop].motion.push((t - reach, Motion::Above { held: Box::new(held.clone()), floor }));
            }
            let world = World { actors, props };
            let (pos, rot) = held.at(t, &world);
            let next = match onto {
                Onto::Prop(s) => {
                    // on a prop: rides with it from now on
                    let (sp, sr) = props[*s].rest(t, &world);
                    let local = sr.transpose() * (pos - sp);
                    let top = props[*s].shape.top();
                    Motion::Ride {
                        support: *s,
                        offset: Vec3 { x: local.x, y: top, z: local.z },
                        yaw: rot.heading() - sr.heading(),
                        drop: (local.y - top).max(0.0),
                        t0: t,
                    }
                }
                _ if pos.y > floor + 1e-3 => Motion::Fall { pos, rot, floor, t0: t },
                _ => Motion::Rest { pos: Vec3 { y: floor, ..pos }, rot },
            };
            if let Some(c) = props[prop].contacts.iter_mut().rev().find(|c| c.2 == actor && c.3 == hand && c.1.is_infinite()) {
                c.1 = t + settle;
            }
            props[prop].motion.push((t, next));
        }
        Action::Push { actor, prop, start, end, hands, reach, settle, height, spread, .. } => {
            let world = World { actors, props };
            let (pos, rot) = props[prop].rest(start, &world);
            let r0 = root_at(&actors[actor], start);
            let (n, tangent) = facing_side(pos, &rot, r0);
            let slide = Motion::Slide { actor, offset: Vec3 { x: pos.x - r0.x, y: 0.0, z: pos.z - r0.z }, y: pos.y, rot };
            let (ep, er) = slide.at(end, &world);
            // which palm is further to the character's left (+X of its frame)
            let left = Mat3::rot_y(actors[actor].facing.at(start)) * Vec3 { x: 1.0, y: 0.0, z: 0.0 };
            let face = surface(&props[prop].shape, n, height);
            let (mut a, mut b) = (face + tangent * spread, face - tangent * spread);
            if (rot * a).dot(left) < 0.0 {
                std::mem::swap(&mut a, &mut b);
            }
            props[prop].motion.push((start, slide));
            props[prop].motion.push((end, Motion::Rest { pos: ep, rot: er }));
            // elbows down and slightly out: so when pushing, not as in a walk's swing
            for (hand, local, side) in [(hands[0], a, 1.0), (hands[1], b, -1.0)] {
                let ch = chain(&actors[actor], hand)?;
                let pole = Vec3 { x: 0.0, y: -1.0, z: 0.0 } + left * (0.6 * side);
                let mut contact = crate::Contact::new(
                    &format!("push:{}:{hand}", props[prop].name),
                    ch,
                    Target::OnProp { prop, local },
                    start - reach,
                    end + settle,
                );
                contact.weight = window(start, end, reach, settle);
                contact.pole = Some(pole);
                contact.approach = Some(rot * n * (0.20 * actors[actor].scale) + Vec3::UP * (0.08 * actors[actor].scale));
                contact.withdrawal = Some((end, contact.approach));
                actors[actor].auto_contacts.push(contact);
            }
        }
        Action::Strike(_) => unreachable!("strikes are compiled by strike()"),
    }
    Ok(())
}
