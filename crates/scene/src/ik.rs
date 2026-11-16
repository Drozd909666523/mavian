//! Inverse kinematics of an arm (or a leg): the end of a chain reaches for a
//! point (a port of the MVP's `engine/ik.py`).
//!
//! Solved by positions, not angles: the shoulder stays where it is, the elbow
//! and the hand are placed so the hand is at the target. The bend plane comes
//! from the pose (where the elbow pointed before IK) or from a pole — so the
//! solution is continuous and does not flip the elbow. The weight blends the
//! pose and the solution smoothly: 0 — as posed, 1 — at the target.

use crate::props::World;
use mavian_core::{Track, Vec3};
use mavian_rig::Figure;

/// Where a hand reaches: a pure function of world time and the scene.
#[derive(Clone, Debug)]
pub enum Target {
    Point(Vec3),
    /// over `at` (by `hover`) until `over`, then smoothly down to it by `down`:
    /// a hand putting a thing down
    Descend {
        at: Vec3,
        over: f64,
        down: f64,
        hover: f64,
    },
    /// a point fixed on a prop (prop index, point in the prop's frame)
    OnProp {
        prop: usize,
        local: Vec3,
    },
    /// a strike's aim: a joint of another character, on the surface facing
    /// the attacker (`surface` from the joint), or the chin — the bottom of
    /// the head at the front, as an uppercut hits from below
    Aim {
        target: usize,
        joint: usize,
        attacker: usize,
        surface: f64,
        chin: bool,
    },
    /// a block: the defender's hand in front of the face, towards the punch
    Block {
        defender: usize,
        head: usize,
        toward: Vec3,
    },
    /// Actual solved blocking fist, rather than its possibly unreachable request.
    BlockContact {
        defender: usize,
        hand: usize,
        offset: Vec3,
    },
    /// another target moved by a vector
    Offset(Box<Target>, Vec3),
}

/// head radius plus half a limb: where a strike's fist or foot centre lands
pub const CHIN_REACH: f64 = 0.19 + 0.06;
/// a block is this far in front of the face (further, a stepping head enters it)
pub const BLOCK_OUT: f64 = 0.22;

/// The ground direction from a to b (unit, x and z).
pub fn flat_dir(a: Vec3, b: Vec3) -> Vec3 {
    let (dx, dz) = (b.x - a.x, b.z - a.z);
    let n = (dx * dx + dz * dz).sqrt();
    let n = if n == 0.0 { 1.0 } else { n };
    Vec3 { x: dx / n, y: 0.0, z: dz / n }
}

impl Target {
    pub fn at(&self, w: f64, world: &World) -> Vec3 {
        match self {
            Target::Point(p) => *p,
            Target::Descend { at, over, down, hover } => {
                let u = ((w - over) / (down - over).max(1e-6)).clamp(0.0, 1.0);
                let u = u * u * (3.0 - 2.0 * u);
                *at + Vec3::UP * (hover * (1.0 - u))
            }
            Target::OnProp { prop, local } => {
                let (pos, rot) = world.props[*prop].rest(w, world);
                pos + rot * *local
            }
            Target::Aim { target, joint, attacker, surface, chin } => {
                let a = &world.actors[*attacker];
                let root = a.figure(w).joints[a.model.root];
                let p = world.actors[*target].figure(w).joints[*joint];
                let f = flat_dir(p, root);
                if *chin {
                    let dir = (Vec3 { x: 0.0, y: -0.17, z: 0.0 } + f * 0.09).norm();
                    return p + dir * CHIN_REACH;
                }
                p + f * *surface
            }
            Target::Block { defender, head, toward } => {
                let h = world.actors[*defender].figure(w).joints[*head];
                h + *toward * -BLOCK_OUT + Vec3 { x: 0.0, y: -0.08, z: 0.0 }
            }
            Target::BlockContact { defender, hand, offset } => {
                let actor = &world.actors[*defender];
                let f = actor.figure_ik(w, world);
                let elbow = actor.model.rig.parent(*hand).expect("blocking arm");
                f.joints[elbow].lerp(f.joints[*hand], 0.55) + *offset
            }
            Target::Offset(t, by) => t.at(w, world) + *by,
        }
    }
}

/// A chain of three joints (shoulder, elbow, hand) reaching for a target.
#[derive(Clone, Debug)]
pub struct Reach {
    pub chain: [usize; 3],
    pub target: Target,
    /// 0 — the pose, 1 — the target
    pub weight: Track<f64>,
    /// where the elbow points (world vector); none — as in the pose
    pub pole: Option<Vec3>,
}

/// A constraint: an arm does not enter another character's head. If a bone
/// is closer than `gap` to it, the elbow and the hand move away by exactly
/// what is missing — as an advancing opponent knocks a guard aside. The
/// other's head is taken without its IK (else a loop: their arms avoid ours).
#[derive(Clone, Debug)]
pub struct Avoid {
    pub other: usize,
    pub head: usize,
    pub chains: Vec<[usize; 3]>,
    pub gap: f64,
}

/// Place the elbow and the hand so the hand reaches `target` (with `weight`).
pub fn solve(joints: &mut [Vec3], chain: [usize; 3], target: Vec3, weight: f64, pole: Option<Vec3>) {
    if weight <= 0.0 {
        return;
    }
    let [a, b, c] = chain;
    let (s, e, h) = (joints[a], joints[b], joints[c]);
    let (l1, l2) = ((e - s).length(), (h - e).length());
    let to = target - s;
    let d = to.length();
    if d < 1e-9 || l1 < 1e-9 || l2 < 1e-9 {
        return;
    }
    let u = to * (1.0 / d);
    // no further than the chain reaches
    let d = ((l1 - l2).abs() + 1e-4).max(d.min((l1 + l2) * 0.999));
    // the bend plane: from the pole or from the pose (where the elbow pointed
    // off the shoulder–target line)
    let mut bend = match pole {
        Some(p) => p - u * p.dot(u),
        None => e - (s + u * (e - s).dot(u)),
    };
    if bend.length() < 1e-6 {
        bend = u.cross(Vec3::UP);
        if bend.length() < 1e-6 {
            bend = Vec3 { x: 1.0, y: 0.0, z: 0.0 };
        }
    }
    let n = bend.norm();
    let along = (l1 * l1 - l2 * l2 + d * d) / (2.0 * d);
    let up = (l1 * l1 - along * along).max(0.0).sqrt();
    let e2 = s + u * along + n * up;
    let h2 = s + u * d;
    joints[b] = e.lerp(e2, weight);
    joints[c] = h.lerp(h2, weight);
}

fn seg(p: Vec3, a: Vec3, b: Vec3) -> (f64, f64) {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.dot(ab).max(1e-9)).clamp(0.0, 1.0);
    ((p - (a + ab * t)).length(), t)
}

impl Avoid {
    pub fn apply(&self, joints: &mut [Vec3], w: f64, world: &World) {
        let head = world.actors[self.other].figure(w).joints[self.head];
        for &[a, b, c] in &self.chains {
            for (j0, j1, mv) in [(a, b, &[b][..]), (b, c, &[b, c][..])] {
                let (p0, p1) = (joints[j0], joints[j1]);
                let (d, t) = seg(head, p0, p1);
                if d >= self.gap {
                    continue;
                }
                let away = p0.lerp(p1, t) - head;
                let n = if away.length() > 1e-9 { away.norm() } else { Vec3::UP };
                let push = n * (self.gap - d);
                for &j in mv {
                    joints[j] = joints[j] + push;
                }
            }
        }
    }
}

/// The figure with its hands at their targets and constraints applied.
pub fn reach_all(fig: &Figure, reaches: &[Reach], avoids: &[Avoid], w: f64, world: &World) -> Figure {
    let mut out = fig.clone();
    for r in reaches {
        solve(&mut out.joints, r.chain, r.target.at(w, world), r.weight.at(w), r.pole);
    }
    for a in avoids {
        a.apply(&mut out.joints, w, world);
    }
    out
}
