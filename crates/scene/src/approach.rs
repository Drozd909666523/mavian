//! Explicit, bounded preparation for prop contacts. Plans are compiled once.
use crate::{Action, Actor, Contact, Motion, Prop, Target, actions, obstacle::Obstacle, props::World};
use mavian_core::{Mat3, Vec3, noise::smooth};
use mavian_rig::{Figure, Part, Side};

#[derive(Clone, Copy, Debug)]
pub struct Approach {
    /// Maximum horizontal world-space translation, metres. One short step may use less.
    pub max_distance: f64,
    /// Preparation ends before the hand begins reaching.
    pub duration: f64,
}
impl Approach {
    pub fn new(max_distance: f64) -> Self {
        Self { max_distance, duration: 0.8 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepStatus {
    Applied,
    Limited,
    Unneeded,
    Blocked,
    PoseLocked,
    AuthoredMotion,
    ContactBusy,
    Unsupported,
    NoTime,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StepReport {
    pub prop: usize,
    pub from: f64,
    pub to: f64,
    pub max_distance: f64,
    pub shift: Vec3,
    pub error_before: f64,
    pub error_after: f64,
    pub status: StepStatus,
}

#[derive(Clone, Debug)]
struct Foot {
    chain: [usize; 3],
    toe: usize,
    from: [Vec3; 2],
}

#[derive(Clone, Debug)]
pub(crate) struct Step {
    pub from: f64,
    pub to: f64,
    delta: Vec3,
    first: usize,
    feet: [Foot; 2],
    pole: Vec3,
    height: f64,
    bend_reserve: f64,
}
impl Step {
    fn new(actor: &Actor, from: f64, to: f64, delta: Vec3) -> Option<Self> {
        let fig = actor.figure(from);
        let foot = |side| {
            let f = actor.model.rig.foot(side)?;
            Some(Foot { chain: actions::chain(actor, f.heel).ok()?, toe: f.toe, from: [fig.joints[f.heel], fig.joints[f.toe]] })
        };
        let feet = [foot(Side::L)?, foot(Side::R)?];
        let first = match actor.planted_foot(from) {
            Some(Side::L) => 1,
            Some(Side::R) => 0,
            None => usize::from(feet[1].from[0].dot(delta) > feet[0].from[0].dot(delta)),
        };
        Some(Self {
            from,
            to,
            delta,
            first,
            feet,
            pole: Mat3::rot_y(actor.facing.at(from)) * Vec3 { x: 0., y: 0., z: 1. },
            height: (delta.length() * 0.2).min(0.05 * actor.scale),
            bend_reserve: 0.008 * actor.scale,
        })
    }
    pub fn active(&self, w: f64) -> bool {
        self.from <= w && w < self.to
    }
    pub fn support(&self, w: f64) -> Side {
        let i = if (w - self.from) / (self.to - self.from) < 0.6 { 1 - self.first } else { self.first };
        if i == 0 { Side::L } else { Side::R }
    }
    pub fn apply(&self, fig: &mut Figure, w: f64) -> f64 {
        if w <= self.from || self.delta.length() < 1e-10 {
            return 0.0;
        }
        let u = ((w - self.from) / (self.to - self.from)).clamp(0., 1.);
        // Transfer weight while both feet are down, then move the leading
        // foot and follow with the trailing foot. This keeps retreating knees
        // from swinging into the nearby object before the body has moved back.
        let shift = self.delta * smooth((u / 0.6).clamp(0., 1.));
        for p in &mut fig.joints {
            *p = *p + shift;
        }
        fig.pivot = fig.pivot + shift;
        if w >= self.to {
            return 0.0;
        }
        let goals: Vec<_> = self
            .feet
            .iter()
            .enumerate()
            .map(|(i, foot)| {
                let t = ((u - if i == self.first { 0.2 } else { 0.6 }) / 0.4).clamp(0., 1.);
                let lift = 16.0 * t * t * (1.0 - t) * (1.0 - t) * self.height;
                foot.from.map(|p| p + self.delta * smooth(t) + Vec3::UP * lift)
            })
            .collect();
        let reference = fig.clone();
        let mut lower = 0.0f64;
        for (foot, goal) in self.feet.iter().zip(&goals) {
            let [hip, knee, heel] = foot.chain;
            let h = fig.joints[hip];
            let length = (fig.joints[knee] - h).length() + (fig.joints[heel] - fig.joints[knee]).length();
            let d = h - goal[0];
            let reach = length - self.bend_reserve * 16.0 * u * u * (1.0 - u) * (1.0 - u);
            let vertical = (reach * reach - d.x * d.x - d.z * d.z).max(0.0).sqrt();
            lower = lower.max(d.y - vertical);
        }
        for p in &mut fig.joints {
            p.y -= lower;
        }
        let mut error = 0.0f64;
        for (foot, goal) in self.feet.iter().zip(&goals) {
            error = error.max(crate::contact::place(&mut fig.joints, &reference, foot.chain, goal[0], Some(self.pole)));
            fig.joints[foot.toe] = fig.joints[foot.chain[2]] + goal[1] - goal[0];
        }
        error
    }
}

type Capsule = (usize, usize, f64, f64);
fn capsules(actor: &Actor, line: f64) -> Vec<Capsule> {
    let s = actor.scale;
    let mut out = Vec::new();
    for part in &actor.model.parts {
        match part {
            Part::Body { a, b, rx, rz, .. } => out.push((*a, *b, rx.0.max(rz.0) * s, rx.1.max(rz.1) * s)),
            Part::Head { joint, radius, .. } | Part::Ball { joint, radius } => out.push((*joint, *joint, radius * s, radius * s)),
            Part::Limb { chain, radii, .. } => {
                for i in 0..chain.len() - 1 {
                    out.push((chain[i], chain[i + 1], radii[i] * s, radii[i + 1] * s));
                }
            }
            Part::Stick { a, b, w0, w1, .. } => out.push((*a, *b, w0 * line * s * 0.5, w1 * line * s * 0.5)),
        }
    }
    out
}

fn obstacles(actors: &[Actor], props: &[Prop], actor: usize, w: f64, fig: &Figure) -> Vec<Obstacle> {
    fn held(m: &Motion, actor: usize, heading: f64, fig: &Figure) -> Option<(Vec3, Mat3)> {
        match m {
            Motion::Held { actor: a, hand, offset, yaw } if *a == actor => {
                Some((fig.joints[*hand] + Mat3::rot_y(heading) * *offset, Mat3::rot_y(heading + yaw)))
            }
            Motion::Above { held: m, floor } => held(m, actor, heading, fig).map(|(p, r)| (Vec3 { y: p.y.max(*floor), ..p }, r)),
            _ => None,
        }
    }
    let world = World { actors, props };
    props
        .iter()
        .enumerate()
        .map(|(prop, p)| {
            let (pos, rot) = p.segment(w).and_then(|m| held(m, actor, actors[actor].facing.at(w), fig)).unwrap_or_else(|| p.rest(w, &world));
            Obstacle { prop, shape: p.shape.clone(), pos, rot }
        })
        .collect()
}
fn gap(o: &Obstacle, fig: &Figure, (a, b, ra, rb): Capsule) -> f64 {
    let mid = fig.joints[a].lerp(fig.joints[b], 0.5);
    let bound = (fig.joints[b] - fig.joints[a]).length() * 0.5 + ra.max(rb);
    let d = o.distance(mid).0 - bound;
    if d > 0.002 {
        return d;
    }
    o.capsule(fig.joints[a], ra, fig.joints[b], rb).0
}

fn safe(actors: &[Actor], props: &[Prop], actor: usize, step: &Step, parts: &[Capsule]) -> bool {
    let samples = ((step.to - step.from) * 120.0).ceil().max(2.0) as usize;
    for n in 0..=samples {
        let w = step.from + (step.to - step.from) * n as f64 / samples as f64;
        let raw = actors[actor].figure(w);
        let mut fig = (*raw).clone();
        if step.apply(&mut fig, w) > 1e-7
            || raw.joints[actors[actor].model.root].y - fig.joints[actors[actor].model.root].y > 0.09 * actors[actor].scale
        {
            return false;
        }
        let original = obstacles(actors, props, actor, w, &raw);
        for o in obstacles(actors, props, actor, w, &fig) {
            for &part in parts {
                let floor = if n == samples { 0.0 } else { gap(&original[o.prop], &raw, part).min(0.0) };
                if gap(&o, &fig, part) < floor - 0.001 {
                    return false;
                }
            }
        }
    }
    true
}

pub(crate) fn prepare(actors: &mut [Actor], props: &[Prop], action: &Action, actions: &[Action], line: f64) -> Result<(), String> {
    let (index, prop, at, reach, height, hands, spread, options) = match *action {
        Action::Grab { actor, hand, prop, at, reach, height, step: Some(s), .. } => (actor, prop, at, reach, height, vec![hand], 0., s),
        Action::Push { actor, prop, start, reach, height, hands, spread, step: Some(s), .. } => {
            (actor, prop, start, reach, height, hands.to_vec(), spread, s)
        }
        _ => return Ok(()),
    };
    if !options.max_distance.is_finite()
        || options.max_distance <= 0.0
        || !options.duration.is_finite()
        || options.duration <= 0.0
        || !at.is_finite()
        || !reach.is_finite()
        || reach < 0.0
    {
        return Err("auto_step needs a positive finite distance and duration, and a finite contact time and nonnegative reach".into());
    }
    let a = actors.get(index).ok_or("auto_step actor does not exist")?;
    let p = props.get(prop).ok_or("auto_step prop does not exist")?;
    let (pos, rot) = p.rest(at, &World { actors, props });
    let raw = a.figure(at);
    let root = raw.joints[a.model.root];
    let (normal, tangent) = actions::facing_side(pos, &rot, root);
    let face = if hands.len() == 1 {
        crate::grip::surface(a, hands[0], &p.shape, normal, height, line)
    } else {
        actions::surface(&p.shape, normal, height)
    };
    let left = Mat3::rot_y(a.facing.at(at)) * Vec3 { x: 1., y: 0., z: 0. };
    let mut local = [face + tangent * spread, face - tangent * spread];
    if (rot * local[0]).dot(left) < 0. {
        local.swap(0, 1);
    }
    let contacts: Vec<_> = hands
        .iter()
        .enumerate()
        .map(|(i, &hand)| {
            let target = pos + rot * local[i];
            Ok((Contact::new(&format!("prepare:{i}"), actions::chain(a, hand)?, Target::Point(target), at - 1., at + 1.), target))
        })
        .collect::<Result<_, String>>()?;
    let mut assistance_contacts = contacts.clone();
    assistance_contacts.extend(
        a.contacts
            .iter()
            .chain(&a.auto_contacts)
            .filter(|c| c.priority == crate::Priority::Support && c.weight_at(at) > 0.0)
            .map(|c| (c.clone(), raw.joints[c.chain[2]])),
    );
    let direction = crate::ik::flat_dir(root, pos);
    let score = |shift: Vec3| {
        let mut fig = (*raw).clone();
        for p in &mut fig.joints {
            *p = *p + shift;
        }
        let obs = obstacles(actors, props, index, at, &fig);
        let assisted = crate::posture::assist(a, at, &fig, &fig, &assistance_contacts, &obs);
        let mut cost = shift.dot(shift) * 0.0001;
        let mut error = 0.0f64;
        for (contact, point) in &contacts {
            let [r, b, c] = contact.chain;
            let length = (fig.joints[b] - fig.joints[r]).length() + (fig.joints[c] - fig.joints[b]).length();
            let distance = (*point - assisted.joints[r]).length();
            let residual = distance - distance.clamp(length * 0.65, length * 0.92);
            cost += residual * residual;
            error = error.max((distance - length).max(0.0));
        }
        (cost, error)
    };
    let (before, error_before) = score(Vec3::ZERO);
    let end = at - reach;
    let start = end - options.duration;
    let mut report = StepReport {
        prop,
        from: start,
        to: end,
        max_distance: options.max_distance,
        shift: Vec3::ZERO,
        error_before,
        error_after: error_before,
        status: StepStatus::Unneeded,
    };
    let finish = |actors: &mut [Actor], report| {
        actors[index].step_reports.push(report);
    };
    if a.pose_lock {
        report.status = StepStatus::PoseLocked;
        finish(actors, report);
        return Ok(());
    }
    let parts = capsules(a, line);
    let zero_clear = obstacles(actors, props, index, at, &raw).iter().all(|o| parts.iter().all(|&p| gap(o, &raw, p) >= -0.001));
    let raw_error = contacts
        .iter()
        .map(|(contact, point)| {
            let [r, b, c] = contact.chain;
            let length = (raw.joints[b] - raw.joints[r]).length() + (raw.joints[c] - raw.joints[b]).length();
            ((*point - raw.joints[r]).length() - length).max(0.0)
        })
        .fold(0.0, f64::max);
    // Do not add a step when explicitly permitted torso assistance already
    // closes the reach deficit, or the original stance is comfortable and clear.
    if zero_clear && (before < 1e-8 || (raw_error > 0.002 && error_before <= 0.002)) {
        finish(actors, report);
        return Ok(());
    }
    if start < 0. {
        report.status = StepStatus::NoTime;
        finish(actors, report);
        return Ok(());
    }
    let Some(zero) = Step::new(a, start, end, Vec3::ZERO) else {
        report.status = StepStatus::Unsupported;
        finish(actors, report);
        return Ok(());
    };
    let from = a.figure(start);
    let mut moving = false;
    let mut busy = a.contacts.iter().chain(&a.auto_contacts).any(|c| c.from < end && start < c.to);
    // Later actions can already reserve this time through their windup/reach.
    // Inspect the complete description, not only contacts compiled so far.
    busy |= actions.iter().filter(|other| !std::ptr::eq(*other, action)).any(|other| {
        let interval = match other {
            Action::Grab { actor, at, reach, settle, .. } | Action::Release { actor, at, reach, settle, .. } if *actor == index => {
                Some((at - reach, at + settle))
            }
            Action::Push { actor, start, end, reach, settle, .. } if *actor == index => Some((start - reach, end + settle)),
            Action::Strike(s) if s.attacker == index || s.target == index => Some((s.at - s.windup * 1.6 - 0.12, s.at + s.hold + s.recover + 0.7)),
            _ => None,
        };
        // A preparation cannot be inserted before an already compiled
        // contact either: that would invalidate its targets/prop offsets.
        interval.is_some_and(|(from, to)| start < to && (from < end || other.time() <= action.time()))
    });
    for sample in 0..=32 {
        let w = start + options.duration * sample as f64 / 32.;
        let f = a.figure(w);
        moving |=
            (f.joints[a.model.root] - from.joints[a.model.root]).length() > 1e-6 * a.scale || (a.facing.at(w) - a.facing.at(start)).abs() > 1e-4;
        for foot in &zero.feet {
            moving |= (f.joints[foot.chain[2]] - foot.from[0]).length() > 1e-6 * a.scale
                || (f.joints[foot.toe] - foot.from[1]).length() > 1e-6 * a.scale
                || foot.from.iter().any(|p| p.y.abs() > 0.005 * a.scale);
        }
        busy |= a.contacts.iter().chain(&a.auto_contacts).any(|c| c.weight_at(w) > 0.0);
    }
    busy |= a.approach_steps.iter().any(|s| s.from < end && start < s.to);
    if moving || busy {
        report.status = if moving { StepStatus::AuthoredMotion } else { StepStatus::ContactBusy };
        finish(actors, report);
        return Ok(());
    }
    // One stationary preparation, bounded by anatomy, the author's distance and
    // a conservative foot speed. No search around an obstacle or extra step.
    let limit = options.max_distance.min(0.35 * a.scale).min(options.duration * 0.6 * a.scale);
    if limit < 0.01 * a.scale {
        report.status = StepStatus::Limited;
        finish(actors, report);
        return Ok(());
    }
    let mut candidates: Vec<_> = (-64..=64)
        .map(|i| {
            let delta = direction * (limit * i as f64 / 64.);
            let (cost, error) = score(delta);
            (cost, delta, error)
        })
        .filter(|(_, d, _)| d.length() >= 0.01 * a.scale)
        .collect();
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.length().total_cmp(&b.1.length())).then(a.1.z.total_cmp(&b.1.z)));
    for (cost, delta, error) in candidates {
        if cost >= before && zero_clear {
            break;
        }
        let step = Step::new(a, start, end, delta).expect("feet checked above");
        if safe(actors, props, index, &step, &parts) {
            report.shift = delta;
            report.error_after = error;
            report.status = if error > 0.002 { StepStatus::Limited } else { StepStatus::Applied };
            actors[index].approach_steps.push(step);
            actors[index].forget();
            finish(actors, report);
            return Ok(());
        }
    }
    report.status = StepStatus::Blocked;
    finish(actors, report);
    Ok(())
}
