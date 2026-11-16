//! Deterministic configuration paths for prop-action arms. Nodes retain exact
//! two-bone lengths; graph edges check both motion and intervening clearance.
use crate::{
    Priority, Shot,
    clearance::{ArmGuide, ArmPoint, binding, body_arm, body_solids},
    obstacle::{clearance, joint_radius},
};
use mavian_core::{Fingerprint, Mat3, StableHasher, Vec3, m};
use mavian_rig::Figure;
use std::{
    collections::BTreeMap,
    sync::{Mutex, OnceLock},
};

type Routes = Vec<(usize, Vec<ArmGuide>)>;
static CACHE: OnceLock<Mutex<BTreeMap<Fingerprint, Routes>>> = OnceLock::new();

#[derive(Clone)]
struct Node {
    elbow: Vec3,
    wrist: Vec3,
    pole: Vec3,
    cost: f64,
    total: f64,
    previous: usize,
    bucket: usize,
    goal: usize,
}
struct Layer {
    at: f64,
    base: Figure,
    nodes: Vec<Node>,
    active: bool,
}
impl Shot {
    fn arm_base(&self, actor: usize, w: f64, except: usize) -> Figure {
        let (base, m) = self.deformed(actor, w);
        let mut base = self.solve_actor_contacts(actor, w, &base, m).0;
        let a = &self.actors[actor];
        for guide in &a.clearance_guides {
            if guide.chain[2] == except {
                continue;
            }
            if let Some(point) = guide.at(w)
                && point.active
            {
                let parent = a.model.rig.parent(guide.chain[0]).unwrap_or(guide.chain[0]);
                let reference = base.clone();
                let pole = base.rots[parent] * point.pole;
                crate::contact::place(&mut base.joints, &reference, guide.chain, point.wrist, Some(pole));
            }
        }
        base
    }
    fn route_free(&self, actor: usize, chain: [usize; 3], w: f64, fig: &Figure, body: bool) -> f64 {
        let a = &self.actors[actor];
        let radii = chain.map(|j| joint_radius(a, j, self.style.line));
        let mut free = clearance(&self.arm_obstacles(actor, chain[2], w, fig), &fig.joints, chain, radii);
        if body {
            for (_, solid) in body_solids(a, fig) {
                free = free.min(body_arm(&solid, fig, chain, radii).0);
            }
        }
        let lift = self.held_lift(actor, chain[2], w, fig);
        if lift > 1e-8 {
            free = free.min(-lift);
        }
        free
    }
    fn route_owned(&self, actor: usize, chain: [usize; 3], w: f64) -> bool {
        self.actors[actor].auto_contacts.iter().any(|c| c.chain == chain && c.weight_at(w) > 0.0)
            || self.props.iter().any(|p| p.segment(w).and_then(binding).is_some_and(|(a, h, _, _)| a == actor && h == chain[2]))
    }
    fn arm_layer(&self, actor: usize, chain: [usize; 3], w: f64, windows: &[(f64, f64)], wide: bool) -> Result<Layer, String> {
        let a = &self.actors[actor];
        let planned = windows.iter().any(|&(from, to)| w >= from && w <= to);
        let base = if planned { self.arm_base(actor, w, chain[2]) } else { (*self.posed(actor, w).0).clone() };
        let [root, elbow, hand] = chain;
        let parent = a.model.rig.parent(root).unwrap_or(root);
        let inverse = base.rots[parent].transpose();
        let ramp = a.auto_contacts.iter().any(|c| c.chain == chain && c.weight_at(w) > 1e-8 && c.weight_at(w) < 1.0 - 1e-8);
        let pinned = a.contacts.iter().any(|c| c.priority == Priority::Support && c.weight_at(w) > 0.0 && c.chain.contains(&hand));
        let free = self.route_free(actor, chain, w, &base, self.route_owned(actor, chain, w));
        let make_node = |fig: &Figure, cost, bucket| Node {
            elbow: fig.joints[elbow],
            wrist: fig.joints[hand],
            pole: inverse * (fig.joints[elbow] - fig.joints[root]).norm(),
            cost,
            total: f64::INFINITY,
            previous: 0,
            bucket,
            goal: usize::MAX,
        };
        if !planned || pinned || (!ramp && free >= -0.0001 && (!wide || !self.route_owned(actor, chain, w))) {
            return Ok(Layer { at: w, nodes: vec![make_node(&base, 0.0, 0)], base, active: false });
        }
        let mut nodes = Vec::new();
        // Include the existing local solve so a planned detour can join its
        // corrected rest pose, rather than forcing a return to obstructed FK.
        let natural = self.posed(actor, w).0;
        if wide && self.route_free(actor, chain, w, &natural, self.route_owned(actor, chain, w)) >= -0.0001 {
            let hd = natural.joints[hand] - base.joints[hand];
            let ed = natural.joints[elbow] - base.joints[elbow];
            nodes.push(make_node(&natural, 8.0 * hd.dot(hd) + ed.dot(ed), 0));
        }
        if free >= 0.002 {
            nodes.push(make_node(&base, 0.0, 0));
        }
        let origin = base.joints[root];
        let initial = base.joints[hand];
        let owned = self.route_owned(actor, chain, w);
        let fixed = a.auto_contacts.iter().any(|c| c.chain == chain && c.weight_at(w) >= 1.0 - 1e-8);
        let mut projected = initial + Vec3::UP * self.held_lift(actor, hand, w, &base);
        if !fixed {
            for obstacle in self.arm_obstacles(actor, hand, w, &base) {
                if self.props[obstacle.prop].segment(w).and_then(binding).is_some_and(|(i, h, _, _)| i == actor && h == hand) {
                    continue;
                }
                let (d, normal) = obstacle.distance(projected);
                let radius = joint_radius(a, hand, self.style.line);
                if d < radius + 0.002 {
                    projected = projected + normal * (radius + 0.002 - d);
                }
            }
        }
        let mut goals = vec![projected];
        if wide && !fixed {
            goals.push(natural.joints[hand]);
        }
        if !fixed {
            let heading = Mat3::rot_y(a.facing.at(w));
            for x in [-1.0, 0.0, 1.0] {
                for y in [-1.0, 0.0, 1.0] {
                    for z in [-1.0, 0.0, 1.0] {
                        if x == 0.0 && y == 0.0 && z == 0.0 {
                            continue;
                        }
                        goals.push(goals[0] + heading * Vec3 { x, y, z } * (0.025 * a.scale));
                    }
                }
            }
            for center in [projected].into_iter().chain(wide.then_some(natural.joints[hand])) {
                for distance in [0.015, 0.04, 0.08, 0.12].into_iter().chain(if wide { &[0.2, 0.3, 0.4][..] } else { &[][..] }.iter().copied()) {
                    for axis in [Vec3 { x: 1.0, y: 0.0, z: 0.0 }, Vec3::UP, Vec3 { x: 0.0, y: 0.0, z: 1.0 }] {
                        for sign in [-1.0, 1.0] {
                            goals.push(center + heading * axis * (distance * sign * a.scale));
                        }
                    }
                }
            }
        }
        for (goal_index, goal) in goals.into_iter().enumerate() {
            let direction = (goal - origin).norm();
            let upper = base.joints[elbow] - origin;
            let mut bend = (upper - direction * upper.dot(direction)).norm();
            if bend.length() < 1e-8 {
                bend = direction.cross(Vec3::UP).norm();
            }
            if bend.length() < 1e-8 {
                bend = Vec3 { x: 1.0, y: 0.0, z: 0.0 };
            }
            let side = direction.cross(bend).norm();
            let angles =
                (0..64).map(|k| k as f64 * std::f64::consts::TAU / 64.0).chain([0.0174533, -0.0174533, 0.0349066, -0.0349066, 0.0698132, -0.0698132]);
            for angle in angles {
                let pole = bend * m::cos(angle) + side * m::sin(angle);
                let mut candidate = base.clone();
                crate::contact::place(&mut candidate.joints, &base, chain, goal, Some(pole));
                if self.route_free(actor, chain, w, &candidate, owned) < 0.002 {
                    continue;
                }
                let hd = candidate.joints[hand] - initial;
                let ed = candidate.joints[elbow] - base.joints[elbow];
                let cost = 8.0 * hd.dot(hd) + ed.dot(ed);
                let bucket = ((angle.rem_euclid(std::f64::consts::TAU) / std::f64::consts::TAU * 32.0).floor() as usize).min(31);
                let mut node = make_node(&candidate, cost, bucket);
                node.goal = goal_index;
                nodes.push(node);
            }
        }
        nodes.sort_by(|a, b| a.cost.total_cmp(&b.cost));
        let mut kept = Vec::new();
        let mut buckets = [false; 32];
        let mut goals = std::collections::BTreeSet::new();
        for (index, n) in nodes.into_iter().enumerate() {
            let new_goal = goals.insert(n.goal);
            if index < 128 || !buckets[n.bucket] || (wide && new_goal) {
                buckets[n.bucket] = true;
                kept.push(n);
            }
        }
        if kept.is_empty() {
            return Err(format!("no clear prop-action arm pose for actor {actor}, hand {hand} at {w:.6}s"));
        }
        Ok(Layer { at: w, base, nodes: kept, active: true })
    }
    fn arm_route(&self, actor: usize, chain: [usize; 3], times: &[f64], windows: &[(f64, f64)], wide: bool) -> Result<ArmGuide, String> {
        let mut layers: Vec<Layer> = Vec::new();
        let a = &self.actors[actor];
        let [root, elbow, hand] = chain;
        let parent = a.model.rig.parent(root).unwrap_or(root);
        for &w in times {
            let mut layer = self.arm_layer(actor, chain, w, windows, wide)?;
            if let Some(previous) = layers.last() {
                let dt = w - previous.at;
                let mid = (w + previous.at) * 0.5;
                let mid_base = self.arm_base(actor, mid, hand);
                let root_motion = (layer.base.joints[root] - previous.base.joints[root]).length();
                let authored = (layer.base.joints[elbow] - previous.base.joints[elbow])
                    .length()
                    .max((layer.base.joints[hand] - previous.base.joints[hand]).length());
                let limit = (6.0 * a.scale * dt).max(0.003 * a.scale) + root_motion + authored;
                let owned = self.route_owned(actor, chain, mid);
                let simple = !layer.active && !previous.active && layer.nodes.len() == 1 && previous.nodes.len() == 1;
                for node in &mut layer.nodes {
                    if simple {
                        node.total = previous.nodes[0].total;
                        node.previous = 0;
                        continue;
                    }
                    let mut edges: Vec<_> = previous
                        .nodes
                        .iter()
                        .enumerate()
                        .filter_map(|(i, old)| {
                            if !old.total.is_finite() {
                                return None;
                            }
                            let ed = node.elbow - old.elbow;
                            let hd = node.wrist - old.wrist;
                            if ed.length().max(hd.length()) > limit {
                                return None;
                            }
                            Some((old.total + node.cost * dt + 0.08 * (ed.dot(ed) + hd.dot(hd)) / dt.max(1e-6), i))
                        })
                        .collect();
                    edges.sort_by(|a, b| a.0.total_cmp(&b.0));
                    for (total, i) in edges {
                        let old = &previous.nodes[i];
                        let mut candidate = mid_base.clone();
                        if layer.active || previous.active {
                            let pole = (mid_base.rots[parent] * old.pole.lerp(node.pole, 0.5)).norm();
                            crate::contact::place(&mut candidate.joints, &mid_base, chain, old.wrist.lerp(node.wrist, 0.5), Some(pole));
                        }
                        if self.route_free(actor, chain, mid, &candidate, owned) < -0.0001 {
                            continue;
                        }
                        node.total = total;
                        node.previous = i;
                        break;
                    }
                }
                if !layer.nodes.iter().any(|n| n.total.is_finite()) {
                    return Err(format!("no continuous prop-action arm route for actor {actor}, hand {hand} at {w:.6}s"));
                }
            } else {
                for node in &mut layer.nodes {
                    node.total = node.cost;
                }
            }
            layers.push(layer);
        }
        let mut index = layers.last().unwrap().nodes.iter().enumerate().min_by(|(_, a), (_, b)| a.total.total_cmp(&b.total)).unwrap().0;
        let mut points = Vec::new();
        for layer in layers.into_iter().rev() {
            let node = &layer.nodes[index];
            points.push(ArmPoint { at: layer.at, wrist: node.wrist, pole: node.pole, active: layer.active });
            index = node.previous;
        }
        points.reverse();
        Ok(ArmGuide { chain, points })
    }
    fn route_key(&self) -> Fingerprint {
        let mut h = StableHasher::new();
        h.f64(self.length).f64(self.style.line).add(&self.ground.is_some()).add(&format!("{:?}", self.actions));
        for a in &self.actors {
            h.f64(a.scale).f64(a.squash).f64(a.body_reach).add(&a.pose_lock).add(&a.model.root);
            h.add(&format!(
                "{:?}{:?}{:?}{:?}",
                a.model.rig.joints(),
                a.model.rig.foot(mavian_rig::Side::L),
                a.model.rig.foot(mavian_rig::Side::R),
                a.model.parts
            ));
            for (name, pose) in a.model.poses() {
                h.add(name).add(&format!("{pose:?}"));
            }
            h.add(&a.model.rest)
                .add(&format!("{:?}{:?}{:?}{:?}{:?}{:?}{:?}", a.poses, a.path, a.facing, a.patches, a.layers, a.auto_layers, a.recoils));
            h.add(&format!("{:?}{:?}{:?}{:?}{:?}", a.contacts, a.auto_contacts, a.reaches, a.avoids, a.approach_steps));
        }
        for p in &self.props {
            h.add(&p.name).add(&p.at).f64(p.turn).add(&format!("{:?}{:?}", p.shape, p.motion));
        }
        h.finish()
    }
    fn difficult_windows(&self, actor: usize, chain: [usize; 3]) -> Vec<(f64, f64)> {
        let mut windows = Vec::new();
        let samples = (self.length * 240.0).ceil().max(1.0) as usize;
        let mut previous: Option<(f64, std::sync::Arc<Figure>)> = None;
        for n in 0..=samples {
            let w = self.length * n as f64 / samples as f64;
            let fig = self.posed(actor, w).0;
            let owned = self.route_owned(actor, chain, w);
            let mut bad = self.route_free(actor, chain, w, &fig, owned) < -0.001;
            if let Some((old_w, ref old)) = previous {
                let root = fig.joints[chain[0]] - old.joints[chain[0]];
                let ed = fig.joints[chain[1]] - old.joints[chain[1]] - root;
                let hd = fig.joints[chain[2]] - old.joints[chain[2]] - root;
                bad |= owned && ed.length().max(hd.length()) > 0.035 * self.actors[actor].scale;
                let mid = (w + old_w) * 0.5;
                let actual = self.posed(actor, mid).0;
                let mut interpolated = (*actual).clone();
                for j in chain {
                    interpolated.joints[j] = old.joints[j].lerp(fig.joints[j], 0.5);
                }
                let radii = chain.map(|j| joint_radius(&self.actors[actor], j, self.style.line));
                let props = self.arm_obstacles(actor, chain[2], mid, &actual);
                bad |= clearance(&props, &interpolated.joints, chain, radii) < -0.001;
            }
            if bad {
                let mut matched = false;
                for c in &self.actors[actor].auto_contacts {
                    if c.chain == chain && w >= c.from - 0.01 && w <= c.to + 0.01 {
                        windows.push(((c.from - 0.05).max(0.0), (c.to + 0.05).min(self.length)));
                        matched = true;
                    }
                }
                if !matched {
                    windows.push(((w - 0.2).max(0.0), (w + 0.2).min(self.length)));
                }
            }
            previous = Some((w, fig));
        }
        windows.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut merged: Vec<(f64, f64)> = Vec::new();
        for (from, to) in windows {
            if let Some(last) = merged.last_mut()
                && from <= last.1
            {
                last.1 = last.1.max(to);
            } else {
                merged.push((from, to));
            }
        }
        merged
    }
    pub(crate) fn plan_arm_clearance(&mut self) -> Result<(), String> {
        let actors: Vec<_> = self
            .actors
            .iter()
            .enumerate()
            .filter_map(|(i, a)| (!a.pose_lock && self.grab_reports.iter().any(|r| r.actor == i && r.attached())).then_some(i))
            .collect();
        if actors.is_empty() {
            return Ok(());
        }
        let key = self.route_key();
        let cache = CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
        if let Some(routes) = cache.lock().unwrap().get(&key).cloned() {
            for (actor, guides) in routes {
                self.actors[actor].clearance_guides = guides;
            }
            return Ok(());
        }
        let samples = (self.length * 120.0).ceil().max(1.0) as usize;
        let mut routes = Vec::new();
        for actor in actors {
            let mut guides = Vec::new();
            for name in ["hand_l", "hand_r"] {
                let Ok(hand) = self.actors[actor].model.rig.joint(name) else { continue };
                let Ok(chain) = crate::actions::chain(&self.actors[actor], hand) else { continue };
                let windows = self.difficult_windows(actor, chain);
                if windows.is_empty() {
                    continue;
                }
                let mut times: Vec<_> = (0..=samples).map(|n| self.length * n as f64 / samples as f64).collect();
                for c in &self.actors[actor].auto_contacts {
                    times.extend([c.from, c.to]);
                    if let Some((at, _)) = c.withdrawal {
                        times.push(at);
                    }
                }
                times.extend(windows.iter().flat_map(|&(a, b)| [a, b]));
                times.retain(|w| *w >= 0.0 && *w <= self.length);
                times.sort_by(f64::total_cmp);
                times.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
                let guide = self.arm_route(actor, chain, &times, &windows, false).or_else(|_| {
                    // A held object may need to remain above a support beyond
                    // the contact fade. Retry without forcing an early join
                    // to the local rest pose, and retain wider wrist detours.
                    let extended = [(windows[0].0, self.length)];
                    self.arm_route(actor, chain, &times, &extended, true)
                })?;
                guides.push(guide);
            }
            routes.push((actor, guides));
        }
        {
            let mut cache = cache.lock().unwrap();
            if cache.len() >= 16 {
                cache.pop_first();
            }
            cache.insert(key, routes.clone());
        }
        for (actor, guides) in routes {
            self.actors[actor].clearance_guides = guides;
        }
        Ok(())
    }
}
