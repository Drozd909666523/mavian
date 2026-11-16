//! The Python description layer: the `mavian` module.
//!
//! A shot is described in Python, the way it was in the MVP, and the module
//! turns the description into the same core structures the Rust shots build.
//! Python only builds the description; frames are computed by the core, so
//! the speed of Python does not matter (05-architecture, §4).
//!
//! The module is embedded in the `mavian` binary: `mavian render shots/x.py`.

use mavian_core::{ClockKind, Ease, Keys, Track, Vec3};
use mavian_models::stickman::stickman;
use mavian_rig::{Model, Side};
use mavian_scene::{
    Action, Actor, Between, CameraRig, Follow, Ground, Label, Onto, Orbit, Patch, Point, PoseKey, Poses, Prop, Shake, Shape, Shot, Style, Watch,
    Wire, pk,
};
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCodeMethods, PyDict, PyList, PyTuple};
use std::ffi::CString;
use std::path::Path;
use std::sync::{Arc, Once, OnceLock};

// --- keys --------------------------------------------------------------------

const EASES: [(&str, Ease); 5] = [("linear", Ease::Linear), ("in", Ease::In), ("out", Ease::Out), ("inout", Ease::InOut), ("hold", Ease::Hold)];

fn ease_of(name: &str) -> PyResult<Ease> {
    EASES.iter().find(|(n, _)| *n == name).map(|(_, e)| *e).ok_or_else(|| PyValueError::new_err(format!("unknown ease: {name}")))
}

fn ease_name(e: Ease) -> &'static str {
    EASES.iter().find(|(_, x)| *x == e).map(|(n, _)| *n).unwrap_or("inout")
}

fn clock_of(name: &str) -> PyResult<ClockKind> {
    match name {
        "world" => Ok(ClockKind::World),
        "shot" => Ok(ClockKind::Shot),
        "progress" => Ok(ClockKind::Progress),
        _ => Err(PyValueError::new_err(format!("unknown clock: {name}"))),
    }
}

/// `(value, "ease")` → value and its ease; anything else keeps the default.
fn split_eased<'py>(v: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Option<Ease>)> {
    if let Ok(t) = v.cast::<PyTuple>()
        && t.len() == 2
        && let Ok(name) = t.get_item(1)?.extract::<String>()
        && let Ok(e) = ease_of(&name)
    {
        return Ok((t.get_item(0)?, Some(e)));
    }
    Ok((v.clone(), None))
}

/// A key track: `keys({0.0: 2, 8.6: (1, "hold")}, ease="inout", clock="world")`.
///
/// Values stay Python objects until the track is used: the same `keys` can
/// hold numbers, points or poses depending on where it is passed.
#[pyclass(name = "Keys", frozen)]
pub struct PyKeys {
    items: Vec<(f64, Py<PyAny>, Option<Ease>)>,
    ease: Ease,
    clock: ClockKind,
}

impl PyKeys {
    fn convert<T: Clone>(&self, py: Python<'_>, f: impl Fn(&Bound<'_, PyAny>) -> PyResult<T>) -> PyResult<Keys<T>> {
        if self.items.is_empty() {
            return Err(PyValueError::new_err("empty key track"));
        }
        let mut k = Keys::new(self.ease).clock(self.clock);
        for (t, v, e) in &self.items {
            k = k.eased(*t, f(v.bind(py))?, e.unwrap_or(self.ease));
        }
        Ok(k)
    }
}

#[pyfunction]
#[pyo3(signature = (table, ease = "inout", clock = "world"))]
fn keys(table: &Bound<'_, PyDict>, ease: &str, clock: &str) -> PyResult<PyKeys> {
    let mut items = Vec::new();
    for (t, v) in table.iter() {
        let (v, e) = split_eased(&v)?;
        items.push((t.extract::<f64>()?, v.unbind(), e));
    }
    Ok(PyKeys { items, ease: ease_of(ease)?, clock: clock_of(clock)? })
}

fn vec3(v: &Bound<'_, PyAny>) -> PyResult<Vec3> {
    let (x, y, z): (f64, f64, f64) = v.extract()?;
    Ok(Vec3 { x, y, z })
}

fn track_f64(v: &Bound<'_, PyAny>) -> PyResult<Track<f64>> {
    match v.cast::<PyKeys>() {
        Ok(k) => Ok(k.get().convert(v.py(), |x| x.extract::<f64>())?.into()),
        Err(_) => Ok(Track::Const(v.extract::<f64>()?)),
    }
}

fn track_vec3(v: &Bound<'_, PyAny>) -> PyResult<Track<Vec3>> {
    match v.cast::<PyKeys>() {
        Ok(k) => Ok(k.get().convert(v.py(), vec3)?.into()),
        Err(_) => Ok(Track::Const(vec3(v)?)),
    }
}

fn track_xz(v: &Bound<'_, PyAny>) -> PyResult<Track<(f64, f64)>> {
    match v.cast::<PyKeys>() {
        Ok(k) => Ok(k.get().convert(v.py(), |x| x.extract::<(f64, f64)>())?.into()),
        Err(_) => Ok(Track::Const(v.extract::<(f64, f64)>()?)),
    }
}

// --- poses -------------------------------------------------------------------

/// A pose key with planting or movement (made by `key`).
#[pyclass(name = "PoseKey", frozen)]
#[derive(Clone)]
pub struct PyPoseKey {
    key: PoseKey,
}

#[pymethods]
impl PyPoseKey {
    fn __repr__(&self) -> String {
        format!("PoseKey({:?}, plant={:?}, move={:?}, toward={:?})", self.key.name, self.key.plant, self.key.mv, self.key.toward)
    }
}

fn side_of(s: &str) -> PyResult<Side> {
    match s {
        "l" => Ok(Side::L),
        "r" => Ok(Side::R),
        _ => Err(PyValueError::new_err(format!("foot must be \"l\" or \"r\", not {s:?}"))),
    }
}

fn pose_key(v: &Bound<'_, PyAny>) -> PyResult<PoseKey> {
    if let Ok(k) = v.cast::<PyPoseKey>() {
        return Ok(k.get().key.clone());
    }
    match v.extract::<String>() {
        Ok(name) => Ok(pk(&name)),
        Err(_) => Err(PyTypeError::new_err("a pose key is a pose name or key(...)")),
    }
}

/// A pose key for a track: `key("pass_l", "linear", plant="l")`, `key("air", move=1.5)`,
/// `key("air", fly=True)` — in the air: the pelvis flies with its take-off speed.
#[pyfunction]
#[pyo3(signature = (name, ease = "inout", plant = None, r#move = None, fly = false))]
fn key<'py>(
    py: Python<'py>,
    name: &str,
    ease: &str,
    plant: Option<&str>,
    r#move: Option<&Bound<'py, PyAny>>,
    fly: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    if fly && (plant.is_some() || r#move.is_some()) {
        return Err(PyValueError::new_err("key: fly=True is neither planted nor moved"));
    }
    let mut k = pk(name);
    if fly {
        k = k.fly();
    }
    if let Some(p) = plant {
        k = k.plant(side_of(p)?);
    }
    if let Some(m) = r#move {
        k = match m.extract::<f64>() {
            Ok(d) => k.forward(d),
            Err(_) => {
                let (dx, dz): (f64, f64) = m.extract()?;
                k.shift(dx, dz)
            }
        };
    }
    ease_of(ease)?;
    PyTuple::new(py, [Py::new(py, PyPoseKey { key: k })?.into_any(), ease.into_pyobject(py)?.into_any().unbind()])
}

/// Walk keys: `keys({**walk(0.0, steps=11, stop="stand"), 6.4: "peer"})`.
#[pyfunction]
#[pyo3(signature = (start, steps, step = 0.5, lead = "l", stop = None, stop_time = 0.35, prefix = ""))]
#[allow(clippy::too_many_arguments)]
fn walk<'py>(
    py: Python<'py>,
    start: f64,
    steps: usize,
    step: f64,
    lead: &str,
    stop: Option<&str>,
    stop_time: f64,
    prefix: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (t, k, e) in mavian_scene::walk(start, steps, step, side_of(lead)?, stop, stop_time, prefix) {
        let v = PyTuple::new(py, [Py::new(py, PyPoseKey { key: k })?.into_any(), ease_name(e).into_pyobject(py)?.into_any().unbind()])?;
        d.set_item(t, v)?;
    }
    Ok(d)
}

/// The stick figure as an actor, as the MVP's `Stickman(...)`:
/// `Stickman(poses="stand", path=(x, 0.0))` is `Actor(STICKMAN, ...)`.
#[pyfunction]
#[pyo3(name = "Stickman", signature = (**kw))]
fn stickman_actor<'py>(py: Python<'py>, kw: Option<&Bound<'py, PyDict>>) -> PyResult<Bound<'py, PyAny>> {
    let m = py.import("mavian")?;
    m.getattr("Actor")?.call((m.getattr("STICKMAN")?,), kw)
}

/// A jump planned from intent (see `jump`): `**j.keys` goes into a pose
/// track; the moments place cameras, slow motion and shakes.
#[pyclass(name = "Jump", frozen)]
pub struct PyJump {
    jump: mavian_scene::Jump,
}

#[pymethods]
impl PyJump {
    /// The jump's pose keys, for `keys({..., **j.keys})`.
    #[getter]
    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (t, k, e) in &self.jump.keys {
            let v = PyTuple::new(py, [Py::new(py, PyPoseKey { key: k.clone() })?.into_any(), ease_name(*e).into_pyobject(py)?.into_any().unbind()])?;
            d.set_item(*t, v)?;
        }
        Ok(d)
    }
    /// the push starts
    #[getter]
    fn push(&self) -> f64 {
        self.jump.push
    }
    /// take-off
    #[getter]
    fn launch(&self) -> f64 {
        self.jump.launch
    }
    /// the top of the flight
    #[getter]
    fn apex(&self) -> f64 {
        self.jump.apex
    }
    /// touching the ground
    #[getter]
    fn landing(&self) -> f64 {
        self.jump.landing
    }
    /// the landing crouch is reached
    #[getter]
    fn end(&self) -> f64 {
        self.jump.end
    }
    /// take-off speed: (forward, up), m/s
    #[getter]
    fn speed(&self) -> (f64, f64) {
        self.jump.speed
    }
    /// how high the pelvis rises, m
    #[getter]
    fn height(&self) -> f64 {
        self.jump.height
    }
    fn __repr__(&self) -> String {
        let j = &self.jump;
        format!(
            "Jump(push={:.3}, launch={:.3}, apex={:.3}, landing={:.3}, end={:.3}, speed=({:.2}, {:.2}) m/s, takeoff at {:.0}% of the push)",
            j.push,
            j.launch,
            j.apex,
            j.landing,
            j.end,
            j.speed.0,
            j.speed.1,
            100.0 * j.takeoff
        )
    }
}

/// A jump from intent: take off at `at`, land `distance` further on (foot to
/// foot), the pelvis rising `height`. The push, flight and landing timings
/// follow from the poses and gravity.
#[pyfunction]
#[pyo3(signature = (model, at, distance, height, plant = "l", scale = 1.0, crouch = "crouch", launch = "leap_launch", air = "leap_air", reach = "leap_reach", land = "land"))]
#[allow(clippy::too_many_arguments)]
fn jump(
    model: &Bound<'_, PyModel>,
    at: f64,
    distance: f64,
    height: f64,
    plant: &str,
    scale: f64,
    crouch: &str,
    launch: &str,
    air: &str,
    reach: &str,
    land: &str,
) -> PyResult<PyJump> {
    let poses = mavian_scene::JumpPoses { crouch: crouch.into(), launch: launch.into(), air: air.into(), reach: reach.into(), land: land.into() };
    let j = mavian_scene::jump(&model.get().model, scale, at, distance, height, side_of(plant)?, &poses).map_err(PyValueError::new_err)?;
    Ok(PyJump { jump: j })
}

/// When a walk started with the same parameters ends.
#[pyfunction]
#[pyo3(signature = (start, steps, step = 0.5, stop_time = 0.0))]
fn walk_end(start: f64, steps: usize, step: f64, stop_time: f64) -> f64 {
    mavian_scene::walk_end(start, steps, step, stop_time)
}

// --- models and actors -------------------------------------------------------

/// A character model (skeleton, parts, poses).
#[pyclass(name = "Model", frozen)]
pub struct PyModel {
    model: Arc<Model>,
}

/// A pose of a model's skeleton (made by `Model.pose`).
#[pyclass(name = "Pose", frozen)]
#[derive(Clone)]
pub struct PyPose {
    pose: mavian_rig::Pose,
}

fn angles_of(v: &Bound<'_, PyAny>) -> PyResult<Vec3> {
    if let Ok(x) = v.extract::<f64>() {
        return Ok(mavian_core::v3(x, 0.0, 0.0));
    }
    if let Ok((x, y)) = v.extract::<(f64, f64)>() {
        return Ok(mavian_core::v3(x, y, 0.0));
    }
    let (x, y, z): (f64, f64, f64) = v.extract().map_err(|_| PyTypeError::new_err("angles: a number, (pitch, roll) or (pitch, roll, twist)"))?;
    Ok(mavian_core::v3(x, y, z))
}

#[pymethods]
impl PyModel {
    #[getter]
    fn name(&self) -> &str {
        &self.model.name
    }

    /// A pose: like `base` (a pose name or Pose), with these joints set:
    /// `MANNEQUIN.pose("stand", arm_r=(40, 6), elbow_r=80)`; `arm=` sets both sides.
    #[pyo3(signature = (base = None, lift = None, **angles))]
    fn pose(&self, base: Option<&Bound<'_, PyAny>>, lift: Option<f64>, angles: Option<&Bound<'_, PyDict>>) -> PyResult<PyPose> {
        let owned;
        let base_pose = match base {
            None => None,
            Some(b) => match b.cast::<PyPose>() {
                Ok(p) => {
                    owned = p.get().pose.clone();
                    Some(&owned)
                }
                Err(_) => Some(self.model.pose(&b.extract::<String>()?).map_err(|e| PyKeyError::new_err(e.0))?),
            },
        };
        let mut pb = self.model.rig.pose();
        if let Some(b) = base_pose {
            pb = pb.base(b);
        }
        if let Some(l) = lift {
            pb = pb.lift(l);
        }
        if let Some(d) = angles {
            for (k, v) in d.iter() {
                pb = pb.set(&k.extract::<String>()?, angles_of(&v)?);
            }
        }
        Ok(PyPose { pose: pb.build().map_err(|e| PyKeyError::new_err(e.0))? })
    }

    /// The pose with left and right swapped.
    fn mirror(&self, pose: &Bound<'_, PyPose>) -> PyPose {
        PyPose { pose: self.model.rig.mirror(&pose.get().pose) }
    }

    /// The same model with more poses: `MANNEQUIN.with_poses({"carry": ...})`.
    fn with_poses(&self, poses: &Bound<'_, PyDict>) -> PyResult<PyModel> {
        let mut m = (*self.model).clone();
        for (k, v) in poses.iter() {
            m.add_pose(&k.extract::<String>()?, v.cast::<PyPose>()?.get().pose.clone()).map_err(|e| PyValueError::new_err(e.0))?;
        }
        Ok(PyModel { model: Arc::new(m) })
    }

    fn __repr__(&self) -> String {
        format!("Model({:?})", self.model.name)
    }
}

/// A character in a shot. Parameters as in the MVP's `Actor`.
#[pyclass(name = "Actor", frozen)]
pub struct PyActor {
    model: Arc<Model>,
    poses: Poses,
    path: Track<(f64, f64)>,
    facing: Track<f64>,
    squash: f64,
    body_reach: f64,
    pose_lock: bool,
    blink: Track<f64>,
    face: bool,
    face_trust: f64,
    scale: f64,
    perspective: f64,
    trails: f64,
    color: Option<String>,
    core: OnceLock<Actor>,
}

impl PyActor {
    fn build(&self) -> Actor {
        let mut a = Actor::new(Arc::clone(&self.model))
            .poses(self.poses.clone())
            .path(self.path.clone())
            .facing(self.facing.clone())
            .squash(self.squash)
            .blink(self.blink.clone())
            .face(self.face)
            .face_trust(self.face_trust)
            .scale(self.scale)
            .perspective(self.perspective)
            .trails(self.trails);
        a.color = self.color.clone();
        a.body_reach = self.body_reach;
        a.pose_lock = self.pose_lock;
        a
    }

    /// The core actor for questions asked while describing (`where`).
    fn core(&self) -> PyResult<&Actor> {
        if self.core.get().is_none() {
            let a = self.build();
            a.validate().map_err(|e| PyValueError::new_err(e.0))?;
            let _ = self.core.set(a);
        }
        Ok(self.core.get().expect("set above"))
    }

    fn where_at(&self, t: f64, joint: Option<&str>) -> PyResult<(f64, f64, f64)> {
        let a = self.core()?;
        if let Some(j) = joint {
            a.model.rig.joint(j).map_err(|e| PyKeyError::new_err(e.0))?;
        }
        let p = a.where_(t, joint);
        Ok((p.x, p.y, p.z))
    }
}

#[pymethods]
impl PyActor {
    #[new]
    #[pyo3(signature = (model, poses = None, path = None, facing = None, squash = 0.0, blink = None, face = true, face_trust = 0.0, scale = 1.0, color = None, perspective = 1.0, trails = 0.0, body_reach = 0.0, pose_lock = false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: &Bound<'_, PyModel>,
        poses: Option<&Bound<'_, PyAny>>,
        path: Option<&Bound<'_, PyAny>>,
        facing: Option<&Bound<'_, PyAny>>,
        squash: f64,
        blink: Option<&Bound<'_, PyAny>>,
        face: bool,
        face_trust: f64,
        scale: f64,
        color: Option<String>,
        perspective: f64,
        trails: f64,
        body_reach: f64,
        pose_lock: bool,
    ) -> PyResult<Self> {
        if perspective <= 0.0 || trails < 0.0 {
            return Err(PyValueError::new_err("perspective must be positive and trails not negative"));
        }
        let poses = match poses {
            None => Poses::Rest,
            Some(p) => match p.cast::<PyKeys>() {
                Ok(k) => Poses::Keys(k.get().convert(p.py(), pose_key)?),
                Err(_) => Poses::Fixed(p.extract::<String>()?),
            },
        };
        if !(0.0..=1.0).contains(&face_trust) {
            return Err(PyValueError::new_err(format!("face_trust must be in [0, 1], got {face_trust}")));
        }
        let actor = PyActor {
            model: Arc::clone(&model.get().model),
            poses,
            path: path.map(track_xz).transpose()?.unwrap_or(Track::Const((0.0, 0.0))),
            facing: facing.map(track_f64).transpose()?.unwrap_or(Track::Const(0.0)),
            squash,
            body_reach,
            pose_lock,
            blink: blink.map(track_f64).transpose()?.unwrap_or(Track::Const(0.0)),
            face,
            face_trust,
            scale,
            perspective,
            trails,
            color: color.filter(|c| !c.is_empty()),
            core: OnceLock::new(),
        };
        // check here, so the error points at the line that made the actor
        actor.core()?;
        Ok(actor)
    }

    /// Where a joint is at world time t; without a name — the model's root joint.
    #[pyo3(name = "where", signature = (t, joint = None))]
    fn where_py(&self, t: f64, joint: Option<&str>) -> PyResult<(f64, f64, f64)> {
        self.where_at(t, joint)
    }
}

/// Where an actor's joint is at world time t — to place cameras from the action.
#[pyfunction]
#[pyo3(name = "where", signature = (actor, t, joint = None))]
fn where_fn(actor: &Bound<'_, PyActor>, t: f64, joint: Option<&str>) -> PyResult<(f64, f64, f64)> {
    actor.get().where_at(t, joint)
}

// --- cameras -----------------------------------------------------------------

/// A point that follows an actor's joint.
#[pyclass(name = "Follow", frozen)]
pub struct PyFollow {
    follow: Follow,
}

#[pymethods]
impl PyFollow {
    #[new]
    #[pyo3(signature = (actor = 0, joint = "pelvis", offset = (0.0, 0.0, 0.0), lag = 0.0, smooth = 0.0))]
    fn new(actor: usize, joint: &str, offset: (f64, f64, f64), lag: f64, smooth: f64) -> Self {
        PyFollow { follow: Follow::new(actor, joint).offset(offset).lag(lag).smooth(smooth) }
    }
}

/// A point between joints of several actors (the middle of a fight).
#[pyclass(name = "Between", frozen)]
pub struct PyBetween {
    between: Between,
}

#[pymethods]
impl PyBetween {
    #[new]
    #[pyo3(signature = (actors = vec![0, 1], joint = "pelvis", offset = None, smooth = 0.0))]
    fn new(actors: Vec<usize>, joint: &str, offset: Option<&Bound<'_, PyAny>>, smooth: f64) -> PyResult<Self> {
        if actors.is_empty() {
            return Err(PyValueError::new_err("Between needs at least one actor"));
        }
        let offset = offset.map(track_vec3).transpose()?.unwrap_or(Track::Const(Vec3::ZERO));
        Ok(PyBetween { between: Between { actors, joint: joint.into(), offset, smooth } })
    }
}

fn point(v: &Bound<'_, PyAny>) -> PyResult<Point> {
    if let Ok(f) = v.cast::<PyFollow>() {
        return Ok(Point::Follow(f.get().follow.clone()));
    }
    if let Ok(b) = v.cast::<PyBetween>() {
        return Ok(Point::Between(b.get().between.clone()));
    }
    Ok(Point::Fixed(track_vec3(v)?))
}

/// Any camera: `Watch`, `Orbit` or `Cuts`.
#[pyclass(name = "Camera", frozen, subclass)]
pub struct PyCamera {
    rig: CameraRig,
}

/// Camera at `pos` looking at `target`.
#[pyfunction]
#[pyo3(name = "Watch", signature = (pos = None, target = None, fov = None, fisheye = false, turn = None, roll = None))]
fn watch(
    pos: Option<&Bound<'_, PyAny>>,
    target: Option<&Bound<'_, PyAny>>,
    fov: Option<&Bound<'_, PyAny>>,
    fisheye: bool,
    turn: Option<&Bound<'_, PyAny>>,
    roll: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyCamera> {
    let mut w = Watch::new(
        pos.map(point).transpose()?.unwrap_or_else(|| (0.0, 1.5, 6.0).into()),
        target.map(point).transpose()?.unwrap_or_else(|| (0.0, 1.1, 0.0).into()),
    );
    if let Some(f) = fov {
        w.fov = track_f64(f)?;
    }
    w.fisheye = fisheye;
    w.turn = turn.map(track_f64).transpose()?;
    if let Some(r) = roll {
        w.roll = track_f64(r)?;
    }
    Ok(PyCamera { rig: w.into() })
}

/// Camera on a circle around the target.
#[pyfunction]
#[pyo3(name = "Orbit", signature = (radius = None, height = None, angle = None, target = None, fov = None, fisheye = false))]
fn orbit(
    radius: Option<&Bound<'_, PyAny>>,
    height: Option<&Bound<'_, PyAny>>,
    angle: Option<&Bound<'_, PyAny>>,
    target: Option<&Bound<'_, PyAny>>,
    fov: Option<&Bound<'_, PyAny>>,
    fisheye: bool,
) -> PyResult<PyCamera> {
    let mut o = Orbit::default();
    if let Some(v) = radius {
        o.radius = track_f64(v)?;
    }
    if let Some(v) = height {
        o.height = track_f64(v)?;
    }
    if let Some(v) = angle {
        o.angle = track_f64(v)?;
    }
    if let Some(v) = target {
        o.target = point(v)?;
    }
    if let Some(v) = fov {
        o.fov = track_f64(v)?;
    }
    o.fisheye = fisheye;
    Ok(PyCamera { rig: o.into() })
}

/// Cuts: `{world time: camera}`. One action, cameras take turns.
#[pyfunction]
#[pyo3(name = "Cuts")]
fn cuts(table: &Bound<'_, PyDict>) -> PyResult<PyCamera> {
    if table.is_empty() {
        return Err(PyValueError::new_err("no cameras"));
    }
    let mut items = Vec::new();
    for (t, c) in table.iter() {
        let c = c.cast::<PyCamera>().map_err(|_| PyTypeError::new_err("Cuts takes cameras"))?;
        items.push((t.extract::<f64>()?, c.get().rig.clone()));
    }
    Ok(PyCamera { rig: CameraRig::cuts(items) })
}

/// Camera shake from a hit.
#[pyclass(name = "Shake", frozen)]
pub struct PyShake {
    shake: Shake,
}

#[pymethods]
impl PyShake {
    #[new]
    #[pyo3(signature = (at, strength = 0.1, decay = 0.3, freq = 16.0))]
    fn new(at: f64, strength: f64, decay: f64, freq: f64) -> Self {
        PyShake { shake: Shake { at, strength, decay, freq } }
    }
}

// --- shot --------------------------------------------------------------------

/// Ground: grid and shadows.
#[pyclass(name = "Ground", frozen)]
pub struct PyGround {
    ground: Ground,
}

#[pymethods]
impl PyGround {
    #[new]
    #[pyo3(signature = (size = 4.0, step = 1.0, width = 1.4, draw_on = 3, shadow = true, center = (0.0, 0.0), boil = 0.5, grid = true))]
    #[allow(clippy::too_many_arguments)]
    fn new(size: f64, step: f64, width: f64, draw_on: u32, shadow: bool, center: (f64, f64), boil: f64, grid: bool) -> Self {
        PyGround { ground: Ground { size, step, width, draw_on, shadow, center, boil, grid } }
    }
}

/// Drawing style of a shot.
#[pyclass(name = "Style", frozen)]
pub struct PyStyle {
    style: Style,
}

#[pymethods]
impl PyStyle {
    #[new]
    #[pyo3(signature = (paper = None, ink = None, ground = None, line = None, depth_gamma = None, zoom = None, halo = None, boil = None, boil_scale = None, pressure = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        paper: Option<String>,
        ink: Option<String>,
        ground: Option<String>,
        line: Option<f64>,
        depth_gamma: Option<f64>,
        zoom: Option<f64>,
        halo: Option<f64>,
        boil: Option<f64>,
        boil_scale: Option<f64>,
        pressure: Option<f64>,
    ) -> Self {
        let d = Style::default();
        PyStyle {
            style: Style {
                paper: paper.unwrap_or(d.paper),
                ink: ink.unwrap_or(d.ink),
                ground: ground.unwrap_or(d.ground),
                line: line.unwrap_or(d.line),
                depth_gamma: depth_gamma.unwrap_or(d.depth_gamma),
                zoom: zoom.unwrap_or(d.zoom),
                halo: halo.unwrap_or(d.halo),
                boil: boil.unwrap_or(d.boil),
                boil_scale: boil_scale.unwrap_or(d.boil_scale),
                pressure: pressure.unwrap_or(d.pressure),
            },
        }
    }
}

/// A shot. Parameters as in the MVP's `Shot`.
#[pyclass(name = "Shot")]
pub struct PyShot {
    camera: CameraRig,
    actors: Vec<Py<PyActor>>,
    fps: u32,
    length: f64,
    seed: u64,
    size: (u32, u32),
    draw_on: Track<f64>,
    time_warp: Track<f64>,
    shakes: Vec<Shake>,
    ground: Option<Ground>,
    style: Style,
    labels: Vec<Label>,
    patches: Vec<PyPatchData>,
    props: Vec<Py<PyProp>>,
    wires: Vec<Wire>,
    fires: Vec<mavian_scene::Fire>,
    actions: Vec<Py<PyAction>>,
    contacts: Vec<Py<PyContact>>,
}

/// A patch as described: the joint by name, resolved against the actor's model.
#[derive(Clone)]
struct PyPatchData {
    actor: usize,
    from: f64,
    to: f64,
    joint: String,
    add: (f64, f64, f64),
    ramp: f64,
}

/// A timed end-effector contact; targets are world points or prop-local points.
#[pyclass(name = "Contact", frozen)]
pub struct PyContact {
    actor: Py<PyActor>,
    joint: String,
    target: Option<Vec3>,
    prop: Option<Py<PyProp>>,
    local: Vec3,
    at: (f64, f64),
    ramp: f64,
    priority: mavian_scene::Priority,
    name: String,
    tolerance: f64,
    pole: Option<Vec3>,
    approach: Option<Vec3>,
}
#[pymethods]
impl PyContact {
    #[new]
    #[pyo3(signature = (actor, joint, at, target = None, local = (0.0,0.0,0.0), ramp = 0.1, priority = "contact", name = String::new(), tolerance = 0.002, pole = None, approach = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        actor: Py<PyActor>,
        joint: String,
        at: (f64, f64),
        target: Option<&Bound<'_, PyAny>>,
        local: (f64, f64, f64),
        ramp: f64,
        priority: &str,
        name: String,
        tolerance: f64,
        pole: Option<(f64, f64, f64)>,
        approach: Option<(f64, f64, f64)>,
    ) -> PyResult<Self> {
        if !at.0.is_finite() || !at.1.is_finite() || at.0 >= at.1 || !ramp.is_finite() || ramp < 0.0 || !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(PyValueError::new_err("Contact: increasing finite interval, nonnegative ramp and positive tolerance required"));
        }
        let priority = match priority {
            "support" => mavian_scene::Priority::Support,
            "contact" => mavian_scene::Priority::Contact,
            "pose" => mavian_scene::Priority::Pose,
            _ => return Err(PyValueError::new_err("Contact priority: support, contact or pose")),
        };
        let (target, prop) = match target {
            None => (None, None),
            Some(t) => {
                if let Ok(p) = t.cast::<PyProp>() {
                    (None, Some(p.clone().unbind()))
                } else {
                    let point: (f64, f64, f64) =
                        t.extract().map_err(|_| PyTypeError::new_err("Contact target: a world point, Prop, or None to pin the joint"))?;
                    (Some(point.into()), None)
                }
            }
        };
        let local: Vec3 = local.into();
        let pole = pole.map(Vec3::from);
        let approach = approach.map(Vec3::from);
        if !mavian_scene::contact::finite(local)
            || target.is_some_and(|p| !mavian_scene::contact::finite(p))
            || pole.is_some_and(|p| !mavian_scene::contact::finite(p))
            || approach.is_some_and(|p| !mavian_scene::contact::finite(p))
        {
            return Err(PyValueError::new_err("Contact points must be finite"));
        }
        Ok(Self { actor, joint, target, prop, local, at, ramp, priority, name, tolerance, pole, approach })
    }
}

// --- props and actions ---------------------------------------------------------

/// A box: `Box(0.72, 0.9, 0.72)` — width (x), height (y), depth (z).
#[pyclass(name = "Box", frozen)]
pub struct PyBox {
    shape: Shape,
}

#[pymethods]
impl PyBox {
    #[new]
    fn new(sx: f64, sy: f64, sz: f64) -> PyResult<Self> {
        if sx <= 0.0 || sy <= 0.0 || sz <= 0.0 {
            return Err(PyValueError::new_err("Box: sizes must be positive"));
        }
        Ok(PyBox { shape: Shape::Box { sx, sy, sz } })
    }
}

/// A ball: `Sphere(0.1)`.
#[pyclass(name = "Sphere", frozen)]
pub struct PySphere {
    shape: Shape,
}

#[pymethods]
impl PySphere {
    #[new]
    fn new(r: f64) -> PyResult<Self> {
        if r <= 0.0 {
            return Err(PyValueError::new_err("Sphere: the radius must be positive"));
        }
        Ok(PySphere { shape: Shape::Sphere { r } })
    }
}

/// A prop: `Prop(Box(...), at=(x, y, z), turn=0, name="cup", fill="#e9dcc3")`;
/// `at` is the middle of its bottom.
#[pyclass(name = "Prop", frozen)]
pub struct PyProp {
    prop: Prop,
}

#[pymethods]
impl PyProp {
    #[new]
    #[pyo3(signature = (shape, at = (0.0, 0.0, 0.0), turn = 0.0, name = String::new(), fill = String::new(), shadow = true, boil = false))]
    fn new(shape: &Bound<'_, PyAny>, at: (f64, f64, f64), turn: f64, name: String, fill: String, shadow: bool, boil: bool) -> PyResult<Self> {
        let shape = match shape.cast::<PyBox>() {
            Ok(b) => b.get().shape.clone(),
            Err(_) => shape.cast::<PySphere>().map_err(|_| PyTypeError::new_err("Prop: the shape is Box or Sphere"))?.get().shape.clone(),
        };
        let mut prop = Prop::new(&name, shape, at.into());
        prop.turn = turn;
        prop.fill = Some(fill).filter(|f| !f.is_empty());
        prop.shadow = shadow;
        prop.boil = boil;
        Ok(PyProp { prop })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.prop.name
    }
}

/// A campfire: stylized flame, flying tongues, sparks, a glow on the ground.
/// `wind` — a number or keys (+ — to the right on screen).
#[pyclass(name = "Fire", frozen)]
pub struct PyFire {
    fire: mavian_scene::Fire,
}

#[pymethods]
impl PyFire {
    #[new]
    #[pyo3(signature = (at = (0.0, 0.0, 0.0), height = 0.9, width = 0.5, seed = 1, rate = 2.2, wind = None, puffs = 6, sparks = 14, glow = "#ff9a3c".to_string(), over = 1.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        at: (f64, f64, f64),
        height: f64,
        width: f64,
        seed: u64,
        rate: f64,
        wind: Option<&Bound<'_, PyAny>>,
        puffs: usize,
        sparks: usize,
        glow: String,
        over: f64,
    ) -> PyResult<Self> {
        let mut fire = mavian_scene::Fire::new(at.into());
        (fire.height, fire.width, fire.seed, fire.rate, fire.puffs, fire.sparks, fire.glow) = (height, width, seed, rate, puffs, sparks, glow);
        fire.over = over;
        if let Some(w) = wind {
            fire.wind = track_f64(w)?;
        }
        Ok(PyFire { fire })
    }
}

/// Line segments drawn in colour (axes, a wire cube).
#[pyclass(name = "Wire", frozen)]
pub struct PyWire {
    wire: Wire,
}

type WireSegment = ((f64, f64, f64), (f64, f64, f64));

#[pymethods]
impl PyWire {
    #[new]
    #[pyo3(signature = (segments, color = "#7a7468", width = 2.0, at = None))]
    fn new(segments: Vec<WireSegment>, color: &str, width: f64, at: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        if !width.is_finite() || width <= 0.0 {
            return Err(PyValueError::new_err("Wire width must be positive"));
        }
        Ok(Self {
            wire: Wire {
                segments: segments.into_iter().map(|(a, b)| (a.into(), b.into())).collect(),
                color: color.into(),
                width,
                at: at.map(point).transpose()?,
            },
        })
    }
}

/// A wire cube.
#[pyfunction]
#[pyo3(name = "cube", signature = (center = (0.0, 0.5, 0.0), size = 1.0, color = "#7a7468"))]
fn cube_fn(center: (f64, f64, f64), size: f64, color: &str) -> PyWire {
    PyWire { wire: mavian_scene::cube(center.into(), size, color) }
}

/// Coordinate axes: X red, Y green (up), Z blue.
#[pyfunction]
#[pyo3(name = "axes", signature = (length = 1.5))]
fn axes_fn(length: f64) -> Vec<PyWire> {
    mavian_scene::axes(length).into_iter().map(|wire| PyWire { wire }).collect()
}

enum OntoData {
    Nothing,
    Point(Vec3),
    Prop(Py<PyProp>),
}

enum ActionData {
    Grab {
        actor: Py<PyActor>,
        hand: String,
        prop: Py<PyProp>,
        at: f64,
        reach: f64,
        settle: f64,
        height: f64,
        step: Option<mavian_scene::Approach>,
    },
    Release {
        actor: Py<PyActor>,
        hand: String,
        prop: Py<PyProp>,
        at: f64,
        onto: OntoData,
        reach: f64,
        settle: f64,
    },
    Push {
        actor: Py<PyActor>,
        prop: Py<PyProp>,
        start: f64,
        end: f64,
        hands: (String, String),
        reach: f64,
        settle: f64,
        height: f64,
        spread: f64,
        step: Option<mavian_scene::Approach>,
    },
    Strike {
        attacker: Py<PyActor>,
        limb: String,
        target: Py<PyActor>,
        blocked: String,
        s: mavian_scene::Strike,
    },
}

/// An intent with props (made by `grab`, `release`, `push`).
#[pyclass(name = "Action", frozen)]
pub struct PyAction {
    data: ActionData,
}

/// Take a prop with a hand at `at`: the hand reaches for its near side.
#[pyfunction]
#[pyo3(signature = (actor, hand, prop, at, reach = 0.4, settle = 0.35, height = 0.6, auto_step = 0.0, step_duration = 0.8))]
#[allow(clippy::too_many_arguments)]
fn grab(
    actor: Py<PyActor>,
    hand: String,
    prop: Py<PyProp>,
    at: f64,
    reach: f64,
    settle: f64,
    height: f64,
    auto_step: f64,
    step_duration: f64,
) -> PyResult<PyAction> {
    Ok(PyAction { data: ActionData::Grab { actor, hand, prop, at, reach, settle, height, step: step_options(auto_step, step_duration)? } })
}

fn step_options(distance: f64, duration: f64) -> PyResult<Option<mavian_scene::Approach>> {
    if !distance.is_finite() || distance < 0.0 || !duration.is_finite() || duration <= 0.0 {
        return Err(PyValueError::new_err("auto_step must be finite and nonnegative; step_duration must be positive and finite"));
    }
    Ok((distance > 0.0).then_some(mavian_scene::Approach { max_distance: distance, duration }))
}

/// Put down what the hand holds at `at`: on a point, on a prop (its near
/// edge) or let go (it falls to the ground).
#[pyfunction]
#[pyo3(signature = (actor, hand, prop, at, onto = None, reach = 0.6, settle = 0.35))]
#[allow(clippy::too_many_arguments)]
fn release(
    actor: Py<PyActor>,
    hand: String,
    prop: Py<PyProp>,
    at: f64,
    onto: Option<&Bound<'_, PyAny>>,
    reach: f64,
    settle: f64,
) -> PyResult<PyAction> {
    let onto = match onto {
        None => OntoData::Nothing,
        Some(o) => match o.cast::<PyProp>() {
            Ok(p) => OntoData::Prop(p.clone().unbind()),
            Err(_) => OntoData::Point(vec3(o)?),
        },
    };
    Ok(PyAction { data: ActionData::Release { actor, hand, prop, at, onto, reach, settle } })
}

/// Push a prop from `start` to `end`: it slides with the character, the
/// palms against its near side.
#[pyfunction]
#[pyo3(signature = (actor, prop, start, end, hands = ("hand_l".to_string(), "hand_r".to_string()), reach = 0.35, settle = 0.35, height = 0.6, spread = 0.17, auto_step = 0.0, step_duration = 0.8))]
#[allow(clippy::too_many_arguments)]
fn push(
    actor: Py<PyActor>,
    prop: Py<PyProp>,
    start: f64,
    end: f64,
    hands: (String, String),
    reach: f64,
    settle: f64,
    height: f64,
    spread: f64,
    auto_step: f64,
    step_duration: f64,
) -> PyResult<PyAction> {
    Ok(PyAction {
        data: ActionData::Push { actor, prop, start, end, hands, reach, settle, height, spread, step: step_options(auto_step, step_duration)? },
    })
}

/// A strike with `limb` of `attacker` into `aim` of `target` at `at`: the
/// reaction, knockback, hit-stop, shake and impact frame follow by themselves.
#[pyfunction]
#[pyo3(signature = (attacker, limb, target, at, aim = "head".to_string(), windup = 0.14, hold = 0.06, recover = 0.3, blocked = String::new(), miss = false, force = 0.3, stop = 0.1, shake = 0.06, flash = false))]
#[allow(clippy::too_many_arguments)]
fn strike(
    attacker: Py<PyActor>,
    limb: String,
    target: Py<PyActor>,
    at: f64,
    aim: String,
    windup: f64,
    hold: f64,
    recover: f64,
    blocked: String,
    miss: bool,
    force: f64,
    stop: f64,
    shake: f64,
    flash: bool,
) -> PyAction {
    let mut s = mavian_scene::Strike::new(0, 0, 0, at);
    (s.aim, s.windup, s.hold, s.recover, s.miss, s.force, s.stop, s.shake, s.flash) = (aim, windup, hold, recover, miss, force, stop, shake, flash);
    PyAction { data: ActionData::Strike { attacker, limb, target, blocked, s } }
}

/// How much further `target` is than `limb` of `attacker` reaches at `at`
/// (≤ 0 — it reaches): to put opponents at striking distance by rule.
#[pyfunction]
#[pyo3(signature = (attacker, limb, target, at, aim = "head"))]
fn reach_gap(attacker: &Bound<'_, PyActor>, limb: &str, target: &Bound<'_, PyActor>, at: f64, aim: &str) -> PyResult<f64> {
    let (a, d) = (attacker.get().core()?, target.get().core()?);
    let limb = a.model.rig.joint(limb).map_err(|e| PyKeyError::new_err(e.0))?;
    mavian_scene::reach_gap(a, limb, d, at, aim).map_err(PyValueError::new_err)
}

/// Text in the frame at a point of the world: `Label("trust = 0", (x, 0.6, 0.0))`.
#[pyclass(name = "Label", frozen)]
pub struct PyLabel {
    label: Label,
}

#[pymethods]
impl PyLabel {
    #[new]
    #[pyo3(signature = (text, at, size = 22.0))]
    fn new(text: String, at: (f64, f64, f64), size: f64) -> Self {
        PyLabel { label: Label { text, at: at.into(), size } }
    }
}

/// A correction of a pose by code over world time (not frames):
/// `Patch(at=(4.0, 4.9), joint="arm_l", add=(0, 55, 0))` — fades in and out
/// over `ramp` seconds.
#[pyclass(name = "Patch", frozen)]
pub struct PyPatch {
    data: PyPatchData,
}

#[pymethods]
impl PyPatch {
    #[new]
    #[pyo3(signature = (at, joint, add, actor = 0, ramp = 0.08))]
    fn new(at: (f64, f64), joint: String, add: (f64, f64, f64), actor: usize, ramp: f64) -> PyResult<Self> {
        if at.1 <= at.0 || ramp < 0.0 {
            return Err(PyValueError::new_err("Patch: at=(from, to) with from < to, ramp not negative"));
        }
        Ok(PyPatch { data: PyPatchData { actor, from: at.0, to: at.1, joint, add, ramp } })
    }
}

#[pymethods]
impl PyShot {
    /// The camera can be set after the shot is made (a camera placed by the
    /// shot's own timeline, as the fight's zoom in a hit-stop).
    #[setter]
    fn set_camera(&mut self, camera: &Bound<'_, PyCamera>) {
        self.camera = camera.get().rig.clone();
    }

    #[getter]
    fn actors(&self, py: Python<'_>) -> Vec<Py<PyActor>> {
        self.actors.iter().map(|a| a.clone_ref(py)).collect()
    }

    /// The shot's clocks, with hit-stops: `shot.timeline().shot_time_of(t)`.
    fn timeline(&self, py: Python<'_>) -> PyResult<PyTimeline> {
        Ok(PyTimeline { tl: self.build(py, "shot")?.timeline() })
    }

    #[new]
    #[pyo3(signature = (camera = None, actors = None, fps = 24, length = 3.0, seed = 1, size = (1280, 720), draw_on = None, time_warp = None, shake = None, ground = None, style = None, labels = None, patches = None, props = None, actions = None, contacts = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        camera: Option<&Bound<'_, PyCamera>>,
        actors: Option<&Bound<'_, PyList>>,
        fps: u32,
        length: f64,
        seed: u64,
        size: (u32, u32),
        draw_on: Option<&Bound<'_, PyAny>>,
        time_warp: Option<&Bound<'_, PyAny>>,
        shake: Option<&Bound<'_, PyAny>>,
        ground: Option<&Bound<'_, PyGround>>,
        style: Option<&Bound<'_, PyStyle>>,
        labels: Option<&Bound<'_, PyList>>,
        patches: Option<&Bound<'_, PyList>>,
        props: Option<&Bound<'_, PyList>>,
        actions: Option<&Bound<'_, PyList>>,
        contacts: Option<&Bound<'_, PyList>>,
    ) -> PyResult<Self> {
        let (mut prop_list, mut wires, mut fires) = (Vec::new(), Vec::new(), Vec::new());
        for x in props.into_iter().flat_map(|l| l.iter()) {
            if let Ok(p) = x.cast::<PyProp>() {
                prop_list.push(p.clone().unbind());
            } else if let Ok(f) = x.cast::<PyFire>() {
                fires.push(f.get().fire.clone());
            } else {
                wires.push(x.cast::<PyWire>().map_err(|_| PyTypeError::new_err("props: Prop, Fire or a wire (cube, axes)"))?.get().wire.clone());
            }
        }
        let action_list = match actions {
            None => Vec::new(),
            Some(l) => l.iter().map(|x| Ok(x.cast::<PyAction>()?.clone().unbind())).collect::<PyResult<_>>()?,
        };
        let actors = match actors {
            None => Vec::new(),
            Some(l) => l.iter().map(|a| Ok(a.cast::<PyActor>()?.clone().unbind())).collect::<PyResult<_>>()?,
        };
        let shakes = match shake {
            None => Vec::new(),
            Some(s) => match s.cast::<PyShake>() {
                Ok(one) => vec![one.get().shake.clone()],
                Err(_) => s.try_iter()?.map(|x| Ok(x?.cast::<PyShake>()?.get().shake.clone())).collect::<PyResult<_>>()?,
            },
        };
        let contact_list =
            contacts.into_iter().flat_map(|l| l.iter()).map(|c| Ok(c.cast::<PyContact>()?.clone().unbind())).collect::<PyResult<Vec<_>>>()?;
        let shot = PyShot {
            camera: camera.map(|c| c.get().rig.clone()).unwrap_or_default(),
            actors,
            fps,
            length,
            seed,
            size,
            draw_on: draw_on.map(track_f64).transpose()?.unwrap_or(Track::Const(2.0)),
            time_warp: time_warp.map(track_f64).transpose()?.unwrap_or(Track::Const(1.0)),
            shakes,
            ground: ground.map(|g| g.get().ground.clone()),
            style: style.map(|s| s.get().style.clone()).unwrap_or_default(),
            labels: match labels {
                None => Vec::new(),
                Some(l) => l.iter().map(|x| Ok(x.cast::<PyLabel>()?.get().label.clone())).collect::<PyResult<_>>()?,
            },
            patches: match patches {
                None => Vec::new(),
                Some(l) => l.iter().map(|x| Ok(x.cast::<PyPatch>()?.get().data.clone())).collect::<PyResult<_>>()?,
            },
            props: prop_list,
            wires,
            fires,
            actions: action_list,
            contacts: contact_list,
        };
        // check here, so the error points at the line that made the shot
        shot.build(py, "shot")?;
        Ok(shot)
    }
}

/// The shot's clocks: frame times in the world and the shot.
#[pyclass(name = "Timeline", frozen)]
pub struct PyTimeline {
    tl: mavian_core::Timeline,
}

#[pymethods]
impl PyTimeline {
    /// Shot time at which the world reaches time w.
    fn shot_time_of(&self, w: f64) -> f64 {
        self.tl.shot_time_of(w)
    }

    fn __len__(&self) -> usize {
        self.tl.len()
    }
}

impl PyShot {
    /// The core shot: fresh actors, so caches never leak between shots.
    pub fn build(&self, py: Python<'_>, name: &str) -> PyResult<Shot> {
        let shot = Shot {
            name: name.into(),
            camera: self.camera.clone(),
            actors: self
                .actors
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    let mut actor = a.bind(py).get().build();
                    for p in self.patches.iter().filter(|p| p.actor == i) {
                        let joint = actor.model.rig.joint(&p.joint).map_err(|e| PyKeyError::new_err(e.0))?;
                        let add = mavian_core::v3(p.add.0, p.add.1, p.add.2);
                        actor = actor.patch(Patch { from: p.from, to: p.to, joint, add, ramp: p.ramp });
                    }
                    Ok(actor)
                })
                .collect::<PyResult<_>>()?,
            fps: self.fps,
            length: self.length,
            seed: self.seed,
            size: self.size,
            draw_on: self.draw_on.clone(),
            time_warp: self.time_warp.clone(),
            shakes: self.shakes.clone(),
            ground: self.ground.clone(),
            style: self.style.clone(),
            labels: self.labels.clone(),
            props: self.props.iter().map(|p| p.bind(py).get().prop.clone()).collect(),
            wires: self.wires.clone(),
            fires: self.fires.clone(),
            actions: Vec::new(),
            stops: Vec::new(),
            flashes: Vec::new(),
            auto_shakes: Vec::new(),
        };
        let mut shot = shot;
        let actor_of = |a: &Py<PyActor>| -> PyResult<usize> {
            self.actors.iter().position(|x| x.is(a)).ok_or_else(|| PyValueError::new_err("an action's actor is not in the shot's actors"))
        };
        let prop_of = |p: &Py<PyProp>| -> PyResult<usize> {
            self.props.iter().position(|x| x.is(p)).ok_or_else(|| PyValueError::new_err("an action's prop is not in the shot's props"))
        };
        let joint =
            |shot: &Shot, i: usize, name: &str| -> PyResult<usize> { shot.actors[i].model.rig.joint(name).map_err(|e| PyKeyError::new_err(e.0)) };
        for c in &self.contacts {
            let c = c.bind(py).get();
            let i = actor_of(&c.actor)?;
            let effector = joint(&shot, i, &c.joint)?;
            let chain = mavian_scene::actions::chain(&shot.actors[i], effector).map_err(PyValueError::new_err)?;
            let anchor = shot.actors[i].figure(c.at.0).joints[effector];
            let target = if let Some(p) = &c.prop {
                mavian_scene::Target::OnProp { prop: prop_of(p)?, local: c.local }
            } else {
                mavian_scene::Target::Point(c.target.unwrap_or(anchor))
            };
            let name = if c.name.is_empty() { format!("contact:{i}:{}", c.joint) } else { c.name.clone() };
            let mut contact = mavian_scene::Contact::new(&name, chain, target, c.at.0, c.at.1);
            contact.priority = c.priority;
            contact.tolerance = c.tolerance;
            contact.pole = c.pole;
            contact.approach = c.approach;
            let ramp = c.ramp.min((c.at.1 - c.at.0) * 0.5);
            if ramp > 0.0 {
                contact.weight = mavian_core::Keys::new(mavian_core::Ease::InOut)
                    .key(c.at.0, 0.0)
                    .key(c.at.0 + ramp, 1.0)
                    .key(c.at.1 - ramp, 1.0)
                    .key(c.at.1, 0.0)
                    .into();
            }
            for side in [mavian_rig::Side::L, mavian_rig::Side::R] {
                if let Some(foot) = shot.actors[i].model.rig.foot(side)
                    && foot.heel == effector
                {
                    let fig = shot.actors[i].figure(c.at.0);
                    contact.follower = Some((foot.toe, fig.joints[foot.toe] - fig.joints[foot.heel]));
                }
            }
            contact.validate(&shot.actors[i].model.rig).map_err(PyValueError::new_err)?;
            shot.actors[i].contacts.push(contact);
            shot.actors[i].forget();
        }
        for a in &self.actions {
            let action = match &a.bind(py).get().data {
                ActionData::Grab { actor, hand, prop, at, reach, settle, height, step } => {
                    let i = actor_of(actor)?;
                    Action::Grab {
                        actor: i,
                        hand: joint(&shot, i, hand)?,
                        prop: prop_of(prop)?,
                        at: *at,
                        reach: *reach,
                        settle: *settle,
                        height: *height,
                        step: *step,
                    }
                }
                ActionData::Release { actor, hand, prop, at, onto, reach, settle } => {
                    let i = actor_of(actor)?;
                    let onto = match onto {
                        OntoData::Nothing => Onto::Nothing,
                        OntoData::Point(p) => Onto::Point(*p),
                        OntoData::Prop(p) => Onto::Prop(prop_of(p)?),
                    };
                    Action::Release { actor: i, hand: joint(&shot, i, hand)?, prop: prop_of(prop)?, at: *at, onto, reach: *reach, settle: *settle }
                }
                ActionData::Strike { attacker, limb, target, blocked, s } => {
                    let (i, d) = (actor_of(attacker)?, actor_of(target)?);
                    let mut s = s.clone();
                    (s.attacker, s.target, s.limb) = (i, d, joint(&shot, i, limb)?);
                    s.blocked = if blocked.is_empty() { None } else { Some(joint(&shot, d, blocked)?) };
                    Action::Strike(s)
                }
                ActionData::Push { actor, prop, start, end, hands, reach, settle, height, spread, step } => {
                    let i = actor_of(actor)?;
                    let hands = [joint(&shot, i, &hands.0)?, joint(&shot, i, &hands.1)?];
                    Action::Push {
                        actor: i,
                        prop: prop_of(prop)?,
                        start: *start,
                        end: *end,
                        hands,
                        reach: *reach,
                        settle: *settle,
                        height: *height,
                        spread: *spread,
                        step: *step,
                    }
                }
            };
            shot.actions.push(action);
        }
        shot.compile().map_err(|e| PyValueError::new_err(e.0))?;
        if let Some(p) = self.patches.iter().find(|p| p.actor >= self.actors.len()) {
            return Err(PyValueError::new_err(format!("Patch for actor {}: the shot has {}", p.actor, self.actors.len())));
        }
        shot.validate().map_err(|e| PyValueError::new_err(e.0))?;
        Ok(shot)
    }
}

// --- module ------------------------------------------------------------------

#[pymodule]
fn mavian(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKeys>()?;
    m.add_class::<PyPoseKey>()?;
    m.add_class::<PyModel>()?;
    m.add_class::<PyActor>()?;
    m.add_class::<PyLabel>()?;
    m.add_class::<PyPose>()?;
    m.add_class::<PyBox>()?;
    m.add_class::<PySphere>()?;
    m.add_class::<PyProp>()?;
    m.add_class::<PyWire>()?;
    m.add_class::<PyFire>()?;
    m.add_class::<PyAction>()?;
    m.add_class::<PyContact>()?;
    m.add_function(wrap_pyfunction!(cube_fn, m)?)?;
    m.add_function(wrap_pyfunction!(axes_fn, m)?)?;
    m.add_function(wrap_pyfunction!(grab, m)?)?;
    m.add_function(wrap_pyfunction!(release, m)?)?;
    m.add_function(wrap_pyfunction!(push, m)?)?;
    m.add_function(wrap_pyfunction!(strike, m)?)?;
    m.add_function(wrap_pyfunction!(reach_gap, m)?)?;
    m.add_class::<PyPatch>()?;
    m.add_class::<PyFollow>()?;
    m.add_class::<PyBetween>()?;
    m.add_class::<PyCamera>()?;
    m.add_class::<PyShake>()?;
    m.add_class::<PyGround>()?;
    m.add_class::<PyStyle>()?;
    m.add_class::<PyShot>()?;
    m.add_function(wrap_pyfunction!(keys, m)?)?;
    m.add_function(wrap_pyfunction!(key, m)?)?;
    m.add_function(wrap_pyfunction!(walk, m)?)?;
    m.add_function(wrap_pyfunction!(walk_end, m)?)?;
    m.add_function(wrap_pyfunction!(jump, m)?)?;
    m.add_function(wrap_pyfunction!(stickman_actor, m)?)?;
    m.add_class::<PyJump>()?;
    m.add_function(wrap_pyfunction!(where_fn, m)?)?;
    m.add_function(wrap_pyfunction!(watch, m)?)?;
    m.add_function(wrap_pyfunction!(orbit, m)?)?;
    m.add_function(wrap_pyfunction!(cuts, m)?)?;
    m.add("STICKMAN", PyModel { model: stickman() })?;
    m.add("MANNEQUIN", PyModel { model: mavian_models::mannequin::mannequin() })?;
    m.add("FIGHTER", PyModel { model: mavian_models::fighter::fighter() })?;
    Ok(())
}

/// Run a shot file and return its shots: `shot` as "base" plus `variants`.
pub fn load_file(path: &Path) -> Result<Vec<(String, Shot)>, String> {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        pyo3::append_to_inittab!(mavian);
        Python::initialize();
    });
    let code = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("shot").to_string();
    Python::attach(|py| {
        let run = || -> PyResult<Vec<(String, Shot)>> {
            // the shot's folder is importable: shared helpers next to shots
            if let Some(dir) = path.parent().and_then(|d| d.to_str()) {
                let sys_path = py.import("sys")?.getattr("path")?;
                sys_path.call_method1("insert", (0, if dir.is_empty() { "." } else { dir }))?;
            }
            let globals = PyDict::new(py);
            globals.set_item("__name__", "__shot__")?;
            globals.set_item("__file__", path.to_string_lossy().as_ref())?;
            globals.set_item("__builtins__", py.import("builtins")?)?;
            let src = CString::new(code.as_str()).map_err(|_| PyValueError::new_err("NUL byte in the shot file"))?;
            let file = CString::new(path.to_string_lossy().as_ref()).unwrap_or_default();
            let compiled = pyo3::types::PyCode::compile(py, &src, &file, pyo3::types::code::PyCodeInput::File)?;
            compiled.run(Some(&globals), None)?;
            let base = globals.get_item("shot")?.ok_or_else(|| PyKeyError::new_err("the shot file must define `shot`"))?;
            let mut out = vec![("base".to_string(), base.cast::<PyShot>()?.borrow().build(py, &stem)?)];
            if let Some(vs) = globals.get_item("variants")? {
                for (name, s) in vs.cast::<PyDict>()?.iter() {
                    let name: String = name.extract()?;
                    let shot = s.cast::<PyShot>()?.borrow().build(py, &format!("{stem}-{name}"))?;
                    out.push((name, shot));
                }
            }
            Ok(out)
        };
        run().map_err(|e| {
            let tb = e.traceback(py).and_then(|t| t.format().ok()).unwrap_or_default();
            format!("{tb}{e}")
        })
    })
}
