//! One continuous take, three cameras with cuts.
//!
//! 1. Side view: the stick figure walks.
//! 2. Front camera, fisheye: he comes almost right up to it, peers
//!    into the lens and turns his head slightly.
//! 3. Camera on the ground: he jumps over it, the camera follows him.
//!
//! There is one action in world time; cameras are placed from where the
//! character will be (`where_`), so changing the steps adjusts everything by itself.

use mavian_core::{Ease, Keys};
use mavian_models::stickman::stickman;
use mavian_rig::Side::L;
use mavian_scene::{Actor, CameraRig, Follow, Ground, JumpPoses, Shake, Shot, Style, Watch, jump, pk, walk};

/// cut to the front camera
const CUT_FRONT: f64 = 3.0;
/// cut to the ground camera
const CUT_LOW: f64 = 8.6;
/// take-off: everything else in the jump follows from it and the distance
const LAUNCH: f64 = 9.30;
/// from the take-off foot to the landing foot, m
const DISTANCE: f64 = 2.4;
/// how high the pelvis rises, m
const HEIGHT: f64 = 0.5;

pub fn make() -> Shot {
    use Ease::*;
    let model = stickman();
    let leap = jump(&model, 1.0, LAUNCH, DISTANCE, HEIGHT, L, &JumpPoses::default()).expect("the jump is possible");
    let land = leap.end;
    let actor = Actor::new(model)
        .path((0.0, -3.4)) // where he starts walking from
        .poses(
            Keys::new(InOut)
                .extend(walk(0.0, 11, 0.5, L, Some("stand"), 0.35, "")) // 11 steps and a stop
                .key(6.40, pk("peer")) // leans towards the lens
                .key(6.95, pk("peer_l")) // head to the left
                .key(7.60, pk("peer_r")) // head to the right
                .key(8.15, pk("peer"))
                .key(8.60, pk("peer"))
                .key(9.05, pk("crouch").plant(L)) // wind-up
                // the jump: push, flight and landing from the take-off and distance
                .extend(leap.keys.clone())
                .key(land + 0.11, pk("land").plant(L))
                .key(land + 0.71, pk("stand").plant(L))
                .key(land + 1.21, pk("look_back")), // looks back at the camera
        )
        .blink(
            Keys::of([(6.60, 0.0), (6.66, 1.0), (6.74, 0.0), (8.00, 0.0), (8.06, 1.0), (8.14, 0.0)]).ease(Linear),
        )
        .squash(0.3);

    let stop = actor.where_(6.0, None); // where he stopped
    let mid = (actor.where_(0.0, None).z + actor.where_(CUT_FRONT, None).z) / 2.0;

    let side = Watch::new((-8.0, 1.3, mid), (0.0, 1.0, mid)).fov(30.0);
    let front = Watch::new((0.0, 1.52, stop.z + 0.62), (0.0, 1.5, stop.z - 2.0)).fov(140.0).fisheye();
    let lens = (0.0, 0.12, stop.z + 1.7); // on the ground, in front of him
    // when he is right above the lens: from where he takes off and how fast
    let over = LAUNCH + (lens.2 - actor.where_(LAUNCH, None).z) / leap.speed.0;
    // follows the pelvis like an operator, over 0.2 s: the push is ~2 g, and
    // a camera locked to it would jerk
    let low = Watch::new(lens, Follow::new(0, "pelvis").offset((0.0, 0.25, 0.0)).smooth(0.2))
        .fov(70.0)
        // while he flies over the camera, it smoothly turns after him
        .turn(Keys::of([(LAUNCH, 180.0), (leap.landing, 0.0)]));

    Shot {
        name: "three_cams".into(),
        fps: 24,
        length: 12.3,
        seed: 5,
        camera: CameraRig::cuts([(0.0, side.into()), (CUT_FRONT, front.into()), (CUT_LOW, low.into())]),
        actors: vec![actor],
        // the jump is on ones
        draw_on: Keys::of([(0.0, 2.0)]).eased(CUT_LOW, 1.0, Hold).into(),
        // slow motion as he passes over the camera
        time_warp: Keys::of([(over - 0.17, 1.0), (over - 0.09, 0.25), (over + 0.17, 0.25), (over + 0.25, 1.0)])
            .ease(Linear)
            .into(),
        shakes: vec![Shake::new(leap.landing, 0.05, 0.25)],
        ground: Some(Ground { size: 8.0, center: (0.0, 1.8), ..Ground::default() }),
        style: Style { boil: 1.5, ..Style::default() },
        ..Shot::default()
    }
}
