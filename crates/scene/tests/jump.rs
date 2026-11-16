//! A jump from intent keeps the pelvis's horizontal speed continuous through
//! take-off, flight and landing; a hand-made flight that lands off its key
//! is caught before the first frame.

use mavian_core::{Ease, Keys};
use mavian_models::stickman::stickman;
use mavian_rig::Side::L;
use mavian_scene::{Actor, JumpPoses, jump, pk};

fn speeds(a: &Actor, from: f64, to: f64) -> Vec<(f64, f64, f64)> {
    let dt = 1.0 / 240.0;
    let mut out = Vec::new();
    let mut t = from;
    while t < to {
        let (p, q) = (a.where_(t, None), a.where_(t + dt, None));
        out.push((t, mavian_core::m::hypot(q.x - p.x, q.z - p.z) / dt, (q.y - p.y) / dt));
        t += dt;
    }
    out
}

#[test]
fn speed_is_continuous_through_the_jump() {
    let model = stickman();
    let j = jump(&model, 1.0, 1.0, 2.4, 0.5, L, &JumpPoses::default()).unwrap();
    assert!(j.push < j.launch && j.launch < j.apex && j.apex < j.landing && j.landing < j.end);
    let a = Actor::new(model).path((0.0, 0.0)).poses(
        Keys::new(Ease::InOut).key(0.0, pk("crouch").plant(L)).extend(j.keys.clone()).key(j.end + 0.5, pk("stand").plant(L)),
    );
    a.validate().unwrap();
    let v = speeds(&a, j.push - 0.1, j.end + 0.2);
    // forward speed: no step anywhere (the impact is vertical)
    for w in v.windows(2) {
        assert!((w[1].1 - w[0].1).abs() < 0.25, "forward speed jumps at {:.3}: {:.2} → {:.2}", w[0].0, w[0].1, w[1].1);
    }
    // in flight: constant forward speed, the take-off speed
    let flight: Vec<_> = v.iter().filter(|s| s.0 > j.launch + 0.01 && s.0 < j.landing - 0.01).collect();
    assert!(!flight.is_empty());
    for s in &flight {
        assert!((s.1 - j.speed.0).abs() < 0.01, "flight speed {:.3} ≠ take-off {:.3}", s.1, j.speed.0);
    }
    // the top of the flight is as high as asked
    let y = |t: f64| a.where_(t, None).y;
    assert!((y(j.apex) - y(j.launch) - 0.5).abs() < 0.01, "rise {:.3}", y(j.apex) - y(j.launch));
}

#[test]
fn a_flight_landing_off_its_key_is_an_error() {
    let a = Actor::new(stickman()).path((0.0, 0.0)).poses(
        Keys::new(Ease::InOut)
            .key(0.0, pk("crouch").plant(L))
            .eased(0.3, pk("leap_launch").plant(L), Ease::In)
            .key(0.6, pk("leap_air").fly())
            .key(2.0, pk("leap_reach").fly()) // far too late: he would be under the ground
            .key(2.3, pk("land").plant(L)),
    );
    let e = a.validate().unwrap_err().0;
    assert!(e.contains("does not land at its key"), "{e}");
}
