use mavian_core::{Mat3, Vec3};
use mavian_scene::{Shape, obstacle::Obstacle};
fn p(x: f64, y: f64, z: f64) -> Vec3 {
    Vec3 { x, y, z }
}
#[test]
fn rotated_box_and_sphere_return_signed_world_distances() {
    let o = Obstacle { prop: 0, shape: Shape::Box { sx: 2.0, sy: 2.0, sz: 1.0 }, pos: p(3., 0., 4.), rot: Mat3::rot_y(90.) };
    let (d, n) = o.distance(p(3.75, 1., 4.));
    assert!((d - 0.25).abs() < 1e-9);
    assert!((n - p(1., 0., 0.)).length() < 1e-9);
    assert!((o.distance(p(3., 1., 4.)).0 + 0.5).abs() < 1e-9);
    let sphere = Obstacle { prop: 1, shape: Shape::Sphere { r: 0.5 }, pos: Vec3::ZERO, rot: Mat3::IDENT };
    assert!((sphere.distance(p(0., 0.5, 0.)).0 + 0.5).abs() < 1e-9);
    assert!((sphere.distance(p(0., 1.2, 0.)).0 - 0.2).abs() < 1e-9);
}
#[test]
fn capsule_query_detects_a_crossing_with_both_endpoints_outside() {
    let o = Obstacle { prop: 0, shape: Shape::Box { sx: 0.2, sy: 0.3, sz: 0.2 }, pos: Vec3::ZERO, rot: Mat3::rot_y(37.) };
    let (a, b) = (p(-1., 0.15, 0.), p(1., 0.15, 0.));
    assert!(o.distance(a).0 > 0.5 && o.distance(b).0 > 0.5);
    let (d, u, _) = o.capsule(a, 0.05, b, 0.05);
    assert!(d < -0.14);
    assert!((u - 0.5).abs() < 0.001);
    assert!(o.capsule(p(-1., 0.6, 0.), 0.05, p(1., 0.6, 0.), 0.05).0 > 0.20);
}
#[test]
fn an_impossible_obstacle_is_reported_instead_of_moving_the_shoulder() {
    use mavian_scene::{Actor, ContactStatus, Prop, Shot};
    let a = Actor::new(mavian_models::mannequin::mannequin());
    let s = a.figure(0.0);
    let shoulder = a.model.rig.joint("arm_r").unwrap();
    let box_prop = Prop::new("blocked-root", Shape::Box { sx: 2., sy: 2., sz: 2. }, s.joints[shoulder] - Vec3::UP);
    let shot = Shot { actors: vec![a], props: vec![box_prop], ..Shot::default() };
    let solved = shot.posed(0, 0.).0;
    assert_eq!(s.joints[shoulder], solved.joints[shoulder]);
    assert!(!shot.obstacle_reports(0.).is_empty());
    assert!(shot.contact_reports(0.).iter().any(|(_, r)| r.status == ContactStatus::Obstructed && r.name.starts_with("collision:")));
}
