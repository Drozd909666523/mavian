use mavian_scene::{Actor,CameraRig,Fire,Prop,Shape,Shot,Watch};
use mavian_core::Vec3;
#[test]
fn sketch_includes_face_filled_head_props_and_fire() {
    let model=mavian_models::stickman::stickman();
    let mut shot=Shot { actors:vec![Actor::new(model.clone())],size:(96,96),length:0.1,camera:CameraRig::Watch(Watch::new((0.0,1.0,4.0),(0.0,1.0,0.0))),..Shot::default() };
    shot.props.push(Prop::new("box",Shape::Box{sx:0.2,sy:0.2,sz:0.2},Vec3::ZERO));
    shot.props[0].fill=Some("#abcdef".into());shot.fires.push(Fire::new(Vec3::ZERO));
    let tl=shot.timeline();let f=shot.frame(&tl,0);
    let face=mavian_render::svg(&shot,&f);
    assert!(face.contains("#abcdef"));assert!(face.contains("fire-0"));
    shot.actors[0].face=false;
    let no_face=mavian_render::svg(&shot,&f);assert_ne!(face,no_face);
    let mut filled=(*model).clone();
    for p in &mut filled.parts { if let mavian_rig::Part::Head { filled,.. }=p { *filled=true; } }
    shot.actors[0].model=std::sync::Arc::new(filled);
    let filled=mavian_render::svg(&shot,&f);assert_ne!(filled,no_face);
    assert!(!filled.contains("NaN"));
}
