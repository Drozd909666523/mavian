use mavian_core::{Vec3,Track};
use mavian_cpu::Cpu;
use mavian_frame::{build,Layers};
use mavian_scene::{Actor,CameraRig,Fire,Follow,Point,Prop,Shape,Shot,Watch,cube};

fn scene()->Shot {
    Shot { actors:vec![Actor::new(mavian_models::stickman::stickman())],camera:CameraRig::Watch(Watch::new((0.0,1.0,4.0),(0.0,0.9,0.0))),size:(96,96),length:0.3,..Shot::default() }
}
#[test]
fn drawing_hold_boil_pressure_and_static_props() {
    let mut s=scene();
    s.props.push(Prop::new("box",Shape::Box{sx:0.3,sy:0.3,sz:0.3},Vec3::ZERO));
    s.draw_on=Track::Const(2.0);
    let tl=s.timeline();let a=s.frame(&tl,0);let b=s.frame(&tl,1);let c=s.frame(&tl,2);
    let mesh=build(&s,&a);let held=build(&s,&b);let changed=build(&s,&c);
    assert_eq!(by_id(&mesh,1),by_id(&held,1));
    assert_ne!(by_id(&mesh,1),by_id(&changed,1));
    assert_eq!(by_id(&mesh,65),by_id(&changed,65));
    s.props[0].boil=true;
    assert_ne!(by_id(&mesh,65),by_id(&build(&s,&c),65));
    let cpu=Cpu::new();let layers=Layers{paint:true,lines:true,ids:false};
    s.style.boil=0.0;s.style.pressure=0.0;
    let plain=cpu.render(&s,&a,layers);let svg=cpu.lines_svg(&s,&a,false);
    s.style.pressure=0.7;
    assert_ne!(plain,cpu.render(&s,&a,layers));
    assert_ne!(svg,cpu.lines_svg(&s,&a,false));
}
fn by_id(m:&mavian_frame::FrameMesh,id:u32)->Vec<[f32;3]> { m.vertices.iter().chain(&m.top).chain(&m.contacts).filter(|v|v.id==id).map(|v|v.pos).collect() }
#[test]
fn attached_wire_and_oriented_box_shadow() {
    let mut s=scene();let mut w=cube(Vec3::ZERO,0.2,"#123456");w.at=Some(Point::Follow(Follow::new(0,"hand_r")));s.wires.push(w);
    let mut p=Prop::new("box",Shape::Box{sx:1.0,sy:0.2,sz:0.3},Vec3::ZERO);p.turn=45.0;s.props.push(p);
    let tl=s.timeline();let f=s.frame(&tl,0);let mesh=build(&s,&f);
    let wire=mesh.curves.iter().find(|c|c.id==66).unwrap();
    let hand=f.figures[0].joints[s.actors[0].model.rig.joint("hand_r").unwrap()];
    assert!((wire.points[0]-hand).length()<0.2);
    let shadow=mesh.shadows.last().unwrap();assert!(shadow[2]>shadow[3]*2.0);assert!(shadow[4].abs()>0.6&&shadow[5].abs()>0.6);
}
#[test]
fn fire_is_clipped_by_nearer_geometry_and_over_changes_clip() {
    let mut s=scene();s.fires.push(Fire::new(Vec3{x:0.0,y:0.1,z:-1.0}));
    let tl=s.timeline();let f=s.frame(&tl,0);let cpu=Cpu::new();
    s.fires[0].over=0.0;let behind=cpu.fires_svg(&s,&f);
    s.fires[0].over=10.0;let front=cpu.fires_svg(&s,&f);
    assert!(behind.contains("clip-path"));assert_ne!(behind,front);
}
#[test]
fn contact_prop_wins_over_hand_in_depth_buffer() {
    let mut s=scene();s.style.boil=0.0;s.style.pressure=0.0;
    let tl=s.timeline();let f=s.frame(&tl,0);let hand=s.actors[0].model.rig.joint("hand_r").unwrap();
    let pos=f.figures[0].joints[hand];
    let mut p=Prop::new("cup",Shape::Sphere{r:0.025},pos-Vec3::UP*0.025+Vec3{x:0.0,y:0.0,z:-0.04});
    s.props.push(p.clone());let cpu=Cpu::new();let layers=Layers{paint:true,lines:false,ids:true};
    let before=cpu.render(&s,&f,layers);
    p.contacts.push((0.0,1.0,0,hand));s.props[0]=p;
    assert_ne!(before,cpu.render(&s,&f,layers));
}

#[test]
fn effects_reuse_geometry_without_changing_the_frame() {
    let mut s=scene();
    s.fires.push(Fire::new(Vec3{x:0.0,y:0.1,z:-1.0}));
    let mut front=Fire::new(Vec3{x:0.2,y:0.0,z:1.0});front.over=0.0;s.fires.push(front);
    let tl=s.timeline();let f=s.frame(&tl,0);let cpu=Cpu::with_threads(2);
    for layers in [Layers{paint:true,lines:true,ids:false},Layers{paint:false,lines:true,ids:false}] {
        let separate=(cpu.render(&s,&f,layers),cpu.fires_svg(&s,&f));
        assert_eq!(separate,cpu.render_with_effects(&s,&f,layers));
    }
    s.fires.clear();assert!(cpu.render_with_effects(&s,&f,Layers{paint:true,lines:true,ids:false}).1.is_empty());
}
