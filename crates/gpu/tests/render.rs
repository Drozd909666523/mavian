//! The GPU renderer draws the same frame the same way, and its line stage
//! works: a mannequin frame has ink lines, paper fills, and no seam inside
//! the silhouette where the arm grows out of the torso.
//!
//! Skipped (passes with a note) on machines without a usable GPU.

use mavian_core::{Ease, Keys};
use mavian_gpu::{Gpu, Layers};
use mavian_models::mannequin::mannequin;
use mavian_scene::{Actor, CameraRig, Ground, Shot, Watch, pk};

fn shot() -> Shot {
    let actor = Actor::new(mannequin()).poses(Keys::new(Ease::InOut).key(0.0, pk("stand")));
    Shot {
        camera: CameraRig::Watch(Watch::new((0.0, 1.1, 3.2), (0.0, 0.9, 0.0)).fov(40.0)),
        actors: vec![actor],
        length: 0.1,
        size: (320, 240),
        ground: Some(Ground::default()),
        ..Shot::default()
    }
}

#[test]
fn same_frame_same_pixels_and_lines_exist() {
    let Ok(mut gpu) = Gpu::new(true) else {
        eprintln!("no GPU here, skipping");
        return;
    };
    let shot = shot();
    shot.validate().unwrap();
    let tl = shot.timeline();
    let f = shot.frame(&tl, 0);
    let layers = Layers { paint: true, lines: true, ids: false };
    let a = gpu.render(&shot, &f, layers);
    let b = gpu.render(&shot, &f, layers);
    assert_eq!(a, b, "the same frame must give the same pixels");
    assert_eq!(a.len(), 320 * 240 * 4);
    let dark = a.as_chunks::<4>().0.iter().filter(|p| p[0] < 60 && p[1] < 60 && p[2] < 60).count();
    assert!(dark > 200, "the outline must be drawn ({dark} dark pixels)");
    // without lines there is no ink at all: fills are paper
    let plain = gpu.render(&shot, &f, Layers { paint: true, lines: false, ids: false });
    let dark = plain.as_chunks::<4>().0.iter().filter(|p| p[0] < 60 && p[1] < 60 && p[2] < 60).count();
    assert_eq!(dark, 0, "with the lines layer off nothing is ink");
}

/// The GPU and the CPU reference draw the same picture: they may differ in
/// rounding and in single pixels decided on a threshold, not in regions. A
/// shader edit without the same edit in `mavian-cpu` fails here.
#[test]
fn matches_the_cpu_reference() {
    let Ok(mut gpu) = Gpu::new(true) else {
        eprintln!("no GPU here, skipping");
        return;
    };
    let shot = shot();
    let tl = shot.timeline();
    let f = shot.frame(&tl, 0);
    for layers in [Layers { paint: true, lines: true, ids: false }, Layers { paint: true, lines: false, ids: true }] {
        let a = gpu.render(&shot, &f, layers);
        let b = mavian_cpu::Cpu::new().render(&shot, &f, layers);
        let visible = a
            .as_chunks::<4>()
            .0
            .iter()
            .zip(b.as_chunks::<4>().0)
            .filter(|(p, q)| (0..3).any(|k| p[k].abs_diff(q[k]) > 24))
            .count();
        assert!(visible <= 20, "{visible} pixels differ visibly between the GPU and the CPU ({layers:?})");
    }
}

/// Exercise the actual hand-over-arm model and contact-prop pass together.
#[test]
fn original_interact_contacts_match_cpu() {
    let Ok(mut gpu)=Gpu::new(true) else { eprintln!("no GPU here, skipping original interact comparison");return; };
    let path=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shots/interact.py");
    let (_,mut shot)=mavian_py::load_file(&path).unwrap().remove(0);shot.size=(320,180);
    let tl=shot.timeline();let cpu=mavian_cpu::Cpu::new();
    for n in [69,70,100,139,140,145] {
        let f=shot.frame(&tl,n);let layers=Layers{paint:true,lines:true,ids:false};
        let a=gpu.render(&shot,&f,layers);let b=cpu.render(&shot,&f,layers);
        let different=a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0).filter(|(p,q)|(0..3).any(|k|p[k].abs_diff(q[k])>24)).count();
        assert!(different<=20,"interact frame {n}: {different} visibly different pixels");
    }
}
#[test]
fn gpu_effect_export_uses_resolved_depth() {
    let Ok(mut gpu)=Gpu::new(true) else { eprintln!("no GPU here, skipping effect export");return; };
    let mut shot=shot();shot.size=(96,96);
    shot.fires.push(mavian_scene::Fire::new(mavian_core::Vec3::ZERO));
    let tl=shot.timeline();let f=shot.frame(&tl,0);let layers=Layers{paint:true,lines:true,ids:false};
    let plain=gpu.render(&shot,&f,layers);
    let (px,svg)=gpu.render_with_effects(&shot,&f,layers);
    assert_eq!(plain,px);assert!(svg.contains("clip-path"));
    let (_,again)=gpu.render_with_effects(&shot,&f,layers);assert_eq!(svg,again);
}
