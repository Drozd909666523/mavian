//! Authoring permissions must affect the actual GPU frame as well as diagnostics.
use mavian_gpu::{Gpu, Layers};
use std::path::Path;

#[test]
fn default_assisted_and_locked_contact_frames_match_cpu() {
    let Ok(mut gpu) = Gpu::new(true) else {
        eprintln!("no GPU here, skipping contact permission rendering");
        return;
    };
    let file = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shots/contacts.py");
    let variants = mavian_py::load_file(&file).unwrap();
    let cpu = mavian_cpu::Cpu::new();
    let layers = Layers { paint: true, lines: true, ids: false };
    let mut rendered = Vec::new();
    for (name, mut shot) in variants {
        shot.size = (320, 240);
        let timeline = shot.timeline();
        let frame = shot.frame(&timeline, 60);
        let actual = gpu.render(&shot, &frame, layers);
        let reference = cpu.render(&shot, &frame, layers);
        let different =
            actual.as_chunks::<4>().0.iter().zip(reference.as_chunks::<4>().0).filter(|(a, b)| (0..3).any(|k| a[k].abs_diff(b[k]) > 24)).count();
        assert!(different <= 20, "contact permissions {name}: {different} visibly different pixels");
        rendered.push((name, actual));
    }
    assert_eq!(rendered.len(), 3);
    for i in 0..rendered.len() {
        for j in i + 1..rendered.len() {
            assert_ne!(rendered[i].1, rendered[j].1, "permissions {} and {} rendered the same pose", rendered[i].0, rendered[j].0);
        }
    }
}

#[test]
fn preparatory_steps_match_cpu_before_during_and_after_the_step() {
    let Ok(mut gpu) = Gpu::new(true) else {
        eprintln!("no GPU here, skipping preparatory step rendering");
        return;
    };
    let file = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shots/approach.py");
    let (_, mut shot) = mavian_py::load_file(&file).unwrap().remove(0);
    shot.size = (480, 270);
    let timeline = shot.timeline();
    let cpu = mavian_cpu::Cpu::new();
    for n in [8, 24, 35, 48] {
        let frame = shot.frame(&timeline, n);
        let layers = Layers { paint: true, lines: true, ids: false };
        let actual = gpu.render(&shot, &frame, layers);
        let reference = cpu.render(&shot, &frame, layers);
        let different =
            actual.as_chunks::<4>().0.iter().zip(reference.as_chunks::<4>().0).filter(|(a, b)| (0..3).any(|k| a[k].abs_diff(b[k]) > 24)).count();
        assert!(different <= 20, "preparatory step frame {n}: {different} visibly different pixels");
    }
}
