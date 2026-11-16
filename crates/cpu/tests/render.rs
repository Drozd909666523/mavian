//! The CPU reference draws the same frame whatever the thread count, and its
//! line stage works: a mannequin frame has ink lines on paper.

use mavian_core::{Ease, Keys};
use mavian_cpu::Cpu;
use mavian_frame::Layers;
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
fn same_pixels_whatever_the_threads() {
    let shot = shot();
    shot.validate().unwrap();
    let tl = shot.timeline();
    let f = shot.frame(&tl, 0);
    let layers = Layers { paint: true, lines: true, ids: false };
    let one = Cpu::with_threads(1).render(&shot, &f, layers);
    let many = Cpu::with_threads(7).render(&shot, &f, layers);
    assert_eq!(one, many, "the picture must not depend on the thread count");
    assert_eq!(one.len(), 320 * 240 * 4);
    let dark = one.as_chunks::<4>().0.iter().filter(|p| p[0] < 60 && p[1] < 60 && p[2] < 60).count();
    assert!(dark > 200, "the outline must be drawn ({dark} dark pixels)");
}

/// The CPU frame is the same bytes on every machine: basic f32 arithmetic is
/// fixed by IEEE 754, the rest comes from the portable `mavian_core::m`. A
/// run on another OS or CPU must give this fingerprint; it changes only with
/// an intentional change of the drawing (bump `mavian_cpu::VERSION` then).
#[test]
fn frozen_fingerprint() {
    let shot = shot();
    let tl = shot.timeline();
    let f = shot.frame(&tl, 0);
    let px = Cpu::with_threads(3).render(&shot, &f, Layers { paint: true, lines: true, ids: false });
    let mut h = mavian_core::StableHasher::new();
    h.bytes(&px);
    assert_eq!(h.finish().to_string(), "a48abcfa92a78ddd4e661b6a81692f48");
}

/// The lines as vectors: the frame's outlines and face become SVG paths,
/// the same text every time.
#[test]
fn lines_as_vectors() {
    let shot = shot();
    let tl = shot.timeline();
    let f = shot.frame(&tl, 0);
    let a = Cpu::with_threads(2).lines_svg(&shot, &f, false);
    let b = Cpu::with_threads(5).lines_svg(&shot, &f, false);
    assert_eq!(a, b, "vector lines must not depend on the thread count");
    assert!(a.starts_with("<svg") && a.trim_end().ends_with("</svg>"));
    let paths = a.matches("<path").count();
    // head ring, eyes, mouth and the body's outline runs at least
    assert!(paths >= 6, "too few paths ({paths})");
    assert!(!a.contains("NaN") && !a.contains("inf"), "bad numbers in the SVG");
}
