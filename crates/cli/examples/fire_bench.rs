//! Compare repeated versus reused geometry on original fire-shot frames.
use mavian_gpu::{Layers, Renderer};
use std::{path::Path, time::Instant};
fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shots/fire.py");
    let (_, shot) = mavian_py::load_file(&root).unwrap().remove(0);
    let tl = shot.timeline();
    let cpu = mavian_cpu::Cpu::with_threads(4);
    let mut renderer: Box<dyn Renderer> = if std::env::args().any(|a| a == "--gpu") {
        Box::new(mavian_gpu::Gpu::new(true).expect("GPU benchmark requires an adapter"))
    } else {
        Box::new(mavian_cpu::Cpu::with_threads(4))
    };
    println!("renderer: {}", renderer.device());
    let layers = Layers { paint: true, lines: true, ids: false };
    let mut old = 0.0;
    let mut new = 0.0;
    // Warm the device and pose caches before measuring either path.
    for n in [0, 72, 120] {
        renderer.render(&shot, &shot.frame(&tl, n), layers);
    }
    for n in [0, 72, 120] {
        let f = shot.frame(&tl, n);
        let t = Instant::now();
        let px = renderer.render(&shot, &f, layers);
        let svg = cpu.fires_svg(&shot, &f);
        let a = t.elapsed().as_secs_f64();
        old += a;
        let t = Instant::now();
        let frame = renderer.render_with_depth(&shot, &f, layers);
        let svg2 = frame.fires_svg(&shot, &f);
        let px2 = frame.rgba;
        let b = t.elapsed().as_secs_f64();
        new += b;
        assert_eq!(px, px2);
        assert_eq!(svg, svg2);
        println!("fire frame {n}: repeated geometry {a:.3}s, reused {b:.3}s; pixels and SVG identical");
    }
    println!("total: {old:.3}s -> {new:.3}s ({:.1}% faster)", 100.0 * (old - new) / old);
}
