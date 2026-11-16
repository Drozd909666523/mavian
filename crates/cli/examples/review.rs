//! Render selected original-shot frames, without changing its choreography.
//! cargo run -p mavian-cli --example review -- shots/interact.py base out/review 60 65 70
use mavian_gpu::Layers;
use std::{fs::File, io::BufWriter, path::Path};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let mut shots = mavian_py::load_file(Path::new(&args[1])).unwrap();
    let (_, shot) = shots.iter_mut().find(|(name, _)| name == &args[2]).unwrap();
    let dir = Path::new(&args[3]);
    std::fs::create_dir_all(dir).unwrap();
    let tl = shot.timeline();
    eprintln!("actions: {:?}", shot.actions);
    let cpu = mavian_cpu::Cpu::new();
    for a in &args[4..] {
        let n: usize = a.parse().unwrap();
        let f = shot.frame(&tl, n);
        let px = cpu.render(shot, &f, Layers { paint: true, lines: true, ids: false });
        let file = BufWriter::new(File::create(dir.join(format!("{n:04}.png"))).unwrap());
        let mut enc = png::Encoder::new(file, shot.size.0, shot.size.1);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header().unwrap().write_image_data(&px).unwrap();
        eprintln!("frame {n}, world {:.3}", f.w);
    }
}
