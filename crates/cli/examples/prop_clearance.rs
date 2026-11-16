//! Sample the original shot's limb/prop clearance independently of the camera.
use std::path::Path;
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let path = Path::new(args.get(1).map(String::as_str).unwrap_or("shots/interact.py"));
    let (_, s) = mavian_py::load_file(path).unwrap().remove(0);
    for (start, end) in [(0.0, 2.5), (2.5, 3.3), (3.3, 5.2), (5.2, 6.2), (6.2, 6.9), (6.9, 9.7), (9.7, 10.7)] {
        let mut worst = std::collections::BTreeMap::<String, (f64, f64, String)>::new();
        for n in 0..=((end - start) * 120.0) as usize {
            let w = start + n as f64 / 120.0;
            for r in s.obstacle_reports(w) {
                let name = s.props[r.prop].name.clone();
                let limb = s.actors[r.actor].model.rig.joints()[r.bone[1]].name.clone();
                let old = worst.entry(name).or_insert((0.0, w, limb.clone()));
                if r.clearance < old.0 {
                    *old = (r.clearance, w, limb);
                }
            }
        }
        println!("{start:.2}..{end:.2}: {worst:?}");
    }
}
