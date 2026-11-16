use mavian_scene::{Action, World};
fn main() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shots/fight.py");
    let (_, s) = mavian_py::load_file(&path).unwrap().remove(0);
    let world = World { actors: &s.actors, props: &s.props };
    for act in &s.actions {
        let Action::Strike(st) = act else { continue };
        println!("strike at {} hold {} blocked {:?} miss {}", st.at, st.hold, st.blocked, st.miss);
        for dt in [-0.06, 0., 0.03, 0.06, 0.1, 0.18] {
            let t = st.at + dt;
            let a = &s.actors[st.attacker];
            let d = &s.actors[st.target];
            let f = a.figure_ik(t, &world);
            let df = d.figure_ik(t, &world);
            let ch = mavian_scene::actions::chain(a, st.limb).unwrap();
            let r = a.reaches.iter().find(|r| r.chain == ch && r.weight.at(t) > 0.5);
            if let Some(r) = r {
                let g = r.target.at(t, &world);
                let end = f.joints[st.limb];
                let head = |actor: &mavian_scene::Actor, fig: &mavian_rig::Figure| fig.joints[actor.model.rig.joint("head").unwrap()];
                println!(
                    "dt {dt:.3} weight {:.3} reach {:.3} error {:.3} heads {:.3} goal {:?} end {:?}",
                    r.weight.at(t),
                    (g - f.joints[ch[0]]).length(),
                    (g - end).length(),
                    (head(a, &f) - head(d, &df)).length(),
                    g,
                    end
                );
            }
        }
    }
}
