//! mavian — building shots.
//!
//!     mavian list                        built-in shots
//!     mavian render shots/three_cams.py  frames → PNG → video and a contact sheet
//!     mavian digest shots/three_cams.py  only the fingerprint of all frames
//!     mavian check shots/three_cams.py   repeatability and built-in checks
//!     mavian dump shots/three_cams.py    time, cameras and poses per frame
//!
//! A shot is a shot file (`shots/x.py`) or a built-in Rust shot (`three_cams`).
//! Options: `-v <variant>`, `--all`, `--no-video`,
//! `--layers paint,lines,sketch` (default `paint,lines` — the GPU renderer;
//! `sketch` — the SVG sketch, alone or over the other layers),
//! `--gpu low-power|high-performance|cpu` (default low-power: the integrated
//! GPU, and the discrete one is not even woken up; `cpu` or `--cpu` — the CPU
//! reference renderer, the same stages without a GPU).
//!
//! `--vector` also writes the lines as SVG (out/<shot>/lines/).
//! `--contacts` overlays contact diagnostics and writes contacts.csv and grabs.csv.
//! `mavian contacts shots/x.py` prints the same contact CSV without rendering.
//! `mavian grabs shots/x.py` prints persistent acquisition decisions, including misses.
//!
//!     mavian compare shots/three_cams.py  every frame on the GPU and the CPU:
//!                                         how far apart they are

use mavian_check::{Status, camera, feet, seed_independence};
use mavian_core::{Fingerprint, StableHasher};
use mavian_cpu::Cpu;
use mavian_gpu::{Gpu, Layers, Renderer};
use mavian_graph::Store;
use mavian_render::{frame_key, overlay_with_fires, svg, svg_with_fires};
use mavian_scene::{FrameState, Shot};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, mpsc};
use std::time::Instant;

const USAGE: &str = "usage: mavian list | render|digest|check|dump|compare|contacts|steps|grabs|placements <shot> [-v variant] [--all] [--no-video] \
                     [--layers paint,lines,sketch] [--gpu low-power|high-performance|cpu] [--cpu] [--vector] [--contacts]\n\
                     <shot> is a shot file (shots/x.py) or a built-in shot";

fn out_root() -> PathBuf {
    let root = std::env::var_os("MAVIAN_OUT").map(PathBuf::from).unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../out"));
    let _ = std::fs::create_dir_all(&root);
    root.canonicalize().unwrap_or(root)
}

/// Compute f(0..n) on all cores; the result is ordered by index.
fn par_map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let next = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut parts: Vec<(usize, T)> = std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut got = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break got;
                        }
                        got.push((i, f(i)));
                    }
                })
            })
            .collect();
        workers.into_iter().flat_map(|w| w.join().expect("worker thread panicked")).collect()
    });
    parts.sort_by_key(|(i, _)| *i);
    parts.into_iter().map(|(_, v)| v).collect()
}

/// Which shots to work on and how.
#[derive(Clone)]
struct Opts {
    name: String,
    variant: Option<String>,
    all: bool,
    video: bool,
    layers: Layers,
    sketch: bool,
    layers_arg: String,
    gpu_arg: String,
    /// also write the lines as vectors: out/<shot>/lines/NNNN.svg
    vector: bool,
    contacts: bool,
}

impl Opts {
    fn gpu(&self) -> bool {
        self.layers.paint || self.layers.lines || self.layers.ids
    }
}

/// Load the selected shots: (variant name, shot). "base" is the file's `shot`.
fn load(o: &Opts) -> Result<Vec<(String, Shot)>, String> {
    let all = if o.name.ends_with(".py") {
        mavian_py::load_file(Path::new(&o.name))?
    } else {
        let shot = mavian_shots::by_name(&o.name).ok_or_else(|| {
            let names: Vec<_> = mavian_shots::all().into_iter().map(|(n, _)| n).collect();
            format!("no shot '{}'; available: {} or a shot file .py", o.name, names.join(", "))
        })?;
        shot.validate().map_err(|e| format!("{}: {e}", o.name))?;
        vec![("base".to_string(), shot)]
    };
    if o.all {
        return Ok(all);
    }
    let want = o.variant.as_deref().unwrap_or("base");
    let names: Vec<String> = all.iter().map(|(n, _)| n.clone()).collect();
    all.into_iter().find(|(n, _)| n == want).map(|v| vec![v]).ok_or_else(|| format!("no variant '{want}'; available: {}", names.join(", ")))
}

fn run(cmd: &mut Command) -> Result<(), String> {
    let out = cmd.stdout(Stdio::null()).stderr(Stdio::piped()).output().map_err(|e| format!("{cmd:?}: {e}"))?;
    if !out.status.success() {
        return Err(format!("{cmd:?}: {}", String::from_utf8_lossy(&out.stderr)));
    }
    Ok(())
}

fn write_png(path: &Path, size: (u32, u32), rgba: &[u8]) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), size.0, size.1);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut w = enc.write_header().map_err(|e| e.to_string())?;
    w.write_image_data(rgba).map_err(|e| e.to_string())
}

fn read_png(path: &Path) -> Result<Vec<u8>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut dec = png::Decoder::new(std::io::BufReader::new(file));
    dec.set_transformations(png::Transformations::EXPAND);
    let mut r = dec.read_info().map_err(|e| e.to_string())?;
    let mut buf = vec![0; r.output_buffer_size().ok_or("png too large")?];
    let info = r.next_frame(&mut buf).map_err(|e| e.to_string())?;
    buf.truncate(info.buffer_size());
    if info.color_type != png::ColorType::Rgba {
        return Err(format!("{}: expected RGBA", path.display()));
    }
    Ok(buf)
}

/// Lay the SVG sketch of frame `f` over pixels (`rgba`), through rsvg.
fn overlay_sketch(shot: &Shot, f: &FrameState, fires: &str, rgba: &mut [u8], tmp: &Path) -> Result<(), String> {
    overlay_svg(&svg_with_fires(shot, f, false, fires), rgba, tmp)
}

/// Lay an SVG page (transparent background) over a frame.
fn overlay_svg(svg: &str, rgba: &mut [u8], tmp: &Path) -> Result<(), String> {
    let svg_path = tmp.with_extension("svg");
    let png_path = tmp.with_extension("over.png");
    std::fs::write(&svg_path, svg).map_err(|e| e.to_string())?;
    run(Command::new("rsvg-convert").arg(&svg_path).arg("-o").arg(&png_path))?;
    let top = read_png(&png_path)?;
    for (d, s) in rgba.as_chunks_mut::<4>().0.iter_mut().zip(top.as_chunks::<4>().0) {
        let a = s[3] as u32;
        for k in 0..3 {
            d[k] = ((s[k] as u32 * a + d[k] as u32 * (255 - a) + 127) / 255) as u8;
        }
    }
    let _ = std::fs::remove_file(&svg_path);
    let _ = std::fs::remove_file(&png_path);
    Ok(())
}

/// Fingerprint of everything frame `f`'s picture depends on, with these
/// layers; `device` — what draws them (renderer version and adapter).
fn key(shot: &Shot, f: &FrameState, o: &Opts, device: &str) -> Fingerprint {
    let mut h = StableHasher::new();
    h.add(&frame_key(shot, f).to_string()).add(&o.layers_arg).add(&o.contacts);
    if o.gpu() {
        h.add(device);
    }
    if o.contacts {
        h.add(&format!("{:?}", shot.contact_reports(f.w)));
        h.add(&format!("{:?}", shot.placement_reports));
    }
    h.finish()
}

type Dev = Option<Box<dyn Renderer>>;

/// Draw frames with the GPU layers: the GPU draws in this thread, PNG
/// encoding and the sketch overlay go to worker threads.
/// `sink(n, rgba)` receives each frame's pixels.
fn gpu_frames(
    shot: &Shot,
    frames: &[usize],
    o: &Opts,
    gpu: &mut dyn Renderer,
    tmp: &Path,
    sink: &(dyn Fn(usize, Vec<u8>) -> Result<(), String> + Sync),
) -> Result<(), String> {
    let tl = shot.timeline();
    let (tx, rx) = mpsc::sync_channel::<(usize, mavian_frame::RenderedFrame)>(8);
    let rx = Mutex::new(rx);
    let failed: Mutex<Option<String>> = Mutex::new(None);
    std::thread::scope(|s| {
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let got = rx.lock().expect("queue").recv();
                    let Ok((n, frame)) = got else { break };
                    let fires = if shot.fires.is_empty() { String::new() } else { frame.fires_svg(shot, &shot.frame(&tl, n)) };
                    let mut px = frame.rgba;
                    let done = (|| {
                        if o.sketch {
                            let f = shot.frame(&tl, n);
                            // the sketch carries its own labels
                            overlay_sketch(shot, &f, &fires, &mut px, &tmp.join(format!("{n:04}")))?;
                        } else if !shot.labels.is_empty() || !shot.fires.is_empty() {
                            let f = shot.frame(&tl, n);
                            if let Some(svg) = overlay_with_fires(shot, &f, &fires) {
                                overlay_svg(&svg, &mut px, &tmp.join(format!("{n:04}")))?;
                            }
                        }
                        if o.contacts {
                            let f = shot.frame(&tl, n);
                            overlay_svg(&mavian_render::contact_svg(shot, &f), &mut px, &tmp.join(format!("{n:04}-contacts")))?;
                        }
                        sink(n, px)
                    })();
                    if let Err(e) = done {
                        failed.lock().expect("errors").get_or_insert(e);
                    }
                }
            });
        }
        for &n in frames {
            let f = shot.frame(&tl, n);
            let frame = gpu.render_with_depth(shot, &f, o.layers);
            if tx.send((n, frame)).is_err() {
                break;
            }
        }
        drop(tx);
    });
    match failed.into_inner().expect("errors") {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Fingerprint of all frames: of the pixels for the GPU layers, of the SVG
/// text for the sketch alone.
fn digest(shot: &Shot, o: &Opts, gpu: &mut Dev) -> Result<String, String> {
    let tl = shot.timeline();
    let n = tl.len();
    let mut h = StableHasher::new();
    if !o.gpu() {
        for (_, f) in par_map(n, |k| (k, svg(shot, &shot.frame(&tl, k)))) {
            h.add(f.as_str());
        }
        return Ok(h.finish().to_string());
    }
    let gpu = open_gpu(gpu, o)?;
    let tmp = std::env::temp_dir().join(format!("mavian-digest-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;
    let frames: Vec<usize> = (0..n).collect();
    let all: Mutex<Vec<(usize, Vec<u8>)>> = Mutex::new(Vec::new());
    gpu_frames(shot, &frames, o, gpu, &tmp, &|k, px| {
        all.lock().expect("frames").push((k, px));
        Ok(())
    })?;
    let _ = std::fs::remove_dir_all(&tmp);
    let mut all = all.into_inner().expect("frames");
    all.sort_by_key(|(k, _)| *k);
    for (_, px) in &all {
        h.bytes(px);
    }
    Ok(h.finish().to_string())
}

fn open_gpu<'a>(gpu: &'a mut Dev, o: &Opts) -> Result<&'a mut dyn Renderer, String> {
    if gpu.is_none() {
        *gpu = Some(if o.gpu_arg == "cpu" {
            Box::new(Cpu::new())
        } else {
            let low = o.gpu_arg != "high-performance";
            Box::new(Gpu::new(low).map_err(|e| format!("{e}\n(use --cpu or --layers sketch to draw without a GPU)"))?)
        });
    }
    Ok(gpu.as_mut().expect("opened above").as_mut())
}

fn render(shot: &Shot, o: &Opts, gpu: &mut Dev) -> Result<(), String> {
    let t0 = Instant::now();
    let name = shot.name.as_str();
    let tl = shot.timeline();
    let root = out_root();
    let out = root.join(name);
    let (svg_dir, png_dir) = (out.join("svg"), out.join("png"));
    if out.exists() {
        std::fs::remove_dir_all(&out).map_err(|e| e.to_string())?;
    }
    for d in [&svg_dir, &png_dir] {
        std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    let store = Store::open(root.join(".cache")).map_err(|e| e.to_string())?;
    let adapter = if o.gpu() { open_gpu(gpu, o)?.device() } else { String::new() };
    let link = |n: usize, cached: &Path| -> Result<(), String> {
        let png = png_dir.join(format!("{n:04}.png"));
        std::fs::hard_link(cached, &png).or_else(|_| std::fs::copy(cached, &png).map(|_| ())).map_err(|e| e.to_string())
    };
    // what is in the cache already, and what must be drawn
    let keys: Vec<Fingerprint> = par_map(tl.len(), |n| key(shot, &shot.frame(&tl, n), o, &adapter));
    let mut todo = Vec::new();
    for (n, k) in keys.iter().enumerate() {
        match store.find(*k, "png") {
            Some(p) => link(n, &p)?,
            None => todo.push(n),
        }
    }
    if o.gpu() {
        let store_frame = |n: usize, px: Vec<u8>| -> Result<(), String> {
            let tmp = png_dir.join(format!("{n:04}.new.png"));
            write_png(&tmp, shot.size, &px)?;
            let p = store.put_file(keys[n], "png", &tmp).map_err(|e| e.to_string())?;
            std::fs::remove_file(&tmp).map_err(|e| e.to_string())?;
            link(n, &p)
        };
        let g = open_gpu(gpu, o)?;
        gpu_frames(shot, &todo, o, g, &svg_dir, &store_frame)?;
    } else {
        let results = par_map(todo.len(), |i| -> Result<(), String> {
            let n = todo[i];
            let f = shot.frame(&tl, n);
            let svg_path = svg_dir.join(format!("{n:04}.svg"));
            let picture = svg(shot, &f);
            let picture = if o.contacts {
                let overlay = mavian_render::contact_svg(shot, &f);
                picture.replacen("</svg>", &format!("{overlay}</svg>"), 1)
            } else {
                picture
            };
            std::fs::write(&svg_path, picture).map_err(|e| e.to_string())?;
            let tmp = png_dir.join(format!("{n:04}.new.png"));
            run(Command::new("rsvg-convert").arg(&svg_path).arg("-o").arg(&tmp))?;
            let p = store.put_file(keys[n], "png", &tmp).map_err(|e| e.to_string())?;
            std::fs::remove_file(&tmp).map_err(|e| e.to_string())?;
            link(n, &p)
        });
        results.into_iter().collect::<Result<Vec<_>, _>>()?;
    }
    if o.contacts {
        std::fs::write(out.join("contacts.csv"), contact_csv(shot)).map_err(|e| e.to_string())?;
        std::fs::write(out.join("grabs.csv"), grab_csv(shot)).map_err(|e| e.to_string())?;
    }
    if !shot.placement_reports.is_empty() {
        std::fs::write(out.join("placements.csv"), placement_csv(shot)).map_err(|e| e.to_string())?;
    }
    if o.vector {
        // the line stage as vectors, from the CPU reference's buffers
        let dir = out.join("lines");
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let cpu = Cpu::new();
        for n in 0..tl.len() {
            let f = shot.frame(&tl, n);
            std::fs::write(dir.join(format!("{n:04}.svg")), cpu.lines_svg(shot, &f, false)).map_err(|e| e.to_string())?;
        }
    }
    if o.video {
        let src = png_dir.join("%04d.png");
        let fps = shot.fps.to_string();
        run(Command::new("ffmpeg")
            .args(["-y", "-framerate", &fps, "-i"])
            .arg(&src)
            .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16"])
            .arg(out.join(format!("{name}.mp4"))))?;
        // contact sheet with frames at equal intervals — see the shot at a glance
        let every = tl.len().div_ceil(24).max(1);
        run(Command::new("ffmpeg")
            .args(["-y", "-framerate", &fps, "-i"])
            .arg(&src)
            .args(["-vf", &format!("select='not(mod(n\\,{every}))',scale=320:-1,tile=6x4"), "-frames:v", "1", "-update", "1"])
            .arg(out.join("sheet.png")))?;
    }
    println!(
        "{name}: {} frames, {:.2} s of shot, {:.2} s of world, layers {} — drawn {}, from cache {}, {:.1} s",
        tl.len(),
        tl.len() as f64 / shot.fps as f64,
        shot.length,
        o.layers_arg,
        todo.len(),
        tl.len() - todo.len(),
        t0.elapsed().as_secs_f64()
    );
    if !adapter.is_empty() {
        println!("  drawn by: {adapter}");
    }
    if o.video {
        println!("  {}", out.join(format!("{name}.mp4")).display());
    }
    Ok(())
}

fn dump(shot: &Shot) {
    let tl = shot.timeline();
    let f = |v: mavian_core::Vec3| format!("{:.5},{:.5},{:.5}", v.x, v.y, v.z);
    for n in 0..tl.len() {
        let (_, c) = shot.camera_for(&tl, n);
        println!("{n} w={:.6} drawn={}", tl.world[n], shot.drawn_frame(&tl, n));
        let fish = if c.fisheye { "True" } else { "False" };
        println!(" cam {} {} {} {} {:.4} {fish}", f(c.pos), f(c.right), f(c.up), f(c.fwd), c.f);
        println!(" pose {}", shot.pose_signature(&tl, n));
    }
}

fn separate_digest(o: &Opts, variant: &str) -> Result<String, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let out = Command::new(exe)
        .args(["digest", &o.name, "-v", variant, "--layers", &o.layers_arg, "--gpu", &o.gpu_arg])
        .output()
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into());
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn contact_csv(shot: &Shot) -> String {
    use std::fmt::Write;
    let mut csv =
        String::from("frame,world,actor,name,priority,status,weight,error_mm,reach_error_mm,target_x,target_y,target_z,actual_x,actual_y,actual_z\n");
    let tl = shot.timeline();
    let mut last = None;
    for n in 0..tl.len() {
        let drawing = shot.drawn_frame(&tl, n);
        if last == Some(drawing) {
            continue;
        }
        last = Some(drawing);
        let w = tl.world[drawing];
        for (actor, r) in shot.contact_reports(w) {
            let name = r.name.replace('"', "\"\"");
            let _ = writeln!(
                csv,
                "{n},{w:.6},{actor},\"{name}\",{:?},{:?},{:.6},{:.6},{:.6},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9}",
                r.priority,
                r.status,
                r.weight,
                r.error * 1000.0,
                r.reach_error * 1000.0,
                r.target.x,
                r.target.y,
                r.target.z,
                r.actual.x,
                r.actual.y,
                r.actual.z
            );
        }
    }
    csv
}

fn contact_dump(shot: &Shot) -> Result<(), String> {
    use std::io::Write;
    std::io::stdout().lock().write_all(contact_csv(shot).as_bytes()).map_err(|e| e.to_string())
}

fn grab_csv(shot: &Shot) -> String {
    use std::fmt::Write;
    let mut csv = String::from("actor,hand,prop,world,status,error_mm,tolerance_mm,target_x,target_y,target_z,actual_x,actual_y,actual_z\n");
    for r in &shot.grab_reports {
        let _ = writeln!(csv, "{},{},{},{:.9},{:?},{:.6},{:.6},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9}",
            r.actor, r.hand, r.prop, r.at, r.status, r.error * 1000.0, r.tolerance * 1000.0,
            r.target.x, r.target.y, r.target.z, r.actual.x, r.actual.y, r.actual.z);
    }
    csv
}

fn placement_csv(shot: &Shot) -> String {
    use std::fmt::Write;
    let mut csv = String::from(
        "actor,hand,prop,support,world,status,error_mm,tolerance_mm,support_gap_mm,edge_margin_mm,target_x,target_y,target_z,actual_x,actual_y,actual_z\n",
    );
    for r in &shot.placement_reports {
        let _ = writeln!(
            csv,
            "{},{},{},{},{:.9},{:?},{:.6},{:.6},{:.6},{:.6},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9}",
            r.actor,
            r.hand,
            r.prop,
            r.support,
            r.at,
            r.status,
            r.error * 1000.0,
            r.tolerance * 1000.0,
            r.support_gap * 1000.0,
            r.edge_margin * 1000.0,
            r.target.x,
            r.target.y,
            r.target.z,
            r.actual.x,
            r.actual.y,
            r.actual.z
        );
    }
    csv
}

fn step_dump(shot: &Shot) {
    println!("actor,prop,from,to,status,max_distance,shift_x,shift_z,error_before_mm,error_after_mm");
    for (actor, a) in shot.actors.iter().enumerate() {
        for r in &a.step_reports {
            println!("{actor},{},{:.6},{:.6},{:?},{:.6},{:.6},{:.6},{:.6},{:.6}", r.prop, r.from, r.to, r.status,
                r.max_distance, r.shift.x, r.shift.z, r.error_before * 1000., r.error_after * 1000.);
        }
    }
}

fn check(o: &Opts, variant: &str, shot: &Shot, gpu: &mut Dev) -> Result<bool, String> {
    println!("{} (layers {}):", shot.name, o.layers_arg);
    let tl = shot.timeline();
    let mut ok = true;
    let (a, b) = (separate_digest(o, variant)?, separate_digest(o, variant)?);
    ok &= a == b;
    println!("[{}] two independent runs: {}… and {}…", if a == b { "ok" } else { "ERROR" }, &a[..16], &b[..16]);
    let here = digest(shot, o, gpu)?;
    ok &= here == a;
    println!("[{}] the same in this process: {}…", if here == a { "ok" } else { "ERROR" }, &here[..16]);
    let one = Opts { variant: Some(variant.into()), all: false, ..o.clone() };
    let mut other = load(&one)?.remove(0).1;
    other.seed += 1;
    let mut reports = vec![feet(shot, &tl), camera(shot, &tl), mavian_check::contacts(shot, &tl), mavian_check::grabs(shot), mavian_check::placements(shot)];
    if !shot.actors.is_empty() {
        reports.push(seed_independence(shot, &other));
    }
    for r in &reports {
        println!("{r}");
        ok &= r.status != Status::Fail;
    }
    Ok(ok)
}

/// Every frame on the GPU and on the CPU reference: how far apart they are.
/// The worst frame goes to out/<shot>/compare/ (GPU, CPU, difference ×8).
fn compare(shot: &Shot, o: &Opts) -> Result<bool, String> {
    if !o.gpu() {
        return Err("compare draws the GPU layers (paint, lines, ids)".into());
    }
    let low = o.gpu_arg != "high-performance";
    let mut gpu = Gpu::new(low)?;
    let cpu = Cpu::new();
    let tl = shot.timeline();
    let (w, h) = shot.size;
    let total = (w * h) as usize;
    // a pixel differs visibly if a channel is off by more than this
    const VISIBLE: u8 = 24;
    let (mut worst, mut worst_n, mut worst_px) = (0usize, 0usize, (Vec::new(), Vec::new()));
    let (mut sum_visible, mut max_diff, mut t_gpu, mut t_cpu) = (0usize, 0u8, 0.0, 0.0);
    for n in 0..tl.len() {
        let f = shot.frame(&tl, n);
        let t = Instant::now();
        let a = gpu.render(shot, &f, o.layers);
        t_gpu += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let b = cpu.render(shot, &f, o.layers);
        t_cpu += t.elapsed().as_secs_f64();
        let mut visible = 0;
        for (pa, pb) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0) {
            let d = (0..3).map(|k| pa[k].abs_diff(pb[k])).max().unwrap_or(0);
            max_diff = max_diff.max(d);
            visible += (d > VISIBLE) as usize;
        }
        sum_visible += visible;
        if visible > worst || worst_px.0.is_empty() {
            (worst, worst_n, worst_px) = (visible, n, (a, b));
        }
    }
    let dir = out_root().join(&shot.name).join("compare");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let (a, b) = &worst_px;
    let diff: Vec<u8> = a
        .as_chunks::<4>()
        .0
        .iter()
        .zip(b.as_chunks::<4>().0)
        .flat_map(|(pa, pb)| {
            let d = (0..3).map(|k| pa[k].abs_diff(pb[k])).max().unwrap_or(0);
            let v = 255 - d.saturating_mul(8);
            [v, v, v, 255]
        })
        .collect();
    write_png(&dir.join(format!("{worst_n:04}-gpu.png")), shot.size, a)?;
    write_png(&dir.join(format!("{worst_n:04}-cpu.png")), shot.size, b)?;
    write_png(&dir.join(format!("{worst_n:04}-diff.png")), shot.size, &diff)?;
    let frames = tl.len();
    let share = |v: usize| 100.0 * v as f64 / total as f64;
    // GPUs rasterize triangle edges on a sub-pixel grid: a few edge pixels
    // may differ, a whole region may not
    let ok = share(worst) < 0.5;
    println!("{} (layers {}): GPU {} against CPU {}", shot.name, o.layers_arg, gpu.adapter, mavian_cpu::VERSION);
    println!(
        "[{}] visibly different pixels (a channel off by > {VISIBLE}): {:.3}% on average, {:.3}% at worst (frame {worst_n}); largest difference {max_diff}",
        if ok { "ok" } else { "ERROR" },
        share(sum_visible) / frames as f64,
        share(worst),
    );
    println!("  time per frame: GPU {:.0} ms, CPU {:.0} ms", 1e3 * t_gpu / frames as f64, 1e3 * t_cpu / frames as f64);
    println!("  {}", dir.display());
    Ok(ok)
}

fn parse(args: &[String]) -> Result<(String, Opts), String> {
    let (cmd, rest) = args.split_first().ok_or(USAGE)?;
    let mut o = Opts {
        name: String::new(),
        variant: None,
        all: false,
        video: true,
        layers: Layers { paint: true, lines: true, ids: false },
        sketch: false,
        layers_arg: "paint,lines".into(),
        gpu_arg: "low-power".into(),
        vector: false,
        contacts: false,
    };
    let mut it = rest.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-v" | "--variant" => o.variant = Some(it.next().ok_or(USAGE)?.clone()),
            "--all" => o.all = true,
            "--no-video" => o.video = false,
            "--layers" => o.layers_arg = it.next().ok_or(USAGE)?.clone(),
            "--gpu" => o.gpu_arg = it.next().ok_or(USAGE)?.clone(),
            "--cpu" => o.gpu_arg = "cpu".into(),
            "--vector" => o.vector = true,
            "--contacts" => o.contacts = true,
            _ if o.name.is_empty() => o.name = a.clone(),
            _ => return Err(USAGE.into()),
        }
    }
    let names: Vec<&str> = o.layers_arg.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    if let Some(bad) = names.iter().find(|n| !["paint", "lines", "sketch", "ids"].contains(n)) {
        return Err(format!("unknown layer '{bad}'; layers: paint, lines, sketch, ids (debug)"));
    }
    if names.is_empty() {
        return Err("no layers to draw".into());
    }
    o.layers = Layers { paint: names.contains(&"paint"), lines: names.contains(&"lines"), ids: names.contains(&"ids") };
    o.sketch = names.contains(&"sketch");
    // a canonical spelling, so the cache key does not depend on the order
    o.layers_arg = ["paint", "lines", "sketch", "ids"].iter().filter(|n| names.contains(n)).copied().collect::<Vec<_>>().join(",");
    if !["low-power", "high-performance", "cpu"].contains(&o.gpu_arg.as_str()) {
        return Err(format!("--gpu is low-power, high-performance or cpu, not '{}'", o.gpu_arg));
    }
    if cmd != "list" && o.name.is_empty() {
        return Err(USAGE.into());
    }
    Ok((cmd.clone(), o))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let run_all = || -> Result<bool, String> {
        let (cmd, o) = parse(&args)?;
        if cmd == "list" {
            for (n, _) in mavian_shots::all() {
                println!("{n}");
            }
            return Ok(true);
        }
        let mut gpu = None;
        let mut ok = true;
        for (variant, shot) in load(&o)? {
            match cmd.as_str() {
                "render" => render(&shot, &o, &mut gpu)?,
                "digest" => println!("{}", digest(&shot, &o, &mut gpu)?),
                "dump" => dump(&shot),
                "contacts" => contact_dump(&shot)?,
                "steps" => step_dump(&shot),
                "grabs" => print!("{}", grab_csv(&shot)),
                "placements" => print!("{}", placement_csv(&shot)),
                "check" => ok &= check(&o, &variant, &shot, &mut gpu)?,
                "compare" => ok &= compare(&shot, &o)?,
                _ => return Err(USAGE.into()),
            }
        }
        Ok(ok)
    };
    // Low power: keep the discrete GPU asleep. The Vulkan loader would load
    // its driver just to list adapters, and that wakes the card (measured on
    // a hybrid laptop: the NVIDIA card goes from suspended to active). Only if
    // the user has not configured the loader themselves.
    if !args.iter().any(|a| a == "high-performance")
        && ["VK_LOADER_DRIVERS_DISABLE", "VK_DRIVER_FILES", "VK_ICD_FILENAMES", "VK_LOADER_DRIVERS_SELECT"]
            .iter()
            .all(|v| std::env::var_os(v).is_none())
    {
        // SAFETY: at the very start of main, before any thread is spawned.
        unsafe { std::env::set_var("VK_LOADER_DRIVERS_DISABLE", "*nvidia*") };
    }
    match run_all() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
