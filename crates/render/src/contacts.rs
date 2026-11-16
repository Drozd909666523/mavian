//! Optional contact diagnostics drawn over the actual rendered pose.
use mavian_scene::{ContactStatus, FrameState, Shot};
use std::fmt::Write;

pub fn contact_svg(shot: &Shot, f: &FrameState) -> String {
    let mut out = format!(r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">"#, shot.size.0, shot.size.1);
    for (actor, r) in shot.contact_reports(f.w) {
        let (Some(a), Some(t)) = (f.cam.project(r.actual), f.cam.project(r.target)) else { continue };
        let color = match r.status {
            ContactStatus::Satisfied => "#008860",
            ContactStatus::Blending => "#307ab5",
            _ => "#d53535",
        };
        let name = r.name.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");
        let _ = write!(
            out,
            r#"<g fill="none" stroke="{color}" stroke-width="1.5"><path d="M{} {}L{} {}M{} {}h10M{} {}v10"/><circle cx="{}" cy="{}" r="4"/></g><text x="{}" y="{}" fill="{color}" font-size="12" font-family="DejaVu Sans, sans-serif">{actor}:{name} {:?} {:.1}mm</text>"#,
            a.x,
            a.y,
            t.x,
            t.y,
            t.x - 5.0,
            t.y,
            t.x,
            t.y - 5.0,
            a.x,
            a.y,
            t.x + 7.0,
            t.y - 7.0,
            r.status,
            1000.0 * r.error
        );
    }
    out.push_str("</svg>");
    out
}
