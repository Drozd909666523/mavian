//! Frame drawing.
//!
//! For now this holds only [`preview`] — a reference CPU preview: sticks and
//! a circle head in SVG, as in the MVP, to see the motion and verify the port.
//! It is not the final renderer. The final one is a GPU renderer with buffers (depth,
//! normal, part ID) and a line stage on top of them (05-architecture, §7).
//!
//! [`frame_key`] — fingerprint of everything a frame's picture depends on: by it
//! a frame is taken from the on-disk cache if its inputs have not changed.

pub mod preview;

pub use preview::{overlay_with_fires, svg_with_fires, frame_key, overlay_layer, labels_svg, svg, svg_layer};

pub mod contacts;
pub use contacts::contact_svg;
