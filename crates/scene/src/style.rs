//! Drawing style of a shot.

#[derive(Clone, Debug)]
pub struct Style {
    pub paper: String,
    pub ink: String,
    pub ground: String,
    /// body line width in world units
    pub line: f64,
    /// how strongly width depends on depth (1 — as in life)
    pub depth_gamma: f64,
    /// how width grows in a close-up: 1 — with the figure, 0 — the same pencil
    /// at any shot size
    pub zoom: f64,
    /// halo width, px
    pub halo: f64,
    /// line "boil" amplitude, px (0 — off)
    pub boil: f64,
    /// boil "wave" size, px
    pub boil_scale: f64,
    /// pressure unevenness (fraction of width)
    pub pressure: f64,
}

impl Default for Style {
    fn default() -> Self {
        Style {
            paper: "#f5f0e6".into(),
            ink: "#1c1b1f".into(),
            ground: "#cbc3b3".into(),
            line: 0.040,
            depth_gamma: 3.0,
            zoom: 0.5,
            halo: 3.0,
            boil: 1.5,
            boil_scale: 48.0,
            pressure: 0.10,
        }
    }
}
