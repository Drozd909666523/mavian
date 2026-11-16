//! The scene over time: actors, cameras, shot.
//!
//! Everything here is a pure function of time: the figure and camera at any moment can
//! be obtained without knowing previous frames.

pub mod actions;
pub mod actor;
mod approach;
pub mod camera;
pub mod contact;
pub mod effects;
pub mod face;
pub mod ik;
pub mod moves;
pub mod obstacle;
mod posture;
pub mod props;
mod recoil;
pub mod shot;
pub mod style;

pub use actions::{Action, Onto, Strike, reach_gap};
pub use actor::{Actor, Layer, Move, Patch, PoseKey, Poses, TRAIL_FROM, TRAIL_STEPS, pk};
pub use camera::{Between, Camera, CameraRig, Follow, Orbit, Point, Projected, Scene, Shake, Watch};
pub use effects::Fire;
pub use face::{FaceShapes, face};
pub use ik::{Avoid, Reach, Target};
pub use moves::{Jump, JumpPoses, jump, walk, walk_end};
pub use props::{Motion, Prop, Shape, Wire, World, axes, cube};
pub use shot::{FrameState, Ground, Label, Shot, ShotError, own_perspective};
pub use style::Style;

pub use approach::{Approach, StepReport, StepStatus};
pub use contact::{Contact, ContactReport, ContactStatus, Priority};
