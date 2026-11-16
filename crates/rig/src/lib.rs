//! The skeleton of any character and the model outside the core.
//!
//! The core does not know which joints a character has: the model lists
//! them (see `mavian-models`). Only general rules live here: poses as angles,
//! forward kinematics, standing on the ground.

mod model;
mod rig;

pub use model::{Face, Model, ModelError, Part};
pub use rig::{Angles, Figure, Foot, Joint, Pose, PoseBuilder, Rig, RigError, Side};
