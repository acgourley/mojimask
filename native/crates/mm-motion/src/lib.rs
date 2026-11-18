//! mm-motion: our own motion estimation (native/docs/porting.md, "write our own motion"; the user's call: domain-specific, not a
//! port of cv2's goodFeaturesToTrack / calcOpticalFlowPyrLK / estimateAffinePartial2D).
//!
//! Two jobs, one toolkit (block matching on image pyramids + a deterministic robust similarity fit):
//! - [`camera`]: one similarity per consecutive frame pair, frame t-1 -> t in source pixels (`mats_v2.npy`,
//!   Python's `track.camera_motion`);
//! - [`head`]: how one head box moved between two frames: the follower's flow position (`track._flow_measure`),
//!   its "lost" check (`follow.face_motion`), and whether the whole frame can be tracked (`follow.Clip.trackable`).
//!
//! [`ffi`] exposes both through a C ABI so the frozen Python can be run with them swapped in (tools/parity/motion_*).
pub mod camera;
pub mod ffi;
pub mod fit;
pub mod head;
pub mod img;
pub mod io;
pub mod matching;
