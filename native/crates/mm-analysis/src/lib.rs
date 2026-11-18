//! mm-analysis: the per-detection measurements the scan and the drawing use, ported from mojimask/:
//!
//! - [`reid`]: RetinaFace landmarks on head crops, the similarity alignment (ALIGN_DST), SFace embeddings (reid.py)
//! - [`faces`]: one face measurement per detection: score, yaw, eyes, px, embedding (scan/faces.py, per detection)
//! - [`appearance`]: OSNet body crops and embeddings (appearance.py)
//! - [`depth`]: Depth Anything, the depth cache's numbers, Depth.near/gap, face_at, Occlusion (depth.py)
//! - [`pose_util`], [`head_pose`]: head geometry from COCO-17 skeletons, roll/yaw per cover (pose_util.py,
//!   head_pose.py's angles)
//! - [`models`]: the sessions (mm-ml), copies per thread as reid.POOL
//! - [`np`]: numpy's summation, median and percentile where results depend on them (mm-core's, shared with mm-scan)
//!
//! Pixels go through mm-vision (bit-exact with the project's cv2). Exact against Python on the same engine (CPU EP);
//! native/tools/parity/analysis_*.py check it.
pub mod appearance;
pub mod depth;
pub mod faces;
pub mod head_pose;
pub mod models;
/// numpy's arithmetic (shared with mm-scan): [`mm_core::np`].
pub use mm_core::np;
pub mod pose_util;
pub mod reid;
