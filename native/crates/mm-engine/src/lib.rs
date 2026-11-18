//! mm-engine: the work behind the UI.
//!
//! - The scan (P5.1 and the decide step's evidence): [`scan`] runs pipeline.py's steps on a video into a project
//!   folder (frames, detection, camera motion, depth, decide), [`detect`] is onnx_core's Core (the three detectors on
//!   worker threads, the plans, the timed plan choice, the CoreML compile cache), [`jpeg`] the project's frames exactly
//!   as cv2 writes and reads them, [`evidence`] mm-scan's evidence traits on mm-analysis and mm-motion (faces, bodies,
//!   depth, motion; motion from Python as an option), [`depth`] the depth cache (depth.compute), [`check`] parity
//!   checks of that evidence against a recording. Command line: `mm-cli scan VIDEO PROJECT` and
//!   `mm-scan-tool scan|detect|decide|depth ...` (src/bin/mm-scan-tool.rs). Parity: native/tools/parity/detect_*.py,
//!   engine2_*.py.
//! - The Python backend ([`sidecar`]): today's engine as a sidecar process, for everything not ported yet.
//! - The stand-in for a scanned project: the PoC dump (`out/poc/data/<clip>`, written by
//!   native/tools/dump_reference.py from today's Python engine) with the user's edits on top ([`dump`], [`project`]),
//!   and progress reporting for long jobs ([`jobs`]).
pub mod backend;
pub mod check;
pub mod depth;
pub mod detect;
pub mod dump;
pub mod evidence;
pub mod jobs;
pub mod native;
/// moved to mm-vision (re-exported here for the scan's callers)
pub use mm_vision::jpeg;
pub mod project;
pub mod scan;
pub mod sidecar;

pub use project::Project;
