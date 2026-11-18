//! mm-core: plain data, rules and the project files, shared by every other crate. No GPU, no media; builds on any OS.
//!
//! - [`time`]: frame t on the constant-rate timeline <-> source presentation time.
//! - [`geometry`]: rectangles and aspect fit.
//! - [`cover`]: the compositor's cover records (`CoverRow`, the kernel's 36 floats), atlas slices, one frame's covers.
//! - [`project`]: the project model; for now the user's edits (moved covers).
//! - [`colour`]: how a decoded frame's YUV maps to RGB (matrix, range, chroma siting).
//! - [`pyjson`]: JSON values that read and write like Python's `json` (project files stay byte-identical).
//! - [`facts`]: the facts log, who's who derived from it, and the Decisions page's answers.
//! - [`ops`], [`edits`]: the edits document, its undo history, and an open project's changes (covers split, re-followed).
//! - [`coords`]: which pixel space a project's files are in. [`projects`]: where projects live, their locks.
//! - [`palettes`]: the emoji sets.
//! - [`np`]: numpy's arithmetic, bit for bit (on macOS through Accelerate, as numpy calls it).
//! - [`privacy`]: which clips are public domain (an agent may look at their pictures).
pub mod colour;
pub mod coords;
pub mod cover;
pub mod edits;
pub mod facts;
pub mod geometry;
pub mod np;
pub mod ops;
pub mod orb;
pub mod palettes;
pub mod paths;
pub mod privacy;
pub mod project;
pub mod projects;
pub mod pyjson;
pub mod time;
