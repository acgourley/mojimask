//! mm-scan: the scan's decide step (Python `mojimask/scan/`), stage by stage, reading and writing the same project
//! files. See native/ARCHITECTURE.md and the per-module docs (each keeps its Python module's rules and evidence).
//!
//! Stages, in pipeline order: [`tracklets`], [`join`] (with [`follow`]), [`faces`], [`graph`], then (newalgo)
//! attach, overlaps, blips, cover/behind, same/questions. Everything learnt from pixels comes through
//! [`evidence`]'s traits; [`np`] reproduces numpy's arithmetic and [`pyset`] Python's set order, so the output files
//! are byte-identical to Python's.
//!
//! Entry point: [`newalgo::run`] (or [`newalgo::run_logged`], which also writes proto/decide_viz.json) with an
//! [`evidence::Evidence`]. Today only [`evidence::replay`] implements it (a recording of a Python run,
//! tools/parity/scan_record.py); mm-motion (Motion: flow, face_motion, trackable, looks) and mm-analysis
//! (FaceEvidence: face_near, face; Appearance: body; Depth: near) supply the real ones.
//!
//! Not ported: the modules' research command lines and census diagnostics (tracklets/join purity, graph.truth_report,
//! questions' answer simulation, cover's census scoring, overlaps' legacy grading and contact sheet), and faces.py's
//! pickle cache (caching belongs behind FaceEvidence).
//!
//! Parity (tools/parity/scan_*.py): every stage's output and every file the step writes are identical to Python's on
//! jazz and jazz_4k (depth off and on) and on 32 jazz frames as photos; decision_bench and photo_bench give the same
//! numbers with this decide step.
pub mod attach;
pub mod behind;
pub mod blips;
pub mod cover;
pub mod evidence;
pub mod faces;
pub mod follow;
pub mod geom;
pub mod graph;
pub mod join;
pub mod newalgo;
pub mod np;
pub mod overlaps;
pub mod pose;
pub mod pyset;
pub mod questions;
pub mod same;
pub mod tracklets;
pub mod viz;
