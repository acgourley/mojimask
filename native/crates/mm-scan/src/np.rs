//! numpy's arithmetic, bit for bit, where the scan's rules read it: [`mm_core::np`] (shared with mm-analysis and
//! mm-core's edits), re-exported so the scan's modules keep calling `np::…`. Its doc comment says what numpy does for
//! each operation and how it's checked (`tools/parity/scan_numcheck.py`).
pub use mm_core::np::*;
