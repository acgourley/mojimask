//! mm-records: what is drawn on every frame, as FrameRecords for mm-render, computed in Rust exactly as the Python
//! helper (native/tools/sidecar/project_server.py, /api/native/records) computes them.
//!
//! - [`records`]: review_server.render_setup + the helper's `record`: the API ([`Records`]);
//! - [`resolve`]: track.resolve (dedupe, bridge, steady), draw_order, shown_pieces, emoji_for, links_for;
//! - [`lighting`]: lighting.py's AUTO estimate (cached as light_est_*.npz / film_est_*.npz, the files Python reads
//!   and writes), film, frame_grain, the user settings;
//! - [`npz`]: np.savez files, with the pickled probe dict.
//!
//! Head pose comes from mm_analysis::head_pose::track_angles, depth cuts from mm_analysis::depth::Occlusion (its
//! maps decoded through mm-media's grey::MovieLuma), frames from mm-vision's JPEG codec (bit-identical to cv2.imread).
//!
//! ## API (for mm-app)
//! ```ignore
//! use mm_records::{Records, records::{Ask, Options}};
//! let mut r = Records::open(project_dir, Options::default())?;      // starts the lighting estimate in the background
//! let ask = Ask { style: None, strict: false, wait: false };       // style None: the edits' cover_style
//! let fr: mm_core::cover::FrameRecord = r.frame_record(t, &ask)?;  // or r.record(t, &ask) -> (JSON, near map)
//! r.set_edits(edits, rev);   // after each edit (mm_core::edits::Project's document and revision)
//! r.reload_edits()?;         // or: re-read PROJECT/edits.json if another writer changed it
//! r.light_ready();           // the estimate is in: records made with wait=false before it are flat; ask again
//! ```
//! Parity: native/tools/parity/records_parity.py (records, byte for byte), records_lighting.py (the estimate).
pub mod lighting;
pub mod npz;
pub mod records;
pub mod resolve;

pub use records::Records;

#[cfg(test)]
mod tests {
    #[test]
    fn base64_matches_python() {
        // base64.b64encode(b"\x00\x01\x02\xff\xfe") == b"AAEC//4="
        assert_eq!(crate::records::base64(&[0, 1, 2, 255, 254]), "AAEC//4=");
        assert_eq!(crate::records::base64(b"ab"), "YWI=");
        assert_eq!(crate::records::base64(b"abc"), "YWJj");
    }

    #[test]
    fn convolve_full() {
        // np.convolve([1, 2, 3], [0, 1, 0.5]) == [0, 1, 2.5, 4, 1.5]
        assert_eq!(crate::resolve::convolve(&[1.0, 2.0, 3.0], &[0.0, 1.0, 0.5]), vec![0.0, 1.0, 2.5, 4.0, 1.5]);
    }
}

#[cfg(test)]
mod pose_tests {
    /// the stand-in photo's own pose, measured (compare with portrait.own_pose: tools in the appnative report)
    #[test]
    #[ignore]
    fn mona_pose() {
        let root = mm_core::paths::root();
        let r = crate::records::measure_pose(&root.join("web/assets/portrait_mona.png"), &mm_core::paths::models(&root)).unwrap();
        println!("rs {} {} {}", r.0, r.1, r.2);
    }
}
