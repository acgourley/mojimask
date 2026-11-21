//! What a scan leaves in a project folder, read in one place.
//!
//! PROJECT/meta.json    the clip: "video" (its path), "frames" (how many the scan read), "width", "height" (the
//!                      scan's pixel space, [`crate::coords`]), "fps", "image" (a photo), and the app's own notes
//! PROJECT/mats_v2.npy  camera motion, N x 2 x 3 float64: row t maps frame t-1 to frame t (identity first)
//! PROJECT/proto_tracks.json  the decide step's boxes and tracks
//! PROJECT/dets/        the detect step's detections
//!
//! About eight crates used to open meta.json themselves, each with its own key names and fallbacks, and three of
//! them carried a copy of the mats_v2.npy shape check. They now ask a [`Scan`]. (The frames themselves come from the
//! video: mm-media's `ScanFrames::for_project`.)
use crate::pyjson::{self, Py};
use std::path::{Path, PathBuf};

pub const META: &str = "meta.json";
pub const MATS: &str = "mats_v2.npy";
pub const PROTO_TRACKS: &str = "proto_tracks.json";

/// A 2x3 affine, row major.
pub type M23 = [[f64; 3]; 2];

/// A project folder and its meta.json.
#[derive(Clone, Debug)]
pub struct Scan {
    pub dir: PathBuf,
    pub meta: Py,
}

impl Scan {
    pub fn open(proj: &Path) -> Result<Scan, String> {
        Ok(Scan { dir: proj.to_path_buf(), meta: pyjson::load(&proj.join(META))? })
    }
    /// How many frames the scan read; None before the detect step has run.
    pub fn frames(&self) -> Option<usize> {
        self.meta.get("frames").as_i64().map(|n| n.max(0) as usize)
    }
    /// The frames' size (width, height) in the scan's pixel space.
    pub fn size(&self) -> Option<(f64, f64)> {
        Some((self.meta.get("width").as_f64()?, self.meta.get("height").as_f64()?))
    }
    /// Frames a second; None when the clip gave none (a photo), and each caller has its own fallback.
    pub fn fps(&self) -> Option<f64> {
        match self.meta.get("fps") {
            v if v.truthy() => v.as_f64(),
            _ => None,
        }
    }
    /// A photo, not a video.
    pub fn photo(&self) -> bool {
        self.meta.get("image").truthy()
    }
    /// The clip the scan read.
    pub fn video(&self) -> Option<&str> {
        self.meta.get("video").as_str()
    }
    /// Camera motion for every frame (mats_v2.npy).
    pub fn mats(&self) -> Result<Vec<M23>, String> {
        let (shape, v) = crate::edits::read_npy_f64(&self.dir.join(MATS))?;
        if shape.len() != 3 || shape[1] != 2 || shape[2] != 3 {
            return Err(format!("{MATS}: shape {shape:?}"));
        }
        Ok(v.chunks_exact(6).map(|m| [[m[0], m[1], m[2]], [m[3], m[4], m[5]]]).collect())
    }
    /// The decide step's result; None when the project was scanned before proto tracks existed (or not yet).
    pub fn proto_tracks(&self) -> Result<Option<Py>, String> {
        let p = self.dir.join(PROTO_TRACKS);
        if !p.exists() {
            return Ok(None);
        }
        pyjson::load(&p).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_meta_and_says_what_is_missing() {
        let d = std::env::temp_dir().join(format!("mm-scan-meta-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        assert!(Scan::open(&d).is_err());
        std::fs::write(d.join(META), r#"{"width": 643, "height": 476, "fps": 0, "video": "/x/jazz.mp4"}"#).unwrap();
        let s = Scan::open(&d).unwrap();
        assert_eq!((s.frames(), s.size(), s.fps(), s.photo(), s.video()), (None, Some((643.0, 476.0)), None, false, Some("/x/jazz.mp4")));
        assert_eq!(s.proto_tracks().unwrap().map(|_| ()), None);
        assert!(s.mats().is_err());
        std::fs::remove_dir_all(&d).unwrap();
    }
}
