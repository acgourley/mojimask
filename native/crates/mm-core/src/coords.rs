//! Which pixel space a project's saved files are in, checked when it opens. Port of `mojimask/coords.py`.
//!
//! Every position a project saves -- detections (dets/*.json), camera motion (mats_v2.npy), the scan's tracks
//! (proto_tracks.json) and the answers that hold positions (edits.json: covers drawn by hand) -- is in the pixels of
//! the frames the scan decoded. That space is recorded in meta.json["space"]:
//!
//! ```text
//! {"w": 643, "h": 476, "rot": 0, "v": 2}
//! ```
//!
//!   w, h  frame size in pixels, upright (after rotation)
//!   rot   the video's rotation (frames are stored upright)
//!   v     how frames were decoded: 2 = at the display size, square pixels (video_io._frames_avf)
//!
//! If the decoder now gives another space, the project can't be used as it is: scan the video again. (Converting
//! older projects' positions was removed with the old pipeline, 2026-11-12: it predated proto_tracks.json and facts.)
use crate::pyjson::{self, Py};
use crate::pydict;
use std::path::Path;

/// video_io.SPACE_VERSION: frames decoded at the display size, square pixels.
pub const SPACE_VERSION: i64 = 2;

/// The space a decoder gives for a video's probe info (video_io.space).
pub fn space(width: i64, height: i64, rotation: i64) -> Py {
    pydict![("w", width), ("h", height), ("rot", rotation), ("v", SPACE_VERSION)]
}

/// (width, height) of a baseline or progressive JPEG, from its SOF marker.
pub fn jpeg_size(path: &Path) -> Option<(i64, i64)> {
    let b = std::fs::read(path).ok()?;
    if b.get(..2)? != [0xff, 0xd8] {
        return None;
    }
    let mut i = 2;
    while i + 4 <= b.len() {
        if b[i] != 0xff {
            i += 1;
            continue;
        }
        let m = b[i + 1];
        if m == 0xff {
            i += 1;
            continue;
        }
        if matches!(m, 0xd8 | 0x01 | 0xd0..=0xd7) {
            i += 2;
            continue;
        }
        let len = u16::from_be_bytes([b[i + 2], b[i + 3]]) as usize;
        if matches!(m, 0xc0..=0xcf) && !matches!(m, 0xc4 | 0xc8 | 0xcc) {
            let h = u16::from_be_bytes([*b.get(i + 5)?, *b.get(i + 6)?]) as i64;
            let w = u16::from_be_bytes([*b.get(i + 7)?, *b.get(i + 8)?]) as i64;
            return Some((w, h));
        }
        i += 2 + len;
    }
    None
}

/// The space a project's files are in: recorded, else read off its first frame image (decoder v1), else None
/// (nothing scanned yet).
pub fn saved_space(work: &Path, meta: &Py) -> Option<Py> {
    if meta.get("space").truthy() {
        return Some(meta.get("space").clone());
    }
    let jpg = work.join("frames").join("00000.jpg");
    if !jpg.exists() {
        return None;
    }
    // unreadable: no space to check against, as if nothing were scanned
    let (w, h) = jpeg_size(&jpg)?;
    // rotation None: not recorded (frames were upright)
    Some(pydict![("w", w), ("h", h), ("rot", meta.get("rotation").clone()), ("v", 1)])
}

fn rot(sp: &Py) -> Py {
    let r = sp.get("rot");
    if r.truthy() { r.clone() } else { Py::Int(0) }
}

/// Older projects didn't record the rotation (None); their frames were stored upright like now, so an unknown
/// rotation matches any.
fn same_rot(a: &Py, b: &Py) -> bool {
    a.get("rot").is_null() || b.get("rot").is_null() || rot(a) == rot(b)
}

pub fn same(a: Option<&Py>, b: Option<&Py>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a.get("w") == b.get("w") && a.get("h") == b.get("h") && same_rot(a, b),
        _ => false,
    }
}

/// Called when a project opens: are its saved positions in the space the decoder gives now (`now`, from
/// [`space`])? Returns a note for the reviewer when they aren't, else None. Records the space in meta.json the first
/// time (or when the decoder version moved on with the same pixels).
pub fn ensure(work: &Path, now: &Py) -> Result<Option<String>, String> {
    let mp = work.join("meta.json");
    let mut meta = if mp.exists() { pyjson::load(&mp)? } else { Py::Dict(Default::default()) };
    let Some(src) = saved_space(work, &meta) else { return Ok(None) };
    if same(Some(&src), Some(now)) {
        if meta.get("space") != now {
            meta.as_dict_mut().ok_or("meta.json: not an object")?.insert("space", now.clone());
            pyjson::write_atomic(&mp, &meta.dumps()).map_err(|e| e.to_string())?;
        }
        return Ok(None);
    }
    Ok(Some(format!(
        "This project was scanned from frames of another shape ({}x{}, {} degrees; the video now gives {}x{}, {} degrees): \
         import the video again for a new scan.",
        src.get("w").py_str(), src.get("h").py_str(), rot(&src).py_str(),
        now.get("w").py_str(), now.get("h").py_str(), now.get("rot").py_str()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spaces_compare_like_python() {
        let a = space(643, 476, 0);
        let old = Py::parse(r#"{"w": 643, "h": 476, "rot": null, "v": 1}"#).unwrap();
        assert!(same(Some(&a), Some(&old))); // unknown rotation matches any
        assert!(!same(Some(&a), Some(&space(643, 476, 90))));
        assert!(!same(Some(&a), Some(&space(476, 643, 0))));
        assert!(!same(None, Some(&a)));
    }

    #[test]
    fn ensure_records_and_warns() {
        let d = std::env::temp_dir().join(format!("mm_coords_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("meta.json"), r#"{"fps": 24.0, "space": {"w": 10, "h": 20, "rot": 0, "v": 1}}"#).unwrap();
        assert_eq!(ensure(&d, &space(10, 20, 0)).unwrap(), None);
        assert_eq!(std::fs::read_to_string(d.join("meta.json")).unwrap(),
                   r#"{"fps": 24.0, "space": {"w": 10, "h": 20, "rot": 0, "v": 2}}"#);
        let note = ensure(&d, &space(20, 10, 0)).unwrap().unwrap();
        assert!(note.contains("(10x20, 0 degrees; the video now gives 20x10, 0 degrees)"));
        std::fs::remove_dir_all(&d).unwrap();
    }
}
