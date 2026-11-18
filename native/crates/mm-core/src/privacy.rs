//! Which clips are public domain (AGENTS.md, "Privacy"): `test_clips/jazz.mp4`, a silent film, and the clips made
//! from it under `out/poc/clips/`. An agent may look at their pictures (devtools' `/grab`, readbacks); for every
//! other clip the video's pixels must never leave the app. Paths are compared after resolving symlinks and `..`, so
//! a copy of jazz elsewhere is *not* public: the rule is about where a file is, which a person controls.
use std::path::{Path, PathBuf};

fn real(p: &Path) -> Option<PathBuf> {
    std::fs::canonicalize(p).ok()
}

pub fn is_public_clip(root: &Path, video: &Path) -> bool {
    let (Some(root), Some(v)) = (real(root), real(video)) else { return false };
    if Some(&v) == real(&root.join("test_clips/jazz.mp4")).as_ref() {
        return true;
    }
    match real(&root.join("out/poc/clips")) {
        Some(dir) => v.starts_with(&dir) && v != dir,
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn allowlist() {
        let tmp = std::env::temp_dir().join(format!("mm-core-privacy-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("test_clips")).unwrap();
        std::fs::create_dir_all(tmp.join("out/poc/clips")).unwrap();
        std::fs::create_dir_all(tmp.join("out/native")).unwrap();
        for f in ["test_clips/jazz.mp4", "test_clips/party.mp4", "out/poc/clips/jazz_4k.mp4", "out/native/jazz.mp4"] {
            std::fs::write(tmp.join(f), b"x").unwrap();
        }
        assert!(is_public_clip(&tmp, &tmp.join("test_clips/jazz.mp4")));
        assert!(is_public_clip(&tmp, &tmp.join("out/poc/clips/jazz_4k.mp4")));
        assert!(is_public_clip(&tmp, &tmp.join("out/poc/clips/../clips/jazz_4k.mp4")));
        assert!(!is_public_clip(&tmp, &tmp.join("test_clips/party.mp4")));
        assert!(!is_public_clip(&tmp, &tmp.join("out/native/jazz.mp4")), "a copy elsewhere is not public");
        assert!(!is_public_clip(&tmp, &tmp.join("missing.mp4")));
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
