//! Where the app's files are. Two places, never mixed:
//!
//! - **the root**: read-only resources laid out as in a checkout: `web/` (strings, icons, brand, fonts, pictures),
//!   `models/` (`onnx/` and the depth model), `test_clips/jazz.mp4` (the sample). In a checkout it is the repository;
//!   in the bundle it is `Contents/Resources` (signed: never written).
//! - **the data folder**: what the app writes: the projects, the store's `state.json`, its caches. From a checkout:
//!   `out/native/` (projects in `out/native/projects`); bundled: `~/Library/Application Support/Mojimask` (projects in
//!   its `Projects/`, the Python app's folder: the same project format), CoreML's compiles in
//!   `~/Library/Caches/org.mojimask.app/coreml`.
//!
//! Nothing here names the build machine in a release build: the developer's fallbacks (the main checkout's models and
//! venv, for a worktree that has none) exist only in debug builds.
use std::path::{Path, PathBuf};

/// The main checkout, when `root` is a git worktree (debug builds only: a worktree has no models of its own).
/// A worktree's `.git` is a file, "gitdir: <main>/.git/worktrees/<name>".
#[cfg(debug_assertions)]
fn dev_main(root: &Path) -> Option<PathBuf> {
    let g = std::fs::read_to_string(root.join(".git")).ok()?;
    let dir = g.strip_prefix("gitdir:")?.trim();
    let i = dir.find("/.git/worktrees/")?;
    Some(PathBuf::from(&dir[..i]))
}

fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/tmp"))
}

/// The resources root: MM_ROOT, else the bundle's Contents/Resources (beside this executable's MacOS/), else the
/// checkout around this executable or the working directory, else the working directory.
pub fn root() -> PathBuf {
    if let Some(r) = std::env::var_os("MM_ROOT") {
        return PathBuf::from(r);
    }
    let exe = std::env::current_exe().ok().and_then(|e| e.canonicalize().ok());
    if let Some(res) = exe.as_ref().and_then(|e| e.parent()).and_then(|m| m.parent()).map(|c| c.join("Resources")) {
        if res.join("web").is_dir() {
            return res;
        }
    }
    let starts = [std::env::current_dir().ok(), exe.and_then(|e| e.parent().map(Path::to_path_buf))];
    for s in starts.into_iter().flatten() {
        let mut d = Some(s.as_path());
        while let Some(p) = d {
            if p.join("native/ARCHITECTURE.md").exists() {
                return p.to_path_buf();
            }
            d = p.parent();
        }
    }
    std::env::current_dir().unwrap_or_default()
}

/// The files the app shows and draws with (fonts, the words, icons, brand, palettes, emoji eyes, pictures):
/// `native/web` in a checkout, the bundle's `Resources/web`.
pub fn web(root: &Path) -> PathBuf {
    if bundled(root) { root.join("web") } else { root.join("native/web") }
}

/// Is this root a bundle's resources (read-only), rather than a checkout?
pub fn bundled(root: &Path) -> bool {
    !root.join("native").is_dir()
}

/// The data folder (see the module doc).
pub fn data_dir(root: &Path) -> PathBuf {
    if bundled(root) { home().join("Library/Application Support/Mojimask") } else { root.join("out/native") }
}

/// The projects folder: MOJIMASK_PROJECTS, else the data folder's (Projects/ bundled, projects/ in a checkout).
pub fn projects(root: &Path) -> PathBuf {
    if let Some(p) = std::env::var_os("MOJIMASK_PROJECTS") {
        return PathBuf::from(p);
    }
    data_dir(root).join(if bundled(root) { "Projects" } else { "projects" })
}

/// The caches (CoreML's compiles go in its coreml/): MOJIMASK_CACHE, else ~/Library/Caches/org.mojimask.app
/// bundled, out/native/cache in a checkout.
pub fn cache_dir(root: &Path) -> PathBuf {
    if let Some(p) = std::env::var_os("MOJIMASK_CACHE") {
        return PathBuf::from(p);
    }
    if bundled(root) { home().join("Library/Caches/org.mojimask.app") } else { root.join("out/native/cache") }
}

/// The models folder: MOJIMASK_MODELS, else the root's models/ (debug builds: else the main checkout's).
pub fn models(root: &Path) -> PathBuf {
    if let Some(p) = std::env::var_os("MOJIMASK_MODELS") {
        return PathBuf::from(p);
    }
    let here = root.join("models");
    #[cfg(debug_assertions)]
    if let Some(m) = dev_main(root).filter(|m| !here.join("onnx").is_dir() && m.join("models/onnx").is_dir()) {
        return m.join("models");
    }
    here
}

