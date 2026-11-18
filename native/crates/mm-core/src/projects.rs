//! Where projects live, and which project server has one open. Port of `mojimask/projects.py` (and the listing
//! half of decisions_cli.py).
//!
//! A project is a folder holding meta.json (the video's timeline), the scan's files (dets/, frames/, mats_v2.npy,
//! proto_tracks.json, decisions_<kind>.json) and what people said (facts.json, edits.json). Folders:
//! - `MOJIMASK_PROJECTS` when set;
//! - the Mac app's library, ~/Library/Application Support/Mojimask/Projects;
//! - a checkout's out/projects.
//!
//! A running project server holds an exclusive lock on PROJECT/.server.lock for its life and writes {"pid", "port"}
//! in it; the OS drops the lock when the process ends.
use crate::pyjson::{self, Py};
use std::fs::{File, OpenOptions, TryLockError};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const LOCK: &str = ".server.lock";

/// The Mac app's projects.
pub fn library() -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
    home.join("Library/Application Support/Mojimask/Projects")
}

/// A checkout's projects (`root`/out/projects).
pub fn checkout(root: &Path) -> PathBuf {
    root.join("out").join("projects")
}

/// Give an existing folder the name's current case (the app was "MojiMask" until 2026-11-15, now "Mojimask").
/// This Mac's disk ignores case, so the old folder still works under the new name, but would keep showing the old one.
pub fn fix_folder_case(path: &Path) {
    let (Some(parent), Some(name)) = (path.parent(), path.file_name()) else { return };
    let name = name.to_string_lossy();
    if let Ok(rd) = std::fs::read_dir(parent) {
        for e in rd.flatten() {
            let n = e.file_name().to_string_lossy().into_owned();
            if n != name && n.to_lowercase() == name.to_lowercase() {
                let _ = std::fs::rename(e.path(), path);
            }
        }
    }
}

/// Project folders. MOJIMASK_PROJECTS when set. Otherwise [the one this app uses]: the library in the app
/// (`in_app`), else the checkout's out/projects. all=true: every place projects can be (the library and, from a
/// checkout, out/projects) that exists, for tools that look for a project wherever it is.
pub fn roots(all: bool, in_app: bool, checkout_root: Option<&Path>) -> Vec<PathBuf> {
    if let Some(p) = std::env::var_os("MOJIMASK_PROJECTS").filter(|p| !p.is_empty()) {
        return vec![PathBuf::from(p)];
    }
    let co = checkout_root.map(checkout);
    if !all {
        return vec![if in_app || std::env::var("MOJIMASK_APP").as_deref() == Ok("1") || co.is_none() {
            library()
        } else {
            co.unwrap()
        }];
    }
    let mut v = vec![library()];
    if !in_app {
        v.extend(co);
    }
    v.into_iter().filter(|r| r.is_dir()).collect()
}

/// (root, folder name) of every project (a folder with meta.json) under the roots, names sorted.
pub fn all_projects(roots: &[PathBuf]) -> Vec<(PathBuf, String)> {
    let mut out = vec![];
    for r in roots {
        let Ok(rd) = std::fs::read_dir(r) else { continue };
        let mut names: Vec<String> = rd.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        names.sort();
        for n in names {
            if r.join(&n).join("meta.json").exists() {
                out.push((r.clone(), n));
            }
        }
    }
    out
}

/// A project by folder name, or the start of one if only one matches.
pub fn find(roots: &[PathBuf], project: &str) -> Result<(PathBuf, String), String> {
    let all = all_projects(roots);
    let mut hits: Vec<_> = all.iter().filter(|(_, n)| n == project).cloned().collect();
    if hits.is_empty() {
        hits = all.iter().filter(|(_, n)| n.starts_with(project)).cloned().collect();
    }
    match hits.len() {
        0 => Err(format!("no project {project:?} (see: projects)")),
        1 => Ok(hits.remove(0)),
        n => Err(format!("{project:?} matches {n} projects: {}",
                         hits.iter().map(|(_, n)| n.as_str()).collect::<Vec<_>>().join(", "))),
    }
}

/// The project server holding this project's lock ({"pid", "port"}), or None if no server has it open.
pub fn lock_holder(proj: &Path) -> Option<Py> {
    let path = proj.join(LOCK);
    let mut f = File::open(&path).ok()?;
    match f.try_lock() {
        Ok(()) => {
            let _ = f.unlock();
            None
        }
        Err(TryLockError::WouldBlock) => {
            let mut s = String::new();
            f.seek(SeekFrom::Start(0)).ok()?;
            f.read_to_string(&mut s).ok()?;
            let v = Py::parse(if s.is_empty() { "{}" } else { &s }).ok()?;
            v.truthy().then_some(v)
        }
        Err(_) => None,
    }
}

/// Take a project's lock for this process (as a project server does), writing {"pid": ..} (and "port" if serving).
/// The lock is held while the returned file is open. Err with the holder when another process has it.
pub fn take_lock(proj: &Path, port: Option<u16>) -> Result<File, String> {
    let mut f = OpenOptions::new().create(true).read(true).append(true).open(proj.join(LOCK)).map_err(|e| e.to_string())?;
    if f.try_lock().is_err() {
        let mut s = String::new();
        let _ = f.seek(SeekFrom::Start(0)).and_then(|_| f.read_to_string(&mut s));
        return Err(format!("another project server already has this project open: {}", s.trim()));
    }
    f.set_len(0).map_err(|e| e.to_string())?;
    let mut v = crate::pydict![("pid", std::process::id() as i64)];
    if let Some(p) = port {
        v.as_dict_mut().unwrap().insert("port", Py::Int(p as i64));
    }
    f.write_all(v.dumps().as_bytes()).map_err(|e| e.to_string())?;
    f.flush().map_err(|e| e.to_string())?;
    Ok(f)
}

/// A project's meta.json.
pub fn meta(proj: &Path) -> Result<Py, String> {
    pyjson::load(&proj.join("meta.json"))
}

/// The name shown for a project: its label, else its video's file name, else the folder's.
pub fn display_name(meta: &Py, folder: &str) -> String {
    if meta.get("label").truthy() {
        return meta.get("label").py_str();
    }
    let v = meta.get("video").as_str().unwrap_or(folder);
    Path::new(v).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| v.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn listing_finding_and_locks() {
        let d = std::env::temp_dir().join(format!("mm_projects_{}", std::process::id()));
        for n in ["jazz_a", "jazz_b", "other"] {
            std::fs::create_dir_all(d.join(n)).unwrap();
            std::fs::write(d.join(n).join("meta.json"), r#"{"video": "/x/jazz.mp4"}"#).unwrap();
        }
        std::fs::create_dir_all(d.join("not_a_project")).unwrap();
        let roots = vec![d.clone()];
        assert_eq!(all_projects(&roots).len(), 3);
        assert_eq!(find(&roots, "oth").unwrap().1, "other");
        assert!(find(&roots, "jazz").unwrap_err().contains("matches 2 projects"));
        assert!(find(&roots, "nope").is_err());
        assert_eq!(display_name(&meta(&d.join("other")).unwrap(), "other"), "jazz.mp4");
        let p = d.join("other");
        assert!(lock_holder(&p).is_none());
        let held = take_lock(&p, Some(9400)).unwrap();
        // (flock locks belong to the open file: a second open in this process sees it as held, like another process)
        assert_eq!(lock_holder(&p).unwrap().get("port"), &Py::Int(9400));
        assert!(take_lock(&p, None).is_err());
        drop(held);
        assert!(lock_holder(&p).is_none());
        std::fs::remove_dir_all(&d).unwrap();
    }
}
