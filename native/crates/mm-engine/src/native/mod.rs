//! The Rust engine behind the app (`--engine native`): app_server.py's and review_server.py's routes answered in
//! this process, with no Python (unless the scan's motion setting is `python`: then the scan's camera motion and the
//! follower's motion, and a placed cover's flow, are asked of Python, exactly as the Python scan computes them).
//!
//! **The library** (app_server.py): `/api/app`, `/api/projects` (the same folder layout, `sizes()` and `scanned()`
//! rules), import (`/api/import_path`, the sample clip with its starting settings), the scan's status
//! (`/api/status/<id>`: steps, info, time left, resuming an unfinished scan) and its "Deciding who's who" events
//! (`/api/projects/<id>/decide_viz`), cancel, clear cache, delete (to the Trash), clone fresh (developer mode), relink,
//! and the pictures (`thumb.jpg`, `preview.jpg`, `head.jpg`).
//!
//! **The scan**: [`crate::scan::scan`] on a thread, its reports as app_server's STATE, its viz events live.
//!
//! **An open project** ([`project::Proj`], review_server.py): one mm_core::edits::Project per open project, the single
//! writer of edits.json and facts.json; it holds `.server.lock` as review_server does, so a Python server can't open
//! the same project at the same time (and this one refuses a project a Python server has open). Draw records from
//! mm-records ([`project::Proj::records`]).
//!
//! Exports are drawn and encoded by the app itself (mm-app's export), as with the Python engine.
pub mod follow;
pub mod http;
pub mod project;
pub mod whose;

use crate::backend::Backend;
use anyhow::{anyhow, bail, Result};
use mm_core::pyjson::{self, Py};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

pub use project::Proj;

/// How the app started its engine.
#[derive(Clone, Debug)]
pub struct Config {
    /// the repository root: web/ (brand, portrait stand-in), test_clips/jazz.mp4 (the sample)
    pub root: PathBuf,
    /// MOJIMASK_PROJECTS
    pub projects: PathBuf,
    pub dev: bool,
    /// MOJIMASK_DEPTH (None: off)
    pub depth: Option<String>,
    /// the scan's options (models, motion, plan, ...)
    pub scan: crate::scan::Options,
    /// the port range for the pictures' HTTP server
    pub ports: (u16, u16),
}

/// One answer: status, content type, body.
pub struct Reply {
    pub status: u16,
    pub ctype: &'static str,
    pub body: Vec<u8>,
    pub cache: Option<&'static str>,
}

impl Reply {
    pub fn json(v: &Value) -> Reply {
        Reply { status: 200, ctype: "application/json", body: serde_json::to_vec(v).unwrap(), cache: None }
    }
    pub fn err(status: u16, msg: impl Into<String>) -> Reply {
        let mut r = Reply::json(&json!({"error": msg.into()}));
        r.status = status;
        r
    }
    pub fn bytes(ctype: &'static str, body: Vec<u8>) -> Reply {
        Reply { status: 200, ctype, body, cache: None }
    }
}

/// A route's failure: its status and message (review_server's 400 / 404 / 500).
pub struct Fail(pub u16, pub String);

impl From<anyhow::Error> for Fail {
    fn from(e: anyhow::Error) -> Fail {
        Fail(500, format!("{e:#}"))
    }
}

impl From<mm_core::edits::Refused> for Fail {
    fn from(e: mm_core::edits::Refused) -> Fail {
        Fail(e.code, e.error)
    }
}

pub type Res = std::result::Result<Reply, Fail>;

/// app_server's STATE[id]: a scan's progress.
#[derive(Default)]
pub struct ScanSt {
    pub steps: Map<String, Value>,
    pub info: Map<String, Value>,
    pub error: Option<String>,
    pub ready: bool,
    pub current: String,
    pub name: Option<String>,
    pub started: Map<String, Value>,
    pub viz: Option<Vec<Value>>,
    pub cancel: bool,
}

pub struct Server {
    pub cfg: Config,
    pub port: u16,
    pub started: f64,
    scans: Mutex<HashMap<String, Arc<Mutex<ScanSt>>>>,
    open: Mutex<HashMap<String, Arc<Proj>>>,
    /// one project opening at a time (the status poll and a page may both ask)
    opening: Mutex<()>,
    /// draws frame t of a project with its covers as the export does (BGR, the frame's size), for thumbnails; set by
    /// the app (mm-render lives there)
    pub drawer: Mutex<Option<Arc<Drawer>>>,
    /// the face and body models (whose-face questions), opened on first use
    models: Mutex<Option<Arc<mm_analysis::models::Models>>>,
}

/// (project id, frame, style) -> (w, h, BGR) of the frame as drawn
pub type Drawer = dyn Fn(&str, usize, &str) -> Option<(usize, usize, Vec<u8>)> + Send + Sync;

pub const KEEP: [&str; 9] = ["meta.json", "edits.json", "dets", "mats_v2.npy", ".server.lock", "facts.json", "proto_tracks.json", "proto", "thumb.jpg"];
const SAMPLE_EDITS: &str = r#"{"light": {"saturation": 0.0}}"#;
const TAIL_RATIO: f64 = 2.2;
const TAIL_RANGE: (f64, f64) = (1.0, 3.5);

fn kept(name: &str) -> bool {
    KEEP.contains(&name) || (name.starts_with("decisions_") && name.ends_with(".json"))
}

pub fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

fn mtime(p: &Path) -> Option<f64> {
    std::fs::metadata(p).ok()?.modified().ok()?.duration_since(UNIX_EPOCH).ok().map(|d| d.as_secs_f64())
}

fn read_json(p: &Path) -> Option<Value> {
    serde_json::from_slice(&std::fs::read(p).ok()?).ok()
}

/// `json.dump(obj, f)` to a temporary file, then a rename.
fn write_json(p: &Path, v: &Value) -> Result<()> {
    pyjson::write_atomic(p, &Py::from_serde(v).dumps())?;
    Ok(())
}

pub fn du(p: &Path) -> u64 {
    let Ok(m) = std::fs::symlink_metadata(p) else { return 0 };
    if m.is_file() {
        return m.len();
    }
    let mut total = 0;
    if let Ok(rd) = std::fs::read_dir(p) {
        for e in rd.flatten() {
            let q = e.path();
            match e.file_type() {
                Ok(t) if t.is_dir() => total += du(&q),
                Ok(t) if t.is_file() => total += e.metadata().map(|m| m.len()).unwrap_or(0),
                _ => {}
            }
        }
    }
    total
}

/// Percent-decoding for a query value.
pub fn unquote(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            let hex = |c: u8| (c as char).to_digit(16);
            if let (Some(h), Some(l)) = (hex(b[i + 1]), hex(b[i + 2])) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(if b[i] == b'+' { b' ' } else { b[i] });
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

pub fn query(q: &str) -> HashMap<String, String> {
    q.split('&').filter(|s| !s.is_empty()).filter_map(|kv| {
        let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
        Some((unquote(k), unquote(v)))
    }).collect()
}

fn valid_id(pid: &str) -> bool {
    !pid.is_empty() && pid != "." && pid != ".." && pid.chars().all(|c| c.is_ascii_alphanumeric() || "._-".contains(c))
}

impl Server {
    /// Start the engine: its pictures' HTTP server on a free port in the range.
    pub fn start(cfg: Config) -> Result<Arc<Server>> {
        std::fs::create_dir_all(&cfg.projects)?;
        mm_media::decode::init();
        let (l, port) = http::bind(cfg.ports.0, cfg.ports.1).ok_or_else(|| anyhow!("no free port in {:?}", cfg.ports))?;
        let s = Arc::new(Server {
            cfg,
            port,
            started: now(),
            scans: Mutex::default(),
            open: Mutex::default(),
            opening: Mutex::new(()),
            drawer: Mutex::new(None),
            models: Mutex::new(None),
        });
        http::serve(l, Arc::downgrade(&s));
        Ok(s)
    }

    pub fn proj_dir(&self, pid: &str) -> std::result::Result<PathBuf, Fail> {
        if !valid_id(pid) {
            return Err(Fail(400, "bad project id".into()));
        }
        Ok(self.cfg.projects.join(pid))
    }

    fn meta(&self, pid: &str) -> Value {
        self.proj_dir(pid).ok().and_then(|d| read_json(&d.join("meta.json"))).unwrap_or(json!({}))
    }

    pub fn scanned(&self, pid: &str) -> bool {
        let Ok(d) = self.proj_dir(pid) else { return false };
        ["mats_v2.npy", "dets/pose_head.json", "proto_tracks.json"].iter().all(|f| d.join(f).exists())
    }

    fn sizes(&self, pid: &str) -> Value {
        let Ok(d) = self.proj_dir(pid) else { return json!({}) };
        let meta = self.meta(pid);
        let copy = if meta["copied"].as_bool().unwrap_or(false) { meta["video"].as_str().map(PathBuf::from) } else { None };
        let (mut kept_b, mut cache, mut copy_b) = (0u64, 0u64, 0u64);
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().to_string();
            if copy.as_ref().is_some_and(|c| *c == p) {
                copy_b += du(&p);
            } else if kept(&name) {
                kept_b += du(&p);
            } else {
                cache += du(&p);
            }
        }
        json!({"kept": kept_b, "cache": cache, "copy": copy_b})
    }

    pub fn models(&self) -> Arc<mm_analysis::models::Models> {
        let mut m = self.models.lock().unwrap();
        m.get_or_insert_with(|| {
            let o = &self.cfg.scan;
            crate::evidence::models(&o.models.join("onnx"), o.analysis_cpu, o.cache.as_ref().map(|c| c.join("analysis")))
        })
        .clone()
    }

    // ------------------------------------------------------------------------------------------------ dispatch
    /// One request, as app_server.py (and, under /p/<id>/, the project's review_server.py) answers it.
    pub fn handle(self: &Arc<Self>, method: &str, path: &str, body: &[u8]) -> Reply {
        let r = (|| -> Res {
            let body: Value = if body.is_empty() { json!({}) } else { serde_json::from_slice(body).map_err(|e| Fail(400, e.to_string()))? };
            let (p, q) = path.split_once('?').unwrap_or((path, ""));
            if let Some(rest) = p.strip_prefix("/p/") {
                let (pid, sub) = rest.split_once('/').map(|(a, b)| (a, format!("/{b}"))).unwrap_or((rest, "/".into()));
                let proj = self.project(pid)?;
                return proj.handle(self, method, &sub, &query(q), &body);
            }
            if method == "GET" { self.get(p, &query(q)) } else { self.post(p, &body) }
        })();
        match r {
            Ok(r) => r,
            Err(Fail(code, msg)) => {
                if code >= 500 {
                    crate::native::log(&format!("{method} {path}: {msg}"));
                }
                Reply::err(code, msg)
            }
        }
    }

    /// The open project `pid` (opened now if it isn't).
    pub fn project(self: &Arc<Self>, pid: &str) -> std::result::Result<Arc<Proj>, Fail> {
        if let Some(p) = self.open.lock().unwrap().get(pid) {
            return Ok(p.clone());
        }
        let _g = self.opening.lock().unwrap();
        if let Some(p) = self.open.lock().unwrap().get(pid) {
            return Ok(p.clone());
        }
        let d = self.proj_dir(pid)?;
        if !self.scanned(pid) {
            return Err(Fail(404, "not scanned yet".into()));
        }
        let t0 = std::time::Instant::now();
        let p = Arc::new(Proj::open(self, pid, &d).map_err(|e| Fail(500, format!("{e:#}")))?);
        log(&format!("opened {pid} in {:.2} s", t0.elapsed().as_secs_f64()));
        self.open.lock().unwrap().insert(pid.to_string(), p.clone());
        Ok(p)
    }

    /// Close a project (its lock released): relink, clear cache, delete.
    fn close(&self, pid: &str) {
        self.open.lock().unwrap().remove(pid);
    }

    pub fn is_open(&self, pid: &str) -> Option<Arc<Proj>> {
        self.open.lock().unwrap().get(pid).cloned()
    }

    fn get(self: &Arc<Self>, p: &str, q: &HashMap<String, String>) -> Res {
        if p == "/api/app" {
            let mut info = json!({"app": false, "dev": self.cfg.dev, "projects_dir": self.cfg.projects.to_string_lossy(), "engine": "native"});
            if self.cfg.dev {
                info["depth"] = json!(self.cfg.depth.clone().unwrap_or_else(|| "off".into()));
                info["started"] = json!(self.started);
                info["stale"] = json!(false);
                info["motion"] = json!(match &self.cfg.scan.motion {
                    crate::scan::Motion::Native => "native".to_string(),
                    crate::scan::Motion::Replay(r) => format!("replay:{}", r.display()),
                });
            }
            return Ok(Reply::json(&info));
        }
        if p == "/api/projects" {
            return Ok(Reply::json(&self.library()));
        }
        if let Some(rest) = p.strip_prefix("/api/projects/") {
            let (pid, what) = rest.split_once('/').unwrap_or((rest, ""));
            let d = self.proj_dir(pid)?;
            match what {
                "preview.jpg" | "thumb.jpg" => {
                    let b = std::fs::read(d.join(what)).map_err(|_| Fail(404, "no picture yet".into()))?;
                    return Ok(Reply::bytes("image/jpeg", b));
                }
                "head.jpg" => return self.head(&d, q),
                "decide_viz" => {
                    let since: usize = q.get("since").and_then(|s| s.parse().ok()).unwrap_or(0);
                    if let Some(st) = self.scans.lock().unwrap().get(pid).cloned() {
                        if let Some(v) = &st.lock().unwrap().viz {
                            return Ok(Reply::json(&json!({"events": v.get(since..).unwrap_or(&[]), "live": true})));
                        }
                    }
                    let ev = read_json(&d.join("proto/decide_viz.json")).and_then(|v| v.as_array().cloned()).unwrap_or_default();
                    return Ok(Reply::json(&json!({"events": ev.get(since..).unwrap_or(&[]), "live": false})));
                }
                _ => {}
            }
        }
        if let Some(pid) = p.strip_prefix("/api/status/") {
            return self.status(pid);
        }
        Err(Fail(404, "not found".into()))
    }

    fn post(self: &Arc<Self>, p: &str, body: &Value) -> Res {
        if p == "/api/import_path" {
            if body["sample"].as_bool().unwrap_or(false) {
                let sample = self.cfg.root.join("test_clips/jazz.mp4");
                let pid = self.new_project(&sample.to_string_lossy(), "jazz.mp4", None, Some(serde_json::from_str(SAMPLE_EDITS).unwrap()));
                return Ok(Reply::json(&json!({"id": pid})));
            }
            let raw = body["path"].as_str().unwrap_or("");
            let path = expand(raw);
            if !path.is_file() {
                return Err(Fail(400, "file not found".into()));
            }
            let path = std::path::absolute(&path).unwrap_or(path);
            let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            let pid = self.new_project(&path.to_string_lossy(), &name, None, None);
            return Ok(Reply::json(&json!({"id": pid})));
        }
        if let Some(rest) = p.strip_prefix("/api/projects/") {
            let (pid, act) = rest.split_once('/').unwrap_or((rest, ""));
            let d = self.proj_dir(pid)?;
            match act {
                "clone_fresh" if self.cfg.dev => return Ok(Reply::json(&self.clone_fresh(pid)?)),
                "cancel" => {
                    let st = self.scans.lock().unwrap().get(pid).cloned();
                    match st {
                        Some(st) if !st.lock().unwrap().ready && !self.scanned(pid) => {
                            st.lock().unwrap().cancel = true;
                            return Ok(Reply::json(&json!({"ok": true})));
                        }
                        _ => return Err(Fail(400, "no scan to cancel".into())),
                    }
                }
                "clear_cache" | "delete" | "relink" => {
                    if !d.is_dir() {
                        return Err(Fail(404, "unknown project".into()));
                    }
                    if act == "relink" {
                        let path = expand(body["path"].as_str().unwrap_or(""));
                        if !path.is_file() {
                            return Err(Fail(400, "file not found".into()));
                        }
                        let mut meta = self.meta(pid);
                        meta["video"] = json!(std::path::absolute(&path).unwrap_or(path).to_string_lossy());
                        if let Some(o) = meta.as_object_mut() {
                            o.remove("copied");
                        }
                        write_json(&d.join("meta.json"), &meta)?;
                        self.close(pid);
                        self.scans.lock().unwrap().remove(pid);
                        return Ok(Reply::json(&json!({"ok": true})));
                    }
                    self.close(pid);
                    self.scans.lock().unwrap().remove(pid);
                    if act == "delete" {
                        trash(&d);
                        return Ok(Reply::json(&json!({"ok": true})));
                    }
                    let meta = self.meta(pid);
                    let copy = if meta["copied"].as_bool().unwrap_or(false) { meta["video"].as_str().map(PathBuf::from) } else { None };
                    for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
                        let name = e.file_name().to_string_lossy().to_string();
                        let q = e.path();
                        if kept(&name) || name.starts_with("before_space") || name.starts_with("edits_before") || copy.as_ref() == Some(&q) {
                            continue;
                        }
                        let _ = if q.is_dir() { std::fs::remove_dir_all(&q) } else { std::fs::remove_file(&q) };
                    }
                    return Ok(Reply::json(&json!({"ok": true, "sizes": self.sizes(pid)})));
                }
                _ => {}
            }
        }
        Err(Fail(404, "not found".into()))
    }

    // ------------------------------------------------------------------------------------------------ the library
    fn library(&self) -> Value {
        let mut out = vec![];
        let mut ids: Vec<String> = std::fs::read_dir(&self.cfg.projects).into_iter().flatten().flatten()
            .map(|e| e.file_name().to_string_lossy().to_string()).collect();
        ids.sort();
        for pid in ids {
            let d = self.cfg.projects.join(&pid);
            let cancelled = self.scans.lock().unwrap().get(&pid).is_some_and(|s| s.lock().unwrap().cancel);
            if !d.join("meta.json").exists() || cancelled {
                continue;
            }
            let meta = self.meta(&pid);
            let used = ["edits.json", "meta.json"].iter().filter_map(|f| mtime(&d.join(f))).fold(0.0, f64::max);
            let video = meta["video"].as_str().unwrap_or("").to_string();
            let name = meta["label"].as_str().map(String::from).unwrap_or_else(|| {
                if video.is_empty() { pid.clone() } else { Path::new(&video).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or(pid.clone()) }
            });
            out.push(json!({
                "id": pid, "name": name, "scanned": self.scanned(&pid),
                "duration": meta["duration"], "width": meta["width"], "height": meta["height"],
                "photo": meta["image"].as_bool().unwrap_or(false),
                "video": meta["video"], "video_found": Path::new(&video).is_file(),
                "copied": meta["copied"].as_bool().unwrap_or(false), "last_used": used, "sizes": self.sizes(&pid),
                "path": d.to_string_lossy(), "thumb": mtime(&d.join("thumb.jpg")).map(|t| t as i64).unwrap_or(0),
            }));
        }
        out.sort_by(|a, b| b["last_used"].as_f64().unwrap_or(0.0).total_cmp(&a["last_used"].as_f64().unwrap_or(0.0)));
        Value::Array(out)
    }

    /// app_server's head.jpg: one head from a scanned frame (?t=&x=&y=&s= as fractions of the frame's width, px).
    fn head(&self, d: &Path, q: &HashMap<String, String>) -> Res {
        let num = |k: &str| q.get(k).and_then(|v| v.parse::<f64>().ok());
        let (Some(t), Some(x), Some(y), Some(s)) = (q.get("t").and_then(|v| v.parse::<i64>().ok()), num("x"), num("y"), num("s")) else {
            return Err(Fail(400, "t, x, y, s".into()));
        };
        let px = q.get("px").and_then(|v| v.parse::<i64>().ok()).unwrap_or(96).clamp(16, 256) as usize;
        let im = mm_vision::jpeg::read_bgr(&d.join("frames").join(format!("{t:05}.jpg"))).map_err(|_| Fail(404, "no frame".into()))?;
        let w = im.w as f64;
        let r = (s * w * 0.7).max(4.0);
        let k = px as f64 / (2.0 * r);
        // np.float32 matrix; INTER_AREA isn't a warpAffine mode: cv2 draws it linear
        let m = [k, 0.0, px as f64 / 2.0 - x * w * k, 0.0, k, px as f64 / 2.0 - y * w * k].map(|v| v as f32 as f64);
        let img = mm_vision::Mat::from_vec(im.h, im.w, 3, im.data);
        let out = mm_vision::warp_affine_u8(&img, &m, px, px, false, mm_vision::Border::Constant, [0.0; 4]);
        let mut r = Reply::bytes("image/jpeg", mm_vision::jpeg::encode_bgr(&out.data, px, px, 82));
        r.cache = Some("max-age=600");
        Ok(r)
    }

    fn clone_fresh(&self, pid: &str) -> std::result::Result<Value, Fail> {
        let src = self.proj_dir(pid)?;
        if !self.scanned(pid) {
            return Ok(json!({"error": "only a scanned project can be cloned"}));
        }
        let mut meta = self.meta(pid);
        let new = format!("{}_fresh{}", pid.split("_fresh").next().unwrap_or(pid), strftime("%H%M%S"));
        let dst = self.proj_dir(&new)?;
        std::fs::create_dir_all(&dst).map_err(|e| Fail(500, e.to_string()))?;
        let skip = ["edits.json", "facts.json", ".server.lock", "review_server.log", "final.mp4", "final.jpg"];
        for e in std::fs::read_dir(&src).into_iter().flatten().flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if skip.contains(&name.as_str()) || name.starts_with("facts.discarded") {
                continue;
            }
            // clonefile (cp -c), as app_server does
            let ok = std::process::Command::new("cp").arg("-cR").arg(e.path()).arg(dst.join(&name)).status().map(|s| s.success()).unwrap_or(false);
            if !ok {
                let _ = std::process::Command::new("cp").arg("-R").arg(e.path()).arg(dst.join(&name)).status();
            }
        }
        let video = meta["video"].as_str().unwrap_or("").to_string();
        if meta["copied"].as_bool().unwrap_or(false) && Path::new(&video).parent() == Some(src.as_path()) {
            meta["video"] = json!(dst.join(Path::new(&video).file_name().unwrap()).to_string_lossy());
        }
        let label = meta["label"].as_str().map(String::from).unwrap_or_else(|| Path::new(if video.is_empty() { pid } else { &video }).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default());
        meta["label"] = json!(format!("{} (fresh test copy)", label.trim_end_matches(" (fresh test copy)")));
        write_json(&dst.join("meta.json"), &meta)?;
        Ok(json!({"id": new}))
    }

    // ------------------------------------------------------------------------------------------------ the scan
    fn new_project(self: &Arc<Self>, video: &str, name: &str, pid: Option<String>, edits: Option<Value>) -> String {
        let pid = pid.unwrap_or_else(|| slug(name));
        let d = self.cfg.projects.join(&pid);
        let _ = std::fs::create_dir_all(&d);
        let st = ScanSt { current: "prepare".into(), name: Some(name.to_string()), ..Default::default() };
        let st = Arc::new(Mutex::new(st));
        self.scans.lock().unwrap().insert(pid.clone(), st.clone());
        let (s, p, v) = (self.clone(), pid.clone(), video.to_string());
        std::thread::Builder::new().name("mm-native-scan".into()).spawn(move || s.scan(&p, &v, st, edits)).unwrap();
        pid
    }

    fn scan(self: &Arc<Self>, pid: &str, video: &str, st: Arc<Mutex<ScanSt>>, start_edits: Option<Value>) {
        struct Cancelled;
        let proj = self.cfg.projects.join(pid);
        st.lock().unwrap().viz = Some(vec![]);
        let o = self.cfg.scan.clone();
        let st2 = st.clone();
        let mut report = move |step: &str, frac: f64, info: Value| {
            let mut s = st2.lock().unwrap();
            if s.cancel {
                drop(s);
                std::panic::panic_any(Cancelled);
            }
            s.steps.insert(step.to_string(), json!(frac));
            if let Some(m) = info.as_object() {
                for (k, v) in m {
                    s.info.insert(k.clone(), v.clone());
                }
            }
            s.current = step.to_string();
            s.started.entry(step.to_string()).or_insert(json!(now()));
        };
        let st3 = st.clone();
        let sink = move |e: &Py| {
            if let Some(v) = st3.lock().unwrap().viz.as_mut() {
                v.push(e.to_serde());
            }
        };
        let t0 = std::time::Instant::now();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| crate::scan::scan_live(video, &proj, &o, &mut report, Box::new(sink))));
        match r {
            Err(p) if p.is::<Cancelled>() => {
                trash(&proj);
                self.scans.lock().unwrap().remove(pid);
                return;
            }
            Err(p) => {
                let msg = p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or("the scan stopped".into());
                self.scan_failed(&proj, &st, &msg);
                return;
            }
            Ok(Err(e)) => {
                self.scan_failed(&proj, &st, &format!("{e:#}"));
                return;
            }
            Ok(Ok(v)) => log(&format!("scan {pid}: {:.1} s ({})", t0.elapsed().as_secs_f64(), v["result"])),
        }
        if let Some(e) = start_edits {
            let p = proj.join("edits.json");
            let mut ed = pyjson::load(&p).unwrap_or(Py::Dict(Default::default()));
            if let (Some(d), Some(add)) = (ed.as_dict_mut(), e.as_object()) {
                for (k, v) in add {
                    d.insert(k.clone(), Py::from_serde(v));
                }
            }
            let _ = pyjson::write_atomic(&p, &ed.dumps());
        }
        if st.lock().unwrap().cancel {
            trash(&proj);
            self.scans.lock().unwrap().remove(pid);
            return;
        }
        {
            let mut s = st.lock().unwrap();
            s.current = "review".into();
            s.started.entry("review".to_string()).or_insert(json!(now()));
        }
        self.start_review(pid, &st);
        self.learn_tail(&st);
    }

    fn scan_failed(&self, proj: &Path, st: &Arc<Mutex<ScanSt>>, msg: &str) {
        let _ = std::fs::OpenOptions::new().create(true).append(true).open(proj.join("scan.log")).and_then(|mut f| {
            use std::io::Write;
            writeln!(f, "{}\n{msg}\n", strftime("%Y-%m-%d %H:%M:%S"))
        });
        log(&format!("scan {}: {msg}", proj.display()));
        st.lock().unwrap().error = Some(msg.to_string());
    }

    /// The project's server is "ready" once it's open here (review_server's start-up: the edits, the facts, its
    /// state warmed up).
    fn start_review(self: &Arc<Self>, pid: &str, st: &Arc<Mutex<ScanSt>>) {
        let video = self.meta(pid)["video"].as_str().unwrap_or("").to_string();
        if !Path::new(&video).is_file() {
            st.lock().unwrap().error = Some("missing_video".into());
            return;
        }
        st.lock().unwrap().steps.insert("review".into(), json!(0.0));
        match self.project(pid) {
            Ok(p) => {
                // warm up, as review_server does before it answers
                let _ = p.state_now();
                let mut s = st.lock().unwrap();
                s.ready = true;
                s.steps.insert("review".into(), json!(1.0));
            }
            Err(Fail(_, e)) => st.lock().unwrap().error = Some(e),
        }
    }

    fn timings(&self) -> PathBuf {
        let parent = self.cfg.projects.parent().map(Path::to_path_buf).unwrap_or_default();
        parent.join("scan_timings.json")
    }

    fn tail_ratio(&self) -> f64 {
        read_json(&self.timings()).and_then(|v| v["tail_ratio"].as_f64()).map(|r| r.clamp(TAIL_RANGE.0, TAIL_RANGE.1)).unwrap_or(TAIL_RATIO)
    }

    fn learn_tail(&self, st: &Arc<Mutex<ScanSt>>) {
        let s = st.lock().unwrap();
        let (Some(d), Some(m)) = (s.started.get("detect").and_then(Value::as_f64), s.started.get("motion").and_then(Value::as_f64)) else { return };
        if !s.ready {
            return;
        }
        let (detect, tail) = (m - d, now() - m);
        if detect < 20.0 {
            return;
        }
        let r = 0.5 * self.tail_ratio() + 0.5 * (tail / detect).clamp(TAIL_RANGE.0, TAIL_RANGE.1);
        let _ = write_json(&self.timings(), &json!({"tail_ratio": (r * 1000.0).round() / 1000.0}));
    }

    fn time_left(&self, s: &ScanSt) -> Option<i64> {
        let n = now();
        if s.ready {
            return None;
        }
        let d = s.started.get("detect").and_then(Value::as_f64)?;
        match s.started.get("motion").and_then(Value::as_f64) {
            None => {
                let eta = s.info.get("eta_s").and_then(Value::as_f64)?;
                Some((eta + self.tail_ratio() * (n - d + eta)).round() as i64)
            }
            Some(m) => {
                let left = self.tail_ratio() * (m - d) - (n - m);
                (left > 3.0).then(|| left.round() as i64)
            }
        }
    }

    fn status(self: &Arc<Self>, pid: &str) -> Res {
        let d = self.proj_dir(pid)?;
        let st = {
            let mut scans = self.scans.lock().unwrap();
            if !scans.contains_key(pid) && self.scanned(pid) {
                let mut s = ScanSt { current: "review".into(), ..Default::default() };
                for k in ["prepare", "detect", "motion", "done"] {
                    s.steps.insert(k.into(), json!(1));
                }
                let st = Arc::new(Mutex::new(s));
                scans.insert(pid.to_string(), st.clone());
                let (me, p2) = (self.clone(), pid.to_string());
                std::thread::spawn(move || me.start_review(&p2, &st));
            }
            let video = self.meta(pid)["video"].as_str().unwrap_or("").to_string();
            if !scans.contains_key(pid) && Path::new(&video).is_file() {
                // resume a scan that never finished: what it had worked out (frames, detections) is reused
                let meta = self.meta(pid);
                if meta.get("frames").is_none() {
                    let _ = std::fs::remove_dir_all(d.join("frames"));
                }
                let name = meta["label"].as_str().map(String::from).unwrap_or_else(|| Path::new(&video).file_name().unwrap().to_string_lossy().to_string());
                let st = Arc::new(Mutex::new(ScanSt { current: "prepare".into(), name: Some(name), ..Default::default() }));
                scans.insert(pid.to_string(), st.clone());
                let (me, p2) = (self.clone(), pid.to_string());
                std::thread::Builder::new().name("mm-native-scan".into()).spawn(move || me.scan(&p2, &video, st, None)).unwrap();
            }
            scans.get(pid).cloned()
        };
        let Some(st) = st else { return Err(Fail(404, "unknown project".into())) };
        let s = st.lock().unwrap();
        let mut v = json!({
            "steps": s.steps, "info": s.info, "error": s.error, "review_port": if s.ready { json!(self.port) } else { Value::Null },
            "current": s.current, "viz_n": s.viz.as_ref().map(|v| v.len()).unwrap_or(0), "meta": self.meta(pid),
            "ready": s.ready, "time_left": self.time_left(&s),
        });
        if let Some(n) = &s.name {
            v["name"] = json!(n);
        }
        if !s.started.is_empty() {
            v["started"] = json!(s.started);
        }
        if s.cancel {
            v["cancel"] = json!(true);
        }
        Ok(Reply::json(&v))
    }
}

/// `os.path.abspath(os.path.expanduser(p))`
fn expand(p: &str) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(h) = std::env::var_os("HOME") {
            return PathBuf::from(h).join(rest);
        }
    }
    PathBuf::from(p)
}

/// time.strftime with the local time (libc).
pub fn strftime(fmt: &str) -> String {
    let t = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as libc_time::time_t).unwrap_or(0);
    libc_time::format(t, fmt)
}

mod libc_time {
    #[allow(non_camel_case_types)]
    pub type time_t = i64;
    #[repr(C)]
    struct Tm {
        tm_sec: i32,
        tm_min: i32,
        tm_hour: i32,
        tm_mday: i32,
        tm_mon: i32,
        tm_year: i32,
        tm_wday: i32,
        tm_yday: i32,
        tm_isdst: i32,
        tm_gmtoff: i64,
        tm_zone: *const i8,
    }
    extern "C" {
        fn localtime_r(t: *const time_t, tm: *mut Tm) -> *mut Tm;
        fn strftime(s: *mut u8, max: usize, fmt: *const i8, tm: *const Tm) -> usize;
    }
    pub fn format(t: time_t, fmt: &str) -> String {
        let mut tm: Tm = unsafe { std::mem::zeroed() };
        let f = std::ffi::CString::new(fmt).unwrap();
        let mut buf = [0u8; 128];
        unsafe {
            localtime_r(&t, &mut tm);
            let n = strftime(buf.as_mut_ptr(), buf.len(), f.as_ptr(), &tm);
            String::from_utf8_lossy(&buf[..n]).into_owned()
        }
    }
}

/// app_server.slug: the file's name made safe, plus the time.
fn slug(name: &str) -> String {
    let stem = Path::new(name).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let mut base = String::new();
    let mut last_bad = false;
    for c in stem.chars() {
        if c.is_ascii_alphanumeric() || "._-".contains(c) {
            base.push(c);
            last_bad = false;
        } else if !last_bad {
            base.push('_');
            last_bad = true;
        }
    }
    let base: String = base.chars().take(40).collect();
    let base = if base.is_empty() { "video".to_string() } else { base };
    format!("{base}_{}", strftime("%Y%m%d-%H%M%S"))
}

/// Move to the Trash (recoverable), like Finder; a plain delete when that fails.
fn trash(p: &Path) {
    if let Some(h) = std::env::var_os("HOME") {
        let bin = PathBuf::from(h).join(".Trash");
        let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let mut dst = bin.join(&name);
        let mut i = 1;
        while dst.exists() {
            dst = bin.join(format!("{name} {i}"));
            i += 1;
        }
        if std::fs::rename(p, &dst).is_ok() {
            return;
        }
    }
    let _ = std::fs::remove_dir_all(p);
}

pub fn log(s: &str) {
    eprintln!("[native engine] {s}");
}

// ------------------------------------------------------------------------------------------------ the app's client
/// The app's handle on the native engine: the same calls as the Python sidecar's, answered in process.
pub struct Client(pub Arc<Server>);

impl Client {
    fn call(&self, method: &str, path: &str, body: &[u8]) -> Result<Vec<u8>> {
        let r = self.0.handle(method, path, body);
        if !(200..300).contains(&r.status) {
            let msg = serde_json::from_slice::<Value>(&r.body).ok().and_then(|v| v["error"].as_str().map(String::from))
                .unwrap_or_else(|| String::from_utf8_lossy(&r.body).chars().take(200).collect());
            bail!("{path}: {} {msg}", r.status);
        }
        Ok(r.body)
    }
}

impl Backend for Client {
    fn engine(&self) -> &'static str {
        "native"
    }
    fn base(&self) -> String {
        format!("http://127.0.0.1:{}", self.0.port)
    }
    fn get_bytes(&self, path: &str) -> Result<Vec<u8>> {
        self.call("GET", path, &[])
    }
    fn get(&self, path: &str) -> Result<Value> {
        Ok(serde_json::from_slice(&self.call("GET", path, &[])?)?)
    }
    fn post(&self, path: &str, body: &Value, _slow: bool) -> Result<Value> {
        Ok(serde_json::from_slice(&self.call("POST", path, &serde_json::to_vec(body)?)?)?)
    }
    fn alive(&self) -> bool {
        true
    }
    fn pid(&self) -> u32 {
        std::process::id()
    }
    fn native(&self) -> Option<Arc<Server>> {
        Some(self.0.clone())
    }
}
