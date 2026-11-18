//! Projects, scans, and the open project's Decisions and People, over the engine (mm_engine::native, through
//! mm_engine::backend's routes, the ones the Python app's pages called): the library (`/api/projects`), import (`/api/import_path`), the scan's progress
//! (`/api/status/<id>`, polled) and its "Deciding who's who" events (`/api/projects/<id>/decide_viz`), then the
//! project's own server through the `/p/<id>/` proxy (`api/init`, `api/decisions`, `api/facts`, `api/undo`).
//!
//! Everything that talks to the engine runs on one worker thread, in order (an answer, then the next key): commands
//! queue work ([`queue`]) and return at once; the state changes when the replies arrive. The scan is polled by a
//! thread of its own.
use crate::decide::{self, KeyAction};
use crate::people;
use crate::text::{v, Text};
use crate::{log, Shared};
use anyhow::{anyhow, bail, Result};
use mm_engine::backend::Backend;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

pub type Job = Box<dyn FnOnce(&Arc<Shared>) + Send>;

/// What the store keeps about the library and the open project, outside the state tree.
#[derive(Default)]
pub struct Lib {
    pub sc: Option<Arc<dyn Backend>>,
    /// the scan on show: its id, and a generation (a newer scan or leaving stops the poller)
    pub scan_id: String,
    pub scan_gen: u64,
    /// "Deciding who's who" events received for it (scan/viz.py)
    pub viz: Vec<Value>,
    /// the page has shown every event (or gave up waiting): the project can open
    pub viz_drained: bool,
    /// the open project
    pub pid: String,
    pub doc: Value,
    pub decisions: Vec<Value>,
    pub missing: bool,
    pub boxes: HashMap<String, Option<BTreeMap<usize, [f64; 4]>>>,
    pub dec_ui: decide::Ui,
    pub people_ui: people::Ui,
    pub palettes: Value,
    /// the edits' revision the Editor's covers were made from
    pub editor_rev: Value,
    /// the Editor's own choices (selection, overlays, a lighting drag)
    pub ed: crate::editor::Ui,
    /// the Export page's choices
    pub xp: crate::export::PageUi,
    /// after a restart: the page and frame to show once the project has opened again
    pub restore: Option<(String, usize)>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct LibraryState {
    /// the sidecar answers
    pub ready: bool,
    /// "" while starting; why it couldn't start
    pub error: String,
    pub dev: bool,
    pub projects_dir: String,
    pub projects_dir_short: String,
    pub projects: Vec<ProjectCard>,
    /// "{size} on disk"
    pub on_disk: String,
    /// the last "Clear cache" result
    pub message: String,
    /// "Clear 2 caches older than 30 days (1.2 GB)" ("" when there are none)
    pub clear_stale: String,
    /// the import button's label while a file is being started ("" otherwise)
    pub busy: String,
    /// developer mode: what the server is running
    pub dev_line: String,
    pub stale_code: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ProjectCard {
    pub id: String,
    pub name: String,
    pub meta: String,
    pub sizes: String,
    pub sizes_tip: String,
    pub video: String,
    pub video_found: bool,
    pub scanned: bool,
    pub copied: bool,
    pub cache: u64,
    pub cache_text: String,
    pub copy_text: String,
    pub path: String,
    /// its picture (a frame with its covers), "" before its first scan finished
    pub thumb: String,
    /// public domain (jazz): an agent may look at its pictures
    pub public: bool,
    pub open_label: String,
    /// not opened for OLD_DAYS and has a cache
    pub stale: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ScanState {
    pub id: String,
    /// a project scanned before: a plain "Opening…"
    pub opening: bool,
    pub opening_name: String,
    /// "jazz.mp4 · 5 s · 643×476"
    pub file: String,
    pub headline: String,
    pub pct: String,
    pub left: String,
    /// the five steps, each 0..1
    pub steps: Vec<f64>,
    pub step_names: Vec<String>,
    pub stepcap: String,
    pub stats: Vec<(String, String)>,
    pub note: String,
    pub error: String,
    pub missing_video: bool,
    pub badge: String,
    /// how far through the clip, 0..1, and the clock either side
    pub timeline: f64,
    pub tl_left: String,
    pub tl_right: String,
    pub photo: bool,
    /// the live preview (reloaded as `preview_n` changes) and the picture's shape
    pub preview: String,
    pub preview_n: u64,
    pub aspect: f64,
    pub can_cancel: bool,
    /// "Deciding who's who" is drawn (the page asks `viz_events`); `viz_n` events so far
    pub viz: bool,
    pub viz_n: usize,
    /// the scan is done: the page shows what's left of the events, then calls `viz_drained`
    pub finishing: bool,
    /// the head pictures' URL, without the query: `?t=&x=&y=&s=&px=`
    pub head_url: String,
    pub public: bool,
    pub ready: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ProjectState {
    pub id: String,
    pub open: bool,
    pub name: String,
    /// the project server, through app_server's proxy
    pub base: String,
    pub public: bool,
    pub w: f64,
    pub h: f64,
    pub n: usize,
    pub fps: f64,
    pub photo: bool,
    pub video: String,
    pub can_undo: bool,
    pub can_redo: bool,
    /// the bar's counts: questions to answer, people
    pub open_count: i64,
    pub people_count: i64,
    pub error: String,
}

// ---------------------------------------------------------------------------------------------------- plumbing
/// Start the worker thread (and the sidecar, when configured).
pub fn start(sh: &Arc<Shared>, rx: mpsc::Receiver<Job>) {
    let s2 = sh.clone();
    std::thread::Builder::new()
        .name("mm-library".into())
        .spawn(move || {
            while let Ok(job) = rx.recv() {
                job(&s2);
            }
        })
        .unwrap();
    if let Ok(p) = std::fs::read(mm_core::paths::web(&sh.root()).join("palettes.json")) {
        sh.lib.lock().unwrap().palettes = serde_json::from_slice(&p).unwrap_or(Value::Null);
    }
    if sh.cfg.sidecar {
        queue(sh, |sh| {
            if let Err(e) = start_sidecar(sh) {
                log::error(&format!("sidecar: {e:#}"));
                sh.st.lock().unwrap().library.error = format!("{e:#}");
                sh.changed(&["library"]);
            }
        });
    }
}

pub fn queue(sh: &Arc<Shared>, f: impl FnOnce(&Arc<Shared>) + Send + 'static) {
    let _ = sh.work.lock().unwrap().send(Box::new(f));
}

/// Queue work that can fail: a failure is logged and shown where `on_err` puts it.
fn queue_try(sh: &Arc<Shared>, what: &'static str, f: impl FnOnce(&Arc<Shared>) -> Result<()> + Send + 'static) {
    queue(sh, move |sh| {
        if let Err(e) = f(sh) {
            log::warn(&format!("{what}: {e:#}"));
            let mut st = sh.st.lock().unwrap();
            match what {
                "answer" | "decisions" => st.decisions.error = format!("{e:#}"),
                "people" => {}
                _ => st.library.message = format!("{e:#}"),
            }
            drop(st);
            sh.changed(&["library", "decisions"]);
        }
    });
}

pub fn sidecar(sh: &Shared) -> Result<Arc<dyn Backend>> {
    sh.lib.lock().unwrap().sc.clone().ok_or_else(|| anyhow!("the engine isn't running yet"))
}

/// The engine: the Rust one in this process (mm_engine::native).
fn start_sidecar(sh: &Arc<Shared>) -> Result<()> {
    start_native(sh)
}

/// The scan's motion setting (`--motion native|python`; python for now: native motion asks one extra question on
/// jazz, and switching is the user's call).
pub fn motion_setting(sh: &Shared) -> String {
    if sh.cfg.motion.is_empty() { "python".into() } else { sh.cfg.motion.clone() }
}

/// The Rust engine (mm_engine::native): app_server's and review_server's routes in this process.
fn start_native(sh: &Arc<Shared>) -> Result<()> {
    let root = sh.cfg.root.clone();
    let projects = if sh.cfg.projects.is_empty() { mm_core::paths::projects(Path::new(&root)).to_string_lossy().into_owned() } else { sh.cfg.projects.clone() };
    let python = if sh.cfg.python.is_empty() { mm_core::paths::python(Path::new(&root)) } else { sh.cfg.python.clone() };
    let cache = mm_core::paths::cache_dir(Path::new(&root)).to_string_lossy().into_owned();
    let mut args = vec!["--root".to_string(), root.clone(), "--python".into(), python, "--motion".into(), motion_setting(sh), "--cache".into(), cache];
    let depth = std::env::var("MOJIMASK_DEPTH").ok().filter(|d| !d.is_empty() && d != "off");
    if let Some(d) = &depth {
        args.extend(["--depth".into(), d.clone()]);
    }
    let scan = mm_engine::scan::Options::from_args(&args)?;
    let cfg = mm_engine::native::Config {
        root: root.clone().into(),
        projects: projects.clone().into(),
        dev: sh.dev(),
        depth,
        scan,
        ports: (9300, 9400),
    };
    let t0 = Instant::now();
    let srv = mm_engine::native::Server::start(cfg)?;
    // thumbnails: a frame drawn as the export draws it (mm-render here)
    let weak = Arc::downgrade(sh);
    *srv.drawer.lock().unwrap() = Some(Arc::new(move |pid: &str, t: usize, style: &str| {
        let sh = weak.upgrade()?;
        crate::records::draw_project_frame(&sh, pid, t, style).map_err(|e| log::warn(&format!("thumb: {e:#}"))).ok()
    }));
    log::info(&format!("engine: native (motion {}, depth {}), pictures on {} ({:.2} s; projects {projects})", motion_setting(sh),
                       srv.cfg.depth.as_deref().unwrap_or("off"), srv.port, t0.elapsed().as_secs_f64()));
    sh.lib.lock().unwrap().sc = Some(Arc::new(mm_engine::native::Client(srv)));
    ready_library(sh)
}

fn ready_library(sh: &Arc<Shared>) -> Result<()> {
    {
        let mut st = sh.st.lock().unwrap();
        st.library.ready = true;
        st.library.error.clear();
    }
    sh.changed(&["library"]);
    refresh_library(sh)
}

/// Stop the sidecar (the app quitting): its project servers stop with it.
pub fn stop(sh: &Shared) {
    sh.lib.lock().unwrap().scan_gen += 1;
    let sc = sh.lib.lock().unwrap().sc.take();
    drop(sc);
}

// ---------------------------------------------------------------------------------------------------- formatting
pub fn fmt_bytes(tx: &Text, b: f64) -> String {
    if b >= 1e9 {
        tx.t("unit.gb", &v(&[("n", format!("{:.1}", b / 1e9))]))
    } else if b >= 1e6 {
        tx.t("unit.mb", &v(&[("n", (b / 1e6).round())]))
    } else {
        tx.t("unit.kb", &v(&[("n", (b / 1e3).round().max(1.0))]))
    }
}

/// "Nov 14": a day in this Mac's time zone (toLocaleDateString, {month: "short", day: "numeric"})
fn short_date(secs: f64) -> String {
    let t = secs as libc::time_t;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    unsafe { libc::localtime_r(&t, &mut tm) };
    const M: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    format!("{} {}", M[tm.tm_mon.clamp(0, 11) as usize], tm.tm_mday)
}

fn now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

fn tilde(p: &str) -> String {
    match std::env::var("HOME") {
        Ok(h) if p.starts_with(&h) => format!("~{}", &p[h.len()..]),
        _ => p.to_string(),
    }
}

fn js_round(x: f64) -> i64 {
    (x + 0.5).floor() as i64
}

fn thousands(n: i64) -> String {
    let s = n.abs().to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    if n < 0 {
        format!("-{out}")
    } else {
        out
    }
}

// ---------------------------------------------------------------------------------------------------- the library
const OLD_DAYS: f64 = 30.0;

pub fn refresh_library(sh: &Arc<Shared>) -> Result<()> {
    let sc = sidecar(sh)?;
    let ps = sc.get("/api/projects")?;
    let app = sc.get("/api/app")?;
    let tx = sh.text.clone();
    let root = sh.root();
    let mut cards = vec![];
    let (mut total, mut stale_n, mut stale_b) = (0.0, 0, 0.0);
    for p in ps.as_array().into_iter().flatten() {
        let sz = |k: &str| p["sizes"][k].as_f64().unwrap_or(0.0);
        total += sz("kept") + sz("cache") + sz("copy");
        let used = p["last_used"].as_f64().unwrap_or(0.0);
        if now() - used > OLD_DAYS * 86400.0 && sz("cache") > 0.0 {
            stale_n += 1;
            stale_b += sz("cache");
        }
        let photo = p["photo"].as_bool().unwrap_or(false);
        let scanned = p["scanned"].as_bool().unwrap_or(false);
        let found = p["video_found"].as_bool().unwrap_or(false);
        let mut meta = vec![];
        if photo {
            meta.push(tx.s("projects.photo"));
        } else if let Some(d) = p["duration"].as_f64().filter(|d| *d > 0.0) {
            meta.push(tx.t("unit.s", &v(&[("n", js_round(d))])));
        }
        if let (Some(w), Some(h)) = (p["width"].as_i64(), p["height"].as_i64()) {
            meta.push(format!("{w}×{h}"));
        }
        if used > 0.0 {
            meta.push(short_date(used));
        }
        if !scanned {
            meta.push(tx.s("projects.scan_not_finished"));
        }
        let mut sizes = vec![
            tx.t("projects.size_work", &v(&[("size", fmt_bytes(&tx, sz("kept")))])),
            tx.t("projects.size_cache", &v(&[("size", if sz("cache") > 0.0 { fmt_bytes(&tx, sz("cache")) } else { tx.s("projects.cache_cleared") })])),
        ];
        if sz("copy") > 0.0 {
            sizes.push(tx.t("projects.size_video", &v(&[("size", fmt_bytes(&tx, sz("copy")))])));
        }
        if !found {
            sizes.push(tx.s("projects.video_not_found"));
        }
        let id = p["id"].as_str().unwrap_or("").to_string();
        let video = p["video"].as_str().unwrap_or("").to_string();
        let thumb = p["thumb"].as_i64().unwrap_or(0);
        let mut tip = tx.s("projects.sizes_tip");
        if !found {
            tip += &format!(" {}", tx.t("projects.sizes_tip_missing", &v(&[("path", &video)])));
        }
        let stale = now() - used > OLD_DAYS * 86400.0 && sz("cache") > 0.0;
        cards.push(ProjectCard {
            stale,
            public: mm_core::privacy::is_public_clip(&root, std::path::Path::new(&video)),
            name: p["name"].as_str().unwrap_or(&id).to_string(),
            meta: meta.join(" · "),
            sizes: sizes.join(" · "),
            sizes_tip: tip,
            video_found: found,
            scanned,
            copied: p["copied"].as_bool().unwrap_or(false),
            cache: sz("cache") as u64,
            cache_text: if sz("cache") > 0.0 { fmt_bytes(&tx, sz("cache")) } else { String::new() },
            copy_text: fmt_bytes(&tx, sz("copy")),
            path: p["path"].as_str().unwrap_or("").to_string(),
            thumb: if thumb > 0 { sc.url(&format!("/api/projects/{id}/thumb.jpg?v={thumb}")) } else { String::new() },
            open_label: tx.s(if !found {
                "projects.locate"
            } else if scanned {
                "projects.open"
            } else {
                "projects.resume"
            }),
            video,
            id,
        });
    }
    let dir = app["projects_dir"].as_str().unwrap_or("").to_string();
    {
        let mut st = sh.st.lock().unwrap();
        let l = &mut st.library;
        l.ready = true;
        l.dev = app["dev"].as_bool().unwrap_or(false) && sh.dev();
        l.projects_dir_short = tilde(&dir);
        l.projects_dir = dir;
        l.projects = cards;
        l.on_disk = tx.t("projects.on_disk", &v(&[("size", fmt_bytes(&tx, total))]));
        l.clear_stale = if stale_n > 0 {
            tx.t("projects.clear_stale", &v(&[("n", stale_n.to_string()), ("days", OLD_DAYS.to_string()), ("size", fmt_bytes(&tx, stale_b))]))
        } else {
            String::new()
        };
        l.dev_line = if l.dev {
            format!("Dev · engine {} · scan motion {} · depth {} · server since {}", sc.engine(), motion_setting(sh), app["depth"].as_str().unwrap_or("off"),
                    short_date(app["started"].as_f64().unwrap_or(0.0)))
        } else {
            String::new()
        };
        l.stale_code = app["stale"].as_bool().unwrap_or(false);
    }
    sh.changed(&["library"]);
    Ok(())
}

pub fn cmd_refresh(sh: &Arc<Shared>) -> Value {
    queue_try(sh, "library", refresh_library);
    json!("queued")
}

/// Import a video or photo where it is (`/api/import_path`), or the sample clip; its scan starts and shows.
pub fn cmd_import(sh: &Arc<Shared>, path: Option<String>) -> Result<Value> {
    if !sh.st.lock().unwrap().library.busy.is_empty() {
        bail!("an import is already starting");
    }
    sh.st.lock().unwrap().library.busy = sh.text.s("start.starting_scan");
    sh.changed(&["library"]);
    queue(sh, move |sh| {
        let r = (|| -> Result<String> {
            let sc = sidecar(sh)?;
            let body = match &path {
                Some(p) => json!({"path": p}),
                None => json!({"sample": true}),
            };
            let r = sc.post("/api/import_path", &body, false)?;
            r["id"].as_str().map(String::from).ok_or_else(|| anyhow!("{}", r["error"].as_str().unwrap_or("no project id")))
        })();
        sh.st.lock().unwrap().library.busy.clear();
        match r {
            Ok(id) => {
                log::info(&format!("import {}: project {id}", path.as_deref().unwrap_or("(sample)")));
                start_scan(sh, &id, None);
            }
            Err(e) => {
                let msg = sh.text.s("start.err.open_file");
                sh.st.lock().unwrap().library.message = format!("{msg} ({e:#})");
                sh.changed(&["library"]);
            }
        }
    });
    Ok(json!("importing"))
}

/// The ⋯ menu's actions: clear (the cache), delete (to the Trash), clone (developer mode), relink {path}.
pub fn cmd_project_action(sh: &Arc<Shared>, id: String, act: String, path: Option<String>) -> Result<Value> {
    queue_try(sh, "library", move |sh| {
        let sc = sidecar(sh)?;
        let tx = sh.text.clone();
        let card = sh.st.lock().unwrap().library.projects.iter().find(|p| p.id == id).cloned().unwrap_or_default();
        match act.as_str() {
            "clear" => {
                sc.post(&format!("/api/projects/{id}/clear_cache"), &json!({}), false)?;
                sh.st.lock().unwrap().library.message = tx.t("projects.freed_one", &v(&[("size", card.cache_text.clone()), ("name", card.name.clone())]));
            }
            "clear_stale" => {
                let olds: Vec<ProjectCard> = sh.st.lock().unwrap().library.projects.iter().filter(|p| p.stale).cloned().collect();
                for p in &olds {
                    sc.post(&format!("/api/projects/{}/clear_cache", p.id), &json!({}), false)?;
                }
                let b: f64 = olds.iter().map(|p| p.cache as f64).sum();
                sh.st.lock().unwrap().library.message = tx.t("projects.freed_many", &v(&[("n", olds.len().to_string()), ("size", fmt_bytes(&tx, b))]));
            }
            "delete" => {
                sc.post(&format!("/api/projects/{id}/delete"), &json!({}), false)?;
                sh.st.lock().unwrap().library.message.clear();
            }
            "clone" => {
                let r = sc.post(&format!("/api/projects/{id}/clone_fresh"), &json!({}), false)?;
                if let Some(e) = r["error"].as_str() {
                    bail!("Couldn't clone “{}”: {e}", card.name);
                }
            }
            "relink" => {
                let p = path.ok_or_else(|| anyhow!("relink: no path"))?;
                sc.post(&format!("/api/projects/{id}/relink"), &json!({"path": crate::commands::local_path(&p)}), false)?;
            }
            _ => bail!("no action '{act}'"),
        }
        sh.changed(&["library"]);
        refresh_library(sh)
    });
    Ok(json!("queued"))
}

// ---------------------------------------------------------------------------------------------------- the scan
const STEPS: [&str; 5] = ["prepare", "detect", "motion", "decide", "review"];

/// Show a project's scan (or, scanned before, a plain "Opening…") and poll it until its server is ready; then open it.
pub fn start_scan(sh: &Arc<Shared>, id: &str, opening: Option<String>) {
    let gen = {
        let mut l = sh.lib.lock().unwrap();
        l.scan_gen += 1;
        l.scan_id = id.to_string();
        l.viz.clear();
        l.viz_drained = false;
        l.scan_gen
    };
    let base = sh.lib.lock().unwrap().sc.as_ref().map(|s| s.base()).unwrap_or_default();
    {
        let tx = &sh.text;
        let mut st = sh.st.lock().unwrap();
        st.scan = ScanState {
            id: id.to_string(),
            opening: opening.is_some(),
            opening_name: opening.unwrap_or_default(),
            headline: tx.s("scan.getting_ready"),
            note: tx.s("scan.getting_ready"),
            pct: "0%".into(),
            steps: vec![0.0; 5],
            step_names: STEPS.iter().map(|k| tx.s(&format!("scan.step.{k}"))).collect(),
            aspect: 1.6,
            can_cancel: true,
            head_url: format!("{base}/api/projects/{id}/head.jpg"),
            ..Default::default()
        };
        st.app.page = "scan".into();
    }
    sh.changed(&["scan", "app.page"]);
    let s2 = sh.clone();
    let id = id.to_string();
    std::thread::Builder::new().name("mm-scan".into()).spawn(move || poll_scan(&s2, &id, gen)).unwrap();
}

fn poll_scan(sh: &Arc<Shared>, id: &str, gen: u64) {
    let live = || sh.lib.lock().unwrap().scan_gen == gen;
    let mut fails = 0;
    let mut last_preview = Instant::now() - Duration::from_secs(10);
    while live() {
        let Ok(sc) = sidecar(sh) else { return };
        let st = match sc.get(&format!("/api/status/{id}")) {
            Ok(s) => {
                fails = 0;
                s
            }
            Err(e) => {
                fails += 1;
                if fails < 5 {
                    std::thread::sleep(Duration::from_secs(fails));
                    continue;
                }
                let msg = sh.text.t("scan.err.lost", &v(&[("error", format!("{e:#}"))]));
                let mut s = sh.st.lock().unwrap();
                s.scan.opening = false;
                s.scan.error = msg;
                drop(s);
                sh.changed(&["scan"]);
                return;
            }
        };
        if !live() {
            return;
        }
        let ready = st["ready"].as_bool().unwrap_or(false);
        let current = st["current"].as_str().unwrap_or("").to_string();
        let opening = sh.st.lock().unwrap().scan.opening;
        let image = st["meta"]["image"].as_bool().unwrap_or(false);
        let error = st["error"].as_str().unwrap_or("").to_string();
        // "Deciding who's who": its events, from the start of the decide step to the end of the scan
        let viz_on = sh.st.lock().unwrap().scan.viz;
        let want_viz = !opening && !image && error.is_empty() && (current == "decide" || viz_on);
        if want_viz {
            let n = sh.lib.lock().unwrap().viz.len();
            if let Ok(r) = sc.get(&format!("/api/projects/{id}/decide_viz?since={n}")) {
                if let Some(evs) = r["events"].as_array() {
                    let mut l = sh.lib.lock().unwrap();
                    if l.scan_gen == gen && l.viz.len() == n {
                        l.viz.extend(evs.iter().cloned());
                    }
                }
            }
        }
        let preview_due = last_preview.elapsed() >= Duration::from_millis(800);
        if preview_due {
            last_preview = Instant::now();
        }
        scan_view(sh, sc.as_ref(), &st, want_viz, preview_due);
        if !error.is_empty() {
            return;
        }
        if ready {
            if want_viz {
                // the page shows what's arrived before the project opens (decide_viz.js drained: 4 s at most)
                sh.st.lock().unwrap().scan.finishing = true;
                sh.changed(&["scan"]);
                let t0 = Instant::now();
                while t0.elapsed() < Duration::from_secs(6) && !sh.lib.lock().unwrap().viz_drained && live() {
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
            if !live() {
                return;
            }
            let id = id.to_string();
            queue_try(sh, "open", move |sh| open_project(sh, &id));
            return;
        }
        std::thread::sleep(Duration::from_millis(if opening || current == "review" { 250 } else { 1000 }));
    }
}

/// app.html pollScan: the scan page's words and numbers from /api/status.
fn scan_view(sh: &Arc<Shared>, sc: &dyn Backend, st: &Value, viz: bool, preview_due: bool) {
    let tx = &sh.text;
    let ready = st["ready"].as_bool().unwrap_or(false);
    let current = st["current"].as_str().unwrap_or("");
    let at = STEPS.iter().position(|k| *k == current).unwrap_or(0);
    let i = &st["info"];
    let m = &st["meta"];
    let frame = i["frame"].as_f64().unwrap_or(0.0);
    let frames = i["frames"].as_f64().unwrap_or(0.0);
    let image = m["image"].as_bool().unwrap_or(false);
    let error = st["error"].as_str().unwrap_or("").to_string();
    let steps: Vec<f64> = STEPS.iter().map(|k| if ready { 1.0 } else { st["steps"][*k].as_f64().unwrap_or(0.0) }).collect();
    let left = st["time_left"].as_f64();
    let eta = left.map(|l| if l > 90.0 { tx.t("unit.min", &v(&[("n", js_round(l / 60.0))])) } else { tx.t("unit.s", &v(&[("n", l as i64)])) });
    let next: Vec<String> = STEPS.iter().skip(at + 1).take(2).map(|k| tx.s(&format!("scan.step_next.{k}"))).collect();
    let clock = |sec: f64| format!("{}:{:02}", (sec / 60.0).floor() as i64, (sec % 60.0).floor() as i64);
    let name = st["name"].as_str().map(String::from).unwrap_or_else(|| m["video"].as_str().unwrap_or("").rsplit('/').next().unwrap_or("").to_string());
    let mut file = vec![name];
    if image {
        file.push(tx.s("scan.file_photo"));
    } else if let Some(d) = m["duration"].as_f64().filter(|d| *d > 0.0) {
        file.push(tx.t("unit.s", &v(&[("n", js_round(d))])));
    }
    if let (Some(w), Some(h)) = (m["width"].as_i64(), m["height"].as_i64()) {
        file.push(format!("{w}×{h}"));
    }
    let head = |k: &str| tx.s(&format!("scan.head.{k}"));
    let headline = if !error.is_empty() {
        tx.s("scan.head.stopped")
    } else if ready {
        head("review")
    } else if STEPS.contains(&current) {
        head(current)
    } else {
        tx.s("scan.getting_ready")
    };
    let note = if ready || (st["steps"]["detect"].as_f64().unwrap_or(0.0) >= 1.0 && current == "review") {
        head("review")
    } else if current == "prepare" {
        head("prepare")
    } else if current == "detect" && frame == 0.0 {
        tx.s("scan.loading_models_html").replace("<br>", "\n")
    } else if current == "motion" {
        head("motion")
    } else {
        tx.s("scan.getting_ready")
    };
    let pct = format!("{}%", js_round(if ready { 1.0 } else { st["steps"][current].as_f64().unwrap_or(0.0) } * 100.0));
    let left_text = if ready || !error.is_empty() {
        String::new()
    } else if let Some(e) = &eta {
        tx.t("scan.left", &v(&[("eta", e)]))
    } else if ["motion", "decide", "review"].contains(&current) {
        tx.s("scan.finishing")
    } else {
        String::new()
    };
    let mut stepcap = tx.t("scan.step_of", &v(&[("n", at + 1), ("total", STEPS.len())]));
    if !next.is_empty() {
        stepcap += &format!(" · {}", tx.t("scan.next", &v(&[("steps", next.join(", "))])));
    }
    let dash = "–".to_string();
    let people_now = if frame > 0.0 { i["people_now"].as_i64().map(|n| n.to_string()).unwrap_or(dash.clone()) } else { dash.clone() };
    let stats = if image {
        vec![
            (tx.s("scan.stat.photo"), m["width"].as_i64().map(|w| format!("{w}×{}", m["height"].as_i64().unwrap_or(0))).unwrap_or("…".into())),
            (tx.s("scan.stat.people"), people_now),
        ]
    } else {
        vec![
            (
                tx.s("scan.stat.frames"),
                if frame > 0.0 { tx.t("scan.stat.frames_value", &v(&[("done", thousands(frame as i64)), ("total", thousands(frames as i64))])) } else { dash.clone() },
            ),
            (tx.s("scan.stat.speed"), if frame > 0.0 { tx.t("scan.stat.speed_value", &v(&[("fps", idstr_num(&i["fps"]))])) } else { dash.clone() }),
            (tx.s("scan.stat.people"), people_now),
        ]
    };
    let done = if frame > 0.0 && frames > 0.0 { frame / frames } else { 0.0 };
    let duration = m["duration"].as_f64().unwrap_or(0.0);
    let id = sh.st.lock().unwrap().scan.id.clone();
    let video = m["video"].as_str().unwrap_or("");
    {
        let mut s = sh.st.lock().unwrap();
        let x = &mut s.scan;
        x.file = file.join(" · ");
        x.headline = headline;
        x.note = note;
        x.pct = pct;
        x.left = left_text;
        x.steps = steps;
        x.stepcap = stepcap;
        x.stats = stats;
        x.photo = image;
        x.ready = ready;
        x.public = mm_core::privacy::is_public_clip(&sh.root(), std::path::Path::new(video));
        x.missing_video = error == "missing_video";
        x.error = if x.missing_video { tx.t("scan.err.missing_video", &v(&[("path", video)])) } else { error.clone() };
        if !error.is_empty() {
            x.opening = false;
        }
        x.timeline = done;
        x.tl_left = if duration > 0.0 { clock(done * duration) } else { String::new() };
        x.tl_right = if duration > 0.0 { clock(duration) } else { String::new() };
        x.can_cancel = !ready && current != "review" && error.is_empty();
        if let (Some(w), Some(h)) = (m["width"].as_f64(), m["height"].as_f64()) {
            if !viz && h > 0.0 {
                x.aspect = (w / h).max(0.5);
            }
        }
        if viz {
            x.aspect = 1.6;
        }
        x.viz = viz;
        x.viz_n = sh.lib.lock().unwrap().viz.len();
        let live = current == "detect" && frame > 0.0 && !image;
        x.badge = if live { tx.t("scan.badge_frame", &v(&[("frame", thousands(frame as i64)), ("total", thousands(frames as i64))])) } else { String::new() };
        // the scan writes preview.jpg from its first detected frame on: asking sooner only gets a 404
        let drawn = frame > 0.0 || (current != "prepare" && current != "detect");
        if preview_due && drawn && !viz && !ready {
            x.preview = sc.url(&format!("/api/projects/{id}/preview.jpg"));
            x.preview_n += 1;
        }
    }
    sh.changed(&["scan"]);
}

fn idstr_num(v: &Value) -> String {
    decide::idstr(v)
}

pub fn cmd_cancel_scan(sh: &Arc<Shared>) -> Result<Value> {
    let id = sh.st.lock().unwrap().scan.id.clone();
    if id.is_empty() {
        bail!("no scan to cancel");
    }
    queue_try(sh, "library", move |sh| {
        sidecar(sh)?.post(&format!("/api/projects/{id}/cancel"), &json!({}), false)?;
        leave_project(sh);
        refresh_library(sh)
    });
    Ok(json!("cancelling"))
}

/// The "Deciding who's who" events from `since` (not in the state tree: they only grow).
pub fn viz_events(sh: &Shared, since: usize) -> Value {
    let l = sh.lib.lock().unwrap();
    json!({"events": l.viz.get(since..).unwrap_or(&[]), "n": l.viz.len()})
}

// ---------------------------------------------------------------------------------------------------- the open project
pub fn cmd_open(sh: &Arc<Shared>, id: &str) -> Result<Value> {
    let card = sh.st.lock().unwrap().library.projects.iter().find(|p| p.id == id).cloned();
    let opening = card.as_ref().filter(|c| c.scanned).map(|c| c.name.clone());
    start_scan(sh, id, opening);
    Ok(json!(id))
}

fn open_project(sh: &Arc<Shared>, id: &str) -> Result<()> {
    let sc = sidecar(sh)?;
    let p = format!("/p/{id}");
    let doc = sc.get_slow(&format!("{p}/api/init"))?;
    let name = sc.get(&format!("{p}/api/formats"))?["name"].as_str().unwrap_or(id).to_string();
    let dr = sc.get_slow(&format!("{p}/api/decisions"))?;
    let meta_video = {
        // the project's video, for the Editor and the privacy rule
        let st = sc.get(&format!("/api/status/{id}")).unwrap_or(Value::Null);
        st["meta"]["video"].as_str().unwrap_or("").to_string()
    };
    let public = mm_core::privacy::is_public_clip(&sh.root(), std::path::Path::new(&meta_video));
    let mut ds = dr["decisions"].as_array().cloned().unwrap_or_default();
    decide::sort(&mut ds);
    let any_open = ds.iter().any(|d| decide::by(d) == "open");
    {
        let mut l = sh.lib.lock().unwrap();
        l.pid = id.to_string();
        l.doc = doc.clone();
        l.decisions = ds;
        l.missing = dr["missing"].as_bool().unwrap_or(false);
        l.boxes.clear();
        l.dec_ui = decide::Ui::default();
        if !any_open {
            l.dec_ui.filter = if sh.dev() { "all".into() } else { "user".into() };
        }
        l.people_ui = people::Ui::default();
        l.ed = crate::editor::Ui::default();
        l.xp = crate::export::PageUi::default();
    }
    *sh.recs.lock().unwrap() = None;
    {
        let mut st = sh.st.lock().unwrap();
        st.project = ProjectState {
            id: id.to_string(),
            open: true,
            name,
            base: sc.url(&p),
            public,
            w: doc["w"].as_f64().unwrap_or(0.0),
            h: doc["h"].as_f64().unwrap_or(0.0),
            n: doc["n"].as_u64().unwrap_or(0) as usize,
            fps: doc["fps"].as_f64().unwrap_or(30.0),
            photo: doc["photo"].as_bool().unwrap_or(false),
            video: meta_video.clone(),
            ..Default::default()
        };
        st.app.page = if any_open { "decisions".into() } else { "editor".into() };
    }
    // after a project's first scan: the finished video first (app.html startResult), once, until Export or Refine is
    // picked (edits.result "pending" -> "done"; a fresh test copy has none yet)
    let first = matches!(doc["edits"]["result"].as_str(), None | Some("pending"));
    if first {
        result_view(sh, &doc, ds_open(sh));
        sh.st.lock().unwrap().app.page = "result".into();
        let _ = sc.post(&format!("{p}/api/ops"), &json!({"ops": [{"op": "set", "path": ["result"], "value": "pending"}], "label": "Result", "undoable": false}), true);
    }
    log::info(&format!("opened project {id} ({} decisions, {})", sh.lib.lock().unwrap().decisions.len(), if any_open { "questions open" } else { "none open" }));
    redraw(sh)?;
    // its picture on the Projects page, as its covers are now (not waited for)
    let t = busiest_frame(&doc);
    let (sc2, p2) = (sc.clone(), p.clone());
    std::thread::spawn(move || {
        let _ = sc2.post(&format!("{p2}/api/thumb"), &json!({"t": t, "if_missing": true}), true);
    });
    // the Editor shows the project's video with its covers drawn from the project's draw records (records.rs)
    if !meta_video.is_empty() {
        open_editor(sh, &meta_video);
        if first {
            // the result page seeks to its frame as soon as it shows, which can be before the video above opened
            // (opening a new video starts it at frame 0): seek again now, or the page waits on frame 0 for ever
            let _ = crate::commands::seek(sh, busiest_frame(&doc) as i64);
        }
    }
    sh.changed(&["project", "app.page"]);
    let back = sh.lib.lock().unwrap().restore.take();
    if let Some((page, t)) = back {
        if !first && page != "projects" && page != "scan" && page != "result" {
            let _ = cmd_go(sh, &page);
            let _ = crate::commands::seek(sh, t as i64);
        }
    }
    Ok(())
}

/// Open the project's video in the Editor (keeping the frame shown when it's the same video), then fetch its draw
/// records and the lighting estimate's state.
pub fn open_editor(sh: &Arc<Shared>, video: &str) {
    let t = sh.st.lock().unwrap().editor.t;
    let same = sh.st.lock().unwrap().editor.video.path == std::fs::canonicalize(video).map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
    match crate::commands::open(sh, video, "") {
        Ok(_) => {
            if same {
                sh.st.lock().unwrap().editor.t = t;
            }
            sh.st.lock().unwrap().editor.project = true;
            {
                let mut l = sh.lib.lock().unwrap();
                l.editor_rev = l.doc["rev"].clone();
            }
            crate::records::refresh(sh);
            crate::editor::rebuild(sh);
            let s2 = sh.clone();
            std::thread::spawn(move || crate::editor::fetch_light(&s2));
        }
        Err(e) => log::warn(&format!("editor: {e:#}")),
    }
}

fn ds_open(sh: &Shared) -> usize {
    sh.lib.lock().unwrap().decisions.iter().filter(|d| decide::by(d) == "open").count()
}

/// The result screen (app.html's s-result): the busiest frame with a ring on each face big enough to recognise,
/// the steps, then the covered video, the styles to pick from (Portrait only if it's already chosen) and the two ways on.
pub fn result_view(sh: &Shared, doc: &Value, open: usize) {
    let tx = &sh.text;
    let t = busiest_frame(doc);
    let base = sh.st.lock().unwrap().project.base.clone();
    let rings: Vec<Value> = doc["resolved"][t]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|b| {
            let a = b.as_array()?;
            (a.get(4)? == "obs" && a[3].as_f64()? >= 24.0).then(|| {
                let d = a[3].as_f64().unwrap() * 1.35;
                json!({"x": a[1].as_f64().unwrap() - d / 2.0, "y": a[2].as_f64().unwrap() - d / 2.0, "d": d})
            })
        })
        .collect();
    let style = doc["edits"]["cover_style"].as_str().unwrap_or("emoji").to_string();
    let styles: Vec<Value> = [("emoji", "😎"), ("blur", "◌"), ("pixelate", "▦"), ("solid", "●"), ("portrait", "🖼️")]
        .iter()
        .filter(|(k, _)| *k != "portrait" || *k == style)
        .map(|(k, ic)| json!({"k": k, "icon": ic, "name": tx.s(&format!("style.{k}")), "hint": tx.s(&format!("style.{k}.hint")), "on": *k == style}))
        .collect();
    let photo = doc["photo"].as_bool().unwrap_or(false);
    let v = json!({
        "t": t, "w": doc["w"], "h": doc["h"], "raw": format!("{base}/api/frames/{t:05}.jpg"), "rings": rings,
        "steps": [tx.s("result.step.scan"), tx.s("result.step.people"), tx.s("result.step.cover")],
        "covering": tx.s("result.covering"), "done": tx.s("result.done"), "head": tx.s("result.head"), "sub": tx.s("result.sub"),
        "styles": styles, "style": style,
        "export": tx.s("result.export"), "export_sub": tx.s(if photo { "result.export_sub_photo" } else { "result.export_sub_video" }),
        "refine": tx.s("result.refine"),
        "refine_sub": if open > 0 { tx.t("result.refine_sub_questions", &v(&[("n", open)])) } else { tx.s("result.refine_sub_none") },
        "photo": photo,
    });
    sh.st.lock().unwrap().result = v;
    sh.changed(&["result"]);
}

/// The result screen's way on: Export or Refine (its questions first); edits.result becomes "done" (bookkeeping).
pub fn cmd_result_done(sh: &Arc<Shared>, to: &str) -> Result<Value> {
    let page = if to == "export" { "export" } else { "decisions" };
    queue_try(sh, "library", |sh| {
        let pid = sh.lib.lock().unwrap().pid.clone();
        let r = sidecar(sh)?.post(&format!("/p/{pid}/api/ops"), &json!({"ops": [{"op": "set", "path": ["result"], "value": "done"}], "label": "Result", "undoable": false}), true)?;
        crate::editor::took(sh, &r);
        Ok(())
    });
    cmd_go(sh, page)
}

/// The result screen's style choice (edits.cover_style, shared with Edit and Export).
pub fn cmd_result_style(sh: &Arc<Shared>, style: String) -> Result<Value> {
    crate::editor::dispatch(sh, "editor_style", &json!({"style": style}))?;
    let s2 = style.clone();
    queue(sh, move |sh| {
        let (doc, open) = (sh.lib.lock().unwrap().doc.clone(), ds_open(sh));
        result_view(sh, &doc, open);
        let _ = s2;
    });
    Ok(json!(style))
}

/// The clip's busiest frame: most faces big enough to recognise, from the middle 80% (app.html busiestFrame).
fn busiest_frame(doc: &Value) -> usize {
    let n = doc["n"].as_u64().unwrap_or(1) as usize;
    let mut best = n / 2;
    let mut score = -1.0;
    let rs = doc["resolved"].as_array().cloned().unwrap_or_default();
    for t in (n as f64 * 0.1).floor() as usize..(n as f64 * 0.9).ceil() as usize {
        let bs: Vec<f64> = rs
            .get(t)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|b| b.as_array())
            .filter(|b| b.len() > 4 && b[4] == "obs" && b[3].as_f64().unwrap_or(0.0) >= 24.0)
            .map(|b| b[3].as_f64().unwrap_or(0.0))
            .collect();
        let v = bs.len() as f64 + bs.iter().map(|s| s.min(200.0)).sum::<f64>() / 2000.0;
        if v > score {
            score = v;
            best = t;
        }
    }
    best
}

/// Leave the open project (and any scan on show): back to Projects.
pub fn leave_project(sh: &Arc<Shared>) {
    {
        let mut l = sh.lib.lock().unwrap();
        l.scan_gen += 1;
        l.pid.clear();
        l.doc = Value::Null;
        l.decisions.clear();
        l.viz.clear();
        l.ed = crate::editor::Ui::default();
    }
    *sh.recs.lock().unwrap() = None;
    {
        let mut st = sh.st.lock().unwrap();
        st.project = Default::default();
        st.scan = Default::default();
        st.decisions = Default::default();
        st.people = Default::default();
        st.app.page = "projects".into();
    }
    sh.changed(&["project", "scan", "decisions", "people", "app.page"]);
}

pub fn cmd_go(sh: &Arc<Shared>, page: &str) -> Result<Value> {
    if page == "projects" {
        // its picture as its covers are now, then the list
        let (pid, doc) = {
            let l = sh.lib.lock().unwrap();
            (l.pid.clone(), l.doc.clone())
        };
        leave_project(sh);
        queue_try(sh, "library", move |sh| {
            if !pid.is_empty() {
                let _ = sidecar(sh)?.post(&format!("/p/{pid}/api/thumb"), &json!({"t": busiest_frame(&doc)}), true);
            }
            refresh_library(sh)
        });
        return Ok(json!(page));
    }
    let open = sh.st.lock().unwrap().project.open;
    if ["decisions", "people", "export", "result"].contains(&page) && !open {
        bail!("no project open");
    }
 sh.st.lock().unwrap().app.page = page.to_string();
    sh.changed(&["app.page"]);
    if page == "editor" && open {
        // the Editor's covers follow the answers: made again when the project's edits changed
        queue_try(sh, "editor", |sh| {
            let (rev, er) = {
                let l = sh.lib.lock().unwrap();
                (l.doc["rev"].clone(), l.editor_rev.clone())
            };
            let video = sh.st.lock().unwrap().project.video.clone();
            let opened = sh.st.lock().unwrap().editor.video.path == std::fs::canonicalize(&video).map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
            if !video.is_empty() && !opened {
                open_editor(sh, &video);
            } else if rev != er || sh.lib.lock().unwrap().xp.strict {
                // (higher-risk mode on the Export page drew solid covers: the Editor draws the project's own)
                sh.lib.lock().unwrap().editor_rev = rev;
                crate::records::refresh(sh);
            }
            crate::editor::rebuild(sh);
            Ok(())
        });
    }
    if page == "export" && open {
        crate::export::page_dispatch(sh, "export_page", &json!({}))?;
    }
    if page == "decisions" || page == "people" {
        // fresh each time: answers elsewhere (the CLI, an undo) change who's who
        queue_try(sh, "decisions", reload);
    }
    Ok(json!(page))
}

/// Fetch the open project's edits and decisions again, and redraw both pages.
pub fn reload(sh: &Arc<Shared>) -> Result<()> {
    let pid = sh.lib.lock().unwrap().pid.clone();
    if pid.is_empty() {
        return Ok(());
    }
    let sc = sidecar(sh)?;
    let doc = sc.get_slow(&format!("/p/{pid}/api/init"))?;
    let dr = sc.get_slow(&format!("/p/{pid}/api/decisions"))?;
    let mut ds = dr["decisions"].as_array().cloned().unwrap_or_default();
    decide::sort(&mut ds);
    {
        let mut l = sh.lib.lock().unwrap();
        if l.pid != pid {
            return Ok(());
        }
        l.doc = doc;
        l.decisions = ds;
        l.missing = dr["missing"].as_bool().unwrap_or(false);
    }
    redraw(sh)?;
    // the Editor follows (an undo, an answer elsewhere): its boxes now, its covers once the records are in
    let (rev, er) = {
        let l = sh.lib.lock().unwrap();
        (l.doc["rev"].clone(), l.editor_rev.clone())
    };
    if rev != er && sh.st.lock().unwrap().editor.project {
        sh.lib.lock().unwrap().editor_rev = rev;
        crate::records::refresh(sh);
    }
    crate::editor::rebuild(sh);
    Ok(())
}

/// Make the Decisions and People pages' models from what the store has (fetching the boxes the selected question's
/// loops need first).
pub fn redraw(sh: &Arc<Shared>) -> Result<()> {
    let (pid, want) = {
        let l = sh.lib.lock().unwrap();
        (l.pid.clone(), decide::wanted_boxes(&l.decisions, &l.dec_ui).into_iter().filter(|k| !l.boxes.contains_key(k)).collect::<Vec<_>>())
    };
    if pid.is_empty() {
        return Ok(());
    }
    if !want.is_empty() {
        if let Ok(sc) = sidecar(sh) {
            let r = sc.get(&format!("/p/{pid}/api/boxes?ids={}", want.join(","))).unwrap_or(Value::Null);
            let mut l = sh.lib.lock().unwrap();
            for k in want {
                let m = r.get(&k).and_then(Value::as_object).map(|o| {
                    o.iter()
                        .filter_map(|(t, b)| {
                            let a = b.as_array()?;
                            Some((t.parse::<usize>().ok()?, [a.first()?.as_f64()?, a.get(1)?.as_f64()?, a.get(2)?.as_f64()?, a.get(3)?.as_f64()?]))
                        })
                        .collect::<BTreeMap<_, _>>()
                });
                l.boxes.insert(k, m);
            }
        }
    }
    let (base, dev) = {
        let st = sh.st.lock().unwrap();
        (st.project.base.clone(), sh.dev())
    };
    let tx = sh.text.clone();
    let (dec, ppl, can_undo, can_redo) = {
        let mut guard = sh.lib.lock().unwrap();
        let l = &mut *guard;
        let fps = l.doc["fps"].as_f64().unwrap_or(30.0);
        let cx = decide::Ctx {
            text: &tx,
            dev,
            fps,
            base: &base,
            w: l.doc["w"].as_f64().unwrap_or(0.0),
            h: l.doc["h"].as_f64().unwrap_or(0.0),
            doc: &l.doc,
            palettes: &l.palettes,
            boxes: &l.boxes,
            missing: l.missing,
        };
        let dec = decide::view(&l.decisions, &mut l.dec_ui, &cx);
        let pc = people::Ctx { text: &tx, base: &base, doc: &l.doc, decisions: &l.decisions, palettes: &l.palettes };
        let ppl = people::view(&l.people_ui, &pc);
        (dec, ppl, l.doc["can_undo"].as_bool().unwrap_or(false), l.doc["can_redo"].as_bool().unwrap_or(false))
    };
    {
        let mut st = sh.st.lock().unwrap();
        let pending = st.decisions.pending;
        st.project.open_count = dec.open as i64;
        st.project.people_count = ppl.n as i64;
        st.project.can_undo = can_undo;
        st.project.can_redo = can_redo;
        st.decisions = dec;
        st.decisions.pending = pending;
        st.people = ppl;
    }
    sh.changed(&["decisions", "people", "project"]);
    Ok(())
}

// ---------------------------------------------------------------------------------------------------- decisions
pub fn cmd_dec_filter(sh: &Arc<Shared>, filter: Option<String>, kind: Option<String>) -> Value {
    {
        let mut l = sh.lib.lock().unwrap();
        if let Some(f) = filter {
            l.dec_ui.filter = f;
        }
        if let Some(k) = kind {
            l.dec_ui.kind = k;
        }
        l.dec_ui.cur = 0;
    }
    queue_try(sh, "decisions", redraw);
    json!("ok")
}

/// A picture clicked in Decisions or People: that moment in Edit, with its box selected (app.html editAt).
pub fn cmd_edit_at(sh: &Arc<Shared>, t: usize, piece: String) -> Result<Value> {
    cmd_go(sh, "editor")?;
    crate::commands::seek(sh, t as i64)?;
    queue(sh, move |sh| {
        // (after the Editor has its video: open_editor runs on this thread first)
        let _ = crate::commands::seek(sh, t as i64);
        let _ = crate::editor::dispatch(sh, "editor_select", &json!({"piece": piece}));
    });
    Ok(json!({"t": t}))
}

/// Only one box's decisions (the Editor's R, People's "Why"); None: every decision. `nearby`: flip the grey-box ones.
pub fn cmd_dec_piece(sh: &Arc<Shared>, piece: Option<String>) -> Result<Value> {
    {
        let mut l = sh.lib.lock().unwrap();
        l.dec_ui.piece = piece.clone();
        l.dec_ui.nearby = false;
        if piece.is_some() {
            l.dec_ui.filter = "all".into();
            l.dec_ui.kind = "all".into();
        }
        l.dec_ui.cur = 0;
    }
    queue_try(sh, "decisions", redraw);
    Ok(json!(piece))
}

pub fn cmd_dec_nearby(sh: &Arc<Shared>) -> Value {
    {
        let mut l = sh.lib.lock().unwrap();
        l.dec_ui.nearby = !l.dec_ui.nearby;
        l.dec_ui.cur = 0;
    }
    queue_try(sh, "decisions", redraw);
    json!("ok")
}

pub fn cmd_dec_select(sh: &Arc<Shared>, i: usize) -> Value {
    sh.lib.lock().unwrap().dec_ui.cur = i;
    queue_try(sh, "decisions", redraw);
    json!(i)
}

pub fn cmd_dec_key(sh: &Arc<Shared>, key: &str) -> Result<Value> {
    let a = {
        let l = sh.lib.lock().unwrap();
        decide::key(&l.decisions, &l.dec_ui, key)
    };
    match a {
        KeyAction::Move(i) => Ok(cmd_dec_select(sh, i)),
        KeyAction::Answer(id, val) => cmd_answer(sh, id, Some(val)),
        KeyAction::None => Ok(json!(null)),
    }
}

/// Answer a decision (None: clear the answer), as the page does: POST api/decisions {id, answer}; the reply carries
/// the record as answered and the covers it leaves. Who's-who answers open the next pairs: the list is fetched again.
pub fn cmd_answer(sh: &Arc<Shared>, id: String, answer: Option<String>) -> Result<Value> {
    sh.st.lock().unwrap().decisions.pending += 1;
    sh.changed(&["decisions"]);
    queue(sh, move |sh| {
        let r = answer_now(sh, &id, answer.as_deref());
        {
            let mut st = sh.st.lock().unwrap();
            st.decisions.pending = st.decisions.pending.saturating_sub(1);
            if let Err(e) = &r {
                st.decisions.error = sh.text.t("common.err.save", &v(&[("error", format!("{e:#}"))]));
            }
        }
        if let Err(e) = r {
            log::warn(&format!("answer {id}: {e:#}"));
        }
        sh.changed(&["decisions"]);
    });
    Ok(json!("queued"))
}

fn answer_now(sh: &Arc<Shared>, id: &str, answer: Option<&str>) -> Result<()> {
    let pid = sh.lib.lock().unwrap().pid.clone();
    if pid.is_empty() {
        bail!("no project open");
    }
    let was_open = sh.lib.lock().unwrap().decisions.iter().any(|d| decide::by(d) == "open");
    let r = sidecar(sh)?.post(&format!("/p/{pid}/api/decisions"), &json!({"id": id, "answer": answer}), true)?;
    let kind = {
        let mut l = sh.lib.lock().unwrap();
        for k in ["resolved", "edits", "hidden", "persons", "rev", "can_undo", "can_redo"] {
            if let Some(x) = r.get(k) {
                l.doc[k] = x.clone();
            }
        }
        let mut kind = String::new();
        if let Some(d) = l.decisions.iter_mut().find(|d| decide::idstr(&d["id"]) == id) {
            if let (Some(o), Some(rec)) = (d.as_object_mut(), r.as_object()) {
                for (k, x) in rec {
                    if !["resolved", "edits", "hidden", "persons", "rev", "can_undo", "can_redo"].contains(&k.as_str()) {
                        o.insert(k.clone(), x.clone());
                    }
                }
                // a cleared answer: the record says so by not having one
                if r.get("by").is_none() {
                    o.remove("by");
                    o.remove("answer");
                }
            }
            kind = d["kind"].as_str().unwrap_or("").to_string();
        }
        kind
    };
    sh.st.lock().unwrap().decisions.error.clear();
    log::info(&format!("answered {id}: {}", answer.unwrap_or("(cleared)")));
    if kind == "same" || kind == "whose" {
        let dr = sidecar(sh)?.get_slow(&format!("/p/{pid}/api/decisions"))?;
        let mut ds = dr["decisions"].as_array().cloned().unwrap_or_default();
        decide::sort(&mut ds);
        sh.lib.lock().unwrap().decisions = ds;
    }
    redraw(sh)?;
    let (none_open, filter) = {
        let l = sh.lib.lock().unwrap();
        (!l.decisions.iter().any(|d| decide::by(d) == "open"), l.dec_ui.filter.clone())
    };
    if was_open && none_open && filter == "open" {
        // the last open question just got its answer: on to People, as the Next button does
        let s2 = sh.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(900));
            if s2.st.lock().unwrap().app.page == "decisions" {
                let _ = cmd_go(&s2, "people");
            }
        });
    }
    Ok(())
}

/// The selected question's picture `pic`: the frames it loops through.
pub fn cmd_loop_frames(sh: &Arc<Shared>, pic: usize) -> Value {
    let base = sh.st.lock().unwrap().project.base.clone();
    let l = sh.lib.lock().unwrap();
    let cx = decide::Ctx {
        text: &sh.text,
        dev: sh.dev(),
        fps: l.doc["fps"].as_f64().unwrap_or(30.0),
        base: &base,
        w: l.doc["w"].as_f64().unwrap_or(0.0),
        h: l.doc["h"].as_f64().unwrap_or(0.0),
        doc: &l.doc,
        palettes: &l.palettes,
        boxes: &l.boxes,
        missing: l.missing,
    };
    decide::loop_frames(&l.decisions, &l.dec_ui, &cx, pic)
}

/// Undo / redo the project's last change (its server's history, shared by every page).
pub fn cmd_undo(sh: &Arc<Shared>, redo: bool) -> Result<Value> {
    queue_try(sh, "decisions", move |sh| {
        let pid = sh.lib.lock().unwrap().pid.clone();
        if pid.is_empty() {
            bail!("no project open");
        }
        sidecar(sh)?.post(&format!("/p/{pid}/api/{}", if redo { "redo" } else { "undo" }), &json!({}), true)?;
        reload(sh)
    });
    Ok(json!("queued"))
}

// ---------------------------------------------------------------------------------------------------- people
/// What a card says about a person: show (bool), name (text), emoji. A fact about their boxes (people.html say).
pub fn cmd_people_say(sh: &Arc<Shared>, tid: String, kind: String, value: Value) -> Result<Value> {
    if !["show", "name", "emoji"].contains(&kind.as_str()) {
        bail!("people_say: kind is show, name or emoji");
    }
    queue(sh, move |sh| {
        let r = (|| -> Result<()> {
            let (pid, pieces) = {
                let st = sh.st.lock().unwrap();
                let all: Vec<people::Card> = st.people.cards.iter().chain(st.people.brief.iter()).cloned().collect();
                (sh.lib.lock().unwrap().pid.clone(), people::pieces_of(&all, &tid))
            };
            if pieces.is_empty() {
                bail!("no person {tid}");
            }
            let label = match kind.as_str() {
                "show" => if value.as_bool().unwrap_or(false) { "unmask" } else { "mask" },
                k => k,
            };
            let r = sidecar(sh)?.post(
                &format!("/p/{pid}/api/facts"),
                &json!({"facts": [{"kind": kind, "pieces": pieces, "value": value}], "label": label, "page": "people"}),
                true,
            )?;
            let mut l = sh.lib.lock().unwrap();
            if let (Some(o), Some(rec)) = (l.doc.as_object_mut(), r.as_object()) {
                for (k, x) in rec {
                    o.insert(k.clone(), x.clone());
                }
            }
            l.people_ui.errors.remove(&tid);
            if kind == "emoji" {
                l.people_ui.palette_open.remove(&tid);
            }
            Ok(())
        })();
        if let Err(e) = r {
            log::warn(&format!("people: {e:#}"));
            sh.lib.lock().unwrap().people_ui.errors.insert(tid.clone(), sh.text.t("people.err.save", &v(&[("error", format!("{e:#}"))])));
        }
        let _ = redraw(sh);
    });
    Ok(json!("queued"))
}

/// A card's open parts: "palette" (its emoji choice), "strip" (their faces), or "brief" (the brief ones listed).
pub fn cmd_people_toggle(sh: &Arc<Shared>, what: &str, tid: &str) -> Result<Value> {
    {
        let mut l = sh.lib.lock().unwrap();
        let u = &mut l.people_ui;
        let flip = |s: &mut std::collections::BTreeSet<String>| {
            if !s.remove(tid) {
                s.insert(tid.to_string());
            }
        };
        match what {
            "palette" => flip(&mut u.palette_open),
            "strip" => flip(&mut u.strip_open),
            "brief" => u.brief = !u.brief,
            _ => bail!("people_toggle: palette, strip or brief"),
        }
    }
    queue_try(sh, "people", redraw);
    Ok(json!("ok"))
}

/// Mark the sidecar as gone (tests, a crash): the pages say so.
pub fn sidecar_alive(sh: &Shared) -> bool {
    sh.lib.lock().unwrap().sc.as_ref().is_some_and(|s| s.alive())
}

pub fn viz_drained(sh: &Shared) -> Value {
    sh.lib.lock().unwrap().viz_drained = true;
    json!(true)
}
