//! The app's self-test (`Mojimask --check [DIR]`, like the Python app's desktop.py --check): the native engine scans
//! the sample clip (test_clips/jazz.mp4 under the root) into DIR/projects, then the store exports that project with its
//! covers as the Export page does (records from mm-records, drawn by mm-render, encoded by mm-media). One JSON report,
//! its last key "ok". The scan's motion is native (no Python); depth is on when the depth model is there.
use crate::Shared;
use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub fn run(sh: &Arc<Shared>, dir: Option<&str>) -> Value {
    let (dir, keep) = match dir.filter(|d| !d.is_empty()) {
        Some(d) => (PathBuf::from(d), true),
        None => (std::env::temp_dir().join(format!("mojimask-check-{}", std::process::id())), false),
    };
    let t0 = Instant::now();
    let r = check(sh, &dir);
    let mut v = match r {
        Ok(v) => v,
        Err(e) => json!({"error": format!("{e:#}")}),
    };
    let ok = v.get("error").is_none();
    v["seconds"] = json!((t0.elapsed().as_secs_f64() * 100.0).round() / 100.0);
    v["dir"] = json!(if keep { dir.to_string_lossy().into_owned() } else { "removed".into() });
    if !keep {
        let _ = std::fs::remove_dir_all(&dir);
    }
    v["ok"] = json!(ok);
    v
}

fn check(sh: &Arc<Shared>, dir: &Path) -> Result<Value> {
    let root = PathBuf::from(&sh.cfg.root);
    let clip = root.join("test_clips/jazz.mp4");
    if !clip.is_file() {
        bail!("no sample clip at {}", clip.display());
    }
    std::fs::create_dir_all(dir)?;
    let models = mm_core::paths::models(&root);
    let depth = models.join("depth-anything-v2-small/model.onnx").exists().then(|| "anything-small".to_string());
    let mut args: Vec<String> = ["--root", &root.to_string_lossy(), "--models", &models.to_string_lossy(), "--motion", "native",
        "--cache", &mm_core::paths::cache_dir(&root).to_string_lossy(), "--engine-cache", &dir.join("engine.json").to_string_lossy()]
        .iter().map(|s| s.to_string()).collect();
    if let Some(d) = &depth {
        args.extend(["--depth".into(), d.clone()]);
    }
    let scan = mm_engine::scan::Options::from_args(&args)?;
    let cfg = mm_engine::native::Config { root: root.clone(), projects: dir.join("projects"), dev: false, depth: depth.clone(), scan, ports: (9300, 9400) };
    let srv = mm_engine::native::Server::start(cfg)?;
    let client = Arc::new(mm_engine::native::Client(srv));
    sh.lib.lock().unwrap().sc = Some(client.clone());
    // the scan
    let t0 = Instant::now();
    let id = client.import(&json!({"path": clip.to_string_lossy()}))?["id"].as_str().map(String::from)
        .ok_or_else(|| anyhow!("import: no project id"))?;
    let st = loop {
        let st = client.status(&id)?;
        if let Some(e) = st["error"].as_str() {
            bail!("scan: {e}");
        }
        if st["ready"].as_bool().unwrap_or(false) {
            break st;
        }
        if t0.elapsed() > Duration::from_secs(900) {
            bail!("scan: no result in 15 min");
        }
        std::thread::sleep(Duration::from_millis(250));
    };
    let scan_s = t0.elapsed().as_secs_f64();
    let doc = client.init(&id)?;
    let covers: usize = doc["resolved"].as_array().map(|f| f.iter().map(|x| x.as_array().map_or(0, |a| a.len())).sum()).unwrap_or(0);
    // the export, with covers
    {
        let mut s = sh.st.lock().unwrap();
        s.project.fps = st["meta"]["fps"].as_f64().unwrap_or(30.0);
    }
    let out = dir.join("jazz_export.mp4");
    let t1 = Instant::now();
    let job = mm_engine::jobs::Job::default();
    let n = crate::export::run_project(sh, &job, &id, &clip.to_string_lossy(), &out.to_string_lossy(), &crate::export::Opts::default())?;
    let bytes = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    if n == 0 || bytes == 0 {
        bail!("export: {n} frames, {bytes} bytes");
    }
    Ok(json!({"root": root.to_string_lossy(), "engine": "native", "motion": "native", "depth": depth, "project": id,
              "scan_seconds": (scan_s * 100.0).round() / 100.0, "covers": covers, "export_frames": n, "export_bytes": bytes,
              "export_seconds": (t1.elapsed().as_secs_f64() * 100.0).round() / 100.0}))
}
