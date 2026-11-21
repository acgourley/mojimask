//! The scan of one video (the app's "Scan" step): port of mojimask/pipeline.py's `run`.
//!
//! Steps (each reports progress):
//!   prepare   read size/fps/rotation, extract upright frames (PROJECT/frames/NNNNN.jpg, the review screens and the
//!             scan's evidence read them)
//!   detect    one pass over the frames running every model in the profile ([`crate::detect::Core`]); writes
//!             dets/<name>.json, dets/pose_head.json and a live preview (frame + a box on each face) for the scan
//!             screen. Frames come straight from the decoder (lossless, upright, the "detect" colour), not from the
//!             JPEGs, which shift borderline detections.
//!   motion    camera motion between frames (mats_v2.npy), by the motion setting ([`Motion`]): mm-motion's camera
//!             fit on the detect step's frames as they are decoded
//!   depth     with a depth model on, the depth cache when the project has none ([`crate::depth`], native)
//!   decide    mm-scan's newalgo with this crate's evidence ([`crate::evidence`]); the follower's motion evidence
//!             from mm-motion (native) or a recording (replay, decide only)
//!   done
//!
//! Profile "core": YOLOv5-Face (tiled) + RetinaFace-R50 + pose (full keypoints -> head boxes, dets/pose_head.json).
//!
//! Differences from Python, none in what the scan reads: one decode pass makes both the JPEGs and the detections
//! (Python decodes twice: the same frames); the preview's boxes are drawn without anti-aliasing; the timed engine
//! plan is remembered in `engine_native.json` (our ONNX Runtime isn't Python's).
use crate::detect::{self, Boxes, Core, CORE};
use crate::jpeg;
use anyhow::{anyhow, Context, Result};
use mm_core::pyjson::{Dict, Py};
use mm_media::convert::Colour;
use mm_core::pyjson::float_repr as py_repr;
use mm_vision::Mat;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc};
use std::time::Instant;

pub const PREVIEW_MIN_SCORE: f64 = 0.3;
pub const HIT_SCORE: f64 = 0.5;
pub const PERSON_SCORE: f64 = 0.3;
/// meta.json keys a rescan keeps: the app's own notes and the frames' space
pub const KEEP_META: [&str; 3] = ["copied", "space", "label"];
pub const SPACE_VERSION: i64 = 2;

pub type Report<'a> = &'a mut dyn FnMut(&str, f64, serde_json::Value);

#[derive(Clone, Debug)]
pub struct Options {
    /// models/ (the onnx/ folder and the depth model's)
    pub models: PathBuf,
    /// where CoreML's compiles go (`<cache>/coreml`); None: no compile cache
    pub cache: Option<PathBuf>,
    /// the timed plan's memory (default: engine_native.json beside the projects folder)
    pub engine_cache: Option<PathBuf>,
    /// a fixed plan instead of the timed one ("cpu" for parity)
    pub plan: Option<String>,
    /// the repo root (or the bundle's Resources)
    pub root: PathBuf,
    /// MOJIMASK_DEPTH for the decide step (None: off)
    pub depth: Option<String>,
    /// the analysis models all on the CPU (parity with a CPU Python run)
    pub analysis_cpu: bool,
    /// where camera motion and the follower's motion evidence come from
    pub motion: Motion,
}

impl Options {
    /// The scan's command-line options (mm-scan-tool, mm-cli scan):
    ///   --plan cpu|default|gpu|ane  --analysis-cpu  --depth NAME|off  --motion native|replay  --replay REC
    ///   --models DIR (default $MOJIMASK_MODELS, else ROOT/models)  --cache DIR (default mm_core::paths::cache_dir)
    ///   --engine-cache FILE  --root DIR
    pub fn from_args(a: &[String]) -> Result<Options> {
        let flag = |name: &str| a.iter().position(|x| x == name).and_then(|i| a.get(i + 1).cloned());
        // (mm_core::paths: the root, MOJIMASK_MODELS, the caches; no build-machine paths in a release build)
        let root = match flag("--root") {
            Some(r) => PathBuf::from(r),
            None => mm_core::paths::root(),
        };
        let models = flag("--models").map(PathBuf::from).unwrap_or_else(|| mm_core::paths::models(&root));
        let replay = flag("--replay");
        let motion = match flag("--motion") {
            Some(m) => Motion::parse(&m, replay.as_deref())?,
            None => match &replay {
                Some(r) => Motion::Replay(PathBuf::from(r)),
                None => Motion::Native,
            },
        };
        Ok(Options {
            models,
            cache: Some(flag("--cache").map(PathBuf::from).unwrap_or_else(|| mm_core::paths::cache_dir(&root))),
            engine_cache: flag("--engine-cache").map(PathBuf::from),
            plan: flag("--plan"),
            root,
            depth: flag("--depth").filter(|d| d != "off"),
            analysis_cpu: a.iter().any(|x| x == "--analysis-cpu"),
            motion,
        })
    }
}

/// The motion setting (`--motion native|replay`).
#[derive(Clone, Debug, PartialEq)]
pub enum Motion {
    /// mm-motion: our own camera motion and head following (no Python)
    Native,
    /// the decide step's motion evidence replayed from a recording (tools/parity/scan_record.py's rec/); decide only
    Replay(PathBuf),
}

impl Motion {
    /// "native" or "replay:REC" (or "replay" with `rec`).
    pub fn parse(s: &str, rec: Option<&str>) -> Result<Motion> {
        match s {
            "native" => Ok(Motion::Native),
            "replay" => rec.map(|r| Motion::Replay(PathBuf::from(r))).ok_or_else(|| anyhow!("--motion replay needs a recording (--replay REC)")),
            s if s.starts_with("replay:") => Ok(Motion::Replay(PathBuf::from(&s[7..]))),
            s => Err(anyhow!("--motion {s}: native or replay")),
        }
    }
}

/// Python's json.dumps of a list of lists of floats (repr floats, ", " separators).
fn rows_json(rows: &[Vec<f64>]) -> String {
    let mut s = String::from("[");
    for (i, r) in rows.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push('[');
        for (j, v) in r.iter().enumerate() {
            if j > 0 {
                s.push_str(", ");
            }
            s.push_str(&py_repr(*v));
        }
        s.push(']');
    }
    s.push(']');
    s
}

/// dets/<name>.json exactly as pipeline.py's json.dump writes it.
pub fn dets_json(w: u32, h: u32, fps: f64, space: &Py, frames: &[Boxes]) -> String {
    let fr: Vec<String> = frames.iter().map(|f| rows_json(f)).collect();
    format!("{{\"w\": {w}, \"h\": {h}, \"fps\": {}, \"space\": {}, \"frames\": [{}]}}", py_repr(fps), space.dumps(), fr.join(", "))
}

fn write_atomic(path: &Path, data: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, data).with_context(|| tmp.display().to_string())?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// pipeline.draw_preview: the frame with a box on each face (the app's yellow), one box a head.
fn preview(frame: &Mat<u8>, found: &[Boxes; 3]) -> Vec<u8> {
    let mut f = frame.clone();
    let w = (frame.rows.max(frame.cols) as f64 / 640.0).round_ties_even().max(1.0) as i64;
    let mut drawn: Vec<[f64; 4]> = vec![];
    let col = [63u8, 210, 255];
    for boxes in &found[..2] {
        for b in boxes {
            let (cx, cy) = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0);
            if b[4] < PREVIEW_MIN_SCORE || drawn.iter().any(|d| d[0] <= cx && cx <= d[2] && d[1] <= cy && cy <= d[3]) {
                continue;
            }
            drawn.push([b[0], b[1], b[2], b[3]]);
            let (x1, y1, x2, y2) = (b[0] as i64, b[1] as i64, b[2] as i64, b[3] as i64);
            for y in y1 - w / 2..=y2 + w / 2 {
                for x in x1 - w / 2..=x2 + w / 2 {
                    let edge = (x - x1).abs() <= w / 2 || (x - x2).abs() <= w / 2 || (y - y1).abs() <= w / 2 || (y - y2).abs() <= w / 2;
                    if edge && x >= 0 && y >= 0 && (x as usize) < f.cols && (y as usize) < f.rows {
                        let o = (y as usize * f.cols + x as usize) * 3;
                        f.data[o..o + 3].copy_from_slice(&col);
                    }
                }
            }
        }
    }
    jpeg::encode_bgr(&f.data, f.cols, f.rows, 80)
}

/// video_io.probe + the project's meta, in Python's key order.
fn meta_for(video: &str, proj: &Path) -> Result<(Dict, mm_media::Probe)> {
    let old = proj.join("meta.json");
    let old = if old.exists() { mm_core::pyjson::load(&old).map_err(|e| anyhow!(e))? } else { Py::Dict(Dict::new()) };
    // scanned before: meta.json counts its frames (projects from before 2026-11-20 also have frames/)
    let have_frames = old.get("frames").as_i64().is_some() || proj.join("frames/00000.jpg").exists();
    let keep_fps = if have_frames { old.get("fps").as_f64() } else { None };
    let p = mm_media::probe(video, keep_fps)?;
    let mut m = Dict::new();
    m.insert("width", Py::Int(p.width as i64));
    m.insert("height", Py::Int(p.height as i64));
    m.insert("fps", Py::Float(p.fps));
    m.insert("duration", Py::Float(p.duration));
    m.insert("video_start", Py::Float(p.video_start));
    m.insert("video_duration", Py::Float(p.video_duration));
    m.insert("rotation", Py::Int(p.rotation as i64));
    m.insert("has_audio", Py::Bool(p.has_audio));
    if p.image {
        m.insert("image", Py::Bool(true));
    }
    if let Some(od) = old.as_dict() {
        for k in KEEP_META {
            if let Some(v) = od.get(k) {
                m.insert(k, v.clone());
            }
        }
    }
    let abs = std::fs::canonicalize(video).unwrap_or_else(|_| PathBuf::from(video));
    m.insert("video", Py::Str(abs.to_string_lossy().to_string()));
    if !have_frames {
        let mut s = Dict::new();
        s.insert("w", Py::Int(p.width as i64));
        s.insert("h", Py::Int(p.height as i64));
        s.insert("rot", Py::Int(p.rotation as i64));
        s.insert("v", Py::Int(SPACE_VERSION));
        m.insert("space", Py::Dict(s));
    }
    Ok((m, p))
}

fn write_meta(proj: &Path, m: &Dict) -> Result<()> {
    write_atomic(&proj.join("meta.json"), Py::Dict(m.clone()).dumps().as_bytes())
}

/// video_io.frames(color="detect")'s picture for a decoded frame: mm-media's (convert::frame_bgr: AVFoundation's 8-bit
/// NV12, through VideoToolbox's pixel transfer when the stream decodes to something else, then video_io.nv12_to_bgr
/// at the stored size, cv2.resize(INTER_AREA) to the display size, upright).
pub fn detect_frame(f: &mm_media::Frame, probe: &mm_media::Probe, tags: Option<&str>) -> Result<Mat<u8>> {
    let img = mm_media::convert::frame_bgr(f, probe, tags, Colour::Detect)?;
    Ok(Mat::from_vec(img.h as usize, img.w as usize, 3, img.data))
}

/// What the detect step measured.
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct DetectStats {
    pub frames: usize,
    pub plan: String,
    /// seconds in the detection loop (decode overlapped), and per frame
    pub seconds: f64,
    pub per_frame_ms: f64,
    /// model load / plan timing before the loop
    pub setup_seconds: f64,
    /// native camera motion (mats_v2.npy written beside detection): its thread's seconds
    pub camera_seconds: Option<f64>,
}

/// prepare + detect (pipeline.run's first two steps). Writes dets/, preview.jpg, meta.json. Until 2026-11-20 it also
/// wrote every frame to frames/NNNNN.jpg; later steps now read the video (mm_media::frames).
pub fn prepare_and_detect(video: &str, proj: &Path, o: &Options, report: Report) -> Result<DetectStats> {
    mm_media::decode::init();
    std::fs::create_dir_all(proj.join("dets"))?;
    let (mut meta, probe) = meta_for(video, proj)?;
    write_meta(proj, &meta)?;
    report("prepare", 0.0, Py::Dict(meta.clone()).to_serde());

    // one decode pass: frames for the detectors
    let (ftx, frx) = mpsc::sync_channel::<Result<Arc<Mat<u8>>>>(6);
    let v = video.to_string();
    let decoder = std::thread::Builder::new().name("scan-decode".into()).spawn(move || {
        let run = || -> Result<()> {
            if mm_media::photo::is_image(&v) {
                let img = mm_media::photo::read_image(&v)?;
                let img = mm_media::convert::Bgr { w: img.w, h: img.h, data: img.data };
                let _ = ftx.send(Ok(Arc::new(Mat::from_vec(img.h as usize, img.w as usize, 3, img.data))));
                return Ok(());
            }
            let mut d = mm_media::Decoder::open_with(&v, None)?;
            let tags = d.colour_tags().map(|s| s.to_string());
            while let Some((_, f)) = d.next()? {
                if ftx.send(Ok(Arc::new(detect_frame(&f, &d.probe, tags.as_deref())?))).is_err() {
                    break;
                }
            }
            Ok(())
        };
        if let Err(e) = run() {
            let _ = ftx.send(Err(e));
        }
    })?;

    // reading the video and detecting are one pass here, so "prepare" ends where detecting starts (as pipeline.py
    // reports it); the frame count arrives with "detect" 1.0 below
    report("prepare", 1.0, Py::Dict(meta.clone()).to_serde());
    report("detect", 0.0, serde_json::json!({"note": "loading models"}));
    let t_setup = Instant::now();
    let mut sample: Vec<Arc<Mat<u8>>> = vec![];
    let mut it = frx.into_iter();
    while sample.len() < 5 {
        match it.next() {
            Some(f) => sample.push(f?),
            None => break,
        }
    }
    let engine_cache = o.engine_cache.clone().unwrap_or_else(|| {
        let a = std::fs::canonicalize(proj).unwrap_or_else(|_| proj.to_path_buf());
        a.parent().unwrap_or(Path::new(".")).join("..").join("engine_native.json")
    });
    let onnx = o.models.join("onnx");
    let (core, engine) = match &o.plan {
        Some(p) => {
            let c = Core::new(p, &onnx, o.cache.clone())?;
            if let Some(f) = sample.first() {
                c.detect(f.clone())?; // load (+ compile), as fastest_core's first frame
            }
            (c, serde_json::json!({"plan": p, "fixed": true}))
        }
        None => detect::fastest_core(&sample, Some(&engine_cache), &onnx, o.cache.clone(), &["default", "ane"])?,
    };
    meta.insert("engine", Py::from_serde(&engine));
    write_meta(proj, &meta)?;
    let setup = t_setup.elapsed().as_secs_f64();
    report("detect", 0.0, serde_json::json!({"engine": core.plan}));

    // native camera motion on the same frames, beside the detectors (track.camera_motion reads video_io.frames)
    let (ctx, crx) = mpsc::channel::<Arc<Mat<u8>>>();
    let camera = (o.motion == Motion::Native).then(|| {
        std::thread::Builder::new().name("scan-camera".into()).spawn(move || {
            let t0 = Instant::now();
            let mut cam = crate::evidence::NativeCamera::default();
            for f in crx {
                cam.push(&f);
            }
            (cam, t0.elapsed().as_secs_f64())
        })
    });
    let camera = camera.transpose()?;
    let mut out: [Vec<Boxes>; 3] = Default::default();
    let n_hint = probe.n.max(1);
    let t0 = Instant::now();
    let mut faces_seen = 0usize;
    let mut t = 0usize;
    let mut last: Option<(Arc<Mat<u8>>, [Boxes; 3])> = None;
    for f in sample.into_iter().map(Ok).chain(it) {
        let f = f?;
        if camera.is_some() {
            let _ = ctx.send(f.clone());
        }
        let found = core.detect(f.clone())?;
        faces_seen += found[..2].iter().map(|b| b.iter().filter(|r| r[4] >= HIT_SCORE).count()).sum::<usize>();
        let people = found[2].iter().filter(|p| p[4] >= PERSON_SCORE).count();
        for (i, b) in found.iter().enumerate() {
            out[i].push(b.clone());
        }
        if t % 8 == 0 {
            write_atomic(&proj.join("preview.jpg"), &preview(&f, &found))?;
            let el = t0.elapsed().as_secs_f64();
            report("detect", (t + 1) as f64 / n_hint as f64, serde_json::json!({"frame": t + 1, "frames": n_hint,
                "fps": ((t + 1) as f64 / el * 10.0).round() / 10.0, "face_hits": faces_seen, "people_now": people}));
        }
        last = Some((f, found));
        t += 1;
    }
    drop(ctx);
    decoder.join().map_err(|_| anyhow!("decoder thread"))?;
    let mut camera_s = None;
    if let Some(c) = camera {
        let (cam, secs) = c.join().map_err(|_| anyhow!("camera thread"))?;
        cam.save(&proj.join("mats_v2.npy"))?;
        camera_s = Some(secs);
    }
    if let Some((f, found)) = &last {
        write_atomic(&proj.join("preview.jpg"), &preview(f, found))?;
    }
    let n = t;
    let secs = t0.elapsed().as_secs_f64();
    // pipeline.py's key order: ..., frames, engine
    let engine = meta.remove("engine");
    meta.insert("frames", Py::Int(n as i64));
    if let Some(e) = engine {
        meta.insert("engine", e);
    }
    write_meta(proj, &meta)?;
    report("detect", 1.0, Py::Dict(meta.clone()).to_serde());
    let (w, h, fps) = (probe.width, probe.height, probe.fps);
    let space = meta.get("space").cloned().unwrap_or(Py::Null);
    for (i, name) in CORE.iter().enumerate() {
        std::fs::write(proj.join("dets").join(format!("{name}.json")), dets_json(w, h, fps, &space, &out[i]))?;
    }
    let heads: Vec<Boxes> = out[2].iter().map(|fr| fr.iter().filter_map(|p| mm_analysis::pose_util::head_box(p)).map(|h| h.to_vec()).collect()).collect();
    std::fs::write(proj.join("dets/pose_head.json"), dets_json(w, h, fps, &space, &heads))?;
    report("detect", 1.0, serde_json::json!({"frame": t, "frames": n}));
    Ok(DetectStats { frames: t, plan: core.plan.clone(), seconds: secs, per_frame_ms: secs / t.max(1) as f64 * 1000.0, setup_seconds: setup, camera_seconds: camera_s })
}

/// The decide step with real evidence. Returns newalgo's result (plus timings) and, for parity runs, its stages and
/// viz events.
pub fn decide(proj: &Path, o: &Options) -> Result<(Dict, mm_scan::newalgo::Stages, mm_scan::viz::Viz, serde_json::Value)> {
    decide_with(proj, o, None)
}

/// [`decide`], each viz event also handed to `sink` as it happens (the app's "Deciding who's who" animation).
pub fn decide_with(proj: &Path, o: &Options, sink: Option<Box<dyn FnMut(&Py)>>) -> Result<(Dict, mm_scan::newalgo::Stages, mm_scan::viz::Viz, serde_json::Value)> {
    decide_beside(proj, o, sink, None)
}

/// [`decide_with`] while the depth step may still be running (`depth_step`): the rules wait for it where they first
/// ask about depth ([`crate::evidence::LazyDepth`]).
fn decide_beside(proj: &Path, o: &Options, sink: Option<Box<dyn FnMut(&Py)>>, depth_step: Option<crate::evidence::DepthStep>)
                 -> Result<(Dict, mm_scan::newalgo::Stages, mm_scan::viz::Viz, serde_json::Value)> {
    use crate::evidence as ev;
    let frames = Arc::new(ev::Frames::for_project(proj)?);
    let f0 = frames.get(0);
    let models = ev::models(&o.models.join("onnx"), o.analysis_cpu, o.cache.as_ref().map(|c| c.join("analysis")));
    let mut faces = ev::RealFaces::new(frames.clone(), models.clone(), Some(proj));
    // the detections faces.measure will ask about (the tracklets' boxes, as newalgo builds them), measured ahead in
    // threads; the scan itself asks one at a time
    let t_pre = Instant::now();
    if let Ok(clip) = mm_scan::follow::Clip::open(proj) {
        if let Ok(d) = mm_scan::tracklets::detections(proj) {
            let tl = mm_scan::tracklets::build(&d, &clip.mats, mm_scan::tracklets::R);
            let mut all: Vec<(i64, mm_scan::geom::Cs)> = tl.values().flat_map(|v| v.obs.iter().map(|(t, c)| (*t, *c))).collect();
            all.sort_by(|a, b| a.0.cmp(&b.0));
            faces.prefetch(&all, std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8));
        }
    }
    let prefetch_s = t_pre.elapsed().as_secs_f64();
    let mut usage = None;
    let motion_box: Box<dyn mm_scan::evidence::Motion> = match &o.motion {
        Motion::Replay(rec) => {
            let (e, u) = mm_scan::evidence::replay_counted(rec, None).map_err(|e| anyhow!(e))?;
            usage = Some(u);
            e.motion
        }
        Motion::Native => Box::new(ev::NativeMotion::new(frames.clone())),
    };
    let mut depth_waited = None;
    let depth: Option<Box<dyn mm_scan::evidence::Depth>> = match &o.depth {
        Some(name) => {
            if depth_step.is_none() && !crate::depth::cached(proj, name) {
                return Err(anyhow!("depth {name}: no cache in {}/dets", proj.display()));
            }
            let d = ev::LazyDepth::new(proj, name, (f0.rows, f0.cols), depth_step)?;
            depth_waited = Some(d.waited.clone());
            Some(Box::new(d))
        }
        None => None,
    };
    let mut evd = mm_scan::evidence::Evidence {
        motion: motion_box,
        faces: Box::new(faces),
        appearance: Box::new(ev::RealBody { frames: frames.clone(), models: models.clone() }),
        depth,
    };
    let t0 = Instant::now();
    let (res, stages, viz) = match sink {
        None => mm_scan::newalgo::run_logged(proj, &mut evd).map_err(|e| anyhow!(e))?,
        Some(s) => {
            // newalgo.run_logged with a live sink
            let mut viz = mm_scan::viz::Viz::default();
            viz.sink = Some(s);
            let r = mm_scan::newalgo::run(proj, &mut evd, &mut viz);
            viz.sink = None;
            let p = proj.join("proto").join("decide_viz.json");
            let _ = std::fs::create_dir_all(p.parent().unwrap());
            let _ = mm_core::pyjson::write_atomic(&p, &Py::List(viz.events.clone()).dumps());
            let (res, st) = r.map_err(|e| anyhow!(e))?;
            (res, st, viz)
        }
    };
    let secs = t0.elapsed().as_secs_f64();
    drop(evd); // saves the face cache
    let info = serde_json::json!({
        "seconds": secs,
        "faces_prefetch_seconds": prefetch_s,
        "stage_seconds": stages.seconds.iter().map(|(k, s)| (k.to_string(), serde_json::json!((s * 100.0).round() / 100.0))).collect::<serde_json::Map<_, _>>(),
        "frames": frames.counts(),
        "waited_for_depth_seconds": depth_waited.and_then(|w| *w.lock().unwrap()),
        "motion_replay [boundary, recorded, used]": usage.map(|u| u.report()),
    });
    Ok((res, stages, viz, info))
}

/// The whole scan of a new (or rescanned) project: Rust detection, depth and decide; motion by `o.motion` (native:
/// no Python at all).
pub fn scan(video: &str, proj: &Path, o: &Options, report: Report) -> Result<serde_json::Value> {
    scan_impl(video, proj, o, report, None)
}

/// [`scan`] with the decide step's viz events handed to `sink` as they happen (the app's scan page).
pub fn scan_live(video: &str, proj: &Path, o: &Options, report: Report, sink: Box<dyn FnMut(&Py)>) -> Result<serde_json::Value> {
    scan_impl(video, proj, o, report, Some(sink))
}

fn scan_impl(video: &str, proj: &Path, o: &Options, report: Report, sink: Option<Box<dyn FnMut(&Py)>>) -> Result<serde_json::Value> {
    if let Motion::Replay(_) = o.motion {
        return Err(anyhow!("--motion replay is for the decide step only"));
    }
    let t0 = Instant::now();
    let det = prepare_and_detect(video, proj, o, report)?;
    report("motion", 0.0, serde_json::json!({}));
    let tm = Instant::now();
    let motion_s = tm.elapsed().as_secs_f64(); // native: 0 here (the camera ran beside detection: detect.camera_seconds)
    report("motion", 1.0, serde_json::json!({}));
    // The depth step. With MOJIMASK_DEPTH_BESIDE=1 it runs beside the decide step: decide's first stages (tracklets,
    // join, faces, graph) never ask about depth, so they start now and the rules wait for the cache where they first
    // need it (evidence::LazyDepth). The result is the same either way (jazz, 8 scans: the same proto_tracks.json).
    // It is not the default: on jazz it took 11 s off the best scan (53.3 -> 42.4 s) and 5 s off the median of 8
    // and 9 (76.0 -> 70.9 s), but 2 of the 9 took 190 s and 262 s, on a machine busy with other GPU work -- to be
    // measured again on a quiet one before it is turned on. (The scan page has no depth step of its own: it shows
    // "getting ready" for as long as this takes.)
    let mut depth_step = None;
    let depth_took = Arc::new(std::sync::Mutex::new(None::<f64>));
    if let Some(name) = o.depth.clone().filter(|n| !crate::depth::cached(proj, n)) {
        fn make(proj: &Path, name: &str, o: &Options, progress: &mut dyn FnMut(f64)) -> Result<f64> {
            let td = Instant::now();
            let m = crate::depth::models(&o.models, o.analysis_cpu, o.cache.as_ref().map(|c| c.join("analysis")));
            crate::depth::compute(proj, name, &m, &o.models, progress)?;
            Ok(td.elapsed().as_secs_f64())
        }
        if std::env::var_os("MOJIMASK_DEPTH_BESIDE").is_some() {
            let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let (stop2, proj2, o2, took) = (stop.clone(), proj.to_path_buf(), o.clone(), depth_took.clone());
            let thread = std::thread::Builder::new().name("mm-depth".into()).spawn(move || -> Result<f64> {
                let secs = make(&proj2, &name, &o2, &mut |_| {
                    if stop2.load(std::sync::atomic::Ordering::Relaxed) {
                        panic!("depth: the scan was stopped");
                    }
                })?;
                *took.lock().unwrap() = Some(secs);
                Ok(secs)
            })?;
            depth_step = Some(crate::evidence::DepthStep { thread, stop });
        } else {
            report("depth", 0.0, serde_json::json!({}));
            let secs = make(proj, &name, o, &mut |f| report("depth", f, serde_json::json!({})))?;
            *depth_took.lock().unwrap() = Some(secs);
        }
    }
    report("decide", 0.0, serde_json::json!({}));
    let (res, _, _, info) = decide_beside(proj, o, sink, depth_step)?;
    let depth_s = *depth_took.lock().unwrap();
    let secs = t0.elapsed().as_secs_f64();
    report("done", 1.0, serde_json::json!({"seconds": secs.round()}));
    Ok(serde_json::json!({"detect": det, "motion": format!("{:?}", o.motion), "motion_seconds": motion_s, "depth_seconds": depth_s,
        "decide": info, "result": Py::Dict(res).to_serde(), "seconds": secs}))
}
