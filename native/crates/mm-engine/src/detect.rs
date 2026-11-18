//! Detection: the three core models on ONNX Runtime (port of mojimask/onnx_core.py's Core, plans and CoreML cache).
//!
//! From onnx_core's docstring (kept true here):
//! - Working canvas: every frame is resized so its long side is CANVAS (1280) px before the face models see it, so
//!   the cost per frame is the same for a 360p clip and a 4K one (the 360x640 test clip is upscaled 2x). Model input
//!   shapes are pinned to that canvas when a session opens; CoreML needs static shapes (RetinaFace with a free input
//!   size gives wrong boxes on the GPU path and is 3x slower).
//! - Engines (Apple; other platforms use "cpu"): "gpu" CoreML ML Program, CPU+GPU, fp32 (fastest on big GPUs);
//!   "ane" CoreML NeuralNetwork, CPU+Neural Engine (fp16, about the same speed on every M-series chip); "cpu" plain
//!   ONNX Runtime CPU. The models run concurrently, so a plan that splits them across GPU and Neural Engine uses both
//!   at once.
//! - Models (models/onnx/): yolov5m-face.onnx (YOLOv5-Face-M, GPL-3.0, fixed 640x640: whole frame + 640 tiles over
//!   the canvas, merged by NMS 0.5), retinaface-r50.onnx (RetinaFace ResNet-50, MIT, on the canvas), yolo11x-pose.onnx
//!   (YOLO11x-pose, AGPL-3.0, Ultralytics rect letterbox at 1280). Pre/post-processing follows uniface (face
//!   models) and Ultralytics (pose).
//! - Output formats: face detectors [x1, y1, x2, y2, score] in source pixels; pose_full
//!   [x1, y1, x2, y2, score, k0x, k0y, k0c, ..., k16x, k16y, k16c] (COCO-17, keypoints rounded to 2 decimals).
//!
//! Exactness: pixels through mm-vision (bit-exact with the project's cv2), post-processing through mm-ml (numpy's
//! float widths, its argsort order). With the same execution provider the boxes equal Python's
//! (native/tools/parity/detect_dets.py).
//!
//! **Plans and the timed choice** (onnx_core.fastest_core): each plan is timed on a few real frames and the fastest
//! kept, remembered per Mac model, ONNX Runtime version and plan table in a JSON file (ours: `engine_native.json`
//! beside the projects folder, since our ONNX Runtime differs from Python's). Too few frames to time (a photo): the
//! remembered plan, else the default.
//!
//! **CoreML cache** (onnx_core._coreml_cache): CoreML's compiled copy of each model goes in a folder keyed by the
//! model file (name, size and a hash of its first and last MiB: builds copy the models with new dates), its pinned
//! input sizes, the engine and the ONNX Runtime version, so a changed model never loads a stale compile. Compiles not
//! used for PRUNE_DAYS are deleted (each use marks its folder). Our own base folder (the native app's compiles aren't
//! Python's), `<base>/coreml`; a cache that won't load is deleted and compiled afresh.
use anyhow::{anyhow, Result};
use mm_ml::post;
use mm_ml::rt::{self, Ep};
use mm_vision::{resize_u8, Interp, Mat};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

pub const CANVAS: usize = 1280;
pub const PRUNE_DAYS: u64 = 14;
/// The models' names in dets/ (pipeline.CORE), in Core's order.
pub const CORE: [&str; 3] = ["yolov5m_face_tiled", "retinaface_r50_2x", "pose_full"];
/// What our ONNX Runtime is (the plan and CoreML caches are keyed by it).
pub const ORT_VERSION: &str = "onnxruntime-1.31/ort-2.0.0-rc.15";

/// onnx_core.PLANS: model -> engine.
pub const PLANS: [(&str, [Ep; 3]); 4] = [
    ("default", [Ep::Gpu, Ep::Ane, Ep::Gpu]),
    ("gpu", [Ep::Gpu, Ep::Gpu, Ep::Gpu]),
    ("ane", [Ep::Ane, Ep::Ane, Ep::Ane]),
    ("cpu", [Ep::Cpu, Ep::Cpu, Ep::Cpu]),
];

pub fn plan(name: &str) -> Option<[Ep; 3]> {
    PLANS.iter().find(|(n, _)| *n == name).map(|(_, p)| *p)
}

/// CoreML is there on a Mac; elsewhere every plan is "cpu" (onnx_core.default_plan).
pub fn coreml_available() -> bool {
    cfg!(target_os = "macos")
}

pub fn default_plan() -> String {
    if !coreml_available() {
        return "cpu".into();
    }
    std::env::var("MOJIMASK_PLAN").ok().filter(|p| plan(p).is_some()).unwrap_or_else(|| "default".into())
}

// ------------------------------------------------------------------------------------------------ CoreML cache

/// FNV-1a 64: our cache key's hash (any stable hash does; md5 isn't in the workspace).
fn fnv(data: &[u8], mut h: u64) -> u64 {
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
const FNV0: u64 = 0xcbf29ce484222325;

/// The folder for CoreML's compiled copy of a model (module doc), made and marked used; old ones pruned.
pub fn coreml_cache(base: &Path, model: &Path, ep: Ep, dims: &[(String, i64)]) -> Result<PathBuf> {
    use std::io::{Read, Seek, SeekFrom};
    let size = std::fs::metadata(model)?.len();
    let mut f = std::fs::File::open(model)?;
    let mut head = vec![0u8; (1 << 20).min(size as usize)];
    f.read_exact(&mut head)?;
    f.seek(SeekFrom::Start(size.saturating_sub(1 << 20)))?;
    let mut tail = vec![];
    f.read_to_end(&mut tail)?;
    let content = fnv(&tail, fnv(&head, FNV0));
    let mut dims = dims.to_vec();
    dims.sort();
    let name = model.file_name().unwrap().to_string_lossy().to_string();
    let key = format!("{name}|{size}|{content:016x}|{dims:?}|{}|{ORT_VERSION}", ep.name());
    let stem = model.file_stem().unwrap().to_string_lossy().to_string();
    let base = base.join("coreml");
    let d = base.join(format!("{stem}-{:016x}", fnv(key.as_bytes(), FNV0)));
    std::fs::create_dir_all(&d)?;
    // mark it used (the folder's mtime), then prune the ones unused for PRUNE_DAYS
    let _ = std::fs::File::open(&d).and_then(|h| h.set_modified(SystemTime::now()));
    let old = SystemTime::now() - Duration::from_secs(PRUNE_DAYS * 86400);
    if let Ok(rd) = std::fs::read_dir(&base) {
        for e in rd.flatten() {
            let p = e.path();
            if p != d && p.is_dir() && e.metadata().and_then(|m| m.modified()).is_ok_and(|t| t < old) {
                let _ = std::fs::remove_dir_all(&p);
            }
        }
    }
    Ok(d)
}

/// onnx_core.session: CPU, or CoreML with its cache (a cache that won't load: deleted, compiled afresh).
fn open(env: &ort::environment::Environment, model: &Path, ep: Ep, dims: &[(String, i64)], cache: Option<&Path>) -> Result<ort::session::Session> {
    if ep == Ep::Cpu || !coreml_available() {
        return rt::session(env, model, Ep::Cpu, dims, None).map_err(|e| anyhow!("{}: {e}", model.display()));
    }
    let dir = match cache {
        Some(b) => Some(coreml_cache(b, model, ep, dims)?),
        None => None,
    };
    match rt::session(env, model, ep, dims, dir.as_deref()) {
        Ok(s) => Ok(s),
        Err(e) => {
            eprintln!("detect: {} would not load with its CoreML cache ({e}); compiling afresh", model.display());
            if let Some(d) = &dir {
                let _ = std::fs::remove_dir_all(d);
            }
            rt::session(env, model, ep, dims, None).map_err(|e| anyhow!("{}: {e}", model.display()))
        }
    }
}

// ------------------------------------------------------------------------------------------------ pre-processing

/// onnx_core.to_canvas: long side -> CANVAS (cubic up, area down). Returns (canvas, k = canvas px per source px).
pub fn to_canvas(frame: &Mat<u8>) -> (Mat<u8>, f64) {
    let (h, w) = (frame.rows, frame.cols);
    let k = CANVAS as f64 / h.max(w) as f64;
    if (k - 1.0).abs() < 1e-6 {
        return (frame.clone(), 1.0);
    }
    let (nw, nh) = ((w as f64 * k).round_ties_even() as usize, (h as f64 * k).round_ties_even() as usize);
    (resize_u8(frame, nw, nh, 0.0, 0.0, if k > 1.0 { Interp::Cubic } else { Interp::Area }), k)
}

pub struct Letterboxed {
    pub shape: [usize; 4],
    pub x: Vec<f32>,
    pub s: f64,
    pub pw: i64,
    pub ph: i64,
}

/// onnx_core.letterbox: fit size (keep aspect, linear), pad 114 centred; rect=false pads to size x size (uniface),
/// rect=true only to a multiple of 32 (Ultralytics). NCHW RGB float / 255.
pub fn letterbox(img: &Mat<u8>, size: usize, rect: bool) -> Letterboxed {
    let stride = 32usize;
    let (h, w) = (img.rows as f64, img.cols as f64);
    let s = (size as f64 / h).min(size as f64 / w);
    let (nh, nw, hh, ww, ph, pw);
    if rect {
        nh = (h * s).round_ties_even() as usize;
        nw = (w * s).round_ties_even() as usize;
        hh = nh + (size - nh) % stride;
        ww = nw + (size - nw) % stride;
        ph = (((hh - nh) as f64) / 2.0 - 0.1).round_ties_even() as usize;
        pw = (((ww - nw) as f64) / 2.0 - 0.1).round_ties_even() as usize;
    } else {
        nh = (h * s) as usize;
        nw = (w * s) as usize;
        hh = size;
        ww = size;
        ph = (size - nh) / 2;
        pw = (size - nw) / 2;
    }
    let r = resize_u8(img, nw, nh, 0.0, 0.0, Interp::Linear);
    let mut x = vec![114.0f32 / 255.0; 3 * hh * ww];
    let plane = hh * ww;
    for y in 0..nh {
        for xx in 0..nw {
            let p = &r.data[(y * nw + xx) * 3..(y * nw + xx) * 3 + 3];
            let o = (y + ph) * ww + xx + pw;
            x[o] = p[2] as f32 / 255.0;
            x[plane + o] = p[1] as f32 / 255.0;
            x[2 * plane + o] = p[0] as f32 / 255.0;
        }
    }
    Letterboxed { shape: [1, 3, hh, ww], x, s, pw: pw as i64, ph: ph as i64 }
}

/// onnx_core.retina_input: BGR minus (104, 117, 123), NCHW float32.
pub fn retina_input(img: &Mat<u8>) -> Vec<f32> {
    let plane = img.rows * img.cols;
    let mut x = vec![0f32; 3 * plane];
    let mean = [104.0f32, 117.0, 123.0];
    for i in 0..plane {
        for c in 0..3 {
            x[c * plane + i] = img.data[i * 3 + c] as f32 - mean[c];
        }
    }
    x
}

/// numpy's `big[y:y+T, x:x+T]` (clipped at the edges).
fn tile(big: &Mat<u8>, x: usize, y: usize, t: usize) -> Mat<u8> {
    let (w, h) = (t.min(big.cols - x), t.min(big.rows - y));
    big.roi(x, y, w, h)
}

// ------------------------------------------------------------------------------------------------ the models

/// One ONNX session per input shape (onnx_core._Shaped).
struct Shaped {
    path: PathBuf,
    ep: Ep,
    s: HashMap<Vec<usize>, ort::session::Session>,
}

impl Shaped {
    fn run(&mut self, env: &ort::environment::Environment, cache: Option<&Path>, shape: &[usize], x: Vec<f32>, dims: &[(String, i64)]) -> Result<Vec<(Vec<usize>, Vec<f32>)>> {
        if !self.s.contains_key(shape) {
            let s = open(env, &self.path, self.ep, dims, cache)?;
            self.s.insert(shape.to_vec(), s);
        }
        rt::run(self.s.get_mut(shape).unwrap(), shape, x).map_err(|e| anyhow!("{}: {e}", self.path.display()))
    }
}

/// A detection result: one row per box (see the module doc for the layouts).
pub type Boxes = Vec<Vec<f64>>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    YoloFace,
    RetinaFace,
    Pose,
}

impl Kind {
    pub fn file(self) -> &'static str {
        match self {
            Kind::YoloFace => "yolov5m-face.onnx",
            Kind::RetinaFace => "retinaface-r50.onnx",
            Kind::Pose => "yolo11x-pose.onnx",
        }
    }
}

/// One model with its pre/post-processing (onnx_core.YoloFace, RetinaFace, Pose).
pub struct Model {
    kind: Kind,
    m: Shaped,
    env: ort::environment::Environment,
    cache: Option<PathBuf>,
    pri: HashMap<(usize, usize), Vec<[f32; 4]>>,
}

impl Model {
    pub fn new(kind: Kind, models_dir: &Path, ep: Ep, cache: Option<PathBuf>) -> Model {
        Model { kind, m: Shaped { path: models_dir.join(kind.file()), ep, s: HashMap::new() }, env: rt::env(), cache, pri: HashMap::new() }
    }

    fn yolo_detect(&mut self, img: &Mat<u8>) -> Result<Vec<[f64; 5]>> {
        let lb = letterbox(img, 640, false);
        let o = self.m.run(&self.env, self.cache.as_deref(), &lb.shape, lb.x, &[])?;
        Ok(post::yolo_face(&o[0].1, 0.05, 0.4, lb.s, lb.pw, lb.ph))
    }

    pub fn run(&mut self, frame: &Mat<u8>, big: &Mat<u8>, k: f64) -> Result<Boxes> {
        match self.kind {
            Kind::YoloFace => {
                // whole frame (catches big faces) + overlapping 640 tiles over the canvas, merged by NMS 0.5
                let mut out: Vec<[f64; 5]> = self.yolo_detect(frame)?;
                let t = 640usize;
                let grid = |n: usize| {
                    let m = n.saturating_sub(t);
                    let mut v: Vec<usize> = (0..m).step_by(t / 2).collect();
                    v.push(0);
                    v.push(m);
                    v.sort();
                    v.dedup();
                    v
                };
                for y in grid(big.rows) {
                    for x in grid(big.cols) {
                        for b in self.yolo_detect(&tile(big, x, y, t))? {
                            out.push([(b[0] + x as f64) / k, (b[1] + y as f64) / k, (b[2] + x as f64) / k, (b[3] + y as f64) / k, b[4]]);
                        }
                    }
                }
                if out.is_empty() {
                    return Ok(vec![]);
                }
                let bb: Vec<[f64; 4]> = out.iter().map(|b| [b[0], b[1], b[2], b[3]]).collect();
                let sc: Vec<f64> = out.iter().map(|b| b[4]).collect();
                Ok(post::nms(&bb, &sc, 0.5).into_iter().map(|i| out[i].to_vec()).collect())
            }
            Kind::RetinaFace => {
                let (h, w) = (big.rows, big.cols);
                let dims = vec![("batch_size".to_string(), 1), ("height".to_string(), h as i64), ("width".to_string(), w as i64)];
                let o = self.m.run(&self.env, self.cache.as_deref(), &[1, 3, h, w], retina_input(big), &dims)?;
                let pri = self.pri.entry((h, w)).or_insert_with(|| post::anchors(h, w));
                Ok(post::retina(&o[0].1, &o[1].1, pri, w, h, k, 0.05, 0.4).into_iter().map(|b| b.to_vec()).collect())
            }
            Kind::Pose => {
                let lb = letterbox(frame, 1280, true);
                let dims = vec![("batch".to_string(), 1), ("height".to_string(), lb.shape[2] as i64), ("width".to_string(), lb.shape[3] as i64)];
                let shape = lb.shape;
                let o = self.m.run(&self.env, self.cache.as_deref(), &shape, lb.x, &dims)?;
                Ok(post::pose(&o[0].1, o[0].0[2], 0.1, 0.5, lb.s, lb.pw, lb.ph))
            }
        }
    }
}

type Job = (Arc<Mat<u8>>, Arc<(Mat<u8>, f64)>);

/// onnx_core.Core: the three models, each on its own worker thread (Python's ThreadPoolExecutor(3)), so one frame's
/// models run at once and a plan that splits them across GPU and Neural Engine uses both.
pub struct Core {
    pub plan: String,
    tx: Vec<mpsc::Sender<Job>>,
    rx: Vec<mpsc::Receiver<Result<Boxes>>>,
}

impl Core {
    pub fn new(plan_name: &str, models_dir: &Path, cache: Option<PathBuf>) -> Result<Core> {
        let p = plan(plan_name).ok_or_else(|| anyhow!("no plan {plan_name}"))?;
        let mut tx = vec![];
        let mut rx = vec![];
        for (i, kind) in [Kind::YoloFace, Kind::RetinaFace, Kind::Pose].into_iter().enumerate() {
            let (jt, jr) = mpsc::channel::<Job>();
            let (rt_, rr) = mpsc::channel::<Result<Boxes>>();
            let (dir, cache, ep) = (models_dir.to_path_buf(), cache.clone(), p[i]);
            std::thread::Builder::new().name(format!("detect-{}", CORE[i])).spawn(move || {
                let mut m = Model::new(kind, &dir, ep, cache);
                for (frame, big) in jr {
                    let r = m.run(&frame, &big.0, big.1);
                    if rt_.send(r).is_err() {
                        break;
                    }
                }
            })?;
            tx.push(jt);
            rx.push(rr);
        }
        Ok(Core { plan: plan_name.to_string(), tx, rx })
    }

    /// One frame through every model: [yolov5m_face_tiled, retinaface_r50_2x, pose_full].
    pub fn detect(&self, frame: Arc<Mat<u8>>) -> Result<[Boxes; 3]> {
        let big = Arc::new(to_canvas(&frame));
        for t in &self.tx {
            t.send((frame.clone(), big.clone())).map_err(|_| anyhow!("detect worker gone"))?;
        }
        let mut out: [Boxes; 3] = Default::default();
        for (i, r) in self.rx.iter().enumerate() {
            out[i] = r.recv().map_err(|_| anyhow!("detect worker gone"))??;
        }
        Ok(out)
    }
}

// ------------------------------------------------------------------------------------------------ fastest_core

pub fn machine_model() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "hw.model"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

fn plans_key() -> String {
    let s: String = PLANS.iter().map(|(n, p)| format!("{n}:{}/{}/{};", p[0].name(), p[1].name(), p[2].name())).collect();
    format!("{:012x}", fnv(s.as_bytes(), FNV0) & 0xffff_ffff_ffff)
}

/// onnx_core.fastest_core: time each plan on a few real frames, keep the fastest, remember it per machine in `cache`
/// (JSON). Returns (Core, info).
pub fn fastest_core(sample: &[Arc<Mat<u8>>], cache: Option<&Path>, models_dir: &Path, coreml: Option<PathBuf>, plans: &[&str]) -> Result<(Core, serde_json::Value)> {
    use serde_json::json;
    if !coreml_available() {
        return Ok((Core::new("cpu", models_dir, coreml)?, json!({"plan": "cpu"})));
    }
    let key = json!({"machine": machine_model(), "ort": ORT_VERSION, "plans": plans_key()});
    if let Some(c) = cache.filter(|c| c.exists()) {
        if let Ok(v) = std::fs::read(c).map_err(|e| e.to_string()).and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).map_err(|e| e.to_string())) {
            let same = ["machine", "ort", "plans"].iter().all(|k| v.get(*k) == key.get(*k));
            if let Some(p) = v.get("plan").and_then(|p| p.as_str()).filter(|p| same && plan(p).is_some()) {
                let mut info = v.clone();
                info["cached"] = json!(true);
                return Ok((Core::new(p, models_dir, coreml)?, info));
            }
        }
    }
    if sample.len() < 3 {
        let p = default_plan();
        return Ok((Core::new(&p, models_dir, coreml)?, json!({"plan": p, "measured": false})));
    }
    let mut best: Option<(Core, f64)> = None;
    let mut fps = serde_json::Map::new();
    for &p in plans {
        let core = Core::new(p, models_dir, coreml.clone())?;
        core.detect(sample[0].clone())?; // load + compile
        let t0 = Instant::now();
        for f in &sample[1..] {
            core.detect(f.clone())?;
        }
        let dt = t0.elapsed().as_secs_f64();
        let v = if dt > 0.0 { ((sample.len() - 1) as f64 / dt * 100.0).round() / 100.0 } else { 0.0 };
        fps.insert(p.to_string(), json!(v));
        if best.as_ref().is_none_or(|b| v > b.1) {
            best = Some((core, v));
        }
    }
    let (core, _) = best.unwrap();
    let mut info = json!({"plan": core.plan, "fps": fps});
    for k in ["machine", "ort", "plans"] {
        info[k] = key[k].clone();
    }
    if let Some(c) = cache {
        let tmp = c.with_extension("json.tmp");
        if std::fs::write(&tmp, serde_json::to_vec(&info)?).is_ok() {
            let _ = std::fs::rename(&tmp, c);
        }
    }
    Ok((core, info))
}
