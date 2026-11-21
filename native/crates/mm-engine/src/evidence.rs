//! The scan's evidence from pixels, measured in Rust: mm-scan's `FaceEvidence`, `Appearance` and `Depth` traits on
//! mm-analysis, reading the project's frames straight from its video in the detect colour (the detectors' own pixels;
//! [`Frames`], mm_media::frames). Until 2026-11-20 they were read from PROJECT/frames/NNNNN.jpg as the Python scan's
//! cv2.imread did.
//!
//! Kept from Python:
//! - **faces** (scan/faces.py `face`, follow.py `face_near`): RetinaFace landmarks on the head crop (reid.face_crop),
//!   the nearest confident face; `face` adds SFace for frontal faces. face_near's head box is computed in float32 as
//!   numpy does it there (the face box / k + x0, face_to_head, to_cs on float32 scalars). The per-project face cache
//!   (faces.measure's) lives here, behind FaceEvidence: `proto/faces_native.json`, keyed by frame and box (exact
//!   floats), so a refresh with the same boxes measures nothing again.
//! - **bodies** (appearance.crop + reid_embed): the scan only asks when a group has 2 or more crops (questions._looks,
//!   in mm-scan), and asks for the group's crops in one call (`Appearance::bodies`), embedded as one batch as Python
//!   does. OSNet runs on the CPU (appearance.py), and ONNX Runtime's CPU results depend on the batch a crop runs in (up
//!   to 1.8e-7 between a crop alone and in its group's batch, measured on jazz's 267 crops), so the batch matters.
//! - **depth** (depth.Depth.near): the cached grey movie and sidecar (dets/depth_<model>.*, made by [`crate::depth`]),
//!   decoded through mm-media.
//! - **motion** (flow, face_motion, trackable, looks), by the scan's motion setting: [`NativeMotion`] (mm-motion's head
//!   following, our own motion, plus follow.patch/ncc ported exactly) or replayed from a recording
//!   ([`mm_scan::evidence::replay`]). `looks` takes both patches' frame and box. Camera motion ([`NativeCamera`],
//!   mats_v2.npy) is mm-motion's camera fit, fed by the detect step's frames.
//! - Face and motion evidence answer `&self` from several threads (the join's follows run in threads, as join.py's):
//!   their caches are behind locks and keyed by the call's arguments, so answers never depend on call order.
use anyhow::{anyhow, Result};
use mm_analysis::models::{Engines, Models};
use mm_analysis::{appearance, depth as adepth, faces as afaces, reid};
use mm_core::pyjson::Py;
use mm_ml::rt::Ep;
use mm_scan::evidence::{self as sev, Appearance, Depth, FaceEvidence, FaceM, FlowM, Motion};
use mm_scan::geom::Cs;
use mm_scan::np::M23;
use mm_vision::Mat;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex};

/// The project's frames: slot t's picture straight from the video, in the detect colour (mm_media::frames).
pub use mm_media::frames::ScanFrames as Frames;

fn face_m(f: afaces::Face) -> FaceM {
    FaceM { score: f.score, yaw: f.yaw, eyes: f.eyes, px: f.px, emb: f.emb }
}

/// faces.face: mm_analysis::faces::face (its embedding normalised as numpy does, through Accelerate's sdot).
pub fn face(m: &Models, img: &Mat<u8>, c: &Cs) -> Option<afaces::Face> {
    afaces::face(m, img, c)
}

/// follow.FACE_NEAR_ENGINE (env FACE_NEAR_ENGINE, default "ane").
fn face_near_engine() -> Ep {
    Ep::parse(&std::env::var("FACE_NEAR_ENGINE").unwrap_or_else(|_| "ane".into()))
}

/// follow.face_near: (score, head box) of the most confident face near box c's centre, else (0, None).
pub fn face_near(m: &Models, img: &Mat<u8>, c: &Cs, ep: Ep) -> (f64, Option<Cs>) {
    let Some(cr) = reid::face_crop(img, c) else { return (0.0, None) };
    let f = reid::landmarks(m, &cr.img, reid::LANDMARK_SIZE, 0.3, 0.4, Some(ep));
    let Some(i) = afaces::nearest(&f.boxes, cr.k, cr.x0, cr.y0, c) else { return (0.0, None) };
    let b = f.boxes[i];
    // float32 throughout, as numpy's scalars: b / k (k a Python float, weak) + x0 (int)
    let (k, x0, y0) = (cr.k as f32, cr.x0 as f32, cr.y0 as f32);
    let fb = [b[0] / k + x0, b[1] / k + y0, b[2] / k + x0, b[3] / k + y0];
    // track.face_to_head on float32 scalars (max: the first unless the second is greater)
    let pymax = |a: f32, b: f32| if b > a { b } else { a };
    let (cx, cy) = ((fb[0] + fb[2]) / 2.0, (fb[1] + fb[3]) / 2.0);
    let s = pymax(fb[2] - fb[0], fb[3] - fb[1]) * 1.35f32;
    let dy = 0.08f32 * s;
    let h = [cx - s / 2.0, cy - s / 2.0 - dy, cx + s / 2.0, cy + s / 2.0 - dy];
    // track.to_cs -> a float32 array
    let cs = [(h[0] + h[2]) / 2.0, (h[1] + h[3]) / 2.0, pymax(h[2] - h[0], h[3] - h[1])];
    (b[4] as f64, Some([cs[0] as f64, cs[1] as f64, cs[2] as f64]))
}

fn key(vals: &[f64]) -> String {
    sev::key(vals)
}

/// FaceEvidence on mm-analysis, with the per-project cache.
pub struct RealFaces {
    pub frames: Arc<Frames>,
    pub models: Arc<Models>,
    cache_path: Option<PathBuf>,
    cache: Mutex<HashMap<String, Option<FaceM>>>,
    near: Mutex<HashMap<String, (f64, Option<Cs>)>>,
    dirty: std::sync::atomic::AtomicBool,
    /// detections measured (face) and seconds spent, over every thread
    pub measured: std::sync::atomic::AtomicUsize,
    pub micros: std::sync::atomic::AtomicU64,
}

pub const FACE_CACHE: &str = "faces_native.json";
/// 2 (2026-11-20): measured on the video's frames; version 1's were measured on PROJECT/frames' JPEG copies.
const FACE_CACHE_VERSION: i64 = 2;

impl RealFaces {
    /// `proj`: where the cache lives (proto/faces_native.json); None: no cache.
    pub fn new(frames: Arc<Frames>, models: Arc<Models>, proj: Option<&Path>) -> RealFaces {
        let cache_path = proj.map(|p| p.join("proto").join(FACE_CACHE));
        let mut cache = HashMap::new();
        if let Some(p) = cache_path.as_ref().filter(|p| p.exists()) {
            if let Ok(doc) = mm_core::pyjson::load(p) {
                if doc.get("version").as_i64() == Some(FACE_CACHE_VERSION) {
                    if let Some(d) = doc.get("faces").as_dict() {
                        for (k, v) in d.iter() {
                            cache.insert(k.to_string(), face_from_py(v));
                        }
                    }
                }
            }
        }
        RealFaces {
            frames,
            models,
            cache_path,
            cache: Mutex::new(cache),
            near: Mutex::new(HashMap::new()),
            dirty: Default::default(),
            measured: Default::default(),
            micros: Default::default(),
        }
    }

    /// Measure these detections ahead, on `threads` threads (faces.measure's thread pool: Python measures a frame's
    /// detections per thread, 8 at a time). The scan then finds them in the cache; a detection it asks about that
    /// wasn't prefetched is measured then, so the result never depends on this.
    pub fn prefetch(&mut self, dets: &[(i64, Cs)], threads: usize) {
        let todo: Vec<(String, i64, Cs)> = dets
            .iter()
            .map(|(t, c)| (key(&[*t as f64, c[0], c[1], c[2]]), *t, *c))
            .filter(|(k, _, _)| !self.cache.lock().unwrap().contains_key(k))
            .collect();
        if todo.is_empty() {
            return;
        }
        let t0 = std::time::Instant::now();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let results: Mutex<Vec<(String, Option<FaceM>)>> = Mutex::new(Vec::with_capacity(todo.len()));
        std::thread::scope(|s| {
            for _ in 0..threads.max(1) {
                s.spawn(|| loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some((k, t, c)) = todo.get(i) else { break };
                    let img = self.frames.get(*t);
                    let v = face(&self.models, &img, c).map(face_m);
                    results.lock().unwrap().push((k.clone(), v));
                });
            }
        });
        let n = todo.len();
        let mut c = self.cache.lock().unwrap();
        for (k, v) in results.into_inner().unwrap() {
            c.insert(k, v);
        }
        self.measured.fetch_add(n, Relaxed);
        self.micros.fetch_add(t0.elapsed().as_micros() as u64, Relaxed);
        self.dirty.store(true, Relaxed);
    }

    pub fn save(&mut self) -> Result<()> {
        let Some(p) = &self.cache_path else { return Ok(()) };
        if !self.dirty.load(Relaxed) {
            return Ok(());
        }
        let mut faces = mm_core::pyjson::Dict::new();
        let cache = self.cache.lock().unwrap();
        let mut keys: Vec<&String> = cache.keys().collect();
        keys.sort();
        for k in keys {
            faces.insert(k.clone(), mm_scan::faces::face_json(cache[k].as_ref()));
        }
        drop(cache);
        let mut doc = mm_core::pyjson::Dict::new();
        doc.insert("version", Py::Int(FACE_CACHE_VERSION));
        doc.insert("faces", Py::Dict(faces));
        std::fs::create_dir_all(p.parent().unwrap())?;
        let tmp = p.with_extension("json.tmp");
        std::fs::write(&tmp, Py::Dict(doc).dumps())?;
        std::fs::rename(&tmp, p)?;
        self.dirty.store(false, Relaxed);
        Ok(())
    }
}

fn face_from_py(v: &Py) -> Option<FaceM> {
    if v.is_null() {
        return None;
    }
    let f = |k: &str| v.get(k).as_f64().unwrap();
    Some(FaceM {
        score: f("score"),
        yaw: f("yaw"),
        eyes: f("eyes"),
        px: f("px"),
        emb: v.get("emb").as_list().map(|l| l.iter().map(|x| x.as_f64().unwrap() as f32).collect()),
    })
}

impl Drop for RealFaces {
    fn drop(&mut self) {
        if let Err(e) = self.save() {
            eprintln!("faces cache: {e}");
        }
    }
}

impl FaceEvidence for RealFaces {
    fn face_near(&self, t: i64, c: &Cs) -> (f64, Option<Cs>) {
        let k = key(&[t as f64, c[0], c[1], c[2]]);
        if let Some(v) = self.near.lock().unwrap().get(&k) {
            return *v;
        }
        let img = self.frames.get(t);
        let ep = if self.models.engines.landmarks == Ep::Cpu { Ep::Cpu } else { face_near_engine() };
        let v = face_near(&self.models, &img, c, ep);
        self.near.lock().unwrap().insert(k, v);
        v
    }
    fn face(&self, t: i64, c: &Cs) -> Option<FaceM> {
        let k = key(&[t as f64, c[0], c[1], c[2]]);
        if let Some(v) = self.cache.lock().unwrap().get(&k) {
            return v.clone();
        }
        let t0 = std::time::Instant::now();
        let img = self.frames.get(t);
        let v = face(&self.models, &img, c).map(face_m);
        self.micros.fetch_add(t0.elapsed().as_micros() as u64, Relaxed);
        self.measured.fetch_add(1, Relaxed);
        self.cache.lock().unwrap().insert(k, v.clone());
        self.dirty.store(true, Relaxed);
        v
    }
}

/// Appearance on mm-analysis: appearance.crop of the body box (no padding) and OSNet.
pub struct RealBody {
    pub frames: Arc<Frames>,
    pub models: Arc<Models>,
}

impl Appearance for RealBody {
    fn body(&self, t: i64, bx: &[f64; 4]) -> Vec<f32> {
        let img = self.frames.get(t);
        let c = appearance::crop(&img, bx, 0.0).expect("body crop (the scan checked it)");
        appearance::reid_embed(&self.models, &[c]).remove(0)
    }
    /// One OSNet batch for the group's crops, as questions._looks runs appearance.reid_embed.
    fn bodies(&self, crops: &[(i64, [f64; 4])]) -> Vec<Vec<f32>> {
        let cs: Vec<Mat<u8>> = crops
            .iter()
            .map(|(t, bx)| appearance::crop(&self.frames.get(*t), bx, 0.0).expect("body crop (the scan checked it)"))
            .collect();
        appearance::reid_embed(&self.models, &cs)
    }
}

// ------------------------------------------------------------------------------------------------ depth

/// The depth movie's reader (moved to mm-media).
pub use mm_media::grey::MovieLuma;

/// mm-scan's Depth evidence on mm_analysis::depth::Depth.
pub struct RealDepth {
    pub d: adepth::Depth,
}

impl RealDepth {
    /// depth.Depth(proj, name) from the cache in PROJECT/dets (which must exist).
    pub fn open(proj: &Path, name: &str, frame_hw: (usize, usize)) -> Result<RealDepth> {
        let d = adepth::Depth::open_cache(proj, name, frame_hw, |p| Box::new(MovieLuma { path: p.to_path_buf(), fps: adepth::MOVIE_FPS }))
            .map_err(|e| anyhow!(e))?;
        Ok(RealDepth { d })
    }
}

impl Depth for RealDepth {
    fn near(&mut self, t: i64, c: &Cs, f: f64) -> f64 {
        self.d.near_f(t, c, f)
    }
    fn name(&self) -> &str {
        &self.d.name
    }
    fn margins(&self) -> (f64, f64, f64) {
        (self.d.front, self.d.alone, self.d.apart)
    }
}


// ------------------------------------------------------------------------------------------------ motion, native

/// follow.CORE (the flow position's core of the box) and follow.FLOW_CORE (face_motion's): x1, x2, y1, y2 fractions.
pub const CORE: [f64; 4] = [0.2, 0.8, 0.15, 0.85];
pub const FLOW_CORE: [f64; 4] = [0.15, 0.85, 0.1, 0.9];
/// follow.TS: the side of the "looks the same" patches
pub const TS: usize = 32;

/// follow.patch: box c's square resampled to TS x TS (cv2.warpAffine, linear, black outside), as float32.
pub fn patch(g: &Mat<u8>, c: &Cs) -> Vec<f32> {
    let (ts, s) = (TS as f64, c[2]);
    // np.array([[TS / s, 0, TS / 2 - c[0] * TS / s], [0, TS / s, TS / 2 - c[1] * TS / s]], np.float32)
    let m = [ts / s, 0.0, ts / 2.0 - c[0] * ts / s, 0.0, ts / s, ts / 2.0 - c[1] * ts / s].map(|v| v as f32 as f64);
    let w = mm_vision::warp_affine_u8(g, &m, TS, TS, false, mm_vision::Border::Constant, [0.0; 4]);
    w.data.iter().map(|&v| v as f32).collect()
}

/// follow._OV: ncc's weights, an oval inside the patch (float32 0/1).
fn oval() -> &'static [f32] {
    static OV: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    OV.get_or_init(|| {
        let ts = TS as f64;
        let mut v = vec![0f32; TS * TS];
        for y in 0..TS {
            for x in 0..TS {
                let a = (x as f64 - (ts - 1.0) / 2.0) / (ts * 0.42);
                let b = (y as f64 - (ts - 1.0) / 2.0) / (ts * 0.48);
                v[y * TS + x] = if a.powi(2) + b.powi(2) <= 1.0 { 1.0 } else { 0.0 };
            }
        }
        v
    })
}

/// follow.ncc of two patches: weighted zero-mean normalised cross-correlation, float32 as numpy computes it (pairwise
/// sums, mm_core::np::sum).
pub fn ncc(a: &[f32], b: &[f32]) -> f64 {
    use mm_core::np::sum;
    let w = oval();
    let ws = sum(w);
    let aw: Vec<f32> = a.iter().zip(w).map(|(x, y)| x * y).collect();
    let bw: Vec<f32> = b.iter().zip(w).map(|(x, y)| x * y).collect();
    let (am, bm) = (sum(&aw) / ws, sum(&bw) / ws);
    let da: Vec<f32> = a.iter().zip(w).map(|(x, y)| (x - am) * y).collect();
    let db: Vec<f32> = b.iter().zip(w).map(|(x, y)| (x - bm) * y).collect();
    let sq = |v: &[f32]| sum(&v.iter().map(|x| x * x).collect::<Vec<f32>>());
    let den = (sq(&da) * sq(&db)).sqrt();
    if den as f64 > 1e-6 {
        let p: Vec<f32> = da.iter().zip(&db).map(|(x, y)| x * y).collect();
        (sum(&p) / den) as f64
    } else {
        0.0
    }
}

/// The follower's motion evidence from mm-motion (our own motion, not cv2's corners, LK and RANSAC): flow and
/// face_motion from `head::head_motion` (with follow.CORE and follow.FLOW_CORE), trackable from `head::trackable`,
/// looks from follow.patch + follow.ncc ported exactly. Frames are the video's (Frames::grey:
/// cv2.cvtColor(BGR2GRAY) (mm-vision, exact). The same swap as tools/parity/motion_bench.py's "ours_follow" variant
/// (flow's move rounded to float32, as cv2's points are). Answers are cached by their arguments.
pub struct NativeMotion {
    frames: Arc<Frames>,
    heads: Mutex<HashMap<String, mm_motion::head::HeadMotion>>,
    trk: Mutex<HashMap<(i64, i64), f64>>,
    params: mm_motion::head::HeadParams,
}

impl NativeMotion {
    pub fn new(frames: Arc<Frames>) -> NativeMotion {
        NativeMotion {
            frames,
            heads: Mutex::new(HashMap::new()),
            trk: Mutex::new(HashMap::new()),
            params: Default::default(),
        }
    }

    pub fn gray(&self, t: i64) -> Arc<Mat<u8>> {
        self.frames.grey(t)
    }

    fn head(&self, t0: i64, t1: i64, c: &Cs, core: [f64; 4]) -> mm_motion::head::HeadMotion {
        let k = format!("{}{}", key(&[t0 as f64, t1 as f64, c[0], c[1], c[2]]), key(&core));
        if let Some(h) = self.heads.lock().unwrap().get(&k) {
            return h.clone();
        }
        let (a, b) = (self.gray(t0), self.gray(t1));
        let h = mm_motion::head::head_motion(
            mm_motion::head::Gray { w: a.cols, h: a.rows, d: &a.data },
            mm_motion::head::Gray { w: b.cols, h: b.rows, d: &b.data },
            *c,
            core,
            &self.params,
        );
        self.heads.lock().unwrap().insert(k, h.clone());
        h
    }
}

impl Motion for NativeMotion {
    fn flow(&self, t0: i64, t1: i64, c: &Cs) -> Option<FlowM> {
        let h = self.head(t0, t1, c, CORE);
        let d = h.d?;
        Some(FlowM { dx: d[0] as f32 as f64, dy: d[1] as f32 as f64, n: h.n_good as i64, spread: h.spread })
    }
    fn face_motion(&self, t0: i64, t1: i64, c: &Cs) -> (Option<M23>, String) {
        use mm_motion::head::Kind;
        let h = self.head(t0, t1, c, FLOW_CORE);
        let kind = match h.kind {
            Kind::Ok => "ok",
            Kind::NoTexture => "no texture",
            Kind::Lost => "lost",
            Kind::Incoherent => "incoherent",
        };
        let m = (h.kind == Kind::Ok).then(|| [[h.m[0], h.m[1], h.m[2]], [h.m[3], h.m[4], h.m[5]]]);
        (m, kind.to_string())
    }
    fn trackable(&self, a: i64, b: i64) -> f64 {
        let k = (a.min(b), a.max(b));
        if let Some(v) = self.trk.lock().unwrap().get(&k) {
            return *v;
        }
        let (g0, g1) = (self.gray(k.0), self.gray(k.1));
        let v = mm_motion::head::trackable(
            mm_motion::head::Gray { w: g0.cols, h: g0.rows, d: &g0.data },
            mm_motion::head::Gray { w: g1.cols, h: g1.rows, d: &g1.data },
        );
        self.trk.lock().unwrap().insert(k, v);
        v
    }
    fn looks(&self, t: i64, c: &Cs, ta: i64, ca: &Cs) -> f64 {
        ncc(&patch(&self.gray(t), c), &patch(&self.gray(ta), ca))
    }
}

/// Camera motion for the whole clip with mm-motion (mats_v2.npy rows: frame t-1 -> t, identity first), fed frame by
/// frame in decode order (the detect step's frames: video_io.frames(color="detect"), as track.camera_motion reads).
pub struct NativeCamera {
    tracker: mm_motion::camera::CameraTracker,
    params: mm_motion::camera::CameraParams,
    pub rows: Vec<f64>,
}

impl Default for NativeCamera {
    fn default() -> Self {
        let p = mm_motion::camera::CameraParams::default();
        NativeCamera { tracker: mm_motion::camera::CameraTracker::new(p.clone()), params: p, rows: vec![] }
    }
}

impl NativeCamera {
    pub fn push(&mut self, bgr: &Mat<u8>) {
        let f = mm_motion::camera::prepare_bgr(bgr, &self.params);
        let (m, _) = self.tracker.push(f);
        self.rows.extend_from_slice(&m);
    }
    /// mats_v2.npy (N x 2 x 3 float64).
    pub fn save(&self, path: &Path) -> Result<()> {
        let n = self.rows.len() / 6;
        mm_vision::npy::write(path, &mm_vision::npy::Arr::F64(vec![n, 2, 3], self.rows.clone())).map_err(|e| anyhow!("{e}"))
    }
}

/// The analysis models for the scan: Python's engines (reid: landmarks "ane", SFace "gpu"; OSNet "cpu"), or all on
/// the CPU. CoreML compiles go under `cache`.
pub fn models(models_dir: &Path, cpu: bool, cache: Option<PathBuf>) -> Arc<Models> {
    Arc::new(Models::new(models_dir, if cpu { Engines::cpu() } else { Engines::default_plan() }, cache))
}

