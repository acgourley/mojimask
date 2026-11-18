//! Depth: how near the camera each part of a frame is -- a cue for "who is in front". Off by default from a
//! checkout; the Mac app (desktop.py, desktop.APP_DEPTH) turns anything-small on.
//!
//! ```text
//! MOJIMASK_DEPTH=off | anything-small | anything-base | pro     (environment; the scan and refresh.py read it)
//! ```
//!
//! The scan's behind rule (scan/behind.py) decides a held cover is hidden behind someone from skeletons:
//! the box sits on another person's body, and that person is clearly nearer (shoulders wider than the face) or the
//! face lost its own skeleton. With a depth model on, "nearer" is measured instead: the box's middle against the
//! same person's face where they were detected (see Depth.near), and a box that reads much nearer than their face
//! is hidden even without a skeleton over it (the dancer's arm when the pose detector missed it).
//!
//! Models (models/README.md has sources and hashes):
//!   anything-small  Depth Anything V2 Small, ONNX on onnxruntime (the Mac's GPU, 32-59 ms a 518 px frame). Apache-2.0.
//!   anything-base   Depth Anything V2 Base, ONNX. CC-BY-NC-4.0: non-commercial, not shippable.
//!   pro             Apple Depth Pro (transformers + torch on the Mac's GPU, ~1.4 s a frame). Apple's research model
//!                   licence: research only, not shippable. Needs experiments/requirements.txt.
//! Only anything-small is packaged (packaging/build_mac.sh, since 2026-11-14); the non-shippable ones never will
//! be: they're for trying out, in case their terms change. (This port runs the ONNX ones; "pro" needs torch and is
//! not ported.)
//!
//! Measured 2026-11-12 with anything-small: on jazz depth agrees with the flow rule and sees the dancer over seated
//! faces; on dark handheld party clips visible faces' own depth wanders, so a low threshold would uncover them.
//! Depth can't see a hand or fan held at a face (same depth).
//!
//! Each model's maps are computed once per project and cached as a grey movie, PROJECT/dets/depth_<model>.mp4, at
//! the model's own resolution (nearness: inverse depth, larger = nearer; each frame's range in 8 bits), with
//! depth_<model>.json beside it: per frame, how to turn the decoded bytes back into nearness (fitted on that frame
//! after decoding) and the frame's nearness percentiles. 2026-11-13: an H.264 movie at 12 Mb/s is ~4x smaller than
//! the 256 px float16 array it replaced and ~15-20x more accurate at depth edges. The scan's readings use it shrunk
//! to CACHE_SIDE (Depth.maps); drawing uses it whole (Depth.full).
//!
//! ## Partial occlusion
//! The part of a cover that's behind something nearer is drawn as the video, not the emoji. Jazz's dancer reaching
//! across a seated face: the emoji was drawn whole on top of her arm. With a depth model on, the drawing (gpu_draw's
//! shader, or track.draw_frame on the CPU) shows the frame's own pixels where the depth map reads clearly nearer
//! than that face does around then (Occlusion.face). Occlusion.plan decides, once per frame, which covers are cut
//! and against what; both drawing paths use it. Safety, since a wrong cut shows a face:
//!   - a face detected in this frame needs the model's "alone" gap, a held one its "front" gap, its rim only
//!     "apart" (FACE_ZONE); drawn covers (manual) are never cut;
//!   - the occluder is trimmed by ~1/CACHE_SIDE of the frame (the map eroded), so the emoji runs a little past
//!     the edge;
//!   - pixels under any other cover are never restored (a nearer person's own face stays under their own emoji);
//!   - a cover is cut only when at least OCC_MIN of its square is in front in a majority of frames t-OCC_STEADY ..
//!     t+OCC_STEADY: brief flickers of the depth model don't cut, or uncut, an emoji (on/off switches, 2026-11-13:
//!     jazz 384 -> 269, party_clip_c 329 -> 263, IMG_1043 1993 -> 1345 with a 5-frame window);
//!   - not when most of its face zone reads in front unless another, nearer person is within reach (OCC_SELF,
//!     REACH: otherwise it's the person's own head or arm).
//! Edges: the shader brings the map to full size with a joint bilateral upsample; the CPU fallback with a guided
//! filter. (The CPU fallback, Occlusion.front, and the coloured depth picture, Occlusion.picture, only draw: they
//! are mm-render's, not ported here.)
//!
//! ## Port notes
//! Port of mojimask/depth.py with numpy's dtypes (NEP 50): the model input is normalised in float64 then cast,
//! Depth.maps are built in float64 (decoded bytes times the frame's float64 fit), area-resized in float64 and
//! stored as float32; Occlusion.map is float64 (mm_vision's `resize_f64` INTER_AREA and `erode` on f64, both exact
//! against cv2 in tools/parity/vision_check.py). The movie (writing and decoding) is mm-media's: [`compute`] and [`Depth`] take
//! it through the [`GreyMovie`] and [`LumaSource`] traits. Parity: native/tools/parity/analysis_depth.py.
use crate::models::Models;
use crate::np;
use mm_ml::rt::Ep;
use mm_vision::{erode, resize_f64, resize_u8, swap_rb, Interp, Mat};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Mutex;

/// Thresholds per model, in Depth.near units, set from tools/bench/depth_bench.py (2026-11-12):
///   front  a box on another skeleton's body reading this much nearer than its face does around then is behind them
///   alone  a box reading this much nearer is behind someone even with no skeleton over it
///   apart  two people reading this far apart are different people (same-person questions answered by rule)
/// Depth Pro read visible faces nearer far more often (jazz: 6.5% of frames by >= 0.5 vs 0.6% for Small; 3 of 20
/// labelled visible held frames at 0.5, none at 0.75) and put two of your same-person pairs >= 0.15 apart (none
/// >= 0.5), so its thresholds are higher.
#[derive(Clone, Copy, Debug)]
pub struct ModelInfo {
    pub name: &'static str,
    pub onnx: bool,
    pub path: &'static str,
    pub licence: &'static str,
    pub front: f64,
    pub alone: f64,
    pub apart: f64,
}

pub const MODELS: [ModelInfo; 3] = [
    ModelInfo { name: "anything-small", onnx: true, path: "models/depth-anything-v2-small/model.onnx", licence: "Apache-2.0", front: 0.25, alone: 0.5, apart: 0.15 },
    ModelInfo { name: "anything-base", onnx: true, path: "models/depth-anything-v2-base/model.onnx", licence: "CC-BY-NC-4.0", front: 0.25, alone: 0.5, apart: 0.15 },
    // 2026-11-13, per-frame face depth: 0.5 uncovered 2/20 labelled
    ModelInfo { name: "pro", onnx: false, path: "models/depth-pro", licence: "Apple ML Research Model licence (research only)", front: 0.75, alone: 1.0, apart: 0.5 },
];

pub fn model(name: &str) -> Option<&'static ModelInfo> {
    MODELS.iter().find(|m| m.name == name)
}

/// Depth Anything's input, longest side (a multiple of 14)
pub const SIDE: usize = 518;
/// Depth.maps (the readings: near, gap, face depth) are this size; drawing uses the full map
pub const CACHE_SIDE: usize = 256;
/// the depth movie: frame i at i / MOVIE_FPS s
pub const MOVIE_BITRATE: u32 = 12_000_000;
pub const MOVIE_KEYFRAMES: u32 = 15;
pub const MOVIE_FPS: f64 = 30.0;
/// full-size frames decoded at a time (Depth.full)
pub const CHUNK: usize = 16;

/// The model MOJIMASK_DEPTH names, or None (off). Err for a name it doesn't know.
pub fn setting() -> Result<Option<&'static str>, String> {
    let v = std::env::var("MOJIMASK_DEPTH").unwrap_or_default().trim().to_lowercase();
    if ["", "off", "0", "no", "none"].contains(&v.as_str()) {
        return Ok(None);
    }
    match model(&v) {
        Some(m) => Ok(Some(m.name)),
        None => Err(format!("MOJIMASK_DEPTH={v:?}: one of off, {}", MODELS.iter().map(|m| m.name).collect::<Vec<_>>().join(", "))),
    }
}

// ---------------------------------------------------------------- the model (depth._Onnx)

/// The model's input size (h, w) for a frame: longest side SIDE, each side rounded to a multiple of 14.
pub fn input_size(h: usize, w: usize) -> (usize, usize) {
    let k = SIDE as f64 / h.max(w) as f64;
    let r = |v: usize| 14.max(np::pyround(v as f64 * k / 14.0) as usize * 14);
    (r(h), r(w))
}

/// The model input for a BGR frame: RGB, cubic resize, x / 255 in float32, then (x - mean) / std in float64,
/// cast to float32; NCHW. Returns (input, h, w).
pub fn onnx_input(bgr: &Mat<u8>) -> (Vec<f32>, usize, usize) {
    let (h, w) = input_size(bgr.rows, bgr.cols);
    let x = resize_u8(&swap_rb(bgr), w, h, 0.0, 0.0, Interp::Cubic);
    let mean = [0.485f64, 0.456, 0.406];
    let std = [0.229f64, 0.224, 0.225];
    let plane = h * w;
    let mut out = vec![0f32; 3 * plane];
    for i in 0..plane {
        for c in 0..3 {
            let v = (x.data[i * 3 + c] as f32 / 255.0) as f64;
            out[c * plane + i] = ((v - mean[c]) / std[c]) as f32;
        }
    }
    (out, h, w)
}

/// Depth Anything (ONNX) on the Mac's GPU through ONNX Runtime's CoreML provider, as onnx_core runs the face
/// detectors (CPU elsewhere). The input size is pinned per frame shape -- with it free, CoreML silently fell back
/// to the CPU -- and compiled once per shape (cached by onnx_core). Same output as the CPU; 32-59 ms a frame vs
/// 205-290 (2026-11-13). Apple's own Core ML build was a little faster (~23 ms) but takes a fixed 518 x 392 input:
/// stretching frames into it cost accuracy (3 wrong "different people" at 0.15, visible faces reading nearer x6 on
/// party_clip_c). Returns relative inverse depth (larger = nearer), h x w.
pub fn run_onnx(m: &Models, model_path: &Path, bgr: &Mat<u8>, ep: Option<Ep>) -> Mat<f32> {
    let (x, h, w) = onnx_input(bgr);
    let dims = vec![("batch_size".to_string(), 1), ("height".to_string(), h as i64), ("width".to_string(), w as i64)];
    let out = m.run_path(model_path, ep.unwrap_or(m.engines.depth), &dims, &[1, 3, h, w], x);
    let (shape, d) = out.into_iter().next().unwrap();
    let (oh, ow) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    Mat::from_vec(oh, ow, 1, d)
}

// ---------------------------------------------------------------- the cache (depth.compute)

/// What one frame adds to the cache: its 8-bit picture (padded to even sizes, the last row/column repeated), its
/// range, percentiles and the every-4th-pixel sample the fit uses.
pub struct Encoded {
    pub q: Mat<u8>,
    pub lo: f64,
    pub hi: f64,
    pub pct: [f64; 3],
    pub sample: Vec<f32>,
}

/// One nearness map -> its movie frame (grey, even size) and numbers (depth.compute's loop body).
pub fn encode(n: &Mat<f32>) -> Encoded {
    let (h, w) = (n.rows, n.cols);
    let lo = n.data.iter().cloned().fold(f32::INFINITY, f32::min) as f64;
    let hi = n.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let (lo32, den32) = (lo as f32, np::pymax(hi - lo, 1e-9) as f32);
    let q: Vec<u8> = n.data.iter().map(|&v| ((v - lo32) / den32 * 255.0f32).round_ties_even() as u8).collect();
    let q = Mat::from_vec(h, w, 1, q);
    let q = mm_vision::copy_make_border(&q, 0, h % 2, 0, w % 2, mm_vision::Border::Replicate, &[]);
    let p = np::percentile_f32(&n.data, &[10.0, 50.0, 90.0]);
    let mut sample = Vec::new();
    for y in (0..h).step_by(4) {
        for x in (0..w).step_by(4) {
            sample.push(n.data[y * w + x]);
        }
    }
    Encoded { q, lo, hi, pct: [p[0], p[1], p[2]], sample }
}

/// depth.compute's line: np.polyfit(x, y, 1) -> (a, b) ([`polyfit1`], bit for bit on the Mac); (0, mean(y)) when x
/// is constant, as depth.compute does (y's float32 mean).
pub fn fit_line(x: &[f64], y: &[f64]) -> (f64, f64) {
    let (mn, mx) = x.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    if !(mx - mn > 0.0) {
        let yf: Vec<f32> = y.iter().map(|&v| v as f32).collect();
        return (0.0, np::mean(&yf) as f64);
    }
    polyfit1(x, y)
}

// np.polyfit(x, y, 1) bit for bit: numpy scales the Vandermonde matrix's columns and solves with LAPACK's dgelsd
// (SVD), here Accelerate's own dgelsd (the project's numpy links Accelerate's `$NEWLAPACK$ILP64` LAPACK), called as
// numpy's _umath_linalg calls it (a workspace query first). The column sums are sums of byte codes squared, exact in
// float64 in any order. Off the Mac: the normal equations (equal only to the last few bits).
#[cfg(target_os = "macos")]
mod lapack {
    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        #[link_name = "dgelsd$NEWLAPACK$ILP64"]
        pub fn dgelsd(m: *const i64, n: *const i64, nrhs: *const i64, a: *mut f64, lda: *const i64, b: *mut f64, ldb: *const i64,
                      s: *mut f64, rcond: *const f64, rank: *mut i64, work: *mut f64, lwork: *const i64, iwork: *mut i64,
                      info: *mut i64);
    }
}

/// numpy.linalg.lstsq(a, b, rcond)[0] for an m x 2 matrix (columns c0, c1) and one right-hand side, m >= 2:
/// Accelerate's dgelsd with numpy's workspace query.
#[cfg(target_os = "macos")]
fn lstsq2(c0: &[f64], c1: &[f64], y: &[f64], rcond: f64) -> [f64; 2] {
    let m = c0.len() as i64;
    let (n, nrhs, lda, ldb) = (2i64, 1i64, m.max(1), m.max(2));
    let mut a: Vec<f64> = c0.iter().chain(c1.iter()).cloned().collect(); // Fortran order
    let mut b = vec![0f64; ldb as usize];
    b[..y.len()].copy_from_slice(y);
    let mut s = [0f64; 2];
    let mut rank = 0i64;
    let mut info = 0i64;
    let mut wq = 0f64;
    let mut iwq = 0i64;
    unsafe {
        let (mut a2, mut b2) = (a.clone(), b.clone());
        lapack::dgelsd(&m, &n, &nrhs, a2.as_mut_ptr(), &lda, b2.as_mut_ptr(), &ldb, s.as_mut_ptr(), &rcond, &mut rank, &mut wq, &-1,
                       &mut iwq, &mut info);
        let lwork = wq as i64;
        let mut work = vec![0f64; lwork.max(1) as usize];
        let mut iwork = vec![0i64; iwq.max(1) as usize];
        lapack::dgelsd(&m, &n, &nrhs, a.as_mut_ptr(), &lda, b.as_mut_ptr(), &ldb, s.as_mut_ptr(), &rcond, &mut rank, work.as_mut_ptr(),
                       &lwork, iwork.as_mut_ptr(), &mut info);
    }
    assert_eq!(info, 0, "dgelsd: info {info}");
    [b[0], b[1]]
}

/// np.polyfit(x, y, 1) -> (slope, intercept), as numpy computes it (see above). Off the Mac: the normal equations.
pub fn polyfit1(x: &[f64], y: &[f64]) -> (f64, f64) {
    #[cfg(target_os = "macos")]
    {
        let rcond = x.len() as f64 * f64::EPSILON;
        // vander(x, 2) = [x, 1]; scale = sqrt((lhs * lhs).sum(axis=0))
        let mut s0 = 0f64;
        for v in x {
            s0 += v * v;
        }
        let mut s1 = 0f64;
        for _ in x {
            s1 += 1.0;
        }
        let (k0, k1) = (s0.sqrt(), s1.sqrt());
        let c0: Vec<f64> = x.iter().map(|v| v / k0).collect();
        let c1: Vec<f64> = x.iter().map(|_| 1.0 / k1).collect();
        let c = lstsq2(&c0, &c1, y, rcond);
        (c[0] / k0, c[1] / k1)
    }
    #[cfg(not(target_os = "macos"))]
    {
        normal_equations(x, y)
    }
}


/// The least-squares line by the normal equations (off the Mac).
#[allow(dead_code)]
fn normal_equations(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let (sx, sy) = (x.iter().sum::<f64>(), y.iter().sum::<f64>());
    let (mxv, myv) = (sx / n, sy / n);
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for (a, b) in x.iter().zip(y) {
        sxx += (a - mxv) * (a - mxv);
        sxy += (a - mxv) * (b - myv);
    }
    let a = sxy / sxx;
    (a, myv - a * mxv)
}

/// The movie writer (mm-media): grey frames as BGR, H.264 at MOVIE_BITRATE, keyframes every MOVIE_KEYFRAMES.
pub trait GreyMovie {
    fn write(&mut self, grey: &Mat<u8>);
    fn close(&mut self);
}

/// The movie reader (mm-media's grey::MovieLuma; video_io.luma): (frame index, Y plane) from frame `start`, `count`
/// frames or to the end. (Defined in mm-vision, so mm-media can implement it without depending on this crate.)
pub use mm_vision::luma::LumaSource;

/// The sidecar (depth_<model>.json): per frame [a, b, p10, p50, p90].
pub struct Sidecar {
    pub model: String,
    pub shape: (usize, usize),
    pub frames: Vec<[f64; 5]>,
}

impl Sidecar {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({"model": self.model, "shape": [self.shape.0, self.shape.1], "frames": self.frames})
    }
    pub fn from_json(v: &serde_json::Value) -> Sidecar {
        let fr = v["frames"].as_array().unwrap().iter().map(|r| {
            let r = r.as_array().unwrap();
            [0, 1, 2, 3, 4].map(|i| r[i].as_f64().unwrap())
        });
        Sidecar {
            model: v["model"].as_str().unwrap_or("").to_string(),
            shape: (v["shape"][0].as_u64().unwrap() as usize, v["shape"][1].as_u64().unwrap() as usize),
            frames: fr.collect(),
        }
    }
}

/// Nearness for every frame (depth.compute): `run` gives each frame's map, `movie` takes the 8-bit frames, then
/// `read_back` decodes what was written and each frame's bytes -> nearness line is fitted on it. The caller writes
/// the movie and sidecar files (atomically, as depth.compute does) and reports progress.
pub fn compute(
    frames: impl Iterator<Item = Mat<u8>>,
    model_name: &str,
    mut run: impl FnMut(&Mat<u8>) -> Mat<f32>,
    movie: &mut dyn GreyMovie,
    read_back: impl FnOnce() -> Vec<(usize, Mat<u8>)>,
    mut progress: impl FnMut(usize),
) -> Result<Sidecar, String> {
    let mut info = Vec::new();
    let mut samples = Vec::new();
    let mut shape = None;
    for (i, f) in frames.enumerate() {
        let n = run(&f);
        shape.get_or_insert((n.rows, n.cols));
        let e = encode(&n);
        movie.write(&e.q);
        info.push(e.pct);
        samples.push(e.sample);
        progress(i + 1);
    }
    movie.close();
    let (h, w) = shape.ok_or("depth: no frames")?;
    let mut fit = Vec::new();
    for (i, y) in read_back() {
        if i >= samples.len() {
            break;
        }
        let mut x = Vec::new();
        for r in (0..h).step_by(4) {
            for c in (0..w).step_by(4) {
                x.push(y.data[r * y.cols + c] as f64);
            }
        }
        let s: Vec<f64> = samples[i].iter().map(|&v| v as f64).collect();
        let (a, b) = fit_line(&x, &s);
        fit.push([a, b]);
    }
    if fit.len() != samples.len() {
        return Err(format!("depth movie: wrote {} frames, read back {}", samples.len(), fit.len()));
    }
    let frames = fit.iter().zip(&info).map(|(f, q)| [f[0], f[1], q[0], q[1], q[2]]).collect();
    Ok(Sidecar { model: model_name.to_string(), shape: (h, w), frames })
}

// ---------------------------------------------------------------- Depth

/// near(t, c): how much nearer the middle of box c ([cx, cy, s], frame pixels) is at frame t than the frame's
/// middle depth, in units of the frame's depth spread (p90 - p10 of nearness). Scale-free, so the three models (one
/// metric, two relative) are compared the same way: a box's reading minus that face's own depth around then
/// (face_at).
pub struct Depth {
    pub name: String,
    pub front: f64,
    pub alone: f64,
    pub apart: f64,
    /// a, b, p10, p50, p90 per frame
    pub fr: Vec<[f64; 5]>,
    /// the model's map size (h, w)
    pub shape: (usize, usize),
    /// readings grid: one map per frame, CACHE_SIDE on the long side, float32
    pub maps: Vec<Mat<f32>>,
    /// Depth.maps px per frame px
    pub k: f64,
    pub p50: Vec<f64>,
    pub spread: Vec<f64>,
    movie: Option<Box<dyn LumaSource>>,
    chunks: Mutex<Vec<(usize, HashMap<usize, Mat<u8>>)>>,
}

impl Depth {
    /// From the cached sidecar and the decoded movie. `frame_hw`: the project's frame size (PROJECT/frames).
    pub fn new(name: &str, side: &Sidecar, movie: Box<dyn LumaSource>, frame_hw: (usize, usize)) -> Depth {
        let maps = Self::build_maps(side, movie.luma(0, None), frame_hw);
        Self::from_parts(name, side, maps, frame_hw, Some(movie))
    }

    /// depth.Depth(proj, name) from the cache in PROJECT/dets (dets/depth_<name>.json and .mp4, which must exist);
    /// `movie` reads dets/depth_<name>.mp4 (mm_media::grey::MovieLuma). `frame_hw`: the project's frame size.
    pub fn open_cache(proj: &Path, name: &str, frame_hw: (usize, usize), movie: impl FnOnce(&Path) -> Box<dyn LumaSource>) -> Result<Depth, String> {
        let side_p = proj.join("dets").join(format!("depth_{name}.json"));
        let mp = proj.join("dets").join(format!("depth_{name}.mp4"));
        if !side_p.exists() || !mp.exists() {
            return Err(format!("depth {name}: no cache in {}/dets (dets/depth_{name}.mp4 and .json)", proj.display()));
        }
        let v: serde_json::Value = std::fs::read(&side_p).map_err(|e| e.to_string()).and_then(|b| serde_json::from_slice(&b).map_err(|e| e.to_string()))?;
        let side = Sidecar::from_json(&v);
        Ok(Depth::new(name, &side, movie(&mp), frame_hw))
    }

    /// The readings grid from decoded luma frames: bytes * a + b in float64, area-shrunk in float64, float32.
    pub fn build_maps(side: &Sidecar, luma: Vec<(usize, Mat<u8>)>, frame_hw: (usize, usize)) -> Vec<Mat<f32>> {
        let (h, w) = side.shape;
        let (fh, fw) = frame_hw;
        let k = CACHE_SIDE as f64 / fh.max(fw) as f64;
        let (sw, sh) = (1.max(np::pyround(fw as f64 * k)) as usize, 1.max(np::pyround(fh as f64 * k)) as usize);
        let mut maps = Vec::new();
        for (i, y) in luma {
            if i < side.frames.len() {
                let (a, b) = (side.frames[i][0], side.frames[i][1]);
                let mut v = Vec::with_capacity(h * w);
                for r in 0..h {
                    for c in 0..w {
                        v.push(y.data[r * y.cols + c] as f32 as f64 * a + b);
                    }
                }
                let s = resize_f64(&Mat::from_vec(h, w, 1, v), sw, sh, 0.0, 0.0, Interp::Area);
                maps.push(Mat::from_vec(sh, sw, 1, s.data.into_iter().map(|x| x as f32).collect()));
            }
        }
        maps
    }

    pub fn from_parts(name: &str, side: &Sidecar, maps: Vec<Mat<f32>>, frame_hw: (usize, usize), movie: Option<Box<dyn LumaSource>>) -> Depth {
        let m = model(name).expect("depth model");
        let k = maps.first().map(|m| m.cols).unwrap_or(0) as f64 / frame_hw.1 as f64;
        Depth {
            name: name.to_string(),
            front: m.front,
            alone: m.alone,
            apart: m.apart,
            fr: side.frames.clone(),
            shape: side.shape,
            k,
            p50: side.frames.iter().map(|f| f[3]).collect(),
            spread: side.frames.iter().map(|f| np::pymax(f[4] - f[2], 1e-6)).collect(),
            maps,
            movie,
            chunks: Mutex::new(Vec::new()),
        }
    }

    /// Frame t's nearness at the model's own resolution (float32), for drawing. Decoded CHUNK frames at a time
    /// (the editor and Export ask for neighbouring frames), the last few chunks kept.
    pub fn full(&self, t: usize) -> Option<Mat<f32>> {
        let movie = self.movie.as_ref()?;
        let c = t / CHUNK;
        let y = {
            let mut ch = self.chunks.lock().unwrap();
            if !ch.iter().any(|(k, _)| *k == c) {
                let got: HashMap<usize, Mat<u8>> =
                    movie.luma(c * CHUNK, Some(CHUNK)).into_iter().filter(|(i, _)| *i < (c + 1) * CHUNK).collect();
                if ch.len() >= 4 {
                    ch.remove(0);
                }
                ch.push((c, got));
            }
            ch.iter().find(|(k, _)| *k == c).unwrap().1.get(&t).cloned()
        }?;
        Some(self.full_from_luma(t, &y))
    }

    /// bytes * a + b in float32 (Python's float() makes the fit's numbers weak scalars).
    pub fn full_from_luma(&self, t: usize, y: &Mat<u8>) -> Mat<f32> {
        let (h, w) = self.shape;
        let (a, b) = (self.fr[t][0] as f32, self.fr[t][1] as f32);
        let mut v = Vec::with_capacity(h * w);
        for r in 0..h {
            for c in 0..w {
                v.push(y.data[r * y.cols + c] as f32 * a + b);
            }
        }
        Mat::from_vec(h, w, 1, v)
    }

    pub fn near(&self, t: i64, c: &[f64; 3]) -> f64 {
        self.near_f(t, c, 0.25)
    }

    pub fn near_f(&self, t: i64, c: &[f64; 3], f: f64) -> f64 {
        if t < 0 || t as usize >= self.maps.len() {
            return f64::NAN;
        }
        let m = &self.maps[t as usize];
        let (h, w) = (m.rows as f64, m.cols as f64);
        let (x, y, s) = (c[0] * self.k, c[1] * self.k, np::pymax(c[2] * self.k, 2.0));
        let pymin = |a: f64, b: f64| if b < a { b } else { a };
        let x1 = np::pymax(x - f * s, 0.0) as i64;
        let x2 = pymin(x + f * s + 1.0, w) as i64;
        let y1 = np::pymax(y - f * s, 0.0) as i64;
        let y2 = pymin(y + f * s + 1.0, h) as i64;
        if x2 <= x1 || y2 <= y1 {
            return f64::NAN;
        }
        let mut v = Vec::with_capacity(((x2 - x1) * (y2 - y1)) as usize);
        for r in y1..y2 {
            v.extend_from_slice(&m.row(r as usize)[x1 as usize..x2 as usize]);
        }
        (np::median(&v) as f64 - self.p50[t as usize]) / self.spread[t as usize]
    }

    /// How far apart two boxes' readings are ({t: box} each) where they come closest in time: k frames of each
    /// nearest that moment. The same person seen twice (your same-person answers) never read more than 0.12 apart
    /// on jazz, party_clip_c and IMG_1043 with Depth Anything; MODELS "apart", used by scan/same.py.
    pub fn gap(&self, a: &BTreeMap<i64, [f64; 3]>, b: &BTreeMap<i64, [f64; 3]>, k: usize) -> f64 {
        if a.is_empty() || b.is_empty() {
            return f64::NAN;
        }
        let mut best: Option<(i64, i64, i64)> = None;
        for &p in a.keys() {
            for &q in b.keys() {
                let cand = ((p - q).abs(), p, q);
                if best.is_none_or(|bb| cand < bb) {
                    best = Some(cand);
                }
            }
        }
        let (_, x, y) = best.unwrap();
        let nearest = |m: &BTreeMap<i64, [f64; 3]>, at: i64| -> Vec<f64> {
            let mut ts: Vec<i64> = m.keys().copied().collect();
            ts.sort_by_key(|t| (t - at).abs()); // stable: ties in time order, as sorted(..., key=)
            ts.iter().take(k).map(|t| self.near(*t, &m[t])).collect()
        };
        (np::nanmedian(&nearest(a, x)) - np::nanmedian(&nearest(b, y))).abs()
    }
}

// ---------------------------------------------------------------- partial occlusion

pub const OCC_SOFT: f64 = 0.15;
pub const OCC_MIN: f64 = 0.03;
/// a cover is cut if it's in front in a majority of the frames within OCC_STEADY of t
pub const OCC_STEADY: i64 = 2;
/// If most of a cover's face zone (FACE_ZONE[0]) reads in front of its face, it's only cut when another person,
/// clearly nearer (the model's "apart"), is within REACH of their head widths sideways: then it's them in front
/// (jazz frame 51, the dancer's arm across two seated faces). Otherwise it's this person's own head or arm reading
/// nearer than their face's estimate, and cutting shows their face -- the party clip, frame 359: a hand on the
/// head, the emoji cut away over the face it covered. Something that isn't a person (a pillar, a cup) still cuts
/// while it leaves most of the face zone uncovered.
pub const OCC_SELF: f64 = 0.5;
pub const REACH: f64 = 4.0;
/// frames either side for a face's own depth (Occlusion.face)
pub const FACE_WINDOW: f64 = 15.0;
/// How much nearer counts as in front depends on where in the cover: the face itself (eyes to chin: a circle of
/// FACE_ZONE[0] head sizes around a point track.FACE_DY head sizes below the head box's centre, where
/// track.face_to_head puts it) needs the model's "alone"/"front" margin, since a wrong cut there shows a face; past
/// FACE_ZONE[1] (hair, ears, the emoji's overhang) the model's "apart" is enough -- the depth gap at which two boxes
/// are never the same person. The editor's depth view draws the same zone (review_server.depth_frame sends both).
/// Jazz frame 126: the dancer's arm reads ~0.25 nearer than the seated faces it crosses, under the 0.5 a detected
/// face needs, so their emoji were drawn over her arm.
pub const FACE_ZONE: (f64, f64) = (0.40, 0.55);
/// track.FACE_DY: the face's centre sits this many head sizes below its head box's
pub const FACE_DY: f64 = 0.08;

/// Distance of pixel (x, y) (frame px) from head box c's face centre, in head sizes.
pub fn zone_r(x: f64, y: f64, c: &[f64; 3]) -> f64 {
    (x - c[0]).hypot(y - (c[1] + FACE_DY * c[2])) / np::pymax(c[2], 1e-6)
}

/// The margin at distance r (zone_r): face_margin on the face, rim past it.
pub fn margin_at(r: f64, face_margin: f64, rim: f64) -> f64 {
    let w = np::clip((r - FACE_ZONE.0) / (FACE_ZONE.1 - FACE_ZONE.0), 0.0, 1.0);
    let rm = if face_margin < rim { face_margin } else { rim }; // python min(rim, face_margin)
    face_margin + (rm - face_margin) * w
}

/// A face's depth reading at frame t from its own detected readings v at frames ts (sorted, finite): the median of
/// those within FACE_WINDOW frames of t (past a gap with none, the window starts at the nearest ones), never the
/// one at t itself -- that is where an occluder would be. At least 5 readings (the nearest ones) when it has them.
/// Used by Occlusion.face (drawing) and scan/behind.py (which held frames are behind someone).
///
/// The fewer-than-5 case takes np.argsort(d)[:5] (numpy's default, unstable sort); here ties go in index order,
/// which is what numpy's sort does for the small arrays tried (analysis_depth.py checks random cases with ties).
pub fn face_at(ts: &[i64], v: &[f64], t: i64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let d: Vec<f64> = ts.iter().map(|&u| if u == t { f64::INFINITY } else { (u - t).abs() as f64 }).collect();
    if !d.iter().any(|x| x.is_finite()) {
        return np::median(v);
    }
    let dmin = d.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut sel: Vec<f64> = d.iter().zip(v).filter(|(x, _)| **x <= FACE_WINDOW + dmin - 1.0).map(|(_, &y)| y).collect();
    if sel.len() < 5 {
        let nf = d.iter().filter(|x| x.is_finite()).count();
        let mut idx: Vec<usize> = (0..d.len()).collect();
        idx.sort_by(|&a, &b| d[a].partial_cmp(&d[b]).unwrap());
        sel = idx[..5.min(nf)].iter().map(|&i| v[i]).collect();
    }
    np::median(&sel)
}

/// One cover in a frame: track id, head box, kind ("obs", "coast", ...), piece (track id when absent).
pub type Item = crate::head_pose::Item;

/// What plan() decided for one cover it cuts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cut {
    pub face: f64,
    pub margin: f64,
    pub rim: f64,
    pub head: f64,
}

/// Why plan() decided what it did for one cover it weighed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Why {
    pub face: f64,
    pub margin: f64,
    pub rim: f64,
    pub itself: bool,
    pub in_front: f64,
    pub someone_nearer: bool,
}

/// A float64 map (Occlusion.map).
#[derive(Clone)]
pub struct Map64 {
    pub h: usize,
    pub w: usize,
    pub d: Vec<f64>,
}

/// The depth an [`Occlusion`] reads: borrowed (tests, the scan) or shared ([`Occlusion::shared`]: an owner that
/// keeps the occlusion, e.g. mm-records).
pub enum DepthRef<'a> {
    Borrowed(&'a Depth),
    Shared(std::sync::Arc<Depth>),
}

impl std::ops::Deref for DepthRef<'_> {
    type Target = Depth;
    fn deref(&self) -> &Depth {
        match self {
            DepthRef::Borrowed(d) => d,
            DepthRef::Shared(d) => d,
        }
    }
}

/// Per frame: plan(t, items, scale, shape) -> (near map, {item index: cut}) decides which covers are cut;
/// nearness(t, piece, c) orders covers far to near. (front(), the CPU fallback's alpha, is mm-render's.)
pub struct Occlusion<'a> {
    pub d: DepthRef<'a>,
    /// each piece's detected boxes: {piece: {t: [cx, cy, s]}}
    pub obs: HashMap<String, BTreeMap<i64, [f64; 3]>>,
    readings: Mutex<HashMap<String, (Vec<i64>, Vec<f64>)>>,
    faces: Mutex<HashMap<(String, i64), f64>>,
    maps: Mutex<Vec<(i64, std::sync::Arc<Map64>)>>,
    /// full-resolution nearness by frame, when the depth has no movie (tests feed decoded frames)
    pub full_override: Option<Box<dyn Fn(usize) -> Mat<f32> + Send + Sync + 'a>>,
}

fn base_piece(piece: &str) -> String {
    piece.split('~').next().unwrap().to_string()
}

impl<'a> Occlusion<'a> {
    pub fn new(d: &'a Depth, obs: HashMap<String, BTreeMap<i64, [f64; 3]>>) -> Occlusion<'a> {
        Occlusion { d: DepthRef::Borrowed(d), obs, readings: Mutex::default(), faces: Mutex::default(), maps: Mutex::default(), full_override: None }
    }

    /// The same, owning a share of the depth (no borrow to keep alive).
    pub fn shared(d: std::sync::Arc<Depth>, obs: HashMap<String, BTreeMap<i64, [f64; 3]>>) -> Occlusion<'static> {
        Occlusion { d: DepthRef::Shared(d), obs, readings: Mutex::default(), faces: Mutex::default(), maps: Mutex::default(), full_override: None }
    }

    /// (frames, readings): the piece's detected boxes read on the depth map (finite ones), sorted by frame. A part
    /// you split off reads like its track.
    pub fn readings(&self, piece: &str) -> (Vec<i64>, Vec<f64>) {
        let p = base_piece(piece);
        if let Some(r) = self.readings.lock().unwrap().get(&p) {
            return r.clone();
        }
        let mut ts = Vec::new();
        let mut v = Vec::new();
        if let Some(o) = self.obs.get(&p) {
            for (&t, c) in o {
                let x = self.d.near(t, c);
                if x.is_finite() {
                    ts.push(t);
                    v.push(x);
                }
            }
        }
        self.readings.lock().unwrap().insert(p, (ts.clone(), v.clone()));
        (ts, v)
    }

    /// The face's depth reading at frame t: the median of its own detected readings within FACE_WINDOW frames of t
    /// (past any gap with none: the window starts at the nearest ones), never the reading at t itself. A median, so
    /// readings spoiled by an arm crossing the face (still detected) don't count while they're under half the
    /// window; local, so it follows someone walking toward the camera. On three clips it beat one number per
    /// person, line-interpolating the nearest readings, and a head-size model (2026-11-12).
    pub fn face(&self, piece: &str, t: i64) -> f64 {
        let key = (base_piece(piece), t);
        if let Some(v) = self.faces.lock().unwrap().get(&key) {
            return *v;
        }
        let (ts, v) = self.readings(piece);
        let r = face_at(&ts, &v, t);
        let mut f = self.faces.lock().unwrap();
        if f.len() > 20000 {
            f.clear();
        }
        f.insert(key, r);
        r
    }

    /// How near a cover's face is at frame t, to draw covers far to near: its own depth around then, else its box's
    /// reading now (a drawn cover), else -inf.
    pub fn nearness(&self, t: i64, piece: &str, c: &[f64; 3]) -> f64 {
        let mut v = self.face(piece, t);
        if !v.is_finite() {
            v = self.d.near(t, c);
        }
        if v.is_finite() {
            v
        } else {
            f64::NEG_INFINITY
        }
    }

    fn full(&self, t: usize) -> Mat<f32> {
        match &self.full_override {
            Some(f) => f(t),
            None => self.d.full(t).expect("depth movie frame"),
        }
    }

    /// Frame t's nearness at the model's own resolution, over the frame's median in units of its spread, eroded by
    /// ~1/CACHE_SIDE of the frame (the occluder trimmed so the face's edge stays covered). A few frames kept.
    pub fn map(&self, t: i64) -> std::sync::Arc<Map64> {
        if let Some((_, m)) = self.maps.lock().unwrap().iter().find(|(k, _)| *k == t) {
            return m.clone();
        }
        let f = self.full(t as usize);
        let (p50, sp) = (self.d.p50[t as usize], self.d.spread[t as usize]);
        let m: Vec<f64> = f.data.iter().map(|&v| (v as f64 - p50) / sp).collect();
        let g = 1.max(np::pyround(f.rows.max(f.cols) as f64 / CACHE_SIDE as f64)) as usize;
        let e = erode(&Mat::from_vec(f.rows, f.cols, 1, m), 2 * g + 1, 2 * g + 1);
        let e = std::sync::Arc::new(Map64 { h: f.rows, w: f.cols, d: e.data });
        let mut ms = self.maps.lock().unwrap();
        while ms.len() >= 6 {
            ms.remove(0);
        }
        ms.push((t, e.clone()));
        e
    }

    /// (fraction of cover c's square reading in front of `face` at frame t, whether most of its face zone does --
    /// OCC_SELF -- and guard is on), at map resolution.
    #[allow(clippy::too_many_arguments)]
    pub fn in_front(&self, t: i64, c: &[f64; 3], scale: f64, face: f64, margin: f64, shape: (usize, usize), guard: bool) -> (f64, bool) {
        let m = self.map(t);
        let (mh, mw) = (m.h as i64, m.w as i64);
        let (kx, ky) = (mw as f64 / shape.1 as f64, mh as f64 / shape.0 as f64);
        let h = c[2] * scale / 2.0;
        let x1 = (((c[0] - h) * kx) as i64).max(0);
        let x2 = (((c[0] + h) * kx).ceil() as i64 + 1).min(mw);
        let y1 = (((c[1] - h) * ky) as i64).max(0);
        let y2 = (((c[1] + h) * ky).ceil() as i64 + 1).min(mh);
        if x2 <= x1 || y2 <= y1 {
            return (0.0, false);
        }
        let n = ((x2 - x1) * (y2 - y1)) as usize;
        let mut d = Vec::with_capacity(n);
        let mut r = Vec::with_capacity(n);
        for yy in y1..y2 {
            let py = (yy as f64 + 0.5) / ky;
            for xx in x1..x2 {
                d.push(m.d[(yy * mw + xx) as usize] - face);
                r.push(zone_r((xx as f64 + 0.5) / kx, py, c));
            }
        }
        if guard {
            let (mut nz, mut ok) = (0usize, 0usize);
            for (dv, rv) in d.iter().zip(&r) {
                if *rv < FACE_ZONE.0 {
                    nz += 1;
                    ok += (*dv >= margin) as usize;
                }
            }
            if nz > 0 && ok as f64 / nz as f64 > OCC_SELF {
                return (0.0, true);
            }
        }
        let p: Vec<f64> =
            d.iter().zip(&r).map(|(dv, rv)| np::clip((dv - margin_at(*rv, margin, self.d.apart)) / OCC_SOFT + 0.5, 0.0, 1.0)).collect();
        (np::mean(&p), false)
    }

    /// Another cover in the frame whose face reads clearly nearer (the model's "apart") than `face`, within REACH of
    /// its own head widths sideways of cover i.
    fn someone_nearer(&self, i: usize, items: &[Item], faces: &[f64], face: f64) -> bool {
        let c = items[i].c;
        items.iter().enumerate().any(|(j, x)| j != i && faces[j] - face >= self.d.apart && (x.c[0] - c[0]).abs() <= REACH * x.c[2])
    }

    /// items: the frame's covers -> (near map for frame t or None, {index: cut}). A cover is cut when at least
    /// OCC_MIN of its square is in front in a majority of the frames within OCC_STEADY of t, and not when most of its
    /// face zone is (OCC_SELF; module notes). why: filled with each cover weighed.
    pub fn plan(
        &self,
        t: i64,
        items: &[Item],
        scale: f64,
        shape: (usize, usize),
        mut why: Option<&mut BTreeMap<usize, Why>>,
    ) -> (Option<std::sync::Arc<Map64>>, BTreeMap<usize, Cut>) {
        let n = self.d.maps.len() as i64;
        if !(0 <= t && t < n) {
            return (None, BTreeMap::new());
        }
        let mut out = BTreeMap::new();
        let faces: Vec<f64> = items.iter().map(|x| self.nearness(t, x.piece(), &x.c)).collect();
        for (i, it) in items.iter().enumerate() {
            if it.kind != "obs" && it.kind != "coast" {
                continue;
            }
            let piece = it.piece();
            let face = self.face(piece, t);
            if !face.is_finite() {
                continue;
            }
            let margin = if it.kind == "obs" { self.d.alone } else { self.d.front };
            let guard = !self.someone_nearer(i, items, &faces, face);
            let (now, itself) = self.in_front(t, &it.c, scale, face, margin, shape, guard);
            if let Some(w) = why.as_deref_mut() {
                w.insert(i, Why { face, margin, rim: self.d.apart, itself, in_front: now, someone_nearer: !guard });
            }
            if itself {
                continue; // its own head or arm in front of its face (OCC_SELF): never cut this frame
            }
            // the vote reuses frame t's box c on the neighbouring frames' maps, on purpose: the window is a few
            // frames, and it asks whether the depth model flickers where the cover is now
            let mut vote: BTreeMap<i64, bool> = BTreeMap::new();
            vote.insert(t, now >= OCC_MIN);
            for u in (t - OCC_STEADY).max(0)..=(t + OCC_STEADY).min(n - 1) {
                if let std::collections::btree_map::Entry::Vacant(e) = vote.entry(u) {
                    let f = self.face(piece, u);
                    e.insert(self.in_front(u, &it.c, scale, f, margin, shape, guard).0 >= OCC_MIN);
                }
            }
            // past the clip's ends the edge frame counts again: a cut starting on the last frame isn't a flicker
            let votes = (t - OCC_STEADY..=t + OCC_STEADY).filter(|&u| vote[&u.clamp(0, n - 1)]).count() as i64;
            if votes < OCC_STEADY + 1 {
                continue;
            }
            out.insert(i, Cut { face, margin, rim: self.d.apart, head: it.c[2] });
        }
        let m = if out.is_empty() { None } else { Some(self.map(t)) };
        (m, out)
    }

    /// Occlusion.picture: frame t's depth map coloured for the editor's depth view (turbo: blue far, red near, over
    /// the frame's p10..p90), BGR at the model's resolution. numpy's arithmetic: the float32 map against the float64
    /// p10/p90 (NumPy scalars, so float64 throughout), then `astype(uint8)` truncates.
    pub fn picture(&self, t: usize) -> Mat<u8> {
        let m = self.full(t);
        let (lo, hi) = (self.d.fr[t][2], self.d.fr[t][4]);
        let den = np::pymax(hi - lo, 1e-6);
        let mut v = Vec::with_capacity(m.data.len() * 3);
        for &x in &m.data {
            let q = ((x as f64 - lo) / den * 0.8 + 0.1).clamp(0.0, 1.0);
            v.extend_from_slice(&TURBO[(q * 255.0) as usize]);
        }
        Mat::from_vec(m.rows, m.cols, 3, v)
    }

    /// For the editor's depth view: per cover, what plan decided at frame t and from what (as JSON, the shape
    /// Occlusion.explain returns: piece, box, kind, cut, and for weighed covers face, margin, rim, self, in_front,
    /// someone_nearer; face and in_front rounded to 3 places).
    pub fn explain(&self, t: i64, items: &[Item], scale: f64, shape: (usize, usize)) -> Vec<serde_json::Value> {
        let mut why = BTreeMap::new();
        let (_, cuts) = self.plan(t, items, scale, shape, Some(&mut why));
        items
            .iter()
            .enumerate()
            .map(|(i, it)| {
                let mut e = serde_json::json!({"piece": it.piece(), "box": it.c, "kind": it.kind, "cut": cuts.contains_key(&i)});
                if let Some(w) = why.get(&i) {
                    let o = e.as_object_mut().unwrap();
                    o.insert("face".into(), serde_json::json!(np::pyround_n(w.face, 3)));
                    o.insert("margin".into(), serde_json::json!(w.margin));
                    o.insert("rim".into(), serde_json::json!(w.rim));
                    o.insert("self".into(), serde_json::json!(w.itself));
                    o.insert("in_front".into(), serde_json::json!(np::pyround_n(w.in_front, 3)));
                    o.insert("someone_nearer".into(), serde_json::json!(w.someone_nearer));
                }
                e
            })
            .collect()
    }
}

/// cv2.COLORMAP_TURBO as applyColorMap gives it: BGR for each byte (the project's OpenCV, dumped by
/// `cv2.applyColorMap(np.arange(256, dtype=np.uint8)[None], cv2.COLORMAP_TURBO)`).
const TURBO: [[u8; 3]; 256] = [
    [59, 18, 48], [67, 21, 50], [74, 24, 51], [81, 27, 52], [88, 30, 53], [95, 33, 54], [102, 36, 55], [109, 39, 56],
    [115, 42, 57], [121, 45, 58], [128, 47, 59], [134, 50, 60], [139, 53, 61], [145, 56, 62], [151, 59, 63], [156, 62, 63],
    [162, 64, 64], [167, 67, 65], [172, 70, 65], [177, 73, 66], [181, 75, 66], [186, 78, 67], [191, 81, 68], [195, 84, 68],
    [199, 86, 68], [203, 89, 69], [207, 92, 69], [211, 94, 69], [214, 97, 70], [218, 100, 70], [221, 102, 70], [224, 105, 70],
    [227, 107, 70], [230, 110, 71], [233, 113, 71], [235, 115, 71], [238, 118, 71], [240, 120, 71], [242, 123, 71], [244, 125, 70],
    [246, 128, 70], [248, 130, 70], [250, 133, 70], [251, 135, 70], [252, 138, 69], [253, 140, 69], [254, 143, 68], [254, 145, 67],
    [255, 148, 66], [255, 150, 65], [255, 153, 64], [254, 155, 62], [254, 158, 61], [253, 160, 59], [252, 163, 58], [251, 165, 56],
    [250, 168, 55], [248, 171, 53], [247, 173, 51], [245, 175, 49], [244, 178, 47], [242, 180, 46], [240, 183, 44], [238, 185, 42],
    [235, 188, 40], [233, 190, 39], [231, 192, 37], [228, 195, 35], [226, 197, 34], [223, 199, 32], [221, 201, 31], [218, 203, 30],
    [216, 205, 28], [213, 208, 27], [210, 210, 26], [208, 212, 26], [205, 213, 25], [202, 215, 24], [200, 217, 24], [197, 219, 24],
    [194, 221, 24], [192, 222, 24], [189, 224, 24], [187, 226, 25], [185, 227, 25], [182, 228, 26], [180, 230, 28], [178, 231, 29],
    [175, 233, 31], [172, 234, 32], [170, 235, 34], [167, 236, 37], [164, 238, 39], [161, 239, 42], [158, 240, 44], [155, 241, 47],
    [152, 242, 50], [148, 243, 53], [145, 244, 56], [142, 245, 60], [138, 246, 63], [135, 247, 67], [132, 248, 70], [128, 248, 74],
    [125, 249, 78], [122, 250, 82], [118, 250, 85], [115, 251, 89], [111, 252, 93], [108, 252, 97], [105, 253, 101], [102, 253, 105],
    [98, 254, 109], [95, 254, 113], [92, 254, 117], [89, 254, 121], [86, 255, 125], [83, 255, 128], [81, 255, 132], [78, 255, 136],
    [75, 255, 139], [73, 255, 143], [71, 255, 146], [68, 254, 150], [66, 254, 153], [64, 254, 156], [63, 253, 159], [61, 253, 161],
    [60, 252, 164], [58, 252, 167], [57, 251, 169], [56, 251, 172], [55, 250, 175], [54, 249, 177], [54, 248, 180], [53, 247, 183],
    [53, 246, 185], [52, 245, 188], [52, 244, 190], [52, 243, 193], [52, 241, 195], [52, 240, 198], [52, 239, 200], [52, 237, 203],
    [52, 236, 205], [52, 234, 208], [53, 233, 210], [53, 231, 212], [53, 229, 215], [54, 228, 217], [54, 226, 219], [55, 224, 221],
    [55, 223, 223], [55, 221, 225], [56, 219, 227], [56, 217, 229], [57, 215, 231], [57, 213, 233], [57, 211, 235], [58, 209, 236],
    [58, 207, 238], [58, 205, 239], [58, 203, 241], [58, 201, 242], [58, 199, 244], [58, 197, 245], [58, 195, 246], [58, 193, 247],
    [57, 190, 248], [57, 188, 249], [57, 186, 250], [56, 184, 251], [55, 182, 251], [54, 179, 252], [54, 177, 252], [53, 174, 253],
    [52, 172, 253], [51, 169, 254], [50, 167, 254], [49, 164, 254], [48, 161, 254], [47, 158, 254], [45, 155, 254], [44, 153, 254],
    [43, 150, 254], [42, 147, 254], [41, 144, 254], [39, 141, 253], [38, 138, 253], [37, 135, 252], [35, 132, 252], [34, 129, 251],
    [33, 126, 251], [31, 123, 250], [30, 120, 249], [29, 117, 249], [28, 114, 248], [26, 111, 247], [25, 108, 246], [24, 105, 245],
    [23, 102, 244], [21, 99, 243], [20, 96, 242], [19, 93, 241], [18, 91, 240], [17, 88, 239], [16, 85, 237], [15, 83, 236],
    [14, 80, 235], [13, 78, 234], [12, 75, 232], [12, 73, 231], [11, 71, 229], [10, 69, 228], [10, 67, 226], [9, 65, 225],
    [8, 63, 223], [8, 61, 221], [7, 59, 220], [7, 57, 218], [6, 55, 216], [6, 53, 214], [5, 51, 212], [5, 49, 210],
    [5, 47, 208], [4, 45, 206], [4, 43, 204], [4, 42, 202], [3, 40, 200], [3, 38, 197], [3, 37, 195], [2, 35, 193],
    [2, 33, 190], [2, 32, 188], [2, 30, 185], [2, 29, 183], [1, 27, 180], [1, 26, 178], [1, 24, 175], [1, 23, 172],
    [1, 22, 169], [1, 20, 167], [1, 19, 164], [1, 18, 161], [1, 16, 158], [1, 15, 155], [1, 14, 152], [1, 13, 149],
    [1, 11, 146], [1, 10, 142], [2, 9, 139], [2, 8, 136], [2, 7, 133], [2, 6, 129], [2, 5, 126], [3, 4, 122],
];
