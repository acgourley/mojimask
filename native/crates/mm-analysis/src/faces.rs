//! The scan's face measurements: one per detection, with how comparable it is. The graph (graph.py) and the
//! same-person questions (questions.py) take their face evidence from these, and the "clear face" test
//! (graph.CLEAR_FACE) is their score.
//!
//! For every detection of every tracklet: RetinaFace (reid.landmarks) on the crop around it (reid.face_crop) finds
//! the most confident face within NEAR (0.35) head sizes of the box centre, its 5 landmarks, and SFace (reid.embed)
//! on the aligned face. Comparability comes from the landmarks:
//!   yaw    (nose x - midpoint of the eyes x) / eye distance: ~0 facing the camera, larger as the head turns
//!   eyes   eye distance / face box width: shrinks as the head turns to profile
//!   px     face box height in pixels
//! A face is "frontal" (comparable) when the landmark detector is confident, it is facing the camera and big
//! enough. Face evidence is only taken between frontal faces: a profile against a frontal face scored as low as
//! two strangers on twirl.
//!
//! Port of the per-detection part of mojimask/scan/faces.py (`nearest`, `face`, `frontal`). `measure()` -- the
//! per-project cache (faces_retina_sface.pkl, keyed by each entry's box), the thread pool over frames and the viz
//! events -- belongs to the scan (mm-scan), which calls [`face`] per detection.
use crate::models::Models;
use crate::reid;
use mm_vision::Mat;

pub const NEAR: f64 = 0.35;
pub const CACHE: &str = "faces_retina_sface";
pub const CACHE_VERSION: i64 = 2;
/// FRONTAL = dict(score=0.6, yaw=0.25, px=24)
pub const FRONTAL_SCORE: f64 = 0.6;
pub const FRONTAL_YAW: f64 = 0.25;
pub const FRONTAL_PX: f64 = 24.0;

/// One face measurement. `eyes` is a float32 in Python when the face box is wider than 1 px (numpy scalar
/// arithmetic), a float otherwise; it is kept here as the f64 of that value.
#[derive(Clone, Debug)]
pub struct Face {
    pub score: f64,
    pub yaw: f64,
    pub eyes: f64,
    pub px: f64,
    /// L2-normalised SFace embedding (float32), only for frontal faces
    pub emb: Option<Vec<f32>>,
}

/// Index of the most confident face box (crop coordinates, from reid.face_crop) whose centre is within NEAR head
/// sizes of box c's centre, or None.
pub fn nearest(boxes: &[[f32; 5]], k: f64, x0: i64, y0: i64, c: &[f64; 3]) -> Option<usize> {
    let (k32, x032, y032) = (k as f32, x0 as f32, y0 as f32);
    let mut best: Option<usize> = None;
    for (i, b) in boxes.iter().enumerate() {
        // float32 until c (a float64 array) comes in
        let dx = ((b[0] + b[2]) / 2.0 / k32 + x032) as f64 - c[0];
        let dy = ((b[1] + b[3]) / 2.0 / k32 + y032) as f64 - c[1];
        let d = dx.hypot(dy);
        if d <= NEAR * c[2] && (best.is_none() || b[4] > boxes[best.unwrap()][4]) {
            best = Some(i);
        }
    }
    best
}

pub fn frontal(f: &Face) -> bool {
    f.score >= FRONTAL_SCORE && f.yaw.abs() <= FRONTAL_YAW && f.px >= FRONTAL_PX
}

/// The face measurement for detection c (cx, cy, s) in img: the most confident face within NEAR head sizes of its
/// centre, or None.
pub fn face(m: &Models, img: &Mat<u8>, c: &[f64; 3]) -> Option<Face> {
    let cr = reid::face_crop(img, c)?;
    let f = reid::landmarks(m, &cr.img, reid::LANDMARK_SIZE, 0.3, 0.4, None);
    measure_one(m, &cr, &f, c)
}

/// [`face`] after the landmark model (for tests that feed recorded outputs).
pub fn measure_one(m: &Models, cr: &reid::Crop, f: &reid::Faces, c: &[f64; 3]) -> Option<Face> {
    let i = nearest(&f.boxes, cr.k, cr.x0, cr.y0, c)?;
    let (b, kp) = (f.boxes[i], f.kps[i]);
    let mut out = numbers(&b, &kp, cr.k);
    if frontal(&out) {
        // the embedding is only ever compared for frontal faces (graph, questions): half the time saved
        let e = reid::embed(m, &reid::align(&cr.img, &kp));
        out.emb = Some(normalise(&e));
    }
    Some(out)
}

/// score, yaw, eyes, px from a face box and its landmarks, with numpy's float32/float64 mix.
pub fn numbers(b: &[f32; 5], kp: &[[f32; 2]; 5], k: f64) -> Face {
    let (el, er, nose) = (kp[0], kp[1], kp[2]);
    let ed = ((er[0] - el[0]).hypot(er[1] - el[1])) as f64 + 1e-6;
    let ed32 = ed as f32;
    let yaw = ((nose[0] - (el[0] + er[0]) / 2.0) / ed32) as f64;
    let bw = b[2] - b[0];
    // ed / max(1.0, b[2] - b[0]): a float32 when the box is wider than 1 (the python float is weak), else a float
    let eyes = if bw > 1.0 { (ed32 / bw) as f64 } else { ed / 1.0 };
    let px = ((b[3] - b[1]) / k as f32) as f64;
    Face { score: b[4] as f64, yaw, eyes, px, emb: None }
}

/// e / np.linalg.norm(e) in float32. numpy's norm of a float32 vector is sqrt(e.dot(e)), and its dot is the BLAS it
/// was built with (the project's numpy: Accelerate). [`crate::np::norm_f32`] calls Accelerate's own `cblas_sdot`,
/// which gives the same bits (checked on 3,000 random vectors of 7-1000 floats; summing in any fixed order in Rust
/// agreed only 50-85% of the time). Elsewhere: a plain float32 sum.
pub fn normalise(e: &[f32]) -> Vec<f32> {
    let n = crate::np::norm_f32(e);
    e.iter().map(|&v| v / n).collect()
}
