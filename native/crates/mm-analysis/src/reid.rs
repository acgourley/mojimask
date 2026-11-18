//! Faces for comparing people: RetinaFace finds a face and its 5 landmarks, SFace embeds the aligned face (the
//! scan's face check and face likeness, scan/faces.py, and its follow check, follow.py, use these).
//!
//! Both are permissively licensed: RetinaFace-R50 is the app's own face detector (biubug6/Pytorch_Retinaface, MIT;
//! the scan's detector and this share onnx_core.retina_decode; this decodes its landmark head too), SFace is OpenCV
//! Zoo's face_recognition_sface_2021dec (Apache-2.0). They replaced InsightFace's SCRFD + ArcFace (non-commercial
//! only) on 2026-11-14, measured: a better face check on every clip tried, the same safety on the decisions bench,
//! ~10% more questions. The old pair is kept for comparisons in experiments/insightface_models.py.
//!
//! Port of mojimask/reid.py. Numerics follow numpy's dtypes there (NEP 50): the decode is float32, NMS float64,
//! the alignment matrix float64. Pixels go through mm-vision (cv2-exact resize and warpAffine).
//! Parity: native/tools/parity/analysis_reid.py.
use crate::models::Models;
use mm_ml::rt::Ep;
use mm_vision::{resize_u8, warp_affine_u8, Border, Interp, Mat};

/// 112x112 alignment template (SFace's too).
pub const ALIGN_DST: [[f32; 2]; 5] =
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]];
/// RetinaFace input: crops are resized to this square (CoreML needs one fixed shape).
pub const LANDMARK_SIZE: usize = 192;

/// Faces found by [`landmarks`]: boxes x1, y1, x2, y2, score and 5 landmarks (eyes, nose, mouth corners, image left
/// first), float32 as Python's arrays.
#[derive(Clone, Debug, Default)]
pub struct Faces {
    pub boxes: Vec<[f32; 5]>,
    pub kps: Vec<[[f32; 2]; 5]>,
}

/// RetinaFace's input: BGR minus (104, 117, 123), NCHW (onnx_core.retina_input).
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

/// Faces in img (BGR) -> boxes and kps. RetinaFace-R50 on img resized to size x size: BGR minus (104, 117, 123),
/// priors as onnx_core.anchors, variances 0.1 / 0.2 (biubug6). Its scores run higher than SCRFD's did (real faces
/// ~0.99): the scan's thresholds were kept as they were, which passes more faces (safer; see the module docs).
pub fn landmarks(m: &Models, img: &Mat<u8>, size: usize, thresh: f32, iou: f64, engine: Option<Ep>) -> Faces {
    let ep = engine.unwrap_or(m.engines.landmarks);
    let (h, w) = (img.rows, img.cols);
    let interp = if h.max(w) > size { Interp::Area } else { Interp::Linear };
    let sq = resize_u8(img, size, size, 0.0, 0.0, interp);
    let dims = vec![("batch_size".to_string(), 1), ("height".to_string(), size as i64), ("width".to_string(), size as i64)];
    let out = m.run(crate::models::RETINAFACE, ep, &dims, &[1, 3, size, size], retina_input(&sq));
    decode_landmarks(&out[0].1, &out[1].1, &out[2].1, size, w, h, thresh, iou)
}

/// onnx_core.to_canvas: long side -> CANVAS (1280) px (cubic up, area down). Returns (canvas, k = canvas px per
/// source px).
pub fn to_canvas(frame: &Mat<u8>) -> (Mat<u8>, f64) {
    const CANVAS: usize = 1280;
    let (h, w) = (frame.rows, frame.cols);
    let k = CANVAS as f64 / h.max(w) as f64;
    if (k - 1.0).abs() < 1e-6 {
        return (frame.clone(), 1.0);
    }
    let (nw, nh) = ((w as f64 * k).round_ties_even() as usize, (h as f64 * k).round_ties_even() as usize);
    (resize_u8(frame, nw, nh, 0.0, 0.0, if k > 1.0 { Interp::Cubic } else { Interp::Area }), k)
}

/// onnx_core.RetinaFace.best_face (engine "cpu"): (box [x1, y1, x2, y2], 5 face points) of the most confident face in a
/// BGR image, in its pixels, or None below `min_score`. One face, no NMS (portrait.py: the chosen photo's own face).
pub fn best_face(m: &Models, img: &Mat<u8>, min_score: f32) -> Option<([f32; 4], [[f32; 2]; 5])> {
    let (big, k) = to_canvas(img);
    let (h, w) = (big.rows, big.cols);
    let dims = vec![("batch_size".to_string(), 1), ("height".to_string(), h as i64), ("width".to_string(), w as i64)];
    let out = m.run(crate::models::RETINAFACE, Ep::Cpu, &dims, &[1, 3, h, w], retina_input(&big));
    let (loc, conf, lm) = (&out[0].1, &out[1].1, &out[2].1);
    let pri = mm_ml::post::anchors(h, w);
    // sc.argmax(): the first of the highest
    let mut i = 0;
    for j in 0..pri.len() {
        if conf[j * 2 + 1] > conf[i * 2 + 1] {
            i = j;
        }
    }
    if conf[i * 2 + 1] < min_score {
        return None;
    }
    let (p, l) = (&pri[i], &loc[i * 4..i * 4 + 4]);
    let (wf, hf, kf) = (w as f32, h as f32, k as f32);
    let cx = p[0] + l[0] * 0.1f32 * p[2];
    let cy = p[1] + l[1] * 0.1f32 * p[3];
    let ww = p[2] * (l[2] * 0.2f32).exp();
    let hh = p[3] * (l[3] * 0.2f32).exp();
    let b = [(cx - ww / 2.0) * wf / kf, (cy - hh / 2.0) * hf / kf, (cx + ww / 2.0) * wf / kf, (cy + hh / 2.0) * hf / kf];
    let mut pts = [[0f32; 2]; 5];
    for j in 0..5 {
        pts[j][0] = (p[0] + lm[i * 10 + j * 2] * 0.1f32 * p[2]) * wf / kf;
        pts[j][1] = (p[1] + lm[i * 10 + j * 2 + 1] * 0.1f32 * p[3]) * hf / kf;
    }
    Some((b, pts))
}

/// The part of [`landmarks`] after the model: loc (n,4), conf (n,2), lm (n,10) for a size x size input, boxes in a
/// W x H image.
#[allow(clippy::too_many_arguments)]
pub fn decode_landmarks(loc: &[f32], conf: &[f32], lm: &[f32], size: usize, w: usize, h: usize, thresh: f32, iou: f64) -> Faces {
    let pri = mm_ml::post::anchors(size, size);
    let (wf, hf) = (w as f32, h as f32);
    let mut boxes = Vec::new();
    let mut kps = Vec::new();
    for i in 0..pri.len() {
        let sc = conf[i * 2 + 1];
        if sc < thresh {
            continue; // m = sc >= thresh
        }
        let p = &pri[i];
        let l = &loc[i * 4..i * 4 + 4];
        let cx = p[0] + l[0] * 0.1f32 * p[2];
        let cy = p[1] + l[1] * 0.1f32 * p[3];
        let ww = p[2] * (l[2] * 0.2f32).exp();
        let hh = p[3] * (l[3] * 0.2f32).exp();
        boxes.push([(cx - ww / 2.0) * wf, (cy - hh / 2.0) * hf, (cx + ww / 2.0) * wf, (cy + hh / 2.0) * hf, sc]);
        let mut k = [[0f32; 2]; 5];
        for j in 0..5 {
            k[j][0] = (p[0] + lm[i * 10 + j * 2] * 0.1f32 * p[2]) * wf;
            k[j][1] = (p[1] + lm[i * 10 + j * 2 + 1] * 0.1f32 * p[3]) * hf;
        }
        kps.push(k);
    }
    if boxes.is_empty() {
        return Faces::default();
    }
    let b64: Vec<[f64; 4]> = boxes.iter().map(|b| [b[0] as f64, b[1] as f64, b[2] as f64, b[3] as f64]).collect();
    let s64: Vec<f64> = boxes.iter().map(|b| b[4] as f64).collect();
    let keep = mm_ml::post::nms(&b64, &s64, iou);
    Faces { boxes: keep.iter().map(|&i| boxes[i]).collect(), kps: keep.iter().map(|&i| kps[i]).collect() }
}

/// Least-squares similarity transform src -> dst (Umeyama; what skimage's SimilarityTransform does), as a 2x3
/// matrix. Python takes the SVD of the 2x2 covariance; for 2x2 it has a closed form: with A = cov,
/// p = (a00 + a11) / 2, q = (a10 - a01) / 2, the rotation is atan2(q, p) whether det(A) is positive or negative
/// (Umeyama's reflection fix gives the same rotation), and S @ d = 2 hypot(p, q). Equal to numpy's LAPACK result up
/// to the last bits of a double.
pub fn similarity(src: &[[f64; 2]], dst: &[[f64; 2]]) -> [f64; 6] {
    let n = src.len() as f64;
    let mean = |p: &[[f64; 2]]| {
        let (mut x, mut y) = (0.0, 0.0);
        for v in p {
            x += v[0];
            y += v[1];
        }
        [x / n, y / n]
    };
    let (ms, md) = (mean(src), mean(dst));
    // A = dd.T @ sd / n
    let mut a = [[0f64; 2]; 2];
    for (s, d) in src.iter().zip(dst) {
        let (sx, sy, dx, dy) = (s[0] - ms[0], s[1] - ms[1], d[0] - md[0], d[1] - md[1]);
        a[0][0] += dx * sx;
        a[0][1] += dx * sy;
        a[1][0] += dy * sx;
        a[1][1] += dy * sy;
    }
    for r in a.iter_mut() {
        for v in r.iter_mut() {
            *v /= n;
        }
    }
    let p = (a[0][0] + a[1][1]) / 2.0;
    let q = (a[1][0] - a[0][1]) / 2.0;
    let hq = p.hypot(q);
    let (c, s) = if hq > 0.0 { (p / hq, q / hq) } else { (1.0, 0.0) };
    // sd.var(0).sum()
    let mut var = 0.0;
    for j in 0..2 {
        let m = src.iter().map(|v| v[j] - ms[j]).map(|v| v * v).sum::<f64>() / n;
        var += m;
    }
    let k = 2.0 * hq / var;
    let r = [[k * c, -k * s], [k * s, k * c]];
    let t = [md[0] - (r[0][0] * ms[0] + r[0][1] * ms[1]), md[1] - (r[1][0] * ms[0] + r[1][1] * ms[1])];
    [r[0][0], r[0][1], t[0], r[1][0], r[1][1], t[1]]
}

/// The similarity matrix for 5 landmarks onto ALIGN_DST (float64, as align() makes it).
pub fn align_matrix(kps: &[[f32; 2]; 5]) -> [f64; 6] {
    let src: Vec<[f64; 2]> = kps.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
    let dst: Vec<[f64; 2]> = ALIGN_DST.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
    similarity(&src, &dst)
}

/// The 112x112 face aligned from its 5 landmarks.
pub fn align(img: &Mat<u8>, kps: &[[f32; 2]; 5]) -> Mat<u8> {
    warp_affine_u8(img, &align_matrix(kps), 112, 112, false, Border::Constant, [0.0; 4])
}

/// 112x112 aligned face (BGR) -> 128-d SFace embedding (not normalised), as OpenCV's FaceRecognizerSF: RGB, 0-255.
pub fn embed(m: &Models, aligned: &Mat<u8>) -> Vec<f32> {
    let plane = 112 * 112;
    let mut x = vec![0f32; 3 * plane];
    for i in 0..plane {
        for c in 0..3 {
            x[c * plane + i] = aligned.data[i * 3 + (2 - c)] as f32;
        }
    }
    let out = m.run(crate::models::SFACE, m.engines.sface, &[], &[1, 3, 112, 112], x);
    out[0].1.clone()
}

/// A picture cut out around a head ([`face_crop`]).
pub struct Crop {
    pub img: Mat<u8>,
    pub k: f64,
    pub x0: i64,
    pub y0: i64,
}

/// The picture around head box c=[cx,cy,s], 0.9 head sizes each way, padded black past the edges and scaled up by
/// k to at least 224 px tall; x0, y0 its top left in the picture. None if nothing is left. (landmarks() resizes it
/// again to LANDMARK_SIZE: the 224 is older than that and the face check was measured with it, so it stays until a
/// benchmark says otherwise.)
pub fn face_crop(img: &Mat<u8>, c: &[f64; 3]) -> Option<Crop> {
    let r = c[2] * 0.9;
    let (x1, y1, x2, y2) = ((c[0] - r) as i64, (c[1] - r) as i64, (c[0] + r) as i64, (c[1] + r) as i64);
    let crop = pad_crop(img, x1, y1, x2, y2)?;
    let k = crate::np::pymax(1.0, 224.0 / crop.rows as f64);
    Some(Crop { img: resize_u8(&crop, 0, 0, k, k, Interp::Cubic), k, x0: x1, y0: y1 })
}

/// `img[y1:y2, x1:x2]` of the image padded with black as far as needed (numpy's pad then slice); None if empty.
pub fn pad_crop(img: &Mat<u8>, x1: i64, y1: i64, x2: i64, y2: i64) -> Option<Mat<u8>> {
    if x2 <= x1 || y2 <= y1 {
        return None;
    }
    let (w, h, cn) = ((x2 - x1) as usize, (y2 - y1) as usize, img.cn);
    let mut out = Mat::new(h, w, cn);
    for yy in 0..h {
        let sy = y1 + yy as i64;
        if sy < 0 || sy >= img.rows as i64 {
            continue;
        }
        let (a, b) = (x1.max(0), x2.min(img.cols as i64));
        if b <= a {
            continue;
        }
        let src = &img.row(sy as usize)[a as usize * cn..b as usize * cn];
        let o = (a - x1) as usize * cn;
        out.row_mut(yy)[o..o + src.len()].copy_from_slice(src);
    }
    Some(out)
}
