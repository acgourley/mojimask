//! Boxes and camera motion, as `mojimask/track.py` computes them (numpy's arithmetic, `np`).
//!
//! A box is a square `[cx, cy, s]` ("cs"); detectors give `[x1, y1, x2, y2]`. Camera motion: `mats_v2.npy`, one 2x3
//! affine per frame taking frame t-1 coordinates to frame t.
use crate::np::{self, M23, M33};

pub type Cs = [f64; 3];
pub type Xyxy = [f64; 4];

/// track.to_cs: box -> centre, size (square).
pub fn to_cs(b: &Xyxy) -> Cs {
    [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0, (b[2] - b[0]).max(b[3] - b[1])]
}

fn pymax(a: f64, b: f64) -> f64 {
    // Python's max(a, b): the first unless the second is greater
    if b > a {
        b
    } else {
        a
    }
}
fn pymin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// track.from_cs (k = 1).
pub fn from_cs(c: &Cs) -> Xyxy {
    let h = c[2] * 1.0 / 2.0;
    [c[0] - h, c[1] - h, c[0] + h, c[1] + h]
}

/// track.iou.
pub fn iou(a: &Xyxy, b: &Xyxy) -> f64 {
    let (x1, y1) = (pymax(a[0], b[0]), pymax(a[1], b[1]));
    let (x2, y2) = (pymin(a[2], b[2]), pymin(a[3], b[3]));
    let inter = pymax(0.0, x2 - x1) * pymax(0.0, y2 - y1);
    let ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    if ua > 0.0 {
        inter / ua
    } else {
        0.0
    }
}

pub const FACE_DY: f64 = 0.08;

/// track.face_to_head: face box -> square head box.
pub fn face_to_head(b: &Xyxy) -> Xyxy {
    let (cx, cy) = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0);
    let s = pymax(b[2] - b[0], b[3] - b[1]) * 1.35;
    [cx - s / 2.0, cy - s / 2.0 - FACE_DY * s, cx + s / 2.0, cy + s / 2.0 - FACE_DY * s]
}

/// track.inside_frac: fraction of square box c inside a W x H frame.
pub fn inside_frac(c: &Cs, w: f64, h: f64) -> f64 {
    let [x1, y1, x2, y2] = from_cs(c);
    let iw = pymax(0.0, pymin(x2, w) - pymax(x1, 0.0));
    let ih = pymax(0.0, pymin(y2, h) - pymax(y1, 0.0));
    iw * ih / pymax((x2 - x1) * (y2 - y1), 1e-6)
}

/// track._heading_out: at own velocity v, would box c be (almost) out of shot within 20 frames?
pub fn heading_out(c: &Cs, v: [f64; 2], w: f64, h: f64) -> bool {
    np::norm(&v) >= 1.0 && inside_frac(&[c[0] + v[0] * 20.0, c[1] + v[1] * 20.0, c[2]], w, h) < 0.05
}

/// track.warp_cs: [cx, cy, s] moved by M.
pub fn warp_cs(c: &Cs, m: &M23) -> Cs {
    let p = np::gemv(m, [c[0], c[1], 1.0]);
    let s = np::det2(m[0][0], m[0][1], m[1][0], m[1][1]).abs().sqrt();
    [p[0], p[1], c[2] * s]
}

fn ext(m: &M23) -> M33 {
    [m[0], m[1], [0.0, 0.0, 1.0]]
}

const EYE: M33 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// track.compose: affine taking frame a coords -> frame b coords (a < b), as 3x3.
pub fn compose33(mats: &[M23], a: i64, b: i64) -> M33 {
    let mut m = EYE;
    for t in a + 1..=b {
        m = np::matmul33(&ext(&mats[t as usize]), &m);
    }
    m
}

pub fn compose(mats: &[M23], a: i64, b: i64) -> M23 {
    let m = compose33(mats, a, b);
    [m[0], m[1]]
}

/// follow.Clip.camera: affine taking frame a coords to frame b coords (either order).
pub fn camera(mats: &[M23], a: i64, b: i64) -> M23 {
    if b >= a {
        compose(mats, a, b)
    } else {
        let m = np::inv33(&ext(&compose(mats, b, a)));
        [m[0], m[1]]
    }
}

/// graph.to_room: box c at frame t in frame 0's coordinates: (xy, size). `inv` caches the inverse motions by frame.
pub fn to_room(mats: &[M23], t: i64, c: &Cs, inv: &mut std::collections::HashMap<i64, M33>) -> ([f64; 2], f64) {
    let m = *inv.entry(t).or_insert_with(|| {
        let m = if t > 0 { ext(&compose(mats, 0, t)) } else { EYE };
        np::inv33(&m)
    });
    let p = np::gemv(&m, [c[0], c[1], 1.0]);
    ([p[0], p[1]], c[2] * np::det2(m[0][0], m[0][1], m[1][0], m[1][1]).abs().sqrt())
}

/// overlaps.covered: fraction of box a (cx, cy, size) inside box b.
pub fn covered(a: &Cs, b: &Cs) -> f64 {
    let (aa, bb) = (from_cs(a), from_cs(b));
    let w = pymax(0.0, pymin(aa[2], bb[2]) - pymax(aa[0], bb[0]));
    let h = pymax(0.0, pymin(aa[3], bb[3]) - pymax(aa[1], bb[1]));
    w * h / ((aa[2] - aa[0]) * (aa[3] - aa[1]))
}

// ---- detections: track.fused_frames
pub const FACE_DETS: [&str; 8] =
    ["retinaface_r50_2x", "scrfd10g_2x", "yolov5m_face_tiled", "yolo11l_face", "yolo12l_face", "sam3_face", "egoblur_gen2", "apple_vision"];
/// pipeline.CORE_DETS: the app's detectors and their thresholds.
pub const CORE_DETS: [(&str, f64); 3] = [("yolov5m_face_tiled", 0.3), ("retinaface_r50_2x", 0.3), ("pose_head", 0.35)];
pub const WEAK_DET: f64 = 0.15;

/// One fused box: (box, score, sources).
#[derive(Clone, Debug)]
pub struct Fused {
    pub b: Xyxy,
    pub s: f64,
    pub srcs: Vec<String>,
}

/// track.fuse: cluster by IoU (> thr), score-weighted average box; score 1 - prod(1 - best score per source).
pub fn fuse(mut boxes: Vec<(Xyxy, f64, String)>, thr: f64) -> Vec<Fused> {
    boxes.sort_by(|a, b| (-a.1).partial_cmp(&-b.1).unwrap()); // stable, like sorted()
    struct Cl {
        b: Xyxy,
        m: Vec<(Xyxy, f64, String)>,
    }
    let mut cl: Vec<Cl> = vec![];
    for (b, s, src) in boxes {
        let mut hit = false;
        for c in cl.iter_mut() {
            if iou(&c.b, &b) > thr {
                c.m.push((b, s, src.clone()));
                let w: Vec<f64> = c.m.iter().map(|m| m.1).collect();
                let ws = np::sum(&w);
                let mut acc = [0.0; 4];
                for (i, m) in c.m.iter().enumerate() {
                    for j in 0..4 {
                        let v = m.0[j] * w[i];
                        acc[j] = if i == 0 { v } else { acc[j] + v };
                    }
                }
                c.b = [acc[0] / ws, acc[1] / ws, acc[2] / ws, acc[3] / ws];
                hit = true;
                break;
            }
        }
        if !hit {
            cl.push(Cl { b, m: vec![(b, s, src)] });
        }
    }
    cl.into_iter()
        .map(|c| {
            let mut best: Vec<(String, f64)> = vec![];
            for (_, s, src) in &c.m {
                match best.iter_mut().find(|(k, _)| k == src) {
                    Some(e) => e.1 = pymax(e.1, *s),
                    None => best.push((src.clone(), pymax(0.0, *s))),
                }
            }
            let mut p = 1.0 - best[0].1;
            for (_, s) in &best[1..] {
                p *= 1.0 - s;
            }
            let mut srcs: Vec<String> = best.iter().map(|x| x.0.clone()).collect();
            srcs.sort();
            Fused { b: c.b, s: 1.0 - p, srcs }
        })
        .collect()
}

/// track.fused_frames: per frame, the fused boxes of the detectors in spec (name, threshold), face boxes as head boxes.
pub fn fused_frames(detdir: &std::path::Path, spec: &[(&str, f64)]) -> Result<Vec<Vec<Fused>>, String> {
    use mm_core::pyjson;
    let mut raw = vec![];
    for (name, _) in spec {
        let d = pyjson::load(&detdir.join(format!("{name}.json")))?;
        let frames: Vec<Vec<Vec<f64>>> = d
            .get("frames")
            .as_list()
            .ok_or("dets: no frames")?
            .iter()
            .map(|f| f.as_list().unwrap().iter().map(|b| b.as_list().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()).collect())
            .collect();
        raw.push(frames);
    }
    let n = raw[0].len();
    let mut out = vec![];
    for t in 0..n {
        let mut boxes = vec![];
        for (i, (name, th)) in spec.iter().enumerate() {
            for e in &raw[i][t] {
                let s = e[e.len() - 1];
                let b = [e[0], e[1], e[2], e[3]];
                assert_eq!(e.len(), 5, "fused_frames: [x1, y1, x2, y2, score]");
                if s >= *th {
                    let b = if FACE_DETS.contains(name) { face_to_head(&b) } else { b };
                    boxes.push((b, s, name.to_string()));
                }
            }
        }
        out.push(fuse(boxes, 0.3));
    }
    Ok(out)
}
