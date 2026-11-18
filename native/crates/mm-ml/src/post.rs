//! Post-processing ported from mojimask/onnx_core.py: nms, YOLOv5-Face decode, RetinaFace anchors + decode,
//! YOLO11-pose decode. Same float widths as numpy there (f32 arrays, f64 inside nms).

/// onnx_core.nms: greedy, best score first (stable order on ties), scores <= 0 dropped.
pub fn nms(b: &[[f64; 4]], s: &[f64], iou: f64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..s.len()).filter(|&i| s[i] > 0.0).collect();
    order.sort_by(|&a, &c| s[c].partial_cmp(&s[a]).unwrap()); // stable, descending
    let area: Vec<f64> = b.iter().map(|r| (r[2] - r[0]) * (r[3] - r[1])).collect();
    let mut keep: Vec<usize> = Vec::new();
    'outer: for &i in &order {
        for &k in &keep {
            let iw = b[i][2].min(b[k][2]) - b[i][0].max(b[k][0]);
            let ih = b[i][3].min(b[k][3]) - b[i][1].max(b[k][1]);
            let inter = if iw > 0.0 && ih > 0.0 { iw * ih } else { 0.0 };
            if inter / (area[i] + area[k] - inter) > iou {
                continue 'outer;
            }
        }
        keep.push(i);
    }
    keep
}

/// YoloFace.detect after the model: p is (25200, 16) rows; returns [x1,y1,x2,y2,score] in the letterboxed image's source px.
pub fn yolo_face(p: &[f32], conf: f32, iou: f64, s: f64, pw: i64, ph: i64) -> Vec<[f64; 5]> {
    let rows: Vec<&[f32]> = p.chunks_exact(16).filter(|r| r[4] >= conf).collect();
    let b32: Vec<[f32; 4]> = rows.iter().map(|r| [r[0] - r[2] / 2.0, r[1] - r[3] / 2.0, r[0] + r[2] / 2.0, r[1] + r[3] / 2.0]).collect();
    let b64: Vec<[f64; 4]> = b32.iter().map(|b| b.map(|v| v as f64)).collect();
    let sc: Vec<f64> = rows.iter().map(|r| r[4] as f64).collect();
    let (s32, pw32, ph32) = (s as f32, pw as f32, ph as f32);
    nms(&b64, &sc, iou)
        .into_iter()
        .map(|k| {
            let b = b32[k];
            [((b[0] - pw32) / s32) as f64, ((b[1] - ph32) / s32) as f64, ((b[2] - pw32) / s32) as f64, ((b[3] - ph32) / s32) as f64, rows[k][4] as f64]
        })
        .collect()
}

/// onnx_core.anchors(h, w): [n,4] cx, cy, w, h as fractions.
pub fn anchors(h: usize, w: usize) -> Vec<[f32; 4]> {
    let mut out = Vec::new();
    for (step, ms) in [(8usize, [16.0, 32.0]), (16, [64.0, 128.0]), (32, [256.0, 512.0])] {
        let (fh, fw) = ((h as f64 / step as f64).ceil() as usize, (w as f64 / step as f64).ceil() as usize);
        for i in 0..fh {
            for j in 0..fw {
                for m in ms {
                    out.push([
                        ((j as f64 + 0.5) * step as f64 / w as f64) as f32,
                        ((i as f64 + 0.5) * step as f64 / h as f64) as f32,
                        (m / w as f64) as f32,
                        (m / h as f64) as f32,
                    ]);
                }
            }
        }
    }
    out
}

/// onnx_core.retina_decode for one prior.
pub fn retina_box(loc: &[f32], pri: &[f32; 4], w: f32, h: f32) -> [f32; 4] {
    let cx = pri[0] + loc[0] * 0.1f32 * pri[2];
    let cy = pri[1] + loc[1] * 0.1f32 * pri[3];
    let ww = pri[2] * (loc[2] * 0.2f32).exp();
    let hh = pri[3] * (loc[3] * 0.2f32).exp();
    [(cx - ww / 2.0) * w, (cy - hh / 2.0) * h, (cx + ww / 2.0) * w, (cy + hh / 2.0) * h]
}

/// RetinaFace.__call__ after the model: loc (n,4), conf (n,2) for a W x H canvas; boxes divided by k.
pub fn retina(loc: &[f32], conf: &[f32], pri: &[[f32; 4]], cw: usize, ch: usize, k: f64, thr: f32, iou: f64) -> Vec<[f64; 5]> {
    let mut idx: Vec<usize> = (0..pri.len()).filter(|&i| conf[i * 2 + 1] > thr).collect();
    if idx.is_empty() {
        return vec![];
    }
    let b: Vec<[f32; 4]> = idx.iter().map(|&i| retina_box(&loc[i * 4..i * 4 + 4], &pri[i], cw as f32, ch as f32)).collect();
    let sc: Vec<f32> = idx.iter().map(|&i| conf[i * 2 + 1]).collect();
    // sc.argsort()[::-1][:5000]: numpy's introsort order (ties included), reversed
    let mut order = crate::npsort::argsort_f32(&sc);
    order.reverse();
    order.truncate(5000);
    idx = order.clone();
    let b2: Vec<[f64; 4]> = idx.iter().map(|&i| b[i].map(|v| v as f64)).collect();
    let s2: Vec<f64> = idx.iter().map(|&i| sc[i] as f64).collect();
    let k32 = k as f32;
    nms(&b2, &s2, iou)
        .into_iter()
        .take(750)
        .map(|j| {
            let bb = b[idx[j]];
            [(bb[0] / k32) as f64, (bb[1] / k32) as f64, (bb[2] / k32) as f64, (bb[3] / k32) as f64, sc[idx[j]] as f64]
        })
        .collect()
}

/// Pose.__call__ after the model: out is (56, A) (batch 0); returns [x1,y1,x2,y2,score, 17*(x,y,c)].
pub fn pose(out: &[f32], a: usize, conf: f32, iou: f64, s: f64, pw: i64, ph: i64) -> Vec<Vec<f64>> {
    let at = |r: usize, i: usize| out[r * a + i];
    let idx: Vec<usize> = (0..a).filter(|&i| at(4, i) >= conf).collect();
    let b32: Vec<[f32; 4]> = idx
        .iter()
        .map(|&i| [at(0, i) - at(2, i) / 2.0, at(1, i) - at(3, i) / 2.0, at(0, i) + at(2, i) / 2.0, at(1, i) + at(3, i) / 2.0])
        .collect();
    let b64: Vec<[f64; 4]> = b32.iter().map(|b| b.map(|v| v as f64)).collect();
    let sc: Vec<f64> = idx.iter().map(|&i| at(4, i) as f64).collect();
    let (s32, pw32, ph32) = (s as f32, pw as f32, ph as f32);
    let r2 = |v: f32| py_round2(v as f64);
    nms(&b64, &sc, iou)
        .into_iter()
        .map(|k| {
            let i = idx[k];
            let b = b32[k];
            let mut v = vec![((b[0] - pw32) / s32) as f64, ((b[1] - ph32) / s32) as f64, ((b[2] - pw32) / s32) as f64, ((b[3] - ph32) / s32) as f64, at(4, i) as f64];
            for j in 0..17 {
                v.push(r2((at(5 + j * 3, i) - pw32) / s32));
                v.push(r2((at(6 + j * 3, i) - ph32) / s32));
                v.push(r2(at(7 + j * 3, i)));
            }
            v
        })
        .collect()
}

/// Python's `round(x, 2)`: the double nearest to x correctly rounded to 2 decimals (half to even on the exact binary
/// value), which `(x * 100).round() / 100` is not (x * 100 rounds first).
pub fn py_round2(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{:.2}", x).parse().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn round2_like_python() {
        // exact ties go to even; 2.675 is 2.67499999... in binary
        for (x, want) in [(0.125, 0.12), (0.375, 0.38), (2.675, 2.67), (1.005, 1.0), (-0.125, -0.12), (0.285, 0.28), (-0.0, -0.0)] {
            assert_eq!(py_round2(x), want, "{x}");
        }
    }
    #[test]
    fn argsort_matches_numpy_small() {
        // np.argsort(np.float32([3, 1, 2, 1, 3, 0])) == [5, 1, 3, 2, 0, 4]
        assert_eq!(crate::npsort::argsort_f32(&[3.0, 1.0, 2.0, 1.0, 3.0, 0.0]), vec![5, 1, 3, 2, 0, 4]);
    }
}
