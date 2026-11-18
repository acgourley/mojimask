//! Body appearance for re-joining people across long gaps, when faces don't show clearly.
//!
//! Person re-ID (OSNet, at the end: crop, body_box, reid_embed), used by the identity questions
//! (scan/questions.py) and the review server. On ONNX Runtime (onnx_core) -- no PyTorch. The DINOv2-small
//! embedder it was compared against lives on in experiments/dinov2_appearance.py, for the finished experiments only.
//!
//! OSNet-AIN x1.0 trained on MSMT17 (torchreid, MIT code; Kaiyang Zhou et al.), exported to ONNX:
//! models/onnx/osnet-ain-x1_0-msmt17.onnx, 256x128 RGB in, 512-d out. Made to match the same person across
//! cameras by clothing and build; the "AIN" variant normalises away lighting. On the test clip's split-track
//! check it ranks a person's other half first 48% of the time and in the top five 75% (DINOv2 on the head + upper
//! body: 34% / 56%), on the WHOLE body as the pose skeleton frames it.
//!
//! Port of mojimask/appearance.py. Parity: native/tools/parity/analysis_reid.py.
use crate::models::Models;
use mm_vision::{resize_u8, Interp, Mat};

/// ImageNet normalisation (reid_embed).
pub const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Crop [x1,y1,x2,y2] (+pad fraction) with black fill outside the frame.
pub fn crop(frame: &Mat<u8>, b: &[f64; 4], pad: f64) -> Option<Mat<u8>> {
    let (w, h) = (b[2] - b[0], b[3] - b[1]);
    let (x1, y1, x2, y2) = ((b[0] - pad * w) as i64, (b[1] - pad * h) as i64, (b[2] + pad * w) as i64, (b[3] + pad * h) as i64);
    // numpy: the frame padded by P on every side, then sliced; a slice can come out empty or clipped at the end
    let (fh, fw) = (frame.rows as i64, frame.cols as i64);
    let p = 0.max(-x1).max(-y1).max(x2 - fw).max(y2 - fh);
    let (ph, pw) = (fh + 2 * p, fw + 2 * p);
    let sl = |a: i64, b: i64, n: i64| -> (i64, i64) {
        // python slice [a:b] on length n (a, b may be negative)
        let f = |v: i64| if v < 0 { (v + n).max(0) } else { v.min(n) };
        let (a, b) = (f(a), f(b));
        (a, b.max(a))
    };
    let (ya, yb) = sl(y1 + p, y2 + p, ph);
    let (xa, xb) = sl(x1 + p, x2 + p, pw);
    if yb <= ya || xb <= xa {
        return None;
    }
    crate::reid::pad_crop(frame, xa - p, ya - p, xb - p, yb - p)
}

/// Whole-body box for a head c=[cx, cy, s]: the detection box of the person's own pose skeleton when there is one,
/// else a head-relative guess (shoulders ~3 heads wide, body ~6 heads tall).
pub fn body_box(c: &[f64; 3], pose: Option<&[f64]>) -> [f64; 4] {
    if let Some(p) = pose {
        return [p[0], p[1], p[2], p[3]];
    }
    [c[0] - 1.6 * c[2], c[1] - 0.8 * c[2], c[0] + 1.6 * c[2], c[1] + 5.5 * c[2]]
}

/// The network input for one crop: 128x256 linear resize, RGB, (x / 255 - MEAN) / STD in float32, CHW.
pub fn reid_input(c: &Mat<u8>) -> Vec<f32> {
    let r = resize_u8(c, 128, 256, 0.0, 0.0, Interp::Linear);
    let plane = 128 * 256;
    let mut x = vec![0f32; 3 * plane];
    for i in 0..plane {
        for ch in 0..3 {
            let v = r.data[i * 3 + (2 - ch)] as f32 / 255.0;
            x[ch * plane + i] = (v - MEAN[ch]) / STD[ch];
        }
    }
    x
}

/// crops: whole-body BGR images (unmasked) -> (n, 512) L2-normalised re-ID features (one batch, as Python runs it).
pub fn reid_embed(m: &Models, crops: &[Mat<u8>]) -> Vec<Vec<f32>> {
    if crops.is_empty() {
        return vec![];
    }
    let mut x = Vec::with_capacity(crops.len() * 3 * 256 * 128);
    for c in crops {
        x.extend(reid_input(c));
    }
    let out = m.run(crate::models::OSNET, m.engines.osnet, &[], &[crops.len(), 3, 256, 128], x);
    let (shape, v) = &out[0];
    let d = shape[1];
    v.chunks_exact(d)
        .map(|r| {
            // v / np.linalg.norm(v, axis=1): sqrt of numpy's pairwise sum of squares, float32
            let sq: Vec<f32> = r.iter().map(|&a| a * a).collect();
            let n = crate::np::sum(&sq).sqrt();
            r.iter().map(|&a| a / n).collect()
        })
        .collect()
}
