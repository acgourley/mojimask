//! Small geometry helpers: `pointPolygonTest` (behind.py), `invertAffineTransform` (track.py) and `transform`
//! (video_io.py's NV12 -> BGR matrix).

use crate::mat::Mat;

/// `cv2.pointPolygonTest(poly, pt, measureDist=False)` for a float32 contour: +1 inside, -1 outside, 0 on an edge
/// or vertex. `pt` is rounded to f32 first, as cv2 does (Point2f). Same branch order and double arithmetic as
/// OpenCV's float branch.
pub fn point_polygon_test(poly: &[[f32; 2]], pt: [f32; 2]) -> f64 {
    let total = poly.len();
    if total == 0 {
        return -1.0;
    }
    let (px, py) = (pt[0], pt[1]);
    let mut counter = 0i32;
    let mut v = poly[total - 1];
    for &p in poly {
        let v0 = v;
        v = p;
        if (v0[1] <= py && v[1] <= py) || (v0[1] > py && v[1] > py) || (v0[0] < px && v[0] < px) {
            if py == v[1] && (px == v[0] || (py == v0[1] && ((v0[0] <= px && px <= v[0]) || (v[0] <= px && px <= v0[0])))) {
                return 0.0;
            }
            continue;
        }
        let mut dist = ((py - v0[1]) as f64) * ((v[0] - v0[0]) as f64) - ((px - v0[0]) as f64) * ((v[1] - v0[1]) as f64);
        if dist == 0.0 {
            return 0.0;
        }
        if v[1] < v0[1] {
            dist = -dist;
        }
        counter += (dist > 0.0) as i32;
    }
    if counter % 2 == 0 {
        -1.0
    } else {
        1.0
    }
}

/// `cv2.invertAffineTransform` for a float64 2x3 matrix (row-major). Singular matrices give zeros, as in OpenCV.
pub fn invert_affine(m: &[f64; 6]) -> [f64; 6] {
    let d = m[0] * m[4] - m[1] * m[3];
    let d = if d != 0.0 { 1.0 / d } else { 0.0 };
    let (a11, a22, a12, a21) = (m[4] * d, m[0] * d, -m[1] * d, -m[3] * d);
    let b1 = -a11 * m[2] - a12 * m[5];
    let b2 = -a21 * m[2] - a22 * m[5];
    [a11, a12, b1, a21, a22, b2]
}

/// `cv2.invertAffineTransform` for a float32 matrix. OpenCV works in softfloat with mixed types: the determinant
/// is float, its reciprocal double; `A = M[i] * D` is a softfloat member operator, so D is rounded to float and the
/// product is float; the offsets `-A11*M2 - A12*M5` are double; the result is stored as float.
pub fn invert_affine_f32(m: &[f32; 6]) -> [f32; 6] {
    let d = (m[0] * m[4] - m[1] * m[3]) as f64;
    let d = if d != 0.0 { 1.0 / d } else { 0.0 };
    let df = d as f32;
    let (a11, a22, a12, a21) = ((m[4] * df) as f64, (m[0] * df) as f64, (-m[1] * df) as f64, (-m[3] * df) as f64);
    let b1 = -a11 * m[2] as f64 - a12 * m[5] as f64;
    let b2 = -a21 * m[2] as f64 - a22 * m[5] as f64;
    [a11 as f32, a12 as f32, b1 as f32, a21 as f32, a22 as f32, b2 as f32]
}

/// `cv2.transform(src, m)` for float32 3-channel input and a 3x3 or 3x4 float32 matrix (row-major). On arm64
/// OpenCV runs the plain C loop `m0*v0 + m1*v1 + m2*v2 + m3`, which clang contracts into fused multiply-adds.
pub fn transform3(src: &Mat<f32>, m: &[f32], cols: usize) -> Mat<f32> {
    assert_eq!(src.cn, 3, "transform3: 3 channels");
    assert!(cols == 3 || cols == 4, "transform3: 3x3 or 3x4");
    let mut mm = [0f32; 12];
    for r in 0..3 {
        for c in 0..cols {
            mm[r * 4 + c] = m[r * cols + c];
        }
    }
    let mut out = Mat::new(src.rows, src.cols, 3);
    for (d, s) in out.data.chunks_exact_mut(3).zip(src.data.chunks_exact(3)) {
        for r in 0..3 {
            let k = &mm[r * 4..r * 4 + 4];
            d[r] = dot3(k, s);
        }
    }
    out
}

#[inline]
fn dot3(k: &[f32], s: &[f32]) -> f32 {
    // ((m0*v0 + m1*v1) + m2*v2) + m3, contracted: fma(m2, v2, fma(m0, v0, m1*v1)) + m3
    k[2].mul_add(s[2], k[0].mul_add(s[0], k[1] * s[1])) + k[3]
}
