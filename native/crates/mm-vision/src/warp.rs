//! `cv2.warpAffine` with INTER_LINEAR, with and without WARP_INVERSE_MAP, BORDER_CONSTANT and BORDER_REPLICATE,
//! for u8 and f32 with 1, 3 or 4 channels (reid.py's face alignment, follow.py's templates, depth.py's crops).
//!
//! OpenCV 5 replaced the old fixed-point `INTER_BITS = 5` tables with float kernels (warp_kernels.simd.hpp,
//! `warpAffineLinearInvoker_*`); this ports those:
//! - Without WARP_INVERSE_MAP the matrix is inverted in double (`cv::warpAffine`), then every coefficient is
//!   rounded to float.
//! - Source coordinates in float: blocks of 8 output pixels (NEON) use `fma(M0, x, fma(y, M1, M2))`, the
//!   `width % 8` tail pixels the scalar expression `x*M0 + y*M1 + M2` as clang contracts it.
//! - `ix = floor(sx)`, `a = sx - ix`; each of the 4 neighbours inside the image is read, outside it is the border
//!   value (CONSTANT) or the clamped pixel (REPLICATE).
//! - `v0 = fma(a, p01 - p00, p00)`, `v1 = fma(a, p11 - p10, p10)`, `v = fma(b, v1 - v0, v0)`; u8 rounds half to
//!   even and saturates.

use crate::mat::{par_rows, Border, Mat};

/// The matrix the kernels use: inverse map, float.
fn kernel_matrix(m: &[f64; 6], inverse: bool) -> [f32; 6] {
    let mut m = *m;
    if !inverse {
        let d = m[0] * m[4] - m[1] * m[3];
        let d = if d != 0.0 { 1.0 / d } else { 0.0 };
        let (a11, a22) = (m[4] * d, m[0] * d);
        m[0] = a11;
        m[1] *= -d;
        m[3] *= -d;
        m[4] = a22;
        let b1 = -m[0] * m[2] - m[1] * m[5];
        let b2 = -m[3] * m[2] - m[4] * m[5];
        m[2] = b1;
        m[5] = b2;
    }
    m.map(|v| v as f32)
}

/// Source coordinates of output pixel (x, y), as OpenCV computes them (vector block or scalar tail).
#[inline]
fn coords(m: &[f32; 6], x: usize, y: usize, vector: bool) -> (f32, f32) {
    let (xf, yf) = (x as f32, y as f32);
    if vector {
        let mx = yf.mul_add(m[1], m[2]);
        let my = yf.mul_add(m[4], m[5]);
        (m[0].mul_add(xf, mx), m[3].mul_add(xf, my))
    } else {
        (xf.mul_add(m[0], yf * m[1]) + m[2], xf.mul_add(m[3], yf * m[4]) + m[5])
    }
}

trait WarpPixel: Copy + Default {
    fn to_f(self) -> f32;
    fn from_f(v: f32) -> Self;
    fn from_border(v: f64) -> Self;
}
impl WarpPixel for u8 {
    #[inline]
    fn to_f(self) -> f32 {
        self as f32
    }
    #[inline]
    fn from_f(v: f32) -> u8 {
        (v.round_ties_even() as i32).clamp(0, 255) as u8
    }
    fn from_border(v: f64) -> u8 {
        crate::mat::Pixel::sat_f64(v)
    }
}
impl WarpPixel for f32 {
    #[inline]
    fn to_f(self) -> f32 {
        self
    }
    #[inline]
    fn from_f(v: f32) -> f32 {
        v
    }
    fn from_border(v: f64) -> f32 {
        v as f32
    }
}

fn warp_linear<T: WarpPixel + Send + Sync>(src: &Mat<T>, m: &[f64; 6], w: usize, h: usize, inverse: bool, border: Border, bval: [f64; 4]) -> Mat<T> {
    assert!(border == Border::Constant || border == Border::Replicate, "warp_affine: CONSTANT or REPLICATE");
    let mk = kernel_matrix(m, inverse);
    let cn = src.cn;
    let (sc, sr) = (src.cols as i64, src.rows as i64);
    let bv: Vec<f32> = (0..cn).map(|c| T::from_border(bval[c]).to_f()).collect();
    let mut out = Mat::new(h, w, cn);
    let vend = w / 8 * 8;
    let step = src.cols * cn;
    par_rows(&mut out.data, w * cn, |y, drow| {
        let mut p = [[0f32; 4]; 4];
        for x in 0..w {
            let (sx, sy) = coords(&mk, x, y, x < vend);
            let (ix, iy) = (sx.floor() as i64, sy.floor() as i64);
            let (a, b) = (sx - ix as f32, sy - iy as f32);
            let o = x * cn;
            if ix >= 0 && iy >= 0 && ix + 1 < sc && iy + 1 < sr {
                // all four neighbours inside: the common case
                let base = iy as usize * step + ix as usize * cn;
                let (r0, r1) = (&src.data[base..base + 2 * cn], &src.data[base + step..base + step + 2 * cn]);
                for c in 0..cn {
                    let (p00, p01, p10, p11) = (r0[c].to_f(), r0[cn + c].to_f(), r1[c].to_f(), r1[cn + c].to_f());
                    let v0 = a.mul_add(p01 - p00, p00);
                    let v1 = a.mul_add(p11 - p10, p10);
                    drow[o + c] = T::from_f(b.mul_add(v1 - v0, v0));
                }
                continue;
            }
            if border == Border::Constant && (ix + 1 < 0 || ix >= sc || iy + 1 < 0 || iy >= sr) {
                for c in 0..cn {
                    drow[o + c] = T::from_f(bv[c]);
                }
                continue;
            }
            for (k, (dy, dx)) in [(0i64, 0i64), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
                let (px, py) = (ix + dx, iy + dy);
                if px >= 0 && px < sc && py >= 0 && py < sr {
                    let s = &src.row(py as usize)[px as usize * cn..];
                    for c in 0..cn {
                        p[k][c] = s[c].to_f();
                    }
                } else if border == Border::Constant {
                    p[k][..cn].copy_from_slice(&bv);
                } else {
                    let (qx, qy) = (px.clamp(0, sc - 1) as usize, py.clamp(0, sr - 1) as usize);
                    let s = &src.row(qy)[qx * cn..];
                    for c in 0..cn {
                        p[k][c] = s[c].to_f();
                    }
                }
            }
            for c in 0..cn {
                let v0 = a.mul_add(p[1][c] - p[0][c], p[0][c]);
                let v1 = a.mul_add(p[3][c] - p[2][c], p[2][c]);
                drow[o + c] = T::from_f(b.mul_add(v1 - v0, v0));
            }
        }
    });
    out
}

/// `cv2.warpAffine(src, M, (w, h), flags=INTER_LINEAR [| WARP_INVERSE_MAP], borderMode, borderValue)` for u8.
pub fn warp_affine_u8(src: &Mat<u8>, m: &[f64; 6], w: usize, h: usize, inverse: bool, border: Border, bval: [f64; 4]) -> Mat<u8> {
    warp_linear(src, m, w, h, inverse, border, bval)
}

/// The same for f32.
pub fn warp_affine_f32(src: &Mat<f32>, m: &[f64; 6], w: usize, h: usize, inverse: bool, border: Border, bval: [f64; 4]) -> Mat<f32> {
    warp_linear(src, m, w, h, inverse, border, bval)
}
