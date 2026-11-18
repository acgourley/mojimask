//! `cv2.GaussianBlur` for u8 and f32, 1-4 channels, default border (BORDER_REFLECT_101), bit-exact with cv2 5.0.0.
//!
//! - **Kernel** (`getGaussianKernelBitExact`): with sigma <= 0 and ksize 1/3/5/7/9 OpenCV's fixed tables
//!   (binomial-like); otherwise sigma = `0.15 * n + 0.35` (fused) when not given, and `exp(-x^2 / (8 sigma^2))`
//!   normalised, all in OpenCV's softdouble arithmetic. Its `exp` is OpenCV's own table-plus-polynomial
//!   (softfloat.cpp `f64_exp`), not libm's; ported here exactly ([`soft_exp`]).
//! - **ksize from sigma:** `cvRound(sigma * (u8 ? 3 : 4) * 2 + 1) | 1`; a dimension of size 1 gets ksize 1.
//! - **u8** (`GaussianBlurFixedPoint`): the kernel is rounded to 8 fractional bits with error diffusion so it sums
//!   to exactly 256; rows are filtered into u16 (8 fractional bits), columns into u32 (16), then
//!   `(v + 2^15) >> 16`. All integer and never saturating, so OpenCV's many special-cased loops (121, 14641, ...)
//!   give the same numbers as this one plain loop.
//! - **f32** (`sepFilter2D` -> FilterEngine): the kernel as float; rows: 3 taps `fma(c, k0, (l + r) * k1)`,
//!   5 taps `fma(l2 + r2, k2, fma(c, k0, (l1 + r1) * k1))`, longer kernels a left-to-right fma chain over the taps;
//!   columns: 3 taps `fma(u + d, k1, c * k0)`, longer ones `c * k0` then `fma(u_k + d_k, k_k, s)` outward. These
//!   are the NEON blocks' operations, and clang contracts the scalar tails to the same ones.

use crate::mat::{border_index, par_rows, Border, Mat};

const EXP_TAB: [u64; 64] = [
    0x3ff0000000000000, 0x3ff02c9a3e778061, 0x3ff059b0d3158574, 0x3ff0874518759bc8,
    0x3ff0b5586cf9890f, 0x3ff0e3ec32d3d1a2, 0x3ff11301d0125b51, 0x3ff1429aaea92de0,
    0x3ff172b83c7d517b, 0x3ff1a35beb6fcb75, 0x3ff1d4873168b9aa, 0x3ff2063b88628cd6,
    0x3ff2387a6e756238, 0x3ff26b4565e27cdd, 0x3ff29e9df51fdee1, 0x3ff2d285a6e4030b,
    0x3ff306fe0a31b715, 0x3ff33c08b26416ff, 0x3ff371a7373aa9cb, 0x3ff3a7db34e59ff7,
    0x3ff3dea64c123422, 0x3ff4160a21f72e2a, 0x3ff44e086061892d, 0x3ff486a2b5c13cd0,
    0x3ff4bfdad5362a27, 0x3ff4f9b2769d2ca7, 0x3ff5342b569d4f82, 0x3ff56f4736b527da,
    0x3ff5ab07dd485429, 0x3ff5e76f15ad2148, 0x3ff6247eb03a5585, 0x3ff6623882552225,
    0x3ff6a09e667f3bcd, 0x3ff6dfb23c651a2f, 0x3ff71f75e8ec5f74, 0x3ff75feb564267c9,
    0x3ff7a11473eb0187, 0x3ff7e2f336cf4e62, 0x3ff82589994cce13, 0x3ff868d99b4492ed,
    0x3ff8ace5422aa0db, 0x3ff8f1ae99157736, 0x3ff93737b0cdc5e5, 0x3ff97d829fde4e50,
    0x3ff9c49182a3f090, 0x3ffa0c667b5de565, 0x3ffa5503b23e255d, 0x3ffa9e6b5579fdbf,
    0x3ffae89f995ad3ad, 0x3ffb33a2b84f15fb, 0x3ffb7f76f2fb5e47, 0x3ffbcc1e904bc1d2,
    0x3ffc199bdd85529c, 0x3ffc67f12e57d14b, 0x3ffcb720dcef9069, 0x3ffd072d4a07897c,
    0x3ffd5818dcfba487, 0x3ffda9e603db3285, 0x3ffdfc97337b9b5f, 0x3ffe502ee78b3ff6,
    0x3ffea4afa2a490da, 0x3ffefa1bee615a27, 0x3fff50765b6e4540, 0x3fffa7c1819e90d8
];

/// OpenCV's softdouble `exp` (softfloat.cpp `f64_exp`), in plain IEEE double arithmetic (softfloat's add/mul/div
/// are correctly rounded, like the hardware's; no fused operations).
pub fn soft_exp(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { x } else { 0.0 };
    }
    let a0c = f64::from_bits(0x3f83ce0f3e46f431);
    let a5 = 1.0 / a0c;
    let a4 = f64::from_bits(0x3fe62e42fefa39f1) / a0c;
    let a3 = f64::from_bits(0x3fcebfbdff82a45a) / a0c;
    let a2 = f64::from_bits(0x3fac6b08d81fec75) / a0c;
    let a1 = f64::from_bits(0x3f83b2a72b4f3cd3) / a0c;
    let a0 = f64::from_bits(0x3f55e7aa1566c2a4) / a0c;
    let prescale = f64::from_bits(0x3ff71547652b82fe) * 64.0;
    let postscale = 1.0 / 64.0;
    let max_val = 3000.0 * 64.0;
    let exp_bits = ((x.to_bits() >> 52) & 0x7ff) as i64;
    let mut x0 = if exp_bits > 1023 + 10 { if x < 0.0 { -max_val } else { max_val } } else { x * prescale };
    let val0 = x0.round_ties_even() as i32;
    let t = ((val0 >> 6) + 1023).clamp(0, 2047) as u64;
    let buf = f64::from_bits(t << 52);
    x0 = (x0 - x0.round_ties_even()) * postscale;
    let poly = ((((a0 * x0 + a1) * x0 + a2) * x0 + a3) * x0 + a4) * x0 + a5;
    buf * a0c * f64::from_bits(EXP_TAB[(val0 & 63) as usize]) * poly
}

/// `getGaussianKernelBitExact`: the kernel in double (softdouble values).
pub fn gaussian_kernel(n: usize, sigma: f64) -> Vec<f64> {
    assert!(n > 0);
    if sigma <= 0.0 {
        let fixed: Option<&[f64]> = match n {
            1 => Some(&[1.0]),
            3 => Some(&[0.25, 0.5, 0.25]),
            5 => Some(&[0.0625, 0.25, 0.375, 0.25, 0.0625]),
            7 => Some(&[0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125]),
            9 => Some(&[4.0 / 256.0, 13.0 / 256.0, 30.0 / 256.0, 51.0 / 256.0, 60.0 / 256.0, 51.0 / 256.0, 30.0 / 256.0, 13.0 / 256.0, 4.0 / 256.0]),
            _ => None,
        };
        if let Some(k) = fixed {
            return k.to_vec();
        }
    }
    let sigma_x = if sigma > 0.0 { sigma } else { (n as f64).mul_add(f64::from_bits(0x3fc3333333333333), f64::from_bits(0x3fd6666666666666)) };
    let scale2x = -0.125 / (sigma_x * sigma_x);
    let n2 = (n - 1) / 2;
    let mut values = vec![0f64; n2 + 1];
    let mut sum = 0.0;
    let mut x = 1 - n as i64;
    for v in values.iter_mut().take(n2) {
        let t = soft_exp(((x * x) as f64) * scale2x);
        *v = t;
        sum += t;
        x += 2;
    }
    sum *= 2.0;
    sum += 1.0;
    if n & 1 == 0 {
        sum += 1.0;
    }
    let mul1 = 1.0 / sum;
    let mut res = vec![0f64; n];
    for i in 0..n2 {
        let t = values[i] * mul1;
        res[i] = t;
        res[n - 1 - i] = t;
    }
    res[n2] = mul1;
    if n & 1 == 0 {
        res[n2 + 1] = res[n2];
    }
    res
}

/// `getGaussianKernelFixedPoint_ED` with 8 fractional bits (ufixedpoint16): sums to exactly 256.
pub fn gaussian_kernel_fixed(k: &[f64]) -> Vec<u32> {
    let n = k.len();
    assert!(n & 1 == 1, "fixed-point Gaussian kernel must be odd");
    let mul = 256.0f64;
    let n2 = n / 2;
    let mut res = vec![0i64; n];
    let mut err = 0.0f64;
    let mut sum = 0i64;
    for i in 0..n2 {
        let adj = k[i] * mul + err;
        let v0 = adj.round_ties_even() as i64;
        err = adj - v0 as f64;
        res[i] = v0;
        res[n - 1 - i] = v0;
        sum += v0;
    }
    res[n2] = 256 - 2 * sum;
    res.into_iter().map(|v| v as u32).collect()
}

/// ksize and sigma as `createGaussianKernels` derives them, then the dimension-of-1 rule from `GaussianBlur`.
fn sizes(rows: usize, cols: usize, kw: usize, kh: usize, s1: f64, s2: f64, is_u8: bool) -> (usize, usize, f64, f64) {
    let (mut kw, mut kh) = (kw, kh);
    // GaussianBlur: 1-pixel dimensions get ksize 1 (non-constant border)
    if rows == 1 {
        kh = 1;
    }
    if cols == 1 {
        kw = 1;
    }
    let s2 = if s2 <= 0.0 { s1 } else { s2 };
    let mul = if is_u8 { 3.0 } else { 4.0 };
    if kw == 0 && s1 > 0.0 {
        kw = ((s1 * mul * 2.0 + 1.0).round_ties_even() as usize) | 1;
    }
    if kh == 0 && s2 > 0.0 {
        kh = ((s2 * mul * 2.0 + 1.0).round_ties_even() as usize) | 1;
    }
    assert!(kw % 2 == 1 && kh % 2 == 1, "GaussianBlur: ksize must be odd and positive");
    (kw, kh, s1.max(0.0), s2.max(0.0))
}

fn kernels(kw: usize, kh: usize, s1: f64, s2: f64) -> (Vec<f64>, Vec<f64>) {
    let kx = gaussian_kernel(kw, s1);
    let ky = if kh == kw && (s1 - s2).abs() < f64::EPSILON { kx.clone() } else { gaussian_kernel(kh, s2) };
    (kx, ky)
}

/// Border index tables: for output position p and tap j, the source index (REFLECT_101).
fn index_table(len: usize, k: usize) -> Vec<usize> {
    let a = (k / 2) as isize;
    let mut t = Vec::with_capacity(len * k);
    for p in 0..len as isize {
        for j in 0..k as isize {
            t.push(border_index(p - a + j, len, Border::Reflect101) as usize);
        }
    }
    t
}

/// A row padded by k/2 pixels on each side with REFLECT_101 (what FilterEngine's row buffer holds).
fn pad_row<T: Copy>(row: &[T], cols: usize, cn: usize, k: usize, out: &mut Vec<T>) {
    let a = (k / 2) as isize;
    out.clear();
    for p in -a..cols as isize + a {
        let q = border_index(p, cols, Border::Reflect101) as usize;
        out.extend_from_slice(&row[q * cn..(q + 1) * cn]);
    }
}

/// `cv2.GaussianBlur(src, (kw, kh), sx, sigmaY=sy)` for u8 (kw = kh = 0 derives the size from sigma).
pub fn gaussian_blur_u8(src: &Mat<u8>, kw: usize, kh: usize, sx: f64, sy: f64) -> Mat<u8> {
    let (kw, kh, s1, s2) = sizes(src.rows, src.cols, kw, kh, sx, sy, true);
    if kw == 1 && kh == 1 {
        return src.clone();
    }
    let (kx, ky) = kernels(kw, kh, s1, s2);
    let (fx, fy) = (gaussian_kernel_fixed(&kx), gaussian_kernel_fixed(&ky));
    let (rows, cols, cn) = (src.rows, src.cols, src.cn);
    let yt = index_table(rows, kh);
    let w = cols * cn;
    // rows: u16 with 8 fractional bits (sum of k_j * src, at most 255 * 256)
    let mut tmp = vec![0u16; rows * w];
    par_rows(&mut tmp, w, |y, d| {
        let mut pad = Vec::with_capacity((cols + kw) * cn);
        pad_row(src.row(y), cols, cn, kw, &mut pad);
        let mut acc = vec![0u32; w];
        for (j, &k) in fx.iter().enumerate() {
            let p = &pad[j * cn..j * cn + w];
            for (a, &v) in acc.iter_mut().zip(p) {
                *a += k * v as u32;
            }
        }
        for (o, a) in d.iter_mut().zip(acc) {
            *o = a as u16;
        }
    });
    // columns: u32 with 16 fractional bits, then (v + 2^15) >> 16
    let mut out = Mat::new(rows, cols, cn);
    par_rows(&mut out.data, w, |y, d| {
        let iy = &yt[y * kh..(y + 1) * kh];
        let mut acc = vec![0u32; w];
        for (j, &k) in fy.iter().enumerate() {
            let r = &tmp[iy[j] * w..(iy[j] + 1) * w];
            for (a, &v) in acc.iter_mut().zip(r) {
                *a += k * v as u32;
            }
        }
        for (o, a) in d.iter_mut().zip(acc) {
            *o = ((a + (1 << 15)) >> 16).min(255) as u8;
        }
    });
    out
}

/// `cv2.GaussianBlur` for f32.
pub fn gaussian_blur_f32(src: &Mat<f32>, kw: usize, kh: usize, sx: f64, sy: f64) -> Mat<f32> {
    let (kw, kh, s1, s2) = sizes(src.rows, src.cols, kw, kh, sx, sy, false);
    if kw == 1 && kh == 1 {
        return src.clone();
    }
    let (kx, ky) = kernels(kw, kh, s1, s2);
    let kx: Vec<f32> = kx.iter().map(|&v| v as f32).collect();
    let ky: Vec<f32> = ky.iter().map(|&v| v as f32).collect();
    sep_filter_symm_f32(src, &kx, &ky)
}

/// FilterEngine's float row filter for a symmetric kernel on one padded row (`p`, k/2 pixels each side).
/// SymmRowSmallFilter (3 or 5 taps) works in blocks of 4 elements (NEON), then pairs, then one last single element
/// if the count is odd, which sums `k0 * c` then `fma(k_j, l_j + r_j, s)` outward (a different order from the
/// blocks'). Longer kernels (RowFilter) use a left-to-right fma chain everywhere.
fn row_filter_f32(p: &[f32], k: &[f32], cn: usize, d: &mut [f32]) {
    let w = d.len();
    let n = k.len();
    let a = n / 2;
    let at = |j: usize| &p[j * cn..j * cn + w];
    match n {
        1 => d.copy_from_slice(at(0)),
        3 => {
            let (l, c, r) = (at(0), at(1), at(2));
            for i in 0..w {
                d[i] = c[i].mul_add(k[1], (l[i] + r[i]) * k[2]);
            }
        }
        5 => {
            let (l2, l1, c, r1, r2) = (at(0), at(1), at(2), at(3), at(4));
            for i in 0..w {
                d[i] = (r2[i] + l2[i]).mul_add(k[4], c[i].mul_add(k[2], (l1[i] + r1[i]) * k[3]));
            }
        }
        _ => {
            let p0 = at(0);
            for i in 0..w {
                d[i] = p0[i] * k[0];
            }
            for j in 1..n {
                let pj = at(j);
                let kj = k[j];
                for i in 0..w {
                    d[i] = pj[i].mul_add(kj, d[i]);
                }
            }
        }
    }
    if (n == 3 || n == 5) && (w - w / 4 * 4) % 2 == 1 {
        let i = w - 1;
        let v = |j: usize| p[j * cn + i];
        let mut s = k[a] * v(a);
        for j in 1..=a {
            s = k[a + j].mul_add(v(a + j) + v(a - j), s);
        }
        d[i] = s;
    }
}

/// FilterEngine's separable float filter for symmetric kernels, BORDER_REFLECT_101. Columns: 3 taps
/// (SymmColumnSmallFilter) `fma(u + d, k1, c * k0)`; otherwise `c * k0` then `fma(u_k + d_k, k_k, s)` outward.
pub fn sep_filter_symm_f32(src: &Mat<f32>, kx: &[f32], ky: &[f32]) -> Mat<f32> {
    let (kw, kh) = (kx.len(), ky.len());
    let (rows, cols, cn) = (src.rows, src.cols, src.cn);
    let yt = index_table(rows, kh);
    let w = cols * cn;
    let mut tmp = vec![0f32; rows * w];
    par_rows(&mut tmp, w, |y, d| {
        let mut pad = Vec::with_capacity((cols + kw) * cn);
        pad_row(src.row(y), cols, cn, kw, &mut pad);
        row_filter_f32(&pad, kx, cn, d);
    });
    let mut out = Mat::new(rows, cols, cn);
    let a = kh / 2;
    par_rows(&mut out.data, w, |y, d| {
        let iy = &yt[y * kh..(y + 1) * kh];
        let r = |j: usize| &tmp[iy[j] * w..(iy[j] + 1) * w];
        match kh {
            1 => d.copy_from_slice(r(0)),
            3 => {
                let (r0, r1, r2) = (r(0), r(1), r(2));
                for i in 0..w {
                    d[i] = (r0[i] + r2[i]).mul_add(ky[2], r1[i] * ky[1]);
                }
            }
            _ => {
                let rc = r(a);
                for i in 0..w {
                    d[i] = rc[i] * ky[a];
                }
                for k in 1..=a {
                    let (up, dn) = (r(a + k), r(a - k));
                    let f = ky[a + k];
                    for i in 0..w {
                        d[i] = (up[i] + dn[i]).mul_add(f, d[i]);
                    }
                }
            }
        }
    });
    out
}
