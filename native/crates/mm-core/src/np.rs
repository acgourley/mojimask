//! numpy's arithmetic, bit for bit, where the Python modules read it (the scan's rules, the analysis modules, the
//! edits' geometry). Shared by mm-scan and mm-analysis (their `np` modules re-export this one).
//!
//! The rules compare floats against thresholds and print them rounded, so a last-bit difference can flip one. What
//! numpy does for each operation used:
//! - `A @ x` (2x3 or 3x3 by a 3-vector) is BLAS `dgemv`, `A @ B` (3x3) is `dgemm`, `x @ y` and `np.linalg.norm` are
//!   `ddot` / `sdot`, `np.linalg.det` is LAPACK `dgetrf` then `sign * exp(sum log|u_ii|)`, `np.linalg.inv` is
//!   `dgesv` against the identity. The project's numpy links Apple's Accelerate (`$NEWLAPACK$ILP64` symbols), so on
//!   macOS these call the same routines in the same library: identical by construction (checked against numpy on
//!   random inputs by `tools/parity/scan_numcheck.py`). Elsewhere a plain implementation stands in (not bit-exact;
//!   Accelerate's kernels use fused multiply-adds in an order of their own).
//! - `arr.sum()` over a contiguous float axis is numpy's pairwise summation ([`sum`], in the array's own width:
//!   float32 or float64); a reduction over axis 0 of a 2-D array adds the rows in order.
//! - `np.mean` divides by the count as a float64 (numpy 2 divides by an `intp` scalar, so float32 means are rounded
//!   twice: to float64, then to float32).
//! - `np.median`: the middle value, or the mean of the two middle ones (in the array's width); `np.percentile`
//!   (linear): numpy's virtual index and `_lerp` (for float32, the difference in float32, the rest in float64).
//! - Python's `round()` ([`pyround`], [`pyround_n`]: correctly rounded, halves to even) is not numpy's `around`
//!   ([`np_round`]); each caller says which one its value gets.
//! - `np.hypot`, `log`, `exp`: the system libm, as numpy calls it.
//!
//! Checked against numpy 2.4 (the project's) by `tools/parity/scan_numcheck.py`, `tools/parity/core_numcheck.py` and
//! the tests below.

#[cfg(target_os = "macos")]
mod accel {
    pub const ROW_MAJOR: i32 = 101;
    pub const COL_MAJOR: i32 = 102;
    pub const NO_TRANS: i32 = 111;
    pub const TRANS: i32 = 112;
    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        #[link_name = "cblas_ddot$NEWLAPACK$ILP64"]
        pub fn cblas_ddot(n: i64, x: *const f64, incx: i64, y: *const f64, incy: i64) -> f64;
        #[link_name = "cblas_sdot$NEWLAPACK$ILP64"]
        pub fn cblas_sdot(n: i64, x: *const f32, incx: i64, y: *const f32, incy: i64) -> f32;
        #[link_name = "cblas_dgemv$NEWLAPACK$ILP64"]
        pub fn cblas_dgemv(order: i32, trans: i32, m: i64, n: i64, alpha: f64, a: *const f64, lda: i64, x: *const f64, incx: i64,
                           beta: f64, y: *mut f64, incy: i64);
        #[link_name = "cblas_dgemm$NEWLAPACK$ILP64"]
        pub fn cblas_dgemm(order: i32, ta: i32, tb: i32, m: i64, n: i64, k: i64, alpha: f64, a: *const f64, lda: i64, b: *const f64,
                           ldb: i64, beta: f64, c: *mut f64, ldc: i64);
        #[link_name = "dgetrf$NEWLAPACK$ILP64"]
        pub fn dgetrf(m: *const i64, n: *const i64, a: *mut f64, lda: *const i64, ipiv: *mut i64, info: *mut i64);
        #[link_name = "dgesv$NEWLAPACK$ILP64"]
        pub fn dgesv(n: *const i64, nrhs: *const i64, a: *mut f64, lda: *const i64, ipiv: *mut i64, b: *mut f64, ldb: *const i64,
                     info: *mut i64);
    }
}

pub type M23 = [[f64; 3]; 2];
pub type M33 = [[f64; 3]; 3];

/// `A @ x` for a row-major R x 3 matrix (R = 2 or 3): numpy's matrix-vector path (`dgemv`, column-major, transposed).
pub fn gemv<const R: usize>(a: &[[f64; 3]; R], x: [f64; 3]) -> [f64; R] {
    let mut y = [0.0; R];
    #[cfg(target_os = "macos")]
    unsafe {
        let flat: Vec<f64> = a.iter().flatten().copied().collect();
        accel::cblas_dgemv(accel::COL_MAJOR, accel::TRANS, 3, R as i64, 1.0, flat.as_ptr(), 3, x.as_ptr(), 1, 0.0, y.as_mut_ptr(), 1);
    }
    #[cfg(not(target_os = "macos"))]
    for i in 0..R {
        y[i] = a[i][0] * x[0] + a[i][1] * x[1] + a[i][2] * x[2];
    }
    y
}

/// `A @ x` for a row-major rows x cols matrix (flat): numpy's matrix-vector path, any size.
pub fn gemv_n(a: &[f64], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), rows * cols);
    assert_eq!(x.len(), cols);
    let mut y = vec![0.0; rows];
    #[cfg(target_os = "macos")]
    unsafe {
        accel::cblas_dgemv(accel::COL_MAJOR, accel::TRANS, cols as i64, rows as i64, 1.0, a.as_ptr(), cols as i64, x.as_ptr(), 1, 0.0,
                           y.as_mut_ptr(), 1);
    }
    #[cfg(not(target_os = "macos"))]
    for i in 0..rows {
        y[i] = (0..cols).map(|j| a[i * cols + j] * x[j]).sum();
    }
    y
}

/// `A @ B` for 3x3 matrices (C order): numpy's matrix-matrix path (`dgemm`, row-major).
pub fn matmul33(a: &M33, b: &M33) -> M33 {
    let mut c = [[0.0; 3]; 3];
    #[cfg(target_os = "macos")]
    unsafe {
        let fa: Vec<f64> = a.iter().flatten().copied().collect();
        let fb: Vec<f64> = b.iter().flatten().copied().collect();
        let mut fc = [0.0f64; 9];
        accel::cblas_dgemm(accel::ROW_MAJOR, accel::NO_TRANS, accel::NO_TRANS, 3, 3, 3, 1.0, fa.as_ptr(), 3, fb.as_ptr(), 3, 0.0,
                           fc.as_mut_ptr(), 3);
        for i in 0..3 {
            for j in 0..3 {
                c[i][j] = fc[i * 3 + j];
            }
        }
    }
    #[cfg(not(target_os = "macos"))]
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

/// `x @ y` / `x.dot(y)` for float64 vectors (`ddot`).
/// numpy's own dot for short vectors (not BLAS): two lanes, no fused multiply-add, then the lanes added. Checked
/// against numpy for lengths up to 4; the scan only uses length 2.
pub fn ddot(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    assert!(x.len() <= 4, "ddot: checked up to length 4");
    let (mut a, mut b, mut i) = (0.0f64, 0.0f64, 0);
    while i + 2 <= x.len() {
        a += x[i] * y[i];
        b += x[i + 1] * y[i + 1];
        i += 2;
    }
    let mut s = a + b;
    while i < x.len() {
        s += x[i] * y[i];
        i += 1;
    }
    s
}

/// numpy's DOUBLE_dot as `_pyarray_correlate` calls it (np.convolve's edges and long kernels): `0. + cblas_ddot`
/// (Accelerate on macOS; a plain loop elsewhere, not bit-exact).
pub fn blas_ddot(x: &[f64], y: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    #[cfg(target_os = "macos")]
    {
        0.0 + unsafe { accel::cblas_ddot(x.len() as i64, x.as_ptr(), 1, y.as_ptr(), 1) }
    }
    #[cfg(not(target_os = "macos"))]
    {
        x.iter().zip(y).fold(0.0, |s, (a, b)| s + a * b)
    }
}

/// np.convolve(a, v) (mode "full"), as numpy computes it: `_pyarray_correlate` with v reversed, the edges by
/// [`blas_ddot`], the middle by `small_correlate` (kernels of 11 taps or fewer: fused multiply-adds in order) or
/// `blas_ddot`. Checked against numpy on 3,000 random cases (mm-records' resolve notes).
pub fn convolve(a: &[f64], v: &[f64]) -> Vec<f64> {
    let (a, v) = if v.len() > a.len() { (v, a) } else { (a, v) };
    let k: Vec<f64> = v.iter().rev().copied().collect();
    let (n1, n2) = (a.len(), k.len());
    let mut out = Vec::with_capacity(n1 + n2 - 1);
    let nl = n2 - 1;
    let (mut n, mut i2) = (1usize, nl);
    for _ in 0..nl {
        out.push(blas_ddot(&a[..n], &k[i2..i2 + n]));
        n += 1;
        i2 = i2.saturating_sub(1);
    }
    let mut i1 = 0;
    for _ in 0..(n1 - n2 + 1) {
        let d = &a[i1..i1 + n2];
        out.push(if n2 <= 11 {
            let mut s = 0.0f64;
            for j in 0..n2 {
                s = d[j].mul_add(k[j], s);
            }
            s
        } else {
            blas_ddot(d, &k)
        });
        i1 += 1;
    }
    let mut n = n2;
    for _ in 0..nl {
        n -= 1;
        out.push(blas_ddot(&a[i1..i1 + n], &k[..n]));
        i1 += 1;
    }
    out
}

/// One row of `np.einsum("nij,nj->ni")` with j = 3: NEON's two lanes, fma(a2, b2, a0 * b0) + a1 * b1.
pub fn einsum_row3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[2].mul_add(b[2], a[0] * b[0]) + a[1] * b[1]
}

/// `x @ y` for float32 vectors (`sdot`; numpy adds the one chunk to a double 0 and rounds back: the same value).
pub fn sdot(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len());
    #[cfg(target_os = "macos")]
    unsafe {
        accel::cblas_sdot(x.len() as i64, x.as_ptr(), 1, y.as_ptr(), 1)
    }
    #[cfg(not(target_os = "macos"))]
    {
        x.iter().zip(y).map(|(a, b)| a * b).sum()
    }
}

/// `np.linalg.norm(x)` for a float64 vector: sqrt(x . x).
pub fn norm(x: &[f64]) -> f64 {
    ddot(x, x).sqrt()
}

/// `np.linalg.norm(x)` for a float32 vector: sqrt in float32.
pub fn norm_f32(x: &[f32]) -> f32 {
    sdot(x, x).sqrt()
}

/// `np.linalg.det` of a 2x2 [[a, b], [c, d]]: LU (dgetrf), then sign * exp(log|u11| + log|u22|).
pub fn det2(a: f64, b: f64, c: f64, d: f64) -> f64 {
    #[cfg(target_os = "macos")]
    {
        let mut m = [a, c, b, d]; // column-major
        let (n, mut ipiv, mut info) = (2i64, [0i64; 2], 0i64);
        unsafe { accel::dgetrf(&n, &n, m.as_mut_ptr(), &n, ipiv.as_mut_ptr(), &mut info) };
        if info != 0 {
            return 0.0;
        }
        let mut sign = if ((ipiv[0] != 1) as i32 + (ipiv[1] != 2) as i32) % 2 == 1 { -1.0 } else { 1.0 };
        let mut logdet = 0.0f64;
        for v in [m[0], m[3]] {
            let mut x = v;
            if x < 0.0 {
                sign = -sign;
                x = -x;
            }
            logdet += x.ln();
        }
        sign * logdet.exp()
    }
    #[cfg(not(target_os = "macos"))]
    {
        a * d - b * c
    }
}

/// `np.linalg.inv` of a 3x3 (dgesv against the identity). Panics on a singular matrix (numpy raises).
pub fn inv33(a: &M33) -> M33 {
    #[cfg(target_os = "macos")]
    {
        let mut m = [0.0f64; 9];
        let mut b = [0.0f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                m[j * 3 + i] = a[i][j];
            }
            b[i * 3 + i] = 1.0;
        }
        let (n, mut ipiv, mut info) = (3i64, [0i64; 3], 0i64);
        unsafe { accel::dgesv(&n, &n, m.as_mut_ptr(), &n, ipiv.as_mut_ptr(), b.as_mut_ptr(), &n, &mut info) };
        assert!(info == 0, "inv33: singular matrix");
        let mut x = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                x[i][j] = b[j * 3 + i];
            }
        }
        x
    }
    #[cfg(not(target_os = "macos"))]
    {
        let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
        let mut x = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let (r0, r1) = ((j + 1) % 3, (j + 2) % 3);
                let (c0, c1) = ((i + 1) % 3, (i + 2) % 3);
                x[i][j] = (a[r0][c0] * a[r1][c1] - a[r0][c1] * a[r1][c0]) / det;
            }
        }
        x
    }
}

/// `np.mean(rows, 0)` of float32 rows: rows added in order, then divided by the count in float64 and rounded back.
pub fn mean_rows_f32(rows: &[&[f32]]) -> Vec<f32> {
    let mut acc = rows[0].to_vec();
    for r in &rows[1..] {
        for (a, b) in acc.iter_mut().zip(r.iter()) {
            *a += *b;
        }
    }
    let n = rows.len() as f64;
    acc.iter().map(|&s| (s as f64 / n) as f32).collect()
}

/// `m / np.linalg.norm(m)` for float32.
pub fn unit_f32(m: &[f32]) -> Vec<f32> {
    let k = norm_f32(m);
    m.iter().map(|v| v / k).collect()
}

/// `np.mean(rows, 0)` of float64 rows (k x D, rows added in order).
pub fn mean_rows(rows: &[Vec<f64>]) -> Vec<f64> {
    let mut acc = rows[0].clone();
    for r in &rows[1..] {
        for (a, b) in acc.iter_mut().zip(r.iter()) {
            *a += *b;
        }
    }
    let n = rows.len() as f64;
    acc.iter().map(|s| s / n).collect()
}

/// `np.percentile(v, q)` (method "linear"): numpy's virtual index and `_lerp`.
pub fn percentile(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    let qq = q / 100.0;
    let vi = (n as f64 - 1.0) * qq;
    let prev = vi.floor();
    let gamma = vi - prev;
    let i = (prev as usize).min(n - 1);
    let j = (i + 1).min(n - 1);
    let (a, b) = (s[i], s[j]);
    let diff = b - a;
    if gamma >= 0.5 {
        b - diff * (1.0 - gamma)
    } else {
        a + diff * gamma
    }
}

/// `np.hypot` (libm).
pub fn hypot(x: f64, y: f64) -> f64 {
    x.hypot(y)
}

/// `round(x, n)` when x is a numpy float64 (numpy's `around`: `rint(x * 10**n) / 10**n`), which differs from Python's
/// correctly rounded `round` on a float near a tie (0.53254999999999997 -> 0.5326, Python 0.5325). Which one a site
/// gets depends on whether its value is still a numpy scalar there; each caller says why it uses this one.
pub fn np_round(x: f64, n: i32) -> f64 {
    let f = 10f64.powi(n);
    (x * f).round_ties_even() / f
}

/// Float types numpy reduces in their own width.
pub trait Fl: Copy + PartialOrd + std::ops::Add<Output = Self> + std::ops::Sub<Output = Self> + std::ops::Div<Output = Self> {
    const ZERO: Self;
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn is_nan(self) -> bool;
}
impl Fl for f32 {
    const ZERO: f32 = 0.0;
    fn from_f64(v: f64) -> f32 {
        v as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }
}
impl Fl for f64 {
    const ZERO: f64 = 0.0;
    fn from_f64(v: f64) -> f64 {
        v
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
}

/// numpy's `pairwise_sum` (umath loops): blocks of 128 with 8 accumulators, halves above that. This is what
/// `a.sum()` gives for a contiguous array (and along the last axis of a C-ordered one).
pub fn sum<T: Fl>(a: &[T]) -> T {
    let n = a.len();
    if n < 8 {
        let mut r = T::ZERO;
        for &v in a {
            r = r + v;
        }
        r
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] = r[j] + a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res = res + a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        sum(&a[..n2]) + sum(&a[n2..])
    }
}

/// `a.mean()` of a contiguous float array (sum in its own width, divided by n).
pub fn mean<T: Fl>(a: &[T]) -> T {
    sum(a) / T::from_f64(a.len() as f64)
}

/// `np.median(a)` (NaN if any NaN; the mean of the two middle values in the array's own width).
pub fn median<T: Fl>(a: &[T]) -> T {
    if a.is_empty() || a.iter().any(|v| v.is_nan()) {
        return T::from_f64(f64::NAN);
    }
    // by selection, not a full sort: the same values in linear time (the grain of a 4K frame takes the median of
    // about 1.5 million numbers, for every frame played)
    let mut v = a.to_vec();
    let n = v.len();
    let (lo, &mut hi, _) = v.select_nth_unstable_by(n / 2, |x, y| x.partial_cmp(y).unwrap());
    if n % 2 == 1 {
        hi
    } else {
        // np.mean of the two: (a + b) / 2 in the array's width; the lower one is the largest left of the middle
        let below = lo.iter().copied().fold(lo[0], |m, x| if x > m { x } else { m });
        (below + hi) / T::from_f64(2.0)
    }
}

/// `np.nanmedian` of f64 values.
pub fn nanmedian(a: &[f64]) -> f64 {
    let v: Vec<f64> = a.iter().copied().filter(|x| !x.is_nan()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    median(&v)
}

/// `np.percentile(a, q)` with the default method "linear", for an f32 array: f64 results, numpy's virtual index
/// `(n - 1) * q` and its `_lerp` (the difference taken in f32, the rest in f64).
pub fn percentile_f32(a: &[f32], qs: &[f64]) -> Vec<f64> {
    let mut v = a.to_vec();
    v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let n = v.len();
    qs.iter()
        .map(|&q| {
            let q = q / 100.0;
            let vi = (n - 1) as f64 * q;
            let prev = vi.floor().clamp(0.0, (n - 1) as f64);
            let next = (prev + 1.0).min((n - 1) as f64);
            let g = vi - vi.floor();
            let (a, b) = (v[prev as usize], v[next as usize]);
            let d = (b - a) as f64;
            if g >= 0.5 {
                b as f64 - d * (1.0 - g)
            } else {
                a as f64 + d * g
            }
        })
        .collect()
}

/// Python's `round(x)` for a float: nearest integer, halves to even.
pub fn pyround(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// Python's `round(x, n)`: the float nearest to x rounded to n decimals (correctly rounded, halves to even).
pub fn pyround_n(x: f64, n: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{:.*}", n, x).parse().unwrap()
}

/// `np.clip(x, lo, hi)` for one value.
pub fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// Python's `max(a, b)`: a unless b is greater.
pub fn pymax(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_matches_numpy_order() {
        // 9 values: 8 accumulators then the tail (numpy: np.float32 sum of these is 1.0000001)
        let a = [1e8f32, 1.0, -1e8, 1.0, 0.5, 0.25, 0.125, 0.0625, 1.0];
        let r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let want = (((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))) + a[8];
        assert_eq!(sum(&a), want);
        assert_eq!(sum(&[1.0f64, 2.0, 3.0]), 6.0);
    }

    #[test]
    fn median_and_percentile() {
        assert_eq!(median(&[3.0f32, 1.0, 2.0, 4.0]), 2.5);
        assert!(median(&[1.0f64, f64::NAN]).is_nan());
        // np.percentile(np.float32([1.5, 2.25, 3.1, 9.7, 0.3]), [10, 50, 90]) = [0.78000003..., 2.25, 7.05999985...]
        let p = percentile_f32(&[1.5, 2.25, 3.1, 9.7, 0.3], &[10.0, 50.0, 90.0]);
        assert_eq!(p[1], 2.25);
        assert!((p[0] - 0.78000003).abs() < 1e-7 && (p[2] - 7.05999985).abs() < 1e-7);
    }

    #[test]
    fn python_rounding() {
        assert_eq!(pyround(2.5), 2);
        assert_eq!(pyround(3.5), 4);
        assert_eq!(pyround_n(0.0005, 3), 0.001); // 0.0005 is a little above 5e-4 in binary
        assert_eq!(pyround_n(-0.0514999, 3), -0.051);
    }
}
