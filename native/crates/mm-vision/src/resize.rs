//! `cv2.resize` (INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA) for u8 and f32 images of 1-4 channels,
//! bit-exact with the project's cv2 5.0.0 (arm64, NEON, Carotene HAL).
//!
//! Which code runs, in OpenCV's order (resize.cpp `cv::resize` -> `hal::resize`):
//! - **Size:** the `dsize` form uses `inv_scale = dsize / ssize`; the `fx, fy` form makes
//!   `dsize = cvRound(ssize * f)` (round half to even, so 643 * 0.5 -> 322) and keeps `inv_scale = f`.
//!   Same size -> a copy.
//! - **Carotene first (u8 only).** AREA at exactly 2x down, 4x down or 2x up (equal in x and y, 1/3/4
//!   channels, size within 0.1 of exact): 2x = `(sum of 4 + 2) >> 2`, 4x = round half to even of `sum / 16`,
//!   2x up = pixel copy. LINEAR on 1 or 4 channels is also Carotene's (`resizeLinearOpenCV`): see
//!   [`carotene_linear_applies`]; that path is ported in `resize_carotene.rs`.
//! - **NEAREST:** `sx = min(floor(dx / fx), w - 1)` in double.
//! - **LINEAR at exactly 2x down** becomes AREA (OpenCV: "INTER_AREA (fast) also is equal to INTER_LINEAR").
//! - **AREA with both scales >= 1:** whole-number scales use `resizeAreaFast` (u8: 2x2 is `(s + 2) >> 2`,
//!   otherwise `rint(sum * (1/area as f32))`; f32: sums in OpenCV's exact association), partial edge cells
//!   `sum / count`. Other scales use `resizeArea` with its weight tables (`computeResizeAreaTab`).
//! - **Everything else** (LINEAR, CUBIC, AREA upscaling as LINEAR with area-style weights) is the separable
//!   `resizeGeneric`: u8 in fixed point (11-bit weights; LINEAR's vertical pass is the NEON
//!   `(mulhi(r0 >> 4, b0) + mulhi(r1 >> 4, b1) + 2) >> 2`; CUBIC's vertical pass is float with fused
//!   multiply-adds for blocks of 8, then the integer formula for the last `width % 8` values), f32 in float with
//!   the same fused/unfused operations as OpenCV's NEON blocks and clang-contracted scalar tails.
//!
//! - **f64** (depth.py's maps): NEAREST and AREA shrinking only (plus LINEAR at exactly 2x down, which is AREA);
//!   the rest panics. `resizeAreaFast_<double, double, ResizeAreaFastNoVec>`: sums in double in OpenCV's
//!   association (`sum += ((a + b) + c) + d`), times the float `1/area`; partial edge cells `(float)sum / count`
//!   (in float, as OpenCV writes it). `resizeArea_<double, double>`: row sums fused (`buf += S * alpha`,
//!   clang-contracted), the vertical sum unfused two lanes at a time for dx + 2 < width (`v_float64`), fused in the
//!   tail.
//!
//! Evidence: native/tools/parity/vision_check.py (jazz frames + random images, every combination above; f64 cases
//! named `*f64*`).

use crate::mat::{par_rows, Mat};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Interp {
    Nearest,
    Linear,
    Cubic,
    Area,
}

const COEF_BITS: i32 = 11;
const COEF_SCALE: f32 = (1 << COEF_BITS) as f32;

/// OpenCV `saturate_cast<int>(double)` (cvRound: half to even).
#[inline]
fn cv_round(v: f64) -> i64 {
    v.round_ties_even() as i64
}

/// Output size and inverse scales exactly as `cv::resize` derives them.
pub fn dst_size(sw: usize, sh: usize, dw: usize, dh: usize, fx: f64, fy: f64) -> (usize, usize, f64, f64) {
    if dw == 0 || dh == 0 {
        assert!(fx > 0.0 && fy > 0.0, "resize: need dsize or fx, fy");
        let w = cv_round(sw as f64 * fx) as usize;
        let h = cv_round(sh as f64 * fy) as usize;
        assert!(w > 0 && h > 0, "resize: empty output");
        (w, h, fx, fy)
    } else {
        (dw, dh, dw as f64 / sw as f64, dh as f64 / sh as f64)
    }
}

/// `cv2.resize` for u8. `dw, dh` = 0 selects the `fx, fy` form.
pub fn resize_u8(src: &Mat<u8>, dw: usize, dh: usize, fx: f64, fy: f64, interp: Interp) -> Mat<u8> {
    let (w, h, isx, isy) = dst_size(src.cols, src.rows, dw, dh, fx, fy);
    if w == src.cols && h == src.rows {
        return src.clone();
    }
    if let Some(out) = carotene_area(src, w, h, isx, isy, interp) {
        return out;
    }
    if interp == Interp::Linear && carotene_linear_applies(src.cols, src.rows, w, h, isx, isy, src.cn) {
        return crate::resize_carotene::resize_linear(src, w, h, 1.0 / isx, 1.0 / isy);
    }
    resize_common(src, w, h, isx, isy, interp)
}

/// `cv2.resize` for f64, NEAREST and AREA shrinking only (LINEAR at exactly 2x down is AREA, as in OpenCV); other
/// cases panic. `dw, dh` = 0 selects the `fx, fy` form.
pub fn resize_f64(src: &Mat<f64>, dw: usize, dh: usize, fx: f64, fy: f64, interp: Interp) -> Mat<f64> {
    let (w, h, isx, isy) = dst_size(src.cols, src.rows, dw, dh, fx, fy);
    if w == src.cols && h == src.rows {
        return src.clone();
    }
    resize_common(src, w, h, isx, isy, interp)
}

/// `cv2.resize` for f32. `dw, dh` = 0 selects the `fx, fy` form.
pub fn resize_f32(src: &Mat<f32>, dw: usize, dh: usize, fx: f64, fy: f64, interp: Interp) -> Mat<f32> {
    let (w, h, isx, isy) = dst_size(src.cols, src.rows, dw, dh, fx, fy);
    if w == src.cols && h == src.rows {
        return src.clone();
    }
    resize_common(src, w, h, isx, isy, interp)
}

/// Carotene's `TEGRA_RESIZE` condition for INTER_LINEAR (u8, 1 or 4 channels).
pub fn carotene_linear_applies(sw: usize, sh: usize, dw: usize, dh: usize, isx: f64, isy: f64, cn: usize) -> bool {
    let supported = match cn {
        1 => sw >= 8 && dw >= 8 && dh >= 8,
        4 => sw >= 2 && dw >= 2 && dh >= 8 && (2 * dw != sw || 2 * dh != sh),
        _ => false,
    };
    let (sw, sh, dwf, dhf) = (sw as f64, sh as f64, dw as f64, dh as f64);
    supported
        && isx > 0.0
        && isy > 0.0
        && (dwf - 0.5) / isx - 0.5 < sw
        && (dhf - 0.5) / isy - 0.5 < sh
        && (dwf + 0.5) / isx + 0.5 >= sw
        && (dhf + 0.5) / isy + 0.5 >= sh
        && (dwf / isx - sw).abs() < 0.1
        && (dhf / isy - sh).abs() < 0.1
}

/// Carotene's AREA path (`resizeAreaOpenCV`): exactly 2x/4x down or 2x up, equal in x and y.
fn carotene_area(src: &Mat<u8>, w: usize, h: usize, isx: f64, isy: f64, interp: Interp) -> Option<Mat<u8>> {
    if interp != Interp::Area || !(src.cn == 1 || src.cn == 3 || src.cn == 4) {
        return None;
    }
    let (wr, hr) = ((1.0 / isx) as f32, (1.0 / isy) as f32);
    if wr != hr || !(wr == 2.0 || wr == 4.0 || wr == 0.5) {
        return None;
    }
    if (w as f64 / isx - src.cols as f64).abs() >= 0.1 || (h as f64 / isy - src.rows as f64).abs() >= 0.1 {
        return None;
    }
    let cn = src.cn;
    let mut out = Mat::new(h, w, cn);
    if wr == 0.5 {
        for y in 0..h {
            let s = src.row(y / 2);
            let d = out.row_mut(y);
            for x in 0..w {
                d[x * cn..(x + 1) * cn].copy_from_slice(&s[(x / 2) * cn..(x / 2 + 1) * cn]);
            }
        }
        return Some(out);
    }
    let k = wr as usize;
    for y in 0..h {
        for x in 0..w {
            for c in 0..cn {
                let mut s = 0u32;
                for yy in 0..k {
                    let r = src.row(y * k + yy);
                    for xx in 0..k {
                        s += r[(x * k + xx) * cn + c] as u32;
                    }
                }
                // OpenCV's unsigned arithmetic: at s == 0 this wraps (0 - 1 + 8 = 7) and still gives 0
                let v = if k == 2 { (s + 2) >> 2 } else { s.wrapping_sub(((16 & !s) >> 4) & 1).wrapping_add(8) >> 4 };
                out.data[(y * w + x) * cn + c] = v as u8;
            }
        }
    }
    Some(out)
}

/// The pixel types resize handles, with OpenCV's per-depth arithmetic.
pub trait ResizePixel: crate::mat::Pixel + Send + Sync {
    fn resize_generic(src: &Mat<Self>, w: usize, h: usize, tabs: &Tabs) -> Mat<Self>;
    fn area_fast(src: &Mat<Self>, w: usize, h: usize, sx: usize, sy: usize) -> Mat<Self>;
    fn area_general(src: &Mat<Self>, w: usize, h: usize, xtab: &[(usize, usize, f32)], ytab: &[(usize, usize, f32)]) -> Mat<Self>;
}

fn resize_common<T: ResizePixel>(src: &Mat<T>, w: usize, h: usize, isx: f64, isy: f64, mut interp: Interp) -> Mat<T> {
    let (sw, sh, cn) = (src.cols, src.rows, src.cn);
    if interp == Interp::Nearest {
        return resize_nn(src, w, h, isx, isy);
    }
    let (scale_x, scale_y) = (1.0 / isx, 1.0 / isy);
    let isc_x = scale_x.round_ties_even();
    let isc_y = scale_y.round_ties_even();
    let is_area_fast = (scale_x - isc_x).abs() < f64::EPSILON && (scale_y - isc_y).abs() < f64::EPSILON;
    if interp == Interp::Linear && is_area_fast && isc_x == 2.0 && isc_y == 2.0 {
        interp = Interp::Area;
    }
    if interp == Interp::Area && scale_x >= 1.0 && scale_y >= 1.0 {
        if is_area_fast {
            return T::area_fast(src, w, h, isc_x as usize, isc_y as usize);
        }
        let xtab = area_tab(sw, w, cn, scale_x);
        let ytab = area_tab(sh, h, 1, scale_y);
        return T::area_general(src, w, h, &xtab, &ytab);
    }
    let tabs = Tabs::new(sw, sh, w, h, cn, scale_x, scale_y, isx, isy, interp);
    T::resize_generic(src, w, h, &tabs)
}

fn resize_nn<T: Copy + Default + Send + Sync>(src: &Mat<T>, w: usize, h: usize, fx: f64, fy: f64) -> Mat<T> {
    let (ifx, ify) = (1.0 / fx, 1.0 / fy);
    let cn = src.cn;
    let xofs: Vec<usize> = (0..w).map(|x| ((x as f64 * ifx).floor() as usize).min(src.cols - 1)).collect();
    let mut out = Mat::new(h, w, cn);
    par_rows(&mut out.data, w * cn, |y, d| {
        let sy = ((y as f64 * ify).floor() as usize).min(src.rows - 1);
        let s = src.row(sy);
        for x in 0..w {
            d[x * cn..(x + 1) * cn].copy_from_slice(&s[xofs[x] * cn..(xofs[x] + 1) * cn]);
        }
    });
    out
}

/// `computeResizeAreaTab` for one axis: (dst index * cn, src index * cn, weight), grouped by dst index. `scale` is
/// the source pixels per output pixel (`ssize / dsize` as `cv::resize` derives it: `1 / (dsize / ssize)`).
pub fn area_tab(ssize: usize, dsize: usize, cn: usize, scale: f64) -> Vec<(usize, usize, f32)> {
    let mut tab = Vec::with_capacity(ssize * 2);
    for dx in 0..dsize {
        let fsx1 = dx as f64 * scale;
        let fsx2 = fsx1 + scale;
        let cell = scale.min(ssize as f64 - fsx1);
        let mut sx2 = fsx2.floor() as i64;
        let mut sx1 = fsx1.ceil() as i64;
        sx2 = sx2.min(ssize as i64 - 1);
        sx1 = sx1.min(sx2);
        if sx1 as f64 - fsx1 > 1e-3 {
            tab.push((dx * cn, (sx1 as usize - 1) * cn, ((sx1 as f64 - fsx1) / cell) as f32));
        }
        for sx in sx1..sx2 {
            tab.push((dx * cn, sx as usize * cn, (1.0 / cell) as f32));
        }
        if fsx2 - sx2 as f64 > 1e-3 {
            tab.push((dx * cn, sx2 as usize * cn, ((fsx2 - sx2 as f64).min(1.0).min(cell) / cell) as f32));
        }
    }
    tab
}

/// The separable filter's tables (`hal::resize` after the special cases): per output column (in elements,
/// channels expanded) the source offset and weights, per output row the source row and weights.
pub struct Tabs {
    pub ksize: usize,
    pub xofs: Vec<i64>,
    pub yofs: Vec<i64>,
    pub alpha: Vec<f32>,  // ksize per output element
    pub beta: Vec<f32>,   // ksize per output row
    pub ialpha: Vec<i32>, // fixed-point versions (u8)
    pub ibeta: Vec<i32>,
    pub xmin: usize, // in elements
    pub xmax: usize,
}

/// OpenCV's `interpolateCubic` (A = -0.75) as clang compiles it for arm64: the polynomials are contracted into
/// fused multiply-adds (checked against the assembly of the same source with the same compiler).
fn interpolate_cubic(x: f32) -> [f32; 4] {
    let x1 = x + 1.0;
    let c0 = x1.mul_add(-0.75, 3.75).mul_add(x1, -6.0).mul_add(x1, 3.0);
    let c1 = {
        let t = 1.25f32.mul_add(x, -2.25);
        x.mul_add(x * t, 1.0)
    };
    let y = 1.0 - x;
    let c2 = {
        let t = 1.25f32.mul_add(y, -2.25);
        y.mul_add(y * t, 1.0)
    };
    [c0, c1, c2, 1.0 - c0 - c1 - c2]
}

/// `saturate_cast<short>(float)`.
#[inline]
fn sat_short(v: f32) -> i32 {
    (v.round_ties_even() as i32).clamp(-32768, 32767)
}

impl Tabs {
    #[allow(clippy::too_many_arguments)]
    fn new(sw: usize, sh: usize, w: usize, h: usize, cn: usize, scale_x: f64, scale_y: f64, isx: f64, isy: f64, interp: Interp) -> Tabs {
        let area_mode = interp == Interp::Area;
        let ksize = if interp == Interp::Cubic { 4 } else { 2 };
        let ksize2 = (ksize / 2) as i64;
        let cubic = interp == Interp::Cubic;
        let (mut xmin, mut xmax) = (0usize, w);
        let width = w * cn;
        let mut xofs = vec![0i64; width];
        let mut alpha = vec![0f32; width * ksize];
        let mut ialpha = vec![0i32; width * ksize];
        let coefs = |f: f32| -> [f32; 4] {
            if cubic {
                interpolate_cubic(f)
            } else {
                [1.0 - f, f, 0.0, 0.0]
            }
        };
        for dx in 0..w {
            let (mut sx, mut fx);
            if !area_mode {
                fx = ((dx as f64 + 0.5) * scale_x - 0.5) as f32;
                sx = fx.floor() as i64;
                fx -= sx as f32;
            } else {
                sx = (dx as f64 * scale_x).floor() as i64;
                fx = ((dx + 1) as f64 - (sx + 1) as f64 * isx) as f32;
                fx = if fx <= 0.0 { 0.0 } else { fx - fx.floor() };
            }
            if sx < ksize2 - 1 {
                xmin = dx + 1;
                if sx < 0 && !cubic {
                    fx = 0.0;
                    sx = 0;
                }
            }
            if sx + ksize2 >= sw as i64 {
                xmax = xmax.min(dx);
                if sx >= sw as i64 - 1 && !cubic {
                    fx = 0.0;
                    sx = sw as i64 - 1;
                }
            }
            let cb = coefs(fx);
            for k in 0..cn {
                xofs[dx * cn + k] = sx * cn as i64 + k as i64;
                for j in 0..ksize {
                    alpha[(dx * cn + k) * ksize + j] = cb[j];
                    ialpha[(dx * cn + k) * ksize + j] = sat_short(cb[j] * COEF_SCALE);
                }
            }
        }
        let mut yofs = vec![0i64; h];
        let mut beta = vec![0f32; h * ksize];
        let mut ibeta = vec![0i32; h * ksize];
        for dy in 0..h {
            let (sy, mut fy);
            if !area_mode {
                fy = ((dy as f64 + 0.5) * scale_y - 0.5) as f32;
                sy = fy.floor() as i64;
                fy -= sy as f32;
            } else {
                sy = (dy as f64 * scale_y).floor() as i64;
                fy = ((dy + 1) as f64 - (sy + 1) as f64 * isy) as f32;
                fy = if fy <= 0.0 { 0.0 } else { fy - fy.floor() };
            }
            yofs[dy] = sy;
            let cb = coefs(fy);
            for j in 0..ksize {
                beta[dy * ksize + j] = cb[j];
                ibeta[dy * ksize + j] = sat_short(cb[j] * COEF_SCALE);
            }
        }
        let _ = sh;
        Tabs { ksize, xofs, yofs, alpha, beta, ialpha, ibeta, xmin: xmin * cn, xmax: xmax * cn }
    }

    /// Source rows for output row dy (clipped like OpenCV's `clip(sy0 - ksize/2 + 1 + k, 0, h)`).
    fn rows(&self, dy: usize, sh: usize) -> [usize; 4] {
        let mut r = [0usize; 4];
        let k2 = (self.ksize / 2) as i64;
        for k in 0..self.ksize {
            let sy = self.yofs[dy] - k2 + 1 + k as i64;
            r[k] = sy.clamp(0, sh as i64 - 1) as usize;
        }
        r
    }
}

/// The horizontal pass for every source row the output needs (in parallel): (row -> slot, buffer of slots).
fn h_rows<B: Copy + Default + Send + Sync>(srows: usize, h: usize, k: usize, width: usize, t: &Tabs, hrow: impl Fn(usize) -> Vec<B> + Sync) -> (Vec<usize>, Vec<B>) {
    let mut slot = vec![usize::MAX; srows];
    let mut needed = Vec::new();
    for dy in 0..h {
        for &r in t.rows(dy, srows).iter().take(k) {
            if slot[r] == usize::MAX {
                slot[r] = needed.len();
                needed.push(r);
            }
        }
    }
    let mut buf = vec![B::default(); needed.len() * width];
    par_rows(&mut buf, width, |i, d| d.copy_from_slice(&hrow(needed[i])));
    (slot, buf)
}

/// Horizontal pass for one source row, generic over the arithmetic (closure per element).
fn hresize_cubic_idx(sx: i64, j: usize, cn: usize, swidth: usize) -> usize {
    let mut sxj = sx - cn as i64 + (j * cn) as i64;
    if (sxj as u64) >= swidth as u64 {
        while sxj < 0 {
            sxj += cn as i64;
        }
        while sxj >= swidth as i64 {
            sxj -= cn as i64;
        }
    }
    sxj as usize
}

impl ResizePixel for u8 {
    fn resize_generic(src: &Mat<u8>, w: usize, h: usize, t: &Tabs) -> Mat<u8> {
        let cn = src.cn;
        let width = w * cn;
        let swidth = src.cols * cn;
        let k = t.ksize;
        let mut out = Mat::new(h, w, cn);
        // horizontal pass in ints (exact: integer dot products)
        let hrow = |r: usize| -> Vec<i32> {
            let s = src.row(r);
            let mut d = vec![0i32; width];
            if k == 2 {
                for dx in 0..width {
                    let sx = t.xofs[dx] as usize;
                    if dx < t.xmax {
                        let a = &t.ialpha[dx * 2..dx * 2 + 2];
                        d[dx] = s[sx] as i32 * a[0] + s[(sx + cn).min(swidth - 1)] as i32 * a[1];
                    } else {
                        d[dx] = s[sx] as i32 * (1 << COEF_BITS);
                    }
                }
            } else {
                for dx in 0..width {
                    let sx = t.xofs[dx];
                    let a = &t.ialpha[dx * 4..dx * 4 + 4];
                    let mut v = 0i32;
                    if dx < t.xmin || dx >= t.xmax {
                        for j in 0..4 {
                            v += s[hresize_cubic_idx(sx, j, cn, swidth)] as i32 * a[j];
                        }
                    } else {
                        let sx = sx as usize;
                        v = s[sx - cn] as i32 * a[0] + s[sx] as i32 * a[1] + s[sx + cn] as i32 * a[2] + s[sx + 2 * cn] as i32 * a[3];
                    }
                    d[dx] = v;
                }
            }
            d
        };
        let (slot, hbuf) = h_rows(src.rows, h, k, width, t, hrow);
        par_rows(&mut out.data, width, |dy, d| {
            let rows = t.rows(dy, src.rows);
            let bufs: Vec<&[i32]> = rows.iter().take(k).map(|&r| &hbuf[slot[r] * width..(slot[r] + 1) * width]).collect();
            let b = &t.ibeta[dy * k..dy * k + k];
            if k == 2 {
                let (s0, s1) = (bufs[0], bufs[1]);
                for x in 0..width {
                    let v = (((s0[x] >> 4) * b[0]) >> 16) + (((s1[x] >> 4) * b[1]) >> 16);
                    d[x] = ((v + 2) >> 2).clamp(0, 255) as u8;
                }
            } else {
                // NEON blocks of 8 in float (fused), then OpenCV's integer FixedPtCast for the tail
                let sc = 1.0f32 / (COEF_SCALE * COEF_SCALE);
                let bf = [b[0] as f32 * sc, b[1] as f32 * sc, b[2] as f32 * sc, b[3] as f32 * sc];
                let vend = width / 8 * 8;
                for x in 0..vend {
                    let mut acc = bufs[3][x] as f32 * bf[3];
                    acc = (bufs[2][x] as f32).mul_add(bf[2], acc);
                    acc = (bufs[1][x] as f32).mul_add(bf[1], acc);
                    acc = (bufs[0][x] as f32).mul_add(bf[0], acc);
                    d[x] = (acc.round_ties_even() as i32).clamp(0, 255) as u8;
                }
                for x in vend..width {
                    let v = bufs[0][x] as i64 * b[0] as i64 + bufs[1][x] as i64 * b[1] as i64 + bufs[2][x] as i64 * b[2] as i64 + bufs[3][x] as i64 * b[3] as i64;
                    d[x] = ((v + (1 << 21)) >> 22).clamp(0, 255) as u8;
                }
            }
        });
        out
    }

    fn area_fast(src: &Mat<u8>, w: usize, h: usize, scx: usize, scy: usize) -> Mat<u8> {
        let cn = src.cn;
        let (sw, sh) = (src.cols * cn, src.rows);
        let area = scx * scy;
        let scale = 1.0f32 / area as f32;
        let dwidth1 = (src.cols / scx) * cn;
        let width = w * cn;
        let fast2 = scx == 2 && scy == 2 && (cn == 1 || cn == 3 || cn == 4);
        let mut out = Mat::new(h, w, cn);
        par_rows(&mut out.data, width, |dy, d| {
            let sy0 = dy * scy;
            if sy0 >= sh {
                d.iter_mut().for_each(|v| *v = 0);
                return;
            }
            let wfull = if sy0 + scy <= sh { dwidth1 } else { 0 };
            let mut dx = 0;
            if fast2 && wfull > 0 {
                let (s0, s1) = (src.row(sy0), src.row(sy0 + 1));
                while dx < wfull {
                    let c = dx % cn;
                    let i = (dx / cn) * 2 * cn + c;
                    d[dx] = ((s0[i] as u32 + s0[i + cn] as u32 + s1[i] as u32 + s1[i + cn] as u32 + 2) >> 2) as u8;
                    dx += 1;
                }
            }
            while dx < wfull {
                let c = dx % cn;
                let sx0 = (dx / cn) * scx * cn + c;
                let mut sum = 0i32;
                for yy in 0..scy {
                    let r = src.row(sy0 + yy);
                    for xx in 0..scx {
                        sum += r[sx0 + xx * cn] as i32;
                    }
                }
                d[dx] = u8_from_f32(sum as f32 * scale);
                dx += 1;
            }
            while dx < width {
                let c = dx % cn;
                let sx0 = (dx / cn) * scx * cn + c;
                let (mut sum, mut count) = (0i32, 0i32);
                for yy in 0..scy {
                    if sy0 + yy >= sh {
                        break;
                    }
                    let r = src.row(sy0 + yy);
                    let mut sx = 0;
                    while sx < scx * cn {
                        if sx0 + sx >= sw {
                            break;
                        }
                        sum += r[sx0 + sx] as i32;
                        count += 1;
                        sx += cn;
                    }
                }
                d[dx] = u8_from_f32(sum as f32 / count as f32);
                dx += 1;
            }
        });
        out
    }

    fn area_general(src: &Mat<u8>, w: usize, h: usize, xtab: &[(usize, usize, f32)], ytab: &[(usize, usize, f32)]) -> Mat<u8> {
        area_general_impl(src, w, h, xtab, ytab, |v| u8_from_f32(v))
    }
}

#[inline]
fn u8_from_f32(v: f32) -> u8 {
    if v.is_nan() {
        0
    } else {
        (v.round_ties_even() as i32).clamp(0, 255) as u8
    }
}

fn area_general_impl<T: crate::mat::Pixel, O: Copy + Default + Send>(
    src: &Mat<T>,
    w: usize,
    h: usize,
    xtab: &[(usize, usize, f32)],
    ytab: &[(usize, usize, f32)],
    store: impl Fn(f32) -> O + Sync,
) -> Mat<O> {
    let cn = src.cn;
    let width = w * cn;
    let mut out = Mat::new(h, w, cn);
    // ytab entries are grouped by output row; each row is independent (its first entry starts from 0)
    let mut ranges = vec![(0usize, 0usize); h];
    let mut j = 0;
    while j < ytab.len() {
        let dy = ytab[j].0;
        let st = j;
        while j < ytab.len() && ytab[j].0 == dy {
            j += 1;
        }
        ranges[dy] = (st, j);
    }
    // NEON for dx + 4 < width (unfused mul then add), scalar (fused) for the rest
    let vend = if width > 4 { (width - 1) / 4 * 4 } else { 0 };
    par_rows(&mut out.data, width, |dy, d| {
        let mut buf = vec![0f32; width];
        let mut sum = vec![0f32; width];
        let (st, en) = ranges[dy];
        for (n, &(_, sy, beta)) in ytab[st..en].iter().enumerate() {
            let s = src.row(sy);
            buf.iter_mut().for_each(|v| *v = 0.0);
            for &(dxn, sxn, a) in xtab {
                for c in 0..cn {
                    buf[dxn + c] = s[sxn + c].to_f32().mul_add(a, buf[dxn + c]);
                }
            }
            if n == 0 {
                for x in 0..width {
                    sum[x] = beta * buf[x];
                }
            } else {
                for x in 0..vend {
                    sum[x] += beta * buf[x];
                }
                for x in vend..width {
                    sum[x] = beta.mul_add(buf[x], sum[x]);
                }
            }
        }
        for (o, &v) in d.iter_mut().zip(&sum) {
            *o = store(v);
        }
    });
    out
}

impl ResizePixel for f32 {
    fn resize_generic(src: &Mat<f32>, w: usize, h: usize, t: &Tabs) -> Mat<f32> {
        let cn = src.cn;
        let width = w * cn;
        let swidth = src.cols * cn;
        let k = t.ksize;
        let mut out = Mat::new(h, w, cn);
        let len0 = t.xmax & !3;
        let hrow = |r: usize| -> Vec<f32> {
            let s = src.row(r);
            let mut d = vec![0f32; width];
            if k == 2 {
                for dx in 0..width {
                    let sx = t.xofs[dx] as usize;
                    let a = &t.alpha[dx * 2..dx * 2 + 2];
                    if dx < len0 {
                        d[dx] = s[sx] * a[0] + s[sx + cn] * a[1];
                    } else if dx < t.xmax {
                        d[dx] = s[sx].mul_add(a[0], s[sx + cn] * a[1]);
                    } else {
                        d[dx] = s[sx];
                    }
                }
            } else {
                for dx in 0..width {
                    let sx = t.xofs[dx];
                    let a = &t.alpha[dx * 4..dx * 4 + 4];
                    if dx < t.xmin || dx >= t.xmax {
                        let mut v = 0f32;
                        for j in 0..4 {
                            v = s[hresize_cubic_idx(sx, j, cn, swidth)].mul_add(a[j], v);
                        }
                        d[dx] = v;
                    } else {
                        let sx = sx as usize;
                        // clang: fma(S3, a3, fma(S2, a2, fma(S0, a0, S1 * a1)))
                        let v = s[sx - cn].mul_add(a[0], s[sx] * a[1]);
                        let v = s[sx + cn].mul_add(a[2], v);
                        d[dx] = s[sx + 2 * cn].mul_add(a[3], v);
                    }
                }
            }
            d
        };
        let (slot, hbuf) = h_rows(src.rows, h, k, width, t, hrow);
        par_rows(&mut out.data, width, |dy, d| {
            let rows = t.rows(dy, src.rows);
            let bufs: Vec<&[f32]> = rows.iter().take(k).map(|&r| &hbuf[slot[r] * width..(slot[r] + 1) * width]).collect();
            let b = &t.beta[dy * k..dy * k + k];
            if k == 2 {
                for x in 0..width {
                    d[x] = bufs[0][x].mul_add(b[0], bufs[1][x] * b[1]);
                }
            } else {
                let vend = width / 4 * 4;
                for x in 0..vend {
                    let mut acc = bufs[3][x] * b[3];
                    acc = bufs[2][x].mul_add(b[2], acc);
                    acc = bufs[1][x].mul_add(b[1], acc);
                    d[x] = bufs[0][x].mul_add(b[0], acc);
                }
                for x in vend..width {
                    let v = bufs[0][x].mul_add(b[0], bufs[1][x] * b[1]);
                    let v = bufs[2][x].mul_add(b[2], v);
                    d[x] = bufs[3][x].mul_add(b[3], v);
                }
            }
        });
        out
    }

    fn area_fast(src: &Mat<f32>, w: usize, h: usize, scx: usize, scy: usize) -> Mat<f32> {
        let cn = src.cn;
        let (sw, sh) = (src.cols * cn, src.rows);
        let area = scx * scy;
        let scale = 1.0f32 / area as f32;
        let dwidth1 = (src.cols / scx) * cn;
        let width = w * cn;
        let fast2 = scx == 2 && scy == 2 && (cn == 1 || cn == 4);
        // element offsets of the area cells, row-major (OpenCV's ofs[])
        let ofs: Vec<(usize, usize)> = (0..scy).flat_map(|yy| (0..scx).map(move |xx| (yy, xx * cn))).collect();
        let mut out = Mat::new(h, w, cn);
        par_rows(&mut out.data, width, |dy, d| {
            let sy0 = dy * scy;
            if sy0 >= sh {
                d.iter_mut().for_each(|v| *v = 0.0);
                return;
            }
            let wfull = if sy0 + scy <= sh { dwidth1 } else { 0 };
            let mut dx = 0;
            if fast2 {
                let (s0, s1) = (src.row(sy0), if sy0 + 1 < sh { src.row(sy0 + 1) } else { src.row(sy0) });
                // NEON: 4 outputs at a time while dx <= w - 4, (r0a + r0b) + (r1a + r1b), times 0.25
                while wfull >= 4 && dx <= wfull - 4 {
                    for i in dx..dx + 4 {
                        let c = i % cn;
                        let p = (i / cn) * 2 * cn + c;
                        d[i] = ((s0[p] + s0[p + cn]) + (s1[p] + s1[p + cn])) * 0.25;
                    }
                    dx += 4;
                }
            }
            while dx < wfull {
                let c = dx % cn;
                let sx0 = (dx / cn) * scx * cn + c;
                let mut sum = 0f32;
                let mut kk = 0;
                while kk + 4 <= area {
                    let v: [f32; 4] = std::array::from_fn(|q| {
                        let (yy, xo) = ofs[kk + q];
                        src.row(sy0 + yy)[sx0 + xo]
                    });
                    sum += ((v[0] + v[1]) + v[2]) + v[3];
                    kk += 4;
                }
                while kk < area {
                    let (yy, xo) = ofs[kk];
                    sum += src.row(sy0 + yy)[sx0 + xo];
                    kk += 1;
                }
                d[dx] = sum * scale;
                dx += 1;
            }
            while dx < width {
                let c = dx % cn;
                let sx0 = (dx / cn) * scx * cn + c;
                let (mut sum, mut count) = (0f32, 0i32);
                for yy in 0..scy {
                    if sy0 + yy >= sh {
                        break;
                    }
                    let r = src.row(sy0 + yy);
                    let mut sx = 0;
                    while sx < scx * cn {
                        if sx0 + sx >= sw {
                            break;
                        }
                        sum += r[sx0 + sx];
                        count += 1;
                        sx += cn;
                    }
                }
                d[dx] = sum / count as f32;
                dx += 1;
            }
        });
        out
    }

    fn area_general(src: &Mat<f32>, w: usize, h: usize, xtab: &[(usize, usize, f32)], ytab: &[(usize, usize, f32)]) -> Mat<f32> {
        area_general_impl(src, w, h, xtab, ytab, |v| v)
    }
}

impl ResizePixel for f64 {
    fn resize_generic(_: &Mat<f64>, _: usize, _: usize, _: &Tabs) -> Mat<f64> {
        panic!("resize_f64: only NEAREST and AREA shrinking are ported for float64");
    }

    fn area_fast(src: &Mat<f64>, w: usize, h: usize, scx: usize, scy: usize) -> Mat<f64> {
        let cn = src.cn;
        let (sw, sh) = (src.cols * cn, src.rows);
        let area = scx * scy;
        let scale = (1.0f32 / area as f32) as f64;
        let dwidth1 = (src.cols / scx) * cn;
        let width = w * cn;
        let ofs: Vec<(usize, usize)> = (0..scy).flat_map(|yy| (0..scx).map(move |xx| (yy, xx * cn))).collect();
        let mut out = Mat::new(h, w, cn);
        par_rows(&mut out.data, width, |dy, d| {
            let sy0 = dy * scy;
            if sy0 >= sh {
                d.iter_mut().for_each(|v| *v = 0.0);
                return;
            }
            let wfull = if sy0 + scy <= sh { dwidth1 } else { 0 };
            let mut dx = 0;
            while dx < wfull {
                let c = dx % cn;
                let sx0 = (dx / cn) * scx * cn + c;
                let at = |k: usize| {
                    let (yy, xo) = ofs[k];
                    src.row(sy0 + yy)[sx0 + xo]
                };
                let mut sum = 0f64;
                let mut kk = 0;
                while kk + 4 <= area {
                    sum += ((at(kk) + at(kk + 1)) + at(kk + 2)) + at(kk + 3);
                    kk += 4;
                }
                while kk < area {
                    sum += at(kk);
                    kk += 1;
                }
                d[dx] = sum * scale;
                dx += 1;
            }
            while dx < width {
                let c = dx % cn;
                let sx0 = (dx / cn) * scx * cn + c;
                let (mut sum, mut count) = (0f64, 0i32);
                for yy in 0..scy {
                    if sy0 + yy >= sh {
                        break;
                    }
                    let r = src.row(sy0 + yy);
                    let mut sx = 0;
                    while sx < scx * cn {
                        if sx0 + sx >= sw {
                            break;
                        }
                        sum += r[sx0 + sx];
                        count += 1;
                        sx += cn;
                    }
                }
                d[dx] = (sum as f32 / count as f32) as f64;
                dx += 1;
            }
        });
        out
    }

    fn area_general(src: &Mat<f64>, w: usize, h: usize, xtab: &[(usize, usize, f32)], ytab: &[(usize, usize, f32)]) -> Mat<f64> {
        let cn = src.cn;
        let width = w * cn;
        let mut out = Mat::new(h, w, cn);
        let mut ranges = vec![(0usize, 0usize); h];
        let mut j = 0;
        while j < ytab.len() {
            let dy = ytab[j].0;
            let st = j;
            while j < ytab.len() && ytab[j].0 == dy {
                j += 1;
            }
            ranges[dy] = (st, j);
        }
        // v_float64 (2 lanes, unfused mul then add) for dx + 2 < width, scalar (fused) for the rest
        let vend = if width > 2 { (width - 1) / 2 * 2 } else { 0 };
        par_rows(&mut out.data, width, |dy, d| {
            let mut buf = vec![0f64; width];
            let (st, en) = ranges[dy];
            for (n, &(_, sy, beta)) in ytab[st..en].iter().enumerate() {
                let beta = beta as f64;
                let s = src.row(sy);
                buf.iter_mut().for_each(|v| *v = 0.0);
                for &(dxn, sxn, a) in xtab {
                    for c in 0..cn {
                        buf[dxn + c] = s[sxn + c].mul_add(a as f64, buf[dxn + c]);
                    }
                }
                if n == 0 {
                    for x in 0..width {
                        d[x] = beta * buf[x];
                    }
                } else {
                    for x in 0..vend {
                        d[x] += beta * buf[x];
                    }
                    for x in vend..width {
                        d[x] = beta.mul_add(buf[x], d[x]);
                    }
                }
            }
        });
        out
    }
}

/// OpenCV's own path (no Carotene) for a `dsize` resize; used by the Carotene placeholder.
pub fn resize_opencv_u8(src: &Mat<u8>, w: usize, h: usize, interp: Interp) -> Mat<u8> {
    let (isx, isy) = (w as f64 / src.cols as f64, h as f64 / src.rows as f64);
    resize_common(src, w, h, isx, isy, interp)
}
