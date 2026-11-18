//! Carotene's `resizeLinearOpenCV` (hal/carotene/src/resize.cpp): what cv2.resize INTER_LINEAR runs for u8 images
//! with 1 or 4 channels on the Mac (see `resize::carotene_linear_applies` for when). Two algorithms, both written
//! here per output pixel (Carotene's 8x8 transposed blocks and its "goto" tails recompute the same values):
//!
//! - `resizeLinearOpenCVchan` (1 channel scaling up or by more than 2x, and every 4-channel case): 7-bit
//!   weights. Rows first: `w = fma(r, hr, 0.5 hr - 0.5)`, top row `floor(w)` with weight `trunc((floor(w) + 1 - w)
//!   * 128)`, value `(top * w + bottom * (128 - w) + 64) >> 7`; then columns the same way from the row results,
//!   with the left index `ceil(x) - 1` (computed as trunc + compare) and the weight `trunc((ceil(x) - x) * 128)`.
//! - `downsample_bilinear_8uc1` (1 channel, scale in (1, 2], source at least 16 wide): 11-bit weights and the
//!   "ugly version matching OpenCV's SSE optimization": per row `v = (h << 11) + (l - h) * cw` for the two source
//!   rows, then `qdmulh(v1 >> 4, rw)`, `qdmulh(v2 >> 4, 2048 - rw)`, halved, summed, `(s + 2) >> 2` saturated.
//!
//! Coordinates use clang's fused multiply-adds (the C expressions and `vmlaq_f32` both compile to fmadd/fmla).

use crate::mat::{par_rows, Mat};

pub fn resize_linear(src: &Mat<u8>, w: usize, h: usize, wr: f64, hr: f64) -> Mat<u8> {
    let (wr, hr) = (wr as f32, hr as f32);
    if src.cn == 1 && !(wr <= 1.0 && hr <= 1.0) && wr <= 2.0 && hr <= 2.0 && src.cols >= 16 {
        downsample_8uc1(src, w, h, wr, hr)
    } else {
        linear_chan(src, w, h, wr, hr)
    }
}

/// `resizeLinearOpenCVchan<1|4>`.
fn linear_chan(src: &Mat<u8>, w: usize, h: usize, wr: f32, hr: f32) -> Mat<u8> {
    let cn = src.cn;
    let (sw, sh) = (src.cols, src.rows);
    let off_x = 0.5f32 * wr - 0.5;
    let off_y = 0.5f32 * hr - 0.5;
    // column table: (left, right, weight of left) per output pixel
    let cols: Vec<(usize, usize, u16)> = (0..w)
        .map(|p| {
            let vw = wr.mul_add(p as f32, off_x);
            let vwi = vw as i32; // fcvtzs: toward zero
            let srch = vwi + ((vwi as f32) < vw) as i32;
            let srcl = srch - 1;
            let wt = ((srch as f32 - vw) * 128.0) as u32 as u16 as u8 as u16; // fcvtzu, then narrowed
            let hi = srch.min(sw as i32 - 1).max(0) as usize; // (min only; srch >= 0 here)
            let lo = srcl.max(0).min(sw as i32 - 1) as usize;
            (lo, hi, wt)
        })
        .collect();
    let mut out = Mat::new(h, w, cn);
    par_rows(&mut out.data, w * cn, |r, d| {
        let mut vrow = vec![0u16; sw * cn];
        let wv = (r as f32).mul_add(hr, off_y);
        let src_row = wv.floor() as i64;
        let src_row2 = src_row + 1;
        let rw = ((src_row2 as f32 - wv) * 128.0) as i32 as u8 as u16;
        let r1 = src_row.max(0) as usize;
        let r2 = src_row2.min(sh as i64 - 1) as usize;
        let (a, b) = (src.row(r1), src.row(r2));
        let rw2 = (128u16.wrapping_sub(rw)) as u8 as u16;
        for e in 0..sw * cn {
            vrow[e] = ((a[e] as u16 * rw + b[e] as u16 * rw2 + 64) >> 7) & 0xff;
        }
        for (p, &(lo, hi, wt)) in cols.iter().enumerate() {
            let wt2 = (128u16.wrapping_sub(wt)) as u8 as u16;
            for c in 0..cn {
                let v = (vrow[lo * cn + c] * wt + vrow[hi * cn + c] * wt2 + 64) >> 7;
                d[p * cn + c] = v as u8;
            }
        }
    });
    out
}

/// `vqdmulhq_s16`: saturate((2 * a * b) >> 16).
#[inline]
fn qdmulh(a: i16, b: i16) -> i16 {
    (((2 * a as i32 * b as i32) >> 16).clamp(-32768, 32767)) as i16
}

/// `downsample_bilinear_8uc1`.
fn downsample_8uc1(src: &Mat<u8>, w: usize, h: usize, wr: f32, hr: f32) -> Mat<u8> {
    const SHIFT: i32 = 11;
    let (sw, sh) = (src.cols as i64, src.rows as i64);
    let off_x = 0.5f32 * wr - 0.5;
    let off_y = 0.5f32 * hr - 0.5;
    let cols: Vec<(usize, usize, i32)> = (0..w)
        .map(|col| {
            let c = (col as f32).mul_add(wr, off_x);
            let col1 = c as i64; // truncation
            let col2 = col1 + 1;
            let cw = ((col2 as f32 - c).mul_add((1 << SHIFT) as f32, 0.5)).floor() as i32 as i16 as i32;
            (col1.max(0) as usize, col2.min(sw - 1) as usize, cw)
        })
        .collect();
    let mut out = Mat::new(h, w, 1);
    for row in 0..h {
        let r = (row as f32).mul_add(hr, off_y);
        let src_row = r.floor() as i64;
        let src_row2 = src_row + 1;
        let rweight = src_row2 as f32 - r;
        let rw = rweight.mul_add((1 << SHIFT) as f32, 0.5).floor() as i32;
        let (vrw, vrw2) = (rw as i16, ((1 << SHIFT) - rw) as i16);
        let s1 = src.row(src_row.max(0) as usize);
        let s2 = src.row(src_row2.min(sh - 1) as usize);
        let d = out.row_mut(row);
        for (x, &(c1, c2, cw)) in cols.iter().enumerate() {
            let v1 = ((s1[c2] as i32) << SHIFT) + (s1[c1] as i32 - s1[c2] as i32) * cw;
            let v2 = ((s2[c2] as i32) << SHIFT) + (s2[c1] as i32 - s2[c2] as i32) * cw;
            let a = qdmulh((v1 >> 4) as i16, vrw);
            let b = qdmulh((v2 >> 4) as i16, vrw2);
            let sum = ((a >> 1) as i32 + (b >> 1) as i32) as i16 as i32;
            d[x] = ((sum + 2) >> 2).clamp(0, 255) as u8;
        }
    }
    out
}
