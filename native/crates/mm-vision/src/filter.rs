//! `cv2.filter2D(src, -1, kernel)` for f32 images (1-4 channels) and an f32 kernel, default anchor (centre),
//! delta 0, default border (BORDER_REFLECT_101): lighting.py's motion blur of premultiplied sprites.
//!
//! - **Kernels under 50 taps (k <= 7 square), exact:** OpenCV's direct `Filter2D` keeps only the non-zero taps,
//!   row-major, and sums `s = fma(src_k, f_k, s)` from 0 in that order (NEON blocks and scalar tail alike).
//! - **50 taps or more (k >= 9), not bit-exact:** on arm64 OpenCV switches to a DFT convolution
//!   (`dftFilter2D`, threshold 50 without SSE3). This runs the same direct sum instead; the difference is float
//!   rounding only (max ~1e-7 of the value, measured by vision_check.py). The plan lists filter2D as drawing-only
//!   (approximate is acceptable, native/docs/porting.md).

use crate::mat::{border_index, par_rows, Border, Mat};

/// Whether cv2 takes the exact direct path for this kernel size (else its DFT path).
pub fn filter2d_is_exact(kw: usize, kh: usize) -> bool {
    kw * kh < 50
}

pub fn filter2d_f32(src: &Mat<f32>, kernel: &Mat<f32>) -> Mat<f32> {
    assert_eq!(kernel.cn, 1, "filter2d: single-channel kernel");
    let (kh, kw) = (kernel.rows, kernel.cols);
    let (ax, ay) = (kw / 2, kh / 2);
    let taps: Vec<(usize, usize, f32)> = (0..kh)
        .flat_map(|y| (0..kw).map(move |x| (y, x)))
        .filter_map(|(y, x)| {
            let f = kernel.data[y * kw + x];
            (f != 0.0).then_some((y, x, f))
        })
        .collect();
    let (rows, cols, cn) = (src.rows, src.cols, src.cn);
    let w = cols * cn;
    // every source row padded by the kernel's width (REFLECT_101), as FilterEngine's ring buffer holds them
    let pw = (cols + kw - 1) * cn;
    let mut padded = vec![0f32; rows * pw];
    par_rows(&mut padded, pw, |y, d| {
        let s = src.row(y);
        for p in 0..cols + kw - 1 {
            let q = border_index(p as isize - ax as isize, cols, Border::Reflect101) as usize;
            d[p * cn..(p + 1) * cn].copy_from_slice(&s[q * cn..(q + 1) * cn]);
        }
    });
    let mut out = Mat::new(rows, cols, cn);
    par_rows(&mut out.data, w, |y, d| {
        for &(ty, tx, f) in &taps {
            let sy = border_index(y as isize - ay as isize + ty as isize, rows, Border::Reflect101) as usize;
            let s = &padded[sy * pw + tx * cn..sy * pw + tx * cn + w];
            for (o, &v) in d.iter_mut().zip(s) {
                *o = v.mul_add(f, *o);
            }
        }
    });
    out
}
