//! `cv2.erode` with a rectangular all-ones kernel (depth.py trims the occluder map by ~1/CACHE_SIDE of the frame).
//!
//! Erosion is a per-channel minimum over the kernel window. OpenCV's default border for erode is BORDER_CONSTANT
//! with `morphologyDefaultBorderValue()` (+max), so pixels outside the image never win: the minimum is taken over
//! the part of the window inside the image. The anchor is the kernel centre (`k / 2`, so even sizes reach one more
//! pixel to the left/top). Order-independent, so exact for u8, f32 and f64 (NaN aside; f64 is depth.py's occluder
//! map).

use crate::mat::{par_rows, Mat, Pixel};

/// `cv2.erode(src, np.ones((kh, kw), np.uint8))`.
pub fn erode<T: Pixel>(src: &Mat<T>, kw: usize, kh: usize) -> Mat<T> {
    let (rows, cols, cn) = (src.rows, src.cols, src.cn);
    let (ax, ay) = (kw / 2, kh / 2);
    let w = cols * cn;
    let big = T::max_value();
    let mn = |a: T, b: T| if b < a { b } else { a };
    // horizontal: min over [x - ax, x - ax + kw) on a row padded with +max
    let mut tmp = Mat::new(rows, cols, cn);
    par_rows(&mut tmp.data, w, |y, d| {
        let mut pad = vec![big; (cols + kw) * cn];
        pad[ax * cn..ax * cn + w].copy_from_slice(src.row(y));
        d.copy_from_slice(&pad[..w]);
        for j in 1..kw {
            for (a, &b) in d.iter_mut().zip(&pad[j * cn..j * cn + w]) {
                *a = mn(*a, b);
            }
        }
    });
    let mut out = Mat::new(rows, cols, cn);
    par_rows(&mut out.data, w, |y, d| {
        let y0 = y.saturating_sub(ay);
        let y1 = (y + kh - ay).min(rows);
        d.copy_from_slice(tmp.row(y0));
        for yy in y0 + 1..y1 {
            for (a, &b) in d.iter_mut().zip(tmp.row(yy)) {
                *a = mn(*a, b);
            }
        }
    });
    out
}
