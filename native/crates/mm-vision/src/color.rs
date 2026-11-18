//! `cv2.cvtColor` for the conversions the engine uses: BGR->GRAY (8-bit) and BGR<->RGB (any depth).
//!
//! The project's OpenCV is built with Carotene (its NEON HAL, on by default for ARM), which takes over 8-bit
//! BGR->GRAY. Carotene uses 15-bit coefficients (R 9798, G 19235, B 3735, rounded shift), not OpenCV's own
//! 14-bit ones (4899, 9617, 1868); the two differ by 1 level on some pixels, and Carotene's is what cv2 returns
//! on the Mac. Checked against cv2 5.0.0 on jazz frames and random images (vision_check.py).

use crate::mat::Mat;

/// `cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)` for 3-channel 8-bit.
pub fn bgr2gray(src: &Mat<u8>) -> Mat<u8> {
    assert_eq!(src.cn, 3, "bgr2gray: 3 channels");
    let mut out = Mat::new(src.rows, src.cols, 1);
    for (d, p) in out.data.iter_mut().zip(src.data.chunks_exact(3)) {
        let v = p[0] as u32 * 3735 + p[1] as u32 * 19235 + p[2] as u32 * 9798;
        *d = ((v + (1 << 14)) >> 15).min(255) as u8;
    }
    out
}

/// `cv2.cvtColor(src, cv2.COLOR_BGR2RGB)` (and RGB2BGR, the same swap) for 3 or 4 channels.
pub fn swap_rb<T: Copy + Default>(src: &Mat<T>) -> Mat<T> {
    assert!(src.cn == 3 || src.cn == 4, "swap_rb: 3 or 4 channels");
    let mut out = src.clone();
    for p in out.data.chunks_exact_mut(src.cn) {
        p.swap(0, 2);
    }
    out
}
