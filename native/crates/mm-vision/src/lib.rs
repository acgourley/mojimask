//! Image functions the engine needs, bit-exact with the project's OpenCV (cv2 5.0.0 built by
//! packaging/build_opencv.sh, arm64, NEON baseline, Carotene HAL on). See native/docs/porting.md, "Scope: porting
//! the OpenCV calls to Rust". Each module states which OpenCV code path it reproduces and how it was checked
//! (native/tools/parity/vision_check.py).
//!
//! OpenCV is Apache-2.0 and Carotene BSD-3; the arithmetic here follows their sources closely.

pub mod border;
pub mod color;
pub mod filter;
pub mod gauss;
pub mod geom;
/// the project's frames (PROJECT/frames/NNNNN.jpg) exactly as cv2.imwrite / cv2.imread (libjpeg-turbo)
pub mod jpeg;
/// The depth movie's reader (implemented by mm-media's grey::MovieLuma; mm_analysis::depth reads through it)
pub mod luma {
    use crate::Mat;
    /// The movie reader (video_io.luma): (frame index, Y plane) from frame `start`, `count` frames or to the end.
    pub trait LumaSource: Send + Sync {
        fn luma(&self, start: usize, count: Option<usize>) -> Vec<(usize, Mat<u8>)>;
    }
}
pub mod mat;
pub mod morph;
pub mod npy;
pub mod resize;
mod resize_carotene;
pub mod warp;

pub use border::copy_make_border;
pub use color::{bgr2gray, swap_rb};
pub use filter::{filter2d_f32, filter2d_is_exact};
pub use gauss::{gaussian_blur_f32, gaussian_blur_u8};
pub use geom::{invert_affine, invert_affine_f32, point_polygon_test, transform3};
pub use mat::{set_num_threads, Border, Mat, Pixel};
pub use morph::erode;
pub use resize::{area_tab, resize_f32, resize_f64, resize_u8, Interp};
pub use warp::{warp_affine_f32, warp_affine_u8};
