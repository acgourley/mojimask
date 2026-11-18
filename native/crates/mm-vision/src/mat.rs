//! A plain interleaved image (OpenCV's dense `Mat` with 1-4 channels), and the border rules shared by the filters.

/// Rows x cols x channels, row-major, channels interleaved (OpenCV's and numpy's HWC layout). No padding.
#[derive(Clone, Debug, PartialEq)]
pub struct Mat<T> {
    pub rows: usize,
    pub cols: usize,
    pub cn: usize,
    pub data: Vec<T>,
}

impl<T: Copy + Default> Mat<T> {
    pub fn new(rows: usize, cols: usize, cn: usize) -> Self {
        Mat { rows, cols, cn, data: vec![T::default(); rows * cols * cn] }
    }
    pub fn from_vec(rows: usize, cols: usize, cn: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), rows * cols * cn, "Mat::from_vec: size");
        Mat { rows, cols, cn, data }
    }
    pub fn filled(rows: usize, cols: usize, cn: usize, v: T) -> Self {
        Mat { rows, cols, cn, data: vec![v; rows * cols * cn] }
    }
    #[inline]
    pub fn row(&self, y: usize) -> &[T] {
        let w = self.cols * self.cn;
        &self.data[y * w..(y + 1) * w]
    }
    #[inline]
    pub fn row_mut(&mut self, y: usize) -> &mut [T] {
        let w = self.cols * self.cn;
        &mut self.data[y * w..(y + 1) * w]
    }
    /// Row stride in elements.
    #[inline]
    pub fn step(&self) -> usize {
        self.cols * self.cn
    }
    /// A copy of the rectangle [x, x+w) x [y, y+h) (numpy's `a[y:y+h, x:x+w]`).
    pub fn roi(&self, x: usize, y: usize, w: usize, h: usize) -> Self {
        assert!(x + w <= self.cols && y + h <= self.rows, "Mat::roi out of range");
        let mut out = Vec::with_capacity(w * h * self.cn);
        for r in y..y + h {
            out.extend_from_slice(&self.row(r)[x * self.cn..(x + w) * self.cn]);
        }
        Mat { rows: h, cols: w, cn: self.cn, data: out }
    }
}

/// The pixel depths the engine uses: u8 and f32 everywhere, f64 for depth.py's maps (resize INTER_AREA, erode).
pub trait Pixel: Copy + Default + PartialOrd + Send + Sync + 'static {
    const IS_U8: bool;
    fn to_f32(self) -> f32;
    fn to_f64(self) -> f64;
    /// OpenCV's `saturate_cast<T>(double)`: round half to even and clamp for u8, plain cast for f32.
    fn sat_f64(v: f64) -> Self;
    fn max_value() -> Self;
}

impl Pixel for u8 {
    const IS_U8: bool = true;
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn sat_f64(v: f64) -> u8 {
        let r = v.round_ties_even();
        if r.is_nan() {
            0
        } else {
            r.clamp(0.0, 255.0) as u8
        }
    }
    fn max_value() -> u8 {
        255
    }
}

impl Pixel for f32 {
    const IS_U8: bool = false;
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn sat_f64(v: f64) -> f32 {
        v as f32
    }
    fn max_value() -> f32 {
        f32::MAX
    }
}

impl Pixel for f64 {
    const IS_U8: bool = false;
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn sat_f64(v: f64) -> f64 {
        v
    }
    fn max_value() -> f64 {
        f64::MAX
    }
}

/// OpenCV border modes used here. `Reflect101` is `BORDER_DEFAULT` (gfedcb|abcdefgh|gfedcba).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Border {
    Constant,
    Replicate,
    Reflect,
    Reflect101,
}

/// `cv::borderInterpolate` for the non-constant modes: where outside index `p` reads from in `0..len`.
#[inline]
pub fn border_index(p: isize, len: usize, border: Border) -> isize {
    let len = len as isize;
    if (p as usize) < len as usize && p >= 0 {
        return p;
    }
    match border {
        Border::Replicate => {
            if p < 0 {
                0
            } else {
                len - 1
            }
        }
        Border::Reflect | Border::Reflect101 => {
            if len == 1 {
                return 0;
            }
            let delta = (border == Border::Reflect101) as isize;
            let mut p = p;
            loop {
                if p < 0 {
                    p = -p - 1 + delta;
                } else {
                    p = len - 1 - (p - len) - delta;
                }
                if p >= 0 && p < len {
                    break;
                }
            }
            p
        }
        Border::Constant => -1,
    }
}

static NUM_THREADS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Like `cv2.setNumThreads`: how many threads the row-parallel functions use (0 = all cores, 1 = none).
/// Results never depend on it (rows are computed independently).
pub fn set_num_threads(n: usize) {
    NUM_THREADS.store(n, std::sync::atomic::Ordering::Relaxed);
}

fn threads_for(work: usize) -> usize {
    let n = match NUM_THREADS.load(std::sync::atomic::Ordering::Relaxed) {
        0 => std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1),
        n => n,
    };
    // below ~64k elements threads cost more than they save (OpenCV's parallel_for_ uses a similar grain)
    n.min(work / 65536).max(1)
}

/// Fills `out` (rows of `row_len` elements) by calling `f(y, row)` for every row, split across threads.
pub fn par_rows<T: Send>(out: &mut [T], row_len: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    if row_len == 0 || out.is_empty() {
        return;
    }
    let rows = out.len() / row_len;
    let nt = threads_for(out.len()).min(rows);
    if nt <= 1 {
        for (y, r) in out.chunks_mut(row_len).enumerate() {
            f(y, r);
        }
        return;
    }
    let per = rows.div_ceil(nt);
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(per * row_len).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (k, r) in chunk.chunks_mut(row_len).enumerate() {
                    f(ci * per + k, r);
                }
            });
        }
    });
}
