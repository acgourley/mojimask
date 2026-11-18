//! `cv2.copyMakeBorder` with BORDER_CONSTANT and BORDER_REPLICATE (depth.py pads its maps, lighting.py pads sprites
//! before the motion blur). Pure copies: exact by construction.

use crate::mat::{border_index, Border, Mat};

/// `cv2.copyMakeBorder(src, top, bottom, left, right, border, value=value)`. `value` is per channel (first `cn`
/// used); ignored for `Replicate`.
pub fn copy_make_border<T: Copy + Default>(
    src: &Mat<T>,
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
    border: Border,
    value: &[T],
) -> Mat<T> {
    let cn = src.cn;
    let (rows, cols) = (src.rows + top + bottom, src.cols + left + right);
    let mut out = Mat::new(rows, cols, cn);
    let mut fill = vec![T::default(); cn];
    for c in 0..cn {
        if let Some(v) = value.get(c) {
            fill[c] = *v;
        }
    }
    let w = src.cols * cn;
    // one padded row per source row (or the constant row), then rows copied
    let fill_row: Vec<T> = (0..cols).flat_map(|_| fill.iter().copied()).collect();
    let make = |r: usize| -> Vec<T> {
        let mut d = Vec::with_capacity(cols * cn);
        let s = src.row(r);
        for x in 0..left {
            let sx = x as isize - left as isize;
            if border == Border::Constant {
                d.extend_from_slice(&fill);
            } else {
                let q = border_index(sx, src.cols, border) as usize;
                d.extend_from_slice(&s[q * cn..(q + 1) * cn]);
            }
        }
        d.extend_from_slice(&s[..w]);
        for x in 0..right {
            if border == Border::Constant {
                d.extend_from_slice(&fill);
            } else {
                let q = border_index((src.cols + x) as isize, src.cols, border) as usize;
                d.extend_from_slice(&s[q * cn..(q + 1) * cn]);
            }
        }
        d
    };
    let ow = cols * cn;
    for y in 0..rows {
        let sy = y as isize - top as isize;
        let d = &mut out.data[y * ow..(y + 1) * ow];
        if border == Border::Constant && (sy < 0 || sy >= src.rows as isize) {
            d.copy_from_slice(&fill_row);
        } else {
            let r = if border == Border::Constant { sy as usize } else { border_index(sy, src.rows, border) as usize };
            if left == 0 && right == 0 {
                d.copy_from_slice(src.row(r));
            } else {
                d.copy_from_slice(&make(r));
            }
        }
    }
    out
}
