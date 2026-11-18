//! C ABI over the camera and head functions, for the parity tools (Python through ctypes; tools/motion/_common.py).
//! Grey images are 8-bit, row-major, `w * h` bytes, no padding. Every function is pure and thread-safe.
#![allow(clippy::missing_safety_doc)]

use crate::camera::{estimate, prepare_gray, production, CameraParams};
use crate::head::{head_motion, trackable, Gray, HeadParams};

unsafe fn gray<'a>(p: *const u8, w: i32, h: i32) -> Gray<'a> {
    Gray { w: w as usize, h: h as usize, d: std::slice::from_raw_parts(p, (w * h) as usize) }
}

/// Camera motion g0 -> g1 (source pixels). out_m: 12 doubles, the raw fit then the production row (2x3 each).
/// info: inlier share, blocks, inliers, sigma. Returns 1 when trusted.
#[no_mangle]
pub unsafe extern "C" fn mm_motion_camera_pair(g0: *const u8, g1: *const u8, w: i32, h: i32, out_m: *mut f64, info: *mut f64) -> i32 {
    let p = CameraParams::default();
    let a = gray(g0, w, h);
    let b = gray(g1, w, h);
    let fa = prepare_gray(a.w, a.h, a.d, &p);
    let fb = prepare_gray(b.w, b.h, b.d, &p);
    let e = estimate(&fa, &fb, &p);
    let prod = production(&e, &p);
    let o = std::slice::from_raw_parts_mut(out_m, 12);
    o[..6].copy_from_slice(&e.m);
    o[6..].copy_from_slice(&prod);
    let i = std::slice::from_raw_parts_mut(info, 4);
    i.copy_from_slice(&[e.inlier_frac, e.blocks as f64, e.inliers as f64, e.sigma]);
    e.ok as i32
}

/// Head motion for box c (cc: cx, cy, s, then the core's x1, x2, y1, y2 fractions). out_m: the similarity (6).
/// out: d x, d y (NaN if none), n_good, spread, inlier share, template ZNCC, textured blocks. Returns the kind:
/// 0 ok, 1 no texture, 2 lost, 3 incoherent.
#[no_mangle]
pub unsafe extern "C" fn mm_motion_head(g0: *const u8, g1: *const u8, w: i32, h: i32, cc: *const f64, out_m: *mut f64, out: *mut f64) -> i32 {
    let c = std::slice::from_raw_parts(cc, 7);
    let r = head_motion(gray(g0, w, h), gray(g1, w, h), [c[0], c[1], c[2]], [c[3], c[4], c[5], c[6]], &HeadParams::default());
    std::slice::from_raw_parts_mut(out_m, 6).copy_from_slice(&r.m);
    let o = std::slice::from_raw_parts_mut(out, 7);
    let (dx, dy) = r.d.map(|d| (d[0], d[1])).unwrap_or((f64::NAN, f64::NAN));
    o.copy_from_slice(&[dx, dy, r.n_good as f64, r.spread, r.inlier_frac, r.template_zncc as f64, r.textured as f64]);
    r.kind as i32
}

/// Share of the picture that tracks g0 -> g1 both ways (follow.Clip.trackable).
#[no_mangle]
pub unsafe extern "C" fn mm_motion_trackable(g0: *const u8, g1: *const u8, w: i32, h: i32) -> f64 {
    trackable(gray(g0, w, h), gray(g1, w, h))
}

/// trackable's counts: blocks, no peak inside the search, weak peak, failed back, good (5 doubles).
#[no_mangle]
pub unsafe extern "C" fn mm_motion_trackable_counts(g0: *const u8, g1: *const u8, w: i32, h: i32, out: *mut f64) {
    let c = crate::head::trackable_counts(gray(g0, w, h), gray(g1, w, h));
    let o = std::slice::from_raw_parts_mut(out, 5);
    for i in 0..5 {
        o[i] = c[i] as f64;
    }
}
