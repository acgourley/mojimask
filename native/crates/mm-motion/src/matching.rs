//! Matching one block of frame a in frame b: normalised cross-correlation (ZNCC) searched over whole-pixel moves, a
//! sub-pixel peak, and a Lucas-Kanade polish that also allows for a change of brightness.
//!
//! ZNCC compares the block's pattern after removing its mean and contrast, so a change of exposure (a phone's auto
//! exposure in a dark room, a flash) doesn't move the match. The polish (inverse compositional, translation only,
//! gain and offset normalised each step) removes ZNCC's pull towards whole pixels.

use crate::img::{Img, Level};

/// A block's pattern: its pixels less their mean, scaled to unit length (so the dot product with another
/// zero-mean pattern is the correlation).
pub struct Template {
    pub x: usize,
    pub y: usize,
    pub b: usize,
    pub t: Vec<f32>,
    /// standard deviation of the raw pixels (0 = flat)
    pub std: f32,
}

impl Template {
    pub fn new(img: &Img, x: usize, y: usize, b: usize) -> Template {
        let mut t = Vec::with_capacity(b * b);
        for yy in y..y + b {
            t.extend_from_slice(&img.d[yy * img.w + x..yy * img.w + x + b]);
        }
        let n = t.len() as f32;
        let mean = t.iter().sum::<f32>() / n;
        let mut ss = 0.0;
        for v in t.iter_mut() {
            *v -= mean;
            ss += *v * *v;
        }
        let norm = ss.sqrt();
        if norm > 1e-6 {
            t.iter_mut().for_each(|v| *v /= norm);
        }
        Template { x, y, b, t, std: (ss / n).sqrt() }
    }
}

/// ZNCC of the template against b's block with top-left (x, y) (whole pixels, inside b).
#[inline]
pub fn zncc_at(tp: &Template, img: &Img, x: usize, y: usize) -> f32 {
    let b = tp.b;
    let (mut s, mut ss, mut c) = (0f32, 0f32, 0f32);
    for j in 0..b {
        let row = &img.d[(y + j) * img.w + x..(y + j) * img.w + x + b];
        let tr = &tp.t[j * b..(j + 1) * b];
        for i in 0..b {
            let v = row[i];
            s += v;
            ss += v * v;
            c += v * tr[i];
        }
    }
    let n = (b * b) as f32;
    let var = ss - s * s / n;
    if var <= 1e-3 {
        return 0.0;
    }
    c / var.sqrt()
}

/// The correlation surface of the template over whole-pixel moves (ox, oy) in [cx - r, cx + r] x [cy - r, cy + r]
/// (moves of the top-left corner). Positions where the block leaves b score NaN.
pub fn surface(tp: &Template, img: &Img, cx: i32, cy: i32, r: i32) -> Vec<f32> {
    let n = (2 * r + 1) as usize;
    let mut out = vec![f32::NAN; n * n];
    for j in 0..n as i32 {
        let y = tp.y as i32 + cy + j - r;
        if y < 0 || y as usize + tp.b > img.h {
            continue;
        }
        for i in 0..n as i32 {
            let x = tp.x as i32 + cx + i - r;
            if x < 0 || x as usize + tp.b > img.w {
                continue;
            }
            out[(j * n as i32 + i) as usize] = zncc_at(tp, img, x as usize, y as usize);
        }
    }
    out
}

/// The peak of a (2r+1)^2 surface with a parabola through it and its neighbours on each axis:
/// (dx, dy) relative to the surface's centre, the peak value, and whether it lies inside (not on the edge).
pub fn peak(s: &[f32], r: i32) -> Option<(f32, f32, f32, bool)> {
    let n = (2 * r + 1) as usize;
    let mut best = -2.0f32;
    let mut bi = usize::MAX;
    for (k, &v) in s.iter().enumerate() {
        if v.is_finite() && v > best {
            best = v;
            bi = k;
        }
    }
    if bi == usize::MAX {
        return None;
    }
    let (i, j) = (bi % n, bi / n);
    let at = |i: usize, j: usize| s[j * n + i];
    let mut inside = true;
    let sub = |a: f32, c: f32, b: f32| -> f32 {
        let den = a - 2.0 * c + b;
        if den.abs() < 1e-9 || !a.is_finite() || !b.is_finite() {
            0.0
        } else {
            (0.5 * (a - b) / den).clamp(-0.5, 0.5)
        }
    };
    let fx = if i > 0 && i + 1 < n {
        sub(at(i - 1, j), best, at(i + 1, j))
    } else {
        inside = false;
        0.0
    };
    let fy = if j > 0 && j + 1 < n {
        sub(at(i, j - 1), best, at(i, j + 1))
    } else {
        inside = false;
        0.0
    };
    Some((i as f32 - r as f32 + fx, j as f32 - r as f32 + fy, best, inside))
}

/// Polish the move (dx, dy) of the template's block from a to b (Lucas-Kanade, inverse compositional, with the
/// target's mean and contrast matched to the template's every step). Returns the polished move, or None if it ran
/// more than `max_shift` from where it started or left the picture.
pub fn polish(lv0: &Level, b_img: &Img, x: usize, y: usize, bs: usize, dx: f32, dy: f32, iters: usize, max_shift: f32) -> Option<(f32, f32)> {
    let a = &lv0.img;
    let n = bs * bs;
    let mut t = Vec::with_capacity(n);
    let mut gx = Vec::with_capacity(n);
    let mut gy = Vec::with_capacity(n);
    for yy in y..y + bs {
        let o = yy * a.w;
        for xx in x..x + bs {
            t.push(a.d[o + xx]);
            gx.push(lv0.gx.d[o + xx]);
            gy.push(lv0.gy.d[o + xx]);
        }
    }
    let nf = n as f32;
    let tm = t.iter().sum::<f32>() / nf;
    let tsd = (t.iter().map(|v| (v - tm) * (v - tm)).sum::<f32>() / nf).sqrt();
    if tsd < 1e-3 {
        return None;
    }
    let (mut h11, mut h12, mut h22) = (0f32, 0f32, 0f32);
    for k in 0..n {
        h11 += gx[k] * gx[k];
        h12 += gx[k] * gy[k];
        h22 += gy[k] * gy[k];
    }
    let det = h11 * h22 - h12 * h12;
    if det.abs() < 1e-6 {
        return None;
    }
    let (mut ux, mut uy) = (dx, dy);
    let mut wv = vec![0f32; n];
    for _ in 0..iters {
        let (fx, fy) = (x as f32 + ux, y as f32 + uy);
        if fx < 0.0 || fy < 0.0 || fx + bs as f32 > b_img.w as f32 - 1.0 || fy + bs as f32 > b_img.h as f32 - 1.0 {
            return None;
        }
        let mut k = 0;
        for j in 0..bs {
            for i in 0..bs {
                wv[k] = b_img.sample(fx + i as f32, fy + j as f32);
                k += 1;
            }
        }
        let wm = wv.iter().sum::<f32>() / nf;
        let wsd = (wv.iter().map(|v| (v - wm) * (v - wm)).sum::<f32>() / nf).sqrt();
        if wsd < 1e-3 {
            return None;
        }
        let g = tsd / wsd;
        let (mut b1, mut b2) = (0f32, 0f32);
        for k in 0..n {
            let e = (wv[k] - wm) * g - (t[k] - tm);
            b1 += gx[k] * e;
            b2 += gy[k] * e;
        }
        let ddx = (h22 * b1 - h12 * b2) / det;
        let ddy = (h11 * b2 - h12 * b1) / det;
        ux -= ddx;
        uy -= ddy;
        if (ux - dx).abs() > max_shift || (uy - dy).abs() > max_shift {
            return None;
        }
        if ddx.abs() + ddy.abs() < 0.005 {
            break;
        }
    }
    Some((ux, uy))
}

/// Coarse-to-fine search of one block (top-left (x, y) on level 0, size bs) starting from the move (dx, dy) in level-0
/// pixels: within `r_top` on level `top`, then within 2 on each finer level. Ok((dx, dy, zncc)) on level 0, or
/// Err(1) when a peak sat on the edge of its search, Err(2) when the level-0 peak is under `min_zncc`.
#[allow(clippy::too_many_arguments)]
pub fn pyramid_search(la: &[Level], lb: &[Level], x: usize, y: usize, bs: usize, dx: f32, dy: f32, top: usize, r_top: i32, min_zncc: f32) -> Result<(f32, f32, f32), u8> {
    let half = (bs as f32 - 1.0) / 2.0;
    let (cx0, cy0) = (x as f32 + half, y as f32 + half);
    let (mut mx, mut my) = (dx, dy);
    let mut last = 0.0;
    for l in (0..=top.min(la.len() - 1)).rev() {
        let k = (1usize << l) as f32;
        let (a, b) = (&la[l].img, &lb[l].img);
        let tx = (cx0 / k - half).round();
        let ty = (cy0 / k - half).round();
        if tx < 0.0 || ty < 0.0 || tx as usize + bs > a.w || ty as usize + bs > a.h {
            continue;
        }
        let tp = Template::new(a, tx as usize, ty as usize, bs);
        let r = if l == top { r_top } else { 2 };
        let (ix, iy) = ((mx / k).round() as i32, (my / k).round() as i32);
        let s = surface(&tp, b, ix, iy, r);
        let Some((fx, fy, v, inside)) = peak(&s, r) else { return Err(1) };
        if !inside {
            return Err(1);
        }
        mx = (ix as f32 + fx) * k;
        my = (iy as f32 + fy) * k;
        last = v;
    }
    if last < min_zncc {
        return Err(2);
    }
    Ok((mx, my, last))
}
