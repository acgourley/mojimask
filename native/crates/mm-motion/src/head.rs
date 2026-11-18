//! How one head moved between two frames: the motion half of the scan's follower (mojimask/scan/follow.py).
//!
//! follow.py asks three motion questions; this module answers them with the same inputs and outputs:
//! - **the flow position** (`track._flow_measure(g0, g1, c, core)` -> (d xy, n_good, spread) or None): where the
//!   head's own motion moves box c (follow's candidate "flow", beside "camera");
//! - **the head's similarity and whether it is lost** (`follow.face_motion(g0, g1, c)` -> (M, kind), kind "ok", "no
//!   texture", "lost" or "incoherent"): used by the follower's lost check and by cover.flows (the steady emoji);
//! - **whether the whole frame tracks** (`follow.Clip.trackable(a, b)` -> share in 0..1).
//!
//! Python's rule for "lost", kept here (follow.py): on a visible face nearly all of its texture agrees on one move
//! (jazz, the dancer bouncing: 0.6-1.0 of the corners; the move lands within 0.01-0.05 head sizes of her next
//! detection, while the detections themselves jump up to 0.2). When someone passes in front, the texture goes with
//! them or fails: under LOST_INL (0.3) agree. That's "lost" -- but only where the rest of the picture tracks fine
//! (trackable >= STEADY_FRAME, 0.6; in a whip pan everything fails and a smeared face can still be there) and the
//! box, with a head's width around it, is inside the picture (follow.py checks those two). Too little texture ("no
//! texture": a dark or flat face) says nothing. Python's trackable: low in a whip pan or a blurred frame
//! (party_clip_c's pan, frames 180-238: 0.00-0.48), never under 0.64 on jazz's steady shot.
//!
//! **How we measure it** (no corners, no RANSAC):
//! 1. Both frames are resampled around the box into a crop where the head is `norm` (48) pixels across and the
//!    crop reaches 1.15 head sizes from its centre (big heads are box-averaged first, so a 4K head isn't aliased;
//!    small heads are enlarged at most 2x).
//! 2. **Template search:** the box's core (the `core` fractions of the box) at half that resolution is searched by
//!    ZNCC over moves up to `reach` (0.6) head sizes. One template for the whole face: it survives blur, a turn and
//!    a change of light that break small patches.
//! 3. **Blocks:** a 4 x 4 grid of 12 px blocks over the core, each searched within 3 px of the template's move,
//!    polished by Lucas-Kanade, then searched back from where it landed; a block counts ("good") only if it comes
//!    back within `fb_px` (1 px, 2% of a head), Python's forward-backward test.
//! 4. A similarity is fitted to the good blocks by IRLS with Tukey weights (fit.rs), starting from the template's
//!    move. Kinds: fewer than 6 textured blocks -> "no texture"; fewer than 4 good ones -> "lost"; under `lost_inl`
//!    (0.4) of the textured blocks within max(1 px, 0.04 head sizes) of the fit -> "incoherent"; else "ok". (Blocks
//!    cover the face evenly where corners crowd onto its strongest features, so the share is taken higher than
//!    Python's 0.3: with 0.3, 92% of heads covered 60-80% by someone passing were called lost; with 0.4, 100%,
//!    Python 98.5%; visible heads "ok" 92.4%, Python 91.3%.)
//! 5. The flow position's move d is the fit's move of the box centre (a pure move if under 5 blocks agree; None when
//!    fewer than 3 are good), n_good the good blocks, spread their median residual (source px).
//!
//! Evidence: tools/motion/heads.py (synthetic heads on known paths: moves, turns, scale, blur, noise, someone passing
//! in front; and jazz's census faces 4 frames apart, followed frame by frame), 2026-11-17: flow position failures
//! (error over 0.1 head sizes or none) 1.5% vs Python's 3.8% on visible heads (2.5x enlarged heads 1.5% vs 6.4%;
//! fast moves of 0.3-0.5 heads 0% vs 12%); real jazz followed 4 frames, mean error 0.043 vs 0.052 head sizes (camera
//! alone 0.097); face_motion's kind the same as Python's on 96% of census faces. trackable: on jazz's pairs 0.67-0.90
//! (Python 0.67-0.91, correlation 0.9), and lower than Python's under heavy horizontal blur (0.50 vs 0.64).

use crate::camera::{self, CameraParams};
use crate::fit::{self, M23};
use crate::img::{Img, Level};
use crate::matching::{peak, polish, surface, Template};

#[derive(Clone, Copy)]
pub struct Gray<'a> {
    pub w: usize,
    pub h: usize,
    pub d: &'a [u8],
}

#[derive(Clone, Debug)]
pub struct HeadParams {
    pub norm: f64,
    pub max_up: f64,
    pub crop_half: f64,
    pub reach: f64,
    pub block: usize,
    pub grid: usize,
    pub min_zncc: f32,
    pub fb_px: f32,
    pub lost_inl: f64,
    pub min_textured: usize,
    pub min_good: usize,
    pub tex_floor: f32,
}

impl Default for HeadParams {
    fn default() -> Self {
        HeadParams {
            norm: 48.0,
            max_up: 2.0,
            crop_half: 1.15,
            reach: 0.6,
            block: 12,
            grid: 4,
            min_zncc: 0.5,
            fb_px: 1.0,
            lost_inl: 0.4,
            min_textured: 6,
            min_good: 4,
            tex_floor: 0.5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Kind {
    Ok = 0,
    NoTexture = 1,
    Lost = 2,
    Incoherent = 3,
}

#[derive(Clone, Debug)]
pub struct HeadMotion {
    pub kind: Kind,
    /// similarity, source pixels, frame a -> b (meaningful when kind is Ok; else the best guess or identity)
    pub m: M23,
    /// the flow position's move of the box centre (follow's "flow"), None when too few blocks track
    pub d: Option<[f64; 2]>,
    pub n_good: usize,
    pub spread: f64,
    pub inlier_frac: f64,
    pub template_zncc: f32,
    pub textured: usize,
}

/// Source region resampled to a crop: crop pixel (i, j) is source (cx + (i - half) / k, cy + (j - half) / k).
fn crop(g: Gray, cx: f64, cy: f64, k: f64, half: usize) -> Img {
    let n = 2 * half + 1;
    let f = if k < 0.5 { (1.0 / k).floor() as usize } else { 1 };
    // the source rectangle the crop needs (plus a cell), box-averaged by f
    let reach = (half as f64 + 2.0) / k + f as f64;
    let x0 = ((cx - reach).floor().max(0.0) as usize / f) * f;
    let y0 = ((cy - reach).floor().max(0.0) as usize / f) * f;
    let x1 = ((cx + reach).ceil().max(0.0) as usize).min(g.w);
    let y1 = ((cy + reach).ceil().max(0.0) as usize).min(g.h);
    let mut out = Img::new(n, n);
    if x1 <= x0 + f || y1 <= y0 + f {
        return out;
    }
    let bw = (x1 - x0) / f;
    let bh = (y1 - y0) / f;
    let mut b = Img::new(bw, bh);
    let inv = 1.0 / (f * f) as f32;
    for by in 0..bh {
        for bx in 0..bw {
            let mut s = 0u32;
            for y in y0 + by * f..y0 + (by + 1) * f {
                let row = &g.d[y * g.w + x0 + bx * f..y * g.w + x0 + (bx + 1) * f];
                s += row.iter().map(|&v| v as u32).sum::<u32>();
            }
            b.d[by * bw + bx] = s as f32 * inv;
        }
    }
    let o = (f as f64 - 1.0) / 2.0;
    for j in 0..n {
        let ys = cy + (j as f64 - half as f64) / k;
        let yb = ((ys - y0 as f64 - o) / f as f64) as f32;
        for i in 0..n {
            let xs = cx + (i as f64 - half as f64) / k;
            let xb = ((xs - x0 as f64 - o) / f as f64) as f32;
            out.d[j * n + i] = b.sample(xb, yb);
        }
    }
    out
}

pub fn head_motion(g0: Gray, g1: Gray, c: [f64; 3], core: [f64; 4], p: &HeadParams) -> HeadMotion {
    let mut out = HeadMotion {
        kind: Kind::NoTexture,
        m: fit::IDENTITY,
        d: None,
        n_good: 0,
        spread: 0.0,
        inlier_frac: 0.0,
        template_zncc: 0.0,
        textured: 0,
    };
    let s = c[2];
    // Python: the core clipped to the picture must be at least 4 px each way
    let (bx1, by1) = (c[0] - s / 2.0, c[1] - s / 2.0);
    let (qx1, qx2) = ((bx1 + core[0] * s).max(0.0), (bx1 + core[1] * s).min(g0.w as f64 - 1.0));
    let (qy1, qy2) = ((by1 + core[2] * s).max(0.0), (by1 + core[3] * s).min(g0.h as f64 - 1.0));
    if s <= 1.0 || qx2 - qx1 < 4.0 || qy2 - qy1 < 4.0 {
        return out;
    }
    let k = (p.norm / s).min(p.max_up);
    let sn = s * k;
    let half = (p.crop_half * sn).ceil() as usize;
    let a0 = Level::new(crop(g0, c[0], c[1], k, half));
    let b0 = Level::new(crop(g1, c[0], c[1], k, half));
    // the core in crop pixels (clipped to the picture, as Python clips it)
    let to_crop = |x: f64, cc: f64| (x - cc) * k + half as f64;
    let (cx1, cx2) = (to_crop(qx1, c[0]), to_crop(qx2, c[0]));
    let (cy1, cy2) = (to_crop(qy1, c[1]), to_crop(qy2, c[1]));

    // 2. template search at half resolution
    let a1 = Level::new(a0.img.down2());
    let b1 = Level::new(b0.img.down2());
    let tx = ((cx1 / 2.0).round() as usize).min(a1.img.w - 2);
    let ty = ((cy1 / 2.0).round() as usize).min(a1.img.h - 2);
    let tw = (((cx2 - cx1) / 2.0).round() as usize).max(3).min(a1.img.w - tx);
    let th = (((cy2 - cy1) / 2.0).round() as usize).max(3).min(a1.img.h - ty);
    let tsz = tw.min(th);
    // ZNCC wants a square block: the core's middle square, plus the rest when the core is taller than wide
    let mut best: Option<(f32, f32, f32)> = None;
    let r = ((p.reach * sn / 2.0).ceil() as i32).max(3);
    let starts: Vec<(usize, usize)> = if th > tw {
        vec![(tx, ty), (tx, ty + th - tsz)]
    } else if tw > th {
        vec![(tx, ty), (tx + tw - tsz, ty)]
    } else {
        vec![(tx, ty)]
    };
    let mut sum: Option<Vec<f32>> = None;
    for (x, y) in starts {
        let tp = Template::new(&a1.img, x, y, tsz);
        if tp.std < 0.5 {
            continue;
        }
        let sf = surface(&tp, &b1.img, 0, 0, r);
        sum = Some(match sum {
            None => sf,
            Some(acc) => acc.iter().zip(&sf).map(|(a, b)| a + b).collect(),
        });
    }
    let parts = if th != tw { 2.0 } else { 1.0 };
    if let Some(sf) = &sum {
        let sf: Vec<f32> = sf.iter().map(|v| v / parts).collect();
        if let Some((dx, dy, v, _)) = peak(&sf, r) {
            best = Some((2.0 * dx, 2.0 * dy, v));
        }
    }
    let (tdx, tdy, tz) = best.unwrap_or((0.0, 0.0, 0.0));
    out.template_zncc = tz;

    // 3. blocks over the core, forward and back
    let bs = p.block;
    let g = p.grid;
    let (cw, ch) = ((cx2 - cx1) / g as f64, (cy2 - cy1) / g as f64);
    let (mut ps, mut qs) = (vec![], vec![]);
    let mut textured = 0;
    let n = a0.img.w;
    for j in 0..g {
        for i in 0..g {
            let mx = cx1 + (i as f64 + 0.5) * cw;
            let my = cy1 + (j as f64 + 0.5) * ch;
            let x = (mx - bs as f64 / 2.0).round().clamp(1.0, (n - bs - 1) as f64) as usize;
            let y = (my - bs as f64 / 2.0).round().clamp(1.0, (n - bs - 1) as f64) as usize;
            if a0.texture(x, y, bs) < p.tex_floor {
                continue;
            }
            textured += 1;
            let tp = Template::new(&a0.img, x, y, bs);
            let (ix, iy) = (tdx.round() as i32, tdy.round() as i32);
            let sf = surface(&tp, &b0.img, ix, iy, 3);
            let Some((fx, fy, v, inside)) = peak(&sf, 3) else { continue };
            if v < p.min_zncc || !inside {
                continue;
            }
            let (mut ux, mut uy) = (ix as f32 + fx, iy as f32 + fy);
            if let Some((a, b)) = polish(&a0, &b0.img, x, y, bs, ux, uy, 8, 1.0) {
                ux = a;
                uy = b;
            }
            // back: the block where it landed (whole pixels), searched in frame a
            let (lx, ly) = (x as f32 + ux, y as f32 + uy);
            let (rx, ry) = (lx.round(), ly.round());
            if rx < 1.0 || ry < 1.0 || rx as usize + bs + 1 > n || ry as usize + bs + 1 > n {
                continue;
            }
            let (rxu, ryu) = (rx as usize, ry as usize);
            let tb = Template::new(&b0.img, rxu, ryu, bs);
            let (ex, ey) = (x as f32 - rx, y as f32 - ry); // the move back that would be consistent
            let (jx, jy) = (ex.round() as i32, ey.round() as i32);
            let sb = surface(&tb, &a0.img, jx, jy, 2);
            let Some((gx, gy, vb, inb)) = peak(&sb, 2) else { continue };
            if vb < p.min_zncc || !inb {
                continue;
            }
            let (mut vx, mut vy) = (jx as f32 + gx, jy as f32 + gy);
            if let Some((a, b)) = polish(&b0, &a0.img, rxu, ryu, bs, vx, vy, 8, 1.0) {
                vx = a;
                vy = b;
            }
            if ((vx - ex).powi(2) + (vy - ey).powi(2)).sqrt() > p.fb_px {
                continue;
            }
            let half_b = (bs as f64 - 1.0) / 2.0;
            ps.push([x as f64 + half_b, y as f64 + half_b]);
            qs.push([x as f64 + half_b + ux as f64, y as f64 + half_b + uy as f64]);
        }
    }
    out.textured = textured;
    out.n_good = ps.len();
    if textured < p.min_textured {
        out.kind = Kind::NoTexture;
        return out;
    }
    // crop -> source: x_n = k (x_s - c) + half
    let hf = half as f64;
    let a: M23 = [k, 0.0, hf - k * c[0], 0.0, k, hf - k * c[1]];
    let to_src = |m: &M23| fit::compose(&fit::invert(&a), &fit::compose(m, &a));
    if ps.len() < 3 {
        out.kind = Kind::Lost;
        return out;
    }
    let init: M23 = [1.0, 0.0, tdx as f64, 0.0, 1.0, tdy as f64];
    let prior = vec![1.0; ps.len()];
    let fr = fit::fit_robust(&ps, &qs, &prior, &init, 0.3, 30, ps.len() < 5);
    let thr = (k).max(0.04 * sn);
    let inl = fr.r.iter().filter(|&&r| r < thr).count();
    out.inlier_frac = inl as f64 / textured as f64;
    let ms = to_src(&fr.m);
    out.m = ms;
    let (px, py) = fit::apply(&ms, c[0], c[1]);
    out.d = Some([px - c[0], py - c[1]]);
    let mut rs = fr.r.clone();
    rs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    out.spread = rs[rs.len() / 2] / k;
    out.kind = if ps.len() < p.min_good {
        Kind::Lost
    } else if out.inlier_frac < p.lost_inl {
        Kind::Incoherent
    } else {
        Kind::Ok
    };
    out
}

/// Share of the picture's textured blocks that track a -> b both ways (follow.Clip.trackable): the camera's move is
/// found first; each well-textured block is searched within 6 working px of where it puts it (people moving on their
/// own mostly still track), polished, then searched back (within 1.5 px, Python's forward-backward limit).
pub fn trackable(g0: Gray, g1: Gray) -> f64 {
    let c = trackable_counts(g0, g1);
    if c[0] == 0 {
        0.0
    } else {
        c[4] as f64 / c[0] as f64
    }
}

const TRACK_R: i32 = 6;
const TRACK_TEX: f32 = 0.3;

/// [blocks in the picture, no peak inside the search, peak under 0.5, failed back, good].
pub fn trackable_counts(g0: Gray, g1: Gray) -> [usize; 5] {
    let p = CameraParams::default();
    let fa = camera::prepare_gray(g0.w, g0.h, g0.d, &p);
    let fb = camera::prepare_gray(g1.w, g1.h, g1.d, &p);
    let e = camera::estimate(&fa, &fb, &p);
    let m = e.m_work;
    let (a, b) = (&fa.levels[0], &fb.levels[0]);
    let bs = p.block;
    let mut blocks = camera::lay_blocks(a, bs, p.grid, 2);
    // corners, not edges: only blocks with a good share of the picture's typical texture (an edge's ZNCC peak
    // slides along it)
    let mut ts: Vec<f32> = blocks.iter().map(|b| b.2).collect();
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if !ts.is_empty() {
        let thr = TRACK_TEX * ts[ts.len() * 3 / 4];
        blocks.retain(|b| b.2 >= thr);
    }
    let half = (bs as f64 - 1.0) / 2.0;
    let mut cnt = [0usize; 5];
    for &(x, y, _) in &blocks {
        let (cx, cy) = (x as f64 + half, y as f64 + half);
        let (px, py) = fit::apply(&m, cx, cy);
        if px - half < 0.0 || py - half < 0.0 || px + half > (b.img.w - 1) as f64 || py + half > (b.img.h - 1) as f64 {
            continue;
        }
        cnt[0] += 1;
        let tp = Template::new(&a.img, x, y, bs);
        let (ix, iy) = ((px - cx).round() as i32, (py - cy).round() as i32);
        let sf = surface(&tp, &b.img, ix, iy, TRACK_R);
        let Some((fx, fy, v, inside)) = peak(&sf, TRACK_R) else { continue };
        if !inside {
            cnt[1] += 1;
            continue;
        }
        if v < 0.5 {
            cnt[2] += 1;
            continue;
        }
        let (mut ux, mut uy) = (ix as f32 + fx, iy as f32 + fy);
        if let Some((q, r)) = polish(a, &b.img, x, y, bs, ux, uy, 8, 1.0) {
            ux = q;
            uy = r;
        }
        let (rx, ry) = ((x as f32 + ux).round(), (y as f32 + uy).round());
        if rx < 0.0 || ry < 0.0 || rx as usize + bs >= b.img.w || ry as usize + bs >= b.img.h {
            cnt[3] += 1;
            continue;
        }
        let tb = Template::new(&b.img, rx as usize, ry as usize, bs);
        let (ex, ey) = (x as f32 - rx, y as f32 - ry);
        let (jx, jy) = (ex.round() as i32, ey.round() as i32);
        let sb = surface(&tb, &a.img, jx, jy, 2);
        let Some((gx, gy, vb, inb)) = peak(&sb, 2) else { continue };
        if vb < 0.5 || !inb {
            cnt[3] += 1;
            continue;
        }
        let (mut vx, mut vy) = (jx as f32 + gx, jy as f32 + gy);
        if let Some((q, r)) = polish(b, &a.img, rx as usize, ry as usize, bs, vx, vy, 8, 1.0) {
            vx = q;
            vy = r;
        }
        if ((vx - ex).powi(2) + (vy - ey).powi(2)).sqrt() <= 1.5 {
            cnt[4] += 1;
        } else {
            cnt[3] += 1;
        }
    }
    cnt
}
