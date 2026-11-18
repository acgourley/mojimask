//! Camera motion: one similarity per pair of consecutive frames, mapping frame t-1's pixels to frame t's (full-
//! resolution source pixels, the convention and units of `mats_v2.npy`; `track.camera_motion` in Python).
//!
//! **The domain.** Handheld phone video at parties: the camera drifts, turns and pans (sometimes whips), people move
//! in front of it (often most of the middle of the picture), the light is low (noise, auto exposure), fast moves
//! blur, and the rolling shutter skews a fast pan. We need the background's move only, to within a fraction of a
//! pixel, and a clear "don't know" when the frame can't say.
//!
//! **The design: coarse-to-fine block matching with a robust fit.** No corners, no RANSAC, no random numbers.
//! 1. Both frames go to grey (`cv2`'s BGR->GRAY weights, mm-vision), are averaged down to a working size (long side
//!    at most `work_long`, 960 px; 643 x 476 jazz stays as is) and built into a pyramid (binomial filter, halving)
//!    down to a coarsest level of about 48-96 px.
//! 2. **Coarse vote** (coarsest level): every textured block's ZNCC surface over all moves up to a quarter of the
//!    picture is summed. Each block votes for every move it fits, so the reach is large (a whip pan of a quarter of
//!    the frame per frame) and the background, the largest thing moving together, makes a strong peak even when
//!    people cover much of the picture. The strongest local peaks (up to 8, each at least 0.3 of the best) are
//!    candidates.
//! 3. **Refine** (every level, coarse to fine): a grid of blocks (up to 24 x 16; in each cell the best-textured of
//!    five positions, by the structure tensor's smaller eigenvalue) is searched within a few pixels of where the
//!    current similarity puts it (ZNCC, which ignores brightness and contrast changes; 3 px, 6 px on the level under
//!    the coarsest, where a turn or zoom not yet fitted shows at the edges), then a similarity is fitted by IRLS
//!    with Tukey weights (fit.rs) starting from the current one. The coarsest level fits a move only; a turn or a
//!    scale needs 12 matches and may change by at most 8% / 0.1 rad per level. Blocks on people moving on their own
//!    end with weight 0. On the finest level each block's move is polished by Lucas-Kanade before the last fit.
//!    **Candidates** are refined down to level 2 in vote order, and the one most blocks agree with there goes on; the
//!    search stops at the first candidate 60% of the blocks agree with (most real frames: one candidate). A large
//!    patch moving as one (someone close to the camera) can out-vote a background that turns or zooms, whose blocks'
//!    moves spread over several translations; the background's similarity still gathers the most blocks.
//! 4. **Trust**: the share of textured blocks within 0.5% of the picture's width of the fit (Python's RANSAC
//!    threshold, 2 px at half resolution, is 0.6% of jazz's width). Under `min_inlier_frac` (0.4, Python's own
//!    guard), or fewer than 8 such blocks, the pair is "unknown" and production gives the identity, as Python does
//!    ("a hand filling the frame").
//! 5. Production then keeps Python's other guard: scale is clipped to 0.97-1.03 per frame.
//!
//! **Why not dense direct alignment** (Gauss-Newton on every pixel): it needs a start within a pixel or two of the
//! answer at each level and gives one vote per pixel, so a large foreground mover pulls it; blocks with a robust fit
//! keep each region's evidence separate and reject it whole. **Why not sparse corners + LK** (Python's): corner
//! responses vanish on blurred or dark frames, LK's reach is bounded by its window x pyramid, and RANSAC is random.
//! **Why CPU first:** at a 960 px working size a pair costs about 5 ms (blocks searched on up to 4 threads) and
//! preparing a 4K frame (grey, averaged down) 10 ms; decoding the frame costs more. The block search is regular and
//! moves to WGSL if the scan needs it.
//!
//! **Rolling shutter** isn't modelled (a similarity can't skew); a skewed frame's blocks disagree a little and the
//! fit's sigma grows, which the trust measure absorbs.
//!
//! Evidence: tools/motion/known_motion.py (jazz frames moved by known transforms, with blur, noise, JPEG and moving
//! foreground patches), tools/motion/jazz_mats.py (agreement with Python's mats on jazz and jazz_4k) and the
//! decisions benchmark (tools/parity/motion_bench.py), 2026-11-17:
//! - known motion, 630 pairs (643 x 476 and 1280 x 720; 5 moves x 7 troubles x 0/2/4 people): what mats_v2.npy
//!   would hold, mean endpoint error 7.7 px and 25% failures (over 1 px or 0.5% of the width), Python 13.9 px and 34%;
//!   the fit alone, without the guards, 0.02 px vs Python's 0.06 px with nobody in front, failing 10% vs 9% overall
//!   (ours fails most on 4 people covering half the frame while the camera turns 2-5 degrees; it then says
//!   "unknown", 16% of pairs vs Python's 29%).
//! - jazz: within 0.31 px (mean; max 0.64) of Python's mats; lines frames up slightly better (median grey difference
//!   3.45 vs 3.50, identity 6.49). jazz_4k: Python trusts only 10 of 126 pairs (identity elsewhere), ours all 126
//!   (3.28 vs 6.16, identity 6.37).

use crate::fit::{self, M23};
use crate::img::{pyramid, Img, Level};
use crate::matching::{peak, polish, surface, Template};

#[derive(Clone, Debug)]
pub struct CameraParams {
    /// working image: long side at most this many pixels (integer box averaging)
    pub work_long: usize,
    /// coarsest pyramid level: long side at least this
    pub min_coarse: usize,
    pub max_levels: usize,
    /// at most this many block cells across and down
    pub grid: (usize, usize),
    pub block: usize,
    pub coarse_block: usize,
    /// coarse search reach, as a share of the coarsest level's long side
    pub coarse_reach: f64,
    pub refine_r: i32,
    /// coarse vote peaks refined as candidates
    pub candidates: usize,
    pub min_zncc: f32,
    /// residual under which a block agrees with the fit, as a share of the working long side (Python's RANSAC
    /// threshold, 2 px at half resolution, is 0.6% of jazz's width)
    pub inlier_rel: f64,
    pub min_inlier_frac: f64,
    pub min_inliers: usize,
    /// production: per-frame scale kept within this range (Python: 0.97-1.03)
    pub clip_scale: Option<(f64, f64)>,
}

impl Default for CameraParams {
    fn default() -> Self {
        CameraParams {
            work_long: 960,
            min_coarse: 48,
            max_levels: 6,
            grid: (24, 16),
            block: 12,
            coarse_block: 8,
            coarse_reach: 0.25,
            refine_r: 3,
            candidates: 8,
            min_zncc: 0.5,
            inlier_rel: 0.005,
            min_inlier_frac: 0.4,
            min_inliers: 8,
            clip_scale: Some((0.97, 1.03)),
        }
    }
}

/// A frame ready for matching: its working pyramid and how it maps to the source pixels.
pub struct CamFrame {
    pub levels: Vec<Level>,
    /// box factor from source to working pixels
    pub f: usize,
    pub w: usize,
    pub h: usize,
}

pub fn prepare_gray(w: usize, h: usize, gray: &[u8], p: &CameraParams) -> CamFrame {
    let f = w.max(h).div_ceil(p.work_long).max(1);
    let base = if f == 1 { Img::from_u8(w, h, gray) } else { box_down_u8(w, h, gray, f) };
    CamFrame { levels: pyramid(base, p.min_coarse, p.max_levels), f, w, h }
}

/// Grey from BGR (cv2's weights, bit-exact with mm-vision) and prepare.
pub fn prepare_bgr(bgr: &mm_vision::Mat<u8>, p: &CameraParams) -> CamFrame {
    let g = mm_vision::bgr2gray(bgr);
    prepare_gray(g.cols, g.rows, &g.data, p)
}

fn box_down_u8(w: usize, h: usize, px: &[u8], f: usize) -> Img {
    let ow = w / f;
    let oh = h / f;
    let mut out = Img::new(ow, oh);
    let inv = 1.0 / (f * f) as f32;
    let mut acc = vec![0u32; ow];
    for oy in 0..oh {
        acc.iter_mut().for_each(|a| *a = 0);
        for y in oy * f..(oy + 1) * f {
            let row = &px[y * w..y * w + ow * f];
            for (ox, a) in acc.iter_mut().enumerate() {
                *a += row[ox * f..(ox + 1) * f].iter().map(|&v| v as u32).sum::<u32>();
            }
        }
        for ox in 0..ow {
            out.d[oy * ow + ox] = acc[ox] as f32 * inv;
        }
    }
    out
}

#[derive(Clone, Debug, Default)]
pub struct CamEstimate {
    /// the fitted similarity, source pixels, frame a -> frame b (identity if nothing could be fitted)
    pub m: M23,
    /// the same in working pixels (level 0 of the pyramids)
    pub m_work: M23,
    /// trusted (enough blocks agree)
    pub ok: bool,
    pub inlier_frac: f64,
    /// textured blocks that stayed in the picture, and how many agree with m
    pub blocks: usize,
    pub inliers: usize,
    /// the fit's residual scale, working pixels
    pub sigma: f64,
    /// the coarse vote's move, source pixels, and its peak (mean ZNCC)
    pub coarse: [f64; 2],
    pub vote: f32,
}

/// Blocks laid on a level: in each grid cell the best-textured of five positions, if it has texture at all.
pub fn lay_blocks(lv: &Level, b: usize, grid: (usize, usize), margin: usize) -> Vec<(usize, usize, f32)> {
    let (w, h) = (lv.img.w, lv.img.h);
    if w < b + 2 * margin + 1 || h < b + 2 * margin + 1 {
        return vec![];
    }
    let cols = grid.0.min((w - 2 * margin) / (b + 2)).max(1);
    let rows = grid.1.min((h - 2 * margin) / (b + 2)).max(1);
    let cw = (w - 2 * margin) as f32 / cols as f32;
    let ch = (h - 2 * margin) as f32 / rows as f32;
    let mut out = vec![];
    for r in 0..rows {
        for c in 0..cols {
            let cx = margin as f32 + (c as f32 + 0.5) * cw;
            let cy = margin as f32 + (r as f32 + 0.5) * ch;
            let mut best = (0usize, 0usize, -1f32);
            for (ox, oy) in [(0.0, 0.0), (-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)] {
                let x = (cx + ox * cw - b as f32 / 2.0).round().clamp(margin as f32, (w - b - margin) as f32) as usize;
                let y = (cy + oy * ch - b as f32 / 2.0).round().clamp(margin as f32, (h - b - margin) as f32) as usize;
                let t = lv.texture(x, y, b);
                if t > best.2 {
                    best = (x, y, t);
                }
            }
            out.push(best);
        }
    }
    // texture: an absolute floor (noise) and a share of the picture's typical texture
    let mut ts: Vec<f32> = out.iter().map(|b| b.2).collect();
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p75 = ts[(ts.len() * 3) / 4];
    let thr = (0.05 * p75).max(0.25);
    out.retain(|b| b.2 >= thr);
    out
}

/// The coarse vote: every textured block's ZNCC surface over moves up to r, summed. Its strongest local peaks (up
/// to `k`, each at least half the best) are the candidate moves, best first: (dx, dy, vote).
fn coarse_vote(a: &Level, b: &Level, bs: usize, grid: (usize, usize), r: i32, k: usize) -> Vec<(f32, f32, f32)> {
    let blocks = lay_blocks(a, bs, grid, 1);
    if blocks.is_empty() {
        return vec![];
    }
    let n = (2 * r + 1) as usize;
    let mut acc = vec![0f32; n * n];
    let mut cnt = vec![0f32; n * n];
    for &(x, y, _) in &blocks {
        let tp = Template::new(&a.img, x, y, bs);
        let s = surface(&tp, &b.img, 0, 0, r);
        for k in 0..n * n {
            if s[k].is_finite() {
                acc[k] += s[k].max(0.0);
                cnt[k] += 1.0;
            }
        }
    }
    let floor = 0.5 * blocks.len() as f32;
    let vote: Vec<f32> = acc.iter().zip(&cnt).map(|(&a, &c)| if c > 0.0 { a / c.max(floor) } else { f32::NAN }).collect();
    // local maxima (over a 5 x 5 neighbourhood), strongest first
    let at = |i: i32, j: i32| -> f32 {
        if i < 0 || j < 0 || i >= n as i32 || j >= n as i32 { f32::NAN } else { vote[j as usize * n + i as usize] }
    };
    let mut peaks = vec![];
    for j in 0..n as i32 {
        for i in 0..n as i32 {
            let v = at(i, j);
            if !v.is_finite() {
                continue;
            }
            let mut is_max = true;
            'nb: for dj in -2..=2 {
                for di in -2..=2 {
                    let u = at(i + di, j + dj);
                    if (di, dj) != (0, 0) && u.is_finite() && (u > v || (u == v && (dj, di) < (0, 0))) {
                        is_max = false;
                        break 'nb;
                    }
                }
            }
            if is_max {
                peaks.push((i, j, v));
            }
        }
    }
    peaks.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap().then((a.1, a.0).cmp(&(b.1, b.0))));
    let best = peaks.first().map(|p| p.2).unwrap_or(0.0);
    let sub = |a: f32, c: f32, b: f32| -> f32 {
        let den = a - 2.0 * c + b;
        if den.abs() < 1e-9 || !a.is_finite() || !b.is_finite() { 0.0 } else { (0.5 * (a - b) / den).clamp(-0.5, 0.5) }
    };
    peaks
        .into_iter()
        .filter(|p| p.2 >= 0.3 * best)
        .take(k)
        .map(|(i, j, v)| {
            let fx = sub(at(i - 1, j), v, at(i + 1, j));
            let fy = sub(at(i, j - 1), v, at(i, j + 1));
            (i as f32 - r as f32 + fx, j as f32 - r as f32 + fy, v)
        })
        .collect()
}

struct LevelFit {
    m: M23,
    blocks: usize,
    inliers: usize,
    sigma: f64,
}

/// Search every block near where m puts it, fit a similarity robustly from m.
/// f over the items on a few threads, results in order (so the answer never depends on the threads).
pub fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let n = items.len();
    let threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1).min(PAR_THREADS).min(n / 48).max(1);
    if threads <= 1 {
        return items.iter().map(&f).collect();
    }
    let chunk = n.div_ceil(threads);
    let f = &f;
    std::thread::scope(|sc| {
        let hs: Vec<_> = items.chunks(chunk).map(|c| sc.spawn(move || c.iter().map(f).collect::<Vec<R>>())).collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

/// Threads for one pair's block search (the scan runs other work beside it).
pub const PAR_THREADS: usize = 4;

/// a candidate most blocks agree with (on the level candidates are compared) ends the search
const SURE_SHARE: f64 = 0.6;

const MIN_SIM: usize = 12;
const MAX_STEP_SCALE: f64 = 0.08;
const MAX_STEP_TURN: f64 = 0.1;

fn refine_level(a: &Level, b: &Level, m: &M23, p: &CameraParams, r: i32, finest: bool, move_only: bool) -> LevelFit {
    let bs = p.block;
    let blocks = lay_blocks(a, bs, p.grid, 2);
    let half = (bs as f64 - 1.0) / 2.0;
    // each block: None if it left b's picture where m puts it, Some(None) if no good match, Some(Some(move))
    let one = |&(x, y, _): &(usize, usize, f32)| -> Option<Option<(f64, f64, f64, f64)>> {
        let (cx, cy) = (x as f64 + half, y as f64 + half);
        let (px, py) = fit::apply(m, cx, cy);
        let (dx, dy) = (px - cx, py - cy);
        if px - half < 0.0 || py - half < 0.0 || px + half > (b.img.w - 1) as f64 || py + half > (b.img.h - 1) as f64 {
            return None;
        }
        let tp = Template::new(&a.img, x, y, bs);
        let (ix, iy) = (dx.round() as i32, dy.round() as i32);
        let s = surface(&tp, &b.img, ix, iy, r);
        let Some((fx, fy, v, inside)) = peak(&s, r) else { return Some(None) };
        if v < p.min_zncc || !inside {
            return Some(None);
        }
        let (mut mx, mut my) = (ix as f32 + fx, iy as f32 + fy);
        if finest {
            if let Some((ux, uy)) = polish(a, &b.img, x, y, bs, mx, my, 8, 1.0) {
                mx = ux;
                my = uy;
            }
        }
        Some(Some((cx, cy, cx + mx as f64, cy + my as f64)))
    };
    let found = par_map(&blocks, one);
    let (mut ps, mut qs, mut pw) = (vec![], vec![], vec![]);
    let mut candidates = 0usize;
    for f in found.into_iter().flatten() {
        candidates += 1;
        if let Some((cx, cy, qx, qy)) = f {
            ps.push([cx, cy]);
            qs.push([qx, qy]);
            pw.push(1.0);
        }
    }
    if ps.len() < 3 {
        return LevelFit { m: *m, blocks: candidates, inliers: 0, sigma: f64::NAN };
    }
    // A turn or a scale only from enough matches (MIN_SIM), and never a jump in them from one level to the next
    // (MAX_STEP_SCALE, MAX_STEP_TURN): a dozen blocks on a coarse level, most of them on people, can bend a free
    // similarity into a zoom that isn't there.
    let sig = if finest { 0.2 } else { 0.35 };
    let mut fitr = fit::fit_robust(&ps, &qs, &pw, m, sig, 30, ps.len() < MIN_SIM || move_only);
    let (s0, s1) = (fit::scale_of(m), fit::scale_of(&fitr.m));
    let turn = (fitr.m[3].atan2(fitr.m[0]) - m[3].atan2(m[0])).abs();
    if (s1 / s0 - 1.0).abs() > MAX_STEP_SCALE || turn > MAX_STEP_TURN {
        fitr = fit::fit_robust(&ps, &qs, &pw, m, sig, 30, true);
    }
    let thr = p.inlier_rel * a.img.w.max(a.img.h) as f64;
    let inliers = fitr.r.iter().filter(|&&r| r < thr).count();
    LevelFit { m: fitr.m, blocks: candidates, inliers, sigma: fitr.sigma }
}

/// The working-pixel similarity -> source pixels.
fn to_source(m: &M23, f: usize) -> M23 {
    let f = f as f64;
    let o = (f - 1.0) / 2.0;
    [m[0], m[1], f * m[2] + o - (m[0] * o + m[1] * o), m[3], m[4], f * m[5] + o - (m[3] * o + m[4] * o)]
}

/// Levels `from` down to `to` (inclusive), starting from m on level `from`; the first level is searched twice (its
/// start is only a move). Returns the fit on level `to` (in level-`to` pixels) and its numbers.
fn descend(a: &CamFrame, b: &CamFrame, p: &CameraParams, mut m: M23, from: usize, to: usize) -> (M23, LevelFit) {
    let top = a.levels.len().min(b.levels.len()) - 1;
    let mut last = None;
    for l in (to..=from).rev() {
        let passes = if l == top { 2 } else { 1 };
        for _ in 0..passes {
            // the level under the coarsest searches wider: the coarsest fitted a move only, so a turn or a zoom is
            // still unaccounted for at the edges there (5 degrees: 5 px at 120 px wide)
            let r = if l == from { 3 } else if l + 1 == top { 6 } else { p.refine_r };
            // the coarsest level fits a move only (its blocks are few and wide)
            let lf = refine_level(&a.levels[l], &b.levels[l], &m, p, r, l == 0, l == a.levels.len().min(b.levels.len()) - 1);
            m = lf.m;
            last = Some(lf);
        }
        if l > to {
            m[2] *= 2.0;
            m[5] *= 2.0;
        }
    }
    (m, last.unwrap())
}

pub fn estimate(a: &CamFrame, b: &CamFrame, p: &CameraParams) -> CamEstimate {
    let nl = a.levels.len().min(b.levels.len());
    let top = nl - 1;
    let la = &a.levels[top];
    let r = ((p.coarse_reach * la.img.w.max(la.img.h) as f64).round() as i32).max(4);
    let mut out = CamEstimate { m: fit::IDENTITY, m_work: fit::IDENTITY, ..Default::default() };
    let cands = coarse_vote(la, &b.levels[top], p.coarse_block, p.grid, r, p.candidates);
    if cands.is_empty() {
        return out;
    }
    // Each candidate move is refined down to level 2; the one most blocks agree with there goes on to level 0. (A
    // large patch moving as one -- someone close to the camera -- can out-vote a background that turns or zooms,
    // whose blocks' moves differ; the background's similarity still gathers the most blocks.)
    let mid = if top >= 3 { 2 } else { top.saturating_sub(1) };
    let mut best: Option<(usize, M23, LevelFit, (f32, f32, f32))> = None;
    for &(dx, dy, v) in &cands {
        let (m, lf) = descend(a, b, p, [1.0, 0.0, dx as f64, 0.0, 1.0, dy as f64], top, mid);
        let share = if lf.blocks > 0 { lf.inliers as f64 / lf.blocks as f64 } else { 0.0 };
        if best.as_ref().map(|b| lf.inliers > b.0).unwrap_or(true) {
            best = Some((lf.inliers, m, lf, (dx, dy, v)));
        }
        // most frames: the strongest peak is the background and most blocks agree; the others aren't tried
        if share >= SURE_SHARE {
            break;
        }
    }
    let (_, mut m, mut lf, (dx, dy, v)) = best.unwrap();
    let k = (1usize << top) as f64;
    out.coarse = [dx as f64 * k * a.f as f64, dy as f64 * k * a.f as f64];
    out.vote = v;
    if mid > 0 {
        m[2] *= 2.0;
        m[5] *= 2.0;
        let (m0, lf0) = descend(a, b, p, m, mid - 1, 0);
        m = m0;
        lf = lf0;
    }
    out.m_work = m;
    out.m = to_source(&m, a.f);
    out.blocks = lf.blocks;
    out.inliers = lf.inliers;
    out.sigma = lf.sigma;
    out.inlier_frac = if lf.blocks > 0 { lf.inliers as f64 / lf.blocks as f64 } else { 0.0 };
    out.ok = out.inlier_frac >= p.min_inlier_frac && out.inliers >= p.min_inliers;
    out
}

/// What goes in mats_v2.npy: the identity when the pair can't be trusted, else the fit with its scale clipped.
pub fn production(e: &CamEstimate, p: &CameraParams) -> M23 {
    if !e.ok {
        return fit::IDENTITY;
    }
    let mut m = e.m;
    if let Some((lo, hi)) = p.clip_scale {
        let s = fit::scale_of(&m);
        let k = s.clamp(lo, hi) / s;
        for i in [0, 1, 3, 4] {
            m[i] *= k;
        }
    }
    m
}

/// Frames in, `mats_v2.npy` rows out (the first is the identity), keeping one prepared frame.
pub struct CameraTracker {
    pub p: CameraParams,
    prev: Option<CamFrame>,
}

impl CameraTracker {
    pub fn new(p: CameraParams) -> Self {
        CameraTracker { p, prev: None }
    }

    /// The row for this frame (identity for the first) and the estimate behind it.
    pub fn push(&mut self, frame: CamFrame) -> (M23, Option<CamEstimate>) {
        let r = match &self.prev {
            None => (fit::IDENTITY, None),
            Some(a) => {
                let e = estimate(a, &frame, &self.p);
                (production(&e, &self.p), Some(e))
            }
        };
        self.prev = Some(frame);
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A picture with texture at several scales and no repeats: hashed noise, blurred at two sizes.
    fn texture(w: usize, h: usize) -> Vec<u8> {
        let noise: Vec<u8> = (0..w * h)
            .map(|i| {
                let mut z = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
                z ^= z >> 29;
                z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                (z >> 56) as u8
            })
            .collect();
        let m = mm_vision::Mat::from_vec(h, w, 1, noise);
        let fine = mm_vision::gaussian_blur_u8(&m, 0, 0, 1.5, 1.5);
        let coarse = mm_vision::gaussian_blur_u8(&m, 0, 0, 6.0, 6.0);
        fine.data
            .iter()
            .zip(&coarse.data)
            .map(|(&f, &c)| (128.0 + (f as f64 - 128.0) * 3.0 + (c as f64 - 128.0) * 12.0).clamp(0.0, 255.0) as u8)
            .collect()
    }

    #[test]
    fn recovers_known_similarity() {
        let (w, h) = (480, 360);
        let src = mm_vision::Mat::from_vec(h, w, 1, texture(w, h));
        let truth = fit::compose(&[1.0, 0.0, 9.5, 0.0, 1.0, -4.25], &{
            // 1.5 degrees and 1% about the middle
            let r = fit::similarity(1.01, 0.026, 0.0, 0.0);
            let (qx, qy) = fit::apply(&r, 240.0, 180.0);
            [r[0], r[1], 240.0 - qx, r[3], r[4], 180.0 - qy]
        });
        let moved = mm_vision::warp_affine_u8(&src, &truth, w, h, false, mm_vision::Border::Replicate, [0.0; 4]);
        let p = CameraParams::default();
        let a = prepare_gray(w, h, &src.data, &p);
        let b = prepare_gray(w, h, &moved.data, &p);
        let e = estimate(&a, &b, &p);
        assert!(e.ok, "{e:?}");
        for (x, y) in [(40.0, 40.0), (440.0, 40.0), (240.0, 180.0), (40.0, 320.0), (440.0, 320.0)] {
            let (u, v) = fit::apply(&e.m, x, y);
            let (tu, tv) = fit::apply(&truth, x, y);
            assert!(((u - tu).powi(2) + (v - tv).powi(2)).sqrt() < 0.1, "at ({x}, {y}): {:?} vs {:?}", e.m, truth);
        }
    }
}
