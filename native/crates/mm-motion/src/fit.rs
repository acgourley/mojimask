//! Similarity transforms (move, turn, scale) and their robust fit to point matches.
//!
//! A similarity is `x' = a x - b y + tx, y' = b x + a y + ty` (scale `sqrt(a^2 + b^2)`, angle `atan2(b, a)`), kept
//! as the 2x3 row-major matrix `[a, -b, tx, b, a, ty]`, the layout of `mats_v2.npy`.
//!
//! The fit is iteratively reweighted least squares with Tukey's biweight: each round solves the weighted least squares
//! in closed form, measures every match's residual, takes the scale of the residuals (first from their 30th
//! percentile, so most matches may be outliers; then from the weighted inliers; floored at `sigma_min` so sub-pixel
//! agreement isn't over-trusted), and gives each match the weight `(1 - (r / c)^2)^2` for `r < c = 4.685 sigma`,
//! else 0. The first three rounds fit a move only (the start's turn and scale kept), so the weights settle on one
//! group before a turn or a scale can be bent towards another. Matches on things that move on their own (people in front
//! of the camera) get weight 0 once their residual passes c. Deterministic: no sampling, the same input gives the
//! same answer. It needs a starting transform no worse than the outliers are far (the coarse search gives one).

pub type M23 = [f64; 6];

pub const IDENTITY: M23 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

#[inline]
pub fn apply(m: &M23, x: f64, y: f64) -> (f64, f64) {
    (m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5])
}

pub fn scale_of(m: &M23) -> f64 {
    (m[0] * m[4] - m[1] * m[3]).abs().sqrt()
}

/// a after b (apply b first).
pub fn compose(a: &M23, b: &M23) -> M23 {
    [
        a[0] * b[0] + a[1] * b[3],
        a[0] * b[1] + a[1] * b[4],
        a[0] * b[2] + a[1] * b[5] + a[2],
        a[3] * b[0] + a[4] * b[3],
        a[3] * b[1] + a[4] * b[4],
        a[3] * b[2] + a[4] * b[5] + a[5],
    ]
}

pub fn invert(m: &M23) -> M23 {
    let det = m[0] * m[4] - m[1] * m[3];
    let (a, b, c, d) = (m[4] / det, -m[1] / det, -m[3] / det, m[0] / det);
    [a, b, -(a * m[2] + b * m[5]), c, d, -(c * m[2] + d * m[5])]
}

/// The similarity from its parts: scale, angle (radians), translation.
pub fn similarity(scale: f64, angle: f64, tx: f64, ty: f64) -> M23 {
    let (s, c) = angle.sin_cos();
    [scale * c, -scale * s, tx, scale * s, scale * c, ty]
}

/// Weighted least-squares similarity taking p to q. None if the weights or the spread of p are degenerate.
pub fn fit_weighted(p: &[[f64; 2]], q: &[[f64; 2]], w: &[f64]) -> Option<M23> {
    let sw: f64 = w.iter().sum();
    if sw <= 1e-12 {
        return None;
    }
    let (mut px, mut py, mut qx, mut qy) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..p.len() {
        px += w[i] * p[i][0];
        py += w[i] * p[i][1];
        qx += w[i] * q[i][0];
        qy += w[i] * q[i][1];
    }
    px /= sw;
    py /= sw;
    qx /= sw;
    qy /= sw;
    let (mut den, mut na, mut nb) = (0.0, 0.0, 0.0);
    for i in 0..p.len() {
        let (x, y) = (p[i][0] - px, p[i][1] - py);
        let (u, v) = (q[i][0] - qx, q[i][1] - qy);
        den += w[i] * (x * x + y * y);
        na += w[i] * (x * u + y * v);
        nb += w[i] * (x * v - y * u);
    }
    if den <= 1e-9 {
        // one point (or all at one place): a pure move
        return Some([1.0, 0.0, qx - px, 0.0, 1.0, qy - py]);
    }
    let a = na / den;
    let b = nb / den;
    Some([a, -b, qx - (a * px - b * py), b, a, qy - (b * px + a * py)])
}

/// A pure translation fit (weighted mean of the moves).
pub fn fit_translation(p: &[[f64; 2]], q: &[[f64; 2]], w: &[f64]) -> Option<M23> {
    let sw: f64 = w.iter().sum();
    if sw <= 1e-12 {
        return None;
    }
    let (mut dx, mut dy) = (0.0, 0.0);
    for i in 0..p.len() {
        dx += w[i] * (q[i][0] - p[i][0]);
        dy += w[i] * (q[i][1] - p[i][1]);
    }
    Some([1.0, 0.0, dx / sw, 0.0, 1.0, dy / sw])
}

const WARM: usize = 3;

/// The best move with m's turn and scale kept.
pub fn fit_translation_keep(p: &[[f64; 2]], q: &[[f64; 2]], w: &[f64], m: &M23) -> Option<M23> {
    let sw: f64 = w.iter().sum();
    if sw <= 1e-12 {
        return None;
    }
    let (mut tx, mut ty) = (0.0, 0.0);
    for i in 0..p.len() {
        tx += w[i] * (q[i][0] - (m[0] * p[i][0] + m[1] * p[i][1]));
        ty += w[i] * (q[i][1] - (m[3] * p[i][0] + m[4] * p[i][1]));
    }
    Some([m[0], m[1], tx / sw, m[3], m[4], ty / sw])
}

pub struct RobustFit {
    pub m: M23,
    /// the final Tukey weight of each match (0 = rejected)
    pub w: Vec<f64>,
    /// each match's residual under m
    pub r: Vec<f64>,
    pub sigma: f64,
}

pub fn residuals(m: &M23, p: &[[f64; 2]], q: &[[f64; 2]]) -> Vec<f64> {
    p.iter()
        .zip(q)
        .map(|(a, b)| {
            let (x, y) = apply(m, a[0], a[1]);
            ((x - b[0]).powi(2) + (y - b[1]).powi(2)).sqrt()
        })
        .collect()
}

/// IRLS with Tukey weights from `init`. `prior` weights each match (texture, match quality); `translation_only`
/// fits a pure move (used when the matches span too little to pin a turn or a scale).
pub fn fit_robust(
    p: &[[f64; 2]],
    q: &[[f64; 2]],
    prior: &[f64],
    init: &M23,
    sigma_min: f64,
    iters: usize,
    translation_only: bool,
) -> RobustFit {
    let mut m = *init;
    let mut w = prior.to_vec();
    let mut sigma = sigma_min;
    for it in 0..iters {
        let r = residuals(&m, p, q);
        // the scale of the inliers (per axis): first from the 30th percentile of all residuals (a 2-D residual's
        // 30th percentile is 0.845 sigma; a low quantile so most matches may be outliers), then from the weighted
        // residuals of the matches still in play
        if it == 0 {
            let mut live: Vec<f64> = r.iter().zip(prior).filter(|(_, &pw)| pw > 0.0).map(|(&x, _)| x).collect();
            live.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q30 = if live.is_empty() { 0.0 } else { live[(live.len() * 3) / 10] };
            sigma = (q30 / 0.845).max(sigma_min);
        } else {
            let (mut num, mut den) = (0.0, 0.0);
            for i in 0..p.len() {
                num += w[i] * r[i] * r[i];
                den += w[i];
            }
            if den > 0.0 {
                sigma = (num / (2.0 * den)).sqrt().max(sigma_min);
            }
        }
        let c = 4.685 * sigma;
        for i in 0..p.len() {
            let u = r[i] / c;
            w[i] = if u < 1.0 { prior[i] * (1.0 - u * u).powi(2) } else { 0.0 };
        }
        // the first rounds move only (keeping the start's turn and scale): the weights settle on one group before
        // a turn or a scale may be fitted from it (with few matches close together, a free similarity can bend
        // towards outliers)
        let next = if translation_only || it < WARM { fit_translation_keep(p, q, &w, &m) } else { fit_weighted(p, q, &w) };
        match next {
            Some(n) => {
                let change = (n[2] - m[2]).abs() + (n[5] - m[5]).abs() + (n[0] - m[0]).abs() * 100.0 + (n[3] - m[3]).abs() * 100.0;
                m = n;
                if change < 1e-4 && it >= WARM {
                    break;
                }
            }
            None => break,
        }
    }
    let r = residuals(&m, p, q);
    RobustFit { m, w, r, sigma }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_similarity_with_outliers() {
        let truth = similarity(1.02, 0.03, 5.0, -3.0);
        let mut p = vec![];
        let mut q = vec![];
        for i in 0..10 {
            for j in 0..8 {
                let (x, y) = (i as f64 * 40.0, j as f64 * 40.0);
                p.push([x, y]);
                let (u, v) = apply(&truth, x, y);
                // a third of them move on their own
                if (i + j) % 3 == 0 {
                    q.push([u + 25.0, v - 12.0]);
                } else {
                    q.push([u, v]);
                }
            }
        }
        let prior = vec![1.0; p.len()];
        // started from the right move only: the turn and scale are found from the matches
        let f = fit_robust(&p, &q, &prior, &[1.0, 0.0, 5.0, 0.0, 1.0, -3.0], 0.1, 30, false);
        for k in 0..6 {
            assert!((f.m[k] - truth[k]).abs() < 1e-6, "{:?} vs {:?}", f.m, truth);
        }
    }

    #[test]
    fn invert_compose() {
        let m = similarity(0.9, -0.2, 3.0, 7.0);
        let i = compose(&invert(&m), &m);
        for k in 0..6 {
            assert!((i[k] - IDENTITY[k]).abs() < 1e-12);
        }
    }
}
