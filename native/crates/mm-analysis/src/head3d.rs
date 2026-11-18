//! Full 3-D head pose (experiment): yaw, pitch and roll from the five head keypoints of a COCO-17 skeleton (nose,
//! eyes, ears) that the scan already saves in dets/pose_full.json. No new model.
//!
//! head_pose.rs measures only roll (the eye line) and a 2-D "how far the nose sits along the eye line" yaw, which
//! is all a flat emoji card needs. A 3-D emoji needs the whole rotation, so here a rigid head model (MODEL, in
//! interocular distances) is fitted to the keypoints under scaled-orthographic projection:
//!   p_i = s * (R X_i).xy + t,   R = Rz(roll) Rx(pitch) Ry(yaw)
//! by damped Gauss-Newton on (yaw, pitch, roll, ln s, tx, ty), each keypoint weighted by its confidence, with a weak
//! pull of every angle to neutral (five points, two of them often hidden, leave pitch loosely held). Three starts
//! (head turned left, front, right) and the lowest cost wins.
//!
//! Conventions (camera space, as head_pose.rs): x image right, y image down, z towards the camera.
//!   yaw   + = face points towards image right
//!   pitch + = face points down
//!   roll  + = clockwise in the image (head_pose.rs's roll: the angle of the eye line)
//! The model's +x side is the subject's left (COCO "left eye"), seen on image right when facing the camera.
//!
//! The head model and the rotation are mm_core::orb's (shared with the drawing). [`track_poses`] smooths the fits
//! along each tracked head for the "orb" cover style (developer only, an experiment towards 3-D emoji);
//! mm-records' pose-orbs tool draws them on a project's frames to judge them by eye.
use crate::pose_util::{self as pu, LEAR, LEYE, NOSE, REAR, REYE};
pub use mm_core::orb::{apply, apply_t, mul, rot, Mat3, HEAD_R, MODEL};
use std::collections::BTreeMap;

/// keypoints used in the fit: below this confidence a point is left out
pub const FIT_VIS: f64 = 0.3;
/// prior on each angle [yaw, pitch, roll]: residual = angle / PRIOR_RAD (radians), next to keypoint residuals in
/// interocular units. Weak on yaw and roll (the eyes, nose and ears hold them); firmer on pitch, which rests on how
/// far the nose drops below the eyes and is easily traded for scale.
pub const PRIOR_RAD: [f64; 3] = [8.0, 3.0, 8.0];

#[derive(Clone, Copy, Debug)]
pub struct Pose3 {
    /// radians
    pub yaw: f64,
    pub pitch: f64,
    pub roll: f64,
    /// pixels per interocular distance
    pub s: f64,
    /// image position of the head centre
    pub t: [f64; 2],
    /// rms keypoint error, in interocular distances
    pub err: f64,
    /// keypoints used
    pub n: usize,
}

impl Pose3 {
    pub fn r(&self) -> Mat3 {
        rot(self.yaw, self.pitch, self.roll)
    }
    /// image position and depth (pixels, + towards the camera) of a head-model point
    pub fn project(&self, x: [f64; 3]) -> [f64; 3] {
        let v = apply(&self.r(), x);
        [self.s * v[0] + self.t[0], self.s * v[1] + self.t[1], self.s * v[2]]
    }
}

/// q = [yaw, pitch, roll, ln s, tx, ty]; residuals in interocular units (pixels / s0), then the priors
fn residuals(q: &[f64; 6], obs: &[(usize, [f64; 2], f64)], s0: f64, out: &mut Vec<f64>) {
    out.clear();
    let r = rot(q[0], q[1], q[2]);
    let s = q[3].exp();
    for &(i, p, w) in obs {
        let v = apply(&r, MODEL[i]);
        let w = w.sqrt();
        out.push(w * (s * v[0] + q[4] - p[0]) / s0);
        out.push(w * (s * v[1] + q[5] - p[1]) / s0);
    }
    for k in 0..3 {
        out.push(q[k] / PRIOR_RAD[k]);
    }
}

fn solve6(mut a: [[f64; 6]; 6], mut b: [f64; 6]) -> Option<[f64; 6]> {
    for c in 0..6 {
        let p = (c..6).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-12 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..6 {
            let f = a[r][c] / a[c][c];
            for k in c..6 {
                a[r][k] -= f * a[c][k];
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = [0.0; 6];
    for c in (0..6).rev() {
        x[c] = (b[c] - (c + 1..6).map(|k| a[c][k] * x[k]).sum::<f64>()) / a[c][c];
    }
    Some(x)
}

fn cost(q: &[f64; 6], obs: &[(usize, [f64; 2], f64)], s0: f64, buf: &mut Vec<f64>) -> f64 {
    residuals(q, obs, s0, buf);
    buf.iter().map(|r| r * r).sum()
}

fn refine(mut q: [f64; 6], obs: &[(usize, [f64; 2], f64)], s0: f64) -> ([f64; 6], f64) {
    let (mut r0, mut r1) = (Vec::new(), Vec::new());
    let mut c = cost(&q, obs, s0, &mut r0);
    let mut lambda = 1e-3;
    for _ in 0..60 {
        residuals(&q, obs, s0, &mut r0);
        // numeric jacobian (angles in radians, ln s, t in pixels: step sizes to match)
        let steps = [1e-5, 1e-5, 1e-5, 1e-6, 1e-4 * s0, 1e-4 * s0];
        let m = r0.len();
        let mut jac = vec![[0.0; 6]; m];
        for k in 0..6 {
            let mut qk = q;
            qk[k] += steps[k];
            residuals(&qk, obs, s0, &mut r1);
            for i in 0..m {
                jac[i][k] = (r1[i] - r0[i]) / steps[k];
            }
        }
        let mut jtj = [[0.0; 6]; 6];
        let mut jtr = [0.0; 6];
        for i in 0..m {
            for a in 0..6 {
                jtr[a] -= jac[i][a] * r0[i];
                for b in 0..6 {
                    jtj[a][b] += jac[i][a] * jac[i][b];
                }
            }
        }
        let mut improved = false;
        for _ in 0..8 {
            let mut a = jtj;
            for d in 0..6 {
                a[d][d] += lambda * (jtj[d][d] + 1e-9);
            }
            let Some(dq) = solve6(a, jtr) else { break };
            let mut qn = q;
            for k in 0..6 {
                qn[k] += dq[k];
            }
            let cn = cost(&qn, obs, s0, &mut r1);
            if cn < c {
                let gain = c - cn;
                q = qn;
                c = cn;
                lambda = (lambda * 0.3).max(1e-9);
                improved = gain > 1e-12;
                break;
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }
    (q, c)
}

/// Fit the head model to one pose_full row ([x1,y1,x2,y2,score, 17 x (x,y,c)]). None when fewer than three of the
/// five head keypoints are seen.
pub fn fit(p: &[f64]) -> Option<Pose3> {
    let k = pu::kps(p);
    let obs: Vec<(usize, [f64; 2], f64)> =
        [NOSE, LEYE, REYE, LEAR, REAR].into_iter().filter(|&i| k[i][2] > FIT_VIS).map(|i| (i, [k[i][0], k[i][1]], k[i][2])).collect();
    if obs.len() < 3 {
        return None;
    }
    let seen = |i: usize| obs.iter().find(|o| o.0 == i).map(|o| o.1);
    let d = |a: [f64; 2], b: [f64; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    // starting scale: the widest spread of the seen points over the model's spread of the same points (a turned
    // head only narrows the image, so this errs small; the fit takes it from there)
    let (mut wi, mut wm) = (0.0f64, 0.0f64);
    for a in &obs {
        for b in &obs {
            wi = wi.max(d(a.1, b.1));
            let (ma, mb) = (MODEL[a.0], MODEL[b.0]);
            wm = wm.max(((ma[0] - mb[0]).powi(2) + (ma[1] - mb[1]).powi(2)).sqrt());
        }
    }
    let s0 = (wi / wm.max(0.3)).max(2.0);
    let roll0 = match (seen(LEYE), seen(REYE)) {
        (Some(a), Some(b)) if a[0] > b[0] => (a[1] - b[1]).atan2(a[0] - b[0]),
        _ => 0.0,
    };
    let mean = |f: fn(&[f64; 2]) -> f64| obs.iter().map(|o| f(&o.1)).sum::<f64>() / obs.len() as f64;
    let (mx, my) = (mean(|p| p[0]), mean(|p| p[1]));
    let mut best: Option<([f64; 6], f64)> = None;
    for yaw0 in [-1.0f64, -0.4, 0.0, 0.4, 1.0] {
        for pitch0 in [-0.3f64, 0.3] {
            // centre the model's projected points on the observed ones
            let r = rot(yaw0, pitch0, roll0);
            let (mut cx, mut cy) = (0.0, 0.0);
            for o in &obs {
                let v = apply(&r, MODEL[o.0]);
                cx += v[0];
                cy += v[1];
            }
            let n = obs.len() as f64;
            let q0 = [yaw0, pitch0, roll0, s0.ln(), mx - s0 * cx / n, my - s0 * cy / n];
            let (q, c) = refine(q0, &obs, s0);
            if best.map_or(true, |b| c < b.1) {
                best = Some((q, c));
            }
        }
    }
    let (q, _) = best?;
    let pose = Pose3 { yaw: q[0], pitch: q[1], roll: q[2], s: q[3].exp(), t: [q[4], q[5]], err: 0.0, n: obs.len() };
    let mut e2 = 0.0;
    for o in &obs {
        let v = pose.project(MODEL[o.0]);
        e2 += (v[0] - o.1[0]).powi(2) + (v[1] - o.1[1]).powi(2);
    }
    Some(Pose3 { err: (e2 / obs.len() as f64).sqrt() / pose.s, ..pose })
}

/// {piece: {t: (yaw_deg, pitch_deg, roll_deg)}} for every cover in per_frame (resolve() output): each cover's pose
/// skeleton (pose_util::pose_for_head, as head_pose::track_angles finds it) fitted, then smoothed along the piece
/// with a Gaussian of sigma frames, each fit weighted by how well it fits (1 / (1 + err / 0.05)), with a weak pull to
/// neutral (prior) so frames with no skeleton drift back to facing the camera.
pub fn track_poses(per_frame: &[Vec<crate::head_pose::Item>], poses: &[Vec<Vec<f64>>], sigma: f64, prior: f64) -> BTreeMap<String, BTreeMap<usize, (f64, f64, f64)>> {
    let mut samples: BTreeMap<String, Vec<(f64, f64, [f64; 3])>> = BTreeMap::new(); // (t, weight, angles)
    let mut present: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (t, items) in per_frame.iter().enumerate() {
        let rows: &[Vec<f64>] = if t < poses.len() { &poses[t] } else { &[] };
        for it in items {
            let piece = it.piece().to_string();
            present.entry(piece.clone()).or_default().push(t);
            if let Some(f) = pu::pose_for_head(rows, &it.c, 0.25).and_then(fit) {
                samples.entry(piece).or_default().push((t as f64, 1.0 / (1.0 + f.err / 0.05), [f.yaw, f.pitch, f.roll]));
            }
        }
    }
    let r = 3.0 * sigma;
    let mut out = BTreeMap::new();
    for (piece, ts) in present {
        let s = samples.remove(&piece).unwrap_or_default();
        let mut res = BTreeMap::new();
        for t in ts {
            let (mut w, mut a) = (prior, [0.0f64; 3]);
            for &(ts, ws, ang) in s.iter().filter(|x| (x.0 - t as f64).abs() <= r) {
                let g = ws * (-0.5 * ((ts - t as f64) / sigma).powi(2)).exp();
                for k in 0..3 {
                    a[k] += g * ang[k];
                }
                w += g;
            }
            res.insert(t, (a[0].to_degrees() / w, a[1].to_degrees() / w, a[2].to_degrees() / w));
        }
        out.insert(piece, res);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row_for(pose: &Pose3, hide: &[usize]) -> Vec<f64> {
        let mut p = vec![0.0; 5 + 17 * 3];
        for i in 0..5 {
            let v = pose.project(MODEL[i]);
            p[5 + i * 3] = v[0];
            p[5 + i * 3 + 1] = v[1];
            p[5 + i * 3 + 2] = if hide.contains(&i) { 0.0 } else { 0.9 };
        }
        p
    }

    #[test]
    fn recovers_a_known_pose() {
        for (yaw, pitch, roll) in [(0.0, 0.0, 0.0), (0.5, 0.2, 0.1), (-0.6, -0.15, -0.2), (0.3, 0.3, 0.0)] {
            let truth = Pose3 { yaw, pitch, roll, s: 30.0, t: [200.0, 150.0], err: 0.0, n: 5 };
            let f = fit(&row_for(&truth, &[])).unwrap();
            // the prior pulls a little towards neutral
            assert!((f.yaw - yaw).abs() < 0.08, "yaw {} vs {}", f.yaw, yaw);
            assert!((f.pitch - pitch).abs() < 0.08, "pitch {} vs {}", f.pitch, pitch);
            assert!((f.roll - roll).abs() < 0.05, "roll {} vs {}", f.roll, roll);
            assert!((f.s - 30.0).abs() < 2.0);
        }
    }

    #[test]
    fn profile_with_one_ear() {
        // turned well to image right: the left ear (image right side when facing) goes out of sight
        let truth = Pose3 { yaw: 1.1, pitch: 0.0, roll: 0.0, s: 30.0, t: [200.0, 150.0], err: 0.0, n: 4 };
        let f = fit(&row_for(&truth, &[LEAR])).unwrap();
        assert!(f.yaw > 0.9, "{:?}", f);
    }
}
