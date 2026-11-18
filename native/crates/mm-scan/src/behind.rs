//! Behind someone: a held cover (a person's box on a frame where their face wasn't detected) that sits on a nearer
//! person's body is a face hidden behind them -- not covered there, so no emoji is drawn over whoever is in front.
//! (Port of `mojimask/scan/behind.py`, with `depth.face_at`; depth readings are behind [`crate::evidence::Depth`].)
//!
//! jazz (2026-11-12): the dancer in front of the band hid seated faces for a second at a time; their covers were held
//! in place (cover follows across gaps of up to attach::MAX_GAP frames) and drew their emoji on her dress and arms.
//!
//! A frame is behind when
//!   - the box isn't detected there (only held covers are ever dropped: a face that shows is a face);
//!   - no weak face detection is on it either (low: everything any detector half-believes, >= track.WEAK_DET -- a face
//!     that's visible but missed by the main detectors usually leaves one; jazz frame 92's seated women did);
//!   - FRAC of it lies on another skeleton's body: its torso (shoulders to hips, a little wider), arms and legs (as
//!     LIMB-wide capsules) or head -- never its own skeleton (pose_util.pose_for_head);
//!   - and that person is in front: clearly nearer the camera (shoulders >= IN_FRONT x the hidden face's size;
//!     shoulder width is about one face size, median 1.08 on jazz, 0.9 on the party clips), or the hidden face has lost
//!     its own skeleton too (jazz frame 115: the dancer back among the band, their size, both faces behind her
//!     skeleton-less). Either cue alone dropped visible faces: on-body alone took seated neighbours' arms (369 of 822
//!     held frames on jazz); no-skeleton without the weak-detection check took two seated women whose skeletons the
//!     pose detector also missed.
//! Runs: a kept stretch of <= CLOSE frames between two behind frames is behind too, so an emoji doesn't blink back
//! for a frame or two while the person is still hidden.
//! Measured on held frames: jazz 274 of 822 behind, party clips a / b / c 18 / 44 / 22, IMG_1043 148 of 4021; by eye
//! the party ones are torsos, hair and dark corners, no face.
//!
//! With a depth model on (MOJIMASK_DEPTH, depth.py) depth only adds: "in front" also holds when the box reads `front`
//! nearer than the face's own depth around then (face_at; Depth.near units: the frame's depth spread), and a box
//! reading `alone` nearer is behind someone even with no skeleton over it (the pose detector missed the arm). Both are
//! per model (depth.MODELS). Replacing the shoulder test with depth put emoji back on jazz's dancer where depth missed
//! her dress (8@83-91, 16@116-121). tools/bench/depth_bench.py: with Small, visible faces read >= 0.5 nearer in
//! 0.2-1.1% of frames.
use crate::evidence::Depth;
use crate::geom::{self, Cs, Fused};
use crate::np;
use crate::pose::{self, Pose, VIS};
use indexmap::IndexMap;
use std::collections::BTreeSet;

pub const FRAC: f64 = 0.5;
pub const IN_FRONT: f64 = 1.4;
pub const LIMB: f64 = 0.2;
pub const CLOSE: i64 = 2;
pub const FACE_WINDOW: f64 = 15.0;
const LIMBS: [(usize, usize); 8] = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)];

fn grid() -> [f64; 7] {
    // np.linspace(0.1, 0.9, 7)
    let step = (0.9f64 - 0.1) / 6.0;
    let mut g = [0.0; 7];
    for (i, x) in g.iter_mut().enumerate() {
        *x = i as f64 * step + 0.1;
    }
    g[6] = 0.9;
    g
}

/// behind.on_body: which of pts fall on skeleton p's body (torso, limbs, head).
pub fn on_body(p: &Pose, pts: &[[f64; 2]]) -> Vec<bool> {
    let k = pose::kps(p);
    let v: Vec<bool> = k.iter().map(|x| x[2] > VIS).collect();
    let none = vec![false; pts.len()];
    if !(v[5] && v[6]) {
        return none;
    }
    let sh = [[k[5][0], k[5][1]], [k[6][0], k[6][1]]];
    let sw = np::norm(&[sh[0][0] - sh[1][0], sh[0][1] - sh[1][1]]);
    if sw < 6.0 {
        return none;
    }
    let hips = if v[11] && v[12] {
        [[k[11][0], k[11][1]], [k[12][0], k[12][1]]]
    } else {
        [[sh[0][0] + 0.0, sh[0][1] + 1.6 * sw], [sh[1][0] + 0.0, sh[1][1] + 1.6 * sw]]
    };
    let poly = [sh[0], sh[1], hips[1], hips[0]];
    let mut mean = [0.0; 2];
    for j in 0..2 {
        mean[j] = (((poly[0][j] + poly[1][j]) + poly[2][j]) + poly[3][j]) / 4.0;
    }
    let poly32: Vec<[f32; 2]> =
        poly.iter().map(|q| [(mean[0] + (q[0] - mean[0]) * 1.15) as f32, (mean[1] + (q[1] - mean[1]) * 1.15) as f32]).collect();
    let mut hit: Vec<bool> = pts.iter().map(|q| mm_vision::geom::point_polygon_test(&poly32, [q[0] as f32, q[1] as f32]) >= 0.0).collect();
    for (a, b) in LIMBS {
        if v[a] && v[b] {
            let (pa, pb) = ([k[a][0], k[a][1]], [k[b][0], k[b][1]]);
            let d = [pb[0] - pa[0], pb[1] - pa[1]];
            let rel: Vec<f64> = pts.iter().flat_map(|q| [q[0] - pa[0], q[1] - pa[1]]).collect();
            let proj = np::gemv_n(&rel, pts.len(), 2, &d);
            let dd = np::ddot(&d, &d);
            let den = if 1e-6 > dd { 1e-6 } else { dd };
            for (i, q) in pts.iter().enumerate() {
                let t = (proj[i] / den).max(0.0).min(1.0);
                let (x, y) = (pa[0] + t * d[0], pa[1] + t * d[1]);
                if np::hypot(q[0] - x, q[1] - y) < LIMB * sw {
                    hit[i] = true;
                }
            }
        }
    }
    if let Some(h) = pose::head(p) {
        for (i, q) in pts.iter().enumerate() {
            if np::hypot(q[0] - h.0, q[1] - h.1) < 0.55 * h.2 {
                hit[i] = true;
            }
        }
    }
    hit
}

/// behind.face_there: a weak detection on box c, about its place and size.
pub fn face_there(c: &Cs, low: &[Fused]) -> bool {
    low.iter().any(|f| {
        let bc = geom::to_cs(&f.b);
        np::hypot(bc[0] - c[0], bc[1] - c[1]) < 0.5 * c[2] && 0.5 < bc[2] / c[2] && bc[2] / c[2] < 2.0
    })
}

/// behind.is_behind. nearer: how much nearer the box reads than the face does around then (depth on), else None;
/// margins: the depth model's (front, alone).
pub fn is_behind(c: &Cs, entries: &[Pose], low: &[Fused], nearer: Option<f64>, margins: Option<(f64, f64)>) -> bool {
    if face_there(c, low) {
        return false;
    }
    if let (Some(n), Some((_, alone))) = (nearer, margins) {
        if n >= alone {
            return true;
        }
    }
    let own = pose::pose_for_head(entries, c);
    let [x0, y0, x1, y1] = geom::from_cs(c);
    let g = grid();
    let pts: Vec<[f64; 2]> = g.iter().flat_map(|a| g.iter().map(move |b| [x0 + a * (x1 - x0), y0 + b * (y1 - y0)])).collect();
    for (i, p) in entries.iter().enumerate() {
        if Some(i) == own || p[4] < 0.3 {
            continue;
        }
        let k = pose::kps(p);
        if !(k[5][2] > VIS && k[6][2] > VIS) {
            continue;
        }
        let shw = np::norm(&[k[5][0] - k[6][0], k[5][1] - k[6][1]]);
        let not_nearer = match (nearer, margins) {
            (Some(n), Some((front, _))) => n < front,
            _ => true,
        };
        if own.is_some() && shw < IN_FRONT * c[2] && not_nearer {
            continue;
        }
        let hit = on_body(p, &pts);
        if hit.iter().filter(|h| **h).count() as f64 / hit.len() as f64 >= FRAC {
            return true;
        }
    }
    false
}

/// depth.face_at: a face's depth reading at frame t from its own detected readings v at frames ts (sorted, finite):
/// the median of those within FACE_WINDOW frames of t (past a gap with none, the window starts at the nearest ones),
/// never the one at t itself -- that is where an occluder would be. At least 5 readings (the nearest ones) when it
/// has them. (numpy's argsort is stable at these sizes; ties beyond would follow its sort.)
pub fn face_at(ts: &[i64], v: &[f64], t: i64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let d: Vec<f64> = ts.iter().map(|x| if *x == t { f64::INFINITY } else { (x - t).abs() as f64 }).collect();
    if !d.iter().any(|x| x.is_finite()) {
        return np::median(v);
    }
    let dmin = d.iter().copied().fold(f64::INFINITY, f64::min);
    let lim = FACE_WINDOW + dmin - 1.0;
    let mut sel: Vec<f64> = d.iter().zip(v).filter(|(x, _)| **x <= lim).map(|(_, y)| *y).collect();
    if sel.len() < 5 {
        let nfin = d.iter().filter(|x| x.is_finite()).count();
        let mut idx: Vec<usize> = (0..d.len()).collect();
        idx.sort_by(|a, b| d[*a].partial_cmp(&d[*b]).unwrap());
        sel = idx[..5.min(nfin)].iter().map(|i| v[*i]).collect();
    }
    np::median(&sel)
}

/// behind.behind_frames: the held frames of one person's cover that are behind someone. low: weak detections per frame.
pub fn behind_frames(cover: &IndexMap<i64, Cs>, obs: &IndexMap<i64, Cs>, poses: &[Vec<Pose>], low: &[Vec<Fused>],
                     mut depth: Option<&mut Box<dyn Depth>>) -> BTreeSet<i64> {
    let mut held: Vec<i64> = cover.keys().copied().filter(|t| !obs.contains_key(t)).collect();
    held.sort();
    let mut face: Option<(Vec<i64>, Vec<f64>)> = None;
    let margins = depth.as_ref().map(|d| {
        let m = d.margins();
        (m.0, m.1)
    });
    if let Some(dep) = depth.as_mut() {
        if !obs.is_empty() {
            let mut ts: Vec<i64> = obs.keys().copied().collect();
            ts.sort();
            let v: Vec<f64> = ts.iter().map(|t| dep.near(*t, &obs[t], 0.25)).collect();
            let keep: Vec<usize> = (0..ts.len()).filter(|i| v[*i].is_finite()).collect();
            face = Some((keep.iter().map(|i| ts[*i]).collect(), keep.iter().map(|i| v[*i]).collect()));
        }
    }
    let mut out = BTreeSet::new();
    for &t in &held {
        let n = match (&face, depth.as_mut()) {
            (Some((fts, fv)), Some(dep)) if !fv.is_empty() => {
                let x = dep.near(t, &cover[&t], 0.25) - face_at(fts, fv, t);
                if x.is_finite() {
                    Some(x)
                } else {
                    None
                }
            }
            _ => None,
        };
        if t as usize >= poses.len() && n.is_none() {
            continue;
        }
        let entries: &[Pose] = if (t as usize) < poses.len() { &poses[t as usize] } else { &[] };
        let lw: &[Fused] = if (t as usize) < low.len() { &low[t as usize] } else { &[] };
        if is_behind(&cover[&t], entries, lw, n, margins) {
            out.insert(t);
        }
    }
    let hs: BTreeSet<i64> = held.iter().copied().collect();
    let snapshot: Vec<i64> = out.iter().copied().collect();
    for t in snapshot {
        for gap in 1..=CLOSE {
            if out.contains(&(t + gap + 1)) && (1..=gap).all(|i| hs.contains(&(t + i))) {
                for i in 1..=gap {
                    out.insert(t + i);
                }
            }
        }
    }
    out
}
