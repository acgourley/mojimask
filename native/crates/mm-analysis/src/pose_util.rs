//! Pose helpers: head geometry and facing state from COCO-17 keypoints.
//!
//! Keypoints: 0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows, 9/10 wrists, 11/12 hips, ...
//! A pose entry from experiments/detect.py "pose_full": [x1,y1,x2,y2,score, k0x,k0y,k0c, ..., k16x,k16y,k16c].
//!
//! Port of mojimask/pose_util.py (all float64, as numpy there; np.linalg.norm of a 2-vector is
//! sqrt(x*x + y*y), checked). Parity: native/tools/parity/analysis_pose.py.
use crate::np;

pub const NOSE: usize = 0;
pub const LEYE: usize = 1;
pub const REYE: usize = 2;
pub const LEAR: usize = 3;
pub const REAR: usize = 4;
pub const LSH: usize = 5;
pub const RSH: usize = 6;
pub const LEL: usize = 7;
pub const REL: usize = 8;
pub const LWR: usize = 9;
pub const RWR: usize = 10;
pub const VIS: f64 = 0.3;

pub type Kps = [[f64; 3]; 17];

pub fn kps(p: &[f64]) -> Kps {
    let mut k = [[0f64; 3]; 17];
    for i in 0..17 {
        for j in 0..3 {
            k[i][j] = p[5 + i * 3 + j];
        }
    }
    k
}

/// np.linalg.norm(a - b) for 2-d points.
pub fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    let (x, y) = (a[0] - b[0], a[1] - b[1]);
    (x * x + y * y).sqrt()
}

fn xy(k: &Kps, i: usize) -> [f64; 2] {
    [k[i][0], k[i][1]]
}

/// mean over rows (numpy's axis-0 reduction: sequential), of x and y
fn mean_xy(k: &Kps, idx: &[usize]) -> [f64; 2] {
    let (mut x, mut y) = (0.0, 0.0);
    for &i in idx {
        x += k[i][0];
        y += k[i][1];
    }
    [x / idx.len() as f64, y / idx.len() as f64]
}

fn mean_c(k: &Kps, idx: &[usize]) -> f64 {
    let v: Vec<f64> = idx.iter().map(|&i| k[i][2]).collect();
    np::mean(&v)
}

/// Head estimate from one pose entry -> (cx, cy, head_width, confidence) or None.
/// Head width is in 'head widths' via anthropometric ratios (shoulders ~2.2, ears ~0.8,
/// eyes ~0.42 head widths); callers calibrate per person against face boxes.
pub fn head(p: &[f64]) -> Option<(f64, f64, f64, f64)> {
    let k = kps(p);
    let c = |i: usize| k[i][2];
    let headpts: Vec<usize> = [NOSE, LEYE, REYE, LEAR, REAR].into_iter().filter(|&i| c(i) > VIS).collect();
    let mut widths = Vec::new();
    if c(LSH) > VIS && c(RSH) > VIS {
        widths.push(dist(xy(&k, LSH), xy(&k, RSH)) / 2.2);
    }
    if c(LEAR) > VIS && c(REAR) > VIS {
        widths.push(dist(xy(&k, LEAR), xy(&k, REAR)) / 0.8);
    }
    if c(LEYE) > VIS && c(REYE) > VIS {
        widths.push(dist(xy(&k, LEYE), xy(&k, REYE)) / 0.42);
    }
    if widths.is_empty() {
        return None;
    }
    let hw = np::median(&widths);
    let (cx, cy, conf);
    if headpts.len() >= 2 {
        let m = mean_xy(&k, &headpts);
        cx = m[0];
        cy = m[1];
        conf = mean_c(&k, &headpts);
    } else if c(LSH) > VIS && c(RSH) > VIS {
        // back of head / occluded face: above the shoulders
        let mid = mean_xy(&k, &[LSH, RSH]);
        cx = mid[0];
        cy = mid[1] - 1.1 * hw;
        conf = mean_c(&k, &[LSH, RSH]) * 0.5;
    } else {
        return None;
    }
    Some((cx, cy, hw, conf))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Facing {
    Front,
    Profile,
    Away,
    Unknown,
}

impl Facing {
    pub fn name(self) -> &'static str {
        match self {
            Facing::Front => "front",
            Facing::Profile => "profile",
            Facing::Away => "away",
            Facing::Unknown => "unknown",
        }
    }
}

/// 'front', 'profile', 'away' or 'unknown' from face keypoint visibility.
///
/// As Python computes it, `eyes = (c[LEYE] > 0.5) + (c[REYE] > 0.5)` adds two numpy bools, which is a logical or
/// (True, never 2): so "front" is never returned, and a face with both eyes and the nose seen is "profile".
/// Kept for parity; the fix (count the eyes) belongs in pose_util.py first. Its one caller is the review server's
/// skeleton view (review_server.py), which only shows it.
pub fn facing(p: &[f64]) -> Facing {
    let k = kps(p);
    let c = |i: usize| k[i][2];
    let eyes = ((c(LEYE) > 0.5) || (c(REYE) > 0.5)) as i32;
    if c(NOSE) > 0.5 && eyes == 2 {
        return Facing::Front;
    }
    if c(NOSE) > 0.4 || eyes >= 1 {
        return Facing::Profile;
    }
    if (c(LEAR) > 0.4 || c(REAR) > 0.4 || (c(LSH) > 0.5 && c(RSH) > 0.5)) && c(NOSE) < 0.2 && eyes == 0 {
        return Facing::Away;
    }
    Facing::Unknown
}

/// Head box [x1, y1, x2, y2, score] from one skeleton's COCO keypoints k (17x3: x, y, confidence) and its person
/// box [x1, y1, x2, y2] (experiments/detect.py's "pose_head"): around the visible head points (nose, eyes, ears)
/// when at least two are seen; one head point sized by the shoulders; else above the shoulders (back of the head).
pub fn head_from_pose(k: &Kps, b: &[f64]) -> Option<[f64; 5]> {
    let c = |i: usize| k[i][2];
    let head: Vec<usize> = (NOSE..=REAR).filter(|&i| c(i) > VIS).collect();
    let sh: Vec<usize> = [LSH, RSH].into_iter().filter(|&i| c(i) > VIS).collect();
    let shw = if sh.len() == 2 { dist(xy(k, LSH), xy(k, RSH)) } else { 0.0 };
    let (cx, cy, s, score);
    if head.len() >= 2 {
        let m = mean_xy(k, &head);
        let xs: Vec<f64> = head.iter().map(|&i| k[i][0]).collect();
        let ptp = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - xs.iter().cloned().fold(f64::INFINITY, f64::min);
        // python max(a, b, c, d): the first of the largest
        let mut span = ptp;
        for v in [0.55 * shw, 0.10 * (b[3] - b[1]), 8.0] {
            span = np::pymax(span, v);
        }
        cx = m[0];
        cy = m[1];
        s = span * 1.6;
        score = mean_c(k, &head);
    } else if head.len() == 1 && shw != 0.0 {
        cx = k[head[0]][0];
        cy = k[head[0]][1];
        s = 0.8 * shw;
        score = c(head[0]) * 0.7;
    } else if shw != 0.0 {
        // back of head / occluded face: extrapolate above the shoulders
        let mid = mean_xy(k, &sh);
        cx = mid[0];
        cy = mid[1] - 0.75 * shw;
        s = 0.8 * shw;
        score = mean_c(k, &sh) * 0.4;
    } else {
        return None;
    }
    Some([cx - s / 2.0, cy - s / 2.0, cx + s / 2.0, cy + s / 2.0, score])
}

/// pose_full entry -> head box [x1,y1,x2,y2,score] (pose_head rule, score scaled by sqrt(person score)).
pub fn head_box(entry: &[f64]) -> Option<[f64; 5]> {
    let mut h = head_from_pose(&kps(entry), &entry[..4])?;
    // python's `** 0.5` is libm pow(); LLVM would turn powf(0.5) into sqrt, which differs in the last bit
    h[4] *= entry[4].powf(std::hint::black_box(0.5));
    Some(h)
}

/// A head-box centre from a skeleton (measured on detected faces with a skeleton): head keypoints (nose/eyes/ears
/// mean) -> centre +0.02 up (MAD 0.02); shoulder midpoint -> centre 0.75 up, 0.06 left (MAD 0.08 / 0.19; in
/// shoulder widths it varies ~3x with body turn, so box size is the unit).
pub const HEAD_FROM_KP: [f64; 2] = [0.0, -0.023];
pub const HEAD_FROM_SHOULDERS: [f64; 2] = [-0.056, -0.746];

/// (head-box centre, confidence q in [0,1]) from a pose entry p for a head of box size s: confidence-weighted
/// head keypoints, or the shoulders (0.75 box sizes above their midpoint) at half their confidence -- whichever is
/// more confident. (The editor snaps a followed box to it.)
pub fn pose_head(p: &[f64], s: f64) -> ([f64; 2], f64) {
    let k = kps(p);
    let mut cs: Vec<f64> = (NOSE..=REAR).map(|i| k[i][2]).collect();
    cs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q_h = (cs[3] + cs[4]) / 2.0;
    let (mut wx, mut wy, mut ws) = (0.0, 0.0, 0.0);
    for i in NOSE..=REAR {
        wx += k[i][0] * k[i][2];
        wy += k[i][1] * k[i][2];
    }
    // h[:, 2].sum(): 5 values, sequential
    for i in NOSE..=REAR {
        ws += k[i][2];
    }
    let den = np::pymax(ws, 1e-6);
    let xy_h = [wx / den + HEAD_FROM_KP[0] * s, wy / den + HEAD_FROM_KP[1] * s];
    let q_s = 0.5 * k[LSH][2].min(k[RSH][2]);
    let xy_s = [
        (k[LSH][0] + k[RSH][0]) / 2.0 + HEAD_FROM_SHOULDERS[0] * s,
        (k[LSH][1] + k[RSH][1]) / 2.0 + HEAD_FROM_SHOULDERS[1] * s,
    ];
    if q_h >= q_s {
        (xy_h, q_h)
    } else {
        (xy_s, q_s)
    }
}

/// head sizes
pub const OWN_POSE_TOL: f64 = 0.5;

/// The pose_full entry (one frame's list) whose head estimate is nearest head box c = [cx, cy, s], or None.
/// Only a skeleton whose head is within OWN_POSE_TOL head sizes counts as this head's own. On jazz census truth
/// a skeleton's real owner is within 0.31 head sizes 9 times in 10 and the next-nearest head is beyond 0.65 9 times
/// in 10; the old 0.8 let 18 of 131 heads in a crowd take a neighbour's skeleton, giving two people the same body
/// picture (body similarity 1.0 between strangers).
pub fn pose_for_head<'a>(entries: &'a [Vec<f64>], c: &[f64; 3], min_score: f64) -> Option<&'a [f64]> {
    let mut best: Option<(f64, &[f64])> = None;
    for p in entries {
        if p[4] < min_score {
            continue;
        }
        if let Some(h) = head(p) {
            let d = (h.0 - c[0]).hypot(h.1 - c[1]);
            if d < OWN_POSE_TOL * np::pymax(c[2], h.2 * 1.3) && best.is_none_or(|b| d < b.0) {
                best = Some((d, p));
            }
        }
    }
    best.map(|b| b.1)
}
