//! Pose helpers the scan uses: head geometry from COCO-17 keypoints (part of Python `mojimask/pose_util.py`).
//!
//! Keypoints: 0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows, 9/10 wrists, 11/12 hips, ...
//! A pose entry ("pose_full"): [x1,y1,x2,y2,score, k0x,k0y,k0c, ..., k16x,k16y,k16c].
use crate::geom::Cs;
use crate::np;

pub const VIS: f64 = 0.3;
pub const OWN_POSE_TOL: f64 = 0.5;
const NOSE: usize = 0;
const LEYE: usize = 1;
const REYE: usize = 2;
const LEAR: usize = 3;
const REAR: usize = 4;
const LSH: usize = 5;
const RSH: usize = 6;

pub type Pose = Vec<f64>;

/// pose_util.kps: 17 x (x, y, confidence).
pub fn kps(p: &Pose) -> Vec<[f64; 3]> {
    (0..17).map(|i| [p[5 + 3 * i], p[6 + 3 * i], p[7 + 3 * i]]).collect()
}

fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    np::norm(&[a[0] - b[0], a[1] - b[1]])
}

/// pose_util.head: (cx, cy, head width, confidence) or None. Head width via anthropometric ratios (shoulders ~2.2,
/// ears ~0.8, eyes ~0.42 head widths).
pub fn head(p: &Pose) -> Option<(f64, f64, f64, f64)> {
    let k = kps(p);
    let c: Vec<f64> = k.iter().map(|x| x[2]).collect();
    let headpts: Vec<usize> = [NOSE, LEYE, REYE, LEAR, REAR].into_iter().filter(|&i| c[i] > VIS).collect();
    let mut widths = vec![];
    if c[LSH] > VIS && c[RSH] > VIS {
        widths.push(dist(&k[LSH], &k[RSH]) / 2.2);
    }
    if c[LEAR] > VIS && c[REAR] > VIS {
        widths.push(dist(&k[LEAR], &k[REAR]) / 0.8);
    }
    if c[LEYE] > VIS && c[REYE] > VIS {
        widths.push(dist(&k[LEYE], &k[REYE]) / 0.42);
    }
    if widths.is_empty() {
        return None;
    }
    let hw = np::median(&widths);
    if headpts.len() >= 2 {
        let n = headpts.len() as f64;
        let (mut sx, mut sy, mut sc) = (0.0, 0.0, 0.0);
        for &i in &headpts {
            sx += k[i][0];
            sy += k[i][1];
            sc += c[i];
        }
        Some((sx / n, sy / n, hw, sc / n))
    } else if c[LSH] > VIS && c[RSH] > VIS {
        let mid = [(k[LSH][0] + k[RSH][0]) / 2.0, (k[LSH][1] + k[RSH][1]) / 2.0];
        Some((mid[0], mid[1] - 1.1 * hw, hw, (c[LSH] + c[RSH]) / 2.0 * 0.5))
    } else {
        None
    }
}

/// pose_util.pose_for_head: the entry (index into one frame's list) whose head estimate is nearest head box c, or None.
/// Only a skeleton whose head is within OWN_POSE_TOL head sizes counts as this head's own. On jazz census truth a
/// skeleton's real owner is within 0.31 head sizes 9 times in 10 and the next-nearest head is beyond 0.65 9 times in
/// 10; the old 0.8 let 18 of 131 heads in a crowd take a neighbour's skeleton, giving two people the same body picture
/// (body similarity 1.0 between strangers).
pub fn pose_for_head(entries: &[Pose], c: &Cs) -> Option<usize> {
    let mut best: Option<(f64, usize)> = None;
    for (i, p) in entries.iter().enumerate() {
        if p[4] < 0.25 {
            continue;
        }
        if let Some(h) = head(p) {
            let d = np::hypot(h.0 - c[0], h.1 - c[1]);
            let hs = h.2 * 1.3;
            let m = if hs > c[2] { hs } else { c[2] };
            if d < OWN_POSE_TOL * m && best.map(|b| d < b.0).unwrap_or(true) {
                best = Some((d, i));
            }
        }
    }
    best.map(|b| b.1)
}

/// The project's skeletons per frame (dets/pose_full.json), or None when the project has none.
pub fn load(proj: &std::path::Path) -> Result<Option<Vec<Vec<Pose>>>, String> {
    let p = proj.join("dets").join("pose_full.json");
    if !p.exists() {
        return Ok(None);
    }
    let d = mm_core::pyjson::load(&p)?;
    Ok(Some(
        d.get("frames")
            .as_list()
            .ok_or("pose_full: frames")?
            .iter()
            .map(|f| f.as_list().unwrap().iter().map(|e| e.as_list().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()).collect())
            .collect(),
    ))
}
