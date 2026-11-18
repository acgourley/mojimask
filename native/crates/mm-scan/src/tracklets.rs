//! The scan's first step after detection: tracklets. join.py chains them, graph.py groups the chains into people.
//! (Port of `mojimask/scan/tracklets.py`.)
//!
//! A tracklet is a chain of detections linked only where the link is certain. Per frame t:
//!   - predict each open tracklet: its last box, moved by the camera (mats_v2.npy) and its own velocity (from its
//!     last two detections, camera removed);
//!   - a detection is a candidate for a tracklet if its centre is within R head sizes of the prediction and its
//!     size is within SIZE of the tracklet's;
//!   - link only a mutual unique pair: the tracklet has exactly one candidate, and that detection has exactly one
//!     candidate tracklet;
//!   - a tracklet with no candidate may skip one frame (a detector miss), carried by the camera; a tracklet with
//!     two or more candidates, or one whose detection is claimed by another tracklet too, ends ("ambiguous");
//!   - every detection not linked starts a tracklet.
//! Detections: the app's fused heads at the app's per-detector thresholds (pipeline.CORE_DETS).
//!
//! (The Python module's command line -- counts, lengths and purity against a census -- isn't ported: it is a research
//! tool, not part of the decide step.)
use crate::geom::{self, Cs};
use crate::np::{self, M23};
use std::collections::BTreeMap;
use std::path::Path;

pub const R: f64 = 0.5;
pub const SIZE: (f64, f64) = (0.7, 1.4);
pub const SKIP: i64 = 1;

#[derive(Clone, Debug)]
pub struct Tracklet {
    pub obs: BTreeMap<i64, Cs>,
    pub score: BTreeMap<i64, f64>,
    pub v: [f64; 2],
    pub why: &'static str,
}

impl Tracklet {
    pub fn first(&self) -> i64 {
        *self.obs.keys().next().unwrap()
    }
    pub fn last(&self) -> i64 {
        *self.obs.keys().next_back().unwrap()
    }
}

/// Tracklets by id (ids are 0.., in creation order: the dict's order in Python).
pub type Tracklets = BTreeMap<i64, Tracklet>;

/// tracklets.detections: per frame, (cs box, score) of the fused core detections.
pub fn detections(proj: &Path) -> Result<Vec<Vec<(Cs, f64)>>, String> {
    let dd = proj.join("dets");
    let spec: Vec<(&str, f64)> = geom::CORE_DETS.iter().copied().filter(|(n, _)| dd.join(format!("{n}.json")).exists()).collect();
    Ok(geom::fused_frames(&dd, &spec)?.into_iter().map(|f| f.into_iter().map(|x| (geom::to_cs(&x.b), x.s)).collect()).collect())
}

/// tracklets.build.
pub fn build(dets: &[Vec<(Cs, f64)>], mats: &[M23], r: f64) -> Tracklets {
    let mut tl: Tracklets = BTreeMap::new();
    let mut open: Vec<i64> = vec![];
    let mut nid = 0i64;
    for (t, d) in dets.iter().enumerate() {
        let t = t as i64;
        let mut preds = vec![];
        for k in &open {
            let tr = &tl[k];
            let last = tr.last();
            let mut c = geom::warp_cs(&tr.obs[&last], &geom::compose(mats, last, t));
            c[0] += tr.v[0] * (t - last) as f64;
            c[1] += tr.v[1] * (t - last) as f64;
            preds.push(c);
        }
        let cand_t: Vec<Vec<usize>> = preds
            .iter()
            .map(|p| {
                d.iter()
                    .enumerate()
                    .filter(|(_, (c, _))| np::hypot(c[0] - p[0], c[1] - p[1]) < r * p[2] && SIZE.0 < c[2] / p[2] && c[2] / p[2] < SIZE.1)
                    .map(|(j, _)| j)
                    .collect()
            })
            .collect();
        let mut cand_d: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, js) in cand_t.iter().enumerate() {
            for &j in js {
                cand_d.entry(j).or_default().push(i);
            }
        }
        let mut linked_d = std::collections::HashSet::new();
        let mut still = vec![];
        for (i, js) in cand_t.iter().enumerate() {
            let k = open[i];
            let tr = tl.get_mut(&k).unwrap();
            if js.len() == 1 && cand_d[&js[0]].len() == 1 {
                let j = js[0];
                let last = tr.last();
                let c = d[j].0;
                if t - last == 1 {
                    let w = geom::warp_cs(&tr.obs[&last], &mats[t as usize]);
                    let moved = [c[0] - w[0], c[1] - w[1]];
                    tr.v = [0.5 * tr.v[0] + 0.5 * moved[0], 0.5 * tr.v[1] + 0.5 * moved[1]];
                }
                tr.obs.insert(t, c);
                tr.score.insert(t, d[j].1);
                linked_d.insert(j);
                still.push(k);
            } else if js.is_empty() && t - tr.last() <= SKIP {
                still.push(k);
            } else {
                tr.why = if !js.is_empty() { "ambiguous" } else { "lost" };
            }
        }
        for (j, (c, s)) in d.iter().enumerate() {
            if linked_d.contains(&j) {
                continue;
            }
            tl.insert(nid, Tracklet { obs: BTreeMap::from([(t, *c)]), score: BTreeMap::from([(t, *s)]), v: [0.0, 0.0], why: "end" });
            still.push(nid);
            nid += 1;
        }
        open = still;
    }
    tl
}
