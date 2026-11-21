//! Following a cover you placed (review_server's propagate): mm_core::edits::Follow for the native engine.
//!
//! - **flow_step** (move a box from one frame to the next by the flow inside it): mm-motion's head motion (our own;
//!   the box moves by its similarity, its size changes by at most 10% a frame, as review_server.flow_step clipped it).
//! - **face_evidence** (what a followed box snaps to): review_server.face_evidence and _someone_else, ported: the
//!   closest weak detection of about the box's size (track.fused_frames at WEAK_DET), or a confident skeleton head,
//!   within PROP_SNAP box sizes, never a neighbour's tracked face.
use crate::evidence::NativeMotion;
use mm_core::edits::{Follow, PROP_SNAP};
use mm_records::resolve::{from_cs, Track};
use mm_scan::evidence::Motion as _;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Cfg {
    pub work: PathBuf,
}

impl Cfg {
    pub fn new(_o: &crate::scan::Options, work: &std::path::Path) -> Cfg {
        Cfg { work: work.to_path_buf() }
    }
}

pub struct Follower {
    flow: NativeMotion,
    w: f64,
    h: f64,
    auto: Arc<Vec<Track>>,
    boxes: Vec<HashMap<i64, [f64; 3]>>,
    obs: Vec<HashSet<i64>>,
    poses: Arc<Vec<Vec<Vec<f64>>>>,
    low: Vec<Vec<mm_scan::geom::Fused>>,
}

pub fn make(cfg: &Cfg, auto: Arc<Vec<Track>>, poses: Arc<Vec<Vec<Vec<f64>>>>, _n: usize, w: usize, h: usize) -> Box<dyn Follow + Send> {
    let flow = NativeMotion::new(Arc::new(crate::evidence::Frames::for_project(&cfg.work).expect("the project's meta.json")));
    let spec: Vec<(&str, f64)> = mm_scan::geom::CORE_DETS.iter().map(|(n, _)| (*n, mm_scan::geom::WEAK_DET)).collect();
    let low = mm_scan::geom::fused_frames(&cfg.work.join("dets"), &spec).unwrap_or_default();
    Box::new(Follower {
        flow,
        w: w as f64,
        h: h as f64,
        boxes: auto.iter().map(|t| t.cover.iter().copied().collect()).collect(),
        obs: auto.iter().map(|t| t.obs.iter().copied().collect()).collect(),
        auto,
        poses,
        low,
    })
}

impl Follower {
    /// _someone_else: is detection bc at frame u the face of another tracked box, one that the frame before was more
    /// than a third of a box away from where this cover was (prev)?
    fn someone_else(&self, u: i64, step: i64, prev: [f64; 3], bc: [f64; 3]) -> bool {
        for (i, _) in self.auto.iter().enumerate() {
            let Some(ab) = self.boxes[i].get(&u) else { continue };
            if !self.obs[i].contains(&u) || (ab[0] - bc[0]).hypot(ab[1] - bc[1]) > 0.25 * ab[2] {
                continue;
            }
            if let Some(pb) = self.boxes[i].get(&(u - step)) {
                if (pb[0] - prev[0]).hypot(pb[1] - prev[1]) > prev[2] / 3.0 {
                    return true;
                }
            }
        }
        false
    }

    fn native_step(m: &NativeMotion, from: i64, to: i64, c: [f64; 3], w: f64, h: f64) -> Option<[f64; 3]> {
        let b = from_cs(&c);
        let (x1, y1) = (b[0].round_ties_even().max(0.0), b[1].round_ties_even().max(0.0));
        let (x2, y2) = (b[2].round_ties_even().min(w - 1.0), b[3].round_ties_even().min(h - 1.0));
        if x2 - x1 < 4.0 || y2 - y1 < 4.0 {
            return None;
        }
        let (mm, _) = m.face_motion(from, to, &c);
        let mm = mm?;
        let nx = mm[0][0] * c[0] + mm[0][1] * c[1] + mm[0][2];
        let ny = mm[1][0] * c[0] + mm[1][1] * c[1] + mm[1][2];
        let sc = (mm[0][0] * mm[1][1] - mm[0][1] * mm[1][0]).abs().sqrt().clamp(0.9, 1.1);
        Some([nx, ny, c[2] * sc])
    }
}

impl Follow for Follower {
    fn flow_step(&mut self, from: i64, to: i64, c: [f64; 3]) -> Option<[f64; 3]> {
        Self::native_step(&self.flow, from, to, c, self.w, self.h)
    }

    fn face_evidence(&mut self, u: i64, c: [f64; 3], prev: [f64; 3], step: i64) -> Option<[f64; 3]> {
        let mut best: Option<(f64, [f64; 3])> = None;
        for f in self.low.get(u as usize).map(|v| v.as_slice()).unwrap_or(&[]) {
            let bc = mm_scan::geom::to_cs(&f.b);
            let d = (bc[0] - c[0]).hypot(bc[1] - c[1]) / c[2];
            if d < PROP_SNAP && 0.6 < bc[2] / c[2] && bc[2] / c[2] < 1.6 && best.is_none_or(|b| d < b.0) {
                if self.someone_else(u, step, prev, bc) {
                    continue;
                }
                best = Some((d, bc));
            }
        }
        for p in self.poses.get(u as usize).map(|v| v.as_slice()).unwrap_or(&[]) {
            if p.get(4).copied().unwrap_or(0.0) < 0.25 {
                continue;
            }
            let (xy, q) = mm_analysis::pose_util::pose_head(p, c[2]);
            let d = (xy[0] - c[0]).hypot(xy[1] - c[1]) / c[2];
            if q >= 0.5 && d < PROP_SNAP && best.is_none_or(|b| d < b.0) {
                best = Some((d, [xy[0], xy[1], c[2]]));
            }
        }
        best.map(|b| b.1)
    }
}
