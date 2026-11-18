//! The scan's face measurements: one per detection, with how comparable it is. The graph (graph) and the
//! same-person questions (questions) take their face evidence from these, and the "clear face" test (graph::CLEAR_FACE)
//! is their score. (Port of `mojimask/scan/faces.py`; the models -- RetinaFace landmarks and SFace -- are behind
//! [`crate::evidence::FaceEvidence::face`].)
//!
//! For every detection of every tracklet: RetinaFace (reid.landmarks) on the crop around it (reid.face_crop) finds the
//! most confident face within NEAR (0.35) head sizes of the box centre, its 5 landmarks, and SFace (reid.embed) on the
//! aligned face. Comparability comes from the landmarks:
//!   yaw    (nose x - midpoint of the eyes x) / eye distance: ~0 facing the camera, larger as the head turns
//!   eyes   eye distance / face box width: shrinks as the head turns to profile
//!   px     face box height in pixels
//! A face is "frontal" (comparable) when the landmark detector is confident, it is facing the camera and big enough.
//! Face evidence is only taken between frontal faces: a profile against a frontal face scored as low as two
//! strangers on twirl.
//!
//! Python caches the measurements per project (proto/faces_retina_sface.pkl, a pickle); this port measures every
//! detection through the evidence trait and leaves caching to the implementation behind it.
use crate::evidence::{Evidence, FaceM};
use crate::tracklets::Tracklets;
use crate::viz::Viz;
use mm_core::pyjson::{Dict, Py};
use std::collections::{BTreeMap, HashMap};

pub const NEAR: f64 = 0.35;
pub const FRONTAL_SCORE: f64 = 0.6;
pub const FRONTAL_YAW: f64 = 0.25;
pub const FRONTAL_PX: f64 = 24.0;

/// {(tracklet, t): measurement or None}
pub type Faces = HashMap<(i64, i64), Option<FaceM>>;

pub fn frontal(f: Option<&FaceM>) -> bool {
    match f {
        Some(f) => f.score >= FRONTAL_SCORE && f.yaw.abs() <= FRONTAL_YAW && f.px >= FRONTAL_PX,
        None => false,
    }
}

/// F.get((k, t)) as an Option<&FaceM>.
pub fn get(fs: &Faces, k: i64, t: i64) -> Option<&FaceM> {
    fs.get(&(k, t)).and_then(|f| f.as_ref())
}

/// faces.measure (without the cache): every detection of every tracklet, frame by frame. Also returns the order
/// Python's dict has them in (by frame, then tracklet), for the stage dump.
pub fn measure(tl: &Tracklets, ev: &mut Evidence, viz: &mut Viz) -> (Faces, Vec<(i64, i64)>) {
    let mut by_frame: BTreeMap<i64, Vec<(i64, [f64; 3])>> = BTreeMap::new();
    let mut n = 0;
    for (k, v) in tl {
        for (t, c) in &v.obs {
            by_frame.entry(*t).or_default().push((*k, *c));
            n += 1;
        }
    }
    viz.emit("faces_start", vec![("frames", Py::Int(by_frame.len() as i64)), ("detections", Py::Int(n))]);
    let mut out = Faces::new();
    let mut order = vec![];
    for (t, ds) in &by_frame {
        let mut d = vec![];
        for (k, c) in ds {
            let f = ev.faces.face(*t, c);
            d.push(Py::List(vec![Py::Int(*k), Py::Int(if frontal(f.as_ref()) { 2 } else if f.is_some() { 1 } else { 0 })]));
            out.insert((*k, *t), f);
            order.push((*k, *t));
        }
        viz.emit("faces", vec![("t", Py::Int(*t)), ("d", Py::List(d))]);
    }
    (out, order)
}

pub fn face_json(f: Option<&FaceM>) -> Py {
    match f {
        None => Py::Null,
        Some(f) => {
            let mut d = Dict::new();
            d.insert("score", Py::Float(f.score));
            d.insert("yaw", Py::Float(f.yaw));
            d.insert("eyes", Py::Float(f.eyes));
            d.insert("px", Py::Float(f.px));
            d.insert("emb", f.emb.as_ref().map(|e| Py::List(e.iter().map(|v| Py::Float(*v as f64)).collect())).unwrap_or(Py::Null));
            Py::Dict(d)
        }
    }
}

/// The stage dump (stage_faces.json): [[k, t, measurement], ...] in Python's dict order.
pub fn faces_json(fs: &Faces, order: &[(i64, i64)]) -> Py {
    Py::List(order.iter().map(|(k, t)| Py::List(vec![Py::Int(*k), Py::Int(*t), face_json(get(fs, *k, *t))])).collect())
}
