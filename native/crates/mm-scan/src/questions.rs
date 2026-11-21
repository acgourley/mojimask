//! The scan's same-person questions: which pairs of people to ask about after the graph has merged what it can.
//! (Port of `mojimask/scan/questions.py`; the OSNet body feature is behind [`crate::evidence::Appearance`].)
//!
//! A question is worth asking where a "yes" would join pieces of a real person and a "yes" is plausible -- not
//! wherever evidence is merely mixed. For every pair of real groups (>= MIN_DET detections, a clear face at least once)
//! that are never on screen together:
//!   motion  at every hand-over between the two (one group's last sighting before the other's next), the room
//!           distance must be reachable: <= 1 head size + V head sizes per frame of the gap; otherwise not asked.
//!           Closer than that -> evidence for.
//!   face    frontal-view SFace means (comparable views only): face_term, 3.6 x (sim - 0.49), within [-2, 1.5].
//!   body    OSNet on the person's own skeleton box (a hint: 0.92 AUC on jazz): body_term, 5 x (sim - 0.8),
//!           within [-1, 1].
//!           Beyond graph::MAX_GAP_S the closeness is scaled by MAX_GAP_S / gap: room positions come from composed
//!           camera motion, which drifts (graph.rs), so over a long gap "close" is mostly luck. (2026-11-17, the 50 s
//!           party clip at 4K: most same-person questions were body-only pairs a long gap apart, carried over the ask
//!           line by a motion term near +0.93. Measured in native/tools/bench/rule_bench.py; numbers in the commit.)
//! BODY_ONLY: two people with no face comparison (no frontal view of one of them) are asked about only when their
//!           bodies match at 0.9 or more (term +0.5): at a party, clothes alike are common and the body model is a
//!           hint, not an identity (2026-11-17: jazz census, all 10 truly-same pairs kept; numbers in the commit).
//! Motion alone never makes a question (in a crowd a neighbour fits as well); asked if the sum is > 0, best first by
//! sum x the smaller group's size (how much a yes would join). After a yes the two groups merge and the candidates are
//! recomputed. same turns these into the Decisions page's records; the editor's "your box" questions
//! (review_server.likeness) use the same face and body terms.
//!
//! face_term: 0.49 is where ArcFace's 0.3 was, the same share of different people above it on the jazz census; the
//! slope scaled by the two models' same/different spreads, 0.30 -> 0.33: 4 -> 3.6.
//!
//! (The Python module's answer simulation from census labels -- its command line -- isn't ported.)
use crate::evidence::Evidence;
use crate::faces;
use crate::geom::{self, Cs};
use crate::graph::{self, State};
use crate::np::{self, M33};
use crate::pose::{self, Pose};
use crate::pyset::PySet;
use indexmap::IndexMap;
use std::collections::{BTreeSet, HashMap};

pub(crate) const V: f64 = 0.2;
pub(crate) const BODY_ONLY: f64 = 0.9;

pub fn face_term(sim: f64) -> f64 {
    (3.6 * (sim - 0.49)).max(-2.0).min(1.5)
}

pub fn body_term(sim: f64) -> f64 {
    (5.0 * (sim - 0.8)).max(-1.0).min(1.0)
}

/// questions.clear_frames: the frames where a group (its chains ms) shows a clear face.
pub(crate) fn clear_frames(st: &State, ms: &PySet) -> BTreeSet<i64> {
    let mut out = BTreeSet::new();
    for c in ms.iter() {
        for k in &st.chains[&c] {
            for t in st.tl[k].obs.keys() {
                if faces::get(&st.fs, *k, *t).map(|f| f.score).unwrap_or(0.0) >= graph::CLEAR_FACE {
                    out.insert(*t);
                }
            }
        }
    }
    out
}

/// questions.is_real: enough detections and a clear face at least once.
pub(crate) fn is_real(n_detections: usize, clear_face: bool) -> bool {
    n_detections >= graph::MIN_DET && clear_face
}

/// One group as the questions see it (questions.groups_data).
#[derive(Clone, Debug)]
pub(crate) struct Group {
    pub ms: PySet,
    pub obs: IndexMap<i64, Cs>,
    pub ts: Vec<i64>,
    pub face: Option<Vec<f32>>,
    pub body: Option<Vec<f32>>,
    pub real: bool,
    pub room: HashMap<i64, ([f64; 2], f64)>,
}

/// questions.groups_data: the real groups; fragments: also the shorter ones (> 2 detections), marked real false.
pub(crate) fn groups_data(st: &mut State, poses: &[Vec<Pose>], ev: &mut Evidence, fragments: bool) -> IndexMap<i64, Group> {
    let mut inv: HashMap<i64, M33> = HashMap::new();
    let mut out = IndexMap::new();
    let mut groups = vec![];
    for (&r, ms) in &st.members {
        let obs = st.group_obs(ms);
        let real = is_real(obs.len(), !clear_frames(st, ms).is_empty());
        if !real && !(fragments && obs.len() > 2) {
            continue;
        }
        let mut ts: Vec<i64> = obs.keys().copied().collect();
        ts.sort();
        groups.push((r, ms.clone(), obs, ts, real));
    }
    // the looks not yet measured (st.looks keeps them by member set), each group's on its own thread: every group's
    // crops are still one OSNet batch, so the features are the same as measured one group after another
    let todo: Vec<usize> = (0..groups.len()).filter(|&i| !st.looks.contains_key(&groups[i].1.sorted())).collect();
    let ev_: &Evidence = ev;
    let st_: &State = st;
    let measured = crate::join::par_map(&todo, crate::join::threads(), |&i| {
        let (_, ms, obs, ts, _) = &groups[i];
        looks(st_, ms, obs, ts, poses, ev_)
    });
    for (i, l) in todo.into_iter().zip(measured) {
        st.looks.insert(groups[i].1.sorted(), l);
    }
    for (r, ms, obs, ts, real) in groups {
        let key = ms.sorted();
        let (face, body) = st.looks[&key].clone();
        let room = ts.iter().map(|t| (*t, geom::to_room(&st.mats, *t, &obs[t], &mut inv))).collect();
        out.insert(r, Group { ms: ms.copy(), obs, ts, face, body, real, room });
    }
    out
}

/// questions._looks: a group's mean frontal SFace embedding and mean OSNet body feature, each None with fewer than two.
fn looks(st: &State, ms: &PySet, obs: &IndexMap<i64, Cs>, ts: &[i64], poses: &[Vec<Pose>], ev: &Evidence)
         -> (Option<Vec<f32>>, Option<Vec<f32>>) {
    let mut embs: Vec<&[f32]> = vec![];
    for c in ms.iter() {
        for k in &st.chains[&c] {
            for t in st.tl[k].obs.keys() {
                let f = faces::get(&st.fs, *k, *t);
                if faces::frontal(f) {
                    embs.push(f.unwrap().emb.as_ref().unwrap());
                }
            }
        }
    }
    let face = if embs.len() >= 2 { Some(np::unit_f32(&np::mean_rows_f32(&embs))) } else { None };
    let step = (ts.len() / 8).max(1);
    let mut crops: Vec<(i64, [f64; 4])> = vec![];
    for t in ts.iter().step_by(step) {
        let Some(i) = pose::pose_for_head(&poses[*t as usize], &obs[t]) else { continue };
        let p = &poses[*t as usize][i];
        let bx = [p[0], p[1], p[2], p[3]]; // appearance.body_box with a skeleton: its own box
        // appearance.crop: int() truncation; None when empty; used when at least 16 px each way
        let (x1, y1, x2, y2) = (bx[0].trunc() as i64, bx[1].trunc() as i64, bx[2].trunc() as i64, bx[3].trunc() as i64);
        let (h, w) = ((y2 - y1).max(0), (x2 - x1).max(0));
        if h == 0 || w == 0 || h.min(w) < 16 {
            continue;
        }
        crops.push((*t, bx));
    }
    let body = if crops.len() >= 2 {
        let rows: Vec<Vec<f32>> = ev.appearance.bodies(&crops);
        let r: Vec<&[f32]> = rows.iter().map(|v| v.as_slice()).collect();
        Some(np::unit_f32(&np::mean_rows_f32(&r)))
    } else {
        None
    };
    (face, body)
}

/// Python's f"{x:+.2f}".
fn sf(x: f64) -> String {
    format!("{:+.2}", x)
}

/// questions.score: (score, why) or None. motion_only: the least motion score (1 - closeness) that makes a question
/// without face or body evidence (None: never). A fragment right on a person's path is worth asking about even when
/// its face and body are too blurred to compare; two whole people in a crowd are not.
pub(crate) fn score(a: &Group, b: &Group, apart_groups: bool, motion_only: Option<f64>, fps: f64) -> Option<(f64, String)> {
    if apart_groups {
        return None;
    }
    let mut seq: Vec<(i64, u8)> = a.ts.iter().map(|t| (*t, 0u8)).chain(b.ts.iter().map(|t| (*t, 1u8))).collect();
    seq.sort();
    let mut m = 0.0f64;
    for w in seq.windows(2) {
        let ((t0, g0), (t1, g1)) = (w[0], w[1]);
        if g0 == g1 {
            continue;
        }
        let (p0, s0) = if g0 == 0 { a.room[&t0] } else { b.room[&t0] };
        let (p1, s1) = if g1 == 0 { a.room[&t1] } else { b.room[&t1] };
        let d = np::hypot(p0[0] - p1[0], p0[1] - p1[1]) / ((s0 + s1) / 2.0);
        let reach = 1.0 + V * (t1 - t0) as f64;
        if d > reach {
            return None;
        }
        // the fit counts in full within MAX_GAP_S, where room positions hold, and fades as 1/gap beyond it
        let trust = (graph::MAX_GAP_S * fps / (t1 - t0) as f64).min(1.0);
        m = m.max((1.0 - d / reach) * trust);
    }
    let mut parts = vec![];
    let mut s = 0.0 + m;
    parts.push(format!("motion fits ({})", sf(m)));
    let has_face = a.face.is_some() && b.face.is_some();
    let mut body_sim = None;
    if let (Some(x), Some(y)) = (&a.face, &b.face) {
        let sim = np::sdot(x, y) as f64;
        let f = face_term(sim);
        s += f;
        parts.push(format!("faces {} ({})", graph::f(sim, 2), sf(f)));
    }
    if let (Some(x), Some(y)) = (&a.body, &b.body) {
        let sim = np::sdot(x, y) as f64;
        body_sim = Some(sim);
        let f = body_term(sim);
        s += f;
        parts.push(format!("bodies {} ({})", graph::f(sim, 2), sf(f)));
    }
    if parts.len() == 1 && motion_only.map(|mo| m < mo).unwrap_or(true) {
        return None;
    }
    // two whole people with no face comparison: the body alone must match closely (fragments keep their own rule)
    if motion_only.is_none() && !has_face && body_sim.map(|b| b < BODY_ONLY).unwrap_or(false) {
        return None;
    }
    Some((s, parts.join(", ")))
}
