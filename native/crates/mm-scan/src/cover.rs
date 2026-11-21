//! The scan's covers: one box per person per frame they're on screen, detected or not (the decide step's last stage,
//! called by newalgo's tracks_for). (Port of `mojimask/scan/cover.py`; its census scoring command line isn't ported.)
//!
//! For every group the graph found (people and everything else it covers):
//!   detected  the detections themselves
//!   gaps      for real people only (as for ends): between two detections at most attach::MAX_GAP frames apart, follow
//!             forward from the earlier one and backward from the later one (follow), blended by time
//!             (nearer side counts more)
//!   ends      for real people only (>= MIN_DET detections, a clear face at least once): up to END frames past the
//!             first and last detection, followed (a cover stays a moment too long rather than ending early), stopping
//!             when the box leaves the picture, or after END_HOLDS frames of holding still with no evidence of the face
//!             (holding is right mid-track, for someone hidden; at a track's start a held box sat on twirl's father's
//!             legs waiting for the girl to appear).
//!   behind    a held frame on a nearer person's body is a face hidden behind them: not covered (behind), so the
//!             hidden person's emoji isn't drawn over whoever is in front. Kept per person in build()'s "behind".
//!   lost      gaps and ends stop where following loses the face: its corners stop tracking while the rest of the picture
//!             tracks fine (follow's face_motion) -- someone passed in front. Frames neither side reaches are uncovered
//!             and go in "behind" too, so the app doesn't bridge over them.
//!   blips     blips and never-a-face pieces are covered where detected, plus BLIP_HOLD frames either side carried by
//!             the camera (not followed); extending them by following smeared covers over nothing (jazz: 934 vs 62).
//!
//! Flows (for the app's steady emoji, track.steady): how each cover's head moved to the next frame (follow's
//! face_motion); the app smooths a person's boxes after carrying them along this, so a fast real move (jazz: the
//! dancer bobbing a head's height every 8 frames, halved by smoothing that knew only the camera) is followed, and only
//! the detector's wobble is averaged away. Kept where it agrees with the next box (within FLOW_AGREE head sizes, scale
//! within FLOW_SCALE).
use crate::attach::{self, MAX_GAP};
use crate::behind;
use crate::blips;
use crate::evidence::Evidence;
use crate::follow::{self, Clip, FPath};
use crate::geom::{self, Cs, Fused};
use crate::graph::{self, State};
use crate::np;
use crate::overlaps;
use crate::pose::Pose;
use crate::pyset::PySet;
use crate::questions;
use crate::viz::{r as rnd, Viz};
use indexmap::IndexMap;
use mm_core::pyjson::{py_round, Dict, Py};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub const END: i64 = 12;
pub const END_HOLDS: i64 = 2;
pub const MIN_DET: usize = graph::MIN_DET;
pub const LOST_MIN: usize = 3;
pub const BLIP_HOLD: i64 = 4;
pub const BLIP_SCORE: f64 = 0.6;
pub const FLOW_AGREE: f64 = 0.5;
pub const FLOW_SCALE: (f64, f64) = (0.8, 1.25);

/// A Counter, in first-touched order.
#[derive(Default, Debug, Clone)]
pub struct Counter(pub IndexMap<String, i64>);
impl Counter {
    pub fn add(&mut self, k: &str, n: i64) {
        *self.0.entry(k.to_string()).or_insert(0) += n;
    }
}

/// What decide_groups hands on.
pub struct Decided {
    pub st: State,
    pub members0: IndexMap<i64, PySet>,
    pub attached: IndexMap<i64, i64>,
    /// the real groups attachment saw (questions::groups_data), as the stage dump writes them
    pub gd: Py,
}

/// cover.decide_groups: the groups after the graph, attachment and the overlap and blip rules. Records saved unless
/// write is false.
pub fn decide_groups(proj: &Path, mut st: State, poses: Option<&[Vec<Pose>]>, photo: bool, clip: &Clip, ev: &mut Evidence, viz: &mut Viz,
                     write: bool) -> Result<Decided, String> {
    let empty: Vec<Vec<Pose>> = vec![];
    let gd = questions::groups_data(&mut st, poses.unwrap_or(&empty), ev, false);
    let att = attach::attach(&st, &gd);
    let emb = |e: &Option<Vec<f32>>| e.as_ref().map(|v| Py::List(v.iter().map(|x| Py::Float(*x as f64)).collect())).unwrap_or(Py::Null);
    let gd_json = Py::Dict(
        gd.iter()
            .map(|(r, g)| {
                let mut d = Dict::new();
                d.insert("face", emb(&g.face));
                d.insert("body", emb(&g.body));
                d.insert("ms", Py::List(g.ms.iter().map(Py::Int).collect()));
                (r.to_string(), Py::Dict(d))
            })
            .collect(),
    );
    for (f, r) in &att {
        let mf = st.members.shift_remove(f).unwrap();
        st.members.get_mut(r).unwrap().update(&mf);
    }
    let members0: IndexMap<i64, PySet> = st.members.iter().map(|(r, ms)| (*r, ms.copy())).collect();
    let mut members = std::mem::take(&mut st.members);
    overlaps::auto(proj, &st, &mut members, ev.depth.as_mut(), write)?;
    blips::auto(proj, &st, &mut members, poses, photo, clip, ev, write)?;
    blips::auto_tracks(proj, &st, &mut members, write)?;
    st.members = members;
    let mut people = Dict::new();
    for (r, ms) in &st.members {
        people.insert(r.to_string(), Py::List(ms.sorted().into_iter().map(Py::Int).collect()));
    }
    let mut dropped: Vec<i64> = members0.iter().filter(|(r, _)| !st.members.contains_key(*r)).flat_map(|(_, ms)| ms.to_vec()).collect();
    dropped.sort();
    let real: Vec<Py> = st
        .members
        .iter()
        .filter(|(_, ms)| ms.iter().map(|c| st.info[&c].obs.len()).sum::<usize>() >= MIN_DET && st.clear_face(ms))
        .map(|(r, _)| Py::Int(*r))
        .collect();
    viz.emit("people", vec![
        ("people", Py::Dict(people)),
        ("dropped", Py::List(dropped.into_iter().map(Py::Int).collect())),
        ("attached", Py::Int(att.len() as i64)),
        ("real", Py::List(real)),
    ]);
    Ok(Decided { st, members0, attached: att, gd: gd_json })
}

/// cover.cover_blip: covered where detected; a confident one also holds BLIP_HOLD frames either side, carried by the
/// camera.
fn cover_blip(clip: &Clip, obs: &IndexMap<i64, Cs>, cv: &mut IndexMap<i64, Cs>, st: &State, ms: &PySet, how: &mut Counter) {
    let mut ts: Vec<i64> = obs.keys().copied().collect();
    ts.sort();
    for &t in &ts {
        if ts.len() > 2 {
            let mut best = f64::NEG_INFINITY;
            for c in ms.iter() {
                let s = st.info[&c].score.get(&t).copied().unwrap_or(0.0);
                if s > best {
                    best = s;
                }
            }
            if best < BLIP_SCORE {
                continue;
            }
        }
        for d in [1i64, -1] {
            let mut c = obs[&t];
            for k in 1..=BLIP_HOLD {
                let u = t + d * k;
                if !(0 <= u && u < clip.n) {
                    break;
                }
                c = geom::warp_cs(&c, &clip.camera(u - d, u));
                if !cv.contains_key(&u) {
                    cv.insert(u, c);
                    how.add("blip hold", 1);
                }
            }
        }
    }
}

/// cover.cover_person: gaps followed from both sides, ends followed outward. Returns the frames where following lost
/// the face from both sides.
fn cover_person(clip: &Clip, ev: &Evidence, obs: &IndexMap<i64, Cs>, cv: &mut IndexMap<i64, Cs>, low: &[Vec<Fused>], how: &mut Counter)
                -> BTreeSet<i64> {
    let seen = |u: i64, c: &Cs| (u as usize) < low.len() && behind::face_there(c, &low[u as usize]);
    let mut ts: Vec<i64> = obs.keys().copied().collect();
    ts.sort();
    let mut lost = BTreeSet::new();
    for w in ts.windows(2) {
        let (a, b) = (w[0], w[1]);
        if b - a <= 1 || b - a > MAX_GAP {
            continue;
        }
        let fp = follow::follow(clip, ev, a, &obs[&a], 1, b - a - 1, None, true, Some(&seen));
        let bp = follow::follow(clip, ev, b, &obs[&b], -1, b - a - 1, None, true, Some(&seen));
        let (mut fw, mut bw) = (fp.boxes.clone(), bp.boxes.clone());
        if fp.why == "lost" || bp.why == "lost" {
            let miss: BTreeSet<i64> = (a + 1..b).filter(|u| !fw.contains_key(u) && !bw.contains_key(u)).collect();
            if miss.len() < LOST_MIN {
                fw = follow::follow(clip, ev, a, &obs[&a], 1, b - a - 1, None, false, None).boxes;
                bw = follow::follow(clip, ev, b, &obs[&b], -1, b - a - 1, None, false, None).boxes;
            } else {
                lost.extend(miss);
            }
        }
        for u in a + 1..b {
            let wt = (u - a) as f64 / (b - a) as f64;
            let v = match (fw.get(&u), bw.get(&u)) {
                (Some(f), Some(g)) => [(1.0 - wt) * f[0] + wt * g[0], (1.0 - wt) * f[1] + wt * g[1], (1.0 - wt) * f[2] + wt * g[2]],
                (Some(f), None) => *f,
                (None, Some(g)) => *g,
                (None, None) => continue,
            };
            cv.insert(u, v);
            how.add("gap", 1);
        }
    }
    for (start, d) in [(ts[ts.len() - 1], 1i64), (ts[0], -1)] {
        let p: FPath = follow::follow(clip, ev, start, &obs[&start], d, END, None, true, Some(&seen));
        if p.why == "lost" {
            how.add("end lost", 1);
            let (from, to) = (start + d * (p.boxes.len() as i64 + 1), start + d * (END + 1));
            let mut u = from;
            while (d > 0 && u < to) || (d < 0 && u > to) {
                if 0 <= u && u < clip.n {
                    lost.insert(u);
                }
                u += d;
            }
        }
        let mut holds = 0;
        let mut us: Vec<i64> = p.boxes.keys().copied().collect();
        us.sort_by_key(|u| d * u);
        for u in us {
            if p.how[&u] == "hold" {
                holds += 1;
                if holds > END_HOLDS {
                    break;
                }
            }
            if !cv.contains_key(&u) {
                cv.insert(u, p.boxes[&u]);
                how.add("end", 1);
            }
        }
    }
    lost
}

/// The decide step's groups and their covers (cover.build).
pub struct Built {
    pub covers: IndexMap<i64, IndexMap<i64, Cs>>,
    pub how: Counter,
    pub behind: IndexMap<i64, Vec<i64>>,
    pub flow: IndexMap<i64, BTreeMap<i64, [f64; 4]>>,
    pub depth: Option<String>,
    pub d: Decided,
}

/// cover.build.
#[allow(clippy::too_many_arguments)]
pub fn build(proj: &Path, st: State, poses: Option<&[Vec<Pose>]>, photo: bool, clip: &Clip, ev: &mut Evidence, viz: &mut Viz, write: bool)
             -> Result<Built, String> {
    let d = decide_groups(proj, st, poses, photo, clip, ev, viz, write)?;
    let st = &d.st;
    let w = clip.w;
    let spec: Vec<(&str, f64)> = geom::CORE_DETS.iter().map(|(n, _)| (*n, geom::WEAK_DET)).collect();
    let low = geom::fused_frames(&proj.join("dets"), &spec)?;
    let empty: Vec<Vec<Pose>> = vec![];
    let poses_all = poses.unwrap_or(&empty);
    let mut covers = IndexMap::new();
    let mut how = Counter::default();
    let mut behind_ = IndexMap::new();
    // each group's cover is followed on its own thread (join::par_map, results in group order); its counts are kept
    // apart and added in group order, so `how` comes out as when the groups ran one after another
    let groups: Vec<(i64, IndexMap<i64, Cs>)> = st.members.iter().map(|(&r, ms)| (r, st.group_obs(ms))).collect();
    let ev_: &Evidence = ev;
    let followed = crate::join::par_map(&groups, crate::join::threads(), |(r, obs)| {
        let ms = &st.members[r];
        let mut keys: Vec<i64> = obs.keys().copied().collect();
        keys.sort();
        let mut cv: IndexMap<i64, Cs> = keys.iter().map(|t| (*t, obs[t])).collect();
        let mut h = Counter::default();
        let lost = if obs.len() < MIN_DET || !st.clear_face(ms) {
            cover_blip(clip, obs, &mut cv, st, ms, &mut h);
            BTreeSet::new()
        } else {
            cover_person(clip, ev_, obs, &mut cv, &low, &mut h)
        };
        (cv, lost, h)
    });
    for ((r, obs), (mut cv, mut lost, h)) in groups.into_iter().zip(followed) {
        how.add("detected", obs.len() as i64);
        for (k, n) in h.0 {
            how.add(&k, n);
        }
        let gone = behind::behind_frames(&cv, &obs, poses_all, &low, ev.depth.as_mut());
        for u in &gone {
            cv.shift_remove(u);
        }
        lost.retain(|u| !cv.contains_key(u));
        how.add("behind", gone.len() as i64);
        how.add("lost", lost.len() as i64);
        if !gone.is_empty() || !lost.is_empty() {
            let all: BTreeSet<i64> = gone.union(&lost).copied().collect();
            behind_.insert(r, all.into_iter().collect::<Vec<_>>());
        }
        let mut fill_ts: Vec<i64> = cv.keys().copied().filter(|t| !obs.contains_key(t)).collect();
        fill_ts.sort();
        viz.emit("cover", vec![
            ("r", Py::Int(r)),
            ("fill", Py::List(fill_ts.iter().map(|t| Py::List(vec![Py::Int(*t), rnd(cv[t][0] / w, 4)])).collect())),
        ]);
        covers.insert(r, cv);
    }
    let flow = flows(ev, &covers);
    let depth = ev.depth.as_ref().map(|d| d.name().to_string());
    Ok(Built { covers, how, behind: behind_, flow, depth, d })
}

/// cover.flows: how each cover's head moved to the next frame, where that agrees with the next box. Every
/// (cover, frame) step is independent, so the head motions are measured on threads (join::par_map, results in order).
fn flows(ev: &Evidence, covers: &IndexMap<i64, IndexMap<i64, Cs>>) -> IndexMap<i64, BTreeMap<i64, [f64; 4]>> {
    let mut steps: Vec<(i64, i64, Cs, Cs)> = vec![];
    for (r, cv) in covers {
        let mut ts: Vec<i64> = cv.keys().copied().collect();
        ts.sort();
        steps.extend(ts.into_iter().filter_map(|t| cv.get(&(t + 1)).map(|n| (*r, t, cv[&t], *n))));
    }
    let ms = crate::join::par_map(&steps, crate::join::threads(), |(_, t, c, _)| ev.motion.face_motion(*t, t + 1, c).0);
    let mut out: IndexMap<i64, BTreeMap<i64, [f64; 4]>> = covers.keys().map(|r| (*r, BTreeMap::new())).collect();
    for ((r, t, c, n), m) in steps.iter().zip(ms) {
        let Some(m) = m else { continue };
        let q = np::gemv(&m, [c[0], c[1], 1.0]);
        let k = np::det2(m[0][0], m[0][1], m[1][0], m[1][1]).abs().sqrt();
        if np::hypot(q[0] - n[0], q[1] - n[1]) <= FLOW_AGREE * c[2] && FLOW_SCALE.0 <= k && k <= FLOW_SCALE.1 {
            out[r].insert(*t, [py_round(m[0][0], 4), py_round(m[1][0], 4), py_round(m[0][2], 2), py_round(m[1][2], 2)]);
        }
    }
    out
}

pub fn members_stage(d: &Decided) -> Py {
    let mut x = Dict::new();
    x.insert("members", graph::members_json(&d.st.members));
    x.insert("members0", graph::members_json(&d.members0));
    Py::Dict(x)
}

pub fn build_stage(b: &Built) -> Py {
    let fl = |c: &[f64]| Py::List(c.iter().map(|v| Py::Float(*v)).collect());
    let mut x = Dict::new();
    x.insert("covers", Py::Dict(b.covers.iter().map(|(r, cv)| (r.to_string(), Py::Dict(cv.iter().map(|(t, c)| (t.to_string(), fl(c))).collect()))).collect()));
    x.insert("behind", Py::Dict(b.behind.iter().map(|(r, v)| (r.to_string(), Py::List(v.iter().map(|t| Py::Int(*t)).collect()))).collect()));
    x.insert("flow", Py::Dict(b.flow.iter().map(|(r, f)| (r.to_string(), Py::Dict(f.iter().map(|(t, m)| (t.to_string(), fl(m))).collect()))).collect()));
    x.insert("how", Py::Dict(b.how.0.iter().map(|(k, v)| (k.clone(), Py::Int(*v))).collect()));
    x.insert("depth", b.depth.clone().map(Py::Str).unwrap_or(Py::Null));
    Py::Dict(x)
}
