//! The scan's identity graph: which chains (tracklets joined by continuity, join) are one person.
//! (Port of `mojimask/scan/graph.py`.)
//!
//! Evidence between two chains A and B (A before B, never on screen together):
//!   face   mean SFace embedding of each chain's frontal faces (faces::frontal: comparable views only), when both have
//!          >= MIN_FACES of them. Same person: >= SAME_FACE; different: < DIFF_FACE.
//!   room   each detection mapped to frame 0 through the camera motion. A chain "stays put" if its spread is
//!          < STILL head sizes over at least MIN_STILL_S seconds (a moment can't show it). Two chains that stay
//!          put: same spot (< SPOT head sizes apart) is evidence of the same person; > FAR apart, of different people.
//!   never  chains on screen at the same time (any common frame) are different people.
//! Rules, strongest first; merged greedily, a merge refused if any two members of the two groups overlap in time:
//!   1. faces >= SAME_FACE and not both staying put far apart
//!   2. both stay put, same spot, faces not clearly different
//! Everything with some evidence for and none decisive becomes a question: faces in [ASK_FACE, SAME_FACE), or the
//! same spot with too little time to show staying put.
//! Thresholds come from jazz census frames 0-28 (face, room position); frames 32-124 are a partly held-out test of
//! them.
//!
//! The constants' evidence:
//! - SAME_FACE, DIFF_FACE, ASK_FACE: cosine between two chains' mean frontal SFace embeddings. Moved from ArcFace's
//!   0.55 / 0.25 / 0.4 on 2026-11-14 to the values with the same error rates on the jazz census's face groups (1%
//!   different people above SAME, 1% of one person below DIFF; experiments/face_model_compare.py --dump).
//! - MIN_DET: a real person is seen in at least this many frames, with a clear face (clear_face) at least once.
//! - CLEAR_FACE: the face check (faces::measure) scores a detection at least this (measured with SCRFD's face check;
//!   kept unchanged for RetinaFace's after a four-clip bench, 2026-11-14).
//! - MAX_GAP_S: seconds between two pieces for the room-position rules: positions in frame-0 coordinates come from
//!   composing camera motion, which drifts -- on the party clip (50 s, handheld) unrelated pieces 30 s apart landed
//!   on the 'same spot'.
//! - DUP_IOU: two chains this overlapped in every shared frame are one head detected twice.
//! - CROWD (head sizes): a spot identifies a person only if no other real head that could be them came this near it
//!   around the time (party clip: a bearded man and the woman dancing beside him were merged by position; 6 of 18
//!   graded room merges were wrong). 1.0 and 0.75 let 2 of those back in.
//! - SPOT_SIZE: position rules (same spot, appears where someone stayed put, ends where the other starts) need the two
//!   heads to be about the same size: distances are in their average head size, so a box 4x too big (jazz: a 153 px
//!   detection over the dancer, joined to a 38 px woman sitting behind her) made a far-off spot look near, and then
//!   the woman's box at that frame covered -- and hid -- what was inside it. A person who stays put keeps their size;
//!   one who walks toward the camera is joined by their face, not their position.
//! - REACH, SPEED: faces never join two pieces nobody could walk between: within MAX_GAP_S (where room positions
//!   hold), where one ends and the other starts may be at most REACH head sizes apart plus SPEED head sizes a second
//!   (a head ~20 cm: 5/s is a brisk walk). jazz: SFace scored person 32 (leaving at the right edge) and person 42
//!   (entering at the left 0.67 s later, 19 head sizes away) 0.69, over SAME_FACE, and the emoji flew across the
//!   picture. Every right face pair on jazz restarts within 1.7 head sizes; with this, 24/24 right merges kept, the
//!   wrong one gone, questions 7 -> 6 (the one between different people gone); the same for SPEED 3 to 8
//!   (2026-11-14).
//!
//! Known (jazz, 2026-11-20, frames from the video): census person 12 ends up as two people, boxes 8 and 357. At frame
//! 106 the detector boxed that head twice: box 395 (the end of chain 357-363-395) and box 397, one frame long and half
//! on the head (IoU 0.27 with 395: under DUP_IOU, so two heads "on screen together"). join's follow back from the
//! next tracklet (415) lands on 397, not 395, and on the video's pixels 397's own follow lands on 415 too, so 397
//! joins the chain that carries on. Both chains then match person 8's face (0.86); the first merged refuses the
//! second. On the JPEG copies' pixels 397's follow drifted, nothing joined it, and both merged. Not fixed: no
//! threshold here does it without giving up something measured -- SAME_FACE 0.62 to 0.70 and ASK_FACE 0.53, 0.61
//! give the same benchmark line (asked 14, split 1); SAME_FACE 0.60 asks 12 and still splits, under the 1% it was
//! set by; CLEAR_FACE 0.55 asks 13 (one overlap question settled, rightly) and 0.40 asks 15, one question either
//! way on one clip. The benchmark's bench_last.json now names who is split ("people").
//!
//! (The Python module's census diagnostics -- truth_report, the command line -- aren't ported.)
use crate::faces::{self, Faces};
use crate::geom::{self, Cs};
use crate::join::Chains;
use crate::np::{self, M23, M33};
use crate::pyset::PySet;
use crate::tracklets::Tracklets;
use crate::viz::Viz;
use indexmap::IndexMap;
use mm_core::pyjson::{Dict, Py};
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub const MIN_FACES: usize = 2;
pub const SAME_FACE: f64 = 0.66;
pub const DIFF_FACE: f64 = 0.33;
pub const ASK_FACE: f64 = 0.57;
pub const STILL: f64 = 0.5;
pub const MIN_STILL_S: f64 = 1.0;
pub const SPOT: f64 = 0.5;
pub const FAR: f64 = 2.0;
pub const MIN_DET: usize = 6;
pub const CLEAR_FACE: f64 = 0.5;
pub const MAX_GAP_S: f64 = 4.0;
pub const DUP_IOU: f64 = 0.4;
pub const CROWD: f64 = 1.5;
pub const SPOT_SIZE: f64 = 1.6;
pub const REACH: f64 = 2.0;
pub const SPEED: f64 = 5.0;

/// One chain's summary (graph.chain_info).
#[derive(Clone, Debug)]
pub struct ChainInfo {
    pub ts: BTreeSet<i64>,
    pub a: i64,
    pub b: i64,
    pub obs: IndexMap<i64, Cs>,
    pub score: IndexMap<i64, f64>,
    pub size: f64,
    pub ctr: [f64; 2],
    pub end: [f64; 2],
    pub start: [f64; 2],
    pub still: bool,
    pub spread: f64,
    pub face: Option<Vec<f32>>,
    pub nfaces: usize,
    pub fps: f64,
    pub room: BTreeMap<i64, [f64; 2]>,
}

/// graph.clear_face: a clear face at least once among the detections of chains ms.
pub fn clear_face<'a>(fs: &Faces, chains: &Chains, tl: &Tracklets, ms: impl IntoIterator<Item = i64>) -> bool {
    ms.into_iter().any(|c| {
        chains[&c].iter().any(|k| tl[k].obs.keys().any(|t| faces::get(fs, *k, *t).map(|f| f.score).unwrap_or(0.0) >= CLEAR_FACE))
    })
}

fn median_col(rows: &[[f64; 3]], j: usize) -> f64 {
    np::median(&rows.iter().map(|r| r[j]).collect::<Vec<_>>())
}

/// graph.chain_info.
pub fn chain_info(tl: &Tracklets, chains: &Chains, fs: &Faces, mats: &[M23], fps: f64) -> IndexMap<i64, ChainInfo> {
    let mut info = IndexMap::new();
    let mut inv: HashMap<i64, M33> = HashMap::new();
    for (&c, seq) in chains {
        let mut obs: IndexMap<i64, Cs> = IndexMap::new();
        let mut score: IndexMap<i64, f64> = IndexMap::new();
        for k in seq {
            for (t, x) in &tl[k].obs {
                obs.insert(*t, *x);
                score.insert(*t, tl[k].score[t]);
            }
        }
        let mut ts: Vec<i64> = obs.keys().copied().collect();
        ts.sort();
        let room: Vec<[f64; 3]> = ts
            .iter()
            .map(|t| {
                let (xy, s) = geom::to_room(mats, *t, &obs[t], &mut inv);
                [xy[0], xy[1], s]
            })
            .collect();
        let size = median_col(&room, 2);
        let ctr = [median_col(&room, 0), median_col(&room, 1)];
        let d: Vec<f64> = room.iter().map(|r| np::hypot(r[0] - ctr[0], r[1] - ctr[1])).collect();
        let spread = np::median(&d) / size;
        let embs: Vec<&[f32]> = seq
            .iter()
            .flat_map(|k| tl[k].obs.keys().map(move |t| (*k, *t)))
            .filter_map(|(k, t)| {
                let f = faces::get(fs, k, t);
                if faces::frontal(f) {
                    Some(f.unwrap().emb.as_ref().unwrap().as_slice())
                } else {
                    None
                }
            })
            .collect();
        let face = if embs.len() >= MIN_FACES { Some(np::unit_f32(&np::mean_rows_f32(&embs))) } else { None };
        let n = room.len();
        info.insert(c, ChainInfo {
            ts: ts.iter().copied().collect(),
            a: ts[0],
            b: ts[n - 1],
            obs,
            score,
            size,
            ctr,
            end: [room[n - 1][0], room[n - 1][1]],
            start: [room[0][0], room[0][1]],
            still: spread < STILL && (ts[n - 1] - ts[0]) as f64 / fps >= MIN_STILL_S,
            spread,
            nfaces: embs.len(),
            face,
            fps,
            room: ts.iter().zip(&room).map(|(t, r)| (*t, [r[0], r[1]])).collect(),
        });
    }
    info
}

/// Python's f"{x:.nf}".
pub fn f(x: f64, n: usize) -> String {
    format!("{:.*}", n, x)
}

/// What the evidence says about two chains: "same", "different", "ask" or "none", a score, and why.
#[derive(Clone, Debug)]
pub struct Ev {
    pub d: &'static str,
    pub s: f64,
    pub why: String,
}

fn ev(d: &'static str, s: f64, why: String) -> Ev {
    Ev { d, s, why }
}

/// graph.evidence: for chains A, B that don't overlap in time.
pub fn evidence(a: &ChainInfo, b: &ChainInfo) -> Ev {
    let sim = match (&a.face, &b.face) {
        (Some(x), Some(y)) => Some(np::sdot(x, y) as f64),
        _ => None,
    };
    let size = (a.size + b.size) / 2.0;
    let (first, second) = if a.a < b.a { (a, b) } else { (b, a) };
    let gap_d = np::hypot(first.end[0] - second.start[0], first.end[1] - second.start[1]) / size;
    let ctr_d = np::hypot(a.ctr[0] - b.ctr[0], a.ctr[1] - b.ctr[1]) / size;
    let both_still = a.still && b.still;
    let gap_s = (second.a - first.b) as f64 / a.fps;
    let near_in_time = gap_s <= MAX_GAP_S;
    if let Some(sim) = sim {
        if sim >= ASK_FACE && near_in_time && gap_d > REACH + SPEED * gap_s {
            return ev("different", 0.0, format!("faces {}, but {} head sizes apart {} s later", f(sim, 2), f(gap_d, 1), f(gap_s, 2)));
        }
        if sim >= SAME_FACE && !(both_still && ctr_d > FAR) {
            return ev("same", 2.0 + sim, format!("faces {}", f(sim, 2)));
        }
    }
    let (mx, mn) = (a.size.max(b.size), a.size.min(b.size));
    if mx > SPOT_SIZE * mn {
        if let Some(sim) = sim {
            if sim < DIFF_FACE {
                return ev("different", 0.0, format!("faces {}", f(sim, 2)));
            }
            if sim >= ASK_FACE {
                return ev("ask", sim + 0.0, format!("faces {}, heads {}x apart in size", f(sim, 2), f(mx / mn, 1)));
            }
        }
        return ev("none", 0.0, String::new());
    }
    let not_diff = sim.map(|s| s >= DIFF_FACE).unwrap_or(true);
    if both_still && near_in_time && ctr_d < SPOT && not_diff {
        return ev("same", 1.0 + (SPOT - ctr_d), format!("same spot ({} apart), both stay put", f(ctr_d, 2)));
    }
    let anchor_still = (a.still && b.spread < STILL) || (b.still && a.spread < STILL);
    if anchor_still && !both_still && ctr_d < SPOT && near_in_time && not_diff {
        return ev("same", 0.5 + (SPOT - ctr_d), format!("appears where a person who stays put was ({} apart)", f(ctr_d, 2)));
    }
    if let Some(sim) = sim {
        if sim < DIFF_FACE {
            return ev("different", 0.0, format!("faces {}", f(sim, 2)));
        }
    }
    if both_still && near_in_time && ctr_d > FAR {
        return ev("different", 0.0, format!("{} head sizes apart, both stay put", f(ctr_d, 1)));
    }
    if sim.map(|s| s >= ASK_FACE).unwrap_or(false) || (gap_d < SPOT && !both_still) {
        let s = sim.unwrap_or(0.0) + (1.0 - if 1.0 < gap_d { 1.0 } else { gap_d });
        let lead = match sim {
            Some(sim) => format!("faces {}", f(sim, 2)),
            None => "no comparable faces".to_string(),
        };
        return ev("ask", s, format!("{lead}, {} apart where one ends and the other starts", f(gap_d, 2)));
    }
    ev("none", 0.0, String::new())
}

/// The graph's state, as later rules read it (graph.run(state=True)'s `st`).
pub struct State {
    pub info: IndexMap<i64, ChainInfo>,
    pub members: IndexMap<i64, PySet>,
    pub chains: Chains,
    pub tl: Tracklets,
    pub fs: Faces,
    pub mats: Vec<M23>,
    pub fps: f64,
    /// graph.run's result counts
    pub res: Vec<(String, i64)>,
    /// questions.groups_data's cache of (face, body) by group
    pub looks: HashMap<Vec<i64>, (Option<Vec<f32>>, Option<Vec<f32>>)>,
}

impl State {
    pub fn same_head(&self, x: i64, y: i64) -> bool {
        let (ix, iy) = (&self.info[&x], &self.info[&y]);
        let mut common = ix.ts.intersection(&iy.ts).peekable();
        common.peek().is_some()
            && common.all(|t| geom::iou(&geom::from_cs(&ix.obs[t]), &geom::from_cs(&iy.obs[t])) > DUP_IOU)
    }
    /// On screen at the same time, as two different heads.
    pub fn apart(&self, x: i64, y: i64) -> bool {
        self.info[&x].ts.intersection(&self.info[&y].ts).next().is_some() && !self.same_head(x, y)
    }
    pub fn any_apart(&self, a: &PySet, b: &PySet) -> bool {
        a.iter().any(|x| b.iter().any(|y| self.apart(x, y)))
    }
    /// A group's boxes: its chains' detections, walked in set order (a later chain's box wins a shared frame).
    pub fn group_obs(&self, ms: &PySet) -> IndexMap<i64, Cs> {
        let mut obs = IndexMap::new();
        for c in ms.iter() {
            for (t, x) in &self.info[&c].obs {
                obs.insert(*t, *x);
            }
        }
        obs
    }
    pub fn clear_face(&self, ms: &PySet) -> bool {
        clear_face(&self.fs, &self.chains, &self.tl, ms.iter())
    }
    pub fn n_obs(&self, ms: &PySet) -> usize {
        ms.iter().map(|c| self.info[&c].ts.len()).sum()
    }
}

/// graph.run(state=True): the graph's groups and state.
pub fn run(tl: Tracklets, chains: Chains, fs: Faces, mats: Vec<M23>, fps: f64, viz: &mut Viz) -> State {
    let info = chain_info(&tl, &chains, &fs, &mats, fps);
    let keys: Vec<i64> = {
        let mut k: Vec<i64> = info.keys().copied().collect();
        k.sort();
        k
    };
    let mut st = State { info, members: IndexMap::new(), chains, tl, fs, mats, fps, res: vec![], looks: HashMap::new() };
    let mut group: HashMap<i64, i64> = keys.iter().map(|c| (*c, *c)).collect();
    st.members = keys.iter().map(|c| (*c, PySet::of(&[*c]))).collect();
    fn find(group: &HashMap<i64, i64>, mut c: i64) -> i64 {
        while group[&c] != c {
            c = group[&c];
        }
        c
    }
    let real = |st: &State, r: i64| -> bool {
        let ms = &st.members[&r];
        st.n_obs(ms) >= MIN_DET && st.clear_face(ms)
    };
    let mut at: BTreeMap<i64, Vec<(i64, [f64; 2])>> = BTreeMap::new();
    for c in &keys {
        for (t, p) in &st.info[c].room {
            at.entry(*t).or_default().push((*c, *p));
        }
    }
    let g_apart = |st: &State, group: &HashMap<i64, i64>, c: i64, r: i64| -> bool { st.any_apart(&st.members[&find(group, c)], &st.members[&r]) };
    let w = mm_core::pyjson::py_round_int(fps);
    let crowded = |st: &State, group: &HashMap<i64, i64>, x: i64, y: i64| -> bool {
        let (first, second) = if st.info[&x].a < st.info[&y].a { (x, y) } else { (y, x) };
        let (fi, si) = (&st.info[&first], &st.info[&second]);
        let size = (fi.size + si.size) / 2.0;
        for t in fi.b - w..si.a + w + 1 {
            for (c, p) in at.get(&t).map(|v| v.as_slice()).unwrap_or(&[]) {
                let fc = find(group, *c);
                if fc == find(group, x) || fc == find(group, y) || !real(st, fc) {
                    continue;
                }
                if t < si.a
                    && np::hypot(p[0] - si.start[0], p[1] - si.start[1]) < CROWD * size
                    && !g_apart(st, group, *c, find(group, second))
                    && g_apart(st, group, *c, find(group, first))
                {
                    return true;
                }
                if t > fi.b
                    && np::hypot(p[0] - fi.end[0], p[1] - fi.end[1]) < CROWD * size
                    && !g_apart(st, group, *c, find(group, first))
                    && g_apart(st, group, *c, find(group, second))
                {
                    return true;
                }
            }
        }
        false
    };
    let mut pairs: Vec<(&'static str, f64, i64, i64, String)> = vec![];
    for (i, &a) in keys.iter().enumerate() {
        for &b in &keys[i + 1..] {
            if st.info[&a].ts.intersection(&st.info[&b].ts).next().is_some() {
                if st.same_head(a, b) {
                    pairs.push(("same", 3.0, a, b, "one head detected twice".into()));
                }
                continue;
            }
            let e = evidence(&st.info[&a], &st.info[&b]);
            if e.d != "none" {
                pairs.push((e.d, e.s, a, b, e.why));
            }
        }
    }
    let mut ch = Dict::new();
    for c in &keys {
        ch.insert(c.to_string(), Py::List(st.chains[c].iter().map(|k| Py::Int(*k)).collect()));
    }
    viz.emit("chains", vec![("chains", Py::Dict(ch))]);
    let n = keys.len() as i64;
    viz.emit("pairs", vec![
        ("compared", Py::Int(n * (n - 1) / 2)),
        ("same", Py::Int(pairs.iter().filter(|p| p.0 == "same").count() as i64)),
        ("ask", Py::Int(pairs.iter().filter(|p| p.0 == "ask").count() as i64)),
    ]);
    let mut same: Vec<_> = pairs.iter().filter(|p| p.0 == "same").cloned().collect();
    same.sort_by(|p, q| (-p.1).partial_cmp(&-q.1).unwrap());
    let (mut merges, mut refused) = (0i64, 0i64);
    for (_, s, a, b, why) in same {
        let (ra, rb) = (find(&group, a), find(&group, b));
        if ra == rb {
            continue;
        }
        if st.any_apart(&st.members[&ra], &st.members[&rb]) {
            refused += 1;
            viz.emit("refuse", vec![("a", Py::Int(a)), ("b", Py::Int(b)), ("why", Py::Str(why))]);
            continue;
        }
        if s < 2.0 && why != "one head detected twice" && crowded(&st, &group, a, b) {
            pairs.push(("ask", s, a, b, why + ", but another head was near that spot"));
            continue;
        }
        group.insert(rb, ra);
        let mb = st.members.shift_remove(&rb).unwrap();
        st.members.get_mut(&ra).unwrap().update(&mb);
        merges += 1;
        viz.emit("merge", vec![("a", Py::Int(a)), ("b", Py::Int(b)), ("why", Py::Str(why))]);
    }
    let mut asks: Vec<_> = pairs.iter().filter(|p| p.0 == "ask").cloned().collect();
    asks.sort_by(|p, q| (-p.1).partial_cmp(&-q.1).unwrap());
    let mut seen_q = std::collections::HashSet::new();
    let mut n_asks = 0i64;
    for (_, _, a, b, why) in asks {
        let (ra, rb) = (find(&group, a), find(&group, b));
        let key = (ra.min(rb), ra.max(rb));
        if ra == rb || seen_q.contains(&key) || !(real(&st, ra) && real(&st, rb)) {
            continue;
        }
        if st.any_apart(&st.members[&ra], &st.members[&rb]) {
            continue;
        }
        seen_q.insert(key);
        n_asks += 1;
        viz.emit("ask", vec![("a", Py::Int(a)), ("b", Py::Int(b)), ("why", Py::Str(why))]);
    }
    st.res = vec![
        ("chains".into(), keys.len() as i64),
        ("people found".into(), st.members.len() as i64),
        ("merges".into(), merges),
        ("refused (on screen together)".into(), refused),
        ("questions".into(), n_asks),
    ];
    st
}

pub fn members_json(members: &IndexMap<i64, PySet>) -> Py {
    Py::List(members.iter().map(|(r, ms)| Py::List(vec![Py::Int(*r), Py::List(ms.iter().map(Py::Int).collect())])).collect())
}

/// The stage dump (stage_graph.json).
pub fn graph_json(st: &State) -> Py {
    let fl = |v: &[f64]| Py::List(v.iter().map(|x| Py::Float(*x)).collect());
    let mut info = Dict::new();
    for (c, i) in &st.info {
        let mut d = Dict::new();
        d.insert("a", Py::Int(i.a));
        d.insert("b", Py::Int(i.b));
        d.insert("size", Py::Float(i.size));
        d.insert("ctr", fl(&i.ctr));
        d.insert("end", fl(&i.end));
        d.insert("start", fl(&i.start));
        d.insert("still", Py::Bool(i.still));
        d.insert("spread", Py::Float(i.spread));
        d.insert("face", i.face.as_ref().map(|e| Py::List(e.iter().map(|v| Py::Float(*v as f64)).collect())).unwrap_or(Py::Null));
        d.insert("nfaces", Py::Int(i.nfaces as i64));
        info.insert(c.to_string(), Py::Dict(d));
    }
    let mut d = Dict::new();
    d.insert("res", Py::Dict(st.res.iter().map(|(k, v)| (k.clone(), Py::Int(*v))).collect()));
    d.insert("members", members_json(&st.members));
    d.insert("info", Py::Dict(info));
    Py::Dict(d)
}
