//! Overlap questions: a box stuck to a numbered person's face at the same moment -- a second box on the same face,
//! a box on their neck or hair, or someone partly hidden behind them. The "same person?" questions (questions)
//! can't reach these: two boxes on screen together are always different people to the graph.
//! (Port of `mojimask/scan/overlaps.py`; its legacy grading against out/probe/gt_*.json and the contact sheet aren't
//! ported.)
//!
//! A piece (a group the graph found, numbered or grey) is STUCK to person R when its box overlaps R's box by at least
//! STUCK of its own area in at least half the frames it was detected in. Then, cheapest answer first:
//!   inside    a grey piece sitting >= INSIDE within R's box in every frame it was detected: R's cover already blurs
//!             it, so it isn't drawn while R is covered (whatever it is, nothing is uncovered). Its decision rides on
//!             R's: if R is ever shown, it must be covered again (party clip: 4 of 33 are someone else's face hidden
//!             behind R). Numbered pieces are asked instead -- party piece 516 sat fully inside #416's box and is a
//!             different person in the grades; dropping it would take them off the people list.
//!   asked     otherwise, if detected in >= MIN_ASK frames: "Is this box part of #R's face, not a face, or someone
//!             else?" Numbered pieces first (a wrong one is an extra person in the list), then by detected frames.
//!   short     a grey piece detected in under MIN_ASK frames rides on R's cover like an inside one (SHORT_RIDES). Jazz
//!             census (2026-11-13): of 21, 18 were no face at all and 2 a second box on R's own face, each drawn as its
//!             own emoji for a frame or two; the one other person's face (P6 beside P4) stays under R's cover.
//!             Numbered short pieces are left as they are (still covered).
//! Answers:
//!   same face     the piece joins R: R keeps its own box where both were detected, takes the piece's elsewhere
//!   not a face    the piece is dropped (no cover)
//!   someone else  kept as it is
//!
//! STUCK, INSIDE and MIN_ASK were first set on the party clip, twirl and jazz graded by position; the rules since
//! (SHORT_RIDES, FACE_RULES) were measured on jazz's census (tools/truth/jazz.json) with decision_bench.py, which is
//! how to check any change to them.
//!
//! The face rules (FACE_RULES), the same signal as blips' track rule: a piece that never shows a clear face (face
//! check >= questions CLEAR_FACE) and whose detector median is under SURE_SCORE is not a face: dropped, riding on R's
//! cover. A numbered piece showing its own clear face in >= CLEAR_FRAMES frames and at least half of them is someone
//! else: kept. Jazz census (2026-11-13): 4 of 4 and 4 of 4 of the asked ones; a wrong "someone else" only means two
//! emoji on one face. (Measured with SCRFD's face check; kept unchanged for RetinaFace's after a four-clip bench,
//! 2026-11-14.)
//!
//! Writes every overlap decision as a record to PROJECT/decisions_overlap.json (the Decisions page shows them; your
//! answers are facts, facts.py, applied over these records by facts.answers()).
use crate::evidence::Depth;
use crate::geom::{self, covered, Cs};
use crate::graph::State;
use crate::np;
use crate::pyset::PySet;
use crate::questions::{clear_frames, is_real};
use indexmap::IndexMap;
use mm_core::facts as fx;
use mm_core::pyjson::{py_round, Dict, Py};
use std::collections::BTreeMap;
use std::path::Path;

pub const STUCK: f64 = 0.3;
pub const INSIDE: f64 = 0.8;
pub const MIN_ASK: usize = 3;
pub const SHORT_RIDES: bool = true;
pub const FACE_RULES: bool = true;
pub const SURE_SCORE: f64 = 0.8;
pub const CLEAR_FRAMES: usize = 3;
pub const FILE: &str = "decisions_overlap.json";

pub struct G {
    pub obs: IndexMap<i64, Cs>,
    pub real: bool,
    pub clear: usize,
    pub median_detector: f64,
}

/// overlaps.groups_of.
pub fn groups_of(st: &State, members: &IndexMap<i64, PySet>) -> IndexMap<i64, G> {
    members
        .iter()
        .map(|(r, ms)| {
            let obs = st.group_obs(ms);
            let clear = clear_frames(st, ms);
            let scores: Vec<f64> = ms.iter().flat_map(|c| st.info[&c].score.values().copied().collect::<Vec<_>>()).collect();
            let g = G {
                real: is_real(obs.len(), !clear.is_empty()),
                obs,
                clear: clear.len(),
                median_detector: if scores.is_empty() { 0.0 } else { np::median(&scores) },
            };
            (*r, g)
        })
        .collect()
}

pub fn fl1(c: &Cs) -> Py {
    Py::List(geom::from_cs(c).iter().map(|v| Py::Float(py_round(*v, 1))).collect())
}

pub fn pct(x: f64) -> String {
    format!("{:.0}%", x * 100.0)
}

/// overlaps.shown: frames to show a decision by.
fn shown(g: &G, p: &G) -> Py {
    let mut own: Vec<i64> = g.obs.keys().copied().collect();
    own.sort();
    let both: Vec<i64> = own.iter().copied().filter(|t| p.obs.contains_key(t)).collect();
    let stuck: Vec<i64> = both.iter().copied().filter(|t| covered(&g.obs[t], &p.obs[t]) >= STUCK).collect();
    let ts = if !stuck.is_empty() { stuck } else if !both.is_empty() { both } else { own };
    let mut picks: Vec<i64> = vec![];
    for t in [ts[0], ts[ts.len() / 2], ts[ts.len() - 1]] {
        if !picks.contains(&t) {
            picks.push(t);
        }
    }
    Py::List(
        picks
            .iter()
            .map(|t| {
                let mut d = Dict::new();
                d.insert("t", Py::Int(*t));
                d.insert("piece", fl1(&g.obs[t]));
                d.insert("person", p.obs.get(t).map(fl1).unwrap_or(Py::Null));
                Py::Dict(d)
            })
            .collect(),
    )
}

/// overlaps.decision_record.
pub fn decision_record(piece: i64, person: i64, evidence: Dict, why: String, show: Py, answer: Option<&str>, by: Option<String>) -> Dict {
    let mut d = Dict::new();
    d.insert("id", Py::Str(format!("overlap:{piece}")));
    d.insert("kind", Py::Str("overlap".into()));
    let mut about = Dict::new();
    about.insert("piece", Py::Int(piece));
    about.insert("person", Py::Int(person));
    d.insert("about", Py::Dict(about));
    d.insert("question", Py::Str("Is the green box part of the grey box's face, not a face, or someone else?".into()));
    let mut opts = vec![fx::A_SAME_FACE, fx::A_NOT_FACE, fx::A_SOMEONE_ELSE];
    if by.is_some() {
        opts.extend(["covered by the person's box (while they are)", "left as it is"]);
    }
    d.insert("options", Py::List(opts.into_iter().map(Py::from).collect()));
    d.insert("evidence", Py::Dict(evidence));
    d.insert("why", Py::Str(why));
    d.insert("show", show);
    let a = answer.map(Py::from).unwrap_or(Py::Null);
    let b = by.map(Py::Str).unwrap_or(Py::Null);
    d.insert("answer", a.clone());
    d.insert("by", b.clone());
    d.insert("rule_answer", a);
    d.insert("rule_by", b);
    d
}

/// overlaps.find: (inside {piece: R}, decisions).
pub fn find(groups: &IndexMap<i64, G>, mut depth: Option<&mut Box<dyn Depth>>) -> (IndexMap<i64, i64>, Vec<Dict>) {
    let mut at: BTreeMap<i64, Vec<(i64, Cs)>> = BTreeMap::new();
    for (r, g) in groups {
        if g.real {
            for (t, o) in &g.obs {
                at.entry(*t).or_default().push((*r, *o));
            }
        }
    }
    let mut inside = IndexMap::new();
    let mut asks: Vec<i64> = vec![];
    let mut decisions: Vec<Dict> = vec![];
    for (&r, g) in groups {
        let n = g.obs.len();
        let mut qs = PySet::new();
        for t in g.obs.keys() {
            for (q, _) in at.get(t).map(|v| v.as_slice()).unwrap_or(&[]) {
                if *q != r {
                    qs.add(*q);
                }
            }
        }
        let mut sorted_obs: Vec<i64> = g.obs.keys().copied().collect();
        sorted_obs.sort();
        let mut best: Option<(f64, f64, i64, f64)> = None;
        for big_r in qs.iter() {
            let ob = &groups[&big_r].obs;
            let cov: Vec<f64> = sorted_obs.iter().map(|t| if let Some(o) = ob.get(t) { covered(&g.obs[t], o) } else { 0.0 }).collect();
            let frac = cov.iter().filter(|c| **c >= STUCK).count() as f64 / n as f64;
            let low = cov.iter().copied().fold(f64::INFINITY, |a, b| if b < a { b } else { a });
            let better = match best {
                None => true,
                Some((bf, bl, _, _)) => frac > bf || (frac == bf && low > bl),
            };
            if better {
                let pos: Vec<f64> = cov.iter().copied().filter(|c| *c > 0.0).collect();
                best = Some((frac, low, big_r, if pos.is_empty() { 0.0 } else { np::median(&pos) }));
            }
        }
        let Some((frac, low, big_r, med)) = best else { continue };
        if frac < 0.5 {
            continue;
        }
        let mut ev = Dict::new();
        ev.insert("numbered", Py::Bool(g.real));
        ev.insert("frames", Py::Int(n as i64));
        ev.insert("frames_overlapping", Py::Float(py_round(frac, 2)));
        ev.insert("median_overlap", Py::Float(py_round(med, 2)));
        // min(cov) is a numpy float64 (covered() of numpy boxes) unless it is 0.0: numpy's round
        ev.insert("least_overlap", Py::Float(np::np_round(low, 2)));
        ev.insert("first", Py::Int(*g.obs.keys().min().unwrap()));
        ev.insert("last", Py::Int(*g.obs.keys().max().unwrap()));
        let why = format!("{}, {n} frames, overlaps the person's box in {} of them (median {} of its box)",
                          if g.real { "numbered" } else { "grey" }, pct(frac), pct(med));
        let show = shown(g, &groups[&big_r]);
        if low >= INSIDE && !g.real {
            inside.insert(r, big_r);
            decisions.push(decision_record(r, big_r, ev, why, show, Some("covered by the person's box (while they are)"),
                                           Some(format!("rule: inside >= {}", INSIDE))));
        } else if n >= MIN_ASK && FACE_RULES && g.clear == 0 && g.median_detector < SURE_SCORE {
            inside.insert(r, big_r);
            ev.insert("clear_face_frames", Py::Int(0));
            decisions.push(decision_record(r, big_r, ev, why, show, Some(fx::A_NOT_FACE),
                                           Some(format!("rule: never a clear face (median detector under {})", SURE_SCORE))));
        } else if n >= MIN_ASK && FACE_RULES && g.real && g.clear >= CLEAR_FRAMES && g.clear as f64 >= n as f64 / 2.0 {
            ev.insert("clear_face_frames", Py::Int(g.clear as i64));
            decisions.push(decision_record(r, big_r, ev, why, show, Some(fx::A_SOMEONE_ELSE),
                                           Some(format!("rule: its own clear face in {} of {n} frames", g.clear))));
        } else if n >= MIN_ASK {
            if let Some(dep) = depth.as_mut() {
                let mut gaps = vec![];
                for t in g.obs.keys() {
                    if let Some(o) = groups[&big_r].obs.get(t) {
                        let x = (dep.near(*t, &g.obs[t], 0.25) - dep.near(*t, o, 0.25)).abs();
                        if x.is_finite() {
                            gaps.push(x);
                        }
                    }
                }
                if !gaps.is_empty() {
                    ev.insert("depth_gap", Py::Float(py_round(np::median(&gaps), 2)));
                }
            }
            asks.push(r);
            decisions.push(decision_record(r, big_r, ev, why, show, None, None));
        } else if !g.real && SHORT_RIDES {
            inside.insert(r, big_r);
            decisions.push(decision_record(r, big_r, ev, why, show, Some("covered by the person's box (while they are)"),
                                           Some(format!("rule: under {MIN_ASK} frames, on the person's box"))));
        } else {
            decisions.push(decision_record(r, big_r, ev, why, show, Some("left as it is"), Some(format!("rule: under {MIN_ASK} frames"))));
        }
    }
    let mut ranked = asks.clone();
    ranked.sort_by_key(|r| (!groups[r].real, -(groups[r].obs.len() as i64)));
    let rank: std::collections::HashMap<i64, i64> = ranked.iter().enumerate().map(|(i, r)| (*r, i as i64 + 1)).collect();
    for d in decisions.iter_mut() {
        let p = d.get("about").unwrap().get("piece").as_i64().unwrap();
        d.insert("rank", rank.get(&p).map(|x| Py::Int(*x)).unwrap_or(Py::Null));
    }
    decisions.sort_by_key(|d| {
        let rk = d.get("rank").unwrap().as_i64();
        (rk.is_none(), rk.unwrap_or(0), -d.get("evidence").unwrap().get("frames").as_i64().unwrap())
    });
    (inside, decisions)
}

/// overlaps.save_records: {"decisions": [...]}, indent 1.
pub fn save_records(path: &Path, decisions: &[Dict]) -> Result<(), String> {
    let mut d = Dict::new();
    d.insert("decisions", Py::List(decisions.iter().map(|x| Py::Dict(x.clone())).collect()));
    mm_core::pyjson::write_atomic(path, &Py::Dict(d).dumps_indent(1)).map_err(|e| e.to_string())
}

/// overlaps.auto: the automatic answers applied to the groups (pieces inside a person's box, pieces never a face
/// dropped), records saved unless write is false.
pub fn auto(proj: &Path, st: &State, members: &mut IndexMap<i64, PySet>, depth: Option<&mut Box<dyn Depth>>, write: bool)
            -> Result<Vec<Dict>, String> {
    let (inside, decisions) = find(&groups_of(st, members), depth);
    if write {
        save_records(&proj.join(FILE), &decisions)?;
    }
    for r in inside.keys() {
        members.shift_remove(r);
    }
    Ok(decisions)
}
