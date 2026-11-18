//! The scan's same-person decision records (kind "same"): two people the scan counted, never on screen together,
//! that may be one person (questions' score: motion fits at every hand-over, plus face and body similarity).
//! Answers are facts (facts.py kinds "same" / "different"), and facts.answers() re-picks the open questions from them
//! each time the page asks, so a join or split made anywhere counts. (Port of `mojimask/scan/same.py`.)
//!
//! Every pair scoring >= MIN_SCORE is recorded (below it -- motion fits but faces or bodies disagree -- nothing is
//! recorded); each person is in at most one open question at a time (best pair first): answering one changes the
//! others (party_clip_c: person 5 was in 8 of 29 candidate pairs). select() makes the scan's own starting picks: pairs
//! scoring under facts.ASK_SCORE (0.8) are answered "different people" by rule (you can still answer otherwise), pairs
//! already joined or kept apart are settled, and the next pair for each person is asked.
//!
//! Fragments: groups too short (or never a clear face) to count as people, covered as someone of their own
//! (party_clip_c: 176 and 181, two bits of the red-hat guy 74, each drew its own emoji on his face with nothing
//! asking). Each is asked against the people it could be (best FRAGMENT_PAIRS first), motion alone enough when it fits
//! closely.
//!
//! With a depth model on (MOJIMASK_DEPTH, depth.py): two people whose heads read the model's "apart" (depth.MODELS) or
//! more apart where they come closest in time are different people, answered by rule (you can still answer otherwise).
//! depth_bench.py: your same-person answers on jazz / party_clip_c / IMG_1043 never read more than 0.12 apart with
//! Depth Anything; at 0.15, 32 of 67 answered questions would have been settled, none wrongly.
use crate::evidence::{Depth, Evidence};
use crate::geom::{covered, Cs};
use crate::graph::{self, State};
use crate::np;
use crate::overlaps::{self, fl1};
use crate::pose::Pose;
use crate::questions::{self, Group};
use crate::viz::Viz;
use indexmap::IndexMap;
use mm_core::facts as fx;
use mm_core::pyjson::{float_repr, py_round, Dict, Py};
use std::collections::{HashMap, HashSet};
use std::path::Path;

pub const MIN_SCORE: f64 = 0.5;
pub const FRAGMENT_PAIRS: usize = 2;
pub const FRAGMENT_MOTION: f64 = 0.5;
pub const FILE: &str = "decisions_same.json";
pub const DEPTH_RULE: &str = "rule: at different depths";
pub const ON_HEAD: f64 = 0.5;

/// same._on_head: every frame the two share, their boxes overlap by >= ON_HEAD of the smaller: one head seen twice.
fn on_head(a: &Group, b: &Group) -> bool {
    a.obs.iter().filter(|(t, _)| b.obs.contains_key(*t)).all(|(t, x)| {
        let y = &b.obs[t];
        covered(x, y).max(covered(y, x)) >= ON_HEAD
    })
}

/// depth.Depth.gap: how far apart two boxes' readings are where they come closest in time (k = 3 frames of each).
fn depth_gap(dep: &mut Box<dyn Depth>, a: &IndexMap<i64, Cs>, b: &IndexMap<i64, Cs>) -> f64 {
    let mut ta: Vec<i64> = a.keys().copied().collect();
    let mut tb: Vec<i64> = b.keys().copied().collect();
    ta.sort();
    tb.sort();
    if ta.is_empty() || tb.is_empty() {
        return f64::NAN;
    }
    let mut best = (i64::MAX, 0, 0);
    for p in &ta {
        for q in &tb {
            let c = ((p - q).abs(), *p, *q);
            if c < best {
                best = c;
            }
        }
    }
    let (_, x, y) = best;
    let mut sa = ta.clone();
    sa.sort_by_key(|t| (t - x).abs());
    let mut sb = tb.clone();
    sb.sort_by_key(|t| (t - y).abs());
    let na: Vec<f64> = sa.iter().take(3).map(|t| dep.near(*t, &a[t], 0.25)).collect();
    let nb: Vec<f64> = sb.iter().take(3).map(|t| dep.near(*t, &b[t], 0.25)).collect();
    (np::nanmedian(&na) - np::nanmedian(&nb)).abs()
}

type Cand = (f64, i64, i64, String);

fn cmp_desc(x: &Cand, y: &Cand) -> std::cmp::Ordering {
    y.0.partial_cmp(&x.0).unwrap().then(y.1.cmp(&x.1)).then(y.2.cmp(&x.2)).then(y.3.cmp(&x.3))
}

/// same.find: the same-person records for the people as they now stand (and the fragments), saved to PROJECT/FILE.
pub fn find(proj: &Path, st: &mut State, poses: &[Vec<Pose>], ev: &mut Evidence, viz: &mut Viz) -> Result<Vec<Dict>, String> {
    let gd = questions::groups_data(st, poses, ev, true);
    let mut cands: Vec<Cand> = vec![];
    let mut frag: IndexMap<i64, Vec<Cand>> = IndexMap::new();
    let keys: Vec<i64> = gd.keys().copied().collect();
    for &r1 in &keys {
        for &r2 in &keys {
            if r1 >= r2 {
                continue;
            }
            let (a, b) = (&gd[&r1], &gd[&r2]);
            if !a.real && !b.real {
                continue;
            }
            let one_frag = !(a.real && b.real);
            let together = st.any_apart(&a.ms, &b.ms) && !(one_frag && on_head(a, b));
            let sc = questions::score(a, b, together, if one_frag { Some(FRAGMENT_MOTION) } else { None }, st.fps);
            if let Some((s, why)) = sc {
                if s >= MIN_SCORE {
                    if one_frag {
                        let (p, f) = if a.real { (r1, r2) } else { (r2, r1) };
                        frag.entry(f).or_default().push((s, p, f, why));
                        continue;
                    }
                    let (p, q) = if a.ts[0] <= b.ts[0] { (r1, r2) } else { (r2, r1) };
                    cands.push((s, p, q, why));
                }
            }
        }
    }
    for (_, mut cs) in frag {
        cs.sort_by(cmp_desc);
        cands.extend(cs.into_iter().take(FRAGMENT_PAIRS));
    }
    cands.sort_by(cmp_desc);
    let mut out: Vec<Dict> = vec![];
    let best = |g: &Group| -> i64 {
        let mut b = g.ts[0];
        for t in &g.ts {
            if g.obs[t][2] > g.obs[&b][2] {
                b = *t;
            }
        }
        b
    };
    for (s, a, b, why) in cands {
        let (ga, gb) = (&gd[&a], &gd[&b]);
        let mut near = ga.ts[0];
        for t in &ga.ts {
            if (t - gb.ts[0]).abs() < (near - gb.ts[0]).abs() {
                near = *t;
            }
        }
        let mut show = vec![];
        let mut seen = vec![];
        for t in [best(ga), near] {
            if seen.contains(&t) {
                continue;
            }
            seen.push(t);
            let mut d = Dict::new();
            d.insert("t", Py::Int(t));
            d.insert("person", fl1(&ga.obs[&t]));
            d.insert("piece", Py::Null);
            show.push(Py::Dict(d));
        }
        let mut seen = vec![];
        for t in [gb.ts[0], best(gb)] {
            if seen.contains(&t) {
                continue;
            }
            seen.push(t);
            let mut d = Dict::new();
            d.insert("t", Py::Int(t));
            d.insert("piece", fl1(&gb.obs[&t]));
            d.insert("person", Py::Null);
            show.push(Py::Dict(d));
        }
        let mut d = Dict::new();
        d.insert("id", Py::Str(format!("same:{a}:{b}")));
        d.insert("kind", Py::from("same"));
        let mut about = Dict::new();
        about.insert("piece", Py::Int(b));
        about.insert("person", Py::Int(a));
        d.insert("about", Py::Dict(about));
        d.insert("question",
                 Py::from("Is the green box the same person as the grey one? (Any part of their head counts: face, side, back or hair.)"));
        d.insert("options", Py::List([fx::A_SAME, fx::A_DIFFERENT, fx::A_JOINED, fx::A_KEPT_APART, fx::A_WAITING].iter().map(|x| Py::from(*x)).collect()));
        let mut e = Dict::new();
        e.insert("score", Py::Float(py_round(s, 2)));
        e.insert("fragment", Py::Bool(!gb.real));
        e.insert("frames", Py::Int((ga.ts.len() + gb.ts.len()) as i64));
        e.insert("first", Py::Int(ga.ts[0]));
        e.insert("last", Py::Int(gb.ts[gb.ts.len() - 1]));
        e.insert("gap_frames", Py::Int(gb.ts[0] - ga.ts[ga.ts.len() - 1]));
        let mut depth_apart = None;
        if let Some(dep) = ev.depth.as_mut() {
            let g = depth_gap(dep, &ga.obs, &gb.obs);
            e.insert("depth_gap", Py::Float(py_round(g, 2)));
            let (_, _, apart) = dep.margins();
            if g >= apart {
                depth_apart = Some(format!("{DEPTH_RULE} ({}: {} apart, the same person reads under {})", dep.name(), graph::f(g, 2), float_repr(apart)));
            }
        }
        d.insert("evidence", Py::Dict(e));
        d.insert("why", Py::Str(why));
        d.insert("show", Py::List(show));
        for k in ["answer", "by", "rule_answer", "rule_by"] {
            d.insert(k, Py::Null);
        }
        if let Some(x) = depth_apart {
            d.insert("depth_apart", Py::Str(x));
        }
        out.push(d);
    }
    select(&mut out);
    let asked: Vec<Py> = out
        .iter()
        .filter(|d| d.get("by").unwrap().is_null())
        .map(|d| {
            let ab = d.get("about").unwrap();
            Py::List(vec![ab.get("person").clone(), ab.get("piece").clone(), d.get("evidence").unwrap().get("score").clone()])
        })
        .collect();
    viz.emit("same", vec![("asked", Py::List(asked)), ("candidates", Py::Int(out.len() as i64))]);
    overlaps::save_records(&proj.join(FILE), &out)?;
    Ok(out)
}

/// same.select: which unanswered pairs are asked now, from your answers so far. Your "same person" answers join people
/// (transitively); then, best score first: a pair already joined is settled ("joined already"); a pair one of your
/// "different people" answers keeps apart is settled ("kept apart"); otherwise it's asked unless either person already
/// has an open question ("waiting": asked once that one is answered).
pub fn select(ds: &mut [Dict]) {
    let mut parent: HashMap<i64, i64> = HashMap::new();
    fn find(parent: &mut HashMap<i64, i64>, mut x: i64) -> i64 {
        parent.entry(x).or_insert(x);
        while parent[&x] != x {
            let gp = parent[&parent[&x]];
            parent.insert(x, gp);
            x = gp;
        }
        x
    }
    let about = |d: &Dict, k: &str| d.get("about").unwrap().get(k).as_i64().unwrap();
    let user = |d: &Dict| d.get("by").and_then(|b| b.as_str()) == Some("user");
    let answer = |d: &Dict| d.get("answer").and_then(|a| a.as_str()).map(|s| s.to_string());
    for d in ds.iter() {
        if user(d) && answer(d).as_deref() == Some(fx::A_SAME) {
            let (p, q) = (find(&mut parent, about(d, "piece")), find(&mut parent, about(d, "person")));
            parent.insert(p, q);
        }
    }
    let mut apart: HashSet<(i64, i64)> = HashSet::new();
    for d in ds.iter() {
        if user(d) && answer(d).as_deref() == Some(fx::A_DIFFERENT) {
            let (p, q) = (find(&mut parent, about(d, "piece")), find(&mut parent, about(d, "person")));
            apart.insert((p.min(q), p.max(q)));
        }
    }
    let mut busy: HashSet<i64> = HashSet::new();
    let mut order: Vec<usize> = (0..ds.len()).collect();
    let score = |d: &Dict| d.get("evidence").unwrap().get("score").as_f64().unwrap();
    order.sort_by(|a, b| (-score(&ds[*a])).partial_cmp(&-score(&ds[*b])).unwrap());
    for i in order {
        if user(&ds[i]) {
            continue;
        }
        let (ga, gb) = (find(&mut parent, about(&ds[i], "person")), find(&mut parent, about(&ds[i], "piece")));
        let d = &mut ds[i];
        let depth_apart = d.get("depth_apart").filter(|x| x.truthy()).cloned();
        let (a, b): (Py, Py) = if depth_apart.is_some() && ga != gb {
            (Py::from(fx::A_DIFFERENT), depth_apart.unwrap())
        } else if score(d) < fx::ASK_SCORE && ga != gb {
            (Py::from(fx::A_DIFFERENT), Py::from(fx::WEAK_RULE))
        } else if ga == gb {
            (Py::from(fx::A_JOINED), Py::from("rule: your other answers make them one person"))
        } else if apart.contains(&(ga.min(gb), ga.max(gb))) {
            (Py::from(fx::A_KEPT_APART), Py::from("rule: you said these two are different people"))
        } else if busy.contains(&ga) || busy.contains(&gb) {
            (Py::from(fx::A_WAITING), Py::from("rule: asked once the open question about this person is answered"))
        } else {
            busy.insert(ga);
            busy.insert(gb);
            (Py::Null, Py::Null)
        };
        d.insert("answer", a.clone());
        d.insert("by", b.clone());
        d.insert("rule_answer", a);
        d.insert("rule_by", b);
    }
}
