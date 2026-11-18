//! "Whose face is this?": review_server.your_box_questions, in Rust.
//!
//! A box you make (a mask you draw, a part you split off) is a person of its own until you say whose face it is: one
//! question per box, with every person as an answer (and "someone new"), best match first. Best = looks most alike:
//! faces (SFace on near-frontal views, as the scan compares them) and bodies (OSNet on the person's own skeleton box:
//! a whip-pan blur defeats the face but rarely the clothes); then nearest in time and place as the tiebreak. Worked out
//! from the edits every time; the answer is a fact (same: your box + that person). A split-off part isn't offered the
//! person it was split from; people you said it isn't aren't offered either. (party_clip_c: P2's mask at 186-212 was
//! P1, last seen at frame 21 -- six seconds off, never asked.)
//!
//! The looks come from mm-analysis (faces::face, appearance::reid_embed) on the default plan: their last bits can
//! differ from the Python server's (CoreML), so two near-equal candidates can swap places.
use super::project::{people_of, piece_kind, Proj};
use super::Server;
use anyhow::Result;
use mm_core::pyjson::{py_round, Dict, Py};
use mm_records::resolve as R;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

pub const YOURS_FRAMES_PER_HEAD: f64 = 30.0;
pub const LOOK_SAMPLES: usize = 8;

/// (face, body) of a box: mean unit embeddings, each None when there's nothing to compare.
pub type Look = (Option<Vec<f32>>, Option<Vec<f32>>);

/// look()'s cache: (piece, its frames) -> look
#[derive(Default)]
pub struct Looks(Mutex<HashMap<(String, Vec<i64>), Look>>);

/// An insertion-ordered {t: c} (Python's dict).
#[derive(Default, Clone)]
struct OMap {
    keys: Vec<i64>,
    v: HashMap<i64, [f64; 3]>,
}

impl OMap {
    fn insert(&mut self, t: i64, c: [f64; 3]) {
        if self.v.insert(t, c).is_none() {
            self.keys.push(t);
        }
    }
    fn len(&self) -> usize {
        self.keys.len()
    }
    fn iter(&self) -> impl Iterator<Item = (i64, [f64; 3])> + '_ {
        self.keys.iter().map(|t| (*t, self.v[t]))
    }
}

fn unit(v: &[f32]) -> Vec<f32> {
    mm_core::np::unit_f32(v)
}

/// review_server.look(piece, boxes).
fn look(srv: &Server, p: &Proj, piece: &str, boxes: &OMap) -> Look {
    let mut ts: Vec<i64> = boxes.keys.clone();
    ts.sort();
    let key = (piece.to_string(), ts.clone());
    if let Some(l) = p.looks.0.lock().unwrap().get(&key) {
        return l.clone();
    }
    let step = (ts.len() / LOOK_SAMPLES).max(1);
    let ts: Vec<i64> = ts.iter().step_by(step).take(LOOK_SAMPLES).copied().collect();
    let models = srv.models();
    let (mut fs, mut crops) = (vec![], vec![]);
    for t in ts {
        let Ok(d) = mm_vision::jpeg::read_bgr(&p.work.join("frames").join(format!("{t:05}.jpg"))) else { continue };
        let img = mm_vision::Mat::from_vec(d.h, d.w, 3, d.data);
        let c = boxes.v[&t];
        if let Some(f) = crate::evidence::face(&models, &img, &c) {
            if let Some(e) = f.emb {
                fs.push(e);
            }
        }
        let sk = if !p.poses.is_empty() && (t as usize) < p.poses.len() {
            mm_analysis::pose_util::pose_for_head(&p.poses[t as usize], &c, 0.25)
        } else {
            None
        };
        let bb = mm_analysis::appearance::body_box(&c, sk);
        if let Some(cr) = mm_analysis::appearance::crop(&img, &bb, 0.0) {
            if cr.rows.min(cr.cols) >= 16 {
                crops.push(cr);
            }
        }
    }
    let face = (!fs.is_empty()).then(|| unit(&mm_core::np::mean_rows_f32(&fs.iter().map(|v| v.as_slice()).collect::<Vec<_>>())));
    let body = (!crops.is_empty()).then(|| {
        let e = mm_analysis::appearance::reid_embed(&models, &crops);
        unit(&mm_core::np::mean_rows_f32(&e.iter().map(|v| v.as_slice()).collect::<Vec<_>>()))
    });
    let out = (face, body);
    p.looks.0.lock().unwrap().insert(key, out.clone());
    out
}

/// review_server.likeness: (score, words) on the scan's scale (questions.score: face_term + body_term).
fn likeness(a: &Look, b: &Look) -> (f64, Vec<String>) {
    let (mut s, mut parts) = (0.0, vec![]);
    if let (Some(x), Some(y)) = (&a.0, &b.0) {
        let sim = mm_core::np::sdot(x, y) as f64;
        s += mm_scan::questions::face_term(sim);
        parts.push(format!("faces {sim:.2}"));
    }
    if let (Some(x), Some(y)) = (&a.1, &b.1) {
        let sim = mm_core::np::sdot(x, y) as f64;
        s += mm_scan::questions::body_term(sim);
        parts.push(format!("bodies {sim:.2}"));
    }
    (s, parts)
}

fn boxj(c: &[f64; 3]) -> Py {
    Py::List(R::from_cs(c).iter().map(|v| Py::Float(py_round(*v, 1))).collect())
}

/// review_server.your_box_questions(edits).
pub fn questions(srv: &Server, p: &Proj, edits: &Py, facts: &[Py]) -> Result<Vec<Py>> {
    let mut e2 = edits.clone();
    if let Some(d) = e2.as_dict_mut() {
        if !d.contains("cuts") {
            d.insert("cuts", Py::Dict(Dict::new()));
        }
    }
    let pf = R::resolve(&p.auto, &e2, &p.mats, p.n, &R::links_for(edits), true)?;
    let gone: HashSet<String> = edits.get("not_face").as_dict().map(|d| d.keys().map(String::from).collect()).unwrap_or_default();
    let manual = edits.get("manual").as_dict().cloned().unwrap_or_default();
    // piece -> {t: c} (its own detections, or your box), in first-seen order
    let mut at_keys: Vec<String> = vec![];
    let mut at: HashMap<String, OMap> = HashMap::new();
    let mut tid_of: HashMap<String, String> = HashMap::new();
    for (t, f) in pf.iter().enumerate() {
        for x in f {
            tid_of.insert(x.piece.clone(), x.tid.clone());
            if x.kind == "obs" || x.kind == "key" || manual.contains(&x.piece) {
                if !at.contains_key(&x.piece) {
                    at_keys.push(x.piece.clone());
                }
                at.entry(x.piece.clone()).or_default().insert(t as i64, x.c);
            }
        }
    }
    let yours: Vec<String> = at_keys.iter().filter(|q| piece_kind(&p.real, q) == "yours" && !gone.contains(*q)).cloned().collect();
    if yours.is_empty() {
        return Ok(vec![]);
    }
    let mut inv = HashMap::new();
    let part = mm_core::facts::partition(facts);
    let names: HashMap<String, String> = edits.get("people").as_dict().map(|d| {
        d.iter().map(|(k, pp)| (format!("@{k}"), pp.get("name").as_str().unwrap_or("").to_string())).collect()
    }).unwrap_or_default();
    let persons: Vec<serde_json::Value> = people_of(&p.real, &pf).into_iter().filter(|q| q["kind"] == "scan" || q["kind"] == "yours").collect();
    let scan_mats: Vec<mm_scan::np::M23> = p.mats.iter().copied().collect();
    let mut out = vec![];
    for mid in &yours {
        let mine = &at[mid];
        let split = mid.contains('~');
        let mut ts: Vec<i64> = mine.keys.clone();
        ts.sort();
        let drawn: Vec<i64> = if let Some(m) = manual.get(mid) {
            let mut s: Vec<i64> = vec![];
            for k in ["keys", "prop"] {
                for (kk, _) in m.get(k).as_dict().map(|d| d.iter().collect::<Vec<_>>()).unwrap_or_default() {
                    if let Ok(u) = kk.trim().parse::<i64>() {
                        if mine.v.contains_key(&u) && !s.contains(&u) {
                            s.push(u);
                        }
                    }
                }
            }
            s.sort();
            if s.is_empty() { ts.clone() } else { s }
        } else {
            ts.clone()
        };
        let did = format!("whose:{mid}");
        let chosen: Option<String> = facts.iter().filter(|f| f.get("decision").as_str() == Some(did.as_str())).last()
            .and_then(|f| f.get("answer").as_str().map(String::from));
        let base = mid.split('~').next().unwrap();
        let mine_tid = tid_of.get(mid).cloned();
        let not_them: HashSet<String> = if split {
            at_keys.iter().filter(|q| q.split('~').next().unwrap() == base && *q != mid && tid_of.contains_key(*q))
                .map(|q| tid_of[q].clone()).filter(|t| Some(t) != mine_tid.as_ref()).collect()
        } else {
            HashSet::new()
        };
        let mut own = OMap::default();
        for t in &drawn {
            own.insert(*t, mine.v[t]);
        }
        for q in &at_keys {
            if q != mid && tid_of.get(q) == tid_of.get(mid) {
                for (t, c) in at[q].iter() {
                    own.insert(t, c);
                }
            }
        }
        let me = look(srv, p, mine_tid.as_deref().unwrap_or(mid), &own);
        let cm = part.c(mid);
        let mut cands: Vec<(f64, Py)> = vec![];
        for pp in &persons {
            let tid = pp["tid"].as_str().unwrap_or("").to_string();
            let ps: Vec<String> = pp["pieces"].as_array().into_iter().flatten().filter_map(|x| x.as_str())
                .filter(|q| at.contains_key(*q) && !gone.contains(*q) && *q != mid).map(String::from).collect();
            if ps.is_empty() {
                continue;
            }
            let is_chosen = chosen.as_ref().is_some_and(|c| ps.contains(c));
            let c0 = part.c(&ps[0]);
            let kept = part.kept.contains(&(cm.clone(), c0.clone())) || part.kept.contains(&(c0.clone(), cm.clone()));
            if !is_chosen && (Some(&tid) == mine_tid.as_ref() || not_them.contains(&tid) || kept) {
                continue;
            }
            let mut obs = OMap::default();
            for q in &ps {
                for (t, c) in at[q].iter() {
                    obs.insert(t, c);
                }
            }
            let mut near: Option<(f64, i64, f64)> = None;
            for tm in &drawn {
                let mut to = obs.keys[0];
                for &u in &obs.keys {
                    if (u - tm).abs() < (to - tm).abs() {
                        to = u;
                    }
                }
                let (p0, s0) = mm_scan::geom::to_room(&scan_mats, *tm, &mine.v[tm], &mut inv);
                let (p1, s1) = mm_scan::geom::to_room(&scan_mats, to, &obs.v[&to], &mut inv);
                let raw = (p0[0] - p1[0]).hypot(p0[1] - p1[1]) / ((s0 + s1) / 2.0);
                let cost = raw + (to - tm).abs() as f64 / YOURS_FRAMES_PER_HEAD;
                if near.is_none_or(|n| cost < n.0) {
                    near = Some((cost, (to - tm).abs(), raw));
                }
            }
            let near = near.unwrap();
            let (sim, words) = likeness(&me, &look(srv, p, &tid, &obs));
            let mut piece = ps[0].clone();
            for q in &ps {
                if at[q].len() > at[&piece].len() {
                    piece = q.clone();
                }
            }
            if is_chosen {
                piece = chosen.clone().unwrap();
            }
            let mut tb = obs.keys[0];
            for &u in &obs.keys {
                if obs.v[&u][2] > obs.v[&tb][2] {
                    tb = u;
                }
            }
            let rank = sim - near.0 / 10.0;
            let why = if words.is_empty() {
                format!("{} frames away, nothing to compare", near.1)
            } else {
                let mut w = words.clone();
                w.push(format!("{} frames away", near.1));
                w.join(", ")
            };
            let mut d = Dict::new();
            d.insert("piece", Py::from(piece.as_str()));
            d.insert("tid", Py::from(tid.as_str()));
            d.insert("name", Py::from(names.get(&tid).cloned().unwrap_or_default()));
            d.insert("t", Py::Int(tb));
            d.insert("box", boxj(&obs.v[&tb]));
            d.insert("rank", Py::Float(rank));
            d.insert("likeness", Py::Float(py_round(sim, 2)));
            d.insert("why", Py::from(why));
            cands.push((rank, Py::Dict(d)));
        }
        cands.sort_by(|a, b| b.0.total_cmp(&a.0)); // stable, as sorted(key=-rank)
        let mut keys: Vec<i64> = manual.get(mid).and_then(|m| m.get("keys").as_dict()).map(|d| {
            d.keys().filter_map(|k| k.trim().parse::<i64>().ok()).filter(|k| mine.v.contains_key(k)).collect()
        }).unwrap_or_default();
        keys.sort();
        if keys.is_empty() {
            keys = vec![drawn[drawn.len() / 2]];
        }
        let mut show = vec![];
        for t in [keys[0], *keys.last().unwrap()] {
            if show.iter().any(|s: &Py| s.get("t").as_i64() == Some(t)) {
                continue;
            }
            let mut d = Dict::new();
            d.insert("t", Py::Int(t));
            d.insert("piece", boxj(&mine.v[&t]));
            d.insert("person", Py::Null);
            show.push(Py::Dict(d));
        }
        let what = if split { "a part you split off" } else { "a mask you added" };
        let mut opts: Vec<Py> = cands.iter().map(|c| c.1.get("piece").clone()).collect();
        opts.push(Py::from(mm_core::facts::A_SOMEONE_NEW));
        let mut ev = Dict::new();
        ev.insert("manual", Py::Bool(!split));
        ev.insert("split", Py::Bool(split));
        ev.insert("frames", Py::from(ts.len()));
        ev.insert("first", Py::Int(ts[0]));
        ev.insert("last", Py::Int(*ts.last().unwrap()));
        ev.insert("candidates", Py::from(cands.len()));
        let mut about = Dict::new();
        about.insert("piece", Py::from(mid.as_str()));
        let mut q = Dict::new();
        q.insert("id", Py::from(did.as_str()));
        q.insert("kind", Py::from("whose"));
        q.insert("about", Py::Dict(about));
        q.insert("question", Py::from(format!("Whose face is this ({what})?")));
        q.insert("options", Py::List(opts));
        q.insert("candidates", Py::List(cands.into_iter().map(|c| c.1).collect()));
        q.insert("evidence", Py::Dict(ev));
        q.insert("why", Py::from(format!("{what}; everyone is offered, the most alike first (faces, then bodies, then nearest in time and place)")));
        q.insert("show", Py::List(show));
        for k in ["answer", "by", "rule_answer", "rule_by"] {
            q.insert(k, Py::Null);
        }
        out.push(Py::Dict(q));
    }
    Ok(out)
}
