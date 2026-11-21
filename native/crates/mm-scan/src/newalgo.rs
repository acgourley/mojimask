//! The scan's decide step, end to end, on a scanned project (the engine runs it after detection). (Port of
//! `mojimask/scan/newalgo.py`.)
//!
//!   join      tracklets joined by continuity (join) -> PROJECT/proto/chains.json
//!   decide    graph, attachment, overlap and blip rules, with any answers already in the project's decision records
//!             (cover) -> PROJECT/decisions_<kind>.json
//!   cover     a cover per person per frame (cover)
//!   tracks    PROJECT/proto_tracks.json: one track per group, its covers made; the review server loads it in place of
//!             its own automatic tracks. Pieces a decision dropped are included, hidden (edits.not_face = "decision"),
//!             so answering them otherwise in Decisions shows them again at once.
//!
//! SCAN_VERSION: box ids, and so every fact a person records about them (facts.py), belong to the scan that made
//! them: a rescan throws the project's facts away. Raise it whenever a change to the scan could renumber or reshape
//! boxes; a project made with another version is rescanned (losing its answers), not patched. (2: graph's position
//! rules need heads of about the same size, SPOT_SIZE. 3, 2026-11-20: the evidence reads frames from the video, not
//! the JPEG copies in PROJECT/frames, so joins, covers and tracks can come out differently: jazz proto_tracks 196 -> 202.)
//!
//! settle, the last pass once covers exist: open questions whose answer can't change the video are settled by a rule.
//!   too small    the box is under TINY_PX: kept covered (no one could tell from it anyway). TINY_PX: head box at its
//!                biggest (90th percentile of detections); under it nobody is recognisable (party clip, reviewer
//!                2026-11-11: nothing under 24 px was a readable face)
//!   inside       every frame of the box is >= INSIDE within another box that stays covered: hidden (that box covers
//!                it; as with overlaps' inside rule, showing that person must cover it again)
//!   asked twice  a box with an open overlap question isn't also asked as a track: the overlap question (part of their
//!                face / not a face / someone else) answers it
//! Your answers are never changed.
use crate::cover::{self, Built};
use crate::evidence::Evidence;
use crate::faces;
use crate::follow::Clip;
use crate::geom;
use crate::graph::{self, State};
use crate::join;
use crate::overlaps;
use crate::pose;
use crate::questions;
use crate::same;
use crate::tracklets;
use crate::viz::Viz;
use mm_core::pyjson::{self, py_round, Dict, Py};
use std::path::Path;

pub const SCAN_VERSION: i64 = 3;
pub const TINY_PX: f64 = 24.0;
pub const INSIDE: f64 = 0.8;
pub const COVERED_BY_ANOTHER: &str = "covered by another box";

fn load(p: &Path) -> Option<Py> {
    if p.exists() {
        pyjson::load(p).ok()
    } else {
        None
    }
}

fn rnd(obs: &indexmap::IndexMap<i64, geom::Cs>) -> Py {
    Py::Dict(obs.iter().map(|(t, o)| (t.to_string(), Py::List(o.iter().map(|v| Py::Float(py_round(*v, 2))).collect()))).collect())
}

/// Every stage's output, for parity dumps, and the seconds each stage took (evidence included).
#[derive(Default)]
pub struct Stages {
    pub items: Vec<(String, Py)>,
    pub seconds: Vec<(&'static str, f64)>,
    lap: Option<std::time::Instant>,
}

impl Stages {
    fn put(&mut self, k: &str, v: Py) {
        self.items.push((k.to_string(), v));
    }
    /// The seconds since the last lap (or the start), as stage `k`.
    fn lap(&mut self, k: &'static str) {
        let now = std::time::Instant::now();
        let t0 = self.lap.replace(now).unwrap_or(now);
        self.seconds.push((k, (now - t0).as_secs_f64()));
    }
}

fn boxes(v: &Py) -> Vec<(String, [f64; 3])> {
    v.as_dict()
        .unwrap()
        .iter()
        .map(|(t, o)| {
            let l = o.as_list().unwrap();
            (t.to_string(), [l[0].as_f64().unwrap(), l[1].as_f64().unwrap(), l[2].as_f64().unwrap()])
        })
        .collect()
}

/// newalgo.settle: returns the ids newly hidden. Rewrites the three decision files.
pub fn settle(proj: &Path, tracks: &Dict, hidden: &[String], overlap: Option<&mut Py>) -> Result<Vec<String>, String> {
    let hidden: std::collections::HashSet<&String> = hidden.iter().collect();
    let mut covered: indexmap::IndexMap<String, std::collections::HashMap<String, [f64; 3]>> = tracks
        .iter()
        .filter(|(k, _)| !hidden.contains(&k.to_string()))
        .map(|(k, t)| (k.to_string(), boxes(t.get("cover")).into_iter().collect()))
        .collect();
    let mut newly: Vec<String> = vec![];
    let mut own_overlap = if overlap.is_none() { load(&proj.join(overlaps::FILE)) } else { None };
    let overlap: Option<&mut Py> = match overlap {
        Some(o) => Some(o),
        None => own_overlap.as_mut(),
    };
    let asked_overlap: std::collections::HashSet<String> = match &overlap {
        Some(o) => o
            .get("decisions")
            .as_list()
            .unwrap()
            .iter()
            .filter(|d| d.get("by").is_null())
            .map(|d| d.get("about").get("piece").py_str())
            .collect(),
        None => Default::default(),
    };
    let mut overlap = overlap;
    for kind in ["track", "overlap", "blip"] {
        let path = proj.join(format!("decisions_{kind}.json"));
        let mut loaded;
        let doc: &mut Py = if kind == "overlap" {
            match overlap.as_mut() {
                Some(o) => o,
                None => continue,
            }
        } else {
            loaded = match load(&path) {
                Some(d) => d,
                None => continue,
            };
            &mut loaded
        };
        let ds = doc.as_dict_mut().unwrap().get_mut("decisions").unwrap();
        let Py::List(ds) = ds else { return Err("decisions: not a list".into()) };
        for d in ds.iter_mut() {
            if !d.get("by").is_null() {
                continue;
            }
            let k = d.get("about").get("piece").py_str();
            let Some(t) = tracks.get(&k) else { continue };
            if hidden.contains(&k) {
                continue;
            }
            let sizes: Vec<f64> = boxes(t.get("obs_c")).iter().map(|(_, o)| o[2]).collect();
            let p90 = crate::np::percentile(&sizes, 90.0);
            let inside = boxes(t.get("cover")).iter().all(|(ts, o)| {
                let mut m = 0.0f64;
                let mut any = false;
                for (kk, cv) in &covered {
                    if *kk == k {
                        continue;
                    }
                    if let Some(b) = cv.get(ts) {
                        let x = geom::covered(o, b);
                        if !any || x > m {
                            m = x;
                        }
                        any = true;
                    }
                }
                (if any { m } else { 0.0 }) >= INSIDE
            });
            let dd = d.as_dict_mut().unwrap();
            let (a, b) = if kind == "track" && asked_overlap.contains(&k) {
                ("kept: covered".to_string(), "rule: asked as an overlap question instead".to_string())
            } else if inside {
                if !newly.contains(&k) {
                    newly.push(k.clone());
                }
                covered.shift_remove(&k);
                (COVERED_BY_ANOTHER.to_string(), format!("rule: inside another covered box (>= {} in every frame)", overlaps::pct(INSIDE)))
            } else if p90 < TINY_PX {
                ("kept: covered".to_string(), "rule: too small to recognise anyone (under 24 px)".to_string())
            } else {
                continue;
            };
            dd.insert("answer", Py::Str(a.clone()));
            dd.insert("by", Py::Str(b.clone()));
            let mut opts = dd.get("options").unwrap().as_list().unwrap().clone();
            for o in [COVERED_BY_ANOTHER, "kept: covered"] {
                if !opts.iter().any(|x| x.as_str() == Some(o)) {
                    opts.push(Py::from(o));
                }
            }
            dd.insert("options", Py::List(opts));
            dd.insert("rule_answer", Py::Str(a));
            dd.insert("rule_by", Py::Str(b));
        }
        pyjson::write_atomic(&path, &doc.dumps_indent(1)).map_err(|e| e.to_string())?;
    }
    Ok(newly)
}

/// newalgo.tracks_for: (tracks, hidden, depth) from the built covers. Writes the same-person records too.
pub fn tracks_for(proj: &Path, b: &mut Built, poses: &[Vec<pose::Pose>], photo: bool, ev: &mut Evidence, viz: &mut Viz, stages: &mut Stages)
                  -> Result<(Dict, Vec<String>), String> {
    let st: &mut State = &mut b.d.st;
    let mut tracks = Dict::new();
    let members: Vec<(i64, crate::pyset::PySet)> = st.members.iter().map(|(r, m)| (*r, m.clone())).collect();
    for (r, ms) in &members {
        let obs = st.group_obs(ms);
        let real = photo || questions::is_real(obs.len(), st.clear_face(ms));
        let mut t = Dict::new();
        t.insert("real", Py::Bool(real));
        t.insert("obs_c", rnd(&obs));
        t.insert("cover", rnd(&b.covers[r]));
        if let Some(bh) = b.behind.get(r) {
            t.insert("behind", Py::List(bh.iter().map(|x| Py::Int(*x)).collect()));
        }
        if let Some(f) = b.flow.get(r).filter(|f| !f.is_empty()) {
            t.insert("flow", Py::Dict(f.iter().map(|(t, m)| (t.to_string(), Py::List(m.iter().map(|v| Py::Float(*v)).collect()))).collect()));
        }
        tracks.insert(r.to_string(), Py::Dict(t));
    }
    let mut ov = load(&proj.join(overlaps::FILE));
    let mut hidden = vec![];
    for (r, ms) in &b.d.members0 {
        if st.members.contains_key(r) {
            continue;
        }
        let obs = st.group_obs(ms);
        let mut t = Dict::new();
        t.insert("real", Py::Bool(false));
        t.insert("dropped", Py::Bool(true));
        t.insert("obs_c", rnd(&obs));
        t.insert("cover", rnd(&obs));
        tracks.insert(r.to_string(), Py::Dict(t));
        hidden.push(r.to_string());
    }
    let newly = settle(proj, &tracks, &hidden, ov.as_mut())?;
    let mut sd = Dict::new();
    sd.insert("hidden_in", Py::List(hidden.iter().map(|h| Py::Str(h.clone())).collect()));
    sd.insert("newly", Py::List(newly.iter().map(|h| Py::Str(h.clone())).collect()));
    stages.put("settle", Py::Dict(sd));
    hidden.extend(newly);
    same::find(proj, st, poses, ev, viz)?;
    Ok((tracks, hidden))
}

pub const EDITS0: &str = r#"{"deleted": [], "trims": {}, "merges": {}, "manual": {}, "emoji": {}, "dismissed": [], "shown": [], "people": {}, "cuts": {}, "cuts_frozen": 1}"#;

/// newalgo.write: PROJECT/proto_tracks.json and a fresh PROJECT/edits.json.
pub fn write(dst: &Path, tracks: Dict, hidden: &[String], extra_edits: Vec<(&str, Py)>, depth: Option<&str>) -> Result<(), String> {
    let mut sorted_h: Vec<String> = hidden.to_vec();
    sorted_h.sort();
    let mut p = Dict::new();
    p.insert("source", Py::Null);
    p.insert("scan_version", Py::Int(SCAN_VERSION));
    p.insert("hidden", Py::List(sorted_h.into_iter().map(Py::Str).collect()));
    p.insert("tracks", Py::Dict(tracks));
    if let Some(d) = depth {
        p.insert("depth", Py::from(d));
    }
    pyjson::write_atomic(&dst.join("proto_tracks.json"), &Py::Dict(p).dumps()).map_err(|e| e.to_string())?;
    let Py::Dict(mut e) = Py::parse(EDITS0)? else { unreachable!() };
    for (k, v) in extra_edits {
        e.insert(k, v);
    }
    e.insert("not_face", Py::Dict(hidden.iter().map(|h| (h.clone(), Py::from("decision"))).collect()));
    pyjson::write_atomic(&dst.join("edits.json"), &Py::Dict(e).dumps()).map_err(|e| e.to_string())
}

/// newalgo._run: the decide step. Returns {"tracks", "people", "hidden"} and each stage's output.
pub fn run(proj: &Path, ev: &mut Evidence, viz: &mut Viz) -> Result<(Dict, Stages), String> {
    let mut stages = Stages { lap: Some(std::time::Instant::now()), ..Default::default() };
    // a rescan starts over: the old facts name the old scan's boxes (and the rules mustn't read old answers)
    for e in std::fs::read_dir(proj).map_err(|e| e.to_string())?.flatten() {
        let n = e.file_name().to_string_lossy().to_string();
        if n == "facts.json" || (n.starts_with("decisions_") && n.ends_with(".json")) {
            std::fs::remove_file(e.path()).map_err(|e| e.to_string())?;
        }
    }
    let clip = Clip::open(proj)?;
    let tl = tracklets::build(&tracklets::detections(proj)?, &clip.mats, tracklets::R);
    stages.put("tracklets", join::tracklets_json(&tl));
    stages.lap("tracklets");
    let j = join::join(tl, &clip, ev, viz, join::LIMIT);
    stages.put("chains", join::chains_json(&j));
    join::save(proj, &j.chains)?;
    stages.lap("join");
    let (fs, order) = faces::measure(&j.tl, ev, viz);
    stages.put("faces", faces::faces_json(&fs, &order));
    stages.lap("faces");
    let meta = pyjson::load(&proj.join("meta.json"))?;
    let fps = match meta.get("fps") {
        v if v.truthy() => v.as_f64().unwrap(),
        _ => 24.0,
    };
    let photo = meta.get("image").truthy();
    let st = graph::run(j.tl, j.chains, fs, clip.mats.clone(), fps, viz);
    stages.put("graph", graph::graph_json(&st));
    stages.lap("graph");
    let poses = pose::load(proj)?;
    let mut b = cover::build(proj, st, poses.as_deref(), photo, &clip, ev, viz, true)?;
    stages.lap("cover");
    let mut att = Dict::new();
    att.insert("att", Py::List(b.d.attached.iter().map(|(f, r)| Py::List(vec![Py::Int(*f), Py::Int(*r)])).collect()));
    att.insert("gd", b.d.gd.clone());
    stages.put("attach", Py::Dict(att));
    stages.put("members", cover::members_stage(&b.d));
    stages.put("build", cover::build_stage(&b));
    let empty = vec![];
    let (tracks, hidden) = tracks_for(proj, &mut b, poses.as_deref().unwrap_or(&empty), photo, ev, viz, &mut stages)?;
    let n_real = tracks.iter().filter(|(_, t)| t.get("real").truthy()).count() as i64;
    let n_tracks = tracks.len() as i64;
    write(proj, tracks, &hidden, vec![("result", Py::from("pending")), ("head_pose", Py::Bool(true))], b.depth.as_deref())?;
    stages.lap("tracks");
    let mut res = Dict::new();
    res.insert("tracks", Py::Int(n_tracks - hidden.len() as i64));
    res.insert("people", Py::Int(n_real));
    res.insert("hidden", Py::Int(hidden.len() as i64));
    Ok((res, stages))
}

/// newalgo.run: the decide step, with its events kept in PROJECT/proto/decide_viz.json.
pub fn run_logged(proj: &Path, ev: &mut Evidence) -> Result<(Dict, Stages, Viz), String> {
    let mut viz = Viz::default();
    let r = run(proj, ev, &mut viz);
    let p = proj.join("proto").join("decide_viz.json");
    let _ = std::fs::create_dir_all(p.parent().unwrap());
    let _ = pyjson::write_atomic(&p, &Py::List(viz.events.clone()).dumps());
    let (res, st) = r?;
    Ok((res, st, viz))
}
