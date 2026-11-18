//! The scan's second step: join tracklets (tracklets) into chains by continuity -- the checked follow (follow)
//! from one tracklet's end landing on another's start. graph then groups the chains into people.
//! (Port of `mojimask/scan/join.py`.)
//!
//! For every tracklet: follow forward from its last detection, and backward from its first, up to LIMIT frames
//! (follow). A follow stops when its box lands (IoU >= MEET) on a detection that starts (forward) or ends (backward)
//! another tracklet. A join needs both directions to agree -- A's forward follow lands on B's start and B's backward
//! follow lands on A's end -- so a false detection (a hand) followed onto a real face doesn't join them unless the
//! face, followed back, also lands on the hand.
//!
//! Writes PROJECT/proto/chains.json: {chain id: [tracklet ids in time order]}.
//!
//! A tracklet's two follows are independent of every other tracklet's, so they run in threads (THREADS, env
//! JOIN_THREADS, as join.py's ThreadPoolExecutor; the face check is most of the time). Results are taken in tracklet
//! order, and the follow events are emitted in that order too (Python emits them as threads finish; the parity tests
//! compare follow events as a multiset).
use crate::evidence::Evidence;
use crate::follow::{self, Clip, FPath};
use crate::geom::{self, Cs};
use crate::tracklets::{self, Tracklets};
use crate::viz::{r, Viz};
use indexmap::IndexMap;
use mm_core::pyjson::{Dict, Py};
use std::collections::HashMap;
use std::path::Path;

pub const LIMIT: i64 = 24;
pub const MEET: f64 = 0.4;

/// join.THREADS: JOIN_THREADS, else min(8, CPUs).
pub fn threads() -> usize {
    std::env::var("JOIN_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8))
        .max(1)
}

/// f over items on `n` threads, results in the items' order.
pub fn par_map<T: Sync, R: Send>(items: &[T], n: usize, f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    if n <= 1 || items.len() <= 1 {
        return items.iter().map(f).collect();
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: std::sync::Mutex<Vec<Option<R>>> = std::sync::Mutex::new((0..items.len()).map(|_| None).collect());
    std::thread::scope(|s| {
        for _ in 0..n.min(items.len()) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(it) = items.get(i) else { break };
                let r = f(it);
                out.lock().unwrap()[i] = Some(r);
            });
        }
    });
    out.into_inner().unwrap().into_iter().map(|r| r.unwrap()).collect()
}

/// Chains by id (the first tracklet's id), in tracklet order: {chain: [tracklet ids in time order]}.
pub type Chains = IndexMap<i64, Vec<i64>>;

pub struct Joined {
    pub tl: Tracklets,
    pub chains: Chains,
    /// join.run's counts: forward lands, backward lands, mutual joins, follows by (direction, why)
    pub info: Vec<(String, Py)>,
}

fn path_viz(p: &FPath, start: i64, w: f64) -> Py {
    let mut us: Vec<i64> = p.boxes.keys().copied().collect();
    us.sort_by_key(|u| (u - start).abs());
    let how: String = us
        .iter()
        .map(|u| match p.how[u] {
            "face at flow" => 'f',
            "face at camera" => 'c',
            "looks the same" => 'l',
            "hold" => 'h',
            "exit" => 'x',
            _ => '?',
        })
        .collect();
    let mut d = Dict::new();
    d.insert("from", Py::Int(start));
    d.insert("x", Py::List(us.iter().map(|u| r(p.boxes[u][0] / w, 4)).collect()));
    d.insert("y", Py::List(us.iter().map(|u| r(p.boxes[u][1] / w, 4)).collect()));
    d.insert("how", Py::Str(how));
    d.insert("why", Py::Str(p.why.into()));
    d.insert("met", p.met.map(|m| Py::Int(m.1)).unwrap_or(Py::Null));
    Py::Dict(d)
}

/// join.run.
pub fn run(proj: &Path, clip: &Clip, ev: &mut Evidence, viz: &mut Viz, limit: i64) -> Result<Joined, String> {
    let tl = tracklets::build(&tracklets::detections(proj)?, &clip.mats, tracklets::R);
    Ok(join(tl, clip, ev, viz, limit))
}

/// join.run after its tracklets are built.
pub fn join(tl: Tracklets, clip: &Clip, ev: &mut Evidence, viz: &mut Viz, limit: i64) -> Joined {
    let w = clip.w;
    let mut pieces = Dict::new();
    // c[i] / W is a numpy float64 here, so Python's round() is numpy's (np::np_round)
    let nr = |x: f64| Py::Float(crate::np::np_round(x, 4));
    for (k, v) in &tl {
        pieces.insert(k.to_string(), Py::List(v.obs.iter().map(|(t, c)| Py::List(vec![Py::Int(*t), nr(c[0] / w), nr(c[1] / w), nr(c[2] / w)])).collect()));
    }
    viz.emit("pieces", vec![("n", Py::Int(clip.n)), ("aspect", r(clip.h / w, 4)), ("pieces", Py::Dict(pieces))]);
    let mut starts: HashMap<i64, Vec<i64>> = HashMap::new();
    let mut ends: HashMap<i64, Vec<i64>> = HashMap::new();
    for (k, v) in &tl {
        starts.entry(v.first()).or_default().push(*k);
        ends.entry(v.last()).or_default().push(*k);
    }
    let meeter = |index: &HashMap<i64, Vec<i64>>, me: i64, u: i64, c: &Cs| -> Option<i64> {
        let mut best = (MEET, None);
        for &k in index.get(&u).map(|v| v.as_slice()).unwrap_or(&[]) {
            if k == me {
                continue;
            }
            let o = geom::iou(&geom::from_cs(c), &geom::from_cs(&tl[&k].obs[&u]));
            if o >= best.0 {
                best = (o, Some(k));
            }
        }
        best.1
    };
    let mut fwd: IndexMap<i64, i64> = IndexMap::new();
    let mut bwd: HashMap<i64, i64> = HashMap::new();
    let mut how: IndexMap<String, i64> = IndexMap::new();
    let keys: Vec<i64> = tl.keys().cloned().collect();
    let evr: &Evidence = ev;
    let done = par_map(&keys, threads(), |&k| {
        let v = &tl[&k];
        let (a, b) = (v.first(), v.last());
        let mf = |u: i64, c: &Cs| meeter(&starts, k, u, c);
        let mb = |u: i64, c: &Cs| meeter(&ends, k, u, c);
        let pf = follow::follow(clip, evr, b, &v.obs[&b], 1, limit, Some(&mf), false, None);
        let pb = follow::follow(clip, evr, a, &v.obs[&a], -1, limit, Some(&mb), false, None);
        (pf, pb)
    });
    for (&k, (pf, pb)) in keys.iter().zip(done) {
        let v = &tl[&k];
        let (a, b) = (v.first(), v.last());
        viz.emit("follow", vec![("k", Py::Int(k)), ("fwd", path_viz(&pf, b, w)), ("bwd", path_viz(&pb, a, w))]);
        *how.entry(format!("('fwd', '{}')", pf.why)).or_insert(0) += 1;
        if let Some(m) = pf.met {
            fwd.insert(k, m.1);
        }
        *how.entry(format!("('bwd', '{}')", pb.why)).or_insert(0) += 1;
        if let Some(m) = pb.met {
            bwd.insert(k, m.1);
        }
    }
    let joins: Vec<(i64, i64)> = fwd.iter().filter(|(a, b)| bwd.get(b) == Some(a)).map(|(a, b)| (*a, *b)).collect();
    let mut order = joins.clone();
    order.sort_by_key(|x| tl[&x.1].first()); // stable
    let mut nxt: IndexMap<i64, i64> = IndexMap::new();
    let mut prev: HashMap<i64, i64> = HashMap::new();
    for (a, b) in order {
        if !nxt.contains_key(&a) && !prev.contains_key(&b) {
            nxt.insert(a, b);
            prev.insert(b, a);
        }
    }
    viz.emit("joins", vec![("pairs", Py::List(nxt.iter().map(|(a, b)| Py::List(vec![Py::Int(*a), Py::Int(*b)])).collect()))]);
    let mut chains = Chains::new();
    for &k in tl.keys() {
        if prev.contains_key(&k) {
            continue;
        }
        let mut seq = vec![k];
        while let Some(n) = nxt.get(seq.last().unwrap()) {
            seq.push(*n);
        }
        chains.insert(k, seq);
    }
    let info = vec![
        ("forward lands".to_string(), Py::Int(fwd.len() as i64)),
        ("backward lands".to_string(), Py::Int(bwd.len() as i64)),
        ("mutual joins".to_string(), Py::Int(joins.len() as i64)),
        ("follows".to_string(), Py::Dict(how.into_iter().map(|(k, v)| (k, Py::Int(v))).collect())),
    ];
    Joined { tl, chains, info }
}

/// join.save: PROJECT/proto/chains.json.
pub fn save(proj: &Path, chains: &Chains) -> Result<(), String> {
    let d: Dict = chains.iter().map(|(c, seq)| (c.to_string(), Py::List(seq.iter().map(|k| Py::Int(*k)).collect()))).collect();
    let p = proj.join("proto").join("chains.json");
    std::fs::create_dir_all(p.parent().unwrap()).map_err(|e| e.to_string())?;
    std::fs::write(&p, Py::Dict(d).dumps()).map_err(|e| e.to_string())
}

/// The tracklets as the recorder writes them (stage_tracklets.json): full precision.
pub fn tracklets_json(tl: &Tracklets) -> Py {
    let f = |c: &[f64]| Py::List(c.iter().map(|v| Py::Float(*v)).collect());
    Py::Dict(
        tl.iter()
            .map(|(k, v)| {
                let mut d = Dict::new();
                d.insert("obs", Py::Dict(v.obs.iter().map(|(t, c)| (t.to_string(), f(c))).collect()));
                d.insert("score", Py::Dict(v.score.iter().map(|(t, s)| (t.to_string(), Py::Float(*s))).collect()));
                d.insert("v", f(&v.v));
                d.insert("why", Py::Str(v.why.into()));
                (k.to_string(), Py::Dict(d))
            })
            .collect(),
    )
}

pub fn chains_json(j: &Joined) -> Py {
    let mut d = Dict::new();
    d.insert("chains", Py::Dict(j.chains.iter().map(|(c, s)| (c.to_string(), Py::List(s.iter().map(|k| Py::Int(*k)).collect()))).collect()));
    d.insert("info", Py::Dict(j.info.iter().cloned().collect()));
    Py::Dict(d)
}
