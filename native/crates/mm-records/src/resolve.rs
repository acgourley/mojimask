//! What's drawn on every frame from the scan's tracks and the edits: the `resolve` half of mojimask/track.py
//! (joins, splits, trims, masks you drew, who's shown, then bridged and steadied), `draw_order`, `shown_pieces`,
//! `emoji_for`, and review_server's `links_for` / `with_cuts`.
//!
//! From track.py (kept true here):
//! - **resolve:** apply the review edits to the automatic tracks. Per frame [(tid, [cx, cy, s], kind, piece)]: tid =
//!   canonical identity after re-ID links and user merges (decides the emoji), piece = the underlying auto/manual
//!   track id; kind in {obs, coast, manual}. Deletes and trims apply ONLY to the pieces listed -- never implicitly to
//!   other pieces linked to them -- so an automatic (possibly wrong) re-ID link can never widen a delete and unmask
//!   someone. not_face {piece: why} / not_real [pieces]: like deleted they get no cover; unlike deleted they keep
//!   their place in the person's group. Automatic links first; the user's own merges override them; named People
//!   (explicit piece lists) override both: their pieces resolve to the person's stable id "@<pid>".
//! - **One box per person** (`_dedupe`), the one with the freshest evidence (frames since its own detection, or
//!   since your keyframe or a face it snapped to) -- not "a cover always wins": a cover followed from one answer slid
//!   off a spinning child onto the furniture and was drawn instead of her detected face. Ties: your keyframe, a
//!   detection, a followed cover, a held box, then the longer-lived. A second box of the same person is kept too when
//!   it is itself fresh (<= SAME_PERSON_FRESH frames) and elsewhere. Boxes mostly overlapping a kept box of someone
//!   else are dropped; a held box yields to someone else's current detection it mostly overlaps (HELD_YIELDS:
//!   party_clip_c frames 145 and 153, two emoji on one face). One person has one face: a fresh second box of theirs
//!   overlapping one already kept (a piece answered "part of their face", or SAME_FACE_OVERLAP) is drawn as one emoji
//!   over both boxes' bounding square (party_clip_c frame 210). A held emoji mostly inside a detected one is dropped.
//! - **Bridging:** a person's mask with a short gap (up to BRIDGE frames, one box each end) is filled moving evenly in
//!   the scene's frame of reference (camera motion taken out), at the larger of the two sizes -- except where the
//!   filled box would mostly sit on someone else's detected face, where the scan found them hidden behind someone
//!   (scan/behind.py), or when the two ends are further apart than BRIDGE_REACH head sizes in the scene (jazz: two
//!   people merged by their faces were bridged by an emoji flying across the picture; across every project's gaps
//!   the largest real one was 1.8 head sizes, median 0.1; that one was 25).
//! - **Steady emoji:** each person's box smoothed over time in the camera's frame of reference (party_clip_c: size
//!   changing 10% a frame one frame in ten): position over STEADY_POS frames, size over STEADY_SIZE frames of the
//!   upper envelope, never below STEADY_KEEP of the box it replaces plus what it takes to still contain it. Carried
//!   along the face's own motion where the scan measured it (track["flow"]): carried by the camera alone, jazz's
//!   dancer's emoji lagged her face by up to 0.28 head sizes.
//!
//! **Port notes (numpy, bit for bit on macOS):** matrix products are mm_core::np's BLAS calls (dgemm, dgemv, dgesv,
//! dgetrf); `np.convolve` is numpy's `_pyarray_correlate`: the edges by `cblas_ddot`, the middle by `small_correlate`
//! for kernels up to 11 taps, whose `s += d * k` clang contracts to a fused multiply-add chain from 0 (checked against
//! numpy on 3,000 random cases); `np.einsum("nij,nj->ni")` with j = 3 is NEON's two lanes: fma(a2, b2, a0 * b0) +
//! a1 * b1 (135,546 cases, no exceptions).
use anyhow::{anyhow, Result};
use mm_core::np;
use mm_core::pyjson::{Dict, Py};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

pub type M23 = [[f64; 3]; 2];
pub type M33 = [[f64; 3]; 3];

pub const KIND_RANK: [(&str, i32); 4] = [("key", 0), ("obs", 1), ("manual", 2), ("coast", 3)];
/// frames
pub const SAME_PERSON_FRESH: i64 = 2;
/// a held box overlapping someone else's current detection this much (of the smaller) is dropped
pub const HELD_YIELDS: f64 = 0.5;
/// two boxes of one person overlapping this much (of the smaller) are one face
pub const SAME_FACE_OVERLAP: f64 = 0.25;
pub const BRIDGE: i64 = 15;
pub const BRIDGE_REACH: f64 = 2.5;
pub const STEADY: bool = true;
pub const STEADY_FLOW: bool = true;
pub const STEADY_POS: f64 = 1.5;
pub const STEADY_SIZE: f64 = 4.0;
pub const STEADY_ENV: usize = 2;
pub const STEADY_KEEP: f64 = 0.9;
pub const EMOJI: [&str; 24] =
    ["😀", "😎", "🤩", "🥳", "😜", "🤪", "😺", "🐸", "🐵", "🦄", "👽", "🤖", "🎃", "👻", "🌝", "🌞", "🍩", "🍕", "🦊", "🐼", "🐯", "🐶", "🐙", "🦋"];
pub const STYLES: [&str; 5] = ["emoji", "blur", "pixelate", "solid", "portrait"];

/// One scan track (review_server.AUTO's entry).
#[derive(Clone, Debug)]
pub struct Track {
    pub id: String,
    /// detected frames, sorted
    pub obs: Vec<i64>,
    pub obs_c: BTreeMap<i64, [f64; 3]>,
    /// the drawn boxes, in the file's order
    pub cover: Vec<(i64, [f64; 3])>,
    pub behind: Vec<i64>,
    /// the head's own move t -> t+1 as a similarity [a, b, tx, ty]
    pub flow: HashMap<i64, [f64; 4]>,
}

/// One drawn box: (tid, c, kind, piece).
#[derive(Clone, Debug, PartialEq)]
pub struct Item {
    pub tid: String,
    pub c: [f64; 3],
    pub kind: &'static str,
    pub piece: String,
}

#[derive(Clone, Debug)]
struct Cand {
    tid: String,
    c: [f64; 3],
    kind: &'static str,
    life: i64,
    piece: String,
    since: i64,
}

// ------------------------------------------------------------------ geometry (track.py)
pub fn from_cs(c: &[f64; 3]) -> [f64; 4] {
    let h = c[2] * 1.0 / 2.0;
    [c[0] - h, c[1] - h, c[0] + h, c[1] + h]
}

fn pmax(a: f64, b: f64) -> f64 {
    // Python's max(a, b): a unless b > a
    if b > a {
        b
    } else {
        a
    }
}
fn pmin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

pub fn iou(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let (x1, y1) = (pmax(a[0], b[0]), pmax(a[1], b[1]));
    let (x2, y2) = (pmin(a[2], b[2]), pmin(a[3], b[3]));
    let inter = pmax(0.0, x2 - x1) * pmax(0.0, y2 - y1);
    let ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    if ua > 0.0 {
        inter / ua
    } else {
        0.0
    }
}

/// Fraction of square a=[cx, cy, s] inside square b.
pub fn frac_inside(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let (a_, b_) = (from_cs(a), from_cs(b));
    let iw = pmax(0.0, pmin(a_[2], b_[2]) - pmax(a_[0], b_[0]));
    let ih = pmax(0.0, pmin(a_[3], b_[3]) - pmax(a_[1], b_[1]));
    iw * ih / pmax((a_[2] - a_[0]) * (a_[3] - a_[1]), 1e-6)
}

/// Overlap of squares a and b as a fraction of the smaller one.
pub fn overlap_of_smaller(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let (a_, b_) = (from_cs(a), from_cs(b));
    let iw = pmax(0.0, pmin(a_[2], b_[2]) - pmax(a_[0], b_[0]));
    let ih = pmax(0.0, pmin(a_[3], b_[3]) - pmax(a_[1], b_[1]));
    iw * ih / pmax(pmin((a_[2] - a_[0]) * (a_[3] - a_[1]), (b_[2] - b_[0]) * (b_[3] - b_[1])), 1e-6)
}

/// Which piece of track tid frame t belongs to, given sorted cut frames.
pub fn piece_at(tid: &str, cuts: &[i64], t: i64) -> String {
    let mut p = tid.to_string();
    for c in cuts {
        if t >= *c {
            p = format!("{tid}~{c}");
        }
    }
    p
}

pub fn canon(tid: &str, merges: &HashMap<String, String>) -> String {
    let mut seen = HashSet::new();
    let mut t = tid.to_string();
    while let Some(n) = merges.get(&t) {
        if seen.contains(&t) {
            break;
        }
        seen.insert(t.clone());
        t = n.clone();
    }
    t
}

/// The scan's track a piece comes from ("88~140" is track 88's).
pub fn base(piece: &str) -> &str {
    piece.split('~').next().unwrap()
}

/// Frames from t to the nearest of the sorted evidence frames (10**6 if there are none).
fn since(frames: &[i64], t: i64) -> i64 {
    if frames.is_empty() {
        return 1_000_000;
    }
    let i = frames.partition_point(|&x| x < t);
    let mut best = i64::MAX;
    for j in [i as i64 - 1, i as i64] {
        if 0 <= j && (j as usize) < frames.len() {
            best = best.min((frames[j as usize] - t).abs());
        }
    }
    best
}

// ------------------------------------------------------------------ edits helpers
fn strs(v: &Py) -> Vec<String> {
    match v {
        Py::List(l) => l.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
        Py::Dict(d) => d.keys().map(str::to_string).collect(),
        _ => vec![],
    }
}

fn str_map(v: &Py) -> Vec<(String, String)> {
    v.as_dict().map(|d| d.iter().filter_map(|(k, v)| v.as_str().map(|s| (k.to_string(), s.to_string()))).collect()).unwrap_or_default()
}

/// edits.get(k) with Python's `or`: a falsy value is absent.
fn get_truthy<'a>(edits: &'a Py, k: &str) -> Option<&'a Py> {
    let v = edits.get(k);
    if v.truthy() {
        Some(v)
    } else {
        None
    }
}

/// review_server.links_for: same-person links for these answers, {piece: root}.
pub fn links_for(edits: &Py) -> HashMap<String, String> {
    let mut must: Vec<(String, String)> = vec![];
    for k in ["merges", "attach"] {
        if let Some(d) = get_truthy(edits, k).and_then(Py::as_dict) {
            for (a, b) in d.iter() {
                let b = b.py_str();
                if a != b {
                    must.push((a.to_string(), b));
                }
            }
        }
    }
    if let Some(people) = get_truthy(edits, "people").and_then(Py::as_dict) {
        for (_, p) in people.iter() {
            let ps: Vec<String> = p.get("pieces").as_list().map(|l| l.iter().map(Py::py_str).collect()).unwrap_or_default();
            for w in ps.windows(2) {
                must.push((w[0].clone(), w[1].clone()));
            }
        }
    }
    let mut parent: Vec<(String, String)> = vec![]; // insertion-ordered dict
    fn idx(parent: &mut Vec<(String, String)>, x: &str) -> usize {
        match parent.iter().position(|(k, _)| k == x) {
            Some(i) => i,
            None => {
                parent.push((x.to_string(), x.to_string()));
                parent.len() - 1
            }
        }
    }
    fn find(parent: &mut Vec<(String, String)>, x: &str) -> String {
        let mut i = idx(parent, x);
        while parent[i].1 != parent[i].0 {
            let p = parent[i].1.clone();
            let pi = idx(parent, &p);
            let gp = parent[pi].1.clone();
            parent[i].1 = gp.clone();
            i = idx(parent, &gp);
        }
        parent[i].0.clone()
    }
    for (a, b) in &must {
        let rb = find(&mut parent, b);
        let ra = find(&mut parent, a);
        let i = idx(&mut parent, &ra);
        parent[i].1 = rb;
    }
    let keys: Vec<String> = parent.iter().map(|(k, _)| k.clone()).collect();
    let mut groups: Vec<(String, Vec<String>)> = vec![];
    for x in keys {
        let r = find(&mut parent, &x);
        match groups.iter_mut().find(|(k, _)| *k == r) {
            Some((_, g)) => g.push(x),
            None => groups.push((r, vec![x])),
        }
    }
    let mut out = HashMap::new();
    for (_, g) in groups {
        if g.len() > 1 {
            let root = g.iter().min_by(|a, b| (a.chars().count(), a.as_str()).cmp(&(b.chars().count(), b.as_str()))).unwrap().clone();
            for x in g {
                out.insert(x, root.clone());
            }
        }
    }
    out
}

/// track.shown_pieces: individually shown pieces plus the pieces LISTED in a Person with show=True (links never
/// extend consent).
pub fn shown_pieces(edits: &Py) -> BTreeSet<String> {
    let mut out: BTreeSet<String> = edits.get("shown").as_list().map(|l| l.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default();
    if let Some(people) = get_truthy(edits, "people").and_then(Py::as_dict) {
        for (_, p) in people.iter() {
            if p.get("show").truthy() {
                if let Some(l) = p.get("pieces").as_list() {
                    out.extend(l.iter().map(Py::py_str));
                }
            }
        }
    }
    out
}

pub fn emoji_for(tid: &str, overrides: &Dict) -> String {
    if let Some(v) = overrides.get(tid) {
        return v.py_str();
    }
    let t = tid.strip_prefix('@').unwrap_or(tid);
    let h: u64 = t.chars().enumerate().map(|(i, ch)| (i as u64 + 1) * ch as u64).sum();
    EMOJI[(h % EMOJI.len() as u64) as usize].to_string()
}

fn num(v: &Py) -> Option<f64> {
    match v {
        Py::Int(i) => Some(*i as f64),
        Py::Float(f) => Some(*f),
        Py::Bool(b) => Some(*b as i64 as f64),
        Py::Str(s) => s.trim().parse().ok(),
        _ => None,
    }
}

fn trim_of(trims: Option<&Dict>, piece: &str, n: i64) -> (f64, f64) {
    if let Some(v) = trims.and_then(|d| d.get(piece)) {
        if v.truthy() {
            if let Some(l) = v.as_list() {
                if l.len() >= 2 {
                    return (num(&l[0]).unwrap_or(0.0), num(&l[1]).unwrap_or((n - 1) as f64));
                }
            }
        }
    }
    (0.0, (n - 1) as f64)
}

// ------------------------------------------------------------------ resolve
/// track.resolve(auto, with_cuts(edits), mats, n, links, dedupe): per-frame drawn boxes.
pub fn resolve(auto: &[Track], edits: &Py, mats: &[M23], n: usize, links: &HashMap<String, String>, dedupe: bool) -> Result<Vec<Vec<Item>>> {
    let ni = n as i64;
    let mut deleted: HashSet<String> = strs(edits.get("deleted")).into_iter().collect();
    for k in ["not_face", "not_real"] {
        if let Some(v) = get_truthy(edits, k) {
            deleted.extend(strs(v));
        }
    }
    let trims = edits.get("trims").as_dict();
    let mut merges: HashMap<String, String> = links.clone();
    for (k, v) in str_map(edits.get("merges")) {
        merges.insert(k, v);
    }
    if let Some(people) = get_truthy(edits, "people").and_then(Py::as_dict) {
        for (pid, p) in people.iter() {
            if let Some(l) = p.get("pieces").as_list() {
                for piece in l.iter().filter_map(Py::as_str) {
                    merges.insert(piece.to_string(), format!("@{pid}"));
                }
            }
        }
    }
    let part_of: HashSet<String> = get_truthy(edits, "part_of").map(strs).unwrap_or_default().into_iter().collect();
    let cuts = get_truthy(edits, "cuts").and_then(Py::as_dict);
    let mut per: Vec<Vec<Cand>> = vec![vec![]; n];
    for tr in auto {
        let obs: HashSet<i64> = tr.obs.iter().copied().collect();
        let mut cs: Vec<i64> = cuts
            .and_then(|d| d.get(&tr.id))
            .and_then(Py::as_list)
            .map(|l| l.iter().filter_map(|x| num(x).map(|f| f as i64)).collect())
            .unwrap_or_default();
        cs.sort();
        let mut seen: HashMap<String, Vec<i64>> = HashMap::new();
        for &t in &tr.obs {
            seen.entry(piece_at(&tr.id, &cs, t)).or_default().push(t);
        }
        for &(t, c) in &tr.cover {
            let piece = piece_at(&tr.id, &cs, t);
            if deleted.contains(&piece) {
                continue;
            }
            let (lo, hi) = trim_of(trims, &piece, ni);
            if lo <= t as f64 && t as f64 <= hi {
                if t < 0 || t >= ni {
                    return Err(anyhow!("track {} has a box at frame {t}, past the clip's {n} frames", tr.id));
                }
                let s = since(seen.get(&piece).map(Vec::as_slice).unwrap_or(&[]), t);
                per[t as usize].push(Cand {
                    tid: canon(&piece, &merges),
                    c,
                    kind: if obs.contains(&t) { "obs" } else { "coast" },
                    life: tr.obs.len() as i64,
                    piece,
                    since: s,
                });
            }
        }
    }
    if let Some(manual) = edits.get("manual").as_dict() {
        for (mid, m) in manual.iter() {
            if deleted.contains(mid) {
                continue;
            }
            let (lo, hi) = trim_of(trims, mid, ni);
            let cm = canon(mid, &merges);
            let keys: BTreeSet<i64> = m.get("keys").as_dict().map(|d| d.keys().filter_map(|k| k.trim().parse().ok()).collect()).unwrap_or_default();
            let mut seen: BTreeSet<i64> = keys.clone();
            if let Some(ev) = get_truthy(m, "ev").and_then(Py::as_list) {
                seen.extend(ev.iter().filter_map(|x| num(x).map(|f| f as i64)));
            }
            let seen: Vec<i64> = seen.into_iter().collect();
            let boxes = mm_core::edits::manual_boxes(m, mats, ni, mm_core::edits::MANUAL_HOLD).map_err(|e| anyhow!("manual cover {mid}: {e:?}"))?;
            for (t, c) in boxes {
                if lo <= t as f64 && t as f64 <= hi && (0..ni).contains(&t) {
                    per[t as usize].push(Cand {
                        tid: cm.clone(),
                        c,
                        kind: if keys.contains(&t) { "key" } else { "manual" },
                        life: 1_000_000,
                        piece: mid.to_string(),
                        since: since(&seen, t),
                    });
                }
            }
        }
    }
    let mut out: Vec<Vec<Item>> = per
        .into_iter()
        .map(|f| {
            if dedupe {
                dedupe_frame(f, &part_of)
            } else {
                f.into_iter().map(|x| Item { tid: x.tid, c: x.c, kind: if x.kind == "key" { "manual" } else { x.kind }, piece: x.piece }).collect()
            }
        })
        .collect();
    let behind: HashSet<(String, i64)> = auto.iter().flat_map(|tr| tr.behind.iter().map(move |&t| (tr.id.clone(), t))).collect();
    if BRIDGE > 0 {
        bridge(&mut out, mats, &behind);
    }
    if STEADY {
        let flows: HashMap<String, &HashMap<i64, [f64; 4]>> = auto.iter().map(|tr| (tr.id.clone(), &tr.flow)).collect();
        steady(&mut out, mats, if STEADY_FLOW && !flows.is_empty() { Some(&flows) } else { None });
    }
    Ok(out)
}

fn rank(kind: &str) -> i32 {
    KIND_RANK.iter().find(|(k, _)| *k == kind).unwrap().1
}

fn dedupe_frame(mut items: Vec<Cand>, part_of: &HashSet<String>) -> Vec<Item> {
    items.sort_by_key(|x| (x.since, rank(x.kind), -x.life)); // stable, as Python's sorted
    let mut keep: Vec<Cand> = vec![];
    for it in items {
        let b = from_cs(&it.c);
        if keep.iter().any(|k| k.tid != it.tid && iou(&b, &from_cs(&k.c)) >= 0.4) {
            continue;
        }
        if it.kind == "coast"
            && keep.iter().any(|k| {
                k.tid != it.tid && k.kind != "coast" && k.since <= SAME_PERSON_FRESH && pmax(frac_inside(&it.c, &k.c), frac_inside(&k.c, &it.c)) >= HELD_YIELDS
            })
        {
            continue;
        }
        let same = keep.iter().any(|k| k.tid == it.tid);
        let touching: Vec<usize> = keep
            .iter()
            .enumerate()
            .filter(|(_, k)| {
                k.tid == it.tid
                    && (part_of.contains(&it.piece) || part_of.contains(&k.piece) || pmax(frac_inside(&it.c, &k.c), frac_inside(&k.c, &it.c)) >= SAME_FACE_OVERLAP)
            })
            .map(|(i, _)| i)
            .collect();
        if !touching.is_empty() {
            if it.since <= SAME_PERSON_FRESH {
                let i = touching[0];
                let kb = from_cs(&keep[i].c);
                let u = [pmin(b[0], kb[0]), pmin(b[1], kb[1]), pmax(b[2], kb[2]), pmax(b[3], kb[3])];
                keep[i].c = [(u[0] + u[2]) / 2.0, (u[1] + u[3]) / 2.0, pmax(u[2] - u[0], u[3] - u[1])];
            }
            continue;
        }
        if !same || it.since <= SAME_PERSON_FRESH {
            keep.push(it);
        }
    }
    let live: Vec<[f64; 3]> = keep.iter().filter(|k| k.kind != "coast").map(|k| k.c).collect();
    keep.into_iter()
        .filter(|k| k.kind != "coast" || !live.iter().any(|l| frac_inside(&k.c, l) >= 0.7))
        .map(|x| Item { tid: x.tid, c: x.c, kind: if x.kind == "key" { "manual" } else { x.kind }, piece: x.piece })
        .collect()
}

// ------------------------------------------------------------------ numpy pieces
fn ext(m: &M23) -> M33 {
    [m[0], m[1], [0.0, 0.0, 1.0]]
}
const EYE: M33 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn scene(mats: &[M23], n: usize) -> Vec<M33> {
    let mut c = vec![EYE; n];
    for t in 1..n {
        let s = if t < mats.len() { ext(&mats[t]) } else { EYE };
        c[t] = np::matmul33(&s, &c[t - 1]);
    }
    c
}

/// sqrt|det m[:2, :2]| (np.linalg.det of the stacked 2x2s: dgetrf each).
fn kdet(m: &M33) -> f64 {
    np::det2(m[0][0], m[0][1], m[1][0], m[1][1]).abs().sqrt()
}

/// One row of np.einsum("nij,nj->ni") with j = 3 (module notes; mm_core::np).
fn ein_row(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    np::einsum_row3(a, b)
}

/// np.convolve(a, v) (mode "full"), as numpy computes it (mm_core::np::convolve).
pub fn convolve(a: &[f64], v: &[f64]) -> Vec<f64> {
    np::convolve(a, v)
}

fn gauss(sig: f64) -> Vec<f64> {
    let r = (3.0 * sig) as i64;
    (-r..=r).map(|x| {
        let q = x as f64 / sig;
        (-0.5 * (q * q)).exp()
    })
    .collect()
}

fn smooth(v: &[f64], g: &[f64]) -> Vec<f64> {
    let r = g.len() / 2;
    let a = convolve(v, g);
    let b = convolve(&vec![1.0; v.len()], g);
    (0..v.len()).map(|i| a[r + i] / b[r + i]).collect()
}

/// Group a frame's items by tid in first-seen order.
fn by_tid(f: &[Item]) -> Vec<(String, Vec<usize>)> {
    let mut cnt: Vec<(String, Vec<usize>)> = vec![];
    for (i, x) in f.iter().enumerate() {
        match cnt.iter_mut().find(|(k, _)| *k == x.tid) {
            Some((_, v)) => v.push(i),
            None => cnt.push((x.tid.clone(), vec![i])),
        }
    }
    cnt
}

pub fn bridge(per_frame: &mut [Vec<Item>], mats: &[M23], behind: &HashSet<(String, i64)>) {
    let n = per_frame.len();
    if n == 0 {
        return;
    }
    let c = scene(mats, n);
    let ci: Vec<M33> = c.iter().map(np::inv33).collect();
    let k: Vec<f64> = c.iter().map(kdet).collect();
    let mut at: Vec<(String, BTreeMap<usize, Vec<usize>>)> = vec![];
    for (t, f) in per_frame.iter().enumerate() {
        for (tid, ix) in by_tid(f) {
            let e = match at.iter().position(|(k, _)| *k == tid) {
                Some(i) => i,
                None => {
                    at.push((tid, BTreeMap::new()));
                    at.len() - 1
                }
            };
            at[e].1.insert(t, ix);
        }
    }
    let mut adds: Vec<(usize, Item)> = vec![];
    for (tid, frames) in &at {
        let ts: Vec<usize> = frames.keys().copied().collect();
        for w in ts.windows(2) {
            let (a, b) = (w[0], w[1]);
            if !(1 < b - a && b - a <= (BRIDGE + 1) as usize) || frames[&a].len() != 1 || frames[&b].len() != 1 {
                continue;
            }
            let xa = &per_frame[a][frames[&a][0]];
            let xb = &per_frame[b][frames[&b][0]];
            let (ca, cb) = (xa.c, xb.c);
            let pa = np::gemv(&ci[a], [ca[0], ca[1], 1.0]);
            let pb = np::gemv(&ci[b], [cb[0], cb[1], 1.0]);
            let sz = pmax(ca[2] / k[a], cb[2] / k[b]);
            if np::hypot(pa[0] - pb[0], pa[1] - pb[1]) > BRIDGE_REACH * (ca[2] / k[a] + cb[2] / k[b]) / 2.0 {
                continue;
            }
            let ends = [base(&xa.piece).to_string(), base(&xb.piece).to_string()];
            let piece = xa.piece.clone();
            for t in a + 1..b {
                if ends.iter().any(|e| behind.contains(&(e.clone(), t as i64))) {
                    continue;
                }
                let w = (t - a) as f64 / (b - a) as f64;
                let m = [0, 1, 2].map(|i| pa[i] * (1.0 - w) + pb[i] * w);
                let q = np::gemv(&c[t], m);
                let cc = [q[0], q[1], sz * k[t]];
                if per_frame[t].iter().any(|x| x.tid != *tid && x.kind != "coast" && overlap_of_smaller(&cc, &x.c) >= HELD_YIELDS) {
                    continue;
                }
                adds.push((t, Item { tid: tid.clone(), c: cc, kind: "coast", piece: piece.clone() }));
            }
        }
    }
    for (t, x) in adds {
        per_frame[t].push(x);
    }
}

pub fn steady(per_frame: &mut [Vec<Item>], mats: &[M23], flows: Option<&HashMap<String, &HashMap<i64, [f64; 4]>>>) {
    let mut one: Vec<(String, Vec<(usize, usize)>)> = vec![];
    for (t, f) in per_frame.iter().enumerate() {
        for (tid, ix) in by_tid(f) {
            if ix.len() == 1 {
                match one.iter_mut().find(|(k, _)| *k == tid) {
                    Some((_, v)) => v.push((t, ix[0])),
                    None => one.push((tid, vec![(t, ix[0])])),
                }
            }
        }
    }
    let (gp, gs) = (gauss(STEADY_POS), gauss(STEADY_SIZE));
    for (_, items) in one {
        let mut runs: Vec<Vec<(usize, usize)>> = vec![];
        let mut cur = vec![items[0]];
        for w in items.windows(2) {
            if w[1].0 == w[0].0 + 1 {
                cur.push(w[1]);
            } else {
                runs.push(std::mem::replace(&mut cur, vec![w[1]]));
            }
        }
        runs.push(cur);
        for run in runs {
            if run.len() < 2 {
                continue;
            }
            let raw: Vec<[f64; 3]> = run.iter().map(|&(t, i)| per_frame[t][i].c).collect();
            let mut g: Vec<M33> = vec![EYE];
            for w in run.windows(2) {
                let ((t, i), (_, i2)) = (w[0], w[1]);
                let (pa, pb) = (base(&per_frame[t][i].piece), base(&per_frame[t + 1][i2].piece));
                let f = match flows {
                    Some(fl) if pa == pb => fl.get(pa).and_then(|m| m.get(&(t as i64))).copied(),
                    _ => None,
                };
                let mut s = if t + 1 < mats.len() { ext(&mats[t + 1]) } else { EYE };
                if let Some([a, b, tx, ty]) = f {
                    let c = per_frame[t][i].c;
                    let want = [a * c[0] - b * c[1] + tx, b * c[0] + a * c[1] + ty];
                    let now = np::gemv(&[s[0], s[1]], [c[0], c[1], 1.0]);
                    s[0][2] += want[0] - now[0];
                    s[1][2] += want[1] - now[1];
                }
                let last = *g.last().unwrap();
                g.push(np::matmul33(&s, &last));
            }
            let gi: Vec<M33> = g.iter().map(np::inv33).collect();
            let k: Vec<f64> = g.iter().map(kdet).collect();
            let len = run.len();
            let p: Vec<[f64; 2]> = (0..len)
                .map(|j| {
                    let x = [raw[j][0], raw[j][1], 1.0];
                    [ein_row(&gi[j][0], &x), ein_row(&gi[j][1], &x)]
                })
                .collect();
            let rs: Vec<f64> = (0..len).map(|j| raw[j][2] / k[j]).collect();
            let env: Vec<f64> = (0..len)
                .map(|j| {
                    let lo = j.saturating_sub(STEADY_ENV);
                    let hi = (j + STEADY_ENV).min(len - 1);
                    rs[lo..=hi].iter().copied().fold(f64::NEG_INFINITY, f64::max)
                })
                .collect();
            let sx = smooth(&p.iter().map(|v| v[0]).collect::<Vec<_>>(), &gp);
            let sy = smooth(&p.iter().map(|v| v[1]).collect::<Vec<_>>(), &gp);
            let ss: Vec<f64> = smooth(&env.iter().map(|v| v.ln()).collect::<Vec<_>>(), &gs).into_iter().map(f64::exp).collect();
            for (j, &(t, i)) in run.iter().enumerate() {
                let x = [sx[j], sy[j], 1.0];
                let q = [ein_row(&g[j][0], &x), ein_row(&g[j][1], &x)];
                let size = ss[j] * k[j];
                let dev = pmax((q[0] - raw[j][0]).abs(), (q[1] - raw[j][1]).abs());
                let floor = STEADY_KEEP * raw[j][2] + 2.0 * dev;
                let size = if size.is_nan() || floor.is_nan() { f64::NAN } else { size.max(floor) };
                per_frame[t][i].c = [q[0], q[1], size];
            }
        }
    }
}

/// track.draw_order: far first. By depth (Occlusion.nearness) when a depth model is on, else by size.
pub fn draw_order(items: &[Item], nearness: Option<&dyn Fn(&Item) -> f64>) -> Vec<Item> {
    let mut v: Vec<(f64, f64, Item)> = items.iter().map(|x| (nearness.map(|f| f(x)).unwrap_or(0.0), x.c[2], x.clone())).collect();
    // Python's sorted on (float, float) tuples: NaN never compares less, so treat it as equal (stable)
    v.sort_by(|a, b| {
        let ka = (a.0, a.1);
        let kb = (b.0, b.1);
        if ka.0 < kb.0 {
            std::cmp::Ordering::Less
        } else if kb.0 < ka.0 {
            std::cmp::Ordering::Greater
        } else if ka.1 < kb.1 {
            std::cmp::Ordering::Less
        } else if kb.1 < ka.1 {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    v.into_iter().map(|x| x.2).collect()
}
