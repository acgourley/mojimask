//! Who's who and what's a face, as one log of facts (the new algorithm's projects). Port of `mojimask/facts.py`.
//!
//! Everything a person says about the boxes -- on the Decisions page, on People, in Edit -- is a fact appended to
//! PROJECT/facts.json. Nothing else stores it: the edits fields the pages draw from (people, emoji, not_face, ...) and
//! the answers the Decisions page shows are computed from the facts by [`derive`] / [`answers`], every time, the same
//! way. So an answer, a join in one page and an undo in another can't disagree: there is one record, and undo takes
//! the last facts back out of it.
//!
//! A fact: {"id", "at" (ms), "kind", "pieces": [...], "value", "by" ("user"), "page", "decision" (the question it
//! answers, if any), "answer" (the Decisions option it stands for, if any), "via" ("agent" when a program answered)}.
//! Kinds:
//!   same       pieces [a, b]   one person
//!   part       pieces [a, b]   a is part of b's face (a chin, neck or hair box): one person, and drawn as one face
//!   different  pieces [a, b]   not one person
//!   face       pieces [a]      value true: a face (masked); false: not a face (no emoji), "why": the reason when false
//!                              ("decision", "you", ...)
//!   show       pieces [...]    value true: unmasked; false: masked (the pieces of the person at the time)
//!   name       pieces [...]    value: their name ("" clears it)
//!   emoji      pieces [...]    value: their emoji ("" clears it), "preset": the emoji preset it was picked under
//!                              (palettes.json key, or "custom"). A pick belongs to its preset: another preset deals
//!                              everyone its own emoji, and coming back brings the picks back.
//!
//! Precedence: newer beats older. Joining is processed newest first: a "same" joins two groups unless a newer
//! "different" already keeps them apart, so the latest word on any pair wins and contradictions resolve the same way
//! every time. The scan's own rules are the baseline underneath (its hidden boxes; its tracks are its people).
//!
//! The facts belong to one scan: facts.json records the scan's version (newalgo.SCAN_VERSION) and a rescan throws
//! them away (box ids are the scan's).
//!
//! ## The Decisions page: each question's answer, from the facts (whichever page they came from)
//! Faces first: whether a box is a face at all is asked before who it is. On jazz (200 tracks) 47 of 58 "different
//! people" answers paired someone with a box later answered "not a face" -- a question that answer settles anyway.
//!
//! Same-person questions not worth asking (2026-11-13). The 50 s party clip asked 179, 169 answered "different": each
//! person walks down their candidates best first, and person 42 got 14 "different" in a row (scores 1.24 down to
//! 0.56). A pair left apart only means one person's emoji may change partway; they stay covered.
//! - [`ASK_SCORE`]: below this match score a pair isn't asked. The lowest score any of your "same person" answers has
//!   had: 0.82 (jazz), 0.87 (party_clip_c), 1.03 (the party clip).
//! - [`STOP_AFTER`]: after this many "different people" answers in a row about one person (since their last join),
//!   their weaker candidates aren't asked. Replayed on the party clip, jazz and party_clip_c, with both rules: no join
//!   missed (2026-11-13). A person counts whichever box they are in, grey or green (2026-11-17: counting only the grey
//!   box let the same person be offered again and again as everyone else's candidate; numbers in the commit).
//!
//! Rust notes: facts and decision records are kept as [`Py`] values (insertion-ordered, like the Python dicts), so
//! the files and the decision lists come out byte-identical. One known difference: `derive` walks the anchors a
//! show/name/emoji fact touches in the order its pieces name them, where Python iterates a `set` (hash order, which
//! changes run to run). It only decides where a person who first appears through such a fact lands in `people`'s
//! key order, never what any person holds.
use crate::palettes;
use crate::pyjson::{write_atomic, Dict, Py};
use crate::pydict;
use std::collections::{HashMap, HashSet};
use std::path::Path;

pub const FILE: &str = "facts.json";
pub const KINDS: [&str; 7] = ["same", "part", "different", "face", "show", "name", "emoji"];
pub const JOIN: [&str; 2] = ["same", "part"];
pub const ATTRS: [&str; 3] = ["show", "name", "emoji"];
// The Decisions page's answers (decisions.json and the facts store them as these strings; decisions.html labels them)
pub const A_FACE: &str = "a face";
pub const A_NOT_FACE: &str = "not a face";
pub const A_SAME: &str = "same person";
pub const A_DIFFERENT: &str = "different people";
pub const A_SAME_FACE: &str = "same face";
pub const A_SOMEONE_ELSE: &str = "someone else";
pub const A_SOMEONE_NEW: &str = "someone new";
pub const A_JOINED: &str = "joined already"; // settled by your other answers
pub const A_KEPT_APART: &str = "kept apart";
pub const A_OPEN: &str = "open"; // same-person questions: (answers() turns "open" into None, asked now)
pub const A_WAITING: &str = "waiting";

pub const FACE_KINDS: [&str; 3] = ["track", "blip", "overlap"];

/// ## The Decisions page's order
/// The one order the app, `mm-cli decisions` and the rules' benchmark all list questions in (so the benchmark asks
/// what a person would be asked next): the open questions first, faces before who's who (a face answer can settle
/// same-person questions), then the weightiest, then the one seen in the most frames. Settled questions keep the
/// order they came in. A question is open while nothing has answered it ("by" is empty).
pub struct PageKey {
    settled: bool,
    rank: u8,
    weight: f64,
    frames: f64,
}

pub fn page_key(open: bool, kind: &str, weight: f64, frames: f64) -> PageKey {
    if !open {
        return PageKey { settled: true, rank: 0, weight: 0.0, frames: 0.0 };
    }
    let rank = match kind {
        "whose" => 0,
        k if FACE_KINDS.contains(&k) => 1,
        "same" => 2,
        _ => 3,
    };
    PageKey { settled: false, rank, weight, frames }
}

/// [`page_key`] of a record as [`answers`] (or the project server) gives it.
pub fn page_key_of(d: &Py) -> PageKey {
    let num = |v: &Py| v.as_f64().unwrap_or(0.0);
    page_key(!d.get("by").truthy(), d.get("kind").as_str().unwrap_or(""), num(d.get("weight")), num(d.get("evidence").get("frames")))
}

/// Put `ds` in the page's order (stable).
pub fn page_sort<T>(ds: &mut [T], key: impl Fn(&T) -> PageKey) {
    ds.sort_by(|a, b| {
        let (a, b) = (key(a), key(b));
        a.settled.cmp(&b.settled).then(a.rank.cmp(&b.rank)).then(b.weight.total_cmp(&a.weight)).then(b.frames.total_cmp(&a.frames))
    });
}
pub const ASK_SCORE: f64 = 0.8;
pub const STOP_AFTER: i64 = 4;
pub const WEAK_RULE: &str = "rule: weak match (score under 0.8), not asked";
pub const STOP_RULE: &str = "rule: you said 'different' to the last 4 best matches of one of these two, so weaker ones aren't asked";
pub const HIDE: [&str; 4] =
    [A_NOT_FACE, "dropped: too weak", "covered by the person's box (while they are)", "covered by another box"];

fn is_join(k: &str) -> bool {
    JOIN.contains(&k)
}
fn is_pairish(k: &str) -> bool {
    is_join(k) || k == "different"
}

/// A fact's fields.
pub fn fid(f: &Py) -> i64 {
    f.get("id").as_i64().unwrap_or(0)
}
pub fn fkind(f: &Py) -> &str {
    f.get("kind").as_str().unwrap_or("")
}
pub fn fpieces(f: &Py) -> Vec<String> {
    f.get("pieces").as_list().map(|l| l.iter().map(Py::py_str).collect()).unwrap_or_default()
}
fn pair_key(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}
fn by_id(facts: &[Py]) -> Vec<&Py> {
    let mut v: Vec<&Py> = facts.iter().collect();
    v.sort_by_key(|f| fid(f));
    v
}

/// The log as a document: {"scan_version", "facts": [...], "last_id"}.
pub fn facts_of(doc: &Py) -> &[Py] {
    doc.get("facts").as_list().map(|v| v.as_slice()).unwrap_or(&[])
}
fn facts_mut(doc: &mut Py) -> &mut Vec<Py> {
    let d = doc.as_dict_mut().expect("facts doc is a dict");
    if !matches!(d.get("facts"), Some(Py::List(_))) {
        d.insert("facts", Py::List(vec![]));
    }
    match d.get_mut("facts") {
        Some(Py::List(l)) => l,
        _ => unreachable!(),
    }
}

/// The log. A log made for another scan version is set aside (facts.discarded.v<its version>.json) and an empty
/// one started. `scan_version` None: take whatever is there.
pub fn load(work: &Path, scan_version: Option<&Py>) -> Result<Py, String> {
    let path = work.join(FILE);
    let doc = if path.exists() { Some(crate::pyjson::load(&path)?) } else { None };
    let stale = match (&doc, scan_version) {
        (None, _) => true,
        (Some(d), Some(v)) => d.get("scan_version") != v,
        _ => false,
    };
    if stale {
        if let Some(d) = &doc {
            let aside = work.join(format!("facts.discarded.v{}.json", d.get("scan_version").py_str()));
            std::fs::rename(&path, aside).map_err(|e| e.to_string())?;
        }
        let d = pydict![("scan_version", scan_version.cloned().unwrap_or(Py::Null)), ("facts", Py::List(vec![]))];
        save(work, &d)?;
        return Ok(d);
    }
    Ok(doc.unwrap())
}

pub fn save(work: &Path, doc: &Py) -> Result<(), String> {
    write_atomic(&work.join(FILE), &doc.dumps_indent(1)).map_err(|e| e.to_string())
}

/// Add facts (dicts with kind, pieces, value[, why, answer]); returns their ids. via="agent": given by a program
/// answering for you (decisions_cli), not by a click. They still count as your answers (by "user") everywhere.
/// `now_ms`: the time stamped on them.
pub fn append(doc: &mut Py, facts: &[Py], page: &str, decision: Option<&str>, via: Option<&str>, now_ms: i64)
    -> Result<Vec<i64>, String> {
    // never reused, even after a retract
    let mut n = facts_of(doc).iter().map(fid).chain([doc.get("last_id").as_i64().unwrap_or(0)]).max().unwrap_or(0);
    let mut ids = vec![];
    for f in facts {
        let k = fkind(f);
        if !KINDS.contains(&k) {
            return Err(format!("unknown fact kind {}", f.get("kind").py_str()));
        }
        n += 1;
        let fd = f.as_dict().ok_or("a fact is a dict")?;
        let mut out = Dict::new();
        for (key, v) in fd.iter() {
            if !["id", "at", "by", "page", "decision", "via"].contains(&key) {
                out.insert(key, v.clone()); // the log's to set
            }
        }
        out.insert("id", Py::Int(n));
        out.insert("at", Py::Int(now_ms));
        out.insert("by", Py::from("user"));
        out.insert("page", Py::from(page));
        if let Some(d) = decision.filter(|d| !d.is_empty()) {
            out.insert("decision", Py::from(d));
        }
        if let Some(v) = via.filter(|v| !v.is_empty()) {
            out.insert("via", Py::from(v));
        }
        let ps = f.get("pieces").as_list().ok_or("pieces: a list")?;
        out.insert("pieces", Py::List(ps.iter().map(|p| Py::Str(p.py_str())).collect()));
        facts_mut(doc).push(Py::Dict(out));
        ids.push(n);
    }
    doc.as_dict_mut().unwrap().insert("last_id", Py::Int(n));
    Ok(ids)
}

pub fn retract(doc: &mut Py, ids: &[i64]) {
    let ids: HashSet<i64> = ids.iter().copied().collect();
    facts_mut(doc).retain(|f| !ids.contains(&fid(f)));
}

/// A person's id: their scan box with the shortest id (a mask you drew, "m1", only when there is none), so joining
/// your mask to someone keeps their id and emoji.
fn anchor_of(group: &[String]) -> String {
    group.iter().min_by(|a, b| (a.starts_with('m'), a.chars().count(), a.as_str()).cmp(&(b.starts_with('m'), b.chars().count(), b.as_str())))
        .cloned().unwrap_or_default()
}

/// {piece: anchor}, in the order Python's dict holds it (first named, first).
pub type Comp = Vec<(String, String)>;

/// The result of [`partition`]: comp in order, a lookup, and the anchor pairs kept apart.
pub struct Partition {
    pub comp: Comp,
    pub of: HashMap<String, String>,
    pub kept: HashSet<(String, String)>,
}

impl Partition {
    /// `comp.get(x, x)`
    pub fn c(&self, x: &str) -> String {
        self.of.get(x).cloned().unwrap_or_else(|| x.to_string())
    }
}

/// {piece: anchor} for every piece a same/different fact names, and the set of anchor pairs kept apart.
pub fn partition(facts: &[Py]) -> Partition {
    let mut names: Vec<String> = vec![];
    let mut idx: HashMap<String, usize> = HashMap::new();
    let mut parent: Vec<usize> = vec![];
    fn find(x: &str, names: &mut Vec<String>, idx: &mut HashMap<String, usize>, parent: &mut Vec<usize>) -> usize {
        let mut i = *idx.entry(x.to_string()).or_insert_with(|| {
            names.push(x.to_string());
            parent.push(names.len() - 1);
            names.len() - 1
        });
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    let mut apart: HashMap<usize, HashSet<usize>> = HashMap::new();
    let mut fs: Vec<&Py> = facts.iter().filter(|f| is_pairish(fkind(f))).collect();
    fs.sort_by_key(|f| -fid(f)); // newest first
    for f in fs {
        let ps = fpieces(f);
        if ps.len() < 2 {
            continue;
        }
        let a = find(&ps[0], &mut names, &mut idx, &mut parent);
        let b = find(&ps[1], &mut names, &mut idx, &mut parent);
        if fkind(f) == "different" {
            if a != b {
                apart.entry(a).or_default().insert(b);
                apart.entry(b).or_default().insert(a);
            }
            continue;
        }
        if a == b || apart.get(&a).is_some_and(|s| s.contains(&b)) {
            continue; // already one, or a newer "different" keeps them apart
        }
        parent[a] = b;
        let moved = apart.remove(&a).unwrap_or_default();
        let into = apart.entry(b).or_default();
        into.extend(moved);
        let xs: Vec<usize> = into.iter().copied().collect();
        for x in xs {
            if let Some(s) = apart.get_mut(&x) {
                if s.remove(&a) {
                    s.insert(b);
                }
            }
        }
    }
    let mut order: Vec<usize> = vec![];
    let mut groups: HashMap<usize, Vec<String>> = HashMap::new();
    let n = names.len();
    let mut root = vec![0; n];
    for x in 0..n {
        let r = find(&names[x].clone(), &mut names, &mut idx, &mut parent);
        root[x] = r;
        if !groups.contains_key(&r) {
            order.push(r);
        }
        groups.entry(r).or_default().push(names[x].clone());
    }
    let anchor: HashMap<usize, String> = groups.iter().map(|(r, g)| (*r, anchor_of(g))).collect();
    let comp: Comp = (0..n).map(|x| (names[x].clone(), anchor[&root[x]].clone())).collect();
    let of = comp.iter().cloned().collect();
    let mut kept = HashSet::new();
    for (a, bs) in &apart {
        for b in bs {
            if let (Some(x), Some(y)) = (anchor.get(a), anchor.get(b)) {
                kept.insert(pair_key(x, y));
            }
        }
    }
    Partition { comp, of, kept }
}

/// The preset a pick without one is taken to be from (palettes.json's "default").
pub fn emoji_preset() -> &'static str {
    palettes::default_key()
}

/// The edits fields that come from the facts: people, emoji (per person: the picks made under `preset`, the emoji
/// preset in use), not_face, part_of {piece: piece} (boxes drawn as one face with their person's), and empty merges /
/// not_same / shown / attach (people carries all of it). hidden: the scan's own hidden boxes (its baseline).
/// Returns the fields in Python's order.
pub fn derive(facts: &[Py], hidden: &[String], preset: &Py) -> Vec<(String, Py)> {
    let default = Py::from(emoji_preset());
    let preset = preset.or(&default).clone();
    let part = partition(facts);
    // latest fact per pair, keyed in first-seen order
    let mut latest: Vec<((String, String), &Py)> = vec![];
    for f in by_id(facts) {
        if !is_pairish(fkind(f)) {
            continue;
        }
        let ps = fpieces(f);
        if ps.len() < 2 {
            continue;
        }
        let k = pair_key(&ps[0], &ps[1]);
        match latest.iter_mut().find(|(q, _)| *q == k) {
            Some(e) => e.1 = f,
            None => latest.push((k, f)),
        }
    }
    let mut part_of = Dict::new();
    for (_, f) in &latest {
        let ps = fpieces(f);
        if fkind(f) == "part" && part.of.get(&ps[0]) == part.of.get(&ps[1]) {
            part_of.insert(ps[0].clone(), Py::Str(ps[1].clone()));
        }
    }
    // groups: anchor -> members, in first-seen order
    let mut gorder: Vec<String> = vec![];
    let mut groups: HashMap<String, Vec<String>> = HashMap::new();
    for (x, a) in &part.comp {
        if !groups.contains_key(a) {
            gorder.push(a.clone());
        }
        let g = groups.entry(a.clone()).or_default();
        if !g.contains(x) {
            g.push(x.clone());
        }
    }
    let mut attrs: HashMap<String, Dict> = HashMap::new(); // anchor -> {kind: value}, newest fact touching the group wins
    for f in by_id(facts) {
        let k = fkind(f);
        if !ATTRS.contains(&k) {
            continue;
        }
        let mut seen: Vec<String> = vec![];
        for p in fpieces(f) {
            let a = part.c(&p);
            if !seen.contains(&a) {
                seen.push(a);
            }
        }
        for a in seen {
            if !groups.contains_key(&a) {
                gorder.push(a.clone());
                groups.insert(a.clone(), vec![a.clone()]);
            }
            if k == "emoji" && *f.get("preset").or(&default) != preset {
                // under another preset: not worn now, but they stay a person (the same id in every preset)
                attrs.entry(a).or_default().setdefault("picked", Py::Bool(true));
                continue;
            }
            attrs.entry(a).or_default().insert(k, f.get("value").clone());
        }
    }
    let mut people = Dict::new();
    let mut emoji = Dict::new();
    let empty = Dict::new();
    for a in &gorder {
        let g = &groups[a];
        let at = attrs.get(a).unwrap_or(&empty);
        if g.len() < 2 && at.is_empty() {
            continue;
        }
        let mut ps = g.clone();
        ps.sort_by(|x, y| (x.chars().count(), x).cmp(&(y.chars().count(), y)));
        people.insert(
            a.clone(),
            pydict![
                ("name", at.get("name").cloned().unwrap_or(Py::from(""))),
                ("pieces", Py::List(ps.into_iter().map(Py::Str).collect())),
                ("show", at.get("show").map(Py::truthy).unwrap_or(false)),
            ],
        );
        if let Some(e) = at.get("emoji").filter(|e| e.truthy()) {
            emoji.insert(format!("@{a}"), e.clone());
        }
    }
    vec![
        ("people".into(), Py::Dict(people)),
        ("emoji".into(), Py::Dict(emoji)),
        ("not_face".into(), Py::Dict(not_face(facts, hidden))),
        ("part_of".into(), Py::Dict(part_of)),
        ("merges".into(), Py::Dict(Dict::new())),
        ("not_same".into(), Py::List(vec![])),
        ("shown".into(), Py::List(vec![])),
        ("attach".into(), Py::Dict(Dict::new())),
    ]
}

/// {piece: why} for the boxes that aren't faces: the scan's hidden ones, then your face facts, newest last.
pub fn not_face(facts: &[Py], hidden: &[String]) -> Dict {
    let mut nf = Dict::new();
    for p in hidden {
        nf.insert(p.clone(), Py::from("decision"));
    }
    for f in by_id(facts) {
        if fkind(f) != "face" {
            continue;
        }
        let Some(p) = fpieces(f).into_iter().next() else { continue };
        if f.get("value").truthy() {
            nf.remove(&p);
        } else {
            let why = f.get("why");
            nf.insert(p, if why.truthy() { why.clone() } else { Py::from("you") });
        }
    }
    nf
}

fn about_piece(d: &Py) -> String {
    d.get("about").get("piece").py_str()
}
fn about_person(d: &Py) -> Option<String> {
    let q = d.get("about").get("person");
    (!q.is_null()).then(|| q.py_str())
}

/// The facts a decision's answer stands for.
pub fn facts_for_answer(d: &Py, ans: &Py) -> Vec<Py> {
    let p = about_piece(d);
    let q = about_person(d);
    let qv = q.clone().map(Py::Str).unwrap_or(Py::Null);
    let k = d.get("kind").as_str().unwrap_or("");
    let ans_s = ans.as_str().unwrap_or("");
    if k == "whose" {
        // a box you made: whose face it is (a person's box), or someone new (it stays its own person)
        if ans_s == A_SOMEONE_NEW {
            return vec![pydict![("kind", "face"), ("pieces", vec![Py::from(&p)]), ("value", true), ("answer", ans.clone())]];
        }
        return vec![pydict![("kind", "same"), ("pieces", vec![Py::from(&p), Py::Str(ans.py_str())]), ("answer", ans.clone())]];
    }
    if k == "same" {
        let kind = if ans_s == A_SAME { "same" } else { "different" };
        return vec![pydict![("kind", kind), ("pieces", vec![Py::from(&p), qv]), ("answer", ans.clone())]];
    }
    if ans_s == A_NOT_FACE {
        return vec![pydict![("kind", "face"), ("pieces", vec![Py::from(&p)]), ("value", false), ("why", "decision"), ("answer", ans.clone())]];
    }
    let mut out = vec![pydict![("kind", "face"), ("pieces", vec![Py::from(&p)]), ("value", true), ("answer", ans.clone())]];
    if k == "overlap" && ans_s == A_SAME_FACE {
        out.push(pydict![("kind", "part"), ("pieces", vec![Py::from(&p), qv.clone()]), ("answer", ans.clone())]);
    }
    if k == "overlap" && ans_s == A_SOMEONE_ELSE {
        out.push(pydict![("kind", "different"), ("pieces", vec![Py::from(&p), qv]), ("answer", ans.clone())]);
    }
    out
}

fn set(d: &mut Py, k: &str, v: Py) {
    d.as_dict_mut().unwrap().insert(k, v);
}
fn strv(s: &str) -> Py {
    Py::from(s)
}

/// The decision records with their answers as the facts stand: a question answered on the Decisions page shows
/// that answer; one settled elsewhere (a box marked not a face in Edit, two people joined) shows how; the
/// same-person questions are re-picked (one open question per person at a time).
pub fn answers(decisions: &[Py], facts: &[Py], hidden: &[String]) -> Vec<Py> {
    let part = partition(facts);
    let gone: HashSet<String> = not_face(facts, hidden).keys().map(String::from).collect();
    let c = |x: &str| part.c(x);
    let mut latest_face: HashMap<String, &Py> = HashMap::new();
    let mut latest_pair: HashMap<(String, String), &Py> = HashMap::new();
    let mut by_decision: HashMap<String, Vec<&Py>> = HashMap::new(); // decision id -> its facts, in log order
    for f in facts {
        let dd = f.get("decision");
        if !dd.is_null() {
            by_decision.entry(dd.py_str()).or_default().push(f);
        }
    }
    for f in by_id(facts) {
        let ps = fpieces(f);
        if fkind(f) == "face" {
            if let Some(p) = ps.first() {
                latest_face.insert(p.clone(), f);
            }
        } else if is_pairish(fkind(f)) && ps.len() >= 2 {
            latest_pair.insert(pair_key(&ps[0], &ps[1]), f);
        }
    }
    let mut out: Vec<Py> = vec![];
    for d0 in decisions {
        let mut d = d0.clone();
        let dd = d0.as_dict().unwrap();
        let ans = dd.get("rule_answer").cloned().unwrap_or_else(|| d0.get("answer").clone());
        let by = dd.get("rule_by").cloned().unwrap_or_else(|| d0.get("by").clone());
        set(&mut d, "answer", ans);
        set(&mut d, "by", by);
        let p = about_piece(&d);
        let q = about_person(&d);
        let kind = d.get("kind").as_str().unwrap_or("").to_string();
        let qs = q.clone().unwrap_or_else(|| "None".into());
        let mine = by_decision.get(&d.get("id").py_str());
        if let Some(mine) = mine.filter(|m| !m.is_empty()) {
            let last = mine[mine.len() - 1];
            set(&mut d, "answer", last.get("answer").clone());
            set(&mut d, "by", strv("user"));
            if last.get("via").truthy() {
                set(&mut d, "via", last.get("via").clone());
            }
        } else if kind == "whose" {
            // only its own answer settles it
        } else if kind == "same" {
            if let Some(pf) = latest_pair.get(&pair_key(&p, &qs)) {
                set(&mut d, "answer", strv(if is_join(fkind(pf)) { A_SAME } else { A_DIFFERENT }));
                set(&mut d, "by", strv("user"));
            }
        } else {
            let ff = latest_face.get(&p);
            let pf = q.as_ref().and_then(|q| latest_pair.get(&pair_key(&p, q)));
            if let (true, Some(pf)) = (kind == "overlap", pf) {
                set(&mut d, "answer", strv(if is_join(fkind(pf)) { A_SAME_FACE } else { A_SOMEONE_ELSE }));
                set(&mut d, "by", strv("user"));
            } else if let Some(ff) = ff {
                let v = ff.get("value").truthy();
                let a = if v {
                    if kind != "overlap" { strv(A_FACE) } else { d.get("answer").clone() }
                } else {
                    strv(A_NOT_FACE)
                };
                set(&mut d, "answer", a);
                set(&mut d, "by", strv("user"));
                if kind == "overlap" && v && d.get("answer").as_str().is_some_and(|a| HIDE.contains(&a)) {
                    set(&mut d, "answer", Py::Null); // shown again, but whose face is still the question
                    set(&mut d, "by", Py::Null);
                }
            }
        }
        out.push(d);
    }
    // same-person questions not answered directly: settled by the people as they stand, else one open per person,
    // and only once both boxes are known to be faces (faces first)
    let face_open: HashSet<String> = out.iter()
        .filter(|d| FACE_KINDS.contains(&d.get("kind").as_str().unwrap_or("")) && d.get("answer").is_null())
        .map(about_piece).collect();
    let mut busy: HashSet<String> = HashSet::new();
    let same: Vec<usize> = (0..out.len()).filter(|&i| out[i].get("kind").as_str() == Some("same")).collect();
    // person -> "different people" answers to questions about them (either box) since a join
    let mut run: HashMap<String, i64> = HashMap::new();
    let about: HashMap<String, (String, String)> = same.iter()
        .map(|&i| (out[i].get("id").py_str(), (out[i].get("about").get("person").py_str(), about_piece(&out[i])))).collect();
    for f in by_id(facts) {
        let dd = f.get("decision");
        if dd.is_null() {
            continue;
        }
        if let Some((who, other)) = about.get(&dd.py_str()) {
            if is_pairish(fkind(f)) {
                for k in [c(who), c(other)] {
                    let v = if fkind(f) == "different" { run.get(&k).copied().unwrap_or(0) + 1 } else { 0 };
                    run.insert(k, v);
                }
            }
        }
    }
    let score = |d: &Py| d.get("evidence").get("score").as_f64().unwrap_or(0.0);
    for &i in &same {
        let d = &mut out[i];
        if d.get("by").as_str() == Some("user") {
            continue;
        }
        let person = d.get("about").get("person").py_str();
        let piece = about_piece(d);
        let (a, b) = (c(&person), c(&piece));
        if d.get("depth_apart").truthy() && a != b {
            let why = d.get("depth_apart").clone();
            set(d, "answer", strv(A_DIFFERENT)); // the scan's depth rule (MOJIMASK_DEPTH)
            set(d, "by", why);
            continue;
        }
        let (ans, by) = if gone.contains(&person) || gone.contains(&piece) {
            (strv(A_NOT_FACE), strv("rule: one of the two isn't a face (no emoji), so nobody to join"))
        } else if a == b {
            (strv(A_JOINED), strv("rule: your other answers make them one person"))
        } else if part.kept.contains(&pair_key(&a, &b)) {
            (strv(A_KEPT_APART), strv("rule: you said these two are different people"))
        } else if score(d) < ASK_SCORE {
            (strv(A_DIFFERENT), strv(WEAK_RULE))
        } else if run.get(&a).copied().unwrap_or(0).max(run.get(&b).copied().unwrap_or(0)) >= STOP_AFTER {
            (strv(A_DIFFERENT), strv(STOP_RULE))
        } else {
            (strv(A_OPEN), Py::Null)
        };
        set(d, "answer", ans);
        set(d, "by", by);
    }
    let mut pending: HashMap<String, i64> = HashMap::new();
    for &i in &same {
        if out[i].get("answer").as_str() == Some(A_OPEN) {
            for x in [out[i].get("about").get("person").py_str(), about_piece(&out[i])] {
                *pending.entry(x).or_default() += 1;
            }
        }
    }
    for d in out.iter_mut() {
        // a face question's weight: the same-person questions its answer can settle (asked first)
        if FACE_KINDS.contains(&d.get("kind").as_str().unwrap_or("")) {
            let w = pending.get(&about_piece(d)).copied().unwrap_or(0);
            set(d, "weight", Py::Int(w));
        }
    }
    let mut open: Vec<usize> = same.iter().copied().filter(|&i| out[i].get("answer").as_str() == Some(A_OPEN)).collect();
    open.sort_by(|&x, &y| score(&out[y]).partial_cmp(&score(&out[x])).unwrap_or(std::cmp::Ordering::Equal)); // stable
    for i in open {
        let d = &mut out[i];
        let person = d.get("about").get("person").py_str();
        let piece = about_piece(d);
        let (a, b) = (c(&person), c(&piece));
        if face_open.contains(&person) || face_open.contains(&piece) {
            set(d, "answer", strv(A_WAITING));
            set(d, "by", strv("rule: asked once we know both boxes are faces"));
        } else if busy.contains(&a) || busy.contains(&b) {
            set(d, "answer", strv(A_WAITING));
            set(d, "by", strv("rule: asked once the open question about this person is answered"));
        } else {
            set(d, "answer", Py::Null);
            busy.insert(a);
            busy.insert(b);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fact(id: i64, kind: &str, pieces: &[&str], value: Py) -> Py {
        pydict![("id", id), ("kind", kind), ("pieces", pieces.iter().map(|p| Py::from(*p)).collect::<Vec<_>>()), ("value", value)]
    }

    #[test]
    fn newest_word_on_a_pair_wins() {
        // older "same", newer "different": kept apart
        let fs = vec![fact(1, "same", &["5", "7"], Py::Null), fact(2, "different", &["5", "7"], Py::Null)];
        let p = partition(&fs);
        assert_ne!(p.c("5"), p.c("7"));
        assert!(p.kept.contains(&pair_key("5", "7")));
        // newer "same" over an older "different": joined, anchored on the shortest id
        let fs = vec![fact(1, "different", &["5", "17"], Py::Null), fact(2, "same", &["17", "5"], Py::Null)];
        let p = partition(&fs);
        assert_eq!(p.c("17"), "5");
    }

    #[test]
    fn masks_never_anchor_when_a_scan_box_is_there() {
        let fs = vec![fact(1, "same", &["m1", "123"], Py::Null)];
        assert_eq!(partition(&fs).c("m1"), "123");
    }

    #[test]
    fn derive_people_emoji_and_not_face() {
        let mut e = fact(3, "emoji", &["7"], Py::from("🐶"));
        e.as_dict_mut().unwrap().insert("preset", Py::from("zooanon"));
        let fs = vec![
            fact(1, "same", &["5", "7"], Py::Null),
            fact(2, "name", &["7"], Py::from("Ann")),
            e,
            fact(4, "face", &["9"], Py::Bool(false)),
            fact(5, "face", &["3"], Py::Bool(true)),
        ];
        let d: Dict = derive(&fs, &["3".into(), "4".into()], &Py::from("zooanon")).into_iter().collect();
        assert_eq!(d.get("people").unwrap().dumps(), r#"{"5": {"name": "Ann", "pieces": ["5", "7"], "show": false}}"#);
        assert_eq!(d.get("emoji").unwrap().dumps(), r#"{"@5": "\ud83d\udc36"}"#);
        assert_eq!(d.get("not_face").unwrap().dumps(), r#"{"4": "decision", "9": "you"}"#);
        // under another preset the pick isn't worn, but the person stays
        let d: Dict = derive(&fs, &[], &Py::Null).into_iter().collect();
        assert_eq!(d.get("emoji").unwrap().dumps(), "{}");
        assert!(d.get("people").unwrap().as_dict().unwrap().contains("5"));
    }

    #[test]
    fn answers_settle_and_repick() {
        let mk = |id: &str, kind: &str, piece: i64, person: Option<i64>, score: f64| {
            let about = match person {
                Some(q) => pydict![("piece", piece), ("person", q)],
                None => pydict![("piece", piece)],
            };
            pydict![("id", id), ("kind", kind), ("about", about), ("evidence", pydict![("score", score)]),
                    ("answer", Py::Null), ("by", Py::Null), ("rule_answer", Py::Null), ("rule_by", Py::Null)]
        };
        let ds = vec![
            mk("track:5", "track", 5, None, 0.0),
            mk("same:1:5", "same", 5, Some(1), 1.5),
            mk("same:1:6", "same", 6, Some(1), 1.2),
            mk("same:2:7", "same", 7, Some(2), 0.5),
        ];
        let out = answers(&ds, &[], &[]);
        assert!(out[0].get("answer").is_null());
        assert_eq!(out[0].get("weight"), &Py::Int(1));
        assert_eq!(out[1].get("answer").as_str(), Some(A_WAITING)); // 5 isn't known to be a face yet
        assert!(out[2].get("answer").is_null()); // the best open one about person 1
        assert_eq!(out[3].get("by").as_str(), Some(WEAK_RULE));
        // say 5 is a face: same:1:5 is asked, same:1:6 waits on person 1
        let mut f = fact(1, "face", &["5"], Py::Bool(true));
        f.as_dict_mut().unwrap().insert("decision", Py::from("track:5"));
        f.as_dict_mut().unwrap().insert("answer", Py::from(A_FACE));
        let out = answers(&ds, &[f], &[]);
        assert_eq!(out[0].get("answer").as_str(), Some(A_FACE));
        assert!(out[1].get("answer").is_null());
        assert_eq!(out[2].get("answer").as_str(), Some(A_WAITING));
    }

    #[test]
    fn append_and_retract_never_reuse_ids() {
        let mut doc = pydict![("scan_version", 2), ("facts", Py::List(vec![]))];
        let f = pydict![("kind", "face"), ("pieces", vec![Py::Int(5)]), ("value", true)];
        assert_eq!(append(&mut doc, &[f.clone()], "edit", None, Some("agent"), 7).unwrap(), vec![1]);
        retract(&mut doc, &[1]);
        assert_eq!(append(&mut doc, &[f], "edit", Some("track:5"), None, 8).unwrap(), vec![2]);
        assert_eq!(facts_of(&doc)[0].dumps(),
                   r#"{"kind": "face", "pieces": ["5"], "value": true, "id": 2, "at": 8, "by": "user", "page": "edit", "decision": "track:5"}"#);
        assert!(append(&mut doc, &[pydict![("kind", "nope"), ("pieces", Py::List(vec![]))]], "", None, None, 0).is_err());
    }

    #[test]
    fn facts_for_answers() {
        let d = pydict![("id", "overlap:3"), ("kind", "overlap"), ("about", pydict![("piece", 3), ("person", 9)])];
        let fs = facts_for_answer(&d, &Py::from(A_SOMEONE_ELSE));
        assert_eq!(fs.len(), 2);
        assert_eq!(fs[1].dumps(), r#"{"kind": "different", "pieces": ["3", "9"], "answer": "someone else"}"#);
    }

    #[test]
    fn page_order_open_first_faces_before_same_settled_as_they_came() {
        let d = |id: i64, kind: &str, by: Py, w: i64, fr: i64| pydict![("id", id), ("kind", kind), ("by", by), ("weight", w), ("evidence", pydict![("frames", fr)])];
        let mut ds = vec![
            d(1, "same", strv("user"), 9, 9), d(2, "same", Py::Null, 0, 50), d(3, "track", strv("rule: x"), 0, 1),
            d(4, "blip", Py::Null, 0, 5), d(5, "track", Py::Null, 2, 1), d(6, "overlap", Py::Null, 0, 7), d(7, "whose", Py::Null, 0, 0),
        ];
        page_sort(&mut ds, page_key_of);
        let ids: Vec<i64> = ds.iter().map(|d| d.get("id").as_i64().unwrap()).collect();
        assert_eq!(ids, [7, 5, 6, 4, 2, 1, 3]);
    }
}
