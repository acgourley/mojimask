//! The scan's rules measured: how many questions a clip asks you, and how many the rules get wrong. Jazz is scored
//! against its census; the party clips, which have none, by their worst case.
//!
//!     cargo run --release -p mm-bench -- --note "what changed" [--clips jazz,party360,party4k]
//!     rule-bench --history [CLIP]           (every logged jazz run, one line each)
//!     rule-bench --score PROJECT            (a jazz project someone answered, against the census)
//!
//! Each clip's detection, camera motion (mats_v2.npy) and depth cache (anything-small) are made once, by
//! `mm-scan-tool scan`, in out/native/rules/<clip>/base_native. Every run copies that (APFS clones) to
//! out/native/rules/<clip>/run and reruns only the decide step with today's rules (`mm-scan-tool decide`, built
//! first), so a run measures today's rules and nothing else. (Checked against the Python bench it replaces,
//! decision_bench.py + rule_bench.py, 2026-11-17: from the same bases, with Python's motion, the same jazz line,
//! records and party counts.)
//!
//! - jazz (test_clips/jazz.mp4, a public-domain film): the Decisions page played through. Each open question, in the
//!   page's order (faces first, then by weight, then by frames), is answered by the census and the next ones re-picked
//!   from the facts (mm_core::facts::answers), as the page does. The census is native/bench/truth/jazz.json: every
//!   face boxed by hand on every 4th frame (538 faces, 33 people), checked; between two checked frames a person's box
//!   slides from one to the other. A scan box takes the label of the census box it overlaps most (IoU >= 0.3, else
//!   sitting >= 0.7 inside a face), frame by frame, majority. Scored:
//!     asked        questions answered, by kind: the cost to you
//!     rule wrong   questions the rules settled without asking against the census's answer, by harm: face uncovered
//!                  (hid a face and nothing else covers it: the worst), needless cover, double cover, missed / wrong
//!                  join
//!     result       after the replay, on checked frames only: census faces never boxed, boxed but left uncovered,
//!                  people split (one person's emoji changes) or mixed
//!   It prints one line (decision_bench's) and logs the record to native/bench/data/decisions_history.jsonl.
//! - party360 (`party clip.mp4`, 640x360) and party4k (party_clip_4k.mov), in the main checkout's test_clips/: real
//!   guests, so numbers only, no frame is looked at. No census, so the worst case is counted: every face question
//!   answered "a face" (overlaps "someone else"), every same-person question "different people". That is the most
//!   same-person questions a person can be asked. One JSON line each.
//!
//! Every line, with the note, also goes to out/native/rules/runs.jsonl.
//!
//! `--score PROJECT` grades a jazz project that someone (a person, an agent through `mm-cli decisions`) answered: each
//! answer they gave against the census's, and the result their answers leave (the same "result" numbers), beside the
//! census's own play-through of that project as the best it could be. Nothing is logged.
use anyhow::{anyhow, bail, Context, Result};
use mm_core::facts as FX;
use mm_core::pydict;
use mm_core::pyjson::{self, Dict, Py};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const DEPTH: &str = "anything-small";
const REPLAY_CAP: usize = 5000; // questions answered in one replay before it gives up (the same question keeps coming back)
const IOU: f64 = 0.3;
const INSIDE: f64 = 0.7;
const CENSUS_REACH: i64 = 2; // frames from a census frame a "not" box (or a person in only one) is carried
/// the files a decide step writes, left out of the copy of a base
const RESULT: [&str; 5] = ["proto_tracks.json", "edits.json", "facts.json", "bench_last.json", "proto"];

/// The Decisions page's order: faces first (decisions.html)
fn rank(kind: &str) -> i32 {
    match kind {
        "whose" => 0,
        "track" | "overlap" | "blip" => 1,
        _ => 2,
    }
}

// ---------------------------------------------------------------- places
struct Places {
    native: PathBuf,
    root: PathBuf,
    main: PathBuf,
    out: PathBuf,
}

impl Places {
    fn find() -> Result<Places> {
        let native = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize()?;
        let root = native.parent().ok_or_else(|| anyhow!("native/ has no parent"))?.to_path_buf();
        // the main checkout (the party clips and the models live there; a worktree has only jazz): git's common
        // directory's parent
        let g = Command::new("git").arg("-C").arg(&root).args(["rev-parse", "--path-format=absolute", "--git-common-dir"])
            .output()?;
        let common = String::from_utf8_lossy(&g.stdout).trim().to_string();
        let main = Path::new(&common).parent().map(Path::to_path_buf).unwrap_or_else(|| root.clone());
        let out = root.join("out/native/rules");
        Ok(Places { native, root, main, out })
    }
    fn truth(&self) -> PathBuf {
        self.native.join("bench/truth/jazz.json")
    }
    fn history(&self) -> PathBuf {
        self.native.join("bench/data/decisions_history.jsonl")
    }
    fn tool(&self) -> PathBuf {
        self.native.join("target/release/mm-scan-tool")
    }
    fn models(&self) -> PathBuf {
        let here = self.root.join("models");
        if here.join("onnx").is_dir() {
            here
        } else {
            self.main.join("models")
        }
    }
    fn clip(&self, clip: &str) -> Result<PathBuf> {
        Ok(match clip {
            "jazz" => self.root.join("test_clips/jazz.mp4"),
            "party360" => self.main.join("test_clips/party clip.mp4"),
            "party4k" => self.main.join("test_clips/party_clip_4k.mov"),
            c => bail!("unknown clip {c}: jazz, party360 or party4k"),
        })
    }
}

struct Cfg {
    places: Places,
}

impl Cfg {
    /// base_video: scanned with the frames read from the video (2026-11-20; base_native's depth was made from
    /// PROJECT/frames' JPEGs).
    fn base_name(&self) -> &'static str {
        "base_video"
    }
    fn tool_args(&self) -> Vec<String> {
        vec!["--depth".into(), DEPTH.into(), "--models".into(), self.places.models().display().to_string()]
    }
    /// Runs mm-scan-tool; its standard output.
    fn run_tool(&self, args: &[String]) -> Result<String> {
        let mut c = Command::new(self.places.tool());
        c.args(args).args(self.tool_args()).env("MM_SCAN_QUIET", "1");
        let o = c.output().with_context(|| format!("running {}", self.places.tool().display()))?;
        if !o.status.success() {
            let tail = |b: &[u8]| {
                let s = String::from_utf8_lossy(b).to_string();
                s[s.len().saturating_sub(3000)..].to_string()
            };
            bail!("mm-scan-tool {} failed:\n{}{}", args[0], tail(&o.stdout), tail(&o.stderr));
        }
        Ok(String::from_utf8_lossy(&o.stdout).to_string())
    }

    /// The clip's base project (detection, camera motion, depth), scanned the first time.
    fn base(&self, clip: &str) -> Result<PathBuf> {
        let b = self.places.out.join(clip).join(self.base_name());
        if b.join(format!("dets/depth_{DEPTH}.json")).exists() {
            return Ok(b);
        }
        if b.exists() {
            std::fs::remove_dir_all(&b)?;
        }
        std::fs::create_dir_all(&b)?;
        let video = self.places.clip(clip)?;
        eprintln!("{clip}: scanning a base project (once) into {}", b.display());
        self.run_tool(&["scan".into(), video.display().to_string(), b.display().to_string(), "--plan".into(), "default".into()])?;
        Ok(b)
    }

    /// A fresh copy of the base with today's decide step run on it; (project, seconds, each stage's seconds).
    fn decide(&self, clip: &str) -> Result<(PathBuf, f64, Py)> {
        let base = self.base(clip)?;
        let run = self.places.out.join(clip).join("run");
        if run.exists() {
            std::fs::remove_dir_all(&run)?;
        }
        copy_tree(&base, &run)?;
        // the run reads its frames from the clip in this checkout (a base can outlive the worktree that scanned it)
        let mp = run.join("meta.json");
        let mut meta = mm_core::pyjson::load(&mp).map_err(|e| anyhow!(e))?;
        if let Some(d) = meta.as_dict_mut() {
            d.insert("video", Py::Str(std::fs::canonicalize(self.places.clip(clip)?)?.display().to_string()));
        }
        std::fs::write(&mp, meta.dumps())?;
        let t0 = Instant::now();
        let out = self.run_tool(&["decide".into(), run.display().to_string()])?;
        let secs = py_round1(t0.elapsed().as_secs_f64());
        let info: serde_json::Value = serde_json::from_str(out.trim()).unwrap_or_default();
        let mut stages = Py::from_serde(&info["info"]["stage_seconds"]);
        if let Some(d) = stages.as_dict_mut() {
            let pre = info["info"]["faces_prefetch_seconds"].as_f64().unwrap_or(0.0);
            d.insert("faces_prefetch", Py::Float((pre * 100.0).round() / 100.0));
        }
        Ok((run, secs, stages))
    }
}

/// shutil.copytree with RESULT and decisions_*.json left out (at any depth). std::fs::copy clones on APFS.
fn copy_tree(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for e in std::fs::read_dir(src)? {
        let e = e?;
        let name = e.file_name().to_string_lossy().to_string();
        if RESULT.contains(&name.as_str()) || (name.starts_with("decisions_") && name.ends_with(".json")) {
            continue;
        }
        if e.file_type()?.is_dir() {
            copy_tree(&e.path(), &dst.join(&name))?;
        } else {
            std::fs::copy(e.path(), dst.join(&name))?;
        }
    }
    Ok(())
}

fn py_round1(x: f64) -> f64 {
    pyjson::py_round(x, 1)
}

fn now() -> String {
    Command::new("date").arg("+%Y-%m-%d %H:%M").output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

fn load(p: &Path) -> Result<Py> {
    pyjson::load(p).map_err(|e| anyhow!("{}: {e}", p.display()))
}

/// Every decision record of a project (decisions_*.json, in file-name order).
fn decisions(proj: &Path) -> Result<Vec<Py>> {
    let mut names: Vec<String> = std::fs::read_dir(proj)?
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().to_string()))
        .filter(|f| f.starts_with("decisions_") && f.ends_with(".json"))
        .collect();
    names.sort();
    let mut out = vec![];
    for f in names {
        out.extend(load(&proj.join(f))?.get("decisions").as_list().cloned().unwrap_or_default());
    }
    Ok(out)
}

fn strs(v: &Py) -> Vec<String> {
    v.as_list().map(|l| l.iter().map(Py::py_str).collect()).unwrap_or_default()
}

/// A counter that keeps first-seen order (Python's Counter as dict(...) prints it).
#[derive(Default)]
struct Counter(Vec<(String, i64)>);

impl Counter {
    fn add(&mut self, k: &str, n: i64) {
        match self.0.iter_mut().find(|(q, _)| q == k) {
            Some(e) => e.1 += n,
            None => self.0.push((k.to_string(), n)),
        }
    }
    fn total(&self) -> i64 {
        self.0.iter().map(|e| e.1).sum()
    }
    fn py(&self) -> Py {
        Py::Dict(self.0.iter().map(|(k, v)| (k.clone(), Py::Int(*v))).collect())
    }
}

/// The open questions in the Decisions page's order; the first is asked next.
fn open_in_order(out: &[Py]) -> Vec<&Py> {
    let mut op: Vec<&Py> = out.iter().filter(|d| d.get("answer").is_null()).collect();
    let key = |d: &Py| {
        (rank(d.get("kind").as_str().unwrap_or("")), -d.get("weight").as_f64().unwrap_or(0.0),
         -d.get("evidence").get("frames").as_f64().unwrap_or(0.0))
    };
    op.sort_by(|a, b| key(a).partial_cmp(&key(b)).unwrap_or(std::cmp::Ordering::Equal)); // stable, as sorted()
    op
}

// ---------------------------------------------------------------- truth
type Box4 = [f64; 4];

fn iou(a: &Box4, b: &Box4) -> f64 {
    let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h;
    if u > 0.0 {
        w * h / u
    } else {
        0.0
    }
}

/// Share of box a inside box b.
fn inside(a: &Box4, b: &Box4) -> f64 {
    let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    w * h / ((a[2] - a[0]) * (a[3] - a[1])).max(1e-9)
}

fn box4(v: &Py) -> Box4 {
    let l = v.as_list().cloned().unwrap_or_default();
    let f = |i: usize| l.get(i).and_then(Py::as_f64).unwrap_or(0.0);
    [f(0), f(1), f(2), f(3)]
}

#[derive(Clone)]
struct GBox {
    b: Box4,
    l: String, // "P<person>" a face, "H<person>" a head whose face isn't visible, "NF" not a head
}

/// The census as boxes for every frame (n frames), and the checked frames. Between two checked frames a person's box
/// slides from one to the other; a person in only one of them, and "not" boxes, are carried CENSUS_REACH frames.
fn census_frames(path: &Path, n: i64) -> Result<(HashMap<i64, Vec<GBox>>, HashSet<i64>)> {
    let c = load(path)?;
    let mut checked: Vec<i64> = c.get("checked").as_list().map(|l| l.iter().filter_map(Py::as_i64).collect()).unwrap_or_default();
    checked.sort();
    let lab = |k: &str| match k {
        "face" | "tiny" => Some("P"),
        "head" => Some("H"),
        "not" => Some("NF"),
        _ => None,
    };
    let mut at: HashMap<i64, Vec<GBox>> = HashMap::new();
    for &t in &checked {
        let mut v = vec![];
        for b in c.get("frames").get(&t.to_string()).as_list().cloned().unwrap_or_default() {
            let k = b.get("c").as_str().unwrap_or("");
            let pre = lab(k).ok_or_else(|| anyhow!("census frame {t}: box class {k}"))?;
            let l = if k == "not" { pre.to_string() } else { format!("{pre}{}", b.get("p").py_str()) };
            v.push(GBox { b: box4(b.get("b")), l });
        }
        at.insert(t, v);
    }
    let people = |t: Option<i64>| -> Vec<(String, Box4)> {
        // {label: box} for the non-"not" boxes (a later box of the same label replaces an earlier one, in its place)
        let mut d: Vec<(String, Box4)> = vec![];
        if let Some(t) = t {
            for g in &at[&t] {
                if g.l == "NF" {
                    continue;
                }
                match d.iter_mut().find(|(l, _)| *l == g.l) {
                    Some(e) => e.1 = g.b,
                    None => d.push((g.l.clone(), g.b)),
                }
            }
        }
        d
    };
    let mut out = HashMap::new();
    for t in 0..n {
        if let Some(v) = at.get(&t) {
            out.insert(t, v.clone());
            continue;
        }
        let a = checked.iter().copied().filter(|&u| u < t).max();
        let z = checked.iter().copied().filter(|&u| u > t).min();
        let (pa, pz) = (people(a), people(z));
        let mut boxes = vec![];
        let mut labels: Vec<&String> = pa.iter().map(|e| &e.0).collect();
        labels.extend(pz.iter().map(|e| &e.0).filter(|l| !pa.iter().any(|e| &e.0 == *l)));
        for l in labels {
            let ba = pa.iter().find(|e| &e.0 == l).map(|e| e.1);
            let bz = pz.iter().find(|e| &e.0 == l).map(|e| e.1);
            match (ba, bz) {
                (Some(x), Some(y)) => {
                    let (a, z) = (a.unwrap(), z.unwrap());
                    let w = (t - a) as f64 / (z - a) as f64;
                    let mut b = [0.0; 4];
                    for i in 0..4 {
                        b[i] = x[i] * (1.0 - w) + y[i] * w;
                    }
                    boxes.push(GBox { b, l: l.clone() });
                }
                (Some(x), None) if t - a.unwrap() <= CENSUS_REACH => boxes.push(GBox { b: x, l: l.clone() }),
                (None, Some(y)) if z.unwrap() - t <= CENSUS_REACH => boxes.push(GBox { b: y, l: l.clone() }),
                _ => {}
            }
        }
        // the nearer checked frame (the earlier one on a tie, as Python's min keeps the first)
        let near = [a, z].into_iter().flatten().min_by_key(|u| (u - t).abs());
        if let Some(near) = near {
            if (near - t).abs() <= CENSUS_REACH {
                boxes.extend(at[&near].iter().filter(|g| g.l == "NF").cloned());
            }
        }
        out.insert(t, boxes);
    }
    Ok((out, checked.into_iter().collect()))
}

/// Which boxes are faces, and whose: the census.
struct Truth {
    w: f64,
    h: f64,
    pt: Py,
    frames: HashMap<i64, Vec<GBox>>,
    scored: HashSet<i64>,
    lab: std::cell::RefCell<HashMap<String, (Option<String>, f64)>>,
}

impl Truth {
    fn new(census: &Path, proj: &Path) -> Result<Truth> {
        let meta = load(&proj.join("meta.json"))?;
        let pt = load(&proj.join("proto_tracks.json"))?;
        let n = meta.get("frames").as_i64().ok_or_else(|| anyhow!("meta.json: frames"))?;
        let (frames, scored) = census_frames(census, n)?;
        Ok(Truth {
            w: meta.get("width").as_f64().unwrap_or(1.0),
            h: meta.get("height").as_f64().unwrap_or(1.0),
            pt,
            frames,
            scored,
            lab: Default::default(),
        })
    }

    fn track(&self, piece: &str) -> &Py {
        self.pt.get("tracks").get(piece)
    }

    /// A scan box [cx, cy, size] in pixels as fractions of the frame.
    fn frac(&self, c: &Py) -> Box4 {
        let l = c.as_list().cloned().unwrap_or_default();
        let f = |i: usize| l.get(i).and_then(Py::as_f64).unwrap_or(0.0);
        let h = f(2) / 2.0;
        [(f(0) - h) / self.w, (f(1) - h) / self.h, (f(0) + h) / self.w, (f(1) + h) / self.h]
    }

    /// The census label of box b at frame t: a face it overlaps (IoU >= IOU), else a head, else a box called not a
    /// face, else a face it sits inside (a chin or hair box of that face), else None.
    fn match_(&self, t: i64, b: &Box4) -> Option<String> {
        let empty = vec![];
        let gs = self.frames.get(&t).unwrap_or(&empty);
        // max((score, label)): a tie on the score goes to the greater label, as Python's tuple max does
        let best = |want: char, score: &dyn Fn(&Box4) -> f64| -> Option<(f64, String)> {
            let mut best: Option<(f64, String)> = None;
            for g in gs.iter().filter(|g| g.l.starts_with(want)) {
                let v = score(&g.b);
                let better = match &best {
                    None => true,
                    Some((bv, bl)) => v > *bv || (v == *bv && g.l > *bl),
                };
                if better {
                    best = Some((v, g.l.clone()));
                }
            }
            best
        };
        for want in ['P', 'H', 'N'] {
            if let Some((v, l)) = best(want, &|g| iou(b, g)) {
                if v >= IOU {
                    return Some(l);
                }
            }
        }
        match best('P', &|g| inside(b, g)) {
            Some((v, l)) if v >= INSIDE => Some(l),
            _ => None,
        }
    }

    /// (label, share of frames): "P<id>", "H<id>", "NF", or None (no census box there: not a face).
    fn label(&self, piece: &str) -> (Option<String>, f64) {
        if let Some(v) = self.lab.borrow().get(piece) {
            return v.clone();
        }
        let mut c: Vec<(Option<String>, i64)> = vec![];
        let obs = self.track(piece).get("obs_c").as_dict().cloned().unwrap_or_default();
        for (t, o) in obs.iter() {
            let m = self.match_(t.parse().unwrap_or(-1), &self.frac(o));
            let k = if m.as_deref().is_some_and(|m| m.starts_with("NF")) { Some("NF".to_string()) } else { m };
            match c.iter_mut().find(|e| e.0 == k) {
                Some(e) => e.1 += 1,
                None => c.push((k, 1)),
            }
        }
        // Counter.most_common(1): the first key with the highest count
        let mut top: Option<&(Option<String>, i64)> = None;
        for e in &c {
            if top.map_or(true, |t| e.1 > t.1) {
                top = Some(e);
            }
        }
        let (lab, n) = top.cloned().unwrap_or((None, 0));
        let v = (lab, n as f64 / obs.len().max(1) as f64);
        self.lab.borrow_mut().insert(piece.to_string(), v.clone());
        v
    }

    /// The census people a scan track's own boxes land on ({id: frames}, at least 2 frames each): more than one is a
    /// track that is two people.
    fn people_in(&self, piece: &str) -> usize {
        let mut c: HashMap<String, i64> = HashMap::new();
        for (t, o) in self.track(piece).get("obs_c").as_dict().map(|d| d.iter().collect::<Vec<_>>()).unwrap_or_default() {
            if let Some(m) = self.match_(t.parse().unwrap_or(-1), &self.frac(o)) {
                if m.starts_with('P') {
                    *c.entry(m[1..].to_string()).or_default() += 1;
                }
            }
        }
        c.values().filter(|&&v| v >= 2).count()
    }

    /// Whose head this box is (a face or the back of a head), or None.
    fn person(&self, piece: &str) -> Option<String> {
        self.label(piece).0.filter(|l| l.starts_with('P') || l.starts_with('H')).map(|l| l[1..].to_string())
    }

    fn is_face(&self, piece: &str) -> bool {
        self.label(piece).0.is_some_and(|l| l.starts_with('P'))
    }

    /// The census's answer to a question.
    fn answer(&self, d: &Py) -> &'static str {
        let p = d.get("about").get("piece").py_str();
        let q = d.get("about").get("person");
        let q = (!q.is_null()).then(|| q.py_str());
        match d.get("kind").as_str().unwrap_or("") {
            "track" | "blip" => {
                if self.is_face(&p) {
                    FX::A_FACE
                } else {
                    FX::A_NOT_FACE
                }
            }
            "overlap" => {
                if !self.is_face(&p) {
                    FX::A_NOT_FACE
                } else if self.person(&p) == q.as_deref().and_then(|q| self.person(q)) {
                    FX::A_SAME_FACE
                } else {
                    FX::A_SOMEONE_ELSE
                }
            }
            "same" => {
                let a = self.person(&p);
                if a.is_some() && a == q.as_deref().and_then(|q| self.person(q)) {
                    FX::A_SAME
                } else {
                    FX::A_DIFFERENT
                }
            }
            _ => FX::A_SOMEONE_NEW, // "whose"
        }
    }
}

// ---------------------------------------------------------------- replay
struct Replay {
    facts: Vec<Py>,
    asked: Counter,
    log: Vec<Py>,
    last: Vec<Py>, // the records' answers once nothing is open
}

/// Play the Decisions page through: the first open question answered by `oracle`, the facts appended, again.
fn replay(ds: &[Py], hidden: &[String], oracle: &dyn Fn(&Py) -> &'static str) -> Result<Replay> {
    let mut doc = pydict![("facts", Py::List(vec![]))];
    let (mut asked, mut log) = (Counter::default(), vec![]);
    loop {
        let out = FX::answers(ds, FX::facts_of(&doc), hidden);
        let op = open_in_order(&out);
        if op.is_empty() || asked.total() as usize > REPLAY_CAP {
            if !op.is_empty() {
                eprintln!("warning: replay stopped after {REPLAY_CAP} questions, {} still open", op.len());
            }
            let facts = FX::facts_of(&doc).to_vec();
            return Ok(Replay { facts, asked, log, last: out });
        }
        let d = op[0].clone();
        let a = oracle(&d);
        let id = d.get("id").py_str();
        FX::append(&mut doc, &FX::facts_for_answer(&d, &Py::from(a)), "bench", Some(&id), None, 0).map_err(|e| anyhow!(e))?;
        asked.add(d.get("kind").as_str().unwrap_or(""), 1);
        log.push(pydict![("id", id), ("answer", a), ("evidence", d.get("evidence").clone()), ("why", d.get("why").clone())]);
    }
}

// ---------------------------------------------------------------- scoring
/// The covers after the replay: every box not hidden and not answered "not a face", per frame.
struct Covers {
    nf: HashSet<String>,
    comp: FX::Partition,
    at: HashMap<i64, Vec<Box4>>,
    any: HashMap<i64, Vec<Box4>>, // every box the scan made, hidden or not
}

impl Covers {
    fn new(truth: &Truth, facts: &[Py], hidden: &[String]) -> Covers {
        let nf: HashSet<String> = FX::not_face(facts, hidden).keys().map(String::from).collect();
        let (mut at, mut any): (HashMap<i64, Vec<Box4>>, HashMap<i64, Vec<Box4>>) = Default::default();
        for (k, tr) in truth.pt.get("tracks").as_dict().map(|d| d.iter().collect::<Vec<_>>()).unwrap_or_default() {
            for (t, c) in tr.get("cover").as_dict().map(|d| d.iter().collect::<Vec<_>>()).unwrap_or_default() {
                let t: i64 = t.parse().unwrap_or(-1);
                if !nf.contains(k) {
                    at.entry(t).or_default().push(truth.frac(c));
                }
                any.entry(t).or_default().push(truth.frac(c));
            }
        }
        Covers { nf, comp: FX::partition(facts), at, any }
    }

    fn hit(g: &Box4, boxes: Option<&Vec<Box4>>) -> bool {
        boxes.is_some_and(|bs| bs.iter().any(|b| iou(g, b) >= IOU || inside(g, b) >= INSIDE))
    }

    /// Frames where this box's census face has no cover at all.
    fn uncovered(&self, truth: &Truth, piece: &str) -> i64 {
        let mut n = 0;
        for (t, o) in truth.track(piece).get("obs_c").as_dict().map(|d| d.iter().collect::<Vec<_>>()).unwrap_or_default() {
            let t: i64 = t.parse().unwrap_or(-1);
            if let Some(m) = truth.match_(t, &truth.frac(o)).filter(|m| m.starts_with('P')) {
                if let Some(g) = truth.frames[&t].iter().find(|g| g.l == m) {
                    n += !Covers::hit(&g.b, self.at.get(&t)) as i64;
                }
            }
        }
        n
    }
}

/// A rule's answer, and the census answers that agree with it.
fn agrees(said: &str) -> Vec<&str> {
    match said {
        "kept: covered" => vec![FX::A_FACE, FX::A_SAME_FACE, FX::A_SOMEONE_ELSE],
        "dropped: too weak" | "covered by another box" => vec![FX::A_NOT_FACE],
        "covered by the person's box (while they are)" => vec![FX::A_NOT_FACE, FX::A_SAME_FACE],
        "left as it is" => vec![FX::A_SOMEONE_ELSE],
        s => vec![s],
    }
}
const HIDES: [&str; 4] = [FX::A_NOT_FACE, "dropped: too weak", "covered by another box", "covered by the person's box (while they are)"];

/// A rule's name with its numbers taken out ("rule: ... (score 0.62)" -> "rule: ..."), at most 60 characters.
fn rule_name(by: &str) -> String {
    let head = by.split('(').next().unwrap_or("").trim();
    let mut s = String::new();
    let mut in_num = false;
    for ch in head.chars() {
        if ch.is_ascii_digit() {
            if !in_num {
                s.push('N');
            }
            in_num = true;
        } else {
            s.push(ch);
            in_num = false;
        }
    }
    s.chars().take(60).collect()
}

/// Questions the scan's rules settled without asking, against the census, by rule. A wrong answer's harm:
///   face uncovered    hid a face, and nothing else covers it in some frames (the worst)
///   hidden, covered   hid a face, but another cover hides it in every frame (no harm)
///   needless cover    covered a box the census calls not a face, or any box on a checked frame no face matches
///   cover, unlabelled covered a box with no census box there (between checked frames; likely not a face)
///   double cover      left a part of someone's face as its own box (two emoji on one face)
///   missed / wrong join   same-person rules
fn rule_errors(last: &[Py], truth: &Truth, cov: &Covers) -> (Vec<(String, Counter)>, Vec<Py>) {
    let mut by_rule: Vec<(String, Counter)> = vec![];
    let mut wrong = vec![];
    for d in last {
        let by = d.get("by");
        let said = d.get("answer").as_str().map(String::from);
        if by.is_null() || by.as_str() == Some("user") {
            continue;
        }
        let said_s = said.as_deref().unwrap_or("");
        if [FX::A_WAITING, FX::A_JOINED, FX::A_KEPT_APART, FX::A_OPEN].contains(&said_s) && said.is_some() {
            continue;
        }
        let by_s = by.py_str();
        if said_s == FX::A_NOT_FACE && by_s.starts_with("rule: one of the two") {
            continue; // follows from a face answer
        }
        let o = truth.answer(d);
        let rule = rule_name(&by_s);
        let idx = match by_rule.iter().position(|(r, _)| *r == rule) {
            Some(i) => i,
            None => {
                by_rule.push((rule.clone(), Counter::default()));
                by_rule.len() - 1
            }
        };
        if said.is_some() && agrees(said_s).contains(&o) {
            by_rule[idx].1.add("right", 1);
            continue;
        }
        let piece = d.get("about").get("piece").py_str();
        let mut frames = 0;
        let h = if d.get("kind").as_str() == Some("same") {
            if o == FX::A_SAME { "missed join" } else { "wrong join" }
        } else if HIDES.contains(&said_s) {
            frames = cov.uncovered(truth, &piece);
            if frames > 0 { "face uncovered" } else { "hidden, covered" }
        } else if o == FX::A_NOT_FACE {
            let checked = truth.track(&piece).get("obs_c").as_dict()
                .is_some_and(|obs| obs.keys().any(|t| t.parse().is_ok_and(|t: i64| truth.scored.contains(&t))));
            if truth.label(&piece).0.as_deref() == Some("NF") || checked { "needless cover" } else { "cover, unlabelled" }
        } else if said_s == "left as it is" && o == FX::A_SAME_FACE {
            "double cover"
        } else {
            "other"
        };
        by_rule[idx].1.add(h, 1);
        let q = d.get("about").get("person");
        let ql = if q.is_null() { Py::Null } else { truth.label(&q.py_str()).0.map(Py::Str).unwrap_or(Py::Null) };
        wrong.push(pydict![("id", d.get("id").clone()), ("rule", by.clone()), ("said", d.get("answer").clone()),
                           ("truth", o), ("harm", h), ("frames", frames),
                           ("label", vec![truth.label(&piece).0.map(Py::Str).unwrap_or(Py::Null), ql])]);
    }
    (by_rule, wrong)
}

/// After the replay, on checked frames: census face-frames the scan never boxed, those it boxed but left uncovered,
/// and people split (one census person across several, so their emoji changes) or mixed.
fn result(truth: &Truth, cov: &Covers) -> Py {
    let (mut total, mut missed, mut uncovered) = (0i64, 0i64, 0i64);
    let mut who: HashSet<String> = HashSet::new();
    for (t, gs) in &truth.frames {
        if !truth.scored.contains(t) {
            continue;
        }
        for g in gs.iter().filter(|g| g.l.starts_with('P')) {
            total += 1;
            if !Covers::hit(&g.b, cov.any.get(t)) {
                missed += 1;
            } else if !Covers::hit(&g.b, cov.at.get(t)) {
                uncovered += 1;
                who.insert(g.l.clone());
            }
        }
    }
    let keys: Vec<String> = truth.pt.get("tracks").as_dict().map(|d| d.keys().map(String::from).collect()).unwrap_or_default();
    let mut person: HashMap<String, HashSet<Option<String>>> = HashMap::new(); // final person -> census people
    for k in &keys {
        if !cov.nf.contains(k) && truth.is_face(k) {
            person.entry(cov.comp.c(k)).or_default().insert(truth.person(k));
        }
    }
    let mut of: HashMap<Option<String>, HashSet<String>> = HashMap::new();
    for (p, labs) in &person {
        for lab in labs {
            of.entry(lab.clone()).or_default().insert(p.clone());
        }
    }
    let tracks_mixed = keys.iter().filter(|k| !cov.nf.contains(*k) && truth.people_in(k) > 1).count();
    pydict![("face_frames", total), ("not_found_by_scan", missed), ("uncovered_frames", uncovered),
            ("uncovered_people", who.len()), ("people_split", of.values().map(|v| v.len() - 1).sum::<usize>()),
            ("people_mixed", person.values().filter(|v| v.len() > 1).count()), ("tracks_mixed", tracks_mixed),
            ("people", person.len()), ("truth_people", of.len())]
}

/// A fingerprint of the code a run measures (the rules, the facts, the evidence, this bench): FNV-1a over the files.
fn fingerprint(native: &Path) -> String {
    fn rs_files(d: &Path, out: &mut Vec<PathBuf>) {
        let mut es: Vec<PathBuf> = std::fs::read_dir(d).map(|r| r.filter_map(|e| e.ok().map(|e| e.path())).collect()).unwrap_or_default();
        es.sort();
        for p in es {
            if p.is_dir() {
                rs_files(&p, out);
            } else if p.extension().is_some_and(|e| e == "rs") {
                out.push(p);
            }
        }
    }
    let mut files = vec![];
    rs_files(&native.join("crates/mm-scan/src"), &mut files);
    for f in ["crates/mm-core/src/facts.rs", "crates/mm-engine/src/evidence.rs", "crates/mm-engine/src/scan.rs", "crates/mm-bench/src/main.rs"] {
        files.push(native.join(f));
    }
    let mut h: u64 = 0xcbf29ce484222325;
    for f in files {
        for b in std::fs::read(&f).unwrap_or_default() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    format!("{h:016x}")[..10].to_string()
}

/// Jazz against its census: the record (decision_bench's fields), the per-rule table, the wrong answers, the log.
fn jazz(cfg: &Cfg, proj: &Path, secs: f64, note: &str) -> Result<Py> {
    let truth = Truth::new(&cfg.places.truth(), proj)?;
    let hidden = strs(truth.pt.get("hidden"));
    let ds = decisions(proj)?;
    let r = replay(&ds, &hidden, &|d| truth.answer(d))?;
    let cov = Covers::new(&truth, &r.facts, &hidden);
    let (rules, wrong) = rule_errors(&r.last, &truth, &cov);
    let mut harms = Counter::default();
    for w in &wrong {
        harms.add(w.get("harm").as_str().unwrap_or(""), 1);
    }
    let asked = r.asked.total();
    let mut rec = pydict![("at", now()), ("clip", "jazz"), ("oracle", "gt"), ("truth", "census"),
                          ("code", fingerprint(&cfg.places.native)), ("depth", DEPTH), ("note", note), ("scan_s", secs),
                          ("decisions", ds.len()), ("asked", asked), ("asked_by_kind", r.asked.py()),
                          ("rule_wrong", harms.py()),
                          ("uncovered_by_rules", wrong.iter().map(|w| w.get("frames").as_i64().unwrap_or(0)).sum::<i64>()),
                          ("result", result(&truth, &cov))];
    if asked as usize > REPLAY_CAP {
        rec.as_dict_mut().unwrap().insert("truncated", Py::Bool(true));
    }
    let rules_py = Py::Dict(rules.iter().map(|(k, v)| (k.clone(), v.py())).collect());
    let last = pydict![("record", rec.clone()), ("rules", rules_py), ("wrong", Py::List(wrong)), ("asked", Py::List(r.log))];
    std::fs::write(proj.join("bench_last.json"), last.dumps_indent(1))?;
    Ok(rec)
}

/// decision_bench's one line per run: when, settings, questions, harms, coverage, note.
fn summary(r: &Py) -> String {
    let res = r.get("result");
    let w = r.get("rule_wrong");
    let k = r.get("asked_by_kind");
    let wi = |n: &str| w.get(n).as_i64().unwrap_or(0);
    let ri = |n: &str| res.get(n).as_i64().unwrap_or(0);
    let kinds: Vec<String> = ["track", "blip", "overlap", "same", "whose"].iter()
        .filter(|n| k.get(n).truthy()).map(|n| format!("{n} {}", k.get(n).py_str())).collect();
    let truth = r.get("truth").as_str().unwrap_or("old grade");
    let depth = if r.get("depth").as_str() != Some("off") { "on " } else { "off" };
    let note = r.get("note").as_str().unwrap_or("");
    format!("{}  {:<8} {:<9} depth {depth}  asked {:3} ({})  needless covers {:2}  faces uncovered by rules {}  uncovered {}/{}  \
             never boxed {}  people split {} mixed {} (tracks {})",
            r.get("at").py_str(), r.get("clip").py_str(), truth, r.get("asked").as_i64().unwrap_or(0), kinds.join(" "),
            wi("needless cover") + wi("cover, unlabelled"), wi("face uncovered"), ri("uncovered_frames"), ri("face_frames"),
            ri("not_found_by_scan"), ri("people_split"), ri("people_mixed"), ri("tracks_mixed"))
        + &if note.is_empty() { String::new() } else { format!("  | {note}") }
}

/// No census: every face question "a face" (overlaps "someone else"), every same-person question "different people".
fn worst_case(proj: &Path, secs: f64, clip: &str) -> Result<Py> {
    let pt = load(&proj.join("proto_tracks.json"))?;
    let hidden = strs(pt.get("hidden"));
    let ds = decisions(proj)?;
    let pick = |kind: &str| match kind {
        "overlap" => FX::A_SOMEONE_ELSE,
        "same" => FX::A_DIFFERENT,
        _ => FX::A_FACE,
    };
    let r = replay(&ds, &hidden, &|d| pick(d.get("kind").as_str().unwrap_or("")))?;
    let mut by = Counter::default();
    for d in r.last.iter().filter(|d| d.get("kind").as_str() == Some("same")) {
        if let Some(b) = d.get("by").as_str().filter(|b| !b.is_empty() && *b != "user") {
            let k: String = b.split(" (").next().unwrap_or("").chars().take(60).collect();
            by.add(&k, 1);
        }
    }
    let kinds = ["track", "blip", "overlap", "same"];
    let records: Dict = kinds.iter().map(|k| (k.to_string(), Py::from(ds.iter().filter(|d| d.get("kind").as_str() == Some(k)).count()))).collect();
    let strong = ds.iter().filter(|d| d.get("kind").as_str() == Some("same")
        && d.get("evidence").get("score").as_f64().unwrap_or(0.0) >= FX::ASK_SCORE).count();
    Ok(pydict![("clip", clip), ("decide_s", secs), ("records", records), ("same_records_score_0.8+", strong),
               ("asked", r.asked.total()), ("asked_by_kind", r.asked.py()), ("same_settled_by", by.py())])
}

fn append_line(path: &Path, line: &str) -> Result<()> {
    use std::io::Write;
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(f, "{line}")?;
    Ok(())
}

/// `--score PROJECT`: the answers given in a jazz project (facts.json, by a person or an agent) against the census.
fn score(census: &Path, proj: &Path) -> Result<Py> {
    let truth = Truth::new(census, proj)?;
    let hidden = strs(truth.pt.get("hidden"));
    let ds = decisions(proj)?;
    let doc = if proj.join(FX::FILE).exists() { load(&proj.join(FX::FILE))? } else { pydict![("facts", Py::List(vec![]))] };
    let facts = FX::facts_of(&doc).to_vec();
    let out = FX::answers(&ds, &facts, &hidden);
    let (mut right, mut asked) = (Counter::default(), Counter::default());
    let mut wrong = vec![];
    for d in out.iter().filter(|d| d.get("by").as_str() == Some("user")) {
        let kind = d.get("kind").as_str().unwrap_or("");
        let (given, census) = (d.get("answer").py_str(), truth.answer(d));
        asked.add(kind, 1);
        if given == census {
            right.add(kind, 1);
        } else {
            wrong.push(pydict![("id", d.get("id").clone()), ("given", given), ("census", census)]);
        }
    }
    let open = out.iter().filter(|d| !d.get("by").truthy()).count();
    let cov = Covers::new(&truth, &facts, &hidden);
    let best = replay(&ds, &hidden, &|d| truth.answer(d))?;
    let best_cov = Covers::new(&truth, &best.facts, &hidden);
    Ok(pydict![("project", proj.display().to_string()), ("answered", asked.total()), ("right", right.total()),
               ("answered_by_kind", asked.py()), ("right_by_kind", right.py()), ("still_open", open),
               ("wrong", Py::List(wrong)), ("result", result(&truth, &cov)),
               ("census_result", result(&truth, &best_cov)), ("census_asked", best.asked.total())])
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().position(|a| a == name).map(|i| args.get(i + 1).cloned().unwrap_or_default());
    let places = Places::find()?;
    if let Some(i) = args.iter().position(|a| a == "--history") {
        let clip = args.get(i + 1).filter(|a| !a.starts_with("--"));
        let text = std::fs::read_to_string(places.history()).unwrap_or_default();
        for l in text.lines().filter(|l| !l.trim().is_empty()) {
            let r = Py::parse(l).map_err(|e| anyhow!(e))?;
            if clip.map_or(true, |c| r.get("clip").as_str() == Some(c)) {
                println!("{}", summary(&r));
            }
        }
        return Ok(());
    }
    if let Some(p) = flag("--score") {
        println!("{}", score(&places.truth(), Path::new(&p))?.dumps_indent(1));
        return Ok(());
    }
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("rule-bench --note TEXT [--clips jazz,party360,party4k] [--no-build]\nrule-bench --history [CLIP]\nrule-bench --score PROJECT");
        return Ok(());
    }
    let note = flag("--note").unwrap_or_default();
    let clips = flag("--clips").unwrap_or_else(|| "jazz,party360,party4k".into());
    if !args.iter().any(|a| a == "--no-build") {
        // today's decide step (this binary is today's facts already: cargo run builds it)
        let ok = Command::new("cargo").args(["build", "-q", "--release", "-p", "mm-engine"]).current_dir(&places.native).status()?;
        if !ok.success() {
            bail!("cargo build -p mm-engine failed");
        }
    }
    let cfg = Cfg { places };
    std::fs::create_dir_all(&cfg.places.out)?;
    let mut lines = vec![];
    for clip in clips.split(',').map(str::trim).filter(|c| !c.is_empty()) {
        cfg.places.clip(clip)?;
        let (proj, secs, stages) = cfg.decide(clip)?;
        let row = if clip == "jazz" {
            let mut rec = jazz(&cfg, &proj, secs, &note)?;
            rec.as_dict_mut().unwrap().insert("stage_s", stages.clone());
            append_line(&cfg.places.history(), &rec.dumps())?;
            let line = summary(&rec);
            println!("{line}");
            pydict![("clip", clip), ("line", line), ("record", rec)]
        } else {
            let mut row = worst_case(&proj, secs, clip)?;
            row.as_dict_mut().unwrap().insert("stage_s", stages.clone());
            println!("{}", row.dumps());
            row
        };
        let mut l = pydict![("at", now()), ("note", note.as_str()),
                            ("facts", "native"), ("motion", "native")];
        for (k, v) in row.as_dict().unwrap().iter() {
            l.as_dict_mut().unwrap().insert(k, v.clone());
        }
        lines.push(l);
    }
    for l in lines {
        append_line(&cfg.places.out.join("runs.jsonl"), &l.dumps())?;
    }
    Ok(())
}
