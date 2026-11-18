//! `mm-cli decisions`: answer a project's Decisions from a program (an agent, a script) instead of clicking. Port of
//! `decisions_cli.py`, with the same commands and JSON:
//!
//! ```text
//! mm-cli decisions projects
//! mm-cli decisions questions <project> [--all] [--kind track|blip|overlap|same|whose] [--limit N]
//!                            [--strips DIR [--allow-faces]]
//! mm-cli decisions answer <project> <question id> "<choice>"
//! mm-cli decisions clear <project> <question id>
//! ```
//!
//! Everything prints JSON. <project> is a project's folder name, or the start of one if only one matches, or a path
//! to a project folder.
//!
//! Unlike decisions_cli.py it needs no running app: it works on the project folder itself, through the same model
//! the app uses (mm_core::edits): answers are saved to the project's facts log (tagged via: "agent"), counted as your
//! answers everywhere, and shown on the Decisions page with an "answered by an agent" note. While a project server has
//! the project open (it holds PROJECT/.server.lock) the folder is that server's: answer through it instead
//! (decisions_cli.py), or close it.
//!
//! Questions give the numbers the scan knows (frames, sizes, scores, box positions). With `--strips DIR` each also gets
//! a picture, DIR/<id>.png, for a program that can see: the question's crops and its whole frame, boxes outlined
//! (strips.rs). Those show people's faces to whatever reads them, so they are made only for a public-domain clip
//! (jazz), or with `--allow-faces`: the footage's owner chose to send them (AGENTS.md, Privacy).
//!
//! Questions come in the Decisions page's order: is it a face first (tracks, overlaps, blips), then who's who.
//! Answering one can settle or open others (the same-person questions are re-picked), so read the list again after
//! answering.
//!
//! Not yet: "whose face is this?" questions for boxes you made (they need the vision port, P5).
//!
//! Two more, for the parity tests (native/tools/parity/model_*.py), not for people:
//!   session <project folder> <steps.jsonl> [--follow recording.json]   apply page requests in one process
//!   replay-facts <project folder> [facts.json]                          derive + answers after each fact
//!   answers-serve                                                       facts.answers over stdin (benches): first
//!                                                                       line {"decisions", "hidden"}, then one
//!                                                                       facts list per line -> one answers list
use anyhow::{anyhow, Result};
use mm_core::edits::{Follow, Project, Refused};
use mm_core::facts as FX;
use mm_core::pyjson::{self, Dict, Py};
use mm_core::{projects, pydict};
use std::path::{Path, PathBuf};

const RANK: [(&str, i64); 5] = [("whose", 0), ("track", 1), ("overlap", 1), ("blip", 1), ("same", 2)];

fn choices(kind: &str) -> Option<Vec<&'static str>> {
    Some(match kind {
        "track" | "blip" => vec![FX::A_FACE, FX::A_NOT_FACE],
        "overlap" => vec![FX::A_SAME_FACE, FX::A_NOT_FACE, FX::A_SOMEONE_ELSE],
        "same" => vec![FX::A_SAME, FX::A_DIFFERENT],
        _ => return None,
    })
}

fn meaning(kind: &str) -> Option<&'static str> {
    Some(match kind {
        "track" => "Is the green box (piece) a face? A face from any side counts.",
        "blip" => "Is the green box (piece), seen only briefly, a face?",
        "overlap" => "The green box (piece) overlaps the grey box (person, a covered face): part of that face, not a face \
                      at all (a hand, clothes, background), or someone else's face?",
        "same" => "Are the green box (piece) and the grey box (person) the same person? Any part of the head counts.",
        "whose" => "A box the user drew: whose face is it? Answer with a person's piece id, or \"someone new\".",
        _ => return None,
    })
}

struct Fail(String);

fn fail<T>(msg: impl Into<String>) -> std::result::Result<T, Fail> {
    Err(Fail(msg.into()))
}

/// Entry point: `args` are what follows `decisions`. Prints JSON; exits 1 with {"error"} on failure.
pub fn main(args: &[String]) -> ! {
    match run(args) {
        Ok(()) => std::process::exit(0),
        Err(Fail(msg)) => {
            println!("{}", pydict![("error", msg)].dumps());
            std::process::exit(1)
        }
    }
}

fn usage() -> Fail {
    Fail("usage: mm-cli decisions (projects | questions <project> [--all] [--kind K] [--limit N] [--strips DIR [--allow-faces]] | answer <project> <id> <choice> | clear <project> <id>)".into())
}

fn checkout_root() -> Option<PathBuf> {
    let mut d = std::env::current_exe().ok()?;
    loop {
        if d.join("native/ARCHITECTURE.md").exists() {
            return Some(d);
        }
        if !d.pop() {
            break;
        }
    }
    let mut d = std::env::current_dir().ok()?;
    loop {
        if d.join("native/ARCHITECTURE.md").exists() {
            return Some(d);
        }
        if !d.pop() {
            return None;
        }
    }
}

fn roots() -> Vec<PathBuf> {
    projects::roots(true, false, checkout_root().as_deref())
}

/// (folder, name) of a project named on the command line.
fn find(project: &str) -> std::result::Result<(PathBuf, String), Fail> {
    let p = Path::new(project);
    if project.contains('/') && p.join("meta.json").exists() {
        let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| project.into());
        return Ok((p.to_path_buf(), name));
    }
    projects::find(&roots(), project).map(|(r, n)| (r.join(&n), n)).map_err(Fail)
}

/// What a program needs to answer: no pictures.
pub fn slim(d: &Py) -> Py {
    let kind = d.get("kind").as_str().unwrap_or("");
    let by = d.get("by");
    let answered_by = if !by.truthy() {
        Py::Null
    } else if by.as_str() == Some("user") {
        Py::from(if d.get("via").as_str() == Some("agent") { "agent" } else { "user" })
    } else {
        Py::from("rule")
    };
    let mut out = pydict![
        ("id", d.get("id").clone()),
        ("kind", d.get("kind").clone()),
        ("meaning", meaning(kind).map(Py::from).unwrap_or_else(|| d.get("question").clone())),
        ("piece", d.get("about").get("piece").clone()),
        ("person", d.get("about").get("person").clone()),
        ("choices", choices(kind).map(|c| Py::List(c.into_iter().map(Py::from).collect())).unwrap_or_else(|| d.get("options").clone())),
        ("why", d.get("why").clone()),
        ("evidence", d.get("evidence").clone()),
        ("boxes", d.get("show").clone()), // [{t: frame, piece: [x0, y0, x1, y1], person: [...]}] in video pixels
        ("answer", d.get("answer").clone()),
        ("answered_by", answered_by),
    ];
    if d.get("rule_answer").truthy() && d.get("rule_by").truthy() {
        out.as_dict_mut().unwrap().insert("rule", pydict![("answer", d.get("rule_answer").clone()), ("why", d.get("rule_by").clone())]);
    }
    out
}

/// decisions.html's order: open first; faces before who's who; the weightiest, then the longest, first.
pub fn ordered(mut ds: Vec<Py>) -> Vec<Py> {
    let key = |d: &Py| {
        let rank = RANK.iter().find(|(k, _)| Some(*k) == d.get("kind").as_str()).map(|(_, r)| *r).unwrap_or(3);
        let w = d.get("weight").or(&Py::Int(0)).as_f64().unwrap_or(0.0);
        let fr = d.get("evidence").get("frames").or(&Py::Int(0)).as_f64().unwrap_or(0.0);
        (d.get("by").truthy(), rank, -w, -fr)
    };
    ds.sort_by(|a, b| key(a).partial_cmp(&key(b)).unwrap_or(std::cmp::Ordering::Equal));
    ds
}

/// The scan's decision records as saved (decisions_<kind>.json, in name order); None when there are none.
fn records(proj: &Path) -> Result<Option<Vec<Py>>, String> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(proj).map_err(|e| e.to_string())?.flatten().map(|e| e.path())
        .filter(|p| p.file_name().map(|n| n.to_string_lossy()).is_some_and(|n| n.starts_with("decisions_") && n.ends_with(".json")))
        .collect();
    files.sort();
    if files.is_empty() {
        return Ok(None);
    }
    let mut ds = vec![];
    for f in files {
        ds.extend(pyjson::load(&f)?.get("decisions").as_list().cloned().unwrap_or_default());
    }
    Ok(Some(ds))
}

/// The scan's records with their answers, read without opening the project for writing (a server may hold it).
fn read_decisions(proj: &Path) -> Result<Option<Vec<Py>>, String> {
    let Some(ds) = records(proj)? else { return Ok(None) };
    let facts = if proj.join(FX::FILE).exists() { pyjson::load(&proj.join(FX::FILE))? } else { pydict![("facts", Py::List(vec![]))] };
    let hidden = hidden_of(proj)?;
    Ok(Some(FX::answers(&ds, FX::facts_of(&facts), &hidden)))
}

fn hidden_of(proj: &Path) -> Result<Vec<String>, String> {
    let proto = pyjson::load(&proj.join("proto_tracks.json"))?;
    Ok(match proto.get("hidden") {
        Py::Null => {
            let e = if proj.join("edits.json").exists() { pyjson::load(&proj.join("edits.json"))? } else { Py::Null };
            let mut h: Vec<String> = e.get("not_face").as_dict().map(|d| d.iter().filter(|(_, v)| v.as_str() == Some("decision"))
                .map(|(k, _)| k.to_string()).collect()).unwrap_or_default();
            h.sort();
            h
        }
        l => l.as_list().map(|l| l.iter().map(Py::py_str).collect()).unwrap_or_default(),
    })
}

fn run(args: &[String]) -> std::result::Result<(), Fail> {
    let cmd = args.first().ok_or_else(usage)?.as_str();
    let rest = &args[1..];
    match cmd {
        "projects" => {
            let mut out = vec![];
            for (r, n) in projects::all_projects(&roots()) {
                let p = r.join(&n);
                let meta = projects::meta(&p).map_err(Fail)?;
                out.push(pydict![
                    ("project", n.as_str()),
                    ("name", projects::display_name(&meta, &n)),
                    ("open_in_app", projects::lock_holder(&p).is_some_and(|h| h.get("port").truthy())),
                    ("folder", p.to_string_lossy().into_owned()),
                ]);
            }
            println!("{}", Py::List(out).dumps_indent(1));
        }
        "questions" => {
            let (mut all, mut kind, mut limit, mut pos) = (false, None::<String>, None::<i64>, vec![]);
            let (mut strips_dir, mut allow_faces) = (None::<PathBuf>, false);
            let mut i = 0;
            while i < rest.len() {
                match rest[i].as_str() {
                    "--all" => all = true,
                    "--kind" => {
                        i += 1;
                        let k = rest.get(i).ok_or_else(usage)?;
                        if !RANK.iter().any(|(r, _)| r == k) {
                            return fail(format!("--kind: one of {}", ["blip", "overlap", "same", "track", "whose"].join(", ")));
                        }
                        kind = Some(k.clone());
                    }
                    "--strips" => {
                        i += 1;
                        strips_dir = Some(PathBuf::from(rest.get(i).ok_or_else(usage)?));
                    }
                    "--allow-faces" => allow_faces = true,
                    "--limit" => {
                        i += 1;
                        limit = Some(rest.get(i).and_then(|s| s.parse().ok()).ok_or_else(usage)?);
                    }
                    x => pos.push(x.to_string()),
                }
                i += 1;
            }
            let [project] = pos.as_slice() else { return Err(usage()) };
            let (proj, name) = find(project)?;
            let Some(ds) = read_decisions(&proj).map_err(Fail)? else {
                return fail("this project has no decisions yet (still scanning, or scanned before decisions existed)");
            };
            let ds = ordered(ds);
            let n_open = ds.iter().filter(|d| !d.get("by").truthy()).count();
            let mut ds: Vec<Py> = ds.into_iter()
                .filter(|d| (all || !d.get("by").truthy()) && kind.as_deref().is_none_or(|k| d.get("kind").as_str() == Some(k)))
                .collect();
            if let Some(l) = limit {
                let n = ds.len() as i64;
                ds.truncate((if l < 0 { (n + l).max(0) } else { l.min(n) }) as usize);
            }
            let mut qs: Vec<Py> = ds.iter().map(slim).collect();
            if let Some(dir) = &strips_dir {
                let meta = projects::meta(&proj).map_err(Fail)?;
                let video = meta.get("video").as_str().unwrap_or("").to_string();
                let public = checkout_root().is_some_and(|r| mm_core::privacy::is_public_clip(&r, Path::new(&video)));
                if !public && !allow_faces {
                    return fail("--strips: this clip isn't public domain, so its strips show real people's faces to whatever \
                                 reads them; pass --allow-faces only if the footage's owner chose to send them");
                }
                let w = meta.get("width").as_f64().unwrap_or(0.0);
                for q in qs.iter_mut() {
                    let st = crate::strips::write(&proj, w, q, dir).unwrap_or_else(|e| pydict![("error", e)]);
                    q.as_dict_mut().unwrap().insert("strip", st);
                }
            }
            println!("{}", pydict![("project", name.as_str()), ("open", n_open), ("questions", qs)].dumps_indent(1));
        }
        "answer" | "clear" => {
            let want = if cmd == "answer" { 3 } else { 2 };
            if rest.len() != want {
                return Err(usage());
            }
            let (proj, _) = find(&rest[0])?;
            let id = &rest[1];
            let choice = if cmd == "answer" {
                let c = rest[2].clone();
                let kind = id.split(':').next().unwrap_or("");
                if let Some(cs) = choices(kind) {
                    if !cs.contains(&c.as_str()) {
                        return fail(format!("{kind} questions take one of: {}", cs.join(", ")));
                    }
                }
                Py::Str(c)
            } else {
                Py::Null
            };
            let _lock = match projects::take_lock(&proj, None) {
                Ok(l) => l,
                Err(_) => return fail("the app has this project open: answer through it (decisions_cli.py), or close it and try again"),
            };
            let mut p = Project::open(&proj, mm_core::edits::system_clock()).map_err(Fail)?;
            let r = p.decide(&pydict![("id", id.as_str()), ("answer", choice), ("via", "agent")]).map_err(|e| Fail(e.error))?;
            println!("{}", slim(&r).dumps_indent(1));
        }
        "session" => session(rest).map_err(|e| Fail(format!("{e:#}")))?,
        "replay-facts" => replay_facts(rest).map_err(|e| Fail(format!("{e:#}")))?,
        "answers-serve" => answers_serve().map_err(|e| Fail(format!("{e:#}")))?,
        _ => return Err(usage()),
    }
    Ok(())
}

/// FNV-1a 64: a cheap fingerprint the parity scripts compute the same way.
pub fn fnv(b: &[u8]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for x in b {
        h ^= *x as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

/// A recording of review_server's flow_step / face_evidence calls, replayed in order (parity only).
struct Recorded {
    calls: Vec<Py>,
    i: usize,
    mismatches: Vec<String>,
}

fn arr(v: &Py) -> Option<[f64; 3]> {
    let l = v.as_list()?;
    Some([l.first()?.as_f64()?, l.get(1)?.as_f64()?, l.get(2)?.as_f64()?])
}

fn close(a: [f64; 3], b: &Py) -> bool {
    arr(b).is_some_and(|b| (0..3).all(|i| (a[i] - b[i]).abs() <= 1e-9 * (1.0 + a[i].abs())))
}

impl Recorded {
    fn next(&mut self, what: &str, check: impl Fn(&Py) -> bool) -> Option<[f64; 3]> {
        let Some(c) = self.calls.get(self.i).cloned() else {
            self.mismatches.push(format!("call {}: {what} past the end of the recording", self.i));
            return None;
        };
        self.i += 1;
        if c.get("fn").as_str() != Some(what) || !check(&c) {
            self.mismatches.push(format!("call {}: {what}, recorded {}", self.i - 1, c.dumps()));
        }
        arr(c.get("out"))
    }
}

impl Follow for Recorded {
    fn flow_step(&mut self, from: i64, to: i64, c: [f64; 3]) -> Option<[f64; 3]> {
        self.next("flow", |r| r.get("from").as_i64() == Some(from) && r.get("to").as_i64() == Some(to) && close(c, r.get("c")))
    }
    fn face_evidence(&mut self, u: i64, c: [f64; 3], prev: [f64; 3], step: i64) -> Option<[f64; 3]> {
        self.next("face", |r| r.get("u").as_i64() == Some(u) && r.get("step").as_i64() == Some(step) && close(c, r.get("c")) && close(prev, r.get("prev")))
    }
}

fn file_hash(p: &Path) -> String {
    std::fs::read(p).map(|b| fnv(&b)).unwrap_or_else(|_| "missing".into())
}

fn refused(e: Refused) -> Py {
    pydict![("error", e.error), ("code", e.code as i64)]
}

/// Apply a script of page requests to a project in one process (the undo history lives in the process, as in a
/// project server). Each line: {"at": ms, "do": ops|facts|decide|undo|redo|split|undo_correction|propagate|refollow,
/// "body": {...the request body}}. Prints one JSON line per step: its result and fingerprints of facts.json,
/// edits.json and the decision list.
fn session(args: &[String]) -> Result<()> {
    let proj = PathBuf::from(args.first().ok_or_else(|| anyhow!("session: which project folder?"))?);
    let script = std::fs::read_to_string(args.get(1).ok_or_else(|| anyhow!("session: which steps file?"))?)?;
    let mut follow = Recorded { calls: vec![], i: 0, mismatches: vec![] };
    if let Some(i) = args.iter().position(|a| a == "--follow") {
        let rec = pyjson::load(Path::new(args.get(i + 1).ok_or_else(|| anyhow!("--follow FILE"))?)).map_err(|e| anyhow!(e))?;
        follow.calls = rec.as_list().cloned().unwrap_or_default();
    }
    let now = std::sync::Arc::new(std::sync::atomic::AtomicI64::new(0));
    let n2 = now.clone();
    let mut p = Project::open(&proj, Box::new(move || n2.load(std::sync::atomic::Ordering::SeqCst))).map_err(|e| anyhow!(e))?;
    let fingerprints = |p: &Project| -> Result<Dict> {
        let ds = p.decisions().map_err(|e| anyhow!(e))?.unwrap_or_default();
        Ok([("facts", file_hash(&p.work.join("facts.json"))), ("edits", file_hash(&p.work.join("edits.json"))),
            ("decisions", fnv(Py::List(ds).dumps().as_bytes()))].into_iter().map(|(k, v)| (k.to_string(), Py::Str(v))).collect())
    };
    println!("{}", pydict![("i", -1), ("result", Py::Null), ("hash", fingerprints(&p)?)].dumps());
    for (i, line) in script.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let step = Py::parse(line).map_err(|e| anyhow!("line {}: {e}", i + 1))?;
        if let Some(at) = step.get("at").as_i64() {
            now.store(at, std::sync::atomic::Ordering::SeqCst);
        }
        let body = step.get("body").clone();
        let ok = || pydict![("ok", true)];
        let num = |k: &str| body.get(k).as_i64().unwrap_or(-1);
        let result = match step.get("do").as_str().unwrap_or("") {
            "ops" => p.post_ops(&body).map(|_| ok()).unwrap_or_else(refused),
            "facts" => p.post_facts(&body).map(|_| ok()).unwrap_or_else(refused),
            "decide" => p.decide(&body).unwrap_or_else(refused),
            "undo" => pydict![("undone", p.undo_last(false).map(|l| l.map(Py::Str).unwrap_or(Py::Null)).unwrap_or_else(refused))],
            "redo" => pydict![("redone", p.undo_last(true).map(|l| l.map(Py::Str).unwrap_or(Py::Null)).unwrap_or_else(refused))],
            "split" => p.split_cover(&body.get("piece").py_str(), num("t")).map(|n| pydict![("new", n)]).unwrap_or_else(refused),
            "undo_correction" => p.undo_correction(&body.get("piece").py_str(), num("t"), &mut follow).map(|_| ok()).unwrap_or_else(refused),
            "refollow" => p.refollow_covers(&mut follow).map(|_| ok()).unwrap_or_else(refused),
            "propagate" => match arr(body.get("c")) {
                Some(c) => p.propagate(num("t"), c, &mut follow).unwrap_or_else(refused),
                None => refused(Refused { code: 400, error: "c: [cx, cy, s]".into() }),
            },
            x => return Err(anyhow!("line {}: unknown step {x:?}", i + 1)),
        };
        println!("{}", pydict![("i", i), ("result", result), ("hash", fingerprints(&p)?)].dumps());
    }
    println!("{}", pydict![("end", true), ("follow_calls", follow.i), ("follow_recorded", follow.calls.len()),
                           ("follow_mismatches", follow.mismatches.iter().take(20).map(Py::from).collect::<Vec<_>>())].dumps());
    Ok(())
}

/// For a facts log (the project's, or FILE): after each fact in log order, the derived edits fields and the
/// decision list's fingerprint (one JSON line each).
/// facts.answers over a pipe, for a program that plays the Decisions page through: the decision records
/// and hidden boxes once, then a facts list per line, each answered with the records' answers as one line of JSON.
fn answers_serve() -> Result<()> {
    use std::io::{BufRead, Write};
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let head = Py::parse(&lines.next().ok_or_else(|| anyhow!("answers-serve: no records line"))??).map_err(|e| anyhow!(e))?;
    let recs = head.get("decisions").as_list().cloned().unwrap_or_default();
    let hidden: Vec<String> = head.get("hidden").as_list().map(|l| l.iter().map(Py::py_str).collect()).unwrap_or_default();
    let mut out = std::io::stdout().lock();
    for line in lines {
        let facts = Py::parse(&line?).map_err(|e| anyhow!(e))?;
        let fs = facts.as_list().cloned().unwrap_or_default();
        writeln!(out, "{}", Py::List(FX::answers(&recs, &fs, &hidden)).dumps())?;
        out.flush()?;
    }
    Ok(())
}

fn replay_facts(args: &[String]) -> Result<()> {
    let proj = PathBuf::from(args.first().ok_or_else(|| anyhow!("replay-facts: which project folder?"))?);
    let file = args.get(1).map(PathBuf::from).unwrap_or_else(|| proj.join(FX::FILE));
    let facts = pyjson::load(&file).map_err(|e| anyhow!(e))?;
    let facts = FX::facts_of(&facts).to_vec();
    let hidden = hidden_of(&proj).map_err(|e| anyhow!(e))?;
    let edits = pyjson::load(&proj.join("edits.json")).unwrap_or(Py::Null);
    let p = records(&proj).map_err(|e| anyhow!(e))?.unwrap_or_default();
    for k in 0..=facts.len() {
        let fs = &facts[..k];
        let derived: Dict = FX::derive(fs, &hidden, edits.get("palette")).into_iter().collect();
        let ans = FX::answers(&p, fs, &hidden);
        println!("{}", pydict![("k", k), ("derive", derived), ("answers", fnv(Py::List(ans).dumps().as_bytes()))].dumps());
    }
    Ok(())
}
