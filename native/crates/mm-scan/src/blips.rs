//! Blips: pieces seen in only 1-2 frames (<= MAX_DET detections). In the app each one is a mask that pops for a frame
//! or two. Never asked in a video. (Port of `mojimask/scan/blips.py`.) The rule, in the order the code applies it:
//!   1. big, no face   the box is BIG_PX or more and the face check never reaches FACE_FAILS: dropped (in a photo, asked
//!                     unless step 3 applies)
//!   2. sure           detector >= SURE_SCORE, face check >= SURE_FACE, or two kinds of evidence (detector >= MIN_SCORE,
//!                     face check >= FACE_OK, a body pose estimate with its head on the box: pose_util.pose_for_head).
//!                     In a video, then followed (FOLLOW_CHECK): runs into a tracked person ("met") or finds no face
//!                     ("nothing") -> dropped; lands on a face nobody tracks -> kept. In a photo, kept.
//!   3. weak           no kind of evidence at all: dropped
//!   4. borderline     one weak kind of evidence: dropped in a video, asked in a photo (PHOTO_ASK)
//! A kept blip is covered and held BLIP_HOLD frames either side (cover).
//!
//! Evidence (2026-11-12): borderline blips used to be asked, and almost never were faces: across every answered project
//! 152 "not a face" to 7 faces, with no evidence telling the two apart, and they were most of the questions left (jazz:
//! 32 of 88). So a face seen only as a borderline blip goes uncovered for that frame or two; the record stays (Rules)
//! and "a face" there covers it. Real people's detections score median 1.00 (face check median 0.69); blips score median
//! ~0.4 with face check 0, on nothing labelled or on a hand, neck, hair or arm beside a face. The skeleton matters: among
//! blips dropped without it, party 56, jazz 16 and twirl 5 had a skeleton head on them and no other cover, and by eye
//! many were blurred real faces or heads. Real people's own detections have a skeleton head 64-86% of the time; a "sits
//! on a torso" rule was rejected (16% of jazz's real faces sit on someone else's torso). The face-check numbers here
//! were measured with SCRFD's face check; the thresholds were kept unchanged for RetinaFace's after a four-clip bench
//! (2026-11-14), which passes more faces (safer).
//!
//! BIG_PX, FACE_FAILS: a blip whose box is BIG_PX or more and never comes near passing the face check (best <
//! FACE_FAILS) is dropped, whatever the detector or skeleton say. At that size the check is decisive. Jazz census
//! (2026-11-13), per detection: real faces >= 40 px pass 85-100%, non-faces 0% (under 30 px faces pass only 21%, so
//! small boxes are left alone). On jazz it drops 6 kept blips: a dancer's back (82 px), shirt fronts, hands, a back of
//! a head; no face.
//!
//! The follow check (FOLLOW_CHECK), for a blip the rule above would keep. Kept blips were kept for the passing face: a
//! person crossing the picture whom no track follows. So test that directly. Follow it FOLLOW_FRAMES either way: if it
//! runs into a tracked person's box, it's a stray piece of someone covered anyway; if it never lands on a face (face
//! check), it's junk; only "a face nobody tracks" keeps it. Jazz census (2026-11-13), all 129 blips followed: 0 lead to
//! an untracked face; the 9 face blips are 8 strays of tracked people + 1 lost at the frame edge (already dropped);
//! of 116 not-faces, 26 run into a person and 90 find nothing. On the kept ones, needless covers 14 -> 4, nothing new
//! uncovered. Real people's smallest heads (27-30 px) are still detected 9-50 times in proper tracks.
//!
//! A photo (a clip of one frame, video_io) sees every face once, so everything in it is a "blip", and the video priors
//! don't hold: there a face is a long track and a one-frame box is almost never one; here every face is a one-frame box.
//! And nothing moves, so there's nothing to follow. In a photo the clear boxes are kept without the follow check, those
//! nothing speaks for are dropped as in a video, and the borderline ones are asked (PHOTO_ASK) rather than dropped: a
//! face missed in a photo is in the whole result, and a photo has few questions. Measured on jazz's census frames as
//! photos (photo_bench.py, 2026-11-14).
//!
//! Every blip is a decision record (overlaps' shape, kind "blip") in PROJECT/decisions_blip.json; the app's decisions
//! page shows them. The rule's answer is "dropped: too weak" or "kept: covered" (it can't tell a face, only how strong
//! the evidence is); your answer ("a face" / "not a face") is a fact (facts.py) and overrides it.
//!
//! Longer tracks (find_tracks): kept by rule when the face evidence is strong (a clear face in CLEAR_FRAMES frames or
//! more, or a median detector score of SURE_SCORE), dropped by rule when they never show a clear face (NEVER_CLEAR);
//! otherwise asked, "Is the green box a face?", and covered until answered. (Party clip, 2026-11-12: a lamp counted as
//! a person -- 17 detections, median detector 0.49, the face check passed once -- with no record to explain it or
//! answer it.) NEVER_CLEAR: a track that never shows one clear face (face check >= FACE_OK) and whose detector median
//! is under SURE_SCORE is dropped by rule, like a weak blip (overridable: "a face" covers it). Jazz census
//! (2026-11-13): all 27 such tracks were hands, a sousaphone bell, shirt fronts or a violin; real faces of the same
//! size (24-50 px) show a clear face in dozens of frames, and the 2 faces with none are kept by their detector median.
//! The party clip's lamp passed the face check once, so it is still asked.
use crate::evidence::Evidence;
use crate::faces;
use crate::follow::{self, Clip};
use crate::geom::{covered, Cs};
use crate::graph::{self, State};
use crate::np;
use crate::overlaps::{self, fl1};
use crate::pose::{self, Pose};
use crate::pyset::PySet;
use indexmap::IndexMap;
use mm_core::facts as fx;
use mm_core::pyjson::{py_round, py_round_int, Dict, Py};
use std::collections::BTreeMap;
use std::path::Path;

pub const MAX_DET: usize = 2;
pub const MIN_SCORE: f64 = 0.6;
pub const FACE_OK: f64 = graph::CLEAR_FACE;
pub const BIG_PX: f64 = 40.0;
pub const FACE_FAILS: f64 = 0.3;
pub const FOLLOW_CHECK: bool = true;
pub const FOLLOW_FRAMES: i64 = 12;
pub const FOLLOW_FACES: usize = 2;
pub const SURE_SCORE: f64 = overlaps::SURE_SCORE;
pub const SURE_FACE: f64 = 0.6;
pub const PHOTO_ASK: bool = true;
pub const FILE: &str = "decisions_blip.json";
pub const CLEAR_FRAMES: usize = overlaps::CLEAR_FRAMES;
pub const TRACK_FILE: &str = "decisions_track.json";
pub const NEVER_CLEAR: bool = true;
pub const NEVER_RULE: &str = "rule: never a clear face (none of its frames passes the face check; median detector under 0.8)";

fn options(extra: &[&str]) -> Py {
    Py::List([fx::A_FACE, fx::A_NOT_FACE].iter().chain(extra.iter()).map(|s| Py::from(*s)).collect())
}

fn group_scores(st: &State, ms: &PySet) -> IndexMap<i64, f64> {
    let mut score = IndexMap::new();
    for c in ms.iter() {
        for (t, s) in &st.info[&c].score {
            score.insert(*t, *s);
        }
    }
    score
}

fn face_scores(st: &State, ms: &PySet) -> Vec<f64> {
    let mut fs = vec![];
    for c in ms.iter() {
        for k in &st.chains[&c] {
            for t in st.tl[k].obs.keys() {
                fs.push(faces::get(&st.fs, *k, *t).map(|f| f.score).unwrap_or(0.0));
            }
        }
    }
    fs
}

fn pymax(v: &[f64]) -> f64 {
    let mut m = v[0];
    for x in &v[1..] {
        if *x > m {
            m = *x;
        }
    }
    m
}

/// blips.follows: "met" if following the blip runs into a tracked person's box, else "untracked face" if it lands on a
/// face in >= FOLLOW_FACES steps, else "nothing".
fn follows(clip: &Clip, ev: &mut Evidence, at: &BTreeMap<i64, Vec<(i64, Cs)>>, r: i64, obs: &IndexMap<i64, Cs>) -> &'static str {
    let meet = |u: i64, c: &Cs| -> Option<i64> {
        at.get(&u)?.iter().find(|(q, o)| *q != r && covered(c, o).max(covered(o, c)) >= 0.5).map(|(q, _)| *q)
    };
    let mut ts: Vec<i64> = obs.keys().copied().collect();
    ts.sort();
    let mut n = 0;
    for (direction, t0) in [(-1i64, ts[0]), (1, ts[ts.len() - 1])] {
        let p = follow::follow(clip, ev, t0, &obs[&t0], direction, FOLLOW_FRAMES, Some(&meet), true, None);
        if p.met.is_some() {
            return "met";
        }
        n += p.how.values().filter(|h| h.starts_with("face")).count();
    }
    if n >= FOLLOW_FACES {
        "untracked face"
    } else {
        "nothing"
    }
}

/// blips.find: the decision records for the groups (after attachment and overlaps).
pub fn find(st: &State, members: &IndexMap<i64, PySet>, poses: Option<&[Vec<Pose>]>, photo: bool, clip: &Clip, ev: &mut Evidence)
            -> Vec<Dict> {
    let mut at: BTreeMap<i64, Vec<(i64, Cs)>> = BTreeMap::new();
    if FOLLOW_CHECK && !photo {
        for (r, ms) in members {
            let obs = st.group_obs(ms);
            if obs.len() > MAX_DET {
                for (t, o) in &obs {
                    at.entry(*t).or_default().push((*r, *o));
                }
            }
        }
    }
    let mut decisions: Vec<Dict> = vec![];
    for (&r, ms) in members {
        let obs = st.group_obs(ms);
        let score = group_scores(st, ms);
        if obs.len() > MAX_DET {
            continue;
        }
        let best = pymax(&score.values().copied().collect::<Vec<_>>());
        let fsc = face_scores(st, ms);
        let face = if fsc.is_empty() { 0.0 } else { pymax(&fsc) };
        let head = match poses {
            Some(p) if !p.is_empty() => obs.iter().any(|(t, o)| (*t as usize) < p.len() && pose::pose_for_head(&p[*t as usize], o).is_some()),
            _ => false,
        };
        let kinds = (best >= MIN_SCORE) as i32 + (face >= FACE_OK) as i32 + head as i32;
        let weak = kinds == 0;
        let size = np::median(&obs.values().map(|o| o[2]).collect::<Vec<_>>());
        let seen_big = size >= BIG_PX && face < FACE_FAILS;
        let mut sure = !seen_big && (best >= SURE_SCORE || face >= SURE_FACE || kinds >= 2);
        let led = if sure && FOLLOW_CHECK && !photo { Some(follows(clip, ev, &at, r, &obs)) } else { None };
        if led.is_some() && led != Some("untracked face") {
            sure = false;
        }
        let ask = photo && PHOTO_ASK && !sure && !weak;
        let mut ts: Vec<i64> = obs.keys().copied().collect();
        ts.sort();
        let mut d = Dict::new();
        d.insert("id", Py::Str(format!("blip:{r}")));
        d.insert("kind", Py::from("blip"));
        let mut about = Dict::new();
        about.insert("piece", Py::Int(r));
        d.insert("about", Py::Dict(about));
        d.insert("question", Py::from("Is the green box a face?"));
        d.insert("options", options(&["dropped: too weak", "kept: covered"]));
        let mut e = Dict::new();
        e.insert("frames", Py::Int(ts.len() as i64));
        e.insert("first", Py::Int(ts[0]));
        e.insert("last", Py::Int(ts[ts.len() - 1]));
        e.insert("detector_score", Py::Float(py_round(best, 2)));
        e.insert("skeleton_head", Py::Bool(head));
        e.insert("face_check", Py::Float(py_round(face, 2)));
        e.insert("size", Py::Int(py_round_int(obs[&ts[0]][2])));
        d.insert("evidence", Py::Dict(e));
        d.insert("why", Py::Str(format!("seen in {} frame{}, detector {}, face check {}, {} skeleton head on it", ts.len(),
                                         if ts.len() > 1 { "s" } else { "" }, graph::f(best, 2), graph::f(face, 2),
                                         if head { "a" } else { "no" })));
        d.insert("show", Py::List(ts.iter().map(|t| {
            let mut s = Dict::new();
            s.insert("t", Py::Int(*t));
            s.insert("piece", fl1(&obs[t]));
            s.insert("person", Py::Null);
            Py::Dict(s)
        }).collect()));
        let answer = if sure { Py::from("kept: covered") } else if ask { Py::Null } else { Py::from("dropped: too weak") };
        let by = if ask {
            Py::Null
        } else if sure && led.is_some() {
            Py::from("rule: clear, and following it finds a face nobody else covers")
        } else if led == Some("met") {
            Py::from("rule: following it runs into a tracked person (that person's cover is there)")
        } else if led == Some("nothing") {
            Py::from("rule: following it finds no face")
        } else if sure {
            Py::from("rule: clear (detector >= 0.8, face check >= 0.6, or two kinds of evidence)")
        } else if seen_big {
            Py::Str(format!("rule: big enough to check ({} px), and no face found in it", graph::f(size, 0)))
        } else if weak {
            Py::from("rule: nothing speaks for it (detector < 0.6, face check < 0.5, no skeleton head)")
        } else {
            Py::from("rule: one weak kind of evidence, seen once or twice (almost never a face; not asked)")
        };
        d.insert("answer", answer.clone());
        d.insert("by", by.clone());
        d.insert("rule_answer", answer);
        d.insert("rule_by", by);
        decisions.push(d);
    }
    decisions.sort_by(|a, b| {
        let ka = (!a.get("answer").unwrap().is_null(), -a.get("evidence").unwrap().get("detector_score").as_f64().unwrap());
        let kb = (!b.get("answer").unwrap().is_null(), -b.get("evidence").unwrap().get("detector_score").as_f64().unwrap());
        ka.partial_cmp(&kb).unwrap()
    });
    decisions
}

fn dropped(d: &Dict) -> bool {
    matches!(d.get("answer").and_then(|a| a.as_str()), Some(a) if a == fx::A_NOT_FACE || a == "dropped: too weak")
}

/// blips.auto: drop the blips the rule says aren't faces; save the records (unless write is false).
#[allow(clippy::too_many_arguments)]
pub fn auto(proj: &Path, st: &State, members: &mut IndexMap<i64, PySet>, poses: Option<&[Vec<Pose>]>, photo: bool, clip: &Clip,
            ev: &mut Evidence, write: bool) -> Result<Vec<Dict>, String> {
    let decisions = find(st, members, poses, photo, clip, ev);
    if write {
        overlaps::save_records(&proj.join(FILE), &decisions)?;
    }
    for d in &decisions {
        if dropped(d) {
            members.shift_remove(&d.get("about").unwrap().get("piece").as_i64().unwrap());
        }
    }
    Ok(decisions)
}

/// blips.find_tracks.
pub fn find_tracks(st: &State, members: &IndexMap<i64, PySet>) -> Vec<Dict> {
    let mut out: Vec<Dict> = vec![];
    for (&r, ms) in members {
        let obs = st.group_obs(ms);
        let score = group_scores(st, ms);
        if obs.len() <= MAX_DET {
            continue;
        }
        let fs = face_scores(st, ms);
        let clear = fs.iter().filter(|f| **f >= FACE_OK).count();
        let med = np::median(&score.values().copied().collect::<Vec<_>>());
        let sure = clear >= CLEAR_FRAMES || med >= SURE_SCORE;
        let never = NEVER_CLEAR && clear == 0 && !sure;
        let mut ts: Vec<i64> = obs.keys().copied().collect();
        ts.sort();
        let mut best = ts[0];
        for t in &ts {
            if score.get(t).copied().unwrap_or(0.0) > score.get(&best).copied().unwrap_or(0.0) {
                best = *t;
            }
        }
        let mut show: Vec<i64> = vec![ts[0], best, ts[ts.len() - 1]];
        show.sort();
        show.dedup();
        let mut d = Dict::new();
        d.insert("id", Py::Str(format!("track:{r}")));
        d.insert("kind", Py::from("track"));
        let mut about = Dict::new();
        about.insert("piece", Py::Int(r));
        d.insert("about", Py::Dict(about));
        d.insert("question", Py::from("Is the green box a face?"));
        d.insert("options", options(&["kept: covered", "dropped: too weak"]));
        let mut e = Dict::new();
        e.insert("frames", Py::Int(ts.len() as i64));
        e.insert("first", Py::Int(ts[0]));
        e.insert("last", Py::Int(ts[ts.len() - 1]));
        e.insert("clear_face_frames", Py::Int(clear as i64));
        e.insert("median_detector", Py::Float(py_round(med, 2)));
        e.insert("best_face_check", Py::Float(py_round(if fs.is_empty() { 0.0 } else { pymax(&fs) }, 2)));
        e.insert("size", Py::Int(py_round_int(np::median(&obs.values().map(|o| o[2]).collect::<Vec<_>>()))));
        d.insert("evidence", Py::Dict(e));
        d.insert("why", Py::Str(format!("{} frames, a clear face in {clear}, median detector {}", ts.len(), graph::f(med, 2))));
        d.insert("show", Py::List(show.iter().map(|t| {
            let mut s = Dict::new();
            s.insert("t", Py::Int(*t));
            s.insert("piece", fl1(&obs[t]));
            s.insert("person", Py::Null);
            Py::Dict(s)
        }).collect()));
        let answer = if sure { Py::from("kept: covered") } else if never { Py::from("dropped: too weak") } else { Py::Null };
        let by = if sure {
            Py::from("rule: clear (a clear face in >= 3 frames, or median detector >= 0.8)")
        } else if never {
            Py::from(NEVER_RULE)
        } else {
            Py::Null
        };
        d.insert("answer", answer.clone());
        d.insert("by", by.clone());
        d.insert("rule_answer", answer);
        d.insert("rule_by", by);
        out.push(d);
    }
    out.sort_by_key(|d| (!d.get("answer").unwrap().is_null(), -d.get("evidence").unwrap().get("frames").as_i64().unwrap()));
    out
}

/// blips.auto_tracks.
pub fn auto_tracks(proj: &Path, st: &State, members: &mut IndexMap<i64, PySet>, write: bool) -> Result<Vec<Dict>, String> {
    let decisions = find_tracks(st, members);
    if write {
        overlaps::save_records(&proj.join(TRACK_FILE), &decisions)?;
    }
    for d in &decisions {
        if dropped(d) {
            members.shift_remove(&d.get("about").unwrap().get("piece").as_i64().unwrap());
        }
    }
    Ok(decisions)
}
