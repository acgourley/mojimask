//! A project's edits document and facts log, with their shared undo history. Port of the "edits document" and
//! "facts" parts of `mojimask/review_server.py` (apply_ops, undo_last, sync_facts, change_facts, the /api/ops,
//! /api/facts and /api/decisions handlers, split_cover, undo_correction, propagate).
//!
//! The scan (its decide step) leaves PROJECT/proto_tracks.json: the people it found, their covers made. What people
//! say about the boxes is a log of facts ([`crate::facts`], PROJECT/facts.json); the edits (PROJECT/edits.json) hold
//! the rest (masks drawn, splits, trims, style) and the fields derived from the facts.
//!
//! **The edits document** ([`crate::ops`]): pages send small operations on paths; a batch is applied all or
//! nothing, saved at once (`json.dump`, write + rename) and is one step on the undo history (the last 200 kept; a new
//! change clears redo). Bookkeeping (`undoable: false`) isn't a step.
//!
//! **Facts:** who's who and what's a face is one log. The edits fields in [`crate::ops::DERIVED`] are derived from it
//! after every change ([`Project::sync_facts`]) and never written by pages: /api/ops drops ops on them. Undo takes the
//! last facts back out of the log, like any other change. A change of emoji preset (`palette`) re-derives too:
//! hand-picked emoji belong to the preset they were picked under.
//!
//! **Covers you place** (edits.manual[id]: keys you placed, prop followed from them, ev the frames with face
//! evidence, pv the follower's version):
//! - [`Project::split_cover`]: split a cover in two at a frame: the part before t stays mid, the rest becomes a new
//!   cover, each with explicit first/last frames so neither carries on over the other. A half without a keyframe
//!   gets one where its path is, at its first (or last) frame. The new cover joins mid's card (its named Person, else
//!   a link to mid), like a split track; you then move whichever half is wrong. One step on undo.
//! - [`Project::undo_correction`]: take back where you placed cover mid on frame t and re-follow the cover from your
//!   other placements, so the path that placement made goes too, not just the one frame. The cover's only placement
//!   can't go (use Not a face, or end / start the box). One undo step.
//! - [`propagate`]: follow a box you placed forward and backward from frame t. Each frame: local flow moves it; a
//!   weak detection or skeleton head within half a box is taken as the face and the box snaps to it (those frames
//!   are the evidence). It stops when flow fails, when it has gone PROP_BLIND_S without evidence, or when an
//!   automatic emoji already covers it for 3 frames (handoff). Plain flow used to run the full 60 frames each way: on
//!   a child being spun it slid onto the dress, the furniture and the dad, and track.resolve drew it instead of her
//!   detected face (twirl.mp4, 2026-11-10).
//!
//! Not here yet (they need the vision port, P5): the optical flow and face evidence [`propagate`] runs on (behind
//! the [`Follow`] trait), and the "whose face is this?" questions for boxes you made (they compare faces and
//! bodies with SFace / OSNet). Decision lists here hold the scan's records only.
use crate::facts as FX;
use crate::ops::{self, DERIVED};
use crate::pyjson::{self, py_round, py_round_int, write_atomic, Dict, Py};
use crate::pydict;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

pub const EDITS: &str = "edits.json";
pub const DECISIONS: &str = "decisions_"; // decisions_<kind>.json
pub const UNDO_KEEP: usize = 200;
pub const PROP_VERSION: i64 = 3; // edits.manual[id].pv: covers followed by this version of propagate()
pub const PROP_SNAP: f64 = 0.5; // box sizes: a weak detection or skeleton head this close is the face -> snap to it
pub const PROP_BLIND_S: f64 = 0.5; // seconds a cover may run on without any such evidence before it stops
pub const MANUAL_MAX_GAP: i64 = 90; // frames: a longer gap between two of a cover's boxes is left empty
pub const MANUAL_HOLD: i64 = 10;

/// One step on the undo history.
#[derive(Clone, Debug)]
pub enum Step {
    Ops { ops: Vec<Py>, label: String },
    Facts { remove: Vec<i64>, restore: Vec<Py>, label: String },
}

impl Step {
    fn label(&self) -> &str {
        match self {
            Step::Ops { label, .. } | Step::Facts { label, .. } => label,
        }
    }
}

/// An error a page gets back (the server's 400 / 404).
#[derive(Clone, Debug, PartialEq)]
pub struct Refused {
    pub code: u16,
    pub error: String,
}

fn bad(e: impl Into<String>) -> Refused {
    Refused { code: 400, error: e.into() }
}

/// What a camera / face follower gives [`propagate`] (P5 ports it; tests and parity replay a recording).
pub trait Follow {
    /// Move square box c=[cx,cy,s] from frame `from` to frame `to` by median optical flow of points inside it;
    /// None when tracking is unreliable (review_server.flow_step).
    fn flow_step(&mut self, from: i64, to: i64, c: [f64; 3]) -> Option<[f64; 3]>;
    /// Where frame u shows a face for a box near c: the closest weak detection of about that size, or a confident
    /// skeleton head, within PROP_SNAP box sizes (never a neighbour's tracked face) (review_server.face_evidence).
    fn face_evidence(&mut self, u: i64, c: [f64; 3], prev: [f64; 3], step: i64) -> Option<[f64; 3]>;
}

/// The project's frames and the scan's covers: what drawing-free geometry needs (split_cover, propagate).
pub struct Geometry {
    pub n: i64,
    pub w: f64,
    pub h: f64,
    pub fps: f64,
    /// camera motion: frame t-1 -> t, 2x3 (mats_v2.npy)
    pub mats: Vec<[[f64; 3]; 2]>,
    /// the scan's covers per track: {t: [cx, cy, s]}
    pub auto: Vec<HashMap<i64, [f64; 3]>>,
}

/// A float64 C-order .npy file (mats_v2.npy) -> (shape, values).
pub fn read_npy_f64(path: &Path) -> Result<(Vec<usize>, Vec<f64>), String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err("not a .npy file".into());
    }
    let (hl, off) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let head = std::str::from_utf8(&b[off..off + hl]).map_err(|e| e.to_string())?;
    if !head.contains("'descr': '<f8'") || head.contains("'fortran_order': True") {
        return Err(format!("npy: want little-endian float64, C order: {head}"));
    }
    let sh = head.split("'shape': (").nth(1).and_then(|s| s.split(')').next()).ok_or("npy: no shape")?;
    let shape: Vec<usize> = sh.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    let vals = b[off + hl..].chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    Ok((shape, vals))
}

impl Geometry {
    pub fn load(work: &Path) -> Result<Geometry, String> {
        let meta = pyjson::load(&work.join("meta.json"))?;
        let (shape, v) = read_npy_f64(&work.join("mats_v2.npy"))?;
        if shape.len() != 3 || shape[1] != 2 || shape[2] != 3 {
            return Err(format!("mats_v2.npy: shape {shape:?}"));
        }
        let mats = v.chunks_exact(6).map(|m| [[m[0], m[1], m[2]], [m[3], m[4], m[5]]]).collect::<Vec<_>>();
        let proto = pyjson::load(&work.join("proto_tracks.json"))?;
        let mut auto = vec![];
        if let Some(ts) = proto.get("tracks").as_dict() {
            for (_, tr) in ts.iter() {
                let mut m = HashMap::new();
                if let Some(c) = tr.get("cover").as_dict() {
                    for (t, v) in c.iter() {
                        if let (Ok(t), Some(b)) = (t.parse::<i64>(), box3(v)) {
                            m.insert(t, b);
                        }
                    }
                }
                auto.push(m);
            }
        }
        let n = meta.get("frames").as_i64().unwrap_or(mats.len() as i64);
        Ok(Geometry {
            n,
            w: meta.get("width").as_f64().unwrap_or(0.0),
            h: meta.get("height").as_f64().unwrap_or(0.0),
            fps: meta.get("fps").as_f64().unwrap_or(30.0),
            mats,
            auto,
        })
    }
}

fn box3(v: &Py) -> Option<[f64; 3]> {
    let l = v.as_list()?;
    Some([l.first()?.as_f64()?, l.get(1)?.as_f64()?, l.get(2)?.as_f64()?])
}

// ---- the camera-compensated geometry of covers you place (track.py's warp_cs / compose / manual_boxes)
// numpy's own arithmetic (crate::np): `A @ B` is dgemm, `M @ x` dgemv, `np.linalg.det` dgetrf then
// sign * exp(sum log|u_ii|), so a moved cover lands on the same bits as Python's.
type M23 = crate::np::M23;
type M33 = crate::np::M33;

/// Affine taking frame a coords -> frame b coords (a < b) (track.compose: `M = vstack([mats[t], [0, 0, 1]]) @ M`).
pub fn compose(mats: &[M23], a: i64, b: i64) -> M23 {
    let mut m: M33 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for t in a + 1..=b {
        let x = mats[t as usize];
        m = crate::np::matmul33(&[x[0], x[1], [0.0, 0.0, 1.0]], &m);
    }
    [m[0], m[1]]
}

/// [cx, cy, s] moved by M (track.warp_cs: `M @ [cx, cy, 1]`, s * sqrt|det M[:, :2]|).
pub fn warp_cs(c: [f64; 3], m: &M23) -> [f64; 3] {
    let p = crate::np::gemv(m, [c[0], c[1], 1.0]);
    let s = crate::np::det2(m[0][0], m[0][1], m[1][0], m[1][1]).abs().sqrt();
    [p[0], p[1], c[2] * s]
}

/// cv2.invertAffineTransform.
pub fn invert_affine(m: &M23) -> M23 {
    let d = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let d = if d != 0.0 { 1.0 / d } else { 0.0 };
    let (a11, a22, a12, a21) = (m[1][1] * d, m[0][0] * d, -m[0][1] * d, -m[1][0] * d);
    let b1 = -a11 * m[0][2] - a12 * m[1][2];
    let b2 = -a21 * m[0][2] - a22 * m[1][2];
    [[a11, a12, b1], [a21, a22, b2]]
}

fn frames_of(d: &Py) -> Result<Vec<(i64, [f64; 3])>, Refused> {
    let mut out = vec![];
    if let Some(d) = d.as_dict() {
        for (k, v) in d.iter() {
            let t: i64 = k.trim().parse().map_err(|_| bad(format!("invalid literal for int() with base 10: '{k}'")))?;
            out.push((t, box3(v).ok_or_else(|| bad("a box: [cx, cy, s]"))?));
        }
    }
    Ok(out)
}

/// A cover you placed: your keyframes + followed frames; camera-compensated interpolation between them (gaps up to
/// MANUAL_MAX_GAP frames), and `hold` frames of camera-carried extension at both ends (track.manual_boxes).
pub fn manual_boxes(m: &Py, mats: &[M23], n: i64, hold: i64) -> Result<BTreeMap<i64, [f64; 3]>, Refused> {
    let mut pts: BTreeMap<i64, [f64; 3]> = frames_of(m.get("prop"))?.into_iter().collect();
    if !m.as_dict().is_some_and(|d| d.contains("keys")) {
        return Err(bad("'keys'"));
    }
    let keys: BTreeMap<i64, [f64; 3]> = frames_of(m.get("keys"))?.into_iter().collect();
    pts.extend(keys.iter().map(|(t, c)| (*t, *c)));
    if pts.is_empty() {
        return Ok(BTreeMap::new());
    }
    // A followed path may only change the box's size so much: 0.67x to 1.5x the nearest keyframe. (Flow
    // allows +-10% per frame, which compounded over a long path once grew a face box past the frame.)
    if !keys.is_empty() {
        for (t, c) in pts.iter_mut() {
            if !keys.contains_key(t) {
                let k = keys.keys().min_by_key(|k| (*k - t).abs()).unwrap(); // (first nearest, in order)
                let s0 = keys[k][2];
                c[2] = c[2].max(0.67 * s0).min(1.5 * s0);
            }
        }
    }
    let ts: Vec<i64> = pts.keys().copied().collect();
    let mut out = pts.clone();
    for w in ts.windows(2) {
        let (t0, t1) = (w[0], w[1]);
        if t1 - t0 > MANUAL_MAX_GAP {
            continue;
        }
        for t in t0 + 1..t1 {
            let al = (t - t0) as f64 / (t1 - t0) as f64;
            let fwd = warp_cs(pts[&t0], &compose(mats, t0, t));
            let bwd = warp_cs(pts[&t1], &invert_affine(&compose(mats, t, t1)));
            out.insert(t, [0, 1, 2].map(|i| (1.0 - al) * fwd[i] + al * bwd[i]));
        }
    }
    let mut c = pts[&ts[0]];
    let mut t = ts[0] - 1;
    while t >= (ts[0] - hold).max(0) {
        c = warp_cs(c, &invert_affine(&mats[(t + 1) as usize]));
        out.entry(t).or_insert(c);
        t -= 1;
    }
    let last = *ts.last().unwrap();
    let mut c = pts[&last];
    for t in last + 1..=(last + hold).min(n - 1) {
        c = warp_cs(c, &mats[t as usize]);
        out.entry(t).or_insert(c);
    }
    let (lo, hi) = (m.get("from").as_i64(), m.get("to").as_i64()); // a cover split in two: each half ends where the other begins
    if lo.is_some() || hi.is_some() {
        out.retain(|t, _| lo.is_none_or(|lo| *t >= lo) && hi.is_none_or(|hi| *t <= hi));
    }
    Ok(out)
}

fn from_cs(c: [f64; 3]) -> [f64; 4] {
    let h = c[2] * 1.0 / 2.0;
    [c[0] - h, c[1] - h, c[0] + h, c[1] + h]
}

fn iou(a: [f64; 4], b: [f64; 4]) -> f64 {
    let (x1, y1) = (a[0].max(b[0]), a[1].max(b[1]));
    let (x2, y2) = (a[2].min(b[2]), a[3].min(b[3]));
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    if ua > 0.0 { inter / ua } else { 0.0 }
}

/// Follow a box you placed (a cover) forward and backward from frame t -> (prop {frame: box} in the order found,
/// ev [frames]). See the module docs for the rules.
pub fn propagate(g: &Geometry, f: &mut dyn Follow, t: i64, c: [f64; 3], limit: i64) -> (Vec<(i64, [f64; 3])>, Vec<i64>) {
    let mut prop = vec![];
    let mut ev = vec![];
    let blind_max = 3.max(py_round_int(PROP_BLIND_S * g.fps));
    for step in [1i64, -1] {
        let (mut cur, mut covered, mut blind) = (c, 0, 0);
        for k in 1..=limit {
            let u = t + step * k;
            if !(0 <= u && u < g.n) {
                break;
            }
            let Some(mut nxt) = f.flow_step(u - step, u, cur) else { break };
            if !(0.0 <= nxt[0] && nxt[0] < g.w && 0.0 <= nxt[1] && nxt[1] < g.h) {
                break;
            }
            if let Some(seen) = f.face_evidence(u, nxt, cur, step) {
                nxt = seen;
                blind = 0;
                ev.push(u);
            } else {
                blind += 1;
                if blind > blind_max {
                    break;
                }
            }
            nxt[2] = nxt[2].max(0.67 * c[2]).min(1.5 * c[2]); // size stays near the placed box's
            cur = nxt;
            prop.push((u, cur.map(|v| py_round(v, 1))));
            let b = from_cs(cur);
            // an automatic box on this face: overlapping, or centred on it (a cover can be bigger than the
            // face, e.g. seeded from a box that drifted off a nearer, bigger face, so overlap alone misses it)
            let on = g.auto.iter().filter_map(|tr| tr.get(&u)).any(|ab| {
                iou(from_cs(*ab), b) > 0.4 || (ab[0] - cur[0]).hypot(ab[1] - cur[1]) < 0.25 * ab[2].min(cur[2])
            });
            covered = if on { covered + 1 } else { 0 };
            if covered >= 3 {
                break;
            }
        }
    }
    ev.sort();
    (prop, ev)
}

/// prop as the server sends and saves it: {"<frame>": [cx, cy, s]}.
pub fn prop_json(prop: &[(i64, [f64; 3])]) -> Dict {
    prop.iter().map(|(u, c)| (u.to_string(), Py::List(c.iter().map(|v| Py::Float(*v)).collect()))).collect()
}

/// An open project: its edits, facts and undo history (the state a project server keeps).
pub struct Project {
    pub work: PathBuf,
    pub edits: Py,
    pub rev: i64,
    pub undo: Vec<Step>,
    pub redo: Vec<Step>,
    pub facts: Py,
    /// the scan's own hidden boxes (its baseline for not_face)
    pub hidden: Vec<String>,
    /// milliseconds since 1970 (facts are stamped with it)
    pub clock: Box<dyn FnMut() -> i64 + Send>,
    geometry: Option<Geometry>,
}

pub fn system_clock() -> Box<dyn FnMut() -> i64 + Send> {
    Box::new(|| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0))
}

impl Project {
    /// Open a project as its server starts: the edits (blanks filled in), the facts (a log for another scan version
    /// set aside), and the edits as the facts stand (saved if that changed them).
    pub fn open(work: &Path, clock: Box<dyn FnMut() -> i64 + Send>) -> Result<Project, String> {
        let proto = pyjson::load(&work.join("proto_tracks.json"))
            .map_err(|_| format!("{}: no proto_tracks.json -- scanned before the current algorithm; scan the video again", work.display()))?;
        let mut edits = pydict![
            ("deleted", Py::List(vec![])), ("trims", Dict::new()), ("merges", Dict::new()), ("manual", Dict::new()),
            ("emoji", Dict::new()), ("dismissed", Py::List(vec![])), ("shown", Py::List(vec![])), ("people", Dict::new()),
        ];
        let ep = work.join(EDITS);
        {
            let d = edits.as_dict_mut().unwrap();
            if ep.exists() {
                // a partial file (e.g. just a starting palette) fills in the rest
                let file = pyjson::load(&ep)?;
                for (k, v) in file.as_dict().ok_or("edits.json: not an object")?.iter() {
                    d.insert(k, v.clone());
                }
            } else {
                d.insert("head_pose", Py::Bool(true)); // no edits yet (a fresh test copy): as a new project starts, tilt & turn on
            }
            d.insert("version", Py::Int(ops::EDITS_VERSION));
        }
        let rev = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0);
        let sv = proto.as_dict().and_then(|d| d.get("scan_version")).cloned().unwrap_or(Py::Int(0));
        let facts = FX::load(work, if sv.is_null() { None } else { Some(&sv) })?;
        let hidden = match proto.get("hidden") {
            Py::Null => {
                // a scan from before the list was kept: its hidden boxes are the edits' "decision" ones
                let mut h: Vec<String> = edits.get("not_face").as_dict().map(|d| d.iter()
                    .filter(|(_, v)| v.as_str() == Some("decision")).map(|(k, _)| k.to_string()).collect()).unwrap_or_default();
                h.sort();
                h
            }
            l => l.as_list().map(|l| l.iter().map(Py::py_str).collect()).unwrap_or_default(),
        };
        let mut p = Project { work: work.to_path_buf(), edits, rev, undo: vec![], redo: vec![], facts, hidden, clock, geometry: None };
        p.sync_facts()?; // the edits as the facts stand (a log set aside for another scan version starts empty)
        Ok(p)
    }

    pub fn geometry(&mut self) -> Result<&Geometry, String> {
        if self.geometry.is_none() {
            self.geometry = Some(Geometry::load(&self.work)?);
        }
        Ok(self.geometry.as_ref().unwrap())
    }

    pub fn facts(&self) -> &[Py] {
        FX::facts_of(&self.facts)
    }

    fn save_edits(&self) -> Result<(), String> {
        write_atomic(&self.work.join(EDITS), &self.edits.dumps()).map_err(|e| e.to_string())
    }

    fn remember(&mut self, step: Step) {
        self.undo.push(step);
        if self.undo.len() > UNDO_KEEP {
            let n = self.undo.len() - UNDO_KEEP;
            self.undo.drain(..n);
        }
        self.redo.clear();
    }

    /// Apply a batch of ops as one change: saved at once, one undo step (if `undoable`). Returns (the new revision,
    /// the ops that would take this change back).
    pub fn apply_ops(&mut self, ops: &[Py], label: &str, undoable: bool) -> Result<(i64, Vec<Py>), Refused> {
        let (work, inverse) = ops::apply_all(&self.edits, ops).map_err(bad)?; // all or nothing
        if inverse.is_empty() {
            return Ok((self.rev, inverse));
        }
        self.edits = work;
        self.rev += 1;
        if undoable {
            self.remember(Step::Ops { ops: inverse.clone(), label: label.to_string() });
        }
        self.save_edits().map_err(|e| Refused { code: 500, error: e })?;
        Ok((self.rev, inverse))
    }

    /// Write the facts' consequences into the edits (not an undo step of its own).
    pub fn sync_facts(&mut self) -> Result<(), String> {
        let preset = self.edits.get("palette").clone(); // hand-picked emoji belong to the preset they were picked under
        let ops: Vec<Py> = FX::derive(self.facts(), &self.hidden, &preset).into_iter()
            .map(|(k, v)| pydict![("op", "set"), ("path", vec![Py::Str(k)]), ("value", v)]).collect();
        self.apply_ops(&ops, "", false).map(|_| ()).map_err(|e| e.error)
    }

    fn save_facts(&self) -> Result<(), Refused> {
        FX::save(&self.work, &self.facts).map_err(|e| Refused { code: 500, error: e })
    }

    /// Retract facts (ids) and add new ones as one change and one undo step. Returns the new facts' ids.
    pub fn change_facts(&mut self, add: &[Py], remove: &[i64], label: &str, page: &str, decision: Option<&str>,
                        via: Option<&str>) -> Result<Vec<i64>, Refused> {
        let gone: Vec<Py> = self.facts().iter().filter(|f| remove.contains(&FX::fid(f))).cloned().collect();
        FX::retract(&mut self.facts, remove);
        let palette = self.edits.get("palette").clone();
        let default = Py::from(FX::emoji_preset());
        let add: Vec<Py> = add.iter().map(|f| {
            // an emoji pick is for the preset in use
            let mut f = f.clone();
            if f.get("kind").as_str() == Some("emoji") && !f.get("preset").truthy() {
                f.as_dict_mut().unwrap().insert("preset", palette.or(&default).clone());
            }
            f
        }).collect();
        let now = (self.clock)();
        let ids = FX::append(&mut self.facts, &add, page, decision, via, now).map_err(bad)?;
        self.save_facts()?;
        if !ids.is_empty() || !gone.is_empty() {
            self.remember(Step::Facts { remove: ids.clone(), restore: gone, label: label.to_string() });
        }
        self.sync_facts().map_err(|e| Refused { code: 500, error: e })?;
        Ok(ids)
    }

    /// Apply a history step (an undo or a redo); returns the step that takes it back.
    fn take(&mut self, step: Step) -> Result<Step, Refused> {
        match step {
            Step::Facts { remove, restore, label } => {
                let gone: Vec<Py> = self.facts().iter().filter(|f| remove.contains(&FX::fid(f))).cloned().collect();
                FX::retract(&mut self.facts, &remove);
                let restored: Vec<i64> = restore.iter().map(FX::fid).collect();
                let mut all: Vec<Py> = self.facts().to_vec();
                all.extend(restore);
                all.sort_by_key(FX::fid);
                self.facts.as_dict_mut().unwrap().insert("facts", Py::List(all));
                self.save_facts()?;
                self.sync_facts().map_err(|e| Refused { code: 500, error: e })?;
                Ok(Step::Facts { remove: restored, restore: gone, label })
            }
            Step::Ops { ops, label } => {
                let (_, back) = self.apply_ops(&ops, "", false)?;
                if ops.iter().any(|o| first_path(o) == "palette") {
                    self.sync_facts().map_err(|e| Refused { code: 500, error: e })?; // (the preset came back: so do the picks made under it)
                }
                Ok(Step::Ops { ops: back, label })
            }
        }
    }

    /// Undo the last change (from any page), or redo the last undone one. Returns its label ("last change" when it
    /// had none), or None if there is nothing to undo (redo).
    pub fn undo_last(&mut self, redo: bool) -> Result<Option<String>, Refused> {
        let step = match if redo { self.redo.pop() } else { self.undo.pop() } {
            Some(s) => s,
            None => return Ok(None),
        };
        let label = step.label().to_string();
        let back = self.take(step)?;
        if redo { self.undo.push(back) } else { self.redo.push(back) }
        Ok(Some(if label.is_empty() { "last change".into() } else { label }))
    }

    /// POST /api/ops: {ops, label, undoable}: a page's changes. Ops on the fields the facts own are dropped.
    pub fn post_ops(&mut self, body: &Py) -> Result<(), Refused> {
        let ops = body.get("ops").as_list().cloned().unwrap_or_default();
        let mut keep = vec![];
        for o in &ops {
            if o.as_dict().is_none() {
                return Err(bad(format!("'{}' object has no attribute 'get'", if o.is_null() { "NoneType" } else { "value" })));
            }
            if !DERIVED.contains(&first_path(o).as_str()) {
                keep.push(o.clone()); // who's who comes from the facts: pages say it with /api/facts
            }
        }
        let label = body.get("label").as_str().unwrap_or("").to_string();
        let undoable = body.as_dict().and_then(|d| d.get("undoable")).map(Py::truthy).unwrap_or(true);
        self.apply_ops(&keep, &label, undoable)?; // bookkeeping (edits.result) isn't undoable
        if keep.iter().any(|o| first_path(o) == "palette") {
            self.sync_facts().map_err(|e| Refused { code: 500, error: e })?; // another emoji preset: its own picks take the place of the last one's
        }
        Ok(())
    }

    /// POST /api/facts: {facts: [{kind, pieces, value}], label, page}: what a page says about boxes.
    pub fn post_facts(&mut self, body: &Py) -> Result<(), Refused> {
        let add = body.get("facts").as_list().cloned().unwrap_or_default();
        let label = body.get("label").as_str().unwrap_or("").to_string();
        let page = body.get("page").as_str().unwrap_or("").to_string();
        self.change_facts(&add, &[], &label, &page, None, None).map(|_| ())
    }

    fn decision_files(&self) -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = std::fs::read_dir(&self.work).map(|rd| rd.flatten().map(|e| e.path())
            .filter(|p| p.file_name().map(|n| n.to_string_lossy()).is_some_and(|n| n.starts_with(DECISIONS) && n.ends_with(".json")))
            .collect()).unwrap_or_default();
        v.sort();
        v
    }

    /// The scan's decision records as saved (decisions_<kind>.json, files in name order); None when there are none
    /// (still scanning, or scanned before decisions existed).
    pub fn decision_records(&self) -> Result<Option<Vec<Py>>, String> {
        let files = self.decision_files();
        if files.is_empty() {
            return Ok(None);
        }
        let mut out = vec![];
        for f in files {
            out.extend(pyjson::load(&f)?.get("decisions").as_list().cloned().unwrap_or_default());
        }
        Ok(Some(out))
    }

    /// GET /api/decisions: every record with its answer as the facts stand. None: no decisions yet.
    pub fn decisions(&self) -> Result<Option<Vec<Py>>, String> {
        Ok(self.decision_records()?.map(|ds| FX::answers(&ds, self.facts(), &self.hidden)))
    }

    /// POST /api/decisions: {id, answer[, via]}: answer (or re-answer, or clear with answer null) a decision, as
    /// facts. Returns the record with its new answer.
    pub fn decide(&mut self, body: &Py) -> Result<Py, Refused> {
        self.decide_with(body, &[])
    }

    /// [`Project::decide`] over the scan's records and then `extra` (review_server's your_box_questions_now(): the
    /// "whose face is this?" questions for boxes you made, worked out by the caller).
    pub fn decide_with(&mut self, body: &Py, extra: &[Py]) -> Result<Py, Refused> {
        let via = body.get("via").clone();
        if !(via.is_null() || via.as_str() == Some("agent")) {
            return Err(bad("via must be \"agent\""));
        }
        let mut sources: Vec<Option<PathBuf>> = self.decision_files().into_iter().map(Some).collect();
        sources.push(None);
        for f in sources {
            let src = match &f {
                Some(f) => pyjson::load(f).map_err(|e| Refused { code: 500, error: e })?,
                None => pydict![("decisions", Py::List(extra.to_vec()))],
            };
            let Some(d) = src.get("decisions").as_list().and_then(|l| l.iter().find(|x| x.get("id") == body.get("id"))).cloned() else {
                continue;
            };
            let ans = body.get("answer").clone();
            if !ans.is_null() && !d.get("options").as_list().is_some_and(|o| o.contains(&ans)) {
                return Err(bad("not an option"));
            }
            let did = d.get("id").py_str();
            let old: Vec<i64> = self.facts().iter().filter(|f| f.get("decision").py_str() == did && !f.get("decision").is_null())
                .map(FX::fid).collect();
            let add = if ans.is_null() { vec![] } else { FX::facts_for_answer(&d, &ans) };
            let label = if ans.truthy() { format!("Answer: {}", ans.py_str()) } else { "Clear answer".into() };
            self.change_facts(&add, &old, &label, "decisions", Some(&did), via.as_str())?;
            return Ok(FX::answers(&[d], self.facts(), &self.hidden).remove(0));
        }
        Err(Refused { code: 404, error: "no such decision".into() })
    }

    /// Split cover mid (edits.manual) at frame t (see the module docs). -> the new cover's id.
    pub fn split_cover(&mut self, mid: &str, t: i64) -> Result<String, Refused> {
        let e = self.edits.clone();
        let m = e.get("manual").get(mid).clone();
        if !m.truthy() {
            return Err(bad(format!("no cover {mid}")));
        }
        let g = self.geometry().map_err(|e| Refused { code: 500, error: e })?;
        let n = g.n;
        let bx = manual_boxes(&m, &g.mats, n, MANUAL_HOLD)?;
        let ts: Vec<i64> = bx.keys().copied().filter(|u| 0 <= *u && *u < n).collect();
        if ts.is_empty() || !(ts[0] < t && t <= *ts.last().unwrap()) {
            let (a, b) = ts.first().zip(ts.last()).map(|(a, b)| (a.to_string(), b.to_string())).unwrap_or(("-".into(), "-".into()));
            return Err(bad(format!("cover {mid} runs {a}..{b}: can't split at {t}")));
        }
        let rnd = |c: [f64; 3]| Py::List(c.iter().map(|v| Py::Float(py_round(*v, 1))).collect());
        let mut halves = vec![];
        for (lo, hi) in [(ts[0], t - 1), (t, *ts.last().unwrap())] {
            let part = |d: &Py| -> Result<Dict, Refused> {
                let mut out = Dict::new();
                if let Some(d) = d.as_dict() {
                    for (k, v) in d.iter() {
                        let u: i64 = k.trim().parse().map_err(|_| bad(format!("invalid literal for int() with base 10: '{k}'")))?;
                        if lo <= u && u <= hi {
                            out.insert(k, v.clone());
                        }
                    }
                }
                Ok(out)
            };
            let (mut keys, mut prop) = (part(m.get("keys"))?, part(m.get("prop"))?);
            if keys.is_empty() {
                let at = if lo == t { lo } else { hi };
                let c = *bx.get(&at).ok_or_else(|| bad(at.to_string()))?;
                keys.insert(at.to_string(), rnd(c));
                prop.remove(&at.to_string());
            }
            let ev: Vec<Py> = m.get("ev").as_list().map(|l| l.iter().filter(|u| u.as_f64().is_some_and(|u| lo as f64 <= u && u <= hi as f64)).cloned().collect()).unwrap_or_default();
            let mut h = pydict![("keys", keys), ("prop", prop), ("ev", ev), ("from", lo), ("to", hi)];
            if let Some(pv) = m.as_dict().and_then(|d| d.get("pv")) {
                h.as_dict_mut().unwrap().insert("pv", pv.clone());
            }
            halves.push(h);
        }
        let top = e.get("manual").as_dict().map(|d| d.keys().filter_map(|k| {
            let r: String = k.chars().skip(1).collect();
            (!r.is_empty() && r.chars().all(|c| c.is_ascii_digit())).then(|| r.parse::<i64>().unwrap_or(0))
        }).max().unwrap_or(0)).unwrap_or(0).max(0);
        let new = format!("m{}", top + 1);
        let mut opl = vec![
            pydict![("op", "set"), ("path", vec![Py::from("manual"), Py::from(mid)]), ("value", halves[0].clone())],
            pydict![("op", "set"), ("path", vec![Py::from("manual"), Py::from(&new)]), ("value", halves[1].clone())],
        ];
        let pid = e.get("people").as_dict().and_then(|ps| ps.iter()
            .find(|(_, pp)| pp.get("pieces").as_list().is_some_and(|l| l.contains(&Py::from(mid)))).map(|(p, _)| p.to_string()));
        match pid {
            Some(pid) => opl.push(pydict![("op", "add"), ("path", vec![Py::from("people"), Py::from(pid), Py::from("pieces")]), ("value", new.as_str())]),
            None => opl.push(pydict![("op", "set"), ("path", vec![Py::from("merges"), Py::from(&new)]), ("value", mid)]),
        }
        let why = e.get("not_face").get(mid);
        if why.truthy() {
            // a box marked not a face: so is its other half, until you say otherwise
            opl.push(pydict![("op", "set"), ("path", vec![Py::from("not_face"), Py::from(&new)]), ("value", why.clone())]);
        }
        self.apply_ops(&opl, &format!("split {mid} at {t}"), true)?;
        Ok(new)
    }

    /// Take back where you placed cover mid on frame t, and re-follow it from your other placements (module docs).
    pub fn undo_correction(&mut self, mid: &str, t: i64, follow: &mut dyn Follow) -> Result<(), Refused> {
        let m = self.edits.get("manual").get(mid).clone();
        if !m.truthy() {
            return Err(bad(format!("no cover {mid}")));
        }
        let mut keys: Vec<(i64, Py)> = vec![];
        if let Some(d) = m.get("keys").as_dict() {
            for (k, v) in d.iter() {
                let u: i64 = k.trim().parse().map_err(|_| bad(format!("invalid literal for int() with base 10: '{k}'")))?;
                match keys.iter_mut().find(|(x, _)| *x == u) {
                    Some(e) => e.1 = v.clone(),
                    None => keys.push((u, v.clone())),
                }
            }
        }
        if !keys.iter().any(|(k, _)| *k == t) {
            return Err(bad(format!("you didn't place {mid} on frame {t}")));
        }
        if keys.len() == 1 {
            return Err(bad("that's the only place this box was set: use Not a face, or Start / End box here"));
        }
        let rest: Vec<(i64, Py)> = keys.into_iter().filter(|(k, _)| *k != t).collect();
        self.geometry().map_err(|e| Refused { code: 500, error: e })?;
        let g = self.geometry.as_ref().unwrap();
        let mut prop = Dict::new();
        let mut ev: Vec<i64> = vec![];
        for (k, c) in &rest {
            // in the order they were placed; the first to reach a frame sets it
            let c = box3(c).ok_or_else(|| bad("a box: [cx, cy, s]"))?;
            let (p, e_) = propagate(g, follow, *k, c, 60);
            for (u, v) in p {
                if !rest.iter().any(|(r, _)| *r == u) && !prop.contains(&u.to_string()) {
                    prop.insert(u.to_string(), Py::List(v.iter().map(|x| Py::Float(*x)).collect()));
                }
            }
            ev.extend(e_);
        }
        ev.sort();
        ev.dedup();
        let restd: Dict = rest.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
        let path = |k: &str| Py::List(vec![Py::from("manual"), Py::from(mid), Py::from(k)]);
        self.apply_ops(&[
            pydict![("op", "set"), ("path", path("keys")), ("value", restd)],
            pydict![("op", "set"), ("path", path("prop")), ("value", prop)],
            pydict![("op", "set"), ("path", path("ev")), ("value", ev.into_iter().map(Py::Int).collect::<Vec<_>>())],
            pydict![("op", "set"), ("path", path("pv")), ("value", PROP_VERSION)],
        ], &format!("undo correction on {mid} at {t}"), true)?;
        Ok(())
    }

    /// POST /api/propagate: {t, c}: a placed box followed -> {prop, ev, pv}. Changes nothing.
    pub fn propagate(&mut self, t: i64, c: [f64; 3], follow: &mut dyn Follow) -> Result<Py, Refused> {
        let g = self.geometry().map_err(|e| Refused { code: 500, error: e })?;
        if !(0 <= t && t < g.n) || c[2] <= 0.0 {
            return Err(bad("t: a frame, c: [cx, cy, s]"));
        }
        let (prop, ev) = propagate(g, follow, t, c, 60);
        Ok(pydict![("prop", prop_json(&prop)), ("ev", ev.into_iter().map(Py::Int).collect::<Vec<_>>()), ("pv", PROP_VERSION)])
    }

    /// Once per cover, when a project opens: re-follow covers made by an older propagate() (no edits.manual[id].pv)
    /// from their own keyframes, so a project made before the change gets the shorter, evidence-snapped paths too.
    /// Like a page building a cover (app.html coverFor), each keyframe is followed in turn, first one first, and the
    /// first keyframe to reach a frame sets it. Saved as one change that undo skips: it only re-derives paths from
    /// your keyframes and changes no answer. Returns how many covers were re-followed.
    pub fn refollow_covers(&mut self, follow: &mut dyn Follow) -> Result<usize, Refused> {
        let manual = self.edits.get("manual").as_dict().cloned().unwrap_or_default();
        self.geometry().map_err(|e| Refused { code: 500, error: e })?;
        let g = self.geometry.as_ref().unwrap();
        let mut opl = vec![];
        let mut n = 0;
        for (mid, m) in manual.iter() {
            if m.get("pv") == &Py::Int(PROP_VERSION) || !m.get("keys").truthy() {
                continue;
            }
            let keys = m.get("keys").as_dict().cloned().unwrap_or_default();
            let mut prop = Dict::new();
            let mut ev: Vec<i64> = vec![];
            for (t, c) in keys.iter() {
                let t: i64 = t.trim().parse().map_err(|_| bad(format!("invalid literal for int() with base 10: '{t}'")))?;
                let c = box3(c).ok_or_else(|| bad("a box: [cx, cy, s]"))?;
                let (p, e) = propagate(g, follow, t, c, 60);
                for (u, v) in p {
                    let k = u.to_string();
                    if !keys.contains(&k) && !prop.contains(&k) {
                        prop.insert(k, Py::List(v.iter().map(|x| Py::Float(*x)).collect()));
                    }
                }
                ev.extend(e);
            }
            ev.sort();
            ev.dedup();
            let path = |k: &str| Py::List(vec![Py::from("manual"), Py::from(mid), Py::from(k)]);
            opl.push(pydict![("op", "set"), ("path", path("prop")), ("value", prop)]);
            opl.push(pydict![("op", "set"), ("path", path("ev")), ("value", ev.into_iter().map(Py::Int).collect::<Vec<_>>())]);
            opl.push(pydict![("op", "set"), ("path", path("pv")), ("value", PROP_VERSION)]);
            n += 1;
        }
        if !opl.is_empty() {
            self.apply_ops(&opl, "", false)?;
        }
        Ok(n)
    }
}

fn first_path(o: &Py) -> String {
    match o.get("path") {
        Py::List(l) if !l.is_empty() => l[0].py_str(),
        Py::Str(s) if !s.is_empty() => s.chars().next().unwrap().to_string(),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Still; // a follower that never moves and never sees a face
    impl Follow for Still {
        fn flow_step(&mut self, _: i64, _: i64, c: [f64; 3]) -> Option<[f64; 3]> {
            Some(c)
        }
        fn face_evidence(&mut self, _: i64, _: [f64; 3], _: [f64; 3], _: i64) -> Option<[f64; 3]> {
            None
        }
    }

    fn project() -> (PathBuf, Project) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let d = std::env::temp_dir().join(format!("mm_edits_{}_{}", std::process::id(), N.fetch_add(1, Ordering::SeqCst)));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("meta.json"), r#"{"width": 100, "height": 100, "fps": 24.0, "frames": 40}"#).unwrap();
        std::fs::write(d.join("proto_tracks.json"), r#"{"scan_version": 2, "hidden": ["9"], "tracks": {"1": {"cover": {"30": [50, 50, 10]}}}}"#).unwrap();
        std::fs::write(d.join("decisions_track.json"), r#"{"decisions": [{"id": "track:5", "kind": "track", "about": {"piece": 5}, "options": ["a face", "not a face"], "evidence": {}, "answer": null, "by": null, "rule_answer": null, "rule_by": null}]}"#).unwrap();
        // identity camera motion
        let mut npy = b"\x93NUMPY\x01\x00".to_vec();
        let mut head = "{'descr': '<f8', 'fortran_order': False, 'shape': (40, 2, 3), }".to_string();
        while (10 + head.len() + 1) % 64 != 0 {
            head.push(' ');
        }
        head.push('\n');
        npy.extend((head.len() as u16).to_le_bytes());
        npy.extend(head.as_bytes());
        for _ in 0..40 {
            for v in [1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0] {
                npy.extend(v.to_le_bytes());
            }
        }
        std::fs::write(d.join("mats_v2.npy"), npy).unwrap();
        let mut t = 1000;
        let p = Project::open(&d, Box::new(move || { t += 1000; t })).unwrap();
        (d, p)
    }

    #[test]
    fn open_derives_and_saves() {
        let (d, p) = project();
        let e = pyjson::load(&d.join(EDITS)).unwrap();
        assert_eq!(e.get("not_face").dumps(), r#"{"9": "decision"}"#);
        assert_eq!(e.get("version"), &Py::Int(2));
        assert_eq!(e.get("head_pose"), &Py::Bool(true));
        assert_eq!(std::fs::read_to_string(d.join("facts.json")).unwrap(), "{\n \"scan_version\": 2,\n \"facts\": []\n}");
        drop(p);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn ops_undo_redo_and_stray_fields() {
        let (d, mut p) = project();
        p.post_ops(&Py::parse(r#"{"ops": [{"op": "set", "path": ["cover_style"], "value": "blur"},
                                          {"op": "set", "path": ["not_face", "3"], "value": "you"}], "label": "style"}"#).unwrap()).unwrap();
        assert_eq!(p.edits.get("cover_style").as_str(), Some("blur"));
        assert!(!p.edits.get("not_face").as_dict().unwrap().contains("3")); // the facts own it
        assert_eq!(p.undo_last(false).unwrap().as_deref(), Some("style"));
        assert!(p.edits.get("cover_style").is_null());
        assert_eq!(p.undo_last(true).unwrap().as_deref(), Some("style"));
        assert_eq!(p.edits.get("cover_style").as_str(), Some("blur"));
        assert_eq!(p.undo_last(true).unwrap(), None);
        // a bad batch changes nothing
        assert!(p.post_ops(&Py::parse(r#"{"ops": [{"op": "set", "path": ["x"], "value": 1}, {"op": "add", "path": ["x"], "value": 1}]}"#).unwrap()).is_err());
        assert!(p.edits.get("x").is_null());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn decide_clear_and_undo_through_facts() {
        let (d, mut p) = project();
        let r = p.decide(&Py::parse(r#"{"id": "track:5", "answer": "not a face", "via": "agent"}"#).unwrap()).unwrap();
        assert_eq!(r.get("answer").as_str(), Some("not a face"));
        assert_eq!(r.get("via").as_str(), Some("agent"));
        assert_eq!(p.edits.get("not_face").get("5").as_str(), Some("decision"));
        assert!(p.decide(&Py::parse(r#"{"id": "track:5", "answer": "maybe"}"#).unwrap()).is_err());
        assert_eq!(p.decide(&Py::parse(r#"{"id": "track:6", "answer": "a face"}"#).unwrap()).unwrap_err().code, 404);
        assert_eq!(p.undo_last(false).unwrap().as_deref(), Some("Answer: not a face"));
        assert!(p.edits.get("not_face").get("5").is_null());
        p.undo_last(true).unwrap();
        let r = p.decide(&Py::parse(r#"{"id": "track:5", "answer": null}"#).unwrap()).unwrap();
        assert!(r.get("answer").is_null());
        assert!(p.facts().is_empty());
        assert_eq!(p.facts.get("last_id"), &Py::Int(1));
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn split_and_undo_correction() {
        let (d, mut p) = project();
        p.post_ops(&Py::parse(r#"{"ops": [{"op": "set", "path": ["manual", "m1"], "value": {"keys": {"10": [20, 20, 10], "20": [30, 30, 10]}, "prop": {"11": [21, 21, 10]}, "ev": [11, 25]}}]}"#).unwrap()).unwrap();
        let new = p.split_cover("m1", 15).unwrap();
        assert_eq!(new, "m2");
        assert_eq!(p.edits.get("manual").get("m1").dumps(), r#"{"keys": {"10": [20, 20, 10]}, "prop": {"11": [21, 21, 10]}, "ev": [11], "from": 0, "to": 14}"#);
        assert_eq!(p.edits.get("manual").get("m2").dumps(), r#"{"keys": {"20": [30, 30, 10]}, "prop": {}, "ev": [25], "from": 15, "to": 30}"#);
        assert_eq!(p.edits.get("merges").get("m2").as_str(), Some("m1"));
        assert!(p.split_cover("m1", 40).is_err());
        p.post_ops(&Py::parse(r#"{"ops": [{"op": "set", "path": ["manual", "m3"], "value": {"keys": {"10": [20, 20, 10], "30": [50, 50, 10]}}}]}"#).unwrap()).unwrap();
        p.undo_correction("m3", 30, &mut Still).unwrap();
        let m3 = p.edits.get("manual").get("m3");
        assert_eq!(m3.get("keys").dumps(), r#"{"10": [20, 20, 10]}"#);
        // still: no evidence, so it runs 12 blind frames (0.5 s at 24 fps) each way, then stops
        assert_eq!(m3.get("prop").as_dict().unwrap().len(), 12 + 10);
        assert_eq!(m3.get("pv"), &Py::Int(3));
        assert!(p.undo_correction("m3", 10, &mut Still).unwrap_err().error.starts_with("that's the only place"));
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn refollow_only_old_covers() {
        let (d, mut p) = project();
        p.post_ops(&Py::parse(r#"{"ops": [{"op": "set", "path": ["manual", "m1"], "value": {"keys": {"10": [20, 20, 10]}}},
                                          {"op": "set", "path": ["manual", "m2"], "value": {"keys": {"10": [20, 20, 10]}, "pv": 3}}]}"#).unwrap()).unwrap();
        let undo = p.undo.len();
        assert_eq!(p.refollow_covers(&mut Still).unwrap(), 1);
        assert_eq!(p.edits.get("manual").get("m1").get("prop").as_dict().unwrap().len(), 12 + 10); // 12 blind frames forward, 10 back to frame 0
        assert!(p.edits.get("manual").get("m2").get("prop").is_null());
        assert_eq!(p.undo.len(), undo); // not an undo step
        assert_eq!(p.refollow_covers(&mut Still).unwrap(), 0);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn affine_helpers() {
        let m = [[2.0, 0.0, 1.0], [0.0, 2.0, -1.0]];
        let i = invert_affine(&m);
        let c = warp_cs(warp_cs([3.0, 4.0, 5.0], &m), &i);
        assert!((c[0] - 3.0).abs() < 1e-12 && (c[1] - 4.0).abs() < 1e-12 && (c[2] - 5.0).abs() < 1e-12);
        let mats = vec![m; 4];
        let k = compose(&mats, 0, 2);
        assert_eq!(k, [[4.0, 0.0, 3.0], [0.0, 4.0, -3.0]]);
    }
}
