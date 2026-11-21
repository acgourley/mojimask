//! The scan's boundaries with the pictures: everything it learns from pixels comes through these traits, so the rules
//! themselves are pure and testable. Real implementations come from mm-motion (motion) and mm-analysis (faces, bodies,
//! depth); [`replay`] answers from a recording of a Python run (`tools/parity/scan_record.py`) and panics on a call
//! that wasn't recorded, so a parity run can't silently drift.
use crate::geom::Cs;
use crate::np::M23;
use mm_core::pyjson::{float_repr, Py};
use std::collections::HashMap;
use std::path::Path;

/// track._flow_measure: median corner flow in the core of box c, frame t0 -> t1: (dx, dy, n good, spread px).
#[derive(Clone, Copy, Debug)]
pub struct FlowM {
    pub dx: f64,
    pub dy: f64,
    pub n: i64,
    pub spread: f64,
}

/// How a head moved between two frames, and whether it can be followed (follow.face_motion, Clip.trackable), plus the
/// follower's "still looks the same" test (follow.ncc of two 32x32 patches).
///
/// Motion and face evidence take `&self` and are `Sync`: the join's follows run in threads, as join.py's do (each
/// tracklet's two follows are independent of every other's). An implementation answers each call from its arguments
/// alone (caches are keyed by them), so the order calls come in never changes an answer.
pub trait Motion: Send + Sync {
    fn flow(&self, t0: i64, t1: i64, c: &Cs) -> Option<FlowM>;
    /// (M, kind): M the similarity taking box c's head in frame t0 to t1 when kind is "ok"; kind "ok", "no texture",
    /// "lost" or "incoherent".
    fn face_motion(&self, t0: i64, t1: i64, c: &Cs) -> (Option<M23>, String);
    /// Share of corners over the whole picture that track a -> b both ways.
    fn trackable(&self, a: i64, b: i64) -> f64;
    /// follow.ncc(patch(frame t, box c), patch(frame ta, box ca)).
    fn looks(&self, t: i64, c: &Cs, ta: i64, ca: &Cs) -> f64;
}

/// One face measurement (faces.face): landmark score, yaw, eye distance / width, face height px, SFace embedding
/// (frontal faces only).
#[derive(Clone, Debug)]
pub struct FaceM {
    pub score: f64,
    pub yaw: f64,
    pub eyes: f64,
    pub px: f64,
    pub emb: Option<Vec<f32>>,
}

pub trait FaceEvidence: Send + Sync {
    /// follow.face_near: (score, head box) of the most confident face near box c's centre at frame t, else (0, None).
    fn face_near(&self, t: i64, c: &Cs) -> (f64, Option<Cs>);
    /// faces.face: the measurement for detection c at frame t, or None.
    fn face(&self, t: i64, c: &Cs) -> Option<FaceM>;
}

pub trait Appearance: Send + Sync {
    /// appearance.reid_embed of appearance.crop(frame t, body box): the L2-normalised OSNet feature.
    fn body(&self, t: i64, bx: &[f64; 4]) -> Vec<f32>;
    /// A group's crops in one call (questions._looks runs appearance.reid_embed on them as one batch, and ONNX Runtime's
    /// CPU results depend on the batch a crop runs in). Default: each crop alone.
    fn bodies(&self, crops: &[(i64, [f64; 4])]) -> Vec<Vec<f32>> {
        crops.iter().map(|(t, bx)| self.body(*t, bx)).collect()
    }
}

pub trait Depth: Send + Sync {
    /// depth.Depth.near(t, c, f).
    fn near(&mut self, t: i64, c: &Cs, f: f64) -> f64;
    fn name(&self) -> &str;
    /// depth.MODELS[name]: (front, alone, apart).
    fn margins(&self) -> (f64, f64, f64);
}

/// Everything the scan reads from pixels.
pub struct Evidence {
    pub motion: Box<dyn Motion>,
    pub faces: Box<dyn FaceEvidence>,
    pub appearance: Box<dyn Appearance>,
    pub depth: Option<Box<dyn Depth>>,
}

/// The recording's key for a call: its arguments as float64s, Python's repr, as json.dumps writes a list.
pub fn key(vals: &[f64]) -> String {
    let parts: Vec<String> = vals.iter().map(|v| float_repr(*v)).collect();
    format!("[{}]", parts.join(", "))
}

fn load(rec: &Path, f: &str) -> Result<HashMap<String, Py>, String> {
    let p = mm_core::pyjson::load(&rec.join(format!("calls_{f}.json")))?;
    Ok(p.as_dict().ok_or("calls: not a dict")?.iter().map(|(k, v)| (k.to_string(), v.clone())).collect())
}

/// Which recorded calls a replay has answered, by boundary: a parity run checks it used every one (the same control
/// flow as Python's, not only the same outputs).
#[derive(Clone, Default)]
pub struct Usage(pub std::sync::Arc<std::sync::Mutex<HashMap<&'static str, (usize, std::collections::HashSet<String>)>>>);

impl Usage {
    fn hit(&self, what: &'static str, total: usize, k: String) {
        let mut m = self.0.lock().unwrap();
        let e = m.entry(what).or_insert((total, Default::default()));
        e.1.insert(k);
    }
    /// {boundary: (recorded, used)}
    pub fn report(&self) -> Vec<(&'static str, usize, usize)> {
        let mut v: Vec<_> = self.0.lock().unwrap().iter().map(|(k, (n, u))| (*k, *n, u.len())).collect();
        v.sort();
        v
    }
}

fn get<'a>(m: &'a HashMap<String, Py>, usage: &Usage, what: &'static str, k: &[f64]) -> &'a Py {
    let s = key(k);
    let v = m.get(&s).unwrap_or_else(|| panic!("replay: {what}{s} wasn't recorded (the Rust scan asked something Python didn't)"));
    usage.hit(what, m.len(), s);
    v
}

fn cs(v: &Py) -> Cs {
    let l = v.as_list().unwrap();
    [l[0].as_f64().unwrap(), l[1].as_f64().unwrap(), l[2].as_f64().unwrap()]
}

pub(crate) struct ReplayMotion {
    usage: Usage,
    flow: HashMap<String, Py>,
    motion: HashMap<String, Py>,
    trackable: HashMap<String, Py>,
    looks: HashMap<String, Py>,
}
pub(crate) struct ReplayFaces {
    usage: Usage,
    face_near: HashMap<String, Py>,
    face: HashMap<String, Py>,
}
pub(crate) struct ReplayBody {
    usage: Usage,
    body: HashMap<String, Py>,
}
pub(crate) struct ReplayDepth {
    usage: Usage,
    depth: HashMap<String, Py>,
    name: String,
}

impl Motion for ReplayMotion {
    fn flow(&self, t0: i64, t1: i64, c: &Cs) -> Option<FlowM> {
        let v = get(&self.flow, &self.usage, "flow", &[t0 as f64, t1 as f64, c[0], c[1], c[2]]);
        let l = v.as_list()?;
        Some(FlowM { dx: l[0].as_f64().unwrap(), dy: l[1].as_f64().unwrap(), n: l[2].as_i64().unwrap(), spread: l[3].as_f64().unwrap() })
    }
    fn face_motion(&self, t0: i64, t1: i64, c: &Cs) -> (Option<M23>, String) {
        let v = get(&self.motion, &self.usage, "motion", &[t0 as f64, t1 as f64, c[0], c[1], c[2]]).as_list().unwrap();
        let m = v[0].as_list().map(|l| {
            let f: Vec<f64> = l.iter().map(|x| x.as_f64().unwrap()).collect();
            [[f[0], f[1], f[2]], [f[3], f[4], f[5]]]
        });
        (m, v[1].as_str().unwrap().to_string())
    }
    fn trackable(&self, a: i64, b: i64) -> f64 {
        get(&self.trackable, &self.usage, "trackable", &[a as f64, b as f64]).as_f64().unwrap()
    }
    fn looks(&self, t: i64, c: &Cs, ta: i64, ca: &Cs) -> f64 {
        get(&self.looks, &self.usage, "looks", &[t as f64, c[0], c[1], c[2], ta as f64, ca[0], ca[1], ca[2]]).as_f64().unwrap()
    }
}

impl FaceEvidence for ReplayFaces {
    fn face_near(&self, t: i64, c: &Cs) -> (f64, Option<Cs>) {
        let v = get(&self.face_near, &self.usage, "face_near", &[t as f64, c[0], c[1], c[2]]).as_list().unwrap();
        (v[0].as_f64().unwrap(), if v[1].is_null() { None } else { Some(cs(&v[1])) })
    }
    fn face(&self, t: i64, c: &Cs) -> Option<FaceM> {
        let v = get(&self.face, &self.usage, "face", &[t as f64, c[0], c[1], c[2]]);
        if v.is_null() {
            return None;
        }
        let f = |k: &str| v.get(k).as_f64().unwrap();
        Some(FaceM {
            score: f("score"),
            yaw: f("yaw"),
            eyes: f("eyes"),
            px: f("px"),
            emb: v.get("emb").as_list().map(|l| l.iter().map(|x| x.as_f64().unwrap() as f32).collect()),
        })
    }
}

impl Appearance for ReplayBody {
    fn body(&self, t: i64, bx: &[f64; 4]) -> Vec<f32> {
        get(&self.body, &self.usage, "body", &[t as f64, bx[0], bx[1], bx[2], bx[3]]).as_list().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
    }
}

impl Depth for ReplayDepth {
    fn near(&mut self, t: i64, c: &Cs, f: f64) -> f64 {
        get(&self.depth, &self.usage, "depth", &[t as f64, c[0], c[1], c[2], f]).as_f64().unwrap()
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn margins(&self) -> (f64, f64, f64) {
        match self.name.as_str() {
            "anything-small" | "anything-base" => (0.25, 0.5, 0.15),
            "pro" => (0.75, 1.0, 0.5),
            n => panic!("depth model {n}"),
        }
    }
}

/// Evidence answered from a recording (rec/calls_*.json), with a handle on which recorded calls were used. depth: the
/// depth model the recording ran with, or None.
pub fn replay_counted(rec: &Path, depth: Option<&str>) -> Result<(Evidence, Usage), String> {
    let u = Usage::default();
    let ev = Evidence {
        motion: Box::new(ReplayMotion {
            usage: u.clone(),
            flow: load(rec, "flow")?,
            motion: load(rec, "motion")?,
            trackable: load(rec, "trackable")?,
            looks: load(rec, "looks")?,
        }),
        faces: Box::new(ReplayFaces { usage: u.clone(), face_near: load(rec, "face_near")?, face: load(rec, "face")? }),
        appearance: Box::new(ReplayBody { usage: u.clone(), body: load(rec, "body")? }),
        depth: match depth {
            Some(n) => Some(Box::new(ReplayDepth { usage: u.clone(), depth: load(rec, "depth")?, name: n.to_string() })),
            None => None,
        },
    };
    Ok((ev, u))
}
