//! Parity checks of the Rust evidence against a recording of a Python scan (tools/parity/scan_record.py): each
//! recorded call asked again of [`crate::evidence`], so a difference in the decide step's output can be told apart
//! as the evidence's (a value differs here) or the order of calls (every value agrees here).
use crate::evidence as ev;
use crate::scan::Options;
use mm_core::pyjson::Py;
use mm_scan::evidence::{Appearance, Depth, FaceEvidence, Motion};
use serde_json::json;
use std::path::Path;
use std::sync::Arc;

fn calls(rec: &Path, f: &str) -> Vec<(Vec<f64>, Py)> {
    let p = rec.join(format!("calls_{f}.json"));
    if !p.exists() {
        return vec![];
    }
    let d = mm_core::pyjson::load(&p).unwrap();
    d.as_dict().unwrap().iter().map(|(k, v)| {
        let key: Vec<f64> = Py::parse(k).unwrap().as_list().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        (key, v.clone())
    }).collect()
}

#[derive(Default)]
struct Tally {
    n: usize,
    exact: usize,
    max_abs: f64,
    first_bad: Vec<String>,
}

impl Tally {
    fn add(&mut self, key: &[f64], py: &[f64], rs: &[f64], structural: bool) {
        self.n += 1;
        let d = if py.len() == rs.len() { py.iter().zip(rs).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) } else { f64::INFINITY };
        if structural && d == 0.0 && py.len() == rs.len() && py.iter().zip(rs).all(|(a, b)| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())) {
            self.exact += 1;
        } else {
            self.max_abs = self.max_abs.max(if d.is_nan() { 0.0 } else { d });
            if self.first_bad.len() < 5 {
                self.first_bad.push(format!("{key:?}: py {:?} rs {:?}", &py[..py.len().min(6)], &rs[..rs.len().min(6)]));
            }
        }
    }
    fn json(&self) -> serde_json::Value {
        json!({"calls": self.n, "exact": self.exact, "max_abs_diff": self.max_abs, "first_differing": self.first_bad})
    }
}

fn fl(v: &Py) -> Vec<f64> {
    match v {
        Py::List(l) => l.iter().flat_map(fl).collect(),
        Py::Null => vec![],
        x => vec![x.as_f64().unwrap()],
    }
}

pub fn evidence(rec: &Path, proj: &Path, o: &Options, only: Option<&str>) -> serde_json::Value {
    let frames = Arc::new(ev::Frames::for_project(proj).expect("the project's meta.json"));
    let models = ev::models(&o.models.join("onnx"), o.analysis_cpu, o.cache.as_ref().map(|c| c.join("analysis")));
    let faces = ev::RealFaces::new(frames.clone(), models.clone(), None);
    let body = ev::RealBody { frames: frames.clone(), models: models.clone() };
    let mut out = serde_json::Map::new();
    let want = |k: &str| only.is_none_or(|o| o.split(',').any(|x| x == k));
    if want("face") {
        let t0 = std::time::Instant::now();
        let mut t = Tally::default();
        let mut emb = Tally::default();
        for (k, v) in calls(rec, "face") {
            let r = faces.face(k[0] as i64, &[k[1], k[2], k[3]]);
            let (py_some, rs_some) = (!v.is_null(), r.is_some());
            if py_some != rs_some {
                t.add(&k, &[py_some as i64 as f64], &[rs_some as i64 as f64], true);
                continue;
            }
            let Some(r) = r else { t.add(&k, &[], &[], true); continue };
            let p = [v.get("score").as_f64().unwrap(), v.get("yaw").as_f64().unwrap(), v.get("eyes").as_f64().unwrap(), v.get("px").as_f64().unwrap()];
            t.add(&k, &p, &[r.score, r.yaw, r.eyes, r.px], true);
            let pe = fl(v.get("emb"));
            let re: Vec<f64> = r.emb.as_ref().map(|e| e.iter().map(|x| *x as f64).collect()).unwrap_or_default();
            if !pe.is_empty() || !re.is_empty() {
                emb.add(&k, &pe, &re, true);
            }
        }
        out.insert("face (score, yaw, eyes, px)".into(), t.json());
        out.insert("face emb".into(), emb.json());
        out.insert("face seconds".into(), json!(t0.elapsed().as_secs_f64()));
    }
    if want("face_near") {
        let t0 = std::time::Instant::now();
        let mut t = Tally::default();
        for (k, v) in calls(rec, "face_near") {
            let (s, h) = faces.face_near(k[0] as i64, &[k[1], k[2], k[3]]);
            let mut rs = vec![s];
            if let Some(h) = h {
                rs.extend(h);
            }
            t.add(&k, &fl(&v), &rs, true);
        }
        out.insert("face_near".into(), t.json());
        out.insert("face_near seconds".into(), json!(t0.elapsed().as_secs_f64()));
    }
    if want("body") {
        let mut t = Tally::default();
        for (k, v) in calls(rec, "body") {
            let r = body.body(k[0] as i64, &[k[1], k[2], k[3], k[4]]);
            t.add(&k, &fl(&v), &r.iter().map(|x| *x as f64).collect::<Vec<_>>(), true);
        }
        out.insert("body".into(), t.json());
    }
    if want("depth") {
        let c = calls(rec, "depth");
        if !c.is_empty() {
            let f0 = frames.get(0);
            match ev::RealDepth::open(proj, o.depth.as_deref().unwrap_or("anything-small"), (f0.rows, f0.cols)) {
                Ok(mut d) => {
                    let mut t = Tally::default();
                    for (k, v) in c {
                        let r = d.near(k[0] as i64, &[k[1], k[2], k[3]], k[4]);
                        t.add(&k, &fl(&v), &[r], true);
                    }
                    out.insert("depth".into(), t.json());
                }
                Err(e) => {
                    out.insert("depth".into(), json!(format!("{e}")));
                }
            }
        }
    }
    // the follower's motion from mm-motion (evidence::NativeMotion; asked only with --only naming them): `looks` is
    // ported exactly (follow.patch + ncc); flow, face_motion and trackable are our own motion, not cv2's, so they are
    // compared, not required to agree
    let only_named = |k: &str| only.is_some_and(|o| o.split(',').any(|x| x == k));
    let nm = ev::NativeMotion::new(frames.clone());
    if only_named("looks") {
        let mut t = Tally::default();
        for (k, v) in calls(rec, "looks") {
            let r = nm.looks(k[0] as i64, &[k[1], k[2], k[3]], k[4] as i64, &[k[5], k[6], k[7]]);
            t.add(&k, &fl(&v), &[r], true);
        }
        out.insert("looks".into(), t.json());
    }
    if only_named("motion") {
        let (mut n, mut same, mut kinds) = (0usize, 0usize, std::collections::BTreeMap::<String, usize>::new());
        for (k, v) in calls(rec, "motion") {
            let (_, kind) = nm.face_motion(k[0] as i64, k[1] as i64, &[k[2], k[3], k[4]]);
            let pk = v.as_list().unwrap()[1].as_str().unwrap().to_string();
            n += 1;
            same += (pk == kind) as usize;
            *kinds.entry(format!("{pk} -> {kind}")).or_default() += 1;
        }
        out.insert("motion kinds".into(), json!({"calls": n, "same kind": same, "python -> ours": kinds}));
    }
    if only_named("flow") {
        let (mut n, mut both, mut none_py, mut none_rs, mut errs) = (0usize, 0usize, 0usize, 0usize, vec![]);
        for (k, v) in calls(rec, "flow") {
            n += 1;
            let r = nm.flow(k[0] as i64, k[1] as i64, &[k[2], k[3], k[4]]);
            match (v.as_list(), r) {
                (Some(p), Some(r)) => {
                    both += 1;
                    errs.push(((p[0].as_f64().unwrap() - r.dx).hypot(p[1].as_f64().unwrap() - r.dy)) / k[4]);
                }
                (None, Some(_)) => none_py += 1,
                (Some(_), None) => none_rs += 1,
                _ => {}
            }
        }
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |f: f64| errs.get(((errs.len() as f64 - 1.0) * f) as usize).cloned().unwrap_or(0.0);
        out.insert("flow".into(), json!({"calls": n, "both moved": both, "none in python only": none_py, "none in ours only": none_rs,
            "move difference / head size: median": q(0.5), "p90": q(0.9), "max": q(1.0)}));
    }
    if only_named("trackable") {
        let mut d = vec![];
        for (k, v) in calls(rec, "trackable") {
            d.push((v.as_f64().unwrap(), nm.trackable(k[0] as i64, k[1] as i64)));
        }
        let lo = |i: usize| d.iter().map(|x| if i == 0 { x.0 } else { x.1 }).fold(f64::INFINITY, f64::min);
        out.insert("trackable".into(), json!({"calls": d.len(), "python min": lo(0), "ours min": lo(1),
            "max |diff|": d.iter().map(|x| (x.0 - x.1).abs()).fold(0.0, f64::max)}));
    }
    serde_json::Value::Object(out)
}
