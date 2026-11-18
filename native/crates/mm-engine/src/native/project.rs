//! One open project: review_server.py's routes, in Rust.
//!
//! - **The single writer.** [`Proj::doc`] (mm_core::edits::Project) owns edits.json, facts.json and the undo history;
//!   the project's `.server.lock` is held while it's open (a Python review server on the same project is refused, and
//!   this refuses a project one has open). Every change answers doc_reply: the new edits, revision, can undo / redo,
//!   and what they resolve to ([`Proj::state`]: review_server.state, through mm-records' track.resolve).
//! - **Decisions** are the scan's records plus the "whose face is this?" questions for boxes you made
//!   ([`super::whose`]), with their answers as the facts stand.
//! - **Covers you place** are followed ([`super::follow`]): flow from Python's cv2 (motion `python`, review_server's
//!   flow_step exactly) or mm-motion's head motion (motion `native`); the face evidence they snap to is ported.
//! - **Draw records** ([`Proj::records_json`], `/api/native/records`): mm-records, kept in step with the edits.
//! - Pictures: the frames (`/api/frames/NNNNN.jpg`), the project's thumbnail (drawn by the app's own drawer, as
//!   `/api/thumb` does), the portrait's picture.
//!
//! Not here: the portrait photo's preparation
//! (`POST /api/portrait`), Export labels' download link; exports are the app's own.
use super::{follow, now, whose, Fail, Reply, Res, Server};
use anyhow::{anyhow, Result};
use mm_core::edits::{Follow, Project};
use mm_core::pyjson::{py_round, Dict, Py};
use mm_records::resolve::{self as R, Item, Track};
use mm_records::Records;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub const SCALE: f64 = 1.25;
pub const PERSON_KINDS: [&str; 3] = ["scan", "yours", "short"];

pub struct Proj {
    pub id: String,
    pub work: PathBuf,
    pub meta: Value,
    _lock: std::fs::File,
    pub doc: Mutex<Project>,
    pub recs: Mutex<Records>,
    pub n: usize,
    pub w: usize,
    pub h: usize,
    pub fps: f64,
    pub photo: bool,
    pub auto: Arc<Vec<Track>>,
    pub mats: Arc<Vec<R::M23>>,
    pub poses: Arc<Vec<Vec<Vec<f64>>>>,
    /// the scan's numbered people (proto_tracks.json tracks with "real")
    pub real: HashSet<String>,
    follow: Mutex<Option<Box<dyn Follow + Send>>>,
    follow_cfg: follow::Cfg,
    pub looks: whose::Looks,
    yours: Mutex<Option<((i64, Vec<i64>), Vec<Py>)>>,
    depth_on: bool,
    dev: bool,
}

fn j(p: &Py) -> Value {
    p.to_serde()
}

fn r1(x: f64) -> Value {
    json!(py_round(x, 1))
}

pub fn piece_kind(real: &HashSet<String>, piece: &str) -> &'static str {
    if piece.starts_with('m') || piece.contains('~') {
        "yours"
    } else if real.contains(piece) {
        "scan"
    } else {
        "short"
    }
}

/// review_server.people_of: every person drawn in pf, most important first.
pub fn people_of(real: &HashSet<String>, pf: &[Vec<Item>]) -> Vec<Value> {
    struct P {
        tid: String,
        pieces: Vec<String>,
        frames: i64,
        kind: usize,
    }
    let mut by: Vec<P> = vec![];
    let mut at: HashMap<String, usize> = HashMap::new();
    for f in pf {
        for x in f {
            let i = *at.entry(x.tid.clone()).or_insert_with(|| {
                by.push(P { tid: x.tid.clone(), pieces: vec![], frames: 0, kind: 2 });
                by.len() - 1
            });
            let p = &mut by[i];
            if !p.pieces.contains(&x.piece) {
                p.pieces.push(x.piece.clone());
            }
            p.frames += 1;
            let k = PERSON_KINDS.iter().position(|k| *k == piece_kind(real, &x.piece)).unwrap();
            if k < p.kind {
                p.kind = k;
            }
        }
    }
    by.sort_by_key(|p| (p.kind, -p.frames)); // stable: first seen first among equals
    by.into_iter()
        .map(|mut p| {
            p.pieces.sort_by(|a, b| (a.chars().count(), a).cmp(&(b.chars().count(), b)));
            json!({"tid": p.tid, "pieces": p.pieces, "frames": p.frames, "kind": PERSON_KINDS[p.kind]})
        })
        .collect()
}

fn with_cuts(edits: &Py) -> Py {
    let mut e = edits.clone();
    if let Some(d) = e.as_dict_mut() {
        if !d.contains("cuts") {
            d.insert("cuts", Py::Dict(Dict::new()));
        }
    }
    e
}

impl Proj {
    pub fn open(srv: &Server, id: &str, work: &Path) -> Result<Proj> {
        // a holder may be on its way out (the app quit and opened again at once, or `mmctl restart`, which starts the
        // new process before the old one exits): give it a few seconds to let go before saying it's open elsewhere
        let t0 = std::time::Instant::now();
        let lock = loop {
            match mm_core::projects::take_lock(work, Some(srv.port)) {
                Ok(l) => break l,
                Err(e) if t0.elapsed() >= std::time::Duration::from_secs(5) => return Err(anyhow!(e)),
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(100)),
            }
        };
        let meta: Value = serde_json::from_slice(&std::fs::read(work.join("meta.json"))?)?;
        let doc = Project::open(work, mm_core::edits::system_clock()).map_err(|e| anyhow!(e))?;
        let opts = mm_records::records::Options {
            depth: srv.cfg.depth.clone(),
            web: srv.cfg.root.join("web"),
            scale: SCALE,
            light_now: true,
            models: Some(srv.cfg.scan.models.clone()),
        };
        let mut recs = Records::open(work, opts)?;
        recs.set_edits(doc.edits.clone(), doc.rev);
        let p = &recs.proj;
        let proto = mm_core::pyjson::load(&work.join("proto_tracks.json")).map_err(|e| anyhow!(e))?;
        let real = proto.get("tracks").as_dict().map(|d| d.iter().filter(|(_, t)| t.get("real").truthy()).map(|(k, _)| k.to_string()).collect()).unwrap_or_default();
        let depth_on = srv.cfg.depth.as_ref().is_some_and(|d| crate::depth::cached(work, d));
        let me = Proj {
            id: id.to_string(),
            work: work.to_path_buf(),
            n: p.n,
            w: p.w,
            h: p.h,
            fps: p.fps,
            photo: meta["image"].as_bool().unwrap_or(false),
            auto: Arc::new(p.auto.clone()),
            mats: Arc::new(p.mats.clone()),
            poses: Arc::new(p.poses.clone()),
            meta,
            _lock: lock,
            doc: Mutex::new(doc),
            recs: Mutex::new(recs),
            real,
            follow: Mutex::new(None),
            follow_cfg: follow::Cfg::new(&srv.cfg.scan, work),
            looks: whose::Looks::default(),
            yours: Mutex::new(None),
            depth_on,
            dev: srv.cfg.dev,
        };
        // review_server's start: re-follow covers made by an older propagate()
        {
            let mut d = me.doc.lock().unwrap();
            let needs = d.edits.get("manual").as_dict().is_some_and(|m| m.iter().any(|(_, c)| c.get("pv") != &Py::Int(mm_core::edits::PROP_VERSION) && c.get("keys").truthy()));
            if needs {
                let mut f = me.follower();
                match d.refollow_covers(f.as_mut()) {
                    Ok(n) if n > 0 => super::log(&format!("re-followed {n} covers (propagate v{})", mm_core::edits::PROP_VERSION)),
                    Err(e) => super::log(&format!("refollow: {}", e.error)),
                    _ => {}
                }
                *me.follow.lock().unwrap() = Some(f);
                let (e, r) = (d.edits.clone(), d.rev);
                me.recs.lock().unwrap().set_edits(e, r);
            }
        }
        Ok(me)
    }

    fn follower(&self) -> Box<dyn Follow + Send> {
        if let Some(f) = self.follow.lock().unwrap().take() {
            return f;
        }
        follow::make(&self.follow_cfg, self.auto.clone(), self.poses.clone(), self.n, self.w, self.h)
    }

    // -------------------------------------------------------------------------------------------- state
    /// review_server.state(edits): resolved covers per frame, the hidden boxes, the people.
    pub fn state(&self, edits: &Py) -> Result<Value> {
        let links = R::links_for(edits);
        let pf = R::resolve(&self.auto, &with_cuts(edits), &self.mats, self.n, &links, true)?;
        let shown = R::shown_pieces(edits);
        let resolved: Vec<Value> = pf
            .iter()
            .map(|f| Value::Array(f.iter().map(|x| json!([x.tid, r1(x.c[0]), r1(x.c[1]), r1(x.c[2]), x.kind, x.piece, shown.contains(&x.piece)])).collect()))
            .collect();
        let mut gone: HashSet<String> = edits.get("not_face").as_dict().map(|d| d.keys().map(String::from).collect()).unwrap_or_default();
        gone.extend(edits.get("not_real").as_list().map(|l| l.iter().map(Py::py_str).collect::<Vec<_>>()).unwrap_or_default());
        let mut hidden = vec![];
        if !gone.is_empty() {
            let mut e2 = with_cuts(edits);
            let d = e2.as_dict_mut().unwrap();
            d.insert("not_face", Py::Dict(Dict::new()));
            d.insert("not_real", Py::List(vec![]));
            let pf2 = R::resolve(&self.auto, &e2, &self.mats, self.n, &links, false)?;
            for (t, f) in pf2.iter().enumerate() {
                for x in f {
                    if gone.contains(&x.piece) {
                        hidden.push(json!([t, x.tid, r1(x.c[0]), r1(x.c[1]), r1(x.c[2]), x.kind, x.piece]));
                    }
                }
            }
        }
        Ok(json!({"resolved": resolved, "hidden": hidden, "persons": people_of(&self.real, &pf)}))
    }

    pub fn state_now(&self) -> Result<Value> {
        let e = self.doc.lock().unwrap().edits.clone();
        self.state(&e)
    }

    /// doc_reply(extra): the edits, their revision, undo / redo, and what they resolve to.
    pub fn doc_reply(&self, extra: Value) -> Result<Value> {
        let (edits, rev, cu, cr) = {
            let d = self.doc.lock().unwrap();
            (d.edits.clone(), d.rev, !d.undo.is_empty(), !d.redo.is_empty())
        };
        let mut out = json!({"rev": rev, "edits": j(&edits), "can_undo": cu, "can_redo": cr});
        let o = out.as_object_mut().unwrap();
        if let Some(x) = extra.as_object() {
            for (k, v) in x {
                o.insert(k.clone(), v.clone());
            }
        }
        if let Some(s) = self.state(&edits)?.as_object() {
            for (k, v) in s {
                o.insert(k.clone(), v.clone());
            }
        }
        Ok(out)
    }

    /// After a change: records follow the edits.
    fn changed(&self) {
        let (e, r) = {
            let d = self.doc.lock().unwrap();
            (d.edits.clone(), d.rev)
        };
        self.recs.lock().unwrap().set_edits(e, r);
    }

    // -------------------------------------------------------------------------------------------- decisions
    /// your_box_questions_now: kept until the edits or the facts change.
    pub fn yours(&self, srv: &Server) -> Result<Vec<Py>> {
        let (edits, key) = {
            let d = self.doc.lock().unwrap();
            (d.edits.clone(), (d.rev, d.facts().iter().map(mm_core::facts::fid).collect::<Vec<_>>()))
        };
        if let Some((k, v)) = self.yours.lock().unwrap().as_ref() {
            if *k == key {
                return Ok(v.clone());
            }
        }
        let facts = self.doc.lock().unwrap().facts().to_vec();
        let out = whose::questions(srv, self, &edits, &facts)?;
        *self.yours.lock().unwrap() = Some((key, out.clone()));
        Ok(out)
    }

    fn decisions(&self, srv: &Server) -> Result<Value> {
        let recs = self.doc.lock().unwrap().decision_records().map_err(|e| anyhow!(e))?;
        let Some(mut ds) = recs else { return Ok(json!({"decisions": [], "missing": true})) };
        ds.extend(self.yours(srv)?);
        let d = self.doc.lock().unwrap();
        let a = mm_core::facts::answers(&ds, d.facts(), &d.hidden);
        Ok(json!({"decisions": a.iter().map(j).collect::<Vec<_>>()}))
    }

    // -------------------------------------------------------------------------------------------- routes
    pub fn handle(&self, srv: &Arc<Server>, method: &str, path: &str, q: &HashMap<String, String>, body: &Value) -> Res {
        if method == "GET" { self.get(srv, path, q) } else { self.post(srv, path, body) }
    }

    fn get(&self, srv: &Arc<Server>, path: &str, q: &HashMap<String, String>) -> Res {
        let clamp = |s: &str| -> std::result::Result<usize, Fail> {
            let t: i64 = s.split('.').next().unwrap_or("").parse().map_err(|_| Fail(400, format!("not a number: {s:?}")))?;
            Ok(t.clamp(0, self.n as i64 - 1) as usize)
        };
        match path {
            "/api/init" => {
                let extra = json!({"n": self.n, "w": self.w, "h": self.h, "fps": self.fps, "scale": SCALE, "frames": "/api/frames",
                                   "photo": self.photo, "notices": [], "dev": self.dev, "depth": self.depth_on});
                return Ok(Reply::json(&self.doc_reply(extra)?));
            }
            "/api/edits" => return Ok(Reply::json(&self.doc_reply(json!({}))?)),
            "/api/rev" => return Ok(Reply::json(&json!({"rev": self.doc.lock().unwrap().rev}))),
            "/api/render" => return Ok(Reply::json(&json!({"status": "idle", "progress": 0.0, "file": null, "path": null, "error": null}))),
            "/api/light" => {
                let mut r = self.recs.lock().unwrap();
                if !r.light_ready() {
                    return Ok(Reply::json(&json!({"ready": false, "progress": 0.0, "error": null})));
                }
                let l = r.lighting_json().map(|p| p.to_serde()).unwrap_or(json!({"ready": false, "progress": 0.0, "error": null}));
                return Ok(Reply::json(&l));
            }
            "/api/formats" => return Ok(Reply::json(&self.formats())),
            "/api/decisions" => return Ok(Reply::json(&self.decisions(srv)?)),
            "/api/export_labels" if self.dev => return Ok(Reply::json(&self.export_labels()?)),
            "/api/portrait" => {
                let edits = self.doc.lock().unwrap().edits.clone();
                let web = srv.cfg.root.join("web");
                let key = mm_records::records::key_for(mm_records::records::project_photo(&self.work, &edits));
                let path = mm_records::records::photo_path(&key, &web).ok_or_else(|| Fail(404, "no photo".into()))?;
                let png = mm_records::records::portrait_sprite_png(Path::new(&path), 120).map_err(|e| Fail(500, format!("{e:#}")))?;
                let mut r = Reply::bytes("image/png", png);
                r.cache = Some("no-store");
                return Ok(r);
            }
            "/api/native/setup" => return Ok(Reply::json(&self.setup(srv))),
            "/api/native/records" => {
                let lo: usize = q.get("from").and_then(|v| v.parse().ok()).unwrap_or(0);
                let hi: usize = q.get("to").and_then(|v| v.parse().ok()).unwrap_or(self.n);
                let yes = |k: &str| q.get(k).is_some_and(|v| v == "1" || v == "true");
                let ask = mm_records::records::Ask { style: q.get("style").cloned(), strict: yes("strict"), wait: yes("wait") };
                return Ok(Reply::json(&self.records_json(lo, hi, &ask)?));
            }
            _ => {}
        }
        if let Some(name) = path.strip_prefix("/api/frames/") {
            let ok = name.len() == 9 && name.ends_with(".jpg") && name[..5].bytes().all(|b| b.is_ascii_digit());
            let b = ok.then(|| std::fs::read(self.work.join("frames").join(name)).ok()).flatten().ok_or_else(|| Fail(404, "no such frame".into()))?;
            return Ok(Reply::bytes("image/jpeg", b));
        }
        if let Some(t) = path.strip_prefix("/api/pose/") {
            return Ok(Reply::json(&self.pose_frame(clamp(t)?)));
        }
        if let Some(name) = path.strip_prefix("/api/depth/") {
            // the editor's depth view (review_server: depth_ui.js, MOJIMASK_DEPTH on): the picture or the covers
            let no = || Fail(404, "no depth".into());
            let t = clamp(name.split('.').next().unwrap_or(""))?;
            let mut r = self.recs.lock().unwrap();
            if name.ends_with(".jpg") {
                return Ok(Reply::bytes("image/jpeg", r.depth_picture(t).ok_or_else(no)?));
            }
            return Ok(Reply::json(&r.depth_view(t).map_err(|e| Fail(500, format!("{e:#}")))?.ok_or_else(no)?));
        }
        if path == "/api/boxes" {
            let ids: Vec<&str> = q.get("ids").map(|s| s.split(',').collect()).unwrap_or_default();
            let mut out = serde_json::Map::new();
            for k in ids {
                if let Some(tr) = self.auto.iter().find(|t| t.id == k) {
                    let (Some(a), Some(b)) = (tr.obs.first(), tr.obs.last()) else {
                        out.insert(k.to_string(), json!({}));
                        continue;
                    };
                    let mut m = serde_json::Map::new();
                    for (t, c) in &tr.cover {
                        if a <= t && t <= b {
                            m.insert(t.to_string(), Value::Array(R::from_cs(c).iter().map(|v| r1(*v)).collect()));
                        }
                    }
                    out.insert(k.to_string(), Value::Object(m));
                }
            }
            return Ok(Reply::json(&Value::Object(out)));
        }
        Err(Fail(404, "not found".into()))
    }

    fn post(&self, srv: &Arc<Server>, path: &str, body: &Value) -> Res {
        let b = Py::from_serde(body);
        let int = |k: &str| -> std::result::Result<i64, Fail> {
            body[k].as_i64().or_else(|| body[k].as_f64().filter(|f| f.fract() == 0.0).map(|f| f as i64))
                .or_else(|| body[k].as_str().and_then(|s| s.trim().parse().ok()))
                .ok_or_else(|| Fail(400, format!("not a number: {}", body[k])))
        };
        let reply = match path {
            "/api/ops" => {
                self.doc.lock().unwrap().post_ops(&b)?;
                self.changed();
                self.doc_reply(json!({}))?
            }
            "/api/facts" => {
                self.doc.lock().unwrap().post_facts(&b)?;
                self.changed();
                self.doc_reply(json!({}))?
            }
            "/api/decisions" => {
                let extra = self.yours(srv)?;
                let rec = self.doc.lock().unwrap().decide_with(&b, &extra)?;
                self.changed();
                let mut r = self.doc_reply(json!({}))?;
                if let (Some(o), Some(x)) = (r.as_object_mut(), rec.to_serde().as_object()) {
                    for (k, v) in x {
                        o.insert(k.clone(), v.clone());
                    }
                }
                r
            }
            "/api/undo" | "/api/redo" => {
                let redo = path == "/api/redo";
                let l = self.doc.lock().unwrap().undo_last(redo)?;
                self.changed();
                self.doc_reply(json!({if redo { "redone" } else { "undone" }: l}))?
            }
            "/api/undo_correction" => {
                let (piece, t) = (body["piece"].as_str().map(String::from).unwrap_or_else(|| body["piece"].to_string()), int("t")?);
                let mut f = self.follower();
                let r = self.doc.lock().unwrap().undo_correction(&piece, t, f.as_mut());
                *self.follow.lock().unwrap() = Some(f);
                r?;
                self.changed();
                self.doc_reply(json!({}))?
            }
            "/api/split_cover" => {
                let (piece, t) = (body["piece"].as_str().map(String::from).unwrap_or_else(|| body["piece"].to_string()), int("t")?);
                let new = self.doc.lock().unwrap().split_cover(&piece, t)?;
                self.changed();
                self.doc_reply(json!({"new": new}))?
            }
            "/api/propagate" => {
                let t = int("t")?;
                let c: Vec<f64> = body["c"].as_array().map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default();
                if c.len() != 3 {
                    return Err(Fail(400, "c: [cx, cy, s]".into()));
                }
                if !(0 <= t && t < self.n as i64) || c[2] <= 0.0 {
                    return Err(Fail(400, "t: a frame, c: [cx, cy, s]".into()));
                }
                let mut f = self.follower();
                let r = self.doc.lock().unwrap().propagate(t, [c[0], c[1], c[2]], f.as_mut());
                *self.follow.lock().unwrap() = Some(f);
                r?.to_serde()
            }
            "/api/thumb" => self.thumb(srv, body)?,
            "/api/render" => return Err(Fail(400, "exports are made by the app".into())),
            _ => return Err(Fail(404, "not found".into())),
        };
        Ok(Reply::json(&reply))
    }

    // -------------------------------------------------------------------------------------------- pictures, records
    fn thumb(&self, srv: &Server, body: &Value) -> std::result::Result<Value, Fail> {
        let out = self.work.join("thumb.jpg");
        if body["if_missing"].as_bool().unwrap_or(false) && out.exists() {
            return Ok(json!({"ok": true}));
        }
        let t = body["t"].as_i64().unwrap_or(0).clamp(0, self.n as i64 - 1) as usize;
        let style = self.doc.lock().unwrap().edits.get("cover_style").as_str().unwrap_or("emoji").to_string();
        let style = if style.is_empty() { "emoji".into() } else { style };
        let drawer = srv.drawer.lock().unwrap().clone().ok_or_else(|| Fail(500, "no drawer".into()))?;
        let (w, h, bgr) = drawer(&self.id, t, &style).ok_or_else(|| Fail(404, "no such frame".into()))?;
        let mut im = mm_vision::Mat::from_vec(h, w, 3, bgr);
        let k = 400.0 / w.max(h) as f64;
        if k < 1.0 {
            let (nw, nh) = ((w as f64 * k).round_ties_even() as usize, (h as f64 * k).round_ties_even() as usize);
            im = mm_vision::resize_u8(&im, nw, nh, 0.0, 0.0, mm_vision::Interp::Area);
        }
        let tmp = self.work.join("thumb.jpg.tmp.jpg");
        std::fs::write(&tmp, mm_vision::jpeg::encode_bgr(&im.data, im.cols, im.rows, 85)).map_err(|e| Fail(500, e.to_string()))?;
        std::fs::rename(&tmp, &out).map_err(|e| Fail(500, e.to_string()))?;
        Ok(json!({"ok": true}))
    }

    /// project_server.setup(): the drawing constants and files.
    pub fn setup(&self, srv: &Server) -> Value {
        let web = srv.cfg.root.join("web");
        let brand = read_brand(&web).unwrap_or(json!({}));
        let font = web.join("fonts").join(format!("{}.woff2", brand["font"].as_str().unwrap_or("Nunito")));
        let (rev, style) = {
            let d = self.doc.lock().unwrap();
            (d.rev, d.edits.get("cover_style").as_str().filter(|s| !s.is_empty()).unwrap_or("emoji").to_string())
        };
        let light_ready = self.recs.lock().unwrap().light_ready();
        json!({"TEX": 384, "SLICES": 48, "SIGMA_S": 0.9, "SIGMA_R": 0.08, "FACE_ZONE": [0.40, 0.55], "FACE_DY": 0.08,
               "OCC_SOFT": mm_analysis::depth::OCC_SOFT, "CANVAS": 1.5, "eyes": web.join("emoji_eyes.json").to_string_lossy(),
               "brand": brand.clone(), "font_file": font.to_string_lossy(), "name": brand["name"], "W": self.w, "H": self.h,
               "n": self.n, "fps": self.fps, "rev": rev, "style": style, "depth": self.depth_on, "light_ready": light_ready,
               "portrait_photo": mm_records::records::photo_path(mm_records::records::PORTRAIT_KEY, &web)})
    }

    /// project_server.records(lo, hi, style, wait, strict), with its cache key.
    pub fn records_json(&self, lo: usize, hi: usize, ask: &mm_records::records::Ask) -> Result<Value> {
        let mut r = self.recs.lock().unwrap();
        let v = r.records(lo, hi, ask)?;
        let mut out = v.to_serde();
        let key = format!("{}|{}|{}|{}|{}", out["rev"], out["style"], ask.strict, out["light_ready"], ask.wait);
        out["key"] = json!(key);
        Ok(out)
    }

    /// Every frame's record, made here (no JSON text): (records, depth maps where a cover is cut, portrait photos by
    /// key, {key, rev, style, light_ready}).
    #[allow(clippy::type_complexity)]
    pub fn frame_records(&self, ask: &mm_records::records::Ask)
        -> Result<(Vec<mm_core::cover::FrameRecord>, Vec<Option<mm_records::records::Near>>, HashMap<String, String>, Value)> {
        let mut r = self.recs.lock().unwrap();
        let mut frs = Vec::with_capacity(self.n);
        let mut near = Vec::with_capacity(self.n);
        let mut photos = HashMap::new();
        for t in 0..self.n {
            let (p, nr) = r.record(t, ask)?;
            if let Some(d) = p.get("photos").as_dict() {
                for (k, v) in d.iter() {
                    photos.insert(k.to_string(), v.py_str());
                }
            }
            frs.push(serde_json::from_value(p.to_serde())?);
            near.push(nr);
        }
        let v = r.records(0, 0, ask)?.to_serde();
        let key = format!("{}|{}|{}|{}|{}", v["rev"], v["style"], ask.strict, v["light_ready"], ask.wait);
        Ok((frs, near, photos, json!({"key": key, "rev": v["rev"], "style": v["style"], "light_ready": v["light_ready"]})))
    }

    fn formats(&self) -> Value {
        let mut d = mm_media::formats::describe();
        let mut sizes = serde_json::Map::new();
        for p in mm_media::formats::PRESETS.iter() {
            let mut m = serde_json::Map::new();
            for (f, _) in mm_media::formats::FITS {
                let (w, h) = mm_media::formats::output_size(p.id, self.w as u32, self.h as u32, f, self.photo);
                m.insert(f.to_string(), json!([w, h]));
            }
            sizes.insert(p.id.to_string(), Value::Object(m));
        }
        let video = self.meta["video"].as_str().unwrap_or("").to_string();
        let vp = Path::new(&video);
        let inside = vp.starts_with(&self.work);
        let o = d.as_object_mut().unwrap();
        o.insert("sizes".into(), Value::Object(sizes));
        o.insert("fps".into(), json!(self.fps));
        o.insert("frames".into(), json!(self.n));
        o.insert("photo".into(), json!(self.photo));
        o.insert("seconds".into(), json!(self.n as f64 / self.fps));
        o.insert("name".into(), json!(vp.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default()));
        o.insert("video".into(), json!(video));
        o.insert("save_dir".into(), json!(if inside { String::new() } else { vp.parent().map(|p| p.to_string_lossy().to_string()).unwrap_or_default() }));
        d
    }

    /// review_server.pose_frame: skeletons for the editor's pose overlay.
    fn pose_frame(&self, t: usize) -> Value {
        use mm_analysis::pose_util as PU;
        let mut out = vec![];
        for p in self.poses.get(t).map(|v| v.as_slice()).unwrap_or(&[]) {
            if p.get(4).copied().unwrap_or(0.0) < 0.25 {
                continue;
            }
            let h = PU::head(p);
            out.push(json!({"body": null, "facing": PU::facing(p).name(),
                            "head": h.map(|h| vec![r1(h.0), r1(h.1), r1(h.2)]), "score": json!(py_round(p[4], 2)),
                            "kps": p.get(5..56.min(p.len())).unwrap_or(&[]).iter().map(|v| r1(*v)).collect::<Vec<_>>()}));
        }
        Value::Array(out)
    }

    /// review_server.export_labels (developer mode): clean face / not-face labels, written to labels_export.json.
    fn export_labels(&self) -> Result<Value> {
        let edits = self.doc.lock().unwrap().edits.clone();
        let links = R::links_for(&edits);
        let pf = R::resolve(&self.auto, &with_cuts(&edits), &self.mats, self.n, &links, true)?;
        let shown = R::shown_pieces(&edits);
        let people = edits.get("people");
        let mut deleted: Vec<String> = edits.get("deleted").as_list().map(|l| l.iter().map(Py::py_str).collect()).unwrap_or_default();
        deleted.extend(edits.get("not_face").as_dict().map(|d| d.keys().map(String::from).collect::<Vec<_>>()).unwrap_or_default());
        deleted.retain(|d| self.auto.iter().any(|t| t.id == *d));
        let (mut nf_total, mut f_total) = (0, 0);
        let mut frames = vec![];
        for (t, f) in pf.iter().enumerate() {
            let faces: Vec<Value> = f.iter().map(|x| {
                let name = x.tid.strip_prefix('@').map(|p| people.get(p).get("name").to_serde()).unwrap_or(Value::Null);
                json!({"person": x.tid, "name": name, "piece": x.piece, "box": R::from_cs(&x.c).iter().map(|v| r1(*v)).collect::<Vec<_>>(),
                       "kind": x.kind, "shown": shown.contains(&x.piece)})
            }).collect();
            let mut nf = vec![];
            for d in &deleted {
                let tr = self.auto.iter().find(|tr| tr.id == *d).unwrap();
                if tr.obs.contains(&(t as i64)) {
                    if let Some((_, c)) = tr.cover.iter().find(|(u, _)| *u == t as i64) {
                        nf.push(json!({"piece": d, "box": R::from_cs(c).iter().map(|v| r1(*v)).collect::<Vec<_>>()}));
                    }
                }
            }
            f_total += faces.len();
            nf_total += nf.len();
            frames.push(json!({"t": t, "faces": faces, "not_faces": nf}));
        }
        let doc = json!({"video": self.meta["video"], "w": self.w, "h": self.h, "fps": self.fps, "frames": frames});
        mm_core::pyjson::write_atomic(&self.work.join("labels_export.json"), &Py::from_serde(&doc).dumps())?;
        Ok(json!({"file": "api/labels_export", "faces": f_total, "not_faces": nf_total}))
    }
}

/// about.BRAND: the object in web/brand.js.
pub fn read_brand(web: &Path) -> Option<Value> {
    let src = std::fs::read_to_string(web.join("brand.js")).ok()?;
    let at = src.find("const BRAND")?;
    let open = at + src[at..].find('{')?;
    let mut de = serde_json::Deserializer::from_str(&src[open..]).into_iter::<Value>();
    de.next()?.ok()
}

#[allow(dead_code)]
fn _unused() -> f64 {
    now()
}
