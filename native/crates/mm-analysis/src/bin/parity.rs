//! mm-analysis against Python: the Rust half of native/tools/parity/analysis_*.py.
//!
//!   mm-analysis-parity <reid|appearance|depth-model|depth-encode|depth-maps|pose> <dir>
//!
//! Each command reads <dir>/manifest.json and the inputs Python wrote there (frames as .npy, decoded by cv2 so the
//! JPEG decoder is not under test), runs the Rust port, compares pixels with Python's arrays where they're files,
//! and writes <dir>/rust.json (numbers, per-call timings) for the Python script to compare.
use mm_analysis::{appearance, depth, faces, head_pose, models, pose_util, reid};
use mm_ml::rt::Ep;
use mm_vision::{npy, Mat};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn load_u8(p: &Path) -> Mat<u8> {
    match npy::read(p).unwrap_or_else(|e| panic!("{e}")) {
        npy::Arr::U8(s, d) => {
            let cn = if s.len() == 3 { s[2] } else { 1 };
            Mat::from_vec(s[0], s[1], cn, d)
        }
        _ => panic!("{}: not u8", p.display()),
    }
}

fn load_f32(p: &Path) -> (Vec<usize>, Vec<f32>) {
    match npy::read(p).unwrap_or_else(|e| panic!("{e}")) {
        npy::Arr::F32(s, d) => (s, d),
        npy::Arr::F64(s, d) => (s, d.into_iter().map(|v| v as f32).collect()),
        _ => panic!("{}: not float", p.display()),
    }
}

/// differing values between two u8 images (shape mismatch: -1)
fn diff_u8(a: &Mat<u8>, b: &Mat<u8>) -> i64 {
    if a.rows != b.rows || a.cols != b.cols || a.cn != b.cn {
        return -1;
    }
    a.data.iter().zip(&b.data).filter(|(x, y)| x != y).count() as i64
}

fn engines(man: &Value) -> models::Engines {
    match man["ep"].as_str().unwrap_or("cpu") {
        "cpu" => models::Engines::cpu(),
        _ => models::Engines::default_plan(),
    }
}

fn open_models(man: &Value) -> models::Models {
    let cache = man["cache"].as_str().map(PathBuf::from);
    models::Models::new(Path::new(man["models"].as_str().unwrap()), engines(man), cache)
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn c3(v: &Value) -> [f64; 3] {
    [v[0].as_f64().unwrap(), v[1].as_f64().unwrap(), v[2].as_f64().unwrap()]
}

fn reid_cmd(dir: &Path, man: &Value) -> Value {
    let m = open_models(man);
    let mut frames: HashMap<i64, Mat<u8>> = HashMap::new();
    let mut out = Vec::new();
    let mut times: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let py = dir.join("py");
    // warm-up: open the sessions outside the timings
    {
        let d = &man["dets"][0];
        let t = d[1].as_i64().unwrap();
        let f = load_u8(&dir.join(format!("frames/{t:05}.npy")));
        if let Some(cr) = reid::face_crop(&f, &c3(&d[2])) {
            reid::landmarks(&m, &cr.img, reid::LANDMARK_SIZE, 0.3, 0.4, None);
            reid::embed(&m, &reid::align(&cr.img, &[[40.0, 50.0], [70.0, 50.0], [56.0, 70.0], [42.0, 90.0], [70.0, 90.0]]));
        }
    }
    for d in man["dets"].as_array().unwrap() {
        let i = d[0].as_i64().unwrap();
        let t = d[1].as_i64().unwrap();
        let c = c3(&d[2]);
        let f = frames.entry(t).or_insert_with(|| load_u8(&dir.join(format!("frames/{t:05}.npy"))));
        let t0 = Instant::now();
        let full = faces::face(&m, f, &c);
        times.entry("face").or_default().push(ms(t0));
        let t0 = Instant::now();
        let cr = reid::face_crop(f, &c);
        times.entry("face_crop").or_default().push(ms(t0));
        let mut e = json!({"i": i});
        let Some(cr) = cr else {
            e["crop"] = Value::Null;
            out.push(e);
            continue;
        };
        let pc = py.join(format!("crop_{i}.npy"));
        e["crop_diff"] = json!(if pc.exists() { diff_u8(&cr.img, &load_u8(&pc)) } else { -2 });
        e["k"] = json!(cr.k);
        e["x0"] = json!(cr.x0);
        e["y0"] = json!(cr.y0);
        let t0 = Instant::now();
        let fs = reid::landmarks(&m, &cr.img, reid::LANDMARK_SIZE, 0.3, 0.4, None);
        times.entry("landmarks").or_default().push(ms(t0));
        e["boxes"] = json!(fs.boxes);
        e["kps"] = json!(fs.kps);
        let near = faces::nearest(&fs.boxes, cr.k, cr.x0, cr.y0, &c);
        e["nearest"] = json!(near);
        if let Some(j) = near {
            let n = faces::numbers(&fs.boxes[j], &fs.kps[j], cr.k);
            e["score"] = json!(n.score);
            e["yaw"] = json!(n.yaw);
            e["eyes"] = json!(n.eyes);
            e["px"] = json!(n.px);
            e["frontal"] = json!(faces::frontal(&n));
            e["matrix"] = json!(reid::align_matrix(&fs.kps[j]));
            let t0 = Instant::now();
            let al = reid::align(&cr.img, &fs.kps[j]);
            times.entry("align").or_default().push(ms(t0));
            let pa = py.join(format!("aligned_{i}.npy"));
            e["aligned_diff"] = json!(if pa.exists() { diff_u8(&al, &load_u8(&pa)) } else { -2 });
            let t0 = Instant::now();
            let emb = reid::embed(&m, &al);
            times.entry("embed").or_default().push(ms(t0));
            e["emb_raw"] = json!(emb);
            e["emb"] = json!(faces::normalise(&emb));
        }
        e["face_frontal"] = json!(full.as_ref().map(faces::frontal));
        e["face_found"] = json!(full.is_some());
        out.push(e);
    }
    json!({"dets": out, "times": times})
}

fn appearance_cmd(dir: &Path, man: &Value) -> Value {
    let m = open_models(man);
    let mut frames: HashMap<i64, Mat<u8>> = HashMap::new();
    let mut out = Vec::new();
    let mut times: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let mut first = true;
    for (bi, b) in man["batches"].as_array().unwrap().iter().enumerate() {
        let mut crops = Vec::new();
        let mut diffs = Vec::new();
        for (j, r) in b.as_array().unwrap().iter().enumerate() {
            let t = r[0].as_i64().unwrap();
            let bx = [r[1].as_f64().unwrap(), r[2].as_f64().unwrap(), r[3].as_f64().unwrap(), r[4].as_f64().unwrap()];
            let f = frames.entry(t).or_insert_with(|| load_u8(&dir.join(format!("frames/{t:05}.npy"))));
            let t0 = Instant::now();
            let cr = appearance::crop(f, &bx, 0.0);
            times.entry("crop").or_default().push(ms(t0));
            let p = dir.join(format!("py/body_{bi}_{j}.npy"));
            match cr {
                Some(c) => {
                    diffs.push(if p.exists() { diff_u8(&c, &load_u8(&p)) } else { -2 });
                    crops.push(c);
                }
                None => diffs.push(if p.exists() { -3 } else { 0 }),
            }
        }
        if first {
            appearance::reid_embed(&m, &crops); // session open
            first = false;
        }
        let t0 = Instant::now();
        let v = appearance::reid_embed(&m, &crops);
        times.entry("reid_embed").or_default().push(ms(t0));
        out.push(json!({"crop_diffs": diffs, "emb": v}));
    }
    json!({"batches": out, "times": times})
}

fn depth_model_cmd(dir: &Path, man: &Value) -> Value {
    let m = open_models(man);
    let path = PathBuf::from(man["model"].as_str().unwrap());
    let mut out = Vec::new();
    let mut times = Vec::new();
    for (n, t) in man["frames"].as_array().unwrap().iter().enumerate() {
        let t = t.as_i64().unwrap();
        let f = load_u8(&dir.join(format!("frames/{t:05}.npy")));
        let (x, h, w) = depth::onnx_input(&f);
        let (_, px) = load_f32(&dir.join(format!("py/din_{t}.npy")));
        let in_diff = x.iter().zip(&px).filter(|(a, b)| a.to_bits() != b.to_bits()).count() + x.len().abs_diff(px.len());
        if n == 0 {
            depth::run_onnx(&m, &path, &f, None);
        }
        let t0 = Instant::now();
        let d = depth::run_onnx(&m, &path, &f, None);
        times.push(ms(t0));
        npy::write(&dir.join(format!("rust/depth_{t}.npy")), &npy::Arr::F32(vec![d.rows, d.cols], d.data)).unwrap();
        out.push(json!({"t": t, "h": h, "w": w, "input_diff": in_diff}));
    }
    json!({"frames": out, "times": times})
}

fn depth_encode_cmd(dir: &Path, man: &Value) -> Value {
    let mut out = Vec::new();
    for t in man["frames"].as_array().unwrap() {
        let t = t.as_i64().unwrap();
        let (s, d) = load_f32(&dir.join(format!("py/depth_{t}.npy")));
        let e = depth::encode(&Mat::from_vec(s[0], s[1], 1, d));
        let q = load_u8(&dir.join(format!("py/q_{t}.npy")));
        let pq = Mat::from_vec(q.rows, q.cols, 1, q.data.clone());
        out.push(json!({"t": t, "q_diff": diff_u8(&e.q, &pq), "pct": e.pct, "lo": e.lo, "hi": e.hi}));
    }
    // the fit: decoded luma vs samples (Python's numbers in the manifest)
    let mut fits = Vec::new();
    if let Some(cases) = man["fit_cases"].as_array() {
        for c in cases {
            let x: Vec<f64> = c["x"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            let y: Vec<f64> = c["y"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            let (a, b) = depth::fit_line(&x, &y);
            fits.push(json!([a, b]));
        }
    }
    json!({"frames": out, "fits": fits})
}

fn obs_map(v: &Value) -> BTreeMap<i64, [f64; 3]> {
    v.as_object().unwrap().iter().map(|(k, c)| (k.parse().unwrap(), c3(c))).collect()
}

fn items_of(v: &Value) -> Vec<head_pose::Item> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| head_pose::Item {
            tid: x[0].as_str().map(String::from).unwrap_or_else(|| x[0].to_string()),
            c: c3(&x[1]),
            kind: x[2].as_str().unwrap().to_string(),
            piece: x.get(3).map(|p| p.as_str().map(String::from).unwrap_or_else(|| p.to_string())),
        })
        .collect()
}

struct NpyLuma(Vec<Mat<u8>>);
impl depth::LumaSource for NpyLuma {
    fn luma(&self, start: usize, count: Option<usize>) -> Vec<(usize, Mat<u8>)> {
        let end = count.map(|c| (start + c).min(self.0.len())).unwrap_or(self.0.len());
        (start..end).map(|i| (i, self.0[i].clone())).collect()
    }
}

fn depth_maps_cmd(dir: &Path, man: &Value) -> Value {
    let side = depth::Sidecar::from_json(&serde_json::from_str(&std::fs::read_to_string(man["sidecar"].as_str().unwrap()).unwrap()).unwrap());
    let luma = match npy::read(&dir.join("py/luma.npy")).unwrap() {
        npy::Arr::U8(s, d) => {
            let n = s[1] * s[2];
            (0..s[0]).map(|i| Mat::from_vec(s[1], s[2], 1, d[i * n..(i + 1) * n].to_vec())).collect::<Vec<_>>()
        }
        _ => panic!("luma"),
    };
    let fhw = (man["frame_hw"][0].as_u64().unwrap() as usize, man["frame_hw"][1].as_u64().unwrap() as usize);
    let t0 = Instant::now();
    let dp = depth::Depth::new(man["model_name"].as_str().unwrap(), &side, Box::new(NpyLuma(luma)), fhw);
    let t_maps = ms(t0);
    let (mh, mw) = (dp.maps[0].rows, dp.maps[0].cols);
    let mut all = Vec::new();
    for m in &dp.maps {
        all.extend_from_slice(&m.data);
    }
    npy::write(&dir.join("rust/maps.npy"), &npy::Arr::F32(vec![dp.maps.len(), mh, mw], all)).unwrap();
    // full(t) for a few frames
    for t in man["full_frames"].as_array().unwrap() {
        let t = t.as_u64().unwrap() as usize;
        let f = dp.full(t).unwrap();
        npy::write(&dir.join(format!("rust/full_{t}.npy")), &npy::Arr::F32(vec![f.rows, f.cols], f.data)).unwrap();
    }
    let t0 = Instant::now();
    let near: Vec<f64> = man["near"].as_array().unwrap().iter().map(|q| dp.near(q[0].as_i64().unwrap(), &c3(&q[1]))).collect();
    let t_near = ms(t0) / near.len().max(1) as f64;
    let gaps: Vec<f64> = man["gaps"].as_array().unwrap().iter().map(|g| dp.gap(&obs_map(&g[0]), &obs_map(&g[1]), 3)).collect();
    let face_at: Vec<f64> = man["face_at"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            let ts: Vec<i64> = c[0].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
            let v: Vec<f64> = c[1].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            depth::face_at(&ts, &v, c[2].as_i64().unwrap())
        })
        .collect();
    // Occlusion
    let obs: HashMap<String, BTreeMap<i64, [f64; 3]>> =
        man["occ_obs"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), obs_map(v))).collect();
    let occ = depth::Occlusion::new(&dp, obs);
    let shape = (fhw.0, fhw.1);
    let scale = man["occ_scale"].as_f64().unwrap();
    let mut plans = Vec::new();
    let mut t_plan = Vec::new();
    for fr in man["occ_frames"].as_array().unwrap() {
        let t = fr[0].as_i64().unwrap();
        let items = items_of(&fr[1]);
        let t0 = Instant::now();
        let ex = occ.explain(t, &items, scale, shape);
        t_plan.push(ms(t0));
        let nearness: Vec<f64> = items.iter().map(|it| occ.nearness(t, it.piece(), &it.c)).collect();
        plans.push(json!({"t": t, "explain": ex, "nearness": nearness.iter().map(|v| if v.is_finite() { json!(v) } else { json!(null) }).collect::<Vec<_>>()}));
    }
    // a few eroded maps
    for t in man["occ_maps"].as_array().unwrap() {
        let t = t.as_i64().unwrap();
        let m = occ.map(t);
        npy::write(&dir.join(format!("rust/occmap_{t}.npy")), &npy::Arr::F64(vec![m.h, m.w], m.d.clone())).unwrap();
    }
    let nan = |v: &Vec<f64>| v.iter().map(|x| if x.is_nan() { json!(null) } else { json!(x) }).collect::<Vec<_>>();
    json!({"near": nan(&near), "gaps": nan(&gaps), "face_at": nan(&face_at), "plans": plans,
           "times": {"maps_ms": t_maps, "near_ms": t_near, "explain_ms": t_plan}})
}

fn pose_cmd(_dir: &Path, man: &Value) -> Value {
    let rows: Vec<Vec<Vec<f64>>> = man["poses"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f.as_array().unwrap().iter().map(|p| p.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()).collect())
        .collect();
    let mut per_row = Vec::new();
    let t0 = Instant::now();
    for f in &rows {
        for p in f {
            let h = pose_util::head(p);
            let hb = pose_util::head_box(p);
            let (xy, q) = pose_util::pose_head(p, 40.0);
            let a = head_pose::raw_angles(p);
            per_row.push(json!({"head": h.map(|h| [h.0, h.1, h.2, h.3]), "facing": pose_util::facing(p).name(), "head_box": hb,
                                "pose_head": [xy[0], xy[1], q], "raw": [a.0, a.1, a.2, a.3]}));
        }
    }
    let t_rows = ms(t0);
    let mut pfh = Vec::new();
    let t0 = Instant::now();
    for q in man["pose_for_head"].as_array().unwrap() {
        let t = q[0].as_u64().unwrap() as usize;
        let r = pose_util::pose_for_head(&rows[t], &c3(&q[1]), 0.25);
        pfh.push(json!(r.map(|p| rows[t].iter().position(|x| x.as_slice() == p).unwrap())));
    }
    let t_pfh = ms(t0);
    let per_frame: Vec<Vec<head_pose::Item>> = man["per_frame"].as_array().unwrap().iter().map(items_of).collect();
    let t0 = Instant::now();
    let ang = head_pose::track_angles(&per_frame, &rows, 4.0, 0.15);
    let t_track = ms(t0);
    let ang: BTreeMap<String, BTreeMap<String, Value>> = ang
        .into_iter()
        .map(|(k, v)| (k, v.into_iter().map(|(t, a)| (t.to_string(), json!([a.0, a.1, a.2]))).collect()))
        .collect();
    let mirror: Vec<bool> = man["mirror"].as_array().unwrap().iter().map(|q| head_pose::mirror_for(q[0].as_str().unwrap(), q[1].as_i64().unwrap() as i32)).collect();
    json!({"rows": per_row, "pose_for_head": pfh, "track_angles": ang, "mirror": mirror,
           "times": {"rows_ms": t_rows, "pose_for_head_ms": t_pfh, "track_angles_ms": t_track}})
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (cmd, dir) = (a[1].as_str(), PathBuf::from(&a[2]));
    let man: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
    std::fs::create_dir_all(dir.join("rust")).unwrap();
    let _ = Ep::Cpu;
    let out = match cmd {
        "reid" => reid_cmd(&dir, &man),
        "appearance" => appearance_cmd(&dir, &man),
        "depth-model" => depth_model_cmd(&dir, &man),
        "depth-encode" => depth_encode_cmd(&dir, &man),
        "depth-maps" => depth_maps_cmd(&dir, &man),
        "pose" => pose_cmd(&dir, &man),
        _ => panic!("command {cmd}"),
    };
    std::fs::write(dir.join("rust.json"), serde_json::to_string(&out).unwrap()).unwrap();
}
