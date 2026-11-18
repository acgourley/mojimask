//! Camera motion for a folder of frames (00000.png/.jpg, ...) -> mats_v2.npy (N x 2 x 3, float64), the same file the
//! Python scan writes, plus a JSON line per pair (fit, trust, timing) for the parity tools.
//!
//!     motion_mats FRAMES_DIR OUT.npy [--diag OUT.json] [--limit N] [--work-long PX] [--raw]
//! --raw: write the fitted similarity even where it isn't trusted (no identity fallback, no scale clip).
use mm_motion::camera::{prepare_bgr, production, CameraParams, CameraTracker};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: motion_mats FRAMES_DIR OUT.npy [--diag OUT.json] [--limit N] [--work-long PX] [--raw]");
        std::process::exit(2);
    }
    let mut diag = None;
    let mut limit = usize::MAX;
    let mut p = CameraParams::default();
    let mut raw = false;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--diag" => { diag = Some(args[i + 1].clone()); i += 1; }
            "--limit" => { limit = args[i + 1].parse().unwrap(); i += 1; }
            "--work-long" => { p.work_long = args[i + 1].parse().unwrap(); i += 1; }
            "--raw" => raw = true,
            a => panic!("unknown argument {a}"),
        }
        i += 1;
    }
    let mut files: Vec<_> = std::fs::read_dir(&args[1]).unwrap().filter_map(|e| {
        let p = e.ok()?.path();
        let ext = p.extension()?.to_str()?.to_lowercase();
        (ext == "png" || ext == "jpg").then_some(p)
    }).collect();
    files.sort();
    files.truncate(limit);
    let mut tr = CameraTracker::new(p.clone());
    let mut rows = vec![];
    let mut lines = vec![];
    let (mut t_prep, mut t_est) = (0.0, 0.0);
    for (k, f) in files.iter().enumerate() {
        let im = mm_motion::io::read_bgr(f);
        let t0 = Instant::now();
        let fr = prepare_bgr(&im, &p);
        let t1 = Instant::now();
        let (m, e) = tr.push(fr);
        let t2 = Instant::now();
        let (dp, de) = ((t1 - t0).as_secs_f64() * 1e3, (t2 - t1).as_secs_f64() * 1e3);
        if k > 0 {
            t_prep += dp;
            t_est += de;
        }
        let m = match (&e, raw) { (Some(e), true) => e.m, _ => m };
        rows.extend_from_slice(&m);
        if let Some(e) = e {
            lines.push(serde_json::json!({"t": k, "m": e.m, "ok": e.ok, "inlier_frac": e.inlier_frac, "blocks": e.blocks,
                "inliers": e.inliers, "sigma": e.sigma, "coarse": e.coarse, "vote": e.vote, "prod": production(&e, &p),
                "prep_ms": dp, "est_ms": de}));
        }
    }
    let n = files.len();
    mm_vision::npy::write(std::path::Path::new(&args[2]), &mm_vision::npy::Arr::F64(vec![n, 2, 3], rows)).unwrap();
    let pairs = (n.max(1) - 1).max(1) as f64;
    let summary = serde_json::json!({"frames": n, "prep_ms_per_frame": t_prep / pairs, "est_ms_per_pair": t_est / pairs,
        "ms_per_pair": (t_prep + t_est) / pairs, "untrusted": lines.iter().filter(|l| !l["ok"].as_bool().unwrap()).count()});
    if let Some(d) = diag {
        std::fs::write(d, serde_json::to_string(&serde_json::json!({"summary": summary, "pairs": lines})).unwrap()).unwrap();
    }
    println!("{summary}");
}
