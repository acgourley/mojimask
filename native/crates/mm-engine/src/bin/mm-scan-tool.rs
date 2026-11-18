//! The native scan from the command line (mm-engine's scan.rs), and its developer pieces. Parity scripts in
//! native/tools/parity/detect_*.py call it.
//!
//!     mm-scan-tool scan VIDEO PROJECT [options]        the whole scan: detect, motion, depth, decide
//!     mm-scan-tool detect VIDEO PROJECT [options]      prepare + detect only (frames/, dets/, meta.json; with
//!                                                      --motion native also mats_v2.npy)
//!     mm-scan-tool decide PROJECT [options] [--replay REC] [--dump DIR]
//!                                                      the decide step with Rust evidence; motion replayed from REC
//!                                                      (scan_record.py's rec/), else by --motion
//!     mm-scan-tool depth PROJECT [NAME]                the depth cache (depth.compute) natively
//!     mm-scan-tool evidence-check REC PROJECT [--only face,face_near,body,depth,looks,motion,flow,trackable]
//!     mm-scan-tool jpeg-dec IN.jpg OUT.raw             cv2.imread's pixels (BGR rows), raw
//!     mm-scan-tool jpeg-enc IN.raw W H OUT.jpg         cv2.imwrite(quality 90) of raw BGR rows
//!
//! options: --plan cpu|default|gpu|ane (a fixed detection plan; default: timed, as fastest_core)
//!          --analysis-cpu (the analysis models on the CPU)  --depth NAME (MOJIMASK_DEPTH)
//!          --models DIR (default ROOT/../models or $MOJIMASK_MODELS)  --cache DIR (CoreML compiles)
//!          --python PATH  --root DIR (the repo, for Python's motion)
//!          --motion native|python|replay (camera motion and the follower's motion evidence; default python;
//!          replay with --replay REC, decide only)
use mm_core::pyjson::Py;
use mm_engine::jpeg;
use mm_engine::scan::{self, Options};
use std::path::PathBuf;

fn flag(a: &[String], name: &str) -> Option<String> {
    a.iter().position(|x| x == name).map(|i| a[i + 1].clone())
}

fn options(a: &[String]) -> Options {
    Options::from_args(a).unwrap_or_else(|e| {
        eprintln!("{e:#}");
        std::process::exit(2)
    })
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut report = |step: &str, f: f64, info: serde_json::Value| {
        if std::env::var_os("MM_SCAN_QUIET").is_none() {
            eprintln!("{step:8} {:4.0}% {info}", f * 100.0);
        }
    };
    match a.get(1).map(|s| s.as_str()) {
        Some("jpeg-dec") => {
            let d = jpeg::read_bgr(std::path::Path::new(&a[2])).unwrap();
            std::fs::write(&a[3], &d.data).unwrap();
            println!("{} {}", d.w, d.h);
        }
        Some("jpeg-enc") => {
            let raw = std::fs::read(&a[2]).unwrap();
            let (w, h): (usize, usize) = (a[3].parse().unwrap(), a[4].parse().unwrap());
            std::fs::write(&a[5], jpeg::encode_bgr(&raw, w, h, 90)).unwrap();
        }
        Some("detect") => {
            let o = options(&a);
            let st = scan::prepare_and_detect(&a[2], std::path::Path::new(&a[3]), &o, &mut report).unwrap_or_else(|e| {
                eprintln!("detect: {e:#}");
                std::process::exit(1)
            });
            println!("{}", serde_json::to_string(&st).unwrap());
        }
        Some("decide") => {
            let o = options(&a);
            let proj = PathBuf::from(&a[2]);
            let (res, stages, viz, info) = scan::decide(&proj, &o).unwrap_or_else(|e| {
                eprintln!("decide: {e:#}");
                std::process::exit(1)
            });
            if let Some(d) = flag(&a, "--dump") {
                let d = PathBuf::from(d);
                std::fs::create_dir_all(&d).unwrap();
                for (k, v) in &stages.items {
                    std::fs::write(d.join(format!("stage_{k}.json")), v.dumps()).unwrap();
                }
                std::fs::write(d.join("events.json"), Py::List(viz.events.clone()).dumps()).unwrap();
            }
            println!("{}", serde_json::json!({"result": Py::Dict(res).to_serde(), "info": info}));
        }
        Some("evidence-check") => {
            // every recorded face / face_near / body / depth call (scan_record.py's rec/) asked of the Rust evidence
            let o = options(&a);
            let rec = PathBuf::from(&a[2]);
            let proj = PathBuf::from(&a[3]);
            let only = flag(&a, "--only");
            let r = mm_engine::check::evidence(&rec, &proj, &o, only.as_deref());
            println!("{}", serde_json::to_string_pretty(&r).unwrap());
        }
        Some("frames-dump") => {
            // the detect frames (video_io.frames(color="detect")) as raw BGR: OUT/<t>.raw (debugging)
            mm_media::decode::init();
            let mut d = mm_media::Decoder::open(&a[2]).unwrap();
            let tags = d.colour_tags().map(|s| s.to_string());
            let n: usize = a[4].parse().unwrap();
            std::fs::create_dir_all(&a[3]).unwrap();
            while let Some((t, f)) = d.next().unwrap() {
                if t >= n {
                    break;
                }
                let img = mm_engine::scan::detect_frame(&f, &d.probe, tags.as_deref()).unwrap();
                std::fs::write(format!("{}/{t}.raw", a[3]), &img.data).unwrap();
                println!("{t} {} {} {:?}", img.cols, img.rows, tags);
            }
        }
        Some("luma-dump") => {
            // the depth movie's Y planes as the evidence reads them: OUT/<i>.raw (debugging)
            use mm_analysis::depth::LumaSource;
            let src = mm_engine::evidence::MovieLuma { path: PathBuf::from(&a[2]), fps: mm_analysis::depth::MOVIE_FPS };
            let n: usize = a[4].parse().unwrap();
            std::fs::create_dir_all(&a[3]).unwrap();
            for (i, y) in src.luma(0, Some(n)) {
                std::fs::write(format!("{}/{i}.raw", a[3]), &y.data).unwrap();
                println!("{i} {} {}", y.cols, y.rows);
            }
        }
        Some("depth") => {
            // depth.compute(PROJECT, NAME) natively: PROJECT/dets/depth_NAME.mp4 and .json
            let o = options(&a);
            let proj = PathBuf::from(&a[2]);
            let name = a.get(3).cloned().unwrap_or_else(|| "anything-small".into());
            let models = mm_engine::depth::models(&o.models, o.analysis_cpu, o.cache.as_ref().map(|c| c.join("analysis")));
            let t0 = std::time::Instant::now();
            mm_engine::depth::compute(&proj, &name, &models, &o.models, &mut |_| {}).unwrap_or_else(|e| {
                eprintln!("depth: {e:#}");
                std::process::exit(1)
            });
            println!("{}", serde_json::json!({"seconds": t0.elapsed().as_secs_f64()}));
        }
        Some("polyfit") => {
            // np.polyfit(x, y, 1) for the parity check: IN.json [[x...], [y...]] -> [a, b]
            let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&a[2]).unwrap()).unwrap();
            let f = |i: usize| v[i].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect::<Vec<f64>>();
            let (p, q) = mm_engine::depth::polyfit1(&f(0), &f(1));
            println!("[{}, {}]", mm_core::pyjson::float_repr(p), mm_core::pyjson::float_repr(q));
        }
        Some("scan") => {
            let o = options(&a);
            let r = scan::scan(&a[2], std::path::Path::new(&a[3]), &o, &mut report).unwrap_or_else(|e| {
                eprintln!("scan: {e:#}");
                std::process::exit(1)
            });
            println!("{r}");
        }
        _ => {
            eprintln!("usage: see the module doc (mm-scan-tool scan|detect|decide|jpeg-dec|jpeg-enc)");
            std::process::exit(2);
        }
    }
}
