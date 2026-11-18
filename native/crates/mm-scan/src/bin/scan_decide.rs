//! The scan's decide step on a project, from a recording of the pictures' evidence (parity runs):
//!
//!     scan_decide PROJECT --replay REC [--depth NAME] [--dump DIR]
//!
//! Writes the project's decide-step files (as Python's newalgo.run) and, with --dump, each stage's output in the
//! recorder's format (stage_*.json) and the events (events.json). Prints the step's result as JSON.
use mm_core::pyjson::Py;
use mm_scan::{evidence, newalgo};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let proj = PathBuf::from(&args[1]);
    let (mut rec, mut dump, mut depth) = (None, None, None);
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--replay" => rec = Some(PathBuf::from(&args[i + 1])),
            "--dump" => dump = Some(PathBuf::from(&args[i + 1])),
            "--depth" => depth = Some(args[i + 1].clone()),
            a => panic!("unknown argument {a}"),
        }
        i += 2;
    }
    let rec = rec.expect("--replay REC");
    let (mut ev, usage) = evidence::replay_counted(&rec, depth.as_deref()).unwrap();
    let t0 = std::time::Instant::now();
    let (res, stages, viz) = match newalgo::run_logged(&proj, &mut ev) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("scan_decide: {e}");
            std::process::exit(1);
        }
    };
    let secs = t0.elapsed().as_secs_f64();
    if let Some(d) = &dump {
        std::fs::create_dir_all(d).unwrap();
        for (k, v) in &stages.items {
            std::fs::write(d.join(format!("stage_{k}.json")), v.dumps()).unwrap();
        }
        std::fs::write(d.join("events.json"), Py::List(viz.events.clone()).dumps()).unwrap();
    }
    let mut out = res;
    out.insert("seconds", Py::Float(secs));
    // recorded calls used, by boundary: [recorded, used]
    out.insert("replay", Py::Dict(usage.report().into_iter().map(|(k, n, u)| (k.to_string(), Py::List(vec![Py::Int(n as i64), Py::Int(u as i64)]))).collect()));
    println!("{}", Py::Dict(out).dumps());
}
