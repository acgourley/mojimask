//! `mm-cli scan VIDEO PROJECT [--plan PLAN] [--depth [NAME]]`: the whole scan of a video into
//! a project folder (pipeline.run's steps; mm_engine::scan), headless. Progress goes to stderr, one line a report;
//! the result (each step's numbers, newalgo's counts, seconds) is one JSON line on stdout.
//!
//! - `--plan`: a fixed detection plan (cpu, default, gpu, ane); without it the timed choice (onnx_core.fastest_core).
//! - `--depth`: a depth model (`anything-small` when no name follows; `off` for none). The project's depth cache is made
//!   natively when it has none.
//! - Camera motion and head following are mm-motion's (our own).
//! - The other options of `mm-scan-tool` (--models, --cache, --analysis-cpu, --root) are accepted too.
use anyhow::Result;
use serde_json::json;

pub fn main(args: &[String]) -> ! {
    let r = run(args);
    match r {
        Ok(v) => {
            println!("{v}");
            std::process::exit(0)
        }
        Err(e) => {
            println!("{}", json!({"ok": false, "error": format!("{e:#}")}));
            std::process::exit(1)
        }
    }
}

fn run(args: &[String]) -> Result<serde_json::Value> {
    let pos: Vec<&String> = {
        let mut v = vec![];
        let mut i = 0;
        while i < args.len() {
            let a = &args[i];
            if a.starts_with("--") {
                // flags with a value (all but --analysis-cpu, and --depth with no name after it)
                let takes = a != "--analysis-cpu" && !(a == "--depth" && args.get(i + 1).is_none_or(|n| n.starts_with("--")));
                i += if takes { 2 } else { 1 };
            } else {
                v.push(a);
                i += 1;
            }
        }
        v
    };
    if pos.len() != 2 {
        anyhow::bail!("usage: mm-cli scan VIDEO PROJECT [--plan PLAN] [--depth [NAME]]");
    }
    // a bare --depth means the shipped model
    let mut a: Vec<String> = vec![];
    for (i, x) in args.iter().enumerate() {
        a.push(x.clone());
        if x == "--depth" && args.get(i + 1).is_none_or(|n| n.starts_with("--")) {
            a.push("anything-small".into());
        }
    }
    let o = mm_engine::scan::Options::from_args(&a)?;
    let quiet = std::env::var_os("MM_SCAN_QUIET").is_some();
    let mut report = |step: &str, f: f64, info: serde_json::Value| {
        if !quiet {
            eprintln!("{step:8} {:4.0}% {info}", f * 100.0);
        }
    };
    let proj = std::path::Path::new(pos[1].as_str());
    std::fs::create_dir_all(proj)?;
    let r = mm_engine::scan::scan(pos[0], proj, &o, &mut report)?;
    Ok(json!({"ok": true, "scan": r}))
}
