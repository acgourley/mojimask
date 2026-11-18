//! mm-cli: the app's store with no window. Same commands as the UI and the agent hooks.
//!
//!   mm-cli [--root DIR] [--video CLIP [--data DUMP]] OP...
//!     call CMD [JSON]     run a command, print its result
//!     state [PATH]        print the state (or the subtree at a dotted path)
//!     wait-ready          wait until the video is open and probed
//!     render PNG          draw the frame asked for (with its covers) on our own GPU device and save it
//!                         (public-domain clips only: test_clips/jazz.mp4, out/poc/clips/)
//!     wait-export         wait until an export started with `call export` ends; print state.export
//!     help                list the store's commands
//!   mm-cli decisions ...   a project's Decisions, answered from the command line (decisions.rs)
//!   mm-cli scan VIDEO PROJECT [--plan PLAN] [--depth [NAME]]
//!                          the whole scan of a video into a project folder, headless (scan.rs)
//! Every result is one JSON line. Exit 1 on the first failing op.
use anyhow::{anyhow, bail, Result};
use mm_app::{App, Config};
use serde_json::{json, Value};
use std::time::Duration;

mod decisions;
mod strips;
mod scan;

fn main() {
    if let Err(e) = run() {
        println!("{}", json!({"ok": false, "error": format!("{e:#}")}));
        std::process::exit(1);
    }
}

fn find_root() -> String {
    let mut d = std::env::current_dir().unwrap();
    loop {
        if d.join("native/ARCHITECTURE.md").exists() {
            return d.to_string_lossy().into_owned();
        }
        if !d.pop() {
            return std::env::current_dir().unwrap().to_string_lossy().into_owned();
        }
    }
}

fn run() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("decisions") {
        decisions::main(&args[1..]); // the project model only: no store, no window (decisions.rs)
    }
    if args.first().map(String::as_str) == Some("scan") {
        scan::main(&args[1..]); // the engine only: no store, no window (scan.rs)
    }
    let mut opt = |k: &str| -> Option<String> {
        let i = args.iter().position(|a| a == k)?;
        args.remove(i);
        (i < args.len()).then(|| args.remove(i))
    };
    let root = opt("--root").unwrap_or_else(find_root);
    let video = opt("--video");
    let data = opt("--data").unwrap_or_default();
    if args.is_empty() {
        bail!("usage: mm-cli [--root DIR] [--video CLIP [--data DUMP]] (call CMD [JSON] | state [PATH] | wait-ready | render PNG | wait-export | help)...");
    }
    let app = App::new(Config { root, ..Default::default() }, Box::new(|_| {}));
    if let Some(v) = video {
        app.call("open", &json!({"video": v, "data": data}))?;
        wait_ready(&app)?;
    }
    let mut i = 0;
    while i < args.len() {
        let op = args[i].as_str();
        i += 1;
        let next_is_arg = |i: usize| i < args.len() && !is_op(&args[i]);
        match op {
            "call" => {
                let cmd = args.get(i).ok_or_else(|| anyhow!("call: which command?"))?.clone();
                i += 1;
                let a: Value = if next_is_arg(i) {
                    i += 1;
                    serde_json::from_str(&args[i - 1])?
                } else {
                    json!({})
                };
                let r = app.call(&cmd, &a)?;
                println!("{}", json!({"ok": true, "cmd": cmd, "result": r}));
            }
            "state" => {
                let p = if next_is_arg(i) {
                    i += 1;
                    args[i - 1].clone()
                } else {
                    String::new()
                };
                println!("{}", app.state(if p == "." { "" } else { &p }));
            }
            "wait-ready" => {
                wait_ready(&app)?;
                println!("{}", json!({"ok": true, "video": app.state("editor.video")}));
            }
            "render" => {
                let png = args.get(i).ok_or_else(|| anyhow!("render: where to?"))?.clone();
                i += 1;
                if !app.wait_frame(Duration::from_secs(20)) {
                    bail!("render: the decoder didn't deliver frame {}", app.state("editor.t"));
                }
                let r = app.render_to_png(&png)?;
                println!("{}", json!({"ok": true, "png": png, "frame": r.shown_t, "composite_ms": r.composite_ms}));
            }
            "wait-export" => {
                app.wait_until(Duration::from_secs(600), |s| s["export"]["state"] != "running");
                println!("{}", app.state("export"));
            }
            "help" => println!("{}", app.call("help", &json!({}))?),
            x => bail!("unknown op '{x}'"),
        }
    }
    Ok(())
}

fn is_op(s: &str) -> bool {
    matches!(s, "call" | "state" | "wait-ready" | "render" | "wait-export" | "help")
}

fn wait_ready(app: &App) -> Result<()> {
    if !app.wait_until(Duration::from_secs(20), |s| s["editor"]["video"]["ready"] == true || s["editor"]["status"].as_str().is_some_and(|x| x.starts_with("can't"))) {
        bail!("the video didn't open in 20 s");
    }
    if app.state("editor.video.ready") != true {
        bail!("{}", app.state("editor.status"));
    }
    Ok(())
}
