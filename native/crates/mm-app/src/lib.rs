//! mm-app: the store. All app state lives here as one serialisable tree; the UI reads it and changes it only through
//! commands (`call("seek", {"t": 10})`). The Qt host, mm-cli and the agent hooks all go through the same `call`, so
//! an agent drives the app exactly as a click does (native/ARCHITECTURE.md, "Three rules").
//!
//! - [`state`]: the tree's shape. [`commands`]: every command, one function each.
//! - [`decoder`]: the preview's decoder thread (latest request wins). [`player`]: playback's clock.
//! - [`render`]: draws the shown frame into the host's texture (or our own, headless).
//! - [`export`]: the export job. [`ffi`]: the C ABI (`include/mm_app.h`).
//!
//! Change notifications: every mutation names the state paths it touched; the host is told (`MmNotifyFn`) and
//! re-reads the state. "@render" means the video item has a new picture to draw.
pub mod check;
pub mod commands;
pub mod decide;
pub mod decoder;
pub mod editor;
pub mod export;
pub mod ffi;
pub mod library;
pub mod log;
pub mod people;
pub mod player;
pub mod records;
pub mod render;
pub mod state;
pub mod text;

use mm_engine::jobs::Job;
use mm_engine::Project;
use mm_media::Probe;
use serde_json::Value;
use state::State;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};

/// How the host started us.
#[derive(Clone, Debug, Default, serde::Deserialize)]
#[serde(default)]
pub struct Config {
    /// the repository root (for the privacy allowlist and out/)
    pub root: String,
    /// where the store saves what it restores after a restart (out/native/state.json)
    pub state_file: String,
    /// restore that state now
    pub restore: bool,
    pub dev: bool,
    /// start the engine (mm_engine::native) for the Projects, Decisions and People pages (off: the Editor alone)
    pub sidecar: bool,
    /// the projects folder it uses (MOJIMASK_PROJECTS); default `<root>/out/native/projects`
    pub projects: String,
}

/// Called with the JSON array of changed paths, from any thread.
pub type NotifyFn = Box<dyn Fn(&str) + Send + Sync>;

#[doc(hidden)]
pub struct Shared {
    pub cfg: Config,
    pub st: Mutex<State>,
    pub project: Mutex<Option<Project>>,
    /// bumped by every open: work for an older video is dropped
    pub gen: AtomicU64,
    pub probe: Mutex<Option<(u64, Probe)>>,
    pub newest: Mutex<Option<decoder::Decoded>>,
    /// covers changed: draw the shown frame again
    pub dirty: AtomicBool,
    pub readback: Mutex<Option<String>>,
    pub dec: Mutex<mpsc::Sender<decoder::Req>>,
    pub play_gen: AtomicU64,
    pub perf: Mutex<render::Perf>,
    pub export: Mutex<Job>,
    pub gpu: Mutex<Option<render::GpuState>>,
    pub notify: NotifyFn,
    /// the library, the open project and the sidecar (library.rs)
    pub lib: Mutex<library::Lib>,
    /// work for the library's thread, in order
    pub work: Mutex<mpsc::Sender<library::Job>>,
    /// native/web/strings.json, icons.json, brand.json
    pub text: Arc<text::Text>,
    /// the open project's draw records (records.rs): what the Editor draws
    pub recs: Mutex<Option<Arc<records::Recs>>>,
    /// lighting's grain per shown frame, measured off the decoder's thread (decoder.rs Grain)
    pub grain: decoder::Grain,
    /// the pages show their developer parts (starts as --dev; `set_dev` turns it off to see what everyone sees, the
    /// hooks staying on)
    pub ui_dev: AtomicBool,
}

impl Shared {
    pub fn dev(&self) -> bool {
        self.ui_dev.load(Ordering::SeqCst)
    }
}

impl Shared {
    pub fn changed(&self, paths: &[&str]) {
        (self.notify)(&serde_json::to_string(paths).unwrap());
    }
    pub fn gen(&self) -> u64 {
        self.gen.load(Ordering::SeqCst)
    }
    pub fn root(&self) -> PathBuf {
        PathBuf::from(&self.cfg.root)
    }
}

/// The store.
pub struct App {
    pub(crate) sh: Arc<Shared>,
}

impl App {
    pub fn new(cfg: Config, notify: NotifyFn) -> App {
        let (tx, rx) = mpsc::channel();
        let (wtx, wrx) = mpsc::channel();
        let text = text::Text::load(std::path::Path::new(&cfg.root)).unwrap_or_else(|e| {
            log::warn(&format!("native/web/strings.json: {e:#}"));
            text::Text::default()
        });
        let cfg_dev = cfg.dev;
        // depth on by default when its model is installed, as the Python Mac app does (desktop.py APP_DEPTH):
        // partial occlusion in scans and covers, and the Editor's Depth view. MOJIMASK_DEPTH=off turns it off. The
        // engine, the records and scans all read MOJIMASK_DEPTH, so it's set once here, before they start.
        if std::env::var_os("MOJIMASK_DEPTH").is_none() {
            // mm_analysis::depth::MODELS[0], the one model the app ships (packaging/make_app.sh DEPTH_REL)
            let models = mm_core::paths::models(std::path::Path::new(&cfg.root));
            if models.join("depth-anything-v2-small/model.onnx").exists() {
                std::env::set_var("MOJIMASK_DEPTH", "anything-small");
            }
        }
        let mut st = State::default();
        st.app.dev = cfg.dev;
        st.app.root = cfg.root.clone();
        st.library.dev = cfg.dev;
        let sh = Arc::new(Shared {
            cfg,
            st: Mutex::new(st),
            project: Mutex::new(None),
            gen: AtomicU64::new(0),
            probe: Mutex::new(None),
            newest: Mutex::new(None),
            dirty: AtomicBool::new(false),
            readback: Mutex::new(None),
            dec: Mutex::new(tx),
            play_gen: AtomicU64::new(0),
            perf: Mutex::new(Default::default()),
            export: Mutex::new(Job::default()),
            gpu: Mutex::new(None),
            notify,
            lib: Mutex::new(library::Lib::default()),
            work: Mutex::new(wtx),
            text: Arc::new(text),
            recs: Mutex::new(None),
            grain: decoder::Grain::default(),
            ui_dev: AtomicBool::new(cfg_dev),
        });
        let s2 = sh.clone();
        std::thread::Builder::new().name("mm-decode".into()).spawn(move || decoder::run(rx, s2)).unwrap();
        decoder::start_grain(&sh);
        library::start(&sh, wrx);
        let app = App { sh };
        if app.sh.cfg.restore {
            if let Err(e) = commands::restore(&app.sh) {
                log::warn(&format!("restore: {e:#}"));
            }
        }
        app
    }

    /// The state tree, or the subtree at a dotted path.
    /// Only the top-level section a path names is serialized (QML re-reads the sections a change note names, up to
    /// 30 times a second while playing: the whole tree is ~270 KB on a party clip, the editor ~15 KB).
    pub fn state(&self, path: &str) -> Value {
        let (top, rest) = path.split_once('.').unwrap_or((path, ""));
        let v = {
            let st = self.sh.st.lock().unwrap();
            let j = |x: Result<Value, serde_json::Error>| x.unwrap();
            match top {
                "" => j(serde_json::to_value(&*st)),
                "app" => j(serde_json::to_value(&st.app)),
                "editor" => j(serde_json::to_value(&st.editor)),
                "export" => j(serde_json::to_value(&st.export)),
                "readback" => j(serde_json::to_value(&st.readback)),
                "library" => j(serde_json::to_value(&st.library)),
                "scan" => j(serde_json::to_value(&st.scan)),
                "project" => j(serde_json::to_value(&st.project)),
                "decisions" => j(serde_json::to_value(&st.decisions)),
                "people" => j(serde_json::to_value(&st.people)),
                "export_page" => st.export_page.clone(),
                "result" => st.result.clone(),
                _ => return Value::Null,
            }
        };
        if rest.is_empty() {
            return v;
        }
        let mut cur = &v;
        for k in rest.split('.') {
            cur = match cur {
                Value::Object(m) => match m.get(k) {
                    Some(x) => x,
                    None => return Value::Null,
                },
                Value::Array(a) => match k.parse::<usize>().ok().and_then(|i| a.get(i)) {
                    Some(x) => x,
                    None => return Value::Null,
                },
                _ => return Value::Null,
            };
        }
        cur.clone()
    }

    /// Run a command.
    pub fn call(&self, cmd: &str, args: &Value) -> anyhow::Result<Value> {
        commands::dispatch(&self.sh, cmd, args)
    }

    /// Wait (polling) until `f(state)` is true, up to `timeout`.
    pub fn wait_until(&self, timeout: std::time::Duration, f: impl Fn(&Value) -> bool) -> bool {
        let t0 = std::time::Instant::now();
        while t0.elapsed() < timeout {
            if f(&self.state("")) {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        false
    }

    /// Wait until the decoder has delivered the frame asked for (headless use: then `render_to_png`).
    pub fn wait_frame(&self, timeout: std::time::Duration) -> bool {
        let t0 = std::time::Instant::now();
        while t0.elapsed() < timeout {
            let t = self.sh.st.lock().unwrap().editor.t;
            if self.sh.newest.lock().unwrap().as_ref().is_some_and(|d| d.t == t && d.gen == self.sh.gen()) {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        false
    }

    /// Headless: draw the shown frame on a device of our own and save it as a PNG (public-domain clips only).
    pub fn render_to_png(&self, path: &str) -> anyhow::Result<render::RenderInfo> {
        render::draw_headless(&self.sh, path)
    }
}

impl Drop for App {
    fn drop(&mut self) {
        self.sh.play_gen.fetch_add(1, Ordering::SeqCst);
        let _ = self.sh.dec.lock().unwrap().send(decoder::Req::Quit);
        commands::persist(&self.sh);
        library::stop(&self.sh);
    }
}
