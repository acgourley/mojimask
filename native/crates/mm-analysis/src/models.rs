//! The analysis models and their ONNX Runtime sessions, opened the way onnx_core.session() opens them (mm_ml::rt):
//! RetinaFace-R50 for landmarks (reid.LANDMARK_ENGINE "ane"), SFace (reid.ENGINE "gpu"), OSNet (always "cpu"),
//! Depth Anything V2 Small ("gpu").
//!
//! reid.POOL (MOJIMASK_FACE_POOL, default 3): copies of each model, because one copy runs one call at a time however
//! many threads ask. Each thread is handed a copy in turn and keeps it (reid._copies).
//!
//! `Engines::cpu()` puts every model on the CPU (bit-identical to Python's CPU sessions; mm_ml::rt's PreferCPU).
use mm_ml::rt::{self, Ep};
use ort::session::Session;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{atomic::AtomicUsize, atomic::Ordering, Arc, Mutex};

pub const RETINAFACE: &str = "retinaface-r50.onnx";
pub const SFACE: &str = "sface-2021dec.onnx";
pub const OSNET: &str = "osnet-ain-x1_0-msmt17.onnx";

/// Where each model runs (onnx_core.ENGINES names).
#[derive(Clone, Copy, Debug)]
pub struct Engines {
    pub landmarks: Ep,
    pub sface: Ep,
    pub osnet: Ep,
    pub depth: Ep,
}

impl Engines {
    /// Python's defaults: reid.LANDMARK_ENGINE "ane", reid.ENGINE "gpu", appearance "cpu", depth._Onnx "gpu".
    pub fn default_plan() -> Engines {
        Engines { landmarks: Ep::Ane, sface: Ep::Gpu, osnet: Ep::Cpu, depth: Ep::Gpu }
    }
    pub fn cpu() -> Engines {
        Engines { landmarks: Ep::Cpu, sface: Ep::Cpu, osnet: Ep::Cpu, depth: Ep::Cpu }
    }
}

type Pool = Arc<Vec<Mutex<Session>>>;

pub struct Models {
    env: ort::environment::Environment,
    /// models/onnx
    pub dir: PathBuf,
    /// the CoreML compile cache (onnx_core._coreml_cache's base); None = none
    pub cache: Option<PathBuf>,
    pub engines: Engines,
    /// copies of each model (MOJIMASK_FACE_POOL, default 3): threads share them
    pub pool: usize,
    /// a model's own number of copies, by file name (e.g. the depth model runs on one thread: one session, since
    /// every CoreML session costs its load time) ([`Models::set_pool`])
    pub pools: HashMap<String, usize>,
    sessions: Mutex<HashMap<String, Pool>>,
}

thread_local! {
    static SLOT: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}
static NEXT: AtomicUsize = AtomicUsize::new(0);

impl Models {
    pub fn new(dir: &Path, engines: Engines, cache: Option<PathBuf>) -> Models {
        let pool = std::env::var("MOJIMASK_FACE_POOL").ok().and_then(|v| v.parse().ok()).unwrap_or(3usize).max(1);
        Models { env: rt::env(), dir: dir.to_path_buf(), cache, engines, pool, pools: HashMap::new(), sessions: Mutex::new(HashMap::new()) }
    }

    /// `n` copies of model `file` (its file name, e.g. depth's onnx), instead of `pool`.
    pub fn set_pool(&mut self, file: &str, n: usize) {
        self.pools.insert(file.to_string(), n.max(1));
    }

    fn pool_of(&self, path: &Path) -> usize {
        let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        self.pools.get(&name).copied().unwrap_or(self.pool)
    }

    /// This thread's copy of a model (opened on first use, `pool` copies or its own [`Models::set_pool`]), locked for
    /// one run.
    pub fn with<R>(&self, path: &Path, ep: Ep, dims: &[(String, i64)], f: impl FnOnce(&mut Session) -> R) -> R {
        let key = format!("{}|{}|{:?}", path.display(), ep.name(), dims);
        let pool = {
            let mut s = self.sessions.lock().unwrap();
            if !s.contains_key(&key) {
                let cache = self.cache.as_ref().map(|c| {
                    let stem = path.file_stem().unwrap().to_string_lossy().to_string();
                    let d: String = dims.iter().map(|(n, v)| format!("-{n}{v}")).collect();
                    c.join(format!("{stem}-{}{d}", ep.name()))
                });
                let v: Vec<Mutex<Session>> = (0..self.pool_of(path))
                    .map(|_| Mutex::new(rt::session(&self.env, path, ep, dims, cache.as_deref()).expect("open model")))
                    .collect();
                s.insert(key.clone(), Arc::new(v));
            }
            s[&key].clone()
        };
        let i = SLOT.with(|c| {
            if c.get().is_none() {
                c.set(Some(NEXT.fetch_add(1, Ordering::Relaxed)));
            }
            c.get().unwrap()
        }) % pool.len();
        let mut g = pool[i].lock().unwrap();
        f(&mut g)
    }

    pub fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    pub fn run(&self, name: &str, ep: Ep, dims: &[(String, i64)], shape: &[usize], x: Vec<f32>) -> Vec<(Vec<usize>, Vec<f32>)> {
        let p = self.path(name);
        self.with(&p, ep, dims, |s| rt::run(s, shape, x).expect("model run"))
    }

    pub fn run_path(&self, path: &Path, ep: Ep, dims: &[(String, i64)], shape: &[usize], x: Vec<f32>) -> Vec<(Vec<usize>, Vec<f32>)> {
        self.with(path, ep, dims, |s| rt::run(s, shape, x).expect("model run"))
    }
}
