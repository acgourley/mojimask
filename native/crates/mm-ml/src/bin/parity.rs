//! Raw model outputs from Rust vs Python's on the same inputs and execution provider (poc/ml's parity check).
//!   mm-ml-parity <ref dir: out/poc/results/ml/ref> [out.json] [--coreml] [--models a,b]
//! CPU vs CPU always; with --coreml also each model's default plan (CoreML) vs Python's CoreML output.
//! Prints a JSON summary: per model and output, max |rust - python|; exit 1 if any CPU comparison isn't exact.
use mm_ml::{npy, rt};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let r = PathBuf::from(args.get(1).expect("usage: mm-ml-parity <ref dir> [out.json] [--coreml] [--models a,b]"));
    let out = args.get(2).filter(|a| !a.starts_with("--")).cloned();
    let coreml = args.iter().any(|a| a == "--coreml");
    let only: Option<Vec<String>> =
        args.iter().position(|a| a == "--models").map(|i| args[i + 1].split(',').map(String::from).collect());
    let man: Value = serde_json::from_str(&std::fs::read_to_string(r.join("manifest.json")).expect("manifest.json")).unwrap();
    let models_dir = Path::new(man["models_dir"].as_str().unwrap());
    let cache = std::env::var("MM_ML_CACHE").map(PathBuf::from).unwrap_or_else(|_| r.join("../cache_native"));
    let env = rt::env();
    let mut sessions: HashMap<(String, &'static str, Vec<usize>), ort::session::Session> = HashMap::new();
    let mut worst: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    let t0 = Instant::now();
    let mut cases = 0;
    for c in man["cases"].as_array().unwrap() {
        let m = c["model"].as_str().unwrap().to_string();
        if only.as_ref().is_some_and(|o| !o.contains(&m)) {
            continue;
        }
        let d = r.join(c["dir"].as_str().unwrap());
        let x = npy::read(&d.join("input.npy")).unwrap();
        let def = rt::Ep::parse(c["engine"].as_str().unwrap());
        let dims: Vec<(String, i64)> =
            c["dims"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_i64().unwrap())).collect();
        let mut eps = vec![rt::Ep::Cpu];
        if coreml && def != rt::Ep::Cpu {
            eps.push(def);
        }
        for ep in eps {
            let key = (m.clone(), ep.name(), x.shape.clone());
            if !sessions.contains_key(&key) {
                let path = models_dir.join(man["models"][&m]["path"].as_str().unwrap());
                let cd = cache.join(format!("{m}-{}", ep.name()));
                let s = rt::session(&env, &path, ep, &dims, Some(&cd)).unwrap_or_else(|e| panic!("{m}: {e}"));
                sessions.insert(key.clone(), s);
            }
            let o = rt::run(sessions.get_mut(&key).unwrap(), &x.shape, x.data.clone()).unwrap();
            let tag = if ep == rt::Ep::Cpu { "cpu" } else { "coreml" };
            for (i, (shape, data)) in o.iter().enumerate().take(c["n_out"].as_u64().unwrap() as usize) {
                let py = npy::read(&d.join(format!("{tag}_{i}.npy"))).unwrap();
                assert_eq!(shape, &py.shape, "{m} output {i} shape");
                let e = data.iter().zip(&py.data).map(|(a, b)| (*a as f64 - *b as f64).abs()).fold(0.0, f64::max);
                let w = worst.entry(format!("{m}|rust_{tag}_vs_py_{tag}|out{i}")).or_insert((0.0, 0));
                w.0 = w.0.max(e);
                w.1 += data.len();
            }
        }
        cases += 1;
    }
    let cpu_exact = worst.iter().filter(|(k, _)| k.contains("rust_cpu")).all(|(_, v)| v.0 == 0.0);
    let res = json!({
        "cases": cases, "seconds": t0.elapsed().as_secs_f64(), "cpu_exact": cpu_exact,
        "max_abs": worst.iter().map(|(k, v)| (k.clone(), json!({"max_abs": v.0, "n": v.1}))).collect::<serde_json::Map<_, _>>(),
    });
    let s = serde_json::to_string_pretty(&res).unwrap();
    if let Some(o) = out {
        std::fs::write(o, &s).unwrap();
    }
    println!("{s}");
    std::process::exit(if cpu_exact { 0 } else { 1 });
}
