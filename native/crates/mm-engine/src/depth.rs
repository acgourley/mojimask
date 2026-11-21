//! The depth cache of a new project (depth.compute), computed natively: the model on every frame
//! (mm_analysis::depth::run_onnx), each map's 8-bit picture into a grey movie (mm-media's [`GreyWriter`], as
//! video_io.writer), then the movie decoded back ([`mm_media::grey::luma`], as video_io.luma) and each frame's bytes ->
//! nearness line fitted on it. Writes PROJECT/dets/depth_<model>.mp4 and .json as depth.compute does (temporary files
//! renamed at the end; nothing half-written left behind on failure).
//!
//! The fit is np.polyfit(x, y, 1) bit for bit (mm_analysis::depth::polyfit1, used by fit_line).
//!
//! Parity: tools/parity/engine2_depth.py (the sidecar and every decoded frame, against Python's depth.compute).
use anyhow::{anyhow, Result};
use mm_analysis::depth as adepth;
use mm_analysis::models::Models;
use mm_media::grey::GreyWriter;
use std::path::{Path, PathBuf};

/// np.polyfit(x, y, 1) -> (slope, intercept), bit for bit on the Mac (moved to mm_analysis::depth::polyfit1).
pub use mm_analysis::depth::polyfit1;

/// depth.cache_paths
pub fn cache_paths(proj: &Path, name: &str) -> (PathBuf, PathBuf) {
    (proj.join("dets").join(format!("depth_{name}.mp4")), proj.join("dets").join(format!("depth_{name}.json")))
}

pub fn cached(proj: &Path, name: &str) -> bool {
    let (m, s) = cache_paths(proj, name);
    m.exists() && s.exists()
}

/// Python's json.dump of the sidecar: {"model": ..., "shape": [h, w], "frames": [[a, b, p10, p50, p90], ...]}.
fn sidecar_json(side: &adepth::Sidecar) -> String {
    use mm_core::pyjson::float_repr as r;
    let rows: Vec<String> = side.frames.iter().map(|f| format!("[{}]", f.iter().map(|v| r(*v)).collect::<Vec<_>>().join(", "))).collect();
    format!("{{\"model\": {}, \"shape\": [{}, {}], \"frames\": [{}]}}", serde_json::to_string(&side.model).unwrap(), side.shape.0, side.shape.1, rows.join(", "))
}

/// The models for [`compute`]: the depth model runs on one thread, so one session of it (Models opens
/// MOJIMASK_FACE_POOL copies of each other model, and every CoreML session costs its load time). `cpu`: on the CPU instead of the
/// GPU (depth._Onnx's "gpu").
pub fn models(models_dir: &Path, cpu: bool, cache: Option<PathBuf>) -> Models {
    use mm_analysis::models::Engines;
    let mut m = Models::new(&models_dir.join("onnx"), if cpu { Engines::cpu() } else { Engines::default_plan() }, cache);
    for i in &adepth::MODELS {
        if let Some(f) = std::path::Path::new(i.path).file_name() {
            m.set_pool(&f.to_string_lossy(), 1);
        }
    }
    m
}

/// depth.compute(proj, name): nearness for every frame of the project's video (detect colour, mm_media::frames), cached as a grey movie + sidecar. Does
/// nothing when the cache exists. `progress(fraction)` while computing. `models`: [`models`].
pub fn compute(proj: &Path, name: &str, models: &Models, models_dir: &Path, progress: &mut dyn FnMut(f64)) -> Result<()> {
    let (movie, side_p) = cache_paths(proj, name);
    if cached(proj, name) {
        return Ok(());
    }
    let info = adepth::model(name).ok_or_else(|| anyhow!("depth model {name}"))?;
    if !info.onnx {
        return Err(anyhow!("depth {name}: only the ONNX models are ported"));
    }
    // MODELS[name]["path"] is relative to the repo; the models folder holds models/<...>
    let model_path = models_dir.join(info.path.trim_start_matches("models/"));
    if !model_path.exists() {
        return Err(anyhow!("MOJIMASK_DEPTH={name}: {} is missing (models/README.md says where it comes from)", model_path.display()));
    }
    // every frame in order, from the video (one decoder, a few frames kept)
    let frames = mm_media::frames::ScanFrames::for_project(proj)?.sized(1, 256 << 20);
    let n = mm_core::scan::Scan::open(proj).map_err(|e| anyhow!(e))?.frames().unwrap_or(0);
    if n == 0 {
        return Err(anyhow!("depth {name}: no frames in {}/meta.json", proj.display()));
    }
    let tmp = PathBuf::from(format!("{}.tmp.mp4", movie.display()));
    let side_tmp = PathBuf::from(format!("{}.tmp", side_p.display()));
    let mut run = || -> Result<()> {
        let mut w: Option<GreyWriter> = None;
        let mut shape = (0, 0);
        let mut info = vec![];
        let mut samples: Vec<Vec<f32>> = vec![];
        let (mut t_read, mut t_model, mut t_write) = (0f64, 0f64, 0f64);
        let t_all = std::time::Instant::now();
        for i in 0..n {
            let tt = std::time::Instant::now();
            let img = frames.get(i as i64);
            t_read += tt.elapsed().as_secs_f64();
            let tt = std::time::Instant::now();
            let d = adepth::run_onnx(models, &model_path, &img, None);
            t_model += tt.elapsed().as_secs_f64();
            if i == 0 && std::env::var_os("MM_DEPTH_TIMES").is_some() {
                eprintln!("depth: first frame {:.1} s ({}x{} -> {}x{})", tt.elapsed().as_secs_f64(), img.cols, img.rows, d.cols, d.rows);
            }
            let tt = std::time::Instant::now();
            if w.is_none() {
                shape = (d.rows, d.cols);
                let (h2, w2) = (d.rows + d.rows % 2, d.cols + d.cols % 2);
                w = Some(GreyWriter::new(&tmp, w2, h2, adepth::MOVIE_FPS, adepth::MOVIE_BITRATE, adepth::MOVIE_KEYFRAMES)?);
            }
            let e = adepth::encode(&d);
            w.as_mut().unwrap().write(&e.q)?;
            info.push(e.pct);
            samples.push(e.sample);
            t_write += tt.elapsed().as_secs_f64();
            progress((i + 1) as f64 / n as f64);
        }
        w.take().unwrap().close()?;
        let t_back = std::time::Instant::now();
        let (h, wd) = shape;
        let mut fit = vec![];
        for (i, y) in mm_media::grey::luma(&tmp, adepth::MOVIE_FPS, 0, None)? {
            if i >= samples.len() {
                break;
            }
            let mut x = vec![];
            for r in (0..h).step_by(4) {
                for c in (0..wd).step_by(4) {
                    x.push(y.data[r * y.cols + c] as f64);
                }
            }
            let s: Vec<f64> = samples[i].iter().map(|&v| v as f64).collect();
            let (mn, mx) = x.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
            let (a, b) = if mx - mn > 0.0 {
                polyfit1(&x, &s)
            } else {
                // float(samples[i].mean()): numpy's float32 mean
                (0.0, mm_core::np::mean(&samples[i]) as f64)
            };
            fit.push([a, b]);
        }
        if std::env::var_os("MM_DEPTH_TIMES").is_some() {
            eprintln!("depth: read {t_read:.1} s, model {t_model:.1} s, encode+write {t_write:.1} s, loop {:.1} s, read back + fit {:.1} s",
                      t_all.elapsed().as_secs_f64() - t_back.elapsed().as_secs_f64(), t_back.elapsed().as_secs_f64());
        }
        if fit.len() != n {
            return Err(anyhow!("depth movie: wrote {} frames, read back {}", n, fit.len()));
        }
        let side = adepth::Sidecar {
            model: name.to_string(),
            shape,
            frames: fit.iter().zip(&info).map(|(f, q)| [f[0], f[1], q[0], q[1], q[2]]).collect(),
        };
        std::fs::write(&side_tmp, sidecar_json(&side))?;
        std::fs::rename(&tmp, &movie)?;
        std::fs::rename(&side_tmp, &side_p)?;
        Ok(())
    };
    let r = run();
    for p in [&tmp, &side_tmp] {
        let _ = std::fs::remove_file(p);
    }
    r
}
