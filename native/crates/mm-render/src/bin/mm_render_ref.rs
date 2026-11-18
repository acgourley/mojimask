//! mm-render-ref: draw frames from a draw reference dump (native/tools/parity/draw_ref.py) to PNG, headless, for the
//! parity scripts.
//!
//!     mm-render-ref --ref out/native/draw/ref --out out/native/draw/rust [--configs a,b] [--frames 0,16]
//!                   [--py-sprites] [--rows]
//!
//! For each config and frame: in/NNNNN.png + <config>/NNNNN.json (the records) -> <out>/<config>/NNNNN.png.
//!   --py-sprites  use Python's emoji_sprite pictures (ref/sprites/*.png) instead of drawing them with CoreText
//!   --rows        also write the compositor rows as <out>/<config>/NNNNN.rows.json
//!   --sprites     only draw every emoji the dump has (CoreText) to <out>/sprites/<key>.png and stop
use anyhow::{anyhow, Context, Result};
use mm_core::cover::{Eyes, FrameRecord, RenderMeta};
use mm_render::draw::Drawer;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let refd = PathBuf::from(arg(&args, "--ref").ok_or_else(|| anyhow!("--ref DIR"))?);
    let outd = PathBuf::from(arg(&args, "--out").ok_or_else(|| anyhow!("--out DIR"))?);
    let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(refd.join("configs.json"))?)?;
    let keys: HashMap<String, String> = serde_json::from_value(cfg["keys"].clone())?;
    if args.iter().any(|a| a == "--sprites") {
        std::fs::create_dir_all(outd.join("sprites"))?;
        for (hex, ch) in &keys {
            if mm_core::cover::is_portrait(ch) {
                continue;
            }
            let t0 = std::time::Instant::now();
            let s = mm_render::sprites::emoji(ch, mm_render::sprites::SPRITE)?;
            s.save(outd.join("sprites").join(format!("{hex}.png")))?;
            eprintln!("{hex} {:?}", t0.elapsed());
        }
        return Ok(());
    }
    let eyes = Eyes::from_json(&std::fs::read_to_string(cfg["eyes"].as_str().unwrap())?)?;
    let meta = RenderMeta {
        tex: cfg["TEX"].as_u64().unwrap() as u32,
        slices: cfg["SLICES"].as_u64().unwrap() as u32,
        sigma_s: cfg["SIGMA_S"].as_f64().unwrap() as f32,
        sigma_r: cfg["SIGMA_R"].as_f64().unwrap() as f32,
        face_zone: [cfg["FACE_ZONE"][0].as_f64().unwrap() as f32, cfg["FACE_ZONE"][1].as_f64().unwrap() as f32],
        face_dy: cfg["FACE_DY"].as_f64().unwrap() as f32,
    };
    let configs: Vec<String> = match arg(&args, "--configs") {
        Some(c) => c.split(',').map(String::from).collect(),
        None => cfg["configs"].as_object().unwrap().keys().cloned().collect(),
    };
    let py_sprites = args.iter().any(|a| a == "--py-sprites");
    let write_rows = args.iter().any(|a| a == "--rows");
    let photo_path = cfg["portrait_photo"].as_str().map(String::from);
    let sprite_dir = refd.join("sprites");
    let key_hex: HashMap<String, String> = keys.iter().map(|(h, k)| (k.clone(), h.clone())).collect();
    let pictures: mm_render::draw::PictureSource = Box::new(move |key: &str| {
        if py_sprites || !cfg!(target_os = "macos") {
            let h = key_hex.get(key).ok_or_else(|| anyhow!("no sprite for {key}"))?;
            return Ok(image::open(sprite_dir.join(format!("{h}.png")))?.to_rgba8());
        }
        #[cfg(target_os = "macos")]
        {
            let pp = photo_path.clone();
            mm_render::sprites::for_key(key, &|_k| {
                let p = pp.as_ref().ok_or_else(|| anyhow!("no portrait photo"))?;
                Ok(image::open(p)?.to_rgba8())
            })
        }
        #[cfg(not(target_os = "macos"))]
        unreachable!()
    });
    let gpu = mm_render::Gpu::headless()?;
    let fmt = wgpu::TextureFormat::Rgba8Unorm;
    let mut drawer = Drawer::new(&gpu.device, &gpu.queue, &meta, fmt, eyes, pictures);
    let mark_cfg = mm_render::mark::MarkStyle::from_brand(&cfg)?;
    let mut marker: Option<mm_render::mark::Marker> = None;
    for c in &configs {
        let frames: Vec<u64> = match arg(&args, "--frames") {
            Some(f) => f.split(',').map(|x| x.parse().unwrap()).collect(),
            None => cfg["configs"][c]["frames"].as_array().ok_or_else(|| anyhow!("no config {c}"))?
                .iter().map(|v| v.as_u64().unwrap()).collect(),
        };
        std::fs::create_dir_all(outd.join(c))?;
        for t in frames {
            let rp = refd.join(c).join(format!("{t:05}.json"));
            let rec: FrameRecord = serde_json::from_slice(&std::fs::read(&rp).with_context(|| rp.display().to_string())?)?;
            let rv: serde_json::Value = serde_json::from_slice(&std::fs::read(&rp)?)?;
            let near = match rv["near"].as_str() {
                Some(f) => {
                    let b = std::fs::read(refd.join(f))?;
                    Some((Arc::new(bytemuck::pod_collect_to_vec::<u8, f32>(&b)), rv["mw"].as_u64().unwrap() as u32,
                          rv["mh"].as_u64().unwrap() as u32))
                }
                None => None,
            };
            let img = image::open(refd.join("in").join(format!("{t:05}.png")))?.to_rgba8();
            let src = mm_render::compositor::upload_rgba(&gpu.device, &gpu.queue, &img);
            if write_rows && (rec.style == "emoji" || rec.style == "portrait") {
                let (rows, pics) = drawer.rows(&rec)?;
                std::fs::write(outd.join(c).join(format!("{t:05}.rows.json")),
                               serde_json::to_string(&serde_json::json!({"rows": rows, "pics": pics}))?)?;
            }
            let mut out = drawer.draw(&gpu.device, &gpu.queue, &src, &rec, near)?;
            if rv["mark"].as_bool() == Some(true) {
                let m = marker.get_or_insert_with(|| mm_render::mark::Marker::new(&gpu.device, fmt, mark_cfg.clone()));
                out = m.draw(&gpu.device, &gpu.queue, out)?;
            }
            save(&gpu, &out, &outd.join(c).join(format!("{t:05}.png")))?;
        }
        eprintln!("{c}: done");
    }
    Ok(())
}

fn save(gpu: &mm_render::Gpu, tex: &wgpu::Texture, p: &Path) -> Result<()> {
    let px = mm_render::readback::read_texture(&gpu.device, &gpu.queue, tex);
    let (w, h) = (tex.width(), tex.height());
    let rgb: Vec<u8> = px.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
    image::RgbImage::from_raw(w, h, rgb).unwrap().save(p)?;
    Ok(())
}
