//! The PoC dump (native/tools/dump_reference.py): meta.json, frames.json, covers.bin (the engine's cover rows),
//! atlas/*.png (cover pictures), near/*.f32 (depth guides). A folder on disk.
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use mm_core::cover::{FrameCovers, RenderMeta, Slice};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Deserialize, Clone, Debug)]
pub struct Meta {
    #[serde(rename = "W")]
    pub w: u32,
    #[serde(rename = "H")]
    pub h: u32,
    pub fps: f64,
    pub n: usize,
    #[serde(rename = "STRIDE")]
    pub stride: usize,
    #[serde(rename = "TEX")]
    pub tex: u32,
    #[serde(rename = "SLICES")]
    pub slices: u32,
    #[serde(rename = "SIGMA_S")]
    pub sigma_s: f32,
    #[serde(rename = "SIGMA_R")]
    pub sigma_r: f32,
    #[serde(rename = "FACE_ZONE")]
    pub face_zone: [f32; 2],
    #[serde(rename = "FACE_DY")]
    pub face_dy: f32,
    pub pictures: HashMap<String, PicMeta>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct PicMeta {
    pub file: String,
    pub size: u32,
}

#[derive(Deserialize, Clone, Debug)]
pub struct FrameRec {
    pub t: usize,
    pub n: usize,
    pub offset: usize,
    pub seed: f32,
    pub soft: f32,
    pub pics: Vec<String>,
    pub mw: u32,
    pub mh: u32,
    pub near: Option<String>,
    pub circles: Vec<[f32; 3]>,
}

pub struct Dump {
    pub dir: PathBuf,
    pub meta: Meta,
    pub frames: Vec<FrameRec>,
    pub covers: Vec<f32>,
    slices: Mutex<HashMap<String, Arc<Slice>>>,
    nears: Mutex<HashMap<usize, Arc<Vec<f32>>>>,
}

impl Dump {
    pub fn load(dir: &str) -> Result<Dump> {
        let dir = PathBuf::from(dir.trim_end_matches('/'));
        let get = |rel: &str| std::fs::read(dir.join(rel)).with_context(|| format!("{}/{rel}", dir.display()));
        let meta: Meta = serde_json::from_slice(&get("meta.json")?)?;
        let frames: Vec<FrameRec> = serde_json::from_slice(&get("frames.json")?)?;
        let cb = get("covers.bin")?;
        let covers: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&cb).to_vec();
        Ok(Dump { dir, meta, frames, covers, slices: Mutex::new(HashMap::new()), nears: Mutex::new(HashMap::new()) })
    }

    fn get(&self, rel: &str) -> Result<Vec<u8>> {
        std::fs::read(self.dir.join(rel)).with_context(|| format!("{}/{rel}", self.dir.display()))
    }

    /// The compositor's constants.
    pub fn render_meta(&self) -> RenderMeta {
        let m = &self.meta;
        RenderMeta { tex: m.tex, slices: m.slices, sigma_s: m.sigma_s, sigma_r: m.sigma_r, face_zone: m.face_zone, face_dy: m.face_dy }
    }

    /// Frame t's covers for the compositor, with `rows` (edited) in place of the engine's when given.
    pub fn frame_covers(&self, t: usize, rows: Option<Vec<f32>>) -> Result<FrameCovers> {
        let t = t.min(self.frames.len() - 1);
        let f = &self.frames[t];
        Ok(FrameCovers {
            rows: rows.unwrap_or_else(|| self.rows(t).to_vec()),
            pics: f.pics.clone(),
            seed: f.seed,
            soft: f.soft,
            near: self.near(t)?.map(|v| (v, f.mw, f.mh)),
        })
    }

    /// Engine rows of frame t (n * STRIDE floats).
    pub fn rows(&self, t: usize) -> &[f32] {
        let f = &self.frames[t];
        let s = self.meta.stride;
        &self.covers[f.offset * s..(f.offset + f.n) * s]
    }

    pub fn near(&self, t: usize) -> Result<Option<Arc<Vec<f32>>>> {
        let f = &self.frames[t];
        let Some(p) = &f.near else { return Ok(None) };
        if let Some(v) = self.nears.lock().unwrap().get(&t) {
            return Ok(Some(v.clone()));
        }
        let b = self.get(p)?;
        let v = Arc::new(bytemuck::pod_collect_to_vec::<u8, f32>(&b));
        self.nears.lock().unwrap().insert(t, v.clone());
        Ok(Some(v))
    }

    pub fn slice(&self, hash: &str) -> Result<Arc<Slice>> {
        if let Some(s) = self.slices.lock().unwrap().get(hash) {
            return Ok(s.clone());
        }
        let pm = &self.meta.pictures[hash];
        let png = self.get(&pm.file)?;
        let img = image::load_from_memory(&png)?.to_rgba8();
        let s = Arc::new(make_slice(&img, self.meta.tex));
        self.slices.lock().unwrap().insert(hash.to_string(), s.clone());
        Ok(s)
    }
}

/// RGBA straight-alpha picture -> atlas slice (as gpu_draw.Compositor.slot + Metal's generateMipmaps, approximately:
/// mips here are a 2x2 box filter computed in f32 from the f16 level above).
pub fn make_slice(img: &image::RgbaImage, tex: u32) -> Slice {
    let img = if img.width() > tex {
        image::imageops::resize(img, tex, tex, image::imageops::FilterType::Triangle)
    } else {
        img.clone()
    };
    let n = img.width();
    let t = tex as usize;
    let mut l0 = vec![half::f16::ZERO; t * t * 4];
    for y in 0..n as usize {
        for x in 0..n as usize {
            let p = img.get_pixel(x as u32, y as u32).0;
            let a = p[3] as f32 / 255.0;
            let i = (y * t + x) * 4;
            // BGR, premultiplied (numpy: float32 / 255 * a, then float16)
            l0[i] = half::f16::from_f32(p[2] as f32 / 255.0 * a);
            l0[i + 1] = half::f16::from_f32(p[1] as f32 / 255.0 * a);
            l0[i + 2] = half::f16::from_f32(p[0] as f32 / 255.0 * a);
            l0[i + 3] = half::f16::from_f32(a);
        }
    }
    let mut levels = vec![l0];
    let mut s = t;
    while s > 1 {
        let ns = (s / 2).max(1);
        let prev = levels.last().unwrap();
        let mut nl = vec![half::f16::ZERO; ns * ns * 4];
        for y in 0..ns {
            for x in 0..ns {
                for c in 0..4 {
                    let mut acc = 0.0f32;
                    let mut k = 0.0;
                    for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        let sy = (y * 2 + dy).min(s - 1);
                        let sx = (x * 2 + dx).min(s - 1);
                        acc += prev[(sy * s + sx) * 4 + c].to_f32();
                        k += 1.0;
                    }
                    nl[(y * ns + x) * 4 + c] = half::f16::from_f32(acc / k);
                }
            }
        }
        levels.push(nl);
        s = ns;
    }
    Slice { n, levels }
}
