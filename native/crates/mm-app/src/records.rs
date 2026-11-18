//! The open project's draw records: what the Editor and the export draw from.
//!
//! The project's server (native/tools/sidecar/project_server.py: review_server.py plus `/api/native/records`) decides
//! every frame's covers exactly as Python's export does (review_server.render_setup, track.draw_order,
//! depth.Occlusion.plan): the head box, the emoji or portrait, the head pose, the lighting estimate and the cover's
//! velocity, the depth cut. mm-render's `Drawer` draws them (track.draw_frame on the GPU). The one thing measured
//! here rather than there is lighting's grain (lighting.frame_grain), because it is read off the very frame drawn.
//!
//! Records are fetched again whenever the project's edits change (their revision), and for the export with the
//! lighting estimate waited for (`wait=1`), as review_server.do_render does.
use crate::library::sidecar;
use crate::{log, Shared};
use anyhow::{anyhow, Context, Result};
use mm_core::cover::{Eyes, FrameRecord, RenderMeta};
use mm_vision::mat::Mat;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// The drawing constants and files (`/api/native/setup`).
#[derive(Clone)]
pub struct Setup {
    pub meta: RenderMeta,
    pub eyes_json: String,
    pub portrait_photo: String,
    /// brand + font_file: the corner mark (mm_render::mark::MarkStyle::from_brand)
    pub brand: Value,
    pub depth: bool,
}

pub struct Recs {
    pub pid: String,
    /// what the records were made from: the edits' revision and the style
    pub key: String,
    pub rev: Value,
    pub style: String,
    pub light_ready: bool,
    pub frames: Vec<FrameRecord>,
    /// the depth guide per frame (values, mw, mh) where a cover is cut
    pub near: Vec<Option<(Arc<Vec<f32>>, u32, u32)>>,
    /// portrait keys -> their photo
    pub photos: HashMap<String, String>,
    pub setup: Setup,
}

impl Recs {
    pub fn get(&self, t: usize) -> Option<(&FrameRecord, Option<(Arc<Vec<f32>>, u32, u32)>)> {
        let t = t.min(self.frames.len().checked_sub(1)?);
        Some((&self.frames[t], self.near[t].clone()))
    }
}

pub fn setup_of(v: &Value) -> Result<Setup> {
    let f = |k: &str| v[k].as_f64().ok_or_else(|| anyhow!("setup: {k}"));
    let meta = RenderMeta {
        tex: f("TEX")? as u32,
        slices: f("SLICES")? as u32,
        sigma_s: f("SIGMA_S")? as f32,
        sigma_r: f("SIGMA_R")? as f32,
        face_zone: [v["FACE_ZONE"][0].as_f64().unwrap_or(0.4) as f32, v["FACE_ZONE"][1].as_f64().unwrap_or(0.55) as f32],
        face_dy: f("FACE_DY")? as f32,
    };
    let eyes_path = v["eyes"].as_str().unwrap_or_default();
    Ok(Setup {
        meta,
        eyes_json: std::fs::read_to_string(eyes_path).with_context(|| eyes_path.to_string())?,
        portrait_photo: v["portrait_photo"].as_str().unwrap_or_default().to_string(),
        brand: serde_json::json!({"brand": v["brand"], "font_file": v["font_file"]}),
        depth: v["depth"].as_bool().unwrap_or(false),
    })
}

/// Fetch every frame's record for the project `pid` (`style`: None = the edits' own; `wait`: for the lighting estimate).
pub fn fetch(sh: &Shared, pid: &str, style: Option<&str>, wait: bool) -> Result<Recs> {
    fetch_with(sh, pid, style, wait, false)
}

/// `strict`: higher-risk mode (solid covers over whole heads, review_server.render_setup).
pub fn fetch_with(sh: &Shared, pid: &str, style: Option<&str>, wait: bool, strict: bool) -> Result<Recs> {
    let sc = sidecar(sh)?;
    if let Some(srv) = sc.native() {
        return fetch_native(&srv, pid, style, wait, strict);
    }
    let base = format!("/p/{pid}/api/native");
    let sv = sc.get_slow(&format!("{base}/setup"))?;
    let setup = setup_of(&sv)?;
    let n = sv["n"].as_u64().unwrap_or(0) as usize;
    let mut frames = Vec::with_capacity(n);
    let mut near = Vec::with_capacity(n);
    let mut photos = HashMap::new();
    let (mut key, mut rev, mut st, mut light_ready) = (String::new(), Value::Null, String::new(), false);
    let mut lo = 0;
    while lo < n.max(1) {
        let hi = (lo + 400).min(n);
        let mut q = format!("{base}/records?from={lo}&to={hi}&wait={}", if wait { 1 } else { 0 });
        if let Some(s) = style {
            q.push_str(&format!("&style={s}"));
        }
        if strict {
            q.push_str("&strict=1");
        }
        let r = sc.get_slow(&q)?;
        if lo > 0 && r["key"].as_str() != Some(key.as_str()) {
            // the edits changed while we read: start again
            return fetch_with(sh, pid, style, wait, strict);
        }
        key = r["key"].as_str().unwrap_or_default().to_string();
        rev = r["rev"].clone();
        st = r["style"].as_str().unwrap_or("emoji").to_string();
        light_ready = r["light_ready"].as_bool().unwrap_or(false);
        for x in r["records"].as_array().into_iter().flatten() {
            if let Some(p) = x["photos"].as_object() {
                for (k, v) in p {
                    photos.insert(k.clone(), v.as_str().unwrap_or_default().to_string());
                }
            }
            near.push(match x["near"].as_str() {
                Some(b) => {
                    let bytes = b64(b)?;
                    Some((Arc::new(bytemuck_f32(&bytes)), x["mw"].as_u64().unwrap_or(1) as u32, x["mh"].as_u64().unwrap_or(1) as u32))
                }
                None => None,
            });
            frames.push(serde_json::from_value::<FrameRecord>(x.clone()).context("a frame record")?);
        }
        if n == 0 {
            break;
        }
        lo = hi;
    }
    Ok(Recs { pid: pid.to_string(), key, rev, style: st, light_ready, frames, near, photos, setup })
}

/// The records straight from the native engine's mm-records (no JSON, no HTTP).
fn fetch_native(srv: &Arc<mm_engine::native::Server>, pid: &str, style: Option<&str>, wait: bool, strict: bool) -> Result<Recs> {
    let p = srv.project(pid).map_err(|f| anyhow!("{pid}: {}", f.1))?;
    let setup = setup_of(&p.setup(srv))?;
    let ask = mm_records::records::Ask { style: style.map(String::from), strict, wait };
    let (frames, near, photos, info) = p.frame_records(&ask)?;
    let near = near.into_iter().map(|n| n.map(|n| (Arc::new(n.data), n.w as u32, n.h as u32))).collect();
    Ok(Recs {
        pid: pid.to_string(),
        key: info["key"].as_str().unwrap_or_default().to_string(),
        rev: info["rev"].clone(),
        style: info["style"].as_str().unwrap_or("emoji").to_string(),
        light_ready: info["light_ready"].as_bool().unwrap_or(false),
        frames,
        near,
        photos,
        setup,
    })
}

/// Frame t of a project drawn as the export draws it, on the project's own JPEG (review_server.preview_frame, for
/// /api/thumb): (w, h, BGR).
pub fn draw_project_frame(sh: &Shared, pid: &str, t: usize, style: &str) -> Result<(usize, usize, Vec<u8>)> {
    let sc = sidecar(sh)?;
    let srv = sc.native().ok_or_else(|| anyhow!("not the native engine"))?;
    let p = srv.project(pid).map_err(|f| anyhow!("{pid}: {}", f.1))?;
    let d = mm_vision::jpeg::read_bgr(&p.work.join("frames").join(format!("{t:05}.jpg")))?;
    let recs = fetch_native(&srv, pid, Some(style), false, false)?;
    let g = mm_render::Gpu::headless()?;
    let (device, queue) = (&g.device, &g.queue);
    let pics = pictures(recs.photos.clone(), recs.setup.portrait_photo.clone());
    let mut drawer = mm_render::draw::Drawer::new(device, queue, &recs.setup.meta, wgpu::TextureFormat::Rgba8Unorm, eyes(&recs.setup)?, pics);
    let Some((rec, near)) = recs.get(t) else { return Ok((d.w, d.h, d.data)) };
    let mut rec = rec.clone();
    if let Some(l) = rec.light.as_mut() {
        l.grain = if l.settings.grain { frame_grain(d.w, d.h, &d.data) } else { 0.0 };
    }
    let mut rgba = image::RgbaImage::new(d.w as u32, d.h as u32);
    for (i, px) in rgba.pixels_mut().enumerate() {
        *px = image::Rgba([d.data[i * 3 + 2], d.data[i * 3 + 1], d.data[i * 3], 255]);
    }
    let src = mm_render::compositor::upload_rgba(device, queue, &rgba);
    let drawn = drawer.draw(device, queue, &src, &rec, near)?;
    let px = mm_render::readback::read_texture(device, queue, &drawn);
    let mut bgr = Vec::with_capacity(d.w * d.h * 3);
    for c in px.chunks_exact(4) {
        bgr.extend_from_slice(&[c[2], c[1], c[0]]);
    }
    Ok((d.w, d.h, bgr))
}

/// Fetch the open project's records again (after an edit, a style change, the lighting estimate arriving) and redraw.
pub fn refresh(sh: &Arc<Shared>) {
    let pid = sh.lib.lock().unwrap().pid.clone();
    if pid.is_empty() {
        return;
    }
    let t0 = std::time::Instant::now();
    // the Export page in higher-risk mode previews what it will export
    let on_export = sh.st.lock().unwrap().app.page == "export";
    let strict = on_export && sh.lib.lock().unwrap().xp.strict;
    match fetch_with(sh, &pid, None, false, strict) {
        Ok(r) => {
            if sh.lib.lock().unwrap().pid != pid {
                return;
            }
            log::info(&format!("records: {} frames of {pid} ({}, rev {}, light {}) in {:.0} ms", r.frames.len(), r.style, r.rev,
                               if r.light_ready { "ready" } else { "estimating" }, t0.elapsed().as_secs_f64() * 1e3));
            let ready = r.light_ready;
            {
                let mut st = sh.st.lock().unwrap();
                st.editor.style = r.style.clone();
                st.editor.records = r.frames.len();
                st.editor.depth_on = r.setup.depth;
            }
            *sh.recs.lock().unwrap() = Some(Arc::new(r));
            sh.dirty.store(true, std::sync::atomic::Ordering::SeqCst);
            sh.changed(&["editor", "@render"]);
            if !ready {
                // the lighting estimate is still being made: look again in a moment (the preview draws flat till then)
                let s2 = sh.clone();
                std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_secs(2));
                    crate::library::queue(&s2, refresh);
                });
            }
        }
        Err(e) => log::warn(&format!("records: {e:#}")),
    }
}

fn b64(s: &str) -> Result<Vec<u8>> {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut inv = [255u8; 256];
    for (i, &c) in T.iter().enumerate() {
        inv[c as usize] = i as u8;
    }
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0);
    for &c in s.as_bytes() {
        if c == b'=' {
            break;
        }
        let v = inv[c as usize];
        if v == 255 {
            return Err(anyhow!("bad base64"));
        }
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

fn bytemuck_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// lighting.frame_grain: robust noise sigma (8-bit units) of a BGR frame: the median deviation from a slightly
/// blurred copy (GaussianBlur sigma 1), read on 16 tiles spread over the frame (frames over 1000 px tall: the tiles
/// halved first), x 1.4826.
pub fn frame_grain(w: usize, h: usize, bgr: &[u8]) -> f64 {
    use mm_vision::gauss::gaussian_blur_f32;
    use mm_vision::resize::{resize_u8, Interp};
    let (th, tw) = ((h / 8).max(16), (w / 8).max(16));
    let img = Mat::from_vec(h, w, 3, bgr.to_vec());
    let mut ds: Vec<f32> = Vec::new();
    for i in 0..4 {
        for j in 0..4 {
            let y = ((2 * i + 1) * h / 8).saturating_sub(th / 2);
            let x = ((2 * j + 1) * w / 8).saturating_sub(tw / 2);
            let (gw, gh) = (tw.min(w - x), th.min(h - y));
            let mut g = img.roi(x, y, gw, gh);
            if h > 1000 {
                g = resize_u8(&g, 0, 0, 0.5, 0.5, Interp::Linear);
            }
            let gf = Mat::from_vec(g.rows, g.cols, 3, g.data.iter().map(|&v| v as f32).collect());
            let b = gaussian_blur_f32(&gf, 0, 0, 1.0, 1.0);
            for r in 2..gf.rows.saturating_sub(2) {
                for c in 2..gf.cols.saturating_sub(2) {
                    for k in 0..3 {
                        let i = (r * gf.cols + c) * 3 + k;
                        ds.push((gf.data[i] - b.data[i]).abs());
                    }
                }
            }
        }
    }
    if ds.is_empty() {
        return 0.0;
    }
    mm_core::np::median(&ds) as f64 * 1.4826
}

/// The picture source for a Drawer: emoji through CoreText, portraits from their photo.
pub fn pictures(photos: HashMap<String, String>, default_photo: String) -> mm_render::draw::PictureSource {
    Box::new(move |key: &str| {
        let path = photos.get(key).cloned().unwrap_or_else(|| default_photo.clone());
        mm_render::sprites::for_key(key, &|_k| Ok(image::open(&path).with_context(|| path.clone())?.to_rgba8()))
    })
}

pub fn eyes(s: &Setup) -> Result<Eyes> {
    Ok(Eyes::from_json(&s.eyes_json)?)
}
