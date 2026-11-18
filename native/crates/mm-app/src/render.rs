//! Drawing the shown frame: decoded planes -> RGB (mm-render yuv + prep) -> covers composited -> the target texture.
//!
//! In the app the target is the Qt Quick item's texture and the device is Qt's own (adopted on the first call, on
//! Qt's render thread; `mm_app_render`). Qt's texture can't be a storage texture, so we composite into our own and
//! blit (one GPU copy, no CPU copies). Headless (mm-cli) the device and target are our own.
//! Work is submitted on the device's queue; the host commits its frame after we return, so Metal orders it after ours.
use crate::{log, Shared};
use anyhow::{bail, Result};
use mm_core::cover::FrameCovers;
use mm_render::compositor::Compositor;
use mm_render::yuv::Yuv;
use mm_render::Gpu;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::time::Instant;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RenderInfo {
    pub shown_t: i32,
    pub new_frame: i32,
    pub composited: i32,
    pub composite_ms: f64,
    pub decode_ms: f64,
    pub latency_ms: f64,
}

pub struct GpuState {
    /// the host queue we adopted (a new window brings a new one)
    pub host_queue: usize,
    pub gpu: Gpu,
    pub report: String,
    importer: mm_media::import::Importer,
    yuv: Option<(u64, Yuv)>,
    comp: Option<(u64, Compositor)>,
    out: Option<wgpu::Texture>,
    target: Option<(usize, wgpu::Texture, bool)>,
    /// (gen, frame) in `yuv.display`
    shown: Option<(u64, usize)>,
    /// draws a project's records (records.rs): (project, records key)
    drawer: Option<(String, mm_render::draw::Drawer)>,
}

impl GpuState {
    pub fn new(gpu: Gpu, host_queue: usize, report: String) -> Result<GpuState> {
        Ok(GpuState {
            importer: mm_media::import::Importer::new(&gpu.device)?,
            gpu,
            host_queue,
            report,
            yuv: None,
            comp: None,
            out: None,
            target: None,
            shown: None,
            drawer: None,
        })
    }
}

impl GpuState {
    pub fn target_key(&self) -> Option<usize> {
        self.target.as_ref().map(|t| t.0)
    }
}

/// Timings for /perf: the last 240 of each.
#[derive(Default)]
pub struct Perf {
    pub composite_ms: VecDeque<f64>,
    pub decode_ms: VecDeque<f64>,
    pub latency_ms: VecDeque<f64>,
    /// the grain thread (decoder.rs): the frame to BGR, then frame_grain
    pub grain_convert_ms: VecDeque<f64>,
    pub grain_ms: VecDeque<f64>,
    pub composites: u64,
    pub new_frames: u64,
    pub zero_copy: u64,
    pub uploads: u64,
    pub gpu: String,
}

pub fn push(q: &mut VecDeque<f64>, v: f64) {
    if q.len() >= 240 {
        q.pop_front();
    }
    q.push_back(v);
}

pub fn stats(q: &VecDeque<f64>) -> Value {
    if q.is_empty() {
        return json!(null);
    }
    let mut v: Vec<f64> = q.iter().cloned().collect();
    v.sort_by(f64::total_cmp);
    let at = |p: f64| (v[((v.len() - 1) as f64 * p).round() as usize] * 100.0).round() / 100.0;
    json!({"n": v.len(), "median": at(0.5), "p95": at(0.95), "max": at(1.0)})
}

impl Perf {
    pub fn summary(&self) -> Value {
        json!({
            "composite_ms": stats(&self.composite_ms), "decode_ms": stats(&self.decode_ms),
            "request_to_composited_ms": stats(&self.latency_ms),
            "grain_convert_ms": stats(&self.grain_convert_ms), "grain_ms": stats(&self.grain_ms), "composites": self.composites,
            "new_frames": self.new_frames, "zero_copy_frames": self.zero_copy, "uploaded_frames": self.uploads,
            "gpu": self.gpu,
        })
    }
}

/// Draw into `target` (an RGBA8 texture the video's size; `writable` = usable as a storage texture).
pub fn draw(sh: &Shared, g: &mut GpuState, target_key: usize, target: Option<(wgpu::Texture, bool)>) -> Result<RenderInfo> {
    let mut res = RenderInfo { shown_t: -1, ..Default::default() };
    let gen = sh.gen();
    let probe = sh.probe.lock().unwrap().clone().filter(|(pg, _)| *pg == gen).map(|(_, p)| p);
    let newest = sh.newest.lock().unwrap().take().filter(|d| d.gen == gen);
    let dirty = sh.dirty.swap(false, Ordering::SeqCst);
    let readback = sh.readback.lock().unwrap().take();
    let Some(p) = probe else {
        if let Some(r) = readback {
            *sh.readback.lock().unwrap() = Some(r);
        }
        return Ok(res);
    };
    let device = g.gpu.device.clone();
    let queue = g.gpu.queue.clone();
    let target_changed = g.target.as_ref().is_none_or(|t| t.0 != target_key);
    if target_changed {
        if let Some((t, w)) = target {
            g.target = Some((target_key, t, w));
        }
    }
    if g.yuv.as_ref().is_none_or(|(yg, _)| *yg != gen) {
        g.yuv = Some((gen, Yuv::new(&device, (p.stored_w, p.stored_h), (p.width, p.height), p.rotation)));
        g.shown = None;
    }
    let dump = {
        let pj = sh.project.lock().unwrap();
        let fit = pj.as_ref().is_some_and(|x| x.covers_fit(p.width, p.height));
        pj.as_ref().and_then(|x| x.dump.clone()).filter(|_| fit)
    };
    if let Some(d) = &dump {
        if g.comp.as_ref().is_none_or(|(cg, _)| *cg != gen) {
            g.comp = Some((gen, Compositor::new(&device, &queue, &d.render_meta(), wgpu::TextureFormat::Rgba8Unorm)));
        }
    } else if g.comp.as_ref().is_none_or(|(cg, _)| *cg != gen) {
        // no covers: the compositor still copies the frame (with an empty atlas)
        let meta = mm_core::cover::RenderMeta { tex: 4, slices: 1, sigma_s: 1.0, sigma_r: 1.0, face_zone: [0.0, 1.0], face_dy: 0.0 };
        g.comp = Some((gen, Compositor::new(&device, &queue, &meta, wgpu::TextureFormat::Rgba8Unorm)));
    }
    let t0 = Instant::now();
    let mut enc = device.create_command_encoder(&Default::default());
    let mut new_src = None;
    if let Some(d) = newest {
        let planes = g.importer.planes(&device, &queue, &d.frame)?;
        g.yuv.as_ref().unwrap().1.encode(&device, &mut enc, &planes.y, &planes.cbcr, &planes.params);
        g.shown = Some((gen, d.t));
        let mut perf = sh.perf.lock().unwrap();
        if planes.zero_copy {
            perf.zero_copy += 1;
        } else {
            perf.uploads += 1;
        }
        new_src = Some(d);
    }
    let Some((_, st)) = g.shown.filter(|(sg, _)| *sg == gen) else { return Ok(res) };
    res.shown_t = st as i32;
    if !(new_src.is_some() || dirty || target_changed || readback.is_some()) {
        return Ok(res);
    }
    let Some((_, ttex, writable)) = &g.target else { bail!("no target texture") };
    if ttex.width() != p.width || ttex.height() != p.height {
        // the host hasn't sized its texture to the video yet: draw when it has
        sh.dirty.store(true, Ordering::SeqCst);
        if let Some(r) = readback {
            *sh.readback.lock().unwrap() = Some(r);
        }
        if new_src.is_some() {
            queue.submit([enc.finish()]);
        }
        res.shown_t = -1;
        return Ok(res);
    }
    let recs = {
        let st = sh.st.lock().unwrap();
        let (pw, ph) = (st.project.w as u32, st.project.h as u32);
        drop(st);
        sh.recs.lock().unwrap().clone().filter(|r| !r.frames.is_empty() && (pw, ph) == (p.width, p.height))
    };
    if !*writable && g.out.as_ref().is_none_or(|o| o.width() != p.width || o.height() != p.height) {
        g.out = Some(mm_render::out_texture(&device, p.width, p.height, wgpu::TextureFormat::Rgba8Unorm));
    }
    if let Some(r) = recs {
        // a project: its records, drawn as the export draws them (mm-render's Drawer)
        queue.submit([enc.finish()]);
        // a new drawer when the pictures it can draw change (a portrait photo chosen: the same count, another file)
        let mut photos: Vec<_> = r.photos.iter().map(|(k, v)| format!("{k}={v}")).collect();
        photos.sort();
        let key = format!("{}|{}|{}", r.pid, r.setup.portrait_photo, photos.join("|"));
        if g.drawer.as_ref().is_none_or(|(k, _)| *k != key) {
            let eyes = crate::records::eyes(&r.setup)?;
            let pics = crate::records::pictures(r.photos.clone(), r.setup.portrait_photo.clone());
            g.drawer = Some((key, mm_render::draw::Drawer::new(&device, &queue, &r.setup.meta, wgpu::TextureFormat::Rgba8Unorm, eyes, pics)));
        }
        let (rec, near) = r.get(st).ok_or_else(|| anyhow::anyhow!("no record for frame {st}"))?;
        let mut rec = crate::editor::live_record(sh, rec);
        if let Some(l) = rec.light.as_mut() {
            l.grain = if l.settings.grain { sh.grain.for_frame(gen, st).unwrap_or(0.0) } else { 0.0 };
        }
        let display = &g.yuv.as_ref().unwrap().1.display;
        let drawn = g.drawer.as_mut().unwrap().1.draw(&device, &queue, display, &rec, near)?;
        enc = device.create_command_encoder(&Default::default());
        let Some((_, ttex, _)) = &g.target else { bail!("no target texture") };
        enc.copy_texture_to_texture(drawn.as_image_copy(), ttex.as_image_copy(),
                                    wgpu::Extent3d { width: p.width, height: p.height, depth_or_array_layers: 1 });
    } else {
    let Some((_, ttex, writable)) = &g.target else { bail!("no target texture") };
    let fc = match (&dump, sh.project.lock().unwrap().as_ref()) {
        (Some(_), Some(pj)) => pj.frame_covers(st)?,
        _ => FrameCovers::default(),
    };
    let comp = &mut g.comp.as_mut().unwrap().1;
    let prepd = match &dump {
        Some(d) => comp.prepare(&fc, &|h| d.slice(h))?,
        None => comp.prepare(&fc, &|_| bail!("no pictures"))?,
    };
    let dst = if *writable { ttex } else { g.out.as_ref().unwrap() };
    let display = &g.yuv.as_ref().unwrap().1.display;
    comp.encode(&mut enc, &display.create_view(&Default::default()), &dst.create_view(&Default::default()), p.width, p.height, &prepd);
    if !*writable {
        enc.copy_texture_to_texture(
            dst.as_image_copy(),
            ttex.as_image_copy(),
            wgpu::Extent3d { width: p.width, height: p.height, depth_or_array_layers: 1 },
        );
    }
    }
    let Some((_, ttex, _)) = &g.target else { bail!("no target texture") };
    queue.submit([enc.finish()]);
    res.composite_ms = t0.elapsed().as_secs_f64() * 1e3;
    res.composited = 1;
    {
        let mut perf = sh.perf.lock().unwrap();
        perf.composites += 1;
        push(&mut perf.composite_ms, res.composite_ms);
        if let Some(d) = &new_src {
            res.new_frame = 1;
            res.decode_ms = d.decode_ms;
            res.latency_ms = d.asked.elapsed().as_secs_f64() * 1e3;
            perf.new_frames += 1;
            push(&mut perf.decode_ms, d.decode_ms);
            push(&mut perf.latency_ms, res.latency_ms);
        }
        perf.gpu = g.report.clone();
    }
    if let Some(path) = readback {
        let r = if sh.st.lock().unwrap().editor.video.public {
            mm_render::readback::save_png(&device, &queue, ttex, false, &path).map(|_| "done".to_string())
        } else {
            Ok("refused".to_string())
        };
        let state = r.unwrap_or_else(|e| format!("failed: {e:#}"));
        log::info(&format!("readback of frame {st} -> {path}: {state}"));
        let mut s = sh.st.lock().unwrap();
        s.readback = crate::state::Readback { state, path, frame: st as i64 };
        drop(s);
        sh.changed(&["readback"]);
    }
    if new_src.is_some() {
        let covers = sh.project.lock().unwrap().as_ref().map(|pj| pj.covers(st)).unwrap_or_default();
        {
            let mut s = sh.st.lock().unwrap();
            s.editor.shown = st as i64;
            s.editor.covers = if s.editor.covers_fit { covers } else { vec![] };
        }
        crate::editor::on_shown(sh, st);
        sh.changed(&["editor.shown", "editor.covers"]);
    }
    g.importer.flush();
    Ok(res)
}

/// Headless: draw the current frame into our own texture and save it (mm-cli).
pub fn draw_headless(sh: &Shared, path: &str) -> Result<RenderInfo> {
    let mut gl = sh.gpu.lock().unwrap();
    if gl.is_none() {
        let gpu = Gpu::headless()?;
        *gl = Some(GpuState::new(gpu, 0, "headless".into())?);
    }
    let g = gl.as_mut().unwrap();
    let (w, h) = {
        let st = sh.st.lock().unwrap();
        (st.editor.video.width, st.editor.video.height)
    };
    if w == 0 {
        bail!("no video ready");
    }
    if !sh.st.lock().unwrap().editor.video.public {
        bail!("refused: the open video isn't a public-domain clip (AGENTS.md, Privacy)");
    }
    let key = (w as usize) << 32 | h as usize;
    let tex = if g.target.as_ref().is_some_and(|t| t.0 == key) {
        None
    } else {
        Some((mm_render::out_texture(&g.gpu.device, w, h, wgpu::TextureFormat::Rgba8Unorm), true))
    };
    sh.dirty.store(true, Ordering::SeqCst);
    let r = draw(sh, g, key, tex)?;
    if r.shown_t >= 0 {
        let t = &g.target.as_ref().unwrap().1;
        mm_render::readback::save_png(&g.gpu.device, &g.gpu.queue, t, false, path)?;
    }
    Ok(r)
}
