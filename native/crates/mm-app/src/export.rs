//! The export job: decode every slot (mm-media), draw it with its covers and the user's edits (mm-render, on a device
//! of its own), encode with the source's audio copied (mm-media). On the Mac the compositor writes straight into the
//! encoder's IOSurface pixel buffers (zero CPU copies); elsewhere it converts to NV12 and reads back.
use crate::{log, Shared};
use anyhow::{anyhow, bail, Result};
use mm_core::cover::FrameCovers;
use mm_engine::jobs::{Job, JobState};
use mm_media::export::Exporter;
use mm_media::Decoder;
use mm_render::compositor::Compositor;
use mm_render::yuv::Yuv;
use mm_render::Gpu;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;

/// What the Export page chose (review_server.do_render's opts: framing and the corner mark).
#[derive(Clone, Debug)]
pub struct Opts {
    pub preset: String,
    pub fit: String,
    pub start: Option<f64>,
    pub end: Option<f64>,
    pub mark: bool,
    /// higher-risk mode: solid covers over whole heads (render_setup's strict)
    pub strict: bool,
}

impl Default for Opts {
    fn default() -> Self {
        Opts { preset: "original".into(), fit: "fill".into(), start: None, end: None, mark: false, strict: false }
    }
}

pub fn start(sh: &Arc<Shared>, out: &str) -> Result<Value> {
    start_with(sh, out, Opts::default())
}

pub fn start_with(sh: &Arc<Shared>, out: &str, opts: Opts) -> Result<Value> {
    if sh.export.lock().unwrap().get().state == JobState::Running {
        bail!("an export is already running");
    }
    let pid = sh.lib.lock().unwrap().pid.clone();
    if !pid.is_empty() && sh.st.lock().unwrap().editor.project {
        return start_project(sh, out, pid, opts);
    }
    let (video, dump, edited, fit) = {
        let pj = sh.project.lock().unwrap();
        let p = pj.as_ref().ok_or_else(|| anyhow!("no video open"))?;
        let st = sh.st.lock().unwrap();
        (p.video.clone(), p.dump.clone(), p.edited_rows(), st.editor.covers_fit)
    };
    let out = std::path::absolute(out)?.to_string_lossy().into_owned();
    if let Some(d) = std::path::Path::new(&out).parent() {
        std::fs::create_dir_all(d)?;
    }
    let job = Job::default();
    job.set(|p| {
        p.state = JobState::Running;
        p.message = format!("exporting to {out}");
    });
    *sh.export.lock().unwrap() = job.clone();
    publish(sh, &job);
    let sh2 = sh.clone();
    let out2 = out.clone();
    std::thread::Builder::new()
        .name("mm-export".into())
        .spawn(move || {
            let t0 = Instant::now();
            let dump = dump.filter(|_| fit);
            let r = run(&sh2, &job, &video, dump.as_deref(), &edited, &out2);
            let secs = t0.elapsed().as_secs_f64();
            job.set(|p| {
                p.seconds = secs;
                match &r {
                    Ok((n, a)) => {
                        p.state = JobState::Done;
                        p.message = format!("exported {n} frames in {secs:.2} s to {out2} (audio: {a} packets to the muxer)");
                    }
                    Err(e) if job.cancelled() => {
                        p.state = JobState::Cancelled;
                        p.message = format!("export cancelled: {e:#}");
                    }
                    Err(e) => {
                        p.state = JobState::Failed;
                        p.message = format!("export failed: {e:#}");
                    }
                }
            });
            log::info(&job.get().message);
            publish(&sh2, &job);
        })?;
    Ok(json!({"out": out}))
}

fn publish(sh: &Shared, job: &Job) {
    sh.st.lock().unwrap().export = job.get();
    sh.changed(&["export"]);
    if !sh.lib.lock().unwrap().pid.is_empty() {
        publish_page(sh);
    }
}

fn run(sh: &Shared, job: &Job, video: &str, dump: Option<&mm_engine::dump::Dump>,
       edited: &std::collections::BTreeMap<usize, Vec<f32>>, out: &str) -> Result<(usize, usize)> {
    let g = Gpu::headless()?;
    let (device, queue) = (&g.device, &g.queue);
    let mut dec = Decoder::open(video)?;
    let p = dec.probe.clone();
    let (w, h) = (p.width, p.height);
    let importer = mm_media::import::Importer::new(device)?;
    let yuv = Yuv::new(device, (p.stored_w, p.stored_h), (w, h), p.rotation);
    let meta = dump.map(|d| d.render_meta()).unwrap_or(mm_core::cover::RenderMeta {
        tex: 4, slices: 1, sigma_s: 1.0, sigma_r: 1.0, face_zone: [0.0, 1.0], face_dy: 0.0,
    });
    #[cfg(target_os = "macos")]
    let fmt = wgpu::TextureFormat::Bgra8Unorm;
    #[cfg(not(target_os = "macos"))]
    let fmt = wgpu::TextureFormat::Rgba8Unorm;
    let mut comp = Compositor::new(device, queue, &meta, fmt);
    let mut ex = Exporter::new(video, out, &p, 24000)?;
    #[cfg(target_os = "macos")]
    let pool = mm_media::mac::cv::Pool::new(w, h, objc2_core_video::kCVPixelFormatType_32BGRA)?;
    #[cfg(target_os = "macos")]
    let cv = mm_media::mac::cv::Importer::new(device)?;
    #[cfg(not(target_os = "macos"))]
    let (nv12, tmp) = (mm_render::nv12::Nv12::new(device, w, h), mm_render::out_texture(device, w, h, fmt));
    let fview = yuv.display.create_view(&Default::default());
    let mut last_note = Instant::now();
    job.set(|pr| pr.total = p.n);
    while let Some((t, f)) = dec.next()? {
        if job.cancelled() {
            ex.cancel();
            bail!("at frame {t}");
        }
        let fc = match dump {
            Some(d) => d.frame_covers(t, edited.get(&t.min(d.frames.len() - 1)).cloned())?,
            None => FrameCovers::default(),
        };
        let prepd = match dump {
            Some(d) => comp.prepare(&fc, &|hsh| d.slice(hsh))?,
            None => comp.prepare(&fc, &|_| bail!("no pictures"))?,
        };
        let planes = importer.planes(device, queue, &f)?;
        let mut enc = device.create_command_encoder(&Default::default());
        yuv.encode(device, &mut enc, &planes.y, &planes.cbcr, &planes.params);
        #[cfg(target_os = "macos")]
        {
            let pb = pool.get()?;
            let dst = cv.import_plane(device, &pb, 0, wgpu::TextureFormat::Bgra8Unorm, true)?;
            comp.encode(&mut enc, &fview, &dst.create_view(&Default::default()), w, h, &prepd);
            queue.submit([enc.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely())?;
            drop(dst);
            ex.push_pixbuf(t, &pb)?;
        }
        #[cfg(not(target_os = "macos"))]
        {
            let tv = tmp.create_view(&Default::default());
            comp.encode(&mut enc, &fview, &tv, w, h, &prepd);
            nv12.encode(device, &mut enc, &tv, w, h);
            queue.submit([enc.finish()]);
            let sl = nv12.read.slice(..);
            sl.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::PollType::wait_indefinitely())?;
            let bytes = sl.get_mapped_range().map_err(|e| anyhow!("map {e:?}"))?.to_vec();
            nv12.read.unmap();
            ex.push_nv12(t, bytes)?;
        }
        drop(planes);
        job.set(|pr| pr.done = t + 1);
        if last_note.elapsed().as_millis() > 100 {
            last_note = Instant::now();
            publish(sh, job);
        }
    }
    let s = ex.finish()?;
    Ok((s.frames, s.audio_packets))
}

// ---------------------------------------------------------------------------------------------------- a project's export
/// Export a project as review_server.do_render -> track.render does: every frame of video_io's timeline (display
/// colours), its covers drawn from the project's draw records (fetched with the lighting estimate waited for), fitted
/// to the output format (formats.fit_frame), the corner mark if asked (track.watermark), encoded by the OS (vtenc)
/// with the source's audio from the export's first frame. A photo exports as a photo.
fn start_project(sh: &Arc<Shared>, out: &str, pid: String, opts: Opts) -> Result<Value> {
    let video = sh.st.lock().unwrap().project.video.clone();
    let out = std::path::absolute(out)?.to_string_lossy().into_owned();
    let lower = out.to_lowercase();
    let photo = sh.st.lock().unwrap().project.photo;
    let ok_ext: &[&str] = if photo { &[".jpg", ".jpeg", ".png"] } else { &[".mp4", ".m4v", ".mov"] };
    if !ok_ext.iter().any(|e| lower.ends_with(e)) {
        bail!("can't save to {out}: a {} ends in {}", if photo { "photo" } else { "video" }, ok_ext.join(", "));
    }
    if let Some(d) = std::path::Path::new(&out).parent() {
        std::fs::create_dir_all(d)?;
    }
    let job = Job::default();
    job.set(|p| {
        p.state = JobState::Running;
        p.message = format!("exporting to {out}");
        p.out = out.clone();
    });
    *sh.export.lock().unwrap() = job.clone();
    publish(sh, &job);
    let sh2 = sh.clone();
    let out2 = out.clone();
    std::thread::Builder::new()
        .name("mm-export".into())
        .spawn(move || {
            let t0 = Instant::now();
            let r = run_project(&sh2, &job, &pid, &video, &out2, &opts);
            let secs = t0.elapsed().as_secs_f64();
            job.set(|p| {
                p.seconds = secs;
                match &r {
                    Ok(n) => {
                        p.state = JobState::Done;
                        p.message = format!("exported {n} frames in {secs:.2} s to {out2}");
                    }
                    Err(e) if job.cancelled() => {
                        p.state = JobState::Cancelled;
                        p.message = format!("export cancelled: {e:#}");
                    }
                    Err(e) => {
                        p.state = JobState::Failed;
                        p.message = format!("export failed: {e:#}");
                    }
                }
            });
            log::info(&job.get().message);
            publish(&sh2, &job);
        })?;
    Ok(json!({"out": out}))
}

/// The corner mark on a BGR picture, in place (track.watermark).
fn watermark(img: &mut mm_media::convert::Bgr, style: &mm_render::mark::MarkStyle) -> Result<()> {
    let px = mm_render::mark::mark_px(img.w, img.h);
    let sp = mm_render::mark::mark_sprite(style, px)?;
    let Some((x, y)) = mm_render::mark::place(img.w, img.h, sp.w, sp.h, px) else { return Ok(()) };
    let w = img.w as usize;
    for r in 0..sp.h {
        for c in 0..sp.w {
            let i = r * sp.w + c;
            let a = sp.alpha[i];
            let o = ((y as usize + r) * w + x as usize + c) * 3;
            for k in 0..3 {
                // BGR picture, RGB sprite colours
                let v = sp.col[i][2 - k] + img.data[o + k] as f32 * (1.0 - a);
                img.data[o + k] = v.clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(())
}

fn bgr_to_rgba(img: &mm_media::convert::Bgr) -> image::RgbaImage {
    let mut v = Vec::with_capacity(img.data.len() / 3 * 4);
    for p in img.data.chunks_exact(3) {
        v.extend_from_slice(&[p[2], p[1], p[0], 255]);
    }
    image::RgbaImage::from_raw(img.w, img.h, v).unwrap()
}

fn rgba_to_bgr(px: &[u8], w: u32, h: u32) -> mm_media::convert::Bgr {
    let mut d = Vec::with_capacity((w * h * 3) as usize);
    for p in px.chunks_exact(4) {
        d.extend_from_slice(&[p[2], p[1], p[0]]);
    }
    mm_media::convert::Bgr { w, h, data: d }
}

pub(crate) fn run_project(sh: &Shared, job: &Job, pid: &str, video: &str, out: &str, o: &Opts) -> Result<usize> {
    job.set(|p| p.message = "preparing (the lighting estimate, the covers)".into());
    publish(sh, job);
    let recs = crate::records::fetch_with(sh, pid, None, true, o.strict)?;
    let g = Gpu::headless()?;
    let (device, queue) = (&g.device, &g.queue);
    let eyes = crate::records::eyes(&recs.setup)?;
    let pics = crate::records::pictures(recs.photos.clone(), recs.setup.portrait_photo.clone());
    let mut drawer = mm_render::draw::Drawer::new(device, queue, &recs.setup.meta, wgpu::TextureFormat::Rgba8Unorm, eyes, pics);
    let mark = if o.mark { Some(mm_render::mark::MarkStyle::from_brand(&recs.setup.brand)?) } else { None };
    let fps = sh.st.lock().unwrap().project.fps;
    let n = recs.frames.len();
    let draw = |drawer: &mut mm_render::draw::Drawer, t: usize, img: &mm_media::convert::Bgr| -> Result<mm_media::convert::Bgr> {
        let Some((rec, near)) = recs.get(t) else { return Ok(img.clone()) };
        let mut rec = rec.clone();
        if let Some(l) = rec.light.as_mut() {
            l.grain = if l.settings.grain { crate::records::frame_grain(img.w as usize, img.h as usize, &img.data) } else { 0.0 };
        }
        let src = mm_render::compositor::upload_rgba(device, queue, &bgr_to_rgba(img));
        let drawn = drawer.draw(device, queue, &src, &rec, near)?;
        let px = mm_render::readback::read_texture(device, queue, &drawn);
        Ok(rgba_to_bgr(&px, img.w, img.h))
    };
    if mm_media::photo::is_image(video) {
        let probe = mm_media::photo::probe(video)?;
        let s = mm_media::formats::settings(&probe, &o.preset, &o.fit, o.start, o.end, Some(1));
        let img = mm_media::photo::read_image(video)?;
        let f = draw(&mut drawer, 0, &img)?;
        let mut f = mm_media::formats::fit_frame_cpu(&f, s.width, s.height, &s.fit);
        if let Some(m) = &mark {
            watermark(&mut f, m)?;
        }
        mm_media::photo::write_image(out, &f, &mm_media::about::name(), &mm_media::about::made_with(), 0.92)?;
        job.set(|p| p.done = 1);
        return Ok(1);
    }
    let mut dec = mm_media::Decoder::open_with(video, Some(fps))?;
    let tags = dec.colour_tags().map(|s| s.to_string());
    let s = mm_media::formats::settings(&dec.probe, &o.preset, &o.fit, o.start, o.end, Some(n));
    let opts = mm_media::export::ExportOptions::from_settings(&dec.probe, &s);
    let mut ex = Exporter::with(video, out, &dec.probe, opts)?;
    job.set(|p| {
        p.total = (s.end - s.start).div_ceil(s.step);
        p.message = format!("{}x{}", s.width, s.height);
    });
    dec.seek(s.start)?;
    let mut last_note = Instant::now();
    let mut done = 0;
    while let Some((t, f)) = dec.next()? {
        if t >= s.end {
            break;
        }
        if job.cancelled() {
            ex.cancel();
            bail!("at frame {t}");
        }
        if t < s.start || (t - s.start) % s.step != 0 {
            continue;
        }
        let img = mm_media::convert::frame_bgr(&f, &dec.probe, tags.as_deref(), mm_media::convert::Colour::Display)?;
        let img = draw(&mut drawer, t, &img)?;
        let mut img = mm_media::formats::fit_frame_cpu(&img, s.width, s.height, &s.fit);
        if let Some(m) = &mark {
            watermark(&mut img, m)?;
        }
        ex.push_nv12(t, mm_media::convert::bgr_to_nv12(&img))?;
        done += 1;
        job.set(|pr| pr.done = done);
        if last_note.elapsed().as_millis() > 100 {
            last_note = Instant::now();
            publish(sh, job);
        }
    }
    ex.finish()?;
    Ok(done)
}

// ---------------------------------------------------------------------------------------------------- the Export page
/// The Export page's choices (app.html's export section).
#[derive(Clone, Debug)]
pub struct PageUi {
    pub preset: String,
    pub fit: String,
    pub start: String,
    pub end: String,
    pub strict: bool,
    /// /api/formats: presets, sizes, the clip's name and folder
    pub formats: Value,
    /// open questions (higher-risk mode needs none)
    pub open: usize,
    pub note: String,
}

impl Default for PageUi {
    fn default() -> Self {
        PageUi { preset: "original".into(), fit: "fill".into(), start: String::new(), end: String::new(), strict: false,
                 formats: Value::Null, open: 0, note: String::new() }
    }
}

pub const PAGE_HELP: &[(&str, &str)] = &[
    ("export_page", "load the Export page's formats (on showing it)"),
    ("export_set", "{preset?, fit?, start?, end?, mark?, strict?}: the Export page's choices (mark is kept with the project)"),
    ("export_run", "{out}: export the open project as chosen to out (.mp4; a photo: .jpg/.png); progress in state.export"),
    ("export_reveal", "show the last export in Finder"),
];

pub fn page_dispatch(sh: &Arc<Shared>, cmd: &str, a: &Value) -> Result<Value> {
    match cmd {
        "export_page" => {
            crate::library::queue(sh, |sh| {
                if let Err(e) = load_page(sh) {
                    log::warn(&format!("export page: {e:#}"));
                }
            });
            Ok(json!("queued"))
        }
        "export_set" => {
            let mut refetch = false;
            {
                let mut l = sh.lib.lock().unwrap();
                let x = &mut l.xp;
                if let Some(p) = a["preset"].as_str() {
                    x.preset = p.into();
                }
                if let Some(f) = a["fit"].as_str() {
                    x.fit = f.into();
                }
                if let Some(v) = a.get("start") {
                    x.start = v.as_str().map(String::from).unwrap_or_else(|| if v.is_null() { String::new() } else { v.to_string() });
                }
                if let Some(v) = a.get("end") {
                    x.end = v.as_str().map(String::from).unwrap_or_else(|| if v.is_null() { String::new() } else { v.to_string() });
                }
                if let Some(b) = a["strict"].as_bool() {
                    refetch = b != x.strict;
                    x.strict = b;
                    x.note = String::new();
                }
            }
            if let Some(m) = a["mark"].as_bool() {
                // the corner mark: remembered per project (edits.watermark, bookkeeping that undo skips)
                crate::library::queue(sh, move |sh| {
                    let pid = sh.lib.lock().unwrap().pid.clone();
                    if let Ok(sc) = crate::library::sidecar(sh) {
                        if let Ok(r) = sc.ops(&pid, &json!({"ops": [{"op": "set", "path": ["watermark"], "value": m}], "label": "Mark", "undoable": false})) {
                            crate::editor::took(sh, &r);
                        }
                    }
                    publish_page(sh);
                });
            }
            if refetch {
                crate::library::queue(sh, |sh| {
                    let _ = count_open(sh);
                    crate::records::refresh(sh);
                    publish_page(sh);
                });
            }
            publish_page(sh);
            Ok(json!("ok"))
        }
        "export_run" => {
            let out = crate::commands::local_path(a["out"].as_str().ok_or_else(|| anyhow!("export_run: missing 'out'"))?);
            let (o, open) = {
                let l = sh.lib.lock().unwrap();
                let x = &l.xp;
                let num = |s: &str| s.trim().parse::<f64>().ok();
                (Opts { preset: x.preset.clone(), fit: x.fit.clone(), start: num(&x.start), end: num(&x.end).filter(|e| *e > 0.0),
                        mark: l.doc["edits"]["watermark"].as_bool().unwrap_or(false), strict: x.strict }, x.open)
            };
            if o.strict && open > 0 {
                let msg = sh.text.t("export.strict_block", &crate::text::v(&[("n", open)]));
                sh.lib.lock().unwrap().xp.note = msg.clone();
                publish_page(sh);
                bail!("{msg}");
            }
            start_with(sh, &out, o)
        }
        "export_reveal" => {
            let out = sh.export.lock().unwrap().get().out;
            if out.is_empty() || !std::path::Path::new(&out).exists() {
                bail!("nothing exported yet");
            }
            std::process::Command::new("open").arg("-R").arg(&out).spawn()?;
            Ok(json!(out))
        }
        _ => bail!("no command '{cmd}'"),
    }
}

fn count_open(sh: &Shared) -> Result<()> {
    let pid = sh.lib.lock().unwrap().pid.clone();
    let dr = crate::library::sidecar(sh)?.decisions(&pid)?;
    let n = dr["decisions"].as_array().into_iter().flatten().filter(|d| d.get("by").is_none_or(Value::is_null)).count();
    sh.lib.lock().unwrap().xp.open = n;
    Ok(())
}

fn load_page(sh: &Arc<Shared>) -> Result<()> {
    let pid = sh.lib.lock().unwrap().pid.clone();
    let f = crate::library::sidecar(sh)?.formats(&pid)?;
    sh.lib.lock().unwrap().xp.formats = f;
    count_open(sh)?;
    publish_page(sh);
    if sh.lib.lock().unwrap().xp.strict {
        crate::records::refresh(sh);
    }
    Ok(())
}

/// Make `export_page` (what the Export page shows) from the choices, the formats and the export job.
pub fn publish_page(sh: &Shared) {
    let tx = &sh.text;
    let (x, edits) = {
        let l = sh.lib.lock().unwrap();
        (l.xp.clone(), l.doc["edits"].clone())
    };
    let f = &x.formats;
    let photo = f["photo"].as_bool().unwrap_or(false);
    let chosen = edits["cover_style"].as_str().unwrap_or("emoji").to_string();
    let style = if x.strict { "solid".to_string() } else { chosen.clone() };
    let icon = match style.as_str() { "emoji" => "😎", "blur" => "◌", "pixelate" => "▦", "solid" => "●", "orb" => "🟡", _ => "🖼️" };
    let mut styles = tx.t("export.covered_with", &crate::text::v(&[("style", format!("{icon} {}", tx.s(&format!("style.{style}"))))]));
    if x.strict && chosen != "solid" {
        styles += &format!(" <span style=\"color:#8b9099\">{}</span>", tx.s("export.higher_risk_mode"));
    }
    let presets: Vec<Value> = f["presets"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|p| {
            let id = p["id"].as_str().unwrap_or("");
            let detail = if photo && id == "original" { tx.s("export.same_as_photo") } else { p["detail"].as_str().unwrap_or("").to_string() };
            json!({"id": id, "label": p["label"], "detail": detail, "on": id == x.preset})
        })
        .collect();
    let size = &f["sizes"][x.preset.as_str()][x.fit.as_str()];
    let (w, h) = (size[0].as_u64().unwrap_or(0), size[1].as_u64().unwrap_or(0));
    let secs = f["seconds"].as_f64().unwrap_or(0.0);
    let summary = if photo {
        format!("{w}×{h} · {}", tx.s("export.out_photo"))
    } else {
        let s0 = x.start.trim().parse::<f64>().unwrap_or(0.0).max(0.0);
        let s1 = x.end.trim().parse::<f64>().ok().filter(|e| *e > 0.0).unwrap_or(secs).min(secs);
        let len = (s1 - s0).max(0.0);
        let l = if len >= 60.0 {
            tx.t("unit.min_s", &crate::text::v(&[("m", (len / 60.0).floor().to_string()), ("s", (len % 60.0).round().to_string())]))
        } else {
            tx.t("unit.s", &crate::text::v(&[("n", format!("{len:.1}"))]))
        };
        let mut s = format!("{w}×{h} · {l} · {}", tx.s("export.out_video"));
        if len > 180.0 && x.preset != "original" {
            s += &format!(" · {}", tx.s("export.too_long"));
        }
        s
    };
    let label = match x.preset.as_str() { "vertical" => "_9x16", "portrait" => "_4x5", "square" => "_1x1", "landscape" => "_16x9", _ => "" };
    let file = tx.t("export.file_name", &crate::text::v(&[
        ("name", f["name"].as_str().map(String::from).unwrap_or_else(|| tx.s("export.file_name_default"))),
        ("format", label.to_string()), ("ext", (if photo { "jpg" } else { "mp4" }).to_string())]));
    let dir = match f["save_dir"].as_str() {
        Some(d) if !d.is_empty() => d.to_string(),
        _ => std::env::var("HOME").map(|h| format!("{h}/{}", if photo { "Pictures" } else { "Movies" })).unwrap_or_default(),
    };
    let job = sh.export.lock().unwrap().get();
    let status = match job.state {
        JobState::Running => tx.t("export.progress", &crate::text::v(&[("pct", if job.total > 0 { 100 * job.done / job.total } else { 0 })])),
        JobState::Done => tx.t("export.saved", &crate::text::v(&[("name", std::path::Path::new(&job.out).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default())])),
        JobState::Failed => tx.t("export.err.failed", &crate::text::v(&[("error", job.message.trim_start_matches("export failed: ").to_string())])),
        JobState::Cancelled => job.message.clone(),
        JobState::Idle => String::new(),
    };
    let strict_note = if x.strict && x.open > 0 { tx.t("export.strict_open", &crate::text::v(&[("n", x.open)])) } else { String::new() };
    let v = json!({
        "loaded": !f.is_null(), "photo": photo, "styles": styles, "presets": presets, "preset": x.preset, "fit": x.fit,
        "fit_shown": x.preset != "original", "out_w": w, "out_h": h, "summary": summary, "start": x.start, "end": x.end,
        "mark": edits["watermark"].as_bool().unwrap_or(false), "strict": x.strict, "open": x.open, "strict_note": strict_note,
        "note": x.note, "file_name": file, "save_dir": dir, "status": status, "state": job.state,
        "progress": if job.total > 0 { job.done as f64 / job.total as f64 } else { 0.0 }, "out": job.out,
        "can_reveal": job.state == JobState::Done && !job.out.is_empty(),
    });
    sh.st.lock().unwrap().export_page = v;
    sh.changed(&["export_page"]);
}
