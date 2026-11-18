//! The preview's decoder thread: one GStreamer pipeline per open video; "show frame t" requests, latest wins.
//! Nearby frames ahead are decoded forward, others are seeks (mm_media::Decoder::frame_at).
use crate::{commands, log, Shared};
use mm_media::{Decoder, Frame};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

pub enum Req {
    Open { path: String, gen: u64 },
    Show { t: usize, gen: u64, asked: Instant },
    Quit,
}

/// A decoded frame waiting for the render thread.
pub struct Decoded {
    pub gen: u64,
    pub t: usize,
    pub frame: Frame,
    pub asked: Instant,
    pub decode_ms: f64,
}

/// Lighting's grain (records::frame_grain) for the frames shown, measured on its own thread: at 4K it takes longer
/// than a frame lasts, and on the decoder's thread it held playback to about 10 fps. While playing, a frame is drawn
/// with the newest grain measured (grain changes slowly), and grain is measured at most every [`PLAYING_GAP`]: at 4K
/// one measure is ~105 ms of a core (debug build: 80 ms to BGR, 25 ms to measure), so measuring every frame kept a
/// core busy for nothing. Each frame's own grain, once measured, is kept and the shown frame redrawn with it, so a
/// paused frame is drawn exactly as the export draws it.
/// While playing, the least time between two grain measures.
const PLAYING_GAP: std::time::Duration = std::time::Duration::from_millis(400);

#[derive(Default)]
pub struct Grain {
    st: std::sync::Mutex<GrainState>,
    wake: std::sync::Condvar,
}

#[derive(Default)]
struct GrainState {
    gen: u64,
    by_t: std::collections::HashMap<usize, f64>,
    last: Option<f64>,
    /// the newest frame waiting to be measured (older ones are dropped)
    todo: Option<(u64, usize, Frame, mm_media::Probe, Option<String>)>,
    /// whether the records (by address) draw lighting with grain: worked out once per records, not per frame
    wanted: Option<(usize, bool)>,
}

impl Grain {
    /// The grain to draw frame t with: its own if measured, else the newest measured.
    pub fn for_frame(&self, gen: u64, t: usize) -> Option<f64> {
        let s = self.st.lock().unwrap();
        if s.gen != gen {
            return None;
        }
        s.by_t.get(&t).copied().or(s.last)
    }
}

pub fn start_grain(sh: &Arc<Shared>) {
    let sh = sh.clone();
    std::thread::Builder::new()
        .name("mm-grain".into())
        .spawn(move || loop {
            let (gen, t, f, probe, tags) = {
                let mut s = sh.grain.st.lock().unwrap();
                loop {
                    if let Some(x) = s.todo.take() {
                        break x;
                    }
                    s = sh.grain.wake.wait(s).unwrap();
                }
            };
            let t0 = Instant::now();
            let Ok(img) = mm_media::convert::frame_bgr(&f, &probe, tags.as_deref(), mm_media::convert::Colour::Display) else { continue };
            let t1 = Instant::now();
            let g = crate::records::frame_grain(img.w as usize, img.h as usize, &img.data);
            {
                let mut p = sh.perf.lock().unwrap();
                crate::render::push(&mut p.grain_convert_ms, (t1 - t0).as_secs_f64() * 1e3);
                crate::render::push(&mut p.grain_ms, t1.elapsed().as_secs_f64() * 1e3);
            }
            {
                let mut s = sh.grain.st.lock().unwrap();
                if s.gen != gen {
                    continue;
                }
                s.by_t.insert(t, g);
                s.last = Some(g);
            }
            // the frame on screen now has its own grain: draw it again (paused, this is the exact picture)
            if gen == sh.gen() && sh.st.lock().unwrap().editor.shown == t as i64 {
                sh.dirty.store(true, std::sync::atomic::Ordering::SeqCst);
                sh.changed(&["@render"]);
            }
            // playing: the frames between are drawn with the newest grain; on a pause the frame last shown (the one
            // waiting) is measured at once
            let until = Instant::now() + PLAYING_GAP;
            while Instant::now() < until && sh.st.lock().unwrap().editor.playing {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        })
        .unwrap();
}

/// Ask for frame t's grain, if the records draw lighting with grain and it isn't known yet.
fn want_grain(sh: &Shared, d: &Decoder, gen: u64, t: usize, f: &Frame) {
    let Some(recs) = sh.recs.lock().unwrap().clone() else { return };
    let mut s = sh.grain.st.lock().unwrap();
    if s.gen != gen {
        *s = GrainState { gen, ..Default::default() };
    }
    let key = Arc::as_ptr(&recs) as usize;
    let want = match s.wanted {
        Some((k, w)) if k == key => w,
        _ => {
            let w = recs.frames.iter().any(|x| x.light.as_ref().is_some_and(|l| l.settings.grain));
            s.wanted = Some((key, w));
            w
        }
    };
    if !want {
        return;
    }
    if s.by_t.contains_key(&t) {
        return;
    }
    s.todo = Some((gen, t, f.clone(), d.probe.clone(), d.colour_tags().map(String::from)));
    sh.grain.wake.notify_one();
}

pub fn run(rx: mpsc::Receiver<Req>, sh: Arc<Shared>) {
    let mut dec: Option<(u64, Decoder)> = None;
    // the last frame delivered: asked for again (a new window, after a QML reload), it's handed over without decoding
    let mut last: Option<(u64, usize, Frame)> = None;
    while let Ok(first) = rx.recv() {
        // coalesce what's queued: the last open, then the last show after it
        let (mut open, mut show) = (None, None);
        let mut r = Some(first);
        while let Some(x) = r {
            match x {
                Req::Quit => return,
                Req::Open { path, gen } => {
                    open = Some((path, gen));
                    show = None;
                }
                Req::Show { t, gen, asked } => show = Some((t, gen, asked)),
            }
            r = rx.try_recv().ok();
        }
        if let Some((path, gen)) = open {
            dec = None;
            let t0 = Instant::now();
            match Decoder::open(&path) {
                Ok(d) => {
                    log::info(&format!("decoder: {} in {:.0} ms: {:?}", path, t0.elapsed().as_secs_f64() * 1e3, d.probe));
                    on_probe(&sh, gen, &d);
                    dec = Some((gen, d));
                }
                Err(e) => {
                    log::error(&format!("decoder: can't open {path}: {e:#}"));
                    if gen == sh.gen() {
                        sh.st.lock().unwrap().editor.status = format!("can't open this video: {e:#}");
                        sh.changed(&["editor.status"]);
                    }
                }
            }
            if show.is_none() && gen == sh.gen() {
                let t = sh.st.lock().unwrap().editor.t;
                show = Some((t, gen, Instant::now()));
            }
        }
        if let (Some((t, gen, asked)), Some((dg, d))) = (show, dec.as_mut()) {
            if gen != *dg || gen != sh.gen() {
                continue;
            }
            let t = t.min(d.probe.n.saturating_sub(1)); // a seek made before the length was known
            if let Some((lg, lt, f)) = &last {
                if *lg == gen && *lt == t {
                    want_grain(&sh, d, gen, t, f);
                    *sh.newest.lock().unwrap() = Some(Decoded { gen, t, frame: f.clone(), asked, decode_ms: 0.0 });
                    sh.changed(&["@render"]);
                    continue;
                }
            }
            let t0 = Instant::now();
            match d.frame_at(t) {
                Ok(Some(frame)) => {
                    last = Some((gen, t, frame.clone()));
                    let decode_ms = t0.elapsed().as_secs_f64() * 1e3;
                    want_grain(&sh, d, gen, t, &frame);
                    *sh.newest.lock().unwrap() = Some(Decoded { gen, t, frame, asked, decode_ms });
                    sh.changed(&["@render"]);
                }
                Ok(None) => log::warn(&format!("decoder: no frame {t}")),
                Err(e) => {
                    log::error(&format!("decoder: frame {t}: {e:#}"));
                    dec = None;
                }
            }
        }
    }
}

fn on_probe(sh: &Arc<Shared>, gen: u64, d: &Decoder) {
    if gen != sh.gen() {
        return;
    }
    let p = d.probe.clone();
    let fit = sh.project.lock().unwrap().as_ref().is_some_and(|pj| pj.covers_fit(p.width, p.height));
    *sh.probe.lock().unwrap() = Some((gen, p.clone()));
    let t = {
        let mut st = sh.st.lock().unwrap();
        let e = &mut st.editor;
        e.video.width = p.width;
        e.video.height = p.height;
        e.video.fps = p.fps;
        e.video.frames = p.n;
        e.video.duration = p.n as f64 / p.fps;
        e.video.has_audio = p.has_audio;
        e.video.ready = true;
        e.covers_fit = fit;
        e.t = e.t.min(p.n.saturating_sub(1));
        e.status = format!(
            "{}x{} · {:.3} fps · {} frames{}",
            p.width,
            p.height,
            p.fps,
            p.n,
            if e.data.is_empty() { "" } else if fit { "" } else { " · the cover data is for another video" }
        );
        e.t
    };
    let _ = t;
    sh.changed(&["editor.video", "editor.covers_fit", "editor.status", "editor.t"]);
    commands::persist(sh);
}
