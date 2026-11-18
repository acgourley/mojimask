//! Video decode through GStreamer onto mojimask's constant-rate timeline.
//!
//! Pipeline: `filesrc ! qtdemux ! queue ! parsebin ! vtdec_hw ! appsink` (caps NV12 for 8-bit sources, P010_10LE for
//! 10-bit ones, system-memory caps so no GL context is made). vtdec_hw's buffers wrap the decoder's CVPixelBuffer
//! (GstAppleCoreVideoMemory, mapped lazily) and carry a GstCoreVideoMeta pointing at it, so the GPU side can import
//! the planes without the CPU touching a pixel (see `cv.rs`).
//!
//! Times: qtdemux applies the edit list as a segment (start = media time of the edit, time = presentation time). The
//! decoder is given an open segment instead (see `Decoder::seek`), and we clip by the demuxer's segment ourselves:
//! frames wholly outside it are dropped, and the frame the edit starts inside has its pts clipped to the edit start
//! (jazz: stored frame 75 starts inside the edit and is shown at t=0, like AVFoundation). We use stream time =
//! segment.to_stream_time(pts) on the demuxer's segment. Slot t shows the latest frame whose (stream time -
//! video_start) * fps rounds to <= t (video_io._frames_avf).
//!
//! Every frame is identified by a label put on it going into the decoder (its own pts and the flush generation),
//! never by the pts the decoder hands out: see `Decoder::seek` for why.
use anyhow::{anyhow, bail, Result};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use mm_core::time::{snap_fps, Timeline};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Probe {
    /// upright display size (rotation applied, square pixels)
    pub width: u32,
    pub height: u32,
    /// coded size as decoded
    pub stored_w: u32,
    pub stored_h: u32,
    pub par: (i32, i32),
    pub fps: f64,
    pub framerate: (i32, i32),
    pub video_start: f64,
    pub video_duration: f64,
    /// clockwise degrees, as AVFoundation's preferredTransform angle (mm's prep convention)
    pub rotation: u32,
    pub n: usize,
    pub format: String,
    pub colorimetry: String,
    /// the source has an audio track (the export copies it)
    pub has_audio: bool,
    /// the asset's duration (all tracks; video_io's "duration")
    pub duration: f64,
    /// a photo: a clip of one frame (video_io: PHOTO_FPS, rotation 0, no audio)
    pub image: bool,
    /// where the timeline facts came from: "mp4" (the movie's boxes, as AVFoundation reads them) or "gst" (caps)
    pub source: String,
}

impl Probe {
    /// video_io.probe's dict: the keys the Python app stores in a project's meta.
    pub fn video_io_json(&self) -> serde_json::Value {
        let mut v = serde_json::json!({
            "width": self.width, "height": self.height, "fps": self.fps, "duration": self.duration,
            "video_start": self.video_start, "video_duration": self.video_duration, "rotation": self.rotation,
            "has_audio": self.has_audio,
        });
        if self.image {
            v["image"] = serde_json::Value::Bool(true);
        }
        v
    }

    pub fn timeline(&self) -> Timeline {
        Timeline { fps: self.fps, video_start: self.video_start, n: self.n }
    }
}

fn snap(fps: f64) -> Option<f64> {
    snap_fps(fps)
}

/// The timeline facts video_io._probe_avf reads from AVFoundation, read from the movie's boxes (mp4.rs).
/// - size: the track header's display size (naturalSize: pixel aspect already applied), rounded; swapped when the
///   matrix turns the picture a quarter turn.
/// - rotation: atan2(b, a) of the track matrix, clockwise degrees (AVFoundation's preferredTransform angle).
/// - fps: the shortest frame duration snapped to a common rate; else the nominal rate (samples / media duration)
///   snapped; else the median gap between presentation times snapped; else the nominal rate; else 30.
/// - video_start, video_duration: the track's time range (leading empty edits; the edits' total).
/// - duration: the movie header's duration.
pub fn probe_movie(path: &std::path::Path) -> Result<Probe> {
    let m = crate::mp4::read_movie(path)?;
    let v = m.video().ok_or_else(|| anyhow!("no video track"))?;
    let [a, b, ..] = v.matrix;
    let rot = ((((b as f64).atan2(a as f64)).to_degrees().round() as i64).rem_euclid(360)) as u32;
    let (mut w, mut h) = (v.tkhd_w.round() as u32, v.tkhd_h.round() as u32);
    let (codec, sw, sh) = crate::mp4::visual_entry(&v.stsd).unwrap_or((*b"????", w, h));
    let par = crate::mp4::pasp(&v.stsd).map(|(x, y)| (x as i32, y as i32)).unwrap_or((1, 1));
    // QuickTime files may leave the track size at the coded size and say the pixel shape only in `pasp`
    // (jazz_hlg10.mov: 1920 wide, 1593:2096, shown 1459); AVFoundation's naturalSize applies it
    if par.0 != par.1 && w == sw && h == sh {
        w = ((sw as f64) * par.0 as f64 / par.1.max(1) as f64).round() as u32;
    }
    if w == 0 || h == 0 {
        w = ((sw as f64) * par.0 as f64 / par.1.max(1) as f64).round() as u32;
        h = sh;
    }
    if rot == 90 || rot == 270 {
        std::mem::swap(&mut w, &mut h);
    }
    let ts = v.media_ts.max(1) as f64;
    let min_d = v.deltas().into_iter().filter(|&d| d > 0).min();
    let nominal = if v.media_dur > 0 { v.n() as f64 / (v.media_dur as f64 / ts) } else { 0.0 };
    let typical = {
        let mut p = v.pts();
        p.sort();
        let mut g: Vec<i64> = p.windows(2).map(|w| w[1] - w[0]).filter(|&d| d > 0).collect();
        g.sort();
        if g.is_empty() {
            None
        } else {
            // numpy's median: the mean of the two middle values for an even count
            let k = g.len();
            let med = if k % 2 == 1 { g[k / 2] as f64 } else { (g[k / 2 - 1] + g[k / 2]) as f64 / 2.0 };
            Some(ts / med)
        }
    };
    let fps = min_d.and_then(|d| snap(ts / d as f64))
        .or_else(|| snap(nominal))
        .or_else(|| snap(typical.unwrap_or(0.0)))
        .unwrap_or(if nominal > 0.0 { nominal } else { 30.0 });
    let (lead, seg) = v.time_range(m.movie_ts);
    let mts = m.movie_ts.max(1) as f64;
    let video_duration = seg as f64 / mts;
    Ok(Probe {
        width: w, height: h, stored_w: sw, stored_h: sh, par, fps,
        framerate: crate::export::rational_fps(fps),
        video_start: lead as f64 / mts, video_duration, rotation: rot,
        n: ((video_duration * fps).round() as usize).max(1),
        format: String::from_utf8_lossy(&codec).into(), colorimetry: String::new(),
        has_audio: m.audio().is_some(), duration: m.movie_dur as f64 / mts, image: false, source: "mp4".into(),
    })
}

/// video_io.probe: a photo's one-frame clip, the movie's boxes for MP4/QuickTime, else what GStreamer says.
/// `keep_fps`: a project's saved timeline rate wins (video_io.probe's `keep`).
pub fn probe(path: &str, keep_fps: Option<f64>) -> Result<Probe> {
    let mut p = if crate::photo::is_image(path) {
        crate::photo::probe(path)?
    } else {
        match probe_movie(std::path::Path::new(path)) {
            Ok(p) => p,
            Err(_) => Decoder::open(path)?.probe.clone(),
        }
    };
    if let Some(f) = keep_fps.filter(|f| *f > 0.0) {
        p.fps = f;
        p.framerate = crate::export::rational_fps(f);
        p.n = ((p.video_duration * f).round() as usize).max(1);
    }
    Ok(p)
}

pub fn init() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| gst::init().expect("gst::init"));
}

/// One decoded stored frame.
#[derive(Clone)]
pub struct Frame {
    /// source pts: stream (presentation) seconds of this stored frame, from the demuxer (not clipped by seeks)
    pub pts: f64,
    /// the pts the decoder reported on the buffer: usually `pts` (unclipped), but GstVideoDecoder rewrites it once
    /// vtdec has handed a frame out of order (see `Decoder::seek`); for debugging only
    pub shown_pts: f64,
    /// the matrix AVFoundation converts this stream with: (kr, kb) from the stream's own tags, or by coded width
    /// when untagged (convert::matrix_for)
    pub matrix: (f32, f32),
    /// full-range codes: only when the stream says so (the decoder's caps default untagged streams to full range
    /// for some sizes; AVFoundation treats them as video range)
    pub full: bool,
    pub buffer: gst::Buffer,
    pub info: gst_video::VideoInfo,
    /// the pts this frame went into the decoder with (segment ns)
    src: u64,
}

/// Each compressed frame going into the decoder is labelled (a GstReferenceTimestampMeta, which GstVideoDecoder
/// copies onto the picture decoded from it) with its own pts and the flush generation it belongs to; `outstanding`
/// lists the pts of this generation's frames the demuxer has sent that are not out of the decoder yet.
#[derive(Default)]
struct Inflight {
    /// flushes seen at the decoder's input (the labels' generation)
    gen: u64,
    /// flushes seen at the demuxer's output (`outstanding`'s generation; the same count once a seek has returned)
    demux_gen: u64,
    /// the demuxer's current segment (the decoder itself is given an open one: see `seek`)
    segment: Option<gst::FormattedSegment<gst::ClockTime>>,
    /// this generation's frames wholly outside the demuxer's segment, which we drop as gst_segment_clip would
    clipped: std::collections::BTreeSet<u64>,
    /// stream time - pts (ns) under the segment each of this generation's frames came in (an edit list of several
    /// edits gives several segments, and the demuxer runs ahead of the decoder)
    offsets: std::collections::HashMap<u64, i128>,
    outstanding: std::collections::BTreeMap<u64, u32>,
    /// the generation in which the decoder last received EOS
    eos_gen: Option<u64>,
}

/// The caps naming our label meta (GstReferenceTimestampMeta's "reference"): timestamp = the frame's pts as it went
/// into the decoder (ns), duration = its flush generation.
fn label_caps() -> &'static gst::Caps {
    static C: std::sync::OnceLock<gst::Caps> = std::sync::OnceLock::new();
    C.get_or_init(|| gst::Caps::new_empty_simple("timestamp/x-mojimask-source"))
}

/// Frames we hold back at most while waiting for an earlier one still in the decoder (a frame the decoder drops
/// never comes out; this bounds the wait). VideoToolbox's reorder depth is at most 16 (HEVC's DPB).
const HOLD_MAX: usize = 20;

#[derive(Default)]
struct Seen {
    rotation: Option<u32>,
    ten_bit: bool,
    /// the parsed stream's colorimetry field, if it has one
    tags: Option<String>,
}

pub struct Decoder {
    pub pipeline: gst::Pipeline,
    sink: gst_app::AppSink,
    pub probe: Probe,
    pending: Option<Frame>,
    cur: Option<Frame>,
    t: usize,
    eos: bool,
    /// what has gone into the decoder since the last flush (see `Inflight`)
    inflight: Arc<Mutex<Inflight>>,
    /// the flush generation the frames we want carry: the number of flushes when the last seek returned
    gen: u64,
    /// frames pulled from the sink but not yet handed out, waiting for any earlier frame still in the decoder
    held: Vec<Frame>,
    /// the last seek (position, flags), to redo it if appsink ends with an EOS from before it
    last_seek: Option<(gst::ClockTime, gst::SeekFlags)>,
    /// the source pts of the last frame `pull` returned since the last seek
    last_src: Option<u64>,
    colour_tags: Option<String>,
}



/// MM_MEDIA_DEBUG=1: say when the decoder's output needed mending (stale pictures, late frames, stale EOS).
fn debug(a: std::fmt::Arguments) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ON.get_or_init(|| std::env::var_os("MM_MEDIA_DEBUG").is_some()) {
        eprintln!("mm-media: {a}");
    }
}

/// Our label on a decoded picture: (pts its compressed frame went into the decoder with, flush generation).
fn label(b: &gst::BufferRef) -> Option<(u64, u64)> {
    b.iter_meta::<gst::ReferenceTimestampMeta>()
        .find(|m| m.reference().is_strictly_equal(label_caps()))
        .map(|m| (m.timestamp().nseconds(), m.duration().map_or(0, |d| d.nseconds())))
}

fn rotation_from_tag(s: &str) -> Option<u32> {
    // GStreamer: "rotate-N" = turn the picture N degrees clockwise to show it; the same number mm's prep takes
    // (AVFoundation's angle for the rot90 clip is 270, and GStreamer's tag is rotate-270).
    s.strip_prefix("rotate-")?.parse().ok()
}

impl Decoder {
    pub fn open(path: &str) -> Result<Decoder> {
        Self::open_with(path, None)
    }

    /// Open a video; `keep_fps`: a project's saved timeline rate (video_io.probe's `keep`).
    pub fn open_with(path: &str, keep_fps: Option<f64>) -> Result<Decoder> {
        init();
        let path = std::fs::canonicalize(path)?;
        let desc = format!(
            "filesrc name=src ! qtdemux name=d d.video_0 ! queue name=vq max-size-buffers=8 ! parsebin ! vtdec_hw name=dec ! \
             appsink name=sink sync=false max-buffers=4 enable-last-sample=false caps=video/x-raw,format={{NV12,P010_10LE}}"
        );
        let pipeline = gst::parse::launch(&desc)?.downcast::<gst::Pipeline>().map_err(|_| anyhow!("not a pipeline"))?;
        pipeline.by_name("src").unwrap().set_property("location", path.to_string_lossy().to_string());
        let sink = pipeline.by_name("sink").unwrap().downcast::<gst_app::AppSink>().unwrap();
        let dec = pipeline.by_name("dec").unwrap();
        let inflight: Arc<Mutex<Inflight>> = Arc::new(Mutex::new(Inflight::default()));
        {
            // label every frame going into the decoder; a flush starts a new generation
            let inflight = inflight.clone();
            dec.static_pad("sink").unwrap().add_probe(gst::PadProbeType::BUFFER | gst::PadProbeType::EVENT_FLUSH | gst::PadProbeType::EVENT_DOWNSTREAM, move |_, info| {
                match &mut info.data {
                    Some(gst::PadProbeData::Buffer(b)) => {
                        if let Some(p) = b.pts() {
                            let st = inflight.lock().unwrap();
                            let gen = st.gen;
                            gst::ReferenceTimestampMeta::add(b.make_mut(), label_caps(), p, gst::ClockTime::from_nseconds(gen));
                        }
                    }
                    Some(gst::PadProbeData::Event(e)) if e.type_() == gst::EventType::FlushStop => {
                        inflight.lock().unwrap().gen += 1;
                    }
                    Some(gst::PadProbeData::Event(e)) if e.type_() == gst::EventType::Segment => {
                        // an open segment for the decoder: it clips by the pts it hands out, which can be another
                        // frame's, and so dropped real frames near the end (see `seek`); we clip by labels instead
                        let mut open = gst::FormattedSegment::<gst::ClockTime>::new();
                        open.set_start(gst::ClockTime::ZERO);
                        let ev = gst::event::Segment::builder(&open).seqnum(e.seqnum()).build();
                        info.data = Some(gst::PadProbeData::Event(ev));
                    }
                    Some(gst::PadProbeData::Event(e)) if e.type_() == gst::EventType::Eos => {
                        let mut st = inflight.lock().unwrap();
                        st.eos_gen = Some(st.gen);
                    }
                    _ => {}
                }
                gst::PadProbeReturn::Ok
            });
        }
        {
            // what the demuxer has sent, in this generation: it sends a B-frame's reference before the B-frame
            // itself, and the decoder could hand the reference out before the B-frame reaches it; the demuxer is
            // ahead of the decoder by the queue, so a frame still to come is already counted here
            let inflight = inflight.clone();
            pipeline.by_name("vq").unwrap().static_pad("sink").unwrap().add_probe(
                gst::PadProbeType::BUFFER | gst::PadProbeType::EVENT_FLUSH | gst::PadProbeType::EVENT_DOWNSTREAM, move |_, info| {
                match &info.data {
                    Some(gst::PadProbeData::Buffer(b)) => {
                        if let Some(p) = b.pts() {
                            let mut st = inflight.lock().unwrap();
                            // gst_segment_clip on pts..pts+duration (40 ms when unknown, as GstVideoDecoder does)
                            let end = p + b.duration().unwrap_or(gst::ClockTime::from_mseconds(40));
                            let clipped = st.segment.as_ref().is_some_and(|s| {
                                s.start().is_some_and(|a| end <= a) || s.stop().is_some_and(|z| p >= z)
                            });
                            if let Some(g) = st.segment.as_ref() {
                                let off = g.time().map_or(0, |c| c.nseconds()) as i128 - g.start().map_or(0, |c| c.nseconds()) as i128;
                                st.offsets.insert(p.nseconds(), off);
                            }
                            if clipped {
                                st.clipped.insert(p.nseconds());
                            } else {
                                *st.outstanding.entry(p.nseconds()).or_insert(0) += 1;
                            }
                        }
                    }
                    Some(gst::PadProbeData::Event(e)) => match e.view() {
                        gst::EventView::FlushStop(_) => {
                            let mut st = inflight.lock().unwrap();
                            st.demux_gen += 1;
                            st.outstanding.clear();
                            st.clipped.clear();
                            st.offsets.clear();
                        }
                        gst::EventView::Segment(sg) => {
                            inflight.lock().unwrap().segment = sg.segment().downcast_ref::<gst::ClockTime>().cloned();
                        }
                        _ => {}
                    },
                    _ => {}
                }
                gst::PadProbeReturn::Ok
            });
        }
        // qtdemux adds every track's pad before the first sample flows: note whether there is audio
        let has_audio = Arc::new(AtomicBool::new(false));
        {
            let has_audio = has_audio.clone();
            pipeline.by_name("d").unwrap().connect_pad_added(move |_, pad| {
                if pad.name().starts_with("audio") {
                    has_audio.store(true, Ordering::SeqCst);
                }
            });
        }
        // Before vtdec negotiates: pick P010 for >8-bit streams; catch the orientation tag.
        let seen = Arc::new(Mutex::new(Seen::default()));
        {
            let seen = seen.clone();
            let sink2 = sink.clone();
            dec.static_pad("sink").unwrap().add_probe(gst::PadProbeType::EVENT_DOWNSTREAM, move |_, info| {
                if let Some(gst::PadProbeData::Event(ev)) = &info.data {
                    match ev.view() {
                        gst::EventView::Caps(c) => {
                            let s = c.caps().structure(0).unwrap();
                            let depth = s.get::<u32>("bit-depth-luma").unwrap_or(8);
                            let ten = depth > 8 || s.get::<&str>("profile").map(|p| p.contains("10")).unwrap_or(false);
                            seen.lock().unwrap().ten_bit = ten;
                            seen.lock().unwrap().tags = s.get::<String>("colorimetry").ok();
                            let fmt = if ten { "P010_10LE" } else { "NV12" };
                            sink2.set_caps(Some(&gst::Caps::builder("video/x-raw").field("format", fmt).build()));
                        }
                        gst::EventView::Tag(tg) => {
                            if let Some(o) = tg.tag().get::<gst::tags::ImageOrientation>() {
                                if let Some(r) = rotation_from_tag(o.get()) {
                                    seen.lock().unwrap().rotation = Some(r);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                gst::PadProbeReturn::Ok
            });
        }
        pipeline.set_state(gst::State::Playing)?;
        let mut d = Decoder { pipeline, sink, probe: Probe::default(), pending: None, cur: None, t: 0, eos: false, inflight, gen: 0, held: Vec::new(), last_seek: None, last_src: None, colour_tags: None };
        let first_sample = loop {
            let s = d.pull_sample()?.ok_or_else(|| anyhow!("no frames"))?;
            if s.buffer().and_then(|b| label(b)).map_or(true, |(_, g)| g == 0) {
                break s;
            }
        };
        let caps = first_sample.caps().ok_or_else(|| anyhow!("no caps"))?.to_owned();
        let info = gst_video::VideoInfo::from_caps(&caps)?;
        let seg = d.inflight.lock().unwrap().segment.clone().ok_or_else(|| anyhow!("no time segment"))?;
        let rot = seen.lock().unwrap().rotation.unwrap_or(0);
        let par = info.par();
        let fr = info.fps();
        let (sw, sh) = (info.width(), info.height());
        let mut dw = ((sw as f64) * par.numer() as f64 / par.denom() as f64).round() as u32;
        let mut dh = sh;
        if rot == 90 || rot == 270 {
            std::mem::swap(&mut dw, &mut dh);
        }
        let mut fps = if fr.numer() > 0 { snap(fr.numer() as f64 / fr.denom() as f64) } else { None };
        // video duration: the edit (segment) length, else the stream duration
        let seg_len = match (seg.start(), seg.stop()) {
            (Some(a), Some(b)) if b > a => Some((b - a).nseconds() as f64 / 1e9),
            _ => None,
        };
        let dur = seg_len.or_else(|| d.pipeline.query_duration::<gst::ClockTime>().map(|c| c.nseconds() as f64 / 1e9)).unwrap_or(0.0);
        if fps.is_none() {
            fps = Some(if fr.numer() > 0 { fr.numer() as f64 / fr.denom() as f64 } else { 30.0 });
        }
        let fps = fps.unwrap();
        let video_start = seg.time().map(|c| c.nseconds() as f64 / 1e9).unwrap_or(0.0);
        d.probe = Probe {
            width: dw, height: dh, stored_w: sw, stored_h: sh, par: (par.numer(), par.denom()), fps,
            framerate: (fr.numer(), fr.denom()), video_start, video_duration: dur, rotation: rot,
            n: ((dur * fps).round() as usize).max(1), format: format!("{:?}", info.format()),
            colorimetry: info.colorimetry().to_string(),
            has_audio: has_audio.load(Ordering::SeqCst),
            duration: d.pipeline.query_duration::<gst::ClockTime>().map(|c| c.nseconds() as f64 / 1e9).unwrap_or(dur),
            image: false,
            source: "gst".into(),
        };
        // the timeline as AVFoundation reads it, when the file's boxes can be read (MP4 / QuickTime)
        if let Ok(m) = probe_movie(&path) {
            let p = &mut d.probe;
            (p.width, p.height, p.fps, p.framerate) = (m.width, m.height, m.fps, m.framerate);
            (p.video_start, p.video_duration, p.rotation, p.n) = (m.video_start, m.video_duration, m.rotation, m.n);
            (p.has_audio, p.duration, p.source) = (m.has_audio, m.duration, m.source);
        }
        if let Some(f) = keep_fps.filter(|f| *f > 0.0) {
            d.probe.fps = f;
            d.probe.framerate = crate::export::rational_fps(f);
            d.probe.n = ((d.probe.video_duration * f).round() as usize).max(1);
        }
        d.colour_tags = seen.lock().unwrap().tags.clone();
        let clipped = first_sample.buffer().and_then(label).is_some_and(|(p, _)| d.inflight.lock().unwrap().clipped.contains(&p));
        if !clipped {
            d.taken(&first_sample);
            let first = d.frame_of(&first_sample)?;
            d.held.push(first);
        }
        d.pending = d.pull()?;
        Ok(d)
    }

    /// The stream's colour tags as the parser reported them (before the decoder fills in defaults): None when the
    /// file doesn't say, which is when AVFoundation picks the matrix by size (colour.rs).
    pub fn colour_tags(&self) -> Option<&str> {
        self.colour_tags.as_deref()
    }

    fn frame_of(&self, s: &gst::Sample) -> Result<Frame> {
        let buffer = s.buffer_owned().ok_or_else(|| anyhow!("sample without buffer"))?;
        let caps = s.caps().ok_or_else(|| anyhow!("no caps"))?;
        let info = gst_video::VideoInfo::from_caps(caps)?;
        // stream time under the demuxer's segment the frame came in (the decoder's own segment is open)
        let (seg, off) = {
            let st = self.inflight.lock().unwrap();
            (st.segment.clone().ok_or_else(|| anyhow!("no segment"))?, label(&buffer).and_then(|(p, _)| st.offsets.get(&p).copied()))
        };
        let shown = buffer.pts().ok_or_else(|| anyhow!("no pts"))?.nseconds();
        // the pts this picture's compressed frame went into the decoder with (the buffer's own pts can be another
        // frame's: see `pull`)
        let src = label(&buffer).map(|(p, _)| p).unwrap_or(shown);
        let off = off.unwrap_or_else(|| seg.time().map_or(0, |c| c.nseconds()) as i128 - seg.start().map_or(0, |c| c.nseconds()) as i128);
        let to_stream = |p: u64| -> f64 { (p as i128 + off) as f64 / 1e9 };
        // stream time, never before the edit's start: the frame the edit list starts inside is shown from slot 0
        // with its pts clipped to the edit start, as AVFoundation reports it. (Not the segment's start: after a
        // key-unit seek qtdemux starts the segment at the seek target, inside the frame shown there.)
        let pts = to_stream(src).max(self.probe.video_start);
        let matrix = crate::convert::matrix_for(self.colour_tags.as_deref(), info.width());
        let full = info.colorimetry().range() == gst_video::VideoColorRange::Range0_255 && self.colour_tags.is_some();
        Ok(Frame { pts, shown_pts: to_stream(shown), matrix, full, buffer, info, src })
    }

    fn pull_sample(&mut self) -> Result<Option<gst::Sample>> {
        if self.eos {
            return Ok(None);
        }
        match self.sink.pull_sample() {
            Ok(s) => Ok(Some(s)),
            Err(_) => {
                if let Some(bus) = self.pipeline.bus() {
                    while let Some(m) = bus.pop() {
                        if let gst::MessageView::Error(e) = m.view() {
                            bail!("gstreamer: {} ({:?})", e.error(), e.debug());
                        }
                    }
                }
                self.eos = true;
                Ok(None)
            }
        }
    }

    /// Count a sample's frame as out of the decoder.
    fn taken(&self, s: &gst::Sample) {
        if let Some((p, g)) = s.buffer().and_then(label) {
            let mut st = self.inflight.lock().unwrap();
            if g == self.gen && st.demux_gen == self.gen {
                if let Some(c) = st.outstanding.get_mut(&p) {
                    *c -= 1;
                    if *c == 0 {
                        st.outstanding.remove(&p);
                    }
                }
            }
        }
    }

    /// The next frame in presentation order. vtdec can hand pictures out of order and under another frame's pts
    /// (see `seek`), so: drop pictures from before the last flush, and hold pictures back until no frame with an
    /// earlier pts is still in the decoder.
    fn pull(&mut self) -> Result<Option<Frame>> {
        let (mut retries, mut lost) = (0, 0);
        loop {
            if let Some((i, m)) = self.held.iter().enumerate().map(|(i, f)| (i, f.src)).min_by_key(|x| x.1) {
                let first_out = self.inflight.lock().unwrap().outstanding.keys().next().copied();
                if self.eos || self.held.len() >= HOLD_MAX || first_out.map_or(true, |o| o >= m) {
                    if !self.eos && self.held.len() >= HOLD_MAX {
                        debug(format_args!("held {HOLD_MAX} frames waiting for pts {first_out:?}: handing out {m}"));
                    }
                    self.last_src = Some(m);
                    return Ok(Some(self.held.remove(i)));
                }
            }
            if self.eos {
                return Ok(None);
            }
            let Some(s) = self.pull_sample()? else {
                // appsink ended: with the decoder's own EOS, or (a race like the stale pictures above) with an EOS
                // from before the last flush, which leaves appsink refusing this generation's frames: seek again
                // and carry on after the last frame handed out
                let (stale, missing) = {
                    let st = self.inflight.lock().unwrap();
                    (st.eos_gen != Some(self.gen), st.outstanding.keys().next().copied())
                };
                if let (false, Some(p)) = (stale, missing) {
                    // the decoder ended without handing out a frame it was given (vtdec's EOS drain race: see
                    // `seek`): decode again from the keyframe before it, carrying on after the last frame handed out
                    lost += 1;
                    if lost <= 2 {
                        let seg = self.inflight.lock().unwrap().segment.clone();
                        let to = seg.as_ref().and_then(|g| g.to_stream_time(gst::ClockTime::from_nseconds(p)))
                            .unwrap_or(gst::ClockTime::from_nseconds(p));
                        debug(format_args!("the decoder ended without frame {p} (and {} more): decoding again from {to}",
                            self.inflight.lock().unwrap().outstanding.len() - 1));
                        let flags = gst::SeekFlags::FLUSH | gst::SeekFlags::KEY_UNIT | gst::SeekFlags::SNAP_BEFORE;
                        self.pipeline.seek_simple(flags, to)?;
                        self.gen = self.inflight.lock().unwrap().gen;
                        self.held.clear();
                        self.eos = false;
                        continue;
                    }
                    debug(format_args!("the decoder ended without frame {p}, again: going on without it"));
                }
                if stale {
                    if let Some((pos, flags)) = self.last_seek {
                        retries += 1;
                        if retries > 3 {
                            bail!("the decoder keeps ending before the end of the stream");
                        }
                        debug(format_args!("appsink ended with an EOS from before the flush: seeking again"));
                        self.pipeline.seek_simple(flags, pos)?;
                        self.gen = self.inflight.lock().unwrap().gen;
                        self.held.clear();
                        self.eos = false;
                    }
                }
                continue;
            };
            if let Some((p, g)) = s.buffer().and_then(label) {
                if g == self.gen && self.inflight.lock().unwrap().clipped.contains(&p) {
                    continue; // outside the segment (before the edit's start, after its end)
                }
                if g != self.gen {
                    debug(format_args!("dropped a picture from before the flush (pts {:?}, generation {g}, want {})", s.buffer().and_then(label).map(|l| l.0), self.gen));
                    continue; // decoded before the flush
                }
            }
            self.taken(&s);
            let f = self.frame_of(&s)?;
            if self.last_src.is_some_and(|l| f.src <= l) {
                continue; // already handed out (decoded again after seeking again)
            }
            self.held.push(f);
        }
    }

    /// Go to slot t: a flushing seek to the keyframe at or before slot t's frame; `next` then decodes forward and
    /// skips the frames before slot t itself, by their source pts.
    ///
    /// A key-unit seek: qtdemux starts its new segment at the target, sends from the keyframe before it, and we drop
    /// the frames wholly before the target as gst_segment_clip would (an ACCURATE seek decodes the same frames).
    ///
    /// **Frames are identified by labels, not by the decoder's pts** (fixed 2026-11-17). Before, a frame reached by
    /// a seek was sometimes not the one decoding forward gives (`mm-media-parity seek-all`: jazz and jazz_rot90 0-3
    /// random slots per 127 seeks, jazz_vfr and jazz_hlg10 their last 2-4 slots every time). Three causes, all in
    /// GStreamer 1.28's vtdec / GstVideoDecoder, all timing-dependent (hashing every frame as it came out hid them):
    /// - **Out of order at the end of the stream** (B-frame clips): at EOS, `gst_vtdec_drain_decoder` sets
    ///   `is_draining` *before* `VTDecompressionSessionWaitForAsynchronousFrames`, and the output thread, seeing
    ///   is_draining, pops the reorder queue's head while VideoToolbox still has frames in flight; a frame with an
    ///   earlier pts then lands after a later one. GstVideoDecoder warns "decreasing timestamp", gives the late
    ///   picture the previous frame's pts, sets `reordered_output` (kept across flushes) and from then on replaces
    ///   every pts with the oldest pending one. So picture and pts disagree for the last few slots: every seek near
    ///   the end decodes to EOS.
    /// - **A picture from before the flush** (jazz, jazz_rot90): now and then the first sample after a flushing seek
    ///   is the picture vtdec's output thread was handing to appsink when the seek came (labelled with the old
    ///   generation: ~5 frames past the last one pulled, i.e. appsink's queue plus one).
    /// - **Frames lost at the end** (B-frame clips, about 1 run of seek-all in 50), two ways. (a) Once
    ///   `reordered_output` is set, GstVideoDecoder guesses pts (previous + duration) and clips the buffer to the segment by that guess; near the
    ///   end a guess lands past the segment's stop (seen: 5.297 and 5.339 s on jazz_hlg10, whose last frame is at
    ///   5.255) and a real frame is dropped. So the decoder gets an open segment (start 0, no stop) and `pull` clips
    ///   by the labels against the demuxer's segment. (b) The last frame in decode order sometimes never comes out
    ///   (jazz_hlg10, about 1 run in 40 even with (a) fixed: the demuxer sent it, the decoder handed out the frames
    ///   after it, then EOS). Reading vtdec.c, the likely path: `gst_vtdec_drain_decoder` waits for VideoToolbox and
    ///   then pauses the output task with `gst_pad_pause_task`, which stops the loop after its current iteration
    ///   whatever the loop's own "only pause once the queue is empty" test says; an iteration that found the queue
    ///   empty just before the last frame landed in it ends the task, and the frame stays in the reorder queue. The
    ///   mend: at EOS, if a frame the demuxer sent is still not out, seek again to it (key unit) and carry on after
    ///   the last frame handed out (at most twice, then go on without it).
    ///
    /// The cure: every compressed frame going into vtdec carries a GstReferenceTimestampMeta with its own pts and
    /// the flush generation (GstVideoDecoder copies untagged metas onto the decoded picture, and VideoToolbox pairs
    /// each picture with its own frame). `pull` drops pictures of an older generation and reorders the rest by
    /// their labelled pts, holding a picture back while a frame with an earlier pts that the demuxer has sent is not
    /// out yet (counted at the demuxer's output, which is ahead of the decoder: a B-frame's reference is sent first
    /// and could come out before the B-frame reaches the decoder). Also mended, not seen to happen: an EOS from
    /// before the flush ending appsink early (seek again). MM_MEDIA_DEBUG=1 logs each mend.
    /// Evidence (2026-11-17): `media_seek.py` 10 runs, 0 wrong slots on every clip; `seek-all` 30 runs on each of
    /// jazz, jazz_rot90, jazz_vfr and jazz_hlg10 (127 seeks each), 0 wrong slots, with 463 stale pictures dropped and
    /// the EOS mend used 5 times (jazz and jazz_hlg10 last frames, a jazz_vfr one). 4K seek (jazz_4k, seek + next):
    /// median 56-60 ms, p95 77-85 ms.
    pub fn seek(&mut self, t: usize) -> Result<()> {
        self.seek_flags(t, gst::SeekFlags::FLUSH | gst::SeekFlags::KEY_UNIT | gst::SeekFlags::SNAP_BEFORE)
    }

    pub fn seek_flags(&mut self, t: usize, flags: gst::SeekFlags) -> Result<()> {
        let target = self.probe.timeline().seek_target(t);
        let pos = gst::ClockTime::from_nseconds((target * 1e9) as u64);
        self.pipeline.seek_simple(flags, pos)?;
        self.last_seek = Some((pos, flags));
        self.last_src = None;
        // the seek has flushed the decoder by the time it returns (qtdemux sends FLUSH_STOP from the seek call)
        self.gen = self.inflight.lock().unwrap().gen;
        self.held.clear();
        self.eos = false;
        self.cur = None;
        self.pending = self.pull()?;
        self.t = t;
        Ok(())
    }

    /// The next stored frame as the decoder hands it out, bypassing the timeline (debugging, tests).
    pub fn pull_frame(&mut self) -> Result<Option<Frame>> {
        if let Some(f) = self.pending.take() {
            return Ok(Some(f));
        }
        self.pull()
    }

    /// The slot the next call to `next` returns.
    pub fn position(&self) -> usize {
        self.t
    }

    /// Slot t's frame: decodes forward when t is a little ahead of the position, else seeks.
    pub fn frame_at(&mut self, t: usize) -> Result<Option<Frame>> {
        if !(self.t <= t && t <= self.t + 12) {
            self.seek(t)?;
        }
        while let Some((s, f)) = self.next()? {
            if s == t {
                return Ok(Some(f));
            }
        }
        Ok(None)
    }

    /// The next slot's (t, frame); the same frame repeats when the video holds one.
    pub fn next(&mut self) -> Result<Option<(usize, Frame)>> {
        let tl = self.probe.timeline();
        if self.t >= tl.n {
            return Ok(None);
        }
        while let Some(f) = &self.pending {
            if tl.starts_by(f.pts, self.t) {
                self.cur = self.pending.take();
                self.pending = self.pull()?;
            } else {
                break;
            }
        }
        if self.cur.is_none() {
            match self.pending.take() {
                None => return Ok(None),
                Some(f) => {
                    self.cur = Some(f);
                    self.pending = self.pull()?;
                }
            }
        }
        let slot = self.t;
        self.t += 1;
        Ok(Some((slot, self.cur.clone().unwrap())))
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        let _ = self.pipeline.set_state(gst::State::Null);
    }
}
