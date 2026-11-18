//! The encoder: `appsrc ! vtenc_h264_hw ! h264parse ! mp4mux`, video only, then [`crate::mp4::finish_export`]
//! adds the source's audio (packets copied byte for byte, priming included) and the app's metadata, and puts the
//! movie header first. The caller draws each slot and pushes it, in slot order:
//! - `push_pixbuf` (Mac, zero-copy): the compositor wrote straight into an IOSurface-backed BGRA pixel buffer from
//!   `pool()`; vtenc hands it to VideoToolbox as is;
//! - `push_nv12`: NV12 bytes (mm-render's rgb2nv12 pass + readback, or [`crate::convert::bgr_to_nv12`]); works with
//!   any encoder.
//!
//! As video_io's writer: H.264 High, a keyframe at least every 2 s, the formats.py bitrate, BT.709 tags; the audio
//! is the source's from the export's first frame for the video's length (video_io.mux), and the only metadata is the
//! app's name (software) and "Made with ..." (description) (about.py).
//!
//! Fixed here: mp4mux dropped the source's 2 AAC priming packets (248 of 250 on jazz_4k, poc/rust/gst/REPORT.md in the
//! unpublished PoC); now all 250 are kept, byte-identical, with the source's "skip 1619 samples" edit
//! (native/tools/parity/media_export.py).
use crate::decode::Probe;
use anyhow::{anyhow, bail, Result};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use mm_core::time::slot_ns;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

/// An exact rational for a timeline rate: the NTSC rates as n*1000/1001, others to the millihertz.
pub fn rational_fps(fps: f64) -> (i32, i32) {
    for base in [15, 24, 30, 48, 60, 120] {
        let ntsc = base as f64 * 1000.0 / 1001.0;
        if (fps - ntsc).abs() < 0.002 {
            return (base * 1000, 1001);
        }
    }
    if (fps - fps.round()).abs() < 1e-6 {
        return (fps.round() as i32, 1);
    }
    let n = (fps * 1000.0).round() as i32;
    let g = gcd(n, 1000);
    (n / g, 1000 / g)
}

fn gcd(a: i32, b: i32) -> i32 {
    if b == 0 { a.abs().max(1) } else { gcd(b, a % b) }
}

/// What to make: size, rate, bitrate and which part of the source's audio goes with it.
#[derive(Clone, Debug)]
pub struct ExportOptions {
    pub width: u32,
    pub height: u32,
    /// output frame rate (exact rational)
    pub framerate: (i32, i32),
    pub bitrate_kbps: u32,
    /// presentation seconds of the source where the export starts (the audio is cut from there)
    pub audio_start: f64,
    /// copy the source's audio (if it has any)
    pub audio: bool,
    /// (software, description) written as metadata; None: the app's (about.rs)
    pub made_by: Option<(String, String)>,
}

impl ExportOptions {
    pub fn original(probe: &Probe, bitrate_kbps: u32) -> ExportOptions {
        let fr = if probe.framerate.0 > 0 { probe.framerate } else { rational_fps(probe.fps) };
        ExportOptions { width: probe.width, height: probe.height, framerate: fr, bitrate_kbps, audio_start: 0.0, audio: true, made_by: None }
    }

    /// From formats.settings: size, rate (after the frame-rate step), bitrate, start.
    pub fn from_settings(probe: &Probe, s: &crate::formats::Settings) -> ExportOptions {
        ExportOptions {
            width: s.width, height: s.height, framerate: rational_fps(s.fps), bitrate_kbps: (s.bitrate / 1000) as u32,
            audio_start: s.start as f64 / probe.fps, audio: true, made_by: None,
        }
    }
}

pub struct Exporter {
    pipeline: gst::Pipeline,
    src: gst_app::AppSrc,
    framerate: (i32, i32),
    clip: PathBuf,
    out: PathBuf,
    tmp: PathBuf,
    opts: ExportOptions,
    has_audio: bool,
    pub width: u32,
    pub height: u32,
    pub frames: usize,
}

pub struct Stats {
    pub frames: usize,
    pub audio_packets: usize,
    /// audio samples the edit list skips at the start (the encoder priming)
    pub audio_skip: i64,
}

impl Exporter {
    /// Start encoding a `probe.width` x `probe.height` video at the source's frame rate into `out`, copying `clip`'s
    /// audio (when it has any).
    pub fn new(clip: &str, out: &str, probe: &Probe, bitrate_kbps: u32) -> Result<Exporter> {
        Self::with(clip, out, probe, ExportOptions::original(probe, bitrate_kbps))
    }

    pub fn with(clip: &str, out: &str, probe: &Probe, opts: ExportOptions) -> Result<Exporter> {
        crate::decode::init();
        let (w, h) = (opts.width, opts.height);
        let fr = opts.framerate;
        let key = ((fr.0 as f64 / fr.1 as f64) * 2.0).round().max(1.0) as u32;
        let desc = format!(
            "appsrc name=src format=time block=true max-buffers=3 caps=video/x-raw,format=NV12,width={w},height={h},\
             framerate={}/{},pixel-aspect-ratio=1/1,colorimetry=bt709,interlace-mode=progressive ! \
             vtenc_h264_hw name=enc bitrate={} max-keyframe-interval={key} ! video/x-h264,profile=high ! h264parse ! \
             mp4mux name=mux ! filesink name=fs",
            fr.0, fr.1, opts.bitrate_kbps
        );
        let out_p = PathBuf::from(out);
        let tmp = out_p.with_extension("video.mp4");
        let pipeline = gst::parse::launch(&desc)?.downcast::<gst::Pipeline>().map_err(|_| anyhow!("pipeline"))?;
        pipeline.by_name("fs").unwrap().set_property("location", tmp.to_string_lossy().to_string());
        let src = pipeline.by_name("src").unwrap().downcast::<gst_app::AppSrc>().unwrap();
        pipeline.set_state(gst::State::Playing)?;
        Ok(Exporter {
            pipeline, src, framerate: fr, clip: std::fs::canonicalize(clip)?, out: out_p, tmp, has_audio: probe.has_audio && opts.audio,
            width: w, height: h, frames: 0, opts,
        })
    }

    /// Push the next output frame (its timestamp is its index at the output rate); see `push_pixbuf` / `push_nv12`.
    pub fn push(&mut self, _t: usize, mut buf: gst::Buffer) -> Result<()> {
        let i = self.frames;
        let (a, b) = (slot_ns(i, self.framerate), slot_ns(i + 1, self.framerate));
        {
            let bm = buf.get_mut().ok_or_else(|| anyhow!("buffer not writable"))?;
            bm.set_pts(gst::ClockTime::from_nseconds(a));
            bm.set_duration(gst::ClockTime::from_nseconds(b - a));
        }
        self.src.push_buffer(buf).map_err(|e| anyhow!("push: {e:?}"))?;
        self.frames += 1;
        Ok(())
    }

    pub fn push_nv12(&mut self, t: usize, bytes: Vec<u8>) -> Result<()> {
        self.push(t, gst::Buffer::from_mut_slice(bytes))
    }

    #[cfg(target_os = "macos")]
    pub fn push_pixbuf(&mut self, t: usize, pb: &objc2_core_foundation::CFRetained<objc2_core_video::CVPixelBuffer>) -> Result<()> {
        let b = crate::mac::cv::wrap_pixbuf(pb)?;
        self.push(t, b)
    }

    /// End the stream, wait for the video to be written, then add the audio and metadata (mp4.rs).
    pub fn finish(self) -> Result<Stats> {
        self.src.end_of_stream().map_err(|e| anyhow!("eos: {e:?}"))?;
        let bus = self.pipeline.bus().unwrap();
        let mut res = Ok(());
        for msg in bus.iter_timed(gst::ClockTime::from_seconds(60)) {
            match msg.view() {
                gst::MessageView::Eos(_) => break,
                gst::MessageView::Error(e) => {
                    res = Err(anyhow!("export: {} ({:?})", e.error(), e.debug()));
                    break;
                }
                _ => {}
            }
        }
        self.pipeline.set_state(gst::State::Null)?;
        if let Err(e) = res {
            let _ = std::fs::remove_file(&self.tmp);
            return Err(e);
        }
        let (sw, de) = self.opts.made_by.clone().unwrap_or_else(|| (crate::about::name(), crate::about::made_with()));
        let meta = [(b"swre", sw), (b"dscp", de)];
        let r = crate::mp4::finish_export(&self.tmp, self.has_audio.then_some(self.clip.as_path()), self.opts.audio_start, &meta, &self.out);
        let _ = std::fs::remove_file(&self.tmp);
        let info = r?;
        Ok(Stats { frames: self.frames, audio_packets: info.audio_packets, audio_skip: info.audio_skip })
    }

    /// Abandon the export and remove what was written.
    pub fn cancel(self) {
        let _ = self.pipeline.set_state(gst::State::Null);
        let _ = std::fs::remove_file(&self.tmp);
        let _ = std::fs::remove_file(&self.out);
    }
}

/// The headless export (no covers): video_io's display frames, fitted (CPU), encoded, audio copied. The reference
/// for the GPU export and the parity test. `progress(fraction)`; `cancel` stops it and removes the partial file.
pub fn export_plain(clip: &str, out: &str, preset: &str, fit: &str, start: Option<f64>, end: Option<f64>,
                    progress: &mut dyn FnMut(f64), cancel: &AtomicBool) -> Result<(crate::formats::Settings, Option<Stats>)> {
    let probe = crate::decode::probe(clip, None)?;
    let s = crate::formats::settings(&probe, preset, fit, start, end, None);
    if probe.image {
        let img = crate::photo::read_image(clip)?;
        let f = crate::formats::fit_frame_cpu(&img, s.width, s.height, &s.fit);
        crate::photo::write_image(out, &f, &crate::about::name(), &crate::about::made_with(), 0.92)?;
        progress(1.0);
        return Ok((s, None));
    }
    let mut dec = crate::Decoder::open(clip)?;
    let tags = dec.colour_tags().map(|s| s.to_string());
    let opts = ExportOptions::from_settings(&dec.probe, &s);
    let mut ex = Exporter::with(clip, out, &dec.probe, opts)?;
    dec.seek(s.start)?;
    while let Some((t, f)) = dec.next()? {
        if t >= s.end {
            break;
        }
        if cancel.load(Ordering::Relaxed) {
            ex.cancel();
            bail!("cancelled");
        }
        if (t - s.start) % s.step != 0 {
            continue;
        }
        let img = crate::convert::frame_bgr(&f, &dec.probe, tags.as_deref(), crate::convert::Colour::Display)?;
        let img = crate::formats::fit_frame_cpu(&img, s.width, s.height, &s.fit);
        ex.push_nv12(t, crate::convert::bgr_to_nv12(&img))?;
        if t % 15 == 0 {
            progress(0.97 * (t - s.start) as f64 / (s.end - s.start).max(1) as f64);
        }
    }
    let st = ex.finish()?;
    progress(1.0);
    Ok((s, Some(st)))
}

