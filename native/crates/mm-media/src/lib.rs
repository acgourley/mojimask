//! mm-media: video in and out through GStreamer, one layer for the engine's scan, the editor and the export, so
//! "frame t" (a source timestamp, mm_core::time) and colour are defined once.
//!
//! - [`decode`]: `filesrc ! qtdemux ! parsebin ! vtdec_hw ! appsink`; `Probe`, `Decoder::seek/next` on the slot timeline.
//! - [`import`]: a decoded frame's planes as wgpu textures (zero-copy on the Mac through `mac::cv`; else one upload),
//!   plus its colour parameters.
//! - [`export`]: `appsrc ! vtenc_h264_hw ! h264parse ! mp4mux`, with the source's AAC copied through untouched.
//!
//! - [`convert`]: a frame as video_io's BGR picture on the CPU (display colour as AVFoundation converts it: on the
//!   Mac through VideoToolbox's pixel transfer, exactly; detect colour as the detectors were tuned on; anamorphic
//!   resize, upright).
//! - [`grey`]: grey data movies (the depth cache): the writer (AVAssetWriter, as video_io.writer) and the Y planes back.
//! - [`photo`]: photos (one-frame clips) through ImageIO; [`formats`]: export presets and fit geometry;
//!   [`mp4`]: the movie's boxes (probe) and the export's final file (audio with priming, metadata).
//!
//! Measured in the PoC (poc/rust/gst/REPORT.md in the unpublished PoC): edit lists right, 52-55 dB vs AVFoundation, 4K
//! seek 71 ms, export 2 s.
pub mod about;
pub mod convert;
pub mod decode;
pub mod export;
pub mod formats;
pub mod grey;
pub mod import;
pub mod mp4;
pub mod photo;
#[cfg(target_os = "macos")]
pub mod mac;

pub use decode::{probe, Decoder, Frame, Probe};
