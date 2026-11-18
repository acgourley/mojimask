//! macOS: GStreamer buffers <-> CVPixelBuffers <-> Metal textures, with no CPU pixel copies; VideoToolbox's pixel
//! transfer for AVFoundation's exact display colour.
pub mod cv;
pub mod vt;
