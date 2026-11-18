//! Grey data movies (the depth cache, depth.py): writing one ([`GreyWriter`], video_io.writer) and reading its Y planes
//! back ([`luma`], video_io.luma). The scan's depth readings are fitted on the decoded bytes, so both sides follow
//! Python's exactly:
//!
//! - **Writing** goes through AVFoundation on the Mac, with video_io._AVFWriter's settings (H.264 High, auto level,
//!   the average bit rate, keyframes at most `keyframes` apart, the expected frame rate, BT.709 tags; BGRA pixel
//!   buffers from the adaptor's pool, alpha 255; frame n at round(n * 90000 / fps) / 90000 s). VideoToolbox is
//!   deterministic for the same input and settings: two Python runs on jazz's depth gave the same decoded bytes (the
//!   .mp4 files differ only in the container's dates), so the movie's frames, and the sidecar fitted on them, equal
//!   Python's (tools/parity/engine2_depth.py). GStreamer's vtenc sets other encoder properties, so it isn't used here.
//!   Not ported to other systems yet.
//! - **Reading** decodes through mm-media's GStreamer decoder. video_io.luma asks AVFoundation for full-range NV12
//!   (kCVPixelFormatType_420YpCbCr8BiPlanarFullRange), so a video-range movie's Y codes come back stretched:
//!   y' = clamp(((y - 16) * 4769 + 2048) >> 12) (255/219 in 12-bit fixed point; fitted on every code 15-235 of jazz's
//!   depth movie, 127 frames: no exceptions; rounding the exact ratio misses codes 159 and 232). The depth sidecar's
//!   fit is made on these codes, so they're reproduced here ([`video_to_full_range`]).
use anyhow::{anyhow, Result};
use mm_vision::Mat;
use std::path::Path;

/// video_io.luma's full-range stretch of a video-range Y code (module doc).
pub fn video_to_full_range(y: u8) -> u8 {
    (((y as i32 - 16) * 4769 + 2048) >> 12).clamp(0, 255) as u8
}

/// The movie's Y planes as AVFoundation hands them out (full range): (frame index round(pts * fps), plane) from frame
/// `start`, `count` frames or to the end (depth._luma over video_io.luma).
pub fn luma(path: &Path, fps: f64, start: usize, count: Option<usize>) -> Result<Vec<(usize, Mat<u8>)>> {
    crate::decode::init();
    let mut d = crate::Decoder::open(&path.to_string_lossy())?;
    let mut out = vec![];
    while let Some(f) = d.pull_frame()? {
        let i = (f.pts * fps).round_ties_even() as usize;
        if i < start {
            continue;
        }
        if count.is_some_and(|n| i >= start + n) {
            break;
        }
        let y = crate::convert::Yuv::from_frame(&f)?;
        let full = f.full;
        let data = y.y.iter().map(|&v| if full { v as u8 } else { video_to_full_range(v as u8) }).collect();
        out.push((i, Mat::from_vec(y.h, y.w, 1, data)));
    }
    Ok(out)
}

/// The depth movie's Y planes through [`luma`] (video_io.luma, with its full-range stretch), as the depth readings'
/// source (mm_analysis::depth::Depth reads it through mm_vision's LumaSource).
pub struct MovieLuma {
    pub path: std::path::PathBuf,
    pub fps: f64,
}

impl mm_vision::luma::LumaSource for MovieLuma {
    fn luma(&self, start: usize, count: Option<usize>) -> Vec<(usize, Mat<u8>)> {
        luma(&self.path, self.fps, start, count).expect("depth movie decode")
    }
}

/// video_io.writer for grey frames: H.264 .mp4, `bitrate` b/s, keyframes at most `keyframes` frames apart.
pub struct GreyWriter {
    #[cfg(target_os = "macos")]
    inner: mac::Writer,
}

impl GreyWriter {
    pub fn new(path: &Path, width: usize, height: usize, fps: f64, bitrate: u32, keyframes: u32) -> Result<GreyWriter> {
        #[cfg(target_os = "macos")]
        {
            Ok(GreyWriter { inner: mac::Writer::new(path, width, height, fps, bitrate, keyframes)? })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (path, width, height, fps, bitrate, keyframes);
            Err(anyhow!("the grey movie writer is only ported for the Mac"))
        }
    }

    /// One frame: a grey (1 channel) picture of the writer's size, written as BGR (cv2.COLOR_GRAY2BGR).
    pub fn write(&mut self, grey: &Mat<u8>) -> Result<()> {
        if grey.cn != 1 {
            return Err(anyhow!("grey movie: {} channels", grey.cn));
        }
        #[cfg(target_os = "macos")]
        {
            self.inner.write(grey)
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(())
        }
    }

    pub fn close(self) -> Result<()> {
        #[cfg(target_os = "macos")]
        {
            self.inner.close()
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(())
        }
    }
}

#[cfg(target_os = "macos")]
mod mac {
    use anyhow::{anyhow, bail, Result};
    use mm_vision::Mat;
    use objc2::rc::Retained;
    use objc2::runtime::AnyObject;
    use objc2_av_foundation::*;
    use objc2_core_video::*;
    use objc2_foundation::{NSDictionary, NSNumber, NSString, NSURL};
    use std::path::Path;
    use std::ptr::NonNull;

    const TIMESCALE: i32 = 90000;

    fn cfkey(k: &objc2_core_foundation::CFString) -> &NSString {
        unsafe { &*(k as *const objc2_core_foundation::CFString as *const NSString) }
    }

    pub struct Writer {
        w: Retained<AVAssetWriter>,
        vin: Retained<AVAssetWriterInput>,
        ad: Retained<AVAssetWriterInputPixelBufferAdaptor>,
        width: usize,
        height: usize,
        fps: f64,
        n: i64,
    }

    impl Writer {
        pub fn new(path: &Path, width: usize, height: usize, fps: f64, bitrate: u32, keyframes: u32) -> Result<Writer> {
            unsafe {
                let _ = std::fs::remove_file(path);
                let abs = if path.is_absolute() { path.to_path_buf() } else { std::env::current_dir()?.join(path) };
                let url = NSURL::fileURLWithPath(&NSString::from_str(&abs.to_string_lossy()));
                let w = AVAssetWriter::assetWriterWithURL_fileType_error(&url, AVFileTypeMPEG4.unwrap()).map_err(|e| anyhow!("writer: {e:?}"))?;
                let props: Retained<NSDictionary<NSString, AnyObject>> = NSDictionary::from_slices(
                    &[AVVideoAverageBitRateKey.unwrap(), AVVideoMaxKeyFrameIntervalKey.unwrap(), AVVideoExpectedSourceFrameRateKey.unwrap(), AVVideoProfileLevelKey.unwrap()],
                    &[&NSNumber::new_i64(bitrate as i64), &NSNumber::new_i64(keyframes as i64), &NSNumber::new_i64(fps.round() as i64), AVVideoProfileLevelH264HighAutoLevel.unwrap()],
                );
                let colour: Retained<NSDictionary<NSString, AnyObject>> = NSDictionary::from_slices(
                    &[AVVideoColorPrimariesKey.unwrap(), AVVideoTransferFunctionKey.unwrap(), AVVideoYCbCrMatrixKey.unwrap()],
                    &[AVVideoColorPrimaries_ITU_R_709_2.unwrap(), AVVideoTransferFunction_ITU_R_709_2.unwrap(), AVVideoYCbCrMatrix_ITU_R_709_2.unwrap()],
                );
                let settings: Retained<NSDictionary<NSString, AnyObject>> = NSDictionary::from_slices(
                    &[AVVideoCodecKey.unwrap(), AVVideoWidthKey.unwrap(), AVVideoHeightKey.unwrap(), AVVideoCompressionPropertiesKey.unwrap(), AVVideoColorPropertiesKey.unwrap()],
                    &[AVVideoCodecTypeH264.unwrap(), &NSNumber::new_i64(width as i64), &NSNumber::new_i64(height as i64), &props, &colour],
                );
                let vin = AVAssetWriterInput::assetWriterInputWithMediaType_outputSettings(AVMediaTypeVideo.unwrap(), Some(&settings));
                vin.setExpectsMediaDataInRealTime(false);
                let pba: Retained<NSDictionary<NSString, AnyObject>> = NSDictionary::from_slices(
                    &[cfkey(kCVPixelBufferPixelFormatTypeKey), cfkey(kCVPixelBufferWidthKey), cfkey(kCVPixelBufferHeightKey)],
                    &[&NSNumber::new_u32(kCVPixelFormatType_32BGRA), &NSNumber::new_i64(width as i64), &NSNumber::new_i64(height as i64)],
                );
                let ad = AVAssetWriterInputPixelBufferAdaptor::assetWriterInputPixelBufferAdaptorWithAssetWriterInput_sourcePixelBufferAttributes(&vin, Some(&pba));
                w.addInput(&vin);
                if !w.startWriting() {
                    bail!("can't start writing: {:?}", w.error());
                }
                w.startSessionAtSourceTime(objc2_core_media::CMTime { value: 0, timescale: TIMESCALE, flags: objc2_core_media::CMTimeFlags::Valid, epoch: 0 });
                Ok(Writer { w, vin, ad, width, height, fps, n: 0 })
            }
        }

        pub fn write(&mut self, grey: &Mat<u8>) -> Result<()> {
            if (grey.cols, grey.rows) != (self.width, self.height) {
                bail!("grey movie: frame {}x{}, writer {}x{}", grey.cols, grey.rows, self.width, self.height);
            }
            unsafe {
                while !self.vin.isReadyForMoreMediaData() {
                    if self.w.status() != AVAssetWriterStatus::Writing {
                        bail!("encode failed: {:?}", self.w.error());
                    }
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
                let pool = self.ad.pixelBufferPool().ok_or_else(|| anyhow!("no pixel buffer pool"))?;
                let mut pb: *mut CVPixelBuffer = std::ptr::null_mut();
                let r = CVPixelBufferPool::create_pixel_buffer(None, &pool, NonNull::from(&mut pb));
                if r != kCVReturnSuccess || pb.is_null() {
                    bail!("no pixel buffer ({r})");
                }
                let pb = objc2_core_foundation::CFRetained::from_raw(NonNull::new_unchecked(pb));
                if CVPixelBufferLockBaseAddress(&pb, CVPixelBufferLockFlags(0)) != kCVReturnSuccess {
                    bail!("CVPixelBufferLockBaseAddress");
                }
                let base = CVPixelBufferGetBaseAddress(&pb) as *mut u8;
                let stride = CVPixelBufferGetBytesPerRow(&pb);
                for y in 0..self.height {
                    let row = std::slice::from_raw_parts_mut(base.add(y * stride), self.width * 4);
                    let src = &grey.data[y * self.width..(y + 1) * self.width];
                    for (px, &v) in row.chunks_exact_mut(4).zip(src) {
                        px.copy_from_slice(&[v, v, v, 255]);
                    }
                }
                CVPixelBufferUnlockBaseAddress(&pb, CVPixelBufferLockFlags(0));
                let when = objc2_core_media::CMTime {
                    value: (self.n as f64 * TIMESCALE as f64 / self.fps).round_ties_even() as i64,
                    timescale: TIMESCALE,
                    flags: objc2_core_media::CMTimeFlags::Valid,
                    epoch: 0,
                };
                if !self.ad.appendPixelBuffer_withPresentationTime(&pb, when) {
                    bail!("encode failed: {:?}", self.w.error());
                }
            }
            self.n += 1;
            Ok(())
        }

        pub fn close(self) -> Result<()> {
            unsafe {
                self.vin.markAsFinished();
                let (tx, rx) = std::sync::mpsc::channel::<()>();
                let blk = block2::RcBlock::new(move || {
                    let _ = tx.send(());
                });
                self.w.finishWritingWithCompletionHandler(&blk);
                rx.recv()?;
                if self.w.status() != AVAssetWriterStatus::Completed {
                    bail!("encode failed: {:?}", self.w.error());
                }
            }
            Ok(())
        }
    }
}
