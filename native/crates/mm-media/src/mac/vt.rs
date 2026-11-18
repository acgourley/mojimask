//! macOS: the display colour exactly as AVFoundation gives it: VideoToolbox's pixel transfer (VTPixelTransferSession)
//! from the decoder's CVPixelBuffer (NV12 / P010) to 32BGRA.
//!
//! An AVAssetReaderTrackOutput asked for kCVPixelFormatType_32BGRA hands back what VTPixelTransferSession makes of
//! the decoder's YUV buffer, given the colour attachments AVFoundation puts on it. Measured 2026-11-17: AVFoundation's
//! own NV12/P010 output of each frame, run through a VTPixelTransferSession into an IOSurface-backed 32BGRA buffer
//! (no colour attachments on it, so no colour matching), equals its 32BGRA output on every pixel: jazz, a 4K iPhone
//! clip (BT.709), and the synthetic BT.709 8/10-bit, HLG, PQ, P3, untagged HD and SD clips. Its 8-bit video-range
//! NV12 (what video_io's detect colour starts from) of a 10-bit stream is the same transfer to 420v.
//! The destination must be IOSurface-backed like AVFoundation's: into a plain CVPixelBufferCreate buffer VideoToolbox
//! converts another way (off by up to 14 levels on jazz, 255 on the synthetic clips' block edges).
//!
//! From vtdec's buffer instead of AVFoundation's: identical on every frame of every parity clip and the 4K clip, for
//! both colours (2026-11-17). Speed: see convert.rs.
//!
//! The attachments that matter are the ones AVFoundation sets: the matrix, primaries, transfer function and chroma
//! siting. vtdec's buffers don't carry the same set, so we set them on the source buffer before the transfer, from
//! the stream's tags (GStreamer colorimetry), or for an untagged stream from AVFoundation's guess ("ColorInfoGuessedBy:
//! VideoToolbox"): the BT.601 matrix with SMPTE-C primaries when the coded width is 704 or less, else BT.709; transfer
//! BT.709 either way. Chroma siting: left (co-sited horizontally), what AVFoundation reports for every clip measured.
use anyhow::{bail, Result};
use objc2_core_foundation::{CFRetained, CFString, CFType};
use objc2_core_video::*;
use std::cell::RefCell;
use std::ffi::c_void;

#[link(name = "VideoToolbox", kind = "framework")]
extern "C" {
    fn VTPixelTransferSessionCreate(allocator: *const c_void, session_out: *mut *mut c_void) -> i32;
    fn VTPixelTransferSessionTransferImage(session: *mut c_void, source: *const CVPixelBuffer, destination: *const CVPixelBuffer) -> i32;
    fn VTPixelTransferSessionInvalidate(session: *mut c_void);
}
extern "C" {
    fn CFRelease(cf: *const c_void);
}

struct Session(*mut c_void);
impl Drop for Session {
    fn drop(&mut self) {
        unsafe {
            VTPixelTransferSessionInvalidate(self.0);
            CFRelease(self.0);
        }
    }
}

thread_local! {
    // one session per thread (sessions aren't documented as thread-safe), and its destination buffer per format
    static SESSION: RefCell<Option<(Session, Vec<((usize, usize, u32), CFRetained<CVPixelBuffer>)>)>> = const { RefCell::new(None) };
}

/// The colour attachments AVFoundation sets on a decoded buffer: (matrix, primaries, transfer).
pub struct Tags {
    pub matrix: &'static CFString,
    pub primaries: &'static CFString,
    pub transfer: &'static CFString,
}

/// AVFoundation's attachments for a stream with GStreamer colorimetry `tags` (None: untagged) and coded width.
pub fn tags_for(tags: Option<&str>, coded_w: u32) -> Tags {
    unsafe {
        let t = tags.unwrap_or("");
        // "range:matrix:transfer:primaries" with GstVideoColorMatrix / GstVideoTransferFunction / GstVideoColorPrimaries codes
        let parts: Vec<&str> = t.split(':').collect();
        let code = |i: usize| if parts.len() == 4 { parts[i].parse::<u32>().ok() } else { None };
        let (kr, _) = crate::convert::matrix_for(tags, coded_w);
        let matrix = if kr == 0.2627 {
            kCVImageBufferYCbCrMatrix_ITU_R_2020
        } else if kr == 0.299 {
            kCVImageBufferYCbCrMatrix_ITU_R_601_4
        } else if kr == 0.212 {
            kCVImageBufferYCbCrMatrix_SMPTE_240M_1995
        } else {
            kCVImageBufferYCbCrMatrix_ITU_R_709_2
        };
        // GstVideoTransferFunction: 5 BT709, 12 BT2020_12, 14 SMPTE2084 (PQ), 15 ARIB_STD_B67 (HLG), 16 BT601, 11 BT2020_10
        let transfer = if t.starts_with("bt2100-pq") || code(2) == Some(14) {
            kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ
        } else if t.starts_with("bt2100-hlg") || code(2) == Some(15) {
            kCVImageBufferTransferFunction_ITU_R_2100_HLG
        } else {
            kCVImageBufferTransferFunction_ITU_R_709_2
        };
        // GstVideoColorPrimaries: 1 BT709, 4 SMPTE170M, 9 BT2020, 11 SMPTERP431 (DCI-P3), 12 SMPTEEG432 (P3 D65)
        let primaries = if t.starts_with("bt2020") || t.starts_with("bt2100") || code(3) == Some(9) {
            kCVImageBufferColorPrimaries_ITU_R_2020
        } else if code(3) == Some(12) {
            kCVImageBufferColorPrimaries_P3_D65
        } else if code(3) == Some(11) {
            kCVImageBufferColorPrimaries_DCI_P3
        } else if kr == 0.299 {
            kCVImageBufferColorPrimaries_SMPTE_C
        } else {
            kCVImageBufferColorPrimaries_ITU_R_709_2
        };
        Tags { matrix, primaries, transfer }
    }
}

/// MM_VT_DEBUG=1: print each source buffer's attachments before and after ours are set (once per process).
fn debug_attachments(pb: &CVPixelBuffer, when: &str) {
    if std::env::var_os("MM_VT_DEBUG").is_none() {
        return;
    }
    let a = pb.attachments(CVAttachmentMode::ShouldPropagate);
    eprintln!("mm-media vt {when}: {a:?}");
}

/// `pb` (vtdec's NV12/P010 buffer) converted by VideoToolbox's pixel transfer to `fourcc` (32BGRA for the display
/// colour, 420v for the detect colour of a stream that doesn't decode to 420v), with AVFoundation's attachments set
/// on `pb` first. The returned buffer is this thread's, reused by its next call: read it before then.
pub fn transfer(pb: &CVPixelBuffer, tags: &Tags, fourcc: u32) -> Result<CFRetained<CVPixelBuffer>> {
    static ONCE: std::sync::Once = std::sync::Once::new();
    let first = !ONCE.is_completed();
    if first {
        debug_attachments(pb, "decoder's attachments");
    }
    // vtdec's own attachments (VideoToolbox's guess) give the same picture for 8-bit streams, but not for 10-bit
    // ones (HLG/PQ display off by up to 15 levels, the 420v for detect by up to 83): set AVFoundation's
    unsafe {
        let mode = CVAttachmentMode::ShouldPropagate;
        let set = |k: &CFString, v: &CFString| pb.set_attachment(k, v.as_ref() as &CFType, mode);
        set(kCVImageBufferYCbCrMatrixKey, tags.matrix);
        set(kCVImageBufferColorPrimariesKey, tags.primaries);
        set(kCVImageBufferTransferFunctionKey, tags.transfer);
        set(kCVImageBufferChromaLocationTopFieldKey, kCVImageBufferChromaLocation_Left);
        set(kCVImageBufferChromaLocationBottomFieldKey, kCVImageBufferChromaLocation_Left);
    }
    if first {
        debug_attachments(pb, "with ours");
        ONCE.call_once(|| {});
    }
    let (w, h) = (CVPixelBufferGetWidth(pb), CVPixelBufferGetHeight(pb));
    SESSION.with(|s| {
        let mut s = s.borrow_mut();
        if s.is_none() {
            let mut p: *mut c_void = std::ptr::null_mut();
            let r = unsafe { VTPixelTransferSessionCreate(std::ptr::null(), &mut p) };
            if r != 0 || p.is_null() {
                bail!("VTPixelTransferSessionCreate: {r}");
            }
            *s = Some((Session(p), Vec::new()));
        }
        let (sess, dsts) = s.as_mut().unwrap();
        let i = match dsts.iter().position(|(k, _)| *k == (w, h, fourcc)) {
            Some(i) => i,
            None => {
                // IOSurface-backed, like AVFoundation's buffers: a plain malloc'd destination takes another
                // conversion path in VideoToolbox, off by up to 14 levels on jazz
                let out = super::cv::Pool::new(w as u32, h as u32, fourcc)?.get()?;
                dsts.retain(|(k, _)| k.2 != fourcc);
                dsts.push(((w, h, fourcc), out));
                dsts.len() - 1
            }
        };
        let d = dsts[i].1.clone();
        let r = unsafe { VTPixelTransferSessionTransferImage(sess.0, pb, &*d) };
        if r != 0 {
            bail!("VTPixelTransferSessionTransferImage: {r}");
        }
        Ok(d)
    })
}

/// `pb` as AVFoundation's 32BGRA, packed to BGR, at the stored size.
pub fn to_bgr(pb: &CVPixelBuffer, tags: &Tags) -> Result<crate::convert::Bgr> {
    let d = transfer(pb, tags, kCVPixelFormatType_32BGRA)?;
    let (w, h) = (CVPixelBufferGetWidth(&d), CVPixelBufferGetHeight(&d));
    unsafe {
        if CVPixelBufferLockBaseAddress(&d, CVPixelBufferLockFlags::ReadOnly) != kCVReturnSuccess {
            bail!("CVPixelBufferLockBaseAddress");
        }
        let base = CVPixelBufferGetBaseAddress(&d) as *const u8;
        let stride = CVPixelBufferGetBytesPerRow(&d);
        let mut data = vec![0u8; w * h * 3];
        for y in 0..h {
            let row = std::slice::from_raw_parts(base.add(y * stride), w * 4);
            for (o, p) in data[y * w * 3..(y + 1) * w * 3].chunks_exact_mut(3).zip(row.chunks_exact(4)) {
                o.copy_from_slice(&p[..3]);
            }
        }
        CVPixelBufferUnlockBaseAddress(&d, CVPixelBufferLockFlags::ReadOnly);
        Ok(crate::convert::Bgr { w: w as u32, h: h as u32, data })
    }
}
