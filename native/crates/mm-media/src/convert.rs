//! A decoded frame as an upright BGR picture on the CPU, the way video_io gives it: the reference for the GPU's
//! YUV pass, the one-copy fallback, and the scan's frames ("detect") until those move to the GPU.
//!
//! **On the Mac both colours are exact** (every frame of every parity clip and a 4K iPhone clip, 2026-11-17):
//! - display: VideoToolbox's pixel transfer of the decoder's CVPixelBuffer to 32BGRA (mac::vt), which is what
//!   AVFoundation's 32BGRA output is. 7 ms per 4K frame, against 65 ms for the fitted model below.
//! - detect: AVFoundation's 8-bit video-range NV12 (the decoder's own buffer, or for 10-bit and full-range streams
//!   VideoToolbox's transfer of it to 420v), then video_io.nv12_to_bgr below. 33 ms per 4K frame.
//!
//! The fitted model below is the display colour off the Mac (and with MM_DISPLAY_CPU=1, or without a
//! CVPixelBuffer): within 2 levels of AVFoundation's, never exact.
//!
//! **The fitted "display" model: what AVFoundation's 32BGRA output does** (measured 2026-11-17 by fitting AVFoundation's
//! BGRA against its own NV12/P010 output of the same frames, on synthetic clips of random flat 16x16 YUV blocks
//! with every tag combination, then checked on jazz):
//! - **Matrix only.** Y'CbCr -> R'G'B' with the stream's matrix; no transfer or primaries conversion at all: a
//!   BT.2020 HLG or PQ clip and a P3 clip come out through their matrix alone (the fit is affine to within the
//!   rounding, mean |error| 0.25 levels, on every class).
//! - **Untagged streams:** BT.601 when the coded width is 704 or less, else BT.709 (640, 704x480 and 704x576 got
//!   601; 708x476 (jazz), 720x480, 720x576, 768 and up got 709). GStreamer guesses by height instead, so the tags
//!   are read before the decoder fills them in.
//! - **Fixed-point quirks**, the bulk of the gap the PoC's textbook WGSL left (55 dB):
//!   8-bit: luma gain 1.16422 (textbook 255/219 = 1.16438), chroma gains x 0.9960 (about 254/255), and per-channel
//!   offsets at black of R -0.19, G +0.11, B -0.25 levels;
//!   10-bit: luma gain 1.16542 (per 8-bit level), chroma x 1.0, offsets R +0.22, G +0.55, B +0.18.
//! - **Chroma:** bilinear, sampled as if co-sited with luma both ways (x/2, y/2): 65.5 dB on jazz_4k against 62.3
//!   for left-sited and 61.0 for centred.
//! - **Rounding:** to nearest, then clamped.
//!
//! Against video_io (PSNR over whole frames): textbook 709 + centred chroma 54-55 dB; this model 63-66 dB on 8-bit
//! jazz frames, 70 dB on the HLG clip.
//!
//! **"detect" colour:** video_io.nv12_to_bgr exactly: nearest-neighbour chroma and the fitted BT.601 matrix the
//! detectors were tuned on, computed in f32 and rounded half to even (np.rint). video_io asks AVFoundation for NV12
//! (not BGRA) here and converts it itself with cv2.transform, so this is about OpenCV's arithmetic, not Apple's: our
//! OpenCV (packaging/build_opencv.sh, clang on arm64, where transform_32f's SIMD path is compiled out) contracts the
//! sum into two FMAs, `fma(m2, Cr, fma(m0, Y, m1*Cb)) + m3`. Found 2026-11-17 on a 4K iPhone clip: the plain sum
//! differed from video_io by one level on 50 (frame 1) to 410 (frame 151) pixels, almost all green (the only row
//! with three non-zero terms); jazz, near-grey, never hit a tie. Emulating each evaluation order in f64 against
//! cv2's output on that clip: plain 50/23/410 pixels off, fma(m2, fma(m1, m0*Y)) 4/12/380, this order 0/0/0.
//! An OpenCV built another way (x86 SIMD, -ffp-contract=off) would need the plain sum.
//!
//! **Size and turn:** a stored size that isn't the display size (anamorphic pixels) is resized like
//! `cv2.resize(INTER_AREA)`; then the picture is turned upright (np.rot90).
use anyhow::{anyhow, Result};
use gstreamer_video as gst_video;
use gst_video::prelude::*;

/// Which conversion: video_io's `color` argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Colour {
    Display,
    Detect,
}

/// A BGR uint8 picture, rows packed.
#[derive(Clone)]
pub struct Bgr {
    pub w: u32,
    pub h: u32,
    pub data: Vec<u8>,
}

/// The decoded planes, as codes (8-bit values, or 10-bit values for P010) on the CPU.
pub struct Yuv {
    pub w: usize,
    pub h: usize,
    pub depth: u32,
    pub y: Vec<u16>,
    /// interleaved Cb, Cr at half size
    pub uv: Vec<u16>,
}

impl Yuv {
    /// The frame's planes. On the Mac, read through the decoder's CVPixelBuffer (the same memory the zero-copy GPU
    /// import aliases) when it has one; else through GStreamer's mapping.
    pub fn from_frame(f: &crate::Frame) -> Result<Yuv> {
        #[cfg(target_os = "macos")]
        if std::env::var("MM_GST_MAP").is_err() {
            if let Some(pb) = crate::mac::cv::pixbuf(f.buffer.as_ref()) {
                return Self::from_pixbuf(&pb, f.info.format() == gst_video::VideoFormat::P01010le);
            }
        }
        Self::from_gst(f)
    }

    /// The 8-bit video-range NV12 video_io's detect colour starts from: it asks AVFoundation for
    /// kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, which for a stream that decodes to something else (10-bit
    /// P010, full range) is VideoToolbox's pixel transfer of the decoded buffer (mac::vt; measured identical on the
    /// 10-bit SDR, HLG and PQ clips). Off the Mac, or without a CVPixelBuffer: the decoded planes, 10-bit codes / 4.
    pub fn for_detect(f: &crate::Frame, tags: Option<&str>) -> Result<Yuv> {
        #[cfg(target_os = "macos")]
        if std::env::var("MM_GST_MAP").is_err() {
            if let Some(pb) = crate::mac::cv::pixbuf(f.buffer.as_ref()) {
                use objc2_core_video::*;
                let v = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
                if CVPixelBufferGetPixelFormatType(&pb) == v {
                    return Self::from_pixbuf(&pb, false);
                }
                let t = crate::mac::vt::tags_for(tags, CVPixelBufferGetWidth(&pb) as u32);
                let d = crate::mac::vt::transfer(&pb, &t, v)?;
                return Self::from_pixbuf(&d, false);
            }
        }
        let _ = tags;
        Self::from_frame(f)
    }

    #[cfg(target_os = "macos")]
    fn from_pixbuf(pb: &objc2_core_video::CVPixelBuffer, ten: bool) -> Result<Yuv> {
        use objc2_core_video::*;
        unsafe {
            if CVPixelBufferLockBaseAddress(pb, CVPixelBufferLockFlags::ReadOnly) != kCVReturnSuccess {
                anyhow::bail!("CVPixelBufferLockBaseAddress");
            }
            let (w, h) = (CVPixelBufferGetWidthOfPlane(pb, 0), CVPixelBufferGetHeightOfPlane(pb, 0));
            let (cw, ch) = (CVPixelBufferGetWidthOfPlane(pb, 1), CVPixelBufferGetHeightOfPlane(pb, 1));
            let mut y = vec![0u16; w * h];
            let mut uv = vec![0u16; cw * ch * 2];
            for (p, out, pw, ph) in [(0usize, &mut y, w, h), (1, &mut uv, cw * 2, ch)] {
                let base = CVPixelBufferGetBaseAddressOfPlane(pb, p) as *const u8;
                let stride = CVPixelBufferGetBytesPerRowOfPlane(pb, p);
                for r in 0..ph {
                    let row = std::slice::from_raw_parts(base.add(r * stride), pw * if ten { 2 } else { 1 });
                    for c in 0..pw {
                        out[r * pw + c] = if ten { u16::from_le_bytes([row[2 * c], row[2 * c + 1]]) >> 6 } else { row[c] as u16 };
                    }
                }
            }
            CVPixelBufferUnlockBaseAddress(pb, CVPixelBufferLockFlags::ReadOnly);
            Ok(Yuv { w, h, depth: if ten { 10 } else { 8 }, y, uv })
        }
    }

    fn from_gst(f: &crate::Frame) -> Result<Yuv> {
        let vf = gst_video::VideoFrameRef::from_buffer_ref_readable(f.buffer.as_ref(), &f.info).map_err(|_| anyhow!("map"))?;
        let ten = f.info.format() == gst_video::VideoFormat::P01010le;
        let (w, h) = (f.info.width() as usize, f.info.height() as usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let mut y = vec![0u16; w * h];
        let mut uv = vec![0u16; cw * ch * 2];
        for (p, out, pw, ph) in [(0u32, &mut y, w, h), (1, &mut uv, cw * 2, ch)] {
            let stride = vf.plane_stride()[p as usize] as usize;
            let data = vf.plane_data(p).map_err(|_| anyhow!("plane"))?;
            for r in 0..ph {
                let row = &data[r * stride..];
                for c in 0..pw {
                    out[r * pw + c] = if ten { u16::from_le_bytes([row[2 * c], row[2 * c + 1]]) >> 6 } else { row[c] as u16 };
                }
            }
        }
        Ok(Yuv { w, h, depth: if ten { 10 } else { 8 }, y, uv })
    }
}

/// The stream's matrix (kr, kb), from its tags or, untagged, AVFoundation's rule by coded width.
pub fn matrix_for(tags: Option<&str>, coded_w: u32) -> (f32, f32) {
    let t = tags.unwrap_or("");
    // GStreamer's colorimetry: a name, or "range:matrix:transfer:primaries" (GstVideoColorMatrix: 3 BT709,
    // 4 BT601, 5 SMPTE240M, 6 BT2020)
    let parts: Vec<&str> = t.split(':').collect();
    let code = if parts.len() == 4 { parts[1].parse::<u32>().ok() } else { None };
    if t.starts_with("bt2020") || t.starts_with("bt2100") || code == Some(6) {
        return (0.2627, 0.0593);
    }
    if t.starts_with("bt601") || t.starts_with("smpte170m") || code == Some(4) {
        return (0.299, 0.114);
    }
    if t.starts_with("smpte240m") || code == Some(5) {
        return (0.212, 0.087);
    }
    if t.starts_with("bt709") || t.starts_with("sRGB") || code == Some(3) {
        return (0.2126, 0.0722);
    }
    // untagged (or matrix unknown): AVFoundation's rule by coded width
    if coded_w <= 704 { (0.299, 0.114) } else { (0.2126, 0.0722) }
}

/// AVFoundation's 32BGRA conversion as a 3x4 affine matrix on codes in 8-bit units (rows R, G, B; columns
/// Y, Cb, Cr, 1): out = M * (Y, Cb, Cr, 1), rounded. For 10-bit, divide the codes by 4 first. `full`: full-range
/// codes (not measured against AVFoundation: textbook).
pub fn display_matrix(kr: f32, kb: f32, depth: u32, full: bool) -> [[f32; 4]; 3] {
    let kg = 1.0 - kr - kb;
    let n = [[0.0, 2.0 * (1.0 - kr)], [-2.0 * (1.0 - kb) * kb / kg, -2.0 * (1.0 - kr) * kr / kg], [2.0 * (1.0 - kb), 0.0]];
    let (gy, gc, off, y0, c0) = if full {
        (1.0f32, 1.0f32, [0.0f32; 3], 0.0f32, 128.0f32)
    } else if depth > 8 {
        (1.16542, 255.0 / 224.0, [0.22, 0.55, 0.18], 16.0, 128.0)
    } else {
        (1.16422, 255.0 / 224.0 * 0.9960, [-0.19, 0.11, -0.25], 16.0, 128.0)
    };
    let mut m = [[0f32; 4]; 3];
    for c in 0..3 {
        m[c][0] = gy;
        m[c][1] = n[c][0] * gc;
        m[c][2] = n[c][1] * gc;
        m[c][3] = off[c] - gy * y0 - (m[c][1] + m[c][2]) * c0;
    }
    m
}

/// video_io._YUV2BGR (rows B, G, R): the detectors' BT.601 conversion.
pub const DETECT_BGR: [[f32; 4]; 3] = [
    [1.1642, 2.0167, 0.0, -278.2338],
    [1.1645, -0.3897, -0.8121, 133.9585],
    [1.1642, 0.0, 1.5956, -224.693],
];

/// YUV -> BGR at the stored size.
pub fn yuv_to_bgr(yuv: &Yuv, colour: Colour, matrix: [[f32; 4]; 3]) -> Bgr {
    let (w, h) = (yuv.w, yuv.h);
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut out = vec![0u8; w * h * 3];
    let s = if yuv.depth > 8 { 0.25f32 } else { 1.0 };
    match colour {
        Colour::Detect => {
            // rows on threads (each row on its own; the scan converts every frame it reads: 40 ms per 4K frame on
            // one core)
            mm_vision::mat::par_rows(&mut out, w * 3, |r, orow| {
                for c in 0..w {
                    let y = yuv.y[r * w + c] as f32 * s;
                    let k = ((r / 2) * cw + c / 2) * 2;
                    let (u, v) = (yuv.uv[k] as f32 * s, yuv.uv[k + 1] as f32 * s);
                    for (i, row) in DETECT_BGR.iter().enumerate() {
                        // cv2.transform in f32, then np.rint (half to even). The project's OpenCV is built with
                        // clang on arm64, whose default -ffp-contract=on fuses transform_'s
                        // `m0*v0 + m1*v1 + m2*v2 + m3` into fma(m2, v2, fma(m0, v0, m1*v1)) + m3: the plain
                        // sum differed on ~50-400 pixels per 4K party frame (see the module doc).
                        let x = row[2].mul_add(v, row[0].mul_add(y, row[1] * u)) + row[3];
                        orow[c * 3 + i] = x.round_ties_even().clamp(0.0, 255.0) as u8;
                    }
                }
            });
        }
        Colour::Display => {
            let uvat = |x: usize, y: usize| -> (f32, f32) {
                let k = (y.min(ch - 1) * cw + x.min(cw - 1)) * 2;
                (yuv.uv[k] as f32 * s, yuv.uv[k + 1] as f32 * s)
            };
            for r in 0..h {
                let (y0, fy) = (r / 2, if r % 2 == 1 { 0.5f32 } else { 0.0 });
                for c in 0..w {
                    let (x0, fx) = (c / 2, if c % 2 == 1 { 0.5f32 } else { 0.0 });
                    let (a, b, cc, d) = (uvat(x0, y0), uvat(x0 + 1, y0), uvat(x0, y0 + 1), uvat(x0 + 1, y0 + 1));
                    let lerp = |p: f32, q: f32, f: f32| p + (q - p) * f;
                    let u = lerp(lerp(a.0, b.0, fx), lerp(cc.0, d.0, fx), fy);
                    let v = lerp(lerp(a.1, b.1, fx), lerp(cc.1, d.1, fx), fy);
                    let y = yuv.y[r * w + c] as f32 * s;
                    for ch in 0..3 {
                        let m = matrix[ch];
                        let x = m[0] * y + m[1] * u + m[2] * v + m[3];
                        // rows are R, G, B; BGR out
                        out[(r * w + c) * 3 + (2 - ch)] = (x + 0.5).floor().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
    }
    Bgr { w: w as u32, h: h as u32, data: out }
}

/// cv2.resize(INTER_AREA) on BGR uint8, bit-exact with the project's cv2: mm-vision's resize (OpenCV's area tables,
/// its integer-factor fast path and Carotene's 2x/4x path, and the area-weighted bilinear when enlarging). This used to
/// be a local copy of the general area path only, off by one level on ~35 pixels a jazz frame.
pub fn resize_area(src: &Bgr, dw: u32, dh: u32) -> Bgr {
    if (src.w, src.h) == (dw, dh) {
        return src.clone();
    }
    let m = mm_vision::Mat::from_vec(src.h as usize, src.w as usize, 3, src.data.clone());
    let r = mm_vision::resize_u8(&m, dw as usize, dh as usize, 0.0, 0.0, mm_vision::Interp::Area);
    Bgr { w: dw, h: dh, data: r.data }
}

/// Bilinear resize with half-pixel centres (cv2.INTER_LINEAR's geometry; float maths, not its fixed point).
pub fn resize_linear(src: &Bgr, dw: u32, dh: u32) -> Bgr {
    let (sw, sh) = (src.w as usize, src.h as usize);
    let (dwu, dhu) = (dw as usize, dh as usize);
    let (fx, fy) = (sw as f32 / dwu as f32, sh as f32 / dhu as f32);
    let mut out = vec![0u8; dwu * dhu * 3];
    for y in 0..dhu {
        let sy = ((y as f32 + 0.5) * fy - 0.5).max(0.0);
        let y0 = (sy.floor() as usize).min(sh - 1);
        let y1 = (y0 + 1).min(sh - 1);
        let ay = sy - y0 as f32;
        for x in 0..dwu {
            let sx = ((x as f32 + 0.5) * fx - 0.5).max(0.0);
            let x0 = (sx.floor() as usize).min(sw - 1);
            let x1 = (x0 + 1).min(sw - 1);
            let ax = sx - x0 as f32;
            for c in 0..3 {
                let p = |yy: usize, xx: usize| src.data[(yy * sw + xx) * 3 + c] as f32;
                let v = (p(y0, x0) * (1.0 - ax) + p(y0, x1) * ax) * (1.0 - ay) + (p(y1, x0) * (1.0 - ax) + p(y1, x1) * ax) * ay;
                out[(y * dwu + x) * 3 + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Bgr { w: dw, h: dh, data: out }
}

/// Turn upright: `rot` clockwise degrees (video_io._upright).
pub fn upright(img: &Bgr, rot: u32) -> Bgr {
    let (w, h) = (img.w as usize, img.h as usize);
    match rot {
        90 | 270 => {
            // output row ny (h wide) on threads: 90 clockwise takes source (x = ny, y = h-1-nx); 270 (x = w-1-ny, y = nx)
            let mut out = vec![0u8; w * h * 3];
            mm_vision::mat::par_rows(&mut out, h * 3, |ny, orow| {
                for nx in 0..h {
                    let (x, y) = if rot == 90 { (ny, h - 1 - nx) } else { (w - 1 - ny, nx) };
                    let s = (y * w + x) * 3;
                    orow[nx * 3..nx * 3 + 3].copy_from_slice(&img.data[s..s + 3]);
                }
            });
            Bgr { w: h as u32, h: w as u32, data: out }
        }
        180 => {
            let mut out = img.data.clone();
            for (i, px) in img.data.chunks(3).enumerate() {
                let j = w * h - 1 - i;
                out[j * 3..j * 3 + 3].copy_from_slice(px);
            }
            Bgr { w: img.w, h: img.h, data: out }
        }
        _ => img.clone(),
    }
}

/// video_io.frames(color=...)'s picture for a decoded frame: converted, resized to the display size, upright.
pub fn frame_bgr(f: &crate::Frame, probe: &crate::Probe, tags: Option<&str>, colour: Colour) -> Result<Bgr> {
    let img = stored_bgr(f, tags, colour)?;
    let (dw, dh) = if probe.rotation == 90 || probe.rotation == 270 { (probe.height, probe.width) } else { (probe.width, probe.height) };
    let img = if (img.w, img.h) != (dw, dh) { resize_area(&img, dw, dh) } else { img };
    Ok(upright(&img, probe.rotation))
}

/// The display colour through VideoToolbox's pixel transfer (mac::vt), exactly AVFoundation's: on the Mac, when the
/// frame has a CVPixelBuffer, unless MM_DISPLAY_CPU is set (then the fitted model below). None: not this path.
#[allow(unused_variables)]
fn display_vt(f: &crate::Frame, tags: Option<&str>, colour: Colour) -> Option<Result<Bgr>> {
    #[cfg(target_os = "macos")]
    if colour == Colour::Display && std::env::var_os("MM_DISPLAY_CPU").is_none() && std::env::var_os("MM_GST_MAP").is_none() {
        let pb = crate::mac::cv::pixbuf(f.buffer.as_ref())?;
        let t = crate::mac::vt::tags_for(tags, objc2_core_video::CVPixelBufferGetWidth(&pb) as u32);
        return Some(crate::mac::vt::to_bgr(&pb, &t));
    }
    None
}

/// video_io's picture at the stored size, before the resize and turn: the detect colour, or the display colour
/// (VideoToolbox's on the Mac; the fitted CPU model off it, or with MM_DISPLAY_CPU).
pub fn stored_bgr(f: &crate::Frame, tags: Option<&str>, colour: Colour) -> Result<Bgr> {
    if let Some(img) = display_vt(f, tags, colour) {
        return img;
    }
    let yuv = if colour == Colour::Detect { Yuv::for_detect(f, tags)? } else { Yuv::from_frame(f)? };
    let (kr, kb) = matrix_for(tags, yuv.w as u32);
    let full = f.info.colorimetry().range() == gst_video::VideoColorRange::Range0_255 && tags.is_some();
    Ok(yuv_to_bgr(&yuv, colour, display_matrix(kr, kb, yuv.depth, full)))
}

/// BGR -> NV12 (BT.709 video range, chroma averaged over each 2x2), for the encoder.
pub fn bgr_to_nv12(img: &Bgr) -> Vec<u8> {
    let (w, h) = (img.w as usize, img.h as usize);
    let mut out = vec![0u8; w * h + w * h / 2];
    let (kr, kb) = (0.2126f32, 0.0722f32);
    let kg = 1.0 - kr - kb;
    let px = |x: usize, y: usize| {
        let i = (y * w + x) * 3;
        (img.data[i + 2] as f32, img.data[i + 1] as f32, img.data[i] as f32)
    };
    for y in 0..h {
        for x in 0..w {
            let (r, g, b) = px(x, y);
            out[y * w + x] = (16.0 + 219.0 / 255.0 * (kr * r + kg * g + kb * b)).round().clamp(0.0, 255.0) as u8;
        }
    }
    for y in 0..h / 2 {
        for x in 0..w / 2 {
            let (mut r, mut g, mut b) = (0.0, 0.0, 0.0);
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let p = px(2 * x + dx, 2 * y + dy);
                r += p.0 / 4.0;
                g += p.1 / 4.0;
                b += p.2 / 4.0;
            }
            let yy = kr * r + kg * g + kb * b;
            let cb = (b - yy) / (2.0 * (1.0 - kb));
            let cr = (r - yy) / (2.0 * (1.0 - kr));
            let i = w * h + y * w + 2 * x;
            out[i] = (128.0 + 224.0 / 255.0 * cb).round().clamp(0.0, 255.0) as u8;
            out[i + 1] = (128.0 + 224.0 / 255.0 * cr).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn untagged_matrix_rule() {
        assert_eq!(matrix_for(None, 704), (0.299, 0.114));
        assert_eq!(matrix_for(None, 708), (0.2126, 0.0722));
        assert_eq!(matrix_for(Some("bt2100-hlg"), 1920), (0.2627, 0.0593));
        assert_eq!(matrix_for(Some("bt601"), 1920), (0.299, 0.114));
    }

    #[test]
    fn black_and_white() {
        let m = display_matrix(0.2126, 0.0722, 8, false);
        let at = |y: f32| (0..3).map(|c| m[c][0] * y + m[c][1] * 128.0 + m[c][2] * 128.0 + m[c][3]).collect::<Vec<_>>();
        assert!(at(16.0).iter().all(|v| v.abs() < 0.3));
        assert!(at(235.0).iter().all(|v| (v - 255.0).abs() < 0.5));
    }

    #[test]
    fn area_same_as_box_for_halving() {
        let src = Bgr { w: 4, h: 2, data: (0..24).map(|i| (i * 10) as u8).collect() };
        let d = resize_area(&src, 2, 1);
        // each output = mean of a 2x2 block
        let mean = |c: usize, x0: usize| (src.data[x0 * 3 + c] as f32 + src.data[(x0 + 1) * 3 + c] as f32
            + src.data[12 + x0 * 3 + c] as f32 + src.data[12 + (x0 + 1) * 3 + c] as f32) / 4.0;
        assert_eq!(d.data[0], mean(0, 0).round_ties_even() as u8);
        assert_eq!(d.data[3], mean(0, 2).round_ties_even() as u8);
    }

    #[test]
    fn rotations() {
        let img = Bgr { w: 2, h: 1, data: vec![1, 1, 1, 2, 2, 2] };
        let r = upright(&img, 90);
        assert_eq!((r.w, r.h), (1, 2));
        assert_eq!(r.data, vec![1, 1, 1, 2, 2, 2]);
        let r = upright(&img, 270);
        assert_eq!(r.data, vec![2, 2, 2, 1, 1, 1]);
    }
}
