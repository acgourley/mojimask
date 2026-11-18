//! Output formats for sharing (formats.py): a few presets with the shapes social apps expect, and how the picture is
//! fitted into them. Files are H.264 video + AAC audio in .mp4 (plays everywhere). Covers are drawn on the full
//! frame first, then the frame is cropped or fitted, so framing can never uncover anyone.
//!
//! Presets never upscale: a 720p clip exported as "Vertical 9:16" stays at its own resolution in that shape
//! (upscaling adds file size, not detail). Frame rates above 60 are reduced to ~30 for sharing.
//!
//! Here: the presets, the sizes and the fit geometry ([`FitPlan`]), exactly as formats.py computes them (Python's
//! `round` is half to even, so is ours). The drawing belongs to the GPU (mm-render: the draw stream's blur); this
//! module also has a CPU version of `fit_frame` ([`fit_frame_cpu`]) for tests and the headless export. Its blur is
//! a plain separable Gaussian (sigma 6, OpenCV's kernel size and reflect-101 border, f32), not cv2's fixed-point
//! one: the blurred background is 8x smaller and darkened, so this is a picture match, not a bit match.
use crate::convert::{resize_area, resize_linear, Bgr};
use crate::decode::Probe;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct Preset {
    pub id: &'static str,
    pub label: &'static str,
    pub detail: &'static str,
    pub aspect: Option<(u32, u32)>,
    pub long: u32,
}

pub const PRESETS: [Preset; 5] = [
    Preset { id: "original", label: "Original", detail: "Same shape, size and frame rate as your clip", aspect: None, long: 0 },
    Preset { id: "vertical", label: "Vertical 9:16", detail: "Reels, TikTok, YouTube Shorts, Stories", aspect: Some((9, 16)), long: 1920 },
    Preset { id: "portrait", label: "Portrait 4:5", detail: "Instagram and Facebook feed posts", aspect: Some((4, 5)), long: 1350 },
    Preset { id: "square", label: "Square 1:1", detail: "Square feed posts", aspect: Some((1, 1)), long: 1080 },
    Preset { id: "landscape", label: "Landscape 16:9", detail: "YouTube, X, Facebook, LinkedIn", aspect: Some((16, 9)), long: 1920 },
];
pub const FITS: [(&str, &str); 2] = [("fill", "Crop to fill"), ("fit", "Whole frame, blurred background")];

pub fn preset(id: &str) -> &'static Preset {
    PRESETS.iter().find(|p| p.id == id).unwrap_or(&PRESETS[0])
}

/// Python's round(): half to even.
fn pround(x: f64) -> i64 {
    x.round_ties_even() as i64
}

fn even(x: f64) -> u32 {
    (pround(x / 2.0) * 2).max(2) as u32
}

/// (width, height) of the export (even, as H.264 needs), or for a photo any size.
pub fn output_size(preset_id: &str, src_w: u32, src_h: u32, fit: &str, photo: bool) -> (u32, u32) {
    let p = preset(preset_id);
    let ev = |x: f64| if photo { pround(x).max(1) as u32 } else { even(x) };
    let (sw, sh) = (src_w as f64, src_h as f64);
    let Some((a, b)) = p.aspect else { return (ev(sw), ev(sh)) };
    let (a, b, long) = (a as f64, b as f64, p.long as f64);
    let (w, h) = if a >= b { (long, long * b / a) } else { (long * a / b, long) };
    let k = if fit == "fill" {
        let (cw, _ch) = if sw / sh > a / b { (sh * a / b, sh) } else { (sw, sw * b / a) };
        (cw / w).min(1.0)
    } else {
        (sw / w).max(sh / h).min(1.0)
    };
    (ev(w * k), ev(h * k))
}

#[derive(Clone, Debug, Serialize)]
pub struct Settings {
    pub preset: String,
    pub fit: String,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub step: usize,
    pub start: usize,
    pub end: usize,
    pub bitrate: u64,
    pub seconds: f64,
}

/// Everything the exporter needs, from the clip's probe and the user's choices (formats.settings).
pub fn settings(meta: &Probe, preset_id: &str, fit: &str, start: Option<f64>, end: Option<f64>, frames: Option<usize>) -> Settings {
    let fps = meta.fps;
    let n = frames.unwrap_or_else(|| (pround(meta.video_duration * fps) as usize).max(1));
    let s = (pround(start.unwrap_or(0.0) * fps)).clamp(0, n as i64 - 1) as usize;
    let e = match end {
        None => n,
        Some(x) if x == 0.0 => n,
        Some(x) => (pround(x * fps).min(n as i64)).max(s as i64 + 1) as usize,
    };
    let step = if preset_id == "original" || fps <= 60.0 { 1 } else { (pround(fps / 30.0) as usize).max(1) };
    let (w, h) = output_size(preset_id, meta.width, meta.height, fit, meta.image);
    let out_fps = fps / step as f64;
    let bitrate = (40e6f64).min((2e6f64).max(0.12 * w as f64 * h as f64 * out_fps)) as u64;
    Settings {
        preset: preset_id.into(), fit: fit.into(), width: w, height: h, fps: out_fps, step, start: s, end: e, bitrate,
        seconds: (e - s) as f64 / fps,
    }
}

/// A resize then a crop: scale the picture to `scaled`, then take `size` from `crop` (top-left) (formats._cover).
#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub struct Cover {
    pub scaled: (u32, u32),
    pub crop: (u32, u32),
    pub size: (u32, u32),
    /// shrinking (INTER_AREA) rather than enlarging (INTER_LINEAR)
    pub shrink: bool,
}

pub fn cover(w: u32, h: u32, tw: u32, th: u32) -> Cover {
    let s = (tw as f64 / w as f64).max(th as f64 / h as f64);
    let nw = (tw as i64).max(pround(w as f64 * s)) as u32;
    let nh = (th as i64).max(pround(h as f64 * s)) as u32;
    Cover { scaled: (nw, nh), crop: ((nw - tw) / 2, (nh - th) / 2), size: (tw, th), shrink: s < 1.0 }
}

/// How a `src` frame becomes the `out` frame (formats.fit_frame), for the GPU or the CPU.
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum FitPlan {
    /// same size: nothing to do
    Same,
    /// centre crop of the scaled frame
    Fill { cover: Cover },
    /// the whole frame scaled to `fg` at `at`, over a background: the frame shrunk to `small` (area), cropped to fill
    /// `bg_cover`, blurred (Gaussian sigma `sigma`), enlarged (bilinear) to the output and darkened by `darken`
    Fit { fg: (u32, u32), at: (u32, u32), fg_shrink: bool, small: (u32, u32), bg_cover: Cover, sigma: f64, darken: f32 },
}

pub fn fit_plan(w: u32, h: u32, ow: u32, oh: u32, fit: &str) -> FitPlan {
    if (w, h) == (ow, oh) {
        return FitPlan::Same;
    }
    if fit == "fill" {
        return FitPlan::Fill { cover: cover(w, h, ow, oh) };
    }
    let small = ((w / 8).max(2), (h / 8).max(2));
    let bg_cover = cover(small.0, small.1, (ow / 8).max(2), (oh / 8).max(2));
    let s = (ow as f64 / w as f64).min(oh as f64 / h as f64);
    let fg = ((pround(w as f64 * s)).max(2) as u32, (pround(h as f64 * s)).max(2) as u32);
    FitPlan::Fit { fg, at: ((ow - fg.0) / 2, (oh - fg.1) / 2), fg_shrink: s < 1.0, small, bg_cover, sigma: 6.0, darken: 0.6 }
}

fn crop(img: &Bgr, x: u32, y: u32, w: u32, h: u32) -> Bgr {
    let mut data = Vec::with_capacity((w * h * 3) as usize);
    for r in y..y + h {
        let s = ((r * img.w + x) * 3) as usize;
        data.extend_from_slice(&img.data[s..s + (w * 3) as usize]);
    }
    Bgr { w, h, data }
}

fn apply_cover(img: &Bgr, c: &Cover) -> Bgr {
    let r = if c.shrink { resize_area(img, c.scaled.0, c.scaled.1) } else { resize_linear(img, c.scaled.0, c.scaled.1) };
    crop(&r, c.crop.0, c.crop.1, c.size.0, c.size.1)
}

/// Separable Gaussian blur, OpenCV's kernel size for 8-bit (2*round(4 sigma)+1) and BORDER_REFLECT_101.
pub fn gaussian_blur(img: &Bgr, sigma: f64) -> Bgr {
    let k = ((sigma * 4.0 * 2.0 + 1.0).round() as i64) | 1;
    let r = (k / 2) as i64;
    let mut wts: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f64 / (2.0 * sigma * sigma)).exp() as f32).collect();
    let sum: f32 = wts.iter().sum();
    wts.iter_mut().for_each(|w| *w /= sum);
    let (w, h) = (img.w as i64, img.h as i64);
    let refl = |i: i64, n: i64| -> i64 {
        if n == 1 {
            return 0;
        }
        let mut i = i;
        while i < 0 || i >= n {
            if i < 0 {
                i = -i;
            }
            if i >= n {
                i = 2 * n - 2 - i;
            }
        }
        i
    };
    let mut tmp = vec![0f32; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let mut a = 0f32;
                for (j, wt) in wts.iter().enumerate() {
                    let xx = refl(x + j as i64 - r, w);
                    a += wt * img.data[((y * w + xx) * 3 + c) as usize] as f32;
                }
                tmp[((y * w + x) * 3 + c) as usize] = a;
            }
        }
    }
    let mut out = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let mut a = 0f32;
                for (j, wt) in wts.iter().enumerate() {
                    let yy = refl(y + j as i64 - r, h);
                    a += wt * tmp[((yy * w + x) * 3 + c) as usize];
                }
                out[((y * w + x) * 3 + c) as usize] = a.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Bgr { w: img.w, h: img.h, data: out }
}

/// formats.fit_frame on the CPU (see the module doc for how exact it is).
pub fn fit_frame_cpu(img: &Bgr, ow: u32, oh: u32, fit: &str) -> Bgr {
    match fit_plan(img.w, img.h, ow, oh, fit) {
        FitPlan::Same => img.clone(),
        FitPlan::Fill { cover } => apply_cover(img, &cover),
        FitPlan::Fit { fg, at, fg_shrink, small, bg_cover, sigma, darken } => {
            let sm = apply_cover(&resize_area(img, small.0, small.1), &bg_cover);
            let bg = resize_linear(&gaussian_blur(&sm, sigma), ow, oh);
            let mut out = Bgr { w: ow, h: oh, data: bg.data.iter().map(|&v| (v as f32 * darken) as u8).collect() };
            let f = if fg_shrink { resize_area(img, fg.0, fg.1) } else { resize_linear(img, fg.0, fg.1) };
            for r in 0..fg.1 {
                let d = (((at.1 + r) * ow + at.0) * 3) as usize;
                let s = ((r * fg.0) * 3) as usize;
                out.data[d..d + (fg.0 * 3) as usize].copy_from_slice(&f.data[s..s + (fg.0 * 3) as usize]);
            }
            out
        }
    }
}

/// Presets + fits for the UI (formats.describe).
pub fn describe() -> serde_json::Value {
    serde_json::json!({
        "presets": PRESETS.iter().map(|p| {
            let mut v = serde_json::json!({"id": p.id, "label": p.label, "detail": p.detail, "aspect": p.aspect.map(|(a, b)| vec![a, b])});
            if p.long > 0 { v["long"] = p.long.into(); }
            v
        }).collect::<Vec<_>>(),
        "fits": FITS.iter().map(|(k, v)| (k.to_string(), serde_json::Value::from(*v))).collect::<serde_json::Map<_, _>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_like_python() {
        // formats.output_size in Python (2026-11-17); round() is half to even there
        assert_eq!(output_size("original", 643, 476, "fill", false), (644, 476));
        assert_eq!(output_size("vertical", 643, 476, "fit", false), (644, 1144));
        assert_eq!(output_size("square", 643, 476, "fill", false), (476, 476));
        assert_eq!(output_size("square", 643, 476, "fill", true), (476, 476));
        assert_eq!(output_size("landscape", 3840, 2160, "fill", false), (1920, 1080));
    }
}
