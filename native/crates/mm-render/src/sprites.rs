//! Cover pictures ("sprites"), as track.emoji_sprite and portrait.sprite draw them, and their atlas slices.
//!
//! - **Emoji** (`emoji`, macOS): the emoji picture every cover uses (export and live preview alike), cropped to its
//!   ink, size x size RGBA. Drawn by macOS's own text engine (CoreText, through AppKit, exactly as
//!   `track._emoji_glyph_macos`: an NSAttributedString in Apple Color Emoji at 160 px, drawAtPoint centred on a 288 px
//!   canvas): the same drawing the browser does, so joined emoji (ZWJ sequences like woman + microphone, skin tones,
//!   flags) come out as ONE picture. Then un-premultiplied, cropped to alpha > 0 and resized to 160 with Pillow's
//!   Lanczos ([`pil_resize_rgba`], a port of Pillow's fixed-point resampler, premultiplied as Pillow does for RGBA).
//!   Not ported: the HEAD_ONLY cut for the sheep and ram (they only exist as whole animals); they are drawn whole.
//! - **Portrait** (`portrait`): the photo cropped to its cut-out (alpha > 24), centred on a transparent square, resized
//!   (Lanczos) to twice the emoji size (portrait.sprite).
//! - **Atlas slice** (`slice`): a straight-alpha RGBA picture -> the compositor's slice: BGR premultiplied f16 in the
//!   top-left n x n of a TEX x TEX slice (pictures over TEX are first resized to TEX, INTER_AREA), with mips (2x2 box,
//!   as Metal's generateMipmaps).
use anyhow::{anyhow, Result};
use mm_core::cover::Slice;

/// emoji_sprite's size; portraits are drawn at twice it.
pub const SPRITE: u32 = 160;

// ------------------------------------------------------------------ Pillow's resampler (Resample.c), Lanczos

const PRECISION_BITS: u32 = 32 - 8 - 2;

fn lanczos(x: f64) -> f64 {
    fn sinc(x: f64) -> f64 {
        if x == 0.0 {
            1.0
        } else {
            let x = x * std::f64::consts::PI;
            x.sin() / x
        }
    }
    if (-3.0..3.0).contains(&x) {
        sinc(x) * sinc(x / 3.0)
    } else {
        0.0
    }
}

/// Pillow's precompute_coeffs + normalize_coeffs_8bpc: per output pixel (first input pixel, count, fixed-point weights).
fn coeffs(in_size: usize, out_size: usize) -> (usize, Vec<(usize, usize)>, Vec<i32>) {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 3.0 * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(out_size);
    let mut kk = vec![0i32; out_size * ksize];
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = (((center + support + 0.5) as i64).min(in_size as i64) as usize) - xmin;
        let mut k = vec![0.0f64; xmax];
        let mut ww = 0.0;
        for (x, kx) in k.iter_mut().enumerate() {
            let w = lanczos((x as f64 + xmin as f64 - center + 0.5) * ss);
            *kx = w;
            ww += w;
        }
        for (x, kx) in k.iter().enumerate() {
            let v = if ww != 0.0 { kx / ww } else { *kx };
            let f = v * (1u64 << PRECISION_BITS) as f64;
            kk[xx * ksize + x] = if v < 0.0 { (-0.5 + f) as i32 } else { (0.5 + f) as i32 };
        }
        bounds.push((xmin, xmax));
    }
    (ksize, bounds, kk)
}

fn clip8(v: i64) -> u8 {
    if v >= (1i64 << PRECISION_BITS << 8) {
        255
    } else if v <= 0 {
        0
    } else {
        (v >> PRECISION_BITS) as u8
    }
}

/// Pillow's ImagingResample (Lanczos) on 4-channel 8-bit pixels: horizontal pass, round to 8 bits, vertical pass.
fn pil_resample4(src: &[u8], w: usize, h: usize, ow: usize, oh: usize) -> Vec<u8> {
    let (kh, bh, kkh) = coeffs(w, ow);
    let (kv, bv, kkv) = coeffs(h, oh);
    let need_h = ow != w;
    let need_v = oh != h;
    let mut tmp;
    let (mid, mw): (&[u8], usize) = if need_h {
        tmp = vec![0u8; ow * h * 4];
        for y in 0..h {
            for (xx, &(xmin, xmax)) in bh.iter().enumerate() {
                let k = &kkh[xx * kh..xx * kh + xmax];
                for c in 0..4 {
                    let mut ss: i64 = 1 << (PRECISION_BITS - 1);
                    for (x, &kx) in k.iter().enumerate() {
                        ss += src[(y * w + x + xmin) * 4 + c] as i64 * kx as i64;
                    }
                    tmp[(y * ow + xx) * 4 + c] = clip8(ss);
                }
            }
        }
        (&tmp[..], ow)
    } else {
        (src, w)
    };
    if !need_v {
        return mid.to_vec();
    }
    let mut out = vec![0u8; mw * oh * 4];
    for (yy, &(ymin, ymax)) in bv.iter().enumerate() {
        let k = &kkv[yy * kv..yy * kv + ymax];
        for x in 0..mw {
            for c in 0..4 {
                let mut ss: i64 = 1 << (PRECISION_BITS - 1);
                for (y, &ky) in k.iter().enumerate() {
                    ss += mid[((y + ymin) * mw + x) * 4 + c] as i64 * ky as i64;
                }
                out[(yy * mw + x) * 4 + c] = clip8(ss);
            }
        }
    }
    out
}

/// PIL `Image.resize(size, LANCZOS)` of an RGBA image: premultiplied (RGBA -> RGBa, MULDIV255), resampled, then
/// un-premultiplied (255 * c / a, truncated; a of 0 or 255 copied).
pub fn pil_resize_rgba(img: &image::RgbaImage, ow: u32, oh: u32) -> image::RgbaImage {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut pm = img.as_raw().clone();
    for p in pm.chunks_exact_mut(4) {
        let a = p[3] as u32;
        for c in &mut p[..3] {
            let t = *c as u32 * a + 128;
            *c = (((t >> 8) + t) >> 8) as u8;
        }
    }
    let mut out = pil_resample4(&pm, w, h, ow as usize, oh as usize);
    for p in out.chunks_exact_mut(4) {
        let a = p[3] as u32;
        if a != 0 && a != 255 {
            for c in &mut p[..3] {
                *c = ((255 * *c as u32) / a).min(255) as u8;
            }
        }
    }
    image::RgbaImage::from_raw(ow, oh, out).unwrap()
}

/// PIL getbbox of the pixels where `keep(alpha)`: (x0, y0, x1, y1) or None.
fn bbox(img: &image::RgbaImage, keep: impl Fn(u8) -> bool) -> Option<(u32, u32, u32, u32)> {
    let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0, 0);
    for (x, y, p) in img.enumerate_pixels() {
        if keep(p.0[3]) {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x + 1);
            y1 = y1.max(y + 1);
        }
    }
    (x0 != u32::MAX).then_some((x0, y0, x1, y1))
}

// ------------------------------------------------------------------ emoji (CoreText through AppKit)

#[cfg(target_os = "macos")]
static GLYPH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// track._emoji_glyph_macos: `ch` drawn by AppKit in Apple Color Emoji at `px`, centred on a (1.8 px)^2 canvas, as
/// straight-alpha RGBA (un-premultiplied in float, truncated, as numpy's astype does).
#[cfg(target_os = "macos")]
pub fn emoji_glyph(ch: &str, px: f64) -> Result<image::RgbaImage> {
    use objc2::rc::Retained;
    use objc2::runtime::AnyObject;
    use objc2::AnyThread;
    use objc2_app_kit::{NSAttributedStringNSStringDrawing, NSFont, NSFontAttributeName, NSGraphicsContext};
    use objc2_core_foundation::CGPoint;
    use objc2_core_graphics::{CGBitmapContextCreate, CGImageAlphaInfo};
    use objc2_foundation::{NSAttributedString, NSDictionary, NSString};
    let _g = GLYPH_LOCK.lock().unwrap();
    let w = (px * 1.8) as usize;
    let h = w;
    let mut buf = vec![0u8; w * h * 4];
    unsafe {
        let cs = objc2_core_graphics::CGColorSpace::new_device_rgb();
        let ctx = CGBitmapContextCreate(buf.as_mut_ptr() as *mut _, w, h, 8, w * 4, cs.as_deref(),
                                        CGImageAlphaInfo::PremultipliedLast.0)
            .ok_or_else(|| anyhow!("no bitmap context"))?;
        let gc = NSGraphicsContext::graphicsContextWithCGContext_flipped(&ctx, false);
        let font = NSFont::fontWithName_size(&NSString::from_str("Apple Color Emoji"), px)
            .ok_or_else(|| anyhow!("no Apple Color Emoji font"))?;
        let fobj: &AnyObject = &font;
        let attrs: Retained<NSDictionary<NSString, AnyObject>> =
            NSDictionary::from_slices(&[NSFontAttributeName], &[fobj]);
        let s = NSAttributedString::initWithString_attributes(NSAttributedString::alloc(), &NSString::from_str(ch),
                                                              Some(&attrs));
        NSGraphicsContext::saveGraphicsState_class();
        NSGraphicsContext::setCurrentContext(Some(&gc));
        let sz = s.size();
        s.drawAtPoint(CGPoint { x: (w as f64 - sz.width) / 2.0, y: (h as f64 - sz.height) / 2.0 });
        gc.flushGraphics();
        NSGraphicsContext::restoreGraphicsState_class();
        drop(gc);
        drop(ctx);
    }
    for p in buf.chunks_exact_mut(4) {
        let a = (p[3] as f64).max(1.0);
        for c in &mut p[..3] {
            *c = (*c as f64 * 255.0 / a).clamp(0.0, 255.0) as u8;
        }
    }
    Ok(image::RgbaImage::from_raw(w as u32, h as u32, buf).unwrap())
}

/// track.emoji_sprite for an emoji: the glyph cropped to its ink (alpha > 0), resized to size x size (Lanczos).
#[cfg(target_os = "macos")]
pub fn emoji(ch: &str, size: u32) -> Result<image::RgbaImage> {
    let g = emoji_glyph(ch, 160.0)?;
    if !g.pixels().any(|p| p.0[3] > 8) {
        return Err(anyhow!("CoreText drew nothing for {ch:?}"));
    }
    let (x0, y0, x1, y1) = bbox(&g, |a| a > 0).unwrap();
    let c = image::imageops::crop_imm(&g, x0, y0, x1 - x0, y1 - y0).to_image();
    Ok(pil_resize_rgba(&c, size, size))
}

/// portrait.sprite: the photo (RGBA, background removed) cropped to its cut-out (alpha > 24), centred on a transparent
/// square, resized to size x size (Lanczos).
pub fn portrait(photo: &image::RgbaImage, size: u32) -> image::RgbaImage {
    let Some((x0, y0, x1, y1)) = bbox(photo, |a| a > 24) else {
        return image::RgbaImage::new(size, size);
    };
    let c = image::imageops::crop_imm(photo, x0, y0, x1 - x0, y1 - y0).to_image();
    let s = c.width().max(c.height());
    let mut sq = image::RgbaImage::new(s, s);
    image::imageops::replace(&mut sq, &c, ((s - c.width()) / 2) as i64, ((s - c.height()) / 2) as i64);
    pil_resize_rgba(&sq, size, size)
}

/// The picture for a cover key: an emoji (CoreText) at SPRITE px, or for a portrait key the photo at twice that
/// (`photo` reads the portrait file: the default photo for portrait::PORTRAIT_KEY).
#[cfg(target_os = "macos")]
pub fn for_key(key: &str, photo: &dyn Fn(&str) -> Result<image::RgbaImage>) -> Result<image::RgbaImage> {
    if mm_core::cover::is_portrait(key) {
        return Ok(portrait(&photo(key)?, SPRITE * 2));
    }
    emoji(key, SPRITE)
}

// ------------------------------------------------------------------ atlas slices

/// cv2.resize INTER_AREA for a (non-integer) shrink of an RGBA u8 picture: each output pixel the area-weighted mean
/// of the input pixels it covers (rounded). Only used for pictures over TEX (never at today's sizes).
fn area_resize(img: &image::RgbaImage, n: u32) -> image::RgbaImage {
    let (w, h) = (img.width() as f64, img.height() as f64);
    let (fx, fy) = (w / n as f64, h / n as f64);
    image::RgbaImage::from_fn(n, n, |ox, oy| {
        let (x0, x1, y0, y1) = (ox as f64 * fx, (ox + 1) as f64 * fx, oy as f64 * fy, (oy + 1) as f64 * fy);
        let mut acc = [0.0f64; 4];
        let mut ws = 0.0;
        let mut y = y0.floor();
        while y < y1 {
            let wy = (y1.min(y + 1.0) - y0.max(y)).max(0.0);
            let mut x = x0.floor();
            while x < x1 {
                let wx = (x1.min(x + 1.0) - x0.max(x)).max(0.0);
                let p = img.get_pixel((x as u32).min(img.width() - 1), (y as u32).min(img.height() - 1)).0;
                for c in 0..4 {
                    acc[c] += wx * wy * p[c] as f64;
                }
                ws += wx * wy;
                x += 1.0;
            }
            y += 1.0;
        }
        image::Rgba(acc.map(|v| (v / ws).round().clamp(0.0, 255.0) as u8))
    })
}

/// A straight-alpha RGBA picture -> its atlas slice (gpu_draw.Compositor.slot): BGR premultiplied f16 in the top-left
/// n x n of the TEX x TEX slice, zero elsewhere; every mip level (2x2 box).
pub fn slice(img: &image::RgbaImage, tex: u32) -> Slice {
    let img = if img.width() > tex { area_resize(img, tex) } else { img.clone() };
    let n = img.width();
    let t = tex as usize;
    let mut l0 = vec![0.0f32; t * t * 4];
    for (x, y, p) in img.enumerate_pixels() {
        let a = p.0[3] as f32 / 255.0;
        let i = (y as usize * t + x as usize) * 4;
        l0[i] = p.0[2] as f32 / 255.0 * a;
        l0[i + 1] = p.0[1] as f32 / 255.0 * a;
        l0[i + 2] = p.0[0] as f32 / 255.0 * a;
        l0[i + 3] = a;
    }
    let to16 = |v: &[f32]| v.iter().map(|&x| half::f16::from_f32(x)).collect::<Vec<_>>();
    let mut levels = vec![to16(&l0)];
    let mut prev = l0.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect::<Vec<f32>>();
    let mut s = t;
    while s > 1 {
        let ns = (s / 2).max(1);
        let mut nl = vec![0.0f32; ns * ns * 4];
        for y in 0..ns {
            for x in 0..ns {
                for c in 0..4 {
                    let mut acc = 0.0f32;
                    for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        acc += prev[((y * 2 + dy).min(s - 1) * s + (x * 2 + dx).min(s - 1)) * 4 + c];
                    }
                    nl[(y * ns + x) * 4 + c] = acc / 4.0;
                }
            }
        }
        let h16 = to16(&nl);
        prev = h16.iter().map(|x| x.to_f32()).collect();
        levels.push(h16);
        s = ns;
    }
    Slice { n, levels }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resize_identity_and_shape() {
        let img = image::RgbaImage::from_fn(10, 7, |x, y| image::Rgba([x as u8 * 20, y as u8 * 30, 7, 255]));
        let r = pil_resize_rgba(&img, 10, 7);
        assert_eq!(r, img);
        let r = pil_resize_rgba(&img, 5, 3);
        assert_eq!(r.dimensions(), (5, 3));
    }
}
