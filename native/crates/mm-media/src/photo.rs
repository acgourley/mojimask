//! Photos (video_io, "photos"): a photo is a clip of one frame. The scan, the questions, the editor and the export
//! all work on frame 0, and the export writes a photo back.
//!
//! Read by macOS's ImageIO (HEIC from iPhones, JPEG, PNG, ...): turned upright by its orientation tag (a
//! "thumbnail" at full size with the transform applied) and colour-managed into sRGB, the space the detectors were
//! tuned in; transparent parts on white. GStreamer has no HEIC decoder, so photos never go through it.
//!
//! Written with none of the original's metadata (no date, place or camera), only which app made it: JPEG (quality
//! 92) with EXIF Software and ImageDescription, or PNG (for a .png name) with Software and Comment text.
use crate::convert::Bgr;
use crate::decode::Probe;
use anyhow::{anyhow, bail, Result};

pub const IMAGE_EXT: [&str; 9] = [".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".webp", ".bmp"];
/// the one frame's nominal rate (the timeline needs one; nothing moves)
pub const PHOTO_FPS: f64 = 30.0;

pub fn is_image(path: &str) -> bool {
    let p = path.to_lowercase();
    IMAGE_EXT.iter().any(|e| p.ends_with(e))
}

pub fn probe(path: &str) -> Result<Probe> {
    let img = read_image(path)?;
    let d = 1.0 / PHOTO_FPS;
    Ok(Probe {
        width: img.w, height: img.h, stored_w: img.w, stored_h: img.h, par: (1, 1), fps: PHOTO_FPS,
        framerate: (30, 1), video_start: 0.0, video_duration: d, rotation: 0, n: 1, format: "image".into(),
        colorimetry: "sRGB".into(), has_audio: false, duration: d, image: true, source: "imageio".into(),
    })
}

#[cfg(target_os = "macos")]
mod imp {
    use super::*;
    use crate::mac::cv::cfkey;
    use objc2::rc::Retained;
    use objc2::runtime::AnyObject;
    use objc2_core_foundation::{CFDictionary, CFRetained, CFString, CFURLPathStyle, CFURL, CGPoint, CGRect, CGSize};
    use objc2_core_graphics::{CGBitmapContextCreate, CGBitmapContextCreateImage, CGColorSpace, CGContext, CGImage, CGImageAlphaInfo};
    use objc2_foundation::{NSDictionary, NSNumber, NSString};
    use objc2_image_io::*;

    fn url(path: &str) -> Result<CFRetained<CFURL>> {
        let abs = std::path::absolute(path)?;
        let s = CFString::from_str(&abs.to_string_lossy());
        CFURL::with_file_system_path(None, Some(&s), CFURLPathStyle::CFURLPOSIXPathStyle, false).ok_or_else(|| anyhow!("bad path"))
    }

    fn cf(d: &Retained<NSDictionary<NSString, AnyObject>>) -> &CFDictionary {
        unsafe { &*(Retained::as_ptr(d) as *const CFDictionary) }
    }

    fn num_prop(props: &NSDictionary<NSString, AnyObject>, key: &CFString) -> usize {
        props.objectForKey(cfkey(key)).and_then(|o| o.downcast::<NSNumber>().ok()).map(|n| n.as_usize()).unwrap_or(0)
    }

    pub fn read_image(path: &str) -> Result<Bgr> {
        unsafe {
            let u = url(path)?;
            let src = CGImageSource::with_url(&u, None).ok_or_else(|| anyhow!("not a picture macOS can read"))?;
            if src.count() == 0 {
                bail!("not a picture macOS can read");
            }
            let side = match src.properties_at_index(0, None) {
                Some(p) => {
                    let p: &NSDictionary<NSString, AnyObject> = &*(CFRetained::as_ptr(&p).as_ptr() as *const NSDictionary<NSString, AnyObject>);
                    num_prop(p, kCGImagePropertyPixelWidth).max(num_prop(p, kCGImagePropertyPixelHeight))
                }
                None => 0,
            };
            let yes = NSNumber::new_bool(true);
            let mut keys = vec![cfkey(kCGImageSourceCreateThumbnailFromImageAlways), cfkey(kCGImageSourceCreateThumbnailWithTransform),
                                cfkey(kCGImageSourceShouldCacheImmediately)];
            let sidev = NSNumber::new_usize(side);
            let mut vals: Vec<&AnyObject> = vec![yes.as_ref(), yes.as_ref(), yes.as_ref()];
            if side > 0 {
                keys.push(cfkey(kCGImageSourceThumbnailMaxPixelSize));
                vals.push(sidev.as_ref());
            }
            let opts: Retained<NSDictionary<NSString, AnyObject>> = NSDictionary::from_slices(&keys, &vals);
            let img = src.thumbnail_at_index(0, Some(cf(&opts))).ok_or_else(|| anyhow!("not a picture macOS can read"))?;
            let (w, h) = (CGImage::width(Some(&img)), CGImage::height(Some(&img)));
            let mut buf = vec![0u8; w * h * 4];
            let cs = CGColorSpace::with_name(Some(objc2_core_graphics::kCGColorSpaceSRGB)).ok_or_else(|| anyhow!("sRGB"))?;
            let ctx = CGBitmapContextCreate(buf.as_mut_ptr() as *mut _, w, h, 8, w * 4, Some(&cs), CGImageAlphaInfo::NoneSkipLast.0)
                .ok_or_else(|| anyhow!("bitmap context"))?;
            let rect = CGRect { origin: CGPoint { x: 0.0, y: 0.0 }, size: CGSize { width: w as f64, height: h as f64 } };
            CGContext::set_rgb_fill_color(Some(&ctx), 1.0, 1.0, 1.0, 1.0);
            CGContext::fill_rect(Some(&ctx), rect);
            CGContext::draw_image(Some(&ctx), rect, Some(&img));
            drop(ctx);
            let mut data = vec![0u8; w * h * 3];
            for (i, px) in buf.chunks_exact(4).enumerate() {
                data[i * 3] = px[2];
                data[i * 3 + 1] = px[1];
                data[i * 3 + 2] = px[0];
            }
            Ok(Bgr { w: w as u32, h: h as u32, data })
        }
    }

    pub fn write_image(path: &str, img: &Bgr, software: &str, description: &str, quality: f64) -> Result<()> {
        let png = path.to_lowercase().ends_with(".png");
        unsafe {
            let (w, h) = (img.w as usize, img.h as usize);
            let mut buf = vec![255u8; w * h * 4];
            for (i, px) in img.data.chunks_exact(3).enumerate() {
                buf[i * 4] = px[2];
                buf[i * 4 + 1] = px[1];
                buf[i * 4 + 2] = px[0];
            }
            let cs = CGColorSpace::with_name(Some(objc2_core_graphics::kCGColorSpaceSRGB)).ok_or_else(|| anyhow!("sRGB"))?;
            let ctx = CGBitmapContextCreate(buf.as_mut_ptr() as *mut _, w, h, 8, w * 4, Some(&cs), CGImageAlphaInfo::NoneSkipLast.0)
                .ok_or_else(|| anyhow!("bitmap context"))?;
            let cg = CGBitmapContextCreateImage(Some(&ctx)).ok_or_else(|| anyhow!("image"))?;
            let typ = CFString::from_str(if png { "public.png" } else { "public.jpeg" });
            let u = url(path)?;
            let dest = CGImageDestination::with_url(&u, &typ, 1, None).ok_or_else(|| anyhow!("can't write {path}"))?;
            let sw = NSString::from_str(software);
            let de = NSString::from_str(description);
            let props: Retained<NSDictionary<NSString, AnyObject>> = if png {
                let p: Retained<NSDictionary<NSString, AnyObject>> = NSDictionary::from_slices(
                    &[cfkey(kCGImagePropertyPNGSoftware), cfkey(kCGImagePropertyPNGComment)], &[sw.as_ref() as &AnyObject, de.as_ref()]);
                NSDictionary::from_slices(&[cfkey(kCGImagePropertyPNGDictionary)], &[p.as_ref() as &AnyObject])
            } else {
                let t: Retained<NSDictionary<NSString, AnyObject>> = NSDictionary::from_slices(
                    &[cfkey(kCGImagePropertyTIFFSoftware), cfkey(kCGImagePropertyTIFFImageDescription)], &[sw.as_ref() as &AnyObject, de.as_ref()]);
                let q = NSNumber::new_f64(quality);
                NSDictionary::from_slices(&[cfkey(kCGImagePropertyTIFFDictionary), cfkey(kCGImageDestinationLossyCompressionQuality)],
                                          &[t.as_ref() as &AnyObject, q.as_ref()])
            };
            dest.add_image(&cg, Some(cf(&props)));
            if !dest.finalize() {
                bail!("can't write {path}");
            }
            Ok(())
        }
    }
}

#[cfg(target_os = "macos")]
pub use imp::{read_image, write_image};

#[cfg(not(target_os = "macos"))]
pub fn read_image(_path: &str) -> Result<Bgr> {
    bail!("photos are read through ImageIO (macOS only so far)")
}
#[cfg(not(target_os = "macos"))]
pub fn write_image(_path: &str, _img: &Bgr, _software: &str, _description: &str, _quality: f64) -> Result<()> {
    bail!("photos are written through ImageIO (macOS only so far)")
}
