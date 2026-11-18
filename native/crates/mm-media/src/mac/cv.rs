//! macOS: GStreamer buffer -> CVPixelBuffer -> Metal textures per plane -> wgpu, no CPU copy (and back, for vtenc).
//!
//! applemedia (vtdec) attaches a GstCoreVideoMeta to each buffer. It is not public API: its API type is registered as
//! "GstCoreVideoMetaAPI" and its struct is `{ GstMeta meta; CVBufferRef cvbuf; CVPixelBufferRef pixbuf; }`
//! (gst-plugins-bad sys/applemedia/coremediabuffer.h). We read `pixbuf`, then CVMetalTextureCache gives an MTLTexture
//! aliasing each IOSurface plane (Y: R8/R16, CbCr: RG8/RG16), which wgpu-hal wraps without copying.
//! The reverse (`wrap_pixbuf`) attaches the same meta to a buffer we make, so vtenc encodes our CVPixelBuffer as is.
use anyhow::{anyhow, bail, Result};
use gstreamer as gst;
use gstreamer::glib;
use gstreamer::glib::translate::*;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_foundation::CFRetained;
use objc2_core_video::*;
use objc2_foundation::{NSDictionary, NSNumber, NSString};
use objc2_metal::{MTLPixelFormat, MTLTexture, MTLTextureType};
use std::ptr::NonNull;

#[repr(C)]
struct GstCoreVideoMeta {
    meta: gst::ffi::GstMeta,
    cvbuf: *mut CVBuffer,
    pixbuf: *mut CVPixelBuffer,
}

fn api_type() -> Option<glib::ffi::GType> {
    let t = glib::Type::from_name("GstCoreVideoMetaAPI")?;
    Some(t.into_glib())
}

extern "C" {
    fn CFGetTypeID(cf: *const std::ffi::c_void) -> usize;
    fn CVPixelBufferGetTypeID() -> usize;
}

/// Check a private applemedia meta against the struct we read it as: registered under `name`, with the size of
/// `size` bytes (GstMeta + its pointers). Err says what changed.
pub fn check_meta(name: &std::ffi::CStr, size: usize) -> Result<(), String> {
    unsafe {
        let info = gst::ffi::gst_meta_get_info(name.as_ptr());
        if info.is_null() {
            return Err(format!("{name:?} is not registered"));
        }
        let got = (*info).size;
        if got != size {
            return Err(format!("{name:?} is {got} bytes, we read it as {size}: applemedia's struct changed"));
        }
    }
    Ok(())
}

/// Is vtdec's GstCoreVideoMeta still `{ GstMeta; CVBufferRef; CVPixelBufferRef }`? Checked once, the first time a
/// buffer carries the meta (applemedia registers it lazily). On Err the importer falls back to the one-copy upload.
pub fn video_meta_ok() -> &'static Result<(), String> {
    static OK: std::sync::OnceLock<Result<(), String>> = std::sync::OnceLock::new();
    OK.get_or_init(|| {
        let r = check_meta(c"GstCoreVideoMeta", std::mem::size_of::<GstCoreVideoMeta>());
        if let Err(e) = &r {
            eprintln!("mm-media: zero-copy import OFF, falling back to one CPU copy per frame: {e}");
        }
        r
    })
}

/// The CVPixelBuffer behind a vtdec buffer (retained), if it has one and the meta is what we expect: the meta's
/// layout (once), the pointer really being a CVPixelBuffer, and its size matching the buffer's.
pub fn pixbuf(buf: &gst::BufferRef) -> Option<CFRetained<CVPixelBuffer>> {
    let api = api_type()?;
    unsafe {
        let m = gst::ffi::gst_buffer_get_meta(buf.as_mut_ptr(), api) as *mut GstCoreVideoMeta;
        if m.is_null() {
            return None;
        }
        if video_meta_ok().is_err() || (*m).pixbuf.is_null() {
            return None;
        }
        if CFGetTypeID((*m).pixbuf as *const _) != CVPixelBufferGetTypeID() {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| eprintln!("mm-media: GstCoreVideoMeta.pixbuf is not a CVPixelBuffer: zero-copy import OFF"));
            return None;
        }
        Some(CFRetained::retain(NonNull::new_unchecked((*m).pixbuf)))
    }
}

/// Whether a decoded buffer would be imported without a copy (for logs and tests).
pub fn zero_copy_available(buf: &gst::BufferRef) -> bool {
    pixbuf(buf).is_some()
}

#[repr(C)]
struct GstCoreMediaMeta {
    meta: gst::ffi::GstMeta,
    sample_buf: *mut std::ffi::c_void,
    image_buf: *mut CVBuffer,
    pixel_buf: *mut CVPixelBuffer,
    block_buf: *mut std::ffi::c_void,
}

/// A buffer with no memory carrying a GstCoreMediaMeta for `pb`: vtenc (1.28) takes the pixel buffer from that meta
/// (gst_core_media_buffer_get_pixel_buffer) and hands it to VTCompressionSession as is. Not public API: the meta is
/// registered by applemedia as "GstCoreMediaMeta"; struct = { GstMeta; CMSampleBufferRef sample_buf; CVImageBufferRef
/// image_buf; CVPixelBufferRef pixel_buf; CMBlockBufferRef block_buf } (coremediabuffer.h). Its free function
/// CFRelease()s sample_buf unconditionally and CVBufferRelease()s image_buf, so both hold a retain of `pb` (no real
/// CMSampleBuffer is needed: vtenc only reads pixel_buf).
pub fn wrap_pixbuf(pb: &CFRetained<CVPixelBuffer>) -> Result<gst::Buffer> {
    unsafe {
        let mut info = gst::ffi::gst_meta_get_info(c"GstCoreMediaMeta".as_ptr());
        if info.is_null() {
            register_core_media_meta()?;
            info = gst::ffi::gst_meta_get_info(c"GstCoreMediaMeta".as_ptr());
        }
        if info.is_null() {
            bail!("GstCoreMediaMeta not registered (applemedia not loaded yet)");
        }
        if let Err(e) = check_meta(c"GstCoreMediaMeta", std::mem::size_of::<GstCoreMediaMeta>()) {
            bail!("zero-copy encode unavailable: {e}");
        }
        let mut buf = gst::Buffer::new();
        let m = gst::ffi::gst_buffer_add_meta(buf.get_mut().unwrap().as_mut_ptr(), info, std::ptr::null_mut()) as *mut GstCoreMediaMeta;
        if m.is_null() {
            bail!("gst_buffer_add_meta failed");
        }
        let a = CFRetained::into_raw(pb.clone()).as_ptr();
        let b = CFRetained::into_raw(pb.clone()).as_ptr();
        (*m).sample_buf = a as *mut std::ffi::c_void;
        (*m).image_buf = b as *mut CVBuffer;
        (*m).pixel_buf = b;
        Ok(buf)
    }
}

/// applemedia registers GstCoreMediaMeta lazily, the first time it wraps a CMSampleBuffer (vtenc's output). Encode
/// one tiny frame so it exists before we add it to our own buffers.
fn register_core_media_meta() -> Result<()> {
    use gstreamer::prelude::*;
    let p = gst::parse::launch("videotestsrc num-buffers=1 ! video/x-raw,format=NV12,width=64,height=64 ! vtenc_h264 ! fakesink")?;
    p.set_state(gst::State::Playing)?;
    let bus = p.bus().unwrap();
    let _ = bus.timed_pop_filtered(gst::ClockTime::from_seconds(5), &[gst::MessageType::Eos, gst::MessageType::Error]);
    p.set_state(gst::State::Null)?;
    Ok(())
}

pub fn cfkey(k: &objc2_core_foundation::CFString) -> &NSString {
    unsafe { &*(k as *const objc2_core_foundation::CFString as *const NSString) }
}

/// A pool of IOSurface-backed, Metal-compatible pixel buffers (for the encoder's input).
pub struct Pool(CFRetained<CVPixelBufferPool>);
unsafe impl Send for Pool {}

impl Pool {
    pub fn new(w: u32, h: u32, fourcc: u32) -> Result<Pool> {
        unsafe {
            let fmt = NSNumber::new_u32(fourcc);
            let ww = NSNumber::new_u32(w);
            let hh = NSNumber::new_u32(h);
            let yes = NSNumber::new_bool(true);
            let empty: Retained<NSDictionary<NSString, AnyObject>> = NSDictionary::new();
            let d: Retained<NSDictionary<NSString, AnyObject>> = NSDictionary::from_slices(
                &[cfkey(kCVPixelBufferPixelFormatTypeKey), cfkey(kCVPixelBufferWidthKey), cfkey(kCVPixelBufferHeightKey),
                  cfkey(kCVPixelBufferMetalCompatibilityKey), cfkey(kCVPixelBufferIOSurfacePropertiesKey)],
                &[fmt.as_ref() as &AnyObject, ww.as_ref(), hh.as_ref(), yes.as_ref(), empty.as_ref()],
            );
            let cf = &*(Retained::as_ptr(&d) as *const objc2_core_foundation::CFDictionary);
            let mut pool: *mut CVPixelBufferPool = std::ptr::null_mut();
            let r = CVPixelBufferPool::create(None, None, Some(cf), NonNull::from(&mut pool));
            if r != kCVReturnSuccess || pool.is_null() {
                bail!("CVPixelBufferPoolCreate: {r}");
            }
            Ok(Pool(CFRetained::from_raw(NonNull::new_unchecked(pool))))
        }
    }
    pub fn get(&self) -> Result<CFRetained<CVPixelBuffer>> {
        unsafe {
            let mut pb: *mut CVPixelBuffer = std::ptr::null_mut();
            let r = CVPixelBufferPool::create_pixel_buffer(None, &self.0, NonNull::from(&mut pb));
            if r != kCVReturnSuccess || pb.is_null() {
                bail!("CVPixelBufferPoolCreatePixelBuffer: {r}");
            }
            Ok(CFRetained::from_raw(NonNull::new_unchecked(pb)))
        }
    }
}

pub struct Importer {
    cache: CFRetained<CVMetalTextureCache>,
}
unsafe impl Send for Importer {}

struct Keep(#[allow(dead_code)] CFRetained<CVMetalTexture>, #[allow(dead_code)] CFRetained<CVPixelBuffer>);
unsafe impl Send for Keep {}
unsafe impl Sync for Keep {}

impl Importer {
    pub fn new(device: &wgpu::Device) -> Result<Importer> {
        unsafe {
            let hal = device.as_hal::<wgpu::hal::api::Metal>().ok_or_else(|| anyhow!("not a Metal device"))?;
            let mtl = hal.raw_device();
            let mut cache: *mut CVMetalTextureCache = std::ptr::null_mut();
            let r = CVMetalTextureCache::create(None, None, mtl, None, NonNull::from(&mut cache));
            if r != kCVReturnSuccess || cache.is_null() {
                bail!("CVMetalTextureCacheCreate: {r}");
            }
            Ok(Importer { cache: CFRetained::from_raw(NonNull::new_unchecked(cache)) })
        }
    }

    /// Plane `plane` of `pb` as a wgpu texture of `format` sharing its IOSurface. The texture keeps `pb` alive.
    pub fn import_plane(&self, device: &wgpu::Device, pb: &CFRetained<CVPixelBuffer>, plane: usize, format: wgpu::TextureFormat,
                        writable: bool) -> Result<wgpu::Texture> {
        let mtl_fmt = match format {
            wgpu::TextureFormat::R8Unorm => MTLPixelFormat::R8Unorm,
            wgpu::TextureFormat::Rg8Unorm => MTLPixelFormat::RG8Unorm,
            wgpu::TextureFormat::R16Unorm => MTLPixelFormat::R16Unorm,
            wgpu::TextureFormat::Rg16Unorm => MTLPixelFormat::RG16Unorm,
            wgpu::TextureFormat::Bgra8Unorm => MTLPixelFormat::BGRA8Unorm,
            f => bail!("format {f:?}"),
        };
        unsafe {
            let planar = CVPixelBufferGetPlaneCount(pb) > 0;
            let (w, h) = if planar {
                (CVPixelBufferGetWidthOfPlane(pb, plane), CVPixelBufferGetHeightOfPlane(pb, plane))
            } else {
                (CVPixelBufferGetWidth(pb), CVPixelBufferGetHeight(pb))
            };
            let attrs = if writable {
                let usage = NSNumber::new_u64((objc2_metal::MTLTextureUsage::ShaderRead | objc2_metal::MTLTextureUsage::ShaderWrite).0 as u64);
                let d: Retained<NSDictionary<NSString, AnyObject>> = NSDictionary::from_slices(&[cfkey(kCVMetalTextureUsage)], &[usage.as_ref() as &AnyObject]);
                Some(d)
            } else { None };
            let attrs_cf = attrs.as_ref().map(|d| &*(Retained::as_ptr(d) as *const objc2_core_foundation::CFDictionary));
            let mut tex: *mut CVMetalTexture = std::ptr::null_mut();
            let r = CVMetalTextureCache::create_texture_from_image(None, &self.cache, pb, attrs_cf, mtl_fmt, w, h, plane, NonNull::from(&mut tex));
            if r != kCVReturnSuccess || tex.is_null() {
                bail!("CVMetalTextureCacheCreateTextureFromImage plane {plane}: {r}",
                      );
            }
            let cvt: CFRetained<CVMetalTexture> = CFRetained::from_raw(NonNull::new_unchecked(tex));
            let mtex: Retained<ProtocolObject<dyn MTLTexture>> = CVMetalTextureGetTexture(&cvt).ok_or_else(|| anyhow!("no MTLTexture"))?;
            let keep = Keep(cvt, pb.clone());
            let hal_tex = wgpu::hal::metal::Device::texture_from_raw(
                mtex, format, MTLTextureType::Type2D, 1, 1,
                wgpu::hal::CopyExtent { width: w as u32, height: h as u32, depth: 1 },
                Some(Box::new(move || drop(keep))),
            );
            let usage = if writable { wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING } else { wgpu::TextureUsages::TEXTURE_BINDING };
            let desc = wgpu::TextureDescriptor {
                label: Some("cv plane"),
                size: wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            };
            Ok(device.create_texture_from_hal::<wgpu::hal::api::Metal>(hal_tex, &desc, wgpu::TextureUses::RESOURCE))
        }
    }

    pub fn flush(&self) {
        self.cache.flush(0);
    }
}
