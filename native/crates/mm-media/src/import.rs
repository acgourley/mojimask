//! A decoded frame's planes as wgpu textures, and how to turn them into RGB.
//!
//! On the Mac, vtdec's buffers carry the decoder's CVPixelBuffer; CVMetalTextureCache aliases each IOSurface plane as
//! an MTLTexture that wgpu wraps: zero CPU copies (mac::cv, private applemedia meta). Elsewhere, or when the meta is
//! missing, the planes are mapped and uploaded (one CPU copy; 4K at 87-94 fps in the PoC).
use crate::decode::Frame;
use anyhow::{anyhow, Result};
use gstreamer_video as gst_video;
use gst_video::prelude::*;
use mm_core::colour::YuvParams;

/// The stream's matrix, range and chroma siting, from the caps.
pub fn yuv_params(info: &gst_video::VideoInfo) -> YuvParams {
    let c = info.colorimetry();
    let (kr, kb) = match c.matrix() {
        gst_video::VideoColorMatrix::Bt601 => (0.299, 0.114),
        gst_video::VideoColorMatrix::Bt2020 => (0.2627, 0.0593),
        gst_video::VideoColorMatrix::Smpte240m => (0.212, 0.087),
        gst_video::VideoColorMatrix::Fcc => (0.30, 0.11),
        _ => (0.2126, 0.0722),
    };
    let site = info.chroma_site();
    YuvParams {
        kr,
        kb,
        full: c.range() == gst_video::VideoColorRange::Range0_255,
        hcos: site.contains(gst_video::VideoChromaSite::H_COSITED),
        vcos: site.contains(gst_video::VideoChromaSite::V_COSITED),
        depth: if info.format() == gst_video::VideoFormat::P01010le { 10 } else { 8 },
        // AVFoundation's 32BGRA output shows HLG through the matrix only; matching it is the default (PoC: 51.8 dB)
        hlg_to_sdr: false,
        matrix: None,
    }
}

/// The parameters that match video_io's display frames (AVFoundation; convert.rs has the measurements): the matrix
/// AVFoundation picks (the stream's tags, else by coded width; caps guess by height, which gets jazz's 708x476
/// wrong), as the fitted 3x4 affine matrix of `convert::display_matrix` (its sub-level gains and offsets), chroma
/// sampled as co-sited both ways, HLG through the matrix: the same arithmetic as the CPU reference.
pub fn yuv_params_display(f: &Frame) -> YuvParams {
    let mut p = yuv_params(&f.info);
    (p.kr, p.kb) = f.matrix;
    p.hcos = true;
    p.vcos = true;
    p.full = f.full;
    p.matrix = Some(crate::convert::display_matrix(p.kr, p.kb, p.depth, p.full));
    p
}

pub struct Planes {
    pub y: wgpu::Texture,
    pub cbcr: wgpu::Texture,
    pub params: YuvParams,
    pub zero_copy: bool,
}

pub struct Importer {
    #[cfg(target_os = "macos")]
    cv: crate::mac::cv::Importer,
    pub force_upload: bool,
}

impl Importer {
    pub fn new(device: &wgpu::Device) -> Result<Importer> {
        Ok(Importer {
            #[cfg(target_os = "macos")]
            cv: crate::mac::cv::Importer::new(device)?,
            force_upload: std::env::var("MM_FORCE_UPLOAD").is_ok(),
        })
    }

    pub fn planes(&self, device: &wgpu::Device, queue: &wgpu::Queue, f: &Frame) -> Result<Planes> {
        let params = yuv_params_display(f);
        let ten = params.depth > 8;
        let (yf, cf) = if ten {
            (wgpu::TextureFormat::R16Unorm, wgpu::TextureFormat::Rg16Unorm)
        } else {
            (wgpu::TextureFormat::R8Unorm, wgpu::TextureFormat::Rg8Unorm)
        };
        #[cfg(target_os = "macos")]
        if !self.force_upload {
            if let Some(pb) = crate::mac::cv::pixbuf(&f.buffer) {
                let y = self.cv.import_plane(device, &pb, 0, yf, false)?;
                let cbcr = self.cv.import_plane(device, &pb, 1, cf, false)?;
                return Ok(Planes { y, cbcr, params, zero_copy: true });
            }
        }
        let vf = gst_video::VideoFrameRef::from_buffer_ref_readable(f.buffer.as_ref(), &f.info).map_err(|_| anyhow!("map"))?;
        let mut out = vec![];
        for (p, fmt) in [(0u32, yf), (1, cf)] {
            let (w, h) = (vf.comp_width(p), vf.comp_height(p));
            let t = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("plane"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: fmt,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let stride = vf.plane_stride()[p as usize] as u32;
            let data = vf.plane_data(p).map_err(|_| anyhow!("plane"))?;
            queue.write_texture(
                t.as_image_copy(),
                data,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(stride), rows_per_image: Some(h) },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            out.push(t);
        }
        let cbcr = out.pop().unwrap();
        Ok(Planes { y: out.pop().unwrap(), cbcr, params, zero_copy: false })
    }

    /// Let go of textures the cache keeps for frames already shown.
    pub fn flush(&self) {
        #[cfg(target_os = "macos")]
        self.cv.flush();
    }
}
