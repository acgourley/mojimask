//! Drawing one frame from its records, in any style: track.draw_frame on the GPU.
//!
//! `Drawer::draw(src, record, near)` -> a new texture with the frame's covers drawn:
//!   - emoji / portrait: the records become compositor rows (`FrameRecord::rows`: eyes on the person's eyes, head pose
//!     drawn by the shader, lighting applied) over one atlas slice per picture (emoji through CoreText, the portrait
//!     photo), composited with motion blur, grain and the depth cut (compose.wgsl);
//!   - blur / pixelate / solid: `styles` over each cover's circle, far to near; with a depth guide, the part of a cover
//!     in front of the face shows the video again (the compositor's cut, run over the styled picture).
//!
//! Pictures are made on first use and kept (by key) for the Drawer's life.
use crate::compositor::{out_texture, Compositor};
use crate::styles::Styles;
use anyhow::{anyhow, Result};
use mm_core::cover::{Eyes, FrameCovers, FrameRecord, RenderMeta, Slice};
use std::collections::HashMap;
use std::sync::Arc;

/// Makes the picture for a key (straight RGBA): `sprites::for_key` on the Mac, or pictures loaded from files.
pub type PictureSource = Box<dyn Fn(&str) -> Result<image::RgbaImage> + Send>;

pub struct Drawer {
    pub comp: Compositor,
    styles: Styles,
    eyes: Eyes,
    meta: RenderMeta,
    format: wgpu::TextureFormat,
    pictures: PictureSource,
    slices: HashMap<String, (Arc<Slice>, u32)>,
}

impl Drawer {
    /// `format`: the frames' format (rgba8unorm or bgra8unorm); sources given to `draw` must be in it and have
    /// TEXTURE_BINDING | COPY_SRC.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, meta: &RenderMeta, format: wgpu::TextureFormat, eyes: Eyes,
               pictures: PictureSource) -> Drawer {
        Drawer {
            comp: Compositor::new(device, queue, meta, format),
            styles: Styles::new(device, format),
            eyes,
            meta: meta.clone(),
            format,
            pictures,
            slices: HashMap::new(),
        }
    }

    /// The picture for `key` as an atlas slice, and its size in px (made once).
    pub fn picture(&mut self, key: &str) -> Result<(Arc<Slice>, u32)> {
        if let Some(s) = self.slices.get(key) {
            return Ok(s.clone());
        }
        let img = (self.pictures)(key)?;
        let v = (Arc::new(crate::sprites::slice(&img, self.meta.tex)), img.width());
        self.slices.insert(key.to_string(), v.clone());
        Ok(v)
    }

    /// The compositor rows for an emoji/portrait record (and its picture keys).
    pub fn rows(&mut self, rec: &FrameRecord) -> Result<(Vec<f32>, Vec<String>)> {
        for k in rec.pics() {
            self.picture(&k)?;
        }
        let sizes: HashMap<String, u32> = self.slices.iter().map(|(k, v)| (k.clone(), v.1)).collect();
        Ok(rec.rows(&self.eyes, &|k| sizes.get(k).copied().unwrap_or(crate::sprites::SPRITE), self.meta.tex))
    }

    /// Draw `rec` over `src` (W x H, the Drawer's format) into a new texture. `near`: the depth guide (values, mw, mh)
    /// when any cover is cut by depth.
    pub fn draw(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Texture, rec: &FrameRecord,
                near: Option<(Arc<Vec<f32>>, u32, u32)>) -> Result<wgpu::Texture> {
        let (w, h) = (src.width(), src.height());
        let out = out_texture(device, w, h, self.format);
        match rec.style.as_str() {
            "emoji" | "portrait" => {
                let (rows, pics) = self.rows(rec)?;
                let fc = FrameCovers { rows, pics, seed: rec.seed as f32, soft: rec.soft as f32, near };
                let slices = &self.slices;
                let p = self.comp.prepare(&fc, &|k| {
                    slices.get(k).map(|s| s.0.clone()).ok_or_else(|| anyhow!("no picture for {k}"))
                })?;
                queue.submit([]);
                let mut enc = device.create_command_encoder(&Default::default());
                let sv = src.create_view(&Default::default());
                let ov = out.create_view(&Default::default());
                self.comp.encode(&mut enc, &sv, &ov, w, h, &p);
                queue.submit([enc.finish()]);
            }
            "blur" | "pixelate" | "solid" => {
                let mut enc = device.create_command_encoder(&Default::default());
                enc.copy_texture_to_texture(src.as_image_copy(), out.as_image_copy(), src.size());
                queue.submit([enc.finish()]);
                let styled = self.styles.apply(device, queue, out, &rec.circles(), &rec.style);
                if near.is_none() || rec.covers.iter().all(|c| c.occ.is_none()) {
                    return Ok(styled);
                }
                // the depth cut over the styled picture: rows with no picture, only the cut
                let mut rows = Vec::new();
                for c in &rec.covers {
                    let mut p = [0.0f32; mm_core::cover::STRIDE];
                    p[0] = -1e6;
                    p[1] = -1e6;
                    p[2] = 1.0;
                    p[25] = c.c[0] as f32;
                    p[26] = c.c[1] as f32;
                    p[27] = (c.c[2] * rec.scale / 2.0) as f32;
                    p[28] = p[27];
                    if let Some(o) = &c.occ {
                        p[22] = 1.0;
                        p[23] = o.face as f32;
                        p[24] = o.margin as f32;
                        p[32] = o.rim as f32;
                        p[33] = o.head as f32;
                    }
                    rows.extend(p);
                }
                let fc = FrameCovers { rows, pics: vec![], seed: rec.seed as f32, soft: rec.soft as f32, near };
                let p = self.comp.prepare(&fc, &|k| Err(anyhow!("no picture {k}")))?;
                let out2 = out_texture(device, w, h, self.format);
                let mut enc = device.create_command_encoder(&Default::default());
                let sv = src.create_view(&Default::default());
                let bv = styled.create_view(&Default::default());
                let ov = out2.create_view(&Default::default());
                self.comp.encode_over(&mut enc, &sv, &bv, &ov, w, h, &p);
                queue.submit([enc.finish()]);
                return Ok(out2);
            }
            s => return Err(anyhow!("unknown style {s}")),
        }
        Ok(out)
    }
}
