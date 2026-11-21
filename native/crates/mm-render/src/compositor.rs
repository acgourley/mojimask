//! The compositor: the WGSL port of `mojimask/gpu_draw.py`'s Metal kernel (compose.wgsl: `guide` + `compose`), its
//! picture atlas (an rgba16f texture array with mips, LRU by last use) and per-frame uploads. Output format is a
//! parameter: Qt Quick's item texture is RGBA8, the encoder's pixel buffers BGRA8. Same maths as the Metal source;
//! parity on jazz_4k vs today's Metal output: 79.6 dB min PSNR (poc/rust/REPORT.md in the unpublished PoC).
//! Beyond the Metal kernel: rows may be posed (head pose drawn in the shader from the base sprite: mm-core cover.rs
//! documents the row), and `encode_over` draws the covers over another picture than the frame (the blur, pixelate
//! and solid styles' result, for their depth cut).
use anyhow::Result;
use mm_core::cover::{FrameCovers, RenderMeta, Slice, STRIDE};
use std::collections::HashMap;
use std::sync::Arc;

/// Covers for one frame, ready for the GPU: rows with P[3] remapped to the atlas slot, depth map.
pub struct Prepared {
    pub rows: Vec<f32>,
    pub n: u32,
    pub seed: f32,
    pub soft: f32,
    pub near: Option<(Arc<Vec<f32>>, u32, u32)>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FrameU {
    w: u32,
    h: u32,
    n: u32,
    mw: u32,
    mh: u32,
    kx: f32,
    ky: f32,
    seed: f32,
    soft: f32,
    ss: f32,
    sr: f32,
    z0: f32,
    z1: f32,
    _p: [u32; 3],
}

pub struct Compositor {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    meta: RenderMeta,
    bgl: wgpu::BindGroupLayout,
    guide: wgpu::ComputePipeline,
    compose: wgpu::ComputePipeline,
    atlas: wgpu::Texture,
    atlas_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    ubuf: wgpu::Buffer,
    cbuf: wgpu::Buffer,
    nbuf: wgpu::Buffer,
    gbuf: wgpu::Buffer,
    // atlas slot cache: picture hash -> slot, LRU by last use (frame counter)
    slots: HashMap<String, (u32, u64)>,
    tick: u64,
    pub uploads: u64,
}

fn buf(device: &wgpu::Device, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor { label: None, size: size.max(16), usage, mapped_at_creation: false })
}

impl Compositor {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, meta: &RenderMeta, out_format: wgpu::TextureFormat) -> Compositor {
        let fmt = if out_format == wgpu::TextureFormat::Rgba8Unorm { "rgba8unorm" } else { "bgra8unorm" };
        let src = include_str!("compose.wgsl").replace("@FACE_DY@", &format!("{:?}", meta.face_dy))
            .replace("texture_storage_2d<bgra8unorm, write>", &format!("texture_storage_2d<{fmt}, write>"));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compose"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        use wgpu::BindingType as B;
        let st = |ro: bool| B::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: ro },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let e = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compose"),
            entries: &[
                e(0, B::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                }),
                e(1, st(true)),
                e(2, B::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }),
                e(3, st(true)),
                e(4, st(false)),
                e(5, B::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                }),
                e(6, B::Sampler(wgpu::SamplerBindingType::Filtering)),
                e(7, B::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: out_format,
                    view_dimension: wgpu::TextureViewDimension::D2,
                }),
                e(8, B::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                }),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let mk = |ep: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(ep),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(ep),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (guide, compose) = (mk("guide"), mk("compose"));
        let levels = 32 - meta.tex.leading_zeros();
        let atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("atlas"),
            size: wgpu::Extent3d { width: meta.tex, height: meta.tex, depth_or_array_layers: meta.slices },
            mip_level_count: levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let atlas_view = atlas.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("atlas"),
            address_mode_u: wgpu::AddressMode::ClampToBorder,
            address_mode_v: wgpu::AddressMode::ClampToBorder,
            address_mode_w: wgpu::AddressMode::ClampToBorder,
            border_color: Some(wgpu::SamplerBorderColor::TransparentBlack),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        use wgpu::BufferUsages as U;
        Compositor {
            device: device.clone(),
            queue: queue.clone(),
            meta: meta.clone(),
            guide,
            compose,
            atlas,
            atlas_view,
            sampler,
            ubuf: buf(device, std::mem::size_of::<FrameU>() as u64, U::UNIFORM | U::COPY_DST),
            cbuf: buf(device, 36 * 4 * 64, U::STORAGE | U::COPY_DST),
            nbuf: buf(device, 1024 * 1024 * 4, U::STORAGE | U::COPY_DST),
            gbuf: buf(device, 1024 * 1024 * 16, U::STORAGE),
            bgl,
            slots: HashMap::new(),
            tick: 0,
            uploads: 0,
        }
    }

    fn upload_slice(&mut self, slot: u32, s: &Slice) {
        let tex = self.meta.tex;
        for (lv, data) in s.levels.iter().enumerate() {
            let sz = (tex >> lv).max(1);
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.atlas,
                    mip_level: lv as u32,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: slot },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(data),
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(sz * 8), rows_per_image: Some(sz) },
                wgpu::Extent3d { width: sz, height: sz, depth_or_array_layers: 1 },
            );
        }
        self.uploads += 1;
    }

    /// One frame's covers -> Prepared: uploads any picture not already in the atlas (fetched through `pic`) and
    /// remaps each row's picture index to its atlas slot.
    pub fn prepare(&mut self, fc: &FrameCovers, pic: &dyn Fn(&str) -> Result<Arc<Slice>>) -> Result<Prepared> {
        self.tick += 1;
        let mut out = fc.rows.clone();
        let mut local_to_slot = Vec::new();
        let in_frame: std::collections::HashSet<&String> = fc.pics.iter().collect();
        for h in &fc.pics {
            if let Some(e) = self.slots.get_mut(h) {
                e.1 = self.tick;
                local_to_slot.push(e.0);
                continue;
            }
            let slot = if (self.slots.len() as u32) < self.meta.slices {
                self.slots.len() as u32
            } else {
                // evict the least recently used picture that this frame doesn't use
                let (k, (s, _)) = self
                    .slots
                    .iter()
                    .filter(|(k, _)| !in_frame.contains(k))
                    .min_by_key(|(_, v)| v.1)
                    .map(|(k, v)| (k.clone(), *v))
                    .ok_or_else(|| anyhow::anyhow!("more pictures in one frame than the atlas holds"))?;
                self.slots.remove(&k);
                s
            };
            let s = pic(h)?;
            self.upload_slice(slot, &s);
            self.slots.insert(h.clone(), (slot, self.tick));
            local_to_slot.push(slot);
        }
        for r in out.chunks_mut(STRIDE) {
            let li = r[3] as usize;
            r[3] = *local_to_slot.get(li).unwrap_or(&0) as f32;
        }
        Ok(Prepared { n: (out.len() / STRIDE) as u32, rows: out, seed: fc.seed, soft: fc.soft, near: fc.near.clone() })
    }

    fn ensure(&mut self, which: u8, need: u64) {
        use wgpu::BufferUsages as U;
        let cur = match which {
            0 => &self.cbuf,
            1 => &self.nbuf,
            _ => &self.gbuf,
        };
        if cur.size() >= need {
            return;
        }
        let nb = match which {
            2 => buf(&self.device, need, U::STORAGE),
            _ => buf(&self.device, need, U::STORAGE | U::COPY_DST),
        };
        match which {
            0 => self.cbuf = nb,
            1 => self.nbuf = nb,
            _ => self.gbuf = nb,
        }
    }

    /// Record the composite of `src` (the frame, W x H, any 8-bit unorm format) into `dst` (bgra8unorm storage).
    /// Uploads go through queue.write_*: submit before encoding the next frame.
    pub fn encode(&mut self, enc: &mut wgpu::CommandEncoder, src: &wgpu::TextureView, dst: &wgpu::TextureView,
                  w: u32, h: u32, p: &Prepared) {
        self.encode_over(enc, src, src, dst, w, h, p)
    }

    /// As `encode`, but the covers are drawn over `base` (e.g. the blur/pixelate/solid styles' picture) while `src`
    /// stays the original frame: what the depth cut shows again, and the depth guide's colours.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_over(&mut self, enc: &mut wgpu::CommandEncoder, src: &wgpu::TextureView, base: &wgpu::TextureView,
                       dst: &wgpu::TextureView, w: u32, h: u32, p: &Prepared) {
        let rows_bytes = (p.rows.len() * 4) as u64;
        self.ensure(0, rows_bytes.max(16));
        if !p.rows.is_empty() {
            self.queue.write_buffer(&self.cbuf, 0, bytemuck::cast_slice(&p.rows));
        }
        let (mw, mh) = match &p.near {
            Some((v, mw, mh)) => {
                self.ensure(1, (v.len() * 4) as u64);
                self.ensure(2, (*mw as u64) * (*mh as u64) * 16);
                self.queue.write_buffer(&self.nbuf, 0, bytemuck::cast_slice(v));
                (*mw, *mh)
            }
            None => (1, 1),
        };
        let fu = FrameU {
            w,
            h,
            n: p.n,
            mw,
            mh,
            kx: mw as f32 / w as f32,
            ky: mh as f32 / h as f32,
            seed: p.seed,
            soft: p.soft,
            ss: self.meta.sigma_s,
            sr: self.meta.sigma_r,
            z0: self.meta.face_zone[0],
            z1: self.meta.face_zone[1],
            _p: [0; 3],
        };
        self.queue.write_buffer(&self.ubuf, 0, bytemuck::bytes_of(&fu));
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                wgpu::BindGroupEntry { binding: 1, resource: self.cbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.nbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.gbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.atlas_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(dst) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(base) },
            ],
        });
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("compose"), timestamp_writes: None });
        cp.set_bind_group(0, &bg, &[]);
        if p.near.is_some() {
            cp.set_pipeline(&self.guide);
            cp.dispatch_workgroups(mw.div_ceil(16), mh.div_ceil(16), 1);
        }
        cp.set_pipeline(&self.compose);
        cp.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
    }
}

/// A texture the compositor can write and the UI can show / we can read back.
pub fn out_texture(device: &wgpu::Device, w: u32, h: u32, format: wgpu::TextureFormat) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("out"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// An rgba8 texture holding an RGB(A) picture (for frames loaded from PNG).
pub fn upload_rgba(device: &wgpu::Device, queue: &wgpu::Queue, img: &image::RgbaImage) -> wgpu::Texture {
    let (w, h) = img.dimensions();
    let t = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("in"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    queue.write_texture(
        t.as_image_copy(),
        img.as_raw(),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    t
}

/// BGRA bytes -> RGB image.
pub fn bgra_to_rgb(w: u32, h: u32, bgra: &[u8]) -> image::RgbImage {
    let mut v = Vec::with_capacity((w * h * 3) as usize);
    for p in bgra.chunks_exact(4) {
        v.extend_from_slice(&[p[2], p[1], p[0]]);
    }
    image::RgbImage::from_raw(w, h, v).unwrap()
}
