//! Bi-planar YUV 4:2:0 (the decoder's own NV12/P010 planes) -> RGB, rgba8 at the stored size (yuv.wgsl), then the
//! prep pass to the upright display size when the stored picture isn't already it (anamorphic, rotated).
//!
//! Two conversions: the textbook one from (kr, kb, range) with an optional HLG-to-SDR, or, when YuvParams::matrix is
//! set, a 3x4 affine matrix on 8-bit-unit codes rounded floor(x + 0.5): AVFoundation's own fixed-point conversion as
//! mm_media::convert measured it (the CPU reference, convert::yuv_to_bgr, does the same arithmetic).
use crate::prep::{frame_texture, Prep};
use mm_core::colour::YuvParams;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct U {
    w: u32, h: u32, cw: u32, ch: u32, depth: u32, full: u32, hcos: u32, vcos: u32, kr: f32, kb: f32, transfer: u32,
    /// 1: convert with `m` (YuvParams::matrix)
    use_m: u32,
    m: [[f32; 4]; 3],
}

pub struct Yuv {
    pipe: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    prep: Prep,
    /// stored-size rgba8 (yuv output) when a prep pass follows
    stored: Option<wgpu::Texture>,
    /// upright display-size rgba8: what the compositor reads
    pub display: wgpu::Texture,
    rotation: u32,
}

impl Yuv {
    /// For frames stored at `stored` (w, h) shown upright at `display` (w, h) after `rotation` degrees clockwise.
    pub fn new(device: &wgpu::Device, stored: (u32, u32), display: (u32, u32), rotation: u32) -> Yuv {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("yuv"),
            source: wgpu::ShaderSource::Wgsl(include_str!("yuv.wgsl").into()),
        });
        let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("yuv"), layout: None, module: &module, entry_point: Some("yuv"),
            compilation_options: Default::default(), cache: None,
        });
        let needs_prep = stored != display || rotation != 0;
        Yuv {
            bgl: pipe.get_bind_group_layout(0),
            pipe,
            prep: Prep::new(device),
            stored: needs_prep.then(|| frame_texture(device, stored.0, stored.1)),
            display: frame_texture(device, display.0, display.1),
            rotation,
        }
    }

    /// Record the passes turning planes (y, cbcr) into `self.display`. Keep the planes alive until the GPU is done.
    pub fn encode(&self, device: &wgpu::Device, enc: &mut wgpu::CommandEncoder, y: &wgpu::Texture, c: &wgpu::Texture, p: &YuvParams) {
        let u = U {
            w: y.width(), h: y.height(), cw: c.width(), ch: c.height(), depth: p.depth, full: p.full as u32,
            hcos: p.hcos as u32, vcos: p.vcos as u32, kr: p.kr, kb: p.kb, transfer: p.hlg_to_sdr as u32,
            use_m: p.matrix.is_some() as u32, m: p.matrix.unwrap_or([[0.0; 4]; 3]),
        };
        let ub = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&u), usage: wgpu::BufferUsages::UNIFORM });
        let out = self.stored.as_ref().unwrap_or(&self.display);
        let (yv, cv, ov) = (y.create_view(&Default::default()), c.create_view(&Default::default()), out.create_view(&Default::default()));
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&yv) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&cv) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&ov) },
                wgpu::BindGroupEntry { binding: 3, resource: ub.as_entire_binding() },
            ],
        });
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&self.pipe);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(u.w.div_ceil(16), u.h.div_ceil(16), 1);
        }
        if let Some(s) = &self.stored {
            self.prep.encode(device, enc, s, &self.display, self.rotation);
        }
    }
}
