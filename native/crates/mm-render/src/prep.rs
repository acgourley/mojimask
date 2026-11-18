//! GPU resize (INTER_AREA weights) + rotation of decoded frames to the timeline's upright display picture (prep.wgsl).
use wgpu::util::DeviceExt;

pub struct Prep {
    bgl: wgpu::BindGroupLayout,
    pipe: wgpu::ComputePipeline,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct U { sw: u32, sh: u32, dw: u32, dh: u32, ow: u32, oh: u32, rot: u32, _p: u32 }

/// An rgba8unorm texture the prep pass writes and the compositor reads.
pub fn frame_texture(device: &wgpu::Device, w: u32, h: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("frame"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

impl Prep {
    pub fn new(device: &wgpu::Device) -> Prep {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prep"),
            source: wgpu::ShaderSource::Wgsl(include_str!("prep.wgsl").into()),
        });
        let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prep"),
            layout: None,
            module: &module,
            entry_point: Some("prep"),
            compilation_options: Default::default(),
            cache: None,
        });
        Prep { bgl: pipe.get_bind_group_layout(0), pipe }
    }

    /// src: stored-size decoded frame; dst: upright display size. (dw, dh): display size in stored orientation.
    pub fn encode(&self, device: &wgpu::Device, enc: &mut wgpu::CommandEncoder, src: &wgpu::Texture, dst: &wgpu::Texture, rot: u32) {
        let (ow, oh) = (dst.width(), dst.height());
        let (dw, dh) = if rot == 90 || rot == 270 { (oh, ow) } else { (ow, oh) };
        let u = U { sw: src.width(), sh: src.height(), dw, dh, ow, oh, rot, _p: 0 };
        let ub = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&u),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let sv = src.create_view(&Default::default());
        let dv = dst.create_view(&Default::default());
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&sv) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&dv) },
                wgpu::BindGroupEntry { binding: 2, resource: ub.as_entire_binding() },
            ],
        });
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&self.pipe);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(ow.div_ceil(16), oh.div_ceil(16), 1);
    }
}
