//! RGB -> NV12 bytes on the GPU (rgb2nv12.wgsl, BT.709 video range), for encoders that can't take our pixel buffers.
use wgpu::util::DeviceExt;

pub struct Nv12 {
    pipe: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    out: wgpu::Buffer,
    pub read: wgpu::Buffer,
    pub size: u64,
}

impl Nv12 {
    pub fn new(device: &wgpu::Device, w: u32, h: u32) -> Nv12 {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rgb2nv12"), source: wgpu::ShaderSource::Wgsl(include_str!("rgb2nv12.wgsl").into()),
        });
        let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rgb2nv12"), layout: None, module: &module, entry_point: Some("main"),
            compilation_options: Default::default(), cache: None,
        });
        let size = (w * h + w * h / 2) as u64;
        let out = device.create_buffer(&wgpu::BufferDescriptor { label: Some("nv12"), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let read = device.create_buffer(&wgpu::BufferDescriptor { label: Some("nv12 read"), size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        Nv12 { bgl: pipe.get_bind_group_layout(0), pipe, out, read, size }
    }

    /// Record the conversion of `src` (w x h) into `self.read` (map it after submitting).
    pub fn encode(&self, device: &wgpu::Device, enc: &mut wgpu::CommandEncoder, src: &wgpu::TextureView, w: u32, h: u32) {
        let ub = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&[w, h, 0u32, 0u32]), usage: wgpu::BufferUsages::UNIFORM });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                wgpu::BindGroupEntry { binding: 1, resource: self.out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: ub.as_entire_binding() },
            ],
        });
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&self.pipe);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups((w / 4).div_ceil(16), (h / 2).div_ceil(16), 1);
        }
        enc.copy_buffer_to_buffer(&self.out, 0, &self.read, 0, self.size);
    }
}
