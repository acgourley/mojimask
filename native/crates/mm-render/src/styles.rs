//! The non-emoji covers (track.cover) on the GPU: blur, pixelate, or solid (opaque, most robust: blur can sometimes be
//! partly reversed), each over a circle of diameter c[2] * scale. Circles are applied one after another (ping-pong), in
//! drawing order, each over the result of the last, like the CPU loop:
//!   - blur: cv2.GaussianBlur twice with a k x k kernel, k = max(int(r) | 1, 3) (cv2.getGaussianKernel(k, 0)), on the
//!     circle's bounding box (x1 = max(int(cx - r), 0) ... x2 = min(int(cx + r) + 1, W)), reflect-101 borders inside
//!     that box, each pass rounded to 8 bits;
//!   - pixelate: 5 blocks across the face (INTER_AREA down, INTER_NEAREST back up);
//!   - solid: (40, 40, 40).
//! Pixels inside the circle ((x - cx)^2 + (y - cy)^2 <= r^2, pixel indices) take the new value.
//! Parity (poc/rust/REPORT.md in the unpublished PoC, jazz frames 0-2): blur 72 dB (max error 1), pixelate and solid
//! identical.
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct U {
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    cx: f32,
    cy: f32,
    r: f32,
    mode: u32,
    k: i32,
    bw: i32,
    bh: i32,
    _p: i32,
}

pub struct Styles {
    bgl: wgpu::BindGroupLayout,
    pipes: std::collections::HashMap<&'static str, wgpu::ComputePipeline>,
}

/// cv2.getGaussianKernel(k, 0)
fn gauss(k: i32) -> Vec<f32> {
    match k {
        3 => return vec![0.25, 0.5, 0.25],
        5 => return vec![0.0625, 0.25, 0.375, 0.25, 0.0625],
        7 => return vec![0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125],
        _ => {}
    }
    let sigma = 0.3 * ((k as f64 - 1.0) * 0.5 - 1.0) + 0.8;
    let c = (k - 1) as f64 / 2.0;
    let v: Vec<f64> = (0..k).map(|i| (-((i as f64 - c).powi(2)) / (2.0 * sigma * sigma)).exp()).collect();
    let s: f64 = v.iter().sum();
    v.iter().map(|x| (x / s) as f32).collect()
}

impl Styles {
    /// `format`: the frames' texture format (rgba8unorm or bgra8unorm); the texture given to `apply` must be in it.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Styles {
        let fmt = if format == wgpu::TextureFormat::Bgra8Unorm { "bgra8unorm" } else { "rgba8unorm" };
        let src = include_str!("styles.wgsl").replace("texture_storage_2d<rgba8unorm, write>",
                                                       &format!("texture_storage_2d<{fmt}, write>"));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("styles"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        use wgpu::BindingType as B;
        let e = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let st = |ro: bool| B::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: ro },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                e(0, B::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                }),
                e(1, B::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format,
                    view_dimension: wgpu::TextureViewDimension::D2,
                }),
                e(2, B::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }),
                e(3, st(false)),
                e(4, st(false)),
                e(5, st(true)),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let mut pipes = std::collections::HashMap::new();
        for ep in ["blur_h_src", "blur_h_s2", "blur_v", "pix_down", "combine"] {
            pipes.insert(
                ep,
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(ep),
                    layout: Some(&pl),
                    module: &module,
                    entry_point: Some(ep),
                    compilation_options: Default::default(),
                    cache: None,
                }),
            );
        }
        Styles { bgl, pipes }
    }

    /// Apply `style` ("blur" | "pixelate" | "solid") for each circle [cx, cy, d] to `img` (the format given to new, needs
    /// TEXTURE_BINDING | STORAGE_BINDING | COPY_SRC | COPY_DST); returns the texture holding the result.
    pub fn apply(&self, device: &wgpu::Device, queue: &wgpu::Queue, img: wgpu::Texture, circles: &[[f32; 3]],
                 style: &str) -> wgpu::Texture {
        let (w, h) = (img.width() as i32, img.height() as i32);
        let other = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pingpong"),
            size: img.size(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: img.format(),
            usage: img.usage(),
            view_formats: &[],
        });
        let (mut a, mut b) = (img, other);
        let mode = match style {
            "blur" => 0,
            "pixelate" => 1,
            _ => 2,
        };
        for c in circles {
            let r = c[2] / 2.0;
            let x1 = ((c[0] - r) as i32).max(0);
            let y1 = ((c[1] - r) as i32).max(0);
            let x2 = ((c[0] + r) as i32 + 1).min(w);
            let y2 = ((c[1] + r) as i32 + 1).min(h);
            if x2 - x1 < 2 || y2 - y1 < 2 {
                continue;
            }
            let (rw, rh) = (x2 - x1, y2 - y1);
            let k = ((r as i32) | 1).max(3);
            let bw = 5;
            let bh = ((5.0 * rh as f32 / rw as f32) as i32).max(1);
            let u = U { x1, y1, x2, y2, cx: c[0], cy: c[1], r, mode, k, bw, bh, _p: 0 };
            let ub = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&u),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let sz = (rw * rh * 16).max(16) as u64;
            let s1 = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: sz, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
            let s2 = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: sz, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
            let kwb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&gauss(k)),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let av = a.create_view(&Default::default());
            let bv = b.create_view(&Default::default());
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&av) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&bv) },
                    wgpu::BindGroupEntry { binding: 2, resource: ub.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: s1.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: s2.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: kwb.as_entire_binding() },
                ],
            });
            let mut enc = device.create_command_encoder(&Default::default());
            enc.copy_texture_to_texture(a.as_image_copy(), b.as_image_copy(), a.size());
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_bind_group(0, &bg, &[]);
                let (gx, gy) = ((rw as u32).div_ceil(16), (rh as u32).div_ceil(16));
                let run = |cp: &mut wgpu::ComputePass, n: &str, x: u32, y: u32| {
                    cp.set_pipeline(&self.pipes[n]);
                    cp.dispatch_workgroups(x, y, 1);
                };
                match mode {
                    0 => {
                        run(&mut cp, "blur_h_src", gx, gy);
                        run(&mut cp, "blur_v", gx, gy);
                        run(&mut cp, "blur_h_s2", gx, gy);
                        run(&mut cp, "blur_v", gx, gy);
                    }
                    1 => run(&mut cp, "pix_down", 1, (bh as u32).div_ceil(8)),
                    _ => {}
                }
                run(&mut cp, "combine", gx, gy);
            }
            queue.submit([enc.finish()]);
            std::mem::swap(&mut a, &mut b);
        }
        a
    }
}
