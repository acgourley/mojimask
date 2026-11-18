//! The wgpu device: our own (headless), or adopted from the host so we draw on its device and queue.
use anyhow::Result;

/// Features the compositor needs (bgra8 storage for the encoder's pixel buffers; clamp-to-border for the atlas).
pub fn features() -> wgpu::Features {
    wgpu::Features::BGRA8UNORM_STORAGE | wgpu::Features::ADDRESS_MODE_CLAMP_TO_BORDER
}

pub fn limits() -> wgpu::Limits {
    wgpu::Limits { max_texture_dimension_2d: 8192, ..wgpu::Limits::default() }
}

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Gpu {
    /// A device of our own on the best adapter (the export and the CLI use this).
    pub fn headless() -> Result<Gpu> {
        let inst = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(inst.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mm headless"),
            required_features: features() | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
            required_limits: limits(),
            ..Default::default()
        }))?;
        Ok(Gpu { device, queue })
    }
}

#[cfg(target_os = "macos")]
pub mod metal {
    //! Adopting Qt's Metal objects. Rust builds its wgpu device *from Qt's own MTLDevice and command queue*
    //! (wgpu-hal `device_from_raw` + `queue_from_raw` -> `create_device_from_hal`), so every submit lands on Qt's
    //! queue in commit order, before Qt's frame that samples our texture: no CPU waits (poc/rust/qtshell/REPORT.md in
    //! the unpublished PoC).
    use super::*;
    use anyhow::{anyhow, bail};
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_metal::{MTLCommandQueue, MTLDevice, MTLPixelFormat, MTLTexture, MTLTextureType, MTLTextureUsage};
    use std::ffi::c_void;

    pub struct Adopted {
        pub gpu: Gpu,
        /// what was adopted, for /perf and bug reports
        pub report: String,
    }

    /// # Safety
    /// `dev` and `queue` must be a live `id<MTLDevice>` and `id<MTLCommandQueue>` of the same device.
    pub unsafe fn adopt(dev: *mut c_void, queue: *mut c_void) -> Result<Adopted> {
        let inst = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(inst.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))?;
        let qt_dev: Retained<ProtocolObject<dyn MTLDevice>> =
            Retained::retain(dev as *mut ProtocolObject<dyn MTLDevice>).ok_or_else(|| anyhow!("no MTLDevice from the host"))?;
        let qt_q: Retained<ProtocolObject<dyn MTLCommandQueue>> = Retained::retain(queue as *mut ProtocolObject<dyn MTLCommandQueue>)
            .ok_or_else(|| anyhow!("no MTLCommandQueue from the host"))?;
        let (adapter_dev, name) = {
            let ha = adapter.as_hal::<wgpu::hal::api::Metal>().ok_or_else(|| anyhow!("not Metal"))?;
            let d = ha.raw_device();
            (d as *const _ as *const c_void as usize, d.name().to_string())
        };
        if adapter_dev != dev as usize {
            bail!("wgpu's adapter MTLDevice {adapter_dev:#x} is not the host's {:#x}", dev as usize);
        }
        let features = features() | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
        let limits = limits();
        let hal_dev = wgpu::hal::metal::Device::device_from_raw(qt_dev, features, &limits);
        let hal_q = wgpu::hal::metal::Queue::queue_from_raw(qt_q, 1.0);
        let (device, queue_w) = adapter.create_device_from_hal::<wgpu::hal::api::Metal>(
            wgpu::hal::OpenDevice { device: hal_dev, queue: hal_q },
            &wgpu::DeviceDescriptor { label: Some("mm on host"), required_features: features, required_limits: limits, ..Default::default() },
        )?;
        let report = format!("{name}; host MTLDevice {:#x}, MTLCommandQueue {:#x} adopted", dev as usize, queue as usize);
        Ok(Adopted { gpu: Gpu { device, queue: queue_w }, report })
    }

    /// The host's texture (an `id<MTLTexture>`, RGBA8) as a wgpu texture; `writable` if the compositor can store to it
    /// directly (Qt's lacks ShaderWrite, so we composite elsewhere and blit).
    ///
    /// # Safety
    /// `ptr` must be a live `id<MTLTexture>` made on the adopted device.
    pub unsafe fn wrap_texture(device: &wgpu::Device, ptr: *mut c_void) -> Result<(wgpu::Texture, bool)> {
        let mtex: Retained<ProtocolObject<dyn MTLTexture>> =
            Retained::retain(ptr as *mut ProtocolObject<dyn MTLTexture>).ok_or_else(|| anyhow!("null texture"))?;
        let writable = mtex.usage().contains(MTLTextureUsage::ShaderWrite);
        if mtex.pixelFormat() != MTLPixelFormat::RGBA8Unorm {
            bail!("the host's texture is not RGBA8Unorm ({:?})", mtex.pixelFormat().0);
        }
        let (w, h) = (mtex.width() as u32, mtex.height() as u32);
        let hal = wgpu::hal::metal::Device::texture_from_raw(
            mtex,
            wgpu::TextureFormat::Rgba8Unorm,
            MTLTextureType::Type2D,
            1,
            1,
            wgpu::hal::CopyExtent { width: w, height: h, depth: 1 },
            None,
        );
        let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST;
        if writable {
            usage |= wgpu::TextureUsages::STORAGE_BINDING;
        }
        let desc = wgpu::TextureDescriptor {
            label: Some("host texture"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage,
            view_formats: &[],
        };
        let tex = device.create_texture_from_hal::<wgpu::hal::api::Metal>(hal, &desc, wgpu::TextureUses::UNINITIALIZED);
        Ok((tex, writable))
    }
}
