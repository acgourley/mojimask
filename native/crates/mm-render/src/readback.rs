//! Reading textures back to the CPU (tests, parity checks, the readback export path).
use anyhow::{anyhow, Result};

/// A 4-byte-per-pixel texture's bytes, tightly packed, in the texture's own channel order.
pub fn read_texture(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture) -> Vec<u8> {
    let (w, h) = (tex.width(), tex.height());
    let bpr = (w * 4).div_ceil(256) * 256;
    let b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (bpr * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        tex.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &b,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit([enc.finish()]);
    let sl = b.slice(..);
    sl.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let data = sl.get_mapped_range().expect("map");
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h as usize {
        out.extend_from_slice(&data[y * bpr as usize..y * bpr as usize + (w * 4) as usize]);
    }
    drop(data);
    b.unmap();
    out
}

/// Save an RGBA8 (or BGRA8, `bgra`) texture as an RGB PNG.
pub fn save_png(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture, bgra: bool, path: &str) -> Result<()> {
    let px = read_texture(device, queue, tex);
    let mut rgb = Vec::with_capacity(px.len() / 4 * 3);
    for c in px.chunks_exact(4) {
        if bgra {
            rgb.extend_from_slice(&[c[2], c[1], c[0]]);
        } else {
            rgb.extend_from_slice(&c[..3]);
        }
    }
    image::RgbImage::from_raw(tex.width(), tex.height(), rgb).ok_or_else(|| anyhow!("readback size"))?.save(path)?;
    Ok(())
}
