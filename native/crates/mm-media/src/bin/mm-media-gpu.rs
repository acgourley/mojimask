//! The GPU display conversion for the parity tests (media_decode.py --gpu): decode forward, import each wanted
//! frame's planes (zero-copy on the Mac), run mm-render's yuv (+ prep) pass with `import::yuv_params_display`, read
//! back, save a PNG, and compare against the CPU reference (convert::frame_bgr) of the same frame.
//!
//!   mm-media-gpu CLIP OUTDIR T,T,...
use anyhow::Result;
use mm_media::convert::{frame_bgr, Colour};
use serde_json::json;

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let m: f64 = a.iter().zip(b).map(|(&x, &y)| ((x as f64) - (y as f64)).powi(2)).sum::<f64>() / a.len() as f64;
    if m == 0.0 { 99.0 } else { 10.0 * (255.0f64 * 255.0 / m).log10() }
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (clip, dir) = (&a[1], &a[2]);
    let ts: Vec<usize> = a[3].split(',').map(|s| s.parse().unwrap()).collect();
    std::fs::create_dir_all(dir)?;
    let gpu = mm_render::device::Gpu::headless()?;
    let (device, queue) = (&gpu.device, &gpu.queue);
    let imp = mm_media::import::Importer::new(device)?;
    let mut d = mm_media::Decoder::open(clip)?;
    let tags = d.colour_tags().map(|s| s.to_string());
    let p = d.probe.clone();
    let yuv = mm_render::yuv::Yuv::new(device, (p.stored_w, p.stored_h), (p.width, p.height), p.rotation);
    let mut rows = vec![];
    while let Some((t, f)) = d.next()? {
        if t > *ts.iter().max().unwrap() {
            break;
        }
        if !ts.contains(&t) {
            continue;
        }
        let planes = imp.planes(device, queue, &f)?;
        let mut enc = device.create_command_encoder(&Default::default());
        yuv.encode(device, &mut enc, &planes.y, &planes.cbcr, &planes.params);
        queue.submit([enc.finish()]);
        let rgba = mm_render::readback::read_texture(device, queue, &yuv.display);
        drop(planes);
        imp.flush();
        let (w, h) = (yuv.display.width(), yuv.display.height());
        let bgr: Vec<u8> = rgba.chunks_exact(4).flat_map(|c| [c[2], c[1], c[0]]).collect();
        let rgb: Vec<u8> = rgba.chunks_exact(4).flat_map(|c| [c[0], c[1], c[2]]).collect();
        image::save_buffer(format!("{dir}/{t:05}.png"), &rgb, w, h, image::ColorType::Rgb8)?;
        let cpu = frame_bgr(&f, &d.probe, tags.as_deref(), Colour::Display)?;
        let same = (cpu.w, cpu.h) == (w, h);
        let (db, identical) = if same {
            let id = bgr.chunks_exact(3).zip(cpu.data.chunks_exact(3)).filter(|(x, y)| x == y).count() as f64 / (w * h) as f64;
            (psnr(&bgr, &cpu.data), id * 100.0)
        } else {
            (0.0, 0.0)
        };
        rows.push(json!({"t": t, "w": w, "h": h, "vs_cpu_db": db, "identical_pct": identical, "zero_copy": planes_zero_copy(&f)}));
    }
    println!("{}", json!({"frames": rows}));
    Ok(())
}

fn planes_zero_copy(f: &mm_media::Frame) -> bool {
    #[cfg(target_os = "macos")]
    {
        mm_media::mac::cv::zero_copy_available(f.buffer.as_ref())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = f;
        false
    }
}
