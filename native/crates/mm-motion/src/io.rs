//! Reading and writing frames for the tools (PNG/JPEG <-> BGR Mat, OpenCV's channel order).
use mm_vision::Mat;
use std::path::Path;

pub fn read_bgr(path: &Path) -> Mat<u8> {
    let im = image::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())).to_rgb8();
    let (w, h) = (im.width() as usize, im.height() as usize);
    let mut d = im.into_raw();
    for p in d.chunks_exact_mut(3) {
        p.swap(0, 2);
    }
    Mat::from_vec(h, w, 3, d)
}

pub fn read_gray(path: &Path) -> Mat<u8> {
    mm_vision::bgr2gray(&read_bgr(path))
}

pub fn write_bgr_png(path: &Path, m: &Mat<u8>) {
    let mut d = m.data.clone();
    for p in d.chunks_exact_mut(3) {
        p.swap(0, 2);
    }
    image::save_buffer(path, &d, m.cols as u32, m.rows as u32, image::ColorType::Rgb8).unwrap();
}

/// JPEG round trip at `quality` (the artefacts of a phone's or a scan's frames).
pub fn jpeg_roundtrip(m: &Mat<u8>, quality: u8) -> Mat<u8> {
    let mut d = m.data.clone();
    for p in d.chunks_exact_mut(3) {
        p.swap(0, 2);
    }
    let mut buf = vec![];
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, quality)
        .encode(&d, m.cols as u32, m.rows as u32, image::ExtendedColorType::Rgb8)
        .unwrap();
    let im = image::load_from_memory(&buf).unwrap().to_rgb8();
    let mut d = im.into_raw();
    for p in d.chunks_exact_mut(3) {
        p.swap(0, 2);
    }
    Mat::from_vec(m.rows, m.cols, 3, d)
}
