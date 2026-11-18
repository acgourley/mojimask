//! `mm-cli decisions questions --strips DIR`: a picture per question for a program that can see (a multimodal model),
//! as the Decisions page shows them. A strip is a row of square crops, one per moment the question shows (its
//! `boxes`), each around the green box (the piece asked about) and the grey box (the person it may be), outlined, then
//! the whole frame of the middle moment with the same outlines, for where it is. Every panel is PANEL pixels high.
//!
//! Pictures of people: a strip shows their faces to whatever reads it, and a model behind an API sees them on its
//! company's servers. So strips are made without asking only for a public-domain clip (mm_core::privacy: jazz); for
//! any other clip the caller must pass `--allow-faces`, which says the person whose footage it is chose to send it
//! (AGENTS.md, Privacy).
use image::{imageops, Rgb, RgbImage};
use mm_core::pyjson::Py;
use std::path::{Path, PathBuf};

pub const PANEL: u32 = 320;
const GREEN: Rgb<u8> = Rgb([40, 230, 90]);
const GREY: Rgb<u8> = Rgb([200, 200, 200]);

type Box4 = [f64; 4];

fn box4(v: &Py) -> Option<Box4> {
    let l = v.as_list()?;
    (l.len() == 4).then(|| [0, 1, 2, 3].map(|i| l[i].as_f64().unwrap_or(0.0)))
}

fn scaled(b: Option<Box4>, k: f64) -> Option<Box4> {
    b.map(|b| b.map(|x| x * k))
}

/// One question's strip (its slim JSON), written to DIR/<id>.png; what it shows, for that JSON.
pub fn write(proj: &Path, meta_w: f64, q: &Py, dir: &Path) -> Result<Py, String> {
    let shows: Vec<(i64, Option<Box4>, Option<Box4>)> = q.get("boxes").as_list().map(|l| {
        l.iter().map(|s| (s.get("t").as_i64().unwrap_or(0), box4(s.get("piece")), box4(s.get("person")))).collect()
    }).unwrap_or_default();
    if shows.is_empty() {
        return Err("no boxes to show".into());
    }
    let mut panels: Vec<RgbImage> = vec![];
    for (t, piece, person) in &shows {
        let (img, k) = frame(proj, *t, meta_w)?;
        panels.push(crop_panel(&img, scaled(*piece, k), scaled(*person, k)));
    }
    let (t, piece, person) = shows[(shows.len() - 1) / 2];
    let (img, k) = frame(proj, t, meta_w)?;
    panels.push(wide_panel(&img, scaled(piece, k), scaled(person, k)));

    let gap = 6;
    let w: u32 = panels.iter().map(|p| p.width()).sum::<u32>() + gap * (panels.len() as u32 - 1);
    let mut out = RgbImage::from_pixel(w, PANEL, Rgb([24, 22, 32]));
    let mut x = 0;
    for p in &panels {
        imageops::replace(&mut out, p, x as i64, 0);
        x += p.width() + gap;
    }
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let path: PathBuf = dir.join(format!("{}.png", q.get("id").py_str().replace([':', '/'], "_")));
    out.save(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(mm_core::pydict![
        ("path", path.to_string_lossy().into_owned()),
        ("frames", Py::List(shows.iter().map(|s| Py::Int(s.0)).collect())),
        ("shows", format!("{} close-ups (one per frame in `frames`), then the whole of frame {t}; green outline: the piece, \
                           grey: the person", shows.len())),
    ])
}

/// The scanned frame and the scale from video pixels (the boxes') to its own.
fn frame(proj: &Path, t: i64, meta_w: f64) -> Result<(RgbImage, f64), String> {
    let p = proj.join(format!("frames/{t:05}.jpg"));
    let img = image::open(&p).map_err(|e| format!("{}: {e}", p.display()))?.to_rgb8();
    let k = if meta_w > 0.0 { img.width() as f64 / meta_w } else { 1.0 };
    Ok((img, k))
}

/// A square around the boxes with room around them (decide.rs crop: 1.4 times their extent, at least 70 px), black
/// past the frame's edge, scaled to PANEL.
fn crop_panel(img: &RgbImage, piece: Option<Box4>, person: Option<Box4>) -> RgbImage {
    let bs: Vec<Box4> = piece.iter().chain(person.iter()).cloned().collect();
    let x0 = bs.iter().map(|b| b[0]).fold(f64::INFINITY, f64::min);
    let y0 = bs.iter().map(|b| b[1]).fold(f64::INFINITY, f64::min);
    let x1 = bs.iter().map(|b| b[2]).fold(f64::NEG_INFINITY, f64::max);
    let y1 = bs.iter().map(|b| b[3]).fold(f64::NEG_INFINITY, f64::max);
    let side = ((x1 - x0).max(y1 - y0) * 1.4).max(70.0).ceil();
    let (ox, oy) = (((x0 + x1) - side) / 2.0, ((y0 + y1) - side) / 2.0);
    let s = side as u32;
    let mut sq = RgbImage::new(s, s);
    for y in 0..s {
        for x in 0..s {
            let (sx, sy) = (ox as i64 + x as i64, oy as i64 + y as i64);
            if sx >= 0 && sy >= 0 && (sx as u32) < img.width() && (sy as u32) < img.height() {
                sq.put_pixel(x, y, *img.get_pixel(sx as u32, sy as u32));
            }
        }
    }
    let mut p = imageops::resize(&sq, PANEL, PANEL, imageops::FilterType::CatmullRom);
    let k = PANEL as f64 / side;
    let at = |b: Box4| [(b[0] - ox) * k, (b[1] - oy) * k, (b[2] - ox) * k, (b[3] - oy) * k];
    if let Some(b) = person {
        outline(&mut p, at(b), GREY);
    }
    if let Some(b) = piece {
        outline(&mut p, at(b), GREEN);
    }
    p
}

/// The whole frame, PANEL high.
fn wide_panel(img: &RgbImage, piece: Option<Box4>, person: Option<Box4>) -> RgbImage {
    let k = PANEL as f64 / img.height() as f64;
    let w = ((img.width() as f64 * k).round() as u32).max(1);
    let mut p = imageops::resize(img, w, PANEL, imageops::FilterType::Triangle);
    if let Some(b) = person {
        outline(&mut p, b.map(|x| x * k), GREY);
    }
    if let Some(b) = piece {
        outline(&mut p, b.map(|x| x * k), GREEN);
    }
    p
}

fn outline(p: &mut RgbImage, b: Box4, c: Rgb<u8>) {
    let (w, h) = (p.width() as i64, p.height() as i64);
    let (x0, y0, x1, y1) = (b[0].round() as i64, b[1].round() as i64, b[2].round() as i64, b[3].round() as i64);
    let mut put = |x: i64, y: i64| {
        if x >= 0 && y >= 0 && x < w && y < h {
            p.put_pixel(x as u32, y as u32, c);
        }
    };
    for d in 0..2 {
        for x in x0..=x1 {
            put(x, y0 + d);
            put(x, y1 - d);
        }
        for y in y0..=y1 {
            put(x0 + d, y);
            put(x1 - d, y);
        }
    }
}
