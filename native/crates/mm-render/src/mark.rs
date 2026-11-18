//! The optional corner mark (Export's "Add a small ... mark"; track.watermark and track._mark_sprite).
//!
//! The wordmark as brand.js sets it (about.BRAND: the name's case, typeface, weight, letter-spacing and the two parts'
//! colours, as on the splash and in the app's bar), in the lower right of the OUTPUT picture (drawn after the format's
//! crop, so it's never cut off), a little transparent (MARK_ALPHA 0.85), with a soft shadow so it reads on any footage.
//! Its height follows the picture's short side: px = max(10, round(short * MARK_SIZE)), MARK_SIZE 0.028; margin
//! round(0.8 px) from the edges; skipped if it doesn't fit.
//!
//! The sprite, as Python builds it with Pillow: each letter drawn on its own (basic layout: no kerning) at
//! pad - min(x0, 0) + at(i), at(i) = the advances before it + i * tracking * px, the ascender line at pad - y0 (getbbox);
//! pad = max(2, px // 4). Letters take the first colour up to `split`, then the second. The shadow: the letters' alpha
//! moved down max(1, px // 14) rows (np.roll: it wraps), Gaussian-blurred (sigma max(1, px / 12)) x 0.8; it adds alpha,
//! no colour. col = rgb * a * MARK_ALPHA, alpha = (a + shadow (1 - a)) * MARK_ALPHA; the picture becomes
//! trunc(clip(col + picture * (1 - alpha))).
//!
//! The letters as Pillow's FreeType draws them: the WOFF2 file decoded by CoreText (its tables copied into an sfnt),
//! outlines hinted by skrifa's port of FreeType's autohinter (what Pillow's FreeType uses on this font), whole-pixel
//! advances, then rasterised here by exact area coverage as FreeType's smooth rasteriser
//! does (curves flattened into 16 lines each).
use anyhow::{anyhow, Result};

pub const MARK_SIZE: f64 = 0.028;
pub const MARK_ALPHA: f32 = 0.85;

#[derive(Clone, Debug)]
pub struct MarkStyle {
    pub name: String,
    pub split: usize,
    pub weight: f64,
    pub tracking: f64,
    pub first: [u8; 3],
    pub second: [u8; 3],
    /// the font file's bytes (TTF, OTF, WOFF or WOFF2)
    pub font: Vec<u8>,
}

fn hex(s: &str) -> Result<[u8; 3]> {
    let h = s.trim_start_matches('#');
    let p = |i: usize| u8::from_str_radix(&h[i..i + 2], 16).map_err(|e| anyhow!("colour {s}: {e}"));
    Ok([p(0)?, p(2)?, p(4)?])
}

impl MarkStyle {
    /// From a JSON object with "brand" (brand.js's BRAND) and "font_file" (its font file), as draw_ref.py's configs.json.
    pub fn from_brand(cfg: &serde_json::Value) -> Result<MarkStyle> {
        let b = &cfg["brand"];
        let s = |k: &str| b[k].as_str().ok_or_else(|| anyhow!("brand.{k}"));
        let f = |k: &str| b[k].as_f64().ok_or_else(|| anyhow!("brand.{k}"));
        Ok(MarkStyle {
            name: s("name")?.to_string(),
            split: f("split")? as usize,
            weight: f("weight")?,
            tracking: f("tracking")?,
            first: hex(s("first")?)?,
            second: hex(s("second")?)?,
            font: std::fs::read(cfg["font_file"].as_str().ok_or_else(|| anyhow!("font_file"))?)?,
        })
    }
}

/// The mark's picture: w x h, colour (RGB, premultiplied by a * MARK_ALPHA, 0..255) and alpha per pixel.
#[derive(Clone)]
pub struct MarkSprite {
    pub w: usize,
    pub h: usize,
    pub col: Vec<[f32; 3]>,
    pub alpha: Vec<f32>,
}

// ------------------------------------------------------------------ outlines and an area-coverage rasteriser

#[derive(Clone, Copy, Debug)]
enum Seg {
    Move(f64, f64),
    Line(f64, f64),
    Quad(f64, f64, f64, f64),
    Cubic(f64, f64, f64, f64, f64, f64),
    Close,
}

/// One letter: its outline (font units scaled to px, y up), rounded advance, control box.
struct Glyph {
    segs: Vec<Seg>,
    advance: f64,
    cbox: (f64, f64, f64, f64),
}

/// Exact signed-area coverage of line segments (the font-rs accumulation method; FreeType's smooth rasteriser computes
/// the same areas). Coordinates in pixels, y down.
struct Raster {
    w: usize,
    h: usize,
    a: Vec<f32>,
}

impl Raster {
    fn new(w: usize, h: usize) -> Raster {
        Raster { w, h, a: vec![0.0; w * h + 4] }
    }

    fn line(&mut self, p0: (f32, f32), p1: (f32, f32)) {
        if p0.1 == p1.1 {
            return;
        }
        let (dir, p0, p1) = if p0.1 < p1.1 { (1.0, p0, p1) } else { (-1.0, p1, p0) };
        let dxdy = (p1.0 - p0.0) / (p1.1 - p0.1);
        let mut x = p0.0;
        let y0 = p0.1.max(0.0) as usize;
        if p0.1 < 0.0 {
            x -= p0.1 * dxdy;
        }
        for y in y0..self.h.min(p1.1.ceil() as usize) {
            let ls = y * self.w;
            let dy = ((y + 1) as f32).min(p1.1) - (y as f32).max(p0.1);
            let xnext = x + dxdy * dy;
            let d = dy * dir;
            let (x0, x1) = if x < xnext { (x, xnext) } else { (xnext, x) };
            let x0f = x0.floor();
            let x0i = x0f as i64;
            let x1c = x1.ceil();
            let x1i = x1c as i64;
            let at = |xi: i64| (ls as i64 + xi).clamp(0, (self.w * self.h) as i64) as usize;
            if x1i <= x0i + 1 {
                let xmf = 0.5 * (x + xnext) - x0f;
                self.a[at(x0i)] += d - d * xmf;
                self.a[at(x0i + 1)] += d * xmf;
            } else {
                let s = (x1 - x0).recip();
                let x0r = x0 - x0f;
                let a0 = 0.5 * s * (1.0 - x0r) * (1.0 - x0r);
                let x1r = x1 - x1c + 1.0;
                let am = 0.5 * s * x1r * x1r;
                self.a[at(x0i)] += d * a0;
                if x1i == x0i + 2 {
                    self.a[at(x0i + 1)] += d * (1.0 - a0 - am);
                } else {
                    let a1 = s * (1.5 - x0r);
                    self.a[at(x0i + 1)] += d * (a1 - a0);
                    for xi in x0i + 2..x1i - 1 {
                        self.a[at(xi)] += d * s;
                    }
                    let a2 = a1 + (x1i - x0i - 3) as f32 * s;
                    self.a[at(x1i - 1)] += d * (1.0 - a2 - am);
                }
                self.a[at(x1i)] += d * am;
            }
            x = xnext;
        }
    }

    /// Fill `segs` placed with its origin at (ox, oy) (y down in the raster, the outline's y up).
    fn fill(&mut self, segs: &[Seg], ox: f64, oy: f64) {
        let pt = |x: f64, y: f64| ((ox + x) as f32, (oy - y) as f32);
        let (mut cur, mut start) = ((0.0f32, 0.0f32), (0.0f32, 0.0f32));
        let n = 16;
        for s in segs {
            match *s {
                Seg::Move(x, y) => {
                    if cur != start {
                        self.line(cur, start);
                    }
                    cur = pt(x, y);
                    start = cur;
                }
                Seg::Line(x, y) => {
                    let p = pt(x, y);
                    self.line(cur, p);
                    cur = p;
                }
                Seg::Quad(cx, cy, x, y) => {
                    let (c, p) = (pt(cx, cy), pt(x, y));
                    let p0 = cur;
                    for i in 1..=n {
                        let t = i as f32 / n as f32;
                        let u = 1.0 - t;
                        let q = (u * u * p0.0 + 2.0 * u * t * c.0 + t * t * p.0, u * u * p0.1 + 2.0 * u * t * c.1 + t * t * p.1);
                        self.line(cur, q);
                        cur = q;
                    }
                }
                Seg::Cubic(c1x, c1y, c2x, c2y, x, y) => {
                    let (c1, c2, p) = (pt(c1x, c1y), pt(c2x, c2y), pt(x, y));
                    let p0 = cur;
                    for i in 1..=n {
                        let t = i as f32 / n as f32;
                        let u = 1.0 - t;
                        let (b0, b1, b2, b3) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
                        let q = (b0 * p0.0 + b1 * c1.0 + b2 * c2.0 + b3 * p.0, b0 * p0.1 + b1 * c1.1 + b2 * c2.1 + b3 * p.1);
                        self.line(cur, q);
                        cur = q;
                    }
                }
                Seg::Close => {
                    if cur != start {
                        self.line(cur, start);
                    }
                    cur = start;
                }
            }
        }
        if cur != start {
            self.line(cur, start);
        }
    }

    /// Coverage as FreeType's 8-bit grey levels: min(255, floor(coverage * 256)).
    fn levels(&self) -> Vec<u8> {
        let mut acc = 0.0f32;
        self.a[..self.w * self.h]
            .iter()
            .map(|v| {
                acc += v;
                ((acc.abs().min(1.0) * 256.0) as u32).min(255) as u8
            })
            .collect()
    }
}

// ------------------------------------------------------------------ the font: CoreText decodes, skrifa hints

/// The font as plain sfnt bytes: TTF/OTF as is; WOFF/WOFF2 decoded by CoreText (its tables copied back out into an
/// sfnt; checksums left 0, which skrifa doesn't check).
#[cfg(target_os = "macos")]
fn sfnt(font: &[u8]) -> Result<Vec<u8>> {
    use objc2_core_foundation::CFData;
    use objc2_core_text::{CTFont, CTFontManagerCreateFontDescriptorFromData, CTFontTableOptions};
    if !font.starts_with(b"wOF") {
        return Ok(font.to_vec());
    }
    unsafe {
        let data = CFData::from_bytes(font);
        let desc = CTFontManagerCreateFontDescriptorFromData(&data).ok_or_else(|| anyhow!("CoreText can't read the font"))?;
        let f = CTFont::with_font_descriptor(&desc, 12.0, std::ptr::null());
        let tags = f.available_tables(CTFontTableOptions::NoOptions).ok_or_else(|| anyhow!("no font tables"))?;
        let mut tables: Vec<(u32, Vec<u8>)> = Vec::new();
        for i in 0..tags.count() {
            let tag = tags.value_at_index(i) as usize as u32;
            if let Some(d) = f.table(tag, CTFontTableOptions::NoOptions) {
                tables.push((tag, d.to_vec()));
            }
        }
        tables.sort_by_key(|t| t.0);
        let n = tables.len();
        let mut out = Vec::new();
        out.extend_from_slice(&0x0001_0000u32.to_be_bytes());
        let es = (usize::BITS - 1 - n.leading_zeros()) as u16;
        let sr = (1u16 << es) * 16;
        for v in [n as u16, sr, es, n as u16 * 16 - sr] {
            out.extend_from_slice(&v.to_be_bytes());
        }
        let mut off = 12 + 16 * n;
        for (tag, d) in &tables {
            out.extend_from_slice(&tag.to_be_bytes());
            out.extend_from_slice(&0u32.to_be_bytes());
            out.extend_from_slice(&(off as u32).to_be_bytes());
            out.extend_from_slice(&(d.len() as u32).to_be_bytes());
            off += d.len().div_ceil(4) * 4;
        }
        for (_, d) in &tables {
            out.extend_from_slice(d);
            out.resize(out.len().div_ceil(4) * 4, 0);
        }
        Ok(out)
    }
}

#[cfg(not(target_os = "macos"))]
fn sfnt(font: &[u8]) -> Result<Vec<u8>> {
    if font.starts_with(b"wOF") {
        return Err(anyhow!("WOFF fonts need CoreText (macOS) for now"));
    }
    Ok(font.to_vec())
}

struct Pen(Vec<Seg>);
impl skrifa::outline::OutlinePen for Pen {
    fn move_to(&mut self, x: f32, y: f32) {
        self.0.push(Seg::Move(x as f64, y as f64));
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.0.push(Seg::Line(x as f64, y as f64));
    }
    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        self.0.push(Seg::Quad(cx as f64, cy as f64, x as f64, y as f64));
    }
    fn curve_to(&mut self, a: f32, b: f32, c: f32, d: f32, x: f32, y: f32) {
        self.0.push(Seg::Cubic(a as f64, b as f64, c as f64, d as f64, x as f64, y as f64));
    }
    fn close(&mut self) {
        self.0.push(Seg::Close);
    }
}

/// The name's letters at px, hinted as Pillow's FreeType does with FT_LOAD_DEFAULT (here: its autohinter, smooth normal
/// target), with FreeType's rounded advances; and the ascender (ceil).
fn glyphs(style: &MarkStyle, px: f64) -> Result<(Vec<Glyph>, f64)> {
    use skrifa::instance::{LocationRef, Size};
    use skrifa::outline::{DrawSettings, HintingInstance, HintingOptions};
    use skrifa::MetadataProvider;
    let bytes = sfnt(&style.font)?;
    let font = skrifa::FontRef::new(&bytes).map_err(|e| anyhow!("font: {e}"))?;
    let loc = font.axes().location([("wght", style.weight as f32)]);
    let size = Size::new(px as f32);
    let outlines = font.outline_glyphs();
    // FreeType's autohinter: what Pillow's FreeType 2.14 does with this font (its hinted advances at 13 px, 12 9 4 4 12 8
    // 7 8, and the sprite's 70 x 19, match only with it; with skrifa's choice, the TrueType interpreter, they're 12 8 4 4
    // 12 7 6 8 and 67 x 19)
    let opts = HintingOptions::from(skrifa::outline::Engine::Auto(None));
    let hint = HintingInstance::new(&outlines, size, LocationRef::from(&loc), opts)
        .map_err(|e| anyhow!("hinting: {e:?}"))?;
    let metrics = font.metrics(size, LocationRef::from(&loc));
    let gm = font.glyph_metrics(size, LocationRef::from(&loc));
    let cmap = font.charmap();
    let mut out = Vec::new();
    for ch in style.name.chars() {
        let gid = cmap.map(ch).ok_or_else(|| anyhow!("no glyph for {ch:?}"))?;
        let g = outlines.get(gid).ok_or_else(|| anyhow!("no outline for {ch:?}"))?;
        let mut pen = Pen(Vec::new());
        let am = g.draw(DrawSettings::hinted(&hint, false), &mut pen).map_err(|e| anyhow!("draw: {e:?}"))?;
        let adv = am.advance_width.or_else(|| gm.advance_width(gid)).unwrap_or(0.0) as f64;
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        let mut grow = |x: f64, y: f64| {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        };
        for sg in &pen.0 {
            match *sg {
                Seg::Move(x, y) | Seg::Line(x, y) => grow(x, y),
                Seg::Quad(a, b, x, y) => {
                    grow(a, b);
                    grow(x, y)
                }
                Seg::Cubic(a, b, c, d, x, y) => {
                    grow(a, b);
                    grow(c, d);
                    grow(x, y)
                }
                Seg::Close => {}
            }
        }
        let cbox = if x0 == f64::MAX { (0.0, 0.0, 0.0, 0.0) } else { (x0, y0, x1, y1) };
        out.push(Glyph { segs: pen.0, advance: adv.round(), cbox });
    }
    Ok((out, (metrics.ascent as f64).ceil()))
}

/// cv2.getGaussianKernel(k, sigma) for float images (k > 7 or sigma given: computed, normalised in double).
fn gauss_kernel(sigma: f64) -> Vec<f32> {
    let k = ((sigma * 4.0 * 2.0 + 1.0).round() as i64 | 1) as usize;
    let c = (k as f64 - 1.0) / 2.0;
    let v: Vec<f64> = (0..k).map(|i| (-(i as f64 - c).powi(2) / (2.0 * sigma * sigma)).exp()).collect();
    let s: f64 = v.iter().sum();
    v.iter().map(|x| (x / s) as f32).collect()
}

fn reflect101(i: i64, n: i64) -> usize {
    if n == 1 {
        return 0;
    }
    let mut j = i;
    while j < 0 || j >= n {
        if j < 0 {
            j = -j;
        }
        if j >= n {
            j = 2 * n - 2 - j;
        }
    }
    j as usize
}

/// cv2.GaussianBlur(a, (0, 0), sigma) on a float image (separable, reflect-101 borders).
fn gaussian_blur(a: &[f32], w: usize, h: usize, sigma: f64) -> Vec<f32> {
    let k = gauss_kernel(sigma);
    let r = (k.len() / 2) as i64;
    let mut t = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            t[y * w + x] = k.iter().enumerate().map(|(i, kv)| kv * a[y * w + reflect101(x as i64 + i as i64 - r, w as i64)]).sum();
        }
    }
    let mut o = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            o[y * w + x] = k.iter().enumerate().map(|(i, kv)| kv * t[reflect101(y as i64 + i as i64 - r, h as i64) * w + x]).sum();
        }
    }
    o
}

/// Pillow's DIV255 / BLEND (Paste.c).
fn blend(mask: u32, a: u32, b: u32) -> u8 {
    let t = a * (255 - mask) + b * mask + 128;
    (((t >> 8) + t) >> 8) as u8
}

/// track._mark_sprite: the wordmark at px tall.
pub fn mark_sprite(style: &MarkStyle, px: u32) -> Result<MarkSprite> {
    let (gs, asc) = glyphs(style, px as f64)?;
    let track = style.tracking * px as f64;
    let n = gs.len();
    let at = |i: usize| gs[..i].iter().map(|g| g.advance).sum::<f64>() + i as f64 * track;
    // font.getbbox(word): the letters' control boxes along the pen (no tracking), "la" anchor (top = the ascender)
    let mut pen = 0.0;
    let (mut bx0, mut by0, mut bx1, mut by1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for g in &gs {
        if !g.segs.is_empty() {
            bx0 = bx0.min(pen + g.cbox.0);
            bx1 = bx1.max(pen + g.cbox.2);
            by0 = by0.min(asc - g.cbox.3);
            by1 = by1.max(asc - g.cbox.1);
        }
        pen += g.advance;
    }
    let (x0, y0, x1, y1) = (bx0.floor(), by0.floor(), bx1.ceil(), by1.ceil());
    let x1 = x1.max(at(n - 1) + gs[n - 1].advance);
    let pad = 2.max(px / 4) as f64;
    let w = (x1 - x0.min(0.0)) as usize + 2 * pad as usize;
    let h = (y1 - y0) as usize + 2 * pad as usize;
    let mut rgba = vec![[0u8; 4]; w * h];
    for (i, g) in gs.iter().enumerate() {
        let ink = if i < style.split { style.first } else { style.second };
        let mut r = Raster::new(w, h);
        r.fill(&g.segs, pad - x0.min(0.0) + at(i), pad - y0 + asc);
        for (p, m) in rgba.iter_mut().zip(r.levels()) {
            if m == 0 {
                continue;
            }
            let m = m as u32;
            let cm = 255 - (255 - m) * p[3] as u32 / 255;
            for c in 0..3 {
                p[c] = blend(cm, p[c] as u32, ink[c] as u32);
            }
            p[3] = blend(m, p[3] as u32, 255);
        }
    }
    let a: Vec<f32> = rgba.iter().map(|p| p[3] as f32 / 255.0).collect();
    let dy = 1.max(px / 14) as usize;
    let mut rolled = vec![0.0f32; w * h];
    for y in 0..h {
        let sy = (y + h - dy % h) % h;
        rolled[y * w..(y + 1) * w].copy_from_slice(&a[sy * w..(sy + 1) * w]);
    }
    let shadow: Vec<f32> = gaussian_blur(&rolled, w, h, (px as f64 / 12.0).max(1.0)).iter().map(|v| v * 0.8).collect();
    let col = rgba.iter().zip(&a).map(|(p, &ai)| [0, 1, 2].map(|c| p[c] as f32 * ai * MARK_ALPHA)).collect();
    let alpha = a.iter().zip(&shadow).map(|(&ai, &s)| (ai + s * (1.0 - ai)) * MARK_ALPHA).collect();
    Ok(MarkSprite { w, h, col, alpha })
}

/// Where the mark goes on a W x H picture: (px, x, y) or None when it doesn't fit.
pub fn place(w: u32, h: u32, sprite_w: usize, sprite_h: usize, px: u32) -> Option<(u32, u32)> {
    let m = (px as f64 * 0.8).round_ties_even() as i64;
    let (y, x) = (h as i64 - sprite_h as i64 - m, w as i64 - sprite_w as i64 - m);
    (y >= 0 && x >= 0).then_some((x as u32, y as u32))
}

/// The mark's height for a W x H output picture.
pub fn mark_px(w: u32, h: u32) -> u32 {
    10.max((w.min(h) as f64 * MARK_SIZE).round_ties_even() as u32)
}

const SHADER: &str = r#"
struct U { x: u32, y: u32, w: u32, h: u32 };
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> u: U;
@group(0) @binding(3) var<storage, read> spr: array<vec4<f32>>;   // rgb premultiplied 0..255, alpha
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let d = textureDimensions(src);
    if (g.x >= d.x || g.y >= d.y) { return; }
    var v = textureLoad(src, g.xy, 0);
    if (g.x >= u.x && g.y >= u.y && g.x < u.x + u.w && g.y < u.y + u.h) {
        let s = spr[(g.y - u.y) * u.w + (g.x - u.x)];
        let roi = round(v.rgb * 255.0);
        v = vec4<f32>(floor(clamp(s.rgb + roi * (1.0 - s.a), vec3<f32>(0.0), vec3<f32>(255.0))) / 255.0, 1.0);
    }
    textureStore(dst, g.xy, v);
}
"#;

/// Draws the mark on output pictures (track.watermark), on the GPU; sprites cached by size.
pub struct Marker {
    style: MarkStyle,
    format: wgpu::TextureFormat,
    pipe: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    cache: std::collections::HashMap<u32, MarkSprite>,
}

impl Marker {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, style: MarkStyle) -> Marker {
        let fmt = if format == wgpu::TextureFormat::Bgra8Unorm { "bgra8unorm" } else { "rgba8unorm" };
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mark"),
            source: wgpu::ShaderSource::Wgsl(SHADER.replace("rgba8unorm", fmt).into()),
        });
        use wgpu::BindingType as B;
        let e = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mark"),
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
                e(3, B::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mark"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Marker { style, format, pipe, bgl, cache: Default::default() }
    }

    /// `src` with the mark drawn (a new texture; `src` itself is returned when the mark doesn't fit).
    pub fn draw(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, src: wgpu::Texture) -> Result<wgpu::Texture> {
        use wgpu::util::DeviceExt;
        let (w, h) = (src.width(), src.height());
        let px = mark_px(w, h);
        if !self.cache.contains_key(&px) {
            self.cache.insert(px, mark_sprite(&self.style, px)?);
        }
        let s = &self.cache[&px];
        let Some((x, y)) = place(w, h, s.w, s.h, px) else { return Ok(src) };
        // (WGSL sees every 8-bit format's channels as r, g, b: no swizzle for bgra8unorm)
        let data: Vec<[f32; 4]> = s.col.iter().zip(&s.alpha).map(|(c, &a)| [c[0], c[1], c[2], a]).collect();
        let sb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mark"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let ub = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[x, y, s.w as u32, s.h as u32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let out = crate::compositor::out_texture(device, w, h, self.format);
        let sv = src.create_view(&Default::default());
        let ov = out.create_view(&Default::default());
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&sv) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&ov) },
                wgpu::BindGroupEntry { binding: 2, resource: ub.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: sb.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&self.pipe);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
        }
        queue.submit([enc.finish()]);
        Ok(out)
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    /// prints the layout (compare with Pillow's getlength/getbbox at the same size)
    #[test]
    #[ignore]
    fn layout() {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../web/fonts/Nunito.woff2");
        let st = MarkStyle { name: "mojimask".into(), split: 4, weight: 900.0, tracking: -0.01, first: [255; 3],
                             second: [255, 210, 63], font: std::fs::read(root).unwrap() };
        let (gs, asc) = glyphs(&st, 13.0).unwrap();
        eprintln!("asc {asc} adv {:?}", gs.iter().map(|g| g.advance).collect::<Vec<_>>());
        eprintln!("cbox {:?}", gs.iter().map(|g| g.cbox).collect::<Vec<_>>());
        let s = mark_sprite(&st, 13).unwrap();
        eprintln!("sprite {}x{}", s.w, s.h);
    }
}
