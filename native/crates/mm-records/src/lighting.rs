//! Scene-matched emoji lighting: each emoji is a matte cardboard cut-out standing where the face is.
//!
//! Two layers:
//!   * AUTO estimate (computed once, cached): per frame the scene's colour cast and exposure, per track
//!     piece per frame the light on that face relative to its usual light, plus a light direction.
//!   * USER settings (edits["light"], the review tool's wheels/sliders) applied on top. The same maths
//!     runs in light_ui.js for the live preview -- keep `apply_settings` and `lightFor()` in sync.
//!
//! Model, in LINEAR light (channels BGR in Python, RGB in JSON/JS):
//! ```text
//!     albedo = desat(albedo, saturation)               # the Saturation slider (0 = gray, 1 = the emoji's own)
//!     lit = albedo * (key * ramp(x, y) + ambient)
//!     lit = max(lit, albedo * floor)                  # legibility floor
//!     out = lerp(albedo, lit, mix)                     # mix 0 = flat emoji
//!     out = black + out * (1 - black)                  # FILM: the footage's black level (old film is never black)
//! ```
//!   black = black_auto(t) * user black. Desaturating the albedo BEFORE the light keeps the scene's cast: on sepia
//!   footage gray emoji turn sepia.
//!   - black_auto  = 1st-percentile luminance (linear), smoothed over time
//!   - colour_auto = 90th-percentile pixel chroma / COLOUR_REF, capped at 1 (colour video ~0.6, B&W ~0): only
//!     scales the colour part of the export's grain (none on black-and-white footage). Saturation
//!     is a plain slider: on our clips this measure was only ever ~0 or ~1, so it added nothing.
//!
//!   key_auto = cast(t) * white(t) * resid(t) ** local_k
//!   - cast(t)  = the frame's white-patch colour cast, smoothed over time. ONE light colour per moment, shared
//!     by every face: a face's own colour can't tell its skin from its light, and a colour averaged
//!     over each track's lifetime made two covers on one head differ (pink vs lavender flicker).
//!   - white(t) = white-patch luminance now: a white card is no brighter than the scene's brightest diffuse
//!     surfaces (exposure anchor; carries strobes)
//!   - resid(t) = this face's brightness now vs. its usual, divided by the scene's -- local spots (brightness
//!     only). Comes from the noisy face probe, so it is smoothed and damped (power < 1). Dividing by
//!     the piece's own median cancels skin tone: shading never encodes what the face looked like.
//!
//!   key     = level * key_auto, or with a Light colour picked: its brightness (mean channel) in that colour;
//!   ambient = amb * scene mean colour (always automatic);
//!   ramp    = 1 + ramp_strength * (gx * x + gy * y), from left/right and top/bottom face brightness.
//! Export also adds grain matched to the frame and motion blur along the track's velocity.
//!
//! ## Port notes
//! Port of the estimate half of mojimask/lighting.py (the drawing half, lit_sprite / paste_lit / motion blur, is
//! mm-render's): settings, film_settings, apply_settings, wheel_bgr, the AUTO estimate ([`Lighting`]) and
//! [`frame_grain`]. Bit-identical with numpy on macOS (native/tools/parity/records_lighting.py):
//! - frames are PROJECT/frames/NNNNN.jpg decoded as cv2.imread does (mm_vision::jpeg); `f / 255.0` and `to_lin`
//!   take only 256 values, so they are tabled;
//! - numpy's reductions, as measured on numpy 2.4: `x.mean()` of a contiguous array is one pairwise sum; of a
//!   non-contiguous 2-D slice (`L[:, mx:]`) the buffered iterator copies as many whole rows as fit in its 8192-element
//!   buffer and sums each such block pairwise ([`slice_mean`]); `a.mean(0)` of an (n, 3) array adds the rows in order;
//!   percentiles are numpy's linear method (selection instead of a sort, same values);
//! - `np.median(float32) * 1.4826` stays float32 (NEP 50).
//!
//! The caches have Python's names and contents ([`crate::npz`]): film_est_{n}_{size}_{mtime}.npz (colour, black)
//! and light_est_{md5[:12]}.npz (cast, mean, white, probe), so either side reads the other's.
use crate::npz::{self, F64Array, Member, Pickled};
use anyhow::{anyhow, Result};
use md5::{Digest, Md5};
use mm_core::np;
use mm_core::pyjson::{Dict, Py};
use mm_vision::{gaussian_blur_f32, resize_u8, Interp, Mat};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// 90th-percentile chroma at which footage counts as fully colourful
pub const COLOUR_REF: f64 = 0.4;
/// a frame without its own estimate borrows the piece's nearest one within this many (sent to light_ui.js)
pub const NEAR_FRAMES: i64 = 15;
/// light_colour: None = the estimated light colour (per face, per frame), or [hue_deg, radius 0..1] from the Light
/// colour wheel: HSV (hue, radius * WHEEL_SAT, 1), so the wheel's centre is white light.
pub const WHEEL_SAT: f64 = 0.8;
pub const LOCAL_K: f64 = 0.6;
pub const LOCAL_A: f64 = 0.25;
/// covers whose centres are within this x their size of each other share one light estimate
pub const OVERLAP: f64 = 0.5;

/// lighting.DEFAULTS, in its order. saturation: the emoji's own colours, 0 = gray, 1 = as drawn; black: multiplier
/// on the footage's measured black level (1 = match it, 0 = emoji keep true black); shutter = motion blur length as
/// a fraction of one frame's movement (1.0 = shutter open the whole frame, 0 = off; "blur": False also turns it off);
/// mix ("Strength"): 1 = fully lit, 0 = the flat emoji.
pub fn defaults() -> Dict {
    let mut d = Dict::new();
    d.insert("enabled", Py::Bool(true));
    d.insert("level", Py::Float(1.0));
    d.insert("mix", Py::Float(1.0));
    d.insert("amb", Py::Float(0.15));
    d.insert("ramp", Py::Float(0.8));
    d.insert("floor", Py::Float(0.12));
    d.insert("light_colour", Py::Null);
    d.insert("grain", Py::Bool(true));
    d.insert("blur", Py::Bool(true));
    d.insert("shutter", Py::Float(1.0));
    d.insert("saturation", Py::Float(1.0));
    d.insert("black", Py::Float(1.0));
    d
}

/// `{**DEFAULTS, **((edits or {}).get("light") or {})}`
pub fn settings(edits: &Py) -> Dict {
    let mut d = defaults();
    if let Some(l) = edits.get("light").as_dict() {
        for (k, v) in l.iter() {
            d.insert(k, v.clone());
        }
    }
    d
}

fn num(st: &Dict, k: &str) -> f64 {
    st.get(k).and_then(|v| v.as_f64()).unwrap_or(f64::NAN)
}

/// Python's min(a, b) / max(a, b) on floats (the first unless the second is strictly smaller / larger).
fn pymin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// User saturation + the footage's measured black level x the user's multiplier -> (saturation, black).
/// Mirrors light_ui.js filmFor(). colour_auto isn't used here (only for grain).
pub fn film_settings(_colour_auto: f64, black_auto: f64, st: &Dict) -> (f64, f64) {
    (pymin(np::pymax(num(st, "saturation"), 0.0), 1.0), pymin(np::pymax(black_auto * num(st, "black"), 0.0), 0.5))
}

/// numpy's mean of a few float64 values (pairwise sum: sequential below 8).
fn mean_small(v: &[f64]) -> f64 {
    np::mean(v)
}

/// normalise so mean channel = 1
pub fn chroma(c: &[f64; 3]) -> [f64; 3] {
    let m = np::pymax(mean_small(c), 1e-6);
    [c[0] / m, c[1] / m, c[2] / m]
}

/// Python's float `%` for a positive divisor.
fn pymod(x: f64, y: f64) -> f64 {
    let m = x % y;
    if m != 0.0 && (m < 0.0) != (y < 0.0) {
        m + y
    } else {
        m
    }
}

/// colorsys.hsv_to_rgb
fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (f64, f64, f64) {
    if s == 0.0 {
        return (v, v, v);
    }
    let i = (h * 6.0) as i64; // int() truncates; h >= 0 here
    let f = (h * 6.0) - i as f64;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match i.rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

/// Light colour wheel [hue_deg, radius] -> light colour normalised to mean 1, BGR (light_ui.js wheelRGB).
pub fn wheel_bgr(hue: f64, amt: f64) -> [f64; 3] {
    let (r, g, b) = hsv_to_rgb(pymod(hue, 360.0) / 360.0, pymin(np::pymax(amt, 0.0), 1.0) * WHEEL_SAT, 1.0);
    chroma(&[b, g, r])
}

/// Auto estimate + user settings -> (key, ambient, ramp gradient). Mirrors light_ui.js lightFor().
pub fn apply_settings(key: [f64; 3], grad: [f64; 2], amb_chroma: [f64; 3], st: &Dict) -> ([f64; 3], [f64; 3], [f64; 2]) {
    let level = num(st, "level");
    let mut key = key.map(|k| level * k);
    let lc = st.get("light_colour").cloned().unwrap_or(Py::Null);
    if lc.truthy() {
        let l = lc.as_list().cloned().unwrap_or_default();
        let w = wheel_bgr(l.first().and_then(|v| v.as_f64()).unwrap_or(0.0), l.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0));
        let m = mean_small(&key);
        key = w.map(|c| m * c);
    }
    let amb = num(st, "amb");
    let ramp = num(st, "ramp");
    (key, amb_chroma.map(|c| amb * c), grad.map(|g| g * ramp))
}

// ------------------------------------------------------------------ colour helpers

/// sRGB 0..1 -> linear
pub fn to_lin(x: f64) -> f64 {
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

/// luminance of linear BGR
pub fn lum(b: f64, g: f64, r: f64) -> f64 {
    0.0722 * b + 0.7152 * g + 0.2126 * r
}

/// to_lin(v / 255.0) for every byte
fn lin_lut() -> &'static [f64; 256] {
    static L: OnceLock<[f64; 256]> = OnceLock::new();
    L.get_or_init(|| std::array::from_fn(|v| to_lin(v as f64 / 255.0)))
}

/// A frame as cv2.imread gives it: BGR bytes, rows of w pixels.
pub struct Frame {
    pub w: usize,
    pub h: usize,
    pub bgr: Vec<u8>,
}

impl Frame {
    pub fn read(frames_dir: &Path, t: usize) -> Result<Frame> {
        let p = frames_dir.join(format!("{t:05}.jpg"));
        let d = mm_vision::jpeg::read_bgr(&p).map_err(|e| anyhow!("lighting: can't read frame {}: {e}", p.display()))?;
        Ok(Frame { w: d.w, h: d.h, bgr: d.data })
    }
}

/// numpy's linear percentile (`np.percentile(v, q)`), by selection; v is reordered.
fn percentile_sel(v: &mut [f64], q: f64) -> f64 {
    let n = v.len();
    let qq = q / 100.0;
    let vi = (n as f64 - 1.0) * qq;
    let prev = vi.floor();
    let gamma = vi - prev;
    let i = (prev as usize).min(n - 1);
    let (_, a, rest) = v.select_nth_unstable_by(i, |x, y| x.partial_cmp(y).unwrap());
    let a = *a;
    let b = if i + 1 < n { rest.iter().copied().fold(f64::INFINITY, f64::min) } else { a };
    let diff = b - a;
    if gamma >= 0.5 {
        b - diff * (1.0 - gamma)
    } else {
        a + diff * gamma
    }
}

/// The frame's per-pixel luminance (lum(to_lin(f / 255))).
fn luminance(f: &Frame) -> Vec<f64> {
    let lut = lin_lut();
    f.bgr.chunks_exact(3).map(|p| lum(lut[p[0] as usize], lut[p[1] as usize], lut[p[2] as usize])).collect()
}

/// White-patch cast (top-5% luminance, unclipped), mean colour, white-patch luminance. `l`: the frame's luminance.
pub fn scene_stats(f: &Frame, l: &[f64]) -> ([f64; 3], [f64; 3], f64) {
    let lut = lin_lut();
    let mut ok: Vec<bool> = f.bgr.chunks_exact(3).map(|p| lut[p[0].max(p[1]).max(p[2]) as usize] < 0.95).collect();
    if ok.iter().filter(|&&o| o).count() < 50 {
        ok.iter_mut().for_each(|o| *o = true);
    }
    let mut lok: Vec<f64> = l.iter().zip(&ok).filter(|(_, &o)| o).map(|(&v, _)| v).collect();
    let p95 = percentile_sel(&mut lok, 95.0);
    let mut s = [0.0f64; 3];
    let mut m = [0.0f64; 3];
    let mut lsel = Vec::new();
    for ((p, &lv), &o) in f.bgr.chunks_exact(3).zip(l).zip(&ok) {
        let c = [lut[p[0] as usize], lut[p[1] as usize], lut[p[2] as usize]];
        for j in 0..3 {
            m[j] += c[j];
        }
        if o && lv >= p95 {
            for j in 0..3 {
                s[j] += c[j];
            }
            lsel.push(lv);
        }
    }
    let k = lsel.len() as f64;
    let npx = (f.w * f.h) as f64;
    (chroma(&s.map(|v| v / k)), m.map(|v| v / npx), np::mean(&lsel))
}

/// (colourfulness 0..1, black level linear) of one frame: what the camera/film did, not the light.
pub fn film_stats(f: &Frame, l: &[f64]) -> (f64, f64) {
    let mut ch: Vec<f64> =
        f.bgr.chunks_exact(3).map(|p| p[0].max(p[1]).max(p[2]) as f64 / 255.0 - p[0].min(p[1]).min(p[2]) as f64 / 255.0).collect();
    let c = pymin(percentile_sel(&mut ch, 90.0) / COLOUR_REF, 1.0);
    let mut lv = l.to_vec();
    (c, percentile_sel(&mut lv, 1.0))
}

/// `.mean()` of columns [c0, c1) of a contiguous h x w array (numpy's buffered reduction: blocks of whole rows,
/// as many as fit in 8192 elements, each summed pairwise).
fn slice_mean(a: &[f64], w: usize, h: usize, c0: usize, c1: usize) -> f64 {
    let rl = c1 - c0;
    if rl == w || h == 1 {
        return np::mean(&a[c0 + 0..(h - 1) * w + c1]);
    }
    let per = (8192 / rl).max(1);
    let mut acc = 0.0f64;
    let mut buf = Vec::with_capacity(per * rl);
    let mut r = 0;
    while r < h {
        buf.clear();
        for rr in r..(r + per).min(h) {
            buf.extend_from_slice(&a[rr * w + c0..rr * w + c1]);
        }
        acc += np::sum(&buf);
        r += per;
    }
    acc / (rl * h) as f64
}

/// Mean colour of the inner face, and a left/right, top/bottom brightness gradient.
pub fn face_probe(f: &Frame, c: &[f64; 3]) -> Option<([f64; 3], [f64; 2])> {
    let (wf, hf) = (f.w as f64, f.h as f64);
    let r = c[2] * 0.28; // inner part of the head box: mostly face, little background
    let x1 = np::pymax(c[0] - r, 0.0).trunc() as i64;
    let y1 = np::pymax(c[1] - r, 0.0).trunc() as i64;
    let x2 = pymin(c[0] + r, wf).trunc() as i64;
    let y2 = pymin(c[1] + r, hf).trunc() as i64;
    if x2 - x1 < 3 || y2 - y1 < 3 {
        return None;
    }
    let (x1, y1, x2, y2) = (x1 as usize, y1 as usize, x2 as usize, y2 as usize);
    let (w, h) = (x2 - x1, y2 - y1);
    let lut = lin_lut();
    let mut l = Vec::with_capacity(w * h);
    let mut s = [0.0f64; 3];
    for y in y1..y2 {
        for x in x1..x2 {
            let i = (y * f.w + x) * 3;
            let p = [lut[f.bgr[i] as usize], lut[f.bgr[i + 1] as usize], lut[f.bgr[i + 2] as usize]];
            l.push(lum(p[0], p[1], p[2]));
            for j in 0..3 {
                s[j] += p[j];
            }
        }
    }
    let (mx, my) = (w / 2, h / 2);
    let d = np::pymax(np::mean(&l) * 2.0, 1e-6);
    let right = slice_mean(&l, w, h, mx, w);
    let left = slice_mean(&l, w, h, 0, mx);
    let bottom = np::mean(&l[my * w..]);
    let top = np::mean(&l[..my * w]);
    let g = [np::clip((right - left) / d, -0.5, 0.5), np::clip((bottom - top) / d, -0.5, 0.5)];
    let n = (w * h) as f64;
    Some((s.map(|v| v / n), g))
}

/// EMA over consecutive frames (restarts after gaps).
pub fn smooth_series<const N: usize>(d: &BTreeMap<i64, [f64; N]>, a: f64) -> BTreeMap<i64, [f64; N]> {
    let mut out = BTreeMap::new();
    let mut prev: Option<[f64; N]> = None;
    let mut pt = 0i64;
    for (&t, v) in d {
        let nv = match prev {
            Some(p) if t - pt <= 2 => std::array::from_fn(|j| a * v[j] + (1.0 - a) * p[j]),
            _ => *v,
        };
        out.insert(t, nv);
        prev = Some(nv);
        pt = t;
    }
    out
}

/// Rows over consecutive frames, EMA forwards then backwards: smooth without lagging behind.
pub fn smooth_both_ways<const N: usize>(rows: &[[f64; N]], a: f64) -> Vec<[f64; N]> {
    let f = smooth_series(&rows.iter().enumerate().map(|(t, v)| (t as i64, *v)).collect(), a);
    let b = smooth_series(&f.iter().map(|(&t, v)| (-t, *v)).collect(), a);
    (0..rows.len() as i64).map(|t| b[&-t]).collect()
}

/// Robust noise sigma (8-bit units) of a BGR uint8 frame: the median deviation from a slightly blurred copy, read
/// on 16 tiles spread over the frame (a quarter of it; within ~10% of the whole frame's, 3-4x faster).
pub fn frame_grain(f: &Frame) -> f64 {
    let (hh, ww) = (f.h, f.w);
    let (th, tw) = ((hh / 8).max(16), (ww / 8).max(16));
    let mut ds: Vec<f32> = Vec::new();
    for i in 0..4 {
        for j in 0..4 {
            let y = ((2 * i + 1) * hh / 8).saturating_sub(th / 2);
            let x = ((2 * j + 1) * ww / 8).saturating_sub(tw / 2);
            let (y2, x2) = ((y + th).min(hh), (x + tw).min(ww));
            let mut tile = Vec::with_capacity((y2 - y) * (x2 - x) * 3);
            for r in y..y2 {
                tile.extend_from_slice(&f.bgr[(r * ww + x) * 3..(r * ww + x2) * 3]);
            }
            let mut g = Mat::from_vec(y2 - y, x2 - x, 3, tile);
            if hh > 1000 {
                g = resize_u8(&g, 0, 0, 0.5, 0.5, Interp::Linear);
            }
            let gf = Mat::from_vec(g.rows, g.cols, 3, g.data.iter().map(|&v| v as f32).collect());
            let bl = gaussian_blur_f32(&gf, 0, 0, 1.0, 1.0);
            for r in 2..gf.rows.saturating_sub(2) {
                for c in 2..gf.cols.saturating_sub(2) {
                    for k in 0..3 {
                        let i = (r * gf.cols + c) * 3 + k;
                        ds.push((gf.data[i] - bl.data[i]).abs());
                    }
                }
            }
        }
    }
    (np::median(&ds) * 1.4826f32) as f64
}

// ------------------------------------------------------------------ the estimate

type Probe = Vec<(String, BTreeMap<i64, ([f64; 3], [f64; 2])>)>;

/// Per frame (colourfulness, black level) and per frame/piece scene + face readings, from the caches or measured.
/// snap: {piece: {t: [cx, cy, s]}} automatic track pieces, in the proto_tracks order.
pub struct Lighting {
    snap: Vec<(String, BTreeMap<i64, [f64; 3]>)>,
    snap_ix: HashMap<String, usize>,
    /// footage colourfulness and black level per frame, smoothed
    pub colour: Vec<f64>,
    pub black: Vec<f64>,
    pub cast: Vec<[f64; 3]>,
    pub mean: Vec<[f64; 3]>,
    pub white: Vec<f64>,
    /// per-frame light colour
    pub cast_s: Vec<[f64; 3]>,
    /// per piece (probe order): {t: key}, {t: grad}
    key: Vec<(String, BTreeMap<i64, [f64; 3]>)>,
    grad: Vec<BTreeMap<i64, [f64; 2]>>,
    key_ix: HashMap<String, usize>,
    n: usize,
    /// the cache files used (read or written)
    pub film_cache: PathBuf,
    pub light_cache: PathBuf,
}

fn arr1(v: &[f64]) -> Member {
    Member::F64(F64Array { shape: vec![v.len()], data: v.to_vec() })
}

fn arr3(v: &[[f64; 3]]) -> Member {
    Member::F64(F64Array { shape: vec![v.len(), 3], data: v.iter().flatten().copied().collect() })
}

fn get_f64(ms: &[(String, Member)], name: &str) -> Result<F64Array> {
    match ms.iter().find(|(k, _)| k == name) {
        Some((_, Member::F64(a))) => Ok(a.clone()),
        _ => Err(anyhow!("cache: no float64 {name}")),
    }
}

fn rows3(a: &F64Array) -> Vec<[f64; 3]> {
    a.data.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

/// The light_est cache's key: hashlib.md5(json.dumps({p: sorted(b.items()) for p, b in sorted(snap.items())},
/// default=float)).hexdigest()[:12]
pub fn snap_key(snap: &[(String, BTreeMap<i64, [f64; 3]>)]) -> String {
    let mut ps: Vec<&(String, BTreeMap<i64, [f64; 3]>)> = snap.iter().collect();
    ps.sort_by(|a, b| a.0.cmp(&b.0));
    let mut d = Dict::new();
    for (p, b) in ps {
        let rows = b.iter().map(|(&t, c)| Py::List(vec![Py::Int(t), Py::List(c.iter().map(|&v| Py::Float(v)).collect())])).collect();
        d.insert(p.clone(), Py::List(rows));
    }
    let text = Py::Dict(d).dumps();
    let h = Md5::digest(text.as_bytes());
    h.iter().map(|b| format!("{b:02x}")).collect::<String>()[..12].to_string()
}

fn probe_to_pickle(p: &Probe) -> Pickled {
    let arr = |v: &[f64]| Pickled::Array(F64Array { shape: vec![v.len()], data: v.to_vec() });
    Pickled::ObjArray(
        vec![],
        vec![Pickled::Dict(
            p.iter()
                .map(|(k, m)| {
                    (
                        Pickled::Str(k.clone()),
                        Pickled::Dict(m.iter().map(|(&t, (a, g))| (Pickled::Int(t), Pickled::Tuple(vec![arr(a), arr(g)]))).collect()),
                    )
                })
                .collect(),
        )],
    )
}

fn probe_from_pickle(v: &Pickled) -> Result<Probe> {
    let d = match v {
        Pickled::ObjArray(_, items) if items.len() == 1 => items[0].as_dict(),
        other => other.as_dict(),
    }
    .ok_or_else(|| anyhow!("probe: not a dict"))?;
    let mut out = vec![];
    for (k, m) in d {
        let Pickled::Str(k) = k else { return Err(anyhow!("probe: key")) };
        let mut mm = BTreeMap::new();
        for (t, pr) in m.as_dict().ok_or_else(|| anyhow!("probe: {k}"))? {
            let Pickled::Int(t) = t else { return Err(anyhow!("probe: frame")) };
            let Pickled::Tuple(two) = pr else { return Err(anyhow!("probe: entry")) };
            match (&two[0], &two[1]) {
                (Pickled::Array(a), Pickled::Array(g)) if a.data.len() == 3 && g.data.len() == 2 => {
                    mm.insert(*t, ([a.data[0], a.data[1], a.data[2]], [g.data[0], g.data[1]]));
                }
                _ => return Err(anyhow!("probe: arrays")),
            }
        }
        out.push((k.clone(), mm));
    }
    Ok(out)
}

impl Lighting {
    /// lighting.Lighting(snap, frames_dir, cache_dir, progress).
    pub fn new(snap: &[(String, BTreeMap<i64, [f64; 3]>)], frames_dir: &Path, cache_dir: &Path, progress: Option<&(dyn Fn(f64) + Sync)>) -> Result<Lighting> {
        let n = std::fs::read_dir(frames_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".jpg"))
            .count();
        // film: independent of the tracks, but not of the frames (the first frame's size and time are in the name)
        let tag = if n > 0 {
            let md = std::fs::metadata(frames_dir.join("00000.jpg"))?;
            let mt = md.modified()?.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
            format!("{n}_{}_{}", md.len(), mt as i64)
        } else {
            format!("{n}")
        };
        let film_cache = cache_dir.join(format!("film_est_{tag}.npz"));
        let light_cache = cache_dir.join(format!("light_est_{}.npz", snap_key(snap)));
        let film = if film_cache.exists() {
            let z = npz::load(&film_cache)?;
            Some((get_f64(&z, "colour")?.data, get_f64(&z, "black")?.data))
        } else {
            None
        };
        let scene = if light_cache.exists() {
            let z = npz::load(&light_cache)?;
            let probe = match z.iter().find(|(k, _)| k == "probe") {
                Some((_, Member::Object(v))) => probe_from_pickle(v)?,
                _ => return Err(anyhow!("{}: no probe", light_cache.display())),
            };
            Some((rows3(&get_f64(&z, "cast")?), rows3(&get_f64(&z, "mean")?), get_f64(&z, "white")?.data, probe))
        } else {
            None
        };
        let (film, scene) = Self::measure(snap, frames_dir, n, film, scene, progress)?;
        if !film_cache.exists() {
            std::fs::create_dir_all(cache_dir)?;
            npz::save(&film_cache, &[("colour", arr1(&film.0)), ("black", arr1(&film.1))])?;
        }
        if !light_cache.exists() {
            std::fs::create_dir_all(cache_dir)?;
            npz::save(
                &light_cache,
                &[("cast", arr3(&scene.0)), ("mean", arr3(&scene.1)), ("white", arr1(&scene.2)), ("probe", Member::Object(probe_to_pickle(&scene.3)))],
            )?;
        }
        let sm = |v: &[f64]| -> Vec<f64> {
            smooth_series(&v.iter().enumerate().map(|(t, &x)| (t as i64, [x])).collect(), 0.15).values().map(|x| x[0]).collect()
        };
        let (colour, black) = (sm(&film.0), sm(&film.1));
        let (cast, mean, white, probe) = scene;
        let nn = white.len();
        let cast_s: Vec<[f64; 3]> = smooth_both_ways(&cast, 0.3).iter().map(chroma).collect();
        let mut key = vec![];
        let mut grad = vec![];
        for (p, pr) in &probe {
            if pr.len() < 3 {
                continue;
            }
            let med: [f64; 3] = std::array::from_fn(|j| np::median(&pr.values().map(|v| v.0[j]).collect::<Vec<_>>()));
            let wmed = np::median(&pr.keys().map(|&t| white[t as usize]).collect::<Vec<_>>());
            let mut rs = BTreeMap::new();
            for (&t, v) in pr {
                let scene = white[t as usize] / wmed;
                let q: [f64; 3] = std::array::from_fn(|j| np::clip(v.0[j] / if med[j] >= 1e-4 { med[j] } else { 1e-4 }, 0.2, 3.0));
                let ratio = mean_small(&q);
                rs.insert(t, [np::clip(ratio / scene, 0.33, 3.0)]);
            }
            let resid = smooth_series(&rs, LOCAL_A);
            let k: BTreeMap<i64, [f64; 3]> = pr
                .keys()
                .map(|&t| {
                    let w = white[t as usize];
                    let f = resid[&t][0].powf(LOCAL_K);
                    (t, cast_s[t as usize].map(|c| c * w * f))
                })
                .collect();
            key.push((p.clone(), k));
            grad.push(smooth_series(&pr.iter().map(|(&t, v)| (t, v.1)).collect(), 0.25));
        }
        let snap_v = snap.to_vec();
        let snap_ix = snap_v.iter().enumerate().map(|(i, (p, _))| (p.clone(), i)).collect();
        let key_ix = key.iter().enumerate().map(|(i, (p, _))| (p.clone(), i)).collect();
        let mut l = Lighting {
            snap: snap_v,
            snap_ix,
            colour,
            black,
            cast,
            mean,
            white,
            cast_s,
            key,
            grad,
            key_ix,
            n: nn,
            film_cache,
            light_cache,
        };
        l.share_overlaps();
        Ok(l)
    }

    #[allow(clippy::type_complexity)]
    fn measure(
        snap: &[(String, BTreeMap<i64, [f64; 3]>)],
        frames_dir: &Path,
        n: usize,
        film: Option<(Vec<f64>, Vec<f64>)>,
        scene: Option<(Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<f64>, Probe)>,
        progress: Option<&(dyn Fn(f64) + Sync)>,
    ) -> Result<((Vec<f64>, Vec<f64>), (Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<f64>, Probe))> {
        if let (Some(f), Some(s)) = (&film, &scene) {
            return Ok((f.clone(), s.clone()));
        }
        let (mut colour, mut black) = (vec![1.0; n], vec![0.0; n]);
        let (mut cast, mut mean, mut white) = (vec![[1.0; 3]; n], vec![[0.0; 3]; n], vec![1.0; n]);
        let mut by_t: BTreeMap<i64, Vec<(usize, [f64; 3])>> = BTreeMap::new();
        for (i, (_, b)) in snap.iter().enumerate() {
            for (&t, c) in b {
                by_t.entry(t).or_default().push((i, *c));
            }
        }
        let mut probe: Probe = snap.iter().map(|(p, _)| (p.clone(), BTreeMap::new())).collect();
        // Each frame is measured on its own, so frames are spread over a few threads and put back in order: the
        // result is the same as one by one. A 50 s 4K clip took minutes on one core. Few threads, as each holds a
        // frame's full-size float arrays (about 300 MB at 4K).
        type Row = (Option<(f64, f64)>, Option<([f64; 3], [f64; 3], f64, Vec<(usize, ([f64; 3], [f64; 2]))>)>);
        let rows: Vec<std::sync::Mutex<Option<Result<Row>>>> = (0..n).map(|_| std::sync::Mutex::new(None)).collect();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let done = std::sync::atomic::AtomicUsize::new(0);
        let failed = std::sync::atomic::AtomicBool::new(false);
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).clamp(1, 6);
        let (want_film, want_scene) = (film.is_none(), scene.is_none());
        std::thread::scope(|sc| {
            for _ in 0..threads {
                sc.spawn(|| loop {
                    let t = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if t >= n || failed.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }
                    let row = (|| -> Result<Row> {
                        let f = Frame::read(frames_dir, t)?;
                        let l = luminance(&f);
                        let fs = want_film.then(|| film_stats(&f, &l));
                        let ss = want_scene.then(|| {
                            let (c, m, w) = scene_stats(&f, &l);
                            let pr = by_t.get(&(t as i64)).map(|v| v.as_slice()).unwrap_or(&[]).iter()
                                .filter_map(|(i, c)| face_probe(&f, c).map(|p| (*i, p))).collect();
                            (c, m, w, pr)
                        });
                        Ok((fs, ss))
                    })();
                    if row.is_err() {
                        failed.store(true, std::sync::atomic::Ordering::Relaxed); // the others stop at their next frame
                    }
                    *rows[t].lock().unwrap() = Some(row);
                    // progress by frames finished, so it only goes forward whatever order they finish in
                    let k = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if let Some(p) = progress {
                        if want_scene && k % 50 == 0 {
                            p(k as f64 / n as f64);
                        }
                    }
                });
            }
        });
        if failed.into_inner() {
            // the first error in frame order (frames after a failure may not have been measured)
            let first = rows.into_iter().find_map(|r| match r.into_inner().unwrap() {
                Some(Err(e)) => Some(e),
                _ => None,
            });
            return Err(first.expect("a failed frame"));
        }
        for (t, r) in rows.into_iter().enumerate() {
            let (fs, ss) = r.into_inner().unwrap().expect("every frame measured")?;
            if let Some(v) = fs {
                (colour[t], black[t]) = v;
            }
            if let Some((c, m, w, pr)) = ss {
                (cast[t], mean[t], white[t]) = (c, m, w);
                for (i, v) in pr {
                    probe[i].1.insert(t as i64, v);
                }
            }
        }
        Ok((film.unwrap_or((colour, black)), scene.unwrap_or((cast, mean, white, probe))))
    }

    /// Covers sitting on the same spot at the same moment (two track pieces on one head) are under the same
    /// light: give each group their average, or whichever is drawn on top flips the emoji's shade frame to frame.
    fn share_overlaps(&mut self) {
        let mut by_t: BTreeMap<i64, Vec<(usize, [f64; 3])>> = BTreeMap::new();
        for (ki, (p, keys)) in self.key.iter().enumerate() {
            let sb = &self.snap[self.snap_ix[p]].1;
            for t in keys.keys() {
                if let Some(c) = sb.get(t) {
                    by_t.entry(*t).or_default().push((ki, *c));
                }
            }
        }
        for (t, items) in by_t {
            if items.len() < 2 {
                continue;
            }
            let mut group: HashMap<usize, usize> = items.iter().map(|(p, _)| (*p, *p)).collect();
            fn find(g: &HashMap<usize, usize>, p: usize) -> usize {
                if g[&p] == p {
                    p
                } else {
                    find(g, g[&p])
                }
            }
            for (i, (p, a)) in items.iter().enumerate() {
                for (q, b) in &items[i + 1..] {
                    let m = np::pymax(a[2], b[2]);
                    if (a[0] - b[0]).abs() < OVERLAP * m && (a[1] - b[1]).abs() < OVERLAP * m {
                        let fp = find(&group, *p);
                        let fq = find(&group, *q);
                        group.insert(fq, fp);
                    }
                }
            }
            let mut groups: Vec<(usize, Vec<usize>)> = vec![];
            for (p, _) in &items {
                let r = find(&group, *p);
                match groups.iter_mut().find(|(k, _)| *k == r) {
                    Some((_, v)) => v.push(*p),
                    None => groups.push((r, vec![*p])),
                }
            }
            for (_, ps) in groups {
                if ps.len() < 2 {
                    continue;
                }
                let m = ps.len() as f64;
                let mut k = [0.0f64; 3];
                let mut g = [0.0f64; 2];
                for &p in &ps {
                    let kv = self.key[p].1[&t];
                    let gv = self.grad[p][&t];
                    for j in 0..3 {
                        k[j] += kv[j];
                    }
                    for j in 0..2 {
                        g[j] += gv[j];
                    }
                }
                let (k, g) = (k.map(|v| v / m), g.map(|v| v / m));
                for &p in &ps {
                    self.key[p].1.insert(t, k);
                    self.grad[p].insert(t, g);
                }
            }
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    fn clamp_t(&self, t: i64) -> usize {
        t.clamp(0, self.n as i64 - 1) as usize
    }

    /// (key BGR, grad xy, ambient chroma BGR) before user settings.
    pub fn auto(&self, piece: &str, t: i64) -> ([f64; 3], [f64; 2], [f64; 3]) {
        let tu = self.clamp_t(t);
        let t = tu as i64;
        let ki = self.key_ix.get(piece).copied();
        let (mut k, mut g) = match ki {
            Some(i) => (self.key[i].1.get(&t).copied(), self.grad[i].get(&t).copied()),
            None => (None, None),
        };
        if k.is_none() {
            if let Some(i) = ki {
                // a frame the probe didn't measure (failed probe, or a flow-carried cover past the track's boxes):
                // the piece's nearest measured frame if it is close, so the emoji's colour doesn't jump
                let keys = &self.key[i].1;
                if let Some(&u) = keys.keys().min_by_key(|&&u| (u - t).abs()) {
                    if (u - t).abs() <= NEAR_FRAMES {
                        k = Some(keys[&u]);
                        g = self.grad[i].get(&u).copied();
                    }
                }
            }
        }
        // manual piece / no usable probe: scene cast and exposure only
        let k = k.unwrap_or_else(|| self.cast_s[tu].map(|c| c * self.white[tu]));
        (k, g.unwrap_or([0.0, 0.0]), chroma(&self.mean[tu]))
    }

    /// (colourfulness, black level) of the footage at frame t, before user settings.
    pub fn film(&self, t: i64) -> (f64, f64) {
        let t = self.clamp_t(t);
        (self.colour[t], self.black[t])
    }

    /// The piece's movement per frame from its automatic boxes (half the difference one frame either side).
    pub fn velocity(&self, piece: &str, t: i64) -> (f64, f64) {
        let Some(&i) = self.snap_ix.get(piece) else { return (0.0, 0.0) };
        let b = &self.snap[i].1;
        let Some(cur) = b.get(&t) else { return (0.0, 0.0) };
        let p = b.get(&(t - 1)).unwrap_or(cur);
        let q = b.get(&(t + 1)).unwrap_or(cur);
        ((q[0] - p[0]) / 2.0, (q[1] - p[1]) / 2.0)
    }

    /// Compact payload for the review tool (RGB order): per-frame fallback key + ambient chroma, per-piece per-frame
    /// [kr, kg, kb, gx, gy], and NEAR_FRAMES (light_ui.js borrows estimates as far).
    pub fn to_json(&self) -> Py {
        let r = |x: f64, n: usize| Py::Float(np::pyround_n(x, n));
        let r3 = |v: &[f64; 3]| -> Vec<Py> { v.iter().rev().map(|&x| r(x, 4)).collect() };
        let mut d = Dict::new();
        d.insert("ready", Py::Bool(true));
        d.insert("n", Py::Int(self.n as i64));
        d.insert("defaults", Py::Dict(defaults()));
        d.insert("wheel_sat", Py::Float(WHEEL_SAT));
        d.insert("near_frames", Py::Int(NEAR_FRAMES));
        d.insert("scene_key", Py::List((0..self.n).map(|t| Py::List(r3(&self.cast_s[t].map(|c| c * self.white[t])))).collect()));
        d.insert("scene_amb", Py::List((0..self.n).map(|t| Py::List(r3(&chroma(&self.mean[t])))).collect()));
        d.insert("film", Py::List((0..self.n).map(|t| Py::List(vec![r(self.colour[t], 3), r(self.black[t], 5)])).collect()));
        let mut pieces = Dict::new();
        for (i, (p, k)) in self.key.iter().enumerate() {
            let mut m = Dict::new();
            for (t, kv) in k {
                let mut row = r3(kv);
                row.extend(self.grad[i][t].iter().map(|&g| r(g, 3)));
                m.insert(t.to_string(), Py::List(row));
            }
            pieces.insert(p.clone(), Py::Dict(m));
        }
        d.insert("pieces", Py::Dict(pieces));
        Py::Dict(d)
    }
}
