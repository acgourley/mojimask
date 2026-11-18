//! pose-orbs: draw a little 3-D head (an orb with eyes, nose, mouth and ears) on every posed person of a project, to
//! see by eye how well head3d's fit follows the heads, lit like the scene.
//!
//!   pose-orbs <project dir> [--out DIR] [--smooth SIGMA_FRAMES] [--alpha A] [--dots] [--flat] [--frames a..b]
//!
//! Reads <project>/frames/NNNNN.jpg and <project>/dets/pose_full.json; writes DIR/NNNNN.png, DIR/fits.json and, when
//! ffmpeg is on the PATH, DIR/orbs.mp4 (DIR defaults to out/native/pose_orbs/<project name>).
//!
//! - Pose: head3d::fit per skeleton. Each orb is drawn at the centre and size of the head box the app covers
//!   (pose_util::head_box of the same skeleton: the head sphere's diameter is the box's side), turned by the fit.
//!   Fits are linked frame to frame (nearest head centre); --smooth smooths each track's pose over time with a
//!   Gaussian of that many frames (0, the default, draws the raw fits).
//! - Light: the app's model (lighting.rs, its defaults), from the same measurements: key = the frame's white-patch
//!   cast x white level x this face's brightness against its track's usual (^ LOCAL_K); ambient = AMB x the frame's
//!   mean colour; a floor; the footage's black level. Where the app ramps a flat card by the face's left/right and
//!   top/bottom brightness gradient, the orb is a sphere, so the gradient becomes a light direction (tilted towards
//!   the brighter side) and the sphere is shaded by it (wrapped Lambert: matte, no hard terminator). Light is
//!   smoothed over 2 frames per track. --flat draws unlit colours.
//! - --dots marks the keypoints the fit used (white) and where the raw fit puts them (red).
//!
//! Only for public-domain clips (test_clips/jazz.mp4): it writes the frames themselves.
use anyhow::Context;
use image::{Rgb, RgbImage};
use mm_analysis::head3d::{self, Pose3, HEAD_R, MODEL};
use mm_analysis::pose_util;
use mm_records::lighting::{self as lt, Frame};
use std::path::{Path, PathBuf};

const EAR_R: f64 = 0.32;
const EAR_AT: f64 = 1.22; // ear sphere centres, a little inside MODEL's ear points so they sit on the head
// lighting.rs defaults
const AMB: f64 = 0.15;
const RAMP: f64 = 0.8;
const FLOOR: f64 = 0.12;
/// light direction tilt per unit of ramped gradient (gradient 0.5 x RAMP -> ~40 deg off the view axis)
const TILT: f64 = 2.0;
/// wrapped Lambert: (n.L + WRAP) / (1 + WRAP), scaled so the sphere's mean shade facing the light is ~1
const WRAP: f64 = 0.5;
const SHADE_NORM: f64 = 1.28;
const LIGHT_SIGMA: f64 = 2.0;

struct Head {
    draw: Pose3,
    raw: Pose3,
    row: Vec<f64>,
    /// linear RGB key light, the gradient [gx, gy] (ramped), ambient RGB
    key: [f64; 3],
    grad: [f64; 2],
    amb: [f64; 3],
    /// the probe's face luminance (None: too small to probe)
    probe: Option<([f64; 3], [f64; 2])>,
}

struct Scene {
    cast: [f64; 3],
    mean: [f64; 3],
    white: f64,
    black: f64,
}

fn rgb(bgr: [f64; 3]) -> [f64; 3] {
    [bgr[2], bgr[1], bgr[0]]
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut proj: Option<PathBuf> = None;
    let (mut out, mut smooth, mut alpha, mut dots, mut flat, mut range) = (None, 0.0f64, 1.0f64, false, false, None);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => out = it.next().map(PathBuf::from),
            "--smooth" => smooth = it.next().and_then(|s| s.parse().ok()).unwrap_or(2.0),
            "--alpha" => alpha = it.next().and_then(|s| s.parse().ok()).unwrap_or(1.0),
            "--dots" => dots = true,
            "--flat" => flat = true,
            "--frames" => {
                range = it.next().and_then(|s| s.split_once("..")).and_then(|(a, b)| Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?)))
            }
            _ if proj.is_none() => proj = Some(PathBuf::from(a)),
            _ => anyhow::bail!("unknown argument {a}"),
        }
    }
    let proj = proj.ok_or_else(|| anyhow::anyhow!("usage: pose-orbs <project dir> [--out DIR] [--smooth S] [--alpha A] [--dots] [--flat]"))?;
    let name = proj.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "project".into());
    let out = out.unwrap_or_else(|| PathBuf::from("out/native/pose_orbs").join(&name));
    std::fs::create_dir_all(&out)?;
    let frames_dir = proj.join("frames");

    let pf: serde_json::Value = {
        let pj = proj.join("dets/pose_full.json");
        serde_json::from_slice(&std::fs::read(&pj).with_context(|| format!("reading {}", pj.display()))?)?
    };
    let rows: Vec<Vec<Vec<f64>>> = serde_json::from_value(pf["frames"].clone())?;
    let mut heads: Vec<Vec<Head>> = rows
        .iter()
        .map(|fr| {
            fr.iter()
                .filter_map(|p| {
                    let f = head3d::fit(p)?;
                    let hb = pose_util::head_box(p)?;
                    let side = (hb[2] - hb[0]).max(hb[3] - hb[1]);
                    let draw = Pose3 { s: side / (2.0 * HEAD_R), t: [(hb[0] + hb[2]) / 2.0, (hb[1] + hb[3]) / 2.0], ..f };
                    Some(Head { draw, raw: f, row: p.clone(), key: [0.0; 3], grad: [0.0; 2], amb: [1.0; 3], probe: None })
                })
                .collect()
        })
        .collect();
    let n = heads.len();

    // the scene, per frame (smoothed over time as the app does), and each head's probe
    let mut scene = Vec::with_capacity(n);
    for (t, hs) in heads.iter_mut().enumerate() {
        let f = Frame::read(&frames_dir, t)?;
        let l = luminance(&f);
        let (cast, mean, white) = lt::scene_stats(&f, &l);
        let (_, black) = lt::film_stats(&f, &l);
        scene.push(Scene { cast: rgb(cast), mean: rgb(mean), white, black });
        for h in hs.iter_mut() {
            h.probe = lt::face_probe(&f, &[h.draw.t[0], h.draw.t[1], h.draw.s * 2.0 * HEAD_R]);
        }
    }
    let scene = smooth_scene(&scene, LIGHT_SIGMA);

    let tracks = link(&heads);
    light_tracks(&mut heads, &tracks, &scene);
    if smooth > 0.0 {
        smooth_pose(&mut heads, &tracks, smooth);
    }

    // the fits, for looking at the numbers: [[yaw_deg, pitch_deg, roll_deg, s, tx, ty, err, n, key_lum, gx, gy], ...]
    // per frame (key_lum: the key light's luminance, linear; gx, gy: the ramped brightness gradient)
    let js: Vec<Vec<[f64; 11]>> = heads
        .iter()
        .map(|fr| {
            fr.iter()
                .map(|h| {
                    let (f, r) = (h.draw, h.raw);
                    let kl = lt::lum(h.key[2], h.key[1], h.key[0]);
                    [f.yaw.to_degrees(), f.pitch.to_degrees(), f.roll.to_degrees(), f.s, f.t[0], f.t[1], r.err, r.n as f64, kl, h.grad[0], h.grad[1]]
                })
                .collect()
        })
        .collect();
    std::fs::write(out.join("fits.json"), serde_json::to_string(&js)?)?;

    let (a, b) = range.unwrap_or((0, n));
    let b = b.min(n);
    for t in a..b {
        let f = Frame::read(&frames_dir, t)?;
        let mut img = RgbImage::from_fn(f.w as u32, f.h as u32, |x, y| {
            let i = (y as usize * f.w + x as usize) * 3;
            Rgb([f.bgr[i + 2], f.bgr[i + 1], f.bgr[i]])
        });
        // far heads first, so near ones draw over them
        let mut order: Vec<&Head> = heads[t].iter().collect();
        order.sort_by(|x, y| x.draw.s.total_cmp(&y.draw.s));
        for h in order {
            draw_orb(&mut img, h, if flat { None } else { Some(scene[t].black) }, alpha);
            if dots {
                draw_dots(&mut img, &h.raw, &h.row);
            }
        }
        img.save(out.join(format!("{t:05}.png")))?;
    }
    let all: Vec<&Head> = heads[a..b].iter().flatten().collect();
    let deg = |v: f64| v.to_degrees();
    let span = |g: &dyn Fn(&Head) -> f64| {
        (deg(all.iter().map(|h| g(h)).fold(f64::MAX, f64::min)), deg(all.iter().map(|h| g(h)).fold(f64::MIN, f64::max)))
    };
    let (y, p, r) = (span(&|h| h.draw.yaw), span(&|h| h.draw.pitch), span(&|h| h.draw.roll));
    println!(
        "{} heads over {} frames ({} tracks); rms keypoint error {:.3} interocular (mean); yaw {:.0}..{:.0} deg, pitch {:.0}..{:.0}, roll {:.0}..{:.0}",
        all.len(),
        b - a,
        tracks.len(),
        all.iter().map(|h| h.raw.err).sum::<f64>() / all.len().max(1) as f64,
        y.0,
        y.1,
        p.0,
        p.1,
        r.0,
        r.1
    );
    encode(&out, a, pf["fps"].as_f64().unwrap_or(24.0));
    println!("{}", out.display());
    Ok(())
}

/// lum(to_lin(f / 255)) per pixel
fn luminance(f: &Frame) -> Vec<f64> {
    let lut: Vec<f64> = (0..256).map(|v| lt::to_lin(v as f64 / 255.0)).collect();
    f.bgr.chunks_exact(3).map(|p| lt::lum(lut[p[0] as usize], lut[p[1] as usize], lut[p[2] as usize])).collect()
}

fn gauss_weights(dt: f64, sigma: f64) -> f64 {
    (-0.5 * (dt / sigma).powi(2)).exp()
}

fn smooth_scene(s: &[Scene], sigma: f64) -> Vec<Scene> {
    let r = (3.0 * sigma).ceil() as i64;
    (0..s.len())
        .map(|t| {
            let (mut w, mut c, mut m, mut wh, mut bl) = (0.0, [0.0; 3], [0.0; 3], 0.0, 0.0);
            for u in (t as i64 - r).max(0)..=(t as i64 + r).min(s.len() as i64 - 1) {
                let g = gauss_weights((u - t as i64) as f64, sigma);
                let x = &s[u as usize];
                for k in 0..3 {
                    c[k] += g * x.cast[k];
                    m[k] += g * x.mean[k];
                }
                wh += g * x.white;
                bl += g * x.black;
                w += g;
            }
            Scene { cast: c.map(|v| v / w), mean: m.map(|v| v / w), white: wh / w, black: bl / w }
        })
        .collect()
}

/// Link heads frame to frame (greedy, nearest head centre within a head radius, gaps up to 3 frames):
/// [(t, index in heads[t])] per track.
fn link(heads: &[Vec<Head>]) -> Vec<Vec<(usize, usize)>> {
    let mut tracks: Vec<Vec<(usize, usize)>> = Vec::new();
    for t in 0..heads.len() {
        let mut taken = vec![false; tracks.len()];
        for i in 0..heads[t].len() {
            let f = &heads[t][i].draw;
            let best = tracks
                .iter()
                .enumerate()
                .filter(|(k, tr)| !taken[*k] && t - tr.last().unwrap().0 <= 3)
                .map(|(k, tr)| {
                    let (lt, li) = *tr.last().unwrap();
                    let g = &heads[lt][li].draw;
                    (k, ((f.t[0] - g.t[0]).powi(2) + (f.t[1] - g.t[1]).powi(2)).sqrt() / (HEAD_R * g.s.max(f.s)))
                })
                .filter(|(_, d)| *d < 1.0)
                .min_by(|x, y| x.1.total_cmp(&y.1));
            match best {
                Some((k, _)) => {
                    tracks[k].push((t, i));
                    taken[k] = true;
                }
                None => {
                    tracks.push(vec![(t, i)]);
                    taken.push(true);
                }
            }
        }
    }
    tracks
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let m = v.len() / 2;
    Some(if v.len() % 2 == 1 { v[m] } else { (v[m - 1] + v[m]) / 2.0 })
}

/// Per track: key = cast x white x resid ^ LOCAL_K, resid = (face luminance / its track median) / (white / the
/// white's median over the same frames), so a face's own tone cancels and only its light against the scene's
/// remains; smoothed over time with the gradient.
fn light_tracks(heads: &mut [Vec<Head>], tracks: &[Vec<(usize, usize)>], scene: &[Scene]) {
    for tr in tracks {
        let plum = |h: &Head| h.probe.map(|(c, _)| lt::lum(c[0], c[1], c[2]));
        let pm = median(tr.iter().filter_map(|&(t, i)| plum(&heads[t][i])).collect());
        let wm = median(tr.iter().map(|&(t, _)| scene[t].white).collect()).unwrap_or(1.0).max(1e-6);
        let raw: Vec<(f64, [f64; 2])> = tr
            .iter()
            .map(|&(t, i)| {
                let h = &heads[t][i];
                match (plum(h), pm, h.probe) {
                    (Some(p), Some(m), Some((_, g))) if m > 1e-6 => {
                        let resid = ((p / m) / (scene[t].white / wm)).clamp(0.25, 4.0);
                        (resid, g)
                    }
                    _ => (1.0, [0.0, 0.0]),
                }
            })
            .collect();
        let r = (3.0 * LIGHT_SIGMA).ceil() as i64;
        for &(t, i) in tr {
            let (mut w, mut lr, mut g) = (0.0, 0.0, [0.0; 2]);
            for (k, &(tk, _)) in tr.iter().enumerate() {
                let dt = tk as i64 - t as i64;
                if dt.abs() > r {
                    continue;
                }
                let gw = gauss_weights(dt as f64, LIGHT_SIGMA);
                lr += gw * raw[k].0.ln();
                g[0] += gw * raw[k].1[0];
                g[1] += gw * raw[k].1[1];
                w += gw;
            }
            let resid = (lr / w).exp().powf(lt::LOCAL_K);
            let s = &scene[t];
            let h = &mut heads[t][i];
            h.key = s.cast.map(|c| c * s.white * resid);
            h.grad = [g[0] / w * RAMP, g[1] / w * RAMP];
            h.amb = lt::chroma(&s.mean).map(|c| c * AMB);
        }
    }
}

/// Smooth each track's drawn pose (angles, size, centre) over time with a Gaussian of sigma frames, weighting each
/// fit by its keypoint error.
fn smooth_pose(heads: &mut [Vec<Head>], tracks: &[Vec<(usize, usize)>], sigma: f64) {
    let r = (3.0 * sigma).ceil() as i64;
    for tr in tracks {
        let raw: Vec<(Pose3, f64)> = tr.iter().map(|&(t, i)| (heads[t][i].draw, heads[t][i].raw.err)).collect();
        for &(t, i) in tr {
            let (mut w, mut acc) = (0.0, [0.0f64; 6]);
            for (k, &(tk, _)) in tr.iter().enumerate() {
                let dt = tk as i64 - t as i64;
                if dt.abs() > r {
                    continue;
                }
                let g = gauss_weights(dt as f64, sigma) / (0.05 + raw[k].1);
                let p = raw[k].0;
                for (a, v) in acc.iter_mut().zip([p.yaw, p.pitch, p.roll, p.s.ln(), p.t[0], p.t[1]]) {
                    *a += g * v;
                }
                w += g;
            }
            let f = &mut heads[t][i].draw;
            f.yaw = acc[0] / w;
            f.pitch = acc[1] / w;
            f.roll = acc[2] / w;
            f.s = (acc[3] / w).exp();
            f.t = [acc[4] / w, acc[5] / w];
        }
    }
}

fn norm(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / l, v[1] / l, v[2] / l]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn mix(a: [f64; 3], b: [f64; 3], k: f64) -> [f64; 3] {
    [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * k)
}

/// linear 0..1 -> sRGB byte
fn to_srgb(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    255.0 * if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
}

const YELLOW: [f64; 3] = [255.0, 203.0, 76.0];
const EAR: [f64; 3] = [240.0, 178.0, 60.0];
const EAR_IN: [f64; 3] = [205.0, 120.0, 55.0];
const DARK: [f64; 3] = [62.0, 40.0, 30.0];
const NOSE_C: [f64; 3] = [235.0, 120.0, 60.0];
const MOUTH: [f64; 3] = [150.0, 45.0, 45.0];
const RIM: [f64; 3] = [90.0, 55.0, 20.0];

/// Colour (sRGB bytes) of the head sphere at head-space direction h (unit vector from the head centre).
fn head_colour(h: [f64; 3]) -> [f64; 3] {
    let eye_l = norm(MODEL[1]);
    let eye_r = norm(MODEL[2]);
    let nose = norm(MODEL[0]);
    if dot(h, eye_l) > 0.16f64.cos() || dot(h, eye_r) > 0.16f64.cos() {
        // a white glint up and to the image left of each pupil, fixed to the face
        let gl = norm([eye_l[0] - 0.07, eye_l[1] - 0.07, eye_l[2]]);
        let gr = norm([eye_r[0] - 0.07, eye_r[1] - 0.07, eye_r[2]]);
        if dot(h, gl) > 0.045f64.cos() || dot(h, gr) > 0.045f64.cos() {
            return [255.0, 255.0, 255.0];
        }
        return DARK;
    }
    if dot(h, nose) > 0.11f64.cos() {
        return NOSE_C;
    }
    if h[2] > 0.0 {
        // smile: an arc of elevation in azimuth, corners up
        let az = h[0].atan2(h[2]);
        let el = h[1].asin();
        let mid = 0.50 - 1.1 * az * az;
        if az.abs() < 0.42 && el > mid - 0.05 && el < mid + 0.05 {
            return MOUTH;
        }
    }
    YELLOW
}

/// black: None draws flat colours; Some(the footage's black level) lights the orb as the head's measured light.
fn draw_orb(img: &mut RgbImage, h: &Head, black: Option<f64>, alpha: f64) {
    let f = &h.draw;
    let r = f.r();
    let ears = [[EAR_AT, MODEL[3][1], MODEL[3][2]], [-EAR_AT, MODEL[4][1], MODEL[4][2]]];
    let mut spheres: Vec<([f64; 3], f64, i32)> = vec![([0.0; 3], HEAD_R, -1)];
    for (k, e) in ears.iter().enumerate() {
        spheres.push((*e, EAR_R, k as i32));
    }
    let cam: Vec<([f64; 3], f64, i32)> = spheres.iter().map(|&(c, rad, k)| (f.project(c), rad * f.s, k)).collect();
    let ext = (HEAD_R + EAR_R + 0.3) * f.s;
    let (w, hgt) = img.dimensions();
    let x0 = (f.t[0] - ext).floor().max(0.0) as u32;
    let y0 = (f.t[1] - ext).floor().max(0.0) as u32;
    let x1 = ((f.t[0] + ext).ceil().max(0.0) as u32).min(w);
    let y1 = ((f.t[1] + ext).ceil().max(0.0) as u32).min(hgt);
    // the light: towards the brighter side of the face (image x right, y down, z to the camera)
    let light = norm([TILT * h.grad[0], TILT * h.grad[1], 1.0]);
    let lin = |c: [f64; 3]| c.map(|v| lt::to_lin(v / 255.0));
    for py in y0..y1 {
        for px in x0..x1 {
            // 3x3 supersampling for smooth edges
            let mut acc = [0.0f64; 3];
            let mut cover = 0.0;
            for sy in 0..3 {
                for sx in 0..3 {
                    let (x, y) = (px as f64 + (sx as f64 + 0.5) / 3.0, py as f64 + (sy as f64 + 0.5) / 3.0);
                    let mut best: Option<(f64, [f64; 3], i32, f64)> = None; // (z, normal, which, rim)
                    for &(c, rad, k) in &cam {
                        let (dx, dy) = (x - c[0], y - c[1]);
                        let d2 = dx * dx + dy * dy;
                        if d2 >= rad * rad {
                            continue;
                        }
                        let dz = (rad * rad - d2).sqrt();
                        let z = c[2] + dz;
                        if best.map_or(true, |b| z > b.0) {
                            best = Some((z, [dx / rad, dy / rad, dz / rad], k, d2.sqrt() / rad));
                        }
                    }
                    let Some((_, n, k, rim)) = best else { continue };
                    let hdir = head3d::apply_t(&r, n); // the surface normal in head space
                    let mut base = if k < 0 {
                        head_colour(hdir)
                    } else {
                        // inner ear: the part of the ear sphere facing sideways out of the head
                        let side = if k == 0 { 1.0 } else { -1.0 };
                        if hdir[0] * side > 0.55 && hdir[2] > -0.2 {
                            mix(EAR, EAR_IN, 0.7)
                        } else {
                            EAR
                        }
                    };
                    if rim > 0.93 {
                        base = mix(base, RIM, 0.6);
                    }
                    let c = match black {
                        None => {
                            let lam = 0.45 + 0.55 * dot(n, norm([-0.45, -0.6, 0.75])).max(0.0);
                            base.map(|v| v * lam)
                        }
                        Some(bl) => {
                            // lighting.rs's model with a sphere's shading in place of the card's ramp
                            let alb = lin(base);
                            let shade = SHADE_NORM * ((dot(n, light) + WRAP) / (1.0 + WRAP)).max(0.0);
                            let lit = [0, 1, 2].map(|i| {
                                let v = alb[i] * (h.key[i] * shade + h.amb[i]);
                                let v = v.max(alb[i] * FLOOR);
                                bl + v * (1.0 - bl)
                            });
                            lit.map(to_srgb)
                        }
                    };
                    for i in 0..3 {
                        acc[i] += c[i];
                    }
                    cover += 1.0;
                }
            }
            if cover == 0.0 {
                continue;
            }
            let a = alpha * cover / 9.0;
            let p = img.get_pixel_mut(px, py);
            for i in 0..3 {
                p[i] = (p[i] as f64 * (1.0 - a) + acc[i] / cover * a).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

fn dot_at(img: &mut RgbImage, x: f64, y: f64, c: [u8; 3]) {
    let (w, h) = img.dimensions();
    for dy in -1i64..=1 {
        for dx in -1i64..=1 {
            let (px, py) = (x.round() as i64 + dx, y.round() as i64 + dy);
            if px >= 0 && py >= 0 && (px as u32) < w && (py as u32) < h {
                img.put_pixel(px as u32, py as u32, Rgb(c));
            }
        }
    }
}

fn draw_dots(img: &mut RgbImage, f: &Pose3, row: &[f64]) {
    for i in 0..5 {
        if row[5 + i * 3 + 2] <= head3d::FIT_VIS {
            continue;
        }
        let m = f.project(MODEL[i]);
        dot_at(img, m[0], m[1], [230, 30, 30]);
        dot_at(img, row[5 + i * 3], row[5 + i * 3 + 1], [255, 255, 255]);
    }
}

fn encode(dir: &Path, start: usize, fps: f64) {
    let ok = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-framerate", &format!("{fps}"), "-start_number", &start.to_string(), "-i"])
        .arg(dir.join("%05d.png"))
        .args(["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "18"])
        .arg(dir.join("orbs.mp4"))
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("(no mp4: ffmpeg missing or failed; the PNGs are there)");
    }
}
