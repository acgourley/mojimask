//! The known-motion test set: pairs of frames where the second is the first moved by a known similarity (mm-vision's
//! warpAffine, bit-exact with cv2's), with the troubles of party video added: motion blur, defocus, noise, low
//! light, JPEG, a change of exposure, and "people" (patches of other frames) moving on their own in front.
//!
//!     motion_testset OUT_DIR SRC.png [SRC.png ...]
//! Writes OUT_DIR/<id>_a.png, <id>_b.png and OUT_DIR/manifest.json: [{id, a, b, M (2x3, a -> b, pixels of the
//! pair), motion, degrade, outliers, src, W, H}]. Deterministic (a fixed-seed generator per pair).
use mm_motion::fit::{self, M23};
use mm_motion::io::{jpeg_roundtrip, read_bgr, write_bgr_png};
use mm_vision::{gaussian_blur_u8, warp_affine_u8, Border, Mat};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn uni(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.next()
    }
    fn sym(&mut self, a: f64, b: f64) -> f64 {
        let v = self.uni(a, b);
        if self.next() < 0.5 { -v } else { v }
    }
    fn gauss(&mut self) -> f64 {
        let u = self.next().max(1e-12);
        let v = self.next();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

const MOTIONS: [&str; 5] = ["small", "large", "whip", "rotate", "zoom"];
const DEGRADES: [&str; 7] = ["clean", "motion_blur", "defocus", "noise", "low_light", "jpeg", "exposure"];
const OUTLIERS: [(&str, usize); 3] = [("none", 0), ("people", 2), ("crowd", 4)];

/// The camera's move for a kind, in the pair's pixels (cw x ch), and its blur direction.
fn sample_motion(kind: &str, cw: f64, ch: f64, r: &mut Rng) -> (M23, f64) {
    let deg = std::f64::consts::PI / 180.0;
    let (t, dir, ang, sc) = match kind {
        "small" => (r.uni(0.0, 0.005 * cw), r.uni(0.0, 6.2832), r.sym(0.0, 0.2) * deg, 1.0 + r.sym(0.0, 0.003)),
        "large" => (r.uni(0.02, 0.05) * cw, r.uni(0.0, 6.2832), r.sym(0.0, 0.5) * deg, 1.0 + r.sym(0.0, 0.01)),
        "whip" => {
            let d = r.sym(0.0, 25.0 * deg) + if r.next() < 0.5 { 0.0 } else { std::f64::consts::PI };
            (r.uni(0.08, 0.14) * cw, d, r.sym(0.0, 1.0) * deg, 1.0 + r.sym(0.0, 0.01))
        }
        "rotate" => (r.uni(0.0, 0.01) * cw, r.uni(0.0, 6.2832), r.sym(2.0, 5.0) * deg, 1.0 + r.sym(0.0, 0.005)),
        "zoom" => (r.uni(0.0, 0.01) * cw, r.uni(0.0, 6.2832), r.sym(0.0, 0.3) * deg, 1.0 + r.sym(0.02, 0.06)),
        _ => unreachable!(),
    };
    // turn and scale about a point near the middle, then move
    let (px, py) = (cw / 2.0 + r.sym(0.0, 0.2 * cw), ch / 2.0 + r.sym(0.0, 0.2 * ch));
    let rs = fit::similarity(sc, ang, 0.0, 0.0);
    let (qx, qy) = fit::apply(&rs, px, py);
    let m = [rs[0], rs[1], px - qx + t * dir.cos(), rs[3], rs[4], py - qy + t * dir.sin()];
    (m, dir)
}

/// The source warped by m (pixels of the crop at offset o), averaged over a straight blur path of `blur` px along dir.
fn render(src: &Mat<u8>, m: &M23, o: (f64, f64), blur: f64, dir: f64) -> Mat<u8> {
    let k = if blur > 0.5 { 9 } else { 1 };
    let mut acc = vec![0u32; src.data.len()];
    for i in 0..k {
        let s = if k == 1 { 0.0 } else { (i as f64 / (k - 1) as f64 - 0.5) * blur };
        let mm = [m[0], m[1], m[2] + s * dir.cos(), m[3], m[4], m[5] + s * dir.sin()];
        // crop coords -> source coords: x_s = x_c + o
        let ms = fit::compose(&[1.0, 0.0, o.0, 0.0, 1.0, o.1], &fit::compose(&mm, &[1.0, 0.0, -o.0, 0.0, 1.0, -o.1]));
        let w = warp_affine_u8(src, &ms, src.cols, src.rows, false, Border::Constant, [0.0; 4]);
        for (a, v) in acc.iter_mut().zip(&w.data) {
            *a += *v as u32;
        }
    }
    let data = acc.iter().map(|&a| ((a + k as u32 / 2) / k as u32) as u8).collect();
    Mat::from_vec(src.rows, src.cols, 3, data)
}

fn crop(m: &Mat<u8>, o: (usize, usize), w: usize, h: usize) -> Mat<u8> {
    m.roi(o.0, o.1, w, h)
}

fn paste(dst: &mut Mat<u8>, patch: &Mat<u8>, x: i64, y: i64) {
    for j in 0..patch.rows as i64 {
        let yy = y + j;
        if yy < 0 || yy >= dst.rows as i64 {
            continue;
        }
        for i in 0..patch.cols as i64 {
            let xx = x + i;
            if xx < 0 || xx >= dst.cols as i64 {
                continue;
            }
            let s = ((j as usize) * patch.cols + i as usize) * 3;
            let d = ((yy as usize) * dst.cols + xx as usize) * 3;
            dst.data[d..d + 3].copy_from_slice(&patch.data[s..s + 3]);
        }
    }
}

fn map_px(m: &mut Mat<u8>, f: impl Fn(f64) -> f64) {
    for v in m.data.iter_mut() {
        *v = f(*v as f64).round().clamp(0.0, 255.0) as u8;
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = std::path::PathBuf::from(&args[1]);
    std::fs::create_dir_all(&out).unwrap();
    let srcs: Vec<(String, Mat<u8>)> = args[2..].iter().map(|p| (p.clone(), read_bgr(std::path::Path::new(p)))).collect();
    let mut manifest = vec![];
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    for (si, (sname, src)) in srcs.iter().enumerate() {
        let (sw, sh) = (src.cols, src.rows);
        let (cw, ch) = (((sw as f64) * 0.84).round() as usize, ((sh as f64) * 0.84).round() as usize);
        let o = ((sw - cw) / 2, (sh - ch) / 2);
        let of = (o.0 as f64, o.1 as f64);
        let donor = &srcs[(si + 1) % srcs.len()].1;
        for motion in MOTIONS {
            for degrade in DEGRADES {
                for (oname, npatch) in OUTLIERS {
                    seed = seed.wrapping_add(0x2545_F491_4F6C_DD1D);
                    let mut r = Rng(seed | 1);
                    for _ in 0..4 {
                        r.next();
                    }
                    let (m, mut dir) = sample_motion(motion, cw as f64, ch as f64, &mut r);
                    let blur = if degrade == "motion_blur" { r.uni(0.012, 0.025) * cw as f64 } else { 0.0 };
                    if motion == "small" || motion == "rotate" || motion == "zoom" {
                        dir = r.uni(0.0, 6.2832);
                    }
                    let mut a = crop(&render(src, &fit::IDENTITY, of, blur, dir), o, cw, ch);
                    let mut b = crop(&render(src, &m, of, blur, dir), o, cw, ch);
                    // people: patches of another frame, each moving on its own
                    for _ in 0..npatch {
                        let pw = (r.uni(0.15, 0.25) * cw as f64) as usize;
                        let ph = (r.uni(0.45, 0.75) * ch as f64) as usize;
                        let dx = r.uni(0.0, (donor.cols - pw) as f64) as usize;
                        let dy = r.uni(0.0, (donor.rows - ph) as f64) as usize;
                        let patch = donor.roi(dx, dy, pw, ph);
                        let px = r.uni(0.0, (cw - pw) as f64);
                        let py = r.uni(0.0, (ch - ph) as f64);
                        let vm = r.uni(0.03, 0.07) * cw as f64;
                        let vd = r.uni(0.0, 6.2832);
                        paste(&mut a, &patch, px as i64, py as i64);
                        paste(&mut b, &patch, (px + vm * vd.cos()) as i64, (py + vm * vd.sin()) as i64);
                    }
                    match degrade {
                        "defocus" => {
                            a = gaussian_blur_u8(&a, 13, 13, 2.0, 2.0);
                            b = gaussian_blur_u8(&b, 13, 13, 2.0, 2.0);
                        }
                        "noise" | "low_light" => {
                            let (gain, sd) = if degrade == "noise" { (1.0, 8.0) } else { (0.25, 3.0) };
                            for im in [&mut a, &mut b] {
                                for v in im.data.iter_mut() {
                                    *v = (*v as f64 * gain + sd * r.gauss()).round().clamp(0.0, 255.0) as u8;
                                }
                            }
                        }
                        "jpeg" => {
                            a = jpeg_roundtrip(&a, 30);
                            b = jpeg_roundtrip(&b, 30);
                        }
                        "exposure" => map_px(&mut b, |v| v * 1.4 - 20.0),
                        _ => {}
                    }
                    let id = format!("s{si}_{motion}_{degrade}_{oname}");
                    let (pa, pb) = (out.join(format!("{id}_a.png")), out.join(format!("{id}_b.png")));
                    write_bgr_png(&pa, &a);
                    write_bgr_png(&pb, &b);
                    manifest.push(serde_json::json!({"id": id, "a": pa, "b": pb, "M": m, "motion": motion, "degrade": degrade,
                        "outliers": oname, "src": sname, "W": cw, "H": ch}));
                }
            }
        }
    }
    std::fs::write(out.join("manifest.json"), serde_json::to_string_pretty(&manifest).unwrap()).unwrap();
    println!("{} pairs", manifest.len());
}
