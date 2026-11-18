//! Preprocessing ported from mojimask/onnx_core.py, with cv2.resize reproduced for 8-bit BGR images:
//! INTER_LINEAR (OpenCV's fixed-point path incl. its NEON vertical pass: bit-exact on every size tried),
//! INTER_CUBIC (float vertical pass; off by 1 on ~5 pixels per million, FMA order) and INTER_AREA for whole-number
//! factors (the 4K -> 1280 canvas is exactly /3).

#[derive(Clone)]
pub struct Img {
    pub h: usize,
    pub w: usize,
    pub d: Vec<u8>, // BGR, row-major, 3 channels
}

impl Img {
    pub fn load_bgr(path: &std::path::Path) -> Img {
        let im = image::open(path).expect("png").to_rgb8();
        let (w, h) = (im.width() as usize, im.height() as usize);
        let mut d = im.into_raw();
        for p in d.chunks_exact_mut(3) {
            p.swap(0, 2);
        }
        Img { h, w, d }
    }
    pub fn crop(&self, y: usize, x: usize, h: usize, w: usize) -> Img {
        let (h, w) = (h.min(self.h - y), w.min(self.w - x));
        let mut d = Vec::with_capacity(h * w * 3);
        for r in y..y + h {
            d.extend_from_slice(&self.d[(r * self.w + x) * 3..(r * self.w + x + w) * 3]);
        }
        Img { h, w, d }
    }
}

fn lin_coefs(dsize: usize, ssize: usize, clamp: bool) -> (Vec<i64>, Vec<[i64; 2]>) {
    let scale = ssize as f64 / dsize as f64;
    let mut ofs = vec![0i64; dsize];
    let mut a = vec![[0i64; 2]; dsize];
    for d in 0..dsize {
        let mut fx = ((d as f64 + 0.5) * scale - 0.5) as f32;
        let mut s = fx.floor() as i64;
        fx -= s as f32;
        if clamp {
            if s < 0 {
                fx = 0.0;
                s = 0;
            }
            if s >= ssize as i64 - 1 {
                fx = 0.0;
                s = ssize as i64 - 1;
            }
        }
        let c0 = ((1.0f32 - fx) * 2048.0f32).round_ties_even() as i64;
        a[d] = [c0, 2048 - c0];
        ofs[d] = s;
    }
    (ofs, a)
}

pub fn resize_linear(src: &Img, nw: usize, nh: usize) -> Img {
    let (sx, ax) = lin_coefs(nw, src.w, true);
    let (sy, ay) = lin_coefs(nh, src.h, false);
    let hrow = |r: usize| -> Vec<i64> {
        let row = &src.d[r * src.w * 3..(r + 1) * src.w * 3];
        let mut o = vec![0i64; nw * 3];
        for x in 0..nw {
            let x0 = sx[x] as usize;
            let x1 = (x0 + 1).min(src.w - 1);
            for c in 0..3 {
                o[x * 3 + c] = row[x0 * 3 + c] as i64 * ax[x][0] + row[x1 * 3 + c] as i64 * ax[x][1];
            }
        }
        o
    };
    let mut cache: std::collections::HashMap<usize, Vec<i64>> = Default::default();
    let mut d = vec![0u8; nw * nh * 3];
    let clampr = |v: i64| v.clamp(0, src.h as i64 - 1) as usize;
    for y in 0..nh {
        let (r0, r1) = (clampr(sy[y]), clampr(sy[y] + 1));
        for r in [r0, r1] {
            cache.entry(r).or_insert_with(|| hrow(r));
        }
        let (a, b) = (&cache[&r0], &cache[&r1]);
        let (b0, b1) = (ay[y][0], ay[y][1]);
        for i in 0..nw * 3 {
            // OpenCV VResizeLinearVec_32s8u: (mulhi(r0>>4, b0) + mulhi(r1>>4, b1) + 2) >> 2
            let v = (((a[i] >> 4) * b0) >> 16) + (((b[i] >> 4) * b1) >> 16);
            d[y * nw * 3 + i] = ((v + 2) >> 2).clamp(0, 255) as u8;
        }
        if y > 2 {
            cache.retain(|&k, _| k + 2 >= r0);
        }
    }
    Img { h: nh, w: nw, d }
}

fn cubic_w(x: f32) -> [f32; 4] {
    let a = -0.75f32;
    let c0 = ((a * (x + 1.0) - 5.0 * a) * (x + 1.0) + 8.0 * a) * (x + 1.0) - 4.0 * a;
    let c1 = ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    let c2 = ((a + 2.0) * (1.0 - x) - (a + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
    [c0, c1, c2, 1.0 - c0 - c1 - c2]
}

fn cub_coefs(dsize: usize, ssize: usize) -> (Vec<i64>, Vec<[i64; 4]>) {
    let scale = ssize as f64 / dsize as f64;
    let mut ofs = vec![0i64; dsize];
    let mut a = vec![[0i64; 4]; dsize];
    for d in 0..dsize {
        let mut fx = ((d as f64 + 0.5) * scale - 0.5) as f32;
        let s = fx.floor() as i64;
        fx -= s as f32;
        let w = cubic_w(fx);
        for k in 0..4 {
            a[d][k] = (w[k] * 2048.0).round_ties_even() as i64;
        }
        ofs[d] = s;
    }
    (ofs, a)
}

pub fn resize_cubic(src: &Img, nw: usize, nh: usize) -> Img {
    let (sx, ax) = cub_coefs(nw, src.w);
    let (sy, ay) = cub_coefs(nh, src.h);
    let cl = |v: i64, n: usize| v.clamp(0, n as i64 - 1) as usize;
    // horizontal pass for every source row (ints, scale 2048)
    let mut hx = vec![0i64; src.h * nw * 3];
    for r in 0..src.h {
        let row = &src.d[r * src.w * 3..(r + 1) * src.w * 3];
        for x in 0..nw {
            for c in 0..3 {
                let mut s = 0i64;
                for k in 0..4 {
                    s += row[cl(sx[x] - 1 + k as i64, src.w) * 3 + c] as i64 * ax[x][k];
                }
                hx[(r * nw + x) * 3 + c] = s;
            }
        }
    }
    let sc = 1.0f32 / (2048.0 * 2048.0);
    let mut d = vec![0u8; nw * nh * 3];
    for y in 0..nh {
        let rows: Vec<usize> = (0..4).map(|k| cl(sy[y] - 1 + k as i64, src.h)).collect();
        let b: Vec<f32> = (0..4).map(|k| ay[y][k] as f32 * sc).collect();
        for i in 0..nw * 3 {
            // OpenCV VResizeCubicVec_32s8u: muladd(S0, b0, muladd(S1, b1, muladd(S2, b2, S3 * b3))), fused on NEON
            let mut acc = hx[rows[3] * nw * 3 + i] as f32 * b[3];
            for k in (0..3).rev() {
                acc = (hx[rows[k] * nw * 3 + i] as f32).mul_add(b[k], acc);
            }
            d[y * nw * 3 + i] = acc.round_ties_even().clamp(0.0, 255.0) as u8;
        }
    }
    Img { h: nh, w: nw, d }
}

/// INTER_AREA for whole-number downscale factors (resizeAreaFast: mean of the block, rounded).
pub fn resize_area(src: &Img, nw: usize, nh: usize) -> Img {
    let (fx, fy) = (src.w / nw, src.h / nh);
    assert!(fx * nw == src.w && fy * nh == src.h, "resize_area: only whole-number factors are ported");
    let inv = 1.0f32 / (fx * fy) as f32;
    let mut d = vec![0u8; nw * nh * 3];
    for y in 0..nh {
        for x in 0..nw {
            for c in 0..3 {
                let mut s = 0u32;
                for yy in 0..fy {
                    for xx in 0..fx {
                        s += src.d[((y * fy + yy) * src.w + x * fx + xx) * 3 + c] as u32;
                    }
                }
                d[(y * nw + x) * 3 + c] = (s as f32 * inv).round_ties_even() as u8;
            }
        }
    }
    Img { h: nh, w: nw, d }
}

pub const CANVAS: usize = 1280;

/// onnx_core.to_canvas: long side -> 1280 (cubic up, area down). Returns (canvas, k).
pub fn to_canvas(f: &Img) -> (Img, f64) {
    let k = CANVAS as f64 / f.h.max(f.w) as f64;
    if (k - 1.0).abs() < 1e-6 {
        return (f.clone(), 1.0);
    }
    let (nw, nh) = ((f.w as f64 * k).round_ties_even() as usize, (f.h as f64 * k).round_ties_even() as usize);
    (if k > 1.0 { resize_cubic(f, nw, nh) } else { resize_area(f, nw, nh) }, k)
}

pub struct Letterboxed {
    pub shape: [usize; 4],
    pub x: Vec<f32>,
    pub s: f64,
    pub pw: i64,
    pub ph: i64,
}

/// onnx_core.letterbox: NCHW RGB float /255, pad 114.
pub fn letterbox(img: &Img, size: usize, rect: bool) -> Letterboxed {
    let stride = 32usize;
    let (h, w) = (img.h as f64, img.w as f64);
    let s = (size as f64 / h).min(size as f64 / w);
    let (nh, nw, hh, ww, ph, pw);
    if rect {
        nh = (h * s).round_ties_even() as usize;
        nw = (w * s).round_ties_even() as usize;
        hh = nh + (size - nh) % stride;
        ww = nw + (size - nw) % stride;
        ph = (((hh - nh) as f64) / 2.0 - 0.1).round_ties_even() as usize;
        pw = (((ww - nw) as f64) / 2.0 - 0.1).round_ties_even() as usize;
    } else {
        nh = (h * s) as usize;
        nw = (w * s) as usize;
        hh = size;
        ww = size;
        ph = (size - nh) / 2;
        pw = (size - nw) / 2;
    }
    let r = resize_linear(img, nw, nh);
    let mut x = vec![114.0f32 / 255.0; 3 * hh * ww];
    let plane = hh * ww;
    for y in 0..nh {
        for xx in 0..nw {
            let p = &r.d[(y * nw + xx) * 3..(y * nw + xx) * 3 + 3];
            let o = (y + ph) * ww + xx + pw;
            x[o] = p[2] as f32 / 255.0; // R
            x[plane + o] = p[1] as f32 / 255.0;
            x[2 * plane + o] = p[0] as f32 / 255.0;
        }
    }
    Letterboxed { shape: [1, 3, hh, ww], x, s, pw: pw as i64, ph: ph as i64 }
}

/// onnx_core.retina_input: BGR minus (104, 117, 123), NCHW.
pub fn retina_input(img: &Img) -> Vec<f32> {
    let plane = img.h * img.w;
    let mut x = vec![0f32; 3 * plane];
    let mean = [104.0f32, 117.0, 123.0];
    for i in 0..plane {
        for c in 0..3 {
            x[c * plane + i] = img.d[i * 3 + c] as f32 - mean[c];
        }
    }
    x
}
