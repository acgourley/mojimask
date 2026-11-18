//! Single-channel float images, the pyramid they're searched on, and the few filters the search needs.
//!
//! Coordinates are pixel centres: pixel (i, j) sits at (i, j). Between pyramid levels `x_l = x_(l-1) / 2` exactly
//! (`down2` keeps the even samples of a binomial-filtered image), so a transform moves between levels by scaling its
//! translation by 2. `box_down(f)` averages f x f cells: its pixel i is centred at `f * i + (f - 1) / 2` in the source.

#[derive(Clone, Debug)]
pub struct Img {
    pub w: usize,
    pub h: usize,
    pub d: Vec<f32>,
}

impl Img {
    pub fn new(w: usize, h: usize) -> Img {
        Img { w, h, d: vec![0.0; w * h] }
    }

    pub fn from_u8(w: usize, h: usize, px: &[u8]) -> Img {
        assert_eq!(px.len(), w * h);
        Img { w, h, d: px.iter().map(|&v| v as f32).collect() }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.d[y * self.w + x]
    }

    /// Bilinear sample, clamped to the border.
    #[inline]
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let x = x.clamp(0.0, (self.w - 1) as f32);
        let y = y.clamp(0.0, (self.h - 1) as f32);
        let x0 = (x as usize).min(self.w.saturating_sub(2));
        let y0 = (y as usize).min(self.h.saturating_sub(2));
        let ax = x - x0 as f32;
        let ay = y - y0 as f32;
        let x1 = (x0 + 1).min(self.w - 1);
        let y1 = (y0 + 1).min(self.h - 1);
        let r0 = &self.d[y0 * self.w..];
        let r1 = &self.d[y1 * self.w..];
        let a = r0[x0] + ax * (r0[x1] - r0[x0]);
        let b = r1[x0] + ax * (r1[x1] - r1[x0]);
        a + ay * (b - a)
    }

    /// Binomial [1 4 6 4 1]/16 blur (mirrored border), then every second pixel: the next pyramid level.
    pub fn down2(&self) -> Img {
        let (w, h) = (self.w, self.h);
        let ow = (w + 1) / 2;
        let oh = (h + 1) / 2;
        let k = [1.0f32 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
        let mir = |p: isize, n: usize| -> usize {
            let n = n as isize;
            let mut p = p;
            if n == 1 {
                return 0;
            }
            while p < 0 || p >= n {
                if p < 0 {
                    p = -p;
                }
                if p >= n {
                    p = 2 * (n - 1) - p;
                }
            }
            p as usize
        };
        // horizontal pass on every row, keeping even columns
        let mut tmp = vec![0f32; ow * h];
        for y in 0..h {
            let row = &self.d[y * w..(y + 1) * w];
            for ox in 0..ow {
                let x = (2 * ox) as isize;
                let mut s = 0.0;
                for (i, kv) in k.iter().enumerate() {
                    s += kv * row[mir(x + i as isize - 2, w)];
                }
                tmp[y * ow + ox] = s;
            }
        }
        let mut out = Img::new(ow, oh);
        for oy in 0..oh {
            let y = (2 * oy) as isize;
            let rows: [usize; 5] = std::array::from_fn(|i| mir(y + i as isize - 2, h));
            for ox in 0..ow {
                let mut s = 0.0;
                for (i, kv) in k.iter().enumerate() {
                    s += kv * tmp[rows[i] * ow + ox];
                }
                out.d[oy * ow + ox] = s;
            }
        }
        out
    }

    /// Mean of f x f cells (the remainder at the right and bottom is dropped).
    pub fn box_down(&self, f: usize) -> Img {
        if f <= 1 {
            return self.clone();
        }
        let ow = self.w / f;
        let oh = self.h / f;
        let mut out = Img::new(ow, oh);
        let inv = 1.0 / (f * f) as f32;
        let mut acc = vec![0f32; ow];
        for oy in 0..oh {
            acc.iter_mut().for_each(|a| *a = 0.0);
            for y in oy * f..(oy + 1) * f {
                let row = &self.d[y * self.w..y * self.w + ow * f];
                for (ox, a) in acc.iter_mut().enumerate() {
                    *a += row[ox * f..(ox + 1) * f].iter().sum::<f32>();
                }
            }
            for ox in 0..ow {
                out.d[oy * ow + ox] = acc[ox] * inv;
            }
        }
        out
    }

    /// Central-difference gradients (half the two-pixel difference; one-sided at the border).
    pub fn gradients(&self) -> (Img, Img) {
        let (w, h) = (self.w, self.h);
        let mut gx = Img::new(w, h);
        let mut gy = Img::new(w, h);
        for y in 0..h {
            let ym = y.saturating_sub(1);
            let yp = (y + 1).min(h - 1);
            let sy = if yp - ym == 2 { 0.5 } else { 1.0 };
            for x in 0..w {
                let xm = x.saturating_sub(1);
                let xp = (x + 1).min(w - 1);
                let sx = if xp - xm == 2 { 0.5 } else { 1.0 };
                gx.d[y * w + x] = (self.get(xp, y) - self.get(xm, y)) * sx;
                gy.d[y * w + x] = (self.get(x, yp) - self.get(x, ym)) * sy;
            }
        }
        (gx, gy)
    }
}

/// A pyramid level: the image and its gradients.
pub struct Level {
    pub img: Img,
    pub gx: Img,
    pub gy: Img,
}

impl Level {
    pub fn new(img: Img) -> Level {
        let (gx, gy) = img.gradients();
        Level { img, gx, gy }
    }

    /// Smallest eigenvalue of the structure tensor over the b x b block at (x, y) (top-left), per pixel: how well
    /// the block pins down a 2-D move (an edge or a flat patch scores low).
    pub fn texture(&self, x: usize, y: usize, b: usize) -> f32 {
        let (mut a, mut c, mut d) = (0f32, 0f32, 0f32);
        for yy in y..y + b {
            let o = yy * self.img.w;
            for xx in x..x + b {
                let gx = self.gx.d[o + xx];
                let gy = self.gy.d[o + xx];
                a += gx * gx;
                c += gx * gy;
                d += gy * gy;
            }
        }
        let n = (b * b) as f32;
        let (a, c, d) = (a / n, c / n, d / n);
        let tr = 0.5 * (a + d);
        let det = (0.25 * (a - d) * (a - d) + c * c).sqrt();
        tr - det
    }
}

/// Levels from fine (0) to coarse; each level half the last.
pub fn pyramid(base: Img, min_long: usize, max_levels: usize) -> Vec<Level> {
    let mut imgs = vec![base];
    while imgs.len() < max_levels {
        let last = imgs.last().unwrap();
        if last.w.max(last.h) / 2 < min_long || last.w.min(last.h) < 16 {
            break;
        }
        let next = last.down2();
        imgs.push(next);
    }
    imgs.into_iter().map(Level::new).collect()
}
