//! The project's frames (PROJECT/frames/NNNNN.jpg) the way the Python app writes and reads them:
//! `cv2.imwrite(path, bgr, [IMWRITE_JPEG_QUALITY, 90])` and `cv2.imread(path)`, i.e. OpenCV's bundled libjpeg-turbo
//! (3.1.2) with its defaults. Everything the scan learns from pixels (faces, bodies, depth, flow) is read from these
//! JPEGs, so the decoded pixels must be bit-identical to cv2's, and a Rust-made project's JPEGs must decode to what
//! Python's would.
//!
//! What is reproduced (libjpeg-turbo's integer paths; its NEON versions are bit-exact with them):
//! - **Writing**: baseline, 4:2:0 (Y 2x2, Cb/Cr 1x1), the standard Huffman tables (no optimisation), quantisation
//!   tables scaled for the quality as `jpeg_set_quality` (force_baseline), RGB->YCbCr with jccolor.c's 16-bit tables,
//!   `h2v2_downsample` (alternating bias 1, 2), edges replicated to whole blocks, dummy blocks at the right edge of
//!   the last MCU (DC of the block before, no AC), `jpeg_fdct_islow`, rounding division by quant * 8. Headers as
//!   libjpeg-turbo writes them (JFIF 1.01, density 1:1), so the files come out byte-identical to cv2's.
//! - **Reading**: baseline Huffman (with restart markers), `jpeg_idct_islow` with its range-limit table,
//!   `h2v2_fancy_upsample` (the triangle filter; context rows replicated at the image's top and bottom edges, the
//!   component's real width at the right), jdcolor.c's YCbCr->RGB tables, out as BGR. 4:4:4 and greyscale too.
//!   Progressive and arithmetic-coded files are refused (the app never writes them).
//!
//! Checked: native/tools/parity/detect_jpeg.py (every jazz frame: our files byte-identical to cv2.imwrite's, our
//! decode identical to cv2.imread).
use anyhow::{bail, Result};

const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
    28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61,
    54, 47, 55, 62, 63,
];

/// The JPEG spec's example tables (Annex K), in natural order.
const STD_LUM_Q: [u32; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51,
    87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
];
const STD_CHR_Q: [u32; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99,
];

const DC_LUM_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const DC_CHR_BITS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
const DC_VALS: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const AC_LUM_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];
const AC_LUM_VALS: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14,
    0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09,
    0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a,
    0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65,
    0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88,
    0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9,
    0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca,
    0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
    0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
];
const AC_CHR_BITS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
const AC_CHR_VALS: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32,
    0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16,
    0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64,
    0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86,
    0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8,
    0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
    0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
];

// jfdctint.c / jidctint.c constants (CONST_BITS 13)
const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;
const F_0_298: i32 = 2446;
const F_0_390: i32 = 3196;
const F_0_541: i32 = 4433;
const F_0_765: i32 = 6270;
const F_0_899: i32 = 7373;
const F_1_175: i32 = 9633;
const F_1_501: i32 = 12299;
const F_1_847: i32 = 15137;
const F_1_961: i32 = 16069;
const F_2_053: i32 = 16819;
const F_2_562: i32 = 20995;
const F_3_072: i32 = 25172;

#[inline]
fn descale(x: i32, n: i32) -> i32 {
    (x + (1 << (n - 1))) >> n
}

/// jpeg_set_quality's table for a quality (force_baseline), in natural order.
pub fn quant_table(std: &[u32; 64], quality: u32) -> [u16; 64] {
    let q = quality.clamp(1, 100);
    let scale = if q < 50 { 5000 / q } else { 200 - q * 2 };
    let mut t = [0u16; 64];
    for i in 0..64 {
        let v = ((std[i] * scale + 50) / 100).clamp(1, 255);
        t[i] = v as u16;
    }
    t
}

// ------------------------------------------------------------------------------------------------ encoder

struct HuffEnc {
    code: [u32; 256],
    size: [u8; 256],
}

fn huff_enc(bits: &[u8; 16], vals: &[u8]) -> HuffEnc {
    let mut h = HuffEnc { code: [0; 256], size: [0; 256] };
    let mut code = 0u32;
    let mut k = 0;
    for l in 1..=16 {
        for _ in 0..bits[l - 1] {
            h.code[vals[k] as usize] = code;
            h.size[vals[k] as usize] = l as u8;
            code += 1;
            k += 1;
        }
        code <<= 1;
    }
    h
}

struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    n: u32,
}

impl BitWriter {
    fn put(&mut self, code: u32, size: u32) {
        if size == 0 {
            return;
        }
        self.acc = (self.acc << size) | (code as u64 & ((1u64 << size) - 1));
        self.n += size;
        while self.n >= 8 {
            let b = ((self.acc >> (self.n - 8)) & 0xff) as u8;
            self.out.push(b);
            if b == 0xff {
                self.out.push(0);
            }
            self.n -= 8;
        }
    }
    fn flush(&mut self) {
        // pad with 1 bits (jchuff flush_bits)
        if self.n > 0 {
            let pad = 8 - self.n;
            self.put((1 << pad) - 1, pad);
        }
    }
}

fn fdct_islow(d: &mut [i32; 64]) {
    for r in 0..8 {
        let p = &mut d[r * 8..r * 8 + 8];
        let tmp0 = p[0] + p[7];
        let tmp7 = p[0] - p[7];
        let tmp1 = p[1] + p[6];
        let tmp6 = p[1] - p[6];
        let tmp2 = p[2] + p[5];
        let tmp5 = p[2] - p[5];
        let tmp3 = p[3] + p[4];
        let tmp4 = p[3] - p[4];
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;
        p[0] = (tmp10 + tmp11) << PASS1_BITS;
        p[4] = (tmp10 - tmp11) << PASS1_BITS;
        let z1 = (tmp12 + tmp13) * F_0_541;
        p[2] = descale(z1 + tmp13 * F_0_765, CONST_BITS - PASS1_BITS);
        p[6] = descale(z1 + tmp12 * -F_1_847, CONST_BITS - PASS1_BITS);
        let z1 = tmp4 + tmp7;
        let z2 = tmp5 + tmp6;
        let z3 = tmp4 + tmp6;
        let z4 = tmp5 + tmp7;
        let z5 = (z3 + z4) * F_1_175;
        let tmp4 = tmp4 * F_0_298;
        let tmp5 = tmp5 * F_2_053;
        let tmp6 = tmp6 * F_3_072;
        let tmp7 = tmp7 * F_1_501;
        let z1 = z1 * -F_0_899;
        let z2 = z2 * -F_2_562;
        let z3 = z3 * -F_1_961 + z5;
        let z4 = z4 * -F_0_390 + z5;
        p[7] = descale(tmp4 + z1 + z3, CONST_BITS - PASS1_BITS);
        p[5] = descale(tmp5 + z2 + z4, CONST_BITS - PASS1_BITS);
        p[3] = descale(tmp6 + z2 + z3, CONST_BITS - PASS1_BITS);
        p[1] = descale(tmp7 + z1 + z4, CONST_BITS - PASS1_BITS);
    }
    for c in 0..8 {
        let g = |i: usize| d[i * 8 + c];
        let tmp0 = g(0) + g(7);
        let tmp7 = g(0) - g(7);
        let tmp1 = g(1) + g(6);
        let tmp6 = g(1) - g(6);
        let tmp2 = g(2) + g(5);
        let tmp5 = g(2) - g(5);
        let tmp3 = g(3) + g(4);
        let tmp4 = g(3) - g(4);
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;
        let mut o = [0i32; 8];
        o[0] = descale(tmp10 + tmp11, PASS1_BITS);
        o[4] = descale(tmp10 - tmp11, PASS1_BITS);
        let z1 = (tmp12 + tmp13) * F_0_541;
        o[2] = descale(z1 + tmp13 * F_0_765, CONST_BITS + PASS1_BITS);
        o[6] = descale(z1 + tmp12 * -F_1_847, CONST_BITS + PASS1_BITS);
        let z1 = tmp4 + tmp7;
        let z2 = tmp5 + tmp6;
        let z3 = tmp4 + tmp6;
        let z4 = tmp5 + tmp7;
        let z5 = (z3 + z4) * F_1_175;
        let tmp4 = tmp4 * F_0_298;
        let tmp5 = tmp5 * F_2_053;
        let tmp6 = tmp6 * F_3_072;
        let tmp7 = tmp7 * F_1_501;
        let z1 = z1 * -F_0_899;
        let z2 = z2 * -F_2_562;
        let z3 = z3 * -F_1_961 + z5;
        let z4 = z4 * -F_0_390 + z5;
        o[7] = descale(tmp4 + z1 + z3, CONST_BITS + PASS1_BITS);
        o[5] = descale(tmp5 + z2 + z4, CONST_BITS + PASS1_BITS);
        o[3] = descale(tmp6 + z2 + z3, CONST_BITS + PASS1_BITS);
        o[1] = descale(tmp7 + z1 + z4, CONST_BITS + PASS1_BITS);
        for i in 0..8 {
            d[i * 8 + c] = o[i];
        }
    }
}

/// One component's samples, padded to whole blocks (edges replicated).
struct Plane {
    d: Vec<u8>,
}

fn encode_block(bw: &mut BitWriter, blk: &[i32; 64], last_dc: &mut i32, dc: &HuffEnc, ac: &HuffEnc) {
    let nbits = |v: i32| -> u32 { 32 - (v.unsigned_abs()).leading_zeros() };
    let diff = blk[0] - *last_dc;
    *last_dc = blk[0];
    let n = nbits(diff);
    bw.put(dc.code[n as usize], dc.size[n as usize] as u32);
    if n > 0 {
        let v = if diff < 0 { diff - 1 } else { diff };
        bw.put(v as u32, n);
    }
    let mut run = 0;
    for k in 1..64 {
        let v = blk[ZIGZAG[k]];
        if v == 0 {
            run += 1;
            continue;
        }
        while run > 15 {
            bw.put(ac.code[0xf0], ac.size[0xf0] as u32);
            run -= 16;
        }
        let n = nbits(v);
        let s = (run << 4) as usize + n as usize;
        bw.put(ac.code[s], ac.size[s] as u32);
        bw.put(if v < 0 { (v - 1) as u32 } else { v as u32 }, n);
        run = 0;
    }
    if run > 0 {
        bw.put(ac.code[0], ac.size[0] as u32);
    }
}

fn quantise(samples: &[u8], stride: usize, x0: usize, y0: usize, q: &[u16; 64]) -> [i32; 64] {
    let mut d = [0i32; 64];
    for r in 0..8 {
        for c in 0..8 {
            d[r * 8 + c] = samples[(y0 + r) * stride + x0 + c] as i32 - 128;
        }
    }
    fdct_islow(&mut d);
    for i in 0..64 {
        let qv = (q[i] as i32) << 3;
        let t = d[i];
        d[i] = if t < 0 { -((-t + (qv >> 1)) / qv) } else { (t + (qv >> 1)) / qv };
    }
    d
}

/// `cv2.imencode(".jpg", bgr, [IMWRITE_JPEG_QUALITY, quality])`: rows of BGR u8, w x h.
pub fn encode_bgr(bgr: &[u8], w: usize, h: usize, quality: u32) -> Vec<u8> {
    assert_eq!(bgr.len(), w * h * 3);
    // MCU 16x16: Y blocks per row = width_in_blocks (ceil(w/8)), chroma width ceil(w/2)
    let mcux = w.div_ceil(16);
    let mcuy = h.div_ceil(16);
    let (yw_blocks, yh_blocks) = (w.div_ceil(8), h.div_ceil(8));
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let (cw_blocks, ch_blocks) = (cw.div_ceil(8), ch.div_ceil(8));
    // colour conversion (jccolor.c rgb_ycc_convert: 16-bit tables)
    const SC: i32 = 16;
    let fix = |x: f64| (x * (1i64 << SC) as f64 + 0.5) as i32;
    let half = 1i32 << (SC - 1);
    let cbcr_off = 128i32 << SC;
    let mut yp = vec![0u8; w * h];
    let mut cbp = vec![0u8; w * h];
    let mut crp = vec![0u8; w * h];
    for i in 0..w * h {
        let (b, g, r) = (bgr[i * 3] as i32, bgr[i * 3 + 1] as i32, bgr[i * 3 + 2] as i32);
        yp[i] = ((fix(0.29900) * r + fix(0.58700) * g + fix(0.11400) * b + half) >> SC) as u8;
        cbp[i] = ((-fix(0.16874) * r - fix(0.33126) * g + fix(0.5) * b + cbcr_off + half - 1) >> SC) as u8;
        crp[i] = ((fix(0.5) * r - fix(0.41869) * g - fix(0.08131) * b + cbcr_off + half - 1) >> SC) as u8;
    }
    // Y padded to whole blocks: right edge to width_in_blocks*8, bottom to the MCU rows (both replicated)
    let ypw = yw_blocks * 8;
    let yph = mcuy * 16;
    let mut yplane = Plane { d: vec![0; ypw * yph] };
    for r in 0..yph {
        let sr = r.min(h - 1);
        for c in 0..ypw {
            yplane.d[r * ypw + c] = yp[sr * w + c.min(w - 1)];
        }
    }
    // chroma: h2v2_downsample over rows padded to even height and columns replicated to 2 * output width
    let cpw = cw_blocks * 8;
    let cph = mcuy * 8;
    let mut planes = vec![];
    for src in [&cbp, &crp] {
        let mut p = Plane { d: vec![0; cpw * cph] };
        let rows = ch; // real downsampled rows
        for r in 0..rows {
            let (r0, r1) = ((2 * r).min(h - 1), (2 * r + 1).min(h - 1));
            let mut bias = 1;
            for c in 0..cpw {
                let (c0, c1) = ((2 * c).min(w - 1), (2 * c + 1).min(w - 1));
                let s = src[r0 * w + c0] as i32 + src[r0 * w + c1] as i32 + src[r1 * w + c0] as i32 + src[r1 * w + c1] as i32;
                p.d[r * cpw + c] = ((s + bias) >> 2) as u8;
                bias ^= 3; // 1, 2, 1, 2 ...
            }
        }
        for r in rows..cph {
            let (a, b) = p.d.split_at_mut(r * cpw);
            b[..cpw].copy_from_slice(&a[(rows - 1) * cpw..rows * cpw]);
        }
        planes.push(p);
    }
    let ql = quant_table(&STD_LUM_Q, quality);
    let qc = quant_table(&STD_CHR_Q, quality);
    let (dcl, acl) = (huff_enc(&DC_LUM_BITS, &DC_VALS), huff_enc(&AC_LUM_BITS, &AC_LUM_VALS));
    let (dcc, acc) = (huff_enc(&DC_CHR_BITS, &DC_VALS), huff_enc(&AC_CHR_BITS, &AC_CHR_VALS));
    let mut bw = BitWriter { out: Vec::with_capacity(w * h / 2), acc: 0, n: 0 };
    let mut last = [0i32; 3];
    for my in 0..mcuy {
        for mx in 0..mcux {
            // dummy blocks (jccoefct compress_data): past the right edge, the DC of the block before in the row;
            // a whole row past the bottom, the DC of the MCU's block before that row; no AC
            let mut prev_dc = 0;
            for by in 0..2 {
                let yb = my * 2 + by;
                let row_dc = prev_dc;
                for bx in 0..2 {
                    let xb = mx * 2 + bx;
                    let blk = if yb >= yh_blocks {
                        let mut d = [0i32; 64];
                        d[0] = row_dc;
                        d
                    } else if xb < yw_blocks {
                        quantise(&yplane.d, ypw, xb * 8, yb * 8, &ql)
                    } else {
                        let mut d = [0i32; 64];
                        d[0] = prev_dc;
                        d
                    };
                    prev_dc = blk[0];
                    encode_block(&mut bw, &blk, &mut last[0], &dcl, &acl);
                }
            }
            for (ci, p) in planes.iter().enumerate() {
                let blk = if mx < cw_blocks && my < ch_blocks.max(mcuy) {
                    quantise(&p.d, cpw, mx * 8, my * 8, &qc)
                } else {
                    [0i32; 64]
                };
                encode_block(&mut bw, &blk, &mut last[ci + 1], &dcc, &acc);
            }
        }
    }
    bw.flush();
    let mut o = Vec::with_capacity(bw.out.len() + 700);
    o.extend([0xff, 0xd8]);
    o.extend([0xff, 0xe0, 0, 16, b'J', b'F', b'I', b'F', 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]);
    for (id, q) in [(0u8, &ql), (1, &qc)] {
        o.extend([0xff, 0xdb, 0, 67, id]);
        for k in 0..64 {
            o.push(q[ZIGZAG[k]] as u8);
        }
    }
    o.extend([0xff, 0xc0, 0, 17, 8, (h >> 8) as u8, h as u8, (w >> 8) as u8, w as u8, 3, 1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1]);
    for (tc, bits, vals) in [(0x00u8, &DC_LUM_BITS, &DC_VALS[..]), (0x10, &AC_LUM_BITS, &AC_LUM_VALS[..]), (0x01, &DC_CHR_BITS, &DC_VALS[..]), (0x11, &AC_CHR_BITS, &AC_CHR_VALS[..])] {
        let len = 2 + 1 + 16 + vals.len();
        o.extend([0xff, 0xc4, (len >> 8) as u8, len as u8, tc]);
        o.extend(bits.iter());
        o.extend(vals.iter());
    }
    o.extend([0xff, 0xda, 0, 12, 3, 1, 0x00, 2, 0x11, 3, 0x11, 0, 63, 0]);
    o.extend(bw.out);
    o.extend([0xff, 0xd9]);
    o
}

// ------------------------------------------------------------------------------------------------ decoder

struct HuffDec {
    /// (length, code) -> value, via canonical decoding tables
    maxcode: [i32; 18],
    valptr: [i32; 17],
    mincode: [i32; 17],
    vals: Vec<u8>,
}

fn huff_dec(bits: &[u8; 16], vals: &[u8]) -> HuffDec {
    let mut h = HuffDec { maxcode: [-1; 18], valptr: [0; 17], mincode: [0; 17], vals: vals.to_vec() };
    let mut code = 0i32;
    let mut k = 0i32;
    for l in 1..=16 {
        let n = bits[l - 1] as i32;
        if n > 0 {
            h.valptr[l] = k;
            h.mincode[l] = code;
            code += n;
            k += n;
            h.maxcode[l] = code - 1;
        } else {
            h.maxcode[l] = -1;
        }
        code <<= 1;
    }
    h.maxcode[17] = i32::MAX;
    h
}

struct BitReader<'a> {
    d: &'a [u8],
    pos: usize,
    acc: u64,
    n: u32,
    marker: bool,
}

impl<'a> BitReader<'a> {
    fn fill(&mut self) {
        while self.n <= 56 {
            let mut b = 0u8;
            if !self.marker && self.pos < self.d.len() {
                b = self.d[self.pos];
                if b == 0xff {
                    let nx = self.d.get(self.pos + 1).copied().unwrap_or(0);
                    if nx == 0 {
                        self.pos += 2;
                    } else {
                        self.marker = true; // a marker: feed zeros (libjpeg does the same)
                        b = 0;
                    }
                } else {
                    self.pos += 1;
                }
            }
            self.acc |= (b as u64) << (56 - self.n);
            self.n += 8;
        }
    }
    fn bits(&mut self, k: u32) -> i32 {
        if k == 0 {
            return 0;
        }
        if self.n < k {
            self.fill();
        }
        let v = (self.acc >> (64 - k)) as i32;
        self.acc <<= k;
        self.n -= k;
        v
    }
    fn bit(&mut self) -> i32 {
        self.bits(1)
    }
    fn decode(&mut self, h: &HuffDec) -> Result<u8> {
        let mut code = self.bit();
        let mut l = 1;
        while code > h.maxcode[l] {
            code = (code << 1) | self.bit();
            l += 1;
            if l > 16 {
                bail!("jpeg: bad Huffman code");
            }
        }
        Ok(h.vals[(h.valptr[l] + code - h.mincode[l]) as usize])
    }
    fn restart(&mut self) -> Result<()> {
        // drop the bits left, then expect RSTn
        self.acc = 0;
        self.n = 0;
        self.marker = false;
        while self.pos + 1 < self.d.len() && !(self.d[self.pos] == 0xff && (0xd0..=0xd7).contains(&self.d[self.pos + 1])) {
            self.pos += 1;
        }
        if self.pos + 1 >= self.d.len() {
            bail!("jpeg: restart marker missing");
        }
        self.pos += 2;
        Ok(())
    }
}

fn extend(v: i32, n: u32) -> i32 {
    if n == 0 {
        0
    } else if v < (1 << (n - 1)) {
        v - (1 << n) + 1
    } else {
        v
    }
}

/// jidctint.c's range limit after the IDCT: (x & 1023) through the sample_range_limit table (+ CENTERJSAMPLE).
fn range_limit(x: i32) -> u8 {
    let y = (x & 1023) as usize;
    if y < 128 {
        (y + 128) as u8
    } else if y < 512 {
        255
    } else if y < 896 {
        0
    } else {
        (y - 896) as u8
    }
}

fn idct_islow(coef: &[i32; 64], q: &[u16; 64], out: &mut [u8], stride: usize) {
    let mut ws = [0i32; 64];
    for c in 0..8 {
        let deq = |r: usize| coef[r * 8 + c] * q[r * 8 + c] as i32;
        if (1..8).all(|r| coef[r * 8 + c] == 0) {
            let dc = deq(0) << PASS1_BITS;
            for r in 0..8 {
                ws[r * 8 + c] = dc;
            }
            continue;
        }
        let z2 = deq(2);
        let z3 = deq(6);
        let z1 = (z2 + z3) * F_0_541;
        let tmp2 = z1 + z3 * -F_1_847;
        let tmp3 = z1 + z2 * F_0_765;
        let z2 = deq(0);
        let z3 = deq(4);
        let tmp0 = (z2 + z3) << CONST_BITS;
        let tmp1 = (z2 - z3) << CONST_BITS;
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;
        let (t0, t1, t2, t3) = (deq(7), deq(5), deq(3), deq(1));
        let z1 = t0 + t3;
        let z2 = t1 + t2;
        let z3 = t0 + t2;
        let z4 = t1 + t3;
        let z5 = (z3 + z4) * F_1_175;
        let t0 = t0 * F_0_298;
        let t1 = t1 * F_2_053;
        let t2 = t2 * F_3_072;
        let t3 = t3 * F_1_501;
        let z1 = z1 * -F_0_899;
        let z2 = z2 * -F_2_562;
        let z3 = z3 * -F_1_961 + z5;
        let z4 = z4 * -F_0_390 + z5;
        let t0 = t0 + z1 + z3;
        let t1 = t1 + z2 + z4;
        let t2 = t2 + z2 + z3;
        let t3 = t3 + z1 + z4;
        let n = CONST_BITS - PASS1_BITS;
        ws[c] = descale(tmp10 + t3, n);
        ws[7 * 8 + c] = descale(tmp10 - t3, n);
        ws[8 + c] = descale(tmp11 + t2, n);
        ws[6 * 8 + c] = descale(tmp11 - t2, n);
        ws[2 * 8 + c] = descale(tmp12 + t1, n);
        ws[5 * 8 + c] = descale(tmp12 - t1, n);
        ws[3 * 8 + c] = descale(tmp13 + t0, n);
        ws[4 * 8 + c] = descale(tmp13 - t0, n);
    }
    for r in 0..8 {
        let w = &ws[r * 8..r * 8 + 8];
        let o = &mut out[r * stride..r * stride + 8];
        if w[1..].iter().all(|&v| v == 0) {
            let v = range_limit(descale(w[0], PASS1_BITS + 3));
            o.fill(v);
            continue;
        }
        let z2 = w[2];
        let z3 = w[6];
        let z1 = (z2 + z3) * F_0_541;
        let tmp2 = z1 + z3 * -F_1_847;
        let tmp3 = z1 + z2 * F_0_765;
        let tmp0 = (w[0] + w[4]) << CONST_BITS;
        let tmp1 = (w[0] - w[4]) << CONST_BITS;
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;
        let (t0, t1, t2, t3) = (w[7], w[5], w[3], w[1]);
        let z1 = t0 + t3;
        let z2 = t1 + t2;
        let z3 = t0 + t2;
        let z4 = t1 + t3;
        let z5 = (z3 + z4) * F_1_175;
        let t0 = t0 * F_0_298;
        let t1 = t1 * F_2_053;
        let t2 = t2 * F_3_072;
        let t3 = t3 * F_1_501;
        let z1 = z1 * -F_0_899;
        let z2 = z2 * -F_2_562;
        let z3 = z3 * -F_1_961 + z5;
        let z4 = z4 * -F_0_390 + z5;
        let t0 = t0 + z1 + z3;
        let t1 = t1 + z2 + z4;
        let t2 = t2 + z2 + z3;
        let t3 = t3 + z1 + z4;
        let n = CONST_BITS + PASS1_BITS + 3;
        o[0] = range_limit(descale(tmp10 + t3, n));
        o[7] = range_limit(descale(tmp10 - t3, n));
        o[1] = range_limit(descale(tmp11 + t2, n));
        o[6] = range_limit(descale(tmp11 - t2, n));
        o[2] = range_limit(descale(tmp12 + t1, n));
        o[5] = range_limit(descale(tmp12 - t1, n));
        o[3] = range_limit(descale(tmp13 + t0, n));
        o[4] = range_limit(descale(tmp13 - t0, n));
    }
}

/// h2v2_fancy_upsample over a whole component (real size cw x ch, stride cs) -> 2cw x 2ch.
fn fancy_h2v2(p: &[u8], cs: usize, cw: usize, ch: usize) -> Vec<u8> {
    let (ow, oh) = (cw * 2, ch * 2);
    let mut out = vec![0u8; ow * oh];
    let mut sums = vec![0i32; cw];
    for r in 0..ch {
        for v in 0..2 {
            let other = if v == 0 { r.saturating_sub(1) } else { (r + 1).min(ch - 1) };
            let (a, b) = (&p[r * cs..r * cs + cw], &p[other * cs..other * cs + cw]);
            for c in 0..cw {
                sums[c] = a[c] as i32 * 3 + b[c] as i32;
            }
            let o = &mut out[(2 * r + v) * ow..(2 * r + v + 1) * ow];
            if cw == 1 {
                o[0] = ((sums[0] * 4 + 8) >> 4) as u8;
                o[1] = ((sums[0] * 4 + 7) >> 4) as u8;
                continue;
            }
            o[0] = ((sums[0] * 4 + 8) >> 4) as u8;
            o[1] = ((sums[0] * 3 + sums[1] + 7) >> 4) as u8;
            for c in 1..cw - 1 {
                o[2 * c] = ((sums[c] * 3 + sums[c - 1] + 8) >> 4) as u8;
                o[2 * c + 1] = ((sums[c] * 3 + sums[c + 1] + 7) >> 4) as u8;
            }
            let c = cw - 1;
            o[2 * c] = ((sums[c] * 3 + sums[c - 1] + 8) >> 4) as u8;
            o[2 * c + 1] = ((sums[c] * 4 + 7) >> 4) as u8;
        }
    }
    out
}

/// A decoded picture: BGR rows (or one grey channel).
pub struct Decoded {
    pub w: usize,
    pub h: usize,
    pub channels: usize,
    pub data: Vec<u8>,
}

struct Comp {
    id: u8,
    hs: usize,
    vs: usize,
    tq: usize,
    td: usize,
    ta: usize,
}

/// `cv2.imread(path)` (IMREAD_COLOR): BGR rows. A greyscale file comes out as BGR with equal channels, as cv2 does.
pub fn decode_bgr(d: &[u8]) -> Result<Decoded> {
    let g = decode(d)?;
    if g.channels == 3 {
        return Ok(g);
    }
    let mut data = Vec::with_capacity(g.w * g.h * 3);
    for &v in &g.data {
        data.extend([v, v, v]);
    }
    Ok(Decoded { w: g.w, h: g.h, channels: 3, data })
}

pub fn read_bgr(path: &std::path::Path) -> Result<Decoded> {
    let d = std::fs::read(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    decode_bgr(&d)
}

fn decode(d: &[u8]) -> Result<Decoded> {
    if d.len() < 4 || d[0] != 0xff || d[1] != 0xd8 {
        bail!("jpeg: not a JPEG");
    }
    let mut qt = [[0u16; 64]; 4];
    let mut dc: Vec<Option<HuffDec>> = (0..4).map(|_| None).collect();
    let mut ac: Vec<Option<HuffDec>> = (0..4).map(|_| None).collect();
    let mut comps: Vec<Comp> = vec![];
    let (mut w, mut h) = (0usize, 0usize);
    let mut restart = 0usize;
    let mut i = 2;
    loop {
        if i + 4 > d.len() {
            bail!("jpeg: truncated");
        }
        if d[i] != 0xff {
            i += 1;
            continue;
        }
        let m = d[i + 1];
        if m == 0xff {
            i += 1;
            continue;
        }
        let len = ((d[i + 2] as usize) << 8) | d[i + 3] as usize;
        let seg = &d[i + 4..(i + 2 + len).min(d.len())];
        match m {
            0xdb => {
                let mut k = 0;
                while k < seg.len() {
                    let (pq, tq) = (seg[k] >> 4, (seg[k] & 15) as usize);
                    k += 1;
                    for z in 0..64 {
                        let v = if pq == 0 {
                            k += 1;
                            seg[k - 1] as u16
                        } else {
                            k += 2;
                            ((seg[k - 2] as u16) << 8) | seg[k - 1] as u16
                        };
                        qt[tq][ZIGZAG[z]] = v;
                    }
                }
            }
            0xc4 => {
                let mut k = 0;
                while k < seg.len() {
                    let (tc, th) = (seg[k] >> 4, (seg[k] & 15) as usize);
                    let mut bits = [0u8; 16];
                    bits.copy_from_slice(&seg[k + 1..k + 17]);
                    let n: usize = bits.iter().map(|&b| b as usize).sum();
                    let vals = &seg[k + 17..k + 17 + n];
                    let t = huff_dec(&bits, vals);
                    if tc == 0 {
                        dc[th] = Some(t);
                    } else {
                        ac[th] = Some(t);
                    }
                    k += 17 + n;
                }
            }
            0xc0 | 0xc1 => {
                if seg[0] != 8 {
                    bail!("jpeg: {}-bit samples", seg[0]);
                }
                h = ((seg[1] as usize) << 8) | seg[2] as usize;
                w = ((seg[3] as usize) << 8) | seg[4] as usize;
                let n = seg[5] as usize;
                for c in 0..n {
                    let s = &seg[6 + c * 3..9 + c * 3];
                    comps.push(Comp { id: s[0], hs: (s[1] >> 4) as usize, vs: (s[1] & 15) as usize, tq: s[2] as usize, td: 0, ta: 0 });
                }
            }
            0xc2 | 0xc3 | 0xc5..=0xc7 | 0xc9..=0xcb | 0xcd..=0xcf => bail!("jpeg: only baseline Huffman files are read"),
            0xdd => restart = ((seg[0] as usize) << 8) | seg[1] as usize,
            0xda => {
                let n = seg[0] as usize;
                for k in 0..n {
                    let (cid, t) = (seg[1 + k * 2], seg[2 + k * 2]);
                    let c = comps.iter_mut().find(|c| c.id == cid).ok_or_else(|| anyhow::anyhow!("jpeg: scan component"))?;
                    c.td = (t >> 4) as usize;
                    c.ta = (t & 15) as usize;
                }
                if n != comps.len() {
                    bail!("jpeg: non-interleaved scans are not read");
                }
                return finish(&d[i + 2 + len..], w, h, &comps, &qt, &dc, &ac, restart);
            }
            0xd9 => bail!("jpeg: no scan"),
            _ => {}
        }
        i += 2 + len;
    }
}

#[allow(clippy::too_many_arguments)]
fn finish(data: &[u8], w: usize, h: usize, comps: &[Comp], qt: &[[u16; 64]; 4], dc: &[Option<HuffDec>], ac: &[Option<HuffDec>], restart: usize) -> Result<Decoded> {
    let hmax = comps.iter().map(|c| c.hs).max().unwrap();
    let vmax = comps.iter().map(|c| c.vs).max().unwrap();
    let mcux = w.div_ceil(8 * hmax);
    let mcuy = h.div_ceil(8 * vmax);
    let n = comps.len();
    if n == 1 {
        // a single component is coded block by block, not in MCUs of the sampling factors
    }
    // per component: sample plane of mcux*hs*8 x mcuy*vs*8
    let mut planes: Vec<(usize, usize, Vec<u8>)> = comps.iter().map(|c| {
        let (pw, ph) = (mcux * c.hs * 8, mcuy * c.vs * 8);
        (pw, ph, vec![0u8; pw * ph])
    }).collect();
    let mut br = BitReader { d: data, pos: 0, acc: 0, n: 0, marker: false };
    let mut pred = vec![0i32; n];
    let mut coef = [0i32; 64];
    let decode_block = |br: &mut BitReader, c: &Comp, pred: &mut i32, coef: &mut [i32; 64]| -> Result<()> {
        coef.fill(0);
        let dct = dc[c.td].as_ref().ok_or_else(|| anyhow::anyhow!("jpeg: DC table"))?;
        let act = ac[c.ta].as_ref().ok_or_else(|| anyhow::anyhow!("jpeg: AC table"))?;
        let s = br.decode(dct)? as u32;
        let diff = extend(br.bits(s), s);
        *pred += diff;
        coef[0] = *pred;
        let mut k = 1;
        while k < 64 {
            let rs = br.decode(act)?;
            let (r, s) = ((rs >> 4) as usize, (rs & 15) as u32);
            if s == 0 {
                if r == 15 {
                    k += 16;
                    continue;
                }
                break;
            }
            k += r;
            if k > 63 {
                bail!("jpeg: bad AC run");
            }
            coef[ZIGZAG[k]] = extend(br.bits(s), s);
            k += 1;
        }
        Ok(())
    };
    if n == 1 {
        let c = &comps[0];
        let (bw, bh) = (w.div_ceil(8), h.div_ceil(8));
        let (pw, _, ref mut p) = planes[0];
        let mut count = 0;
        for by in 0..bh {
            for bx in 0..bw {
                if restart > 0 && count > 0 && count % restart == 0 {
                    br.restart()?;
                    pred[0] = 0;
                }
                count += 1;
                decode_block(&mut br, c, &mut pred[0], &mut coef)?;
                idct_islow(&coef, &qt[c.tq], &mut p[by * 8 * pw + bx * 8..], pw);
            }
        }
    } else {
        let mut count = 0;
        for my in 0..mcuy {
            for mx in 0..mcux {
                if restart > 0 && count > 0 && count % restart == 0 {
                    br.restart()?;
                    pred.fill(0);
                }
                count += 1;
                for (ci, c) in comps.iter().enumerate() {
                    for by in 0..c.vs {
                        for bx in 0..c.hs {
                            decode_block(&mut br, c, &mut pred[ci], &mut coef)?;
                            let (pw, _, ref mut p) = planes[ci];
                            let (x, y) = ((mx * c.hs + bx) * 8, (my * c.vs + by) * 8);
                            idct_islow(&coef, &qt[c.tq], &mut p[y * pw + x..], pw);
                        }
                    }
                }
            }
        }
    }
    if n == 1 {
        let (pw, _, p) = &planes[0];
        let mut out = vec![0u8; w * h];
        for r in 0..h {
            out[r * w..(r + 1) * w].copy_from_slice(&p[r * pw..r * pw + w]);
        }
        return Ok(Decoded { w, h, channels: 1, data: out });
    }
    if n != 3 {
        bail!("jpeg: {n} components");
    }
    // bring chroma to full size
    let mut full: Vec<(usize, Vec<u8>)> = vec![];
    for (ci, c) in comps.iter().enumerate() {
        let (pw, _, p) = &planes[ci];
        if c.hs == hmax && c.vs == vmax {
            full.push((*pw, p.clone()));
        } else if hmax == 2 && vmax == 2 && c.hs == 1 && c.vs == 1 {
            // the component's real size (downsampled_width/height)
            let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
            full.push((cw * 2, fancy_h2v2(p, *pw, cw, ch)));
        } else {
            bail!("jpeg: sampling {}x{} of {}x{} isn't read", c.hs, c.vs, hmax, vmax);
        }
    }
    // jdcolor.c ycc_rgb_convert, out as BGR
    const SC: i32 = 16;
    let fix = |x: f64| (x * (1i64 << SC) as f64 + 0.5) as i32;
    let half = 1i32 << (SC - 1);
    let mut cr_r = [0i32; 256];
    let mut cb_b = [0i32; 256];
    let mut cr_g = [0i32; 256];
    let mut cb_g = [0i32; 256];
    for i in 0..256 {
        let x = i as i32 - 128;
        cr_r[i] = (fix(1.40200) * x + half) >> SC;
        cb_b[i] = (fix(1.77200) * x + half) >> SC;
        cr_g[i] = -fix(0.71414) * x;
        cb_g[i] = -fix(0.34414) * x + half;
    }
    let lim = |v: i32| v.clamp(0, 255) as u8;
    let mut out = vec![0u8; w * h * 3];
    for r in 0..h {
        for c in 0..w {
            let y = full[0].1[r * full[0].0 + c] as i32;
            let cb = full[1].1[r * full[1].0 + c] as usize;
            let cr = full[2].1[r * full[2].0 + c] as usize;
            let o = (r * w + c) * 3;
            out[o + 2] = lim(y + cr_r[cr]);
            out[o + 1] = lim(y + ((cb_g[cb] + cr_g[cr]) >> SC));
            out[o] = lim(y + cb_b[cb]);
        }
    }
    Ok(Decoded { w, h, channels: 3, data: out })
}
