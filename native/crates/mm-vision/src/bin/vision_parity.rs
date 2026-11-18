//! Parity check and benchmark for mm-vision against cv2 (native/tools/parity/vision_check.py drives it).
//!
//!   vision_parity check <dump dir> [name filter]   -> one JSON line per case, then a summary line
//!   vision_parity bench <dump dir>                 -> JSON with ms per function on the 4K inputs
use mm_vision::npy::{self, Arr};
use mm_vision::*;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn mat_u8(a: &Arr) -> Mat<u8> {
    let (h, w, c) = a.hwc();
    match a {
        Arr::U8(_, d) => Mat::from_vec(h, w, c, d.clone()),
        _ => panic!("want u8"),
    }
}
fn mat_f32(a: &Arr) -> Mat<f32> {
    let (h, w, c) = a.hwc();
    match a {
        Arr::F32(_, d) => Mat::from_vec(h, w, c, d.clone()),
        _ => panic!("want f32"),
    }
}
fn mat_f64(a: &Arr) -> Mat<f64> {
    let (h, w, c) = a.hwc();
    match a {
        Arr::F64(_, d) => Mat::from_vec(h, w, c, d.clone()),
        _ => panic!("want f64"),
    }
}
fn arr_f64(m: Mat<f64>) -> Arr {
    Arr::F64(shape(&m.rows, &m.cols, m.cn), m.data)
}
fn arr_u8(m: Mat<u8>) -> Arr {
    Arr::U8(shape(&m.rows, &m.cols, m.cn), m.data)
}
fn arr_f32(m: Mat<f32>) -> Arr {
    Arr::F32(shape(&m.rows, &m.cols, m.cn), m.data)
}
fn shape(r: &usize, c: &usize, cn: usize) -> Vec<usize> {
    if cn == 1 {
        vec![*r, *c]
    } else {
        vec![*r, *c, cn]
    }
}
fn f(v: &Value, k: &str) -> f64 {
    v[k].as_f64().unwrap_or_else(|| panic!("arg {k}"))
}
fn u(v: &Value, k: &str) -> usize {
    v[k].as_u64().unwrap_or_else(|| panic!("arg {k}")) as usize
}
fn border(s: &str) -> Border {
    match s {
        "constant" => Border::Constant,
        "replicate" => Border::Replicate,
        _ => panic!("border {s}"),
    }
}
fn interp(s: &str) -> Interp {
    match s {
        "nearest" => Interp::Nearest,
        "linear" => Interp::Linear,
        "cubic" => Interp::Cubic,
        "area" => Interp::Area,
        _ => panic!("interp {s}"),
    }
}

/// Runs one case; None if the function isn't ported.
fn run(fname: &str, a: &Value, ins: &[Arr]) -> Option<Arr> {
    let x = &ins[0];
    Some(match fname {
        "cvt_bgr2gray" => arr_u8(bgr2gray(&mat_u8(x))),
        "cvt_swap_rb" => match x {
            Arr::U8(..) => arr_u8(swap_rb(&mat_u8(x))),
            _ => arr_f32(swap_rb(&mat_f32(x))),
        },
        "resize" => {
            let (dw, dh, fx, fy, it) = (u(a, "dw"), u(a, "dh"), f(a, "fx"), f(a, "fy"), interp(a["interp"].as_str().unwrap()));
            match x {
                Arr::U8(..) => arr_u8(resize_u8(&mat_u8(x), dw, dh, fx, fy, it)),
                Arr::F64(..) => arr_f64(resize_f64(&mat_f64(x), dw, dh, fx, fy, it)),
                _ => arr_f32(resize_f32(&mat_f32(x), dw, dh, fx, fy, it)),
            }
        }
        "warp_affine" => {
            let m: Vec<f64> = a["m"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            let m: [f64; 6] = m.try_into().unwrap();
            let (w, h, inv, b, bv) = (u(a, "w"), u(a, "h"), a["inverse"].as_bool().unwrap(), border(a["border"].as_str().unwrap()), f(a, "bval"));
            match x {
                Arr::U8(..) => arr_u8(warp_affine_u8(&mat_u8(x), &m, w, h, inv, b, [bv; 4])),
                _ => arr_f32(warp_affine_f32(&mat_f32(x), &m, w, h, inv, b, [bv; 4])),
            }
        }
        "gaussian_blur" => {
            let (kw, kh, sx, sy) = (u(a, "kw"), u(a, "kh"), f(a, "sx"), f(a, "sy"));
            match x {
                Arr::U8(..) => arr_u8(gaussian_blur_u8(&mat_u8(x), kw, kh, sx, sy)),
                _ => arr_f32(gaussian_blur_f32(&mat_f32(x), kw, kh, sx, sy)),
            }
        }
        "erode" => {
            let (kw, kh) = (u(a, "kw"), u(a, "kh"));
            match x {
                Arr::U8(..) => arr_u8(erode(&mat_u8(x), kw, kh)),
                Arr::F64(..) => arr_f64(erode(&mat_f64(x), kw, kh)),
                _ => arr_f32(erode(&mat_f32(x), kw, kh)),
            }
        }
        "copy_make_border" => {
            let (t, bo, l, r, b, v) = (u(a, "top"), u(a, "bottom"), u(a, "left"), u(a, "right"), border(a["border"].as_str().unwrap()), f(a, "value"));
            match x {
                Arr::U8(..) => arr_u8(copy_make_border(&mat_u8(x), t, bo, l, r, b, &[v as u8; 4])),
                _ => arr_f32(copy_make_border(&mat_f32(x), t, bo, l, r, b, &[v as f32; 4])),
            }
        }
        "point_polygon_test" => {
            let poly: Vec<[f32; 2]> = match x {
                Arr::F32(_, d) => d.chunks_exact(2).map(|p| [p[0], p[1]]).collect(),
                _ => panic!(),
            };
            let pts: Vec<f64> = match &ins[1] {
                Arr::F64(_, d) => d.clone(),
                _ => panic!(),
            };
            let r: Vec<f64> = pts.chunks_exact(2).map(|p| point_polygon_test(&poly, [p[0] as f32, p[1] as f32])).collect();
            Arr::F64(vec![r.len()], r)
        }
        "invert_affine" => match x {
            Arr::F64(s, d) => Arr::F64(s.clone(), invert_affine(&d[..].try_into().unwrap()).to_vec()),
            Arr::F32(s, d) => Arr::F32(s.clone(), invert_affine_f32(&d[..].try_into().unwrap()).to_vec()),
            _ => panic!(),
        },
        "transform" => {
            let m: Vec<f32> = a["m"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
            arr_f32(transform3(&mat_f32(x), &m, u(a, "cols")))
        }
        "filter2d" => {
            let k = mat_f32(&ins[1]);
            arr_f32(filter2d_f32(&mat_f32(x), &k))
        }
        _ => return None,
    })
}

/// (n differing elements, max abs diff, n elements); NaN == NaN counts as equal.
fn compare(a: &Arr, b: &Arr) -> Result<(usize, f64, usize), String> {
    if a.shape() != b.shape() {
        return Err(format!("shape {:?} vs cv2 {:?}", a.shape(), b.shape()));
    }
    let mut nd = 0;
    let mut mx = 0f64;
    let n;
    match (a, b) {
        (Arr::U8(_, x), Arr::U8(_, y)) => {
            n = x.len();
            for (p, q) in x.iter().zip(y) {
                if p != q {
                    nd += 1;
                    mx = mx.max((*p as f64 - *q as f64).abs());
                }
            }
        }
        (Arr::F32(_, x), Arr::F32(_, y)) => {
            n = x.len();
            for (p, q) in x.iter().zip(y) {
                if p.to_bits() != q.to_bits() && !(p.is_nan() && q.is_nan()) && !(*p == 0.0 && *q == 0.0) {
                    nd += 1;
                    mx = mx.max((*p as f64 - *q as f64).abs());
                }
            }
        }
        (Arr::F64(_, x), Arr::F64(_, y)) => {
            n = x.len();
            for (p, q) in x.iter().zip(y) {
                if p.to_bits() != q.to_bits() && !(p.is_nan() && q.is_nan()) && !(*p == 0.0 && *q == 0.0) {
                    nd += 1;
                    mx = mx.max((p - q).abs());
                }
            }
        }
        _ => return Err("dtype differs".into()),
    }
    Ok((nd, mx, n))
}

fn check(dir: &Path, filter: Option<&str>) {
    let man: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
    let mut by_fn: std::collections::BTreeMap<String, (usize, usize, usize, f64, usize)> = Default::default();
    for c in man["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        if let Some(fl) = filter {
            if !name.contains(fl) {
                continue;
            }
        }
        let fname = c["fn"].as_str().unwrap();
        let ins: Vec<Arr> = c["inputs"].as_array().unwrap().iter().map(|p| npy::read(&dir.join(p.as_str().unwrap())).unwrap()).collect();
        let want = npy::read(&dir.join(c["output"].as_str().unwrap())).unwrap();
        let got = std::panic::catch_unwind(|| run(fname, &c["args"], &ins));
        let e = by_fn.entry(fname.to_string()).or_default();
        e.0 += 1;
        let line = match got {
            Err(_) => {
                e.4 += 1;
                json!({"case": name, "fn": fname, "pass": false, "error": "panic"})
            }
            Ok(None) => {
                e.4 += 1;
                json!({"case": name, "fn": fname, "pass": false, "error": "not ported"})
            }
            Ok(Some(got)) => match compare(&got, &want) {
                Err(msg) => {
                    e.4 += 1;
                    json!({"case": name, "fn": fname, "pass": false, "error": msg})
                }
                Ok((nd, mx, n)) => {
                    if nd == 0 {
                        e.1 += 1;
                    }
                    e.2 += nd;
                    e.3 = e.3.max(mx);
                    json!({"case": name, "fn": fname, "pass": nd == 0, "differing": nd, "of": n, "max_abs_diff": mx})
                }
            },
        };
        println!("{line}");
    }
    let summary: Vec<Value> = by_fn
        .iter()
        .map(|(k, (n, p, nd, mx, err))| json!({"fn": k, "cases": n, "exact": p, "errors": err, "differing_elements": nd, "max_abs_diff": mx}))
        .collect();
    println!("{}", json!({"summary": summary}));
}

fn time<R>(reps: usize, mut f: impl FnMut() -> R) -> f64 {
    let _ = f();
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        std::hint::black_box(f());
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn bench(dir: &Path) {
    let bgr = mat_u8(&npy::read(&dir.join("bench_bgr4k.npy")).unwrap());
    let gray = mat_u8(&npy::read(&dir.join("bench_gray4k.npy")).unwrap());
    let gf = mat_f32(&npy::read(&dir.join("bench_grayf4k.npy")).unwrap());
    let bf = mat_f32(&npy::read(&dir.join("bench_bgrf4k.npy")).unwrap());
    let r = 5;
    let mut out = serde_json::Map::new();
    let mut put = |k: &str, v: f64| {
        out.insert(k.into(), json!((v * 100.0).round() / 100.0));
    };
    put("cvt_bgr2gray", time(r, || bgr2gray(&bgr)));
    put("cvt_bgr2rgb", time(r, || swap_rb(&bgr)));
    put("resize_linear_to1280", time(r, || resize_u8(&bgr, 1280, 720, 0.0, 0.0, Interp::Linear)));
    put("resize_area_to1280", time(r, || resize_u8(&bgr, 1280, 720, 0.0, 0.0, Interp::Area)));
    put("resize_area_to1000", time(r, || resize_u8(&bgr, 1000, 562, 0.0, 0.0, Interp::Area)));
    put("resize_cubic_to1280", time(r, || resize_u8(&bgr, 1280, 720, 0.0, 0.0, Interp::Cubic)));
    put("resize_nearest_half", time(r, || resize_u8(&bgr, 1920, 1080, 0.0, 0.0, Interp::Nearest)));
    put("resize_linear_f05", time(r, || resize_u8(&bgr, 0, 0, 0.5, 0.5, Interp::Linear)));
    put("resize_f32_area_to256", time(r, || resize_f32(&gf, 256, 144, 0.0, 0.0, Interp::Area)));
    let m = [0.9, -0.2, 100.0, 0.2, 0.9, -50.0];
    put("warp_affine_u8c3_4k", time(r, || warp_affine_u8(&bgr, &m, 3840, 2160, false, Border::Constant, [0.0; 4])));
    put("warp_affine_u8c1_4k", time(r, || warp_affine_u8(&gray, &m, 3840, 2160, false, Border::Constant, [0.0; 4])));
    put("warp_affine_f32c1_4k", time(r, || warp_affine_f32(&gf, &m, 3840, 2160, true, Border::Replicate, [0.0; 4])));
    put("gauss_u8c3_k5", time(r, || gaussian_blur_u8(&bgr, 5, 5, 0.0, 0.0)));
    put("gauss_u8c3_s6", time(r, || gaussian_blur_u8(&bgr, 0, 0, 6.0, 0.0)));
    put("gauss_f32c1_s1", time(r, || gaussian_blur_f32(&gf, 0, 0, 1.0, 0.0)));
    put("gauss_f32c3_s1", time(r, || gaussian_blur_f32(&bf, 0, 0, 1.0, 0.0)));
    put("erode_f32_k5", time(r, || erode(&gf, 5, 5)));
    put("copy_make_border_u8c3", time(r, || copy_make_border(&bgr, 8, 8, 8, 8, Border::Replicate, &[0; 4])));
    put("transform_f32c3", time(r, || transform3(&bf, &[1.1642, 2.0167, 0.0, -278.2338, 1.1645, -0.3897, -0.8121, 133.9585, 1.1642, 0.0, 1.5956, -224.693], 4)));
    let k = Mat::filled(9, 9, 1, 1.0f32 / 81.0);
    put("filter2d_f32c1_k9", time(r, || filter2d_f32(&gf, &k)));
    println!("{}", Value::Object(out));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(args.get(2).expect("usage: vision_parity check|bench <dir> [filter]"));
    match args[1].as_str() {
        "check" => check(&dir, args.get(3).map(|s| s.as_str())),
        "bench" => bench(&dir),
        _ => panic!("check|bench"),
    }
}
