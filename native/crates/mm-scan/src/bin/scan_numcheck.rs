//! numpy-exactness checks for mm-scan's np module: reads one JSON case per line from stdin
//! ({"op": ..., "x": [...]}) and prints one result per line. Driven by tools/parity/scan_numcheck.py.
use mm_core::pyjson::Py;
use mm_scan::np;
use std::io::BufRead;

fn fl(v: &Py) -> Vec<f64> {
    v.as_list().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn main() {
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        let c = Py::parse(&line).unwrap();
        let op = c.get("op").as_str().unwrap().to_string();
        let x = fl(c.get("x"));
        if op == "mmcore_repr" {
            println!("{}", Py::Str(Py::List(x.iter().map(|v| Py::Float(*v)).collect()).dumps()).dumps());
            continue;
        }
        if op == "repr" {
            // the recording key text for these values (Python's json.dumps of the list)
            println!("{}", Py::Str(mm_scan::evidence::key(&x)).dumps());
            continue;
        }
        let out: Vec<f64> = match op.as_str() {
            "gemv23" => np::gemv(&[[x[0], x[1], x[2]], [x[3], x[4], x[5]]], [x[6], x[7], x[8]]).to_vec(),
            "gemv33" => np::gemv(&[[x[0], x[1], x[2]], [x[3], x[4], x[5]], [x[6], x[7], x[8]]], [x[9], x[10], x[11]]).to_vec(),
            "matmul33" => {
                let a = [[x[0], x[1], x[2]], [x[3], x[4], x[5]], [x[6], x[7], x[8]]];
                let b = [[x[9], x[10], x[11]], [x[12], x[13], x[14]], [x[15], x[16], x[17]]];
                np::matmul33(&a, &b).iter().flatten().copied().collect()
            }
            "det2" => vec![np::det2(x[0], x[1], x[2], x[3])],
            "inv33" => np::inv33(&[[x[0], x[1], x[2]], [x[3], x[4], x[5]], [x[6], x[7], x[8]]]).iter().flatten().copied().collect(),
            "sum" => vec![np::sum(&x)],
            "mean" => vec![np::mean(&x)],
            "median" => vec![np::median(&x)],
            "p90" => vec![np::percentile(&x, 90.0)],
            "hypot" => vec![np::hypot(x[0], x[1])],
            "norm" => vec![np::norm(&x)],
            "ddot" => {
                let h = x.len() / 2;
                vec![np::ddot(&x[..h], &x[h..])]
            }
            "sdot" => {
                let v: Vec<f32> = x.iter().map(|&a| a as f32).collect();
                let h = v.len() / 2;
                vec![np::sdot(&v[..h], &v[h..]) as f64]
            }
            "meanrows32" => {
                // x: [n, d, values...]
                let (n, d) = (x[0] as usize, x[1] as usize);
                let v: Vec<f32> = x[2..].iter().map(|&a| a as f32).collect();
                let rows: Vec<&[f32]> = (0..n).map(|i| &v[i * d..(i + 1) * d]).collect();
                let m = np::mean_rows_f32(&rows);
                np::unit_f32(&m).iter().map(|&a| a as f64).collect()
            }
            "mmcore_warp" => mm_core::edits::warp_cs([x[6], x[7], x[8]], &[[x[0], x[1], x[2]], [x[3], x[4], x[5]]]).to_vec(),
            "scan_warp" => mm_scan::geom::warp_cs(&[x[6], x[7], x[8]], &[[x[0], x[1], x[2]], [x[3], x[4], x[5]]]).to_vec(),
            "pyset" => {
                // x: [n, v_0..v_{n-1}, then (a, b) index pairs: sets[a] |= sets[b] (b emptied), then -1 and copies]
                let n = x[0] as usize;
                let mut sets: Vec<Option<mm_scan::pyset::PySet>> =
                    (0..n).map(|i| Some(mm_scan::pyset::PySet::of(&[x[1 + i] as i64]))).collect();
                let mut i = 1 + n;
                let mut out = vec![];
                while i + 1 < x.len() {
                    let (a, b) = (x[i] as usize, x[i + 1] as usize);
                    let sb = sets[b].take().unwrap();
                    sets[a].as_mut().unwrap().update(&sb);
                    i += 2;
                }
                for s in sets.iter().flatten() {
                    out.extend(s.iter().map(|v| v as f64));
                    out.push(-1.0);
                    out.extend(s.copy().iter().map(|v| v as f64));
                    out.push(-2.0);
                }
                out
            }
            _ => panic!("op {op}"),
        };
        println!("{}", Py::List(out.into_iter().map(Py::Float).collect()).dumps());
    }
}
