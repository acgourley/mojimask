//! Python- and numpy-exactness checks for mm-core: reads one JSON case per line from stdin ({"op": ..., "x": [...]})
//! and prints one result per line. Driven by tools/parity/core_numcheck.py.
//! - `repr`: `pyjson` writing the list (Python's `json.dumps`, so each float as `repr`), as a JSON string.
//! - `warp`: `edits::warp_cs([x6, x7, x8], [[x0..x2], [x3..x5]])`.
//! - `compose`: `edits::compose` of the 2x3 matrices in x (6 numbers each), from 0 to the last.
use mm_core::pyjson::Py;
use std::io::{BufRead, Write};

fn main() {
    let stdin = std::io::stdin();
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        let c = Py::parse(&line).unwrap();
        let op = c.get("op").as_str().unwrap().to_string();
        let x: Vec<f64> = c.get("x").as_list().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let res: Vec<f64> = match op.as_str() {
            "repr" => {
                writeln!(out, "{}", Py::Str(Py::List(x.iter().map(|v| Py::Float(*v)).collect()).dumps()).dumps()).unwrap();
                continue;
            }
            "warp" => mm_core::edits::warp_cs([x[6], x[7], x[8]], &[[x[0], x[1], x[2]], [x[3], x[4], x[5]]]).to_vec(),
            "compose" => {
                let mats: Vec<[[f64; 3]; 2]> =
                    x.chunks(6).map(|m| [[m[0], m[1], m[2]], [m[3], m[4], m[5]]]).collect();
                mm_core::edits::compose(&mats, 0, mats.len() as i64 - 1).iter().flatten().copied().collect()
            }
            _ => panic!("op {op}"),
        };
        writeln!(out, "{}", Py::List(res.into_iter().map(Py::Float).collect()).dumps()).unwrap();
    }
}
