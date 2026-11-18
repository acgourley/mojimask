//! Minimal .npy reader/writer for little-endian f32, C order (what dump_python.py writes).
use std::{fs, io::Write, path::Path};

pub struct Npy {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub fn read(path: &Path) -> Result<Npy, String> {
    let b = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if &b[..6] != b"\x93NUMPY" {
        return Err(format!("{}: not npy", path.display()));
    }
    let (hlen, off) = if b[6] == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let h = std::str::from_utf8(&b[off..off + hlen]).unwrap();
    let descr = h.split("'descr':").nth(1).unwrap().trim_start();
    let fortran = h.contains("'fortran_order': True");
    if fortran {
        return Err("fortran order".into());
    }
    let s = h.split("'shape':").nth(1).unwrap();
    let s = &s[s.find('(').unwrap() + 1..s.find(')').unwrap()];
    let shape: Vec<usize> = s.split(',').filter(|t| !t.trim().is_empty()).map(|t| t.trim().parse().unwrap()).collect();
    let raw = &b[off + hlen..];
    let n: usize = shape.iter().product();
    let data: Vec<f32> = if descr.starts_with("'<f4'") {
        raw.chunks_exact(4).take(n).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    } else if descr.starts_with("'<f8'") {
        raw.chunks_exact(8).take(n).map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32).collect()
    } else {
        return Err(format!("{}: dtype {descr}", path.display()));
    };
    Ok(Npy { shape, data })
}

#[allow(dead_code)]
pub fn write(path: &Path, shape: &[usize], data: &[f32]) {
    let sh = if shape.len() == 1 { format!("({},)", shape[0]) } else {
        format!("({})", shape.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", "))
    };
    let mut h = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {sh}, }}");
    while (10 + h.len() + 1) % 64 != 0 {
        h.push(' ');
    }
    h.push('\n');
    let mut f = fs::File::create(path).unwrap();
    f.write_all(b"\x93NUMPY\x01\x00").unwrap();
    f.write_all(&(h.len() as u16).to_le_bytes()).unwrap();
    f.write_all(h.as_bytes()).unwrap();
    for v in data {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
}
