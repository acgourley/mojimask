//! Minimal .npy reader and writer (little-endian u8, f32, f64; C order) for the parity tools.
use std::{fs, io::Write, path::Path};

#[derive(Clone, Debug)]
pub enum Arr {
    U8(Vec<usize>, Vec<u8>),
    F32(Vec<usize>, Vec<f32>),
    F64(Vec<usize>, Vec<f64>),
}

impl Arr {
    pub fn shape(&self) -> &[usize] {
        match self {
            Arr::U8(s, _) | Arr::F32(s, _) | Arr::F64(s, _) => s,
        }
    }
    /// (rows, cols, channels) for an image array: (h, w) or (h, w, c).
    pub fn hwc(&self) -> (usize, usize, usize) {
        let s = self.shape();
        match s.len() {
            2 => (s[0], s[1], 1),
            3 => (s[0], s[1], s[2]),
            1 => (1, s[0], 1),
            _ => panic!("not an image: {s:?}"),
        }
    }
}

pub fn read(path: &Path) -> Result<Arr, String> {
    let b = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(format!("{}: not npy", path.display()));
    }
    let (hlen, off) = if b[6] == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let h = std::str::from_utf8(&b[off..off + hlen]).map_err(|e| e.to_string())?;
    if h.contains("'fortran_order': True") {
        return Err("fortran order".into());
    }
    let descr = h.split("'descr':").nth(1).ok_or("no descr")?.trim_start();
    let s = h.split("'shape':").nth(1).ok_or("no shape")?;
    let s = &s[s.find('(').unwrap() + 1..s.find(')').unwrap()];
    let shape: Vec<usize> = s.split(',').filter(|t| !t.trim().is_empty()).map(|t| t.trim().parse().unwrap()).collect();
    let raw = &b[off + hlen..];
    let n: usize = shape.iter().product();
    if descr.starts_with("'|u1'") {
        Ok(Arr::U8(shape, raw[..n].to_vec()))
    } else if descr.starts_with("'<f4'") {
        Ok(Arr::F32(shape, raw.chunks_exact(4).take(n).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()))
    } else if descr.starts_with("'<f8'") {
        Ok(Arr::F64(shape, raw.chunks_exact(8).take(n).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()))
    } else {
        Err(format!("{}: dtype {descr}", path.display()))
    }
}

pub fn write(path: &Path, a: &Arr) -> std::io::Result<()> {
    let (descr, shape) = match a {
        Arr::U8(s, _) => ("|u1", s),
        Arr::F32(s, _) => ("<f4", s),
        Arr::F64(s, _) => ("<f8", s),
    };
    let sh = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!("({})", shape.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", "))
    };
    let mut h = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {sh}, }}");
    while (10 + h.len() + 1) % 64 != 0 {
        h.push(' ');
    }
    h.push('\n');
    let mut f = fs::File::create(path)?;
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&(h.len() as u16).to_le_bytes())?;
    f.write_all(h.as_bytes())?;
    match a {
        Arr::U8(_, d) => f.write_all(d)?,
        Arr::F32(_, d) => d.iter().try_for_each(|v| f.write_all(&v.to_le_bytes()))?,
        Arr::F64(_, d) => d.iter().try_for_each(|v| f.write_all(&v.to_le_bytes()))?,
    }
    Ok(())
}
