//! `np.savez` / `np.load` files: the lighting caches (PROJECT/film_est_*.npz, light_est_*.npz) written so that Python
//! and Rust each read the other's.
//!
//! An .npz is a zip (stored, zip64 extras: `np.savez` opens each member with `force_zip64=True`) of .npy members,
//! written in keyword order. Each member is `np.lib.format.write_array`'s: format 1.0, the header dict with sorted
//! keys and `GROWTH_AXIS_MAX_DIGITS` spare spaces, padded to 64 bytes. Numeric arrays are raw little-endian float64;
//! an object array (light_est's `probe`, a 0-d array holding a dict) is a pickle of the array. Its members come out
//! byte-identical to numpy's for float64 arrays; the pickle is written plainly (no memo, no frames), which
//! `pickle.load` reads to an equal object.
//!
//! The pickle reader covers what Python 3 protocols 2-5 emit for numpy arrays, dicts, tuples, lists, ints, floats,
//! strings and bytes. Values are copies (a memoised object that is mutated later is seen as it was when memoised);
//! numpy's pickles only re-use globals, strings, shapes and dtypes this way, and a dtype's state is not needed (its
//! descr, e.g. "f8", is in its constructor's arguments).
use anyhow::{anyhow, bail, Context, Result};
use std::io::{Read, Write};
use std::path::Path;

/// A float64 C-order array.
#[derive(Clone, Debug, PartialEq)]
pub struct F64Array {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

/// A member of an .npz.
#[derive(Clone, Debug)]
pub enum Member {
    F64(F64Array),
    /// an object array (any shape) as its pickled value
    Object(Pickled),
}

/// A value read from (or to be written to) a pickle.
#[derive(Clone, Debug, PartialEq)]
pub enum Pickled {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    Tuple(Vec<Pickled>),
    List(Vec<Pickled>),
    Dict(Vec<(Pickled, Pickled)>),
    /// a float64 ndarray
    Array(F64Array),
    /// an object ndarray: shape and items (C order)
    ObjArray(Vec<usize>, Vec<Pickled>),
    Global(String, String),
    /// callable(*args), then BUILD with state
    Obj { f: Box<Pickled>, args: Box<Pickled>, state: Option<Box<Pickled>> },
}

impl Pickled {
    pub fn as_dict(&self) -> Option<&[(Pickled, Pickled)]> {
        match self {
            Pickled::Dict(d) => Some(d),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------- .npy

const GROWTH_AXIS_MAX_DIGITS: usize = 21;

fn shape_repr(shape: &[usize]) -> String {
    match shape.len() {
        0 => "()".into(),
        1 => format!("({},)", shape[0]),
        _ => format!("({})", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ")),
    }
}

fn npy_header(descr: &str, shape: &[usize]) -> Vec<u8> {
    let mut h = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {}, }}", shape_repr(shape));
    if !shape.is_empty() {
        h.push_str(&" ".repeat(GROWTH_AXIS_MAX_DIGITS - shape[0].to_string().len()));
    }
    let hlen = h.len() + 1;
    let pad = 64 - ((6 + 2 + 2 + hlen) % 64);
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend(((hlen + pad) as u16).to_le_bytes());
    out.extend(h.as_bytes());
    out.extend(std::iter::repeat_n(b' ', pad));
    out.push(b'\n');
    out
}

/// One .npy member's bytes.
pub fn npy_bytes(m: &Member) -> Vec<u8> {
    match m {
        Member::F64(a) => {
            let mut out = npy_header("<f8", &a.shape);
            for v in &a.data {
                out.extend(v.to_le_bytes());
            }
            out
        }
        Member::Object(v) => {
            let (shape, items) = match v {
                Pickled::ObjArray(s, it) => (s.clone(), it.clone()),
                other => (vec![], vec![other.clone()]),
            };
            let mut out = npy_header("|O", &shape);
            let mut p = Pickler::default();
            p.out.extend([0x80, 4]);
            p.obj_array(&shape, &items);
            p.out.push(b'.');
            out.extend(p.out);
            out
        }
    }
}

fn parse_npy(b: &[u8]) -> Result<Member> {
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        bail!("not a .npy member");
    }
    let (hl, off) = match b[6] {
        1 => (u16::from_le_bytes([b[8], b[9]]) as usize, 10),
        _ => (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12),
    };
    let head = std::str::from_utf8(&b[off..off + hl])?;
    if head.contains("'fortran_order': True") {
        bail!("npy: Fortran order not supported");
    }
    let sh = head.split("'shape': (").nth(1).and_then(|s| s.split(')').next()).ok_or_else(|| anyhow!("npy: no shape"))?;
    let shape: Vec<usize> = sh.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    let body = &b[off + hl..];
    if head.contains("'descr': '<f8'") {
        let data: Vec<f64> = body.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
        let n: usize = shape.iter().product();
        if data.len() < n {
            bail!("npy: truncated");
        }
        return Ok(Member::F64(F64Array { shape, data: data[..n].to_vec() }));
    }
    if head.contains("'descr': '|O'") {
        return Ok(Member::Object(unpickle(body)?));
    }
    bail!("npy: unsupported dtype: {head}")
}

/// `np.load(path, allow_pickle=True)`: the members by name (without ".npy"), in the file's order.
pub fn load(path: &Path) -> Result<Vec<(String, Member)>> {
    let f = std::fs::File::open(path).with_context(|| format!("{}", path.display()))?;
    let mut z = zip::ZipArchive::new(f)?;
    let mut out = vec![];
    for i in 0..z.len() {
        let mut e = z.by_index(i)?;
        let name = e.name().trim_end_matches(".npy").to_string();
        let mut b = vec![];
        e.read_to_end(&mut b)?;
        let m = parse_npy(&b).with_context(|| format!("{}: {name}", path.display()))?;
        out.push((name, m));
    }
    Ok(out)
}

/// `np.savez(path, **members)` (written to a temporary file, then renamed into place).
pub fn save(path: &Path, members: &[(&str, Member)]) -> Result<()> {
    let tmp = path.with_extension("npz.tmp");
    {
        let f = std::fs::File::create(&tmp).with_context(|| format!("{}", tmp.display()))?;
        let mut z = zip::ZipWriter::new(f);
        let opt = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored)
            .large_file(true);
        for (name, m) in members {
            z.start_file(format!("{name}.npy"), opt)?;
            z.write_all(&npy_bytes(m))?;
        }
        z.finish()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

// ---------------------------------------------------------------- pickle writer

#[derive(Default)]
struct Pickler {
    out: Vec<u8>,
}

impl Pickler {
    fn str_(&mut self, s: &str) {
        let b = s.as_bytes();
        if b.len() < 256 {
            self.out.extend([0x8c, b.len() as u8]);
        } else {
            self.out.push(b'X');
            self.out.extend((b.len() as u32).to_le_bytes());
        }
        self.out.extend(b);
    }
    fn global(&mut self, m: &str, n: &str) {
        self.str_(m);
        self.str_(n);
        self.out.push(0x93); // STACK_GLOBAL
    }
    fn int(&mut self, v: i64) {
        if (0..256).contains(&v) {
            self.out.extend([b'K', v as u8]);
        } else if (0..65536).contains(&v) {
            self.out.push(b'M');
            self.out.extend((v as u16).to_le_bytes());
        } else if (i32::MIN as i64..=i32::MAX as i64).contains(&v) {
            self.out.push(b'J');
            self.out.extend((v as i32).to_le_bytes());
        } else {
            self.out.extend([0x8a, 8]); // LONG1, 8 bytes
            self.out.extend(v.to_le_bytes());
        }
    }
    fn bytes(&mut self, b: &[u8]) {
        if b.len() < 256 {
            self.out.extend([b'C', b.len() as u8]);
        } else {
            self.out.push(b'B');
            self.out.extend((b.len() as u32).to_le_bytes());
        }
        self.out.extend(b);
    }
    fn tuple(&mut self, items: &[Pickled]) {
        match items.len() {
            0 => self.out.push(b')'),
            1..=3 => {
                for it in items {
                    self.value(it);
                }
                self.out.push([0x85, 0x86, 0x87][items.len() - 1]);
            }
            _ => {
                self.out.push(b'(');
                for it in items {
                    self.value(it);
                }
                self.out.push(b't');
            }
        }
    }
    fn shape(&mut self, s: &[usize]) {
        self.tuple(&s.iter().map(|&v| Pickled::Int(v as i64)).collect::<Vec<_>>());
    }
    /// numpy.dtype(descr, False, True) with its state (3, order, None, None, None, -1, -1, flags)
    fn dtype(&mut self, descr: &str, order: &str, flags: i64) {
        self.global("numpy", "dtype");
        self.tuple(&[Pickled::Str(descr.into()), Pickled::Bool(false), Pickled::Bool(true)]);
        self.out.push(b'R');
        self.tuple(&[
            Pickled::Int(3),
            Pickled::Str(order.into()),
            Pickled::None,
            Pickled::None,
            Pickled::None,
            Pickled::Int(-1),
            Pickled::Int(-1),
            Pickled::Int(flags),
        ]);
        self.out.push(b'b');
    }
    /// _reconstruct(ndarray, (0,), b'b'), then BUILD (1, shape, dtype, False, data)
    fn array_head(&mut self) {
        self.global("numpy._core.multiarray", "_reconstruct");
        self.global("numpy", "ndarray");
        self.tuple(&[Pickled::Int(0)]);
        self.bytes(b"b");
        self.out.push(0x87);
        self.out.push(b'R');
    }
    fn f64_array(&mut self, a: &F64Array) {
        self.array_head();
        self.out.push(b'(');
        self.int(1);
        self.shape(&a.shape);
        self.dtype("f8", "<", 0);
        self.out.push(0x89);
        let raw: Vec<u8> = a.data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.bytes(&raw);
        self.out.push(b't');
        self.out.push(b'b');
    }
    fn obj_array(&mut self, shape: &[usize], items: &[Pickled]) {
        self.array_head();
        self.out.push(b'(');
        self.int(1);
        self.shape(shape);
        self.dtype("O8", "|", 63);
        self.out.push(0x89);
        self.out.push(b']');
        if !items.is_empty() {
            self.out.push(b'(');
            for it in items {
                self.value(it);
            }
            self.out.push(b'e'); // APPENDS
        }
        self.out.push(b't');
        self.out.push(b'b');
    }
    fn value(&mut self, v: &Pickled) {
        match v {
            Pickled::None => self.out.push(b'N'),
            Pickled::Bool(b) => self.out.push(if *b { 0x88 } else { 0x89 }),
            Pickled::Int(i) => self.int(*i),
            Pickled::Float(f) => {
                self.out.push(b'G');
                self.out.extend(f.to_be_bytes());
            }
            Pickled::Str(s) => self.str_(s),
            Pickled::Bytes(b) => self.bytes(b),
            Pickled::Tuple(t) => self.tuple(t),
            Pickled::List(l) => {
                self.out.push(b']');
                if !l.is_empty() {
                    self.out.push(b'(');
                    for it in l {
                        self.value(it);
                    }
                    self.out.push(b'e');
                }
            }
            Pickled::Dict(d) => {
                self.out.push(b'}');
                for chunk in d.chunks(1000) {
                    self.out.push(b'(');
                    for (k, v) in chunk {
                        self.value(k);
                        self.value(v);
                    }
                    self.out.push(b'u');
                }
            }
            Pickled::Array(a) => self.f64_array(a),
            Pickled::ObjArray(s, it) => self.obj_array(s, it),
            Pickled::Global(m, n) => self.global(m, n),
            Pickled::Obj { f, args, state } => {
                self.value(f);
                self.value(args);
                self.out.push(b'R');
                if let Some(s) = state {
                    self.value(s);
                    self.out.push(b'b');
                }
            }
        }
    }
}

// ---------------------------------------------------------------- pickle reader

/// Read one pickle; numpy arrays come back as [`Pickled::Array`] (float64) or [`Pickled::ObjArray`].
pub fn unpickle(b: &[u8]) -> Result<Pickled> {
    let mut st: Vec<Pickled> = vec![];
    let mut marks: Vec<usize> = vec![];
    let mut memo: std::collections::HashMap<u64, Pickled> = Default::default();
    let mut i = 0usize;
    let take = |i: &mut usize, n: usize| -> Result<&[u8]> {
        if *i + n > b.len() {
            bail!("pickle: truncated");
        }
        let s = &b[*i..*i + n];
        *i += n;
        Ok(s)
    };
    macro_rules! pop {
        () => {
            st.pop().ok_or_else(|| anyhow!("pickle: stack underflow"))?
        };
    }
    macro_rules! since_mark {
        () => {{
            let m = marks.pop().ok_or_else(|| anyhow!("pickle: no mark"))?;
            st.split_off(m)
        }};
    }
    loop {
        let op = take(&mut i, 1)?[0];
        match op {
            0x80 => {
                take(&mut i, 1)?;
            }
            0x95 => {
                take(&mut i, 8)?;
            }
            b'.' => return st.pop().ok_or_else(|| anyhow!("pickle: empty")),
            b'(' => marks.push(st.len()),
            b'N' => st.push(Pickled::None),
            0x88 => st.push(Pickled::Bool(true)),
            0x89 => st.push(Pickled::Bool(false)),
            b'K' => st.push(Pickled::Int(take(&mut i, 1)?[0] as i64)),
            b'M' => {
                let s = take(&mut i, 2)?;
                st.push(Pickled::Int(u16::from_le_bytes([s[0], s[1]]) as i64))
            }
            b'J' => {
                let s = take(&mut i, 4)?;
                st.push(Pickled::Int(i32::from_le_bytes(s.try_into().unwrap()) as i64))
            }
            0x8a => {
                let n = take(&mut i, 1)?[0] as usize;
                let s = take(&mut i, n)?;
                if n > 8 {
                    bail!("pickle: LONG1 too wide");
                }
                let mut v: i64 = 0;
                for (k, &byte) in s.iter().enumerate() {
                    v |= (byte as i64) << (8 * k);
                }
                if n > 0 && n < 8 && s[n - 1] & 0x80 != 0 {
                    v -= 1i64 << (8 * n);
                }
                st.push(Pickled::Int(v))
            }
            b'G' => {
                let s = take(&mut i, 8)?;
                st.push(Pickled::Float(f64::from_be_bytes(s.try_into().unwrap())))
            }
            0x8c => {
                let n = take(&mut i, 1)?[0] as usize;
                st.push(Pickled::Str(String::from_utf8(take(&mut i, n)?.to_vec())?))
            }
            b'X' => {
                let n = u32::from_le_bytes(take(&mut i, 4)?.try_into().unwrap()) as usize;
                st.push(Pickled::Str(String::from_utf8(take(&mut i, n)?.to_vec())?))
            }
            0x8d => {
                let n = u64::from_le_bytes(take(&mut i, 8)?.try_into().unwrap()) as usize;
                st.push(Pickled::Str(String::from_utf8(take(&mut i, n)?.to_vec())?))
            }
            b'C' => {
                let n = take(&mut i, 1)?[0] as usize;
                st.push(Pickled::Bytes(take(&mut i, n)?.to_vec()))
            }
            b'B' => {
                let n = u32::from_le_bytes(take(&mut i, 4)?.try_into().unwrap()) as usize;
                st.push(Pickled::Bytes(take(&mut i, n)?.to_vec()))
            }
            0x8e => {
                let n = u64::from_le_bytes(take(&mut i, 8)?.try_into().unwrap()) as usize;
                st.push(Pickled::Bytes(take(&mut i, n)?.to_vec()))
            }
            0x96 => {
                // BYTEARRAY8
                let n = u64::from_le_bytes(take(&mut i, 8)?.try_into().unwrap()) as usize;
                st.push(Pickled::Bytes(take(&mut i, n)?.to_vec()))
            }
            b')' => st.push(Pickled::Tuple(vec![])),
            0x85 => {
                let a = pop!();
                st.push(Pickled::Tuple(vec![a]))
            }
            0x86 => {
                let b2 = pop!();
                let a = pop!();
                st.push(Pickled::Tuple(vec![a, b2]))
            }
            0x87 => {
                let c = pop!();
                let b2 = pop!();
                let a = pop!();
                st.push(Pickled::Tuple(vec![a, b2, c]))
            }
            b't' => {
                let items = since_mark!();
                st.push(Pickled::Tuple(items))
            }
            b']' => st.push(Pickled::List(vec![])),
            b'l' => {
                let items = since_mark!();
                st.push(Pickled::List(items))
            }
            b'a' => {
                let v = pop!();
                match st.last_mut() {
                    Some(Pickled::List(l)) => l.push(v),
                    _ => bail!("pickle: APPEND to a non-list"),
                }
            }
            b'e' => {
                let items = since_mark!();
                match st.last_mut() {
                    Some(Pickled::List(l)) => l.extend(items),
                    _ => bail!("pickle: APPENDS to a non-list"),
                }
            }
            b'}' => st.push(Pickled::Dict(vec![])),
            b'd' => {
                let items = since_mark!();
                let mut d = vec![];
                let mut it = items.into_iter();
                while let (Some(k), Some(v)) = (it.next(), it.next()) {
                    d.push((k, v));
                }
                st.push(Pickled::Dict(d))
            }
            b's' => {
                let v = pop!();
                let k = pop!();
                match st.last_mut() {
                    Some(Pickled::Dict(d)) => dict_set(d, k, v),
                    _ => bail!("pickle: SETITEM on a non-dict"),
                }
            }
            b'u' => {
                let items = since_mark!();
                match st.last_mut() {
                    Some(Pickled::Dict(d)) => {
                        let mut it = items.into_iter();
                        while let (Some(k), Some(v)) = (it.next(), it.next()) {
                            dict_set(d, k, v);
                        }
                    }
                    _ => bail!("pickle: SETITEMS on a non-dict"),
                }
            }
            0x94 => {
                let v = st.last().ok_or_else(|| anyhow!("pickle: MEMOIZE on empty"))?.clone();
                memo.insert(memo.len() as u64, v);
            }
            b'q' => {
                let k = take(&mut i, 1)?[0] as u64;
                memo.insert(k, st.last().cloned().ok_or_else(|| anyhow!("pickle: empty"))?);
            }
            b'r' => {
                let k = u32::from_le_bytes(take(&mut i, 4)?.try_into().unwrap()) as u64;
                memo.insert(k, st.last().cloned().ok_or_else(|| anyhow!("pickle: empty"))?);
            }
            b'h' => {
                let k = take(&mut i, 1)?[0] as u64;
                st.push(memo.get(&k).cloned().ok_or_else(|| anyhow!("pickle: no memo {k}"))?)
            }
            b'j' => {
                let k = u32::from_le_bytes(take(&mut i, 4)?.try_into().unwrap()) as u64;
                st.push(memo.get(&k).cloned().ok_or_else(|| anyhow!("pickle: no memo {k}"))?)
            }
            0x93 => {
                let n = pop!();
                let m = pop!();
                match (m, n) {
                    (Pickled::Str(m), Pickled::Str(n)) => st.push(Pickled::Global(m, n)),
                    _ => bail!("pickle: STACK_GLOBAL wants two strings"),
                }
            }
            b'c' => {
                let rest = &b[i..];
                let a = rest.iter().position(|&c| c == b'\n').ok_or_else(|| anyhow!("pickle: GLOBAL"))?;
                let m = String::from_utf8(rest[..a].to_vec())?;
                let rest2 = &rest[a + 1..];
                let c = rest2.iter().position(|&c| c == b'\n').ok_or_else(|| anyhow!("pickle: GLOBAL"))?;
                let n = String::from_utf8(rest2[..c].to_vec())?;
                i += a + 1 + c + 1;
                st.push(Pickled::Global(m, n));
            }
            b'R' => {
                let args = pop!();
                let f = pop!();
                st.push(Pickled::Obj { f: Box::new(f), args: Box::new(args), state: None })
            }
            b'b' => {
                let s = pop!();
                match st.pop() {
                    Some(Pickled::Obj { f, args, .. }) => st.push(finish(Pickled::Obj { f, args, state: Some(Box::new(s)) })?),
                    _ => bail!("pickle: BUILD on a non-object"),
                }
            }
            _ => bail!("pickle: opcode {op:#04x} not supported"),
        }
    }
}

fn dict_set(d: &mut Vec<(Pickled, Pickled)>, k: Pickled, v: Pickled) {
    match d.iter_mut().find(|(q, _)| *q == k) {
        Some(e) => e.1 = v,
        None => d.push((k, v)),
    }
}

/// An ndarray reconstruction (`_reconstruct(...)` + BUILD state) as an array value; anything else unchanged.
fn finish(o: Pickled) -> Result<Pickled> {
    let Pickled::Obj { f, state: Some(state), .. } = &o else { return Ok(o) };
    let Pickled::Global(m, n) = f.as_ref() else { return Ok(o) };
    if !(m.ends_with("core.multiarray") && n == "_reconstruct") {
        return Ok(o);
    }
    let Pickled::Tuple(s) = state.as_ref() else { bail!("ndarray state") };
    if s.len() != 5 {
        bail!("ndarray state: {} items", s.len());
    }
    let shape: Vec<usize> = match &s[1] {
        Pickled::Tuple(t) => t
            .iter()
            .map(|v| match v {
                Pickled::Int(i) => Ok(*i as usize),
                _ => Err(anyhow!("ndarray shape")),
            })
            .collect::<Result<_>>()?,
        _ => bail!("ndarray shape"),
    };
    let descr = match &s[2] {
        Pickled::Obj { args, .. } => match args.as_ref() {
            Pickled::Tuple(a) => match a.first() {
                Some(Pickled::Str(d)) => d.clone(),
                _ => bail!("dtype args"),
            },
            _ => bail!("dtype args"),
        },
        _ => bail!("ndarray dtype"),
    };
    if matches!(s[3], Pickled::Bool(true)) {
        bail!("ndarray: Fortran order not supported");
    }
    match (descr.as_str(), &s[4]) {
        ("f8", Pickled::Bytes(raw)) => Ok(Pickled::Array(F64Array {
            shape,
            data: raw.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect(),
        })),
        ("O8", Pickled::List(items)) => Ok(Pickled::ObjArray(shape, items.clone())),
        (d, _) => bail!("ndarray dtype {d} not supported"),
    }
}
