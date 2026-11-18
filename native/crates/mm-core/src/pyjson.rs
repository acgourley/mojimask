//! JSON values that behave like Python's `json` module and `dict`, so the project files the Rust model writes
//! (facts.json, edits.json, meta.json) are byte-identical to the Python app's.
//!
//! What has to match, and why it isn't serde_json:
//! - **Key order is insertion order** (a Python dict): setting an existing key keeps its place, deleting and
//!   re-adding moves it to the end. `people` and `not_face` are written in that order.
//! - **Ints and floats stay apart:** `1` and `1.0` are different text. Floats are written as Python's `repr`
//!   (shortest round trip; `1e-05`, `1e+16`, `5.0`).
//! - **`ensure_ascii`:** everything outside ASCII is written `\uXXXX` (surrogate pairs above U+FFFF), lowercase hex.
//! - **Separators:** `json.dump(x)` writes `", "` and `": "`; with `indent=n`, `","` at line ends.
//! - **Equality** is Python's: `1 == 1.0 == True`, dicts equal whatever their order.
use std::fmt::Write as _;

#[derive(Clone, Debug)]
pub enum Py {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<Py>),
    Dict(Dict),
}

/// An insertion-ordered string-keyed map (a Python dict as JSON gives it).
#[derive(Clone, Debug, Default)]
pub struct Dict(pub Vec<(String, Py)>);

impl Dict {
    pub fn new() -> Self {
        Dict(Vec::new())
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    fn pos(&self, k: &str) -> Option<usize> {
        self.0.iter().position(|(q, _)| q == k)
    }
    pub fn get(&self, k: &str) -> Option<&Py> {
        self.0.iter().find(|(q, _)| q == k).map(|(_, v)| v)
    }
    pub fn get_mut(&mut self, k: &str) -> Option<&mut Py> {
        self.0.iter_mut().find(|(q, _)| q == k).map(|(_, v)| v)
    }
    pub fn contains(&self, k: &str) -> bool {
        self.pos(k).is_some()
    }
    /// `d[k] = v`: a new key goes last, an existing one keeps its place.
    pub fn insert(&mut self, k: impl Into<String>, v: Py) {
        let k = k.into();
        match self.pos(&k) {
            Some(i) => self.0[i].1 = v,
            None => self.0.push((k, v)),
        }
    }
    /// `d.pop(k, None)`
    pub fn remove(&mut self, k: &str) -> Option<Py> {
        self.pos(k).map(|i| self.0.remove(i).1)
    }
    /// `d.setdefault(k, v)`
    pub fn setdefault(&mut self, k: &str, v: Py) -> &mut Py {
        let i = match self.pos(k) {
            Some(i) => i,
            None => {
                self.0.push((k.to_string(), v));
                self.0.len() - 1
            }
        };
        &mut self.0[i].1
    }
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|(k, _)| k.as_str())
    }
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Py)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v))
    }
}

impl FromIterator<(String, Py)> for Dict {
    fn from_iter<I: IntoIterator<Item = (String, Py)>>(it: I) -> Self {
        let mut d = Dict::new();
        for (k, v) in it {
            d.insert(k, v);
        }
        d
    }
}

/// Build a dict from pairs, in order: `dict![("a", Py::Int(1))]`.
#[macro_export]
macro_rules! pydict {
    ($(($k:expr, $v:expr)),* $(,)?) => {{
        #[allow(unused_mut)]
        let mut d = $crate::pyjson::Dict::new();
        $(d.insert($k, $crate::pyjson::Py::from($v));)*
        $crate::pyjson::Py::Dict(d)
    }};
}

impl From<&str> for Py {
    fn from(s: &str) -> Py {
        Py::Str(s.to_string())
    }
}
impl From<String> for Py {
    fn from(s: String) -> Py {
        Py::Str(s)
    }
}
impl From<&String> for Py {
    fn from(s: &String) -> Py {
        Py::Str(s.clone())
    }
}
impl From<bool> for Py {
    fn from(b: bool) -> Py {
        Py::Bool(b)
    }
}
impl From<i64> for Py {
    fn from(i: i64) -> Py {
        Py::Int(i)
    }
}
impl From<i32> for Py {
    fn from(i: i32) -> Py {
        Py::Int(i as i64)
    }
}
impl From<usize> for Py {
    fn from(i: usize) -> Py {
        Py::Int(i as i64)
    }
}
impl From<f64> for Py {
    fn from(f: f64) -> Py {
        Py::Float(f)
    }
}
impl From<Dict> for Py {
    fn from(d: Dict) -> Py {
        Py::Dict(d)
    }
}
impl From<Vec<Py>> for Py {
    fn from(v: Vec<Py>) -> Py {
        Py::List(v)
    }
}
impl From<Option<Py>> for Py {
    fn from(v: Option<Py>) -> Py {
        v.unwrap_or(Py::Null)
    }
}

impl PartialEq for Py {
    /// Python's `==` on JSON values.
    fn eq(&self, o: &Py) -> bool {
        use Py::*;
        match (self, o) {
            (Null, Null) => true,
            (Str(a), Str(b)) => a == b,
            (List(a), List(b)) => a == b,
            (Dict(a), Dict(b)) => a.len() == b.len() && a.iter().all(|(k, v)| b.get(k) == Some(v)),
            _ => match (self.num(), o.num()) {
                (Some(a), Some(b)) => a == b,
                _ => false,
            },
        }
    }
}

impl Py {
    fn num(&self) -> Option<f64> {
        match self {
            Py::Bool(b) => Some(*b as i64 as f64),
            Py::Int(i) => Some(*i as f64),
            Py::Float(f) => Some(*f),
            _ => None,
        }
    }
    /// The value as a number (bools and ints too).
    pub fn as_f64(&self) -> Option<f64> {
        self.num()
    }
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Py::Int(i) => Some(*i),
            Py::Bool(b) => Some(*b as i64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Py::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_dict(&self) -> Option<&Dict> {
        match self {
            Py::Dict(d) => Some(d),
            _ => None,
        }
    }
    pub fn as_dict_mut(&mut self) -> Option<&mut Dict> {
        match self {
            Py::Dict(d) => Some(d),
            _ => None,
        }
    }
    pub fn as_list(&self) -> Option<&Vec<Py>> {
        match self {
            Py::List(l) => Some(l),
            _ => None,
        }
    }
    pub fn is_null(&self) -> bool {
        matches!(self, Py::Null)
    }
    /// `d.get(k)` on a dict; Null for anything else or a missing key.
    pub fn get(&self, k: &str) -> &Py {
        static NULL: Py = Py::Null;
        self.as_dict().and_then(|d| d.get(k)).unwrap_or(&NULL)
    }
    /// Python truthiness.
    pub fn truthy(&self) -> bool {
        match self {
            Py::Null => false,
            Py::Bool(b) => *b,
            Py::Int(i) => *i != 0,
            Py::Float(f) => *f != 0.0,
            Py::Str(s) => !s.is_empty(),
            Py::List(l) => !l.is_empty(),
            Py::Dict(d) => !d.is_empty(),
        }
    }
    /// `x or default`
    pub fn or<'a>(&'a self, default: &'a Py) -> &'a Py {
        if self.truthy() {
            self
        } else {
            default
        }
    }
    /// Python's `str(x)`.
    pub fn py_str(&self) -> String {
        match self {
            Py::Null => "None".into(),
            Py::Bool(true) => "True".into(),
            Py::Bool(false) => "False".into(),
            Py::Int(i) => i.to_string(),
            Py::Float(f) => float_repr(*f),
            Py::Str(s) => s.clone(),
            other => other.dumps(), // (lists and dicts never name a box)
        }
    }
    /// `json.dumps(x)`
    pub fn dumps(&self) -> String {
        let mut s = String::new();
        write_value(&mut s, self, None, false, 0);
        s
    }
    /// `json.dumps(x, sort_keys=True)`: the server's test for "the same value".
    pub fn dumps_sorted(&self) -> String {
        let mut s = String::new();
        write_value(&mut s, self, None, true, 0);
        s
    }
    /// `json.dumps(x, indent=n)`
    pub fn dumps_indent(&self, n: usize) -> String {
        let mut s = String::new();
        write_value(&mut s, self, Some(n), false, 0);
        s
    }
    pub fn parse(text: &str) -> Result<Py, String> {
        let mut p = Parser { b: text.as_bytes(), i: 0 };
        p.ws();
        let v = p.value()?;
        p.ws();
        if p.i != p.b.len() {
            return Err(format!("extra data at {}", p.i));
        }
        Ok(v)
    }
    /// From serde_json (for callers holding one); integers stay integers.
    pub fn from_serde(v: &serde_json::Value) -> Py {
        use serde_json::Value as V;
        match v {
            V::Null => Py::Null,
            V::Bool(b) => Py::Bool(*b),
            V::Number(n) => n.as_i64().map(Py::Int).unwrap_or_else(|| Py::Float(n.as_f64().unwrap_or(f64::NAN))),
            V::String(s) => Py::Str(s.clone()),
            V::Array(a) => Py::List(a.iter().map(Py::from_serde).collect()),
            V::Object(o) => Py::Dict(o.iter().map(|(k, v)| (k.clone(), Py::from_serde(v))).collect()),
        }
    }
    pub fn to_serde(&self) -> serde_json::Value {
        serde_json::from_str(&self.dumps()).unwrap_or(serde_json::Value::Null)
    }
}

/// The digits Python's `repr` writes for a finite, non-zero |f|, and the decimal exponent of the first: the shortest
/// that round-trip and, among those, the nearest to f, halves to even (David Gay's `dtoa` mode 0).
///
/// Rust's shortest `{:e}` gives the right length but not always the right last digit: when two strings of that
/// length are exactly as near (common for float32-derived values: 96.362579345703125 is halfway between
/// ...312 and ...313), it can take the odd one (Python "96.36257934570312", `{:e}` "...313"). So: the length from
/// `{:e}`, then the correctly rounded digits at that length (`{:.*e}`, exact, halves to even) when they round-trip.
/// Checked on 100,000 float32-derived values (tools/parity/core_numcheck.py).
fn repr_digits(a: f64) -> (String, i32) {
    let split = |e: &str| -> (String, i32) {
        let (m, x) = e.split_once('e').unwrap();
        let d: String = m.chars().filter(|c| c.is_ascii_digit()).collect();
        (d.trim_end_matches('0').to_string(), x.parse().unwrap())
    };
    let short = format!("{:e}", a);
    let (d, x) = split(&short);
    let near = format!("{:.*e}", d.len().saturating_sub(1), a);
    if near != short && near.parse::<f64>() == Ok(a) {
        let (dn, xn) = split(&near);
        if !dn.is_empty() {
            return (dn, xn);
        }
    }
    (d, x)
}

/// Python's `repr(float)`: the shortest digits that round-trip (the nearest, halves to even: [`repr_digits`]), fixed
/// notation for exponents -4..16.
pub fn float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let neg = f < 0.0;
    let (digits, exp) = repr_digits(f.abs());
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if (-4..16).contains(&exp) {
        if exp < 0 {
            out.push_str("0.");
            for _ in 0..(-exp - 1) {
                out.push('0');
            }
            out.push_str(&digits);
        } else {
            let ip = exp as usize + 1;
            if digits.len() <= ip {
                out.push_str(&digits);
                for _ in digits.len()..ip {
                    out.push('0');
                }
                out.push_str(".0");
            } else {
                out.push_str(&digits[..ip]);
                out.push('.');
                out.push_str(&digits[ip..]);
            }
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(out, "e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
    out
}

/// Python's `round(x, nd)` for floats (correctly rounded, ties to even on the exact binary value).
pub fn py_round(x: f64, nd: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{:.*}", nd, x).parse().unwrap_or(x)
}

/// Python's `round(x)` (to an int, ties to even).
pub fn py_round_int(x: f64) -> i64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        let t = x.trunc();
        return if (t as i64) % 2 == 0 { t as i64 } else { r as i64 };
    }
    r as i64
}

fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    let _ = write!(out, "\\u{:04x}", u);
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn newline(out: &mut String, indent: usize, level: usize) {
    out.push('\n');
    for _ in 0..indent * level {
        out.push(' ');
    }
}

fn write_value(out: &mut String, v: &Py, indent: Option<usize>, sort: bool, level: usize) {
    let item_sep = if indent.is_some() { "," } else { ", " };
    match v {
        Py::Null => out.push_str("null"),
        Py::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Py::Int(i) => {
            let _ = write!(out, "{i}");
        }
        Py::Float(f) => out.push_str(&float_repr(*f)),
        Py::Str(s) => write_str(out, s),
        Py::List(l) => {
            if l.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, x) in l.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                if let Some(n) = indent {
                    newline(out, n, level + 1);
                }
                write_value(out, x, indent, sort, level + 1);
            }
            if let Some(n) = indent {
                newline(out, n, level);
            }
            out.push(']');
        }
        Py::Dict(d) => {
            if d.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            let mut items: Vec<&(String, Py)> = d.0.iter().collect();
            if sort {
                items.sort_by(|a, b| a.0.cmp(&b.0));
            }
            for (i, (k, x)) in items.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                if let Some(n) = indent {
                    newline(out, n, level + 1);
                }
                write_str(out, k);
                out.push_str(": ");
                write_value(out, x, indent, sort, level + 1);
            }
            if let Some(n) = indent {
                newline(out, n, level);
            }
            out.push('}');
        }
    }
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn err<T>(&self, what: &str) -> Result<T, String> {
        Err(format!("{what} at byte {}", self.i))
    }
    fn lit(&mut self, s: &str, v: Py) -> Result<Py, String> {
        if self.b[self.i..].starts_with(s.as_bytes()) {
            self.i += s.len();
            Ok(v)
        } else {
            self.err("bad literal")
        }
    }
    fn value(&mut self) -> Result<Py, String> {
        if self.i >= self.b.len() {
            return self.err("unexpected end");
        }
        match self.b[self.i] {
            b'n' => self.lit("null", Py::Null),
            b't' => self.lit("true", Py::Bool(true)),
            b'f' => self.lit("false", Py::Bool(false)),
            b'N' => self.lit("NaN", Py::Float(f64::NAN)),
            b'I' => self.lit("Infinity", Py::Float(f64::INFINITY)),
            b'"' => Ok(Py::Str(self.string()?)),
            b'[' => {
                self.i += 1;
                let mut l = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Py::List(l));
                }
                loop {
                    self.ws();
                    l.push(self.value()?);
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Py::List(l));
                        }
                        _ => return self.err("expected , or ]"),
                    }
                }
            }
            b'{' => {
                self.i += 1;
                let mut d = Dict::new();
                self.ws();
                if self.b.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Py::Dict(d));
                }
                loop {
                    self.ws();
                    if self.b.get(self.i) != Some(&b'"') {
                        return self.err("expected a key");
                    }
                    let k = self.string()?;
                    self.ws();
                    if self.b.get(self.i) != Some(&b':') {
                        return self.err("expected :");
                    }
                    self.i += 1;
                    self.ws();
                    let v = self.value()?;
                    d.insert(k, v); // (a repeated key: the last wins, in the first one's place, like Python)
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Py::Dict(d));
                        }
                        _ => return self.err("expected , or }"),
                    }
                }
            }
            _ => self.number(),
        }
    }
    fn number(&mut self) -> Result<Py, String> {
        let s = self.i;
        if self.b[self.i..].starts_with(b"-Infinity") {
            self.i += 9;
            return Ok(Py::Float(f64::NEG_INFINITY));
        }
        let mut float = false;
        while self.i < self.b.len() {
            match self.b[self.i] {
                b'0'..=b'9' | b'-' | b'+' => {}
                b'.' | b'e' | b'E' => float = true,
                _ => break,
            }
            self.i += 1;
        }
        let t = std::str::from_utf8(&self.b[s..self.i]).unwrap();
        if t.is_empty() {
            return self.err("unexpected character");
        }
        if !float {
            if let Ok(i) = t.parse::<i64>() {
                return Ok(Py::Int(i));
            }
        }
        t.parse::<f64>().map(Py::Float).map_err(|_| format!("bad number {t:?}"))
    }
    fn hex4(&mut self) -> Result<u32, String> {
        let h = self.b.get(self.i..self.i + 4).ok_or("short \\u escape")?;
        let v = u32::from_str_radix(std::str::from_utf8(h).map_err(|e| e.to_string())?, 16).map_err(|e| e.to_string())?;
        self.i += 4;
        Ok(v)
    }
    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // "
        let mut out = String::new();
        loop {
            let start = self.i;
            while self.i < self.b.len() && self.b[self.i] != b'"' && self.b[self.i] != b'\\' {
                self.i += 1;
            }
            out.push_str(std::str::from_utf8(&self.b[start..self.i]).map_err(|e| e.to_string())?);
            match self.b.get(self.i) {
                None => return self.err("unterminated string"),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                _ => {
                    self.i += 1;
                    let c = *self.b.get(self.i).ok_or("bad escape")?;
                    self.i += 1;
                    match c {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let mut u = self.hex4()?;
                            if (0xd800..0xdc00).contains(&u) && self.b[self.i..].starts_with(b"\\u") {
                                let save = self.i;
                                self.i += 2;
                                let lo = self.hex4()?;
                                if (0xdc00..0xe000).contains(&lo) {
                                    u = 0x10000 + ((u - 0xd800) << 10) + (lo - 0xdc00);
                                } else {
                                    self.i = save;
                                }
                            }
                            out.push(char::from_u32(u).unwrap_or('\u{fffd}'));
                        }
                        _ => return self.err("bad escape"),
                    }
                }
            }
        }
    }
}

/// Read a JSON file as Python's `json.load` would.
pub fn load(path: &std::path::Path) -> Result<Py, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Py::parse(&s).map_err(|e| format!("{}: {e}", path.display()))
}

/// Replace a file whole: write `path.tmp`, then rename it over the old one (a crash never leaves half a file).
pub fn write_atomic(path: &std::path::Path, text: &str) -> std::io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_like_python() {
        for (f, s) in [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (1e16, "1e+16"),
            (123456789012345.6, "123456789012345.6"),
            (-2.5, "-2.5"),
            (1.5e300, "1.5e+300"),
            (100.0, "100.0"),
            (0.82, "0.82"),
            (96.362579345703125, "96.36257934570312"), // a tie at the shortest length: Python takes the even digit
            (5e-324, "5e-324"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (9.5, "9.5"),
        ] {
            assert_eq!(float_repr(f), s);
        }
    }

    #[test]
    fn round_trip_and_formats() {
        let t = r#"{"b": 1, "a": [1.0, "\u00e9\ud83d\ude00", {}, []], "c": null, "d": true}"#;
        let v = Py::parse(t).unwrap();
        assert_eq!(v.dumps(), t);
        assert_eq!(v.dumps_sorted(), r#"{"a": [1.0, "\u00e9\ud83d\ude00", {}, []], "b": 1, "c": null, "d": true}"#);
        assert_eq!(Py::parse("[1,{\"x\":[]}]").unwrap().dumps_indent(1), "[\n 1,\n {\n  \"x\": []\n }\n]");
    }

    #[test]
    fn dict_order_and_equality() {
        let mut d = Dict::new();
        d.insert("a", Py::Int(1));
        d.insert("b", Py::Int(2));
        d.insert("a", Py::Int(3));
        assert_eq!(d.keys().collect::<Vec<_>>(), ["a", "b"]);
        d.remove("a");
        d.insert("a", Py::Int(1));
        assert_eq!(d.keys().collect::<Vec<_>>(), ["b", "a"]);
        assert_eq!(Py::Int(1), Py::Float(1.0));
        assert_eq!(Py::parse(r#"{"x":1,"y":2}"#).unwrap(), Py::parse(r#"{"y":2,"x":1}"#).unwrap());
    }

    #[test]
    fn rounding() {
        assert_eq!(py_round(0.25, 1), 0.2);
        assert_eq!(py_round(0.35, 1), 0.3); // 0.35 is 0.34999...
        assert_eq!(py_round(2.675, 2), 2.67);
        assert_eq!(py_round_int(11.988), 12);
        assert_eq!(py_round_int(2.5), 2);
        assert_eq!(py_round_int(3.5), 4);
    }
}
