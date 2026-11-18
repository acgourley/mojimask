//! JSON as the Python scripts wrote and read it: objects keep their keys in insertion order (Python dicts do;
//! serde_json's Map sorts them unless a workspace-wide feature is turned on), and the printers reproduce
//! `json.dumps(x)` and `json.dumps(x, indent=1)` byte for byte (", " / ": " separators, `,\n` with indent, `[]` and
//! `{}` for empty ones, non-ASCII escaped as \uXXXX). Numbers that Python printed as floats are carried as their
//! already formatted text (`Raw`). The reader is only for dist/relocate.json (what `relocate` wrote), where the
//! order of `sources` decides the order of rows with the same formula name in the notices.

use anyhow::{bail, Result};

#[derive(Clone, Debug)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    /// a number already formatted the way Python would print it (e.g. "12.3")
    Raw(String),
    Str(String),
    Arr(Vec<Value>),
    Obj(Vec<(String, Value)>),
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
}

/// A Python-style string literal for JSON (ensure_ascii=True).
pub fn quote(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            '\u{8}' => o.push_str("\\b"),
            '\u{c}' => o.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    o.push_str(&format!("\\u{:04x}", u));
                }
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn scalar(v: &Value) -> Option<String> {
    Some(match v {
        Value::Null => "null".into(),
        Value::Bool(b) => if *b { "true" } else { "false" }.into(),
        Value::Int(i) => i.to_string(),
        Value::Raw(r) => r.clone(),
        Value::Str(s) => quote(s),
        _ => return None,
    })
}

/// `json.dumps(v)`
pub fn dumps(v: &Value) -> String {
    if let Some(s) = scalar(v) {
        return s;
    }
    match v {
        Value::Arr(a) => format!("[{}]", a.iter().map(dumps).collect::<Vec<_>>().join(", ")),
        Value::Obj(kv) => format!(
            "{{{}}}",
            kv.iter().map(|(k, v)| format!("{}: {}", quote(k), dumps(v))).collect::<Vec<_>>().join(", ")
        ),
        _ => unreachable!(),
    }
}

/// `json.dumps(v, indent=1)`
pub fn dumps_indent1(v: &Value) -> String {
    let mut o = String::new();
    indent_into(v, 1, &mut o);
    o
}

fn indent_into(v: &Value, level: usize, o: &mut String) {
    if let Some(s) = scalar(v) {
        o.push_str(&s);
        return;
    }
    let nl = format!("\n{}", " ".repeat(level));
    let close = format!("\n{}", " ".repeat(level - 1));
    match v {
        Value::Arr(a) if a.is_empty() => o.push_str("[]"),
        Value::Obj(kv) if kv.is_empty() => o.push_str("{}"),
        Value::Arr(a) => {
            o.push('[');
            for (i, x) in a.iter().enumerate() {
                o.push_str(if i == 0 { "" } else { "," });
                o.push_str(&nl);
                indent_into(x, level + 1, o);
            }
            o.push_str(&close);
            o.push(']');
        }
        Value::Obj(kv) => {
            o.push('{');
            for (i, (k, x)) in kv.iter().enumerate() {
                o.push_str(if i == 0 { "" } else { "," });
                o.push_str(&nl);
                o.push_str(&quote(k));
                o.push_str(": ");
                indent_into(x, level + 1, o);
            }
            o.push_str(&close);
            o.push('}');
        }
        _ => unreachable!(),
    }
}

/// Parse JSON keeping object keys in file order. Numbers come back as `Raw` (their text) unless they are integers.
pub fn parse(s: &str) -> Result<Value> {
    let b = s.as_bytes();
    let mut i = 0;
    let v = parse_value(b, &mut i)?;
    skip_ws(b, &mut i);
    if i != b.len() {
        bail!("trailing data in JSON at byte {i}");
    }
    Ok(v)
}

fn skip_ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && matches!(b[*i], b' ' | b'\n' | b'\r' | b'\t') {
        *i += 1;
    }
}

fn parse_value(b: &[u8], i: &mut usize) -> Result<Value> {
    skip_ws(b, i);
    let Some(&c) = b.get(*i) else { bail!("unexpected end of JSON") };
    match c {
        b'{' => {
            *i += 1;
            let mut kv = Vec::new();
            skip_ws(b, i);
            if b.get(*i) == Some(&b'}') {
                *i += 1;
                return Ok(Value::Obj(kv));
            }
            loop {
                skip_ws(b, i);
                let k = parse_string(b, i)?;
                skip_ws(b, i);
                if b.get(*i) != Some(&b':') {
                    bail!("expected ':' at byte {}", *i);
                }
                *i += 1;
                let v = parse_value(b, i)?;
                kv.push((k, v));
                skip_ws(b, i);
                match b.get(*i) {
                    Some(b',') => *i += 1,
                    Some(b'}') => {
                        *i += 1;
                        return Ok(Value::Obj(kv));
                    }
                    _ => bail!("expected ',' or '}}' at byte {}", *i),
                }
            }
        }
        b'[' => {
            *i += 1;
            let mut a = Vec::new();
            skip_ws(b, i);
            if b.get(*i) == Some(&b']') {
                *i += 1;
                return Ok(Value::Arr(a));
            }
            loop {
                a.push(parse_value(b, i)?);
                skip_ws(b, i);
                match b.get(*i) {
                    Some(b',') => *i += 1,
                    Some(b']') => {
                        *i += 1;
                        return Ok(Value::Arr(a));
                    }
                    _ => bail!("expected ',' or ']' at byte {}", *i),
                }
            }
        }
        b'"' => Ok(Value::Str(parse_string(b, i)?)),
        b't' if b[*i..].starts_with(b"true") => {
            *i += 4;
            Ok(Value::Bool(true))
        }
        b'f' if b[*i..].starts_with(b"false") => {
            *i += 5;
            Ok(Value::Bool(false))
        }
        b'n' if b[*i..].starts_with(b"null") => {
            *i += 4;
            Ok(Value::Null)
        }
        _ => {
            let st = *i;
            while *i < b.len() && matches!(b[*i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                *i += 1;
            }
            if st == *i {
                bail!("bad JSON at byte {st}");
            }
            let t = std::str::from_utf8(&b[st..*i])?;
            Ok(t.parse::<i64>().map(Value::Int).unwrap_or_else(|_| Value::Raw(t.to_string())))
        }
    }
}

fn parse_string(b: &[u8], i: &mut usize) -> Result<String> {
    if b.get(*i) != Some(&b'"') {
        bail!("expected a string at byte {}", *i);
    }
    *i += 1;
    let mut units: Vec<u16> = Vec::new();
    let mut out = String::new();
    let flush = |units: &mut Vec<u16>, out: &mut String| {
        if !units.is_empty() {
            out.push_str(&String::from_utf16_lossy(units));
            units.clear();
        }
    };
    loop {
        let Some(&c) = b.get(*i) else { bail!("unterminated JSON string") };
        match c {
            b'"' => {
                *i += 1;
                flush(&mut units, &mut out);
                return Ok(out);
            }
            b'\\' => {
                let e = *b.get(*i + 1).ok_or_else(|| anyhow::anyhow!("bad escape"))?;
                *i += 2;
                let ch = match e {
                    b'"' => '"',
                    b'\\' => '\\',
                    b'/' => '/',
                    b'b' => '\u{8}',
                    b'f' => '\u{c}',
                    b'n' => '\n',
                    b'r' => '\r',
                    b't' => '\t',
                    b'u' => {
                        let h = std::str::from_utf8(b.get(*i..*i + 4).unwrap_or_default())?;
                        units.push(u16::from_str_radix(h, 16)?);
                        *i += 4;
                        continue;
                    }
                    _ => bail!("bad escape"),
                };
                flush(&mut units, &mut out);
                out.push(ch);
            }
            _ => {
                flush(&mut units, &mut out);
                // copy one UTF-8 character
                let len = match c {
                    0x00..=0x7f => 1,
                    0xc0..=0xdf => 2,
                    0xe0..=0xef => 3,
                    _ => 4,
                };
                out.push_str(std::str::from_utf8(&b[*i..*i + len])?);
                *i += len;
            }
        }
    }
}

/// Python's `repr()` of a list of strings, e.g. `['a', "b'c"]` (used in notices' messages).
pub fn py_repr_list(items: &[String]) -> String {
    let r: Vec<String> = items
        .iter()
        .map(|s| {
            let q = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
            let mut o = String::new();
            o.push(q);
            for c in s.chars() {
                match c {
                    '\\' => o.push_str("\\\\"),
                    '\n' => o.push_str("\\n"),
                    '\r' => o.push_str("\\r"),
                    '\t' => o.push_str("\\t"),
                    c if c == q => {
                        o.push('\\');
                        o.push(c);
                    }
                    c if (c as u32) < 0x20 || c as u32 == 0x7f => o.push_str(&format!("\\x{:02x}", c as u32)),
                    c => o.push(c),
                }
            }
            o.push(q);
            o
        })
        .collect();
    format!("[{}]", r.join(", "))
}

/// Python's `round(x, 1)` printed by json/repr, and `f"{x:.1f}"`: both are the correctly rounded one-decimal text
/// (ties to even on the exact binary value), which Rust's `{:.1}` also gives.
pub fn f1(x: f64) -> String {
    format!("{x:.1}")
}
