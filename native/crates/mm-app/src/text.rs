//! The words, icons and name the pages show, read from the web app's own files so there is one copy of each:
//! `native/web/strings.js` (`STRINGS.en`), `native/web/icons.js` (`ICONS`, Lucide shapes) and `native/web/brand.js`
//! (`BRAND`).
//!
//! [`Text::t`] is strings.js's `t()`: `{placeholders}` filled from vars (`{app}` is the brand's name), and a plural
//! (`{one, other}`) picked by `vars.n` (English rules: 1 is "one"). QML gets the same tables (`ui_text`) for the
//! words it shows as they are; the store uses `t` for text it puts together (counts, sizes, dates).
use anyhow::{anyhow, bail, Context, Result};
use serde_json::{Map, Value};
use std::path::Path;

#[derive(Default, Clone)]
pub struct Text {
    pub strings: Map<String, Value>,
    pub icons: Map<String, Value>,
    pub brand: Map<String, Value>,
}

impl Text {
    pub fn load(root: &Path) -> Result<Text> {
        let read = |f: &str| std::fs::read_to_string(mm_core::paths::web(root).join(f)).with_context(|| format!("web/{f}"));
        let obj = |src: &str, start: &str| -> Result<Map<String, Value>> {
            let i = src.find(start).ok_or_else(|| anyhow!("no '{start}'"))? + start.len();
            let mut p = Parser { s: src.as_bytes(), i };
            p.ws();
            match p.value()? {
                Value::Object(m) => Ok(m),
                _ => bail!("'{start}' isn't an object"),
            }
        };
        Ok(Text {
            strings: obj(&read("strings.js")?, "STRINGS.en =")?,
            icons: obj(&read("icons.js")?, "const ICONS =")?,
            brand: obj(&read("brand.js")?, "const BRAND =")?,
        })
    }

    pub fn name(&self) -> String {
        self.brand.get("name").and_then(Value::as_str).unwrap_or("mojimask").to_string()
    }

    /// strings.js `t(key, vars)`.
    pub fn t(&self, key: &str, vars: &[(&str, String)]) -> String {
        let Some(v) = self.strings.get(key) else { return key.to_string() };
        let n = vars.iter().find(|(k, _)| *k == "n").and_then(|(_, v)| v.parse::<f64>().ok());
        let s = match v {
            Value::String(s) => s.clone(),
            Value::Object(m) => {
                let form = if n == Some(1.0) { "one" } else { "other" };
                m.get(form).or_else(|| m.get("other")).and_then(Value::as_str).unwrap_or(key).to_string()
            }
            _ => return key.to_string(),
        };
        let name = self.name();
        let mut out = String::with_capacity(s.len());
        let b = s.as_bytes();
        let mut i = 0;
        while i < b.len() {
            if b[i] == b'{' {
                if let Some(j) = s[i + 1..].find('}') {
                    let k = &s[i + 1..i + 1 + j];
                    if !k.is_empty() && k.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_') {
                        let rep = vars.iter().find(|(a, _)| *a == k).map(|(_, v)| v.clone()).or_else(|| match k {
                            "app" => Some(name.clone()),
                            "app_file" => Some(name.to_lowercase()),
                            _ => None,
                        });
                        if let Some(r) = rep {
                            out.push_str(&r);
                            i += j + 2;
                            continue;
                        }
                    }
                }
            }
            let ch = s[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
        out
    }

    /// `t` without vars.
    pub fn s(&self, key: &str) -> String {
        self.t(key, &[])
    }
}

/// Vars for [`Text::t`]: `v(&[("n", 3)])`.
pub fn v<T: ToString>(pairs: &[(&'static str, T)]) -> Vec<(&'static str, String)> {
    pairs.iter().map(|(k, x)| (*k, x.to_string())).collect()
}

/// A JavaScript object literal, as these files write them: quoted or bare keys, "double" or 'single' quoted strings,
/// numbers, nested objects and arrays, // and /* */ comments, trailing commas.
struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        loop {
            while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
                self.i += 1;
            }
            if self.s[self.i..].starts_with(b"//") {
                while self.i < self.s.len() && self.s[self.i] != b'\n' {
                    self.i += 1;
                }
            } else if self.s[self.i..].starts_with(b"/*") {
                match self.s[self.i + 2..].windows(2).position(|w| w == b"*/") {
                    Some(p) => self.i += p + 4,
                    None => self.i = self.s.len(),
                }
            } else {
                return;
            }
        }
    }
    fn peek(&self) -> u8 {
        *self.s.get(self.i).unwrap_or(&0)
    }
    fn value(&mut self) -> Result<Value> {
        self.ws();
        match self.peek() {
            b'{' => {
                self.i += 1;
                let mut m = Map::new();
                loop {
                    self.ws();
                    match self.peek() {
                        b'}' => {
                            self.i += 1;
                            return Ok(Value::Object(m));
                        }
                        b'"' | b'\'' => {
                            let k = self.string()?;
                            m.insert(k, self.after_key()?);
                        }
                        c if c.is_ascii_alphanumeric() || c == b'_' || c == b'$' => {
                            let a = self.i;
                            while self.peek().is_ascii_alphanumeric() || self.peek() == b'_' || self.peek() == b'$' {
                                self.i += 1;
                            }
                            let k = String::from_utf8_lossy(&self.s[a..self.i]).into_owned();
                            m.insert(k, self.after_key()?);
                        }
                        c => bail!("unexpected '{}' at {}", c as char, self.i),
                    }
                    self.ws();
                    if self.peek() == b',' {
                        self.i += 1;
                    }
                }
            }
            b'[' => {
                self.i += 1;
                let mut a = vec![];
                loop {
                    self.ws();
                    if self.peek() == b']' {
                        self.i += 1;
                        return Ok(Value::Array(a));
                    }
                    a.push(self.value()?);
                    self.ws();
                    if self.peek() == b',' {
                        self.i += 1;
                    }
                }
            }
            b'"' | b'\'' => Ok(Value::String(self.string()?)),
            _ => {
                let a = self.i;
                while matches!(self.peek(), b'0'..=b'9' | b'.' | b'-' | b'+' | b'e' | b'E' | b'a'..=b'z') {
                    self.i += 1;
                }
                let w = std::str::from_utf8(&self.s[a..self.i])?;
                match w {
                    "true" => Ok(Value::Bool(true)),
                    "false" => Ok(Value::Bool(false)),
                    "null" => Ok(Value::Null),
                    _ => Ok(serde_json::from_str(w).with_context(|| format!("bad value '{w}' at {a}"))?),
                }
            }
        }
    }
    fn after_key(&mut self) -> Result<Value> {
        self.ws();
        if self.peek() != b':' {
            bail!("expected ':' at {}", self.i);
        }
        self.i += 1;
        self.value()
    }
    fn string(&mut self) -> Result<String> {
        let q = self.peek();
        self.i += 1;
        let mut out: Vec<u8> = vec![];
        loop {
            let c = self.peek();
            if c == 0 {
                bail!("unterminated string");
            }
            self.i += 1;
            if c == q {
                return Ok(String::from_utf8(out)?);
            }
            if c == b'\\' {
                let e = self.peek();
                self.i += 1;
                match e {
                    b'n' => out.push(b'\n'),
                    b't' => out.push(b'\t'),
                    b'u' => {
                        let h = std::str::from_utf8(&self.s[self.i..self.i + 4])?;
                        self.i += 4;
                        let ch = char::from_u32(u32::from_str_radix(h, 16)?).unwrap_or('\u{fffd}');
                        out.extend(ch.to_string().as_bytes());
                    }
                    other => out.push(other),
                }
            } else {
                out.push(c);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_the_web_tables() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let t = Text::load(&root).unwrap();
        assert!(t.strings.len() > 300);
        assert!(t.icons.contains_key("chevron-right"));
        assert_eq!(t.t("scan.left", &v(&[("eta", "3 min")])), "about 3 min left");
        assert_eq!(t.t("viz.people.title", &v(&[("n", 1)])), "1 person");
        assert_eq!(t.t("viz.people.title", &v(&[("n", 4)])), "4 people");
        assert_eq!(t.t("legal.line", &[]), format!("{} is free software under the AGPL-3.0, with no warranty.", t.name()));
        assert_eq!(t.t("no.such.key", &[]), "no.such.key");
    }
}
