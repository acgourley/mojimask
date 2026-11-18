//! The words, icons and name the pages show: `native/web/strings.json` (every English word, one key each),
//! `native/web/icons.json` (Lucide shapes) and `native/web/brand.json` (the name and its wordmark).
//!
//! [`Text::t`] is strings.json's `t()`: `{placeholders}` filled from vars (`{app}` is the brand's name), and a plural
//! (`{one, other}`) picked by `vars.n` (English rules: 1 is "one"). QML gets the same tables (`ui_text`) for the
//! words it shows as they are; the store uses `t` for text it puts together (counts, sizes, dates).
use anyhow::{bail, Context, Result};
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
        let read = |f: &str| -> Result<Map<String, Value>> {
            let src = std::fs::read_to_string(mm_core::paths::web(root).join(f)).with_context(|| format!("web/{f}"))?;
            match serde_json::from_str(&src).with_context(|| format!("web/{f}"))? {
                Value::Object(m) => Ok(m),
                _ => bail!("web/{f} isn't an object"),
            }
        };
        Ok(Text { strings: read("strings.json")?, icons: read("icons.json")?, brand: read("brand.json")? })
    }

    pub fn name(&self) -> String {
        self.brand.get("name").and_then(Value::as_str).unwrap_or("mojimask").to_string()
    }

    /// strings.json `t(key, vars)`.
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
