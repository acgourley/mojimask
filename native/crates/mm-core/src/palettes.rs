//! Emoji sets ("palettes") people can pick in the review tool. Port of `mojimask/palettes.py`.
//!
//! Definitions live in native/web/palettes.json (built into this crate), shared with the page (emoji_set_ui.js). Choice
//! is stored in edits:
//! ```text
//! edits["palette"]        = a preset key from palettes.json, or "custom"
//! edits["palette_custom"] = list of emoji (when "custom"; only Custom is editable)
//! ```
//! An unknown key (an old project's retired preset) falls back to DEFAULT.
//! Assignment hash must match review.html emojiFor(): same person -> same slot in whatever set is active.
use crate::pyjson::{Dict, Py};
use std::sync::OnceLock;

const JSON: &str = include_str!("../../../web/palettes.json");

pub struct Preset {
    pub key: String,
    pub label: String,
    pub default: bool,
    pub emoji: Vec<String>,
}

/// The presets, in palettes.json's order.
pub fn presets() -> &'static [Preset] {
    static P: OnceLock<Vec<Preset>> = OnceLock::new();
    P.get_or_init(|| {
        let v = Py::parse(JSON).expect("native/web/palettes.json");
        v.as_list().expect("a list").iter().map(|p| Preset {
            key: p.get("key").py_str(),
            label: p.get("label").as_str().unwrap_or("").to_string(),
            default: p.get("default").truthy(),
            emoji: p.get("emoji").as_list().map(|l| l.iter().map(Py::py_str).collect()).unwrap_or_default(),
        }).collect()
    })
}

/// The preset marked "default": true in palettes.json (the pages and facts read the same flag).
pub fn default_key() -> &'static str {
    let ps = presets();
    &ps.iter().find(|p| p.default).unwrap_or(&ps[0]).key
}

fn preset(key: &str) -> Option<&'static Preset> {
    presets().iter().find(|p| p.key == key)
}

/// The emoji in use for these edits.
pub fn for_edits(edits: &Py) -> Vec<String> {
    let dk = Py::from(default_key());
    let key = edits.get("palette").or(&dk).py_str();
    if key == "custom" && edits.get("palette_custom").truthy() {
        if let Some(l) = edits.get("palette_custom").as_list() {
            return l.iter().map(Py::py_str).collect();
        }
    }
    preset(&key).or_else(|| preset(default_key())).unwrap().emoji.clone()
}

/// A person's emoji from a palette: the same person gets the same slot in every set.
pub fn pick<'a>(tid: &str, pal: &'a [String]) -> &'a str {
    let t = tid.strip_prefix('@').unwrap_or(tid);
    let h: u64 = t.chars().enumerate().map(|(i, ch)| (i as u64 + 1) * ch as u64).sum();
    &pal[(h % pal.len() as u64) as usize]
}

/// {person tid: emoji} for every person in `tids` (the resolved frames' people), honouring per-person overrides
/// (edits.emoji). Passed to the renderer as its overrides, so it needs no palette knowledge.
pub fn assignment<'a>(tids: impl IntoIterator<Item = &'a str>, edits: &Py) -> Dict {
    let pal = for_edits(edits);
    let mut out = Dict::new();
    for t in tids {
        out.insert(t, Py::from(pick(t, &pal)));
    }
    if let Some(e) = edits.get("emoji").as_dict() {
        for (k, v) in e.iter() {
            out.insert(k, v.clone());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_and_fallbacks() {
        assert_eq!(default_key(), "masquerave");
        let e = Py::parse(r#"{"palette": "retired"}"#).unwrap();
        assert_eq!(for_edits(&e), preset("masquerave").unwrap().emoji);
        let e = Py::parse(r#"{"palette": "custom", "palette_custom": ["A", "B"]}"#).unwrap();
        assert_eq!(for_edits(&e), ["A", "B"]);
        let e = Py::parse(r#"{"palette": "custom", "palette_custom": []}"#).unwrap();
        assert_eq!(for_edits(&e), preset("masquerave").unwrap().emoji); // (Python: PRESETS.get("custom", default))
    }

    #[test]
    fn pick_hash_matches_python() {
        // python: sum((i+1)*ord(ch) for i, ch in enumerate("12")) = 1*49 + 2*50 = 149
        let pal: Vec<String> = (0..10).map(|i| i.to_string()).collect();
        assert_eq!(pick("@12", &pal), "9");
        assert_eq!(pick("12", &pal), "9");
        let a = assignment(["5", "7"], &Py::parse(r#"{"palette": "flock", "emoji": {"7": "X"}}"#).unwrap());
        assert_eq!(a.get("5").unwrap().as_str(), Some(pick("5", &preset("flock").unwrap().emoji)));
        assert_eq!(a.get("7").unwrap().as_str(), Some("X"));
    }
}
