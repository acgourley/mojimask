//! `mm-pack sizes APP`: the bundle's size by part, in MB (bytes of regular files, symlinks not counted), as JSON:
//! the total, each file in MacOS/, Frameworks/ grouped (Qt, ICU, GStreamer, GLib with gettext and libffi, the
//! rest), each entry of PlugIns/ and Resources/, the 15 largest frameworks, and the code alone (all but the models
//! and the sample clip). make_app.sh keeps it as dist/sizes.json, to see what a change costs.

use crate::fsutil::*;
use crate::json::{dumps_indent1, f1, Value};
use anyhow::Result;

fn size(path: &str) -> u64 {
    if is_file(path) && !is_link(path) {
        return std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    }
    let mut t = 0;
    for (d, fs) in walk(path) {
        for f in fs {
            let p = join(&d, &f);
            if !is_link(&p) {
                t += std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            }
        }
    }
    t
}

fn mb(b: u64) -> Value {
    Value::Raw(f1(b as f64 / 1e6))
}

fn sorted_list(d: &str) -> Result<Vec<String>> {
    let mut v = listdir(d)?;
    v.sort();
    Ok(v)
}

fn per_entry(d: &str) -> Result<Value> {
    Ok(Value::Obj(sorted_list(d)?.into_iter().map(|f| {
        let s = size(&join(d, &f));
        (f, mb(s))
    }).collect()))
}

fn is_glib(f: &str) -> bool {
    ["glib", "gobject", "gio", "gmodule", "gthread", "intl", "ffi", "pcre2-8"]
        .iter()
        .any(|p| f.starts_with(&format!("lib{p}")))
}

pub fn run(app: &str) -> Result<()> {
    let c = join(app, "Contents");
    let fw = join(&c, "Frameworks");
    let names = ["qt", "icu", "gstreamer", "glib/gettext/ffi", "other libs"];
    let mut groups = [0u64; 5];
    let mut per: Vec<(String, f64, String)> = Vec::new();
    for f in sorted_list(&fw)? {
        let s = size(&join(&fw, &f));
        let r = f1(s as f64 / 1e6);
        per.push((f.clone(), r.parse().unwrap_or(0.0), r));
        let g = if f.starts_with("Qt") {
            0
        } else if f.starts_with("libicu") {
            1
        } else if f.starts_with("libgst") || f.starts_with("liborc") {
            2
        } else if is_glib(&f) {
            3
        } else {
            4
        };
        groups[g] += s;
    }
    let r = join(&c, "Resources");
    // sorted(per.items(), key=-mb)[:15]: a stable sort by the rounded value, largest first
    per.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let largest: Vec<(String, Value)> = per.into_iter().take(15).map(|(f, _, r)| (f, Value::Raw(r))).collect();
    let total = size(&c);
    let out = Value::Obj(vec![
        ("total".into(), mb(total)),
        ("MacOS".into(), per_entry(&join(&c, "MacOS"))?),
        ("Frameworks".into(), Value::Obj(names.iter().zip(groups).map(|(k, v)| (k.to_string(), mb(v))).collect())),
        ("PlugIns".into(), per_entry(&join(&c, "PlugIns"))?),
        ("Resources".into(), per_entry(&r)?),
        ("largest_frameworks".into(), Value::Obj(largest)),
        (
            "code (all but models, clip)".into(),
            mb(total - size(&join(&r, "models")) - size(&join(&r, "test_clips"))),
        ),
    ]);
    println!("{}", dumps_indent1(&out));
    Ok(())
}
