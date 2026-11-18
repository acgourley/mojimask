//! The Editor on a project (web/review.html and its plug-ins: light_ui.js, emoji_set_ui.js, pose_ui.js, depth_ui.js).
//!
//! **One writer.** The project's edits (edits.json) and facts (facts.json) have one owner: its review server, which
//! holds the project's `.server.lock` (review_server.py, "the edits document"). The Editor never writes either file.
//! Every change goes to that server, exactly as review.html sends it:
//!   - plain edits as ops on paths (`/api/ops`: trims, cuts, manual covers, cover_style, light, palette, head_pose);
//!   - who's who and what's a face as facts (`/api/facts`: face, show, emoji);
//!   - covers placed or split by hand through the server's own rules (`/api/propagate`, `/api/split_cover`,
//!     `/api/undo_correction`).
//! Every reply is the server's new document (doc_reply): the store takes it as the truth, then fetches the draw
//! records again (records.rs). Undo and redo are the server's history, shared with Decisions and People.
//! (Switching to mm_core::edits::Project would mean stopping the project's Python server from writing at the same
//! time; that comes with P5, when the engine no longer needs it.)
//!
//! What the Editor draws is the export (records.rs + mm-render's Drawer); the boxes, labels and selection are drawn
//! by QML over it from `editor.view`, which this module makes.
use crate::decide::{emoji_for, idstr, palette_for};
use crate::library::{queue, sidecar};
use crate::text::v;
use crate::{log, Shared};
use anyhow::{anyhow, bail, Result};
use mm_core::cover::FrameRecord;
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

/// The Editor's own choices (not saved with the project).
#[derive(Default)]
pub struct Ui {
    /// the selected box (its track piece) and its person (canonical id)
    pub piece: Option<String>,
    pub person: Option<String>,
    /// V: boxes instead of covers
    pub boxes: bool,
    /// D, B, ?
    pub depth_view: bool,
    pub pose_view: bool,
    pub keys_open: bool,
    /// edits.light while a lighting control is being dragged (drawn live, saved on release)
    pub live_light: Option<Value>,
    pub depth: HashMap<usize, Value>,
    pub pose: HashMap<usize, Value>,
    pub error: String,
    /// the light estimate's /api/light status (ready, progress, error) and its per-frame scene colour
    pub light_status: Value,
    /// developer mode: what Export labels wrote
    pub labels: String,
    /// the timeline's columns, made once per revision of the edits
    pub timeline: Option<(Value, Value)>,
}

/// "orb" (the 3-D head, mm_core::orb) is offered only in developer mode
const STYLES: [&str; 6] = ["emoji", "blur", "pixelate", "solid", "portrait", "orb"];
pub const SLIDERS: [(&str, f64, f64, f64); 7] = [
    ("mix", 0.0, 1.0, 0.05),
    ("level", 0.3, 2.0, 0.05),
    ("ramp", 0.0, 1.5, 0.05),
    ("amb", 0.0, 0.6, 0.01),
    ("saturation", 0.0, 1.0, 0.05),
    ("black", 0.0, 3.0, 0.05),
    ("shutter", 0.0, 2.0, 0.05),
];

fn light_defaults() -> Value {
    json!({"enabled": true, "level": 1.0, "mix": 1.0, "amb": 0.15, "ramp": 0.8, "floor": 0.12, "light_colour": null,
           "grain": true, "blur": true, "shutter": 1.0, "saturation": 1.0, "black": 1.0})
}

/// lighting.settings(edits): the defaults with edits.light on top.
pub fn light_settings(light: &Value) -> Value {
    let mut s = light_defaults();
    if let (Some(o), Some(l)) = (s.as_object_mut(), light.as_object()) {
        for (k, x) in l {
            o.insert(k.clone(), x.clone());
        }
    }
    s
}

/// The record as drawn now: a lighting control being dragged changes the settings live (lighting's sums are in
/// FrameRecord::rows, so the preview is the export's).
pub fn live_record(sh: &Shared, rec: &FrameRecord) -> FrameRecord {
    let mut r = rec.clone();
    let (live, boxes) = {
        let l = sh.lib.lock().unwrap();
        (l.ed.live_light.clone(), l.ed.boxes)
    };
    if boxes {
        r.covers.clear(); // V: the frame without its covers, the boxes on top
    }
    if let Some(l) = live {
        let s = light_settings(&l);
        if let Ok(ls) = serde_json::from_value::<mm_core::colour::LightSettings>(s.clone()) {
            if let Some(fl) = r.light.as_mut() {
                fl.settings = ls.clone();
            }
            // without lighting: the emoji's own saturation (render_setup's `saturation`)
            r.saturation = ls.saturation.clamp(0.0, 1.0);
        }
    }
    r
}

// ---------------------------------------------------------------------------------------------------- the view
fn base_of(p: &str) -> String {
    match p.find(['~', '@']) {
        Some(i) => p[..i].to_string(),
        None => p.to_string(),
    }
}

fn pid_of(tid: &str) -> Option<&str> {
    tid.strip_prefix('@')
}

fn person_label(tx: &crate::text::Text, tid: &str, edits: &Value) -> String {
    if let Some(pid) = pid_of(tid) {
        let name = edits["people"][pid]["name"].as_str().unwrap_or("");
        if !name.is_empty() {
            return name.to_string();
        }
        return tx.t("edit.person", &v(&[("n", pid)]));
    }
    tx.t("edit.unnamed", &v(&[("id", tid)]))
}

fn is_hidden_piece(edits: &Value, p: &str) -> bool {
    edits["not_face"].as_object().is_some_and(|o| o.contains_key(p))
        || edits["not_real"].as_array().is_some_and(|a| a.iter().any(|x| idstr(x) == p))
}

/// (tid, cx, cy, side, kind, piece, shown) of frame t's covers (api/init's resolved)
fn resolved_at(doc: &Value, t: usize) -> Vec<(String, f64, f64, f64, String, String, bool)> {
    doc["resolved"][t]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|x| {
            let a = x.as_array()?;
            Some((idstr(&a[0]), a[1].as_f64()?, a[2].as_f64()?, a[3].as_f64()?, a[4].as_str().unwrap_or("obs").to_string(),
                  idstr(&a[5]), a.get(6).and_then(Value::as_bool).unwrap_or(false)))
        })
        .collect()
}

/// (tid, cx, cy, side, kind, piece) of frame t's hidden boxes (answered Not a face / Not a real person)
fn hidden_at(doc: &Value, t: usize) -> Vec<(String, f64, f64, f64, String, String)> {
    doc["hidden"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|x| {
            let a = x.as_array()?;
            if a.first()?.as_u64()? as usize != t {
                return None;
            }
            Some((idstr(&a[1]), a[2].as_f64()?, a[3].as_f64()?, a[4].as_f64()?, a[5].as_str().unwrap_or("obs").to_string(), idstr(&a[6])))
        })
        .collect()
}

fn person_hue(tid: &str) -> String {
    let mut h: u32 = 0;
    for c in tid.encode_utf16() {
        h = h.wrapping_mul(31).wrapping_add(c as u32);
    }
    format!("hsl({}, 75%, 62%)", h % 360)
}

fn hsl_hex(s: &str) -> String {
    // "hsl(H, 75%, 62%)" -> #rrggbb (QML has no hsl() colour strings)
    let h = s.trim_start_matches("hsl(").split(',').next().and_then(|x| x.trim().parse::<f64>().ok()).unwrap_or(0.0);
    let (sat, l): (f64, f64) = (0.75, 0.62);
    let a = sat * l.min(1.0 - l);
    let f = |n: f64| {
        let k = (n + h / 30.0) % 12.0;
        l - a * (k - 3.0).min(9.0 - k).clamp(-1.0, 1.0)
    };
    format!("#{:02x}{:02x}{:02x}", (f(0.0) * 255.0).round() as u8, (f(8.0) * 255.0).round() as u8, (f(4.0) * 255.0).round() as u8)
}

fn person_pieces(doc: &Value, person: &str) -> Vec<String> {
    let mut set = BTreeSet::new();
    for f in doc["resolved"].as_array().into_iter().flatten() {
        for x in f.as_array().into_iter().flatten() {
            if idstr(&x[0]) == person {
                set.insert(idstr(&x[5]));
            }
        }
    }
    set.into_iter().collect()
}

fn is_shown(doc: &Value, person: &str) -> bool {
    let edits = &doc["edits"];
    if let Some(pid) = pid_of(person) {
        if let Some(p) = edits["people"].get(pid) {
            return p["show"].as_bool().unwrap_or(false);
        }
    }
    let ps = person_pieces(doc, person);
    let shown: BTreeSet<String> = edits["shown"].as_array().into_iter().flatten().map(idstr).collect();
    !ps.is_empty() && ps.iter().all(|p| shown.contains(p))
}

/// Keep the selection on the selected box's CURRENT person (giving it an emoji can make it a named person "@3").
fn refresh_selected(doc: &Value, ui: &mut Ui) {
    let Some(piece) = ui.piece.clone() else { return };
    for f in doc["resolved"].as_array().into_iter().flatten() {
        for x in f.as_array().into_iter().flatten() {
            if idstr(&x[5]) == piece {
                ui.person = Some(idstr(&x[0]));
                return;
            }
        }
    }
    for x in doc["hidden"].as_array().into_iter().flatten() {
        if idstr(&x[6]) == piece {
            ui.person = Some(idstr(&x[1]));
            return;
        }
    }
}

fn fmt1(x: f64) -> String {
    format!("{x:.1}")
}

/// Make `editor.view` for frame t from the project's document and the Editor's choices.
pub fn rebuild(sh: &Shared) {
    let (t, base) = {
        let st = sh.st.lock().unwrap();
        (if st.editor.shown >= 0 { st.editor.shown as usize } else { st.editor.t }, st.project.base.clone())
    };
    let view = {
        let mut guard = sh.lib.lock().unwrap();
        let l = &mut *guard;
        if l.pid.is_empty() || l.doc.is_null() {
            return;
        }
        refresh_selected(&l.doc, &mut l.ed);
        if l.ed.timeline.as_ref().is_none_or(|(r, _)| *r != l.doc["rev"]) {
            l.ed.timeline = Some((l.doc["rev"].clone(), timeline(&l.doc)));
        }
        make_view(sh, &l.doc, &l.ed, &l.decisions, &l.palettes, t, &base)
    };
    sh.st.lock().unwrap().editor.view = view;
    sh.changed(&["editor.view"]);
}

fn make_view(sh: &Shared, doc: &Value, ui: &Ui, decs: &[Value], palettes: &Value, t: usize, base: &str) -> Value {
    let tx = &sh.text;
    let edits = &doc["edits"];
    let edits_live = {
        // a lighting drag: the panel shows the values being dragged
        let mut e = edits.clone();
        if let Some(l) = &ui.live_light {
            e["light"] = l.clone();
        }
        e
    };
    let n = doc["n"].as_u64().unwrap_or(1) as usize;
    let fps = doc["fps"].as_f64().unwrap_or(30.0);
    let scale = doc["scale"].as_f64().unwrap_or(1.25);
    let style = edits["cover_style"].as_str().unwrap_or("emoji").to_string();
    let sel_piece = ui.piece.clone();
    let sel_person = ui.person.clone();
    // the boxes: small (far) first, large (near) on top
    let mut rs = resolved_at(doc, t);
    rs.sort_by(|a, b| a.3.total_cmp(&b.3));
    let mut boxes = vec![];
    for (tid, cx, cy, s, kind, piece, shown_p) in &rs {
        let sel = sel_piece.as_deref() == Some(piece.as_str()) || (sel_person.as_deref() == Some(tid.as_str()) && sel_piece.is_none());
        let same = !sel && sel_person.as_deref() == Some(tid.as_str());
        let label = if *shown_p {
            tx.t("edit.box.shown", &v(&[("piece", piece.clone())]))
        } else if ui.boxes || sel {
            format!("{piece}{}{}", if tid != piece { format!(" ({tid})") } else { String::new() },
                    if kind == "coast" { format!(" {}", tx.t("edit.box.held", &[])) } else { String::new() })
        } else {
            String::new()
        };
        let manual = edits["manual"].get(piece.as_str()).is_some();
        boxes.push(json!({"tid": tid, "piece": piece, "x": cx, "y": cy, "s": s * scale, "kind": kind, "shown": shown_p,
                          "sel": sel, "same": same, "label": label, "hidden": false, "manual": manual,
                          "emoji": emoji_for(tid, edits, palettes)}));
    }
    for (tid, cx, cy, s, kind, piece) in hidden_at(doc, t) {
        let sel = sel_piece.as_deref() == Some(piece.as_str());
        if edits["not_face"][piece.as_str()].as_str() == Some("decision") && !ui.boxes && !sel {
            continue; // V shows them
        }
        let mine = sel_person.as_deref() == Some(tid.as_str());
        let not_real = edits["not_real"].as_array().is_some_and(|a| a.iter().any(|x| idstr(x) == piece));
        let label = tx.t("edit.box.not_face", &[]) + &if not_real { format!(" {}", tx.t("edit.box.not_real", &[])) } else { String::new() };
        boxes.push(json!({"tid": tid, "piece": piece, "x": cx, "y": cy, "s": s * scale, "kind": kind, "shown": false,
                          "sel": sel, "same": mine, "label": label, "hidden": true, "manual": false,
                          "colour": hsl_hex(&person_hue(&tid)), "opacity": if sel { 1.0 } else if mine { 0.85 } else { 0.55 }}));
    }
    let count = rs.len();
    let note = format!("{} · {}",
        if style == "emoji" { tx.t("edit.count_emoji", &v(&[("n", count)])) } else { tx.t("edit.count_covers", &v(&[("n", count)])) },
        if ui.boxes { tx.t("edit.pic.no_covers", &[]) } else { tx.t("edit.pic.export", &[]) });
    let photo = doc["photo"].as_bool().unwrap_or(false);
    let info = if photo { String::new() } else {
        tx.t("edit.frame_info", &v(&[("frame", t.to_string()), ("last", (n.max(1) - 1).to_string()), ("s", format!("{:.2}", t as f64 / fps))]))
    };
    let styles: Vec<Value> = STYLES
        .iter()
        .filter(|k| **k != "orb" || sh.dev() || **k == style)
        .map(|k| {
            let tip = match *k {
                "solid" => tx.t("edit.style_tip.solid", &[]),
                "portrait" => tx.t("edit.style_tip.portrait", &[]),
                _ => tx.t(&format!("style.{k}.hint"), &[]),
            };
            json!({"k": k, "label": tx.t(&format!("style.{k}"), &[]), "tip": tip, "on": *k == style})
        })
        .collect();
    let sel = match (&sel_piece, &sel_person) {
        (Some(p), Some(person)) => selection(sh, doc, decs, palettes, p, person, t),
        _ => Value::Null,
    };
    // the emoji preset (emoji_set_ui.js)
    let presets = palettes.as_array().cloned().unwrap_or_default();
    let custom = edits["palette"].as_str() == Some("custom") && edits["palette_custom"].as_array().is_some_and(|a| !a.is_empty());
    let preset_key = {
        let k = edits["palette"].as_str().unwrap_or("");
        if presets.iter().any(|p| p["key"] == k) {
            k.to_string()
        } else {
            presets.iter().find(|p| p["default"].as_bool() == Some(true)).or(presets.first()).and_then(|p| p["key"].as_str()).unwrap_or("").to_string()
        }
    };
    let cur = palette_for(edits, palettes);
    let n_over = edits["emoji"].as_object().map(|o| o.len()).unwrap_or(0);
    let mut options: Vec<Value> = presets.iter().map(|p| json!({"key": p["key"], "label": p["label"]})).collect();
    options.push(json!({"key": "custom", "label": tx.t("emojiset.custom", &[])}));
    let emojiset = json!({
        "options": options, "current": if custom { "custom".to_string() } else { preset_key.clone() },
        "hint": if custom { tx.t("emojiset.custom_hint", &[]) } else {
            presets.iter().find(|p| p["key"] == preset_key.as_str()).and_then(|p| p["hint"].as_str()).unwrap_or("").to_string() },
        "chips": cur, "custom": custom, "overrides": n_over,
        "overrides_text": if n_over > 0 { tx.t("emojiset.overrides", &v(&[("n", n_over)])) } else { String::new() },
        "head_pose": edits["head_pose"].as_bool().unwrap_or(false),
    });
    // lighting (light_ui.js)
    let ls = light_settings(&edits_live["light"]);
    let sliders: Vec<Value> = SLIDERS
        .iter()
        .map(|(k, lo, hi, step)| {
            let val = if *k == "shutter" && ls["blur"].as_bool() == Some(false) { 0.0 } else { ls[*k].as_f64().unwrap_or(0.0) };
            let pct = ["mix", "saturation", "black"].contains(k);
            json!({"k": k, "label": tx.t(&format!("light.{k}"), &[]), "lo": lo, "hi": hi, "step": step, "value": val,
                   "text": if pct { format!("{}%", (val * 100.0).round()) } else { format!("{val:.2}") },
                   "tip": tip(tx, k)})
        })
        .collect();
    let lstat = &ui.light_status;
    let status = if lstat["ready"].as_bool() == Some(true) { String::new() }
        else if let Some(e) = lstat["error"].as_str() { tx.t("light.err.failed", &v(&[("error", e)])) }
        else { tx.t("light.estimating", &v(&[("pct", (lstat["progress"].as_f64().unwrap_or(0.0) * 100.0).round())])) };
    let swatch = scene_swatch(lstat, t);
    let light = json!({"enabled": ls["enabled"], "grain": ls["grain"], "fixed": !ls["light_colour"].is_null(),
                       "colour": ls["light_colour"], "sliders": sliders, "status": status, "ready": lstat["ready"],
                       "shown": style == "emoji" || style == "portrait" || style == "orb", "swatch": swatch,
                       "tips": {"enabled": tip(tx, "enabled"), "light_colour": tip(tx, "light_colour"), "grain": tip(tx, "grain")}});
    let depth = if ui.depth_view { ui.depth.get(&t).cloned().unwrap_or(Value::Null) } else { Value::Null };
    let pose = if ui.pose_view { ui.pose.get(&t).cloned().unwrap_or(Value::Null) } else { Value::Null };
    json!({
        "t": t, "n": n, "fps": fps, "scale": scale, "photo": photo, "w": doc["w"], "h": doc["h"],
        "boxes": boxes, "style": style, "styles": styles, "view": if ui.boxes { "boxes" } else { "covers" },
        "depth_view": ui.depth_view, "pose_view": ui.pose_view, "depth_on": doc["depth"].as_bool().unwrap_or(false),
        "keys_open": ui.keys_open, "sel": sel, "emojiset": emojiset, "light": light,
        "frame_info": info, "frame_note": note, "timeline": ui.timeline.as_ref().map(|x| x.1.clone()).unwrap_or(Value::Null),
        "depth": depth, "depth_pic": if ui.depth_view { format!("{base}/api/depth/{t}.jpg") } else { String::new() },
        "pose": pose, "error": ui.error, "dev": sh.dev(), "labels": ui.labels,
        "portrait": style == "portrait", "portrait_pic": format!("{base}/api/portrait?k={}", edits["portrait"].as_str().unwrap_or("")),
    })
}

/// The timeline: per column (up to 200), its busiest frame's boxes by kind, detected first (review.html drawTimeline).
fn timeline(doc: &Value) -> Value {
    let n = doc["n"].as_u64().unwrap_or(1) as usize;
    let cols = n.clamp(1, 200);
    let per = n as f64 / cols as f64;
    let order = ["obs", "coast", "manual"];
    let mut tl = vec![];
    let mut maxc = 4;
    for i in 0..cols {
        let mut best: Vec<String> = vec![];
        let lo = (i as f64 * per).floor() as usize;
        let hi = (((i + 1) as f64 * per).floor() as usize).clamp(lo + 1, n);
        for u in lo..hi {
            let ks: Vec<String> = doc["resolved"][u].as_array().into_iter().flatten().map(|b| b[4].as_str().unwrap_or("obs").to_string()).collect();
            if ks.len() > best.len() {
                best = ks;
            }
        }
        let mut ks: Vec<String> = best.into_iter().map(|k| if order.contains(&k.as_str()) { k } else { "obs".into() }).collect();
        ks.sort_by_key(|k| order.iter().position(|o| o == k));
        maxc = maxc.max(ks.len());
        tl.push(ks);
    }
    json!({"cols": tl, "max": maxc})
}

fn tip(tx: &crate::text::Text, k: &str) -> Value {
    json!({"title": tx.t(&format!("light.tip.{k}.title"), &[]), "what": tx.t(&format!("light.tip.{k}.what"), &[]),
           "ends": tx.t(&format!("light.tip.{k}.ends"), &[])})
}

/// This frame's scene light as a colour (light_ui.js frameLight -> sRGB), "" before the estimate.
fn scene_swatch(lstat: &Value, t: usize) -> String {
    let Some(sk) = lstat["scene_key"].as_array() else { return String::new() };
    if sk.is_empty() {
        return String::new();
    }
    let k = &sk[t.min(sk.len() - 1)];
    let c: Vec<f64> = k.as_array().into_iter().flatten().filter_map(Value::as_f64).collect();
    if c.len() < 3 {
        return String::new();
    }
    let m = c.iter().cloned().fold(0.0, f64::max).max(1e-9);
    let srgb = |x: f64| {
        let x = (x / m).clamp(0.0, 1.0);
        255.0 * if x <= 0.0031308 { x * 12.92 } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
    };
    // light_ui.js's scene_key is RGB
    format!("#{:02x}{:02x}{:02x}", srgb(c[0]) as u8, srgb(c[1]) as u8, srgb(c[2]) as u8)
}

/// The panel's Selection block (review.html renderSelApp).
fn selection(sh: &Shared, doc: &Value, decs: &[Value], palettes: &Value, piece: &str, person: &str, t: usize) -> Value {
    let tx = &sh.text;
    let edits = &doc["edits"];
    let fps = doc["fps"].as_f64().unwrap_or(30.0);
    let n = doc["n"].as_u64().unwrap_or(1) as usize;
    let who = person_label(tx, person, edits);
    let man = edits["manual"].get(piece);
    let hidden = is_hidden_piece(edits, piece);
    let mut frames = BTreeSet::new();
    for (u, f) in doc["resolved"].as_array().into_iter().flatten().enumerate() {
        if f.as_array().into_iter().flatten().any(|x| idstr(&x[5]) == piece) {
            frames.insert(u);
        }
    }
    for x in doc["hidden"].as_array().into_iter().flatten() {
        if idstr(&x[6]) == piece {
            frames.insert(x[0].as_u64().unwrap_or(0) as usize);
        }
    }
    let span = match (frames.first(), frames.last()) {
        (Some(a), Some(b)) => tx.t("edit.sel.span", &v(&[("from", fmt1(*a as f64 / fps)), ("to", fmt1(*b as f64 / fps))])),
        _ => String::new(),
    };
    let not_real = edits["not_real"].as_array().is_some_and(|a| a.iter().any(|x| idstr(x) == piece));
    let why = match edits["not_face"][piece].as_str() {
        Some(w) if w != "you" => {
            let k = format!("edit.not_face_why.{w}");
            let txt = tx.t(&k, &[]);
            format!(" {}", tx.t("edit.sel.computers_call", &v(&[("why", if txt == k { w.to_string() } else { txt })])))
        }
        _ => String::new(),
    };
    let trim = edits["trims"][piece].as_array().map(|r| {
        let (a, b) = (r[0].as_f64().unwrap_or(0.0), r[1].as_f64().unwrap_or(0.0));
        let (start, end) = (fmt1(a / fps), fmt1(b / fps));
        let (s, e) = (a > 0.0, (b as usize) < n - 1);
        if s && e { tx.t("edit.trim.both", &v(&[("start", start), ("end", end)])) }
        else if s { tx.t("edit.trim.started", &v(&[("start", start)])) }
        else if e { tx.t("edit.trim.ended", &v(&[("end", end)])) }
        else { tx.t("edit.trim.shortened", &[]) }
    });
    let keys: Vec<usize> = man
        .and_then(|m| m["keys"].as_object())
        .map(|o| { let mut k: Vec<usize> = o.keys().filter_map(|x| x.parse().ok()).collect(); k.sort(); k })
        .unwrap_or_default();
    let here = keys.contains(&t);
    let corr = if man.is_some() {
        json!({"times": keys.iter().map(|f| json!({"t": f, "text": tx.t("unit.s", &v(&[("n", fmt1(*f as f64 / fps))]))})).collect::<Vec<_>>(),
               "prefix": tx.t("edit.corr.set_at", &v(&[("times", "")])),
               "can_undo": here && keys.len() > 1, "first": here && keys.len() == 1})
    } else { Value::Null };
    // the decision records behind the box
    let b = base_of(piece);
    let ds: Vec<&Value> = decs.iter().filter(|d| idstr(&d["about"]["piece"]) == b || idstr(&d["about"]["person"]) == b).collect();
    let rows: Vec<Value> = ds
        .iter()
        .take(6)
        .map(|d| {
            let mine = idstr(&d["about"]["piece"]) == b;
            let by = d["by"].as_str();
            let who = match by { Some("user") => "you", Some(_) => "rule", None => "open" };
            let kind = d["kind"].as_str().unwrap_or("");
            let what = if kind == "overlap" {
                if mine { tx.t("edit.decs.stuck_to", &v(&[("person", idstr(&d["about"]["person"]))])) }
                else { tx.t("edit.decs.stuck_to_it", &v(&[("piece", idstr(&d["about"]["piece"]))])) }
            } else { kind.to_string() };
            json!({"what": what, "answer": d["answer"].as_str().map(String::from).unwrap_or_else(|| tx.t("edit.decs.not_answered", &[])),
                   "by": who, "by_text": tx.t(&format!("edit.decs.by_{}", if who == "you" { "you" } else { who }), &[])})
        })
        .collect();
    json!({
        "piece": piece, "person": person, "who": who, "emoji": emoji_for(person, edits, palettes),
        "box": tx.t("edit.sel.box", &v(&[("piece", piece)])), "by_hand": if man.is_some() { tx.t("edit.sel.by_hand", &[]) } else { String::new() },
        "span": span, "hidden": hidden, "not_real": not_real,
        "no_emoji": tx.t("edit.sel.no_emoji", &v(&[("why", why)])),
        "shown": is_shown(doc, person), "manual": man.is_some(), "trim": trim, "corr": corr,
        "hint": format!("{} {}", if man.is_some() { tx.t("edit.sel.drag_correct", &[]) } else { tx.t("edit.sel.drag_add", &[]) }, tx.t("edit.sel.why_here", &[])),
        "decs": rows, "decs_more": if ds.len() > 6 { tx.t("edit.decs.more", &v(&[("n", ds.len() - 6)])) } else { String::new() },
        "decs_none": tx.t("edit.decs.none", &v(&[("piece", b.clone())])), "base": b,
    })
}

/// A new frame is on screen: its boxes (and the depth / pose overlays, fetched once per frame).
pub fn on_shown(sh: &Shared, t: usize) {
    if sh.lib.lock().unwrap().pid.is_empty() {
        return;
    }
    rebuild(sh);
    let (dv, pv, have_d, have_p) = {
        let l = sh.lib.lock().unwrap();
        (l.ed.depth_view, l.ed.pose_view, l.ed.depth.contains_key(&t), l.ed.pose.contains_key(&t))
    };
    if (dv && !have_d) || (pv && !have_p) {
        let _ = sh.work.lock().unwrap().send(Box::new(move |sh: &Arc<Shared>| overlays(sh, t)));
    }
}

fn overlays(sh: &Arc<Shared>, t: usize) {
    let pid = sh.lib.lock().unwrap().pid.clone();
    let Ok(sc) = sidecar(sh) else { return };
    let (dv, pv) = {
        let l = sh.lib.lock().unwrap();
        (l.ed.depth_view, l.ed.pose_view)
    };
    if dv {
        if let Ok(d) = sc.get(&format!("/p/{pid}/api/depth/{t}")) {
            sh.lib.lock().unwrap().ed.depth.insert(t, d);
        }
    }
    if pv {
        if let Ok(d) = sc.get(&format!("/p/{pid}/api/pose/{t}")) {
            sh.lib.lock().unwrap().ed.pose.insert(t, d);
        }
    }
    rebuild(sh);
}

// ---------------------------------------------------------------------------------------------------- talking to the server
/// Take a doc_reply (any change's answer) as the project's document, then draw it.
pub fn took(sh: &Arc<Shared>, r: &Value) {
    {
        let mut l = sh.lib.lock().unwrap();
        for k in ["edits", "rev", "resolved", "hidden", "persons", "can_undo", "can_redo"] {
            if let Some(x) = r.get(k) {
                l.doc[k] = x.clone();
            }
        }
        let doc = l.doc.clone();
        refresh_selected(&doc, &mut l.ed);
    }
    {
        let mut st = sh.st.lock().unwrap();
        st.project.can_undo = r["can_undo"].as_bool().unwrap_or(false);
        st.project.can_redo = r["can_redo"].as_bool().unwrap_or(false);
    }
    sh.changed(&["project"]);
    rebuild(sh);
    crate::records::refresh(sh);
}

fn post(sh: &Arc<Shared>, path: &str, body: Value) -> Result<Value> {
    let pid = sh.lib.lock().unwrap().pid.clone();
    if pid.is_empty() {
        bail!("no project open");
    }
    sidecar(sh)?.post(&format!("/p/{pid}/api/{path}"), &body, true)
}

/// Send ops (one undo step) and take the reply.
fn ops(sh: &Arc<Shared>, ops: Vec<Value>, label: &str) -> Result<()> {
    let r = post(sh, "ops", json!({"ops": ops, "label": label}))?;
    took(sh, &r);
    Ok(())
}

fn facts(sh: &Arc<Shared>, kind: &str, pieces: Vec<String>, value: Value, label: &str, extra: Value) -> Result<()> {
    let mut f = json!({"kind": kind, "pieces": pieces, "value": value});
    if let (Some(o), Some(e)) = (f.as_object_mut(), extra.as_object()) {
        for (k, x) in e {
            o.insert(k.clone(), x.clone());
        }
    }
    let r = post(sh, "facts", json!({"facts": [f], "label": label, "page": "edit"}))?;
    took(sh, &r);
    refresh_decisions(sh);
    Ok(())
}

fn refresh_decisions(sh: &Arc<Shared>) {
    let pid = sh.lib.lock().unwrap().pid.clone();
    if let Ok(sc) = sidecar(sh) {
        if let Ok(dr) = sc.get_slow(&format!("/p/{pid}/api/decisions")) {
            let mut ds = dr["decisions"].as_array().cloned().unwrap_or_default();
            crate::decide::sort(&mut ds);
            sh.lib.lock().unwrap().decisions = ds;
            let _ = crate::library::redraw(sh);
            rebuild(sh);
        }
    }
}

/// Run editor work on the library thread (in order with every other change); a failure shows in the panel.
fn work(sh: &Arc<Shared>, what: &'static str, f: impl FnOnce(&Arc<Shared>) -> Result<()> + Send + 'static) {
    queue(sh, move |sh| {
        let r = f(sh);
        let msg = match &r {
            Err(e) => {
                log::warn(&format!("editor {what}: {e:#}"));
                format!("{e:#}")
            }
            Ok(_) => String::new(),
        };
        sh.lib.lock().unwrap().ed.error = msg;
        rebuild(sh);
    });
}

fn cur_t(sh: &Shared) -> usize {
    let st = sh.st.lock().unwrap();
    if st.editor.shown >= 0 { st.editor.shown as usize } else { st.editor.t }
}

// ---------------------------------------------------------------------------------------------------- commands
pub const HELP: &[(&str, &str)] = &[
    ("editor_click", "{x, y}: click at video px: select the cover there (smallest first), or a hidden box, or nothing"),
    ("editor_select", "{piece}: select a box by its track piece (null: none)"),
    ("editor_draw", "{x0, y0, x1, y1}: drag out a box (video px): corrects the selected hand-made cover here, else adds one"),
    ("editor_move", "{piece, dx, dy}: move a hand-made cover on the shown frame by (dx, dy) video px (its placement here)"),
    ("editor_act", "{what}: notface | restore | split | trim_start | trim_end | restore_length | undo_correction | emoji | toggle_show"),
    ("editor_style", "{style}: emoji | blur | pixelate | solid | portrait | orb (the edits' cover_style; orb: developer only)"),
    ("editor_toggle", "{what}: view (V) | depth (D) | pose (B) | keys (?)"),
    ("editor_light", "{key, value, live?}: a lighting setting; live: drawn but not saved yet (a slider being dragged)"),
    ("editor_light_end", "save the lighting being dragged (one undo step)"),
    ("editor_light_reset", "lighting back to its defaults"),
    ("editor_palette", "{key}: the emoji preset (or custom)"),
    ("editor_palette_add", "{text}: add emoji to Custom"),
    ("editor_palette_drop", "{i}: drop Custom's emoji i"),
    ("editor_clear_picks", "take back every emoji picked by hand under this preset"),
    ("editor_head_pose", "{on}: tilt & turn emoji with the head"),
    ("editor_portrait", "{path}: Portrait style's photo (a PNG with its background removed)"),
    ("editor_export_labels", "developer mode: write clean face / not-face labels for training (review_server export_labels)"),
    ("editor_key", "{key, shift?}: a key on the Editor: Left Right (shift: 10) Space V D B Escape Delete S R [ ] K E U ?"),
];

pub fn dispatch(sh: &Arc<Shared>, cmd: &str, a: &Value) -> Result<Value> {
    let num = |k: &str| a.get(k).and_then(Value::as_f64).ok_or_else(|| anyhow!("{cmd}: missing number '{k}'"));
    let s = |k: &str| a.get(k).and_then(Value::as_str).map(String::from);
    match cmd {
        "editor_click" => {
            click(sh, num("x")?, num("y")?);
            Ok(json!("ok"))
        }
        "editor_select" => {
            select(sh, s("piece"));
            Ok(json!("ok"))
        }
        "editor_draw" => {
            let (x0, y0, x1, y1) = (num("x0")?, num("y0")?, num("x1")?, num("y1")?);
            draw_box(sh, x0, y0, x1, y1)
        }
        "editor_move" => move_manual(sh, s("piece").ok_or_else(|| anyhow!("editor_move: missing 'piece'"))?, num("dx")?, num("dy")?),
        "editor_act" => act(sh, &s("what").ok_or_else(|| anyhow!("editor_act: missing 'what'"))?),
        "editor_style" => {
            let st = s("style").ok_or_else(|| anyhow!("editor_style: missing 'style'"))?;
            if !STYLES.contains(&st.as_str()) {
                bail!("style: {}", STYLES.join(" | "));
            }
            work(sh, "style", move |sh| ops(sh, vec![json!({"op": "set", "path": ["cover_style"], "value": st})], "style"));
            Ok(json!("queued"))
        }
        "editor_toggle" => {
            let w = s("what").unwrap_or_default();
            {
                let mut l = sh.lib.lock().unwrap();
                match w.as_str() {
                    "view" => l.ed.boxes = !l.ed.boxes,
                    "depth" => l.ed.depth_view = !l.ed.depth_view,
                    "pose" => l.ed.pose_view = !l.ed.pose_view,
                    "keys" => l.ed.keys_open = !l.ed.keys_open,
                    _ => bail!("editor_toggle: view | depth | pose | keys"),
                }
            }
            if w == "view" {
                sh.dirty.store(true, std::sync::atomic::Ordering::SeqCst);
                sh.changed(&["@render"]);
            }
            on_shown(sh, cur_t(sh));
            Ok(json!("ok"))
        }
        "editor_light" => {
            let k = s("key").ok_or_else(|| anyhow!("editor_light: missing 'key'"))?;
            let val = a.get("value").cloned().unwrap_or(Value::Null);
            let live = a.get("live").and_then(Value::as_bool).unwrap_or(false);
            {
                let mut l = sh.lib.lock().unwrap();
                let mut cur = l.ed.live_light.clone().unwrap_or_else(|| l.doc["edits"]["light"].clone());
                if !cur.is_object() {
                    cur = json!({});
                }
                cur[k.as_str()] = val;
                if k == "shutter" {
                    cur["blur"] = json!(true);
                }
                l.ed.live_light = Some(cur);
            }
            sh.dirty.store(true, std::sync::atomic::Ordering::SeqCst);
            sh.changed(&["@render"]);
            rebuild(sh);
            if !live {
                light_end(sh);
            }
            Ok(json!("ok"))
        }
        "editor_light_end" => {
            light_end(sh);
            Ok(json!("queued"))
        }
        "editor_light_reset" => {
            sh.lib.lock().unwrap().ed.live_light = None;
            work(sh, "light", |sh| ops(sh, vec![json!({"op": "del", "path": ["light"]})], "lighting"));
            Ok(json!("queued"))
        }
        "editor_palette" => {
            let k = s("key").ok_or_else(|| anyhow!("editor_palette: missing 'key'"))?;
            work(sh, "palette", move |sh| {
                let custom_empty = sh.lib.lock().unwrap().doc["edits"]["palette_custom"].as_array().is_none_or(|a| a.is_empty());
                let mut o = vec![json!({"op": "set", "path": ["palette"], "value": k})];
                if k == "custom" && custom_empty {
                    o.push(json!({"op": "set", "path": ["palette_custom"], "value": ["🥸"]}));
                }
                ops(sh, o, "emoji preset")
            });
            Ok(json!("queued"))
        }
        "editor_palette_add" => {
            let text = s("text").unwrap_or_default();
            work(sh, "palette", move |sh| {
                let (edits, pal) = {
                    let l = sh.lib.lock().unwrap();
                    (l.doc["edits"].clone(), l.palettes.clone())
                };
                if edits["palette"].as_str() != Some("custom") {
                    bail!("add emoji to Custom (choose Custom first)");
                }
                let mut cur = palette_for(&edits, &pal);
                for g in graphemes_emoji(&text) {
                    if !cur.contains(&g) {
                        cur.push(g);
                    }
                }
                ops(sh, vec![json!({"op": "set", "path": ["palette"], "value": "custom"}),
                             json!({"op": "set", "path": ["palette_custom"], "value": cur})], "emoji preset")
            });
            Ok(json!("queued"))
        }
        "editor_palette_drop" => {
            let i = num("i")? as usize;
            work(sh, "palette", move |sh| {
                let (edits, pal) = {
                    let l = sh.lib.lock().unwrap();
                    (l.doc["edits"].clone(), l.palettes.clone())
                };
                let mut cur = palette_for(&edits, &pal);
                if edits["palette"].as_str() != Some("custom") || cur.len() <= 1 || i >= cur.len() {
                    return Ok(());
                }
                cur.remove(i);
                ops(sh, vec![json!({"op": "set", "path": ["palette_custom"], "value": cur})], "emoji preset")
            });
            Ok(json!("queued"))
        }
        "editor_clear_picks" => {
            work(sh, "palette", |sh| {
                let edits = sh.lib.lock().unwrap().doc["edits"].clone();
                let fs: Vec<Value> = edits["emoji"]
                    .as_object()
                    .into_iter()
                    .flatten()
                    .filter_map(|(tid, _)| {
                        let ps = edits["people"][tid.trim_start_matches('@')]["pieces"].clone();
                        ps.as_array().filter(|a| !a.is_empty())?;
                        Some(json!({"kind": "emoji", "value": "", "pieces": ps}))
                    })
                    .collect();
                if fs.is_empty() {
                    return Ok(());
                }
                let r = post(sh, "facts", json!({"facts": fs, "label": "Clear picked emoji", "page": "edit"}))?;
                took(sh, &r);
                Ok(())
            });
            Ok(json!("queued"))
        }
        "editor_head_pose" => {
            let on = a.get("on").and_then(Value::as_bool).unwrap_or(true);
            work(sh, "pose", move |sh| {
                let op = if on { json!({"op": "set", "path": ["head_pose"], "value": true}) } else { json!({"op": "del", "path": ["head_pose"]}) };
                ops(sh, vec![op], "tilt & turn")
            });
            Ok(json!("queued"))
        }
        "editor_portrait" => {
            let p = crate::commands::local_path(&s("path").ok_or_else(|| anyhow!("editor_portrait: missing 'path'"))?);
            work(sh, "portrait", move |sh| {
                let bytes = std::fs::read(&p)?;
                let r = post(sh, "portrait", json!({"data": format!("data:image/png;base64,{}", b64(&bytes))}))?;
                let file = r["file"].as_str().ok_or_else(|| anyhow!("no photo made"))?.to_string();
                ops(sh, vec![json!({"op": "set", "path": ["portrait"], "value": file})], "portrait")
            });
            Ok(json!("queued"))
        }
        "editor_export_labels" => {
            work(sh, "labels", |sh| {
                let pid = sh.lib.lock().unwrap().pid.clone();
                let r = sidecar(sh)?.get_slow(&format!("/p/{pid}/api/export_labels"));
                sh.lib.lock().unwrap().ed.labels = match r {
                    Ok(r) => format!("labels: {} faces, {} not-faces · labels_export.json", r["faces"], r["not_faces"]),
                    Err(e) => format!("Export labels failed: {e:#}"),
                };
                Ok(())
            });
            Ok(json!("queued"))
        }
        "editor_key" => key(sh, &s("key").unwrap_or_default(), a.get("shift").and_then(Value::as_bool).unwrap_or(false)),
        _ => bail!("no command '{cmd}'"),
    }
}

fn light_end(sh: &Arc<Shared>) {
    let live = sh.lib.lock().unwrap().ed.live_light.clone();
    let Some(l) = live else { return };
    work(sh, "light", move |sh| {
        let r = ops(sh, vec![json!({"op": "set", "path": ["light"], "value": l})], "lighting");
        sh.lib.lock().unwrap().ed.live_light = None;
        r
    });
}

fn select(sh: &Arc<Shared>, piece: Option<String>) {
    {
        let mut l = sh.lib.lock().unwrap();
        l.ed.piece = piece.clone();
        l.ed.person = None;
        if piece.is_some() {
            let doc = l.doc.clone();
            refresh_selected(&doc, &mut l.ed);
        }
    }
    rebuild(sh);
}

fn click(sh: &Arc<Shared>, x: f64, y: f64) {
    let t = cur_t(sh);
    let hit = {
        let l = sh.lib.lock().unwrap();
        let scale = l.doc["scale"].as_f64().unwrap_or(1.25);
        let inside = |cx: f64, cy: f64, s: f64| (x - cx).abs() <= s * scale / 2.0 && (y - cy).abs() <= s * scale / 2.0;
        let mut hits: Vec<_> = resolved_at(&l.doc, t).into_iter().filter(|b| inside(b.1, b.2, b.3)).collect();
        hits.sort_by(|a, b| a.3.total_cmp(&b.3));
        if let Some(h) = hits.first() {
            Some((h.0.clone(), h.5.clone()))
        } else {
            let mut hh: Vec<_> = hidden_at(&l.doc, t).into_iter().filter(|b| inside(b.1, b.2, b.3)).collect();
            hh.sort_by(|a, b| a.3.total_cmp(&b.3));
            hh.first().map(|h| (h.0.clone(), h.5.clone()))
        }
    };
    {
        let mut l = sh.lib.lock().unwrap();
        match hit {
            Some((tid, piece)) => {
                l.ed.person = Some(tid);
                l.ed.piece = Some(piece);
            }
            None => {
                l.ed.person = None;
                l.ed.piece = None;
            }
        }
    }
    rebuild(sh);
}

/// A box dragged out on the frame (review.html mouseup): it corrects the selected hand-made cover on this frame, else
/// it's a new cover; either way the server follows it from here (propagate).
fn draw_box(sh: &Arc<Shared>, x0: f64, y0: f64, x1: f64, y1: f64) -> Result<Value> {
    let t = cur_t(sh);
    let scale = sh.lib.lock().unwrap().doc["scale"].as_f64().unwrap_or(1.25);
    let s = (x1 - x0).abs().max((y1 - y0).abs()) / scale;
    let c = [(x0 + x1) / 2.0, (y0 + y1) / 2.0, s.max(6.0)];
    place(sh, t, c);
    Ok(json!({"t": t, "c": c}))
}

fn place(sh: &Arc<Shared>, t: usize, c: [f64; 3]) {
    work(sh, "place", move |sh| {
        let (sel, manual) = {
            let l = sh.lib.lock().unwrap();
            let sel = l.ed.piece.clone();
            let manual = l.doc["edits"]["manual"].clone();
            (sel, manual)
        };
        let pr = post(sh, "propagate", json!({"t": t, "c": c}))?;
        let (prop, ev) = (pr["prop"].clone(), pr["ev"].clone());
        match sel.filter(|p| manual.get(p.as_str()).is_some()) {
            Some(id) => {
                // setKey: this placement, and the frames it now follows replace what the cover had there
                let mut m = manual[id.as_str()].clone();
                m["keys"][t.to_string()] = json!(c);
                let ts: Vec<i64> = prop.as_object().into_iter().flatten().filter_map(|(k, _)| k.parse().ok()).collect();
                let mut evs: BTreeSet<i64> = m["ev"].as_array().into_iter().flatten().filter_map(Value::as_i64).collect();
                if let (Some(lo), Some(hi)) = (ts.iter().min(), ts.iter().max()) {
                    if let Some(o) = m["prop"].as_object_mut() {
                        o.retain(|k, _| k.parse::<i64>().map_or(true, |k| k < *lo || k > *hi));
                    }
                    evs.retain(|u| u < lo || u > hi);
                }
                if !m["prop"].is_object() {
                    m["prop"] = json!({});
                }
                for (k, x) in prop.as_object().into_iter().flatten() {
                    m["prop"][k.as_str()] = x.clone();
                }
                for u in ev.as_array().into_iter().flatten().filter_map(Value::as_i64) {
                    evs.insert(u);
                }
                m["ev"] = json!(evs.into_iter().collect::<Vec<_>>());
                ops(sh, vec![json!({"op": "set", "path": ["manual", id], "value": m})], "correct a cover")
            }
            None => {
                // addManual: a new cover m<n+1>
                let next = manual.as_object().into_iter().flatten().filter_map(|(k, _)| k.strip_prefix('m')?.parse::<u64>().ok()).max().unwrap_or(0) + 1;
                let id = format!("m{next}");
                let mut m = json!({"keys": {t.to_string(): c}, "prop": prop, "ev": if ev.is_array() { ev } else { json!([]) }});
                if let Some(pv) = pr.get("pv") {
                    m["pv"] = pv.clone();
                }
                ops(sh, vec![json!({"op": "set", "path": ["manual", id.clone()], "value": m})], "add a cover")?;
                let mut l = sh.lib.lock().unwrap();
                l.ed.piece = Some(id.clone());
                l.ed.person = Some(id);
                Ok(())
            }
        }
    });
}

/// Drag a hand-made cover: its placement on this frame moves (same size), and it's followed from there.
fn move_manual(sh: &Arc<Shared>, piece: String, dx: f64, dy: f64) -> Result<Value> {
    let t = cur_t(sh);
    let (c, manual) = {
        let l = sh.lib.lock().unwrap();
        let b = resolved_at(&l.doc, t).into_iter().find(|b| b.5 == piece);
        (b.map(|b| [b.1 + dx, b.2 + dy, b.3]), l.doc["edits"]["manual"].get(piece.as_str()).is_some())
    };
    let c = c.ok_or_else(|| anyhow!("no box {piece} on frame {t}"))?;
    if !manual {
        bail!("{piece} wasn't added by hand: drag out a new box over the face instead");
    }
    {
        let mut l = sh.lib.lock().unwrap();
        l.ed.piece = Some(piece.clone());
        l.ed.person = Some(piece);
    }
    place(sh, t, c);
    Ok(json!({"t": t, "c": c}))
}

fn act(sh: &Arc<Shared>, what: &str) -> Result<Value> {
    let what = what.to_string();
    let t = cur_t(sh);
    let (piece, person) = {
        let mut l = sh.lib.lock().unwrap();
        let doc = l.doc.clone();
        refresh_selected(&doc, &mut l.ed);
        (l.ed.piece.clone(), l.ed.person.clone())
    };
    let (Some(piece), Some(person)) = (piece, person) else { return Ok(json!("nothing selected")) };
    work(sh, "act", move |sh| {
        let (doc, pal) = {
            let l = sh.lib.lock().unwrap();
            (l.doc.clone(), l.palettes.clone())
        };
        let edits = &doc["edits"];
        let n = doc["n"].as_u64().unwrap_or(1) as usize;
        match what.as_str() {
            "trim_start" | "trim_end" => {
                let r = edits["trims"][piece.as_str()].as_array().map(|a| (a[0].as_u64().unwrap_or(0) as usize, a[1].as_u64().unwrap_or(0) as usize)).unwrap_or((0, n - 1));
                let v = if what == "trim_start" { (t, r.1) } else { (r.0, t) };
                ops(sh, vec![json!({"op": "set", "path": ["trims", piece], "value": [v.0, v.1]})], "trim")
            }
            "restore_length" => ops(sh, vec![json!({"op": "del", "path": ["trims", piece]})], "restore length"),
            "undo_correction" => {
                if edits["manual"][piece.as_str()]["keys"].get(t.to_string()).is_none() {
                    return Ok(());
                }
                let r = post(sh, "undo_correction", json!({"piece": piece, "t": t}))?;
                took(sh, &r);
                Ok(())
            }
            "notface" => {
                if is_hidden_piece(edits, &piece) {
                    return Ok(());
                }
                facts(sh, "face", vec![piece], json!(false), "not a face", json!({"why": "you"}))
            }
            "restore" => {
                if !is_hidden_piece(edits, &piece) {
                    return Ok(());
                }
                facts(sh, "face", vec![piece], json!(true), "a face", json!({}))
            }
            "flip" => {
                if is_hidden_piece(edits, &piece) {
                    facts(sh, "face", vec![piece], json!(true), "a face", json!({}))
                } else {
                    facts(sh, "face", vec![piece], json!(false), "not a face", json!({"why": "you"}))
                }
            }
            "split" => {
                let base = piece.split('~').next().unwrap_or("").to_string();
                if base.starts_with('m') {
                    let r = post(sh, "split_cover", json!({"piece": piece, "t": t}))
                        .map_err(|e| anyhow!("{}", sh.text.t("edit.err.split", &v(&[("piece", piece.clone()), ("error", format!("{e:#}"))]))))?;
                    let new = idstr(&r["new"]);
                    took(sh, &r);
                    sh.lib.lock().unwrap().ed.piece = Some(new);
                    return Ok(());
                }
                let mut cuts: BTreeSet<u64> = edits["cuts"][base.as_str()].as_array().into_iter().flatten().filter_map(Value::as_u64).collect();
                cuts.insert(t as u64);
                ops(sh, vec![json!({"op": "set", "path": ["cuts", base.clone()], "value": cuts.into_iter().collect::<Vec<_>>()})], "split")?;
                sh.lib.lock().unwrap().ed.piece = Some(format!("{base}~{t}"));
                Ok(())
            }
            "toggle_show" => {
                let on = !is_shown(&doc, &person);
                facts(sh, "show", person_pieces(&doc, &person), json!(on), if on { "unmask" } else { "mask" }, json!({}))
            }
            "emoji" => {
                let p = palette_for(edits, &pal);
                let cur = emoji_for(&person, edits, &pal);
                let i = p.iter().position(|x| *x == cur).map(|i| i as i64).unwrap_or(-1);
                let next = p[((i + 1) as usize) % p.len()].clone();
                facts(sh, "emoji", person_pieces(&doc, &person), json!(next), "emoji", json!({}))
            }
            w => bail!("editor_act: no action {w}"),
        }
    });
    Ok(json!("queued"))
}

/// The Editor's keys (review.html keydown and its plug-ins).
fn key(sh: &Arc<Shared>, k: &str, shift: bool) -> Result<Value> {
    let d = if shift { 10 } else { 1 };
    match k {
        "Right" | "ArrowRight" => crate::commands::dispatch(sh, "step", &json!({"d": d})),
        "Left" | "ArrowLeft" => crate::commands::dispatch(sh, "step", &json!({"d": -d})),
        "Space" | " " => crate::commands::dispatch(sh, "toggle_play", &json!({})),
        "v" | "V" => dispatch(sh, "editor_toggle", &json!({"what": "view"})),
        "d" | "D" => dispatch(sh, "editor_toggle", &json!({"what": "depth"})),
        "b" | "B" => dispatch(sh, "editor_toggle", &json!({"what": "pose"})),
        "?" => dispatch(sh, "editor_toggle", &json!({"what": "keys"})),
        "Escape" => {
            select(sh, None);
            Ok(json!("ok"))
        }
        "Delete" | "Backspace" => act(sh, "flip"),
        "s" | "S" => act(sh, "split"),
        "[" => act(sh, "trim_start"),
        "]" => act(sh, "trim_end"),
        "k" | "K" => act(sh, "undo_correction"),
        "e" | "E" => act(sh, "emoji"),
        "u" | "U" => act(sh, "toggle_show"),
        "r" | "R" => why(sh),
        _ => Ok(json!(null)),
    }
}

/// R: the selected box's decisions, in Decisions (filtered to its piece).
pub fn why(sh: &Arc<Shared>) -> Result<Value> {
    let piece = sh.lib.lock().unwrap().ed.piece.clone();
    let Some(p) = piece else { return Ok(json!(null)) };
    crate::commands::dispatch(sh, "go", &json!({"page": "decisions", "piece": base_of(&p)}))
}

fn graphemes_emoji(s: &str) -> Vec<String> {
    // emoji_set_ui.js parseEmoji: grapheme clusters that are pictographs. Without a segmenter: a pictograph plus the
    // joiners, variation selectors, skin tones and keycaps that follow it.
    let mut out: Vec<String> = vec![];
    let mut cur = String::new();
    let mut join = false;
    for c in s.chars() {
        let u = c as u32;
        let ext = u == 0xFE0F || u == 0x20E3 || (0x1F3FB..=0x1F3FF).contains(&u) || (0xE0020..=0xE007F).contains(&u);
        if u == 0x200D {
            cur.push(c);
            join = true;
            continue;
        }
        let pict = u >= 0x2190 && !c.is_alphanumeric() || (0x1F1E6..=0x1F1FF).contains(&u);
        if ext || (join && pict) {
            cur.push(c);
            join = false;
            continue;
        }
        if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        if pict {
            cur.push(c);
        }
        join = false;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn b64(b: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(b.len().div_ceil(3) * 4);
    for c in b.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            if i <= c.len() {
                s.push(T[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                s.push('=');
            }
        }
    }
    s
}

/// Fetch the lighting estimate's status (and its scene colours for the swatch), until it's ready.
pub fn fetch_light(sh: &Arc<Shared>) {
    let pid = sh.lib.lock().unwrap().pid.clone();
    let Ok(sc) = sidecar(sh) else { return };
    let Ok(r) = sc.get_slow(&format!("/p/{pid}/api/light")) else { return };
    let ready = r["ready"].as_bool().unwrap_or(false);
    let keep = json!({"ready": ready, "progress": r["progress"], "error": r["error"], "scene_key": r["scene_key"]});
    {
        let mut l = sh.lib.lock().unwrap();
        if l.pid != pid {
            return;
        }
        l.ed.light_status = keep;
    }
    rebuild(sh);
    if !ready && r["error"].is_null() {
        let s2 = sh.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(2));
            fetch_light(&s2);
        });
    }
}
