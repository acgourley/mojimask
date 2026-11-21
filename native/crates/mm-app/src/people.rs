//! The People page's model (`web/people.html`): everyone the scan counted as a person, one card each, worked out from
//! the project server's resolved covers (`/api/init`: per frame [tid, cx, cy, side, kind, piece, shown]) and its one
//! list of persons (review_server people_of: kind scan / yours / short).
//!
//! Everyone starts covered; a card shows someone who's happy to be seen, names them, or picks their emoji. Each is a
//! fact about their boxes (facts.py) sent to `/api/facts`; the server works out the edits from them and replies with
//! who's who now, and the cards are made again from that.
use crate::decide::{by, crop_url, emoji_for, idstr, palette_for};
use crate::text::{v, Text};
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Debug, Default)]
pub struct Ui {
    /// the brief ones are listed
    pub brief: bool,
    /// cards with their emoji choice or their faces open
    pub palette_open: BTreeSet<String>,
    pub strip_open: BTreeSet<String>,
    pub errors: HashMap<String, String>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct PeopleState {
    pub loaded: bool,
    /// "3 · all masked"
    pub count: String,
    pub n: usize,
    pub cards: Vec<Card>,
    pub brief: Vec<Card>,
    /// "4 brief or unclear ones are masked too."
    pub others: String,
    pub brief_button: String,
    pub palette: Vec<String>,
}

/// A picture: frame t cropped to the rectangle (x, y, w, h) in frame pixels, with the face's square outlined.
#[derive(Clone, Debug, Serialize)]
pub struct Pic {
    pub t: usize,
    /// the picture: the frame's rectangle `rect` (decide::crop_url)
    pub url: String,
    /// what `url` shows, [x, y, w, h] in whole frame pixels
    pub rect: [i64; 4],
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
    /// the face [x, y, side]
    pub face: [f64; 3],
    pub tip: String,
    pub piece: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Card {
    pub tid: String,
    pub pieces: Vec<String>,
    pub name: String,
    pub shown: bool,
    pub emoji: String,
    pub emoji_tip: String,
    pub line: String,
    pub why: bool,
    /// "Might also be someone else: answer in Decisions →" ("" when nothing is open about them)
    pub ask: String,
    pub pic: Pic,
    pub strip: Vec<Pic>,
    pub palette_open: bool,
    pub strip_open: bool,
    pub error: String,
}

struct Person {
    tid: String,
    pieces: BTreeSet<String>,
    frames: Vec<(usize, [f64; 3], String, String)>,
    best: Option<(usize, [f64; 3], f64)>,
    shown: bool,
    kind: String,
}

/// people.html paint(): a crop n wide and h high around the face c = [cx, cy, side], k face-widths to each side,
/// slid inside the frame so it isn't padded where it needn't be.
fn crop(c: [f64; 3], k: f64, n: f64, h: f64, iw: f64, ih: f64) -> (f64, f64, f64, f64) {
    let hw = (c[2] * k).max(24.0);
    let hh = hw * h / n;
    let slide = |v: f64, half: f64, size: f64| if 2.0 * half >= size { size / 2.0 } else { v.max(half).min(size - half) };
    let x = slide(c[0], hw, iw);
    let y = slide(c[1], hh, ih);
    (x - hw, y - hh, 2.0 * hw, 2.0 * hh)
}

pub struct Ctx<'a> {
    pub text: &'a Text,
    pub base: &'a str,
    pub doc: &'a Value,
    pub decisions: &'a [Value],
    pub palettes: &'a Value,
}

pub fn view(ui: &Ui, cx: &Ctx) -> PeopleState {
    let tx = cx.text;
    let doc = cx.doc;
    let (w, h) = (doc["w"].as_f64().unwrap_or(1e9), doc["h"].as_f64().unwrap_or(1e9));
    let fps = doc["fps"].as_f64().filter(|f| *f > 0.0).unwrap_or(30.0);
    let photo = doc["photo"].as_bool().unwrap_or(false);
    let edits = &doc["edits"];
    let kind_of: HashMap<String, String> = doc["persons"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|p| (idstr(&p["tid"]), p["kind"].as_str().unwrap_or("short").to_string()))
        .collect();
    let mut by_tid: Vec<Person> = vec![];
    let mut index: HashMap<String, usize> = HashMap::new();
    for (t, f) in doc["resolved"].as_array().into_iter().flatten().enumerate() {
        for x in f.as_array().into_iter().flatten() {
            let Some(a) = x.as_array() else { continue };
            if a.len() < 7 {
                continue;
            }
            let tid = idstr(&a[0]);
            let c = [a[1].as_f64().unwrap_or(0.0), a[2].as_f64().unwrap_or(0.0), a[3].as_f64().unwrap_or(0.0)];
            let kind = a[4].as_str().unwrap_or("").to_string();
            let piece = idstr(&a[5]);
            let i = *index.entry(tid.clone()).or_insert_with(|| {
                by_tid.push(Person {
                    tid: tid.clone(),
                    pieces: BTreeSet::new(),
                    frames: vec![],
                    best: None,
                    shown: false,
                    kind: kind_of.get(&tid).cloned().unwrap_or_else(|| "short".into()),
                });
                by_tid.len() - 1
            });
            let p = &mut by_tid[i];
            p.pieces.insert(piece.clone());
            if a[6].as_bool().unwrap_or(false) {
                p.shown = true;
            }
            let inside = c[0] - c[2] / 2.0 >= 0.0 && c[1] - c[2] / 2.0 >= 0.0 && c[0] + c[2] / 2.0 <= w && c[1] + c[2] / 2.0 <= h;
            let score = c[2] * if kind == "obs" { 2.0 } else { 1.0 } * if inside { 1.0 } else { 0.3 };
            if p.best.is_none_or(|b| score > b.2) {
                p.best = Some((t, c, score));
            }
            p.frames.push((t, c, kind, piece));
        }
    }
    // the scan's and yours first, then by time on screen (a stable sort, as the page's)
    by_tid.sort_by(|a, b| (a.kind == "short").cmp(&(b.kind == "short")).then(b.frames.len().cmp(&a.frames.len())));
    let person_of = |tid: &str| -> Option<&Value> { tid.strip_prefix('@').and_then(|pid| edits.get("people").and_then(|p| p.get(pid))) };
    let card = |p: &Person| -> Card {
        let pm = person_of(&p.tid);
        let name = pm.and_then(|m| m["name"].as_str()).unwrap_or("").to_string();
        let shown = pm.map(|m| m["show"].as_bool().unwrap_or(false)).unwrap_or(p.shown);
        let has = |d: &Value| p.pieces.contains(&idstr(&d["about"]["piece"])) || p.pieces.contains(&idstr(&d["about"]["person"]));
        let decs = cx.decisions.iter().filter(|d| has(d)).count();
        let maybes: Vec<&Value> = cx.decisions.iter().filter(|d| matches!(d["kind"].as_str(), Some("same" | "whose")) && by(d) == "open" && has(d)).collect();
        let mut line = String::new();
        if !photo {
            line = tx.t("people.on_screen", &v(&[("s", format!("{:.1}", p.frames.len() as f64 / fps))]));
        }
        if p.pieces.len() > 1 {
            if !photo {
                line += " · ";
            }
            line += &tx.t("people.joined", &v(&[("n", p.pieces.len())]));
        }
        let emoji = emoji_for(&p.tid, edits, cx.palettes);
        let (bt, bc, _) = p.best.unwrap();
        let (x, y, cw, ch) = crop(bc, 1.4, 348.0, 264.0, w, h);
        let (url, rect) = crop_url(cx.base, bt, x, y, cw, ch);
        let pic = Pic { t: bt, url, rect, x, y, w: cw, h: ch, face: bc, tip: String::new(), piece: String::new() };
        // up to 8 detected moments, spread over their time on screen
        let det: Vec<&(usize, [f64; 3], String, String)> = p.frames.iter().filter(|f| f.2 == "obs").collect();
        let pick: Vec<&(usize, [f64; 3], String, String)> = if det.is_empty() { p.frames.iter().collect() } else { det };
        let n = pick.len().min(8);
        let strip = (0..n)
            .map(|i| {
                let f = pick[i * pick.len() / n];
                let (x, y, cw, ch) = crop(f.1, 1.3, 240.0, 240.0, w, h);
                let (url, rect) = crop_url(cx.base, f.0, x, y, cw, ch);
                Pic {
                    t: f.0,
                    url,
                    rect,
                    x,
                    y,
                    w: cw,
                    h: ch,
                    face: f.1,
                    tip: tx.t("people.strip_tip", &v(&[("s", format!("{:.1}", f.0 as f64 / fps))])),
                    piece: f.3.clone(),
                }
            })
            .collect();
        Card {
            tid: p.tid.clone(),
            pieces: p.pieces.iter().cloned().collect(),
            name,
            shown,
            emoji_tip: tx.t("people.pick_emoji_now", &v(&[("emoji", &emoji)])),
            emoji,
            line,
            why: decs > 0,
            ask: if maybes.is_empty() {
                String::new()
            } else if maybes.iter().any(|d| d["kind"] == "whose") {
                tx.s("people.ask_whose")
            } else {
                tx.s("people.ask_same")
            },
            pic,
            strip,
            palette_open: ui.palette_open.contains(&p.tid),
            strip_open: ui.strip_open.contains(&p.tid),
            error: ui.errors.get(&p.tid).cloned().unwrap_or_default(),
        }
    };
    let ps: Vec<&Person> = by_tid.iter().filter(|p| p.kind != "short").collect();
    let brief: Vec<&Person> = by_tid.iter().filter(|p| p.kind == "short").collect();
    let cards: Vec<Card> = ps.iter().map(|p| card(p)).collect();
    let shown = cards.iter().filter(|c| c.shown).count();
    PeopleState {
        loaded: true,
        count: format!("{} · {}", ps.len(), if shown > 0 { tx.t("people.unmasked", &v(&[("n", shown)])) } else { tx.s("people.all_masked") }),
        n: ps.len(),
        cards,
        brief: if ui.brief { brief.iter().map(|p| card(p)).collect() } else { vec![] },
        others: if brief.is_empty() { String::new() } else { tx.t("people.brief", &v(&[("n", brief.len())])) },
        brief_button: if brief.is_empty() {
            String::new()
        } else {
            tx.s(if ui.brief { "people.hide_brief" } else { "people.show_brief" })
        },
        palette: palette_for(edits, cx.palettes),
    }
}

/// The piece a card's question or "Why" opens Decisions on (people.html: the open same-person question's box).
pub fn pieces_of(cards: &[Card], tid: &str) -> Vec<String> {
    cards.iter().find(|c| c.tid == tid).map(|c| c.pieces.clone()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    #[test]
    fn crop_slides_inside() {
        // a face near the left edge: the crop starts at 0, not before it
        let (x, _, w, _) = super::crop([10.0, 200.0, 40.0], 1.4, 348.0, 264.0, 640.0, 480.0);
        assert_eq!(x, 0.0);
        assert!((w - 112.0).abs() < 1e-9);
    }
}
