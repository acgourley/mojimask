//! The Decisions page's model: what `web/decisions.html` works out in the page, worked out here so QML only lays it
//! out (native/ARCHITECTURE.md: no logic in QML).
//!
//! Every decision is one record, whoever made it: a rule, or you (`by: "user"`). The page lists them (open questions
//! first, the ones that matter most first), shows the selected one large with its pictures, and takes answers: keys
//! 1-3 by slot (1 keep it, 2 not a face / different, 3 someone else), 0-9 for "whose face", J/K or arrows to move.
//! Developer mode adds the rules' answers, filters by kind and the evidence numbers.
//!
//! Pictures: crops of the scanned frames, cut and scaled by the project's server (`/api/frames/NNNNN.jpg?x=&y=&w=&h=`,
//! the page adds `&px=` for the size it draws at: [`crop_url`]), given here as that URL, the square in frame pixels
//! (centre, half side) and the boxes to outline. The selected question's pictures
//! loop through their box's own frames ([`loop_frames`], fetched by the page when the selection changes, kept out of
//! the state tree because it is long).
use crate::text::{v, Text};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};

/// rows drawn in the list
pub const MAX: usize = 80;
/// the longest loop, seconds (a box on screen longer plays faster)
const LOOP_MAX_S: f64 = 6.0;
pub const SOMEONE_NEW: &str = "someone new";

/// What the person chose on the page (not the records).
#[derive(Clone, Debug)]
pub struct Ui {
    pub filter: String,
    pub kind: String,
    pub cur: usize,
    /// only the decisions about one box (the Editor's R, People's "Why"): decisions.html #piece=
    pub piece: Option<String>,
    /// with the piece: also the questions where it is only the grey box
    pub nearby: bool,
}

impl Default for Ui {
    fn default() -> Self {
        Ui { filter: "open".into(), kind: "all".into(), cur: 0, piece: None, nearby: false }
    }
}

fn own(d: &Value, piece: &str) -> bool {
    idstr(&d["about"]["piece"]) == piece
}
fn next_to(d: &Value, piece: &str) -> bool {
    !own(d, piece) && idstr(&d["about"]["person"]) == piece
}
/// decisions.html about(): every record, or (a piece chosen) its own, and with `nearby` the ones it's the grey box in
fn about(d: &Value, ui: &Ui) -> bool {
    match &ui.piece {
        None => true,
        Some(p) => own(d, p) || (ui.nearby && next_to(d, p)),
    }
}

/// The open project, as the pictures need it.
pub struct Ctx<'a> {
    pub text: &'a Text,
    pub dev: bool,
    pub fps: f64,
    /// the project server's base URL (through app_server's proxy): `http://127.0.0.1:9300/p/<id>`
    pub base: &'a str,
    pub w: f64,
    pub h: f64,
    /// /api/init: resolved covers per frame, edits, scale
    pub doc: &'a Value,
    pub palettes: &'a Value,
    /// /api/boxes per piece id: frame -> [x0, y0, x1, y1] (None: a box the server can't follow)
    pub boxes: &'a HashMap<String, Option<BTreeMap<usize, [f64; 4]>>>,
    pub missing: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct DecisionsState {
    pub loaded: bool,
    pub dev: bool,
    /// "{open} to answer · {answered} answered"
    pub count: String,
    pub open: usize,
    pub filters: Vec<Pill>,
    pub kinds: Vec<Pill>,
    pub rows: Vec<Row>,
    pub cur: usize,
    pub hint: String,
    pub detail: Option<Detail>,
    pub empty: Option<Empty>,
    /// answers on their way to the project's server
    pub pending: usize,
    pub error: String,
    /// only one box's decisions (the Editor's R): what the bar above the list says, "" otherwise
    pub piece: String,
    pub piece_text: String,
    pub nearby_text: String,
    pub all_text: String,
    pub split_hint: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Pill {
    pub key: String,
    pub label: String,
    pub n: usize,
    pub on: bool,
}

/// A crop of a scanned frame: a square around (cx, cy), `half` frame pixels to each side, with the boxes outlined.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Crop {
    pub t: usize,
    /// the picture: the frame's rectangle `rect` ([`crop_url`])
    pub url: String,
    /// what `url` shows, [x, y, w, h] in whole frame pixels: the square, rounded outwards
    pub rect: [i64; 4],
    pub cx: f64,
    pub cy: f64,
    pub half: f64,
    /// the green box and the grey box, [x0, y0, x1, y1] in frame pixels
    pub piece: Option<[f64; 4]>,
    pub person: Option<[f64; 4]>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Row {
    pub i: usize,
    pub id: String,
    pub q: String,
    pub why: String,
    pub badge: String,
    /// "user" | "rule" | ""
    pub badge_kind: String,
    pub thumb: Option<Crop>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Pic {
    /// the box a click on it selects in Edit (the green one, else the grey)
    pub piece: String,
    pub label: String,
    /// the caption under the still ("The box · frame 40")
    pub cap: String,
    pub wide: bool,
    pub still: Crop,
}

#[derive(Clone, Debug, Serialize)]
pub struct Answer {
    pub value: String,
    pub label: String,
    pub key: String,
    pub chosen: bool,
    pub rule_chosen: bool,
    pub title: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Cand {
    pub piece: String,
    pub name: String,
    pub why: String,
    pub key: String,
    pub chosen: bool,
    pub pic: Crop,
}

#[derive(Clone, Debug, Serialize)]
pub struct Chip {
    pub text: String,
    /// "" | "rule" | "grade"
    pub cls: String,
    pub title: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Detail {
    pub id: String,
    pub kind: String,
    /// changes whenever the pictures change (the page reloads their loops)
    pub key: String,
    pub title: String,
    pub note: String,
    pub pics: Vec<Pic>,
    pub answers: Vec<Answer>,
    pub cands: Vec<Cand>,
    pub rule_pick: String,
    pub rule_pick_title: String,
    pub why: String,
    pub agent: bool,
    pub chips: Vec<Chip>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Empty {
    pub title: String,
    pub text: String,
    pub next: bool,
}

// ---------------------------------------------------------------------------------------------------- the records
fn s<'a>(d: &'a Value, k: &str) -> &'a str {
    d.get(k).and_then(Value::as_str).unwrap_or("")
}
fn num(d: &Value, k: &str) -> f64 {
    d.get(k).and_then(Value::as_f64).unwrap_or(0.0)
}
/// a JSON scalar as the page would print it (an id may be a number or a string)
pub fn idstr(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        x => x.to_string(),
    }
}
fn box4(v: &Value) -> Option<[f64; 4]> {
    let a = v.as_array()?;
    if a.len() < 4 {
        return None;
    }
    Some([a[0].as_f64()?, a[1].as_f64()?, a[2].as_f64()?, a[3].as_f64()?])
}

/// "user", "rule" or "open"
pub fn by(d: &Value) -> &'static str {
    match d.get("by") {
        Some(Value::String(b)) if b == "user" => "user",
        Some(Value::String(b)) if !b.is_empty() => "rule",
        _ => "open",
    }
}

/// The page's order ([`mm_core::facts::page_sort`], shared with `mm-cli decisions` and the rules' benchmark).
pub fn sort(ds: &mut [Value]) {
    mm_core::facts::page_sort(ds, |d| mm_core::facts::page_key(by(d) == "open", s(d, "kind"), num(d, "weight"), num(&d["evidence"], "frames")));
}

/// The answers a person can give (value, label key), left to right: keys 1, 2, 3.
pub fn opts(kind: &str) -> &'static [(&'static str, &'static str)] {
    match kind {
        "whose" => &[],
        "same" => &[("same person", "decisions.a.same_person"), ("different people", "decisions.a.different_people")],
        "track" | "blip" => &[("a face", "decisions.a.face"), ("not a face", "decisions.a.not_face")],
        _ => &[("same face", "decisions.a.same_face"), ("not a face", "decisions.a.not_face_at_all"), ("someone else", "decisions.a.someone_else")],
    }
}

fn kind_name(kind: &str) -> &'static str {
    match kind {
        "whose" => "Whose face",
        "same" => "Same person",
        "track" => "Tracks",
        "blip" => "Blips",
        _ => "Overlaps",
    }
}
const KINDS: [&str; 5] = ["whose", "same", "overlap", "track", "blip"];

fn question(tx: &Text, d: &Value) -> String {
    match s(d, "kind") {
        "whose" => s(d, "question").to_string(),
        "same" => tx.s("decisions.q.same"),
        "track" | "blip" => tx.s("decisions.q.face"),
        _ => tx.s("decisions.q.overlap"),
    }
}
fn note(tx: &Text, kind: &str) -> String {
    match kind {
        "whose" => tx.s("decisions.note.whose"),
        "same" => tx.s("decisions.note.same"),
        "track" | "blip" => tx.s("decisions.note.face"),
        _ => tx.s("decisions.note.overlap"),
    }
}
fn short_q(tx: &Text, kind: &str) -> String {
    match kind {
        "track" | "blip" => tx.s("decisions.short.face"),
        "same" => tx.s("decisions.short.same"),
        "whose" => tx.s("decisions.q.whose"),
        _ => tx.s("decisions.short.overlap"),
    }
}
pub fn why_text(tx: &Text, d: &Value, fps: f64) -> String {
    let ev = &d["evidence"];
    match s(d, "kind") {
        "same" => tx.s(if ev.get("fragment").and_then(Value::as_bool).unwrap_or(false) { "decisions.why.fragment" } else { "decisions.why.same" }),
        "blip" => tx.t("decisions.why.blip", &v(&[("n", num(ev, "frames"))])),
        "track" => tx.t("decisions.why.track", &v(&[("s", format!("{:.1}", num(ev, "frames") / if fps > 0.0 { fps } else { 30.0 }))])),
        "whose" => tx.s("decisions.why.whose"),
        _ => tx.s("decisions.why.overlap"),
    }
}
fn short_a(tx: &Text, a: &str) -> String {
    let k = match a {
        "a face" | "same person" | "same face" => "decisions.sa.yes",
        "not a face" | "different people" => "decisions.sa.no",
        "someone else" => "decisions.sa.someone_else",
        "someone new" => "decisions.sa.someone_new",
        "covered by the person's box (while they are)" => "decisions.sa.covered_by_box",
        "left as it is" => "decisions.sa.too_short",
        "dropped: too weak" => "decisions.sa.dropped",
        "kept: covered" => "decisions.sa.kept",
        "covered by another box" => "decisions.sa.inside",
        "joined already" => "decisions.sa.joined",
        "kept apart" => "decisions.sa.kept_apart",
        "waiting" => "decisions.sa.waiting",
        _ => return a.to_string(),
    };
    tx.s(k)
}

/// JavaScript's Number formatting for the evidence chips: integers without ".0".
fn jn(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}

/// The evidence chips (developer mode), decisions.html KINDS[kind].chips.
fn chips(kind: &str, ev: &Value) -> Vec<String> {
    let n = |k: &str| num(ev, k);
    let b = |k: &str| ev.get(k).and_then(Value::as_bool).unwrap_or(false);
    let span = format!("frames {}–{}", jn(n("first")), jn(n("last")));
    match kind {
        "whose" => vec![if b("split") { "split off by you".into() } else { "your mask".into() }, format!("{} people offered", jn(n("candidates"))), span],
        "same" => {
            let mut c = vec![format!("score {:.2}", n("score"))];
            if b("fragment") {
                c.push("short stretch".into());
            }
            if b("manual") {
                c.push("your mask".into());
            }
            let g = n("gap_frames");
            c.push(if g < 0.0 { format!("on their head {} frames", jn(1.0 - g)) } else { format!("gap {} frames", jn(g)) });
            c.push(span);
            c
        }
        "track" => vec![
            format!("{} frames", jn(n("frames"))),
            format!("a clear face in {}", jn(n("clear_face_frames"))),
            format!("median detector {:.2}", n("median_detector")),
            format!("best face check {:.2}", n("best_face_check")),
            format!("{}px", jn(n("size"))),
            span,
        ],
        "blip" => vec![
            format!("seen in {} frame{}", jn(n("frames")), if n("frames") > 1.0 { "s" } else { "" }),
            format!("detector {:.2}", n("detector_score")),
            format!("face check {:.2}", n("face_check")),
            if b("skeleton_head") { "skeleton head on it".into() } else { "no skeleton head".into() },
            format!("{}px", jn(n("size"))),
            if n("first") == n("last") { format!("frame {}", jn(n("first"))) } else { span },
        ],
        _ => {
            let mut c = vec![
                format!("{} box", if b("numbered") { "numbered" } else { "grey" }),
                format!("{} frames", jn(n("frames"))),
                format!("overlaps in {}% of them", (100.0 * n("frames_overlapping")).round()),
                format!("median {}% of its box", (100.0 * n("median_overlap")).round()),
            ];
            if ev.get("depth_gap").is_some() {
                c.push(format!("depth gap {:.2}", n("depth_gap")));
            }
            c.push(span);
            c
        }
    }
}

// ---------------------------------------------------------------------------------------------------- the pictures
/// One moment a record shows: frame t and its boxes.
#[derive(Clone, Debug)]
pub struct Still {
    pub t: usize,
    pub piece: Option<[f64; 4]>,
    pub person: Option<[f64; 4]>,
}

/// The moments a record shows, one per set of boxes (the green one, the grey one, or both): its clearest moment
/// (decisions.html stills).
pub fn stills(d: &Value) -> Vec<Still> {
    let mut groups: Vec<(String, Vec<Still>)> = vec![];
    for sh in d.get("show").and_then(Value::as_array).into_iter().flatten() {
        let st = Still { t: sh.get("t").and_then(Value::as_u64).unwrap_or(0) as usize, piece: box4(&sh["piece"]), person: box4(&sh["person"]) };
        let k = format!("{}{}", if st.piece.is_some() { "g" } else { "" }, if st.person.is_some() { "p" } else { "" });
        match groups.iter_mut().find(|(g, _)| *g == k) {
            Some((_, v)) => v.push(st),
            None => groups.push((k, vec![st])),
        }
    }
    let same = s(d, "kind") == "same";
    groups
        .into_iter()
        .map(|(k, ss)| if same { if k == "p" { ss[0].clone() } else { ss[ss.len() - 1].clone() } } else { ss[(ss.len() - 1) / 2].clone() })
        .collect()
}

/// The square around a picture's boxes: centre and half side (decisions.html crop).
pub fn crop(bs: &[[f64; 4]]) -> (f64, f64, f64) {
    if bs.is_empty() {
        return (0.0, 0.0, 35.0);
    }
    let x0 = bs.iter().map(|b| b[0]).fold(f64::INFINITY, f64::min);
    let y0 = bs.iter().map(|b| b[1]).fold(f64::INFINITY, f64::min);
    let x1 = bs.iter().map(|b| b[2]).fold(f64::NEG_INFINITY, f64::max);
    let y1 = bs.iter().map(|b| b[3]).fold(f64::NEG_INFINITY, f64::max);
    ((x0 + x1) / 2.0, (y0 + y1) / 2.0, ((x1 - x0).max(y1 - y0) * 1.4).max(70.0))
}

fn boxes_of(st: &Still) -> Vec<[f64; 4]> {
    st.piece.iter().chain(st.person.iter()).cloned().collect()
}

pub fn frame_url(base: &str, t: usize) -> String {
    format!("{base}/api/frames/{t:05}.jpg")
}

/// The rectangle (x, y, w, h) of frame `t` as a picture, and the whole pixels it covers (rounded outwards; it may
/// reach past the frame, black there). The server cuts it out and scales it down, so a 4K frame is never encoded,
/// sent and decoded whole for a thumbnail (68 + 64 ms a picture at 4K; the page draws several a card and a loop of
/// them). The page appends `&px=N`, the longer side it draws at (components/Picture.qml).
pub fn crop_url(base: &str, t: usize, x: f64, y: f64, w: f64, h: f64) -> (String, [i64; 4]) {
    let (x0, y0) = (x.floor() as i64, y.floor() as i64);
    let r = [x0, y0, ((x + w).ceil() as i64 - x0).max(1), ((y + h).ceil() as i64 - y0).max(1)];
    (format!("{base}/api/frames/{t:05}.jpg?x={}&y={}&w={}&h={}", r[0], r[1], r[2], r[3]), r)
}

impl Crop {
    pub fn new(base: &str, t: usize, (cx, cy, half): (f64, f64, f64), piece: Option<[f64; 4]>, person: Option<[f64; 4]>) -> Crop {
        let (url, rect) = crop_url(base, t, cx - half, cy - half, 2.0 * half, 2.0 * half);
        Crop { t, url, rect, cx, cy, half, piece, person }
    }
}

impl Ctx<'_> {
    fn crop_of(&self, st: &Still) -> Crop {
        Crop::new(self.base, st.t, crop(&boxes_of(st)), st.piece, st.person)
    }

    /// "In the frame": as much of the frame as a square holds, slid to keep the boxes in it. The Decisions page never
    /// draws covers (no emoji there: you're looking at the faces to decide about them), so the frame is shown as it is.
    fn wide_of(&self, st: &Still) -> Crop {
        let half = self.w.min(self.h) / 2.0;
        let (bx, by, _) = crop(&boxes_of(st));
        let cx = bx.max(half).min(self.w - half);
        let cy = by.max(half).min(self.h - half);
        Crop::new(self.base, st.t, (cx, cy, half), st.piece, st.person)
    }
}

// ---------------------------------------------------------------------------------------------------- emoji (page_util.js)
/// The emoji set in use (palettes.py for_edits): a custom set, else the chosen preset, else the default one.
pub fn palette_for<'a>(edits: &'a Value, palettes: &'a Value) -> Vec<String> {
    let strs = |v: &Value| v.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect::<Vec<_>>()).unwrap_or_default();
    if s(edits, "palette") == "custom" {
        let c = strs(&edits["palette_custom"]);
        if !c.is_empty() {
            return c;
        }
    }
    let ps = palettes.as_array().cloned().unwrap_or_default();
    let p = ps
        .iter()
        .find(|p| s(p, "key") == s(edits, "palette") && !s(edits, "palette").is_empty())
        .or_else(|| ps.iter().find(|p| p.get("default").and_then(Value::as_bool).unwrap_or(false)))
        .or(ps.first());
    p.map(|p| strs(&p["emoji"])).filter(|v| !v.is_empty()).unwrap_or_else(|| vec!["😎".into()])
}

/// The emoji a person or box wears: a hand pick, else its slot in the set by a hash of its id (page_util.js emojiFor).
pub fn emoji_for(tid: &str, edits: &Value, palettes: &Value) -> String {
    if let Some(e) = edits.get("emoji").and_then(|m| m.get(tid)).and_then(Value::as_str) {
        if !e.is_empty() {
            return e.to_string();
        }
    }
    let id = tid.strip_prefix('@').unwrap_or(tid);
    let h: u64 = id.encode_utf16().enumerate().map(|(i, c)| (i as u64 + 1) * c as u64).sum();
    let p = palette_for(edits, palettes);
    p[(h % p.len() as u64) as usize].clone()
}

// ---------------------------------------------------------------------------------------------------- the page
pub fn shown<'a>(ds: &'a [Value], ui: &Ui) -> Vec<&'a Value> {
    let mut l: Vec<&Value> = ds
        .iter()
        .filter(|d| about(d, ui) && (ui.filter == "all" || by(d) == ui.filter) && (ui.kind == "all" || s(d, "kind") == ui.kind))
        .collect();
    if let Some(p) = &ui.piece {
        l.sort_by_key(|d| !own(d, p)); // its own questions first (stable)
    }
    l
}

/// The ids /api/boxes must be asked for before the selected question's loops can play.
pub fn wanted_boxes(ds: &[Value], ui: &Ui) -> Vec<String> {
    let l = shown(ds, ui);
    let Some(d) = l.get(ui.cur.min(l.len().saturating_sub(1))) else { return vec![] };
    let mut out = vec![];
    for k in ["piece", "person"] {
        let id = idstr(&d["about"][k]);
        if !id.is_empty() {
            out.push(id);
        }
    }
    out
}

pub fn view(ds: &[Value], ui: &mut Ui, cx: &Ctx) -> DecisionsState {
    let tx = cx.text;
    let mut c: HashMap<&str, usize> = HashMap::new();
    let mut tot: HashMap<&str, usize> = HashMap::new();
    for d in ds {
        *tot.entry(by(d)).or_default() += 1;
        if (ui.kind == "all" || s(d, "kind") == ui.kind) && about(d, ui) {
            *c.entry(by(d)).or_default() += 1;
            *c.entry("all").or_default() += 1;
        }
    }
    let g = |m: &HashMap<&str, usize>, k: &str| *m.get(k).unwrap_or(&0);
    let mut count = tx.t("decisions.count", &v(&[("open", g(&tot, "open")), ("answered", g(&tot, "user"))]));
    if cx.dev {
        count += &format!(" · {} by rules", g(&tot, "rule"));
    }
    let mut fl: Vec<(&str, String)> = vec![("open", tx.s("decisions.filter.open")), ("user", tx.s("decisions.filter.answered"))];
    if cx.dev {
        fl.push(("rule", "Rules".into()));
        fl.push(("all", "All".into()));
    }
    let filters = fl.into_iter().map(|(k, label)| Pill { key: k.into(), label, n: g(&c, k), on: ui.filter == k }).collect();
    let kinds = if cx.dev {
        std::iter::once(Pill { key: "all".into(), label: "Every kind".into(), n: ds.len(), on: ui.kind == "all" })
            .chain(KINDS.iter().map(|k| Pill { key: (*k).into(), label: kind_name(k).into(), n: ds.iter().filter(|d| s(d, "kind") == *k).count(), on: ui.kind == *k }))
            .collect()
    } else {
        vec![]
    };
    let l = shown(ds, ui);
    ui.cur = ui.cur.min(l.len().min(MAX).saturating_sub(1));
    let rows = l
        .iter()
        .take(MAX)
        .enumerate()
        .map(|(i, d)| {
            let who = by(d);
            let ans = if who == "open" {
                String::new()
            } else if s(d, "kind") == "whose" && s(d, "answer") != SOMEONE_NEW {
                tx.s("decisions.chosen")
            } else {
                short_a(tx, s(d, "answer"))
            };
            let ss = stills(d);
            let st = ss.iter().find(|x| x.piece.is_some()).or(ss.first());
            let mut why = why_text(tx, d, cx.fps);
            if cx.dev {
                why += &format!(" · {}", kind_name(s(d, "kind")).to_lowercase());
            }
            Row {
                i,
                id: idstr(&d["id"]),
                q: short_q(tx, s(d, "kind")),
                why,
                badge: if ans.is_empty() { ans } else if who == "rule" { format!("{ans} · {}", tx.s("decisions.rule")) } else { ans },
                badge_kind: if who == "open" { String::new() } else { who.into() },
                thumb: st.map(|st| cx.crop_of(st)),
            }
        })
        .collect();
    let mut hint = String::new();
    if l.len() > MAX {
        hint = tx.t("decisions.more_not_listed", &v(&[("n", l.len() - MAX)])) + " · ";
    }
    if !l.is_empty() {
        hint += &tx.s("decisions.keys_hint");
    }
    let any_open = ds.iter().any(|d| by(d) == "open");
    let (detail, empty) = match l.get(ui.cur) {
        Some(d) => (Some(detail(d, cx)), None),
        None => (
            None,
            Some(if cx.missing {
                Empty { title: tx.s("decisions.empty.missing_title"), text: tx.s("decisions.empty.missing"), next: false }
            } else if ui.filter == "open" && !any_open {
                Empty { title: tx.s("decisions.empty.done_title"), text: tx.s("decisions.empty.done"), next: true }
            } else {
                Empty { title: tx.s("decisions.empty.none_title"), text: tx.s("decisions.empty.none"), next: false }
            }),
        ),
    };
    DecisionsState {
        loaded: true,
        dev: cx.dev,
        count,
        open: g(&tot, "open"),
        filters,
        kinds,
        rows,
        cur: ui.cur,
        hint,
        detail,
        empty,
        pending: 0,
        error: String::new(),
        piece: ui.piece.clone().unwrap_or_default(),
        piece_text: match &ui.piece {
            Some(p) => {
                let mine = ds.iter().filter(|d| own(d, p)).count();
                if mine > 0 { tx.t("decisions.piece.mine_html", &v(&[("n", mine)])) } else { tx.s("decisions.piece.none") }
            }
            None => String::new(),
        },
        nearby_text: match &ui.piece {
            Some(p) => {
                let others = ds.iter().filter(|d| next_to(d, p)).count();
                if others > 0 { tx.t(if ui.nearby { "decisions.piece.hide_nearby" } else { "decisions.piece.show_nearby" }, &v(&[("n", others)])) } else { String::new() }
            }
            None => String::new(),
        },
        all_text: tx.s("decisions.piece.all"),
        split_hint: tx.s("decisions.piece.split_hint"),
    }
}

fn detail(d: &Value, cx: &Ctx) -> Detail {
    let tx = cx.text;
    let kind = s(d, "kind").to_string();
    let ss = stills(d);
    let wide = ss.len() == 1;
    let mut pics = vec![];
    for st in &ss {
        let label = if ss.len() > 1 {
            tx.s(if st.piece.is_some() { "decisions.pic.green" } else { "decisions.pic.grey" })
        } else if st.piece.is_some() && st.person.is_some() {
            tx.s("decisions.pic.boxes")
        } else {
            tx.s("decisions.pic.box")
        };
        let cap = format!("{label} · {}", tx.t("decisions.pic.frame", &v(&[("t", st.t)])));
        let piece = idstr(&d["about"][if st.piece.is_some() { "piece" } else { "person" }]);
        pics.push(Pic { piece, label, cap, wide: false, still: cx.crop_of(st) });
    }
    if wide {
        let label = tx.s("decisions.pic.in_frame");
        let piece = idstr(&d["about"][if ss[0].piece.is_some() { "piece" } else { "person" }]);
        pics.push(Pic { piece, cap: label.clone(), label, wide: true, still: cx.wide_of(&ss[0]) });
    }
    let whose = kind == "whose";
    let mut title = if whose { s(d, "question").to_string() } else { question(tx, d) };
    if let Some(r) = d.get("rank").filter(|r| !r.is_null() && r.as_f64() != Some(0.0)) {
        title = format!("{}. {title}", idstr(r));
    }
    let who = by(d);
    let ruled = who == "rule";
    let answer = s(d, "answer");
    let o: Vec<(String, String)> =
        if whose { vec![(SOMEONE_NEW.into(), tx.s("decisions.a.someone_new"))] } else { opts(&kind).iter().map(|(a, k)| (a.to_string(), tx.s(k))).collect() };
    let answers = o
        .iter()
        .enumerate()
        .map(|(k, (val, label))| {
            let chosen = who == "user" && answer == val;
            let rule_chosen = ruled && answer == val;
            Answer {
                value: val.clone(),
                label: label.clone(),
                key: if whose { "0".into() } else { (k + 1).to_string() },
                chosen,
                rule_chosen,
                title: if rule_chosen { tx.t("decisions.rule_tip", &v(&[("rule", s(d, "by"))])) } else { String::new() },
            }
        })
        .collect();
    let cands = if whose {
        d.get("candidates")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
            .map(|(k, c)| {
                let piece = idstr(&c["piece"]);
                let st = Still { t: c.get("t").and_then(Value::as_u64).unwrap_or(0) as usize, piece: None, person: box4(&c["box"]) };
                Cand {
                    chosen: who == "user" && answer == piece,
                    piece,
                    name: Some(s(c, "name")).filter(|n| !n.is_empty()).map(String::from).unwrap_or_else(|| tx.s("decisions.unnamed")),
                    why: s(c, "why").into(),
                    key: if k < 9 { (k + 1).to_string() } else { String::new() },
                    pic: cx.crop_of(&st),
                }
            })
            .collect()
    } else {
        vec![]
    };
    let (rule_pick, rule_pick_title) = if ruled && !o.iter().any(|(v, _)| v == answer) {
        (format!("{} · {}", short_a(tx, answer), tx.s("decisions.rule")), tx.t("decisions.rule_tip_pick", &v(&[("rule", s(d, "by"))])))
    } else {
        (String::new(), String::new())
    };
    let mut chip_list = vec![];
    if cx.dev {
        if ruled {
            chip_list.push(Chip { text: format!("{answer} · {}", s(d, "by")), cls: "rule".into(), title: s(d, "by").into() });
        }
        for c in chips(&kind, &d["evidence"]) {
            chip_list.push(Chip { text: c, cls: String::new(), title: String::new() });
        }
        if let Some(gr) = d.get("graded").filter(|g| !g.is_null()) {
            chip_list.push(Chip { text: format!("grades: {}", idstr(gr)), cls: "grade".into(), title: "What your position grades (out/probe) say; for checking".into() });
        }
    }
    Detail {
        id: idstr(&d["id"]),
        key: format!("{}:{}", idstr(&d["id"]), s(d, "answer")),
        kind: kind.clone(),
        title,
        note: note(tx, &kind),
        pics,
        answers,
        cands,
        rule_pick,
        rule_pick_title,
        why: why_text(tx, d, cx.fps),
        agent: s(d, "via") == "agent",
        chips: chip_list,
    }
}

/// The frames picture `pic` of the selected question plays: its box's own frames, first detection to last, at most
/// LOOP_MAX_S seconds' worth (decisions.html Loops.plan). Empty: it stays still (a box seen once, one you drew).
pub fn loop_frames(ds: &[Value], ui: &Ui, cx: &Ctx, pic: usize) -> Value {
    let l = shown(ds, ui);
    let Some(d) = l.get(ui.cur) else { return json!([]) };
    let ss = stills(d);
    let wide = pic == ss.len() && ss.len() == 1;
    let Some(st) = ss.get(if wide { 0 } else { pic }) else { return json!([]) };
    let get = |k: &str| -> Option<&BTreeMap<usize, [f64; 4]>> { cx.boxes.get(&idstr(&d["about"][k])).and_then(|m| m.as_ref()) };
    let (pm, qm) = (if st.piece.is_some() { get("piece") } else { None }, if st.person.is_some() { get("person") } else { None });
    let Some(life) = pm.or(qm) else { return json!([]) };
    if cx.fps <= 0.0 || life.is_empty() {
        return json!([]);
    }
    let at = |m: Option<&BTreeMap<usize, [f64; 4]>>, t: usize| -> Option<[f64; 4]> {
        let m = m?;
        if let Some(b) = m.get(&t) {
            return Some(*b);
        }
        m.iter().min_by_key(|(u, _)| (**u as i64 - t as i64).abs()).map(|(_, b)| *b)
    };
    let (a, z) = (*life.keys().next().unwrap(), *life.keys().last().unwrap());
    let step = (((z - a + 1) as f64) / (LOOP_MAX_S * cx.fps)).ceil().max(1.0) as usize;
    let (_, _, half) = crop(&boxes_of(st));
    let mut out = vec![];
    let mut t = a;
    while t <= z {
        let f = Still { t, piece: if st.piece.is_some() { at(pm, t) } else { None }, person: if st.person.is_some() { at(qm, t) } else { None } };
        let c = if wide {
            cx.wide_of(&f)
        } else {
            let (x, y, _) = crop(&boxes_of(&f));
            Crop::new(cx.base, t, (x, y, half), f.piece, f.person)
        };
        out.push(c);
        t += step;
    }
    if out.len() < 2 {
        return json!([]);
    }
    json!({"frames": out, "span": [a, z], "fps": cx.fps})
}

/// A key on the page: what it does (decisions.html keydown). Returns ("answer", id, value) | ("move", ..) | none.
pub enum KeyAction {
    Move(usize),
    Answer(String, String),
    None,
}

pub fn key(ds: &[Value], ui: &Ui, k: &str) -> KeyAction {
    let l: Vec<&Value> = shown(ds, ui).into_iter().take(MAX).collect();
    match k {
        "j" | "Down" | "ArrowDown" => KeyAction::Move((ui.cur + 1).min(l.len().saturating_sub(1))),
        "k" | "Up" | "ArrowUp" => KeyAction::Move(ui.cur.saturating_sub(1)),
        _ => {
            let Some(d) = l.get(ui.cur) else { return KeyAction::None };
            let id = idstr(&d["id"]);
            if s(d, "kind") == "whose" && k.len() == 1 && k.as_bytes()[0].is_ascii_digit() {
                if k == "0" {
                    return KeyAction::Answer(id, SOMEONE_NEW.into());
                }
                let i = k.parse::<usize>().unwrap() - 1;
                return match d["candidates"].get(i) {
                    Some(c) => KeyAction::Answer(id, idstr(&c["piece"])),
                    None => KeyAction::None,
                };
            }
            match k {
                "1" | "2" | "3" => match opts(s(d, "kind")).get(k.parse::<usize>().unwrap() - 1) {
                    Some((a, _)) => KeyAction::Answer(id, a.to_string()),
                    None => KeyAction::None,
                },
                _ => KeyAction::None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn emoji_hash_matches_page_util() {
        let pal = json!([{"key": "a", "default": true, "emoji": ["0", "1", "2", "3", "4"]}]);
        // "12": 1*'1'(49) + 2*'2'(50) = 149; 149 % 5 = 4
        assert_eq!(emoji_for("12", &json!({}), &pal), "4");
        assert_eq!(emoji_for("@12", &json!({}), &pal), "4");
        assert_eq!(emoji_for("12", &json!({"emoji": {"12": "x"}}), &pal), "x");
    }
    #[test]
    fn crop_has_a_floor() {
        assert_eq!(crop(&[[0.0, 0.0, 10.0, 10.0]]), (5.0, 5.0, 70.0));
        assert_eq!(crop(&[[0.0, 0.0, 100.0, 50.0]]), (50.0, 25.0, 140.0));
    }
}
