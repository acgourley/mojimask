//! Cover records: what the compositor draws on one frame.
//!
//! Two levels:
//! - [`CoverRecord`] / [`FrameRecord`]: one cover as the engine describes it, before any drawing decision (head box,
//!   emoji key, head pose, lighting estimate, depth cut). [`FrameRecord::rows`] turns them into rows exactly as
//!   `track.cover_picture` + `gpu_draw.frame_covers` + `gpu_draw.Compositor._cover` do.
//! - [`CoverRow`]: one row of `STRIDE` (36) floats, the layout of `mojimask/gpu_draw.py`'s Metal kernel (and the WGSL
//!   port in mm-render).
//!
//! The row's fields (Python's `P[i]`):
//!   - `[0..3]` the drawn box: x0, y0 (top-left, whole pixels) and its size s in px (moves with the cover);
//!   - `[3]` which picture of the frame's `pics` the cover shows (remapped to an atlas slot by the compositor);
//!   - `[4..6]` motion blur: the movement over the shutter in box sizes (vx/s, vy/s) and the number of taps;
//!   - `[7]` flags (a float holding small integer bits; old rows hold 0 or 1): bit 0 lit, bit 1 posed (the head
//!     pose is drawn by the shader, see below), bit 2 mirrored (posed covers only);
//!   - `[8..11]` the key light (BGR, linear), `[11..14]` the ambient light (BGR), `[14..16]` the ramp gradient
//!     (x, y), `[16]` mix (strength), `[17]` the legibility floor, `[18]` saturation, `[19]` the footage's black level,
//!     `[20]` grain sigma (0..1 units), `[21]` how much of the grain is colour;
//!   - `[22]` depth cut on (1/0), `[23]` the face's depth, `[24]` the face margin, `[32]` the rim's margin ("apart"),
//!     `[33]` the head size (depth.zone_r);
//!   - `[25], [26], [28]`: the cover circle's centre and radius, in video pixels; `[27]` its half side (the cut's
//!     square);
//!   - `[29]` the box's margin in px (motion blur reaches past the box), `[30]` the mip level, `[31]` the picture's
//!     size in the atlas over the atlas size;
//!   - `[34]` roll in degrees and `[35]` yaw (-1..1), quantised as `head_pose.posed_sprite` does (1 degree, 0.05):
//!     posed covers only.
//!
//! **Head pose in the shader.** Python (`head_pose.posed_sprite`) draws each posed picture on the CPU: the sprite
//! tilted by roll and turned by yaw (a perspective "card" turn: narrower, far edge shorter, never below MIN_W of its
//! width) on a transparent canvas CANVAS (1.5) times its size, pasted at the head moved sideways by SHIFT * yaw / 2 head
//! sizes towards where it points. Here the atlas holds only the base sprite (one slice per emoji, not one per posed
//! picture) and the row carries roll, yaw and mirror; the box is the canvas's box (as Python's), and the shader maps
//! each canvas pixel back through the same perspective transform (closed form: the card's verticals stay vertical, so
//! u = (x + w) / (w (2 + c) - c x), v = 1/2 + y (1 + c u) / (2 lh), c = lh / rh - 1) and samples the base sprite.
//! The mip level is the canvas's, in sprite texels.
//! The rest (shape, pose, style) passes through untouched by the editor.
use std::sync::Arc;

pub const STRIDE: usize = 36;
const X: usize = 0;
const Y: usize = 1;
const PIC: usize = 3;
const CX: usize = 25;
const CY: usize = 26;
const R: usize = 28;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoverRow(pub [f32; STRIDE]);

impl CoverRow {
    pub fn from_slice(s: &[f32]) -> CoverRow {
        let mut a = [0.0; STRIDE];
        a.copy_from_slice(&s[..STRIDE]);
        CoverRow(a)
    }
    pub fn pic(&self) -> usize {
        self.0[PIC] as usize
    }
    pub fn set_pic(&mut self, slot: u32) {
        self.0[PIC] = slot as f32;
    }
    /// (cx, cy, r) in video pixels
    pub fn circle(&self) -> (f32, f32, f32) {
        (self.0[CX], self.0[CY], self.0[R])
    }
    /// Move the whole cover (box and circle) by (dx, dy) video pixels.
    pub fn move_by(&mut self, dx: f32, dy: f32) {
        self.0[X] += dx;
        self.0[Y] += dy;
        self.0[CX] += dx;
        self.0[CY] += dy;
    }
}

/// Rows packed as the GPU wants them (n * STRIDE floats) <-> CoverRows.
pub fn unpack(rows: &[f32]) -> Vec<CoverRow> {
    rows.chunks_exact(STRIDE).map(CoverRow::from_slice).collect()
}
pub fn pack(rows: &[CoverRow]) -> Vec<f32> {
    rows.iter().flat_map(|r| r.0).collect()
}

/// One cover picture as an atlas slice: premultiplied BGRA f16, TEX x TEX, every mip level.
pub struct Slice {
    pub n: u32,
    pub levels: Vec<Vec<half::f16>>,
}

/// The compositor's constants for a project (from the engine's meta).
#[derive(Clone, Debug, PartialEq)]
pub struct RenderMeta {
    /// atlas slice size and count
    pub tex: u32,
    pub slices: u32,
    pub sigma_s: f32,
    pub sigma_r: f32,
    pub face_zone: [f32; 2],
    pub face_dy: f32,
}

/// Everything the compositor needs to draw one frame's covers.
#[derive(Clone, Default)]
pub struct FrameCovers {
    /// n * STRIDE floats; `[3]` indexes `pics`
    pub rows: Vec<f32>,
    /// picture ids (hashes) this frame's covers use
    pub pics: Vec<String>,
    pub seed: f32,
    pub soft: f32,
    /// the depth guide: (values, width, height)
    pub near: Option<(Arc<Vec<f32>>, u32, u32)>,
}

impl FrameCovers {
    pub fn n(&self) -> usize {
        self.rows.len() / STRIDE
    }
    /// No covers (the video alone).
    pub fn clear(&mut self) {
        self.rows.clear();
        self.near = None;
    }
}


// ------------------------------------------------------------------ records: the engine's covers, before drawing

/// Python's `round()` (half to even), as `int(round(x))` in track.py / gpu_draw.py.
pub fn py_round(x: f64) -> f64 {
    x.round_ties_even()
}

/// head_pose.CANVAS, SHIFT, MIN_W; track.EYE_KEEP, EYE_GROW_MAX.
pub const CANVAS: f64 = 1.5;
pub const SHIFT: f64 = 0.12;
pub const MIN_W: f64 = 0.8;
pub const EYE_KEEP: f64 = 0.06;
pub const EYE_GROW_MAX: f64 = 1.2;
/// portrait.FRONT: a photo turned less than this counts as front-facing and is never mirrored.
pub const PORTRAIT_FRONT: f64 = 0.15;
/// portrait.PORTRAIT_KEY and PREFIX: the Portrait style's picture keys.
pub const PORTRAIT_KEY: &str = "\u{1F5BC}\u{FE0F}";
pub const PORTRAIT_PREFIX: &str = "portrait:";
/// Row flag bits in `[7]`.
pub const FLAG_LIT: u32 = 1;
pub const FLAG_POSED: u32 = 2;
pub const FLAG_MIRROR: u32 = 4;

pub fn is_portrait(key: &str) -> bool {
    key == PORTRAIT_KEY || key.starts_with(PORTRAIT_PREFIX)
}

/// Emoji drawn in profile (Apple Color Emoji), by the way the head points: -1 = image left, +1 = image right
/// (head_pose.FACING, checked by eye against the rendered glyphs). Front-facing ones and things with no head aren't
/// listed and never flip.
pub fn facing(key: &str) -> i32 {
    const LEFT: &str = "🦄🐧🐳🐴🐎🦓🐕🐩🐈🐖🐄🐂🐑🐏🐐🐃🦬🦒🐘🦣🦏🦛🐪🐫🦙🐅🐆🦍🦮🐓🦃🦆🦢🦤🦅🐦🐤🦜🦩🐊🐢🐍🐉🦖🦕🐬🐟🐠🐡🦈🐋🦭🐁🐀🐿🦫🦔🐇🦘🐛🐜🐝";
    const LEFT_MULTI: [&str; 4] = ["🐈\u{200D}⬛", "🐕\u{200D}🦺", "🕊\u{FE0F}", "🕊"];
    const RIGHT: &str = "🦌🦨🦡🐌🦎";
    let mut cs = key.chars();
    let single = match (cs.next(), cs.next()) {
        (Some(c), None) => Some(c),
        _ => None,
    };
    if let Some(c) = single {
        if LEFT.contains(c) {
            return -1;
        }
        if RIGHT.contains(c) {
            return 1;
        }
    }
    if LEFT_MULTI.contains(&key) {
        return -1;
    }
    0
}

/// head_pose.sprite_pose: the head's (roll, yaw, side) -> (roll, yaw, mirror) to draw cover `key` with. Emoji tilt and
/// turn as the head does (profile emoji mirrored to face the way the head is held: mirror_for); a portrait photo also
/// corrects for its own pose (portrait.match: its own tilt taken off, mirrored when the head faces the other way,
/// the card turn only for the turn the photo doesn't already show).
pub fn sprite_pose(key: &str, roll: f64, yaw: f64, side: i32, own_pose: Option<[f64; 2]>) -> (f64, f64, bool) {
    if is_portrait(key) {
        let [r0, y0] = own_pose.unwrap_or([0.0, 0.0]);
        let f = if y0.abs() < PORTRAIT_FRONT { 0 } else if y0 > 0.0 { 1 } else { -1 };
        let mirror = f != 0 && side != 0 && side != f;
        let shown = if mirror { -y0 } else { y0 };
        return (roll - if mirror { -r0 } else { r0 }, (yaw - shown).clamp(-1.0, 1.0), mirror);
    }
    let f = facing(key);
    (roll, yaw, f != 0 && side != 0 && side != f)
}

/// Where each emoji's eyes are (native/web/emoji_eyes.json): eye_y = the eye line as a fraction of the emoji picture's
/// height, reference = where real eyes sit in a face cover box, from its centre in box sizes.
#[derive(Clone, Debug, Default, serde::Deserialize)]
pub struct Eyes {
    pub reference: f64,
    pub eye_y: std::collections::HashMap<String, f64>,
}

impl Eyes {
    pub fn from_json(s: &str) -> serde_json::Result<Eyes> {
        serde_json::from_str(s)
    }

    /// track.eye_fit: (move, grow) for emoji `key` drawn at `scale` times the face box: move its centre down by `move`
    /// face sizes (negative = up) and draw it `grow` times larger. Lining the emoji's eyes up with the person's: an
    /// emoji whose eyes sit low in its picture is drawn higher, one with eyes high up lower. Moving it uncovers face at
    /// the other edge, so it is drawn larger as far as needed to still reach EYE_KEEP face sizes past the face box, up
    /// to EYE_GROW_MAX; past that it only goes part of the way. (0, 1) for emoji without eyes and for portraits.
    pub fn fit(&self, key: &str, scale: f64) -> (f64, f64) {
        let Some(&e) = self.eye_y.get(key) else { return (0.0, 1.0) };
        if scale <= 0.0 {
            return (0.0, 1.0);
        }
        let (h, m, r) = (0.5 / scale, EYE_KEEP / scale, self.reference / scale);
        let g = EYE_GROW_MAX.min(1.0f64.max((h + m - r) / (1.0 - e)).max((r + h + m) / e));
        let (lo, hi) = (h + m - g / 2.0, g / 2.0 - h - m);
        let d = if lo <= hi { (r - g * (e - 0.5)).max(lo).min(hi) } else { 0.0 };
        (d * scale, g)
    }

    /// track.eye_place: face box c = [cx, cy, s] -> the box to draw emoji `key` in.
    pub fn place(&self, key: &str, c: [f64; 3], scale: f64) -> [f64; 3] {
        let (d, g) = self.fit(key, scale);
        if d == 0.0 && g == 1.0 {
            return c;
        }
        [c[0], c[1] + d * c[2], c[2] * g]
    }
}

/// The lighting estimate for one cover on one frame (lighting.Lighting.auto): key light (BGR, linear, relative to a
/// white card), the left/right, top/bottom brightness gradient, the ambient's colour (normalised to mean 1).
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct LightEstimate {
    pub key: [f64; 3],
    pub grad: [f64; 2],
    pub amb: [f64; 3],
}

/// The depth cut planned for a cover (depth.Occlusion.plan): the face's depth, the face margin, the rim's margin
/// ("apart") and the head size.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct Occ {
    pub face: f64,
    pub margin: f64,
    pub rim: f64,
    pub head: f64,
}

/// One cover as the engine describes it.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct CoverRecord {
    /// head box: centre and size, video px (resolve()'s c[:3])
    pub c: [f64; 3],
    /// the picture: an emoji, or a portrait key (portrait.key_for); not needed for blur/pixelate/solid
    #[serde(default)]
    pub key: String,
    /// the head's smoothed pose (head_pose.track_angles): roll in degrees, yaw -1..1 (+ = facing image right), side
    /// (-1, 0, +1: the held facing); None = drawn upright
    #[serde(default)]
    pub pose: Option<(f64, f64, i32)>,
    /// portraits: the photo's own (roll, yaw) (portrait.own_pose)
    #[serde(default)]
    pub own_pose: Option<[f64; 2]>,
    /// the lighting estimate, when lighting is on
    #[serde(default, rename = "light_auto")]
    pub light: Option<LightEstimate>,
    /// the cover's movement per frame (FrameLighter.velocity: half the difference of its centre one frame either side)
    #[serde(default, rename = "vel_raw")]
    pub vel: Option<[f64; 2]>,
    #[serde(default)]
    pub occ: Option<Occ>,
}

/// The frame's lighting: the user's settings, the footage's (colourfulness, black level) and grain (8-bit sigma).
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct FrameLight {
    pub settings: crate::colour::LightSettings,
    pub film: [f64; 2],
    pub grain: f64,
}

/// One frame's covers as the engine describes them, far to near (track.draw_order), whitelisted people left out.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct FrameRecord {
    /// emoji | portrait | blur | pixelate | solid
    pub style: String,
    /// emoji size vs head box (1.25; strict mode 1.3x)
    pub scale: f64,
    pub seed: f64,
    pub soft: f64,
    /// the emoji's saturation when there's no lighting (lighting's Saturation slider)
    #[serde(default = "one")]
    pub saturation: f64,
    #[serde(default)]
    pub light: Option<FrameLight>,
    pub covers: Vec<CoverRecord>,
}

fn one() -> f64 {
    1.0
}

impl FrameRecord {
    /// The covers as circles [cx, cy, d] (the blur, pixelate and solid styles: track.cover).
    pub fn circles(&self) -> Vec<[f32; 3]> {
        self.covers.iter().map(|c| [c.c[0] as f32, c.c[1] as f32, (c.c[2] * self.scale) as f32]).collect()
    }

    /// The picture keys this frame draws, in first-use order; a row's `[3]` indexes this list.
    pub fn pics(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for c in &self.covers {
            if !out.contains(&c.key) {
                out.push(c.key.clone());
            }
        }
        out
    }

    /// The compositor's rows for an emoji/portrait frame (gpu_draw.frame_covers + Compositor._cover, with the head
    /// pose left to the shader). `size(key)` = the picture's size in px (emoji_sprite: 160; portraits 320); `tex` =
    /// the atlas slice size. Returns (rows, pics): `[3]` indexes pics.
    pub fn rows(&self, eyes: &Eyes, size: &dyn Fn(&str) -> u32, tex: u32) -> (Vec<f32>, Vec<String>) {
        let pics = self.pics();
        let mut rows = Vec::with_capacity(self.covers.len() * STRIDE);
        for cv in &self.covers {
            let w0 = size(&cv.key) as f64;
            let mut p = [0.0f64; STRIDE];
            // track.cover_picture: eyes on the person's eyes (emoji only), then tilt/turn with the head
            let mut cp = if self.style == "emoji" { eyes.place(&cv.key, cv.c, self.scale) } else { cv.c };
            let mut n = w0.min(tex as f64); // the picture's size in the atlas
            let mut canvas = None;
            let mut flags = 0u32;
            if let Some((r, y, side)) = cv.pose {
                let (r, y, mirror) = sprite_pose(&cv.key, r, y, side, cv.own_pose);
                // head_pose.posed_sprite: quantised (1 degree, 0.05 yaw)
                let (r, y) = (py_round(r), py_round(y * 20.0) / 20.0);
                let th = r.to_radians();
                let shift = SHIFT * y / 2.0;
                let off = (shift * th.cos(), shift * th.sin());
                cp = [cp[0] + off.0 * cp[2], cp[1] + off.1 * cp[2], cp[2] * CANVAS];
                let s_canvas = py_round(w0 * CANVAS);
                canvas = Some(s_canvas);
                flags |= FLAG_POSED | if mirror { FLAG_MIRROR } else { 0 };
                p[34] = r;
                p[35] = y;
                n = s_canvas.min(tex as f64);
            }
            let s = py_round(cp[2] * self.scale);
            let bx = if s < 4.0 {
                (-1e6, -1e6, 1.0) // too small to draw (as on the CPU), still counts as a cover
            } else {
                (py_round(cp[0] - s / 2.0), py_round(cp[1] - s / 2.0), s)
            };
            p[0] = bx.0;
            p[1] = bx.1;
            p[2] = bx.2;
            p[3] = pics.iter().position(|k| k == &cv.key).unwrap_or(0) as f64;
            let s = bx.2;
            let mut sat = self.saturation;
            if let (Some(fl), Some(est)) = (&self.light, &cv.light) {
                let st = &fl.settings;
                let (key, amb, grad) = crate::colour::apply_settings(est, st);
                let (lsat, black) = crate::colour::film_settings(fl.film[1], st);
                let shutter = if st.blur { st.shutter } else { 0.0 };
                if shutter > 0.0 {
                    if let Some([vx, vy]) = cv.vel {
                        let (vx, vy) = (vx * shutter, vy * shutter);
                        let ln = vx.hypot(vy);
                        if ln >= 1.0 {
                            p[4] = vx / s;
                            p[5] = vy / s;
                            p[6] = (ln.ceil() * 2.0 + 1.0).clamp(3.0, 64.0);
                            p[29] = ln / 2.0 + 1.0;
                        }
                    }
                }
                flags |= FLAG_LIT;
                p[8..11].copy_from_slice(&key);
                p[11..14].copy_from_slice(&amb);
                p[14..16].copy_from_slice(&grad);
                p[16] = st.mix;
                p[17] = st.floor;
                p[19] = black;
                p[20] = if st.grain { fl.grain } else { 0.0 } / 255.0;
                p[21] = fl.film[0].min(1.0);
                sat = lsat;
            }
            p[29] = p[29].max(1.0);
            p[7] = flags as f64;
            p[18] = sat;
            p[25] = cv.c[0];
            p[26] = cv.c[1];
            p[27] = cv.c[2] * self.scale / 2.0;
            p[28] = p[27];
            if let Some(o) = &cv.occ {
                p[22] = 1.0;
                p[23] = o.face;
                p[24] = o.margin;
                p[32] = o.rim;
                p[33] = o.head;
            }
            // the mip level, in the base sprite's texels; posed: the canvas's (as Python's canvas in the atlas)
            p[30] = if s > 0.0 { (n / s).log2().max(0.0) } else { 0.0 };
            if let Some(sc) = canvas {
                p[30] += (sc / n).log2();
            }
            p[31] = w0.min(tex as f64) / tex as f64;
            rows.extend(p.iter().map(|&v| v as f32));
        }
        (rows, pics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pose_rules() {
        assert_eq!(facing("🦄"), -1);
        assert_eq!(facing("🦌"), 1);
        assert_eq!(facing("😀"), 0);
        assert_eq!(facing("🐕\u{200D}🦺"), -1);
        // a profile emoji facing left, head held right: mirrored
        assert_eq!(sprite_pose("🦄", 5.0, 0.4, 1, None), (5.0, 0.4, true));
        // the portrait's own pose is taken off; a photo turned left (yaw -0.27) on a head facing right is mirrored
        let (r, y, m) = sprite_pose(PORTRAIT_KEY, 0.0, 0.0, 1, Some([-0.68, -0.27]));
        assert!(m && (r - -0.68).abs() < 1e-9 && (y - -0.27).abs() < 1e-9);
        assert_eq!(py_round(2.5), 2.0);
        assert_eq!(py_round(3.5), 4.0);
    }

    #[test]
    fn rows_flat_and_posed() {
        let eyes = Eyes::default();
        let cv = CoverRecord { c: [100.0, 80.0, 40.0], key: "😀".into(), pose: None, own_pose: None, light: None,
                               vel: None, occ: None };
        let mut fr = FrameRecord { style: "emoji".into(), scale: 1.25, seed: 3.0, soft: 0.15, saturation: 1.0,
                                   light: None, covers: vec![cv.clone()] };
        let (rows, pics) = fr.rows(&eyes, &|_| 160, 384);
        assert_eq!(pics, vec!["😀".to_string()]);
        assert_eq!(&rows[0..3], &[75.0, 55.0, 50.0]);
        assert_eq!(rows[7], 0.0);
        assert!((rows[30] - (160.0f32 / 50.0).log2()).abs() < 1e-6);
        fr.covers[0].pose = Some((10.4, 0.33, 0));
        let (rows, _) = fr.rows(&eyes, &|_| 160, 384);
        assert_eq!(rows[7] as u32, FLAG_POSED);
        assert_eq!((rows[34], rows[35]), (10.0, 0.35));
        assert_eq!(rows[2], 75.0); // the canvas: 1.5 x the box
    }
    #[test]
    fn move_keeps_box_and_circle_together() {
        let mut a = [0.0f32; STRIDE];
        a[0] = 10.0;
        a[1] = 20.0;
        a[3] = 2.0;
        a[25] = 15.0;
        a[26] = 25.0;
        a[28] = 5.0;
        let mut r = CoverRow(a);
        r.move_by(1.5, -2.0);
        assert_eq!(r.circle(), (16.5, 23.0, 5.0));
        assert_eq!((r.0[0], r.0[1]), (11.5, 18.0));
        assert_eq!(r.pic(), 2);
        let packed = pack(&[r, r]);
        assert_eq!(packed.len(), 2 * STRIDE);
        assert_eq!(unpack(&packed)[1], r);
    }
}
