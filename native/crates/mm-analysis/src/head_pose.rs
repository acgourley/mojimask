//! Head pose for covers (prototype): tilt (roll) and left/right turn (yaw) per covered head, so the emoji
//! can lean and turn with the face.
//!
//! Source: the COCO-17 pose skeletons the scan already saves (pose_full.json) -- eyes, nose, ears. No new model.
//!   roll  angle of the eye line (subject's right eye -> left eye, i.e. image left -> right when facing camera);
//!         ears as fallback when the eyes aren't both seen.
//!   yaw   where the nose sits along the eye line: 0 = centred (facing camera), +-1 = at/after an eye (~profile).
//!         Sign is in IMAGE terms: + = face pointing towards image right. One-eye profiles use nose vs visible ear.
//! Per track piece the raw angles are smoothed over time (confidence-weighted Gaussian) with a weak pull to
//! neutral, so missing or noisy frames drift back to an upright, front-facing emoji instead of jittering.
//!
//! posed_sprite() (Python) draws any RGBA sprite rotated by roll and turned by yaw (a perspective "card" turn:
//! narrower, far edge shorter), never shrinking the cover below MIN_W of its width. That drawing is mm-render's
//! (a shader transform); this module ports the angles: raw_angles, track_angles, FACING, mirror_for, and the
//! posed-card geometry constants the drawing shares. sprite_pose's portrait branch (portrait.match, a photo's own
//! pose) is not here: it needs the portrait module.
//!
//! Port of mojimask/head_pose.py. Parity: native/tools/parity/analysis_pose.py.
use crate::np;
use crate::pose_util::{self as pu, LEAR, LEYE, NOSE, REAR, REYE};
use std::collections::BTreeMap;

/// keypoints trusted for angles: stricter than pose_util.VIS (0.3) on purpose
pub const POSE_ANGLE_VIS: f64 = 0.4;
/// degrees
pub const MAX_ROLL: f64 = 40.0;
/// sideways move towards the facing direction, in half-widths
pub const SHIFT: f64 = 0.12;
/// turned card keeps at least this fraction of its width (privacy: coverage)
pub const MIN_W: f64 = 0.8;
/// posed sprite canvas, in sprite sizes (room for a 40 deg tilt of the turned card)
pub const CANVAS: f64 = 1.5;
/// flip hysteresis on the smoothed yaw: commit to a side past 0.35, hold it until the head has clearly come back
/// past 0.15 the other way (no flicker near front)
pub const SIDE_ON: f64 = 0.35;
pub const SIDE_OFF: f64 = 0.15;

/// (roll_deg, yaw, roll_conf, yaw_conf) from one pose_full row; confs 0 when not measurable.
pub fn raw_angles(p: &[f64]) -> (f64, f64, f64, f64) {
    let k = pu::kps(p);
    let c = |i: usize| k[i][2];
    let xy = |i: usize| [k[i][0], k[i][1]];
    let (mut roll, mut yaw, mut rc, mut yc) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    if c(LEYE) > POSE_ANGLE_VIS && c(REYE) > POSE_ANGLE_VIS {
        let v = [xy(LEYE)[0] - xy(REYE)[0], xy(LEYE)[1] - xy(REYE)[1]]; // image left -> right when facing the camera
        let d = (v[0] * v[0] + v[1] * v[1]).sqrt();
        if d > 1.5 && v[0] > 0.0 {
            roll = v[1].atan2(v[0]).to_degrees();
            rc = c(LEYE).min(c(REYE));
            if c(NOSE) > POSE_ANGLE_VIS {
                let u = [v[0] / d, v[1] / d];
                let mid = [(xy(LEYE)[0] + xy(REYE)[0]) / 2.0, (xy(LEYE)[1] + xy(REYE)[1]) / 2.0];
                let a = [xy(NOSE)[0] - mid[0], xy(NOSE)[1] - mid[1]];
                let r = (a[0] * u[0] + a[1] * u[1]) / d;
                yaw = np::clip(r / 0.45, -1.0, 1.0);
                yc = rc.min(c(NOSE));
            }
        }
    } else if c(LEAR) > POSE_ANGLE_VIS && c(REAR) > POSE_ANGLE_VIS && xy(LEAR)[0] > xy(REAR)[0] {
        let v = [xy(LEAR)[0] - xy(REAR)[0], xy(LEAR)[1] - xy(REAR)[1]];
        roll = v[1].atan2(v[0]).to_degrees();
        rc = 0.5 * c(LEAR).min(c(REAR));
    }
    if yc == 0.0 && c(NOSE) > POSE_ANGLE_VIS {
        // profile: nose vs the one visible ear
        let ears: Vec<usize> = [LEAR, REAR].into_iter().filter(|&e| c(e) > POSE_ANGLE_VIS).collect();
        if ears.len() == 1 {
            let dx = xy(NOSE)[0] - xy(ears[0])[0];
            if dx.abs() > 1.0 {
                yaw = dx.signum();
                yc = 0.6 * c(NOSE).min(c(ears[0]));
            }
        }
    }
    if roll.abs() > MAX_ROLL {
        rc = 0.0;
    }
    (roll, yaw, rc, yc)
}

/// Emoji drawn in profile (Apple Color Emoji), by the way the head points: -1 = image left, +1 = image right.
/// Checked by eye against the rendered glyphs; front-facing ones (faces, owl, otter, chick ...) and things with no
/// head (food, planets) aren't listed and never flip.
pub fn facing_of(ch: &str) -> i32 {
    const LEFT: &str = "🦄🐧🐳🐴🐎🦓🐕🐩🐈🐖🐄🐂🐑🐏🐐🐃🦬🦒🐘🦣🦏🦛🐪🐫🦙🐅🐆🦍🦮🐓🦃🦆🦢🦤🦅🐦🐤🦜🦩🐊🐢🐍🐉🦖🦕🐬🐟🐠🐡🦈🐋🦭🐁🐀🐿🦫🦔🐇🦘🐛🐜🐝";
    const LEFT_MULTI: [&str; 4] = ["🐈\u{200d}⬛", "🐕\u{200d}🦺", "🕊️", "🕊"]; // more than one code point each
    const RIGHT: &str = "🦌🦨🦡🐌🦎";
    let one = |s: &str| {
        let mut it = ch.chars();
        matches!((it.next(), it.next()), (Some(c), None) if s.contains(c))
    };
    // dict order: the single code points, then the multi ones (later keys win), then the right-facing ones
    if one(RIGHT) {
        1
    } else if LEFT_MULTI.contains(&ch) || one(LEFT) {
        -1
    } else {
        0
    }
}

/// Draw emoji ch mirrored? side = the head's held facing (-1 left, +1 right, 0 undecided).
pub fn mirror_for(ch: &str, side: i32) -> bool {
    let f = facing_of(ch);
    f != 0 && side != 0 && side != f
}

/// One cover in a frame (track.resolve() output): track id, head box, kind, and the piece it belongs to (the track
/// id when absent).
#[derive(Clone, Debug)]
pub struct Item {
    pub tid: String,
    pub c: [f64; 3],
    pub kind: String,
    pub piece: Option<String>,
}

impl Item {
    pub fn piece(&self) -> &str {
        self.piece.as_deref().unwrap_or(&self.tid)
    }
}

/// {piece: {t: (roll_deg, yaw, side)}} for every cover in per_frame (resolve() output), smoothed per piece.
/// side = which way the head is held to face (-1 image left, +1 image right, 0 not yet clear), with
/// hysteresis: decides whether a profile emoji (FACING) is mirrored.
pub fn track_angles(per_frame: &[Vec<Item>], poses: &[Vec<Vec<f64>>], sigma: f64, prior: f64) -> BTreeMap<String, BTreeMap<usize, (f64, f64, i32)>> {
    let mut samples: BTreeMap<String, Vec<[f64; 5]>> = BTreeMap::new();
    let mut present: Vec<(String, Vec<usize>)> = Vec::new(); // insertion order, as the dict
    for (t, items) in per_frame.iter().enumerate() {
        let rows: &[Vec<f64>] = if t < poses.len() { &poses[t] } else { &[] };
        for it in items {
            let piece = it.piece().to_string();
            match present.iter_mut().find(|(p, _)| *p == piece) {
                Some((_, v)) => v.push(t),
                None => present.push((piece.clone(), vec![t])),
            }
            if !rows.is_empty() {
                if let Some(p) = pu::pose_for_head(rows, &it.c, 0.25) {
                    let (a, b, c, d) = raw_angles(p);
                    samples.entry(piece).or_default().push([t as f64, a, b, c, d]);
                }
            }
        }
    }
    let mut out = BTreeMap::new();
    let r = (3.0 * sigma) as i64 as f64;
    for (piece, ts) in &present {
        let s = samples.get(piece).cloned().unwrap_or_default();
        let mut res = BTreeMap::new();
        let mut side = 0i32;
        for &t in ts {
            let (roll, yaw);
            if !s.is_empty() {
                let m: Vec<&[f64; 5]> = s.iter().filter(|r0| (r0[0] - t as f64).abs() <= r).collect();
                let g: Vec<f64> = m.iter().map(|r0| (-0.5 * ((r0[0] - t as f64) / sigma).powi(2)).exp()).collect();
                let wr: Vec<f64> = g.iter().zip(&m).map(|(g, r0)| g * r0[3]).collect();
                let wy: Vec<f64> = g.iter().zip(&m).map(|(g, r0)| g * r0[4]).collect();
                let nr: Vec<f64> = wr.iter().zip(&m).map(|(w, r0)| w * r0[1]).collect();
                let ny: Vec<f64> = wy.iter().zip(&m).map(|(w, r0)| w * r0[2]).collect();
                roll = np::sum(&nr) / (np::sum(&wr) + prior);
                yaw = np::sum(&ny) / (np::sum(&wy) + prior);
            } else {
                roll = 0.0;
                yaw = 0.0;
            }
            if yaw >= SIDE_ON {
                side = 1;
            } else if yaw <= -SIDE_ON {
                side = -1;
            } else if (side == 1 && yaw < -SIDE_OFF) || (side == -1 && yaw > SIDE_OFF) {
                side = 0; // clearly turned back the other way but not yet committed: back to the emoji's own way
            }
            res.insert(t, (roll, yaw, side));
        }
        out.insert(piece.clone(), res);
    }
    out
}

/// Head box c=[cx,cy,s] -> the box to paste a posed sprite canvas into (posed_box).
pub fn posed_box(c: &[f64; 3], off: (f64, f64)) -> [f64; 3] {
    [c[0] + off.0 * c[2], c[1] + off.1 * c[2], c[2] * CANVAS]
}
