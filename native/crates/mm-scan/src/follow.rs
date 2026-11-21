//! The scan's follower: follow a head through frames without (usable) detections -- the checked rule. join uses it to
//! bridge gaps between tracklets, cover to carry covers across gaps and past a track's ends, blips to check a blip.
//! (Port of `mojimask/scan/follow.py`; the pixel work -- corner flow, the face check, the patch likeness -- is behind
//! [`crate::evidence`].)
//!
//! Per frame, two candidate positions: where corner flow moves the box ("flow"), and where the camera alone
//! moves it ("camera", holding still in the room).
//!   1. a face found at the flow position   -> follow, re-centred on that face
//!   2. else a face at the camera position  -> stay, re-centred on that face
//!   3. else the patch at the flow position still looks like the face did when last seen (NCC >= LOOKS)
//!                                           -> follow
//!   4. else                                 -> hold (camera); heading out of the picture -> keep its own velocity
//! Stops: the box is mostly outside the picture ("out"), or `limit` frames; with lost=True (covers, not joins) also where
//! the face is lost ("lost", see face_motion below).
//! "A face found" = RetinaFace (reid.landmarks) finds a face (score >= FACE) whose centre is within NEAR head sizes of the
//! box centre; re-centring moves the box onto that face's head box (track.face_to_head), cancelling drift.
//! Evidence for this rule: jazz labels, and six clips without labels.
//!
//! These thresholds and the frames' pixels (2026-11-20). Since the scan reads its frames from the video and not from
//! JPEG copies, jazz asks 14 questions where it asked 11 and splits one person. Retuning FACE and LOOKS doesn't bring
//! that back: every pair of 0.4, 0.5 and 0.6 gives the same benchmark line on the video's pixels (asked 14, needless
//! covers 5, people split 1), though the chains differ, and on the JPEG copies' pixels FACE 0.4 with
//! LOOKS 0.4, 0.5 and 0.6 all ask 11 (the last splits one person as well). join's MEET: 0.3 splits three people; 0.5 splits none but
//! leaves a second face uncovered. So they stay; what differs is which of two near-equal outcomes a follow or a face
//! match falls into (graph.rs, "Known").
//!
//! The face's own flow (follow.face_motion, behind [`crate::evidence::Motion::face_motion`]): corners in the head box
//! tracked both ways, then one similarity fitted to them. On a visible face nearly all corners agree (jazz, the dancer
//! bouncing: 0.6-1.0 of them; the move it gives lands within 0.01-0.05 head sizes of her next detection, while the
//! detections themselves jump up to 0.2). When someone passes in front, the corners go with them or fail: under
//! LOST_INL agree. That's "lost" -- but only where the rest of the picture tracks fine (Clip.trackable >= STEADY_FRAME;
//! in a whip pan everything fails and a smeared face can still be there; party_clip_c's pan, frames 180-238: 0.00-0.48,
//! never under 0.64 on jazz's steady shot) and the box, with a head's width around it, is inside the picture (corners
//! at the edge fail too: party_clip_b box 120, a face entering at the right edge, was lost while in plain view). Few
//! corners at all ("no texture": a dark or flat face) says nothing.
use crate::evidence::Evidence;
use crate::geom::{self, Cs};
use crate::np::M23;
use std::collections::BTreeMap;
use std::path::Path;

pub const FACE: f64 = 0.5;
pub const LOOKS: f64 = 0.5;
pub const OUT: f64 = 0.3;
pub const TS: i64 = 32;
pub const LOST_INL: f64 = 0.3;
pub const STEADY_FRAME: f64 = 0.6;
pub const LOST_INSIDE: f64 = 0.9;

/// The clip as the follower sees it: camera motion, frame count and size.
pub struct Clip {
    pub mats: Vec<M23>,
    pub n: i64,
    pub w: f64,
    pub h: f64,
}

impl Clip {
    pub fn open(proj: &Path) -> Result<Clip, String> {
        // the frame count and size as the detect step recorded them (Python counted PROJECT/frames and read frame
        // 0's shape, the same numbers)
        let scan = mm_core::scan::Scan::open(proj)?;
        let n = scan.frames().ok_or("meta: frames")? as i64;
        let (w, h) = scan.size().ok_or("meta: width, height")?;
        Ok(Clip { mats: scan.mats()?, n, w, h })
    }
    pub fn camera(&self, a: i64, b: i64) -> M23 {
        geom::camera(&self.mats, a, b)
    }
}

#[derive(Clone, Debug, Default)]
pub struct FPath {
    pub boxes: BTreeMap<i64, Cs>,
    pub how: BTreeMap<i64, &'static str>,
    /// "limit" | "out" | "met" | "clip end" | "lost"
    pub why: &'static str,
    pub met: Option<(i64, i64)>,
}

pub type Meet<'a> = &'a dyn Fn(i64, &Cs) -> Option<i64>;
pub type Seen<'a> = &'a dyn Fn(i64, &Cs) -> bool;

/// follow.follow. lost=true: stop where the face is lost (face_motion), unless a face is found there or seen(u, box)
/// -- a weak detection on it -- says it's still showing.
#[allow(clippy::too_many_arguments)]
pub fn follow(clip: &Clip, ev: &Evidence, start_t: i64, c0: &Cs, direction: i64, limit: i64, meet: Option<Meet>, lost: bool,
              seen: Option<Seen>) -> FPath {
    let mut c = *c0;
    let mut anchor = (start_t, c);
    let mut v = [0.0f64; 2];
    let mut p = FPath { why: "limit", ..Default::default() };
    let mut u = start_t;
    for _ in 0..limit {
        let u2 = u + direction;
        if !(0 <= u2 && u2 < clip.n) {
            p.why = "clip end";
            break;
        }
        let m = clip.camera(u, u2);
        let cam = geom::warp_cs(&c, &m);
        let fl = match ev.motion.flow(u, u2, &c) {
            Some(f) => [c[0] + f.dx, c[1] + f.dy, cam[2]],
            None => cam,
        };
        let (s, head) = ev.faces.face_near(u2, &fl);
        let (new, how): (Cs, &'static str);
        if s >= FACE {
            new = head.unwrap();
            how = "face at flow";
        } else {
            let (s, head) = ev.faces.face_near(u2, &cam);
            if s >= FACE {
                new = head.unwrap();
                how = "face at camera";
            } else if lost
                && matches!(ev.motion.face_motion(u, u2, &c).1.as_str(), "lost" | "incoherent")
                && geom::inside_frac(&[c[0], c[1], 2.0 * c[2]], clip.w, clip.h)
                    .min(geom::inside_frac(&[cam[0], cam[1], 2.0 * cam[2]], clip.w, clip.h))
                    >= LOST_INSIDE
                && ev.motion.trackable(u, u2) >= STEADY_FRAME
                && !seen.map(|f| f(u2, &cam) || f(u2, &fl)).unwrap_or(false)
            {
                p.why = "lost";
                break;
            } else if ev.motion.looks(u2, &fl, anchor.0, &anchor.1) >= LOOKS {
                new = fl;
                how = "looks the same";
            } else if geom::heading_out(&cam, v, clip.w, clip.h) {
                new = [cam[0] + v[0], cam[1] + v[1], cam[2]];
                how = "exit";
            } else {
                new = cam;
                how = "hold";
            }
        }
        if how.starts_with("face") {
            anchor = (u2, new);
        }
        if how != "hold" {
            v = [0.5 * v[0] + 0.5 * (new[0] - cam[0]), 0.5 * v[1] + 0.5 * (new[1] - cam[1])];
        }
        c = new;
        u = u2;
        p.boxes.insert(u, c);
        p.how.insert(u, how);
        if geom::inside_frac(&c, clip.w, clip.h) < OUT {
            p.why = "out";
            break;
        }
        if let Some(meet) = meet {
            if let Some(hit) = meet(u, &c) {
                p.why = "met";
                p.met = Some((u, hit));
                break;
            }
        }
    }
    p
}
