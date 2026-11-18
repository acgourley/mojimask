//! After the graph: attach fragments (groups that aren't real people -- a few detections, or never a clear face)
//! to the real person they belong to. A fast mover (twirl's spinning girl) breaks into such fragments; they are
//! covered either way, but without this they belong to nobody (no person, no consistent emoji).
//! (Port of `mojimask/scan/attach.py`.)
//!
//! A fragment F goes to real person R when
//!   - F and R are never on screen at the same time as different heads;
//!   - motion fits: R's nearest sighting before F and/or after F (within MAX_GAP frames) is reachable from F
//!     (<= 1 head size + V head sizes per frame of the gap), on every side where R has one;
//!   - and F continues R: within CONT_GAP frames of where R left off (or leads into R), a detection >= CONT_SCORE,
//!     within CONT_CLOSE of the reachable distance, a face size within CONT_SIZE of R's, no third head within CROWD
//!     head sizes of F around then (a stray second detection on F's or R's own head doesn't count), and no other
//!     person fits.
//! Body matches (OSNet) and "on the path" (motion alone) were tried and removed: under the party clip's coloured light
//! body matches attached fragments to the wrong person even at >= 0.92 (jazz strangers' p99 is 0.84; that bar doesn't
//! carry over), and motion alone attached a wrong person and junk on jazz. Graded (2026-11-12): party 0 attached; jazz 0;
//! twirl 1 (the spinning girl at 175).
//! A fragment that fits nobody stays its own covered piece -- covered either way; this only gives it a person.
//!
//! CROWD (head sizes): continuation only where no other head is this near the fragment (party: jumps to a neighbour).
//! CONT_GAP: jazz: 0 wrong-person attachments at 3, 6 and 10.
use crate::geom::{self, Cs};
use crate::graph::State;
use crate::np::{self, M33};
use crate::questions::{Group, V};
use indexmap::IndexMap;
use std::collections::{BTreeMap, HashMap};

pub const MAX_GAP: i64 = 24;
pub const CROWD: f64 = 1.5;
pub const CONT_GAP: i64 = 3;
pub const CONT_SCORE: f64 = 0.6;
pub const CONT_CLOSE: f64 = 0.5;
pub const CONT_SIZE: f64 = 1.3;

fn near(a: &Cs, b: &Cs, k: f64) -> bool {
    np::hypot(a[0] - b[0], a[1] - b[1]) < k * (a[2] + b[2]) / 2.0
}

/// attach.attach: {fragment group: person group}, in fragment order. gd: questions::groups_data (the real people).
pub fn attach(st: &State, gd: &IndexMap<i64, Group>) -> IndexMap<i64, i64> {
    let mut inv: HashMap<i64, M33> = HashMap::new();
    let mut out = IndexMap::new();
    let mut at: BTreeMap<i64, Vec<(i64, Cs)>> = BTreeMap::new();
    for (g, gms) in &st.members {
        for c in gms.iter() {
            for (t, x) in &st.info[&c].obs {
                at.entry(*t).or_default().push((*g, *x));
            }
        }
    }
    let alone = |f: i64, r: i64, t: i64, c: &Cs| -> bool {
        for u in t - CONT_GAP..=t + CONT_GAP {
            let here = at.get(&u).map(|v| v.as_slice()).unwrap_or(&[]);
            let own: Vec<&Cs> = here.iter().filter(|(g, _)| *g == f || *g == r).map(|(_, x)| x).collect();
            for (g, x) in here {
                if *g == f || *g == r || !near(x, c, CROWD) || own.iter().any(|o| near(x, o, 1.0)) {
                    continue;
                }
                return false;
            }
        }
        true
    };
    for (&f, ms) in &st.members {
        if gd.contains_key(&f) {
            continue;
        }
        let obs = st.group_obs(ms);
        let mut fts: Vec<i64> = obs.keys().copied().collect();
        fts.sort();
        let mut conf = f64::NEG_INFINITY;
        for c in ms.iter() {
            let i = &st.info[&c];
            for t in i.obs.keys() {
                let s = i.score.get(t).copied().unwrap_or(0.0);
                if s > conf {
                    conf = s;
                }
            }
        }
        let mut conts = vec![];
        for (&r, rg) in gd {
            if st.any_apart(ms, &rg.ms) {
                continue;
            }
            let (mut ok, mut close, mut sides, mut gap, mut ratio) = (true, 9.0f64, 0, 99i64, 1.0f64);
            for (t_edge, d) in [(fts[0], -1i64), (fts[fts.len() - 1], 1)] {
                let mut best: Option<i64> = None;
                for &u in &rg.ts {
                    let x = d * (u - t_edge);
                    if 0 < x && x <= MAX_GAP && best.map(|b| (u - t_edge).abs() < (b - t_edge).abs()).unwrap_or(true) {
                        best = Some(u);
                    }
                }
                let Some(u) = best else { continue };
                let (pf, sf) = geom::to_room(&st.mats, t_edge, &obs[&t_edge], &mut inv);
                let (pr, sr) = geom::to_room(&st.mats, u, &rg.obs[&u], &mut inv);
                let dist = np::hypot(pf[0] - pr[0], pf[1] - pr[1]) / ((sf + sr) / 2.0);
                let reach = 1.0 + V * (u - t_edge).abs() as f64;
                if dist > reach {
                    ok = false;
                    break;
                }
                let x = dist / reach;
                if x < close {
                    close = x;
                }
                sides += 1;
                if (u - t_edge).abs() < gap {
                    gap = (u - t_edge).abs();
                    ratio = sf.max(sr) / sf.min(sr);
                }
            }
            if !ok || sides == 0 {
                continue;
            }
            if gap <= CONT_GAP && conf >= CONT_SCORE && close <= CONT_CLOSE && ratio <= CONT_SIZE && fts.iter().all(|t| alone(f, r, *t, &obs[t])) {
                conts.push(r);
            }
        }
        if conts.len() == 1 {
            out.insert(f, conts[0]);
        }
    }
    out
}
