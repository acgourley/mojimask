//! The project model. For now: the user's edits on top of the engine's covers (a cover moved on a frame). Later the
//! facts log (`mojimask/facts.py`) lives here too, as the source of truth answers and edits derive from.
use crate::cover::{pack, unpack, CoverRow};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One user edit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Edit {
    /// cover `i` of frame `t` moved by (dx, dy) video pixels
    Move { t: usize, i: usize, dx: f32, dy: f32 },
}

/// The edits, in the order made. Replaying them over the engine's rows gives what's drawn.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Edits {
    pub log: Vec<Edit>,
}

impl Edits {
    pub fn push(&mut self, e: Edit) {
        // consecutive moves of the same cover on the same frame merge (a drag is one edit)
        if let (Some(Edit::Move { t, i, dx, dy }), Edit::Move { t: t2, i: i2, dx: dx2, dy: dy2 }) = (self.log.last_mut(), &e) {
            if t == t2 && i == i2 {
                *dx += dx2;
                *dy += dy2;
                return;
            }
        }
        self.log.push(e);
    }

    /// The frames that have edits.
    pub fn frames(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.log.iter().map(|e| match e { Edit::Move { t, .. } => *t }).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Apply the edits of frame t to that frame's engine rows.
    pub fn apply(&self, t: usize, rows: &[f32]) -> Vec<f32> {
        let mut r: Vec<CoverRow> = unpack(rows);
        for e in &self.log {
            match *e {
                Edit::Move { t: et, i, dx, dy } if et == t => {
                    if let Some(c) = r.get_mut(i) {
                        c.move_by(dx, dy);
                    }
                }
                _ => {}
            }
        }
        pack(&r)
    }

    /// All frames' edited rows, for the export.
    pub fn by_frame(&self, rows_of: impl Fn(usize) -> Vec<f32>) -> BTreeMap<usize, Vec<f32>> {
        self.frames().into_iter().map(|t| (t, self.apply(t, &rows_of(t)))).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cover::STRIDE;
    #[test]
    fn drags_merge_and_apply() {
        let mut e = Edits::default();
        e.push(Edit::Move { t: 3, i: 0, dx: 1.0, dy: 0.0 });
        e.push(Edit::Move { t: 3, i: 0, dx: 1.0, dy: 2.0 });
        e.push(Edit::Move { t: 4, i: 1, dx: 5.0, dy: 5.0 });
        assert_eq!(e.log.len(), 2);
        assert_eq!(e.frames(), vec![3, 4]);
        let rows = vec![0.0f32; 2 * STRIDE];
        let a = e.apply(3, &rows);
        assert_eq!((a[25], a[26]), (2.0, 2.0));
        assert_eq!(a[STRIDE + 25], 0.0);
        let b = e.apply(4, &rows);
        assert_eq!(b[STRIDE + 25], 5.0);
        assert_eq!(e.apply(9, &rows), rows);
        let js = serde_json::to_string(&e).unwrap();
        assert_eq!(serde_json::from_str::<Edits>(&js).unwrap(), e);
    }
}
