//! "Frame t": slots on a constant-rate timeline, defined once (native/docs/stack-decision.md, "Found along the way").
//!
//! Slot t covers presentation times `[video_start + (t - 0.5)/fps, video_start + (t + 0.5)/fps)`. A stored frame with
//! stream time `pts` is shown from the first slot whose centre it doesn't start after, and stays until the next stored
//! frame takes over (video_io._frames_avf; the same rule the GStreamer decoder applies). Seeks aim just before slot
//! t's upper boundary so the decoder's first frame out is the one slot t shows.
use serde::{Deserialize, Serialize};

/// Frame rates a camera or editor really writes; a measured rate within 2 % snaps to one of them.
pub const RATES: [f64; 17] =
    [14.985, 15., 23.976, 24., 25., 29.97, 30., 47.952, 48., 50., 59.94, 60., 90., 100., 119.88, 120., 240.];

pub fn snap_fps(fps: f64) -> Option<f64> {
    let r = RATES.iter().cloned().min_by(|a, b| (fps - a).abs().total_cmp(&(fps - b).abs()))?;
    if (fps - r).abs() <= 0.02 * r {
        Some(r)
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct Timeline {
    pub fps: f64,
    /// stream time of slot 0 (the edit list's presentation start)
    pub video_start: f64,
    /// number of slots
    pub n: usize,
}

impl Timeline {
    pub fn new(fps: f64, video_start: f64, duration: f64) -> Timeline {
        Timeline { fps, video_start, n: ((duration * fps).round() as usize).max(1) }
    }

    /// Where to seek for slot t (seconds of stream time).
    pub fn seek_target(&self, t: usize) -> f64 {
        (self.video_start + (t as f64 + 0.5) / self.fps - 1e-6).max(0.0)
    }

    /// Does a stored frame starting at `pts` start at or before slot t (so it is a candidate to show there)?
    pub fn starts_by(&self, pts: f64, t: usize) -> bool {
        (pts - self.video_start) * self.fps < t as f64 + 0.5
    }

    /// The first slot a stored frame at `pts` shows in.
    pub fn first_slot(&self, pts: f64) -> usize {
        let x = (pts - self.video_start) * self.fps;
        ((x - 0.5).floor() + 1.0).max(0.0) as usize
    }

    /// The nominal presentation time of slot t (stream seconds).
    pub fn time_of(&self, t: usize) -> f64 {
        self.video_start + t as f64 / self.fps
    }

    pub fn clamp(&self, t: i64) -> usize {
        t.clamp(0, self.n.saturating_sub(1) as i64) as usize
    }
}

/// Export timestamps: slot t at an exact rational frame rate, in nanoseconds (no drift over long clips).
pub fn slot_ns(t: usize, framerate: (i32, i32)) -> u64 {
    (t as u128 * 1_000_000_000u128 * framerate.1 as u128 / framerate.0.max(1) as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snaps_ntsc() {
        assert_eq!(snap_fps(23.98), Some(23.976));
        assert_eq!(snap_fps(29.5), Some(29.97));
        assert_eq!(snap_fps(37.0), None);
    }

    #[test]
    fn slots_round_trip() {
        let tl = Timeline::new(24000.0 / 1001.0, 0.0, 127.0 * 1001.0 / 24000.0);
        assert_eq!(tl.n, 127);
        for t in 0..tl.n {
            let pts = tl.time_of(t);
            assert_eq!(tl.first_slot(pts), t);
            assert!(tl.starts_by(pts, t));
            assert!(!tl.starts_by(pts, t.saturating_sub(1)) || t == 0);
            assert!(tl.seek_target(t) > pts && tl.seek_target(t) < tl.time_of(t + 1));
        }
    }

    #[test]
    fn edit_list_start() {
        // a clip whose first shown frame starts inside the edit: stream time is clipped to video_start
        let tl = Timeline { fps: 25.0, video_start: 2.0, n: 10 };
        assert_eq!(tl.first_slot(2.0), 0);
        assert_eq!(tl.first_slot(1.99), 0);
        assert_eq!(tl.first_slot(2.04), 1);
        assert_eq!(tl.clamp(-3), 0);
        assert_eq!(tl.clamp(99), 9);
    }

    #[test]
    fn export_ns() {
        assert_eq!(slot_ns(0, (24000, 1001)), 0);
        assert_eq!(slot_ns(24000, (24000, 1001)), 1001 * 1_000_000_000);
    }
}
