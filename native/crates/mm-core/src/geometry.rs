//! Rectangles and the aspect fit the editor uses to place the video in its stage.
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl Rect {
    pub fn new(x: f64, y: f64, w: f64, h: f64) -> Rect {
        Rect { x, y, w, h }
    }
    pub fn contains(&self, px: f64, py: f64) -> bool {
        px >= self.x && py >= self.y && px < self.x + self.w && py < self.y + self.h
    }
    pub fn centre(&self) -> (f64, f64) {
        (self.x + self.w / 2.0, self.y + self.h / 2.0)
    }
    /// A `w` x `h` picture scaled to fit inside `self`, centred.
    pub fn fit(&self, w: f64, h: f64) -> Rect {
        if w <= 0.0 || h <= 0.0 {
            return *self;
        }
        let k = (self.w / w).min(self.h / h);
        let (fw, fh) = (w * k, h * k);
        Rect { x: self.x + (self.w - fw) / 2.0, y: self.y + (self.h - fh) / 2.0, w: fw, h: fh }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fit_letterboxes() {
        let r = Rect::new(0.0, 0.0, 1000.0, 1000.0).fit(3840.0, 2160.0);
        assert!((r.w - 1000.0).abs() < 1e-9);
        assert!((r.h - 562.5).abs() < 1e-9);
        assert!((r.y - 218.75).abs() < 1e-9);
        assert!(r.contains(500.0, 500.0));
        assert!(!r.contains(500.0, 100.0));
    }
}
