//! A project open in the app: the video, the engine's covers (for now the PoC dump), the user's edits.
use crate::dump::Dump;
use anyhow::Result;
use mm_core::cover::{unpack, FrameCovers};
use mm_core::project::{Edit, Edits};
use std::collections::BTreeMap;
use std::sync::Arc;

pub struct Project {
    pub video: String,
    pub dump: Option<Arc<Dump>>,
    pub edits: Edits,
}

/// One cover as the editor shows it: circle in video pixels.
#[derive(Clone, Debug, serde::Serialize)]
pub struct CoverView {
    pub i: usize,
    pub x: f32,
    pub y: f32,
    pub r: f32,
}

impl Project {
    /// `data` = a dump folder, or "" for a video with no covers.
    pub fn open(video: &str, data: &str) -> Result<Project> {
        let dump = if data.is_empty() { None } else { Some(Arc::new(Dump::load(data)?)) };
        Ok(Project { video: video.to_string(), dump, edits: Edits::default() })
    }

    /// Do the covers belong to this video? (The dump was made for a video of its size.)
    pub fn covers_fit(&self, w: u32, h: u32) -> bool {
        self.dump.as_ref().is_some_and(|d| d.meta.w == w && d.meta.h == h)
    }

    /// Frame t's rows with the edits applied.
    pub fn rows(&self, t: usize) -> Vec<f32> {
        let Some(d) = &self.dump else { return vec![] };
        let t = t.min(d.frames.len() - 1);
        self.edits.apply(t, d.rows(t))
    }

    pub fn frame_covers(&self, t: usize) -> Result<FrameCovers> {
        match &self.dump {
            Some(d) => d.frame_covers(t, Some(self.rows(t))),
            None => Ok(FrameCovers::default()),
        }
    }

    pub fn covers(&self, t: usize) -> Vec<CoverView> {
        unpack(&self.rows(t))
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let (x, y, r) = c.circle();
                CoverView { i, x, y, r }
            })
            .collect()
    }

    pub fn move_cover(&mut self, t: usize, i: usize, dx: f32, dy: f32) -> bool {
        if i >= self.covers(t).len() {
            return false;
        }
        self.edits.push(Edit::Move { t, i, dx, dy });
        true
    }

    /// Every edited frame's rows (for the export).
    pub fn edited_rows(&self) -> BTreeMap<usize, Vec<f32>> {
        match &self.dump {
            Some(d) => self.edits.by_frame(|t| d.rows(t.min(d.frames.len() - 1)).to_vec()),
            None => BTreeMap::new(),
        }
    }

    /// Decode the cover pictures in the background so the first frames draw without waiting on PNG decodes.
    pub fn warm(&self) {
        if let Some(d) = &self.dump {
            let d = d.clone();
            std::thread::spawn(move || {
                for f in &d.frames {
                    for h in &f.pics {
                        let _ = d.slice(h);
                    }
                }
            });
        }
    }
}
