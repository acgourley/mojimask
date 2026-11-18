//! Long jobs (export now; the scan later): progress a UI can poll and a way to cancel.
use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    #[default]
    Idle,
    Running,
    Done,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Progress {
    pub state: JobState,
    pub done: usize,
    pub total: usize,
    pub message: String,
    pub seconds: f64,
    /// where the export goes (Reveal in Finder)
    pub out: String,
}

#[derive(Clone, Default)]
pub struct Job {
    pub progress: Arc<Mutex<Progress>>,
    cancel: Arc<AtomicBool>,
}

impl Job {
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }
    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
    pub fn set(&self, f: impl FnOnce(&mut Progress)) {
        f(&mut self.progress.lock().unwrap());
    }
    pub fn get(&self) -> Progress {
        self.progress.lock().unwrap().clone()
    }
}
