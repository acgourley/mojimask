//! The state tree's shape. QML binds to it (`store.state.editor.t`); `/state` and `mm-cli state` print it.
//! Every field is plain data. What survives a restart is `Saved` (written to `Config::state_file`).
use mm_core::project::Edits;
use mm_engine::jobs::Progress;
use mm_engine::project::CoverView;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize)]
pub struct State {
    pub app: AppState,
    pub editor: Editor,
    pub export: Progress,
    pub readback: Readback,
    /// the Projects page (library.rs)
    pub library: crate::library::LibraryState,
    /// a scan under way, or a project opening
    pub scan: crate::library::ScanState,
    /// the open project
    pub project: crate::library::ProjectState,
    pub decisions: crate::decide::DecisionsState,
    pub people: crate::people::PeopleState,
    /// the Export page (export.rs publish_page)
    pub export_page: serde_json::Value,
    /// the result after a project's first scan (library.rs result_view)
    pub result: serde_json::Value,
}

#[derive(Clone, Debug, Serialize)]
pub struct AppState {
    /// the page shown: projects, scan, decisions, people, editor, export
    pub page: String,
    pub pages: Vec<String>,
    pub dev: bool,
    pub root: String,
    /// bumped by `ping`; a cheap way to see a state change round-trip
    pub pings: u64,
}

impl Default for AppState {
    fn default() -> Self {
        AppState {
            page: "projects".into(),
            pages: ["projects", "scan", "decisions", "people", "editor", "export", "result"].map(String::from).to_vec(),
            dev: false,
            root: String::new(),
            pings: 0,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Video {
    pub path: String,
    pub name: String,
    /// upright display size (rotation applied, square pixels)
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub frames: usize,
    /// seconds on the timeline (frames / fps)
    pub duration: f64,
    pub has_audio: bool,
    /// the video is open and probed
    pub ready: bool,
    /// public domain (test_clips/jazz.mp4, out/poc/clips/): an agent may look at its pictures
    pub public: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Editor {
    pub video: Video,
    /// the cover data (a PoC dump folder) or ""
    pub data: String,
    /// the frame asked for
    pub t: usize,
    /// the frame in the video item's texture (-1: none yet)
    pub shown: i64,
    pub playing: bool,
    /// while playing, the frames drawn per second over the last second (0: paused, or not measured yet); the
    /// transport says so when it is under the clip's rate
    pub play_fps: f64,
    /// the selected cover (-1: none)
    pub selected: i64,
    /// the covers of the shown frame, circles in video pixels
    pub covers: Vec<CoverView>,
    /// the cover data was made for this video (same size)
    pub covers_fit: bool,
    /// how many edits (moves) have been made
    pub edits: usize,
    pub status: String,
    /// a project's video (its covers from the project's draw records), not a PoC dump
    pub project: bool,
    /// the records' style, how many frames they cover, the depth model is on
    pub style: String,
    pub records: usize,
    pub depth_on: bool,
    /// the project Editor's view of the shown frame: boxes, the panel (editor.rs)
    pub view: serde_json::Value,
}

impl Default for Editor {
    fn default() -> Self {
        Editor {
            video: Video::default(),
            data: String::new(),
            t: 0,
            shown: -1,
            playing: false,
            play_fps: 0.0,
            selected: -1,
            covers: vec![],
            covers_fit: false,
            edits: 0,
            status: String::new(),
            project: false,
            style: String::new(),
            records: 0,
            depth_on: false,
            view: serde_json::Value::Null,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Readback {
    /// "", "pending", "done", "refused" or "failed: ..."
    pub state: String,
    pub path: String,
    pub frame: i64,
}

/// What a restart restores.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Saved {
    pub page: String,
    pub video: String,
    pub data: String,
    pub t: usize,
    pub selected: i64,
    pub edits: Edits,
    /// the library project open (its page and frame come back once it has opened again)
    #[serde(default)]
    pub project: String,
}
