//! The engine behind the app's pages, whichever it is: the routes `web/app.html` and the project pages call
//! (app_server.py's `/api/...` and, through `/p/<id>/...`, review_server.py's), as JSON in and out.
//!
//! - [`crate::sidecar::Sidecar`]: the Python engine (app_server.py and its review servers) over HTTP (`--engine python`);
//! - [`crate::native::Server`]: the same routes answered in this process by the Rust engine (`--engine native`): the
//!   library, the scan (mm-engine's scan), the edits, facts and decisions (mm_core::edits::Project, the single writer),
//!   the draw records (mm-records), the frames and pictures. No Python runs unless the scan's motion setting asks for
//!   it (`--motion python`).
//!
//! Pictures (frames, thumbnails, the scan's preview and heads) are loaded by QML from [`Backend::url`]s: both engines
//! serve them over HTTP on 127.0.0.1 (the native one from a small server of its own, [`crate::native::http`]).
use anyhow::Result;
use serde_json::Value;

pub trait Backend: Send + Sync {
    /// "python" or "native"
    fn engine(&self) -> &'static str;
    /// http://127.0.0.1:<port>: where its pictures are served
    fn base(&self) -> String;
    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base(), path)
    }
    fn get_bytes(&self, path: &str) -> Result<Vec<u8>>;
    fn get(&self, path: &str) -> Result<Value>;
    /// GET for requests that can take minutes (a project opening)
    fn get_slow(&self, path: &str) -> Result<Value> {
        self.get(path)
    }
    /// POST JSON; `slow`: requests that can take minutes
    fn post(&self, path: &str, body: &Value, slow: bool) -> Result<Value>;
    fn alive(&self) -> bool;
    /// the engine's process (this one for the native engine)
    fn pid(&self) -> u32;
    /// the Rust engine, for what it hands over directly (the draw records, without JSON)
    fn native(&self) -> Option<std::sync::Arc<crate::native::Server>> {
        None
    }
}

impl Backend for crate::sidecar::Sidecar {
    fn engine(&self) -> &'static str {
        "python"
    }
    fn base(&self) -> String {
        self.base.clone()
    }
    fn get_bytes(&self, path: &str) -> Result<Vec<u8>> {
        crate::sidecar::Sidecar::get_bytes(self, path)
    }
    fn get(&self, path: &str) -> Result<Value> {
        crate::sidecar::Sidecar::get(self, path)
    }
    fn get_slow(&self, path: &str) -> Result<Value> {
        crate::sidecar::Sidecar::get_slow(self, path)
    }
    fn post(&self, path: &str, body: &Value, slow: bool) -> Result<Value> {
        crate::sidecar::Sidecar::post(self, path, body, slow)
    }
    fn alive(&self) -> bool {
        crate::sidecar::Sidecar::alive(self)
    }
    fn pid(&self) -> u32 {
        crate::sidecar::Sidecar::pid(self)
    }
}
