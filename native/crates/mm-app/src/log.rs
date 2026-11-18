//! Rust's log lines go to the host (its /logs ring, in order with qDebug and QML warnings) and to stderr.
use std::sync::Mutex;

pub type LogFn = Box<dyn Fn(i32, &str) + Send + Sync>;
static SINK: Mutex<Option<LogFn>> = Mutex::new(None);

pub fn set_sink(f: Option<LogFn>) {
    *SINK.lock().unwrap() = f;
}

pub fn log(level: i32, msg: &str) {
    match &*SINK.lock().unwrap() {
        Some(f) => f(level, msg),
        None => eprintln!("[mm] {msg}"),
    }
}
pub fn info(msg: &str) {
    log(1, msg)
}
pub fn warn(msg: &str) {
    log(2, msg)
}
pub fn error(msg: &str) {
    log(3, msg)
}
