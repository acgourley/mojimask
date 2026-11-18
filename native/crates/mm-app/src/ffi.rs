//! The C ABI (include/mm_app.h). Thin: JSON in and out, so new commands and state never change it.
//! Panics are caught at the boundary and returned as errors.
use crate::render::{draw, GpuState, RenderInfo};
use crate::{log, App, Config};
use serde_json::{json, Value};
use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

pub type MmNotifyFn = Option<extern "C" fn(*mut c_void, *const c_char)>;
pub type MmLogFn = Option<extern "C" fn(*mut c_void, i32, *const c_char)>;

#[derive(Clone, Copy)]
struct Ctx(usize);

fn cstr(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

fn out(s: String) -> *mut c_char {
    CString::new(s.replace('\0', "")).unwrap().into_raw()
}

fn write_err(s: &str, buf: *mut c_char, len: i32) {
    if buf.is_null() || len <= 0 {
        return;
    }
    let n = s.len().min(len as usize - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(s.as_ptr(), buf as *mut u8, n);
        *buf.add(n) = 0;
    }
}

#[no_mangle]
pub extern "C" fn mm_app_abi() -> u32 {
    1
}

#[no_mangle]
pub extern "C" fn mm_app_new(config: *const c_char, notify: MmNotifyFn, logf: MmLogFn, ctx: *mut c_void) -> *mut App {
    let c = Ctx(ctx as usize);
    if let Some(f) = logf {
        log::set_sink(Some(Box::new(move |lv, m| {
            let s = CString::new(m.replace('\0', "")).unwrap();
            f(c.0 as *mut c_void, lv, s.as_ptr());
        })));
    }
    let cfg: Config = serde_json::from_str(&cstr(config)).unwrap_or_else(|e| {
        log::error(&format!("mm_app_new: bad config ({e}); using defaults"));
        Config::default()
    });
    let n: crate::NotifyFn = match notify {
        Some(f) => Box::new(move |paths| {
            let s = CString::new(paths).unwrap();
            f(c.0 as *mut c_void, s.as_ptr());
        }),
        None => Box::new(|_| {}),
    };
    match catch_unwind(AssertUnwindSafe(|| App::new(cfg, n))) {
        Ok(a) => Box::into_raw(Box::new(a)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `app` must come from `mm_app_new` and not be used after.
#[no_mangle]
pub unsafe extern "C" fn mm_app_free(app: *mut App) {
    if !app.is_null() {
        drop(Box::from_raw(app));
    }
    log::set_sink(None);
}

/// # Safety
/// `app` must be live.
#[no_mangle]
pub unsafe extern "C" fn mm_app_state(app: *mut App, path: *const c_char) -> *mut c_char {
    let a = &*app;
    let p = cstr(path);
    match catch_unwind(AssertUnwindSafe(|| a.state(&p))) {
        Ok(v) => out(v.to_string()),
        Err(_) => out("null".into()),
    }
}

/// # Safety
/// `app` must be live.
#[no_mangle]
pub unsafe extern "C" fn mm_app_call(app: *mut App, cmd: *const c_char, args: *const c_char) -> *mut c_char {
    let a = &*app;
    let (cmd, args) = (cstr(cmd), cstr(args));
    let r = catch_unwind(AssertUnwindSafe(|| {
        let v: Value = if args.trim().is_empty() { json!({}) } else { serde_json::from_str(&args)? };
        a.call(&cmd, &v)
    }));
    let v = match r {
        Ok(Ok(v)) => json!({"ok": true, "result": v}),
        Ok(Err(e)) => json!({"ok": false, "error": format!("{e:#}")}),
        Err(_) => json!({"ok": false, "error": format!("{cmd}: panicked")}),
    };
    out(v.to_string())
}

/// # Safety
/// `s` must come from this library.
#[no_mangle]
pub unsafe extern "C" fn mm_app_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

/// # Safety
/// Host render thread; `device`, `queue`, `texture` live Metal objects (see mm_app.h).
#[no_mangle]
pub unsafe extern "C" fn mm_app_render(app: *mut App, device: *mut c_void, queue: *mut c_void, texture: *mut c_void,
                                       res: *mut RenderInfo, err: *mut c_char, errlen: i32) -> i32 {
    let a = &*app;
    let r = catch_unwind(AssertUnwindSafe(|| render_host(a, device, queue, texture)));
    match r {
        Ok(Ok(x)) => {
            if !res.is_null() {
                *res = x;
            }
            0
        }
        Ok(Err(e)) => {
            write_err(&format!("{e:#}"), err, errlen);
            -1
        }
        Err(_) => {
            write_err("render panicked", err, errlen);
            -1
        }
    }
}

#[cfg(target_os = "macos")]
unsafe fn render_host(a: &App, device: *mut c_void, queue: *mut c_void, texture: *mut c_void) -> anyhow::Result<RenderInfo> {
    let mut gl = a.sh.gpu.lock().unwrap();
    if gl.as_ref().is_some_and(|g| g.host_queue != queue as usize) {
        *gl = None;
    }
    if gl.is_none() {
        let ad = mm_render::device::metal::adopt(device, queue)?;
        log::info(&format!("render: {}", ad.report));
        *gl = Some(GpuState::new(ad.gpu, queue as usize, ad.report)?);
        // a new window (a QML reload): ask for the frame again, the decoder hands over the one it has
        let t = a.sh.st.lock().unwrap().editor.t;
        crate::commands::request(&a.sh, t);
    }
    let g = gl.as_mut().unwrap();
    let key = texture as usize;
    let tex = if g_target_is(g, key) { None } else { Some(mm_render::device::metal::wrap_texture(&g.gpu.device, texture)?) };
    draw(&a.sh, g, key, tex)
}

#[cfg(target_os = "macos")]
fn g_target_is(g: &GpuState, key: usize) -> bool {
    g.target_key() == Some(key)
}

#[cfg(not(target_os = "macos"))]
unsafe fn render_host(_: &App, _: *mut c_void, _: *mut c_void, _: *mut c_void) -> anyhow::Result<RenderInfo> {
    anyhow::bail!("mm_app_render: only Metal hosts so far")
}

/// # Safety
/// `app` must be live; call on the render thread.
#[no_mangle]
pub unsafe extern "C" fn mm_app_release_gpu(app: *mut App) {
    let a = &*app;
    *a.sh.gpu.lock().unwrap() = None;
}
