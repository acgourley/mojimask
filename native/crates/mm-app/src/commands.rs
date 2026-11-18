//! Every command the store runs. The UI, mm-cli and the agent hooks call these by name with JSON args.
//! To add a feature: add a function here and a line in `dispatch` (and `HELP`); no C++ or ABI change.
use crate::decoder::Req;
use crate::state::Saved;
use crate::{library, log, player, Shared};
use anyhow::{anyhow, bail, Context, Result};
use mm_engine::Project;
use serde_json::{json, Value};
use std::sync::atomic::Ordering;
use std::sync::Arc;

pub const HELP: &[(&str, &str)] = &[
    ("help", "this list"),
    ("ping", "bump app.pings (a state round trip)"),
    ("check", "{dir?}: the self-test: the native engine scans the sample clip, the store exports it with covers; a JSON report ending ok"),
    ("open", "{video, data?}: open a video, with cover data (a PoC dump folder)"),
    ("seek", "{t}: show frame t (pauses)"),
    ("step", "{d}: show frame t+d (pauses)"),
    ("play", "start playback"),
    ("pause", "stop playback"),
    ("toggle_play", "play or pause"),
    ("select", "{i}: select cover i (-1: none)"),
    ("move_cover", "{i, dx, dy, t?}: move cover i of frame t (default: the shown frame) by (dx, dy) video pixels"),
    ("set_page", "{page}: projects | scan | decisions | people | editor | export | result (as go)"),
    ("edit_at", "{t, piece}: that moment in Edit with that box selected (a picture clicked in Decisions or People)"),
    ("result_done", "{to: export | decisions}: the result screen's Export or Refine (edits.result becomes done)"),
    ("result_style", "{style}: the result screen's cover style"),
    ("go", "{page, piece?}: show a page; projects leaves the open project; decisions with a piece: only its box's"),
    ("set_dev", "{on}: show the pages' developer parts or not (what someone without --dev sees); the hooks stay"),
    ("ui_text", "native/web/strings.json, icons.json and brand.json's tables, for the pages"),
    ("library_refresh", "list the projects again (the sidecar's /api/projects)"),
    ("import", "{path}: start a project from a video or photo where it is (a path or file:// URL); its scan shows"),
    ("import_sample", "start a project from the sample clip (test_clips/jazz.mp4)"),
    ("open_project", "{id}: open a project (scanned: its pages; not finished: the scan resumes)"),
    ("project_action", "{id, act: clear | clear_stale | delete | clone | relink, path?}: the ⋯ menu"),
    ("cancel_scan", "stop the scan on show (the project goes to the Trash)"),
    ("viz_events", "{since}: the scan's 'Deciding who's who' events from `since`"),
    ("viz_drained", "the page has shown every event: the project may open"),
    ("decisions_filter", "{filter?: open | user | rule | all, kind?: all | whose | same | overlap | track | blip}"),
    ("decisions_select", "{i}: select row i"),
    ("decisions_piece", "{piece}: only the decisions about one box (the Editor's R, People's Why); null: all"),
    ("decisions_nearby", "with a piece: also show (or hide) the questions where it is the grey box"),
    ("decisions_key", "{key}: a key on the Decisions page: 1 2 3 (answer by slot), 0-9 (whose face), j k Up Down"),
    ("decisions_answer", "{id, answer}: answer a decision (answer null: clear it)"),
    ("loop_frames", "{pic}: the frames the selected question's picture `pic` plays"),
    ("people_say", "{tid, kind: show | name | emoji, value}: what a People card says about someone"),
    ("people_toggle", "{what: palette | strip | brief, tid?}: open or close a card's emoji choice or faces, or the brief ones"),
    ("reveal", "{path}: show a file or folder in Finder"),
    ("undo", "undo the project's last change (its server's history)"),
    ("redo", "redo it"),
    ("reload_project", "fetch the open project's edits and decisions again"),
    ("export", "{out, preset?, fit?, start?, end?, mark?, strict?}: export the video with its covers to out (.mp4); progress in state.export"),
    ("cancel_export", "stop the export"),
    ("readback", "{path}: save the video item's next drawn picture as a PNG (public-domain clips only)"),
    ("perf", "decode / composite / latency timings"),
    ("time_of", "{t}: frame t's presentation time in the source, seconds"),
    ("save_state", "write what a restart restores"),
    ("restore", "reopen what save_state wrote"),
];

/// Every command, the Editor's included.
pub fn help() -> Value {
    HELP.iter().chain(crate::editor::HELP).chain(crate::export::PAGE_HELP).map(|(k, v)| json!({"cmd": k, "does": v})).collect()
}

pub fn dispatch(sh: &Arc<Shared>, cmd: &str, a: &Value) -> Result<Value> {
    if cmd.starts_with("editor_") {
        return crate::editor::dispatch(sh, cmd, a);
    }
    if ["export_page", "export_set", "export_run", "export_reveal"].contains(&cmd) {
        return crate::export::page_dispatch(sh, cmd, a);
    }
    let num = |k: &str| a.get(k).and_then(Value::as_f64).ok_or_else(|| anyhow!("{cmd}: missing number '{k}'"));
    let s = |k: &str| a.get(k).and_then(Value::as_str).map(String::from);
    match cmd {
        "help" => Ok(help()),
        "check" => Ok(crate::check::run(sh, s("dir").as_deref())),
        "ping" => {
            let n = {
                let mut st = sh.st.lock().unwrap();
                st.app.pings += 1;
                st.app.pings
            };
            sh.changed(&["app.pings"]);
            Ok(json!(n))
        }
        "open" => open(sh, &s("video").ok_or_else(|| anyhow!("open: missing 'video'"))?, &s("data").unwrap_or_default()),
        "seek" => {
            player::pause(sh);
            seek(sh, num("t")? as i64)
        }
        "step" => {
            player::pause(sh);
            let t = sh.st.lock().unwrap().editor.t as i64;
            seek(sh, t + num("d").unwrap_or(1.0) as i64)
        }
        "play" => player::play(sh),
        "pause" => {
            player::pause(sh);
            Ok(json!(false))
        }
        "toggle_play" => {
            if sh.st.lock().unwrap().editor.playing {
                player::pause(sh);
                Ok(json!(false))
            } else {
                player::play(sh)
            }
        }
        "select" => {
            let i = num("i")? as i64;
            sh.st.lock().unwrap().editor.selected = i;
            sh.changed(&["editor.selected"]);
            persist(sh);
            Ok(json!(i))
        }
        "move_cover" => move_cover(sh, num("i")? as usize, num("dx")? as f32, num("dy")? as f32, a.get("t").and_then(Value::as_i64)),
        "set_page" | "go" => {
            let p = s("page").ok_or_else(|| anyhow!("{cmd}: missing 'page'"))?;
            if !sh.st.lock().unwrap().app.pages.contains(&p) {
                bail!("no page '{p}' (pages: {})", sh.st.lock().unwrap().app.pages.join(", "));
            }
            if p == "decisions" {
                // (app.html showDecisions: a piece filters to its box's decisions; none shows every decision)
                library::cmd_dec_piece(sh, s("piece").or_else(|| a.get("piece").filter(|x| x.is_number()).map(|x| x.to_string())))?;
            }
            let r = library::cmd_go(sh, &p)?;
            persist(sh);
            Ok(r)
        }
        "set_dev" => {
            let on = a.get("on").and_then(Value::as_bool).unwrap_or(!sh.dev());
            sh.ui_dev.store(on, Ordering::SeqCst);
            if !on {
                // the developer filters aren't there without it
                let mut l = sh.lib.lock().unwrap();
                if l.dec_ui.filter == "rule" || l.dec_ui.filter == "all" {
                    l.dec_ui.filter = "user".into();
                }
                l.dec_ui.kind = "all".into();
                l.dec_ui.cur = 0;
            }
            {
                let mut st = sh.st.lock().unwrap();
                st.app.dev = on;
                st.library.dev = on;
            }
            sh.changed(&["app", "library"]);
            library::queue(sh, |sh| {
                let _ = library::refresh_library(sh);
                let _ = library::redraw(sh);
                crate::editor::rebuild(sh);
            });
            Ok(json!(on))
        }
        "ui_text" => Ok(json!({"strings": sh.text.strings, "icons": sh.text.icons, "brand": sh.text.brand})),
        "library_refresh" => Ok(library::cmd_refresh(sh)),
        "import" => library::cmd_import(sh, Some(local_path(&s("path").ok_or_else(|| anyhow!("import: missing 'path'"))?))),
        "import_sample" => library::cmd_import(sh, None),
        "open_project" => library::cmd_open(sh, &s("id").ok_or_else(|| anyhow!("open_project: missing 'id'"))?),
        "project_action" => library::cmd_project_action(
            sh,
            s("id").unwrap_or_default(),
            s("act").ok_or_else(|| anyhow!("project_action: missing 'act'"))?,
            s("path"),
        ),
        "cancel_scan" => library::cmd_cancel_scan(sh),
        "viz_events" => Ok(library::viz_events(sh, a.get("since").and_then(Value::as_u64).unwrap_or(0) as usize)),
        "viz_drained" => Ok(library::viz_drained(sh)),
        "decisions_filter" => Ok(library::cmd_dec_filter(sh, s("filter"), s("kind"))),
        "decisions_select" => Ok(library::cmd_dec_select(sh, num("i")? as usize)),
        "decisions_piece" => library::cmd_dec_piece(sh, s("piece").or_else(|| a.get("piece").filter(|x| x.is_number()).map(|x| x.to_string()))),
        "decisions_nearby" => Ok(library::cmd_dec_nearby(sh)),
        "edit_at" => library::cmd_edit_at(sh, num("t")? as usize, s("piece").or_else(|| a.get("piece").map(|x| x.to_string())).unwrap_or_default()),
        "result_done" => library::cmd_result_done(sh, &s("to").unwrap_or_else(|| "export".into())),
        "result_style" => library::cmd_result_style(sh, s("style").ok_or_else(|| anyhow!("result_style: missing 'style'"))?),
        "decisions_key" => library::cmd_dec_key(sh, &s("key").ok_or_else(|| anyhow!("decisions_key: missing 'key'"))?),
        "decisions_answer" => library::cmd_answer(sh, s("id").ok_or_else(|| anyhow!("decisions_answer: missing 'id'"))?, s("answer")),
        "loop_frames" => Ok(library::cmd_loop_frames(sh, a.get("pic").and_then(Value::as_u64).unwrap_or(0) as usize)),
        "people_say" => library::cmd_people_say(
            sh,
            s("tid").ok_or_else(|| anyhow!("people_say: missing 'tid'"))?,
            s("kind").ok_or_else(|| anyhow!("people_say: missing 'kind'"))?,
            a.get("value").cloned().unwrap_or(Value::Null),
        ),
        "people_toggle" => library::cmd_people_toggle(sh, &s("what").unwrap_or_default(), &s("tid").unwrap_or_default()),
        "reveal" => {
            let p = local_path(&s("path").ok_or_else(|| anyhow!("reveal: missing 'path'"))?);
            std::process::Command::new("open").arg("-R").arg(&p).spawn()?;
            Ok(json!(p))
        }
        "undo" => library::cmd_undo(sh, false),
        "redo" => library::cmd_undo(sh, true),
        "reload_project" => {
            library::queue(sh, |sh| {
                if let Err(e) = library::reload(sh) {
                    log::warn(&format!("reload: {e:#}"));
                }
            });
            Ok(json!("queued"))
        }
        "export" => {
            let o = crate::export::Opts {
                preset: s("preset").unwrap_or_else(|| "original".into()),
                fit: s("fit").unwrap_or_else(|| "fill".into()),
                start: a.get("start").and_then(Value::as_f64),
                end: a.get("end").and_then(Value::as_f64),
                mark: a.get("mark").and_then(Value::as_bool).unwrap_or(false),
                strict: a.get("strict").and_then(Value::as_bool).unwrap_or(false),
            };
            crate::export::start_with(sh, &local_path(&s("out").ok_or_else(|| anyhow!("export: missing 'out'"))?), o)
        }
        "cancel_export" => {
            sh.export.lock().unwrap().cancel();
            Ok(json!(true))
        }
        "readback" => readback(sh, &local_path(&s("path").ok_or_else(|| anyhow!("readback: missing 'path'"))?)),
        "perf" => Ok(sh.perf.lock().unwrap().summary()),
        "time_of" => {
            let p = sh.probe.lock().unwrap().clone().ok_or_else(|| anyhow!("no video open"))?.1;
            let t = num("t")? as usize;
            Ok(json!({"t": t, "seconds": p.timeline().time_of(t)}))
        }
        "save_state" => {
            persist(sh);
            Ok(json!(sh.cfg.state_file))
        }
        "restore" => restore(sh),
        _ => bail!("no command '{cmd}' (try 'help')"),
    }
}

/// A path from the UI: a plain path, or a file:// URL (what Qt's dialogs and drops give), percent-decoded.
pub fn local_path(s: &str) -> String {
    let Some(rest) = s.strip_prefix("file://") else { return s.to_string() };
    let b = rest.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&rest[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

pub fn open(sh: &Arc<Shared>, video: &str, data: &str) -> Result<Value> {
    player::pause(sh);
    let (video, data) = (local_path(video), local_path(data));
    let (video, data) = (video.as_str(), data.as_str());
    let video = std::fs::canonicalize(video).with_context(|| format!("no video at {video}"))?.to_string_lossy().into_owned();
    let data = if data.is_empty() {
        String::new()
    } else {
        std::fs::canonicalize(data).with_context(|| format!("no cover data at {data}"))?.to_string_lossy().into_owned()
    };
    let p = Project::open(&video, &data)?;
    p.warm();
    *sh.project.lock().unwrap() = Some(p);
    let gen = sh.gen.fetch_add(1, Ordering::SeqCst) + 1;
    *sh.newest.lock().unwrap() = None;
    *sh.probe.lock().unwrap() = None;
    {
        let mut st = sh.st.lock().unwrap();
        let e = &mut st.editor;
        *e = Default::default();
        e.video.path = video.clone();
        e.video.name = std::path::Path::new(&video).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        e.video.public = mm_core::privacy::is_public_clip(&sh.root(), std::path::Path::new(&video));
        e.data = data.clone();
        e.status = format!("opening {}", e.video.name);
    }
    log::info(&format!("open {video} (data: {})", if data.is_empty() { "none" } else { &data }));
    let _ = sh.dec.lock().unwrap().send(Req::Open { path: video.clone(), gen });
    sh.changed(&["editor"]);
    persist(sh);
    Ok(json!({"video": video, "gen": gen}))
}

/// Show frame t (clamped). Doesn't pause: playback calls it too.
pub fn seek(sh: &Arc<Shared>, t: i64) -> Result<Value> {
    let t = {
        let mut st = sh.st.lock().unwrap();
        // while the video is still opening its length is unknown (0): keep the frame asked for; the decoder
        // clamps it once it knows the length (on_probe), instead of every early seek landing on frame 0
        let n = st.editor.video.frames as i64;
        let t = if n > 0 { t.clamp(0, n - 1) } else { t.max(0) } as usize;
        st.editor.t = t;
        t
    };
    request(sh, t);
    sh.changed(&["editor.t"]);
    if !sh.st.lock().unwrap().editor.playing {
        persist(sh);
    }
    Ok(json!(t))
}

/// Ask the decoder for frame t (the latest request wins).
pub fn request(sh: &Arc<Shared>, t: usize) {
    let _ = sh.dec.lock().unwrap().send(Req::Show { t, gen: sh.gen(), asked: std::time::Instant::now() });
}

fn move_cover(sh: &Arc<Shared>, i: usize, dx: f32, dy: f32, t: Option<i64>) -> Result<Value> {
    player::pause(sh);
    let t = match t {
        Some(t) => t as usize,
        None => {
            let st = sh.st.lock().unwrap();
            if st.editor.shown >= 0 { st.editor.shown as usize } else { st.editor.t }
        }
    };
    let (covers, n) = {
        let mut pj = sh.project.lock().unwrap();
        let p = pj.as_mut().ok_or_else(|| anyhow!("no video open"))?;
        if !p.move_cover(t, i, dx, dy) {
            bail!("frame {t} has no cover {i}");
        }
        (p.covers(t), p.edits.log.len())
    };
    {
        let mut st = sh.st.lock().unwrap();
        st.editor.edits = n;
        if st.editor.shown == t as i64 {
            st.editor.covers = covers.clone();
        }
    }
    sh.dirty.store(true, Ordering::SeqCst);
    sh.changed(&["editor.covers", "editor.edits", "@render"]);
    persist(sh);
    Ok(json!(covers.get(i)))
}

fn readback(sh: &Arc<Shared>, path: &str) -> Result<Value> {
    {
        let mut st = sh.st.lock().unwrap();
        if !st.editor.video.public {
            st.readback.state = "refused".into();
            st.readback.path = path.into();
            drop(st);
            sh.changed(&["readback"]);
            bail!("refused: the open video isn't a public-domain clip (AGENTS.md, Privacy)");
        }
        st.readback = crate::state::Readback { state: "pending".into(), path: path.into(), frame: -1 };
    }
    *sh.readback.lock().unwrap() = Some(path.to_string());
    sh.changed(&["readback", "@render"]);
    Ok(json!("pending"))
}

/// Save what a restart restores (skipped while playing: the frame changes 24 times a second).
pub fn persist(sh: &Shared) {
    if sh.cfg.state_file.is_empty() {
        return;
    }
    let project = sh.lib.lock().unwrap().pid.clone();
    let saved = {
        let st = sh.st.lock().unwrap();
        if st.editor.video.path.is_empty() {
            return;
        }
        let edits = sh.project.lock().unwrap().as_ref().map(|p| p.edits.clone()).unwrap_or_default();
        Saved {
            page: st.app.page.clone(),
            video: st.editor.video.path.clone(),
            data: st.editor.data.clone(),
            t: st.editor.t,
            selected: st.editor.selected,
            edits,
            project: project.clone(),
        }
    };
    let p = std::path::Path::new(&sh.cfg.state_file);
    if let Some(d) = p.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let tmp = p.with_extension("json.tmp");
    if std::fs::write(&tmp, serde_json::to_vec_pretty(&saved).unwrap()).is_ok() {
        let _ = std::fs::rename(&tmp, p);
    }
}

pub fn restore(sh: &Arc<Shared>) -> Result<Value> {
    let saved: Saved = serde_json::from_slice(&std::fs::read(&sh.cfg.state_file).with_context(|| sh.cfg.state_file.clone())?)?;
    if !saved.project.is_empty() && sh.cfg.sidecar {
        // a library project: opened again once the engine is up (queued after it starts), then its page and frame
        let (id, page, t) = (saved.project.clone(), saved.page.clone(), saved.t);
        sh.lib.lock().unwrap().restore = Some((page.clone(), t));
        library::queue(sh, move |sh| {
            if let Err(e) = library::cmd_open(sh, &id) {
                log::warn(&format!("restore {id}: {e:#}"));
            }
        });
        log::info(&format!("restoring project {} on {page} at frame {t}", saved.project));
        return Ok(json!({"project": saved.project, "page": saved.page, "t": saved.t}));
    }
    open(sh, &saved.video, &saved.data)?;
    let n = {
        let mut pj = sh.project.lock().unwrap();
        let p = pj.as_mut().unwrap();
        p.edits = saved.edits.clone();
        p.edits.log.len()
    };
    {
        let mut st = sh.st.lock().unwrap();
        st.editor.t = saved.t;
        st.editor.selected = saved.selected;
        st.editor.edits = n;
        if !saved.page.is_empty() {
            st.app.page = saved.page.clone();
        }
    }
    // the decoder clamps t once the video is probed and shows it (decoder::on_probe asks for editor.t)
    log::info(&format!("restored {} at frame {} with {n} edits", saved.video, saved.t));
    sh.changed(&["app", "editor"]);
    persist(sh);
    Ok(json!({"video": saved.video, "t": saved.t, "edits": n}))
}

#[cfg(test)]
mod tests {
    #[test]
    fn file_urls() {
        assert_eq!(super::local_path("/a/b.mp4"), "/a/b.mp4");
        assert_eq!(super::local_path("file:///a/my%20clip.mp4"), "/a/my clip.mp4");
        assert_eq!(super::local_path("file:///a/%"), "/a/%");
    }
}
