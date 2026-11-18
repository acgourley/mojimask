//! Playback's clock: while playing, a thread moves editor.t on at the clip's rate (wrapping at the end) and asks the
//! decoder for each frame. Frames the decoder can't keep up with are skipped, never queued. Once a second it counts
//! the frames drawn (editor.play_fps), so the transport can say when playback is slower than the clip.
use crate::{commands, Shared};
use anyhow::Result;
use serde_json::{json, Value};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub fn play(sh: &Arc<Shared>) -> Result<Value> {
    let (t0, fps, n) = {
        let mut st = sh.st.lock().unwrap();
        let v = &st.editor.video;
        if !v.ready || v.frames == 0 {
            return Ok(json!(false));
        }
        let x = (st.editor.t, v.fps, v.frames);
        st.editor.playing = true;
        x
    };
    let pg = sh.play_gen.fetch_add(1, Ordering::SeqCst) + 1;
    sh.changed(&["editor.playing"]);
    let sh2 = sh.clone();
    std::thread::Builder::new()
        .name("mm-play".into())
        .spawn(move || {
            let start = Instant::now();
            let mut last = t0;
            let mut tick = (Instant::now(), sh2.perf.lock().unwrap().new_frames);
            while sh2.play_gen.load(Ordering::SeqCst) == pg {
                if tick.0.elapsed() >= Duration::from_secs(1) {
                    let n = sh2.perf.lock().unwrap().new_frames;
                    let rate = (n - tick.1) as f64 / tick.0.elapsed().as_secs_f64();
                    tick = (Instant::now(), n);
                    let mut st = sh2.st.lock().unwrap();
                    if sh2.play_gen.load(Ordering::SeqCst) == pg {
                        st.editor.play_fps = (rate * 10.0).round() / 10.0;
                        drop(st);
                        sh2.changed(&["editor.play_fps"]);
                    }
                }
                let t = (t0 + (start.elapsed().as_secs_f64() * fps) as usize) % n;
                if t != last {
                    last = t;
                    let _ = commands::seek(&sh2, t as i64);
                }
                std::thread::sleep(Duration::from_millis(4));
            }
        })
        .unwrap();
    Ok(json!(true))
}

pub fn pause(sh: &Arc<Shared>) {
    sh.play_gen.fetch_add(1, Ordering::SeqCst);
    let was = {
        let mut st = sh.st.lock().unwrap();
        st.editor.play_fps = 0.0;
        std::mem::replace(&mut st.editor.playing, false)
    };
    if was {
        sh.changed(&["editor.playing"]);
        commands::persist(sh);
    }
}
