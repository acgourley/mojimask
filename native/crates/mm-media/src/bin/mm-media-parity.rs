//! mm-media's side of the parity tests (native/tools/parity/media_*.py). Every command prints JSON.
//!
//!   mm-media-parity probe CLIP...
//!   mm-media-parity decode CLIP OUTDIR display|detect T,T,...     (PNG per slot; also by seeking backwards)
//!   mm-media-parity photo IN OUT.png
//!   mm-media-parity export CLIP OUT PRESET FIT [START END]
//!   mm-media-parity meta-check CLIP
//!   mm-media-parity seek-all CLIP                  (every slot by seek vs forward: media_seek.py)
//!   mm-media-parity frame-at CLIP [N]             (frame_at over a random walk vs forward: forward or seek)
//!   mm-media-parity keys CLIP                      (the slots the keyframes show in)
//!   mm-media-parity seek-bench CLIP [N]            (seek + next latency at N spread-out slots)
//!   mm-media-parity trace CLIP F|sN:K ...          (frames as the decoder hands them out, after seeks; debugging)
use anyhow::{bail, Result};
use mm_media::convert::{frame_bgr, Bgr, Colour};
use serde_json::json;
use std::sync::atomic::AtomicBool;

fn save_png(path: &str, img: &Bgr) -> Result<()> {
    let rgb: Vec<u8> = img.data.chunks_exact(3).flat_map(|p| [p[2], p[1], p[0]]).collect();
    image::save_buffer(path, &rgb, img.w, img.h, image::ColorType::Rgb8)?;
    Ok(())
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let cmd = a.get(1).map(|s| s.as_str()).unwrap_or("");
    match cmd {
        "probe" => {
            let mut out = serde_json::Map::new();
            for c in &a[2..] {
                let v = match mm_media::probe(c, None) {
                    Ok(p) => {
                        let mut v = p.video_io_json();
                        v["_n"] = p.n.into();
                        v["_source"] = p.source.clone().into();
                        v
                    }
                    Err(e) => json!({"error": format!("{e:#}")}),
                };
                out.insert(c.clone(), v);
            }
            println!("{}", serde_json::Value::Object(out));
        }
        "decode" => {
            let (clip, dir, colour) = (&a[2], &a[3], if a[4] == "detect" { Colour::Detect } else { Colour::Display });
            let ts: Vec<usize> = a[5].split(',').map(|s| s.parse().unwrap()).collect();
            std::fs::create_dir_all(dir)?;
            if mm_media::photo::is_image(clip) {
                let img = mm_media::photo::read_image(clip)?;
                save_png(&format!("{dir}/00000.png"), &img)?;
                println!("{}", json!({"frames": [{"t": 0}]}));
                return Ok(());
            }
            let mut d = mm_media::Decoder::open(clip)?;
            let tags = d.colour_tags().map(|s| s.to_string());
            let mut rows = vec![];
            // decoding forward from the start (seeks: see seek-all)
            while let Some((t, f)) = d.next()? {
                if t > *ts.iter().max().unwrap() {
                    break;
                }
                if !ts.contains(&t) {
                    continue;
                }
                let img = frame_bgr(&f, &d.probe, tags.as_deref(), colour)?;
                save_png(&format!("{dir}/{t:05}.png"), &img)?;
                rows.push(json!({"t": t, "pts": f.pts, "shown_pts": f.shown_pts, "w": img.w, "h": img.h, "data_md5": md5ish(&img.data),
                                  "zero_copy": zero_copy(&f)}));
            }
            let seek_rows: Vec<serde_json::Value> = vec![];
            println!("{}", json!({"probe": d.probe, "tags": tags, "frames": rows, "after_seek": seek_rows}));
        }
        "formats" => {
            // formats.settings for every preset and fit, with and without a cut
            let p = mm_media::probe(&a[2], None)?;
            let mut out = vec![];
            for pr in mm_media::formats::PRESETS.iter() {
                for (fit, _) in mm_media::formats::FITS {
                    for (s, e) in [(None, None), (Some(1.0), Some(3.5))] {
                        out.push(mm_media::formats::settings(&p, pr.id, fit, s, e, None));
                    }
                }
            }
            println!("{}", json!(out));
        }
        "photo" => {
            let img = mm_media::photo::read_image(&a[2])?;
            save_png(&a[3], &img)?;
            println!("{}", json!({"w": img.w, "h": img.h}));
        }
        "export" => {
            let start = a.get(6).map(|s| s.parse::<f64>().unwrap());
            let end = a.get(7).map(|s| s.parse::<f64>().unwrap());
            let cancel = AtomicBool::new(false);
            let mut last = 0.0;
            let t0 = std::time::Instant::now();
            let (s, st) = mm_media::export::export_plain(&a[2], &a[3], &a[4], &a[5], start, end, &mut |p| last = p, &cancel)?;
            println!("{}", json!({"settings": s, "frames": st.as_ref().map(|s| s.frames), "audio_packets": st.as_ref().map(|s| s.audio_packets),
                                  "audio_skip": st.as_ref().map(|s| s.audio_skip), "progress": last, "seconds": t0.elapsed().as_secs_f64()}));
        }
        "export-cancel" => {
            // cancel after a few frames: the partial file must be gone
            let cancel = AtomicBool::new(false);
            let r = mm_media::export::export_plain(&a[2], &a[3], "original", "fill", None, None,
                &mut |p| if p > 0.2 { cancel.store(true, std::sync::atomic::Ordering::Relaxed) }, &cancel);
            println!("{}", json!({"cancelled": r.is_err(), "error": r.err().map(|e| e.to_string()),
                                  "out_exists": std::path::Path::new(&a[3]).exists(),
                                  "tmp_exists": std::path::Path::new(&a[3]).with_extension("video.mp4").exists()}));
        }
        "seek-list" => {
            // every frame the decoder hands out after a seek to slot T (debugging)
            let mut d = mm_media::Decoder::open(&a[2])?;
            let t: usize = a[3].parse()?;
            d.seek(t)?;
            let mut rows = vec![];
            let mut held = vec![];
            while let Some(f) = d.pull_frame()? {
                let y = mm_media::convert::Yuv::from_frame(&f)?;
                let ys: Vec<u8> = y.y.iter().map(|&v| v as u8).collect();
                rows.push(json!({"pts": f.pts, "shown": f.shown_pts, "y_md5": md5ish(&ys)}));
                held.push(f);
            }
            for (i, f) in held.iter().enumerate() {
                let y = mm_media::convert::Yuv::from_frame(f)?;
                let ys: Vec<u8> = y.y.iter().map(|&v| v as u8).collect();
                rows[i]["y_md5_later"] = md5ish(&ys).into();
                let img = frame_bgr(f, &d.probe, None, Colour::Display)?;
                rows[i]["bgr_md5"] = md5ish(&img.data).into();
            }
            println!("{}", json!(rows));
        }
        "seek-next" => {
            // frame_at(A) forward, then seek(B) + next(), printing what next gives (debugging)
            let mut d = mm_media::Decoder::open(&a[2])?;
            for spec in a[3..].iter() {
                let (op, t) = spec.split_at(1);
                let t: usize = t.parse()?;
                let f = if op == "f" { d.frame_at(t)?.map(|f| (t, f)) } else { d.seek(t)?; d.next()? };
                let (s, f) = f.unwrap();
                let img = frame_bgr(&f, &d.probe, None, Colour::Display)?;
                println!("{}", json!({"op": spec, "slot": s, "pts": f.pts, "shown": f.shown_pts, "md5": md5ish(&img.data)}));
            }
        }
        "seek-all" => {
            // every slot decoded forward vs reached by a seek: which slots differ (content or source pts)
            let mut d = mm_media::Decoder::open(&a[2])?;
            let n = d.probe.n;
            let mut fwd = vec![];
            while let Some((_, f)) = d.next()? {
                let y = mm_media::convert::Yuv::from_frame(&f)?;
                fwd.push((f.pts, md5ish(&y.y.iter().map(|&v| v as u8).collect::<Vec<_>>())));
            }
            let mut bad = vec![];
            for t in (0..n).rev() {
                d.seek(t)?;
                let (s, f) = d.next()?.unwrap();
                let y = mm_media::convert::Yuv::from_frame(&f)?;
                let m = md5ish(&y.y.iter().map(|&v| v as u8).collect::<Vec<_>>());
                if s != t || (f.pts - fwd[t].0).abs() > 1e-6 || m != fwd[t].1 {
                    bad.push(json!({"t": t, "pts": f.pts, "fwd_pts": fwd[t].0}));
                }
            }
            println!("{}", json!({"n": n, "forward": fwd.len(), "seek_mismatches": bad}));
        }
        "trace" => {
            // ops: "F" pull every frame to the end; "sN:K" seek to slot N and pull K frames (debugging)
            let mut d = mm_media::Decoder::open(&a[2])?;
            for spec in a[3..].iter() {
                let (n, k) = if spec == "F" { (None, usize::MAX) } else {
                    let (x, y) = spec[1..].split_once(':').unwrap();
                    (Some(x.parse::<usize>()?), y.parse::<usize>()?)
                };
                if let Some(n) = n { d.seek(n)?; }
                let mut i = 0;
                while i < k {
                    let Some(f) = d.pull_frame()? else { break };
                    let y = mm_media::convert::Yuv::from_frame(&f)?;
                    println!("{} {i} pts={:.4} shown={:.4} y={}", spec, f.pts, f.shown_pts, md5ish(&y.y.iter().map(|&v| v as u8).collect::<Vec<_>>()));
                    i += 1;
                }
            }
        }
        "frame-at" => {
            // frame_at over a random walk (jumps forward within and past keyframes, back, short steps) vs decoding
            // forward: which slots differ (Decoder::frame_at decodes forward or seeks)
            let mut d = mm_media::Decoder::open(&a[2])?;
            let n = d.probe.n;
            let k: usize = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(300);
            let mut fwd = vec![];
            while let Some((_, f)) = d.next()? {
                let y = mm_media::convert::Yuv::from_frame(&f)?;
                fwd.push((f.pts, md5ish(&y.y.iter().map(|&v| v as u8).collect::<Vec<_>>())));
            }
            let (mut x, mut t, mut bad, mut fwds) = (99u64, 0usize, vec![], 0);
            for _ in 0..k {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let r = (x >> 33) as usize;
                t = match r % 4 {
                    0 => t + 1 + r % 3,
                    1 => t + 13 + r % 40,
                    2 => t.saturating_sub(1 + r % 30),
                    _ => r % n,
                }
                .min(n - 1);
                fwds += d.forward_to(t) as usize;
                let f = d.frame_at(t)?.unwrap();
                let y = mm_media::convert::Yuv::from_frame(&f)?;
                let m = md5ish(&y.y.iter().map(|&v| v as u8).collect::<Vec<_>>());
                if (f.pts - fwd[t].0).abs() > 1e-6 || m != fwd[t].1 {
                    bad.push(json!({"t": t, "pts": f.pts, "fwd_pts": fwd[t].0}));
                }
            }
            println!("{}", json!({"n": n, "asked": k, "went_forward": fwds, "mismatches": bad}));
        }
        "keys" => {
            // the slots the keyframes show in (Decoder::frame_at decodes forward up to the next one)
            let d = mm_media::Decoder::open(&a[2])?;
            println!("{}", json!({"n": d.probe.n, "keys": d.keyframes()}));
        }
        "luma-chunks" => {
            // luma-chunks MOVIE [FPS]: the depth movie's Y planes read whole, then CHUNK (16) frames at a time as
            // mm_analysis::depth::Depth::full reads them; both must agree byte for byte. Prints the times.
            let path = std::path::PathBuf::from(&a[2]);
            let fps: f64 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(30.0);
            let t0 = std::time::Instant::now();
            let all = mm_media::grey::luma(&path, fps, 0, None)?;
            let whole = t0.elapsed().as_secs_f64();
            let t0 = std::time::Instant::now();
            let (mut n, mut wrong, mut slowest) = (0usize, 0usize, 0f64);
            for c in 0..all.len().div_ceil(16) {
                let t1 = std::time::Instant::now();
                let got = mm_media::grey::luma(&path, fps, c * 16, Some(16))?;
                slowest = slowest.max(t1.elapsed().as_secs_f64());
                for (i, y) in got.iter().filter(|(i, _)| *i < (c + 1) * 16) {
                    n += 1;
                    if all.iter().find(|(j, _)| j == i).map(|(_, z)| z.data != y.data).unwrap_or(true) {
                        wrong += 1;
                    }
                }
            }
            let chunks = t0.elapsed().as_secs_f64();
            println!("{}", json!({"frames": all.len(), "chunked": n, "wrong": wrong, "whole_s": whole, "chunks_s": chunks, "slowest_chunk_s": slowest}));
        }
        "frame-cost" => {
            // frame-cost CLIP [N]: what a scan frame costs from the video against from a JPEG, per frame, over the
            // first N frames (default 120): forward decode alone, the detect-colour conversion, and encoding / decoding
            // that picture as the project's quality-90 JPEG.
            let mut d = mm_media::Decoder::open(&a[2])?;
            let tags = d.colour_tags().map(|s| s.to_string());
            let n: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(120);
            let (mut dec, mut conv, mut enc, mut jdec, mut k) = (0f64, 0f64, 0f64, 0f64, 0usize);
            let mut t0 = std::time::Instant::now();
            while let Some((t, f)) = d.next()? {
                dec += t0.elapsed().as_secs_f64();
                if t >= n {
                    break;
                }
                let t1 = std::time::Instant::now();
                let img = mm_media::convert::frame_bgr(&f, &d.probe, tags.as_deref(), mm_media::convert::Colour::Detect)?;
                conv += t1.elapsed().as_secs_f64();
                if t % 10 == 0 {
                    let t1 = std::time::Instant::now();
                    let j = mm_vision::jpeg::encode_bgr(&img.data, img.w as usize, img.h as usize, 90);
                    enc += t1.elapsed().as_secs_f64();
                    let t1 = std::time::Instant::now();
                    let _ = mm_vision::jpeg::decode_bgr(&j)?;
                    jdec += t1.elapsed().as_secs_f64();
                    k += 1;
                }
                t0 = std::time::Instant::now();
            }
            let ms = |x: f64, m: usize| (x / m as f64 * 1e4).round() / 10.0;
            println!("{}", json!({"w": d.probe.width, "h": d.probe.height, "frames": n, "decode_ms": ms(dec, n), "detect_colour_ms": ms(conv, n),
                                  "jpeg_encode_ms": ms(enc, k), "jpeg_decode_ms": ms(jdec, k)}));
        }
        "seek-same" => {
            // seek-same CLIP [K]: every slot's detect-colour picture going forward, then K seeks (fixed pseudo-random
            // slots, frame_at from wherever the decoder is) must give the same bytes: the scan reads frames both ways.
            let mut d = mm_media::Decoder::open(&a[2])?;
            let tags = d.colour_tags().map(|s| s.to_string());
            let k: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(60);
            let det = |f: &mm_media::Frame, p: &mm_media::Probe| mm_media::convert::frame_bgr(f, p, tags.as_deref(), mm_media::convert::Colour::Detect).map(|b| md5ish(&b.data));
            let mut fwd = vec![];
            while let Some((_, f)) = d.next()? {
                fwd.push(det(&f, &d.probe)?);
            }
            let n = fwd.len();
            let (mut x, mut wrong, mut bad) = (777u64, 0usize, vec![]);
            for _ in 0..k {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let t = ((x >> 33) as usize) % n;
                let f = d.frame_at(t)?.ok_or_else(|| anyhow::anyhow!("no frame {t}"))?;
                if det(&f, &d.probe)? != fwd[t] {
                    wrong += 1;
                    bad.push(t);
                }
            }
            println!("{}", json!({"frames": n, "seeks": k, "wrong": wrong, "wrong_slots": bad}));
        }
        "detect-hash" => {
            // detect-hash CLIP [N]: one hash over the first N slots' detect-colour pictures (default all), and the ms
            // per frame the conversion took: to check a change to the conversion leaves its bytes alone.
            let mut d = mm_media::Decoder::open(&a[2])?;
            let tags = d.colour_tags().map(|s| s.to_string());
            let n: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
            let (mut all, mut k, mut secs) = (String::new(), 0usize, 0f64);
            while let Some((t, f)) = d.next()? {
                if t >= n {
                    break;
                }
                let t0 = std::time::Instant::now();
                let b = mm_media::convert::frame_bgr(&f, &d.probe, tags.as_deref(), mm_media::convert::Colour::Detect)?;
                secs += t0.elapsed().as_secs_f64();
                all.push_str(&md5ish(&b.data));
                k += 1;
            }
            println!("{}", json!({"frames": k, "hash": md5ish(all.as_bytes()), "convert_ms": (secs / k as f64 * 1e4).round() / 10.0}));
        }
        "frames-vs-jpeg" => {
            // frames-vs-jpeg VIDEO FRAMES_DIR: every slot from ScanFrames (8 threads, shuffled order: the pool, seeks
            // and the cache as the scan uses them) against the JPEG the old scan wrote for it (frames/NNNNN.jpg).
            // Per slot: mean absolute difference to its own JPEG and to its neighbours' (t-2..t+2); a frame from the
            // wrong time matches a neighbour better.
            let dir = std::path::PathBuf::from(&a[3]);
            let n = std::fs::read_dir(&dir)?.filter(|e| e.as_ref().is_ok_and(|e| e.file_name().to_string_lossy().ends_with(".jpg"))).count();
            let jpg = |t: usize| mm_vision::jpeg::read_bgr(&dir.join(format!("{t:05}.jpg"))).map(|d| d.data);
            let jpgs: Vec<Vec<u8>> = (0..n).map(jpg).collect::<Result<_>>()?;
            let fr = mm_media::frames::ScanFrames::open(&a[2], 8, 1 << 30);
            let mut order: Vec<usize> = (0..n).collect();
            let mut x: u64 = 99;
            for i in (1..n).rev() {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                order.swap(i, (x >> 33) as usize % (i + 1));
            }
            let got: Vec<std::sync::Mutex<Option<Vec<u8>>>> = (0..n).map(|_| std::sync::Mutex::new(None)).collect();
            let next = std::sync::atomic::AtomicUsize::new(0);
            std::thread::scope(|s| {
                for _ in 0..8 {
                    s.spawn(|| loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&t) = order.get(i) else { break };
                        *got[t].lock().unwrap() = Some(fr.get(t as i64).data.clone());
                    });
                }
            });
            let mad = |a: &[u8], b: &[u8]| if a.len() != b.len() { f64::NAN } else { a.iter().zip(b).map(|(&p, &q)| (p as i32 - q as i32).unsigned_abs() as f64).sum::<f64>() / a.len() as f64 };
            let (mut worst, mut wrong_time, mut rows) = (0f64, vec![], vec![]);
            for t in 0..n {
                let g = got[t].lock().unwrap().take().unwrap();
                let own = mad(&g, &jpgs[t]);
                let best = (t.saturating_sub(2)..(t + 3).min(n)).map(|u| (mad(&g, &jpgs[u]), u)).min_by(|a, b| a.0.total_cmp(&b.0)).unwrap();
                worst = worst.max(own);
                if best.1 != t || !own.is_finite() {
                    wrong_time.push(json!({"t": t, "own": own, "best": best.1, "best_mad": best.0}));
                }
                rows.push(own);
            }
            let mean = rows.iter().sum::<f64>() / n as f64;
            println!("{}", json!({"frames": n, "mean_mad": mean, "worst_mad": worst, "wrong_time": wrong_time, "counts": fr.counts()}));
        }
        "seek-bench" => {
            // seek latency: seek(t) + next() to slot t's frame, at N spread-out slots (a fixed pseudo-random order)
            let mut d = mm_media::Decoder::open(&a[2])?;
            let n = d.probe.n;
            let k: usize = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(100);
            let mut x: u64 = 12345;
            let mut ms = vec![];
            for _ in 0..k {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let t = ((x >> 33) as usize) % n;
                let t0 = std::time::Instant::now();
                d.seek(t)?;
                let (s, _) = d.next()?.unwrap();
                ms.push(t0.elapsed().as_secs_f64() * 1e3);
                assert_eq!(s, t);
            }
            let mut v = ms.clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
            println!("{}", json!({"n": k, "median_ms": q(0.5), "p95_ms": q(0.95), "max_ms": q(1.0), "mean_ms": ms.iter().sum::<f64>() / k as f64}));
        }
        "meta-check" => {
            let mut d = mm_media::Decoder::open(&a[2])?;
            let f = d.frame_at(0)?.unwrap();
            #[cfg(target_os = "macos")]
            let layout = mm_media::mac::cv::video_meta_ok().clone();
            #[cfg(not(target_os = "macos"))]
            let layout: Result<(), String> = Err("not a Mac".into());
            println!("{}", json!({"zero_copy": zero_copy(&f), "layout": layout.err()}));
        }
        _ => bail!("usage: mm-media-parity probe|decode|photo|export|export-cancel|meta-check ..."),
    }
    Ok(())
}

fn zero_copy(f: &mm_media::Frame) -> bool {
    #[cfg(target_os = "macos")]
    {
        mm_media::mac::cv::zero_copy_available(f.buffer.as_ref())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = f;
        false
    }
}

/// A cheap content hash (FNV-1a 64) for "same picture" checks.
fn md5ish(d: &[u8]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in d {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}
