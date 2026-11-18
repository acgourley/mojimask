//! The lighting estimate on one project, for native/tools/parity/records_lighting.py.
//!
//!     lighting-check PROJECT [--grain t,t,...] [--settings JSON]
//!
//! Builds (or reads) PROJECT's lighting caches from PROJECT/frames and proto_tracks.json, as review_server does
//! (lighting.Lighting({tid: cover boxes}, PROJECT/frames, PROJECT)), and prints one JSON object: the caches used,
//! to_json()'s text, auto() and film() for every piece (plus one unknown piece) on every frame (and one past each
//! end), and frame_grain on the frames asked for. Floats are written as Python's repr, so the text compares exactly.
use mm_core::pyjson::{self, Dict, Py};
use mm_records::lighting::{apply_settings, film_settings, frame_grain, settings, Frame, Lighting};
use std::collections::BTreeMap;
use std::path::PathBuf;

fn fl(v: &[f64]) -> Py {
    Py::List(v.iter().map(|&x| Py::Float(x)).collect())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let proj = PathBuf::from(args.first().expect("usage: lighting-check PROJECT [--grain t,...]"));
    let grain: Vec<usize> = match args.iter().position(|a| a == "--grain") {
        Some(i) => args[i + 1].split(',').filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).collect(),
        None => vec![],
    };
    let proto = pyjson::load(&proj.join("proto_tracks.json")).map_err(anyhow::Error::msg)?;
    let mut snap: Vec<(String, BTreeMap<i64, [f64; 3]>)> = vec![];
    for (tid, tr) in proto.get("tracks").as_dict().expect("tracks").iter() {
        let mut b = BTreeMap::new();
        for (t, c) in tr.get("cover").as_dict().expect("cover").iter() {
            let l = c.as_list().unwrap();
            b.insert(t.parse::<i64>()?, [0, 1, 2].map(|i| l[i].as_f64().unwrap()));
        }
        snap.push((tid.to_string(), b));
    }
    let t0 = std::time::Instant::now();
    let l = Lighting::new(&snap, &proj.join("frames"), &proj, None)?;
    let secs = t0.elapsed().as_secs_f64();
    let n = l.n() as i64;
    let mut out = Dict::new();
    out.insert("film_cache", Py::Str(l.film_cache.file_name().unwrap().to_string_lossy().into()));
    out.insert("light_cache", Py::Str(l.light_cache.file_name().unwrap().to_string_lossy().into()));
    out.insert("seconds", Py::Float(secs));
    out.insert("to_json", Py::Str(l.to_json().dumps()));
    let mut auto = Dict::new();
    let mut pieces: Vec<String> = snap.iter().map(|(p, _)| p.clone()).collect();
    pieces.push("no-such-piece".into());
    for p in &pieces {
        let rows = (-1..=n)
            .map(|t| {
                let (k, g, a) = l.auto(p, t);
                let v = l.velocity(p, t);
                let mut r = vec![];
                r.extend(k);
                r.extend(g);
                r.extend(a);
                r.extend([v.0, v.1]);
                fl(&r)
            })
            .collect();
        auto.insert(p.clone(), Py::List(rows));
    }
    out.insert("auto", Py::Dict(auto));
    out.insert("film", Py::List((-1..=n).map(|t| {
        let f = l.film(t);
        fl(&[f.0, f.1])
    }).collect()));
    let mut g = Dict::new();
    for t in grain {
        g.insert(t.to_string(), Py::Float(frame_grain(&Frame::read(&proj.join("frames"), t)?)));
    }
    out.insert("grain", Py::Dict(g));
    // user settings on top: --settings '[edits, ...]' -> per edits: settings(), film_settings and apply_settings
    // for the first piece on frame 10
    if let Some(i) = args.iter().position(|a| a == "--settings") {
        let cases = Py::parse(&args[i + 1]).map_err(anyhow::Error::msg)?;
        let mut rows = vec![];
        for e in cases.as_list().unwrap() {
            let st = settings(e);
            let (k, gr, am) = l.auto(&snap[0].0, 10);
            let f = l.film(10);
            let (s, b) = film_settings(f.0, f.1, &st);
            let (k2, a2, g2) = apply_settings(k, gr, am, &st);
            let mut r = vec![s, b];
            r.extend(k2);
            r.extend(a2);
            r.extend(g2);
            rows.push(Py::List(vec![Py::Str(Py::Dict(st).dumps()), fl(&r)]));
        }
        out.insert("settings", Py::List(rows));
    }
    println!("{}", Py::Dict(out).dumps());
    Ok(())
}
