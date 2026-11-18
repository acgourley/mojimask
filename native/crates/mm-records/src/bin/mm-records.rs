//! A project's draw records from the command line (what the Python helper's /api/native/records serves).
//!
//!     mm-records PROJECT [--from A] [--to B] [--style S] [--strict] [--wait] [--depth NAME|off] [--out FILE]
//!
//! One record per line (JSON, as Python's json.dumps writes it), frames [A, B). `--depth` defaults to MOJIMASK_DEPTH.
//! Several asks in one run (the parity script): `--batch FILE`, a JSON list of {"style", "strict", "wait"} objects,
//! each written as a block of lines after a line `# <ask>`; with `--reload` the edits are read again before each.
//! The editor's depth view instead (native/tools/parity/records_depth_view.py): `--depth-view T,T,... --out DIR`
//! writes DIR/<t>.json (review_server.depth_frame) and DIR/<t>.jpg (/api/depth/<t>.jpg) for each frame.
use mm_records::records::{Ask, Options, Records};
use std::io::Write;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut proj = None;
    let (mut lo, mut hi) = (0usize, usize::MAX);
    let mut ask = Ask::default();
    let mut opts = Options::default();
    let mut out: Option<PathBuf> = None;
    let mut batch: Option<PathBuf> = None;
    let mut depth_view: Option<Vec<usize>> = None;
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let mut val = || {
            i += 1;
            args.get(i).cloned().unwrap_or_default()
        };
        match a {
            "--from" => lo = val().parse()?,
            "--to" => hi = val().parse()?,
            "--style" => ask.style = Some(val()),
            "--strict" => ask.strict = true,
            "--wait" => ask.wait = true,
            "--depth" => {
                let v = val();
                opts.depth = if ["off", "", "0", "no", "none"].contains(&v.as_str()) { None } else { Some(v) };
            }
            "--web" => opts.web = PathBuf::from(val()),
            "--out" => out = Some(PathBuf::from(val())),
            "--batch" => batch = Some(PathBuf::from(val())),
            "--depth-view" => depth_view = Some(val().split(',').filter(|s| !s.is_empty()).map(|s| s.parse()).collect::<Result<_, _>>()?),
            _ => proj = Some(PathBuf::from(a)),
        }
        i += 1;
    }
    let proj = proj.ok_or_else(|| anyhow::anyhow!("usage: mm-records PROJECT [--from A --to B --style S --strict --wait --depth NAME --out FILE]"))?;
    let mut r = Records::open(&proj, opts)?;
    if let Some(ts) = depth_view {
        let dir = out.ok_or_else(|| anyhow::anyhow!("--depth-view needs --out DIR"))?;
        std::fs::create_dir_all(&dir)?;
        for t in ts {
            let v = r.depth_view(t)?.ok_or_else(|| anyhow::anyhow!("no depth for this project"))?;
            std::fs::write(dir.join(format!("{t}.json")), serde_json::to_vec(&v)?)?;
            std::fs::write(dir.join(format!("{t}.jpg")), r.depth_picture(t).unwrap())?;
        }
        return Ok(());
    }
    let mut w: Box<dyn Write> = match &out {
        Some(p) => Box::new(std::io::BufWriter::new(std::fs::File::create(p)?)),
        None => Box::new(std::io::BufWriter::new(std::io::stdout())),
    };
    let asks: Vec<Ask> = match &batch {
        Some(p) => {
            let v: serde_json::Value = serde_json::from_slice(&std::fs::read(p)?)?;
            v.as_array()
                .unwrap()
                .iter()
                .map(|a| Ask { style: a["style"].as_str().map(str::to_string), strict: a["strict"].as_bool().unwrap_or(false), wait: a["wait"].as_bool().unwrap_or(true) })
                .collect()
        }
        None => vec![ask],
    };
    for a in &asks {
        if batch.is_some() {
            writeln!(w, "# {:?} {} {}", a.style, a.strict, a.wait)?;
        }
        for t in lo..hi.min(r.n()) {
            let (p, _) = r.record(t, a)?;
            writeln!(w, "{}", p.dumps())?;
        }
    }
    w.flush()?;
    Ok(())
}
