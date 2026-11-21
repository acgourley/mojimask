//! A project's draw records: per frame, what the export draws and how (mm_core::cover::FrameRecord), computed as
//! review_server.render_setup + native/tools/sidecar/project_server.py's `record` do, without Python.
//!
//! What decides a frame's covers (review_server.render_setup, "shared by the export AND the live preview so the two
//! can never drift apart"):
//! - **style** in track.STYLES (emoji when unknown); **strict** (the higher-risk mode) covers whole heads: solid,
//!   1.3x the scale;
//! - **what's drawn** ([`crate::resolve`]): the scan's tracks and the edits (people, merges, splits, trims, hidden,
//!   not_face, manual covers), joined by `links_for`; whitelisted pieces (`shown_pieces`) are left out of the
//!   records;
//! - **order** far to near (track.draw_order): by the depth model's face nearness when depth is on, else by size;
//! - **emoji** per person (palettes.assignment: the palette's pick, then the per-person choices), or the portrait
//!   photo (portrait.key_for of the project's photo) for every head;
//! - **head pose** (head_pose.track_angles, from dets/pose_full.json): always for portraits, for emoji when
//!   edits.head_pose ("tilt & turn") is on;
//! - **lighting** ([`crate::lighting`]): only for pasted pictures (emoji, portrait), when edits.light.enabled; the
//!   AUTO estimate is computed once per set of track boxes and cached in the project (light_est_*.npz,
//!   film_est_*.npz: the same files Python reads and writes). Until it's ready a preview draws flat (`wait`
//!   false), the export waits (`wait` true). Saturation applies without lighting too;
//! - **depth cuts** (depth.Occlusion.plan) when a depth model is on and its maps are cached (dets/depth_<model>.mp4):
//!   the part of a cover behind something nearer is drawn as the video; `near` is the frame's map (float32,
//!   base64), sent only when a cover is cut;
//! - "grain" is left 0: the drawing measures it on the frame it draws (lighting.frame_grain).
//!
//! Records are JSON text identical to the Python helper's (`json.dumps(record(t, ...))`): checked by
//! native/tools/parity/records_parity.py.
//!
//! ## Use
//! ```ignore
//! let mut r = mm_records::Records::open(project_dir, mm_records::Options::default())?;
//! let fr: mm_core::cover::FrameRecord = r.frame_record(t, &mm_records::records::Ask::default())?;
//! r.set_edits(edits_doc, rev);   // or r.reload_edits(): records follow the edits' revision
//! ```
//! Each (edits revision, style, strict, lighting ready, wait) builds the setup once and keeps every frame's record,
//! like review_server's PREVIEW cache.
use crate::lighting::{self as LT, Lighting};
use crate::resolve::{self as R, Item, Track, M23};
use anyhow::{anyhow, Context, Result};
use mm_analysis::depth as DP;
use mm_core::pyjson::{Dict, Py};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

pub const PORTRAIT_KEY: &str = "\u{1F5BC}\u{FE0F}";
/// The orb style (developer only): its picture is drawn this much bigger than an emoji's, as the picture leaves room
/// round the head sphere for the ears (mm_core::orb::HALF): the sphere comes out ~1.15 head boxes across, about an
/// emoji face's size. Its pose is smoothed along each piece over ORB_SIGMA frames with a pull of ORB_PRIOR to
/// neutral (head_pose's track_angles uses 4 and 0.15).
pub const ORB_SCALE: f64 = 1.1;
pub const ORB_SIGMA: f64 = 3.0;
pub const ORB_PRIOR: f64 = 0.15;
pub const PORTRAIT_PREFIX: &str = "portrait:";
/// emoji size vs head box (track.add_args' --scale)
pub const SCALE: f64 = 1.25;
pub const FACE_DETS_FOR_N: [&str; 3] = ["yolov5m_face_tiled", "retinaface_r50_2x", "pose_head"];

/// How to open a project.
#[derive(Clone, Debug)]
pub struct Options {
    /// the depth model (MOJIMASK_DEPTH's names); None = off. Default: MOJIMASK_DEPTH.
    pub depth: Option<String>,
    /// the app's web/ folder (the default portrait photo, native/web/assets/portrait_mona.png)
    pub web: PathBuf,
    /// emoji size vs head box
    pub scale: f64,
    /// start the lighting estimate in the background at once (review_server does)
    pub light_now: bool,
    /// the models folder (its onnx/: RetinaFace, for a portrait photo's own pose); None: a photo without a cached
    /// pose draws as if it faced front. Default: MOJIMASK_MODELS, else the checkout's models/, else the main checkout's.
    pub models: Option<PathBuf>,
}

/// The models folder (mm_core::paths::models: MOJIMASK_MODELS, else the root's models/), when it has onnx/.
pub fn default_models() -> Option<PathBuf> {
    let m = mm_core::paths::models(&mm_core::paths::root());
    m.join("onnx").is_dir().then_some(m)
}

impl Default for Options {
    fn default() -> Self {
        Options {
            depth: DP::setting().ok().flatten().map(str::to_string),
            web: mm_core::paths::web(&mm_core::paths::root()),
            scale: SCALE,
            light_now: true,
            models: default_models(),
        }
    }
}

/// What a caller asks for (the /api/native/records query): style (None: the edits' cover_style), strict, wait for
/// the lighting estimate.
#[derive(Clone, Debug, Default)]
pub struct Ask {
    pub style: Option<String>,
    pub strict: bool,
    pub wait: bool,
}

/// The project as review_server loads it at start.
pub struct Project {
    pub work: PathBuf,
    pub n: usize,
    pub w: usize,
    pub h: usize,
    pub fps: f64,
    pub mats: Vec<M23>,
    pub auto: Vec<Track>,
    /// dets/pose_full.json's frames ([] without it)
    pub poses: Vec<Vec<Vec<f64>>>,
}

fn f3(v: &Py) -> Option<[f64; 3]> {
    let l = v.as_list()?;
    Some([l.first()?.as_f64()?, l.get(1)?.as_f64()?, l.get(2)?.as_f64()?])
}

impl Project {
    pub fn load(work: &Path) -> Result<Project> {
        let meta = mm_core::pyjson::load(&work.join("meta.json")).map_err(|e| anyhow!(e))?;
        let (shape, v) = mm_core::edits::read_npy_f64(&work.join("mats_v2.npy")).map_err(|e| anyhow!(e))?;
        if shape.len() != 3 || shape[1] != 2 || shape[2] != 3 {
            return Err(anyhow!("mats_v2.npy: shape {shape:?}"));
        }
        let mats: Vec<M23> = v.chunks_exact(6).map(|m| [[m[0], m[1], m[2]], [m[3], m[4], m[5]]]).collect();
        let p = work.join("proto_tracks.json");
        if !p.exists() {
            return Err(anyhow!("{}: no proto_tracks.json -- scanned before the current algorithm; scan the video again", work.display()));
        }
        let proto = mm_core::pyjson::load(&p).map_err(|e| anyhow!(e))?;
        let mut auto = vec![];
        for (tid, tr) in proto.get("tracks").as_dict().ok_or_else(|| anyhow!("proto_tracks.json: no tracks"))?.iter() {
            let pairs = |k: &str| -> Vec<(i64, [f64; 3])> {
                tr.get(k).as_dict().map(|d| d.iter().filter_map(|(t, c)| Some((t.trim().parse().ok()?, f3(c)?))).collect()).unwrap_or_default()
            };
            let obs_c: BTreeMap<i64, [f64; 3]> = pairs("obs_c").into_iter().collect();
            let flow = tr
                .get("flow")
                .as_dict()
                .map(|d| {
                    d.iter()
                        .filter_map(|(t, m)| {
                            let l = m.as_list()?;
                            Some((t.trim().parse().ok()?, [l.first()?.as_f64()?, l.get(1)?.as_f64()?, l.get(2)?.as_f64()?, l.get(3)?.as_f64()?]))
                        })
                        .collect()
                })
                .unwrap_or_default();
            auto.push(Track {
                id: tid.to_string(),
                obs: obs_c.keys().copied().collect(),
                obs_c,
                cover: pairs("cover"),
                behind: tr.get("behind").as_list().map(|l| l.iter().filter_map(Py::as_i64).collect()).unwrap_or_default(),
                flow,
            });
        }
        // N: the frames the detectors saw (review_server: len(LOW), the fused detections)
        let mut n = None;
        for d in FACE_DETS_FOR_N {
            let p = work.join("dets").join(format!("{d}.json"));
            if p.exists() {
                let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&p)?)?;
                n = v["frames"].as_array().map(|a| a.len());
                break;
            }
        }
        let n = n.or_else(|| meta.get("frames").as_i64().map(|x| x as usize)).unwrap_or(mats.len());
        let pp = work.join("dets").join("pose_full.json");
        let poses = if pp.exists() {
            let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&pp)?)?;
            v["frames"]
                .as_array()
                .map(|fr| {
                    fr.iter()
                        .map(|rows| rows.as_array().map(|r| r.iter().map(|row| row.as_array().map(|x| x.iter().filter_map(|y| y.as_f64()).collect()).unwrap_or_default()).collect()).unwrap_or_default())
                        .collect()
                })
                .unwrap_or_default()
        } else {
            vec![]
        };
        Ok(Project {
            work: work.to_path_buf(),
            n,
            w: meta.get("width").as_i64().unwrap_or(0) as usize,
            h: meta.get("height").as_i64().unwrap_or(0) as usize,
            fps: meta.get("fps").as_f64().unwrap_or(30.0),
            mats,
            auto,
            poses,
        })
    }

    /// review_server.load_edits: the file over the blank document (head_pose on when there's no file yet).
    pub fn read_edits(&self) -> Result<Py> {
        let mut d = Dict::new();
        for (k, v) in [("deleted", "[]"), ("trims", "{}"), ("merges", "{}"), ("manual", "{}"), ("emoji", "{}"), ("dismissed", "[]"), ("shown", "[]"), ("people", "{}")] {
            d.insert(k, Py::parse(v).unwrap());
        }
        let p = self.work.join(mm_core::edits::EDITS);
        if p.exists() {
            let f = mm_core::pyjson::load(&p).map_err(|e| anyhow!(e))?;
            if let Some(fd) = f.as_dict() {
                for (k, v) in fd.iter() {
                    d.insert(k, v.clone());
                }
            }
        } else {
            d.insert("head_pose", Py::Bool(true));
        }
        Ok(Py::Dict(d))
    }
}

// ---------------------------------------------------------------- portrait (portrait.py's keys and the photo's pose)

/// portrait.project_photo: the project's chosen photo, or None for the default.
pub fn project_photo(work: &Path, edits: &Py) -> Option<String> {
    let name = edits.get("portrait").as_str()?;
    if name.is_empty() || Path::new(name).file_name().map(|f| f.to_string_lossy() != name).unwrap_or(true) {
        return None;
    }
    let p = format!("{}/portrait/{name}", work.to_string_lossy());
    Path::new(&p).is_file().then_some(p)
}

pub fn key_for(path: Option<String>) -> String {
    match path {
        Some(p) => format!("{PORTRAIT_PREFIX}{p}"),
        None => PORTRAIT_KEY.to_string(),
    }
}

/// portrait._path: the photo file for key ch, or None (the stand-in).
pub fn photo_path(ch: &str, web: &Path) -> Option<String> {
    if let Some(p) = ch.strip_prefix(PORTRAIT_PREFIX) {
        let r = std::fs::canonicalize(p).unwrap_or_else(|_| PathBuf::from(p));
        let ok = r.extension().is_some_and(|e| e == "png") && r.parent().and_then(|d| d.file_name()).is_some_and(|d| d == "portrait");
        return ok.then(|| r.to_string_lossy().to_string());
    }
    let d = web.join("assets").join("portrait_mona.png");
    // DEFAULT = os.path.join(paths.WEB, ...): web is an absolute, normalised path in Python
    let d = std::fs::canonicalize(&d).unwrap_or(d);
    d.is_file().then(|| d.to_string_lossy().to_string())
}

/// portrait.own_pose: (roll, yaw) of the photo itself, same conventions as head_pose (yaw + = facing image right).
/// Read from its cached `<photo>.pose.json` when that is newer than the photo; else measured ([`measure_pose`]: the
/// photo's cut-out on grey, RetinaFace's best face on the CPU) and cached there, as Python does. `models`: the models
/// folder (its onnx/); None: not measured, (0, 0).
pub fn own_pose(ch: &str, web: &Path, models: Option<&Path>) -> (f64, f64) {
    let Some(src) = photo_path(ch, web) else { return (0.0, 0.0) };
    let mt = |p: &str| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    let key = (src.clone(), mt(&src));
    static POSES: std::sync::Mutex<Vec<((String, Option<std::time::SystemTime>), (f64, f64))>> = std::sync::Mutex::new(Vec::new());
    if let Some((_, v)) = POSES.lock().unwrap().iter().find(|(k, _)| *k == key) {
        return *v;
    }
    let cache = format!("{src}.pose.json");
    if let (Some(a), Some(b)) = (key.1, mt(&cache)) {
        if b >= a {
            if let Ok(v) = mm_core::pyjson::load(Path::new(&cache)) {
                if let (Some(r), Some(y)) = (v.get("roll").as_f64(), v.get("yaw").as_f64()) {
                    POSES.lock().unwrap().push((key, (r, y)));
                    return (r, y);
                }
            }
        }
    }
    let Some(models) = models else {
        eprintln!("mm-records: {cache} missing and no models folder: the photo's own pose is (0, 0)");
        return (0.0, 0.0);
    };
    let (r, y, found) = match measure_pose(Path::new(&src), models) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("mm-records: measuring {src}: {e:#}");
            (0.0, 0.0, false)
        }
    };
    // json.dump({"roll", "yaw", "found"}) to a temporary file, then os.replace
    let text = format!("{{\"roll\": {}, \"yaw\": {}, \"found\": {}}}", mm_core::pyjson::float_repr(r), mm_core::pyjson::float_repr(y),
                       if found { "true" } else { "false" });
    let tmp = format!("{cache}.tmp");
    if std::fs::write(&tmp, text).and_then(|_| std::fs::rename(&tmp, &cache)).is_err() {
        // not cached on disk: measured again next run
    }
    POSES.lock().unwrap().push((key, (r, y)));
    (r, y)
}

/// portrait.sprite(ch, 480) on grey, then RetinaFace's best face (portrait._face) -> (roll_deg, yaw, found). The
/// photo is cropped to its cut-out (alpha > 24), centred on a transparent square and resized to 480 px with a
/// Lanczos-3 filter (PIL's LANCZOS; the `image` crate's, so the pixels can differ from PIL's by a level or two),
/// composited on 128 grey and truncated to 8 bits. RetinaFace and its decoding are the scan's own (exact).
pub fn measure_pose(src: &Path, models: &Path) -> Result<(f64, f64, bool)> {
    let Some(a) = portrait_sprite(src, 480)? else { return Ok((0.0, 0.0, false)) };
    let mut bgr = Vec::with_capacity(480 * 480 * 3);
    for p in a.pixels() {
        let al = p[3] as f64 / 255.0;
        for c in [2usize, 1, 0] {
            bgr.push((p[c] as f64 * al + 128.0 * (1.0 - al)) as u8);
        }
    }
    let img = mm_vision::Mat::from_vec(480, 480, 3, bgr);
    let m = face_models(models);
    let Some((_, p)) = mm_analysis::reid::best_face(&m, &img, 0.5) else { return Ok((0.0, 0.0, false)) };
    // le, re, nose = image-left eye, image-right eye, nose (float32 points, numpy's float32 arithmetic)
    let (le, re, nose) = (p[0], p[1], p[2]);
    // (numpy 2: float32 arrays and scalars stay float32 against Python floats)
    let v = [re[0] - le[0], re[1] - le[1]];
    let d = (v[0] * v[0] + v[1] * v[1]).sqrt();
    let roll = (v[1] as f64).atan2(v[0] as f64).to_degrees();
    let mid = [(le[0] + re[0]) / 2.0, (le[1] + re[1]) / 2.0];
    let dn = [nose[0] - mid[0], nose[1] - mid[1]];
    let u = [v[0] / d, v[1] / d];
    let dot = dn[0] * u[0] + dn[1] * u[1];
    let yaw = (dot / d / 0.45f32).clamp(-1.0, 1.0) as f64;
    Ok((roll, yaw, true))
}

/// portrait.sprite(ch, size): the photo cropped to its cut-out (alpha > 24), centred on a transparent square, resized
/// (Lanczos-3), RGBA. None when the photo has no visible pixels.
pub fn portrait_sprite(src: &Path, size: u32) -> Result<Option<image::RgbaImage>> {
    let im = image::open(src).with_context(|| src.display().to_string())?.to_rgba8();
    let (w, h) = im.dimensions();
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0u32, 0u32);
    for (x, y, p) in im.enumerate_pixels() {
        if p[3] > 24 {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x + 1);
            y1 = y1.max(y + 1);
        }
    }
    if x1 <= x0 || y1 <= y0 {
        return Ok(None);
    }
    let crop = image::imageops::crop_imm(&im, x0, y0, x1 - x0, y1 - y0).to_image();
    let s = (x1 - x0).max(y1 - y0);
    let mut sq = image::RgbaImage::new(s, s);
    image::imageops::overlay(&mut sq, &crop, ((s - crop.width()) / 2) as i64, ((s - crop.height()) / 2) as i64);
    Ok(Some(image::imageops::resize(&sq, size, size, image::imageops::FilterType::Lanczos3)))
}

/// The portrait's picture for the editor's panel (review_server's GET /api/portrait), as PNG.
pub fn portrait_sprite_png(src: &Path, size: u32) -> Result<Vec<u8>> {
    let im = portrait_sprite(src, size)?.ok_or_else(|| anyhow!("{}: nothing visible", src.display()))?;
    let mut out = std::io::Cursor::new(Vec::new());
    im.write_to(&mut out, image::ImageFormat::Png)?;
    Ok(out.into_inner())
}

/// One CPU RetinaFace (portrait._retinaface), made on first use.
fn face_models(models: &Path) -> std::sync::Arc<mm_analysis::models::Models> {
    use mm_analysis::models::{Engines, Models};
    static M: std::sync::Mutex<Option<std::sync::Arc<Models>>> = std::sync::Mutex::new(None);
    let mut g = M.lock().unwrap();
    g.get_or_insert_with(|| {
        let mut m = Models::new(&models.join("onnx"), Engines::cpu(), None);
        m.pool = 1;
        std::sync::Arc::new(m)
    })
    .clone()
}

// ---------------------------------------------------------------- the setup (review_server.render_setup)

struct Light {
    st: Dict,
    /// per frame: {piece: c} (FrameLighter.boxes; a later item of the same piece wins)
    boxes: Vec<HashMap<String, [f64; 3]>>,
}

struct Setup {
    pf: Vec<Vec<Item>>,
    style: String,
    scale: f64,
    overrides: Dict,
    shown: BTreeSet<String>,
    light: Option<Light>,
    head_pose: Option<BTreeMap<String, BTreeMap<usize, (f64, f64, i32)>>>,
    /// the orb style: {piece: {t: (yaw, pitch, roll)}} in degrees (head3d::track_poses)
    orb_pose: Option<BTreeMap<String, BTreeMap<usize, (f64, f64, f64)>>>,
    portrait: Option<String>,
    saturation: f64,
    key: String,
    recs: HashMap<usize, (Py, Option<Near>)>,
}

/// A frame's depth map, when a cover is cut (Occlusion.map as float32).
#[derive(Clone)]
pub struct Near {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

/// The records of one project. See the module docs.
pub struct Records {
    occl: Option<DP::Occlusion<'static>>,
    pub proj: Project,
    pub opts: Options,
    edits: Py,
    rev: i64,
    edits_stamp: Option<(std::time::SystemTime, u64)>,
    light: Arc<OnceLock<std::result::Result<Lighting, String>>>,
    light_started: bool,
    setup: Option<Setup>,
    /// each frame's draw order and occlusion plan, by what they're made of (see [`Records::planned`]): kept across
    /// setups, so a style change or an edit elsewhere doesn't plan every frame again
    plans: std::sync::Mutex<HashMap<usize, Planned>>,
}

/// A frame's draw order (the shown pieces left out) and occlusion plan, and the hash of what they were made from.
#[derive(Clone)]
struct Planned {
    key: u64,
    items: Vec<Item>,
    near: Option<Arc<DP::Map64>>,
    occ: BTreeMap<usize, DP::Cut>,
}

fn stamp(p: &Path) -> Option<(std::time::SystemTime, u64)> {
    std::fs::metadata(p).ok().map(|m| (m.modified().unwrap_or(std::time::UNIX_EPOCH), m.len()))
}

impl Records {
    /// Open a project folder (PROJECT/proto_tracks.json, mats_v2.npy, meta.json, edits.json; frames from its video).
    pub fn open(work: &Path, opts: Options) -> Result<Records> {
        let proj = Project::load(work)?;
        let edits = proj.read_edits()?;
        let mut occl = None;
        if let Some(name) = opts.depth.as_deref() {
            let cached = ["json", "mp4"].iter().all(|e| work.join("dets").join(format!("depth_{name}.{e}")).exists());
            if cached {
                let f0 = mm_media::frames::ScanFrames::for_project(work)?.sized(1, 0).get(0);
                let d = DP::Depth::open_cache(work, name, (f0.rows, f0.cols), |p| {
                    Box::new(mm_media::grey::MovieLuma { path: p.to_path_buf(), fps: DP::MOVIE_FPS })
                })
                .map_err(|e| anyhow!(e))?;
                let obs: HashMap<String, BTreeMap<i64, [f64; 3]>> = proj.auto.iter().map(|t| (t.id.clone(), t.obs_c.clone())).collect();
                occl = Some(DP::Occlusion::shared(Arc::new(d), obs));
            } else {
                eprintln!("MOJIMASK_DEPTH={name}, but {}/dets/depth_{name}.mp4 isn't computed: covers drawn whole", work.display());
            }
        }
        let stamp0 = stamp(&work.join(mm_core::edits::EDITS));
        let mut r = Records {
            occl,
            proj,
            opts,
            edits,
            rev: 0,
            edits_stamp: stamp0,
            light: Arc::new(OnceLock::new()),
            light_started: false,
            setup: None, plans: Default::default(),
        };
        if r.opts.light_now {
            r.start_light();
        }
        Ok(r)
    }

    /// Start the lighting estimate in the background (once).
    pub fn start_light(&mut self) {
        if self.light_started {
            return;
        }
        self.light_started = true;
        let cell = self.light.clone();
        let snap: Vec<(String, BTreeMap<i64, [f64; 3]>)> = self.proj.auto.iter().map(|t| (t.id.clone(), t.cover.iter().copied().collect())).collect();
        let work = self.proj.work.clone();
        std::thread::spawn(move || {
            let r = (|| -> Result<Lighting> {
                let frames = mm_media::frames::ScanFrames::for_project(&work)?.sized(6, 512 << 20);
                let n = mm_core::pyjson::load(&work.join("meta.json")).map_err(|e| anyhow!(e))?.get("frames").as_i64().unwrap_or(0).max(0) as usize;
                Lighting::new(&snap, &frames, n, &work, None)
            })()
            .map_err(|e| e.to_string());
            if let Err(e) = &r {
                eprintln!("lighting estimate failed: {e}");
            }
            let _ = cell.set(r);
        });
    }

    /// The lighting estimate, if it's ready (Err: it failed: drawn flat).
    pub fn light_ready(&self) -> bool {
        matches!(self.light.get(), Some(Ok(_)))
    }

    fn lighting(&mut self, wait: bool) -> Option<&Lighting> {
        if wait {
            self.start_light();
            return self.light.wait().as_ref().ok();
        }
        self.light.get().and_then(|r| r.as_ref().ok())
    }

    /// The lighting estimate's JSON (/api/lighting), waiting for it.
    pub fn lighting_json(&mut self) -> Option<Py> {
        self.lighting(true).map(|l| l.to_json())
    }

    /// The edits changed (the app's own model): records follow from now on.
    pub fn set_edits(&mut self, edits: Py, rev: i64) {
        self.edits = edits;
        self.rev = rev;
        self.setup = None;
    }

    /// Read PROJECT/edits.json again if it changed on disk (another writer: the Python sidecar, mm-cli).
    pub fn reload_edits(&mut self) -> Result<bool> {
        let s = stamp(&self.proj.work.join(mm_core::edits::EDITS));
        if s == self.edits_stamp {
            return Ok(false);
        }
        self.edits_stamp = s;
        let e = self.proj.read_edits()?;
        self.set_edits(e, self.rev + 1);
        Ok(true)
    }

    pub fn edits(&self) -> &Py {
        &self.edits
    }

    pub fn n(&self) -> usize {
        self.proj.n
    }

    fn ensure(&mut self, ask: &Ask) -> Result<()> {
        let edits = self.edits.clone();
        let mut style = match &ask.style {
            Some(s) if R::STYLES.contains(&s.as_str()) => s.clone(),
            _ => {
                let s = edits.get("cover_style");
                if s.truthy() { s.py_str() } else { "emoji".into() }
            }
        };
        if ask.strict {
            style = "solid".into();
        }
        let ready = self.light_ready();
        let key = format!("{}|{}|{}|{}|{}", self.rev, style, ask.strict, ready, ask.wait);
        if self.setup.as_ref().is_some_and(|s| s.key == key) {
            return Ok(());
        }
        // render_setup(edits, {style, strict}, wait_light=wait)
        let mut rstyle = if R::STYLES.contains(&style.as_str()) { style.clone() } else { "emoji".into() };
        if ask.strict {
            rstyle = "solid".into();
        }
        let mut scale = self.opts.scale * if ask.strict { 1.3 } else { 1.0 };
        if rstyle == "orb" {
            scale *= ORB_SCALE;
        }
        let links = R::links_for(&edits);
        let pf = R::resolve(&self.proj.auto, &edits, &self.proj.mats, self.proj.n, &links, true)?;
        let mut st = LT::settings(&edits);
        if rstyle != "emoji" && rstyle != "portrait" && rstyle != "orb" {
            st.insert("enabled", Py::Bool(false));
        }
        let enabled = st.get("enabled").is_some_and(Py::truthy);
        let have = if enabled { self.lighting(ask.wait).is_some() } else { false };
        let light = if enabled && have {
            let mut full = LT::defaults();
            for (k, v) in st.iter() {
                full.insert(k, v.clone());
            }
            let boxes = pf.iter().map(|f| f.iter().map(|it| (it.piece.clone(), it.c)).collect()).collect();
            Some(Light { st: full, boxes })
        } else {
            None
        };
        let overrides = mm_core::palettes::assignment(pf.iter().flat_map(|f| f.iter().map(|x| x.tid.as_str())), &edits);
        let shown = R::shown_pieces(&edits);
        let head_pose = if rstyle == "portrait" || (rstyle == "emoji" && edits.get("head_pose").truthy()) {
            if self.proj.poses.is_empty() {
                None
            } else {
                let items: Vec<Vec<mm_analysis::head_pose::Item>> = pf
                    .iter()
                    .map(|f| f.iter().map(|x| mm_analysis::head_pose::Item { tid: x.tid.clone(), c: x.c, kind: x.kind.to_string(), piece: Some(x.piece.clone()) }).collect())
                    .collect();
                Some(mm_analysis::head_pose::track_angles(&items, &self.proj.poses, 4.0, 0.15))
            }
        } else {
            None
        };
        let orb_pose = (rstyle == "orb").then(|| {
            let items: Vec<Vec<mm_analysis::head_pose::Item>> = pf
                .iter()
                .map(|f| f.iter().map(|x| mm_analysis::head_pose::Item { tid: x.tid.clone(), c: x.c, kind: x.kind.to_string(), piece: Some(x.piece.clone()) }).collect())
                .collect();
            mm_analysis::head3d::track_poses(&items, &self.proj.poses, ORB_SIGMA, ORB_PRIOR)
        });
        let portrait = (rstyle == "portrait").then(|| key_for(project_photo(&self.proj.work, &edits)));
        let saturation = LT::film_settings(1.0, 0.0, &LT::settings(&edits)).0;
        // (the key uses readiness *after* a wait, as the helper's cache key does)
        let key = format!("{}|{}|{}|{}|{}", self.rev, style, ask.strict, self.light_ready(), ask.wait);
        self.setup = Some(Setup { pf, style: rstyle, scale, overrides, shown, light, head_pose, orb_pose, portrait, saturation, key, recs: HashMap::new() });
        Ok(())
    }

    /// Frame t's record as JSON (project_server.record), and its depth map when a cover is cut.
    pub fn record(&mut self, t: usize, ask: &Ask) -> Result<(Py, Option<Near>)> {
        self.ensure(ask)?;
        if let Some(r) = self.setup.as_ref().unwrap().recs.get(&t) {
            return Ok(r.clone());
        }
        let r = self.build(t)?;
        self.setup.as_mut().unwrap().recs.insert(t, r.clone());
        Ok(r)
    }

    /// Frame t's record as the drawing's type (the near map: [`Records::record`]).
    pub fn frame_record(&mut self, t: usize, ask: &Ask) -> Result<mm_core::cover::FrameRecord> {
        let (p, _) = self.record(t, ask)?;
        Ok(serde_json::from_str(&p.dumps())?)
    }

    /// Frames [lo, hi) as the helper's /api/native/records body (without its cache "key").
    pub fn records(&mut self, lo: usize, hi: usize, ask: &Ask) -> Result<Py> {
        let mut out = vec![];
        for t in lo..hi.min(self.proj.n) {
            out.push(self.record(t, ask)?.0);
        }
        let s = self.setup.as_ref().unwrap();
        let mut d = Dict::new();
        d.insert("rev", Py::Int(self.rev));
        d.insert("style", Py::from(s.style.as_str()));
        d.insert("n", Py::from(self.proj.n));
        d.insert("light_ready", Py::Bool(self.light_ready()));
        d.insert("records", Py::List(out));
        Ok(Py::Dict(d))
    }

    /// review_server.depth_frame, the editor's depth view for frame t: what the depth model decided for each cover
    /// drawn there (Occlusion.explain over the frame's draw order, whitelisted pieces left out), with the face zone's
    /// geometry. None when depth is off or not computed for this project.
    pub fn depth_view(&mut self, t: usize) -> Result<Option<serde_json::Value>> {
        if self.occl.is_none() {
            return Ok(None);
        }
        self.ensure(&Ask { style: None, strict: false, wait: false })?;
        let t = t.min(self.proj.n.saturating_sub(1));
        let s = self.setup.as_ref().unwrap();
        let o = self.occl.as_ref().unwrap();
        let ti = t as i64;
        let near_of = |x: &Item| o.nearness(ti, &x.piece, &x.c);
        let order = R::draw_order(&s.pf[t], Some(&near_of));
        let its: Vec<DP::Item> = order
            .into_iter()
            .filter(|x| !s.shown.contains(&x.piece))
            .map(|x| DP::Item { tid: x.tid.clone(), c: x.c, kind: x.kind.to_string(), piece: Some(x.piece.clone()) })
            .collect();
        let covers = o.explain(ti, &its, s.scale, (self.proj.h, self.proj.w));
        Ok(Some(serde_json::json!({"covers": covers, "model": o.d.name, "face_zone": DP::FACE_ZONE.0, "zone_dy": DP::FACE_DY})))
    }

    /// The depth view's picture for frame t (Occlusion.picture) as review_server sends it: a JPEG at quality 85.
    pub fn depth_picture(&self, t: usize) -> Option<Vec<u8>> {
        let o = self.occl.as_ref()?;
        let m = o.picture(t.min(self.proj.n.saturating_sub(1)));
        Some(mm_vision::jpeg::encode_bgr(&m.data, m.cols, m.rows, 85))
    }

    /// Frame t's draw order and occlusion plan. They depend only on the frame's boxes, the pieces shown, the scale
    /// and the depth (fixed for a project), never on the style: so they're kept by a hash of those, across setups.
    /// (Until 2026-11-18 every setup planned every frame again: a style change on a 4K party clip, 1491 frames,
    /// took 22-54 s in a debug build, ~70% of it planning.)
    fn planned(&self, t: usize) -> Planned {
        use std::hash::{Hash, Hasher};
        let s = self.setup.as_ref().unwrap();
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for x in &s.pf[t] {
            (&x.tid, &x.piece, x.kind, x.c.map(f64::to_bits), s.shown.contains(&x.piece)).hash(&mut h);
        }
        (s.scale.to_bits(), self.proj.h, self.proj.w, self.occl.is_some()).hash(&mut h);
        let key = h.finish();
        if let Some(p) = self.plans.lock().unwrap().get(&t).filter(|p| p.key == key) {
            return p.clone();
        }
        let occl = self.occl.as_ref();
        let ti = t as i64;
        let near_of = |x: &Item| occl.unwrap().nearness(ti, &x.piece, &x.c);
        let order = R::draw_order(&s.pf[t], if occl.is_some() { Some(&near_of) } else { None });
        let items: Vec<Item> = order.into_iter().filter(|x| !s.shown.contains(&x.piece)).collect();
        let (near, occ) = match occl {
            Some(o) => {
                let its: Vec<DP::Item> = items.iter().map(|x| DP::Item { tid: x.tid.clone(), c: x.c, kind: x.kind.to_string(), piece: Some(x.piece.clone()) }).collect();
                o.plan(ti, &its, s.scale, (self.proj.h, self.proj.w), None)
            }
            None => (None, BTreeMap::new()),
        };
        let p = Planned { key, items, near, occ };
        self.plans.lock().unwrap().insert(t, p.clone());
        p
    }

    fn build(&self, t: usize) -> Result<(Py, Option<Near>)> {
        let s = self.setup.as_ref().unwrap();
        let ti = t as i64;
        let Planned { items, near, occ, .. } = self.planned(t);
        let lighting = if s.light.is_some() { self.light.get().and_then(|r| r.as_ref().ok()) } else { None };
        let mut covers = vec![];
        let mut photos = Dict::new();
        for (i, it) in items.iter().enumerate() {
            let piece = &it.piece;
            let mut rec = Dict::new();
            rec.insert("tid", Py::from(it.tid.as_str()));
            rec.insert("piece", Py::from(piece.as_str()));
            rec.insert("c", Py::List(it.c.iter().map(|&v| Py::Float(v)).collect()));
            if s.style == "emoji" || s.style == "portrait" || s.style == "orb" {
                let ch = if s.style == "orb" {
                    // the picture turned as the head is (yaw, pitch); the shader tilts it by roll
                    let (y, p, r) = s.orb_pose.as_ref().and_then(|h| h.get(piece)).and_then(|m| m.get(&t)).copied().unwrap_or((0.0, 0.0, 0.0));
                    rec.insert("pose", Py::List(vec![Py::Float(r), Py::Float(0.0), Py::Int(0)]));
                    mm_core::orb::key(y, p)
                } else if s.style == "portrait" {
                    s.portrait.clone().filter(|p| !p.is_empty()).unwrap_or_else(|| PORTRAIT_KEY.to_string())
                } else {
                    R::emoji_for(&it.tid, &s.overrides)
                };
                rec.insert("key", Py::from(ch.as_str()));
                if let Some(hp) = s.head_pose.as_ref().and_then(|h| h.get(piece)).and_then(|m| m.get(&t)) {
                    rec.insert("pose", Py::List(vec![Py::Float(hp.0), Py::Float(hp.1), Py::Int(hp.2 as i64)]));
                }
                if s.style == "portrait" {
                    let (r, y) = own_pose(&ch, &self.opts.web, self.opts.models.as_deref());
                    rec.insert("own_pose", Py::List(vec![Py::Float(r), Py::Float(y)]));
                    if let Some(p) = photo_path(&ch, &self.opts.web) {
                        photos.insert(ch.clone(), Py::from(p.as_str()));
                    }
                }
                if let (Some(l), Some(lt)) = (&s.light, lighting) {
                    let (k, g, amb) = lt.auto(piece, ti);
                    let mut la = Dict::new();
                    la.insert("key", Py::List(k.iter().map(|&v| Py::Float(v)).collect()));
                    la.insert("grad", Py::List(g.iter().map(|&v| Py::Float(v)).collect()));
                    la.insert("amb", Py::List(amb.iter().map(|&v| Py::Float(v)).collect()));
                    rec.insert("light_auto", Py::Dict(la));
                    let v = velocity(&l.boxes, piece, t);
                    rec.insert("vel_raw", Py::List(vec![Py::Float(v.0), Py::Float(v.1)]));
                }
            }
            if let Some(oc) = occ.get(&i) {
                let mut o = Dict::new();
                o.insert("face", Py::Float(oc.face));
                o.insert("margin", Py::Float(oc.margin));
                o.insert("rim", Py::Float(oc.rim));
                o.insert("head", Py::Float(oc.head));
                rec.insert("occ", Py::Dict(o));
            }
            covers.push(Py::Dict(rec));
        }
        let mut fr = Dict::new();
        fr.insert("t", Py::from(t));
        fr.insert("style", Py::from(s.style.as_str()));
        fr.insert("scale", Py::Float(s.scale));
        fr.insert("seed", Py::Float(t as f64));
        fr.insert("soft", Py::Float(DP::OCC_SOFT));
        fr.insert("saturation", Py::Float(s.saturation));
        fr.insert("light", Py::Null);
        fr.insert("mw", Py::Int(1));
        fr.insert("mh", Py::Int(1));
        fr.insert("near", Py::Null);
        fr.insert("covers", Py::List(covers));
        fr.insert("photos", Py::Dict(photos));
        let mut nr = None;
        if let Some(m) = near {
            fr.insert("mw", Py::from(m.w));
            fr.insert("mh", Py::from(m.h));
            let data: Vec<f32> = m.d.iter().map(|&v| v as f32).collect();
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            fr.insert("near", Py::Str(base64(&bytes)));
            nr = Some(Near { w: m.w, h: m.h, data });
        }
        if let (Some(l), Some(lt)) = (&s.light, lighting) {
            let (c, b) = lt.film(ti);
            let mut d = Dict::new();
            d.insert("settings", Py::Dict(l.st.clone()));
            d.insert("film", Py::List(vec![Py::Float(c), Py::Float(b)]));
            d.insert("grain", Py::Float(0.0));
            fr.insert("light", Py::Dict(d));
        }
        Ok((Py::Dict(fr), nr))
    }
}

/// FrameLighter.velocity: half the difference of the piece's centre one frame either side (its own box where it has
/// none there).
fn velocity(boxes: &[HashMap<String, [f64; 3]>], piece: &str, t: usize) -> (f64, f64) {
    let Some(cur) = boxes.get(t).and_then(|b| b.get(piece)) else { return (0.0, 0.0) };
    let p = if t > 0 { boxes[t - 1].get(piece).unwrap_or(cur) } else { cur };
    let q = if t + 1 < boxes.len() { boxes[t + 1].get(piece).unwrap_or(cur) } else { cur };
    ((q[0] - p[0]) / 2.0, (q[1] - p[1]) / 2.0)
}

/// Standard base64 with padding (Python's base64.b64encode).
pub fn base64(b: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(b.len().div_ceil(3) * 4);
    for ch in b.chunks(3) {
        let n = (ch[0] as u32) << 16 | (*ch.get(1).unwrap_or(&0) as u32) << 8 | *ch.get(2).unwrap_or(&0) as u32;
        s.push(A[(n >> 18) as usize & 63] as char);
        s.push(A[(n >> 12) as usize & 63] as char);
        s.push(if ch.len() > 1 { A[(n >> 6) as usize & 63] as char } else { '=' });
        s.push(if ch.len() > 2 { A[n as usize & 63] as char } else { '=' });
    }
    s
}
