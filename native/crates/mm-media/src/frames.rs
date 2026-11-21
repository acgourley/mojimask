//! The scan's frames straight from the video: [`ScanFrames`] hands out slot t's picture in the detect colour (the
//! same bytes the detectors were given, [`crate::convert::frame_bgr`] with `Colour::Detect`), decoded on demand.
//!
//! Until 2026-11-20 the scan wrote every frame to PROJECT/frames/NNNNN.jpg (quality 90, as the Python scan's
//! cv2.imwrite did) and every later step read the JPEGs back. From the video a 4K frame costs a hardware decode
//! (hidden behind the conversion) and the detect-colour conversion (6 ms on threads), against 64 ms to decode its JPEG
//! (68 ms to encode it during the scan) and 785 MB of JPEGs per 50 s of 4K; and the evidence is now measured on the
//! detectors' own pixels, not on JPEG copies of them (`mm-media-parity frame-cost`).
//!
//! - **Decoders:** a pool, one per thread asking at most. A request takes the idle decoder that reaches slot t by
//!   decoding forward ([`crate::Decoder::forward_to`]: no keyframe between, or a few slots ahead), nearest first;
//!   else a new decoder while the pool is below its size; else the least recently used idle one seeks. A seek and the
//!   forward decode give the same bytes (`mm-media-parity seek-same`: 0 wrong of 60 seeks on jazz, jazz_4k,
//!   party_clip_4k and party clip).
//! - **Cache:** [`FrameCache`], least recently used dropped first, a frame made once however many threads ask at once;
//!   the pictures and their grey copies have their own budgets ([`cache_bytes`]).
//! - **Photos:** a one-frame clip, read once (photo::read_image).
//!
//! Checked 2026-11-20 on jazz: every slot read through the pool (8 threads, shuffled order, 62 of 127 by a seek)
//! matches its own old JPEG better than its neighbours' (mean difference 1.2 of 255, worst 1.3: the JPEG's own loss;
//! `mm-media-parity frames-vs-jpeg`). With MM_FRAMES_VIA_JPEG=1 each picture goes through that JPEG and back, and the
//! decide step's outputs equal the JPEG-era scan's byte for byte (only proto_tracks' scan_version differs): what the
//! rules decide differently on the video's own pixels (jazz: 11 asked -> 14, one person split) is the JPEG's loss
//! gone, not a frame read wrong.
use crate::convert::Colour;
use anyhow::{anyhow, Result};
use mm_vision::Mat;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// Decoded pictures by frame, least recently used dropped first once they pass `budget` bytes. A frame is made once
/// however many threads ask for it at the same time: the others wait for it (the join's and cover's threads follow
/// neighbouring people through the same frames).
///
/// Until 2026-11-20 the scan's caches kept the last 257 frames in the order they were first made, and threads asking
/// for the same frame each made it: on party4k (1491 4K frames) the decide step decoded 12,700 JPEGs, 8.5 per frame;
/// this cache at 8 GB: 6,000, cover 82 s -> 54 s.
pub struct FrameCache<V> {
    budget: usize,
    st: Mutex<CacheState<V>>,
}

struct CacheState<V> {
    slots: HashMap<i64, Arc<OnceLock<Arc<V>>>>,
    /// least recently used first
    order: std::collections::VecDeque<i64>,
    sizes: HashMap<i64, usize>,
    bytes: usize,
}

impl<V> FrameCache<V> {
    pub fn new(budget: usize) -> FrameCache<V> {
        FrameCache { budget, st: Mutex::new(CacheState { slots: HashMap::new(), order: Default::default(), sizes: HashMap::new(), bytes: 0 }) }
    }

    /// Frame t's picture if it's kept (not made).
    pub fn peek(&self, t: i64) -> Option<Arc<V>> {
        self.st.lock().unwrap().slots.get(&t).and_then(|s| s.get().cloned())
    }

    /// Frame t's picture, made by `make` if it isn't kept; `size`: its bytes.
    pub fn get(&self, t: i64, size: impl Fn(&V) -> usize, make: impl FnOnce() -> V) -> Arc<V> {
        let slot = {
            let mut st = self.st.lock().unwrap();
            if let Some(p) = st.order.iter().position(|&k| k == t) {
                st.order.remove(p);
            }
            st.order.push_back(t);
            st.slots.entry(t).or_default().clone()
        };
        let v = slot.get_or_init(|| Arc::new(make())).clone();
        let mut st = self.st.lock().unwrap();
        if st.slots.get(&t).is_some_and(|s| Arc::ptr_eq(s, &slot)) && !st.sizes.contains_key(&t) {
            let n = size(&v);
            st.sizes.insert(t, n);
            st.bytes += n;
        }
        while st.bytes > self.budget && st.order.len() > 1 {
            let Some(o) = st.order.pop_front() else { break };
            st.slots.remove(&o);
            st.bytes -= st.sizes.remove(&o).unwrap_or(0);
        }
        v
    }
}

/// MM_FRAME_CACHE_MB, else a quarter of the machine's memory up to 8 GB; the pictures get three quarters, their grey
/// copies one. party4k's decide: 8 GB 6,000 decodes, cover 54 s; 16 GB 5,200, cover 49 s, 6 GB more resident.
pub fn cache_bytes() -> usize {
    if let Some(mb) = std::env::var("MM_FRAME_CACHE_MB").ok().and_then(|v| v.parse::<usize>().ok()) {
        return mb << 20;
    }
    let mut mem: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    // SAFETY: hw.memsize is a u64; len is its size
    let ok = unsafe { libc::sysctlbyname(c"hw.memsize".as_ptr(), &mut mem as *mut u64 as *mut libc::c_void, &mut len, std::ptr::null_mut(), 0) } == 0;
    if ok && mem > 0 {
        ((mem / 4) as usize).min(8 << 30)
    } else {
        4 << 30
    }
}

/// MM_DECODERS, else 8.
pub fn decoders() -> usize {
    std::env::var("MM_DECODERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8)
}

struct Idle {
    dec: crate::Decoder,
    tags: Option<String>,
    used: u64,
}

struct Pool {
    idle: Vec<Idle>,
    /// decoders open (idle or lent out)
    open: usize,
    clock: u64,
}

/// What the frames cost so far.
#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub struct Counts {
    /// pictures decoded and converted
    pub decoded: usize,
    /// of them, reached by a seek (the rest by decoding forward)
    pub seeks: usize,
    pub decoders: usize,
}

/// A clip's frames for the scan: slot t's picture in the detect colour (module doc).
pub struct ScanFrames {
    video: String,
    /// a photo's one picture, once read
    photo: OnceLock<Option<Arc<Mat<u8>>>>,
    pool: Mutex<Pool>,
    freed: Condvar,
    size: usize,
    bgr: FrameCache<Mat<u8>>,
    grey: FrameCache<Mat<u8>>,
    decoded: AtomicUsize,
    seeks: AtomicUsize,
}

impl ScanFrames {
    /// The clip at `video` (a movie or a photo), up to `decoders` decoders at once, `budget` bytes kept. Nothing is
    /// opened until a frame is asked for.
    pub fn open(video: &str, decoders: usize, budget: usize) -> ScanFrames {
        ScanFrames {
            video: video.to_string(),
            photo: OnceLock::new(),
            pool: Mutex::new(Pool { idle: vec![], open: 0, clock: 0 }),
            freed: Condvar::new(),
            size: decoders.max(1),
            bgr: FrameCache::new(budget / 4 * 3),
            grey: FrameCache::new(budget / 4),
            decoded: AtomicUsize::new(0),
            seeks: AtomicUsize::new(0),
        }
    }

    /// The project's clip (meta.json's "video"), with the default pool and budget.
    pub fn for_project(work: &std::path::Path) -> Result<ScanFrames> {
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(work.join("meta.json"))?)?;
        let video = meta["video"].as_str().ok_or_else(|| anyhow!("{}: meta.json names no video", work.display()))?;
        Ok(ScanFrames::open(video, decoders(), cache_bytes()))
    }

    /// The same clip with `decoders` decoders and `budget` bytes kept (a pass over every frame in order: one
    /// decoder, a few frames kept).
    pub fn sized(self, decoders: usize, budget: usize) -> ScanFrames {
        ScanFrames::open(&self.video, decoders, budget)
    }

    pub fn video(&self) -> &str {
        &self.video
    }

    /// Slot t's picture (BGR, upright, the detect colour).
    pub fn get(&self, t: i64) -> Arc<Mat<u8>> {
        let photo = self.photo.get_or_init(|| {
            crate::photo::is_image(&self.video).then(|| {
                let img = crate::photo::read_image(&self.video).unwrap_or_else(|e| panic!("{}: {e:#}", self.video));
                Arc::new(Mat::from_vec(img.h as usize, img.w as usize, 3, img.data))
            })
        });
        if let Some(p) = photo {
            return p.clone();
        }
        self.bgr.get(t, |m| m.data.len(), || self.decode(t).unwrap_or_else(|e| panic!("frame {t} of {}: {e:#}", self.video)))
    }

    /// [`ScanFrames::get`] with failures (a missing video, a slot past the end) as errors, for answering the app.
    pub fn try_get(&self, t: i64) -> Result<Arc<Mat<u8>>> {
        if crate::photo::is_image(&self.video) {
            if !std::path::Path::new(&self.video).exists() {
                return Err(anyhow!("the photo is missing: {}", self.video));
            }
            return Ok(self.get(t));
        }
        if let Some(m) = self.bgr.peek(t) {
            return Ok(m);
        }
        let m = self.decode(t)?;
        Ok(self.bgr.get(t, |m| m.data.len(), || m))
    }

    /// Slot t's picture in grey (cv2.cvtColor(BGR2GRAY), mm-vision, exact).
    pub fn grey(&self, t: i64) -> Arc<Mat<u8>> {
        self.grey.get(t, |m| m.data.len(), || mm_vision::bgr2gray(&self.get(t)))
    }

    pub fn counts(&self) -> Counts {
        Counts { decoded: self.decoded.load(Relaxed), seeks: self.seeks.load(Relaxed), decoders: self.pool.lock().unwrap().open }
    }

    fn decode(&self, t: i64) -> Result<Mat<u8>> {
        let t = usize::try_from(t).map_err(|_| anyhow!("frame {t}"))?;
        let mut it = self.take(t)?;
        if !it.dec.forward_to(t) {
            self.seeks.fetch_add(1, Relaxed);
        }
        let got = it.dec.frame_at(t).and_then(|f| {
            let f = f.ok_or_else(|| anyhow!("no frame {t}"))?;
            crate::convert::frame_bgr(&f, &it.dec.probe, it.tags.as_deref(), Colour::Detect)
        });
        self.decoded.fetch_add(1, Relaxed);
        let mut p = self.pool.lock().unwrap();
        p.clock += 1;
        it.used = p.clock;
        match &got {
            Ok(_) => p.idle.push(it),
            Err(_) => p.open -= 1, // a decoder that failed is dropped
        }
        drop(p);
        self.freed.notify_one();
        let img = got?;
        if std::env::var_os("MM_FRAMES_VIA_JPEG").is_some() {
            // a check, not a mode: each picture through the old scan's quality-90 JPEG and back, which must give the
            // old frames/NNNNN.jpg pixels and so the JPEG-era scan's results exactly
            let j = mm_vision::jpeg::encode_bgr(&img.data, img.w as usize, img.h as usize, 90);
            let d = mm_vision::jpeg::decode_bgr(&j)?;
            return Ok(Mat::from_vec(d.h, d.w, 3, d.data));
        }
        Ok(Mat::from_vec(img.h as usize, img.w as usize, 3, img.data))
    }

    /// A decoder for slot t (module doc): one that goes forward to it, else a new one, else the least recently used.
    fn take(&self, t: usize) -> Result<Idle> {
        let mut p = self.pool.lock().unwrap();
        loop {
            let fwd = p.idle.iter().enumerate().filter(|(_, d)| d.dec.forward_to(t)).max_by_key(|(_, d)| d.dec.position()).map(|(i, _)| i);
            if let Some(i) = fwd {
                return Ok(p.idle.swap_remove(i));
            }
            if p.open < self.size {
                p.open += 1;
                drop(p);
                crate::decode::init();
                let made = crate::Decoder::open(&self.video).map(|dec| {
                    let tags = dec.colour_tags().map(|s| s.to_string());
                    Idle { dec, tags, used: 0 }
                });
                if made.is_err() {
                    self.pool.lock().unwrap().open -= 1;
                }
                return made;
            }
            if let Some(i) = p.idle.iter().enumerate().min_by_key(|(_, d)| d.used).map(|(i, _)| i) {
                return Ok(p.idle.swap_remove(i));
            }
            p = self.freed.wait(p).unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FrameCache;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn made_once_and_least_recently_used_dropped() {
        let made = AtomicUsize::new(0);
        let c: FrameCache<Vec<u8>> = FrameCache::new(30);
        let get = |t: i64| {
            c.get(t, |v| v.len(), || {
                made.fetch_add(1, Ordering::SeqCst);
                vec![t as u8; 10]
            })
        };
        // eight threads asking for the same frame at once make it once
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| get(1));
            }
        });
        assert_eq!(made.load(Ordering::SeqCst), 1);
        get(2);
        get(3);
        get(1); // 1 is now the most recently used: 2 goes first
        get(4);
        assert_eq!(made.load(Ordering::SeqCst), 4);
        get(1);
        get(3);
        assert_eq!(made.load(Ordering::SeqCst), 4);
        get(2);
        assert_eq!(made.load(Ordering::SeqCst), 5);
    }
}
