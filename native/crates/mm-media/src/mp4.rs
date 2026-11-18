//! MP4 / QuickTime boxes: read a file's movie header (timeline facts AVFoundation reports) and write the export's
//! final file (the encoded video plus the source's audio, copied packet for packet, plus the app's metadata).
//!
//! **Why read boxes ourselves.** video_io's probe is AVFoundation's: the shortest frame duration, the nominal rate,
//! the track's time range (its edit list) and the asset's duration all come straight from the sample tables and the
//! edit list. Reading the same boxes gives the same numbers without decoding anything (GStreamer's caps carry only a
//! framerate, and none at all for some variable-rate files). Probe parity: identical on every clip in out/poc/clips
//! and test_clips/jazz.mp4 (native/tools/parity/media_probe.py, 2026-11-17).
//!
//! **Why write the audio track ourselves.** mp4mux drops the first two AAC packets (the encoder priming) and shifts
//! the rest with an empty edit: 248 of 250 packets on jazz_4k (poc/rust/gst/REPORT.md in the unpublished PoC).
//! AVFoundation's passthrough export keeps all 250 and says "skip 1619 samples" in the edit list, exactly as the source
//! does. So the exporter
//! muxes video only with GStreamer and [`finish_export`] adds the source's audio packets (bytes copied, the source's
//! sample description and roll group kept) with the edit list that starts playback at the same sample, moves the
//! movie header first (AVAssetExportSession's `shouldOptimizeForNetworkUse`), and writes the only metadata the
//! Python export writes: 3GPP `swre` (software = the app's name) and `dscp` (description = "Made with ...") in the
//! movie's user data, language "und" (what AVFoundation writes for the common Software and Description keys).
use anyhow::{anyhow, bail, Context, Result};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// One box: its type and either raw payload (leaf) or children (container).
#[derive(Clone, Debug)]
pub struct Bx {
    pub typ: [u8; 4],
    pub data: Vec<u8>,
    pub kids: Vec<Bx>,
    /// bytes before the children inside a container (e.g. `meta`'s version/flags, `stsd`'s count)
    pub container: bool,
}

const CONTAINERS: [&[u8; 4]; 9] = [b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf", b"udta", b"mvex"];

fn be16(b: &[u8]) -> u16 {
    u16::from_be_bytes([b[0], b[1]])
}
fn be32(b: &[u8]) -> u32 {
    u32::from_be_bytes(b[..4].try_into().unwrap())
}
fn be64(b: &[u8]) -> u64 {
    u64::from_be_bytes(b[..8].try_into().unwrap())
}

impl Bx {
    pub fn leaf(typ: &[u8; 4], data: Vec<u8>) -> Bx {
        Bx { typ: *typ, data, kids: vec![], container: false }
    }
    pub fn cont(typ: &[u8; 4], kids: Vec<Bx>) -> Bx {
        Bx { typ: *typ, data: vec![], kids, container: true }
    }

    /// Parse the boxes in `b`.
    pub fn parse_all(mut b: &[u8]) -> Result<Vec<Bx>> {
        let mut out = vec![];
        while b.len() >= 8 {
            let mut size = be32(b) as u64;
            let typ: [u8; 4] = b[4..8].try_into().unwrap();
            let mut h = 8usize;
            if size == 1 {
                size = be64(&b[8..]);
                h = 16;
            } else if size == 0 {
                size = b.len() as u64;
            }
            if size < h as u64 || size as usize > b.len() {
                bail!("bad box {:?}", String::from_utf8_lossy(&typ));
            }
            let body = &b[h..size as usize];
            if CONTAINERS.iter().any(|c| **c == typ) {
                out.push(Bx { typ, data: vec![], kids: Bx::parse_all(body)?, container: true });
            } else {
                out.push(Bx::leaf(&typ, body.to_vec()));
            }
            b = &b[size as usize..];
        }
        Ok(out)
    }

    pub fn write(&self, out: &mut Vec<u8>) {
        let start = out.len();
        out.extend_from_slice(&[0, 0, 0, 0]);
        out.extend_from_slice(&self.typ);
        if self.container {
            for k in &self.kids {
                k.write(out);
            }
        } else {
            out.extend_from_slice(&self.data);
        }
        let size = (out.len() - start) as u32;
        out[start..start + 4].copy_from_slice(&size.to_be_bytes());
    }

    pub fn size(&self) -> usize {
        let mut v = vec![];
        self.write(&mut v);
        v.len()
    }

    pub fn get(&self, typ: &[u8; 4]) -> Option<&Bx> {
        self.kids.iter().find(|k| &k.typ == typ)
    }
    pub fn get_mut(&mut self, typ: &[u8; 4]) -> Option<&mut Bx> {
        self.kids.iter_mut().find(|k| &k.typ == typ)
    }
    /// The first box along a path of types.
    pub fn path(&self, p: &[&[u8; 4]]) -> Option<&Bx> {
        let mut b = self;
        for t in p {
            b = b.get(t)?;
        }
        Some(b)
    }
    pub fn path_mut(&mut self, p: &[&[u8; 4]]) -> Option<&mut Bx> {
        let mut b = self;
        for t in p {
            b = b.get_mut(t)?;
        }
        Some(b)
    }
}

/// An edit: `seg` movie-timescale ticks of the media starting at media time `media_time` (-1 = empty).
#[derive(Clone, Copy, Debug)]
pub struct Edit {
    pub seg: u64,
    pub media_time: i64,
    pub rate: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Track {
    pub id: u32,
    pub handler: [u8; 4],
    /// tkhd's display size (16.16) and matrix (a b u c d v x y w)
    pub tkhd_w: f64,
    pub tkhd_h: f64,
    pub matrix: [i32; 9],
    pub edits: Vec<Edit>,
    pub media_ts: u32,
    pub media_dur: u64,
    /// (count, delta) decode-time deltas
    pub stts: Vec<(u32, u32)>,
    /// (count, offset) composition offsets
    pub ctts: Vec<(u32, i32)>,
    pub sizes: Vec<u32>,
    /// each sample's byte offset in the file
    pub offsets: Vec<u64>,
    pub stsd: Vec<u8>,
    pub sgpd: Vec<Vec<u8>>,
}

impl Track {
    pub fn n(&self) -> usize {
        self.sizes.len()
    }
    pub fn is_video(&self) -> bool {
        &self.handler == b"vide"
    }
    pub fn is_audio(&self) -> bool {
        &self.handler == b"soun"
    }
    /// Decode times of every sample (media ticks).
    pub fn dts(&self) -> Vec<i64> {
        let mut v = Vec::with_capacity(self.n());
        let mut t = 0i64;
        for &(c, d) in &self.stts {
            for _ in 0..c {
                v.push(t);
                t += d as i64;
            }
        }
        v.truncate(self.n());
        v
    }
    /// Presentation times (media ticks, before the edit list) of every sample.
    pub fn pts(&self) -> Vec<i64> {
        let mut v = self.dts();
        let mut i = 0usize;
        for &(c, o) in &self.ctts {
            for _ in 0..c {
                if i < v.len() {
                    v[i] += o as i64;
                }
                i += 1;
            }
        }
        v
    }
    pub fn deltas(&self) -> Vec<u32> {
        let mut v = vec![];
        for &(c, d) in &self.stts {
            v.extend(std::iter::repeat(d).take(c as usize));
        }
        v.truncate(self.n());
        v
    }
    /// Movie-time start of the media (the leading empty edits) and the presentation duration (movie ticks).
    pub fn time_range(&self, movie_ts: u32) -> (u64, u64) {
        if self.edits.is_empty() {
            let d = (self.media_dur as u128 * movie_ts as u128 / self.media_ts.max(1) as u128) as u64;
            return (0, d);
        }
        let lead: u64 = self.edits.iter().take_while(|e| e.media_time < 0).map(|e| e.seg).sum();
        let dur: u64 = self.edits.iter().skip_while(|e| e.media_time < 0).map(|e| e.seg).sum();
        (lead, dur)
    }
    /// Media time where presentation starts (the first non-empty edit's media time), media ticks.
    pub fn edit_media_start(&self) -> i64 {
        self.edits.iter().find(|e| e.media_time >= 0).map(|e| e.media_time).unwrap_or(0)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Movie {
    pub movie_ts: u32,
    pub movie_dur: u64,
    pub tracks: Vec<Track>,
}

impl Movie {
    pub fn video(&self) -> Option<&Track> {
        self.tracks.iter().find(|t| t.is_video())
    }
    pub fn audio(&self) -> Option<&Track> {
        self.tracks.iter().find(|t| t.is_audio())
    }
}

/// The top-level boxes of a file, with `moov` parsed and `mdat` only located (offset, size).
pub struct FileBoxes {
    pub ftyp: Option<Vec<u8>>,
    pub moov: Bx,
    /// (payload offset, payload length) of each mdat
    pub mdats: Vec<(u64, u64)>,
}

pub fn read_boxes(path: &Path) -> Result<FileBoxes> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let len = f.metadata()?.len();
    let mut pos = 0u64;
    let (mut ftyp, mut moov, mut mdats) = (None, None, vec![]);
    while pos + 8 <= len {
        f.seek(SeekFrom::Start(pos))?;
        let mut h = [0u8; 16];
        f.read_exact(&mut h[..8])?;
        let mut size = be32(&h) as u64;
        let typ: [u8; 4] = h[4..8].try_into().unwrap();
        let mut hl = 8u64;
        if size == 1 {
            f.read_exact(&mut h[8..16])?;
            size = be64(&h[8..]);
            hl = 16;
        } else if size == 0 {
            size = len - pos;
        }
        if size < hl || pos + size > len {
            break;
        }
        match &typ {
            b"ftyp" | b"moov" => {
                let mut body = vec![0u8; (size - hl) as usize];
                f.read_exact(&mut body)?;
                if &typ == b"ftyp" {
                    ftyp = Some(body);
                } else {
                    moov = Some(Bx::cont(b"moov", Bx::parse_all(&body)?));
                }
            }
            b"mdat" => mdats.push((pos + hl, size - hl)),
            _ => {}
        }
        pos += size;
    }
    Ok(FileBoxes { ftyp, moov: moov.ok_or_else(|| anyhow!("no moov box"))?, mdats })
}

fn full(b: &Bx) -> (u8, &[u8]) {
    (b.data[0], &b.data[4..])
}

fn parse_track(t: &Bx) -> Result<Track> {
    let mut tr = Track::default();
    let tkhd = t.get(b"tkhd").ok_or_else(|| anyhow!("no tkhd"))?;
    let (v, d) = full(tkhd);
    let (id_at, rest) = if v == 1 { (16, 16 + 4 + 4 + 8) } else { (8, 8 + 4 + 4 + 4) };
    tr.id = be32(&d[id_at..]);
    let m0 = rest + 8 + 2 + 2 + 2 + 2;
    for i in 0..9 {
        tr.matrix[i] = be32(&d[m0 + 4 * i..]) as i32;
    }
    tr.tkhd_w = be32(&d[m0 + 36..]) as f64 / 65536.0;
    tr.tkhd_h = be32(&d[m0 + 40..]) as f64 / 65536.0;
    if let Some(elst) = t.path(&[b"edts", b"elst"]) {
        let (v, d) = full(elst);
        let n = be32(d) as usize;
        let mut q = 4;
        for _ in 0..n {
            let (seg, mt) = if v == 1 {
                let r = (be64(&d[q..]), be64(&d[q + 8..]) as i64);
                q += 16;
                r
            } else {
                let r = (be32(&d[q..]) as u64, be32(&d[q + 4..]) as i32 as i64);
                q += 8;
                r
            };
            let rate = be32(&d[q..]) as i32 as f64 / 65536.0;
            q += 4;
            tr.edits.push(Edit { seg, media_time: mt, rate });
        }
    }
    let mdia = t.get(b"mdia").ok_or_else(|| anyhow!("no mdia"))?;
    let mdhd = mdia.get(b"mdhd").ok_or_else(|| anyhow!("no mdhd"))?;
    let (v, d) = full(mdhd);
    if v == 1 {
        tr.media_ts = be32(&d[16..]);
        tr.media_dur = be64(&d[20..]);
    } else {
        tr.media_ts = be32(&d[8..]);
        tr.media_dur = be32(&d[12..]) as u64;
    }
    if let Some(h) = mdia.get(b"hdlr") {
        tr.handler = h.data[8..12].try_into().unwrap();
    }
    let stbl = mdia.path(&[b"minf", b"stbl"]).ok_or_else(|| anyhow!("no stbl"))?;
    if let Some(b) = stbl.get(b"stts") {
        let d = &b.data[4..];
        for i in 0..be32(d) as usize {
            tr.stts.push((be32(&d[4 + 8 * i..]), be32(&d[8 + 8 * i..])));
        }
    }
    if let Some(b) = stbl.get(b"ctts") {
        let d = &b.data[4..];
        for i in 0..be32(d) as usize {
            tr.ctts.push((be32(&d[4 + 8 * i..]), be32(&d[8 + 8 * i..]) as i32));
        }
    }
    if let Some(b) = stbl.get(b"stsz") {
        let d = &b.data[4..];
        let (fixed, n) = (be32(d), be32(&d[4..]) as usize);
        tr.sizes = if fixed != 0 { vec![fixed; n] } else { (0..n).map(|i| be32(&d[8 + 4 * i..])).collect() };
    }
    let chunks: Vec<u64> = if let Some(b) = stbl.get(b"stco") {
        let d = &b.data[4..];
        (0..be32(d) as usize).map(|i| be32(&d[4 + 4 * i..]) as u64).collect()
    } else if let Some(b) = stbl.get(b"co64") {
        let d = &b.data[4..];
        (0..be32(d) as usize).map(|i| be64(&d[4 + 8 * i..])).collect()
    } else {
        vec![]
    };
    let mut stsc = vec![];
    if let Some(b) = stbl.get(b"stsc") {
        let d = &b.data[4..];
        for i in 0..be32(d) as usize {
            stsc.push((be32(&d[4 + 12 * i..]), be32(&d[8 + 12 * i..])));
        }
    }
    // sample offsets from chunks
    let mut s = 0usize;
    for (ci, &co) in chunks.iter().enumerate() {
        let c1 = ci as u32 + 1;
        let per = stsc.iter().rev().find(|e| e.0 <= c1).map(|e| e.1).unwrap_or(0);
        let mut o = co;
        for _ in 0..per {
            if s >= tr.sizes.len() {
                break;
            }
            tr.offsets.push(o);
            o += tr.sizes[s] as u64;
            s += 1;
        }
    }
    if let Some(b) = stbl.get(b"stsd") {
        tr.stsd = b.data.clone();
    }
    for k in &stbl.kids {
        if &k.typ == b"sgpd" {
            tr.sgpd.push(k.data.clone());
        }
    }
    Ok(tr)
}

pub fn parse_movie(moov: &Bx) -> Result<Movie> {
    let mvhd = moov.get(b"mvhd").ok_or_else(|| anyhow!("no mvhd"))?;
    let (v, d) = full(mvhd);
    let (movie_ts, movie_dur) = if v == 1 { (be32(&d[16..]), be64(&d[20..])) } else { (be32(&d[8..]), be32(&d[12..]) as u64) };
    let mut tracks = vec![];
    for k in &moov.kids {
        if &k.typ == b"trak" {
            tracks.push(parse_track(k)?);
        }
    }
    Ok(Movie { movie_ts, movie_dur, tracks })
}

pub fn read_movie(path: &Path) -> Result<Movie> {
    parse_movie(&read_boxes(path)?.moov)
}

/// The visual sample entry's pixel aspect ratio (`pasp`), if any.
pub fn pasp(stsd: &[u8]) -> Option<(u32, u32)> {
    let i = stsd.windows(4).position(|w| w == b"pasp")?;
    let d = &stsd[i + 4..];
    if d.len() < 8 {
        return None;
    }
    Some((be32(d), be32(&d[4..])))
}

/// The visual sample entry's codec (avc1, hvc1, ...) and coded size.
pub fn visual_entry(stsd: &[u8]) -> Option<([u8; 4], u32, u32)> {
    // stsd: version/flags(4) count(4), then entry: size(4) type(4) reserved(6) dref(2) pre_defined(16) width(2) height(2)
    if stsd.len() < 8 + 8 + 6 + 2 + 16 + 4 {
        return None;
    }
    let e = &stsd[8..];
    let typ: [u8; 4] = e[4..8].try_into().ok()?;
    Some((typ, be16(&e[8 + 6 + 2 + 16..]) as u32, be16(&e[8 + 6 + 2 + 16 + 2..]) as u32))
}

// ------------------------------------------------------------------ writing the export

fn fullbox(typ: &[u8; 4], ver: u8, flags: u32, body: &[u8]) -> Bx {
    let mut d = vec![ver, (flags >> 16) as u8, (flags >> 8) as u8, flags as u8];
    d.extend_from_slice(body);
    Bx::leaf(typ, d)
}

fn u32s(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_be_bytes()).collect()
}

/// A 3GPP user-data string box (swre, dscp, ...): version/flags, language "und", UTF-8, NUL.
fn udta_string(typ: &[u8; 4], s: &str) -> Bx {
    let mut d = vec![0x55, 0xC4];
    d.extend_from_slice(s.as_bytes());
    d.push(0);
    fullbox(typ, 0, 0, &d)
}

/// Read the user-data strings (swre, dscp) back from a file: for tests and the parity script.
pub fn udta_strings(path: &Path) -> Result<Vec<(String, String)>> {
    let fb = read_boxes(path)?;
    let mut out = vec![];
    if let Some(u) = fb.moov.get(b"udta") {
        for k in &u.kids {
            if k.data.len() >= 7 && (&k.typ == b"swre" || &k.typ == b"dscp") {
                let s = &k.data[6..];
                let s = s.strip_suffix(&[0]).unwrap_or(s);
                out.push((String::from_utf8_lossy(&k.typ).into(), String::from_utf8_lossy(s).into()));
            }
        }
    }
    Ok(out)
}

/// The audio to copy: which source samples, and the edit that starts playback where the source's presentation
/// time `start` (seconds after the audio's own presentation start) is.
struct AudioCut {
    first: usize,
    last: usize,
    media_time: i64,
}

fn cut_audio(a: &Track, start: f64, dur: f64) -> AudioCut {
    let ts = a.media_ts as f64;
    let m0 = a.edit_media_start() + (start * ts).round() as i64;
    let m1 = m0 + (dur * ts).round() as i64;
    let dts = a.dts();
    let deltas = a.deltas();
    let idx = |m: i64| dts.iter().rposition(|&d| d <= m).unwrap_or(0);
    // decoder warm-up: from the start, everything the source has (its edit starts inside packet 1 of 250 on jazz:
    // packet 0 is priming); from a later start, one packet before (AAC's roll distance -1), which the edit then
    // skips: what AVFoundation keeps (jazz from 1.0 s: 119 packets, skip 1539 samples)
    let first = if start <= 0.0 { 0 } else { idx(m0).saturating_sub(1) };
    let mut last = idx((m1 - 1).max(m0));
    if last + 1 < dts.len() && dts[last] + deltas[last] as i64 <= m1 - 1 {
        last += 1;
    }
    AudioCut { first, last: last.min(dts.len().saturating_sub(1)), media_time: m0 - dts[first] }
}

pub struct FinishInfo {
    pub audio_packets: usize,
    pub audio_skip: i64,
}

/// Make the export's final file `out` from GStreamer's video-only `video_mp4`: the source's audio for presentation
/// seconds `[start, start + video duration)` copied in (when `audio_src` has an audio track), `meta` as user-data
/// strings ([(b"swre", name), (b"dscp", description)]), and the movie header first.
pub fn finish_export(video_mp4: &Path, audio_src: Option<&Path>, start: f64, meta: &[(&[u8; 4], String)], out: &Path)
                     -> Result<FinishInfo> {
    let fb = read_boxes(video_mp4)?;
    let mut moov = fb.moov.clone();
    let movie = parse_movie(&moov)?;
    let vt = movie.video().ok_or_else(|| anyhow!("export has no video track"))?.clone();
    let movie_ts = movie.movie_ts;
    let (_, v_seg) = vt.time_range(movie_ts);
    let v_dur = v_seg as f64 / movie_ts as f64;
    // drop mp4mux's own (empty) user data, and any audio track it wrote
    moov.kids.retain(|k| &k.typ != b"udta");
    for k in moov.kids.iter_mut().filter(|k| &k.typ == b"trak") {
        k.kids.retain(|k| &k.typ != b"udta");
    }
    let mut keep = vec![];
    for k in moov.kids.drain(..) {
        if &k.typ == b"trak" {
            let t = parse_track(&k)?;
            if !t.is_video() {
                continue;
            }
        }
        keep.push(k);
    }
    moov.kids = keep;

    // the audio
    let mut audio_bytes = vec![];
    let mut info = FinishInfo { audio_packets: 0, audio_skip: 0 };
    let mut next_id = movie.tracks.iter().map(|t| t.id).max().unwrap_or(1) + 1;
    let mut audio_trak = None;
    let mut a_seg_movie = 0u64;
    if let Some(src) = audio_src {
        let sm = read_movie(src)?;
        if let Some(a) = sm.audio() {
            let cut = cut_audio(a, start, v_dur);
            let mut f = File::open(src)?;
            let mut sizes = vec![];
            for i in cut.first..=cut.last {
                f.seek(SeekFrom::Start(a.offsets[i]))?;
                let mut b = vec![0u8; a.sizes[i] as usize];
                f.read_exact(&mut b)?;
                audio_bytes.extend_from_slice(&b);
                sizes.push(a.sizes[i]);
            }
            let deltas: Vec<u32> = a.deltas()[cut.first..=cut.last].to_vec();
            let media_dur: u64 = deltas.iter().map(|&d| d as u64).sum();
            // stts run-length
            let mut stts: Vec<(u32, u32)> = vec![];
            for d in &deltas {
                match stts.last_mut() {
                    Some((c, x)) if x == d => *c += 1,
                    _ => stts.push((1, *d)),
                }
            }
            let avail = media_dur as i64 - cut.media_time;
            let seg_media = ((v_dur * a.media_ts as f64).round() as i64).min(avail).max(0) as u64;
            a_seg_movie = (seg_media as u128 * movie_ts as u128 / a.media_ts as u128) as u64;
            info = FinishInfo { audio_packets: sizes.len(), audio_skip: cut.media_time };
            let n = sizes.len() as u32;
            let mut stbl = vec![Bx::leaf(b"stsd", a.stsd.clone())];
            let mut b = (stts.len() as u32).to_be_bytes().to_vec();
            for (c, d) in &stts {
                b.extend(u32s(&[*c, *d]));
            }
            stbl.push(fullbox(b"stts", 0, 0, &b));
            stbl.push(fullbox(b"stsc", 0, 0, &u32s(&[1, 1, n, 1])));
            let mut b = u32s(&[0, n]);
            b.extend(u32s(&sizes));
            stbl.push(fullbox(b"stsz", 0, 0, &b));
            stbl.push(fullbox(b"co64", 0, 0, &[1u32.to_be_bytes().to_vec(), 0u64.to_be_bytes().to_vec()].concat()));
            for g in &a.sgpd {
                // the roll group (AAC: roll distance -1) applies to every packet
                let grouping: [u8; 4] = g[4..8].try_into().unwrap();
                stbl.push(Bx::leaf(b"sgpd", g.clone()));
                let mut sb = grouping.to_vec();
                sb.extend(u32s(&[1, n, 1]));
                stbl.push(fullbox(b"sbgp", 0, 0, &sb));
            }
            let mut tk = vec![0u8; 0];
            tk.extend(u32s(&[0, 0, next_id, 0]));
            tk.extend((a_seg_movie as u32).to_be_bytes());
            tk.extend([0u8; 8]);
            tk.extend([0, 0, 0, 1, 1, 0, 0, 0]); // layer 0, alternate group 1, volume 1.0, reserved
            tk.extend(u32s(&[0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000]));
            tk.extend(u32s(&[0, 0]));
            let tkhd = fullbox(b"tkhd", 0, 3, &tk);
            let mut el = 1u32.to_be_bytes().to_vec();
            el.extend((a_seg_movie as u32).to_be_bytes());
            el.extend((cut.media_time as i32).to_be_bytes());
            el.extend(0x0001_0000u32.to_be_bytes());
            let edts = Bx::cont(b"edts", vec![fullbox(b"elst", 0, 0, &el)]);
            let mut md = u32s(&[0, 0, a.media_ts, media_dur as u32]);
            md.extend([0x55, 0xC4, 0, 0]);
            let mdhd = fullbox(b"mdhd", 0, 0, &md);
            let mut hd = u32s(&[0]);
            hd.extend(b"soun");
            hd.extend([0u8; 12]);
            hd.extend(b"SoundHandler\0");
            let hdlr = fullbox(b"hdlr", 0, 0, &hd);
            let smhd = fullbox(b"smhd", 0, 0, &[0, 0, 0, 0]);
            let dref = fullbox(b"dref", 0, 0, &[u32s(&[1]), fullbox_bytes(b"url ", 0, 1, &[])].concat());
            let minf = Bx::cont(b"minf", vec![smhd, Bx::cont(b"dinf", vec![dref]), Bx::cont(b"stbl", stbl)]);
            let mdia = Bx::cont(b"mdia", vec![mdhd, hdlr, minf]);
            audio_trak = Some(Bx::cont(b"trak", vec![tkhd, edts, mdia]));
            next_id += 1;
        }
    }
    if let Some(t) = audio_trak {
        moov.kids.push(t);
    }
    // movie header: duration and next track id
    {
        let mv = moov.get_mut(b"mvhd").ok_or_else(|| anyhow!("no mvhd"))?;
        let dur = v_seg.max(a_seg_movie);
        let v = mv.data[0];
        if v == 1 {
            mv.data[4 + 20..4 + 28].copy_from_slice(&dur.to_be_bytes());
        } else {
            mv.data[4 + 12..4 + 16].copy_from_slice(&(dur as u32).to_be_bytes());
        }
        let l = mv.data.len();
        mv.data[l - 4..].copy_from_slice(&next_id.to_be_bytes());
    }
    if !meta.is_empty() {
        moov.kids.push(Bx::cont(b"udta", meta.iter().map(|(t, s)| udta_string(t, s)).collect()));
    }

    // layout: ftyp, moov, mdat (video bytes, then audio bytes)
    let mut src = File::open(video_mp4)?;
    let mut video_bytes = vec![];
    let mut old_ranges = vec![];
    for &(o, l) in &fb.mdats {
        old_ranges.push((o, l, video_bytes.len() as u64));
        src.seek(SeekFrom::Start(o))?;
        let mut b = vec![0u8; l as usize];
        src.read_exact(&mut b)?;
        video_bytes.extend_from_slice(&b);
    }
    let ftyp = Bx::leaf(b"ftyp", fb.ftyp.clone().unwrap_or_else(|| [b"mp42".to_vec(), u32s(&[1]), b"isommp41mp42".to_vec()].concat()));
    // chunk offsets become 64-bit (the moov size can't depend on them), then patched once its size is known
    promote_co64(&mut moov);
    let head = ftyp.size() as u64 + moov.size() as u64;
    let mdat_hdr = 16u64; // large-size mdat header, so any size fits
    let data0 = head + mdat_hdr;
    let audio_at = data0 + video_bytes.len() as u64;
    for trak in moov.kids.iter_mut().filter(|k| &k.typ == b"trak") {
        let is_video = parse_track(trak)?.is_video();
        let co = trak.path_mut(&[b"mdia", b"minf", b"stbl", b"co64"]).ok_or_else(|| anyhow!("no co64"))?;
        let n = be32(&co.data[4..]) as usize;
        for i in 0..n {
            let p = 8 + 8 * i;
            let old = be64(&co.data[p..]);
            let new = if is_video {
                let (o, _, rel) = old_ranges.iter().find(|(o, l, _)| old >= *o && old < o + l).ok_or_else(|| anyhow!("chunk outside mdat"))?;
                data0 + rel + (old - o)
            } else {
                audio_at + old
            };
            co.data[p..p + 8].copy_from_slice(&new.to_be_bytes());
        }
    }
    let mut out_b = vec![];
    ftyp.write(&mut out_b);
    moov.write(&mut out_b);
    assert_eq!(out_b.len() as u64, head);
    let mdat_len = mdat_hdr + video_bytes.len() as u64 + audio_bytes.len() as u64;
    out_b.extend(1u32.to_be_bytes());
    out_b.extend(b"mdat");
    out_b.extend(mdat_len.to_be_bytes());
    let tmp = out.with_extension("part");
    {
        let mut f = File::create(&tmp)?;
        f.write_all(&out_b)?;
        f.write_all(&video_bytes)?;
        f.write_all(&audio_bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, out)?;
    Ok(info)
}

fn fullbox_bytes(typ: &[u8; 4], ver: u8, flags: u32, body: &[u8]) -> Vec<u8> {
    let mut v = vec![];
    fullbox(typ, ver, flags, body).write(&mut v);
    v
}

fn promote_co64(moov: &mut Bx) {
    for trak in moov.kids.iter_mut().filter(|k| &k.typ == b"trak") {
        if let Some(stbl) = trak.path_mut(&[b"mdia", b"minf", b"stbl"]) {
            for k in stbl.kids.iter_mut() {
                if &k.typ == b"stco" {
                    let n = be32(&k.data[4..]) as usize;
                    let mut d = k.data[..8].to_vec();
                    for i in 0..n {
                        d.extend((be32(&k.data[8 + 4 * i..]) as u64).to_be_bytes());
                    }
                    *k = Bx::leaf(b"co64", d);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_round_trip() {
        let b = Bx::cont(b"moov", vec![Bx::leaf(b"mvhd", vec![1, 2, 3]), Bx::cont(b"udta", vec![udta_string(b"swre", "x")])]);
        let mut v = vec![];
        b.write(&mut v);
        let p = Bx::parse_all(&v).unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].kids[0].data, vec![1, 2, 3]);
        assert_eq!(&p[0].kids[1].kids[0].data[6..], b"x\0");
    }
}
