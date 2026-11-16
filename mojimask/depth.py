"""Depth: how near the camera each part of a frame is -- a cue for "who is in front". Off by default from a checkout;
the Mac app (desktop.py, desktop.APP_DEPTH) turns anything-small on.

    MOJIMASK_DEPTH=off | anything-small | anything-base | pro     (environment; the scan and refresh.py read it)

The scan's behind rule (scan/behind.py) decides a held cover is hidden behind someone from skeletons:
the box sits on another person's body, and that person is clearly nearer (shoulders wider than the face) or the face
lost its own skeleton. With a depth model on, "nearer" is measured instead: the box's middle against the same
person's face where they were detected (see Depth.near), and a box that reads much nearer than their face is
hidden even without a skeleton over it (the dancer's arm when the pose detector missed it).

Models (models/README.md has sources and hashes):
  anything-small  Depth Anything V2 Small, ONNX on onnxruntime (the Mac's GPU, 32-59 ms a 518 px frame). Apache-2.0.
  anything-base   Depth Anything V2 Base, ONNX. CC-BY-NC-4.0: non-commercial, not shippable.
  pro             Apple Depth Pro (transformers + torch on the Mac's GPU, ~1.4 s a frame). Apple's research model
                  licence: research only, not shippable. Needs experiments/requirements.txt.
Only anything-small is packaged (packaging/build_mac.sh, since 2026-11-14); the non-shippable ones never will be:
they're for trying out, in case their terms change.

Measured 2026-11-12 with anything-small: on jazz depth
agrees with the flow rule and sees the dancer over seated faces; on dark handheld party clips visible faces' own
depth wanders, so a low threshold would uncover them. Depth can't see a hand or fan held at a face (same depth).

Each model's maps are computed once per project and cached as a grey movie, PROJECT/dets/depth_<model>.mp4, at the
model's own resolution (nearness: inverse depth, larger = nearer; each frame's range in 8 bits), with
depth_<model>.json beside it: per frame, how to turn the decoded bytes back into nearness (fitted on that frame after
decoding) and the frame's nearness percentiles. 2026-11-13: an H.264 movie at 12 Mb/s is ~4x smaller than the
256 px float16 array it replaced and ~15-20x more accurate at depth edges. The scan's
readings use it shrunk to CACHE_SIDE (Depth.maps); drawing uses it whole (Depth.full).
"""
import json
import threading
import os
import sys
import time

import cv2
import numpy as np

from mojimask import track as T

from mojimask.paths import ROOT as HERE  # models/ are at the root
# Thresholds per model, in Depth.near units, set from tools/bench/depth_bench.py (2026-11-12):
#   front  a box on another skeleton's body reading this much nearer than its face does around then is behind them
#   alone  a box reading this much nearer is behind someone even with no skeleton over it
#   apart  two people reading this far apart are different people (same-person questions answered by rule)
# Depth Pro read visible faces nearer far more often (jazz: 6.5% of frames by >= 0.5 vs 0.6% for Small; 3 of 20
# labelled visible held frames at 0.5, none at 0.75) and put two of your same-person pairs >= 0.15 apart (none
# >= 0.5), so its thresholds are higher.
MODELS = {
    "anything-small": {"kind": "onnx", "path": "models/depth-anything-v2-small/model.onnx", "licence": "Apache-2.0",
                       "front": 0.25, "alone": 0.5, "apart": 0.15},
    "anything-base": {"kind": "onnx", "path": "models/depth-anything-v2-base/model.onnx", "licence": "CC-BY-NC-4.0",
                      "front": 0.25, "alone": 0.5, "apart": 0.15},
    "pro": {"kind": "depthpro", "path": "models/depth-pro", "licence": "Apple ML Research Model licence (research only)",
            "front": 0.75, "alone": 1.0, "apart": 0.5},  # 2026-11-13, per-frame face depth: 0.5 uncovered 2/20 labelled
}
SIDE = 518  # Depth Anything's input, longest side (a multiple of 14)
CACHE_SIDE = 256  # Depth.maps (the readings: near, gap, face depth) are this size; drawing uses the full map
MOVIE_BITRATE, MOVIE_KEYFRAMES, MOVIE_FPS = 12_000_000, 15, 30.0  # the depth movie: frame i at i / MOVIE_FPS s
CHUNK = 16  # full-size frames decoded at a time (Depth.full)


def setting():
    """The model MOJIMASK_DEPTH names, or None (off). ValueError for a name it doesn't know."""
    v = (os.environ.get("MOJIMASK_DEPTH") or "off").strip().lower()
    if v in ("", "off", "0", "no", "none"):
        return None
    if v not in MODELS:
        raise ValueError(f"MOJIMASK_DEPTH={v!r}: one of off, {', '.join(MODELS)}")
    return v


class _Onnx:
    """Depth Anything (ONNX) on the Mac's GPU through ONNX Runtime's CoreML provider, as onnx_core runs the face
    detectors (CPU elsewhere). The input size is pinned per frame shape -- with it free, CoreML silently fell back to
    the CPU -- and compiled once per shape (cached by onnx_core). Same output as the CPU; 32-59 ms a frame vs 205-290
    (2026-11-13). Apple's own Core ML build was a little faster (~23 ms) but takes a fixed 518 x 392 input: stretching
    frames into it cost accuracy (3 wrong "different people" at 0.15, visible faces reading nearer x6 on party_clip_c)."""

    def __init__(self, path):
        self.path, self.sessions = path, {}

    def __call__(self, bgr):
        H, W = bgr.shape[:2]
        k = SIDE / max(H, W)
        w, h = max(14, int(round(W * k / 14)) * 14), max(14, int(round(H * k / 14)) * 14)
        if (h, w) not in self.sessions:
            from mojimask import onnx_core
            self.sessions[(h, w)] = onnx_core.session(self.path, "gpu", dims={"batch_size": 1, "height": h, "width": w})
        s = self.sessions[(h, w)]
        x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        out = s.run(None, {s.get_inputs()[0].name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0]
        return out.reshape(out.shape[-2:])  # relative inverse depth: larger = nearer


class _DepthPro:
    def __init__(self, path):
        import torch
        from transformers import DepthProForDepthEstimation, DepthProImageProcessor
        self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.dtype = torch.float16 if self.dev == "mps" else torch.float32
        self.proc = DepthProImageProcessor.from_pretrained(path)
        self.model = DepthProForDepthEstimation.from_pretrained(path, torch_dtype=self.dtype).to(self.dev).eval()

    def __call__(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = self.proc(images=rgb, return_tensors="pt")["pixel_values"].to(self.dev, self.dtype)
        with self.torch.no_grad():
            out = self.model(pixel_values=x)
        d = self.proc.post_process_depth_estimation(out, target_sizes=[rgb.shape[:2]])[0]["predicted_depth"]
        return 1.0 / np.maximum(d.float().cpu().numpy(), 1e-3)  # metres -> nearness


def _load(name):
    m = MODELS[name]
    path = os.path.join(HERE, m["path"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"MOJIMASK_DEPTH={name}: {m['path']} is missing (models/README.md says where it comes from)")
    return _Onnx(path) if m["kind"] == "onnx" else _DepthPro(path)


def cache_paths(proj, name):
    return os.path.join(proj, "dets", f"depth_{name}.mp4"), os.path.join(proj, "dets", f"depth_{name}.json")


def cached(proj, name):
    return all(os.path.exists(p) for p in cache_paths(proj, name))


def compute(proj, name, progress=None):
    """Nearness for every frame of the project (PROJECT/frames), cached as a grey movie + sidecar (module docstring).
    progress(fraction) while computing."""
    from mojimask import video_io
    movie, side = cache_paths(proj, name)
    if cached(proj, name):
        return
    frames = sorted(f for f in os.listdir(os.path.join(proj, "frames")) if f.endswith(".jpg"))
    if not frames:
        raise RuntimeError(f"depth {name}: no frames in {proj}/frames")
    run, t0, w, info, samples = _load(name), time.time(), None, [], []
    tmp = movie + ".tmp.mp4"
    try:
        for i, f in enumerate(frames):
            n = run(cv2.imread(os.path.join(proj, "frames", f))).astype(np.float32)
            h, wd = n.shape
            if w is None:
                H2, W2 = h + h % 2, wd + wd % 2  # H.264 wants even sizes: the last row/column repeated, cropped on read
                w = video_io.writer(tmp, W2, H2, MOVIE_FPS, MOVIE_BITRATE, keyframes=MOVIE_KEYFRAMES)
                shape = (h, wd)
            lo, hi = float(n.min()), float(n.max())
            q = np.round((n - lo) / max(hi - lo, 1e-9) * 255).astype(np.uint8)
            q = cv2.copyMakeBorder(q, 0, H2 - h, 0, W2 - wd, cv2.BORDER_REPLICATE)
            w.write(cv2.cvtColor(q, cv2.COLOR_GRAY2BGR))
            p10, p50, p90 = np.percentile(n, [10, 50, 90])
            info.append([float(p10), float(p50), float(p90)])
            samples.append(n[::4, ::4].copy())  # to fit bytes -> nearness after decoding
            if progress:
                progress((i + 1) / len(frames))
            if i in (0, 9) or (i + 1) % 100 == 0:
                print(f"depth {name}: {i + 1}/{len(frames)} frames, {time.time() - t0:.0f} s", file=sys.stderr)
        w, done = None, w
        done.close()
        fit = []
        for i, (_, y) in enumerate(_luma(tmp)):  # what the decoder hands back, fitted to what was written, per frame
            if i >= len(samples):
                break
            x = y[:shape[0]:4, :shape[1]:4].astype(np.float64).ravel()
            a, b = np.polyfit(x, samples[i].astype(np.float64).ravel(), 1) if np.ptp(x) > 0 else (0.0, float(samples[i].mean()))
            fit.append([float(a), float(b)])
        if len(fit) != len(frames):
            raise RuntimeError(f"depth movie: wrote {len(frames)} frames, read back {len(fit)}")
        with open(side + ".tmp", "w") as fh:
            json.dump({"model": name, "shape": list(shape), "frames": [f + q for f, q in zip(fit, info)]}, fh)
        os.replace(tmp, movie)
        os.replace(side + ".tmp", side)
    finally:  # on failure: no half-written movie or sidecar left behind (cached() needs both)
        if w is not None:
            try:
                w.close()
            except Exception:
                pass
        for p in (tmp, side + ".tmp"):
            if os.path.exists(p):
                os.remove(p)


def _luma(movie, start=0, count=None):
    """(frame index, Y plane) from the depth movie, from frame `start`."""
    from mojimask import video_io
    end = None if count is None else (start + count - 1) / MOVIE_FPS
    for pts, y in video_io.luma(movie, start / MOVIE_FPS, end):
        i = int(round(pts * MOVIE_FPS))
        if i >= start:
            yield i, y


_LOADED = {}


class Depth:
    """near(t, c): how much nearer the middle of box c ([cx, cy, s], frame pixels) is at frame t than the frame's
    middle depth, in units of the frame's depth spread (p90 - p10 of nearness). Scale-free, so the three models (one
    metric, two relative) are compared the same way: a box's reading minus that face's own depth around then (face_at)."""

    def __init__(self, proj, name, progress=None):
        self.name = name
        self.front, self.alone, self.apart = (MODELS[name][k] for k in ("front", "alone", "apart"))
        compute(proj, name, progress)
        self.movie, side = cache_paths(proj, name)
        key = (self.movie, os.path.getmtime(self.movie))
        im = cv2.imread(os.path.join(proj, "frames", "00000.jpg"))
        if key not in _LOADED:  # one decode per process (the scan builds several Depths)
            meta = json.load(open(side))
            fr = np.array(meta["frames"], np.float64)  # a, b, p10, p50, p90 per frame
            h, w = meta["shape"]
            H, W = im.shape[:2]  # the readings grid has the frame's shape (the model's is rounded to 14 px)
            k = CACHE_SIDE / max(H, W)
            small, maps = (max(1, round(W * k)), max(1, round(H * k))), []
            for i, y in _luma(self.movie):
                if i < len(fr):
                    maps.append(cv2.resize(y[:h, :w].astype(np.float32) * fr[i, 0] + fr[i, 1], small, interpolation=cv2.INTER_AREA))
            _LOADED[key] = (fr, (h, w), np.array(maps, np.float32))
        self.fr, self.shape, self.maps = _LOADED[key]
        self.k = self.maps.shape[2] / im.shape[1]  # Depth.maps px per frame px
        self.p50, self.spread = self.fr[:, 3], np.maximum(self.fr[:, 4] - self.fr[:, 2], 1e-6)
        self._chunks, self._lock = {}, threading.Lock()

    def full(self, t):
        """Frame t's nearness at the model's own resolution (float32), for drawing. Decoded CHUNK frames at a time
        (the editor and Export ask for neighbouring frames), the last few chunks kept."""
        c = t // CHUNK
        with self._lock:
            if c not in self._chunks:
                h, w = self.shape
                got = {i: y[:h, :w] for i, y in _luma(self.movie, c * CHUNK, CHUNK) if i < (c + 1) * CHUNK}
                if len(self._chunks) >= 4:
                    self._chunks.pop(next(iter(self._chunks)))
                self._chunks[c] = got
            y = self._chunks[c].get(t)
        if y is None:
            return None
        return y.astype(np.float32) * float(self.fr[t, 0]) + float(self.fr[t, 1])

    @classmethod
    def for_scan(cls, proj, progress=None):
        name = setting()
        return cls(proj, name, progress) if name else None

    def near(self, t, c, f=0.25):
        if not 0 <= t < len(self.maps):
            return float("nan")
        h, w = self.maps.shape[1:]
        x, y, s = c[0] * self.k, c[1] * self.k, max(c[2] * self.k, 2)
        x1, x2, y1, y2 = int(max(x - f * s, 0)), int(min(x + f * s + 1, w)), int(max(y - f * s, 0)), int(min(y + f * s + 1, h))
        if x2 <= x1 or y2 <= y1:
            return float("nan")
        return float((np.median(self.maps[t, y1:y2, x1:x2]) - self.p50[t]) / self.spread[t])

    def gap(self, a, b, k=3):
        """How far apart two boxes' readings are ({t: box} each) where they come closest in time: k frames of each
        nearest that moment. The same person seen twice (your same-person answers) never read more than 0.12 apart
        on jazz, party_clip_c and IMG_1043 with Depth Anything; MODELS "apart", used by scan/same.py."""
        ta, tb = sorted(a), sorted(b)
        if not ta or not tb:
            return float("nan")
        _, x, y = min((abs(p - q), p, q) for p in ta for q in tb)
        na = [self.near(t, a[t]) for t in sorted(ta, key=lambda t: abs(t - x))[:k]]
        nb = [self.near(t, b[t]) for t in sorted(tb, key=lambda t: abs(t - y))[:k]]
        return abs(float(np.nanmedian(na)) - float(np.nanmedian(nb)))


# ---- partial occlusion: the part of a cover that's behind something nearer is drawn as the video, not the emoji
# Jazz's dancer reaching across a seated face: the emoji was drawn whole on top of her arm. With a depth model on, the
# drawing (gpu_draw's shader, or track.draw_frame on the CPU) shows the frame's own pixels where the depth map reads
# clearly nearer than that face does around then (Occlusion.face). Occlusion.plan decides, once per frame, which covers
# are cut and against what; both drawing paths use it. Safety, since a wrong cut shows a face:
#   - a face detected in this frame needs the model's "alone" gap, a held one its "front" gap, its rim only "apart"
#     (FACE_ZONE); drawn covers (manual) are never cut;
#   - the occluder is trimmed by ~1/CACHE_SIDE of the frame (the map eroded), so the emoji runs a little past the edge;
#   - pixels under any other cover are never restored (a nearer person's own face stays under their own emoji);
#   - a cover is cut only when at least OCC_MIN of its square is in front in a majority of frames t-OCC_STEADY ..
#     t+OCC_STEADY: brief flickers of the depth model don't cut, or uncut, an emoji (on/off switches, 2026-11-13:
#     jazz 384 -> 269, party_clip_c 329 -> 263, IMG_1043 1993 -> 1345 with a 5-frame window);
#   - not when most of its face zone reads in front unless another, nearer person is within reach (OCC_SELF, REACH:
#     otherwise it's the person's own head or arm).
# Edges: the shader brings the map to full size with a joint bilateral upsample; the CPU fallback with a guided filter.
OCC_SOFT, OCC_MIN = 0.15, 0.03
OCC_STEADY = 2  # a cover is cut if it's in front in a majority of the frames within OCC_STEADY of t
# If most of a cover's face zone (FACE_ZONE[0]) reads in front of its face, it's only cut when another person, clearly
# nearer (the model's "apart"), is within REACH of their head widths sideways: then it's them in front (jazz frame 51,
# the dancer's arm across two seated faces). Otherwise it's this person's own head or arm reading nearer than their
# face's estimate, and cutting shows their face -- the party clip, frame 359: a hand on the head, the emoji cut away
# over the face it covered. Something that isn't a person (a pillar, a cup) still cuts while it leaves most of the face
# zone uncovered.
OCC_SELF, REACH = 0.5, 4.0
FACE_WINDOW = 15  # frames either side for a face's own depth (Occlusion.face)
# How much nearer counts as in front depends on where in the cover: the face itself (eyes to chin: a circle of
# FACE_ZONE[0] head sizes around a point track.FACE_DY head sizes below the head box's centre, where
# track.face_to_head puts it) needs the model's "alone"/"front" margin, since a wrong cut there shows a face; past
# FACE_ZONE[1] (hair, ears, the emoji's overhang) the model's "apart" is enough -- the depth gap at which two boxes are
# never the same person. The editor's depth view draws the same zone (review_server.depth_frame sends both).
# Jazz frame 126: the dancer's arm reads ~0.25 nearer than the seated faces it crosses, under the 0.5 a detected face
# needs, so their emoji were drawn over her arm.
FACE_ZONE = (0.40, 0.55)


def zone_r(x, y, c):
    """Distance of pixels x, y (broadcastable arrays, frame px) from head box c's face centre, in head sizes."""
    return np.hypot(x - c[0], y - (c[1] + T.FACE_DY * c[2])) / max(c[2], 1e-6)


def margin_at(r, face_margin, rim):
    """The margin at distance r (zone_r): face_margin on the face, rim past it."""
    w = np.clip((r - FACE_ZONE[0]) / (FACE_ZONE[1] - FACE_ZONE[0]), 0, 1)
    return face_margin + (min(rim, face_margin) - face_margin) * w


def _guided(I, p, r, eps=1e-3):
    """He et al.'s guided filter: p smoothed but following the edges of guide image I (both float32, 0..1)."""
    k = (2 * r + 1, 2 * r + 1)
    mean = lambda x: cv2.boxFilter(x, -1, k)  # noqa: E731
    mI, mp = mean(I), mean(p)
    a = (mean(I * p) - mI * mp) / (mean(I * I) - mI * mI + eps)
    b = mp - a * mI
    return mean(a) * I + mean(b)


def face_at(ts, v, t):
    """A face's depth reading at frame t from its own detected readings v at frames ts (numpy arrays, sorted, finite):
    the median of those within FACE_WINDOW frames of t (past a gap with none, the window starts at the nearest ones),
    never the one at t itself -- that is where an occluder would be. At least 5 readings (the nearest ones) when it
    has them. Used by Occlusion.face (drawing) and scan/behind.py (which held frames are behind someone)."""
    if not len(v):
        return float("nan")
    d = np.abs(ts - t).astype(float)
    d[ts == t] = np.inf
    if not np.isfinite(d).any():
        return float(np.median(v))
    m = d <= FACE_WINDOW + d.min() - 1
    if m.sum() < 5:
        m = np.argsort(d)[:min(5, int(np.isfinite(d).sum()))]
    return float(np.median(v[m]))


class Occlusion:
    """Per frame: plan(t, items, scale, shape) -> (near map, {item index: cut}) decides which covers are cut;
    front(c, scale, cut, frame, near) -> (x1, y1, x2, y2, alpha) is a cut's alpha on the CPU (1 = draw the video).
    nearness(t, piece, c) orders covers far to near."""

    def __init__(self, depth, obs):
        self.D, self.obs = depth, obs  # obs: {piece: {t: [cx, cy, s]}} -- each piece's detected boxes
        self._readings, self._faces, self._maps = {}, {}, {}  # caches shared by request and export threads:
        self._lock = threading.Lock()  # each value is computed into a local and returned from it, never read back

    def readings(self, piece):
        """(frames, readings): the piece's detected boxes read on the depth map (finite ones), sorted by frame. A part
        you split off reads like its track."""
        p = str(piece).split("~")[0]
        r = self._readings.get(p)
        if r is None:
            o = self.obs.get(p) or {}
            ts = sorted(int(t) for t in o)
            v = np.array([self.D.near(t, o[t] if t in o else o[str(t)]) for t in ts])
            ok = np.isfinite(v)
            r = self._readings[p] = (np.array(ts)[ok], v[ok])
        return r

    def face(self, piece, t):
        """The face's depth reading at frame t: the median of its own detected readings within FACE_WINDOW frames of t
        (past any gap with none: the window starts at the nearest ones), never the reading at t itself. A median, so
        readings spoiled by an arm crossing the face (still detected) don't count while they're under half the
        window; local, so it follows someone walking toward the camera. On three clips it beat one number per person,
        line-interpolating the nearest readings, and a head-size model (2026-11-12)."""
        key = (str(piece).split("~")[0], t)
        v = self._faces.get(key)
        if v is None:
            v = face_at(*self.readings(piece), t)
            with self._lock:
                if len(self._faces) > 20000:
                    self._faces.clear()
                self._faces[key] = v
        return v

    def nearness(self, t, piece, c):
        """How near a cover's face is at frame t, to draw covers far to near: its own depth around then, else its box's
        reading now (a drawn cover), else -inf."""
        v = self.face(piece, t)
        if not np.isfinite(v):
            v = self.D.near(t, c)
        return v if np.isfinite(v) else -np.inf

    def map(self, t):
        """Frame t's nearness at the model's own resolution, over the frame's median in units of its spread, eroded by
        ~1/CACHE_SIDE of the frame (the occluder trimmed so the face's edge stays covered). A few frames kept."""
        e = self._maps.get(t)
        if e is None:
            m = (self.D.full(t) - self.D.p50[t]) / self.D.spread[t]
            g = max(1, round(max(m.shape) / CACHE_SIDE))
            e = cv2.erode(m, np.ones((2 * g + 1, 2 * g + 1), np.uint8))
            with self._lock:
                while len(self._maps) >= 6:
                    self._maps.pop(next(iter(self._maps)), None)
                self._maps[t] = e
        return e

    def _in_front(self, t, c, scale, face, margin, shape, guard=True):
        """(fraction of cover c's square reading in front of `face` at frame t, whether most of its face zone does --
        OCC_SELF -- and guard is on), at map resolution."""
        m = self.map(t)
        mh, mw = m.shape
        kx, ky = mw / shape[1], mh / shape[0]
        h = c[2] * scale / 2
        x1, x2 = max(int((c[0] - h) * kx), 0), min(int(np.ceil((c[0] + h) * kx)) + 1, mw)
        y1, y2 = max(int((c[1] - h) * ky), 0), min(int(np.ceil((c[1] + h) * ky)) + 1, mh)
        if x2 <= x1 or y2 <= y1:
            return 0.0, False
        d = m[y1:y2, x1:x2] - face
        r = zone_r(((np.arange(x1, x2) + 0.5) / kx)[None], ((np.arange(y1, y2) + 0.5) / ky)[:, None], c)
        zone = r < FACE_ZONE[0]
        if guard and zone.any() and (d[zone] >= margin).mean() > OCC_SELF:
            return 0.0, True
        return float(np.clip((d - margin_at(r, margin, self.D.apart)) / OCC_SOFT + 0.5, 0, 1).mean()), False

    def _someone_nearer(self, i, items, faces, face):
        """Another cover in the frame whose face reads clearly nearer (the model's "apart") than `face`, within REACH
        of its own head widths sideways of cover i."""
        c = items[i][1]
        return any(j != i and faces[j] - face >= self.D.apart and abs(x[1][0] - c[0]) <= REACH * x[1][2]
                   for j, x in enumerate(items))

    def plan(self, t, items, scale, shape, why=None):
        """items: the frame's covers [(tid, c, kind, piece?), ...] -> (near map for frame t or None, {index: cut}),
        cut = {"face", "margin", "rim", "head"}: the face's depth, the margin on the face and on its rim, and the head
        size. A cover is cut when at least OCC_MIN of its square is in front in a majority of the frames within
        OCC_STEADY of t, and not when most of its face zone is (OCC_SELF; module notes).
        why: a dict to fill, {index: {face, margin, rim, self, in_front, someone_nearer}} for each cover weighed."""
        n = len(self.D.maps)
        if not 0 <= t < n:
            return None, {}
        out = {}
        faces = [self.nearness(t, x[3] if len(x) > 3 else x[0], x[1]) for x in items]
        for i, (tid, c, kind, *rest) in enumerate(items):
            if kind not in ("obs", "coast"):
                continue
            piece = rest[0] if rest else tid
            face = self.face(piece, t)
            if not np.isfinite(face):
                continue
            margin = self.D.alone if kind == "obs" else self.D.front
            guard = not self._someone_nearer(i, items, faces, face)
            now, itself = self._in_front(t, c, scale, face, margin, shape, guard)
            if why is not None:
                why[i] = {"face": face, "margin": margin, "rim": self.D.apart, "self": itself, "in_front": now,
                          "someone_nearer": not guard}
            if itself:
                continue  # its own head or arm in front of its face (OCC_SELF): never cut this frame
            # the vote reuses frame t's box c on the neighbouring frames' maps, on purpose: the window is a few frames,
            # and it asks whether the depth model flickers where the cover is now
            vote = {t: now >= OCC_MIN}
            for u in range(max(t - OCC_STEADY, 0), min(t + OCC_STEADY, n - 1) + 1):
                if u not in vote:
                    vote[u] = self._in_front(u, c, scale, self.face(piece, u), margin, shape, guard)[0] >= OCC_MIN
            # past the clip's ends the edge frame counts again: a cut starting on the last frame isn't a flicker
            if sum(vote[min(max(u, 0), n - 1)] for u in range(t - OCC_STEADY, t + OCC_STEADY + 1)) < OCC_STEADY + 1:
                continue
            out[i] = {"face": face, "margin": margin, "rim": self.D.apart, "head": c[2]}
        return (self.map(t) if out else None), out

    def explain(self, t, items, scale, shape):
        """For the editor's depth view: per cover, what Occlusion decided at frame t and from what. A list of
        {piece, box, kind, face, margin, rim, self, in_front, cut}: face depth, the margins on the face and its rim,
        whether the face zone reads in front of itself (OCC_SELF), the fraction of the square in front now, cut or not."""
        why = {}
        _, cuts = self.plan(t, items, scale, shape, why)
        out = []
        for i, (tid, c, kind, *rest) in enumerate(items):
            piece = rest[0] if rest else tid
            e = {"piece": str(piece), "box": [float(v) for v in c], "kind": kind, "cut": i in cuts}
            if i in why:
                w = why[i]
                e.update(w, face=round(w["face"], 3), in_front=round(w["in_front"], 3))
            out.append(e)
        return out

    def picture(self, t):
        """Frame t's depth map coloured (turbo: blue far, red near, over the frame's p10..p90), BGR uint8 at the
        model's resolution."""
        m = self.D.full(t)
        lo, hi = self.D.fr[t, 2], self.D.fr[t, 4]
        q = np.clip((m - lo) / max(hi - lo, 1e-6) * 0.8 + 0.1, 0, 1)
        return cv2.applyColorMap((q * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

    def front(self, c, scale, cut, frame, near):
        """CPU fallback: the alpha of a planned cut over cover c's square (crop of `frame`), the depth sampled just
        there from `near` (plan's map) and snapped to the picture's edges with a guided filter."""
        H, W = frame.shape[:2]
        h = c[2] * scale / 2
        x1, y1, x2, y2 = max(int(c[0] - h), 0), max(int(c[1] - h), 0), min(int(c[0] + h) + 1, W), min(int(c[1] + h) + 1, H)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        mh, mw = near.shape
        kx, ky = mw / W, mh / H
        M = np.float32([[kx, 0, (x1 + 0.5) * kx - 0.5], [0, ky, (y1 + 0.5) * ky - 0.5]])
        d = cv2.warpAffine(near, M, (x2 - x1, y2 - y1), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_REPLICATE) - cut["face"]
        r = zone_r((np.arange(x1, x2) + 0.5)[None], (np.arange(y1, y2) + 0.5)[:, None], c)
        p = np.clip((d - margin_at(r, cut["margin"], cut["rim"])) / OCC_SOFT + 0.5, 0, 1)
        grey = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        p = np.clip(_guided(grey, p.astype(np.float32), max(2, int(c[2] * scale / 16))), 0, 1)
        return x1, y1, x2, y2, cv2.GaussianBlur(p, (3, 3), 0)
