"""Shared geometry, camera motion, drawing and rendering for the app.

  geometry / fusion   boxes as [cx, cy, size]; detector outputs fused per frame (fused_frames)
  camera motion       frame-to-frame similarity transforms from sparse optical flow (camera_motion, compose)
  resolve             what's drawn on every frame from the scan's tracks and the edits (joins, splits, trims, masks
                      you drew, who's shown), then bridged and steadied
  rendering           emoji sprites and other covers (draw_frame, render)
The scan itself (tracks, people, questions) is the scan package (scan/newalgo.py); it uses the geometry and flow helpers here.
"""
import json
import os
import threading

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mojimask.paths import WEB

# Detectors whose boxes are tight faces (fused_frames turns them into head boxes). The scan uses retinaface_r50_2x
# and yolov5m_face_tiled (pipeline.CORE_DETS); the rest are older or experiments-only (experiments/detect.py).
FACE_DETS = ("retinaface_r50_2x", "scrfd10g_2x", "yolov5m_face_tiled", "yolo11l_face", "yolo12l_face", "sam3_face",
             "egoblur_gen2", "apple_vision")


# ------------------------------------------------------------------ geometry
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def frac_inside(a, b):
    """Fraction of square a=[cx, cy, s] inside square b."""
    A_, B_ = from_cs(a), from_cs(b)
    iw = max(0, min(A_[2], B_[2]) - max(A_[0], B_[0])); ih = max(0, min(A_[3], B_[3]) - max(A_[1], B_[1]))
    return iw * ih / max((A_[2] - A_[0]) * (A_[3] - A_[1]), 1e-6)


def overlap_of_smaller(a, b):
    """Overlap of squares a and b=[cx, cy, s] as a fraction of the smaller one."""
    A_, B_ = from_cs(a), from_cs(b)
    iw = max(0, min(A_[2], B_[2]) - max(A_[0], B_[0])); ih = max(0, min(A_[3], B_[3]) - max(A_[1], B_[1]))
    return iw * ih / max(min((A_[2] - A_[0]) * (A_[3] - A_[1]), (B_[2] - B_[0]) * (B_[3] - B_[1])), 1e-6)


def to_cs(b):  # box -> center, size (square)
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(b[2] - b[0], b[3] - b[1])])


def from_cs(c, k=1.0):
    h = c[2] * k / 2
    return [c[0] - h, c[1] - h, c[0] + h, c[1] + h]


FACE_DY = 0.08  # the face's centre sits this many head sizes below its head box's (depth.zone_r, gpu_draw's shader)


def face_to_head(b):
    """Face box -> square head box (faces boxes are tight around eyes-to-chin)."""
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    s = max(b[2] - b[0], b[3] - b[1]) * 1.35
    return [cx - s / 2, cy - s / 2 - FACE_DY * s, cx + s / 2, cy + s / 2 - FACE_DY * s]


# ------------------------------------------------------------------ fusion
def fuse(boxes, thr=0.3):
    """boxes: list of (box, score, src). Cluster by IoU, score-weighted average box.
    Fused score = 1 - prod(1 - s) over distinct sources (agreement boosts confidence)."""
    boxes = sorted(boxes, key=lambda x: -x[1])
    clusters = []
    for b, s, src in boxes:
        for c in clusters:
            if iou(c["box"], b) > thr:
                c["m"].append((b, s, src))
                w = np.array([m[1] for m in c["m"]])
                c["box"] = list((np.array([m[0] for m in c["m"]]) * w[:, None]).sum(0) / w.sum())
                break
        else:
            clusters.append({"box": list(b), "m": [(b, s, src)]})
    out = []
    for c in clusters:
        best = {}
        for _, s, src in c["m"]:
            best[src] = max(best.get(src, 0), s)
        score = 1 - np.prod([1 - s for s in best.values()])
        out.append((c["box"], float(score), sorted(best)))
    return out


# ------------------------------------------------------------------ camera motion
def camera_motion(video, scale=0.5, meta=None):
    """List of 2x3 affine matrices mapping frame t-1 coords -> frame t coords (on video_io's timeline)."""
    from mojimask import video_io
    mats, prev = [np.eye(2, 3)], None
    for f in video_io.frames(video, meta):
        g = cv2.cvtColor(cv2.resize(f, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            M = np.eye(2, 3)
            p0 = cv2.goodFeaturesToTrack(prev, 400, 0.01, 6)
            if p0 is not None and len(p0) > 10:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None)
                ok_ = st.flatten() == 1
                if ok_.sum() > 10:
                    m, inl = cv2.estimateAffinePartial2D(p0[ok_], p1[ok_], method=cv2.RANSAC,
                                                         ransacReprojThreshold=2.0)
                    # Trust the fit only if most points agree (a hand filling the frame breaks this),
                    # and never let camera "zoom" exceed a few % per frame.
                    if m is not None and inl is not None and inl.mean() > 0.4:
                        M = m.copy()
                        M[:, 2] /= scale
                        sc = np.sqrt(abs(np.linalg.det(M[:, :2])))
                        M[:, :2] /= sc
                        M[:, :2] *= float(np.clip(sc, 0.97, 1.03))
            mats.append(M)
        prev = g
    return mats


def warp_cs(c, M):
    p = M @ np.array([c[0], c[1], 1.0])
    s = np.sqrt(abs(np.linalg.det(M[:, :2])))
    return np.array([p[0], p[1], c[2] * s])


# ------------------------------------------------------------------ tracking
def compose(mats, a, b):
    """Affine taking frame a coords -> frame b coords (a < b)."""
    M = np.eye(3)
    for t in range(a + 1, b + 1):
        M = np.vstack([mats[t], [0, 0, 1]]) @ M
    return M[:2]


# ------------------------------------------------------------------ rendering
EMOJI = list("😀😎🤩🥳😜🤪😺🐸🐵🦄👽🤖🎃👻🌝🌞🍩🍕🦊🐼🐯🐶🐙🦋")


def _emoji_glyph_macos(ch, px=160):
    """ch drawn by macOS's own text engine (CoreText, through AppKit): the same drawing the browser does, so
    joined emoji (ZWJ sequences like woman + microphone = 👩‍🎤, skin tones, flags) come out as ONE picture.
    Pillow without libraqm can't join them and draws the parts side by side. RGBA (straight alpha) or None."""
    try:
        import AppKit
        import Quartz
    except ImportError:
        return None
    W = H = int(px * 1.8)
    ctx = Quartz.CGBitmapContextCreate(None, W, H, 8, W * 4, Quartz.CGColorSpaceCreateDeviceRGB(),
                                       Quartz.kCGImageAlphaPremultipliedLast)
    gc = AppKit.NSGraphicsContext.graphicsContextWithCGContext_flipped_(ctx, False)
    font = AppKit.NSFont.fontWithName_size_("Apple Color Emoji", px)
    if font is None:
        return None
    s = AppKit.NSAttributedString.alloc().initWithString_attributes_(ch, {AppKit.NSFontAttributeName: font})
    AppKit.NSGraphicsContext.saveGraphicsState()
    try:
        AppKit.NSGraphicsContext.setCurrentContext_(gc)
        sz = s.size()
        s.drawAtPoint_(((W - sz.width) / 2, (H - sz.height) / 2))
        gc.flushGraphics()
    finally:
        AppKit.NSGraphicsContext.restoreGraphicsState()
    img = Quartz.CGBitmapContextCreateImage(ctx)
    data = bytes(Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(img)))
    a = np.frombuffer(data, np.uint8).reshape(H, Quartz.CGImageGetBytesPerRow(img) // 4, 4)[:, :W].copy()
    al = a[..., 3:4].astype(np.float32)
    a[..., :3] = np.clip(a[..., :3] * 255.0 / np.maximum(al, 1), 0, 255).astype(np.uint8)  # un-premultiply
    return a


_GLYPH_LOCK = threading.Lock()  # PyObjC loads Quartz/AppKit functions lazily and not thread-safely: the editor's
                                # server draws frames on many threads, and two first-time emoji at once raised
                                # KeyError('CGColorSpaceCreateDeviceRGB') -> a frame with no emoji


def emoji_sprite(ch, size=160):
    """The emoji picture every cover uses (export and live preview alike): cropped to its ink, size x size RGBA.
    A portrait key (portrait.key_for: the Portrait style's photo) gives that photo instead, at twice the size."""
    from mojimask import portrait
    if portrait.is_portrait(ch):
        return portrait.sprite(ch, size * 2)
    with _GLYPH_LOCK:
        a = _emoji_glyph_macos(ch)
    if a is not None and (a[..., 3] > 8).any():
        im = Image.fromarray(a, "RGBA")
    else:  # PyObjC missing (or it drew nothing): Pillow with the same macOS font (joined emoji may come out as parts)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Apple Color Emoji.ttc", 160)
        except OSError:
            from mojimask import about
            raise RuntimeError(f"can't draw emoji: the Apple Color Emoji font isn't there ({about.NAME} needs macOS)")
        im = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((100, 100), ch, font=font, embedded_color=True, anchor="mm")
    im = im.crop(im.getbbox())
    if ch in HEAD_ONLY:
        im = _head_only(im, *HEAD_ONLY[ch])
    return np.array(im.resize((size, size), Image.LANCZOS))  # RGBA


# Lining the emoji's eyes up with the person's (emoji_eyes.json: where each emoji's eyes are, and where real eyes
# sit in a cover box). An emoji whose eyes sit low in its picture (🐰 under its ears) is drawn higher, one with
# eyes high up (🐸) lower. Moving it uncovers face at the other edge, so it is drawn larger as far as needed to
# still reach EYE_KEEP face sizes past the face box (today's margin at scale 1.25 is 0.125), up to EYE_GROW_MAX;
# past that it only goes part of the way. Emoji not listed (no eyes: food) are drawn centred as before.
EYE_KEEP, EYE_GROW_MAX = 0.06, 1.2
_EYES = json.load(open(os.path.join(WEB, "emoji_eyes.json")))
_EYE_FIT = {}


def eye_fit(ch, scale):
    """(move, grow) for emoji ch drawn at `scale` times the face box: move its centre down by move face sizes
    (negative = up) and draw it grow times larger. (0, 1) for emoji without eyes and for portraits."""
    key = (ch, scale)
    if key not in _EYE_FIT:
        e = _EYES["eye_y"].get(ch)
        if e is None or scale <= 0:
            _EYE_FIT[key] = (0.0, 1.0)
        else:  # in drawn-picture sizes (scale x face box): face box half-height h, margin m, real eye line r
            h, m, r = 0.5 / scale, EYE_KEEP / scale, _EYES["reference"] / scale
            g = min(EYE_GROW_MAX, max(1.0, (h + m - r) / (1 - e), (r + h + m) / e))
            lo, hi = h + m - g / 2, g / 2 - h - m  # the centre may move this far and still cover
            d = min(max(r - g * (e - 0.5), lo), hi) if lo <= hi else 0.0
            _EYE_FIT[key] = (d * scale, g)
    return _EYE_FIT[key]


def eye_place(ch, c, scale):
    """Face box c=[cx, cy, s, ...] -> the box to draw emoji ch in (same form, so paste(..., c, scale) draws it
    with its eyes on the person's)."""
    d, g = eye_fit(ch, scale)
    if d == 0 and g == 1:
        return c
    return [c[0], c[1] + d * c[2], c[2] * g, *c[3:]]


# Emoji that only exist as a whole animal (Unicode has no sheep face): cover with just the head, cut from
# the top-left of the picture (where Apple draws it) as (right, bottom) fractions of the ink box, with a soft
# round edge so the cut into the wool doesn't show as a straight line.
HEAD_ONLY = {"🐑": (0.36, 0.38), "🐏": (0.36, 0.38)}


def _head_only(im, fx, fy):
    from PIL import ImageFilter
    W, H = im.size
    hd = im.crop((0, 0, max(1, int(W * fx)), max(1, int(H * fy))))
    w, h = hd.size
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).ellipse((-w * 0.25, -h * 0.25, w * 0.98, h * 0.98), fill=255)
    m = m.filter(ImageFilter.GaussianBlur(w * 0.05))
    hd.putalpha(Image.fromarray(np.minimum(np.array(hd)[..., 3], np.array(m))))
    return hd.crop(hd.getbbox() or (0, 0, w, h))


def paste(frame, sprite, c, k):
    s = int(round(c[2] * k))
    if s < 4:
        return
    sp = cv2.resize(sprite, (s, s), interpolation=cv2.INTER_AREA)
    x0, y0 = int(round(c[0] - s / 2)), int(round(c[1] - s / 2))
    H, W = frame.shape[:2]
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + s, W), min(y0 + s, H)
    if xa >= xb or ya >= yb:
        return
    sp = sp[ya - y0:yb - y0, xa - x0:xb - x0]
    a = sp[:, :, 3:4] / 255.0
    rgb = sp[:, :, 2::-1].astype(float)  # RGBA -> BGR
    frame[ya:yb, xa:xb] = (a * rgb + (1 - a) * frame[ya:yb, xa:xb]).astype(np.uint8)


WEAK_DET = 0.15  # every detector's threshold for the weak boxes covers snap to (cover.py, the editor's LOW)


def fused_frames(detdir, spec):
    """Per frame, the fused boxes (fuse) of the detectors in spec [(name, threshold)], face boxes made head boxes."""
    if not spec:
        raise ValueError("fused_frames: no detectors given")
    raw = {}
    for name, _ in spec:
        with open(f"{detdir}/{name}.json") as fh:
            raw[name] = json.load(fh)["frames"]
    n = len(next(iter(raw.values())))
    frames = []
    for t in range(n):
        boxes = []
        for name, th in spec:
            for *b, s in raw[name][t]:
                if s >= float(th):
                    boxes.append((face_to_head(b) if name in FACE_DETS else b, s, name))
        frames.append(fuse(boxes))
    return frames


# ------------------------------------------------------------------ flow helpers (scan/follow.py, the editor's covers)
def inside_frac(c, W, H):
    """Fraction of square box c inside a W x H frame."""
    x1, y1, x2, y2 = from_cs(c)
    iw, ih = max(0.0, min(x2, W) - max(x1, 0)), max(0.0, min(y2, H) - max(y1, 0))
    return iw * ih / max((x2 - x1) * (y2 - y1), 1e-6)


def _heading_out(c, v, W, H, horizon=20):
    """At own velocity v, would box c be (almost) out of shot within `horizon` frames?"""
    return np.linalg.norm(v) >= 1.0 and inside_frac([c[0] + v[0] * horizon, c[1] + v[1] * horizon, c[2]], W, H) < 0.05


LK_WIN, LK_LEVELS = (15, 15), 3
LK_MARGIN = 160  # px around the points: LK's reach (7 px window x 8 at the top level) plus fast motion


def lk_fb(g0, g1, p0):
    """Forward-backward LK of points p0 from g0 to g1 -> (p1, good mask: tracked both ways within 1.5 px).
    Runs on a crop around the points, not the whole frame: LK's pyramids were rebuilt over the full frame
    for every box and direction, most of the analysis time on high-res video. The crop starts on the
    pyramid's 8-px grid and leaves LK_MARGIN px around the points, so the flow matches the full-frame one
    except for points that move further than the margin in one frame (lost, as they mostly were anyway)."""
    H, W = g0.shape[:2]
    q = p0.reshape(-1, 2)
    step = 1 << LK_LEVELS
    x1 = max(int(q[:, 0].min()) - LK_MARGIN, 0) // step * step
    y1 = max(int(q[:, 1].min()) - LK_MARGIN, 0) // step * step
    x2, y2 = min(int(q[:, 0].max()) + LK_MARGIN + 1, W), min(int(q[:, 1].max()) + LK_MARGIN + 1, H)
    o = np.array([x1, y1], np.float32)
    a, b = g0[y1:y2, x1:x2], g1[y1:y2, x1:x2]
    c0 = (p0 - o).astype(np.float32)
    c1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, c0, None, winSize=LK_WIN, maxLevel=LK_LEVELS)
    cb, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, c1, None, winSize=LK_WIN, maxLevel=LK_LEVELS)
    fb = np.linalg.norm((cb - c0).reshape(-1, 2), axis=1)
    return (c1 + o).astype(np.float32), (st.flatten() == 1) & (st2.flatten() == 1) & (fb < 1.5)


def box_corners(g, x1, y1, x2, y2, n=60, quality=0.005, min_dist=2):
    """cv2.goodFeaturesToTrack inside the box, computed on a crop with a small margin instead of the whole
    frame with a mask (the corner measure only looks a few pixels around each pixel, so it is the same)."""
    H, W = g.shape[:2]
    m = 6
    cx1, cy1, cx2, cy2 = max(x1 - m, 0), max(y1 - m, 0), min(x2 + m, W), min(y2 + m, H)
    mask = np.zeros((cy2 - cy1, cx2 - cx1), np.uint8)
    mask[y1 - cy1:y2 - cy1, x1 - cx1:x2 - cx1] = 255
    p = cv2.goodFeaturesToTrack(np.ascontiguousarray(g[cy1:cy2, cx1:cx2]), n, quality, min_dist, mask=mask)
    return None if p is None else (p + np.array([cx1, cy1], np.float32)).astype(np.float32)


def _flow_measure(g0, g1, c, core):
    """Median LK flow of corners in the core of box c: (d xy, n_good, spread px) or None."""
    H, W = g0.shape[:2]
    x1, y1, x2, y2 = from_cs(c)
    w, h = x2 - x1, y2 - y1
    x1, x2, y1, y2 = x1 + core[0] * w, x1 + core[1] * w, y1 + core[2] * h, y1 + core[3] * h
    x1, y1, x2, y2 = [int(round(v)) for v in (max(x1, 0), max(y1, 0), min(x2, W - 1), min(y2, H - 1))]
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    p0 = cv2.goodFeaturesToTrack(g0[y1:y2, x1:x2], 60, 0.005, 2)
    if p0 is None or len(p0) < 6:
        return None
    p0 = (p0 + np.array([x1, y1], np.float32)).astype(np.float32)
    p1, good = lk_fb(g0, g1, p0)
    if good.sum() < 5:
        return None
    m = (p1 - p0).reshape(-1, 2)[good]
    d = np.median(m, 0)
    return d, int(good.sum()), float(np.median(np.linalg.norm(m - d, axis=1)))


MANUAL_MAX_GAP = 90  # frames: a longer gap between two of a cover's boxes is left empty, not interpolated


def manual_boxes(m, mats, n, hold=10):
    """Manual track: user keyframes + propagated frames; camera-compensated interpolation between
    them (gaps up to MANUAL_MAX_GAP frames), and `hold` frames of camera-carried extension at both ends."""
    pts = {int(t): np.array(c, float) for t, c in m.get("prop", {}).items()}
    keys = {int(t): np.array(c, float) for t, c in m["keys"].items()}
    pts.update(keys)
    if not pts:
        return {}
    # A followed path may only change the box's size so much: 0.67x to 1.5x the nearest keyframe. (Flow
    # allows +-10% per frame, which compounded over a long path once grew a face box past the frame.)
    if keys:
        kt = sorted(keys)
        for t, c in pts.items():
            if t not in keys:
                s0 = keys[min(kt, key=lambda k: abs(k - t))][2]
                c[2] = float(np.clip(c[2], 0.67 * s0, 1.5 * s0))
    ts = sorted(pts)
    out = {t: pts[t] for t in ts}
    for t0, t1 in zip(ts, ts[1:]):
        if t1 - t0 > MANUAL_MAX_GAP:
            continue
        for t in range(t0 + 1, t1):
            al = (t - t0) / (t1 - t0)
            fwd = warp_cs(pts[t0], compose(mats, t0, t))
            bwd = warp_cs(pts[t1], cv2.invertAffineTransform(compose(mats, t, t1)))
            out[t] = (1 - al) * fwd + al * bwd
    c = pts[ts[0]]
    for t in range(ts[0] - 1, max(ts[0] - hold, 0) - 1, -1):
        c = warp_cs(c, cv2.invertAffineTransform(mats[t + 1])); out.setdefault(t, c)
    c = pts[ts[-1]]
    for t in range(ts[-1] + 1, min(ts[-1] + hold, n - 1) + 1):
        c = warp_cs(c, mats[t]); out.setdefault(t, c)
    lo, hi = m.get("from"), m.get("to")  # a cover split in two: each half ends where the other begins
    if lo is not None or hi is not None:
        out = {t: c for t, c in out.items() if (lo is None or t >= lo) and (hi is None or t <= hi)}
    return out


def piece_at(tid, cuts, t):
    """Which piece of track tid frame t belongs to, given sorted cut frames: the part before the first cut
    keeps the track id, each later part is "tid~<cut frame>" (edits.cuts: identity switches split apart)."""
    p = tid
    for c in cuts:
        if t >= c:
            p = f"{tid}~{c}"
    return p


def canon(tid, merges):
    seen = set()
    while tid in merges and tid not in seen:
        seen.add(tid); tid = merges[tid]
    return tid


KIND_RANK = {"key": 0, "obs": 1, "manual": 2, "coast": 3}
SAME_PERSON_FRESH = 2  # frames
HELD_YIELDS = 0.5  # a held box overlapping someone else's current detection this much (of the smaller) is dropped
SAME_FACE_OVERLAP = 0.25  # two boxes of one person overlapping this much (of the smaller) are one face


def resolve(auto, edits, mats, n, links=None, dedupe=True):
    """Apply review edits to the automatic tracks. Returns per-frame [(tid, [cx,cy,s], kind, piece)]:
    tid = canonical identity after re-ID links and user merges (decides the emoji), piece = the
    underlying auto/manual track id. kind in {'obs','coast','manual'}.
    Deletes and trims apply ONLY to the pieces listed -- never implicitly to other pieces linked to
    them -- so an automatic (possibly wrong) re-ID link can never widen a delete and unmask someone."""
    # not_face {piece: why} / not_real [pieces]: boxes the reviewer said aren't a face (hair, a stomach) or
    # aren't a real person (poster, mannequin). Like deleted they get no cover; unlike deleted they keep
    # their place in the person's group, so Review can show them greyed out and undo is one click.
    deleted = set(edits.get("deleted", [])) | set(edits.get("not_face") or {}) | set(edits.get("not_real") or [])
    trims = edits.get("trims", {})
    # automatic re-identification links first; the user's own merges override them; named People
    # (explicit piece lists) override both: their pieces resolve to the person's stable id "@<pid>"
    people = {piece: f"@{pid}" for pid, P in (edits.get("people") or {}).items() for piece in P.get("pieces", [])}
    merges = {**(links or {}), **edits.get("merges", {}), **people}
    per_frame = [[] for _ in range(n)]
    part_of = set(edits.get("part_of") or {})  # boxes answered "part of their face": always drawn as one face with it
    cuts = edits.get("cuts") or {}
    for tid, tr in auto.items():
        obs = set(int(t) for t in tr["obs"])
        cs = sorted(int(c) for c in cuts.get(tid, []))
        seen = {}  # piece -> its detection frames (evidence)
        for t in sorted(obs):
            seen.setdefault(piece_at(tid, cs, t), []).append(t)
        for t, c in tr.get("cover", tr["boxes"]).items():  # cover: the drawn boxes (the scan makes them)
            t = int(t)
            piece = piece_at(tid, cs, t)  # a cut box is several pieces: "tid", "tid~<first frame>", ...
            if piece in deleted:
                continue
            lo, hi = trims.get(piece) or [0, n - 1]
            if lo <= t <= hi:
                per_frame[t].append((canon(piece, merges), list(c), "obs" if t in obs else "coast", len(obs), piece,
                                     _since(seen.get(piece, []), t)))
    for mid, m in edits.get("manual", {}).items():
        if mid in deleted:
            continue
        lo, hi = trims.get(mid) or [0, n - 1]
        cm = canon(mid, merges)
        keys = {int(k) for k in m.get("keys", {})}
        seen = sorted(keys | {int(u) for u in m.get("ev") or []})  # your keyframes + where it snapped to a face
        for t, c in manual_boxes(m, mats, n).items():
            if lo <= t <= hi:
                per_frame[t].append((cm, [float(v) for v in c], "key" if t in keys else "manual", 10 ** 6, mid,
                                     _since(seen, t)))
    for t in range(n):
        if not dedupe:  # every box as is (listing the boxes answered Not a face: none may hide another)
            per_frame[t] = [(tid, c, "manual" if kind == "key" else kind, piece) for tid, c, kind, _, piece, _ in per_frame[t]]
            continue
        per_frame[t] = _dedupe(per_frame[t], part_of)
    if mats is not None:
        # frames where the scan found a person hidden behind someone nearer (scan/behind.py): bridging
        # mustn't put their emoji back there
        behind = {(str(k), t) for k, tr in auto.items() for t in tr.get("behind", ())}
        if BRIDGE:
            bridge(per_frame, mats, behind)
        if STEADY:
            steady(per_frame, mats, {str(k): tr.get("flow") or {} for k, tr in auto.items()})
    return per_frame


def _dedupe(items, part_of):
    """One frame's boxes (resolve's 6-tuples: tid, c, kind, life, piece, frames since evidence) -> the boxes drawn,
    as (tid, c, kind, piece)."""
    # One box per person, the one with the freshest evidence (frames since its own detection, or since your
    # keyframe or a face it snapped to) -- not "a cover always wins": a cover followed from one answer slid
    # off a spinning child onto the furniture and was drawn instead of her detected face. Ties: your
    # keyframe, a detection, a followed cover, a held box, then the longer-lived. A second box of the same
    # person is kept too when it is itself fresh (<= SAME_PERSON_FRESH frames) and elsewhere: two current
    # sightings are both covered, never one uncovered for the other. Boxes mostly overlapping a kept box of
    # someone else are dropped.
    # One person has one face: a fresh second box of theirs that overlaps one already kept (a piece answered
    # "part of their face", or two pieces you joined) is the same face -- one emoji over both boxes (their
    # bounding square), so neither part is uncovered and no second emoji is drawn on top. (party_clip_c,
    # frame 210: box 216 answered part of 208's face got its own copy of the emoji, half on top of it.)
    items = sorted(items, key=lambda x: (x[5], KIND_RANK[x[2]], -x[3]))
    keep = []
    for it in items:
        b = from_cs(it[1])
        if any(k[0] != it[0] and iou(b, from_cs(k[1])) >= 0.4 for k in keep):
            continue
        # a held box (no detection of its own for a while) yields to someone else's current detection it mostly
        # overlaps: the held one is a guess that has drifted onto that face (party_clip_c frames 145 and 153:
        # 74 held over 176's detected face, IoU 0.35, two emoji on one face)
        if it[2] == "coast" and any(k[0] != it[0] and k[2] != "coast" and k[5] <= SAME_PERSON_FRESH
                                    and max(frac_inside(it[1], k[1]), frac_inside(k[1], it[1])) >= HELD_YIELDS for k in keep):
            continue
        same = [k for k in keep if k[0] == it[0]]
        touching = [i for i, k in enumerate(keep) if k[0] == it[0] and (it[4] in part_of or k[4] in part_of  # answered part of their face
                                                                     or max(frac_inside(it[1], k[1]), frac_inside(k[1], it[1])) >= SAME_FACE_OVERLAP)]
        if touching:
            if it[5] <= SAME_PERSON_FRESH:  # a current sighting: the face spans both
                i = touching[0]; kb = from_cs(keep[i][1])
                u = [min(b[0], kb[0]), min(b[1], kb[1]), max(b[2], kb[2]), max(b[3], kb[3])]
                keep[i] = (keep[i][0], [(u[0] + u[2]) / 2, (u[1] + u[3]) / 2, max(u[2] - u[0], u[3] - u[1])], *keep[i][2:])
            continue  # a stale one of theirs on the same face: dropped
        if not same or it[5] <= SAME_PERSON_FRESH:
            keep.append(it)
    # de-clutter: a held (coasting) emoji mostly inside a currently-detected one is a stale duplicate
    live = [k for k in keep if k[2] != "coast"]
    keep = [k for k in keep if k[2] != "coast" or not any(frac_inside(k[1], l[1]) >= 0.7 for l in live)]
    return [(tid, c, "manual" if kind == "key" else kind, piece) for tid, c, kind, _, piece, _ in keep]


def _base(piece):
    """The scan's track a piece comes from ("88~140", a part you split off, is track 88's)."""
    return str(piece).split("~")[0]


# ---- bridging: a person's mask with a short gap (a missed detection between two of their boxes) is filled
# The scan holds a box only a couple of frames past a detection, so a 6-frame miss left 2 bare frames mid-face
# (party_clip_c, person 2, frames 12-13). Gaps up to BRIDGE frames between two boxes of one person (one box each end)
# are filled moving evenly in the scene's frame of reference (camera motion taken out), at the larger of the two
# sizes -- except where the filled box would mostly sit on someone else's detected face (the held-box rule).
# Only where the two ends are near each other in the scene (BRIDGE_REACH head sizes): the scan has already followed
# the face across the gap with optical flow as far as the pictures allow (scan/cover.py), so what is
# left for a straight line is a short remainder. Two far-apart ends have nothing connecting them: jazz, two different
# people merged by their faces (person 32 leaving at the right edge, person 42 entering at the left 16 frames later)
# were bridged by an emoji flying across the picture. Across every project's gaps the largest real one was 1.8 head
# sizes (median 0.1); that one was 25.
BRIDGE, BRIDGE_REACH = 15, 2.5


def _scene(mats, n):
    C = np.empty((n, 3, 3)); C[0] = np.eye(3)
    for t in range(1, n):
        C[t] = (np.vstack([mats[t], [0, 0, 1]]) if t < len(mats) else np.eye(3)) @ C[t - 1]
    return C


def bridge(per_frame, mats, behind=frozenset()):
    n = len(per_frame)
    C = _scene(mats, n)
    Ci = np.linalg.inv(C)
    k = np.sqrt(np.abs(np.linalg.det(C[:, :2, :2])))
    at = {}
    for t, f in enumerate(per_frame):
        cnt = {}
        for x in f:
            cnt.setdefault(x[0], []).append(x)
        for tid, xs in cnt.items():
            at.setdefault(tid, {})[t] = xs
    adds = []
    for tid, frames in at.items():
        ts = sorted(frames)
        for a, b in zip(ts, ts[1:]):
            if not (1 < b - a <= BRIDGE + 1) or len(frames[a]) != 1 or len(frames[b]) != 1:
                continue
            ca, cb = frames[a][0][1], frames[b][0][1]
            pa = Ci[a] @ np.array([ca[0], ca[1], 1.0]); pb = Ci[b] @ np.array([cb[0], cb[1], 1.0])
            sz = max(ca[2] / k[a], cb[2] / k[b])  # scene size, the larger end
            if np.hypot(pa[0] - pb[0], pa[1] - pb[1]) > BRIDGE_REACH * (ca[2] / k[a] + cb[2] / k[b]) / 2:
                continue  # the ends are too far apart for a straight line between them to be the same face
            ends = {_base(frames[a][0][3]), _base(frames[b][0][3])}
            for t in range(a + 1, b):
                if any((e, t) in behind for e in ends):
                    continue  # hidden behind someone there (the scan's behind frames)
                w = (t - a) / (b - a)
                q = C[t] @ (pa * (1 - w) + pb * w)
                c = [float(q[0]), float(q[1]), float(sz * k[t])]
                if any(x[0] != tid and x[2] != "coast" and overlap_of_smaller(c, x[1]) >= HELD_YIELDS for x in per_frame[t]):
                    continue
                adds.append((t, (tid, c, "coast", frames[a][0][3])))
    for t, x in adds:
        per_frame[t].append(x)


# ---- steady emoji: each person's box smoothed over time, in the camera's frame of reference
# A person's box comes from whichever of their pieces is freshest, and detector boxes wobble: the emoji jumped
# (party_clip_c: size changing 10% a frame one frame in ten, 34% one in a hundred; position as much). Each frame's box
# is smoothed with its neighbours' (the same person, one box per frame, consecutive frames) after carrying them into
# this frame with the camera motion, so a pan doesn't drag it: position over a short window (STEADY_POS frames, the
# person's own movement stays), size over a longer one and over the upper envelope (the largest of nearby sizes), so
# it errs larger. It never shrinks below STEADY_KEEP of the box it replaces, plus whatever it takes to still contain
# that box after the position moved: smoothing must not uncover any face.
# Carried along the face's own motion where the scan measured it (STEADY_FLOW; the scan's track["flow"], {t: the head's
# move t -> t+1 as a similarity}, scan/cover.py flows), the camera's otherwise: carried by the camera
# alone, a fast real move counts as wobble -- jazz's dancer bobs a head's height every ~8 frames and her emoji lagged
# her face by up to 0.28 head sizes (median 0.07); along her flow only the detector's own wobble is averaged.
STEADY = True
STEADY_FLOW = True
STEADY_POS, STEADY_SIZE, STEADY_ENV, STEADY_KEEP = 1.5, 4.0, 2, 0.9  # gaussian sigmas (frames), envelope half-width, floor


def steady(per_frame, mats, flows=None):
    """In place. Each run of a person's boxes is averaged in one frame of reference (the run's first frame, every box
    carried back to it step by step); the steps are affine, so averaging there and carrying the result into frame t
    is the same as carrying every neighbour into frame t."""
    flows = flows if STEADY_FLOW else None
    one = {}  # tid -> [(t, i)] where the person has exactly one box
    for t, f in enumerate(per_frame):
        cnt = {}
        for i, x in enumerate(f):
            cnt.setdefault(x[0], []).append(i)
        for tid, ix in cnt.items():
            if len(ix) == 1:
                one.setdefault(tid, []).append((t, ix[0]))

    def gauss(sig):
        r = int(3 * sig); x = np.arange(-r, r + 1)
        return np.exp(-0.5 * (x / sig) ** 2)
    gp, gs = gauss(STEADY_POS), gauss(STEADY_SIZE)

    def smooth(v, g):  # normalised convolution (the run's ends average over what they have)
        r = len(g) // 2  # ("same" mode returns the kernel's length for a run shorter than it)
        return np.convolve(v, g)[r:r + len(v)] / np.convolve(np.ones_like(v), g)[r:r + len(v)]
    for tid, items in one.items():
        runs, cur = [], [items[0]]
        for a, b in zip(items, items[1:]):
            if b[0] == a[0] + 1:
                cur.append(b)
            else:
                runs.append(cur); cur = [b]
        runs.append(cur)
        for run in runs:
            if len(run) < 2:
                continue
            ts = np.array([t for t, _ in run])
            raw = np.array([per_frame[t][i][1] for t, i in run], float)
            G = [np.eye(3)]  # run's first frame -> frame t
            for (t, i), (_, i2) in zip(run, run[1:]):
                pa, pb = _base(per_frame[t][i][3]), _base(per_frame[t + 1][i2][3])
                f = (flows.get(pa) or {}).get(t, (flows.get(pa) or {}).get(str(t))) if flows and pa == pb else None
                S = np.vstack([mats[t + 1], [0, 0, 1]]) if t + 1 < len(mats) else np.eye(3)
                if f is not None:  # the camera's scale and turn, moved so the head lands where its flow took it
                    a, b, tx, ty = f
                    c = per_frame[t][i][1]
                    S[:2, 2] += np.array([a * c[0] - b * c[1] + tx, b * c[0] + a * c[1] + ty]) - S[:2] @ [c[0], c[1], 1]
                G.append(S @ G[-1])
            G = np.array(G)
            Gi = np.linalg.inv(G)
            k = np.sqrt(np.abs(np.linalg.det(G[:, :2, :2])))  # frame-t size per reference size
            p = np.einsum("nij,nj->ni", Gi, np.c_[raw[:, :2], np.ones(len(ts))])[:, :2]  # reference coords
            rs = raw[:, 2] / k
            env = np.lib.stride_tricks.sliding_window_view(np.pad(rs, STEADY_ENV, constant_values=-np.inf), 2 * STEADY_ENV + 1).max(1)
            sx, sy = smooth(p[:, 0], gp), smooth(p[:, 1], gp)
            ss = np.exp(smooth(np.log(env), gs))
            q = np.einsum("nij,nj->ni", G, np.c_[sx, sy, np.ones(len(ts))])[:, :2]  # back into each frame
            size = ss * k
            size = np.maximum(size, STEADY_KEEP * raw[:, 2] + 2 * np.abs(q - raw[:, :2]).max(1))
            for j, (t, i) in enumerate(run):
                tid_, _, kind, piece = per_frame[t][i]
                per_frame[t][i] = (tid_, [float(q[j, 0]), float(q[j, 1]), float(size[j])], kind, piece)


def _since(frames, t):
    """Frames from t to the nearest of the sorted evidence frames (a large number if there are none)."""
    import bisect
    if not frames:
        return 10 ** 6
    i = bisect.bisect_left(frames, t)
    return min(abs(frames[j] - t) for j in (i - 1, i) if 0 <= j < len(frames))


def shown_pieces(edits):
    """Pieces rendered unmasked: individually shown pieces plus the pieces LISTED in a Person with show=True.
    Only listed pieces: a piece that merely resolves to a shown person through an automatic re-ID link
    (e.g. one computed after the decision, or a wrong one) keeps that person's emoji but stays covered.
    Links may group appearances; they never extend consent."""
    out = set(edits.get("shown", []))
    for P in (edits.get("people") or {}).values():
        if P.get("show"):
            out |= {str(x) for x in P.get("pieces", [])}
    return out


def emoji_for(tid, overrides):
    if tid in overrides:
        return overrides[tid]
    h = sum((i + 1) * ord(ch) for i, ch in enumerate(str(tid).removeprefix("@")))  # a person "@x": its anchor box x's
    return EMOJI[h % len(EMOJI)]


STYLES = ("emoji", "blur", "pixelate", "solid", "portrait")  # portrait: one headshot on every head (portrait.py)


BLUR_STEPS = 24  # cover()'s blur: done at r / BLUR_STEPS pixels to one, for a face r pixels in radius


def cover(f, c, scale, style):
    """Non-emoji covers over a circle of diameter c[2]*scale: blur, pixelate, or solid (opaque, most robust:
    blur can sometimes be partly reversed)."""
    H, W = f.shape[:2]
    r = c[2] * scale / 2
    x1, y1, x2, y2 = max(int(c[0] - r), 0), max(int(c[1] - r), 0), min(int(c[0] + r) + 1, W), min(int(c[1] + r) + 1, H)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = f[y1:y2, x1:x2]
    xx, yy = np.arange(x1, x2), np.arange(y1, y2)[:, None]
    m = (xx - c[0]) ** 2 + (yy - c[1]) ** 2 <= r * r  # (the circle; rows and columns broadcast, not a full grid)
    if style == "blur":
        k = max(int(r) | 1, 3)
        # A kernel the face's radius wide, twice: its cost grows with the cube of the face's size (a 4K frame with
        # six faces took 340 ms). Blurred this much, nothing finer than ~r/BLUR_STEPS is left, so the same blur done on
        # a copy shrunk to that grain and enlarged again looks the same, at a few ms. (Faces smaller than that: as is.)
        d = r / BLUR_STEPS
        if d >= 2:
            h, w = roi.shape[:2]
            ks = max(int(k / d) | 1, 3)
            small = cv2.resize(roi, (max(round(w / d), 3), max(round(h / d), 3)), interpolation=cv2.INTER_AREA)
            new = cv2.resize(cv2.GaussianBlur(cv2.GaussianBlur(small, (ks, ks), 0), (ks, ks), 0), (w, h),
                             interpolation=cv2.INTER_LINEAR)
        else:
            new = cv2.GaussianBlur(cv2.GaussianBlur(roi, (k, k), 0), (k, k), 0)
    elif style == "pixelate":
        n = 5  # blocks across the face
        small = cv2.resize(roi, (n, max(int(n * roi.shape[0] / roi.shape[1]), 1)), interpolation=cv2.INTER_AREA)
        new = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:  # solid
        new = np.empty_like(roi); new[:] = (40, 40, 40)
    np.copyto(roi, new, where=m[..., None])  # (roi is a view of f)


def draw_order(items, t, occlusion=None):
    """The frame's covers in drawing order, far first so the nearest is on top: by depth (depth.Occlusion.nearness:
    the face's own depth around then) when a depth model is on, else by size (bigger = nearer)."""
    if occlusion is None:
        return sorted(items, key=lambda x: x[1][2])
    return sorted(items, key=lambda x: (occlusion.nearness(t, x[3] if len(x) > 3 else x[0], x[1]), x[1][2]))


def cover_picture(tid, piece, c, t, sprites, scale, overrides, style, head_pose=None, portrait=None):
    """(sprite, box) for one emoji/portrait cover, as draw_frame and gpu_draw both draw it: the person's emoji (or the
    portrait photo), eyes on their eyes, tilted and turned with the head when head_pose has this piece at t."""
    if style == "portrait":
        from mojimask import portrait as PT
        ch = portrait or PT.PORTRAIT_KEY  # everyone gets the photo; per-person picks don't apply
    else:
        ch = emoji_for(tid, overrides)
    if ch not in sprites:
        sprites[ch] = emoji_sprite(ch)
    sp, cp = sprites[ch], c
    if style == "emoji":  # eyes on the person's eyes (emoji_eyes.json)
        cp = eye_place(ch, c, scale)
    hp = (head_pose or {}).get(piece, {}).get(t)
    if hp is not None:  # tilt/turn with the head: a pre-warped, bigger canvas pasted the usual way
        from mojimask import head_pose as HP
        sp, off = HP.posed_sprite(sp, *HP.sprite_pose(ch, *hp))
        cp = HP.posed_box(cp, off)
    return sp, cp


def draw_frame(f, t, items, sprites, scale=1.25, overrides=None, debug=False, shown=None, light=None, style="emoji", head_pose=None, portrait=None, saturation=1.0,
               occlusion=None):
    """Composite all emoji for frame t onto image f (BGR, modified copy returned).
    saturation: the emoji's own colourfulness (lighting's Saturation slider) when there is no `light` to apply it
    (lighting off, or its estimate not ready yet) -- it's about the emoji, not the light, so it always applies.
    head_pose: optional {piece: {t: (roll_deg, yaw, side)}} (head_pose.track_angles) -- tilt/turn emoji with the
    head, mirroring profile emoji (head_pose.FACING) to face the way the head does.
    portrait: style "portrait"'s photo as a sprite key (portrait.key_for); None = the default photo.
    shown: track pieces whose person is whitelisted (real faces, deliberately left unmasked).
    light: optional lighting.FrameLighter -- shade each emoji to match the scene (else flat).
    occlusion: optional depth.Occlusion (MOJIMASK_DEPTH) -- the part of a cover behind something nearer shows the video;
    never where another cover is drawn.
    Emoji and portraits are drawn on the GPU (gpu_draw, Metal) when there is one; MOJIMASK_GPU=0 keeps the CPU drawing
    below, which stays the reference (and draws the other styles and the debug labels)."""
    shown = shown or set()
    overrides = overrides or {}
    if style in ("emoji", "portrait") and not debug:
        from mojimask import gpu_draw
        out = gpu_draw.draw_frame(f, t, items, sprites, scale, overrides, shown, light, style, head_pose, portrait,
                                  saturation, occlusion)
        if out is not None:
            return out
    orig = f.copy()
    f = f.copy()
    items = draw_order(items, t, occlusion)
    masked = [x for x in items if not (x[3:] and x[3] in shown)]
    near, cuts = occlusion.plan(t, masked, scale, f.shape) if occlusion is not None else (None, {})
    cut_of = {id(masked[i]): cut for i, cut in cuts.items()}  # depth.Occlusion.plan: which covers are cut
    drawn = [x[1] for x in masked]  # every cover drawn this frame
    for item in items:
        tid, c, kind, *rest = item
        if rest and rest[0] in shown:
            continue  # whitelisted: not masked
        if style not in ("emoji", "portrait"):
            cover(f, c, scale, style)
        else:
            sp, cp = cover_picture(tid, rest[0] if rest else tid, c, t, sprites, scale, overrides, style, head_pose,
                                   portrait)
            if light is not None:
                light.paste(f, sp, cp, scale, rest[0] if rest else tid, t)
            else:
                if saturation < 1:
                    from mojimask import lighting
                    sp = lighting.desat_sprite(sp, saturation)
                paste(f, sp, cp, scale)
        cut = cut_of.get(id(item))
        if cut is not None:
            fr = occlusion.front(c, scale, cut, orig, near)
            if fr is not None:
                x1, y1, x2, y2, a = fr
                yy, xx = np.mgrid[y1:y2, x1:x2]
                for o in drawn:  # never under another cover: a nearer person's face stays under their own emoji
                    if o is not c and abs(o[0] - c[0]) < (o[2] + c[2]) * scale and abs(o[1] - c[1]) < (o[2] + c[2]) * scale:
                        a = a * ((xx - o[0]) ** 2 + (yy - o[1]) ** 2 > (o[2] * scale / 2) ** 2)
                a = a[..., None]
                f[y1:y2, x1:x2] = (a * orig[y1:y2, x1:x2] + (1 - a) * f[y1:y2, x1:x2]).astype(np.uint8)
        if debug:
            cv2.putText(f, f"{tid}{'*' if kind == 'coast' else ''}", (int(c[0] - c[2] / 2), int(c[1] - c[2] / 2) - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
    return f


# The optional corner mark (Export's "Add a small ... mark"): the wordmark as brand.js sets it (about.BRAND: the name's
# case, typeface, weight, letter-spacing and the two parts' colours, as on the splash and in the app's bar), in the
# lower right of the OUTPUT picture (drawn after the format's crop, so it's never cut off), a little transparent, with
# a soft shadow so it reads on any footage. Its height follows the picture's short side.
MARK_SIZE, MARK_ALPHA = 0.028, 0.85  # text height / the short side; opacity
_MARKS = {}


def _rgba(hex_colour):
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (255,)


def _mark_sprite(px):
    """The wordmark at px tall, as a premultiplied (BGR, alpha) pair; cached by size."""
    if px not in _MARKS:
        from mojimask import about
        B = about.BRAND
        font = ImageFont.truetype(about.FONT_FILE, px)
        try:
            font.set_variation_by_axes([B["weight"]])
        except (OSError, AttributeError):  # (a FreeType without variable fonts: the default weight)
            pass
        word, track = about.NAME, B["tracking"] * px  # letter-spacing, as CSS: after every letter

        def at(i):  # where letter i starts: the shaped advance up to and including it, less its own (keeps kerning)
            return font.getlength(word[:i + 1]) - font.getlength(word[i]) + i * track
        x0, y0, x1, y1 = font.getbbox(word)
        x1 = max(x1, at(len(word) - 1) + font.getlength(word[-1]))
        pad = max(2, px // 4)
        W, H = int(x1 - min(x0, 0)) + 2 * pad, y1 - y0 + 2 * pad
        text = Image.new("RGBA", (W, H))
        d = ImageDraw.Draw(text)
        colours = [_rgba(B["first"])] * B["split"] + [_rgba(B["second"])] * (len(word) - B["split"])
        for i, ch in enumerate(word):
            d.text((pad - min(x0, 0) + at(i), pad - y0), ch, font=font, fill=colours[i])
        a = np.asarray(text)[:, :, 3].astype(np.float32) / 255
        dy = max(1, px // 14)  # (a shadow a little below the letters, so white text still reads on white)
        shadow = cv2.GaussianBlur(np.roll(a, dy, axis=0), (0, 0), max(1.0, px / 12)) * 0.8
        rgb = np.asarray(text)[:, :, 2::-1].astype(np.float32)  # BGR
        alpha = (a + shadow * (1 - a)) * MARK_ALPHA
        _MARKS[px] = (rgb * a[:, :, None] * MARK_ALPHA, alpha)  # (the shadow is black: it adds alpha, no colour)
    return _MARKS[px]


def watermark(img):
    """img (BGR uint8, the output picture) with the corner mark drawn in place; returns it."""
    H, W = img.shape[:2]
    px = max(10, round(min(H, W) * MARK_SIZE))
    col, alpha = _mark_sprite(px)
    h, w = alpha.shape
    m = round(px * 0.8)  # margin from the edges
    y, x = H - h - m, W - w - m
    if y < 0 or x < 0:
        return img
    roi = img[y:y + h, x:x + w].astype(np.float32)
    img[y:y + h, x:x + w] = np.clip(col + roi * (1 - alpha[:, :, None]), 0, 255).astype(np.uint8)
    return img


def render(video, per_frame, out, audio=None, scale=1.25, overrides=None, debug=False, progress=None, shown=None, light=None, style="emoji", output=None, head_pose=None, meta=None, portrait=None,
           saturation=1.0, occlusion=None, mark=False):
    """Export (a photo: its one frame, saved as a photo): covers drawn on every frame of video_io's timeline (the same frames the scan saw, here
    colour-managed for display), then fitted to the output format (formats.settings: size, crop/fit,
    frame-rate step, trim, bitrate), encoded to H.264 by the OS (AVFoundation), and muxed with the
    original's audio for the same span. No container metadata (date, GPS, device) is carried over.
    light: optional lighting.FrameLighter. mark: the corner mark (watermark) on every output picture."""
    import shutil
    import tempfile
    from mojimask import formats
    from mojimask import video_io
    overrides = overrides or {}
    sprites = {}
    meta = meta or video_io.probe(video)  # the project's timeline (review_server passes it)
    o = output or formats.settings(meta, "original", frames=len(per_frame))
    if meta.get("image"):  # a photo: its one frame, saved as a photo (JPEG or PNG by out's name), with no metadata
        f = draw_frame(next(video_io.frames(video, meta, color="display")), 0, per_frame[0], sprites, scale, overrides,
                       debug, shown, light, style, head_pose, portrait, saturation, occlusion)
        f = formats.fit_frame(f, o["width"], o["height"], o["fit"])
        video_io.write_image(out, watermark(f) if mark else f)
        if progress:
            progress(1.0)
        return
    s, e = o["start"], min(o["end"], len(per_frame))
    tmpdir = tempfile.mkdtemp(prefix="mojimask-")
    try:
        tmp = os.path.join(tmpdir, "video.mp4")
        w = video_io.writer(tmp, o["width"], o["height"], o["fps"], o["bitrate"])
        try:
            for t, f in enumerate(video_io.frames(video, meta, color="display")):
                if t >= e:
                    break
                if t < s or (t - s) % o["step"]:
                    continue
                f = draw_frame(f, t, per_frame[t], sprites, scale, overrides, debug, shown, light, style, head_pose,
                               portrait, saturation, occlusion)
                f = formats.fit_frame(f, o["width"], o["height"], o["fit"])
                w.write(watermark(f) if mark else f)
                if progress and t % 15 == 0:
                    progress(0.97 * (t - s) / max(1, e - s))
        except BaseException:
            try:
                w.close()  # stop the encoder (its own error, if any, would hide the real one)
            except Exception:  # noqa: BLE001
                pass
            raise
        w.close()
        video_io.mux(tmp, audio or video, out, start=s / meta["fps"])
        if progress:
            progress(1.0)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def add_args(ap):
    """The editor's (review_server's) drawing options. app_server always passes --dets (pipeline.CORE_DETS, the
    scan's detectors); the default is the same list."""
    from mojimask import pipeline  # here, not at the top: pipeline imports track inside its steps
    ap.add_argument("--dets", nargs="+", default=list(pipeline.CORE_DETS))
    ap.add_argument("--scale", type=float, default=1.25, help="emoji size vs head box")
