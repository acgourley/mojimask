"""The scan's follower: follow a head through frames without (usable) detections -- the checked rule. join.py
uses it to bridge gaps between tracklets, cover.py to carry covers across gaps and past a track's ends, blips.py
to check a blip.

Per frame, two candidate positions: where corner flow moves the box ("flow"), and where the camera alone
moves it ("camera", holding still in the room).
  1. a face found at the flow position   -> follow, re-centred on that face
  2. else a face at the camera position  -> stay, re-centred on that face
  3. else the patch at the flow position still looks like the face did when last seen (NCC >= LOOKS)
                                          -> follow
  4. else                                 -> hold (camera); heading out of the picture -> keep its own velocity
Stops: the box is mostly outside the picture ("out"), or `limit` frames; with lost=True (covers, not joins) also where
the face is lost ("lost", see face_motion below).
"A face found" = RetinaFace (reid.landmarks) finds a face (score >= FACE) whose centre is within NEAR head sizes of the box centre;
re-centring moves the box onto that face's head box (track.face_to_head), cancelling drift.
Evidence for this rule: jazz labels, and six clips without labels.

    follow(clip, start_t, c0, direction=+1, limit=24, meet=None, lost=False, seen=None) -> Path
clip: Clip (frames, mats). meet(u, c) -> an id if box c at frame u lands on something to stop at (a detection
that starts or ends another tracklet), else None. lost: also stop where the face is lost. seen(u, c) -> true if a
(weak) detection at frame u shows the face is still there, so it isn't lost.
"""
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from mojimask.scan import faces as FA  # noqa: E402
from mojimask import reid  # noqa: E402
from mojimask import track as T  # noqa: E402

FACE, LOOKS, OUT, TS = 0.5, 0.5, 0.3, 32
NEAR = FA.NEAR  # the same crop and nearest-face search as face measuring (reid.face_crop, faces.nearest)
CORE = (0.2, 0.8, 0.15, 0.85)


class Clip:
    def __init__(self, proj):
        self.proj = proj
        self.mats = np.load(f"{proj}/mats_v2.npy")
        self._c, self._g, self._trk = {}, {}, {}
        import threading
        self._lock = threading.Lock()  # follows run in threads (join.py): one frame cache, read and filled safely
        im = self.color(0)
        self.H, self.W = im.shape[:2]
        self.n = len([f for f in os.listdir(f"{proj}/frames") if f.endswith(".jpg")])

    def color(self, t):
        with self._lock:
            im = self._c.get(t)
        if im is None:
            im = cv2.imread(f"{self.proj}/frames/{t:05d}.jpg")
            with self._lock:
                if len(self._c) > 256:
                    self._c.pop(next(iter(self._c)))
                self._c[t] = im
        return im

    def gray(self, t):
        with self._lock:
            g = self._g.get(t)
        if g is None:
            g = cv2.cvtColor(self.color(t), cv2.COLOR_BGR2GRAY)
            with self._lock:
                if len(self._g) > 256:
                    self._g.pop(next(iter(self._g)))
                self._g[t] = g
        return g

    def trackable(self, a, b):
        """Share of corners over the whole picture that track a -> b both ways: low in a whip pan or a blurred
        frame (party_clip_c's pan, frames 180-238: 0.00-0.48), never under 0.64 on jazz's steady shot."""
        key = (min(a, b), max(a, b))
        with self._lock:
            v = self._trk.get(key)
        if v is None:
            g0, g1 = self.gray(key[0]), self.gray(key[1])
            p0 = cv2.goodFeaturesToTrack(g0, 400, 0.01, 8)
            if p0 is None or len(p0) < 8:
                v = 0.0
            else:
                _, ok = T.lk_fb(g0, g1, p0.astype(np.float32))
                v = float(ok.mean())
            with self._lock:
                self._trk[key] = v
        return v

    def camera(self, a, b):
        """Affine taking frame a coords to frame b coords (either order)."""
        if b >= a:
            return T.compose(self.mats, a, b)
        M = np.vstack([T.compose(self.mats, b, a), [0, 0, 1]])
        return np.linalg.inv(M)[:2]


# The follow check only asks "is a face here": it runs on the neural chip, which the graphics chip couldn't keep up
# with (SCRFD, 16,000 checks on jazz: 32-39 s there, 21-22 s here). Since 2026-11-14 the landmarks are RetinaFace's
# (reid.landmarks, 192 px), which runs on the neural chip for face measuring too.
FACE_NEAR_ENGINE = os.environ.get("FACE_NEAR_ENGINE", "ane")


def face_near(img, c):
    """(score, head box [cx, cy, s]) of the most confident face reid.landmarks finds within NEAR head sizes of box c's
    centre, else (0, None)."""
    cr = reid.face_crop(img, c)
    if cr is None:
        return 0.0, None
    crop, k, x0, y0 = cr
    boxes, _ = reid.landmarks(crop, engine=FACE_NEAR_ENGINE)
    i = FA.nearest(boxes, k, x0, y0, c)
    if i is None:
        return 0.0, None
    b = boxes[i]
    fb = [b[0] / k + x0, b[1] / k + y0, b[2] / k + x0, b[3] / k + y0]
    return float(b[4]), T.to_cs(T.face_to_head(fb))


_y, _x = np.mgrid[0:TS, 0:TS]  # ncc's weights: an oval inside the patch
_OV = ((((_x - (TS - 1) / 2) / (TS * 0.42)) ** 2 + ((_y - (TS - 1) / 2) / (TS * 0.48)) ** 2) <= 1).astype(np.float32)


def patch(g, c):
    s = c[2]
    M = np.array([[TS / s, 0, TS / 2 - c[0] * TS / s], [0, TS / s, TS / 2 - c[1] * TS / s]], np.float32)
    return cv2.warpAffine(g, M, (TS, TS), flags=cv2.INTER_LINEAR).astype(np.float32)


def ncc(a, b):
    w = _OV
    am, bm = (a * w).sum() / w.sum(), (b * w).sum() / w.sum()
    da, db = (a - am) * w, (b - bm) * w
    den = np.sqrt((da * da).sum() * (db * db).sum())
    return float((da * db).sum() / den) if den > 1e-6 else 0.0


# ---- the face's own flow: how the head moved between two frames, and whether it can still be followed
# Corners in the head box tracked both ways (track.lk_fb), then one similarity (move, scale, turn) fitted to them.
# On a visible face nearly all corners agree (jazz, the dancer bouncing: 0.6-1.0 of them; the move it gives lands
# within 0.01-0.05 head sizes of her next detection, while the detections themselves jump up to 0.2). When someone
# passes in front, the corners go with them or fail: under LOST_INL agree. That's "lost" -- but only where the rest of
# the picture tracks fine (Clip.trackable >= STEADY_FRAME; in a whip pan everything fails and a smeared face can still
# be there) and the box, with a head's width around it, is inside the picture (corners at the edge fail too: party_clip_b
# box 120, a face entering at the right edge, was lost while in plain view). Few corners at all ("no texture": a dark
# or flat face) says nothing.
FLOW_CORE = (0.15, 0.85, 0.1, 0.9)
LOST_INL, STEADY_FRAME, LOST_INSIDE = 0.3, 0.6, 0.9


def face_motion(g0, g1, c):
    """(M, kind): M the 2x3 similarity taking box c's head in g0 to g1 (None unless kind is "ok"); kind "ok", "no
    texture" (too few corners to say), "lost" (corners didn't track) or "incoherent" (they don't agree on one move)."""
    H, W = g0.shape[:2]
    x1, y1, x2, y2 = T.from_cs(c)
    w, h = x2 - x1, y2 - y1
    x1, x2, y1, y2 = x1 + FLOW_CORE[0] * w, x1 + FLOW_CORE[1] * w, y1 + FLOW_CORE[2] * h, y1 + FLOW_CORE[3] * h
    x1, y1, x2, y2 = [int(round(v)) for v in (max(x1, 0), max(y1, 0), min(x2, W - 1), min(y2, H - 1))]
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None, "no texture"
    p0 = cv2.goodFeaturesToTrack(g0[y1:y2, x1:x2], 80, 0.003, 2)
    if p0 is None or len(p0) < 6:
        return None, "no texture"
    p0 = (p0 + np.array([x1, y1], np.float32)).astype(np.float32)
    p1, good = T.lk_fb(g0, g1, p0)
    if good.sum() < 5:
        return None, "lost"
    a, b = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
    M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=max(1.0, 0.04 * c[2]))
    if M is None:
        return None, "lost"
    if inl.sum() / len(p0) < LOST_INL:
        return None, "incoherent"
    return M, "ok"


@dataclass
class Path:
    boxes: dict = field(default_factory=dict)   # {t: [cx, cy, s]}
    how: dict = field(default_factory=dict)     # {t: "face at flow" | "face at camera" | "looks the same" | "hold" | "exit"}
    why: str = "limit"                          # why it stopped: "limit" | "out" | "met" | "clip end" | "lost"
    met: object = None                          # what meet() returned, and when: (u, id)


def follow(clip, start_t, c0, direction=1, limit=24, meet=None, lost=False, seen=None):
    """lost=True: stop where the face is lost (face_motion), unless a face is found there or seen(u, box) -- a weak
    detection on it -- says it's still showing."""
    c = np.array(c0, float)
    anchor = patch(clip.gray(start_t), c)
    v = np.zeros(2)
    p = Path()
    u = start_t
    for _ in range(limit):
        u2 = u + direction
        if not 0 <= u2 < clip.n:
            p.why = "clip end"
            break
        M = clip.camera(u, u2)
        cam = T.warp_cs(c, M)
        g0, g1 = clip.gray(u), clip.gray(u2)
        m = T._flow_measure(g0, g1, c, CORE)
        fl = np.array([c[0] + m[0][0], c[1] + m[0][1], cam[2]]) if m is not None else cam.copy()
        img = clip.color(u2)
        s, head = face_near(img, fl)
        if s >= FACE:
            new, how = head, "face at flow"
        else:
            s, head = face_near(img, cam)
            if s >= FACE:
                new, how = head, "face at camera"
            elif (lost and face_motion(g0, g1, c)[1] in ("lost", "incoherent")
                  and min(T.inside_frac([c[0], c[1], 2 * c[2]], clip.W, clip.H),
                          T.inside_frac([cam[0], cam[1], 2 * cam[2]], clip.W, clip.H)) >= LOST_INSIDE
                  and clip.trackable(u, u2) >= STEADY_FRAME
                  and not (seen and (seen(u2, cam) or seen(u2, fl)))):
                p.why = "lost"
                break
            elif ncc(patch(g1, fl), anchor) >= LOOKS:
                new, how = fl, "looks the same"
            elif T._heading_out(cam, v, clip.W, clip.H):
                new, how = cam.copy(), "exit"
                new[:2] += v
            else:
                new, how = cam, "hold"
        if how.startswith("face"):
            anchor = patch(g1, new)
        if how != "hold":
            v = 0.5 * v + 0.5 * (new[:2] - cam[:2])
        c = np.array(new, float)
        u = u2
        p.boxes[u], p.how[u] = c.copy(), how
        if T.inside_frac(c, clip.W, clip.H) < OUT:
            p.why = "out"
            break
        if meet is not None:
            hit = meet(u, c)
            if hit is not None:
                p.why, p.met = "met", (u, hit)
                break
    return p
