"""Behind someone: a held cover (a person's box on a frame where their face wasn't detected) that sits on a nearer
person's body is a face hidden behind them -- not covered there, so no emoji is drawn over whoever is in front.

jazz (2026-11-12): the dancer in front of the band hid seated faces for a second at a time; their covers were held
in place (cover.py follows across gaps of up to attach.MAX_GAP frames) and drew their emoji on her dress and arms.

A frame is behind when
  - the box isn't detected there (only held covers are ever dropped: a face that shows is a face);
  - no weak face detection is on it either (low: everything any detector half-believes, >= track.WEAK_DET -- a face that's
    visible but missed by the main detectors usually leaves one; jazz frame 92's seated women did);
  - FRAC of it lies on another skeleton's body: its torso (shoulders to hips, a little wider), arms and legs (as
    LIMB-wide capsules) or head -- never its own skeleton (pose_util.pose_for_head);
  - and that person is in front: clearly nearer the camera (shoulders >= IN_FRONT x the hidden face's size;
    shoulder width is about one face size, median 1.08 on jazz, 0.9 on the party clips), or the hidden face has lost
    its own skeleton too (jazz frame 115: the dancer back among the band, their size, both faces behind her
    skeleton-less). Either cue alone dropped visible faces: on-body alone took seated neighbours' arms (369 of 822
    held frames on jazz); no-skeleton without the weak-detection check took two seated women whose skeletons the
    pose detector also missed.
Runs: a kept stretch of <= CLOSE frames between two behind frames is behind too, so an emoji doesn't blink back
for a frame or two while the person is still hidden.
Measured on held frames: jazz 274 of 822 behind, party clips a / b / c 18 / 44 / 22, IMG_1043 148 of 4021; by eye
the party ones are torsos, hair and dark corners, no face.
"""
import cv2
import numpy as np

from mojimask import depth as DP  # noqa: E402
from mojimask import pose_util as PU  # noqa: E402
from mojimask import track as T  # noqa: E402

FRAC, IN_FRONT, LIMB, CLOSE = 0.5, 1.4, 0.2, 2
# With a depth model on (MOJIMASK_DEPTH, depth.py) depth only adds: "in front" also holds when the box reads `front`
# nearer than the face's own depth around then (depth.face_at; Depth.near units: the frame's depth spread), and a box reading
# `alone` nearer is behind someone even with no skeleton over it (the pose detector missed the arm). Both are per
# model (depth.MODELS). Replacing the shoulder test with depth put emoji back on jazz's dancer where depth missed her
# dress (8@83-91, 16@116-121). tools/bench/depth_bench.py: with Small, visible faces read >= 0.5 nearer in
# 0.2-1.1% of frames.
LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]
_GRID = np.linspace(0.1, 0.9, 7)


def on_body(p, pts):
    """Which of pts (N x 2) fall on skeleton p's body (torso, limbs, head)."""
    k = PU.kps(p)
    v = k[:, 2] > PU.VIS
    if not (v[5] and v[6]):
        return np.zeros(len(pts), bool)
    sh = k[[5, 6], :2]
    sw = float(np.linalg.norm(sh[0] - sh[1]))
    if sw < 6:
        return np.zeros(len(pts), bool)
    hips = k[[11, 12], :2] if v[11] and v[12] else sh + [0, 1.6 * sw]
    poly = np.array([sh[0], sh[1], hips[1], hips[0]])
    poly = (poly.mean(0) + (poly - poly.mean(0)) * 1.15).astype(np.float32)
    hit = np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for x, y in pts])
    for a, b in LIMBS:
        if v[a] and v[b]:
            A, B = k[a, :2], k[b, :2]
            d = B - A
            t = np.clip(((pts - A) @ d) / max(float(d @ d), 1e-6), 0, 1)
            hit |= np.hypot(*(pts - (A + t[:, None] * d)).T) < LIMB * sw
    h = PU.head(p)
    if h:
        hit |= np.hypot(pts[:, 0] - h[0], pts[:, 1] - h[1]) < 0.55 * h[2]
    return hit


def face_there(c, low):
    """A weak detection (low: one frame of T.fused_frames at T.WEAK_DET) on box c: about its place and size."""
    for b, s, srcs in low:
        bc = T.to_cs(b)
        if np.hypot(bc[0] - c[0], bc[1] - c[1]) < 0.5 * c[2] and 0.5 < bc[2] / c[2] < 2:
            return True
    return False


def is_behind(c, entries, low=(), nearer=None, *, front=None, alone=None):
    """Box c = [cx, cy, s] on a frame with skeletons `entries` (pose_full) and weak detections `low`: behind
    someone in front of it? nearer: how much nearer the box reads than the face does around then (depth on), else None;
    with it, front and alone are the depth model's margins (depth.MODELS)."""
    if face_there(c, low):
        return False
    if nearer is not None and nearer >= alone:
        return True
    own = PU.pose_for_head(entries, c)
    x0, y0, x1, y1 = T.from_cs(c)
    pts = np.array([(x0 + a * (x1 - x0), y0 + b * (y1 - y0)) for a in _GRID for b in _GRID])
    for p in entries:
        if p is own or p[4] < 0.3:
            continue
        k = PU.kps(p)
        if not (k[5, 2] > PU.VIS and k[6, 2] > PU.VIS):
            continue
        if (own is not None and np.linalg.norm(k[5, :2] - k[6, :2]) < IN_FRONT * c[2]
                and (nearer is None or nearer < front)):
            continue  # not clearly nearer (by shoulders or by depth), and the face still has its own skeleton
        if on_body(p, pts).mean() >= FRAC:
            return True
    return False


def behind_frames(cover, obs, poses, low, depth=None):
    """The held frames of one person's cover ({t: box}; obs: their detected frames) that are behind someone.
    low: weak detections per frame (T.fused_frames at T.WEAK_DET). depth: a depth.Depth (MOJIMASK_DEPTH on) or None."""
    held = sorted(t for t in cover if t not in obs)
    face = None
    if depth is not None and obs:
        ts = np.array(sorted(obs))
        v = np.array([depth.near(t, obs[t]) for t in ts])
        face = (ts[np.isfinite(v)], v[np.isfinite(v)])
    margins = {"front": depth.front, "alone": depth.alone} if depth is not None else {}

    def nearer(t):  # this box's reading over the face's own depth around then (depth.face_at)
        if face is None or not len(face[1]):
            return None
        v = depth.near(t, cover[t]) - DP.face_at(*face, t)
        return v if np.isfinite(v) else None
    out = set()
    for t in held:
        n = nearer(t)
        if t >= len(poses) and n is None:
            continue  # no skeletons and no depth reading: nothing to go on
        if is_behind(cover[t], poses[t] if t < len(poses) else [], low[t] if t < len(low) else (), n, **margins):
            out.add(t)
    hs = set(held)
    for t in sorted(out):  # close short kept stretches between two behind frames (no blink back)
        for gap in range(1, CLOSE + 1):
            if t + gap + 1 in out and all(t + i in hs for i in range(1, gap + 1)):
                out.update(t + i for i in range(1, gap + 1))
    return out
