"""Motion evidence for the native scan, from Python, until mm-motion supplies it (mm-engine's evidence::PyMotion).

    python -c "$(cat motion_server.py)" PROJECT

One JSON request a line on stdin, one JSON answer a line on stdout, exactly as Python's scan computes them
(follow.Clip on PROJECT/frames and mats_v2.npy):
  {"fn": "flow", "a": [t0, t1, c]}          track._flow_measure(gray t0, gray t1, c, follow.CORE) -> [dx, dy, n, spread] | null
  {"fn": "motion", "a": [t0, t1, c]}        follow.face_motion(gray t0, gray t1, c) -> [M (6 floats) | null, kind]
  {"fn": "trackable", "a": [a, b]}          Clip.trackable(a, b)
  {"fn": "looks", "a": [t, c, ta, ca]}      follow.ncc(patch(gray t, c), patch(gray ta, ca))
  {"fn": "flow_step", "a": [t0, t1, c]}     review_server.flow_step(gray t0, gray t1, c) -> [cx, cy, s] | null (frames read
                                            as review_server reads them: cv2.IMREAD_GRAYSCALE)
  {"fn": "quit"}
"""
import json
import sys

import numpy as np

from mojimask import track as T
from mojimask.scan import follow as FO

import cv2

clip = FO.Clip(sys.argv[1])
_G = {}


def _gray(t):  # review_server.gray
    g = _G.get(t)
    if g is None:
        if len(_G) > 400:
            _G.pop(next(iter(_G)))
        g = _G[t] = cv2.imread(f"{sys.argv[1]}/frames/{t:05d}.jpg", cv2.IMREAD_GRAYSCALE)
    return g


def flow_step(g0, g1, c):  # review_server.flow_step, verbatim
    H, W = g0.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in T.from_cs(c)]
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W - 1), min(y2, H - 1)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    p0 = T.box_corners(g0, x1, y1, x2, y2)
    if p0 is None or len(p0) < 6:
        return None
    p1, good = T.lk_fb(g0, g1, p0)
    if good.sum() < 5:
        return None
    a, b = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
    d = np.median(b - a, axis=0)
    i, j = np.triu_indices(len(a), 1)
    da, db = np.linalg.norm(a[i] - a[j], axis=1), np.linalg.norm(b[i] - b[j], axis=1)
    ok = da > 2
    sc = float(np.clip(np.median(db[ok] / da[ok]), 0.9, 1.1)) if ok.sum() > 3 else 1.0
    return [c[0] + float(d[0]), c[1] + float(d[1]), c[2] * sc]


print("ready", flush=True)
for line in sys.stdin:
    q = json.loads(line)
    f, a = q["fn"], q.get("a")
    if f == "quit":
        break
    if f == "flow":
        m = T._flow_measure(clip.gray(a[0]), clip.gray(a[1]), np.array(a[2], float), FO.CORE)
        out = None if m is None else [float(m[0][0]), float(m[0][1]), int(m[1]), float(m[2])]
    elif f == "motion":
        M, kind = FO.face_motion(clip.gray(a[0]), clip.gray(a[1]), np.array(a[2], float))
        out = [None if M is None else [float(v) for v in np.asarray(M, float).ravel()], kind]
    elif f == "trackable":
        out = float(clip.trackable(a[0], a[1]))
    elif f == "looks":
        out = float(FO.ncc(FO.patch(clip.gray(a[0]), np.array(a[1], float)), FO.patch(clip.gray(a[2]), np.array(a[3], float))))
    elif f == "flow_step":
        out = flow_step(_gray(a[0]), _gray(a[1]), [float(v) for v in a[2]])
    else:
        raise ValueError(f)
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()
