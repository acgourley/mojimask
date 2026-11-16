"""Camera motion v2: background-only homographies with unreliable frames filled from neighbours.

Handheld phone motion is mostly rotation, which a homography models exactly at any scene depth.
People are excluded from the fit (pose person boxes), and frames whose fit is unreliable (strobe,
blur, a hand over the lens) borrow the motion of their reliable neighbours instead of assuming
"no motion".

    python motion.py out/review/frames out/dets/pose_full.json out/motion_h.npy
"""
import json
import sys

import cv2
import numpy as np


def person_mask(shape, persons, pad=0.08, min_score=0.25):
    m = np.full(shape, 255, np.uint8)
    H, W = shape
    for p in persons:
        x1, y1, x2, y2, s = p[:5]
        if s < min_score:
            continue
        w, h = x2 - x1, y2 - y1
        m[max(int(y1 - pad * h), 0):min(int(y2 + pad * h), H), max(int(x1 - pad * w), 0):min(int(x2 + pad * w), W)] = 0
    return m


def estimate(frames_dir, n, persons=None, scale=0.5):
    """Returns (Hs, reliable): Hs[t] maps frame t-1 -> t (3x3, full-res pixels); Hs[0] = I."""
    S = np.diag([scale, scale, 1.0])
    Si = np.linalg.inv(S)
    Hs, rel = [np.eye(3)], [True]
    prev = None
    for t in range(n):
        f = cv2.imread(f"{frames_dir}/{t:05d}.jpg")
        g = cv2.cvtColor(cv2.resize(f, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            g0, m0 = prev
            Ht, ok = np.eye(3), False
            p0 = cv2.goodFeaturesToTrack(g0, 500, 0.01, 5, mask=m0)
            if p0 is not None and len(p0) >= 15:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g, p0, None)
                good = st.flatten() == 1
                if good.sum() >= 15:
                    h, inl = cv2.findHomography(p0[good], p1[good], cv2.RANSAC, 1.5)
                    if h is not None and inl is not None and inl.mean() > 0.5 and inl.sum() >= 12:
                        Ht = Si @ h @ S
                        # sanity: no more than a few % zoom or strong perspective per frame
                        J = Ht[:2, :2] / Ht[2, 2]
                        sc = np.sqrt(abs(np.linalg.det(J)))
                        ok = 0.95 < sc < 1.05 and np.abs(Ht[2, :2]).max() < 1e-3
            Hs.append(Ht / Ht[2, 2])
            rel.append(ok)
        m = person_mask(g.shape, [[v * scale for v in p[:4]] + [p[4]] for p in persons[t]]) if persons else None
        prev = (g, m)
    Hs, rel = np.array(Hs), np.array(rel)
    # fill unreliable frames from the nearest reliable neighbours (average of previous and next)
    idx = np.where(rel)[0]
    for t in np.where(~rel)[0]:
        if t == 0 or not len(idx):
            continue
        before, after = idx[idx < t], idx[idx > t]
        cands = ([Hs[before[-1]]] if len(before) else []) + ([Hs[after[0]]] if len(after) else [])
        Hs[t] = np.mean(cands, 0) if cands else np.eye(3)
        Hs[t] /= Hs[t][2, 2]
    return Hs, rel


# ------------------------------------------------------------------ helpers for [cx, cy, s] boxes
def as3(M):
    M = np.asarray(M, float)
    return M if M.shape == (3, 3) else np.vstack([M, [0, 0, 1]])


def warp_cs(c, M):
    """Move a square box (centre, size) by a 2x3 affine or 3x3 homography (size by local scale)."""
    M = as3(M)
    p = M @ np.array([c[0], c[1], 1.0])
    w = p[2]
    x, y = p[0] / w, p[1] / w
    J = (M[:2, :2] * w - np.outer(p[:2], M[2, :2])) / (w * w)  # Jacobian of the projective map
    s = np.sqrt(abs(np.linalg.det(J)))
    return np.array([x, y, c[2] * s])


def compose(Hs, a, b):
    """Map frame a -> frame b (a <= b)."""
    M = np.eye(3)
    for t in range(a + 1, b + 1):
        M = as3(Hs[t]) @ M
    return M


def carry(c, Hs, a, b):
    """Carry box c from frame a to frame b (either direction) by camera motion only."""
    if b >= a:
        return warp_cs(c, compose(Hs, a, b))
    return warp_cs(c, np.linalg.inv(compose(Hs, b, a)))


if __name__ == "__main__":
    frames_dir, pose_path, out = sys.argv[1:4]
    persons = json.load(open(pose_path))["frames"]
    Hs, rel = estimate(frames_dir, len(persons), persons)
    np.save(out, Hs)
    np.save(out.replace(".npy", "_reliable.npy"), rel)
    print(f"{len(Hs)} frames, {(~rel).sum()} unreliable frames filled from neighbours")
