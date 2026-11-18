"""Shared bits of the motion tools: paths, the Python under test, our library through ctypes.

Only public footage: test_clips/jazz.mp4 and jazz_4k.mp4 (a public-domain film), and files under out/.
"""
import ctypes
import os
import subprocess
import sys


# the main checkout (a worktree has no venv or models of its own): the parent of git's common directory
MAIN_CHECKOUT = os.path.dirname(subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse",
                                                "--path-format=absolute", "--git-common-dir"],
                                               capture_output=True, text=True).stdout.strip())

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(NATIVE)
MAIN = MAIN_CHECKOUT  # the main checkout: its venv and models, read only
PY = os.path.join(MAIN, ".venv", "bin", "python")
OUT = os.path.join(ROOT, "out", "native", "motion")
REL = os.path.join(NATIVE, "target", "release")
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

CLIPS = {"jazz": os.path.join(MAIN, "test_clips", "jazz.mp4"), "jazz_4k": os.path.join(MAIN, "test_clips", "jazz_4k.mp4")}


def frames_dir(clip):
    """Frames decoded by Python's video_io (the scan's own decode), as PNGs under out/native/motion/frames/<clip>."""
    d = os.path.join(OUT, "frames", clip)
    if not os.path.exists(os.path.join(d, "00000.png")):
        import cv2
        from mojimask import video_io
        os.makedirs(d, exist_ok=True)
        meta = video_io.probe(CLIPS[clip])
        for i, f in enumerate(video_io.frames(CLIPS[clip], meta)):
            cv2.imwrite(os.path.join(d, f"{i:05d}.png"), f, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return d


def python_camera_pair(f0, f1):
    """Python's track.camera_motion on exactly two BGR frames (video_io.frames swapped for the pair)."""
    from mojimask import track as T
    from mojimask import video_io
    orig = video_io.frames
    video_io.frames = lambda video, meta=None, color="detect": iter([f0, f1])
    try:
        return np.array(T.camera_motion("pair", meta={}))[1]
    finally:
        video_io.frames = orig


def python_camera_pair_raw(f0, f1, scale=0.5):
    """The same cv2 calls without the guards (no identity fallback, no scale clip): what the fit itself found, or
    None where it found nothing."""
    import cv2
    g0 = cv2.cvtColor(cv2.resize(f0, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(cv2.resize(f1, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(g0, 400, 0.01, 6)
    if p0 is None or len(p0) <= 10:
        return None, 0.0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None)
    ok = st.flatten() == 1
    if ok.sum() <= 10:
        return None, 0.0
    m, inl = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if m is None:
        return None, 0.0
    m = m.copy()
    m[:, 2] /= scale
    return m, float(inl.mean())


class Lib:
    """libmm_motion through ctypes (crates/mm-motion/src/ffi.rs). Images: contiguous uint8 grey (H, W)."""

    def __init__(self, path=None):
        path = path or os.path.join(REL, "libmm_motion.dylib")
        L = ctypes.CDLL(path)
        u8p, dp = ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_double)
        L.mm_motion_camera_pair.argtypes = [u8p, u8p, ctypes.c_int, ctypes.c_int, dp, dp]
        L.mm_motion_camera_pair.restype = ctypes.c_int
        L.mm_motion_head.argtypes = [u8p, u8p, ctypes.c_int, ctypes.c_int, dp, dp, dp]
        L.mm_motion_head.restype = ctypes.c_int
        L.mm_motion_trackable.argtypes = [u8p, u8p, ctypes.c_int, ctypes.c_int]
        L.mm_motion_trackable.restype = ctypes.c_double
        self.L = L

    @staticmethod
    def _u8(a):
        a = np.ascontiguousarray(a, dtype=np.uint8)
        return a, a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    @staticmethod
    def _d(a):
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def camera_pair(self, g0, g1):
        """-> (raw fit 2x3, production 2x3, info dict) for grey frames g0 -> g1."""
        a, pa = self._u8(g0)
        b, pb = self._u8(g1)
        m = np.zeros(12)
        info = np.zeros(8)
        ok = self.L.mm_motion_camera_pair(pa, pb, a.shape[1], a.shape[0], self._d(m), self._d(info))
        return m[:6].reshape(2, 3), m[6:].reshape(2, 3), {"ok": bool(ok), "inlier_frac": info[0], "blocks": int(info[1]),
                                                            "inliers": int(info[2]), "sigma": info[3]}

    KINDS = {0: "ok", 1: "no texture", 2: "lost", 3: "incoherent"}

    def head(self, g0, g1, c, core):
        """-> (kind, M 2x3 or None, flow d (2,) or None, n_good, spread, extra dict) for head box c=[cx, cy, s]."""
        a, pa = self._u8(g0)
        b, pb = self._u8(g1)
        cc = np.array([c[0], c[1], c[2], core[0], core[1], core[2], core[3]], np.float64)
        m = np.zeros(6)
        out = np.zeros(8)
        k = self.L.mm_motion_head(pa, pb, a.shape[1], a.shape[0], self._d(cc), self._d(m), self._d(out))
        kind = self.KINDS[k]
        d = None if np.isnan(out[0]) else out[:2].copy()
        return kind, (m.reshape(2, 3).copy() if kind == "ok" else None), d, int(out[2]), float(out[3]), \
            {"inlier_frac": out[4], "template_zncc": out[5], "textured": int(out[6])}

    def trackable(self, g0, g1):
        a, pa = self._u8(g0)
        b, pb = self._u8(g1)
        return float(self.L.mm_motion_trackable(pa, pb, a.shape[1], a.shape[0]))


def endpoint_error(m_est, m_true, W, H, step=16):
    """Mean and max distance (px) between where the two transforms put a grid of points over the frame."""
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    P = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
    e = np.linalg.norm(np.asarray(m_est) @ P - np.asarray(m_true) @ P, axis=0)
    return float(e.mean()), float(e.max())


def stats(v, fail):
    v = np.asarray(v, float)
    if len(v) == 0:
        return {}
    return {"n": int(len(v)), "mean": round(float(v.mean()), 4), "median": round(float(np.median(v)), 4),
            "p95": round(float(np.percentile(v, 95)), 4), "fail_rate": round(float(np.mean(np.asarray(fail))), 4)}
