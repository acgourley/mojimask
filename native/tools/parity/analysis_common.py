"""Shared by the analysis parity tests (analysis_*.py): paths, the jazz copy, frames as .npy, the Rust binary.

Only jazz (public domain): out/poc/projects/jazz is copied to out/native/analysis/jazz. Frames are decoded here
with cv2.imread and handed to Rust as .npy, so the JPEG decoder is not part of the comparison.
"""
import json
import os
import shutil
import subprocess
import sys
import time

# the main checkout (a worktree has no venv or models of its own): the parent of git's common directory
MAIN_CHECKOUT = os.path.dirname(subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse",
                                                "--path-format=absolute", "--git-common-dir"],
                                               capture_output=True, text=True).stdout.strip())

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(NATIVE)
SRC = os.path.join(ROOT, "out", "poc", "projects", "jazz")
OUT = os.path.join(ROOT, "out", "native", "analysis")
PROJ = os.path.join(OUT, "jazz")
MODELS = os.path.join(MAIN_CHECKOUT, "models/onnx")
DEPTH_MODEL = os.path.join(MAIN_CHECKOUT, "models/depth-anything-v2-small/model.onnx")
CACHE = os.path.join(OUT, "coreml_cache")  # Rust's CoreML compiles (Python keeps its own, onnx_core._coreml_cache)
BIN = os.path.join(NATIVE, "target", "release", "mm-analysis-parity")

sys.path.insert(0, ROOT)


def project():
    if not os.path.exists(os.path.join(PROJ, "meta.json")):
        if os.path.exists(PROJ):
            shutil.rmtree(PROJ)
        shutil.copytree(SRC, PROJ)
    return PROJ


def workdir(name, ep):
    d = os.path.join(OUT, f"{name}_{ep}")
    if os.path.exists(d):
        shutil.rmtree(d)
    for s in ("frames", "py", "rust"):
        os.makedirs(os.path.join(d, s))
    return d


def save_frames(d, ts):
    import cv2
    import numpy as np
    for t in sorted(set(ts)):
        np.save(os.path.join(d, "frames", f"{t:05d}.npy"), cv2.imread(os.path.join(PROJ, "frames", f"{t:05d}.jpg")))


def build():
    subprocess.run(["cargo", "build", "--release", "-q", "-p", "mm-analysis", "--bin", "mm-analysis-parity"], cwd=NATIVE,
                   check=True)


def run_rust(cmd, d, manifest):
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    t0 = time.perf_counter()
    subprocess.run([BIN, cmd, d], check=True)
    wall = time.perf_counter() - t0
    with open(os.path.join(d, "rust.json")) as f:
        return json.load(f), wall


def stats(ms):
    import numpy as np
    if not ms:
        return None
    a = np.array(ms, float)
    return {"n": len(a), "median_ms": round(float(np.median(a)), 3), "mean_ms": round(float(a.mean()), 3)}


def timed(times, key, f, *a, **k):
    t0 = time.perf_counter()
    r = f(*a, **k)
    times.setdefault(key, []).append((time.perf_counter() - t0) * 1e3)
    return r


def tracklet_dets(every=1):
    """[(index, t, [cx, cy, s])] for every detection of jazz's tracklets (as scan/faces.measure sees them)."""
    import numpy as np
    from mojimask.scan import tracklets as TL
    p = project()
    tl = TL.build(TL.detections(p), np.load(os.path.join(p, "mats_v2.npy")), TL.R)
    out = []
    for k, v in tl.items():
        for t, c in v["obs"].items():
            if t % every == 0:
                out.append((len(out), int(t), c, k))
    return out, tl


def ep_arg():
    return "plan" if "--coreml" in sys.argv else "cpu"
