"""The depth cache of a new project made natively (mm-engine depth.rs: mm-analysis's model, mm-media's GreyWriter and
luma) against Python's depth.compute, on jazz's frames.

    .venv/bin/python native/tools/parity/engine2_depth.py [--name jazz_depth] [--model anything-small] [--keep]

1. Python's depth.compute and `mm-scan-tool depth` each make the cache in a fresh project with the same frames
   (out/native/engine2/depth/{py,rs}).
2. The sidecars (dets/depth_<model>.json) byte for byte; the movies' decoded frames: every Y plane of Rust's movie
   (decoded by Python's video_io.luma) against Python's movie's, and Rust's own reading of its movie (`luma-dump`,
   mm-media) against Python's reading of its movie. The .mp4 files themselves differ in the container's dates only.
3. The readings: every Depth.near call recorded in the scan recording (out/native/scan/<name>/rec) asked of the Rust
   evidence over the Rust-made cache (`mm-scan-tool evidence-check --only depth`).
Prints JSON and a PASS/FAIL line (PASS: sidecar identical, every frame identical, every reading identical).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.dirname(NATIVE)
sys.path.insert(0, ROOT)
os.environ.setdefault("MOJIMASK_CACHE", os.path.join(ROOT, "out", "native", "cache", "py"))
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="jazz_depth")
ap.add_argument("--model", default="anything-small")
ap.add_argument("--keep", action="store_true", help="reuse caches already made")
a = ap.parse_args()
assert "jazz" in a.name, "privacy: jazz only"
from mojimask import depth as DP  # noqa: E402

SRC = os.path.join(ROOT, "out", "native", "scan", a.name)
BASE = os.path.join(ROOT, "out", "native", "engine2", "depth")
assert subprocess.run(["cargo", "build", "-q", "--release", "-p", "mm-engine", "--bin", "mm-scan-tool"], cwd=NATIVE).returncode == 0
TOOL = os.path.join(NATIVE, "target", "release", "mm-scan-tool")
out = {"name": a.name, "model": a.model}
times = {}
for who in ("py", "rs"):
    proj = os.path.join(BASE, who)
    if not (a.keep and DP.cached(proj, a.model)):
        if os.path.exists(proj):
            shutil.rmtree(proj)
        os.makedirs(os.path.join(proj, "dets"))
        shutil.copytree(os.path.join(SRC, "input", "frames"), os.path.join(proj, "frames"))
        t0 = time.time()
        if who == "py":
            DP.compute(proj, a.model)
        else:
            r = subprocess.run([TOOL, "depth", proj, a.model], capture_output=True, text=True)
            if r.returncode:
                raise SystemExit(r.stderr[-3000:])
        times[who] = round(time.time() - t0, 1)
out["seconds (model load included)"] = times
P, R = (os.path.join(BASE, w, "dets", f"depth_{a.model}") for w in ("py", "rs"))
out["sidecar identical"] = open(P + ".json", "rb").read() == open(R + ".json", "rb").read()
out["mp4 bytes identical"] = open(P + ".mp4", "rb").read() == open(R + ".mp4", "rb").read()
py_frames = list(DP._luma(P + ".mp4"))
rs_by_py = list(DP._luma(R + ".mp4"))
out["frames"] = len(py_frames)
out["rust movie decoded by Python: frames identical"] = len(py_frames) == len(rs_by_py) and all(
    i == j and np.array_equal(x, y) for (i, x), (j, y) in zip(py_frames, rs_by_py))
dump = os.path.join(BASE, "luma_rs")
if os.path.exists(dump):
    shutil.rmtree(dump)
r = subprocess.run([TOOL, "luma-dump", R + ".mp4", dump, str(len(py_frames) + 5)], capture_output=True, text=True)
rows = [l.split() for l in r.stdout.strip().splitlines()]
same = len(rows) == len(py_frames)
for (i, x), row in zip(py_frames, rows):
    y = np.fromfile(os.path.join(dump, f"{row[0]}.raw"), np.uint8)
    same = same and int(row[0]) == i and y.size == x.size and np.array_equal(y.reshape(x.shape), x)
out["rust movie decoded by Rust (mm-media) vs Python's: frames identical"] = same
# readings: every recorded Depth.near over the Rust-made cache
proj = os.path.join(BASE, "rs_project")
if os.path.exists(proj):
    shutil.rmtree(proj)
shutil.copytree(os.path.join(SRC, "input"), proj)
for ext in (".json", ".mp4"):
    shutil.copy(R + ext, os.path.join(proj, "dets", f"depth_{a.model}{ext}"))
r = subprocess.run([TOOL, "evidence-check", os.path.join(SRC, "rec"), proj, "--depth", a.model, "--only", "depth"], capture_output=True, text=True)
out["readings"] = json.loads(r.stdout)["depth"] if r.returncode == 0 else r.stderr[-2000:]
print(json.dumps(out, indent=1))
rd = out["readings"]
ok = (out["sidecar identical"] and out["rust movie decoded by Python: frames identical"]
      and out["rust movie decoded by Rust (mm-media) vs Python's: frames identical"]
      and isinstance(rd, dict) and rd.get("exact") == rd.get("calls"))
print("PASS" if ok else "FAIL")
