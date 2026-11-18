"""A whole scan of any clip, native against Python, with numbers only (no truth needed, no frame is looked at).

    PYTHONPATH=ROOT <venv>/bin/python native/tools/parity/clip_scan_parity.py VIDEO [--name NAME] [--depth anything-small]
                    [--motion python] [--engines python,rust] [--plan default]

Python's pipeline.run and `mm-cli scan` (--motion python by default, so camera motion and the follower's motion come
from Python and any difference is the port's) each scan VIDEO into a fresh project under out/native/clipscan/<name>/.
Both detect with the same pinned plan and, with depth on, make their own depth cache. Then every file the scan writes
is compared byte for byte (dets/*.json, frames/*.jpg, mats_v2.npy, the chains, decisions and tracks, edits.json); for
a file that differs, the first differing JSON path. Prints JSON (counts and timings only) and a PASS/FAIL line.
For clips of real people: nothing here prints pixels or crops, and the projects stay in out/ (git-ignored).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.dirname(NATIVE)
sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--name")
ap.add_argument("--depth", default="anything-small")
ap.add_argument("--motion", default="python")
ap.add_argument("--plan", default="default")
ap.add_argument("--engines", default="python,rust")
a = ap.parse_args()
VIDEO = os.path.abspath(a.video)
NAME = a.name or os.path.splitext(os.path.basename(VIDEO))[0].replace(" ", "_")
BASE = os.path.join(ROOT, "out", "native", "clipscan", NAME)
os.environ["MOJIMASK_DEPTH"] = a.depth
os.environ.setdefault("MOJIMASK_CACHE", os.path.join(ROOT, "out", "native", "cache", "py"))

from mojimask import onnx_core as OC  # noqa: E402


def pin_python_plan(projects_dir):
    """fastest_core's memory (projects_dir/../engine.json) set to the plan, keyed as onnx_core keys it."""
    key = {"machine": OC.machine_model(), "ort": OC.ort.__version__,
           "plans": hashlib.md5(json.dumps(OC.PLANS, sort_keys=True).encode()).hexdigest()[:12]}
    os.makedirs(projects_dir, exist_ok=True)
    with open(os.path.join(projects_dir, os.pardir, "engine.json"), "w") as f:
        json.dump({"plan": a.plan, **key}, f)


def first_diff(x, y, path=""):
    if type(x) is not type(y) and not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        return f"{path}: {type(x).__name__} vs {type(y).__name__}"
    if isinstance(x, dict):
        if set(x) != set(y):
            return f"{path}: keys only in python {sorted(set(x) - set(y))[:5]}, only in rust {sorted(set(y) - set(x))[:5]}"
        if list(x) != list(y):
            return f"{path}: key order differs"
        for k in x:
            d = first_diff(x[k], y[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(x, list):
        if len(x) != len(y):
            return f"{path}: length {len(x)} vs {len(y)}"
        for i, (p, q) in enumerate(zip(x, y)):
            d = first_diff(p, q, f"{path}[{i}]")
            if d:
                return d
        return None
    if x != y and not (x != x and y != y):
        return f"{path}: {x!r} vs {y!r}"
    return None


times, out = {}, {"video": os.path.basename(VIDEO), "depth": a.depth, "motion": a.motion, "plan": a.plan}
engines = a.engines.split(",")
for engine in engines:
    proj = os.path.join(BASE, engine, "projects", "p")
    t0 = time.time()
    if engine == "python":
        if os.path.exists(proj):
            shutil.rmtree(proj)
        os.makedirs(proj)
        pin_python_plan(os.path.dirname(proj))
        from mojimask import pipeline
        pipeline.run(VIDEO, proj, report=lambda *x: None)
    else:
        if os.path.exists(proj):
            shutil.rmtree(proj)
        os.makedirs(proj)
        assert subprocess.run(["cargo", "build", "-q", "--release", "-p", "mm-cli"], cwd=NATIVE).returncode == 0
        cmd = [os.path.join(NATIVE, "target", "release", "mm-cli"), "scan", VIDEO, proj, "--plan", a.plan,
               "--motion", a.motion, "--python", sys.executable] + (["--depth", a.depth] if a.depth != "off" else [])
        p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "MM_SCAN_QUIET": "1", "PYTHONPATH": ROOT})
        if p.returncode:
            raise SystemExit(p.stdout[-2000:] + p.stderr[-3000:])
        out["rust_scan"] = json.loads(p.stdout.strip().splitlines()[-1])["scan"]
    times[engine] = round(time.time() - t0, 1)
out["scan_seconds"] = times

P = os.path.join(BASE, "python", "projects", "p")
R = os.path.join(BASE, "rust", "projects", "p")
ok = True
if os.path.isdir(P) and os.path.isdir(R):
    files = {}

    def cmp(rel):
        p, q = os.path.join(P, rel), os.path.join(R, rel)
        if not os.path.exists(p) or not os.path.exists(q):
            return f"missing in {'python' if not os.path.exists(p) else 'rust'}"
        pb, qb = open(p, "rb").read(), open(q, "rb").read()
        if pb == qb:
            return "identical"
        if rel.endswith(".json"):
            try:
                return first_diff(json.loads(pb), json.loads(qb)) or "same JSON, different bytes"
            except ValueError:
                pass
        return "differ"

    for f in sorted(set(os.listdir(os.path.join(P, "dets"))) | set(os.listdir(os.path.join(R, "dets")))):
        if not f.endswith(".mp4"):
            files["dets/" + f] = cmp("dets/" + f)
    jp = sorted(f for f in os.listdir(os.path.join(P, "frames")) if f.endswith(".jpg"))
    jr = sorted(f for f in os.listdir(os.path.join(R, "frames")) if f.endswith(".jpg"))
    bad = [f for f in jp if cmp("frames/" + f) != "identical"]
    files["frames/*.jpg"] = "identical" if jp == jr and not bad else f"{len(jp)} vs {len(jr)} frames, {len(bad)} differ (first {bad[:3]})"
    for f in ["mats_v2.npy", "proto/chains.json", "decisions_overlap.json", "decisions_blip.json", "decisions_track.json",
              "decisions_same.json", "proto_tracks.json", "edits.json"]:
        files[f] = cmp(f)
    out["files"] = files
    ok = all(v == "identical" for v in files.values())

    def counts(proj):
        c = {}
        for k in ("overlap", "blip", "track", "same"):
            fp = os.path.join(proj, f"decisions_{k}.json")
            if os.path.exists(fp):
                d = json.load(open(fp))
                recs = d if isinstance(d, list) else d.get("decisions", d.get("records", []))
                c[k] = len(recs)
        return c
    out["decision_records"] = {"python": counts(P), "rust": counts(R)}
print(json.dumps(out, indent=1))
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
