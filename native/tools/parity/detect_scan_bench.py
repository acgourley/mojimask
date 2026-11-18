"""decision_bench's numbers for a whole scan from the video: Python's pipeline.run against the native scan
(`mm-scan-tool scan`: Rust frames, detection and decide with Rust evidence; only camera motion and the follower's
motion evidence still come from Python).

    .venv/bin/python native/tools/parity/detect_scan_bench.py [--clip jazz] [--plan default] [--depth off|anything-small]

tools/bench/decision_bench.py is imported, not edited: its scan step is swapped for a fresh scan of the clip into a
project under out/native/scan/bench_full/ by each engine. Both detect with the same fixed plan (Python's timed choice is
pinned by writing its engine.json for that plan first). With depth on, the depth cache (dets/depth_<model>.*) is made
by Python's depth.compute in both (writing the grey movie isn't ported: mm-media's GreyMovie). Everything after the
scan (the census, the replayed answers, the scoring) is the bench's own Python; its history goes under the fake root,
so tools/bench/data/decisions_history.jsonl isn't touched. Also compares the scans' files (dets, frames, decisions).
Prints both bench lines, the scan times and whether every number agrees.
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
sys.path.insert(0, os.path.join(ROOT, "tools", "bench"))
ap = argparse.ArgumentParser()
ap.add_argument("--clip", default="jazz")
ap.add_argument("--plan", default="default")
ap.add_argument("--depth", default="off")
ap.add_argument("--engines", default="python,rust")
a = ap.parse_args()
assert a.clip == "jazz", "privacy: jazz only"
os.environ["MOJIMASK_DEPTH"] = a.depth
os.environ.setdefault("MOJIMASK_CACHE", os.path.join(ROOT, "out", "native", "cache", "py"))
VIDEO = os.path.join(ROOT, "test_clips", f"{a.clip}.mp4")

import decision_bench as DB  # noqa: E402
from mojimask import onnx_core as OC  # noqa: E402

FAKE = os.path.join(ROOT, "out", "native", "scan", "bench_full", a.clip + ("" if a.depth == "off" else "_depth"))
DB.ROOT = FAKE
DB.HISTORY = os.path.join(FAKE, "history.jsonl")
assert subprocess.run(["cargo", "build", "-q", "--release", "-p", "mm-engine", "--bin", "mm-scan-tool"], cwd=NATIVE).returncode == 0
TOOL = os.path.join(NATIVE, "target", "release", "mm-scan-tool")
PROJ = {}
TIMES = {}


def pin_python_plan(projects_dir):
    """fastest_core's memory (projects_dir/../engine.json) set to the plan, keyed as onnx_core keys it."""
    key = {"machine": OC.machine_model(), "ort": OC.ort.__version__,
           "plans": hashlib.md5(json.dumps(OC.PLANS, sort_keys=True).encode()).hexdigest()[:12]}
    os.makedirs(projects_dir, exist_ok=True)
    with open(os.path.join(projects_dir, os.pardir, "engine.json"), "w") as f:
        json.dump({"plan": a.plan, **key}, f)


def make_scan(engine):
    def scan(clip, detect=False, rescan=True):
        base = os.path.join(FAKE, engine, "out", "bench")
        proj = os.path.join(base, clip)
        if os.path.exists(proj):
            shutil.rmtree(proj)
        os.makedirs(proj)
        t0 = time.time()
        if engine == "python":
            pin_python_plan(base)
            from mojimask import pipeline
            pipeline.run(VIDEO, proj, report=lambda *x: None)
        else:
            if a.depth != "off":
                # the depth cache by Python (frames and dets made by Rust first)
                p = subprocess.run([TOOL, "detect", VIDEO, proj, "--plan", a.plan], capture_output=True, text=True, env={**os.environ, "MM_SCAN_QUIET": "1"})
                if p.returncode:
                    raise SystemExit(p.stderr[-2000:])
                from mojimask import depth as DP
                DP.compute(proj, a.depth)
            cmd = [TOOL, "scan", VIDEO, proj, "--plan", a.plan] + (["--depth", a.depth] if a.depth != "off" else [])
            p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "MM_SCAN_QUIET": "1"})
            if p.returncode:
                raise SystemExit(p.stderr[-3000:])
            TIMES["rust_detail"] = json.loads(p.stdout.strip().splitlines()[-1])
        PROJ[engine] = proj
        TIMES[engine] = round(time.time() - t0, 1)
        return proj, TIMES[engine]
    return scan


lines, recs = {}, {}
engines = a.engines.split(",")
for engine in engines:
    DB.scan = make_scan(engine)
    rec, rules, wrong = DB.run(a.clip, "gt", None, False, f"native scan {engine}", True)
    recs[engine] = rec
    lines[engine] = DB.summary(rec)
out = {"depth": a.depth, "plan": a.plan, "lines": lines, "scan_seconds": {e: TIMES.get(e) for e in engines}}
if "rust_detail" in TIMES:
    d = TIMES["rust_detail"]
    out["rust_steps"] = {"detect_s": round(d["detect"]["seconds"], 1), "detect_ms_per_frame": round(d["detect"]["per_frame_ms"], 1),
                         "motion_s (python)": round(d["motion_seconds"], 1), "decide_s": round(d["decide"]["seconds"], 1)}
ok = True
if len(engines) == 2:
    skip = {"at", "note", "scan_s", "code"}
    differ = [k for k in recs["python"] if k not in skip and recs["python"][k] != recs["rust"].get(k)]
    out["bench_differ"] = differ
    ok = not differ
    files = {}
    P, R = PROJ["python"], PROJ["rust"]
    for f in sorted(os.listdir(os.path.join(P, "dets"))):
        if f.endswith(".mp4"):
            continue
        files["dets/" + f] = open(os.path.join(P, "dets", f), "rb").read() == open(os.path.join(R, "dets", f), "rb").read()
    jp = sorted(os.listdir(os.path.join(P, "frames")))
    files["frames/*.jpg"] = all(open(os.path.join(P, "frames", f), "rb").read() == open(os.path.join(R, "frames", f), "rb").read() for f in jp)
    files["mats_v2.npy"] = open(os.path.join(P, "mats_v2.npy"), "rb").read() == open(os.path.join(R, "mats_v2.npy"), "rb").read()
    for f in ["proto/chains.json", "decisions_overlap.json", "decisions_blip.json", "decisions_track.json", "decisions_same.json",
              "proto_tracks.json"]:
        files[f] = open(os.path.join(P, f), "rb").read() == open(os.path.join(R, f), "rb").read()
    out["files_identical"] = files
print(json.dumps(out, indent=1))
print("PASS" if ok else "FAIL")
