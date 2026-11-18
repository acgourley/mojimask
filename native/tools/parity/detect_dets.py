"""Detection: mm-engine's prepare + detect (crates/mm-engine/src/scan.rs, detect.rs) against pipeline.py's, on jazz.

    .venv/bin/python native/tools/parity/detect_dets.py [--plan cpu|default] [--frames N] [--reuse]

Python side: pipeline.run's prepare and detect steps as written (video_io.probe/extract_jpegs, onnx_core.Core with the
plan, the same json.dump of dets/*.json and pose_head.json), only the plan fixed instead of timed. Rust side:
`mm-scan-tool detect VIDEO PROJECT --plan P`. Compares, in out/native/detect/{py,rs}_<plan>/:
  - dets/*.json: byte for byte, and per model the boxes (count, max |difference|)
  - frames/*.jpg: byte for byte
  - meta.json: the same keys and values (engine aside)
and reports the time per frame of each. Exact on the CPU plan; on CoreML (default plan) both run the same compiled
models, so it is expected exact too. Prints JSON and a PASS/FAIL line.
"""
import argparse
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
ap.add_argument("--plan", default="cpu")
ap.add_argument("--reuse", action="store_true", help="keep an existing Python run")
a = ap.parse_args()
VIDEO = os.path.join(ROOT, "test_clips", "jazz.mp4")
BASE = os.path.join(ROOT, "out", "native", "detect")
PY, RS = os.path.join(BASE, f"py_{a.plan}"), os.path.join(BASE, f"rs_{a.plan}")
os.environ.setdefault("MOJIMASK_CACHE", os.path.join(ROOT, "out", "native", "cache", "py"))  # Python's CoreML compiles


def python_detect(proj):
    """pipeline.run's prepare + detect, plan fixed."""
    from mojimask import onnx_core, video_io
    from mojimask import pose_util as PU
    from mojimask.coords import _write as write_json
    os.makedirs(f"{proj}/frames")
    os.makedirs(f"{proj}/dets")
    meta = video_io.probe(VIDEO)
    meta["video"] = os.path.abspath(VIDEO)
    meta["space"] = video_io.space(meta)
    video_io.extract_jpegs(VIDEO, f"{proj}/frames", meta)
    n = len([x for x in os.listdir(f"{proj}/frames") if x.endswith(".jpg")])
    meta["frames"] = n
    t0 = time.time()
    core = onnx_core.Core(a.plan)
    sample = next(iter(video_io.frames(VIDEO, meta)))
    core(sample)  # load (+ compile), as fastest_core's first frame
    setup = time.time() - t0
    meta["engine"] = {"plan": a.plan, "fixed": True}
    write_json(f"{proj}/meta.json", meta)
    profile = ["yolov5m_face_tiled", "retinaface_r50_2x", "pose_full"]
    out = {name: [] for name in profile}
    H, W = meta["height"], meta["width"]
    dec = video_io.frames(VIDEO, meta)
    t0 = time.time()
    for t in range(n):
        frame = next(dec, None)
        if frame is None:
            for name in profile:
                out[name] += [[] for _ in range(n - t)]
            break
        found = core(frame)
        for name in profile:
            out[name].append(found[name])
    secs = time.time() - t0
    dec.close()
    for name, frames in out.items():
        json.dump({"w": W, "h": H, "fps": meta["fps"], "space": meta.get("space"), "frames": frames}, open(f"{proj}/dets/{name}.json", "w"))
    heads = [[h for h in (PU.head_box(p) for p in fr) if h] for fr in out["pose_full"]]
    json.dump({"w": W, "h": H, "fps": meta["fps"], "space": meta.get("space"), "frames": heads}, open(f"{proj}/dets/pose_head.json", "w"))
    return {"frames": n, "seconds": secs, "per_frame_ms": secs / n * 1000, "setup_seconds": setup}


if not (a.reuse and os.path.exists(os.path.join(PY, "timing.json"))):
    if os.path.exists(PY):
        shutil.rmtree(PY)
    tp = python_detect(PY)
    json.dump(tp, open(os.path.join(PY, "timing.json"), "w"))
tp = json.load(open(os.path.join(PY, "timing.json")))

if os.path.exists(RS):
    shutil.rmtree(RS)
assert subprocess.run(["cargo", "build", "-q", "--release", "-p", "mm-engine", "--bin", "mm-scan-tool"], cwd=NATIVE).returncode == 0
r = subprocess.run([os.path.join(NATIVE, "target", "release", "mm-scan-tool"), "detect", VIDEO, RS, "--plan", a.plan],
                   capture_output=True, text=True, env={**os.environ, "MM_SCAN_QUIET": "1"})
if r.returncode:
    print(r.stderr[-3000:])
    print("FAIL")
    sys.exit(1)
tr = json.loads(r.stdout.strip().splitlines()[-1])

res = {"plan": a.plan, "python": tp, "rust": tr}
ok = True
dets = {}
for f in sorted(os.listdir(os.path.join(PY, "dets"))):
    pb, rb = open(os.path.join(PY, "dets", f), "rb").read(), open(os.path.join(RS, "dets", f), "rb").read()
    p, q = json.loads(pb), json.loads(rb)
    nbox = [sum(len(x) for x in p["frames"]), sum(len(x) for x in q["frames"])]
    worst, frames_differ = 0.0, 0
    for fp, fq in zip(p["frames"], q["frames"]):
        if fp != fq:
            frames_differ += 1
        if len(fp) == len(fq):
            for bp, bq in zip(fp, fq):
                worst = max([worst] + [abs(x - y) for x, y in zip(bp, bq)])
    same = pb == rb
    ok &= same
    dets[f] = {"bytes_identical": same, "boxes [py, rust]": nbox, "frames_differing": frames_differ, "max_abs_diff_aligned": worst}
res["dets"] = dets
jp = sorted(os.listdir(os.path.join(PY, "frames")))
bad = [f for f in jp if open(os.path.join(PY, "frames", f), "rb").read() != open(os.path.join(RS, "frames", f), "rb").read()]
res["frames"] = {"n": len(jp), "n_rust": len(os.listdir(os.path.join(RS, "frames"))), "not_identical": bad[:10]}
ok &= not bad and len(jp) == res["frames"]["n_rust"]
mp, mr = json.load(open(os.path.join(PY, "meta.json"))), json.load(open(os.path.join(RS, "meta.json")))
mp.pop("engine"), mr.pop("engine")
res["meta_equal"] = mp == mr
if mp != mr:
    res["meta"] = {"python": mp, "rust": mr}
ok &= mp == mr
res["speed"] = {"python_ms_per_frame": round(tp["per_frame_ms"], 1), "rust_ms_per_frame": round(tr["per_frame_ms"], 1)}
print(json.dumps(res, indent=1))
print("PASS" if ok else "FAIL")
