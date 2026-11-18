"""The decide step with Rust's own evidence (crates/mm-engine/src/evidence.rs on mm-analysis) against Python's, on a
recording of a Python scan (tools/parity/scan_record.py: jazz, jazz_depth, ...).

    .venv/bin/python native/tools/parity/detect_evidence.py [--name jazz] [--check-only] [--analysis-cpu]

1. evidence: every recorded face / face_near / body / depth call asked again of the Rust evidence (`mm-scan-tool
   evidence-check`): how many agree bit for bit, and the largest difference. This tells a difference in the outputs
   below apart as the evidence's or the order of calls'.
2. decide: `mm-scan-tool decide` on a fresh copy of the recording's input project, faces, bodies and depth measured by
   Rust (default engines, as the recording: landmarks on the Neural Engine, SFace on the GPU, OSNet on the CPU), motion
   replayed from the recording (mm-motion's, not yet ported), or with --motion python asked of Python live. Every stage and every file the step writes compared
   with Python's, as scan_parity.py does.
Prints JSON and a PASS/FAIL line (PASS: every stage and file identical).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.dirname(NATIVE)
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="jazz")
ap.add_argument("--check-only", action="store_true")
ap.add_argument("--skip-check", action="store_true")
ap.add_argument("--analysis-cpu", action="store_true")
ap.add_argument("--motion", default="replay", help="replay (from the recording) or python (asked of Python live: evidence::PyMotion)")
a = ap.parse_args()
BASE = os.path.join(ROOT, "out", "native", "scan", a.name)
REC = os.path.join(BASE, "rec")
summary = json.load(open(os.path.join(REC, "summary.json")))
assert subprocess.run(["cargo", "build", "-q", "--release", "-p", "mm-engine", "--bin", "mm-scan-tool"], cwd=NATIVE).returncode == 0
TOOL = os.path.join(NATIVE, "target", "release", "mm-scan-tool")
extra = (["--depth", summary["depth"]] if summary["depth"] != "off" else []) + (["--analysis-cpu"] if a.analysis_cpu else [])
out = {"name": a.name, "depth": summary["depth"]}
if not a.skip_check:
    r = subprocess.run([TOOL, "evidence-check", REC, os.path.join(BASE, "input")] + extra, capture_output=True, text=True)
    out["evidence"] = json.loads(r.stdout) if r.returncode == 0 else r.stderr[-2000:]
if a.check_only:
    print(json.dumps(out, indent=1))
    sys.exit(0)

RS = os.path.join(BASE, "rust_ev" + ("" if a.motion == "replay" else "_pymotion"))
DUMP = RS + "_rec"
for d in (RS, DUMP):
    if os.path.exists(d):
        shutil.rmtree(d)
shutil.copytree(os.path.join(BASE, "input"), RS)
r = subprocess.run([TOOL, "decide", RS, "--dump", DUMP] + (["--replay", REC] if a.motion == "replay" else []) + extra,
                   capture_output=True, text=True)
out["motion"] = a.motion
out["rust_exit"] = r.returncode
if r.returncode:
    out["stderr"] = r.stderr[-3000:]
else:
    res = json.loads(r.stdout.strip().splitlines()[-1])
    out["rust"] = res


def first_diff(x, y, path=""):
    if type(x) is not type(y) and not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        return f"{path}: {type(x).__name__} vs {type(y).__name__}"
    if isinstance(x, dict):
        if list(x) != list(y):
            kx, ky = set(x), set(y)
            if kx != ky:
                return f"{path}: keys only in python {sorted(kx - ky)[:5]}, only in rust {sorted(ky - kx)[:5]}"
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


stages = {}
for f in sorted(os.listdir(REC)):
    if not f.startswith("stage_") or f == "stage_dets.json":
        continue
    st = f[6:-5]
    rp = os.path.join(DUMP, f)
    if not os.path.exists(rp):
        stages[st] = "not produced"
        continue
    d = first_diff(json.load(open(os.path.join(REC, f))), json.load(open(rp)))
    stages[st] = "exact" if d is None else d
out["stages"] = stages
files = {}
PY = os.path.join(BASE, "py")
for f in ["proto/chains.json", "decisions_overlap.json", "decisions_blip.json", "decisions_track.json", "decisions_same.json",
          "proto_tracks.json", "edits.json"]:
    p, q = os.path.join(PY, f), os.path.join(RS, f)
    if not os.path.exists(q):
        files[f] = "not written"
    else:
        files[f] = "identical" if open(p, "rb").read() == open(q, "rb").read() else first_diff(json.load(open(p)), json.load(open(q))) or "same JSON, different bytes"


def events(path):
    ev = json.load(open(path))
    for e in ev:
        e.pop("ms", None)
    fol = sorted(json.dumps(e, sort_keys=True) for e in ev if e["e"] == "follow")
    return [e for e in ev if e["e"] != "follow"], fol


pv, rv = os.path.join(PY, "proto", "decide_viz.json"), os.path.join(RS, "proto", "decide_viz.json")
if os.path.exists(rv):
    (pe, pf), (re_, rf) = events(pv), events(rv)
    n = min(len(pe), len(re_))
    d = first_diff(pe[:n], re_[:n])
    files["decide_viz events"] = d or ("identical" if len(pe) == len(re_) else f"identical for the first {n} of {len(pe)} (rust {len(re_)})")
    files["decide_viz follow events (multiset)"] = "identical" if pf == rf else f"differ ({len(pf)} vs {len(rf)})"
out["files"] = files
print(json.dumps(out, indent=1))
ok = r.returncode == 0 and all(v == "exact" for v in stages.values()) and all(v == "identical" for v in files.values())
print("PASS" if ok else "FAIL")
