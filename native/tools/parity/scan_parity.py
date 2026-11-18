"""mm-scan's decide step against Python's, stage by stage, on a recording (tools/parity/scan_record.py).

    .venv/bin/python native/tools/parity/scan_parity.py [--name jazz] [--release]

Runs crates/mm-scan's scan_decide on a fresh copy of the recording's input project, answering every question about
pixels from the recording, and compares each stage's output (stage_*.json: exact, floats bit for bit) and every file
the step writes (byte for byte; decide_viz.json without "ms", and the join's "follow" events, which Python's threads
emit in any order, as a multiset). Prints JSON per stage and a PASS/FAIL line.
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
ap.add_argument("--release", action="store_true")
a = ap.parse_args()
BASE = os.path.join(ROOT, "out", "native", "scan", a.name)
REC = os.path.join(BASE, "rec")
summary = json.load(open(os.path.join(REC, "summary.json")))
RS = os.path.join(BASE, "rust")
DUMP = os.path.join(BASE, "rust_rec")
for d in (RS, DUMP):
    if os.path.exists(d):
        shutil.rmtree(d)
shutil.copytree(os.path.join(BASE, "input"), RS)
prof = "release" if a.release else "debug"
r = subprocess.run(["cargo", "build", "-q", "-p", "mm-scan", "--bin", "scan_decide"] + (["--release"] if a.release else []), cwd=NATIVE)
assert r.returncode == 0
cmd = [os.path.join(NATIVE, "target", prof, "scan_decide"), RS, "--replay", REC, "--dump", DUMP]
if summary["depth"] != "off":
    cmd += ["--depth", summary["depth"]]
r = subprocess.run(cmd, capture_output=True, text=True)
out = {"name": a.name, "depth": summary["depth"], "rust_exit": r.returncode}
replay_ok = False
if r.returncode == 0:
    res = json.loads(r.stdout.strip().splitlines()[-1])
    out["rust_seconds"] = round(res["seconds"], 2)
    out["replay_calls [recorded, used]"] = res["replay"]
    # every recorded call used once: Rust asked exactly what Python asked (calls with no recording would have panicked)
    replay_ok = all(n == u for n, u in res["replay"].values()) and \
        all(res["replay"].get(k, [0, 0])[0] == v for k, v in summary["calls"].items() if v)
if r.returncode:
    out["stderr"] = r.stderr[-3000:]


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


pv = os.path.join(PY, "proto", "decide_viz.json")
rv = os.path.join(RS, "proto", "decide_viz.json")
if not os.path.exists(rv):
    rv = os.path.join(DUMP, "events.json")
if os.path.exists(rv):
    (pe, pf), (re_, rf) = events(pv), events(rv)
    n = min(len(pe), len(re_))
    d = first_diff(pe[:n], re_[:n])
    files["decide_viz events"] = d or ("identical" if len(pe) == len(re_) else f"identical for the first {n} of {len(pe)} (rust {len(re_)})")
    files["decide_viz follow events (multiset)"] = "identical" if pf == rf else f"differ ({len(pf)} vs {len(rf)})"
out["files"] = files
print(json.dumps(out, indent=1))
ok = replay_ok and r.returncode == 0 and all(v == "exact" for v in stages.values()) and all(v == "identical" for v in files.values())
print("PASS" if ok else "FAIL")
