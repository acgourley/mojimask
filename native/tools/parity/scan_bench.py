"""decision_bench's numbers with the decide step run by Python or by mm-scan (Rust), side by side.

    .venv/bin/python native/tools/parity/scan_bench.py [--name jazz] [--clip jazz]

tools/bench/decision_bench.py is imported, not edited: its scan step is swapped for one that copies the recording's
input project (tools/parity/scan_record.py; detection already done) and runs the decide step with the chosen engine:
  python  mojimask.scan.newalgo.run, as the bench does
  rust    crates/mm-scan's scan_decide, with the pictures' evidence (flow, faces, bodies, depth) replayed from the
          recording -- the Rust motion and analysis crates aren't plugged in yet
Everything after the scan (the census, the replayed answers, the scoring) is the bench's own Python. Its data root and
history file point under out/native/scan/bench/, so tools/bench/data/decisions_history.jsonl isn't touched.
Prints both bench lines and whether every number agrees.
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
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "bench"))
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="jazz", help="the recording (out/native/scan/NAME)")
ap.add_argument("--clip", default="jazz", help="the census to score against")
a = ap.parse_args()
BASE = os.path.join(ROOT, "out", "native", "scan", a.name)
REC = os.path.join(BASE, "rec")
summary = json.load(open(os.path.join(REC, "summary.json")))
depth = summary["depth"]
os.environ["MOJIMASK_DEPTH"] = depth

import decision_bench as DB  # noqa: E402

FAKE = os.path.join(ROOT, "out", "native", "scan", "bench", a.name)
DB.ROOT = FAKE
DB.HISTORY = os.path.join(FAKE, "history.jsonl")
r = subprocess.run(["cargo", "build", "-q", "--release", "-p", "mm-scan", "--bin", "scan_decide"], cwd=NATIVE)
assert r.returncode == 0


def make_scan(engine):
    def scan(clip, detect=False, rescan=True):
        proj = os.path.join(FAKE, "out", "bench", clip)
        if os.path.exists(proj):
            shutil.rmtree(proj)
        shutil.copytree(os.path.join(BASE, "input"), proj)
        import time
        t0 = time.time()
        if engine == "python":
            from mojimask.scan import newalgo
            newalgo.run(proj)
        else:
            cmd = [os.path.join(NATIVE, "target", "release", "scan_decide"), proj, "--replay", REC]
            if depth != "off":
                cmd += ["--depth", depth]
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode:
                raise SystemExit(p.stderr[-2000:])
        return proj, round(time.time() - t0, 1)
    return scan


lines, recs = {}, {}
for engine in ("python", "rust"):
    DB.scan = make_scan(engine)
    rec, rules, wrong = DB.run(a.clip, "gt", None, False, f"engine {engine}", True)
    recs[engine] = rec
    lines[engine] = DB.summary(rec)
skip = {"at", "note", "scan_s", "code"}
same = {k: recs["python"][k] == recs["rust"][k] for k in recs["python"] if k not in skip}
print(json.dumps({"depth": depth, "python": lines["python"], "rust": lines["rust"],
                  "scan_s": {e: recs[e]["scan_s"] for e in recs}, "differ": [k for k, v in same.items() if not v]}, indent=1))
print("PASS" if all(same.values()) else "FAIL")
