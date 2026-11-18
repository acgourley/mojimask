"""The native scan's rules measured: jazz against its census (tools/bench/decision_bench.py's numbers), other clips by
the worst-case question count (no truth needed).

    PYTHONPATH=ROOT <venv>/bin/python native/tools/bench/rule_bench.py --note "what changed" [--clips jazz,party360,party4k]
                    [--python-facts]

Each clip's detection, camera motion (Python's, --motion python) and depth cache (anything-small) are made once, in
out/native/rules/<clip>/base; every run copies that and reruns only the decide step with today's mm-scan-tool
(`mm-scan-tool decide`), so a run measures today's rules and nothing else.

- jazz: decision_bench's replay and scoring (imported, not edited) on that project, with the Decisions page's answers
  logic (facts.answers) from the native app (`mm-cli decisions answers-serve`), not Python's; --python-facts uses
  Python's (they must agree until a rule in facts changes). Its history goes to out/native/rules/history.jsonl.
- party360 (`party clip.mp4`, 640x360) and party4k (party_clip_4k.mov): no census, so the worst case is counted: every
  face question answered "a face" (overlaps "someone else"), every same-person question "different people". That is
  the most same-person questions a person can be asked; numbers only, no frame is looked at.
Prints one line per clip and appends them, with the note, to out/native/rules/runs.jsonl.
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
sys.path.insert(0, os.path.join(ROOT, "tools", "bench"))
ap = argparse.ArgumentParser()
ap.add_argument("--note", default="")
ap.add_argument("--clips", default="jazz,party360,party4k")
ap.add_argument("--python-facts", action="store_true")
a = ap.parse_args()
DEPTH = "anything-small"
os.environ["MOJIMASK_DEPTH"] = DEPTH
os.environ.setdefault("MOJIMASK_CACHE", os.path.join(ROOT, "out", "native", "cache", "py"))
# the main checkout (the party clips live in its test_clips/; a worktree has only jazz): git's common directory's parent
MAIN = os.path.dirname(subprocess.run(["git", "-C", ROOT, "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                      capture_output=True, text=True).stdout.strip())
CLIPS = {"jazz": os.path.join(ROOT, "test_clips", "jazz.mp4"),
         "party360": os.path.join(MAIN, "test_clips", "party clip.mp4"),
         "party4k": os.path.join(MAIN, "test_clips", "party_clip_4k.mov")}
# projects already scanned natively (Python's motion) that a base can start from instead of a fresh scan
SEED = {"party360": os.path.join(ROOT, "out", "native", "clipscan", "party_clip", "rust", "projects", "p"),
        "party4k": os.path.join(ROOT, "out", "native", "partyfull", "proj")}
OUT = os.path.join(ROOT, "out", "native", "rules")
REL = os.path.join(NATIVE, "target", "release")
assert subprocess.run(["cargo", "build", "-q", "--release", "-p", "mm-cli", "-p", "mm-engine"], cwd=NATIVE).returncode == 0
TOOL = [os.path.join(REL, "mm-scan-tool")]
RESULT = ("proto_tracks.json", "edits.json", "facts.json", "bench_last.json", "proto")

import decision_bench as DB  # noqa: E402
from mojimask import facts as FX  # noqa: E402

PY_ANSWERS = FX.answers


class NativeAnswers:
    """facts.answers by `mm-cli decisions answers-serve`, one process per record list."""
    def __init__(self):
        self.p, self.key = None, None

    def __call__(self, decisions, facts, hidden):
        key = id(decisions)
        if key != self.key:
            self.close()
            self.p = subprocess.Popen([os.path.join(REL, "mm-cli"), "decisions", "answers-serve"], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, text=True, bufsize=1)
            self.p.stdin.write(json.dumps({"decisions": decisions, "hidden": list(hidden or [])}) + "\n")
            self.key = key
        self.p.stdin.write(json.dumps(facts) + "\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def close(self):
        if self.p:
            self.p.stdin.close()
            self.p.wait()
            self.p = None


def base(clip):
    b = os.path.join(OUT, clip, "base")
    if os.path.exists(os.path.join(b, "dets", f"depth_{DEPTH}.json")):
        return b
    if os.path.exists(b):
        shutil.rmtree(b)
    seed = SEED.get(clip)
    if seed and os.path.isdir(seed):
        shutil.copytree(seed, b, ignore=shutil.ignore_patterns(*RESULT, "decisions_*.json", ".lock", "*.log"))
        if not os.path.exists(os.path.join(b, "dets", f"depth_{DEPTH}.json")):
            subprocess.run(TOOL + ["depth", b, DEPTH], check=True, env={**os.environ, "MM_SCAN_QUIET": "1"})
    else:
        os.makedirs(b)
        subprocess.run(TOOL + ["scan", CLIPS[clip], b, "--plan", "default", "--motion", "python", "--depth", DEPTH,
                               "--python", sys.executable, "--root", ROOT], check=True,
                       env={**os.environ, "MM_SCAN_QUIET": "1", "PYTHONPATH": ROOT})
    return b


def decide(clip):
    run = os.path.join(OUT, clip, "run")
    if os.path.exists(run):
        shutil.rmtree(run)
    shutil.copytree(base(clip), run, ignore=shutil.ignore_patterns(*RESULT, "decisions_*.json"))
    t0 = time.time()
    p = subprocess.run(TOOL + ["decide", run, "--depth", DEPTH, "--motion", "python", "--python", sys.executable, "--root", ROOT],
                       capture_output=True, text=True, env={**os.environ, "MM_SCAN_QUIET": "1", "PYTHONPATH": ROOT})
    if p.returncode:
        raise SystemExit(p.stdout[-2000:] + p.stderr[-3000:])
    return run, round(time.time() - t0, 1)


def worst_case(proj):
    """Every face question "a face" (overlaps "someone else"), every same-person question "different people"."""
    pt = json.load(open(os.path.join(proj, "proto_tracks.json")))
    ds = DB.decisions(proj)
    doc, asked = {"facts": []}, {}
    pick = {"track": FX.A_FACE, "blip": FX.A_FACE, "overlap": FX.A_SOMEONE_ELSE, "same": FX.A_DIFFERENT}
    while True:
        out = FX.answers(ds, doc["facts"], pt.get("hidden", []))
        op = sorted((d for d in out if d["answer"] is None),
                    key=lambda d: (DB.RANK[d["kind"]], -d.get("weight", 0), -d["evidence"].get("frames", 0)))
        if not op or sum(asked.values()) > 5000:
            break
        d = op[0]
        FX.append(doc, FX.facts_for_answer(d, pick.get(d["kind"], FX.A_FACE)), "bench", d["id"])
        asked[d["kind"]] = asked.get(d["kind"], 0) + 1
    by = {}
    for d in out:
        if d["kind"] == "same" and d.get("by") and d["by"] != "user":
            k = d["by"].split(" (")[0][:60]
            by[k] = by.get(k, 0) + 1
    return {"records": {k: sum(1 for d in ds if d["kind"] == k) for k in ("track", "blip", "overlap", "same")},
            "same_records_score_0.8+": sum(1 for d in ds if d["kind"] == "same" and d["evidence"].get("score", 0) >= FX.ASK_SCORE),
            "asked": sum(asked.values()), "asked_by_kind": asked, "same_settled_by": by}


native = NativeAnswers()
if not a.python_facts:
    FX.answers = native
DB.ROOT = os.path.join(OUT, "jazz_bench_root")
DB.HISTORY = os.path.join(OUT, "history.jsonl")
os.makedirs(OUT, exist_ok=True)
lines = []
for clip in a.clips.split(","):
    proj, secs = decide(clip)
    if clip == "jazz":
        DB.scan = lambda c, detect=False, rescan=True, _p=proj, _s=secs: (_p, _s)
        rec, rules, wrong = DB.run("jazz", "gt", None, False, a.note + (" (python facts)" if a.python_facts else ""), True)
        row = {"clip": clip, "line": DB.summary(rec), "record": rec}
        print(row["line"])
    else:
        row = {"clip": clip, "decide_s": secs, **worst_case(proj)}
        print(json.dumps(row))
    native.close()
    lines.append({"at": time.strftime("%Y-%m-%d %H:%M"), "note": a.note, "facts": "python" if a.python_facts else "native", **row})
with open(os.path.join(OUT, "runs.jsonl"), "a") as f:
    for r in lines:
        f.write(json.dumps(r, default=str) + "\n")
