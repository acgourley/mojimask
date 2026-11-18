"""mm-scan on photos: jazz's census frames saved as photos (as tools/bench/photo_bench.py makes them), each scanned
on its own, its decide step recorded (scan_record.py --project), replayed in Rust (scan_parity.py), and photo_bench's
numbers summed over the photos with the decide step run by Python and by Rust.

    .venv/bin/python native/tools/parity/scan_photos.py [--every 4]

Photos and their projects: out/native/scan/photos/jazz_<t>.png and .../jazz_<t>/ (detection done once, by the Python
pipeline). Recordings: out/native/scan/photo_jazz_<t>/. Nothing is written to tools/ or out/bench/.
"""
import argparse
import collections
import json
import os
import shutil
import subprocess
import sys


# the main checkout (a worktree has no venv or models of its own): the parent of git's common directory
MAIN_CHECKOUT = os.path.dirname(subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse",
                                                "--path-format=absolute", "--git-common-dir"],
                                               capture_output=True, text=True).stdout.strip())

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.dirname(NATIVE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "bench"))
PY = os.path.join(MAIN_CHECKOUT, ".venv/bin/python")
ap = argparse.ArgumentParser()
ap.add_argument("--every", type=int, default=4)
a = ap.parse_args()
os.environ["MOJIMASK_DEPTH"] = "off"

import cv2  # noqa: E402
import decision_bench as B  # noqa: E402
import photo_bench as PB  # noqa: E402
from mojimask import paths, video_io  # noqa: E402

D = os.path.join(ROOT, "out", "native", "scan", "photos")
os.makedirs(D, exist_ok=True)
census_doc = json.load(open(paths.truth("jazz", ROOT)))
checked = sorted(int(t) for t in census_doc["checked"])[::a.every]
want = {t: os.path.join(D, f"jazz_{t}.png") for t in checked}
if not all(os.path.exists(p) for p in want.values()):
    for t, f in enumerate(video_io.frames(os.path.join(ROOT, "test_clips", "jazz.mp4"), None, color="display")):
        if t in want:
            cv2.imwrite(want[t], f)
        if t >= checked[-1]:
            break
env = dict(os.environ, PYTHONPATH=ROOT)
parity = {}
for t, photo in want.items():
    proj = photo[:-4]
    if not os.path.exists(os.path.join(proj, "dets", "pose_head.json")):
        B.scan_into(photo, proj)  # detection (once) and the decide step
    name = f"photo_jazz_{t}"
    subprocess.run([PY, os.path.join(HERE, "scan_record.py"), "--project", proj, "--name", name], env=env, check=True, capture_output=True)
    r = subprocess.run([PY, os.path.join(HERE, "scan_parity.py"), "--name", name], env=env, capture_output=True, text=True)
    parity[t] = r.stdout.strip().splitlines()[-1]

census, _ = B.census_frames(paths.truth("jazz", ROOT))
lines = {}
for engine in ("python", "rust"):
    asked, harms, res = collections.Counter(), collections.Counter(), collections.Counter()
    for t in checked:
        base = os.path.join(ROOT, "out", "native", "scan", f"photo_jazz_{t}")
        proj = os.path.join(base, f"bench_{engine}")
        if os.path.exists(proj):
            shutil.rmtree(proj)
        shutil.copytree(os.path.join(base, "input"), proj)
        if engine == "python":
            from mojimask.scan import newalgo
            newalgo.run(proj)
        else:
            subprocess.run([os.path.join(NATIVE, "target", "debug", "scan_decide"), proj, "--replay", os.path.join(base, "rec")],
                           check=True, capture_output=True)
        truth = PB.PhotoTruth(proj, census[t])
        ds = B.decisions(proj)
        facts, asked_, log, final = B.replay(ds, truth.pt.get("hidden", []), truth)
        cov = B.Covers(truth, facts)
        rules, wrong = B.rule_errors(final, truth, truth, cov)
        r = B.result(truth, cov)
        asked.update(asked_)
        harms.update(w["harm"] for w in wrong)
        res.update({k: r[k] for k in ("face_frames", "not_found_by_scan", "uncovered_frames")})
    lines[engine] = {"photos": len(checked), "asked": sum(asked.values()), "asked_by_kind": dict(asked), "rule_wrong": dict(harms),
                     "result": dict(res)}
print(json.dumps({"parity": parity, "bench": lines}, indent=1))
print("PASS" if all(v == "PASS" for v in parity.values()) and lines["python"] == lines["rust"] else "FAIL")
