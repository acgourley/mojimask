"""The decisions benchmark (tools/bench/decision_bench.py, unedited) on jazz with our motion swapped in.

Variants, each scanned in its own scratch copy of the main checkout's out/bench/jazz (detection cached there) under
out/native/motion_bench/<variant>/, so neither the user's bench folder nor tools/bench/data/decisions_history.jsonl
is touched (the history line goes to the scratch folder):
  python       the scan as it is (Python's mats_v2.npy, cv2's flow in the follower)
  ours         our camera motion: mats_v2.npy written by mm-motion's motion_mats from video_io's frames, and
               track.camera_motion patched to return them (in case the pipeline asks)
  ours_follow  ours, plus the follower's motion through libmm_motion (ctypes): track._flow_measure,
               follow.face_motion (also cover.flows' steady emoji) and follow.Clip.trackable

    python native/tools/parity/motion_bench.py [--variants python,ours,ours_follow] [--note TEXT]
Prints each variant's bench line and a pass/fail line: pass if every ours* variant is no worse than python on
faces uncovered by rules, frames uncovered, never boxed, people split and mixed, and asks at most 2 more questions.
"""
import argparse
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
NATIVE = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(NATIVE)
MAIN = MAIN_CHECKOUT
PY = os.path.join(MAIN, ".venv", "bin", "python")
OUT = os.path.join(ROOT, "out", "native", "motion_bench")
REL = os.path.join(NATIVE, "target", "release")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(NATIVE, "tools", "motion"))


def scratch(variant):
    """A fresh copy of the main checkout's bench project for jazz (detection cached), and the clip linked in."""
    root = os.path.join(OUT, variant)
    proj = os.path.join(root, "out", "bench", "jazz")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(os.path.join(root, "test_clips"))
    os.symlink(os.path.join(MAIN, "test_clips", "jazz.mp4"), os.path.join(root, "test_clips", "jazz.mp4"))
    shutil.copytree(os.path.join(MAIN, "out", "bench", "jazz"), proj,
                    ignore=shutil.ignore_patterns("sheets", "bench_last.json", "*.lock"))
    return root, proj


def our_mats():
    import _common as C
    npy = os.path.join(C.OUT, "jazz_ours.npy")
    subprocess.run([os.path.join(REL, "motion_mats"), C.frames_dir("jazz"), npy], check=True, capture_output=True)
    return npy


def patch_follow():
    import numpy as np
    import cv2
    import _common as C
    from mojimask import track as T
    from mojimask.scan import follow as FO
    L = C.Lib()

    def flow_measure(g0, g1, c, core):
        kind, M, d, n_good, spread, _ = L.head(g0, g1, c, core)
        if d is None:
            return None
        return np.array(d, np.float32), n_good, spread

    def face_motion(g0, g1, c):
        kind, M, d, n_good, spread, _ = L.head(g0, g1, c, FO.FLOW_CORE)
        return (M if kind == "ok" else None), kind

    def trackable(self, a, b):
        key = (min(a, b), max(a, b))
        with self._lock:
            v = self._trk.get(key)
        if v is None:
            v = L.trackable(self.gray(key[0]), self.gray(key[1]))
            with self._lock:
                self._trk[key] = v
        return v

    T._flow_measure = flow_measure
    FO.face_motion = face_motion
    FO.Clip.trackable = trackable
    _ = cv2


def one(variant, note):
    root, proj = scratch(variant)
    if variant.startswith("ours"):
        import numpy as np
        mats = np.load(our_mats())
        np.save(os.path.join(proj, "mats_v2.npy"), mats)
        from mojimask import track as T
        T.camera_motion = lambda video, scale=0.5, meta=None: [m for m in mats]
    if variant == "ours_follow":
        patch_follow()
    sys.path.insert(0, os.path.join(ROOT, "tools", "bench"))
    import decision_bench as DB
    DB.ROOT = root
    DB.HISTORY = os.path.join(root, "decisions_history.jsonl")
    rec, rules, wrong = DB.run("jazz", "gt", None, False, f"motion_bench {variant}" + (f": {note}" if note else ""), True)
    print("RESULT " + json.dumps({"variant": variant, "line": DB.summary(rec), "record": rec}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="python,ours,ours_follow")
    ap.add_argument("--note", default="")
    ap.add_argument("--one")
    a = ap.parse_args()
    if a.one:
        return one(a.one, a.note)
    results = {}
    for v in a.variants.split(","):
        r = subprocess.run([PY, os.path.abspath(__file__), "--one", v, "--note", a.note], capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": ROOT, "MOJIMASK_DEPTH": "off"})
        line = next((x for x in r.stdout.splitlines() if x.startswith("RESULT ")), None)
        if line is None:
            print(r.stdout[-3000:], r.stderr[-3000:])
            raise SystemExit(f"{v} failed")
        results[v] = json.loads(line[7:])
        print(results[v]["line"])
    json.dump(results, open(os.path.join(OUT, "motion_bench.json"), "w"), indent=1)
    if "python" not in results:
        return
    base = results["python"]["record"]

    def harms(rec):
        res, w = rec["result"], rec["rule_wrong"]
        return {"face uncovered": w.get("face uncovered", 0), "uncovered_frames": res["uncovered_frames"],
                "never boxed": res["not_found_by_scan"], "people split": res["people_split"], "people mixed": res["people_mixed"]}

    ok = True
    for v, r in results.items():
        if v == "python":
            continue
        hb, hv = harms(base), harms(r["record"])
        worse = {k: (hb[k], hv[k]) for k in hb if hv[k] > hb[k]}
        more_asked = r["record"]["asked"] - base["asked"]
        good = not worse and more_asked <= 2
        ok &= good
        print(f"  {v}: asked {r['record']['asked']} ({more_asked:+d} vs python), worse on: {worse or 'nothing'}")
    print(f"{'PASS' if ok else 'FAIL'} motion_bench: our motion no worse than Python's on jazz's decisions benchmark")


if __name__ == "__main__":
    main()
