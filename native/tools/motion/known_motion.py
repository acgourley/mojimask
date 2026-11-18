"""Known-motion test of camera motion: ours (mm-motion) against Python's track.camera_motion (cv2) on the same pairs.

The pairs (crates/mm-motion/src/bin/motion_testset.rs): jazz frames (643 x 476, and 1280 x 720 from jazz_4k's
decode) moved by known similarities with mm-vision's warpAffine (cv2's, bit-exact): small, large, whip, rotate,
zoom; with motion blur, defocus, noise, low light, JPEG, an exposure change; and 0, 2 or 4 "people" (patches of
another frame) moving on their own in front. Measured: the mean endpoint error over a grid of the pair (px), for
  prod  what mats_v2.npy would hold (Python's guards: identity when untrusted, scale clipped to 0.97-1.03; ours the same)
  raw   the fit itself (no guards; Python: None where cv2 found nothing, counted as identity)
A failure is a mean endpoint error over 1 px (or 0.5% of the width, whichever is larger).

    python native/tools/motion/known_motion.py [--regen] [--out out/native/motion/known_motion.json]
Prints a JSON summary and a pass/fail line: pass if ours is no worse than Python on mean error and failure rate (prod).
"""
import argparse
import collections
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

SET = os.path.join(C.OUT, "known")


def sources():
    d = os.path.join(SET, "src")
    os.makedirs(d, exist_ok=True)
    out = []
    jz = C.frames_dir("jazz")
    for t in (10, 50, 100):
        p = os.path.join(d, f"jazz_{t:05d}.png")
        if not os.path.exists(p):
            cv2.imwrite(p, cv2.imread(os.path.join(jz, f"{t:05d}.png")))
        out.append(p)
    k4 = C.frames_dir("jazz_4k")
    for t in (20, 70, 110):
        p = os.path.join(d, f"jazz1280_{t:05d}.png")
        if not os.path.exists(p):
            cv2.imwrite(p, cv2.resize(cv2.imread(os.path.join(k4, f"{t:05d}.png")), (1280, 720), interpolation=cv2.INTER_AREA))
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true")
    ap.add_argument("--out", default=os.path.join(C.OUT, "known_motion.json"))
    a = ap.parse_args()
    man = os.path.join(SET, "pairs", "manifest.json")
    if a.regen or not os.path.exists(man):
        subprocess.run([os.path.join(C.REL, "motion_testset"), os.path.join(SET, "pairs"), *sources()], check=True)
    pairs = json.load(open(man))
    L = C.Lib()
    rows = []
    for p in pairs:
        fa, fb = cv2.imread(p["a"]), cv2.imread(p["b"])
        ga, gb = cv2.cvtColor(fa, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY)
        W, H = p["W"], p["H"]
        Mt = np.array(p["M"]).reshape(2, 3)
        thr = max(1.0, 0.005 * W)
        py_prod = C.python_camera_pair(fa, fb)
        py_raw, py_inl = C.python_camera_pair_raw(fa, fb)
        if py_raw is None:
            py_raw = np.eye(2, 3)
        o_raw, o_prod, info = L.camera_pair(ga, gb)
        r = {"id": p["id"], "res": W, "motion": p["motion"], "degrade": p["degrade"], "outliers": p["outliers"], "thr": thr}
        for name, m in (("py_prod", py_prod), ("py_raw", py_raw), ("ours_prod", o_prod), ("ours_raw", o_raw)):
            r[name] = C.endpoint_error(m, Mt, W, H)[0]
        r["py_trusted"] = py_inl > 0.4
        r["ours_trusted"] = info["ok"]
        rows.append(r)

    def summarise(sel):
        out = {}
        for name in ("ours_prod", "py_prod", "ours_raw", "py_raw"):
            out[name] = C.stats([r[name] for r in sel], [r[name] > r["thr"] for r in sel])
        out["untrusted"] = {"ours": round(float(np.mean([not r["ours_trusted"] for r in sel])), 4),
                            "py": round(float(np.mean([not r["py_trusted"] for r in sel])), 4)}
        return out

    summary = {"pairs": len(rows), "all": summarise(rows), "all_but_zoom": summarise([r for r in rows if r["motion"] != "zoom"])}
    for key in ("motion", "degrade", "outliers", "res"):
        groups = collections.defaultdict(list)
        for r in rows:
            groups[str(r[key])].append(r)
        summary[f"by_{key}"] = {k: summarise(v) for k, v in groups.items()}
    json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)
    s = summary["all"]
    ok = s["ours_prod"]["mean"] <= s["py_prod"]["mean"] and s["ours_prod"]["fail_rate"] <= s["py_prod"]["fail_rate"]
    print(json.dumps(summary, indent=1))
    print(f"{'PASS' if ok else 'FAIL'} known_motion: ours prod mean {s['ours_prod']['mean']} px, fail {s['ours_prod']['fail_rate']}; "
          f"python prod mean {s['py_prod']['mean']} px, fail {s['py_prod']['fail_rate']} ({len(rows)} pairs)")


if __name__ == "__main__":
    main()
