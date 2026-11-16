"""Finished research: the comparison made before the scan switched from SCRFD to RetinaFace (2026-11-14), when the
scan's face-check thresholds were on SCRFD's scale. The app no longer reads these maps (only swap_bench's "-mapped"
variants do); kept as the record of that switch.

Face-check scores of SCRFD and YuNet on the jazz census's faces and not-faces (2026-11-14): how well each tells
a face from a not-face (AUC), and a quantile map from YuNet's score to SCRFD's scale, so the scan's fixed
face-check thresholds (blips.FACE_OK/SURE_FACE, faces.FRONTAL score, cover.py's 0.5) mean the same with YuNet.
Writes out/face_compare/<det>_to_scrfd.json ({"<det>": [...], "scrfd": [...]}: matching quantiles, for np.interp),
for each other detector (yunet, retina).

    .venv/bin/python experiments/face_score_calibrate.py            (fit the map on the jazz census)
    .venv/bin/python experiments/face_score_calibrate.py twirl      (check it on a clip's old grade, every 4th frame:
                                                                     "P.." faces, "NF:" not-faces; numbers only)
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import face_model_compare as FMC  # noqa: E402


def map_path(det):
    return os.path.join(FMC.ROOT, "out", "face_compare", f"{det}_to_scrfd.json")


def main(CLIP=None):
    if "scrfd" not in FMC.DETECTORS:
        raise SystemExit("Needs SCRFD among the detectors (unset DETECTORS, or include scrfd): every map is onto its scale.")
    if CLIP:  # an old position grade: [x1, y1, x2, y2, label] fractions per frame
        g = json.load(open(os.path.join(FMC.ROOT, "out", "probe", f"gt_{CLIP}.json")))
        frames_dir = next(d for d in (os.path.join(FMC.ROOT, "out", "bench", CLIP, "frames"),
                                      os.path.join(FMC.ROOT, "out", "bench_swap", "_dets", CLIP, "frames")) if os.path.isdir(d))
        W, H = g["space"]["w"], g["space"]["h"]
        rows = {t: [{"b": b[:4], "c": "face" if str(b[4]).startswith("P") else "not" if str(b[4]).startswith("NF") else "?"}
                    for b in bs] for t, bs in g["frames"].items() if int(t) % 4 == 0}
    else:
        truth = json.load(open(FMC.TRUTH))
        W, H, frames_dir, rows = truth["w"], truth["h"], FMC.FRAMES, truth["frames"]
    S = {d: {"face": [], "not": []} for d in FMC.DETECTORS}
    for t, bs in rows.items():
        img = cv2.imread(os.path.join(frames_dir, f"{int(t):05d}.jpg"))
        for b in bs:
            if b["c"] not in ("face", "not"):
                continue
            x1, y1, x2, y2 = b["b"][0] * W, b["b"][1] * H, b["b"][2] * W, b["b"][3] * H
            c = [(x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * FMC.HEAD_PER_FACE]
            crop, k, cx, cy = FMC.crop_of(img, c)
            for det in S:
                f = FMC.measure(det, crop, k, cx, cy, c)
                S[det][b["c"]].append(f["score"] if f else 0.0)
    n_face, n_not = len(S["scrfd"]["face"]), len(S["scrfd"]["not"])
    print(f"census boxes: {n_face} faces, {n_not} not-faces")
    for det, v in S.items():
        f, nf = np.array(v["face"]), np.array(v["not"])
        line = f"  {det:6} AUC face vs not {FMC.auc(f, nf):.3f}"
        for thr in (0.5, 0.6, 0.8):
            line += f" | >= {thr}: faces {np.mean(f >= thr):.0%} not-faces {np.mean(nf >= thr):.0%}"
        print(line)
    OTHERS = [d for d in FMC.DETECTORS if d != "scrfd"]
    if CLIP:  # check the jazz maps here
        for d in OTHERS:
            m = json.load(open(map_path(d)))
            S[d + "_mapped"] = {lab: (np.interp(S[d][lab], m[d], m["scrfd"]) * (np.array(S[d][lab]) > 0)).tolist()
                                for lab in ("face", "not")}
        for det in ["scrfd"] + [x for d in OTHERS for x in (d, d + "_mapped")]:
            f, nf = np.array(S[det]["face"]), np.array(S[det]["not"])
            print(f"  {det:13} found faces {np.mean(f > 0):.0%} not-faces {np.mean(nf > 0):.0%}"
                  + "".join(f" | >= {t}: faces {np.mean(f >= t):.0%} not {np.mean(nf >= t):.0%}" for t in (0.5, 0.6)))
        return
    # quantile maps over every box where the detector found something (faces and not-faces pooled)
    q = np.linspace(0, 1, 101)
    ss = np.sort([s for s in S["scrfd"]["face"] + S["scrfd"]["not"] if s > 0])
    for d in OTHERS:
        ys = np.sort([s for s in S[d]["face"] + S[d]["not"] if s > 0])
        xq, yq = np.quantile(ys, q), np.quantile(ss, q)
        xu = np.unique(xq)  # saturated scores (RetinaFace: many at ~1.0) tie: one point per value, the mean target
        m = {d: xu.tolist(), "scrfd": [float(yq[xq == x].mean()) for x in xu]}
        json.dump(m, open(map_path(d), "w"))
        mf = np.interp(S[d]["face"], m[d], m["scrfd"]) * (np.array(S[d]["face"]) > 0)
        mn = np.interp(S[d]["not"], m[d], m["scrfd"]) * (np.array(S[d]["not"]) > 0)
        line = f"  {d} mapped to scrfd's scale"
        for thr in (0.5, 0.6, 0.8):
            line += f" | >= {thr}: faces {np.mean(mf >= thr):.0%} not-faces {np.mean(mn >= thr):.0%}"
        print(line)
        print(f"  map ({d} -> scrfd):", ", ".join(f"{a:.2f}->{b:.2f}" for a, b in zip(m[d][::20], m["scrfd"][::20])))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
