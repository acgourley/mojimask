"""Our camera motion on the real jazz clips, against Python's mats_v2.npy (agreement, not equality), plus speed.

For jazz (643 x 476) and jazz_4k (3840 x 2160; also resized to 1280 x 720) decoded by Python's video_io:
  agreement   mean distance (px, source pixels) between where our mats and Python's put a grid over the frame,
              on the pairs Python trusted (not the identity it writes when it can't tell)
  alignment   with no truth to hand, how well each set of mats lines the frames up: frame t-1 warped by the mat onto
              frame t, the median absolute grey difference over the picture (the background; people moving on their
              own are most of the rest), averaged over pairs. Lower is better. The identity is the floor to beat.
  speed       ms per frame pair (prepare = grey + pyramid of one frame; estimate = the pair), one thread
    python native/tools/motion/jazz_mats.py
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402


def frames_1280():
    d = os.path.join(C.OUT, "frames", "jazz_1280")
    if not os.path.exists(os.path.join(d, "00000.png")):
        os.makedirs(d, exist_ok=True)
        src = C.frames_dir("jazz_4k")
        for f in sorted(os.listdir(src)):
            cv2.imwrite(os.path.join(d, f), cv2.resize(cv2.imread(os.path.join(src, f)), (1280, 720), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return d


def alignment(fdir, mats, scale):
    errs = []
    names = sorted(os.listdir(fdir))
    prev = None
    for t, n in enumerate(names):
        g = cv2.cvtColor(cv2.imread(os.path.join(fdir, n)), cv2.COLOR_BGR2GRAY)
        if scale != 1:
            g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if prev is not None:
            M = mats[t].copy()
            M[:, 2] *= scale
            H, W = g.shape
            w = cv2.warpAffine(prev, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=-1)
            valid = cv2.warpAffine(np.ones_like(prev), M, (W, H), flags=cv2.INTER_NEAREST) > 0
            d = np.abs(w.astype(np.float32) - g.astype(np.float32))[valid]
            errs.append(float(np.median(d)))
        prev = g
    return float(np.mean(errs))


def main():
    out = {}
    ok = True
    for clip, fdir, py_mats, scale in (("jazz", C.frames_dir("jazz"), os.path.join(C.ROOT, "out/poc/projects/jazz/mats_v2.npy"), 1),
                                       ("jazz_1280", frames_1280(), None, 1),
                                       ("jazz_4k", C.frames_dir("jazz_4k"), os.path.join(C.ROOT, "out/poc/projects/jazz_4k/mats_v2.npy"), 0.25)):
        npy = os.path.join(C.OUT, f"{clip}_ours.npy")
        diag = os.path.join(C.OUT, f"{clip}_ours.json")
        r = subprocess.run([os.path.join(C.REL, "motion_mats"), fdir, npy, "--diag", diag], check=True, capture_output=True, text=True)
        speed = json.loads(r.stdout)
        ours = np.load(npy)
        g0 = cv2.imread(os.path.join(fdir, "00000.png"))
        H, W = g0.shape[:2]
        rec = {"size": [W, H], "pairs": len(ours) - 1, "ours_untrusted": speed["untrusted"],
               "ms_per_pair": round(speed["ms_per_pair"], 2), "prepare_ms_per_frame": round(speed["prep_ms_per_frame"], 2),
               "estimate_ms_per_pair": round(speed["est_ms_per_pair"], 2),
               "alignment_ours": round(alignment(fdir, ours, scale), 3),
               "alignment_identity": round(alignment(fdir, np.array([np.eye(2, 3)] * len(ours)), scale), 3)}
        if py_mats:
            py = np.load(py_mats)
            ident = np.array([np.allclose(m, np.eye(2, 3)) for m in py[1:]])
            e = np.array([C.endpoint_error(ours[t], py[t], W, H, step=max(W, H) // 32)[0] for t in range(1, len(py))])
            rec.update({"python_identity_pairs": int(ident.sum()),
                        "agreement_px": {"n": int((~ident).sum()), "mean": round(float(e[~ident].mean()), 3) if (~ident).any() else None,
                                         "p95": round(float(np.percentile(e[~ident], 95)), 3) if (~ident).any() else None,
                                         "max": round(float(e[~ident].max()), 3) if (~ident).any() else None},
                        "alignment_python": round(alignment(fdir, py, scale), 3)})
            ok &= rec["alignment_ours"] <= rec["alignment_python"] + 1e-3
        out[clip] = rec
    json.dump(out, open(os.path.join(C.OUT, "jazz_mats.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    print(f"{'PASS' if ok else 'FAIL'} jazz_mats: ours lines frames up at least as well as Python's mats on jazz and jazz_4k")


if __name__ == "__main__":
    main()
