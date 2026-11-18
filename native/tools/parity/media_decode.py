"""Decode parity: video_io.frames (Python, AVFoundation) vs mm-media (GStreamer + convert.rs, the CPU reference of
the display conversion) on every parity clip at several t: PSNR per frame, for color="display" (target: >= 50 dB
everywhere, plan exit >= 45) and color="detect" (the detectors' frames: share of pixels identical, and PSNR).
Frames are decoded forward from the start; seeking is media_seek.py.

    media_decode.py [--quick]   (--quick: display only, no detect pass)
    media_decode.py --gpu       the GPU display path instead (mm-render's yuv pass on the decoder's planes, imported
                                zero-copy, with import::yuv_params_display): PSNR vs video_io (target >= 57 dB) and
                                vs the CPU reference of the same frame
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np

from _media import NATIVE, OUT, build, clips, psnr, report, rust
from mojimask import video_io

quick = "--quick" in sys.argv
gpu = "--gpu" in sys.argv
build()
if gpu:
    quick = True
    env = dict(os.environ, PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig")
    subprocess.run(["cargo", "build", "--release", "-q", "-p", "mm-media", "--features", "gpu-parity", "--bin", "mm-media-gpu"],
                   cwd=NATIVE, env=env, check=True)


def rust_gpu(clip, d, ts):
    r = subprocess.run([os.path.join(NATIVE, "target", "release", "mm-media-gpu"), clip, d, ",".join(map(str, ts))],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"mm-media-gpu {clip}: {r.stderr.strip()[-2000:]}")
    g = json.loads(r.stdout.strip().splitlines()[-1])
    return {"frames": [dict(f, pts=None) for f in g["frames"]], "after_seek": []}


rows, worst = [], 99.0
detect_rows = []
ok_seek = True
for clip in clips():
    meta = video_io.probe(clip)
    n = max(1, round(meta["video_duration"] * meta["fps"]))
    ts = sorted({0, 1, 2, 10, n // 3, 50, 77, 100, n - 2, n - 1} & set(range(n)))
    name = os.path.splitext(os.path.basename(clip))[0]
    for colour in (["display"] if quick else ["display", "detect"]):
        d = os.path.join(OUT, "decode_gpu" if gpu else "decode", name, colour)
        got = rust_gpu(clip, d, ts) if gpu else rust("decode", clip, d, colour, ",".join(map(str, ts)))
        fwd = {r["t"]: r for r in got["frames"]}
        for r in got["after_seek"]:
            f = fwd[r["t"]]
            if r["data_md5"] != f["data_md5"] or abs(r["pts"] - f["pts"]) > 1e-6:
                ok_seek = False
                rows.append({"clip": name, "t": r["t"], "seek_mismatch": {"fwd": f, "seek": r}})
        want = set(ts)
        for t, img in enumerate(video_io.frames(clip, meta, color=colour)):
            if t not in want:
                continue
            mine = cv2.imread(os.path.join(d, f"{t:05d}.png"))
            if mine is None or mine.shape != img.shape:
                rows.append({"clip": name, "t": t, "colour": colour, "error": f"shape {None if mine is None else mine.shape} vs {img.shape}"})
                worst = 0
                continue
            p = psnr(mine, img)
            if colour == "display":
                worst = min(worst, p)
                rows.append({"clip": name, "t": t, "psnr": round(p, 2), "pts": fwd[t]["pts"], "zero_copy": fwd[t]["zero_copy"],
                             **({"vs_cpu_db": round(fwd[t]["vs_cpu_db"], 2), "vs_cpu_identical_pct": round(fwd[t]["identical_pct"], 3)} if gpu else {})})
            else:
                detect_rows.append({"clip": name, "t": t, "psnr": round(p, 2),
                                    "identical_pct": round(float((mine == img).all(-1).mean() * 100), 3),
                                    "max_diff": int(np.abs(mine.astype(int) - img).max())})
by_clip = {}
for r in rows:
    if "psnr" in r:
        c = by_clip.setdefault(r["clip"], [99, 0, 0])
        c[0] = min(c[0], r["psnr"]); c[1] += r["psnr"]; c[2] += 1
summary = {k: {"min_db": v[0], "mean_db": round(v[1] / v[2], 2)} for k, v in by_clip.items()}
if gpu:
    for k in summary:
        cr = [r for r in rows if r.get("clip") == k and "vs_cpu_db" in r]
        summary[k]["min_vs_cpu_db"] = min(r["vs_cpu_db"] for r in cr)
        summary[k]["min_vs_cpu_identical_pct"] = min(r["vs_cpu_identical_pct"] for r in cr)
        summary[k]["zero_copy"] = all(r["zero_copy"] for r in cr)
det = {}
for r in detect_rows:
    c = det.setdefault(r["clip"], [100, 99, 0])
    c[0] = min(c[0], r["identical_pct"]); c[1] = min(c[1], r["psnr"]); c[2] = max(c[2], r["max_diff"])
os.makedirs(OUT, exist_ok=True)
details = os.path.join(OUT, "decode_gpu_parity.json" if gpu else "decode_parity.json")
json.dump({"display": rows, "detect": detect_rows}, open(details, "w"), indent=1)
sys.exit(report("media_decode_gpu" if gpu else "media_decode", worst >= (57 if gpu else 50), min_display_db=round(worst, 2),
                display=summary, detect={k: {"min_identical_pct": v[0], "min_db": v[1], "max_diff": v[2]} for k, v in det.items()},
                details=details))
