"""Probe parity: video_io.probe (Python, AVFoundation) vs mm_media::probe (the movie's boxes) on every parity clip
and test photo: width, height, fps, duration, video_start, video_duration, rotation, has_audio (and image), plus the
frame count n = round(video_duration * fps). Exact (floats compared to 1e-9)."""
import glob
import sys

from _media import build, clips, report, rust
from mojimask import video_io

build()
paths = clips() + sorted(glob.glob("out/native/media/photos/jazz_*.*"))
got = rust("probe", *paths)
rows, bad = [], 0
for p in paths:
    py = video_io.probe(p)
    py["_n"] = max(1, round(py.get("video_duration", py["duration"]) * py["fps"]))
    rs = got[p]
    diffs = {k: (py[k], rs.get(k)) for k in py
             if not (isinstance(py[k], float) and isinstance(rs.get(k), (int, float)) and abs(py[k] - rs[k]) < 1e-9)
             and py[k] != rs.get(k)}
    bad += bool(diffs)
    rows.append({"clip": p, "same": not diffs, **({"diffs": diffs} if diffs else {"n": py["_n"], "fps": py["fps"]})})
sys.exit(report("media_probe", bad == 0, clips=len(paths), mismatched=bad, rows=rows))
