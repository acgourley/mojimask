"""Export parity: the Python export without covers (video_io frames -> formats.fit_frame -> video_io.writer ->
video_io.mux, see _py_export.py) vs mm-media's export_plain (GStreamer decode, convert.rs, formats.rs CPU fit,
vtenc + mp4mux, then mp4.rs: the source's audio and the metadata) on jazz clips and presets. Checks, each exact
unless noted:
- size, frame count, frame rate; duration within one frame;
- audio: packet count, each packet's bytes (md5) and the first packet's pts (the priming skip) equal Python's export
  and the source's packets in order (all 250 on jazz, priming included);
- metadata: software and description as Python writes them (udta swre/dscp);
- pictures: PSNR of frames 0, middle and last between the two exports (both lossy H.264: >= 35 dB);
- cancel removes the partial files; a photo export writes a JPEG with EXIF Software/ImageDescription like Python's.
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image

from _media import OUT, ROOT, build, psnr, report, rust

PY = sys.executable
build()
d = os.path.join(OUT, "export")
os.makedirs(d, exist_ok=True)


def ffprobe(path, *args):
    return subprocess.run(["ffprobe", "-v", "error", *args, path], capture_output=True, text=True, check=True).stdout


def streams(path):
    j = json.loads(ffprobe(path, "-show_entries", "stream=codec_type,width,height,r_frame_rate,nb_frames,duration,start_time", "-of", "json"))
    return {s["codec_type"]: s for s in j["streams"]}


def packets(path):
    out = ffprobe(path, "-select_streams", "a", "-show_packets", "-show_data_hash", "md5", "-show_entries",
                  "packet=pts,size,data_hash", "-of", "json")
    return [(p["size"], p["data_hash"], int(p["pts"])) for p in json.loads(out)["packets"]]


def udta(path):
    b = open(path, "rb").read()
    out = {}
    for tag in (b"swre", b"dscp"):
        i = b.find(tag)
        if i >= 0:
            size = int.from_bytes(b[i - 4:i], "big")
            out[tag.decode()] = b[i + 4 + 6:i - 4 + size].rstrip(b"\0").decode()
    return out


def frames(path, idx):
    """Frames by index, decoded by ffmpeg (bgr24); the project's OpenCV has no video reading."""
    st = streams(path)["video"]
    w, h = int(st["width"]), int(st["height"])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "bgr24", "-fps_mode", "passthrough", "-"],
                         capture_output=True, check=True).stdout
    n = len(raw) // (w * h * 3)
    return {i: np.frombuffer(raw, np.uint8, w * h * 3, i * w * h * 3).reshape(h, w, 3) for i in idx if i < n}


# formats.settings for every preset x fit (x cut) on every parity clip and a photo: exact
fmt_bad = []
for clip in ["out/poc/clips/jazz.mp4", "out/poc/clips/jazz_4k.mp4", "out/poc/clips/jazz_hlg10.mov", "out/poc/clips/jazz_rot90.mp4",
             "out/poc/clips/jazz_vfr.mp4", os.path.join(OUT, "photos", "jazz_o6.jpg")]:
    if not os.path.exists(clip):
        continue
    from mojimask import formats, video_io
    meta = video_io.probe(clip)
    got = iter(rust("formats", clip))
    for pr in formats.PRESETS:
        for fit in formats.FITS:
            for st, en in [(None, None), (1.0, 3.5)]:
                py, rs = formats.settings(meta, pr["id"], fit, st, en), next(got)
                if any((abs(py[k] - rs[k]) > 1e-9) if isinstance(py[k], float) else py[k] != rs[k] for k in py):
                    fmt_bad.append({"clip": clip, "py": py, "rs": rs})

cases = [("jazz_4k", "out/poc/clips/jazz_4k.mp4", "original", "fill", []),
         ("jazz", "test_clips/jazz.mp4", "original", "fill", []),
         ("jazz_vertical_fit", "test_clips/jazz.mp4", "vertical", "fit", []),
         ("jazz_square_cut", "test_clips/jazz.mp4", "square", "fill", ["1.0", "3.5"]),
         ("jazz_rot90", "out/poc/clips/jazz_rot90.mp4", "original", "fill", [])]
rows, ok = [], True
for name, clip, preset, fit, cut in cases:
    pyo, rso = os.path.join(d, f"py_{name}.mp4"), os.path.join(d, f"rs_{name}.mp4")
    pys = json.loads(subprocess.run([PY, os.path.join(ROOT, "native/tools/parity/_py_export.py"), clip, pyo, preset, fit, *cut],
                                    capture_output=True, text=True, check=True, env=dict(os.environ, PYTHONPATH=ROOT)).stdout.strip().splitlines()[-1])
    rs = rust("export", rso, *[]) if False else rust("export", clip, rso, preset, fit, *cut)
    a, b = streams(pyo), streams(rso)
    src_pk, py_pk, rs_pk = packets(clip), packets(pyo), packets(rso)
    v_fps = eval(a["video"]["r_frame_rate"])
    r = {"case": name, "settings_same": {k: pys[k] == rs["settings"][k] for k in pys},
         "size": [b["video"]["width"], b["video"]["height"]], "size_same": (a["video"]["width"], a["video"]["height"]) == (b["video"]["width"], b["video"]["height"]),
         "frames": [int(a["video"]["nb_frames"]), int(b["video"]["nb_frames"])],
         "rate": [a["video"]["r_frame_rate"], b["video"]["r_frame_rate"]],
         "duration": [float(a["video"]["duration"]), float(b["video"]["duration"])],
         "audio_packets": [len(src_pk), len(py_pk), len(rs_pk)],
         "audio_bytes_same_as_python": [p[:2] for p in py_pk] == [p[:2] for p in rs_pk],
         "audio_first_pts": [py_pk[0][2] if py_pk else None, rs_pk[0][2] if rs_pk else None],
         "audio_in_source_order": any([p[:2] for p in src_pk[i:i + len(rs_pk)]] == [p[:2] for p in rs_pk] for i in range(len(src_pk))),
         "metadata": [udta(pyo), udta(rso)]}
    n = int(a["video"]["nb_frames"])
    fa, fb = frames(pyo, {0, n // 2, n - 1}), frames(rso, {0, n // 2, n - 1})
    r["psnr"] = {k: round(psnr(fa[k], fb[k]), 2) for k in fa if k in fb and fa[k].shape == fb[k].shape}
    good = (all(r["settings_same"].values()) and r["size_same"] and r["frames"][0] == r["frames"][1] and r["rate"][0] == r["rate"][1]
            and abs(r["duration"][0] - r["duration"][1]) <= 1 / v_fps and r["audio_packets"][1] == r["audio_packets"][2]
            and r["audio_bytes_same_as_python"] and r["audio_first_pts"][0] == r["audio_first_pts"][1] and r["audio_in_source_order"]
            and r["metadata"][0] == r["metadata"][1] and len(r["psnr"]) == 3 and min(r["psnr"].values()) >= 35)
    r["pass"] = good
    ok &= good
    rows.append(r)

# cancel
c = rust("export-cancel", "test_clips/jazz.mp4", os.path.join(d, "rs_cancelled.mp4"))
ok &= c["cancelled"] and not c["out_exists"] and not c["tmp_exists"]
# photo export
photo = os.path.join(OUT, "photos", "jazz_o6.jpg")
photo_row = {}
if os.path.exists(photo):
    pyo, rso = os.path.join(d, "py_photo.jpg"), os.path.join(d, "rs_photo.jpg")
    subprocess.run([PY, os.path.join(ROOT, "native/tools/parity/_py_export.py"), photo, pyo, "square", "fill"], check=True,
                   capture_output=True, env=dict(os.environ, PYTHONPATH=ROOT))
    rust("export", photo, rso, "square", "fill")
    ea, eb = Image.open(pyo).getexif(), Image.open(rso).getexif()
    ia, ib = cv2.imread(pyo), cv2.imread(rso)
    photo_row = {"size": [ia.shape[:2], ib.shape[:2]], "exif_python": {k: ea.get(k) for k in (0x0131, 0x010E)},
                 "exif_rust": {k: eb.get(k) for k in (0x0131, 0x010E)},
                 "psnr": round(psnr(ia, ib), 2) if ia.shape == ib.shape else None}
    ok &= photo_row["size"][0] == photo_row["size"][1] and photo_row["exif_python"] == photo_row["exif_rust"] and (photo_row["psnr"] or 0) >= 35
else:
    photo_row = {"skipped": "run media_photos.py first"}
ok &= not fmt_bad
sys.exit(report("media_export", ok, formats_mismatches=fmt_bad, cases=rows, cancel=c, photo=photo_row))
