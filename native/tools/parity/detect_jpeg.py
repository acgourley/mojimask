"""The project's frames: mm-engine's JPEG codec (crates/mm-engine/src/jpeg.rs) against cv2's (libjpeg-turbo).

    .venv/bin/python native/tools/parity/detect_jpeg.py [--frames N]

On jazz (public domain): every frame as video_io.frames gives it (the "detect" colour, what extract_jpegs writes)
  - encode: Rust's file == cv2.imwrite(quality 90)'s, byte for byte
  - decode: Rust's pixels == cv2.imread's, for those files
plus odd sizes cut from them (every width/height mod 16) and grey and noisy pictures, so the edges and dummy blocks are
covered. Prints JSON and a PASS/FAIL line.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.dirname(NATIVE)
sys.path.insert(0, ROOT)
from mojimask import video_io  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=0, help="only the first N frames (0: all)")
a = ap.parse_args()
assert subprocess.run(["cargo", "build", "-q", "-p", "mm-engine", "--bin", "mm-scan-tool"], cwd=NATIVE).returncode == 0
TOOL = os.path.join(NATIVE, "target", "debug", "mm-scan-tool")
tmp = tempfile.mkdtemp(dir=os.path.join(ROOT, "out", "native"))

video = os.path.join(ROOT, "test_clips", "jazz.mp4")
pics = []
for i, f in enumerate(video_io.frames(video)):
    if a.frames and i >= a.frames:
        break
    pics.append(("jazz_%05d" % i, f))
rng = np.random.default_rng(0)
f0 = pics[0][1]
for dh in range(16):
    for dw in (0, 1, 7, 9, 15):
        pics.append((f"cut_{dh}_{dw}", np.ascontiguousarray(f0[: 33 + dh, : 40 + dw])))
pics.append(("noise", rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)))
pics.append(("flat", np.full((17, 17, 3), 200, np.uint8)))
pics.append(("tiny", rng.integers(0, 256, (1, 1, 3), dtype=np.uint8)))

enc_bad, dec_bad, n = [], [], 0
for name, img in pics:
    raw = os.path.join(tmp, name + ".raw")
    img.tofile(raw)
    rs = os.path.join(tmp, name + "_rs.jpg")
    subprocess.run([TOOL, "jpeg-enc", raw, str(img.shape[1]), str(img.shape[0]), rs], check=True)
    py = os.path.join(tmp, name + "_py.jpg")
    cv2.imwrite(py, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if open(rs, "rb").read() != open(py, "rb").read():
        enc_bad.append(name)
    out = os.path.join(tmp, name + "_dec.raw")
    subprocess.run([TOOL, "jpeg-dec", py, out], check=True, capture_output=True)
    ref = cv2.imread(py)
    got = np.fromfile(out, np.uint8).reshape(ref.shape)
    d = np.abs(got.astype(int) - ref)
    if d.max():
        dec_bad.append([name, int((d > 0).sum()), int(d.max())])
    n += 1
for f in os.listdir(tmp):
    os.remove(os.path.join(tmp, f))
os.rmdir(tmp)
ok = not enc_bad and not dec_bad
print(json.dumps({"pictures": n, "encode_not_identical": enc_bad[:20], "n_encode_bad": len(enc_bad),
                  "decode_diff [name, pixels, max]": dec_bad[:20], "n_decode_bad": len(dec_bad)}, indent=1))
print("PASS" if ok else "FAIL")
