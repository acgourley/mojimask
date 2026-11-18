"""Photo parity: video_io.read_image / probe (Python, ImageIO) vs mm_media::photo (Rust, ImageIO through objc2) on
jazz test photos made here: frame 30 of jazz (video_io display frame) written as JPEG with each EXIF orientation
(1, 3, 6, 8: as is, 180, 90 CW, 90 CCW), as PNG, and as HEIC (sips, from the orientation-6 JPEG). Pictures must be
upright and match (PSNR >= 50 dB, expected identical), and probe() must agree exactly."""
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image

from _media import OUT, build, psnr, report, rust
from mojimask import video_io

build()
d = os.path.join(OUT, "photos")
os.makedirs(d, exist_ok=True)
frame = None
for t, f in enumerate(video_io.frames("test_clips/jazz.mp4", color="display")):
    if t == 30:
        frame = f
        break
rgb = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))
# a pixel file stored turned, plus the tag that turns it back (PIL's transpose = what the tag undoes)
turn = {1: None, 3: Image.Transpose.ROTATE_180, 6: Image.Transpose.ROTATE_90, 8: Image.Transpose.ROTATE_270}
paths = []
for o, tr in turn.items():
    img = rgb.transpose(tr) if tr is not None else rgb
    exif = Image.Exif()
    exif[0x0112] = o
    p = os.path.join(d, f"jazz_o{o}.jpg")
    img.save(p, "JPEG", quality=95, exif=exif)
    paths.append(p)
p = os.path.join(d, "jazz.png")
rgb.save(p, "PNG")
paths.append(p)
heic = os.path.join(d, "jazz_o6.heic")
subprocess.run(["sips", "-s", "format", "heic", paths[2], "--out", heic], capture_output=True, check=True)
paths.append(heic)

rows, ok = [], True
for p in paths:
    py = video_io.read_image(p)
    out = p + ".rust.png"
    rust("photo", p, out)
    rs = cv2.imread(out)
    pp, rp = video_io.probe(p), rust("probe", p)[p]
    same_probe = all(abs(pp[k] - rp[k]) < 1e-9 if isinstance(pp[k], float) else pp[k] == rp[k] for k in pp)
    upright = abs(psnr(cv2.resize(py, (frame.shape[1], frame.shape[0])), frame)) if py.shape == frame.shape else None
    r = {"photo": os.path.basename(p), "size": list(py.shape[:2]), "probe_same": same_probe,
         "psnr_rust_vs_python": round(psnr(rs, py), 2) if rs is not None and rs.shape == py.shape else None,
         "python_upright_vs_frame_db": None if upright is None else round(upright, 1)}
    ok &= same_probe and r["psnr_rust_vs_python"] is not None and r["psnr_rust_vs_python"] >= 50
    rows.append(r)
sys.exit(report("media_photos", ok, rows=rows))
