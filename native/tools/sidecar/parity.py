"""Editor and export parity: the native app's drawn frames against Python's export of the same frames.

    <venv>/bin/python native/tools/sidecar/parity.py editor [--frames 0,40,80,126] [--min 45]
    <venv>/bin/python native/tools/sidecar/parity.py export OUT.mp4 [--frames ...] [--preset original --fit fill --mark]

The app must be running with --dev and a public-domain project open in the Editor (jazz). The reference for frame t
is the project's server drawing it as its export does (native/tools/sidecar/project_server.py,
/api/native/export_frame: video_io's display frame, track.draw_frame, formats.fit_frame, track.watermark).
  editor: each frame is shown in the app (seek), read back from the video item's texture at full size (readback);
  export: OUT.mp4 (the app's export, made by the caller) is decoded and frame t compared. H.264 costs a few dB, so the
          comparison is also made against Python's own export (--py-export, made with POST /api/render) decoded
          the same way, and both against the lossless reference.
Prints JSON, the last line "PASS ..." or "FAIL ...". Pictures go to out/native/ui2/parity/.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MMCTL = os.path.join(ROOT, "native", "tools", "mmctl")
OUT = os.path.join(ROOT, "out", "native", "ui2", "parity")


def mm(*args):
    r = subprocess.run([sys.executable, MMCTL, *args], capture_output=True, text=True)
    try:
        return json.loads(r.stdout)
    except ValueError:
        raise SystemExit(f"mmctl {' '.join(args)}: {r.stdout} {r.stderr}")


def psnr(a, b):
    d = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if d == 0 else 10 * np.log10(255 ** 2 / d)


def ref(base, t, q=""):
    with urllib.request.urlopen(f"{base}/api/native/export_frame?t={t}{q}", timeout=600) as r:
        return cv2.imdecode(np.frombuffer(r.read(), np.uint8), cv2.IMREAD_COLOR)


def frames_of(path):
    """Every frame of a video, decoded as the app decodes (video_io, display colours)."""
    sys.path.insert(0, ROOT)
    from mojimask import video_io
    return list(video_io.frames(path, color="display"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["editor", "export"])
    ap.add_argument("video", nargs="?")
    ap.add_argument("--frames", default="0,40,80,126")
    ap.add_argument("--min", type=float, default=45.0)
    ap.add_argument("--preset", default="original")
    ap.add_argument("--fit", default="fill")
    ap.add_argument("--mark", action="store_true")
    ap.add_argument("--py-export", help="Python's own export of the project (an .mp4), to compare H.264 to H.264")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    st = mm("state", "project")["result"]
    if not st.get("open") or not st.get("public"):
        raise SystemExit("open a public-domain project (jazz) first")
    base = st["base"]
    frames = [int(x) for x in a.frames.split(",")]
    res = {"what": a.what, "project": st["id"], "frames": {}}
    if a.what == "editor":
        for t in frames:
            mm("call", "seek", json.dumps({"t": t}))
            mm("wait", f"store.state.editor.shown === {t}", "--timeout", "20000")
            p = os.path.join(OUT, f"editor_{t:05d}.png")
            if os.path.exists(p):
                os.remove(p)
            mm("call", "readback", json.dumps({"path": p}))
            mm("wait", "store.state.readback.state !== 'pending'", "--timeout", "20000")
            got = cv2.imread(p)
            want = ref(base, t, "&preset=none")
            cv2.imwrite(os.path.join(OUT, f"py_{t:05d}.png"), want)
            res["frames"][t] = round(psnr(got, want), 2)
    else:
        q = f"&preset={a.preset}&fit={a.fit}" + ("&mark=1" if a.mark else "")
        ours = frames_of(a.video)
        py = frames_of(a.py_export) if a.py_export else None
        res["n_frames"] = len(ours)
        for t in frames:
            want = ref(base, t, q)
            got = ours[t]
            cv2.imwrite(os.path.join(OUT, f"export_{t:05d}.png"), got)
            row = {"vs_lossless": round(psnr(got, want), 2)}
            if py is not None:
                row["python_export_vs_lossless"] = round(psnr(py[t], want), 2)
                row["vs_python_export"] = round(psnr(got, py[t]), 2)
            res["frames"][t] = row
    print(json.dumps(res, indent=1))
    if a.what == "export" and a.py_export:
        # H.264 costs both exports the same few dB: ours must lose no more than Python's own export does (0.5 dB)
        gap = max(v["python_export_vs_lossless"] - v["vs_lossless"] for v in res["frames"].values())
        print(f"{'PASS' if gap <= 0.5 else 'FAIL'} worst {gap:.2f} dB below Python's own export (max 0.5)")
        return
    worst = min((v if isinstance(v, float) else v["vs_lossless"]) for v in res["frames"].values())
    print(f"{'PASS' if worst >= a.min else 'FAIL'} worst {worst:.2f} dB (min {a.min})")


if __name__ == "__main__":
    main()
