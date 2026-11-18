"""Shared bits of the media parity scripts (media_*.py): paths, the clips, building and running mm-media-parity.

Run every media_*.py with the project's Python (it imports mojimask, the frozen parity target):
    .venv/bin/python native/tools/parity/media_probe.py
"""
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NATIVE = os.path.join(ROOT, "native")
OUT = os.path.join(ROOT, "out", "native", "media")
BIN = os.path.join(NATIVE, "target", "release", "mm-media-parity")
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def clips(extra=True):
    """The parity clips: every clip in out/poc/clips, test_clips/jazz.mp4, and (extra) the synthetic colour clips
    made by media_decode.py --make-colour (random flat YUV blocks with each colour tag; no people)."""
    c = sorted(glob.glob("out/poc/clips/*.mp4") + glob.glob("out/poc/clips/*.mov")) + ["test_clips/jazz.mp4"]
    if extra:
        c += sorted(glob.glob("out/native/media/colour/*.mp4") + glob.glob("out/native/media/colour/*.mov"))
    return c


def build():
    env = dict(os.environ, PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig")
    subprocess.run(["cargo", "build", "--release", "-q", "-p", "mm-media", "--bin", "mm-media-parity"], cwd=NATIVE, env=env, check=True)


def rust(*args):
    r = subprocess.run([BIN, *map(str, args)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"mm-media-parity {' '.join(map(str, args))}: {r.stderr.strip()[-2000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def report(name, ok, **kw):
    print(json.dumps({"test": name, **kw}, default=str))
    print(json.dumps({"test": name, "pass": bool(ok)}))
    return 0 if ok else 1


def psnr(a, b):
    import numpy as np
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    m = d.mean()
    return 99.0 if m == 0 else float(10 * np.log10(255 ** 2 / m))
