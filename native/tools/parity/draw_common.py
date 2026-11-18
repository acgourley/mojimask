"""Shared bits of the draw stream's parity scripts (draw_*.py): paths, running mm-render-ref, PSNR."""
import json
import math
import os
import subprocess
import sys

import numpy as np

NATIVE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROOT = os.path.dirname(NATIVE)
REF = os.path.join(ROOT, "out", "native", "draw", "ref")
RUST = os.path.join(ROOT, "out", "native", "draw", "rust")
BIN = os.path.join(NATIVE, "target", "debug", "mm-render-ref")


def build():
    subprocess.run(["cargo", "build", "-q", "-p", "mm-render", "--bin", "mm-render-ref"], cwd=NATIVE, check=True)


def run(*args):
    subprocess.run([BIN, "--ref", REF, "--out", RUST, *args], check=True)


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else min(99.0, 10 * math.log10(255.0 ** 2 / mse))


def report(name, results, ok):
    out = {"test": name, "pass": bool(ok), **results}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"{name}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
