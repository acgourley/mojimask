"""mm-core's Python- and numpy-exact arithmetic against Python and numpy themselves (fixes stream):

- pyjson's float repr (`pyjson::float_repr`, which every project file and the scan's recording keys go through)
  against Python's `repr`, on float32-derived values (exact ties at the shortest length are common there) and plain
  float64s across the exponent range. Pass: 0 differences.
- edits' `warp_cs` and `compose` (track.warp_cs / track.compose: numpy's dgemv, dgemm and det) against numpy, bit for
  bit, on camera-like matrices.

    cd native && cargo build -p mm-core --bin core_numcheck
    PYTHONPATH=.. ../../../../.venv/bin/python tools/parity/core_numcheck.py [--n 100000]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.abspath(os.path.join(HERE, "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100000, help="float32-derived values for the repr check")
a = ap.parse_args()
rng = np.random.default_rng(11)

f32 = []
k = a.n // 5
f32 += [float(v) for v in rng.uniform(-5000, 5000, k).astype(np.float32)]
f32 += [float(v) for v in rng.uniform(0, 1, k).astype(np.float32)]
f32 += [float(np.float32(v)) / 3840.0 for v in rng.uniform(0, 1, k)]
f32 += [float(v) for v in (rng.normal(0, 1, k) * 10.0 ** rng.integers(-30, 30, k)).astype(np.float32)]
bits = rng.integers(0, 2**32, 2 * (a.n - 4 * k), dtype=np.uint64).astype(np.uint32).view(np.float32)  # any float32
f32 += [float(v) for v in bits[np.isfinite(bits)][:a.n - 4 * k]]
assert len(f32) == a.n
f64 = [float(v) for v in rng.normal(0, 1, 20000) * 10.0 ** rng.integers(-300, 300, 20000)]
f64 += [float(v) for v in rng.integers(0, 2**63, 20000, dtype=np.uint64).view(np.float64) if np.isfinite(v)]


def cam():
    th, s = rng.normal(0, 0.01), 1 + rng.normal(0, 0.01)
    return np.array([[s * np.cos(th), -s * np.sin(th), rng.normal(0, 3)], [s * np.sin(th), s * np.cos(th), rng.normal(0, 3)]])


def compose(mats):
    M = np.eye(3)
    for t in range(1, len(mats)):
        M = np.vstack([mats[t], [0, 0, 1]]) @ M
    return M[:2]


def warp(c, M):  # track.warp_cs
    p = M @ np.array([c[0], c[1], 1.0])
    s = np.sqrt(abs(np.linalg.det(M[:, :2])))
    return np.array([p[0], p[1], c[2] * s])


cases = []  # (kind, op, x, want)
B = 50
for vals, kind in ((f32, "repr_f32"), (f64, "repr_f64")):
    for i in range(0, len(vals), B):
        x = vals[i:i + B]
        cases.append((kind, "repr", x, json.dumps(x)))
for i in range(3000):
    mats = [cam() for _ in range(int(rng.integers(2, 30)))]
    M = compose(mats)
    cases.append(("compose", "compose", [float(v) for m in mats for v in m.ravel()], [float(v) for v in M.ravel()]))
    c = [rng.uniform(0, 3840), rng.uniform(0, 2160), rng.uniform(10, 400)]
    Mw = cam() if i % 2 else M
    cases.append(("warp", "warp", [float(v) for v in Mw.ravel()] + c, [float(v) for v in warp(c, Mw)]))

inp = "\n".join(json.dumps({"op": op, "x": x}) for _, op, x, _ in cases)
r = subprocess.run([os.path.join(NATIVE, "target", "debug", "core_numcheck")], input=inp, capture_output=True, text=True)
if r.returncode:
    print(r.stderr[-2000:])
    sys.exit(1)
outs = [json.loads(x) for x in r.stdout.splitlines()]
assert len(outs) == len(cases)
tot, bad, ex = {}, {}, {}
for (kind, op, x, want), got in zip(cases, outs):
    if op == "repr":
        w, g = want[1:-1].split(", "), got[1:-1].split(", ")  # compare the text, value by value
        n = len(w)
        d = [(a_, b_) for a_, b_ in zip(w, g) if a_ != b_] if want != got else []
        diff = len(d) if len(w) == len(g) else n
    else:
        n, d = 1, ([(want, got)] if want != got else [])
        diff = len(d)
    tot[kind] = tot.get(kind, 0) + n
    if diff:
        bad[kind] = bad.get(kind, 0) + diff
        ex.setdefault(kind, d[:2])
print(json.dumps({"values": tot, "differ": bad, "examples": {k: v[:2] for k, v in ex.items()}}))
print("PASS" if not bad else "FAIL")
