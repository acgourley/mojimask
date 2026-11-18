"""mm-scan's numpy arithmetic (crates/mm-scan/src/np.rs) against numpy itself, bit for bit, on random inputs shaped
like the scan's (camera matrices near the identity, head boxes in pixels, unit embeddings).

    .venv/bin/python native/tools/parity/scan_numcheck.py [--n 3000]
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
ap.add_argument("--n", type=int, default=3000)
a = ap.parse_args()
rng = np.random.default_rng(7)


def cam():
    th, s = rng.normal(0, 0.01), 1 + rng.normal(0, 0.01)
    return np.array([[s * np.cos(th), -s * np.sin(th), rng.normal(0, 3)], [s * np.sin(th), s * np.cos(th), rng.normal(0, 3)]])


def compose_like():
    M = np.eye(3)
    for _ in range(rng.integers(1, 30)):
        M = np.vstack([cam(), [0, 0, 1]]) @ M
    return M


cases = []
for i in range(a.n):
    M = cam() if i % 2 else compose_like()[:2]
    c = np.array([rng.uniform(0, 640), rng.uniform(0, 480), rng.uniform(10, 150)])
    cases.append(("gemv23", list(M.ravel()) + [c[0], c[1], 1.0], lambda x: np.array(x[:6]).reshape(2, 3) @ np.array(x[6:9])))
    W = lambda x: list(np.array(x[:6]).reshape(2, 3) @ np.array([x[6], x[7], 1.0])) + [x[8] * np.sqrt(abs(np.linalg.det(np.array(x[:6]).reshape(2, 3)[:, :2])))]  # noqa: E731
    cases.append(("scan_warp", list(M.ravel()) + list(c), W))
    cases.append(("mmcore_warp", list(M.ravel()) + list(c), W))  # mm-core's own (edits.rs), for the report
    A = compose_like()
    cases.append(("gemv33", list(A.ravel()) + [c[0], c[1], 1.0], lambda x: np.array(x[:9]).reshape(3, 3) @ np.array(x[9:12])))
    B = np.vstack([cam(), [0, 0, 1]])
    cases.append(("matmul33", list(B.ravel()) + list(A.ravel()), lambda x: np.array(x[:9]).reshape(3, 3) @ np.array(x[9:]).reshape(3, 3)))
    cases.append(("det2", list(M[:, :2].ravel()), lambda x: np.linalg.det(np.array(x).reshape(2, 2))))
    cases.append(("inv33", list(A.ravel()), lambda x: np.linalg.inv(np.array(x).reshape(3, 3))))
    n = int(rng.integers(1, 300))
    v = list(rng.normal(100, 50, n))
    cases.append(("sum", v, lambda x: np.array(x).sum()))
    cases.append(("mean", v, lambda x: np.array(x).mean()))
    cases.append(("median", v, lambda x: np.median(x)))
    cases.append(("p90", v, lambda x: np.percentile(x, 90)))
    cases.append(("hypot", list(rng.normal(0, 50, 2)), lambda x: np.hypot(x[0], x[1])))
    k = int(rng.integers(2, 5))
    w = list(rng.normal(0, 30, k))
    cases.append(("norm", w, lambda x: np.linalg.norm(np.array(x))))
    cases.append(("ddot", w + list(rng.normal(0, 30, k)), lambda x: np.array(x[:len(x) // 2]) @ np.array(x[len(x) // 2:])))
    e1 = rng.normal(0, 1, 128).astype(np.float32)
    e2 = rng.normal(0, 1, 128).astype(np.float32)
    e1, e2 = e1 / np.linalg.norm(e1), e2 / np.linalg.norm(e2)
    cases.append(("sdot", [float(x) for x in e1] + [float(x) for x in e2], lambda x: np.array(x[:128], np.float32) @ np.array(x[128:], np.float32)))
    rows = int(rng.integers(2, 40))
    d = 128 if i % 2 else 512
    E = rng.normal(0, 1, (rows, d)).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)

    def mr(x):
        n_, d_ = int(x[0]), int(x[1])
        embs = [np.array(x[2 + j * d_:2 + (j + 1) * d_], np.float32) for j in range(n_)]
        m = np.mean(embs, 0)
        return m / np.linalg.norm(m)
    cases.append(("meanrows32", [rows, d] + [float(x) for x in E.ravel()], mr))

    # Python set iteration order (pyset.rs): groups of chain ids merged with |=, then copied with set()
    n = int(rng.integers(2, 60))
    vals = [int(v) for v in rng.choice(4000, n, replace=False)]
    alive = list(range(n))
    prog = []
    for _ in range(int(rng.integers(0, n))):
        a_, b_ = rng.choice(alive, 2, replace=False)
        prog += [int(a_), int(b_)]
        alive.remove(b_)

    def run_sets(x):
        n_ = int(x[0])
        sets = [{int(x[1 + i])} for i in range(n_)]
        for j in range(1 + n_, len(x) - 1, 2):
            a_, b_ = int(x[j]), int(x[j + 1])
            sets[a_] |= sets[b_]
            sets[b_] = None
        out = []
        for s_ in sets:
            if s_ is not None:
                out += list(s_) + [-1] + list(set(s_)) + [-2]
        return out
    cases.append(("pyset", [n] + vals + prog, run_sets))

    # Python's repr, which keys the recording: float32-derived values (exact ties at the shortest length are common)
    vals = [float(np.float32(rng.uniform(-5000, 5000))) for _ in range(4)] + [float(np.float32(rng.uniform(0, 1))) / 3840.0,
                                                                              float(rng.normal(0, 1e-5)), float(rng.normal(0, 1e17))]
    cases.append(("repr", vals, lambda x: json.dumps([float(v) for v in x])))
    cases.append(("mmcore_repr", vals, lambda x: json.dumps([float(v) for v in x])))  # mm-core's float_repr, for the report

inp = "\n".join(json.dumps({"op": op, "x": x}) for op, x, _ in cases)
r = subprocess.run([os.path.join(NATIVE, "target", "debug", "scan_numcheck")], input=inp, capture_output=True, text=True)
if r.returncode:
    print(r.stderr[-2000:])
    sys.exit(1)
outs = [json.loads(x) for x in r.stdout.splitlines()]
bad = {}
tot = {}
for (op, x, f), got in zip(cases, outs):
    if op in ("repr", "mmcore_repr"):
        want = [f(x)]
    else:
        want = [float(v) for v in np.asarray(f(x), float).ravel()]
    if op in ("repr", "mmcore_repr"):
        got = [got]
    tot[op] = tot.get(op, 0) + 1
    if want != got:
        bad.setdefault(op, []).append((want[:3], got[:3]))
INFO = {"mmcore_warp", "mmcore_repr"}  # not mm-scan's: reported for mm-core's owner (its warp_cs's det isn't numpy's)
res = {"cases": tot, "differ": {k: len(v) for k, v in bad.items()}, "examples": {k: v[:2] for k, v in bad.items()},
       "informational": sorted(INFO)}
print(json.dumps(res))
print("PASS" if not set(bad) - INFO else "FAIL")
