"""The lighting estimate: mm-records' lighting.rs against mojimask/lighting.py, exactly.

    .venv/bin/python native/tools/parity/records_lighting.py [--clips jazz,jazz_4k] [--bin PATH]

For each clip (public domain: out/poc/projects/jazz and jazz_4k), fresh copies under out/native/records/light/<clip>/
(frames/ with their times kept, proto_tracks.json, meta.json; no caches):
  py/     lighting.Lighting({tid: cover boxes}, frames, py/) in this process: computes and writes its caches
  rs/     the Rust bin (lighting-check) on its own copy: computes and writes its caches
  - both caches' names, and every array in them (bit for bit; probe: the same pieces, frames and arrays, in order)
  - to_json() text (json.dumps), settings() / film_settings() / apply_settings() for a few edits, auto() + velocity() for every piece on every frame (and one past each end, and a
    piece that isn't there), film(), frame_grain() on a few frames: exact
  cross reads:
  x_rs/   Python's caches, read by Rust: same output as rs/ (and the files left untouched)
  x_py/   Rust's caches, read by Python: same to_json() as py/
Prints JSON and a PASS/FAIL line.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.dirname(NATIVE)
sys.path.insert(0, ROOT)
from mojimask import lighting as LT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--clips", default="jazz,jazz_4k")
ap.add_argument("--bin", default=None, help="a built lighting-check (default: cargo build -p mm-records)")
a = ap.parse_args()

if a.bin:
    BIN = a.bin
else:
    assert subprocess.run(["cargo", "build", "-q", "-p", "mm-records", "--bin", "lighting-check"], cwd=NATIVE).returncode == 0
    BIN = os.path.join(NATIVE, "target", "debug", "lighting-check")


def fresh(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    shutil.copytree(os.path.join(src, "frames"), os.path.join(dst, "frames"))  # copy2: times kept (the film cache's name)
    for f in ("proto_tracks.json", "meta.json"):
        shutil.copy2(os.path.join(src, f), dst)


def caches(d):
    return sorted(f for f in os.listdir(d) if f.endswith(".npz") and "_est_" in f)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def same_arrays(p, q):
    """Every member of two .npz: same names, dtypes, shapes, bytes (probe: same dict, order and arrays)."""
    za, zb = np.load(p, allow_pickle=True), np.load(q, allow_pickle=True)
    bad = []
    if list(za.files) != list(zb.files):
        return [f"members {za.files} vs {zb.files}"]
    for k in za.files:
        x, y = za[k], zb[k]
        if x.dtype == object:
            dx, dy = x.item(), y.item()
            if list(dx) != list(dy):
                bad.append(f"{k}: pieces differ")
                continue
            for piece in dx:
                if list(dx[piece]) != list(dy[piece]):
                    bad.append(f"{k}[{piece}]: frames differ")
                    continue
                for t in dx[piece]:
                    for u, v in zip(dx[piece][t], dy[piece][t]):
                        if u.dtype != v.dtype or u.shape != v.shape or u.tobytes() != v.tobytes():
                            bad.append(f"{k}[{piece}][{t}]: {u} vs {v}")
        elif x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            nd = int(np.sum(x != y)) if x.shape == y.shape else -1
            bad.append(f"{k}: {nd} values differ (max {float(np.max(np.abs(x - y))) if nd > 0 else 0})")
    return bad


def py_side(d, snap, grain_ts):
    t0 = time.time()
    L = LT.Lighting(snap, os.path.join(d, "frames"), d)
    secs = time.time() - t0
    n = L.n
    auto = {}
    flowt = lambda v: [float(x) for x in v]
    for p in list(snap) + ["no-such-piece"]:
        rows = []
        for t in range(-1, n + 1):
            k, g, am = L.auto(p, t)
            v = L.velocity(p, t)
            rows.append(flowt(k) + flowt(g) + flowt(am) + [float(v[0]), float(v[1])])
        auto[p] = rows
    film = [list(map(float, L.film(t))) for t in range(-1, n + 1)]
    grain = {str(t): float(LT.frame_grain(cv2.imread(os.path.join(d, "frames", f"{t:05d}.jpg")))) for t in grain_ts}
    return {"seconds": secs, "to_json": json.dumps(L.to_json()), "auto": auto, "film": film, "grain": grain,
            "settings": py_settings(L, snap)}


SETTINGS = [None, {}, {"light": {"level": 1.3, "light_colour": [200, 0.5], "amb": 0.3, "ramp": 0.5, "saturation": 1.4, "black": 2}},
            {"light": {"light_colour": [-30.5, 1.5], "black": 0.5, "saturation": -1, "enabled": False}},
            {"light": {"light_colour": [720, 0], "level": 0}}, {"light": None}]


def py_settings(L, snap):
    rows = []
    p = next(iter(snap))
    for e in SETTINGS:
        st = LT.settings(e)
        k, g, am = L.auto(p, 10)
        s, b = LT.film_settings(*L.film(10), st)
        k2, a2, g2 = LT.apply_settings(k, g, am, st)
        rows.append([json.dumps(st), [float(s), float(b)] + [float(x) for x in (*k2, *a2, *g2)]])
    return rows


def rs_side(d, grain_ts):
    r = subprocess.run([BIN, d, "--grain", ",".join(map(str, grain_ts)), "--settings", json.dumps([e or {} for e in SETTINGS])],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"lighting-check failed on {d}:\n{r.stderr[-3000:]}")
    return json.loads(r.stdout)


def diff(py, rs):
    """Field -> number of differing values (exact float equality; NaN equal to NaN)."""
    eq = lambda x, y: x == y or (x != x and y != y)
    out = {"to_json": int(py["to_json"] != rs["to_json"])}
    na = 0
    for p, rows in py["auto"].items():
        rr = rs["auto"].get(p)
        if rr is None or len(rr) != len(rows):
            na += len(rows)
            continue
        na += sum(not eq(x, y) for r1, r2 in zip(rows, rr) for x, y in zip(r1, r2))
    out["auto"] = na
    out["film"] = sum(not eq(x, y) for r1, r2 in zip(py["film"], rs["film"]) for x, y in zip(r1, r2)) + abs(len(py["film"]) - len(rs["film"]))
    out["grain"] = sum(not eq(py["grain"][t], rs["grain"].get(t)) for t in py["grain"])
    out["settings"] = sum(a != b for a, b in zip(py["settings"], rs.get("settings", []))) + abs(len(py["settings"]) - len(rs.get("settings", [])))
    return out


report = {"clips": {}}
ok = True
for clip in a.clips.split(","):
    src = os.path.join(ROOT, "out", "poc", "projects", clip)
    base = os.path.join(ROOT, "out", "native", "records", "light", clip)
    proto = json.load(open(os.path.join(src, "proto_tracks.json")))
    snap = {tid: {int(t): v for t, v in tr["cover"].items()} for tid, tr in proto["tracks"].items()}
    n = len([f for f in os.listdir(os.path.join(src, "frames")) if f.endswith(".jpg")])
    grain_ts = sorted({0, n // 3, n // 2, n - 1})
    dirs = {k: os.path.join(base, k) for k in ("py", "rs", "x_rs", "x_py")}
    for d in dirs.values():
        fresh(src, d)
    py = py_side(dirs["py"], snap, grain_ts)
    rs = rs_side(dirs["rs"], grain_ts)
    c_py, c_rs = caches(dirs["py"]), caches(dirs["rs"])
    r = {"frames": n, "pieces": len(snap), "seconds": {"python": round(py["seconds"], 1), "rust": round(rs["seconds"], 1)},
         "cache_names_equal": c_py == c_rs, "caches": c_py, "cache_arrays": {}, "diff": diff(py, rs)}
    for f in c_py:
        if f in c_rs:
            r["cache_arrays"][f] = same_arrays(os.path.join(dirs["py"], f), os.path.join(dirs["rs"], f)) or "identical"
    # cross reads: Rust on Python's caches, Python on Rust's
    for f in c_py:
        shutil.copy2(os.path.join(dirs["py"], f), dirs["x_rs"])
    before = {f: sha(os.path.join(dirs["x_rs"], f)) for f in c_py}
    xr = rs_side(dirs["x_rs"], grain_ts)
    r["rust_reads_python"] = {"diff": diff(py, xr), "caches_untouched": before == {f: sha(os.path.join(dirs["x_rs"], f)) for f in caches(dirs["x_rs"])}}
    for f in c_rs:
        shutil.copy2(os.path.join(dirs["rs"], f), dirs["x_py"])
    Lx = LT.Lighting(snap, os.path.join(dirs["x_py"], "frames"), dirs["x_py"])
    r["python_reads_rust"] = {"to_json_equal": json.dumps(Lx.to_json()) == py["to_json"], "caches": caches(dirs["x_py"]) == c_rs}
    clip_ok = (r["cache_names_equal"] and all(v == "identical" for v in r["cache_arrays"].values()) and len(r["cache_arrays"]) == 2
               and not any(r["diff"].values()) and not any(r["rust_reads_python"]["diff"].values())
               and r["rust_reads_python"]["caches_untouched"] and all(r["python_reads_rust"].values()))
    r["pass"] = clip_ok
    ok &= clip_ok
    report["clips"][clip] = r

print(json.dumps(report, indent=1))
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
