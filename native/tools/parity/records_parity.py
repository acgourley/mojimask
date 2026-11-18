"""Draw records: mm-records (Rust) against the Python helper (native/tools/sidecar/project_server.py's `records`),
byte for byte, over a seeded random sequence of edit states.

    .venv/bin/python native/tools/parity/records_parity.py [--clips jazz,jazz_4k] [--states 40] [--seed 3] [--bin PATH]

Per clip (public domain only: out/poc/projects/jazz and jazz_4k), a fresh copy under out/native/records/parity/<clip>/
(frames linked, lighting caches removed) is opened by project_server.py in a child Python process (review_server's
startup, as the app's sidecar does; no port is served). That process changes the edits with review_server's own
functions (apply_ops, change_facts, the decisions' facts_for_answer, undo_last, split_cover): palettes and picks,
cover style, lighting settings (on, off, a fixed light colour, the estimate followed), tilt & turn, deleted,
not_real, trims, manual covers (keys, followed frames, evidence), cuts, a chosen portrait photo; facts (face / not a
face, same / different / part, emoji, show, name) and answers to the scan's decisions. After each step it writes
json.dumps(record) for every frame for every ask (each style; strict; the edits' own style), with depth on or off
(alternating, where the clip has depth maps). Then the Rust bin (mm-records --batch) reads the same folder and
writes the same lines.

Also: the lighting caches. A second copy (rs/) without caches is opened by Rust first, so it computes and writes its
own; its arrays are compared with Python's (exact), and its records for the first state with those of the shared
copy.
Prints JSON and a PASS/FAIL line.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time

# the main checkout (a worktree has no venv or models of its own): the parent of git's common directory
MAIN_CHECKOUT = os.path.dirname(subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse",
                                                "--path-format=absolute", "--git-common-dir"],
                                               capture_output=True, text=True).stdout.strip())

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.dirname(NATIVE)
PY = os.path.join(MAIN_CHECKOUT, ".venv/bin/python")
OUT = os.path.join(ROOT, "out", "native", "records", "parity")
VIDEOS = {"jazz": os.path.join(ROOT, "out", "poc", "clips", "jazz.mp4"), "jazz_4k": os.path.join(ROOT, "out", "poc", "clips", "jazz_4k.mp4")}
DEPTH = "anything-small"
ASKS = [{"style": s, "strict": False, "wait": True} for s in ("emoji", "portrait", "blur", "pixelate", "solid")] + \
       [{"style": None, "strict": False, "wait": True}, {"style": "emoji", "strict": True, "wait": True}]
SKIP = {".server.lock", "review_server.log", "frames_view"}


def copy_project(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for name in os.listdir(src):
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if name in SKIP or name.startswith(("light_est_", "film_est_")):
            continue
        if name == "frames":
            os.symlink(s, d)
        elif os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
    web = os.path.join(os.path.dirname(dst), "web", "assets")
    os.makedirs(web, exist_ok=True)
    shutil.copy2(os.path.join(ROOT, "web", "assets", "portrait_mona.png"), web)
    os.makedirs(os.path.join(dst, "portrait"), exist_ok=True)
    shutil.copy2(os.path.join(ROOT, "web", "assets", "portrait_mona.png"), os.path.join(dst, "portrait", "photo.png"))
    return dst


def rust_bin(a):
    if a.bin:
        return a.bin
    r = subprocess.run(["cargo", "build", "-q", "-p", "mm-records", "--bin", "mm-records"], cwd=NATIVE, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("cargo build -p mm-records failed:\n" + r.stderr[-3000:])
    return os.path.join(NATIVE, "target", "debug", "mm-records")


# ------------------------------------------------------------------ child: Python side (one clip per process)
def child(a):
    clip, work = a.child, os.path.join(OUT, a.child, "py")
    asks_path = os.path.join(OUT, clip, "asks.json")
    with open(asks_path, "w") as f:
        json.dump(ASKS, f)
    has_depth = os.path.exists(os.path.join(work, "dets", f"depth_{DEPTH}.mp4"))
    dets = [d + ":0.3" for d in ("yolov5m_face_tiled", "retinaface_r50_2x") if os.path.exists(os.path.join(work, "dets", d + ".json"))]
    sys.argv = ["project_server.py", VIDEOS[clip], os.path.join(work, "dets"), "--workdir", work, "--port", "1", "--dets", *dets]
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(NATIVE, "tools", "sidecar"))
    os.environ["MOJIMASK_DEPTH"] = DEPTH if has_depth else "off"
    import project_server as PS
    from mojimask import portrait as PT
    # the default photo from a copy in out/ (its pose cache is written next to it, not into web/)
    PT.DEFAULT = os.path.join(OUT, clip, "web", "assets", "portrait_mona.png")
    from mojimask import facts as FX
    from mojimask import palettes
    RS = PS.RS
    rng = random.Random(a.seed)
    recs = [d for f in RS._decision_files() for d in RS._load_json(f)["decisions"]]
    tracks = sorted(RS.AUTO, key=lambda q: (len(q), q))
    pieces = sorted({str(d["about"]["piece"]) for d in recs} | set(tracks), key=lambda q: (len(q), q))
    presets = list(palettes.PRESETS) + ["custom"]
    emoji = ["😀", "🐸", "🦊", "🌝", "🍕", "🐑", "🦄", "🖼️"]

    def box_at(tid=None):
        tid = tid or rng.choice(tracks)
        t = rng.choice(sorted(RS.AUTO[tid]["cover"]))
        c = RS.AUTO[tid]["cover"][t]
        return t, [round(float(v) + rng.uniform(-3, 3), 1) for v in c]

    def covers():
        return sorted((RS.load_edits().get("manual") or {}).items())

    def op():
        k = rng.random()
        p = rng.choice(pieces)
        if k < 0.07:
            return {"op": "set", "path": ["palette"], "value": rng.choice(presets)}
        if k < 0.09:
            return {"op": "set", "path": ["palette_custom"], "value": rng.sample(emoji, rng.randint(1, 4))}
        if k < 0.15:
            return {"op": "set", "path": ["cover_style"], "value": rng.choice(["emoji", "blur", "pixelate", "solid", "portrait"])}
        if k < 0.30:
            light = RS.load_edits().get("light") or {}
            kk = rng.choice(["enabled", "level", "mix", "amb", "ramp", "floor", "light_colour", "grain", "blur", "shutter",
                             "saturation", "black"])
            v = {"enabled": rng.random() < 0.6, "light_colour": rng.choice([None, [rng.uniform(0, 400), rng.uniform(-0.2, 1.2)], [30, 1]]),
                 "grain": rng.random() < 0.5, "blur": rng.random() < 0.5}.get(kk, rng.choice([0, 0.5, 1, 1.0, 0.25, 2, 1.7, -0.3]))
            return {"op": "set", "path": ["light"], "value": {**light, kk: v}}
        if k < 0.34:
            return {"op": "set", "path": ["head_pose"], "value": rng.random() < 0.7}
        if k < 0.42:
            return {"op": rng.choice(["add", "remove"]), "path": [rng.choice(["deleted", "not_real"])], "value": p}
        if k < 0.48:
            lo = rng.randint(0, 80)
            return rng.choice([{"op": "set", "path": ["trims", p], "value": [lo, lo + rng.randint(0, 60)]},
                               {"op": "del", "path": ["trims", p]}])
        if k < 0.62:
            n = 1 + max([int(m[1:]) for m, _ in covers() if m[1:].isdigit()] + [0])
            tid = rng.choice(tracks)
            keys = dict(box_at(tid) for _ in range(rng.randint(1, 3)))
            m = {"keys": {str(t): c for t, c in sorted(keys.items())}}
            if rng.random() < 0.5:
                m["prop"] = {str(t + 1): [round(c[0] + 1, 1), round(c[1], 1), c[2]] for t, c in keys.items() if str(t + 1) not in m["keys"]}
                m["ev"] = sorted(t + 1 for t in keys)
                m["pv"] = 3
            return {"op": "set", "path": ["manual", f"m{n}"], "value": m}
        if k < 0.68 and covers():
            mid = rng.choice(covers())[0]
            if rng.random() < 0.3:
                return {"op": "del", "path": ["manual", mid]}
            t, c = box_at()
            return {"op": "set", "path": ["manual", mid, "keys", str(t)], "value": c}
        if k < 0.76:
            return {"op": "set", "path": ["cuts", rng.choice(tracks)], "value": sorted(rng.sample(range(1, 126), rng.randint(1, 2)))}
        if k < 0.82:
            return {"op": "set", "path": ["portrait"], "value": rng.choice(["photo.png", "missing.png", None])}
        return {"op": "set", "path": ["cover_style"], "value": "emoji"}

    def fact():
        k = rng.random()
        p, q = rng.choice(pieces), rng.choice(pieces)
        if k < 0.25:
            return {"kind": "face", "pieces": [p], "value": rng.random() < 0.4}
        if k < 0.55:
            return {"kind": rng.choice(["same", "same", "different", "part"]), "pieces": [p, q]}
        if k < 0.65:
            return {"kind": "name", "pieces": [p], "value": rng.choice(["Ann", "Bo", ""])}
        if k < 0.85:
            return {"kind": "emoji", "pieces": [p], "value": rng.choice(emoji)}
        return {"kind": "show", "pieces": [p], "value": rng.random() < 0.5}

    def step():
        k = rng.random()
        try:
            if k < 0.35:
                ops = [op() for _ in range(rng.choice([1, 1, 2, 3]))]
                RS.apply_ops(ops, "edit")
                if any(o["path"][0] == "palette" for o in ops):
                    RS.sync_facts()
                return "ops", ops
            if k < 0.55:
                fs = [fact() for _ in range(rng.choice([1, 1, 2]))]
                RS.change_facts(fs, label="facts", page="people")
                return "facts", fs
            if k < 0.80:
                d = rng.choice(recs)
                ans = rng.choice(d["options"])
                old = [f["id"] for f in RS.FACTS["facts"] if f.get("decision") == d["id"]]
                RS.change_facts(FX.facts_for_answer(d, ans), old, label="Answer", page="decisions", decision=d["id"])
                return "decide", [d["id"], ans]
            if k < 0.88:
                return "undo", RS.undo_last()
            if k < 0.92:
                return "redo", RS.undo_last(redo=True)
            cs = covers()
            if cs:
                mid, m = rng.choice(cs)
                ks = sorted(int(t) for t in m.get("keys", {})) or [5]
                t = rng.randint(min(ks) - 5, max(ks) + 5)
                return "split", [mid, t, RS.split_cover(mid, t)]
            return "noop", None
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            return "refused", str(e)

    log = []
    for i in range(a.states):
        if i:
            for _ in range(rng.choice([1, 2, 3])):
                log.append(step())
        depth = has_depth and i % 2 == 1
        RS.DEPTH = DEPTH if depth else None
        RS.OCCL.clear()
        t0 = time.time()
        with open(os.path.join(OUT, clip, f"py_{i:03d}.jsonl"), "w") as f:
            for ask in ASKS:
                f.write(f"# {ask['style']!r} {ask['strict']} {ask['wait']}\n")
                r = PS.records(0, RS.N, ask["style"], ask["wait"], ask["strict"])
                for rec in r["records"]:
                    f.write(json.dumps(rec) + "\n")
        shutil.copy2(os.path.join(work, "edits.json"), os.path.join(OUT, clip, f"edits_{i:03d}.json"))
        print(json.dumps({"state": i, "depth": depth, "steps": log[-3:], "py_s": round(time.time() - t0, 2)}, ensure_ascii=True, default=str), flush=True)


# ------------------------------------------------------------------ parent
def compare(pa, ra):
    a = [x for x in open(pa).read().splitlines() if not x.startswith("#")]
    b = [x for x in open(ra).read().splitlines() if not x.startswith("#")]
    if a == b:
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            j = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y)))
            return {"line": i, "python": x[max(0, j - 200):j + 200], "rust": y[max(0, j - 200):j + 200], "lines": [len(a), len(b)]}
    return {"lines": [len(a), len(b)]}


def npz_equal(a, b):
    import numpy as np
    za, zb = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    if sorted(za.files) != sorted(zb.files):
        return False
    for k in za.files:
        x, y = za[k], zb[k]
        if x.dtype == object:
            x, y = x.item(), y.item()
            if list(x) != list(y):
                return False
            for p in x:
                if list(x[p]) != list(y[p]):
                    return False
                for t in x[p]:
                    if any(u.tobytes() != v.tobytes() for u, v in zip(x[p][t], y[p][t])):
                        return False
        elif x.tobytes() != y.tobytes() or x.shape != y.shape:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="jazz,jazz_4k")
    ap.add_argument("--states", type=int, default=40)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--bin", default=None)
    ap.add_argument("--child", default=None)
    a = ap.parse_args()
    if a.child:
        return child(a)
    exe = rust_bin(a)
    report = {}
    ok = True
    for clip in a.clips.split(","):
        src = os.path.join(ROOT, "out", "poc", "projects", clip)
        base = os.path.join(OUT, clip)
        if os.path.exists(base):
            shutil.rmtree(base)
        py, rs = copy_project(src, os.path.join(base, "py")), copy_project(src, os.path.join(base, "rs"))
        has_depth = os.path.exists(os.path.join(py, "dets", f"depth_{DEPTH}.mp4"))
        # Rust first, on its own copy: it computes and writes its own lighting caches
        t0 = time.time()
        r0 = subprocess.run([exe, rs, "--wait", "--depth", "off", "--style", "emoji", "--web", os.path.join(base, "web"), "--out", os.path.join(base, "rs_own_000.jsonl")], capture_output=True, text=True)
        rs_light_s = time.time() - t0
        if r0.returncode:
            sys.exit(f"rust failed: {r0.stderr[-3000:]}")
        t0 = time.time()
        p = subprocess.run([PY, os.path.abspath(__file__), "--child", clip, "--states", str(a.states), "--seed", str(a.seed)],
                           capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ROOT})
        py_s = time.time() - t0
        if p.returncode:
            sys.exit(f"python failed: {p.stdout[-2000:]}\n{p.stderr[-4000:]}")
        states = [json.loads(x) for x in p.stdout.splitlines() if x.startswith("{")]
        diffs, rs_s, lines, kinds = [], 0.0, 0, {}
        for s in states:
            for st in s["steps"]:
                kinds[st[0]] = kinds.get(st[0], 0) + 1
        for s in states:
            i = s["state"]
            shutil.copy2(os.path.join(base, f"edits_{i:03d}.json"), os.path.join(py, "edits.json"))
            t0 = time.time()
            r = subprocess.run([exe, py, "--batch", os.path.join(base, "asks.json"), "--depth", DEPTH if s["depth"] else "off", "--web", os.path.join(base, "web"),
                                "--out", os.path.join(base, f"rs_{i:03d}.jsonl")], capture_output=True, text=True)
            rs_s += time.time() - t0
            if r.returncode:
                diffs.append({"state": i, "rust_error": r.stderr[-2000:]})
                continue
            lines += sum(1 for x in open(os.path.join(base, f"py_{i:03d}.jsonl")) if not x.startswith("#"))
            d = compare(os.path.join(base, f"py_{i:03d}.jsonl"), os.path.join(base, f"rs_{i:03d}.jsonl"))
            if d:
                diffs.append({"state": i, "depth": s["depth"], **d})
        # the lighting caches: Rust's own (rs/) against Python's (py/)
        caches = {}
        for pre in ("light_est_", "film_est_"):
            a_ = sorted(f for f in os.listdir(py) if f.startswith(pre))
            b_ = sorted(f for f in os.listdir(rs) if f.startswith(pre))
            caches[pre] = {"names_equal": a_ == b_, "arrays_equal": a_ == b_ and all(npz_equal(os.path.join(py, x), os.path.join(rs, x)) for x in a_)}
        # Rust's records from its own caches = the shared copy's first state (emoji, depth off)
        own = open(os.path.join(base, "rs_own_000.jsonl")).read().splitlines()
        shared = open(os.path.join(base, "py_000.jsonl")).read().split("\n# ")[0].splitlines()[1:]
        own_equal = own == shared
        cok = not diffs and own_equal and all(c["arrays_equal"] for c in caches.values())
        ok &= cok
        n_depth = sum(s["depth"] for s in states)
        report[clip] = {"pass": cok, "states": len(states), "with_depth": n_depth, "asks_per_state": len(ASKS), "record_lines": lines,
                        "differing_states": len(diffs), "first_differences": diffs[:3], "steps": kinds, "lighting_caches": caches,
                        "own_cache_records_equal": own_equal, "has_depth": has_depth,
                        "seconds": {"python_all": round(py_s, 1), "rust_records": round(rs_s, 1), "rust_first_with_light": round(rs_light_s, 1)}}
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
