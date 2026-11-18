"""Head following: ours (mm-motion's head.rs) against Python's (track._flow_measure, follow.face_motion) on the same
frames. Only jazz (a public-domain film).

1. Synthetic heads on known paths. A census head (tools/truth/jazz.json, face box -> head box) is cut out with an oval
   mask and pasted onto a background (the same frame shifted far, so the head appears once), in frame a at its
   place and in frame b moved by a known transform, while the background moves by a small camera move. Kinds of own
   motion: still, move (0.05-0.25 head sizes), fast (0.3-0.5), scale (8-15%), roll (8-15 degrees), turn (squeezed
   sideways to 80-90%: not a similarity), blur (a move with motion blur), noise, and passed (a move, then 60-80% of
   the head covered in b: following should say lost). Each at the film's size and enlarged 2.5x (big heads).
   Measured: the flow position's error (|d - true move| in head sizes; a failure over 0.1 or no answer), and
   face_motion's kind ("ok" wanted; for passed, "lost" or "incoherent" wanted).
2. Real jazz: every census person seen on checked frames t and t+4, followed frame by frame by the flow position
   (where there's none: the camera, Python's mats), compared with the census head at t+4 (head sizes); and the
   kind face_motion gives each of them from t to t+1, Python's and ours.
    python native/tools/motion/heads.py
"""
import collections
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from mojimask import track as T  # noqa: E402
from mojimask.scan import follow as FO  # noqa: E402

KINDS = ["still", "move", "fast", "scale", "roll", "turn", "blur", "noise", "passed"]


def census():
    tr = json.load(open(os.path.join(C.ROOT, "tools", "truth", "jazz.json")))
    W, H = tr["w"], tr["h"]
    out = {}
    for t, fs in tr["frames"].items():
        for f in fs:
            if f["c"] != "face":
                continue
            b = [f["b"][0] * W, f["b"][1] * H, f["b"][2] * W, f["b"][3] * H]
            out.setdefault(int(t), {})[f["p"]] = T.to_cs(T.face_to_head(b))
    return out, tr["checked"]


def own_motion(kind, s, rng):
    """Head's own 2x3 transform about its centre (0, 0): (A, t)."""
    ang = sc = 0.0
    sx = 1.0
    t = np.zeros(2)
    d = rng.uniform(0, 2 * math.pi)
    if kind in ("move", "blur", "noise", "passed"):
        t = rng.uniform(0.05, 0.25) * s * np.array([math.cos(d), math.sin(d)])
    elif kind == "fast":
        t = rng.uniform(0.3, 0.5) * s * np.array([math.cos(d), math.sin(d)])
    elif kind == "scale":
        sc = rng.choice([-1, 1]) * rng.uniform(0.08, 0.15)
        t = rng.uniform(0, 0.05) * s * np.array([math.cos(d), math.sin(d)])
    elif kind == "roll":
        ang = math.radians(rng.choice([-1, 1]) * rng.uniform(8, 15))
    elif kind == "turn":
        sx = rng.uniform(0.8, 0.9)
        t = rng.uniform(0, 0.1) * s * np.array([math.cos(d), math.sin(d)])
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]]) * (1 + sc)
    A = R @ np.diag([sx, 1.0])
    return A, t


def make_pair(frame, c, kind, rng):
    """Returns (a, b, true 2x3 head transform a -> b in pixels)."""
    H, W = frame.shape[:2]
    s = c[2]
    # background: the frame shifted far (another place), moved a little by the camera between a and b
    sh = np.float32([[1, 0, W * 0.37], [0, 1, H * 0.29]])
    bg = cv2.warpAffine(frame, sh, (W, H), borderMode=cv2.BORDER_REFLECT_101)
    cam = np.float32([[1, 0, rng.uniform(-0.01, 0.01) * W], [0, 1, rng.uniform(-0.01, 0.01) * H]])
    bg_b = cv2.warpAffine(bg, cam, (W, H), borderMode=cv2.BORDER_REFLECT_101)
    A, t = own_motion(kind, s, rng)
    cc = np.array([c[0], c[1]])
    # head transform: about its centre, own motion, then the camera's move
    Mh = np.zeros((2, 3))
    Mh[:, :2] = A
    Mh[:, 2] = cc + t - A @ cc + cam[:, 2]
    yy, xx = np.mgrid[0:H, 0:W]
    mask = ((((xx - c[0]) / (0.46 * s)) ** 2 + ((yy - c[1]) / (0.56 * s)) ** 2) <= 1).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, 0.03 * s))[..., None]
    a = (frame * mask + bg * (1 - mask)).astype(np.float32)
    steps = 9 if kind == "blur" else 1
    hb = np.zeros_like(a)
    mb = np.zeros_like(mask)
    blur = rng.uniform(0.15, 0.3) * s if kind == "blur" else 0.0
    tn = t / (np.linalg.norm(t) + 1e-9)
    for i in range(steps):
        off = (i / max(steps - 1, 1) - 0.5) * blur * tn
        M = Mh.copy()
        M[:, 2] += off
        hb += cv2.warpAffine(frame.astype(np.float32), M, (W, H))
        mb += cv2.warpAffine(mask, M, (W, H))[..., None]
    hb /= steps
    mb /= steps
    b = hb * mb / np.maximum(mb, 1e-6) * mb + bg_b * (1 - mb)
    if kind == "passed":  # someone passes in front: a patch of the background covers most of the head in b
        cov = rng.uniform(0.6, 0.8)
        nc = Mh @ np.array([c[0], c[1], 1.0])
        x1 = int(nc[0] - s / 2 - 0.3 * s)
        y1, y2 = int(nc[1] - s / 2), int(nc[1] + s / 2)
        x2 = int(nc[0] - s / 2 + cov * s)
        patch = cv2.warpAffine(frame, np.float32([[1, 0, -W * 0.21], [0, 1, H * 0.13]]), (W, H), borderMode=cv2.BORDER_REFLECT_101)
        b[max(y1, 0):y2, max(x1, 0):x2] = patch[max(y1, 0):y2, max(x1, 0):x2]
    if kind == "noise":
        a = a + rng.normal(0, 8, a.shape)
        b = b + rng.normal(0, 8, b.shape)
    return np.clip(a, 0, 255).astype(np.uint8), np.clip(b, 0, 255).astype(np.uint8), Mh


def synthetic(L, frames, cen, checked):
    rng = np.random.default_rng(7)
    rows = []
    picks = [(t, p) for t in checked[::3] for p in sorted(cen.get(t, {}))]
    picks = [picks[i] for i in rng.choice(len(picks), size=min(40, len(picks)), replace=False)]
    for big in (1.0, 2.5):
        for t, p in picks:
            f = frames(t)
            c = np.array(cen[t][p], float)
            if big != 1.0:
                f = cv2.resize(f, None, fx=big, fy=big, interpolation=cv2.INTER_CUBIC)
                c = c * big
            H, W = f.shape[:2]
            if T.inside_frac([c[0], c[1], 1.6 * c[2]], W, H) < 0.99:
                continue
            for kind in KINDS:
                a, b, Mh = make_pair(f, c, kind, rng)
                ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
                truth = Mh @ np.array([c[0], c[1], 1.0]) - c[:2]
                s = c[2]
                m = T._flow_measure(ga, gb, c, FO.CORE)
                py_d = None if m is None else m[0]
                _, py_kind = FO.face_motion(ga, gb, c)
                _, _, o_d, _, _, _ = L.head(ga, gb, c, FO.CORE)
                o_kind, _, _, _, _, _ = L.head(ga, gb, c, FO.FLOW_CORE)
                rows.append({"t": t, "p": p, "big": big, "kind": kind, "s": float(s),
                             "py_err": None if py_d is None else float(np.linalg.norm(py_d - truth) / s),
                             "ours_err": None if o_d is None else float(np.linalg.norm(o_d - truth) / s),
                             "py_kind": py_kind, "ours_kind": o_kind})
    return rows


def real(L, frames, cen, checked, mats):
    rows = []
    gray = {}

    def g(u):
        if u not in gray:
            gray[u] = cv2.cvtColor(frames(u), cv2.COLOR_BGR2GRAY)
        return gray[u]

    for t in checked:
        if t + 4 not in cen or t + 4 > 126:
            continue
        for p, c0 in cen.get(t, {}).items():
            if p not in cen[t + 4]:
                continue
            end = cen[t + 4][p]
            res = {}
            for who in ("py", "ours", "camera"):
                c = np.array(c0, float)
                for u in range(t, t + 4):
                    cam = T.warp_cs(c, mats[u + 1])
                    if who == "camera":
                        c = cam
                        continue
                    if who == "py":
                        m = T._flow_measure(g(u), g(u + 1), c, FO.CORE)
                        d = None if m is None else m[0]
                    else:
                        d = L.head(g(u), g(u + 1), c, FO.CORE)[2]
                    c = np.array([c[0] + d[0], c[1] + d[1], cam[2]]) if d is not None else cam
                res[who] = float(np.hypot(c[0] - end[0], c[1] - end[1]) / end[2])
            # face_motion's kind on the first step, both ways of measuring it
            _, pk = FO.face_motion(g(t), g(t + 1), np.array(c0, float))
            ok_ = L.head(g(t), g(t + 1), np.array(c0, float), FO.FLOW_CORE)[0]
            rows.append({"t": t, "p": p, **res, "py_kind": pk, "ours_kind": ok_})
    return rows


def main():
    L = C.Lib()
    fdir = C.frames_dir("jazz")
    cache = {}

    def frames(t):
        if t not in cache:
            cache[t] = cv2.imread(os.path.join(fdir, f"{t:05d}.png"))
        return cache[t]

    cen, checked = census()
    syn = synthetic(L, frames, cen, checked)
    mats = np.load(os.path.join(C.ROOT, "out", "poc", "projects", "jazz", "mats_v2.npy"))
    rl = real(L, frames, cen, checked, mats)

    def flow_stats(rows, who):
        errs = [r[f"{who}_err"] for r in rows]
        v = [e for e in errs if e is not None]
        return {**C.stats(v, [e > 0.1 for e in v]), "no_answer": round(sum(e is None for e in errs) / len(errs), 4),
                "fail_rate": round(sum(e is None or e > 0.1 for e in errs) / len(errs), 4)}

    def kind_stats(rows, who):
        k = collections.Counter(r[f"{who}_kind"] for r in rows)
        return {kk: round(v / len(rows), 3) for kk, v in sorted(k.items())}

    summary = {"synthetic_pairs": len(syn), "by_kind": {}}
    for kind in KINDS:
        sel = [r for r in syn if r["kind"] == kind]
        summary["by_kind"][kind] = {"flow_py": flow_stats(sel, "py"), "flow_ours": flow_stats(sel, "ours"),
                                    "kind_py": kind_stats(sel, "py"), "kind_ours": kind_stats(sel, "ours")}
    for big in (1.0, 2.5):
        sel = [r for r in syn if r["big"] == big and r["kind"] != "passed"]
        summary[f"size_x{big}"] = {"flow_py": flow_stats(sel, "py"), "flow_ours": flow_stats(sel, "ours"),
                                   "ok_py": round(np.mean([r["py_kind"] == "ok" for r in sel]), 3),
                                   "ok_ours": round(np.mean([r["ours_kind"] == "ok" for r in sel]), 3)}
    vis = [r for r in syn if r["kind"] != "passed"]
    passed = [r for r in syn if r["kind"] == "passed"]
    summary["visible"] = {"flow_py": flow_stats(vis, "py"), "flow_ours": flow_stats(vis, "ours"),
                          "ok_py": round(np.mean([r["py_kind"] == "ok" for r in vis]), 3),
                          "ok_ours": round(np.mean([r["ours_kind"] == "ok" for r in vis]), 3)}
    summary["passed_in_front"] = {"lost_py": round(np.mean([r["py_kind"] in ("lost", "incoherent") for r in passed]), 3),
                                  "lost_ours": round(np.mean([r["ours_kind"] in ("lost", "incoherent") for r in passed]), 3)}
    summary["real_jazz_4_frames"] = {w: C.stats([r[w] for r in rl], [r[w] > 0.25 for r in rl]) for w in ("py", "ours", "camera")}
    summary["real_jazz_kinds"] = {"py": kind_stats(rl, "py"), "ours": kind_stats(rl, "ours"),
                                  "same_kind": round(float(np.mean([r["py_kind"] == r["ours_kind"] for r in rl])), 3),
                                  "both_ok_or_both_not": round(float(np.mean([(r["py_kind"] == "ok") == (r["ours_kind"] == "ok") for r in rl])), 3)}
    json.dump({"summary": summary, "synthetic": syn, "real": rl}, open(os.path.join(C.OUT, "heads.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    v = summary["visible"]
    ok = (v["flow_ours"]["fail_rate"] <= v["flow_py"]["fail_rate"] and v["ok_ours"] >= v["ok_py"] - 0.02
          and summary["real_jazz_4_frames"]["ours"]["mean"] <= summary["real_jazz_4_frames"]["py"]["mean"] * 1.05)
    print(f"{'PASS' if ok else 'FAIL'} heads: visible flow fail ours {v['flow_ours']['fail_rate']} py {v['flow_py']['fail_rate']}; "
          f"ok ours {v['ok_ours']} py {v['ok_py']}; passed-in-front lost ours {summary['passed_in_front']['lost_ours']} "
          f"py {summary['passed_in_front']['lost_py']}; real 4-frame error ours {summary['real_jazz_4_frames']['ours']['mean']} "
          f"py {summary['real_jazz_4_frames']['py']['mean']} head sizes")


if __name__ == "__main__":
    main()
