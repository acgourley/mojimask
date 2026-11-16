"""Bench: would a depth model (depth.py, MOJIMASK_DEPTH) answer decisions for you, or hide covers smarter -- and at what cost?

    python tools/bench/depth_bench.py [--models anything-small,anything-base,pro] [--noise PROJECT ...]

Needs each project's depth maps (depth.compute; cached in PROJECT/dets/depth_<model>.mp4 + .json). Three parts:

1. Decisions you answered (PROJECT/facts.json, by "user"), against the project's own scan (proto_tracks.json):
   - "is it a face?"   relief: the box's middle against the ring around it (a head stands out from what's behind
                       it; a print, a fan on a chest, a pattern doesn't), median over the box's detected frames.
   - "same person?"    depth gap: the two boxes' readings where they come closest in time.
   Scored honestly: a threshold is picked on the other projects (the most permissive one making no mistake there)
   and applied to the held-out project: how many of its questions it would have answered, and how many wrongly.
2. Noise on visible faces: every detected frame of every numbered person -- a face that's plainly there -- read
   against that person's usual depth. How often it reads "nearer by T" (what a depth rule would uncover by mistake),
   and how much a face's reading jumps frame to frame.
3. Held frames labelled by eye (tools/bench/data/jazz_held_labels.json: V visible, H hidden, O on the face of
   whoever is in front, N no face): at each threshold, hidden frames a depth rule would catch that today's rules
   keep (gain), and visible ones it would uncover (errors).

Readings are depth.Depth.near: the box's middle, over the frame's median, in units of the frame's depth spread.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mojimask import depth as DP  # noqa: E402

PROJ = os.path.join(ROOT, "out", "projects")
DECISION_PROJECTS = ["jazz_20261112-193341", "party_clip_c_20261112-210106", "IMG_1043_20261109-215455_newalgo131448"]
LABELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "jazz_held_labels.json")
THRESHOLDS = [0.25, 0.5, 0.75, 1.0, 1.5]


def ring(D, t, c, inner=0.75, outer=1.5):
    """Median reading in the ring between inner and outer box half-widths around box c (same units as near)."""
    m = D.maps[t]
    h, w = m.shape
    x, y, s = c[0] * D.k, c[1] * D.k, max(c[2] * D.k, 2)
    x1, x2, y1, y2 = int(max(x - outer * s, 0)), int(min(x + outer * s + 1, w)), int(max(y - outer * s, 0)), int(min(y + outer * s + 1, h))
    if x2 <= x1 or y2 <= y1:
        return float("nan")
    yy, xx = np.mgrid[y1:y2, x1:x2]
    r = np.maximum(np.abs(xx - x), np.abs(yy - y)) / s
    v = m[y1:y2, x1:x2][(r >= inner) & (r <= outer)]
    return float((np.median(v) - D.p50[t]) / D.spread[t]) if len(v) else float("nan")


def tracks(proj):
    p = json.load(open(os.path.join(proj, "proto_tracks.json")))
    return {k: {int(t): c for t, c in tr["obs_c"].items()} for k, tr in p["tracks"].items()}, p["tracks"]


def piece_features(D, obs):
    ts = sorted(obs)
    near = np.array([D.near(t, obs[t]) for t in ts])
    rel = np.array([D.near(t, obs[t]) - ring(D, t, obs[t]) for t in ts])
    return {"relief": float(np.nanmedian(rel)) if len(ts) else np.nan, "near": float(np.nanmedian(near)) if len(ts) else np.nan}


def gap(D, a, b, k=3):
    return D.gap(a, b, k)


def decision_rows(name, model):
    proj = os.path.join(PROJ, name)
    D = DP.Depth(proj, model)
    obs, _ = tracks(proj)
    facts = [f for f in json.load(open(os.path.join(proj, "facts.json")))["facts"] if f.get("by") == "user"]
    rows = []
    for f in facts:
        if f["kind"] == "face" and f["pieces"][0] in obs:
            rows.append({"q": "face", "truth": bool(f["value"]), **piece_features(D, obs[f["pieces"][0]])})
        elif f["kind"] in ("same", "different") and all(p in obs for p in f["pieces"][:2]):
            a, b = f["pieces"][:2]
            rows.append({"q": "same", "truth": f["kind"] == "same", "gap": gap(D, obs[a], obs[b])})
    return rows


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def held_out(rows_by_proj, q, feat, answer_when_low, answer):
    """Leave one project out: threshold from the others (most permissive with no wrong answer there), applied to the
    held-out one. answer_when_low: answer `answer` (a truth value) when feat <= thr (else when >= thr)."""
    out = []
    for name in rows_by_proj:
        train = [r for n, rs in rows_by_proj.items() if n != name for r in rs if r["q"] == q and np.isfinite(r[feat])]
        test = [r for r in rows_by_proj[name] if r["q"] == q and np.isfinite(r[feat])]
        wrong = [r[feat] for r in train if r["truth"] != answer]
        if not wrong or not test:
            out.append((name, 0, 0, len(test), None))
            continue
        thr = min(wrong) - 1e-9 if answer_when_low else max(wrong) + 1e-9
        hit = [r for r in test if (r[feat] <= thr if answer_when_low else r[feat] >= thr)]
        out.append((name, len(hit), sum(r["truth"] != answer for r in hit), len(test), thr))
    return out


def noise(name, model):
    """Visible faces' readings against (a) the person's usual reading (median of all their frames) and (b) the
    face's depth at that frame (depth.Occlusion.face: their own readings within FACE_WINDOW frames, not that one)."""
    proj = os.path.join(PROJ, name) if not os.path.isabs(name) else name
    D = DP.Depth(proj, model)
    obs, raw = tracks(proj)
    O = DP.Occlusion(D, obs)
    dev, devf, jit = [], [], []
    for k, o in obs.items():
        if not raw[k].get("real") or len(o) < 6:
            continue
        ts = sorted(o)
        v = np.array([D.near(t, o[t]) for t in ts])
        dev += list(v - np.nanmedian(v))
        devf += [v[i] - O.face(k, t) for i, t in enumerate(ts)]
        jit += [abs(v[i + 1] - v[i]) for i in range(len(ts) - 1) if ts[i + 1] == ts[i] + 1]
    return np.array(dev), np.array(devf), np.array(jit)


def labelled(model, lab):
    """lab: the held-frame labels (LABELS)."""
    proj = os.path.join(PROJ, lab["project"])
    D = DP.Depth(proj, model)
    obs, _ = tracks(proj)
    O = DP.Occlusion(D, obs)
    out = []
    for x in lab["frames"]:
        own = O.face(x["piece"], x["t"])  # the face's depth at that frame (depth.Occlusion.face)
        out.append({**x, "score": D.near(x["t"], x["box"]) - own})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DP.MODELS))
    ap.add_argument("--noise", nargs="*", default=DECISION_PROJECTS, help="projects (names or paths) for part 2")
    a = ap.parse_args()
    lab = json.load(open(LABELS))
    for model in a.models.split(","):
        have = [n for n in DECISION_PROJECTS if DP.cached(os.path.join(PROJ, n), model)]
        print(f"\n=== {model} ({DP.MODELS[model]['licence']}) -- projects with depth: {len(have)}/{len(DECISION_PROJECTS)}")
        rows = {n: decision_rows(n, model) for n in have}
        allr = [r for rs in rows.values() for r in rs]
        for q, feat in (("face", "relief"), ("face", "near"), ("same", "gap")):
            rr = [r for r in allr if r["q"] == q]
            pos = [r[feat] for r in rr if r["truth"]]
            neg = [r[feat] for r in rr if not r["truth"]]
            print(f"  1. {q:4s} {feat:6s}: {len(pos)} yes / {len(neg)} no, AUC (yes reads higher) {auc(pos, neg):.2f}")
        for q, feat, low, ans, what in (("face", "relief", True, False, "'not a face' when relief is low"),
                                         ("face", "relief", False, True, "'a face' when relief is high"),
                                         ("same", "gap", False, False, "'different people' when the gap is wide")):
            res = held_out(rows, q, feat, low, ans)
            if len(rows) > 1:
                print(f"     held out, {what}: " + "; ".join(f"{n.split('_2026')[0]} {h}/{t} answered, {w} wrong" for n, h, w, t, _ in res))
        sg = [r for r in allr if r["q"] == "same" and np.isfinite(r["gap"])]
        print("     fixed thresholds, 'different people' when the gap is at least: " + "; ".join(
            f"{g}: {sum(r['gap'] >= g for r in sg)}/{len(sg)} answered, {sum(r['gap'] >= g and r['truth'] for r in sg)} wrong"
            for g in (0.1, 0.15, 0.25, 0.5)))
        for n in a.noise:
            p = n if os.path.isabs(n) else os.path.join(PROJ, n)
            if not DP.cached(p, model):
                continue
            dev, devf, jit = noise(n, model)
            dev, devf = dev[np.isfinite(dev)], devf[np.isfinite(devf)]
            print(f"  2. visible faces {os.path.basename(p).split('_2026')[0]:13s}: {len(dev)} frames; reads nearer by >= "
                  + ", ".join(f"{t}: {np.mean(dev >= t):.1%} / {np.mean(devf >= t):.1%}" for t in THRESHOLDS)
                  + f" (vs usual / vs that frame's face depth); frame-to-frame jump median {np.nanmedian(jit):.2f}"
                  f" p90 {np.nanpercentile(jit, 90):.2f}")
        if DP.cached(os.path.join(PROJ, lab["project"]), model):
            L = labelled(model, lab)
            c = collections.Counter(x["label"] for x in L)
            print(f"  3. labelled held frames {dict(c)}")
            for t in THRESHOLDS:
                hit = [x for x in L if np.isfinite(x["score"]) and x["score"] >= t]
                gain = sum(x["label"] in "HO" and x["verdict"] == "kept" for x in hit)
                agree = sum(x["label"] in "HO" and x["verdict"] != "kept" for x in hit)
                bad = sum(x["label"] == "V" for x in hit)
                print(f"     nearer by >= {t}: catches {sum(x['label'] in 'HO' for x in hit)}/{c['H'] + c['O']} hidden "
                      f"({gain} that today's rules keep, {agree} they already drop); uncovers {bad}/{c['V']} visible")


if __name__ == "__main__":
    main()
