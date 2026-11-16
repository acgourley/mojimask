"""The scan's first step after detection: tracklets. join.py chains them, graph.py groups the chains into people.

A tracklet is a chain of detections linked only where the link is certain. Per frame t:
  - predict each open tracklet: its last box, moved by the camera (mats_v2.npy) and its own velocity (from its
    last two detections, camera removed);
  - a detection is a candidate for a tracklet if its centre is within R head sizes of the prediction and its
    size is within SIZE of the tracklet's;
  - link only a mutual unique pair: the tracklet has exactly one candidate, and that detection has exactly one
    candidate tracklet;
  - a tracklet with no candidate may skip one frame (a detector miss), carried by the camera; a tracklet with
    two or more candidates, or one whose detection is claimed by another tracklet too, ends ("ambiguous");
  - every detection not linked starts a tracklet.
Detections: the app's fused heads at the app's per-detector thresholds (pipeline.CORE_DETS).

    python -m mojimask.scan.tracklets PROJECT [--truth NAME] [--r 0.5]

Prints counts and lengths, and with --truth (NAME's census, paths.truth) the purity of tracklets against census
truth, next to the old tracker's (tracks_v1.pkl, when the project has one).
"""
import argparse
import collections
import json
import os
import pickle

import numpy as np

from mojimask import paths
from mojimask.paths import ROOT
from mojimask import pipeline  # noqa: E402
from mojimask import track as T  # noqa: E402

CORE = [(n, float(th)) for n, th in (d.split(":") for d in pipeline.CORE_DETS)]  # the app's detectors and thresholds
R, SIZE, SKIP = 0.5, (0.7, 1.4), 1


def detections(proj):
    spec = [(n, th) for n, th in CORE if os.path.exists(f"{proj}/dets/{n}.json")]
    return [[(T.to_cs(b), s) for b, s, src in f] for f in T.fused_frames(f"{proj}/dets", spec)]


def build(dets, mats, r=R):
    tracklets, open_, nid = {}, [], 0
    for t, D in enumerate(dets):
        preds = []
        for k in open_:
            tr = tracklets[k]
            last = max(tr["obs"])
            c = T.warp_cs(tr["obs"][last], T.compose(mats, last, t))
            c[:2] += tr["v"] * (t - last)
            preds.append(c)
        cand_t = [[j for j, (c, s) in enumerate(D)
                   if np.hypot(*(c[:2] - p[:2])) < r * p[2] and SIZE[0] < c[2] / p[2] < SIZE[1]] for p in preds]
        cand_d = collections.defaultdict(list)
        for i, js in enumerate(cand_t):
            for j in js:
                cand_d[j].append(i)
        linked_t, linked_d, still = set(), set(), []
        for i, js in enumerate(cand_t):
            k = open_[i]
            tr = tracklets[k]
            if len(js) == 1 and len(cand_d[js[0]]) == 1:
                j = js[0]
                last = max(tr["obs"])
                c = np.array(D[j][0], float)
                if t - last == 1:  # own velocity, camera removed
                    moved = c[:2] - T.warp_cs(tr["obs"][last], mats[t])[:2]
                    tr["v"] = 0.5 * tr["v"] + 0.5 * moved
                tr["obs"][t] = c
                tr["score"][t] = D[j][1]
                linked_t.add(i)
                linked_d.add(j)
                still.append(k)
            elif not js and t - max(tr["obs"]) <= SKIP:
                still.append(k)
            else:
                tr["why"] = "ambiguous" if js else "lost"
        for j, (c, s) in enumerate(D):
            if j in linked_d:
                continue
            tracklets[nid] = {"obs": {t: np.array(c, float)}, "score": {t: s}, "v": np.zeros(2), "why": "end"}
            still.append(nid)
            nid += 1
        open_ = still
    return tracklets


def truth_labels(name):
    d = json.load(open(paths.truth(name)))
    W, H = d["w"], d["h"]
    return {int(t): [([b["b"][0] * W, b["b"][1] * H, b["b"][2] * W, b["b"][3] * H], b["c"], b["p"]) for b in d["frames"][str(t)]]
            for t in d["checked"]}


def label(truth, t, c):
    best = (0.3, None)
    for b, cls, p in truth.get(t, []):
        o = T.iou(T.from_cs(c), b)
        if o >= best[0]:
            best = (o, "not" if cls == "not" else f"P{p}")
    return best[1]


def purity(pieces, truth):
    """pieces: {id: {t: cs}}. Returns (pieces seen at >= 2 census frames, impure ones, their labels)."""
    seen, bad = 0, []
    for k, obs in pieces.items():
        labs = [label(truth, t, c) for t, c in obs.items() if t in truth]
        labs = [x for x in labs if x]
        if len(labs) >= 2:
            seen += 1
            if len(set(labs)) > 1:
                bad.append((k, collections.Counter(labs).most_common()))
    return seen, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--truth")
    ap.add_argument("--r", type=float, default=R)
    a = ap.parse_args()
    proj = a.project
    mats = np.load(f"{proj}/mats_v2.npy")
    dets = detections(proj)
    tl = build(dets, mats, a.r)
    lens = np.array([len(v["obs"]) for v in tl.values()])
    why = collections.Counter(v["why"] for v in tl.values())
    print(f"tracklets {len(tl)}: length median {np.median(lens):.0f}, mean {lens.mean():.1f}, >= 24 frames {int((lens >= 24).sum())}; "
          f"ended {dict(why)}; detections {sum(len(D) for D in dets)}")
    out = {str(k): {"obs": {str(t): [round(float(x), 1) for x in c] for t, c in v["obs"].items()}, "why": v["why"]} for k, v in tl.items()}
    os.makedirs(os.path.join(ROOT, "out", "proto"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "out", "proto", f"tracklets_{os.path.basename(proj.rstrip('/'))}.json"), "w"))
    if a.truth:
        truth = truth_labels(a.truth)
        seen, bad = purity({k: v["obs"] for k, v in tl.items()}, truth)
        print(f"tracklets at >= 2 census frames: {seen}, mixed: {len(bad)} {bad[:8]}")
        if os.path.exists(f"{proj}/tracks_v1.pkl"):
            auto = pickle.load(open(f"{proj}/tracks_v1.pkl", "rb"))["auto"]
            seen, bad = purity({k: {t: np.array(tr["obs_c"][t]) for t in tr["obs"]} for k, tr in auto.items()}, truth)
            print(f"today's tracks at >= 2 census frames: {seen}, mixed: {len(bad)} {bad[:8]}")
        # how many pieces each person is in, over the census frames
        per = collections.defaultdict(set)
        for k, v in tl.items():
            for t, c in v["obs"].items():
                if t in truth:
                    lab = label(truth, t, c)
                    if lab and lab != "not":
                        per[lab].add(k)
        n = [len(v) for v in per.values()]
        print(f"people in census: {len(per)}; tracklets per person over frames 0-{max(truth)}: median {np.median(n):.0f}, max {max(n)}")


if __name__ == "__main__":
    main()
