"""The scan's second step: join tracklets (tracklets.py) into chains by continuity -- the checked follow (follow.py)
from one tracklet's end landing on another's start. graph.py then groups the chains into people.

For every tracklet: follow forward from its last detection, and backward from its first, up to LIMIT frames
(scan/follow.py). A follow stops when its box lands (IoU >= MEET) on a detection that starts
(forward) or ends (backward) another tracklet. A join needs both directions to agree -- A's forward follow
lands on B's start and B's backward follow lands on A's end -- so a false detection (a hand) followed onto a
real face doesn't join them unless the face, followed back, also lands on the hand.

    python -m mojimask.scan.join PROJECT [--truth NAME] [--limit 24]

Writes PROJECT/proto/chains.json: {chain id: [tracklet ids in time order]}, and prints counts, purity
against census truth, and how many pieces each labelled person is in.
"""
import argparse
import collections
import json
import os

import numpy as np

from mojimask.scan import follow as FO  # noqa: E402
from mojimask import track as T  # noqa: E402
from mojimask.scan import tracklets as TL  # noqa: E402
from mojimask.scan import viz  # noqa: E402

LIMIT, MEET = 24, 0.4
THREADS = int(os.environ.get("JOIN_THREADS", min(8, os.cpu_count() or 4)))


def run(proj, limit=LIMIT):
    clip = FO.Clip(proj)
    tl = TL.build(TL.detections(proj), clip.mats)
    W = float(clip.W)
    viz.emit("pieces", n=clip.n, aspect=round(clip.H / W, 4),
             pieces={int(k): [[int(t), round(c[0] / W, 4), round(c[1] / W, 4), round(c[2] / W, 4)] for t, c in sorted(v["obs"].items())]
                     for k, v in tl.items()})
    starts, ends = collections.defaultdict(list), collections.defaultdict(list)
    for k, v in tl.items():
        starts[min(v["obs"])].append(k)
        ends[max(v["obs"])].append(k)

    def meeter(index, me):
        def meet(u, c):
            best = (MEET, None)
            for k in index.get(u, []):
                if k == me:
                    continue
                o = T.iou(T.from_cs(c), T.from_cs(tl[k]["obs"][u]))
                if o >= best[0]:
                    best = (o, k)
            return best[1]
        return meet
    fwd, bwd, how = {}, {}, collections.Counter()

    def both(k):  # a tracklet's two follows; independent of every other tracklet's, so they run in threads (the
        v = tl[k]  # face check, ~90% of the time, runs in ONNX Runtime outside Python's lock)
        a, b = min(v["obs"]), max(v["obs"])
        pf, pb = FO.follow(clip, b, v["obs"][b], 1, limit, meeter(starts, k)), FO.follow(clip, a, v["obs"][a], -1, limit, meeter(ends, k))
        viz.emit("follow", k=int(k), fwd=_path_viz(pf, b, W), bwd=_path_viz(pb, a, W))
        return k, pf, pb
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        done = list(ex.map(both, list(tl)))
    for k, pf, pb in done:  # in tracklet order, as before
        how[("fwd", pf.why)] += 1
        if pf.met:
            fwd[k] = pf.met[1]
        how[("bwd", pb.why)] += 1
        if pb.met:
            bwd[k] = pb.met[1]
    joins = {a: b for a, b in fwd.items() if bwd.get(b) == a}
    nxt, prev = {}, {}
    for a, b in sorted(joins.items(), key=lambda x: min(tl[x[1]]["obs"])):
        if a not in nxt and b not in prev:
            nxt[a], prev[b] = b, a
    viz.emit("joins", pairs=[[int(a), int(b)] for a, b in nxt.items()])
    chains = {}
    for k in tl:
        if k in prev:
            continue
        seq = [k]
        while seq[-1] in nxt:
            seq.append(nxt[seq[-1]])
        chains[k] = seq
    return tl, chains, {"forward lands": len(fwd), "backward lands": len(bwd), "mutual joins": len(joins), "follows": dict(how)}


HOW = {"face at flow": "f", "face at camera": "c", "looks the same": "l", "hold": "h", "exit": "x"}


def _path_viz(p, start, W):
    """A follow for the scan screen: frames in order from `start`, x and size per frame, how it saw the head."""
    us = sorted(p.boxes, key=lambda u: abs(u - start))
    return {"from": int(start), "x": [round(float(p.boxes[u][0]) / W, 4) for u in us],
            "y": [round(float(p.boxes[u][1]) / W, 4) for u in us],
            "how": "".join(HOW.get(p.how[u], "?") for u in us), "why": p.why,
            "met": int(p.met[1]) if p.met else None}


def save(proj, chains):
    from mojimask.scan import store
    json.dump({str(c): [int(k) for k in seq] for c, seq in chains.items()}, open(store.path(proj, "chains", "json", write=True), "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--truth")
    ap.add_argument("--limit", type=int, default=LIMIT)
    a = ap.parse_args()
    tl, chains, info = run(a.project, a.limit)
    print(info)
    obs = {c: {t: x for k in seq for t, x in tl[k]["obs"].items()} for c, seq in chains.items()}
    lens = np.array([len(o) for o in obs.values()])
    print(f"tracklets {len(tl)} -> chains {len(chains)}; chain length median {np.median(lens):.0f}, >= 24 frames {int((lens >= 24).sum())}")
    save(a.project, chains)
    if a.truth:
        truth = TL.truth_labels(a.truth)
        seen, bad = TL.purity(obs, truth)
        print(f"chains at >= 2 census frames: {seen}, mixed: {len(bad)} {bad[:8]}")
        per = collections.defaultdict(set)
        for c, o in obs.items():
            for t, x in o.items():
                if t in truth:
                    lab = TL.label(truth, t, x)
                    if lab and lab != "not":
                        per[lab].add(c)
        n = [len(v) for v in per.values()]
        print(f"pieces per labelled person over the census frames: median {np.median(n):.0f}, max {max(n)}, one piece: {sum(x == 1 for x in n)}/{len(n)}")


if __name__ == "__main__":
    main()
