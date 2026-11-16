"""Is it a face? One probability per box from all its evidence, and cover / drop / ask by cost (prototype, 2026-11-13).

The scan's face rules (blips.py, overlaps.py) are thresholds on single signals. This puts the same signals into one
number (2026-11-13):

  evidence   every detection of a box adds a log-likelihood ratio log P(what we saw | face) / P(... | not a face)
             for each signal, from a table fitted on census-labelled detections:
               face check (RetinaFace on the crop, faces.measure), binned, *per size band* (under 30 px it misses most real faces;
               from 40 px it is decisive)
               fused detector score, binned
               a skeleton head on the box (pose_util.pose_for_head)
  the box    log-odds = b0 + w * (sum of its detections' evidence) / n ** (1 - ALPHA). Neighbouring frames of
             one box aren't independent (a blur in one is a blur in the next), so n detections count like n ** ALPHA
             independent ones; b0, w fitted on the boxes (logistic), ALPHA picked by held-out fit.
  decision   with p = P(face) and the box's frames n (those not already under another cover count):
               cost of dropping  p * n * MISS        (a face left uncovered, per frame)
               cost of covering  (1 - p) * n * NEEDLESS   (an emoji on something else, per frame)
               cost of asking    ASK                (your time)
             the cheapest wins.

Truth is the census (paths.truth(CLIP)): detections on checked frames label the evidence table; boxes take
decision_bench.Truth's label. Everything is scored held out: a box's probability comes from a model fitted without
its fold.

    python experiments/proto/evidence.py jazz           (after decision_bench.py jazz: uses out/bench/jazz)
"""
import collections
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "bench"))

SIZE_BANDS = (0, 30, 40, 10 ** 6)
FACE_BINS = (0, 0.3, 0.5, 0.7, 1.01)
DET_BINS = (0, 0.5, 0.8, 1.01)
SMOOTH = 1.0
FOLDS = 5
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def band(x, edges):
    return next(i for i in range(len(edges) - 1) if edges[i] <= x < edges[i + 1])


def detections(proj):
    """{piece: [(t, box_cs, detector, face_check, skeleton)]} for every group the scan made (before any rule dropped
    one), from the scan's own state."""
    from mojimask.scan import cover as CV
    from mojimask import pose_util as PU
    b = CV.build(proj, write=False)  # reruns the scan's rules without saving their records (PROJECT/decisions_*.json)
    st, members0 = b["st"], b["members0"]
    info, F, chains, tl = st["info"], st["F"], st["chains"], st["tl"]
    pp = os.path.join(proj, "dets", "pose_full.json")
    poses = []
    if os.path.exists(pp):
        with open(pp) as f:
            poses = json.load(f)["frames"]
    out = {}
    for r, ms in members0.items():
        obs, score, face = {}, {}, collections.defaultdict(float)
        for c in ms:
            obs.update(info[c]["obs"]); score.update(info[c]["score"])
            for k in chains[c]:
                for t in tl[k]["obs"]:
                    face[t] = max(face[t], (F.get((k, t)) or {}).get("score", 0) or 0)
        out[str(r)] = [(t, o, float(score.get(t, 0)), float(face.get(t, 0)), t < len(poses) and PU.pose_for_head(poses[t], o) is not None)
                       for t, o in sorted(obs.items())]
    return out


def keys(d):
    """The evidence a detection carries, as table keys."""
    t, o, det, face, skel = d
    return [("face", band(o[2], SIZE_BANDS), band(face, FACE_BINS)), ("det", band(det, DET_BINS)), ("skel", skel)]


def fit_table(labelled):
    """labelled: [(detection, is_face)] -> {key: llr}"""
    n = collections.Counter(f for _, f in labelled)
    c = collections.Counter((k, f) for d, f in labelled for k in keys(d))
    space = collections.defaultdict(set)
    for (k, _f) in c:
        space[k[:-1]].add(k)
    table = {}
    for k in {k for k, _ in c}:
        m = len(space[k[:-1]]) or 1
        pf = (c[(k, True)] + SMOOTH) / (n[True] + SMOOTH * m)
        pn = (c[(k, False)] + SMOOTH) / (n[False] + SMOOTH * m)
        table[k] = math.log(pf / pn)
    return table


def box_evidence(dets, table, alpha):
    s = sum(table.get(k, 0.0) for d in dets for k in keys(d))
    return s / len(dets) ** (1 - alpha)


def logistic(x, y, iters=3000, lr=0.5, l2=1e-3):
    """(b0, w) for P(y) = sigmoid(b0 + w x); fitted on standardised x, returned in x's own units."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mu, sd = x.mean(), x.std() or 1.0
    z = (x - mu) / sd
    a, b = 0.0, 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(a + b * z, -30, 30)))
        a -= lr * np.mean(p - y)
        b -= lr * (np.mean((p - y) * z) + l2 * b)
    return a - b * mu / sd, b / sd


def sigmoid(v):
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, v))))


def det_labels(truth, dets):
    """Detections on checked census frames with their truth (face / not); heads and unmatched skipped... unmatched on
    a checked frame is not a face (every face there is boxed)."""
    out = []
    for d in dets:
        t, o = d[0], d[1]
        if t not in truth.scored:
            continue
        m = truth.match(t, truth.frac(o))
        if m and m[0] == "H":
            continue
        out.append((d, bool(m) and m[0] == "P"))
    return out


def held_out(truth, D, pieces, alpha):
    """{piece: P(face)} with each fold's model fitted without it."""
    order = sorted(pieces, key=lambda k: hashlib.md5(str(k).encode()).hexdigest())  # the same folds every run
    folds = [order[i::FOLDS] for i in range(FOLDS)]
    prob = {}
    for f in folds:
        train = [k for k in pieces if k not in set(f)]
        table = fit_table([x for k in train for x in det_labels(truth, D[k])])
        xs = [box_evidence(D[k], table, alpha) for k in train]
        ys = [truth.is_face(k) for k in train]
        b0, w = logistic(xs, ys)
        for k in f:
            prob[k] = sigmoid(b0 + w * box_evidence(D[k], table, alpha))
    return prob


def log_loss(prob, truth, pieces):
    eps = 1e-6
    return -np.mean([math.log(prob[k] + eps) if truth.is_face(k) else math.log(1 - prob[k] + eps) for k in pieces])


def decide(p, n, miss, needless, ask, exposed=None):
    """exposed: the frames where dropping it would show something (not under another cover); default all n."""
    e = n if exposed is None else exposed
    costs = {"drop": p * e * miss, "cover": (1 - p) * n * needless, "ask": ask if e else float("inf")}
    return min(costs, key=costs.get)


SOLID = 0.5  # a box at least this much inside another cover is hidden by it


def exposure(truth, D, solid=SOLID):
    """{piece: frames where it is not under the cover of a box the scan keeps for sure} -- the people with a clear
    face (their covers, held frames included), which no face question touches."""
    from mojimask.scan import overlaps as OV
    sure = {k for k, tr in truth.pt["tracks"].items() if tr.get("real")}
    out = {}
    for k, dets in D.items():
        n = 0
        for t, o, *_ in dets:
            n += not any(OV.covered(o, truth.pt["tracks"][q]["cover"][str(t)]) >= solid
                         for q in sure if q != k and str(t) in truth.pt["tracks"][q]["cover"])
        out[k] = n
    return out


# ---------------------------------------------------------------- compare with the rules (decision_bench replay)
def simulate(proj, truth, prob, D, miss, needless, ask, exposed=None):
    """Replace the rule answers of the blip and track questions with the model's cover / drop / ask, replay with the
    census as the answerer, and score as decision_bench does."""
    import decision_bench as B
    ds = B.decisions(proj)
    hidden = set(truth.pt.get("hidden", []))
    acts = collections.Counter()
    for d in ds:
        k = str(d["about"]["piece"])
        if d["kind"] not in ("blip", "track") or k not in prob:
            continue
        if str(d.get("rule_by") or "").startswith("rule: asked as an overlap"):
            continue
        a = decide(prob[k], len(D[k]), miss, needless, ask, None if exposed is None else exposed[k])
        acts[a] += 1
        if a == "ask":
            d["answer"] = d["rule_answer"] = d["by"] = d["rule_by"] = None
            hidden.discard(k)  # covered until answered
        elif a == "cover":
            d["answer"] = d["rule_answer"] = "kept: covered"; d["by"] = d["rule_by"] = "model"
            hidden.discard(k)
        else:
            d["answer"] = d["rule_answer"] = "dropped: too weak"; d["by"] = d["rule_by"] = "model"
            hidden.add(k)
    truth.pt["hidden"] = sorted(hidden)
    facts, asked, _, final = B.replay(ds, truth.pt["hidden"], truth)
    cov = B.Covers(truth, facts)
    _, wrong = B.rule_errors(final, truth, truth, cov)
    return acts, asked, collections.Counter(w["harm"] for w in wrong), B.result(truth, cov), sum(w["frames"] for w in wrong)


def main(clip):
    import pickle
    import decision_bench as B
    proj = os.path.join(ROOT, "out", "bench", clip)
    cache = os.path.join(proj, "evidence_dets.pkl")
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(os.path.join(proj, "proto_tracks.json")):
        D = pickle.load(open(cache, "rb"))
    else:
        D = detections(proj)
        pickle.dump(D, open(cache, "wb"))
    truth = B.Truth(clip, proj)
    pieces = [k for k in D if truth.person(k) is None or truth.is_face(k)]  # heads (no cover needed) left out
    fits = {a: held_out(truth, D, pieces, a) for a in ALPHAS}
    loss = {a: log_loss(p, truth, pieces) for a, p in fits.items()}
    for a in ALPHAS:
        print(f"alpha {a:.2f}: held-out log loss {loss[a]:.3f}")
    alpha = min(ALPHAS, key=loss.get)
    prob = fits[alpha]
    print(f"using alpha {alpha}")
    base = B.Truth(clip, proj)
    ds = B.decisions(proj)
    facts, asked, _, final = B.replay(ds, base.pt["hidden"], base)
    cov = B.Covers(base, facts)
    _, wrong = B.rule_errors(final, base, base, cov)
    res = B.result(base, cov)
    print(f"{'rules today':28s} asked {sum(asked.values()):3d} {dict(asked)}  harms {dict(collections.Counter(w['harm'] for w in wrong))}  "
          f"uncovered (checked) {res['uncovered_frames']}, by rules (all frames) {sum(w['frames'] for w in wrong)}")
    ex = exposure(base, D)
    print("boxes fully under a sure cover:", sum(1 for k in pieces if ex[k] == 0), "of", len(pieces))
    for miss, ask in ((5, 2), (20, 2), (20, 5), (100, 5), (100, 20)):
        acts, asked, harms, res, unc = simulate(proj, B.Truth(clip, proj), prob, D, miss, 1, ask, ex)
        print(f"model miss={miss:3d} ask={ask:2d}      asked {sum(asked.values()):3d} {dict(asked)}  harms {dict(harms)}  "
              f"uncovered (checked) {res['uncovered_frames']}, by the model (all frames) {unc}   {dict(acts)}")
    return prob, D, truth


if __name__ == "__main__":
    main(sys.argv[1])
