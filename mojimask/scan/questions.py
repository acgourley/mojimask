"""The scan's same-person questions: which pairs of people to ask about after the graph has merged what it can.

A question is worth asking where a "yes" would join pieces of a real person and a "yes" is plausible -- not
wherever evidence is merely mixed. For every pair of real groups (>= MIN_DET detections, a clear face at least once)
that are never on screen together:
  motion  at every hand-over between the two (one group's last sighting before the other's next), the room
          distance must be reachable: <= 1 head size + V head sizes per frame of the gap; otherwise not asked.
          Closer than that -> evidence for.
  face    frontal-view SFace means (comparable views only): face_term, 3.6 x (sim - 0.49), within [-2, 1.5].
  body    OSNet on the person's own skeleton box (a hint: 0.92 AUC on jazz): body_term, 5 x (sim - 0.8),
          within [-1, 1].
Motion alone never makes a question (in a crowd a neighbour fits as well); asked if the sum is > 0, best first by sum x the smaller group's size (how much a yes would join). After a yes
the two groups merge and the candidates are recomputed. same.py turns these into the Decisions page's records; the
editor's "your box" questions (review_server.likeness) use the same face and body terms.

    python -m mojimask.scan.questions PROJECT --truth NAME     (simulates answers from the labels)
"""
import argparse, collections, json
import cv2
import numpy as np
from mojimask import appearance as AP, pose_util as PU  # noqa: E402
from mojimask.scan import faces as FA, graph as G, tracklets as TL  # noqa: E402

V = 0.2  # head sizes per frame a person can move
MIN_DET, CLEAR_FACE = G.MIN_DET, G.CLEAR_FACE  # the "real person" test's thresholds (graph.py)


def face_term(sim):
    """A face likeness (SFace cosine of two mean frontal embeddings) as evidence for "same person".
    (0.49 is where ArcFace's 0.3 was, the same share of different people above it on the jazz census; the slope
    scaled by the two models' same/different spreads, 0.30 -> 0.33: 4 -> 3.6)"""
    return float(np.clip(3.6 * (sim - 0.49), -2, 1.5))


def body_term(sim):
    """A body likeness (OSNet cosine of two mean body features) as evidence for "same person": a hint only."""
    return float(np.clip(5 * (sim - 0.8), -1, 1))


def clear_frames(st, ms):
    """The frames where a group (its tracklet chains ms) shows a clear face."""
    F, chains, tl = st["F"], st["chains"], st["tl"]
    return {t for c in ms for k in chains[c] for t in tl[k]["obs"]
            if (F.get((k, t)) or {}).get("score", 0) >= CLEAR_FACE}


def is_real(n_detections, clear_face):
    """The scan's test for one of its numbered people (People's "scan" cards): enough detections and a clear face at
    least once. newalgo.py marks tracks with it; the same-person questions pair on it."""
    return n_detections >= MIN_DET and bool(clear_face)


def groups_data(proj, st, fragments=False):
    """The real groups (>= MIN_DET detections and a clear face once). fragments: also the shorter ones (> 2
    detections) that are covered as someone of their own -- each marked real False."""
    info, members, mats = st["info"], st["members"], st["mats"]
    poses = json.load(open(f"{proj}/dets/pose_full.json"))["frames"]
    looks = st.setdefault("looks", {})  # {frozenset(chains): (face, body)}: a group the rules left alone isn't redone
    inv = {}
    out = {}
    for r, ms in members.items():
        obs = {}
        for c in ms:
            obs.update(info[c]["obs"])
        real = is_real(len(obs), clear_frames(st, ms))
        if not real and not (fragments and len(obs) > 2):
            continue
        ts = sorted(obs)
        key = frozenset(ms)
        if key not in looks:
            looks[key] = _looks(proj, st, ms, obs, ts, poses)
        face, body = looks[key]
        out[r] = {"ms": set(ms), "obs": obs, "ts": ts, "face": face, "body": body, "real": real,
                  "room": {t: G.to_room(mats, t, obs[t], inv) for t in ts}}
    return out


def _looks(proj, st, ms, obs, ts, poses):
    """(face, body) of a group: its mean frontal SFace embedding and mean OSNet body feature, each None when there are
    fewer than two to average."""
    F, chains, tl = st["F"], st["chains"], st["tl"]
    embs = [F[(k, t)]["emb"] for c in ms for k in chains[c] for t in tl[k]["obs"] if FA.frontal(F.get((k, t)))]
    face = None
    if len(embs) >= 2:
        m = np.mean(embs, 0); face = m / np.linalg.norm(m)
    crops = []
    for t in ts[::max(1, len(ts) // 8)]:
        sk = PU.pose_for_head(poses[t], obs[t])
        if sk is None:
            continue
        img = cv2.imread(f"{proj}/frames/{t:05d}.jpg")
        cr = AP.crop(img, AP.body_box(obs[t], sk))
        if cr is not None and min(cr.shape[:2]) >= 16:
            crops.append(cr)
    body = None
    if len(crops) >= 2:
        v = AP.reid_embed(crops).mean(0); body = v / np.linalg.norm(v)
    return face, body


def score(A, B, apart_groups, motion_only=None):
    """motion_only: the least motion score (1 - closeness) that makes a question without face or body evidence
    (None: never). A fragment right on a person's path is worth asking about even when its face and body are too
    blurred to compare; two whole people in a crowd are not."""
    if apart_groups:
        return None
    seq = sorted([(t, "A") for t in A["ts"]] + [(t, "B") for t in B["ts"]])
    best_close = 9.0
    for (t0, g0), (t1, g1) in zip(seq, seq[1:]):
        if g0 == g1:
            continue
        (p0, s0), (p1, s1) = (A if g0 == "A" else B)["room"][t0], (A if g1 == "A" else B)["room"][t1]
        d = float(np.hypot(*(p0 - p1)) / ((s0 + s1) / 2))
        reach = 1.0 + V * (t1 - t0)
        if d > reach:
            return None
        best_close = min(best_close, d / reach)
    parts, s = [], 0.0
    m = 1.0 - best_close
    s += m; parts.append(f"motion fits ({m:+.2f})")
    if A["face"] is not None and B["face"] is not None:
        sim = float(A["face"] @ B["face"]); f = face_term(sim); s += f; parts.append(f"faces {sim:.2f} ({f:+.2f})")
    if A["body"] is not None and B["body"] is not None:
        sim = float(A["body"] @ B["body"]); b = body_term(sim); s += b; parts.append(f"bodies {sim:.2f} ({b:+.2f})")
    if len(parts) == 1 and (motion_only is None or m < motion_only):  # motion alone: a crowd neighbour fits as well
        return None
    return s, ", ".join(parts)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("project"); ap.add_argument("--truth"); ap.add_argument("--max", type=int, default=30)
    a = ap.parse_args()
    res, st = G.run(a.project, a.truth, state=True)
    gd = groups_data(a.project, st)
    apart = st["apart"]
    truth = TL.truth_labels(a.truth) if a.truth else None

    def lab(g):
        ls = [TL.label(truth, t, c) for t, c in g["obs"].items() if t in truth]
        ls = [x for x in ls if x and x.startswith("P") and x != "PNone"]
        return collections.Counter(ls).most_common(1)[0][0] if ls else None

    def people_one_piece(groups):
        per = collections.defaultdict(int)
        for g in groups.values():
            l = lab(g)
            if l:
                per[l] += 1
        return sum(v == 1 for v in per.values()), len(per)
    groups = dict(gd)
    asked, log = set(), []
    print("real groups:", len(groups), "people in one piece before questions: %d/%d" % people_one_piece(groups))
    for q in range(a.max):
        cands = []
        for r1 in groups:
            for r2 in groups:
                if r1 >= r2 or (r1, r2) in asked:
                    continue
                A, B = groups[r1], groups[r2]
                ap_ = any(apart(x, y) for x in A["ms"] for y in B["ms"])
                sc = score(A, B, ap_)
                if sc and sc[0] > 0:
                    cands.append((sc[0] * min(len(A["ts"]), len(B["ts"])), sc[0], r1, r2, sc[1]))
        if not cands:
            break
        v, s, r1, r2, why = max(cands)
        asked.add((r1, r2))
        la, lb = lab(groups[r1]), lab(groups[r2])
        ans = "can't tell (unlabelled)" if not (la and lb) else ("same" if la == lb else "different")
        log.append((q + 1, ans, la, lb, round(s, 2), why))
        if ans == "same":
            A, B = groups[r1], groups.pop(r2)
            A["ms"] |= B["ms"]; A["obs"].update(B["obs"]); A["ts"] = sorted(A["obs"]); A["room"].update(B["room"])
            for k in ("face", "body"):
                if A[k] is None:
                    A[k] = B[k]
                elif B[k] is not None:
                    m = A[k] + B[k]; A[k] = m / np.linalg.norm(m)
        one, n = people_one_piece(groups)
        if q + 1 in (5, 10, 20, a.max):
            print(f"after {q + 1} questions: people in one piece {one}/{n}")
    for row in log:
        print("  ", row)
    c = collections.Counter(x[1] for x in log)
    print("answers:", dict(c))
    per = collections.defaultdict(list)
    for r, g in groups.items():
        l = lab(g)
        if l:
            per[l].append((min(g["ts"]), max(g["ts"]), len(g["ts"]), g["face"] is not None, g["body"] is not None))
    print("still split:", {p: v for p, v in per.items() if len(v) > 1})


if __name__ == "__main__":
    main()
