"""The scan's identity graph: which chains (tracklets joined by continuity, join.py) are one person.

Evidence between two chains A and B (A before B, never on screen together):
  face   mean SFace embedding of each chain's frontal faces (faces.frontal: comparable views only), when both have
         >= MIN_FACES of them. Same person: >= SAME_FACE; different: < DIFF_FACE.
  room   each detection mapped to frame 0 through the camera motion. A chain "stays put" if its spread is
         < STILL head sizes over at least MIN_STILL_S seconds (a moment can't show it). Two chains that stay
         put: same spot (< SPOT head sizes apart) is evidence of the same person; > FAR apart, of different people.
  never  chains on screen at the same time (any common frame) are different people.
Rules, strongest first; merged greedily, a merge refused if any two members of the two groups overlap in time:
  1. faces >= SAME_FACE and not both staying put far apart
  2. both stay put, same spot, faces not clearly different
Everything with some evidence for and none decisive becomes a question: faces in [ASK_FACE, SAME_FACE), or the
same spot with too little time to show staying put.
Thresholds come from jazz census frames 0-28 (face, room position); frames 32-124 are a
partly held-out test of them.

    python -m mojimask.scan.graph PROJECT [--truth NAME]
"""
import argparse
import collections
import json
import os

import numpy as np

from mojimask.paths import ROOT
from mojimask.scan import faces as FA  # noqa: E402
from mojimask import track as T  # noqa: E402
from mojimask.scan import tracklets as TL  # noqa: E402

# Cosine between two chains' mean frontal SFace embeddings (reid.embed). Moved from ArcFace's 0.55 / 0.25 / 0.4 on
# 2026-11-14 to the values with the same error rates on the jazz census's face groups (1% different people above
# SAME, 1% of one person below DIFF; experiments/face_model_compare.py --dump).
MIN_FACES, SAME_FACE, DIFF_FACE, ASK_FACE = 2, 0.66, 0.33, 0.57
STILL, MIN_STILL_S, SPOT, FAR = 0.5, 1.0, 0.5, 2.0
MIN_DET = 6  # a real person: seen in at least this many frames, with a clear face (clear_face) at least once
CLEAR_FACE = 0.5  # a clear face: the face check (faces.measure) scores a detection at least this (measured with
# SCRFD's face check; kept unchanged for RetinaFace's after a four-clip bench, 2026-11-14)
MAX_GAP_S = 4.0  # seconds between two pieces for the room-position rules: positions in frame-0 coordinates come from
# composing camera motion, which drifts -- on the party clip (50 s, handheld) unrelated pieces 30 s apart landed
# on the 'same spot'
DUP_IOU = 0.4  # two chains this overlapped in every shared frame are one head detected twice
# head sizes: a spot identifies a person only if no other real head that could be them came this near it around
# the time (party clip: a bearded man and the woman dancing beside him were merged by position; 6 of 18 graded
# room merges were wrong). 1.0 and 0.75 let 2 of those back in.
CROWD = 1.5
# Position rules (same spot, appears where someone stayed put, ends where the other starts) need the two heads to be
# about the same size: distances are in their average head size, so a box 4x too big (jazz: a 153 px detection over
# the dancer, joined to a 38 px woman sitting behind her) made a far-off spot look near, and then the woman's box at
# that frame covered -- and hid -- what was inside it. A person who stays put keeps their size; one who walks toward
# the camera is joined by their face, not their position.
SPOT_SIZE = 1.6
# Faces never join two pieces nobody could walk between: within MAX_GAP_S (where room positions hold), where one ends
# and the other starts may be at most REACH head sizes apart plus SPEED head sizes a second (a head ~20 cm: 5/s is a
# brisk walk). jazz: SFace scored person 32 (leaving at the right edge) and person 42 (entering at the left 0.67 s
# later, 19 head sizes away) 0.69, over SAME_FACE, and the emoji flew across the picture. Every right face pair on
# jazz restarts within 1.7 head sizes; with this, 24/24 right merges kept, the wrong one gone, questions 7 -> 6 (the
# one between different people gone); the same for SPEED 3 to 8 (2026-11-14).
REACH, SPEED = 2.0, 5.0


def to_room(mats, t, c, inv):
    """Box c = [cx, cy, s] at frame t in frame 0's coordinates, through the camera motion: (xy, size). inv: a cache
    of the inverse motions by frame, kept by the caller."""
    if t not in inv:
        M = np.vstack([T.compose(mats, 0, t), [0, 0, 1]]) if t > 0 else np.eye(3)
        inv[t] = np.linalg.inv(M)
    p = inv[t] @ np.array([c[0], c[1], 1.0])
    return np.array([p[0], p[1]]), c[2] * np.sqrt(abs(np.linalg.det(inv[t][:2, :2])))


def clear_face(F, chains, tl, ms):
    """A clear face at least once among the detections of chains ms (F: faces.measure)."""
    return any((F.get((k, t)) or {}).get("score", 0) >= CLEAR_FACE for c in ms for k in chains[c] for t in tl[k]["obs"])


def chain_info(tl, chains, F, mats, fps):
    info = {}
    inv = {}
    for c, seq in chains.items():
        obs = {t: x for k in seq for t, x in tl[k]["obs"].items()}
        ts = sorted(obs)
        room = []
        for t in ts:
            xy, s = to_room(mats, t, obs[t], inv)
            room.append([xy[0], xy[1], s])
        room = np.array(room)
        size = float(np.median(room[:, 2]))
        ctr = np.median(room[:, :2], 0)
        spread = float(np.median(np.hypot(*(room[:, :2] - ctr).T)) / size)
        embs = [F[(k, t)]["emb"] for k in seq for t in tl[k]["obs"] if FA.frontal(F.get((k, t)))]
        face = None
        if len(embs) >= MIN_FACES:
            m = np.mean(embs, 0)
            face = m / np.linalg.norm(m)
        score = {t: tl[k]["score"][t] for k in seq for t in tl[k]["obs"]}
        info[c] = {"ts": set(ts), "a": ts[0], "b": ts[-1], "obs": obs, "score": score, "size": size, "ctr": ctr,
                   "end": room[-1, :2], "start": room[0, :2], "still": spread < STILL and (ts[-1] - ts[0]) / fps >= MIN_STILL_S,
                   "spread": spread, "face": face, "nfaces": len(embs), "fps": fps, "room": dict(zip(ts, room[:, :2]))}
    return info


def evidence(A, B):
    """(decision, score, why) for chains A, B that don't overlap in time."""
    sim = float(A["face"] @ B["face"]) if A["face"] is not None and B["face"] is not None else None
    size = (A["size"] + B["size"]) / 2
    first, second = (A, B) if A["a"] < B["a"] else (B, A)
    gap_d = float(np.hypot(*(first["end"] - second["start"])) / size)       # where one ends vs the other starts
    ctr_d = float(np.hypot(*(A["ctr"] - B["ctr"])) / size)
    both_still = A["still"] and B["still"]
    gap_s = (second["a"] - first["b"]) / A["fps"]
    near_in_time = gap_s <= MAX_GAP_S
    if sim is not None and sim >= ASK_FACE and near_in_time and gap_d > REACH + SPEED * gap_s:
        return "different", 0, f"faces {sim:.2f}, but {gap_d:.1f} head sizes apart {gap_s:.2f} s later"
    if sim is not None and sim >= SAME_FACE and not (both_still and ctr_d > FAR):
        return "same", 2 + sim, f"faces {sim:.2f}"
    if max(A["size"], B["size"]) > SPOT_SIZE * min(A["size"], B["size"]):  # positions say nothing (SPOT_SIZE)
        if sim is not None and sim < DIFF_FACE:
            return "different", 0, f"faces {sim:.2f}"
        return ("ask", sim + 0.0, f"faces {sim:.2f}, heads {max(A['size'], B['size']) / min(A['size'], B['size']):.1f}x apart in size") \
            if sim is not None and sim >= ASK_FACE else ("none", 0, "")
    if both_still and near_in_time and ctr_d < SPOT and (sim is None or sim >= DIFF_FACE):
        return "same", 1 + (SPOT - ctr_d), f"same spot ({ctr_d:.2f} apart), both stay put"
    # one piece shows the person stays put (seen >= MIN_STILL_S, not moving); the other, short, appears right there
    # without moving (a seated person seen again after being hidden). Two short pieces prove nothing (twirl).
    anchor_still = (A["still"] and B["spread"] < STILL) or (B["still"] and A["spread"] < STILL)
    if anchor_still and not both_still and ctr_d < SPOT and near_in_time and (sim is None or sim >= DIFF_FACE):
        return "same", 0.5 + (SPOT - ctr_d), f"appears where a person who stays put was ({ctr_d:.2f} apart)"
    if sim is not None and sim < DIFF_FACE:
        return "different", 0, f"faces {sim:.2f}"
    if both_still and near_in_time and ctr_d > FAR:
        return "different", 0, f"{ctr_d:.1f} head sizes apart, both stay put"
    if (sim is not None and sim >= ASK_FACE) or (gap_d < SPOT and not both_still):
        return "ask", (sim or 0) + (1 - min(gap_d, 1)), (f"faces {sim:.2f}" if sim is not None else "no comparable faces") + f", {gap_d:.2f} apart where one ends and the other starts"
    return "none", 0, ""


def run(proj, name_truth=None, state=False):
    """The graph's groups. state: also return what the later rules need (info, members, chains, tl, F, ...).
    name_truth: also check the merges and questions against a census (CLI diagnostics)."""
    mats = np.load(f"{proj}/mats_v2.npy")
    with open(f"{proj}/meta.json") as f:
        fps = json.load(f).get("fps") or 24
    tl = TL.build(TL.detections(proj), mats, TL.R)  # the tracklets join.py made chains.json from (same R)
    from mojimask.scan import store
    with open(store.path(proj, "chains", "json")) as f:
        chains = {int(c): v for c, v in json.load(f).items()}
    F = FA.measure(proj, tl)
    info = chain_info(tl, chains, F, mats, fps)
    keys = sorted(info)
    pairs = []
    group = {c: c for c in keys}  # the groups as merged so far (union-find)
    members = {c: {c} for c in keys}

    def find(c):
        while group[c] != c:
            c = group[c]
        return c

    def real(r):
        ms = members[r]
        return sum(len(info[c]["ts"]) for c in ms) >= MIN_DET and clear_face(F, chains, tl, ms)

    def same_head(x, y):
        common = info[x]["ts"] & info[y]["ts"]
        return common and all(T.iou(T.from_cs(info[x]["obs"][t]), T.from_cs(info[y]["obs"][t])) > DUP_IOU for t in common)

    def apart(x, y):  # on screen at the same time, as two different heads
        return bool(info[x]["ts"] & info[y]["ts"]) and not same_head(x, y)

    at = collections.defaultdict(list)  # frame -> [(chain, room position)]
    for c in keys:
        for t, p in info[c]["room"].items():
            at[t].append((c, p))

    def g_apart(c, r):  # c's group and group r were on screen together as different heads (groups as merged so far)
        return any(apart(u, v) for u in members[find(c)] for v in members[r])

    def crowded(x, y):
        """Someone else could be the same person: another head, known (as merged so far) to be a different person
        from one piece's group, near where the other piece ends or starts and never on screen together with that
        piece's group -- so it could be who that piece continues. A seated neighbour, on screen alongside both, is
        no alternative; nor is another piece of the same person (on screen with neither). Checked at merge time,
        when the stronger merges (faces) have already grown the groups."""
        first, second = (x, y) if info[x]["a"] < info[y]["a"] else (y, x)
        F_, S_ = info[first], info[second]
        size = (F_["size"] + S_["size"]) / 2
        w = int(round(fps))
        for t in range(F_["b"] - w, S_["a"] + w + 1):
            for c, p in at.get(t, ()):
                if find(c) in (find(x), find(y)) or not real(find(c)):
                    continue  # a blip or never-a-face piece is no alternative person
                if t < S_["a"] and np.hypot(*(p - S_["start"])) < CROWD * size and not g_apart(c, find(second)) and g_apart(c, find(first)):
                    return True  # c could be who the later piece continues
                if t > F_["b"] and np.hypot(*(p - F_["end"])) < CROWD * size and not g_apart(c, find(first)) and g_apart(c, find(second)):
                    return True  # c could be where the earlier piece went
        return False

    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if info[a]["ts"] & info[b]["ts"]:
                if same_head(a, b):
                    pairs.append(("same", 3, a, b, "one head detected twice"))
                continue
            d, s, why = evidence(info[a], info[b])
            if d != "none":
                pairs.append((d, s, a, b, why))
    from mojimask.scan import viz
    viz.emit("chains", chains={int(c): [int(k) for k in chains[c]] for c in keys})
    viz.emit("pairs", compared=len(keys) * (len(keys) - 1) // 2, same=sum(p[0] == "same" for p in pairs),
             ask=sum(p[0] == "ask" for p in pairs))
    merges, refused = [], 0
    for d, s, a, b, why in sorted([p for p in pairs if p[0] == "same"], key=lambda p: -p[1]):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if any(apart(x, y) for x in members[ra] for y in members[rb]):
            refused += 1
            viz.emit("refuse", a=int(a), b=int(b), why=why)
            continue
        if s < 2 and why != "one head detected twice" and crowded(a, b):  # a room-position merge, someone else could fit
            pairs.append(("ask", s, a, b, why + ", but another head was near that spot"))
            continue
        group[rb] = ra
        members[ra] |= members.pop(rb)
        merges.append((a, b, why))
        viz.emit("merge", a=int(a), b=int(b), why=why)
    # questions: between whole groups that are real people (seen in >= MIN_DET frames, a clear face at least once:
    # a piece that never shows one stays covered but isn't asked about), one per pair of groups, best evidence first
    seen_q, asks = set(), []
    for d, s, a, b, why in sorted([p for p in pairs if p[0] == "ask"], key=lambda p: -p[1]):
        ra, rb = find(a), find(b)
        key = (min(ra, rb), max(ra, rb))
        if ra == rb or key in seen_q or not (real(ra) and real(rb)):
            continue
        if any(apart(x, y) for x in members[ra] for y in members[rb]):
            continue
        seen_q.add(key)
        asks.append((a, b, why))
        viz.emit("ask", a=int(a), b=int(b), why=why)
    res = {"chains": len(keys), "people found": len(members), "merges": len(merges), "refused (on screen together)": refused,
           "questions": len(asks)}
    if name_truth:
        res.update(truth_report(name_truth, info, members, merges, asks))
    if state:
        return res, {"info": info, "members": members, "apart": apart, "chains": chains, "tl": tl, "F": F, "mats": mats, "fps": fps}
    out = os.path.join(ROOT, "out", "proto")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"graph_{os.path.basename(proj.rstrip('/'))}.json"), "w") as f:
        json.dump({"groups": {str(r): sorted(ms) for r, ms in members.items()}, "merges": merges, "questions": asks}, f, default=str)
    return res


def truth_report(name_truth, info, members, merges, asks):
    """The graph's merges, questions and groups against a census (CLI diagnostics)."""
    res = {}
    truth = TL.truth_labels(name_truth)

    def lab(c):
        ls = [TL.label(truth, t, x) for t, x in info[c]["obs"].items() if t in truth]
        ls = [x for x in ls if x and x.startswith("P") and x != "PNone"]
        return collections.Counter(ls).most_common(1)[0][0] if ls else None
    L = {c: lab(c) for c in sorted(info)}
    mk = collections.Counter("right" if L[a] == L[b] else "WRONG" if L[a] and L[b] else "unlabelled" for a, b, _ in merges)
    res["merges by truth"] = dict(mk)
    res["wrong merges"] = [(a, b, L[a], L[b], why) for a, b, why in merges if L[a] and L[b] and L[a] != L[b]]
    qk = collections.Counter("same" if L[a] == L[b] else "different" if L[a] and L[b] else "unlabelled" for a, b, _ in asks)
    res["questions by truth"] = dict(qk)
    per = collections.defaultdict(set)
    mixed = 0
    for r, ms in members.items():
        ls = collections.Counter(L[c] for c in ms if L[c])
        if len(ls) > 1:
            mixed += 1
        for p in ls:
            per[p].add(r)
    n = [len(v) for v in per.values()]
    res["groups mixing people"] = mixed
    res["pieces per person"] = {"median": float(np.median(n)), "max": max(n), "one piece": f"{sum(x == 1 for x in n)}/{len(n)}"}
    res["people still split"] = sorted(((p, len(v)) for p, v in per.items() if len(v) > 1), key=lambda x: -x[1])
    # why: the best evidence between each split person's groups
    diag = {}
    for p, rs in per.items():
        if len(rs) < 2:
            continue
        rs = sorted(rs)
        out = []
        for i, ra in enumerate(rs):
            for rb in rs[i + 1:]:
                best = None
                for x in members[ra]:
                    for y in members[rb]:
                        if L[x] != p or L[y] != p:
                            continue
                        if info[x]["ts"] & info[y]["ts"]:
                            ev = ("on screen together", 0, f"chains {x},{y} overlap in time")
                        else:
                            ev = evidence(info[x], info[y])
                        rank = {"same": 3, "ask": 2, "none": 1, "different": 0, "on screen together": -1}[ev[0]]
                        if best is None or rank > best[0]:
                            best = (rank, ev[0], ev[2], x, y, info[x]["nfaces"], info[y]["nfaces"], round(info[x]["spread"], 2), round(info[y]["spread"], 2),
                                    info[x]["a"], info[x]["b"], info[y]["a"], info[y]["b"])
                out.append(best[1:] if best else None)
        diag[p] = out
    res["why still split"] = diag
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--truth")
    a = ap.parse_args()
    for k, v in run(a.project, a.truth).items():
        print(f"{k}: {v}")
