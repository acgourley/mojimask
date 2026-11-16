"""The scan's same-person decision records (kind "same"): two people the scan counted, never on screen together,
that may be one person (questions.py's score: motion fits at every hand-over, plus face and body similarity).
Answers are facts (facts.py kinds "same" / "different"), and facts.answers() re-picks the open questions from them
each time the page asks, so a join or split made anywhere counts.

Every pair scoring >= MIN_SCORE is recorded (below it -- motion fits but faces or bodies disagree -- nothing is
recorded); each person is in at most one open question at a time (best pair first): answering one changes the
others (party_clip_c: person 5 was in 8 of 29 candidate pairs). select() makes the scan's own starting picks: pairs
scoring under facts.ASK_SCORE (0.8) are answered "different people" by rule (you can still answer otherwise), pairs
already joined or kept apart are settled, and the next pair for each person is asked.
"""
import os

from mojimask.scan import overlaps as OV  # noqa: E402
from mojimask.scan import questions as Qs  # noqa: E402
from mojimask import track as T  # noqa: E402
from mojimask import facts as FX  # noqa: E402

MIN_SCORE = 0.5
# Fragments: groups too short (or never a clear face) to count as people, covered as someone of their own (party_clip_c:
# 176 and 181, two bits of the red-hat guy 74, each drew its own emoji on his face with nothing asking). Each is asked
# against the people it could be (best FRAGMENT_PAIRS first), motion alone enough when it fits closely.
FRAGMENT_PAIRS, FRAGMENT_MOTION = 2, 0.5
FILE = "decisions_same.json"
OPTIONS = [FX.A_SAME, FX.A_DIFFERENT]
# With a depth model on (MOJIMASK_DEPTH, depth.py): two people whose heads read the model's "apart" (depth.MODELS) or more
# apart where they come closest in time are different people, answered by rule (you can still answer otherwise).
# depth_bench.py: your same-person answers on jazz / party_clip_c / IMG_1043 never read more than 0.12 apart with
# Depth Anything; at 0.15, 32 of 67 answered questions would have been settled, none wrongly.
DEPTH_RULE = "rule: at different depths"
ON_HEAD = 0.5
SETTLED = [FX.A_JOINED, FX.A_KEPT_APART, FX.A_WAITING]


def _on_head(A, B):
    """Every frame the two share, their boxes overlap by >= ON_HEAD of the smaller: one head seen twice (a second
    detector's box, a chin or hair box), not two people side by side."""
    common = set(A["obs"]) & set(B["obs"])
    on = lambda a, b: max(OV.covered(a, b), OV.covered(b, a))  # noqa: E731  (the overlap / the smaller box)
    return all(on(A["obs"][t], B["obs"][t]) >= ON_HEAD for t in common)


def find(proj, st):
    """The same-person records for the people as they now stand (and the fragments), saved to PROJECT/FILE. Your
    answers are facts (facts.py): facts.answers() applies them over these each time the page asks."""
    gd = Qs.groups_data(proj, st, fragments=True)  # the people as they stand after every other decision, and fragments
    apart = st["apart"]
    path = os.path.join(proj, FILE)
    from mojimask import depth as DP
    dep = DP.Depth.for_scan(proj)
    cands = []
    frag = {}
    for r1 in gd:
        for r2 in gd:
            if r1 >= r2:
                continue
            A, B = gd[r1], gd[r2]
            if not A["real"] and not B["real"]:
                continue
            one_frag = not (A["real"] and B["real"])
            # a fragment on screen with a person counts as apart only if it isn't on their head (graph.py's apart()
            # needs two boxes of one head to agree closely; party_clip_c's 176 on 74 overlapped about a third)
            together = any(apart(x, y) for x in A["ms"] for y in B["ms"]) and not (one_frag and _on_head(A, B))
            sc = Qs.score(A, B, together, FRAGMENT_MOTION if one_frag else None)
            if sc and sc[0] >= MIN_SCORE:
                if one_frag:  # the fragment is the green box, the person the grey one
                    a, b = (r1, r2) if A["real"] else (r2, r1)
                    frag.setdefault(b, []).append((sc[0], a, b, sc[1]))
                    continue
                a, b = (r1, r2) if A["ts"][0] <= B["ts"][0] else (r2, r1)  # a: the earlier one
                cands.append((sc[0], a, b, sc[1]))
    for f, cs in frag.items():
        cands += sorted(cs, reverse=True)[:FRAGMENT_PAIRS]
    cands.sort(reverse=True)
    out = []
    best = lambda g: max(g["ts"], key=lambda t: g["obs"][t][2])  # noqa: E731  (the biggest view)
    for s, a, b, why in cands:
        A, B = gd[a], gd[b]
        near = min(A["ts"], key=lambda t: abs(t - B["ts"][0]))  # the person where the other one starts
        show = [{"t": t, "person": [round(float(v), 1) for v in T.from_cs(A["obs"][t])], "piece": None} for t in dict.fromkeys((best(A), near))]
        show += [{"t": t, "piece": [round(float(v), 1) for v in T.from_cs(B["obs"][t])], "person": None} for t in dict.fromkeys((B["ts"][0], best(B)))]
        i = f"same:{a}:{b}"
        d = {"id": i, "kind": "same", "about": {"piece": b, "person": a},
             "question": "Is the green box the same person as the grey one? (Any part of their head counts: face, side, back or hair.)", "options": OPTIONS + SETTLED,
             "evidence": {"score": round(s, 2), "fragment": not gd[b]["real"], "frames": len(A["ts"]) + len(B["ts"]), "first": A["ts"][0], "last": B["ts"][-1],
                          "gap_frames": int(B["ts"][0] - A["ts"][-1])},
             "why": why, "show": show, "answer": None, "by": None, "rule_answer": None, "rule_by": None}
        if dep is not None:
            g = dep.gap(A["obs"], B["obs"])
            d["evidence"]["depth_gap"] = round(g, 2)
            if g >= dep.apart:
                d["depth_apart"] = f"{DEPTH_RULE} ({dep.name}: {g:.2f} apart, the same person reads under {dep.apart})"
        out.append(d)
    select(out)
    from mojimask.scan import viz
    viz.emit("same", asked=[[int(d["about"]["person"]), int(d["about"]["piece"]), float(d["evidence"]["score"])] for d in out if d["by"] is None],
             candidates=len(out))
    OV.save_records(path, out)
    return out


def select(ds):
    """Which unanswered pairs are asked now, from your answers so far (cheap: no images, no models; the review
    server runs it after every same-person answer). Your "same person" answers join people (transitively); then,
    best score first: a pair already joined is settled ("joined already"); a pair one of your "different people"
    answers keeps apart is settled ("kept apart"); otherwise it's asked unless either person already has an open
    question ("waiting": asked once that one is answered)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for d in ds:
        if d.get("by") == "user" and d["answer"] == FX.A_SAME:
            parent[find(d["about"]["piece"])] = find(d["about"]["person"])
    apart = {frozenset((find(d["about"]["piece"]), find(d["about"]["person"]))) for d in ds
             if d.get("by") == "user" and d["answer"] == FX.A_DIFFERENT}
    busy = set()
    for d in sorted(ds, key=lambda d: -d["evidence"]["score"]):
        if d.get("by") == "user":
            continue
        ga, gb = find(d["about"]["person"]), find(d["about"]["piece"])
        if d.get("depth_apart") and ga != gb:
            d["answer"], d["by"] = FX.A_DIFFERENT, d["depth_apart"]
        elif d["evidence"]["score"] < FX.ASK_SCORE and ga != gb:
            d["answer"], d["by"] = FX.A_DIFFERENT, FX.WEAK_RULE  # (facts.answers adds the stop-after rule)
        elif ga == gb:
            d["answer"], d["by"] = FX.A_JOINED, "rule: your other answers make them one person"
        elif frozenset((ga, gb)) in apart:
            d["answer"], d["by"] = FX.A_KEPT_APART, "rule: you said these two are different people"
        elif ga in busy or gb in busy:
            d["answer"], d["by"] = FX.A_WAITING, "rule: asked once the open question about this person is answered"
        else:
            d["answer"], d["by"] = None, None
            busy |= {ga, gb}
        d["rule_answer"], d["rule_by"] = d["answer"], d["by"]
