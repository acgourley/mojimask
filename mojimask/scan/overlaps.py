"""Overlap questions: a box stuck to a numbered person's face at the same moment -- a second box on the same face,
a box on their neck or hair, or someone partly hidden behind them. The "same person?" questions (questions.py)
can't reach these: two boxes on screen together are always different people to the graph.

A piece (a group the graph found, numbered or grey) is STUCK to person R when its box overlaps R's box by at least
STUCK of its own area in at least half the frames it was detected in. Then, cheapest answer first:
  inside    a grey piece sitting >= INSIDE within R's box in every frame it was detected: R's cover already blurs
            it, so it isn't drawn while R is covered (whatever it is, nothing is uncovered). Its decision rides on
            R's: if R is ever shown, it must be covered again (party clip: 4 of 33 are someone else's face hidden
            behind R). Numbered pieces are asked instead -- party piece 516 sat fully inside #416's box and is a
            different person in the grades; dropping it would take them off the people list.
  asked     otherwise, if detected in >= MIN_ASK frames: "Is this box part of #R's face, not a face, or someone
            else?" Numbered pieces first (a wrong one is an extra person in the list), then by detected frames.
  short     a grey piece detected in under MIN_ASK frames rides on R's cover like an inside one (SHORT_RIDES). Jazz
            census (2026-11-13): of 21, 18 were no face at all and 2 a second box on R's own face, each drawn as its
            own emoji for a frame or two; the one other person's face (P6 beside P4) stays under R's cover.
            Numbered short pieces are left as they are (still covered).
Answers:
  same face     the piece joins R: R keeps its own box where both were detected, takes the piece's elsewhere
  not a face    the piece is dropped (no cover)
  someone else  kept as it is
(--gt, legacy, adds "graded": what the old position grades (out/probe/gt_*.json, unreliable by eye) say; it is not an
answer.)

STUCK, INSIDE and MIN_ASK were first set on the party clip, twirl and jazz graded by position; the rules since
(SHORT_RIDES, FACE_RULES) were measured on jazz's census (tools/truth/jazz.json)
with decision_bench.py, which is how to check any change to them.

Writes every overlap decision as a record to PROJECT/decisions_overlap.json (the Decisions page shows them; your
answers are facts, facts.py, applied over these records by facts.answers()).

    python -m mojimask.scan.overlaps PROJECT [--gt out/probe/gt_NAME.json] [--sheet out.jpg]
"""
import argparse
import collections
import json
import os

import numpy as np

from mojimask import facts as FX  # noqa: E402
from mojimask import track as T  # noqa: E402
from mojimask.scan.questions import clear_frames, is_real  # noqa: E402  (the scan's "real person" test)

STUCK, INSIDE, MIN_ASK = 0.3, 0.8, 3
SHORT_RIDES = True
# The face rules (FACE_RULES), the same signal as blips.py's track rule. A piece that never shows a clear face (face
# check >= questions.CLEAR_FACE) and whose detector median is under SURE_SCORE is not a face: dropped, riding on R's
# cover. A numbered piece showing its own clear face in >= CLEAR_FRAMES frames and at least half of them is someone
# else: kept. Jazz census (2026-11-13): 4 of 4 and 4 of 4 of the asked ones; a wrong "someone else" only means two emoji on one face.
# (Measured with SCRFD's face check; kept unchanged for RetinaFace's after a four-clip bench,
# 2026-11-14.)
FACE_RULES, SURE_SCORE, CLEAR_FRAMES = True, 0.8, 3


def covered(a, b):
    """Fraction of box a (cx, cy, size) inside box b."""
    A, B = T.from_cs(a), T.from_cs(b)
    w = max(0.0, min(A[2], B[2]) - max(A[0], B[0]))
    h = max(0.0, min(A[3], B[3]) - max(A[1], B[1]))
    return w * h / ((A[2] - A[0]) * (A[3] - A[1]))


def groups_of(st, members):
    """{group: {"obs": {t: (cx, cy, s)}, "real": numbered or not}} for the graph's groups (after attachment)."""
    info = st["info"]
    out = {}
    for r, ms in members.items():
        obs = {}
        for c in ms:
            obs.update(info[c]["obs"])
        clear = clear_frames(st, ms)
        scores = [v for c in ms for v in info[c]["score"].values()]
        out[r] = {"obs": obs, "real": is_real(len(obs), clear), "clear": len(clear),
                  "median_detector": float(np.median(scores)) if scores else 0.0}
    return out


def find(groups, depth=None):
    """(inside, asks, decisions): inside = {piece: R} dropped without asking; asks = [(piece, R, why)], best first;
    decisions = one record per stuck piece (decision_record), answered or open. depth: a depth.Depth (MOJIMASK_DEPTH) --
    adds the two boxes' depth gap (median over the frames both are detected) to an asked decision's evidence, shown on
    its card. Only shown: on the party clip (2026-11-13) 2 of the 23 asked pieces reading >= 0.25 apart were "part of
    their face" (0.26 and 0.43 apart), so a gap doesn't rule out "same face"."""
    at = collections.defaultdict(list)
    for r, g in groups.items():
        if g["real"]:
            for t, o in g["obs"].items():
                at[t].append((r, o))
    inside, asks, decisions = {}, [], []
    for r, g in groups.items():
        n = len(g["obs"])
        best = None
        for R in {q for t in g["obs"] for q, _ in at.get(t, []) if q != r}:
            ob = groups[R]["obs"]
            cov = [covered(g["obs"][t], ob[t]) if t in ob else 0.0 for t in sorted(g["obs"])]
            frac = sum(c >= STUCK for c in cov) / n
            if best is None or (frac, min(cov)) > best[:2]:
                pos = [c for c in cov if c > 0]  # (none when it only shares frames with R without overlapping)
                best = (frac, min(cov), R, float(np.median(pos)) if pos else 0.0)
        if best is None or best[0] < 0.5:
            continue
        frac, low, R, med = best
        ev = {"numbered": g["real"], "frames": n, "frames_overlapping": round(frac, 2), "median_overlap": round(med, 2),
              "least_overlap": round(low, 2), "first": min(g["obs"]), "last": max(g["obs"])}
        why = (f"{'numbered' if g['real'] else 'grey'}, {n} frames, overlaps the person's box in {frac:.0%} of them"
               f" (median {med:.0%} of its box)")
        show = shown(g, groups[R])
        if low >= INSIDE and not g["real"]:
            inside[r] = R
            decisions.append(decision_record(r, R, ev, why, show, "covered by the person's box (while they are)", f"rule: inside >= {INSIDE}"))
        elif n >= MIN_ASK and FACE_RULES and not g["clear"] and g["median_detector"] < SURE_SCORE:
            inside[r] = R  # dropped like a not-a-face answer (and covered again if R is ever shown)
            decisions.append(decision_record(r, R, {**ev, "clear_face_frames": 0}, why, show, FX.A_NOT_FACE,
                                             f"rule: never a clear face (median detector under {SURE_SCORE})"))
        elif n >= MIN_ASK and FACE_RULES and g["real"] and g["clear"] >= CLEAR_FRAMES and g["clear"] >= n / 2:
            decisions.append(decision_record(r, R, {**ev, "clear_face_frames": g["clear"]}, why, show, FX.A_SOMEONE_ELSE,
                                             f"rule: its own clear face in {g['clear']} of {n} frames"))
        elif n >= MIN_ASK:
            if depth is not None:
                both = [t for t in g["obs"] if t in groups[R]["obs"]]
                gaps = [abs(depth.near(t, g["obs"][t]) - depth.near(t, groups[R]["obs"][t])) for t in both]
                gaps = [x for x in gaps if np.isfinite(x)]
                if gaps:
                    ev["depth_gap"] = round(float(np.median(gaps)), 2)
            asks.append((r, R, why))
            decisions.append(decision_record(r, R, ev, why, show))
        elif not g["real"] and SHORT_RIDES:  # a grey blip on R's box: R's cover is there, so it rides on R's
            inside[r] = R
            decisions.append(decision_record(r, R, ev, why, show, "covered by the person's box (while they are)",
                                             f"rule: under {MIN_ASK} frames, on the person's box"))
        else:
            decisions.append(decision_record(r, R, ev, why, show, "left as it is", f"rule: under {MIN_ASK} frames"))
    asks.sort(key=lambda a: (not groups[a[0]]["real"], -len(groups[a[0]]["obs"])))
    rank = {r: i + 1 for i, (r, _, _) in enumerate(asks)}
    for d in decisions:
        d["rank"] = rank.get(d["about"]["piece"])
    decisions.sort(key=lambda d: (d["rank"] is None, d["rank"] or 0, -d["evidence"]["frames"]))
    return inside, asks, decisions


OPTIONS = [FX.A_SAME_FACE, FX.A_NOT_FACE, FX.A_SOMEONE_ELSE]
FILE = "decisions_overlap.json"  # in the project folder, next to the app's edits; the app's developer page reads it


def shown(g, P):
    """Frames to show a decision by: first, middle and last of the frames where the two boxes overlap (both detected,
    >= STUCK), else where both are detected, else the piece's own; the piece's and the person's boxes as x0, y0, x1, y1
    in the project's frame pixels. (Jazz overlap:140: its first and last detections were frames where the person
    wasn't detected, so the card showed no grey box at all.)"""
    own = sorted(g["obs"])
    both = [t for t in own if t in P["obs"]]
    ts = [t for t in both if covered(g["obs"][t], P["obs"][t]) >= STUCK] or both or own
    out = []
    for t in dict.fromkeys((ts[0], ts[len(ts) // 2], ts[-1])):
        out.append({"t": t, "piece": [round(float(v), 1) for v in T.from_cs(g["obs"][t])],
                    "person": [round(float(v), 1) for v in T.from_cs(P["obs"][t])] if t in P["obs"] else None})
    return out


def decision_record(piece, person, evidence, why, show, answer=None, by=None):
    """One decision, the same shape whichever answerer settles it (a rule, a model, the user, or grades in testing):
    what it's about, the evidence, the options, the answer and who gave it (by: "rule: ...", "user", or None while
    open). show: frames and boxes to show it by."""
    return {"id": f"overlap:{piece}", "kind": "overlap", "about": {"piece": piece, "person": person},
            "question": "Is the green box part of the grey box's face, not a face, or someone else?",
            "options": OPTIONS + (["covered by the person's box (while they are)", "left as it is"] if by else []),
            "evidence": evidence, "why": why, "show": show, "answer": answer, "by": by,
            "rule_answer": answer, "rule_by": by}  # what a rule said, kept when you answer over it


def save_records(path, decisions):
    """Write a decisions_<kind>.json file (written whole, then moved into place). Your answers aren't kept in it: they
    are facts (facts.py), applied over these records by facts.answers()."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"decisions": decisions}, f, indent=1, default=str)
    os.replace(tmp, path)


def auto(proj, st, members, write=True):
    """Apply the automatic answers (grey pieces inside a person's box, pieces that are never a face) to the graph's
    groups and save the records (unless write is False), for the Decisions page. Returns (decisions, extra):
    extra = {person: {t: box}} for pieces answered "same face" (the frames the person had no box of their own); none
    now that answers are facts."""
    from mojimask import depth as DP
    inside, asks, decisions = find(groups_of(st, members), DP.Depth.for_scan(proj))
    if write:
        save_records(os.path.join(proj, FILE), decisions)
    extra = apply(members, st["info"], inside, {})
    return decisions, extra


def apply(members, info, inside, answers):
    """Change the graph's groups in place: answers = {piece: (answer, person)}. Returns {person: {t: box}} for pieces
    joined to a person (only the frames the person lacked)."""
    for r in inside:
        members.pop(r, None)
    extra = collections.defaultdict(dict)
    for r, (ans, R) in answers.items():
        if r not in members or R not in members:
            continue
        if ans == FX.A_NOT_FACE:
            members.pop(r)
        elif ans == FX.A_SAME_FACE:
            own = {t for c in members[R] for t in info[c]["obs"]}
            for c in members.pop(r):
                for t, o in info[c]["obs"].items():
                    if t not in own:
                        extra[R][t] = o
    return extra


def gt_label(gt, obs):
    """A piece's majority grade by position: 'P<id>', 'NF:<why>' or None."""
    W, H = gt["space"]["w"], gt["space"]["h"]
    cnt = collections.Counter()
    for t, x in obs.items():
        b = T.from_cs(x)
        best = None
        for g in gt["frames"].get(str(t), []):
            i = T.iou(b, [g[0] * W, g[1] * H, g[2] * W, g[3] * H])
            if i >= 0.3 and (best is None or i > best[0]):
                best = (i, g[4])
        if best:
            cnt[best[1]] += 1
    return cnt.most_common(1)[0][0] if cnt else None


def main():
    from mojimask.scan import attach as AT
    from mojimask.scan import graph as G
    from mojimask.scan import questions as Qs
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--gt")
    ap.add_argument("--sheet")
    a = ap.parse_args()
    proj = a.project.rstrip("/")
    _, st = G.run(proj, None, state=True)
    members = st["members"]
    for f, r in AT.attach(proj, st, Qs.groups_data(proj, st)).items():
        members[r] |= members.pop(f)
    groups = groups_of(st, members)
    inside, asks, decisions = find(groups)
    print(f"{len(groups)} groups ({sum(g['real'] for g in groups.values())} numbered): {len(inside)} dropped inside a person's box"
          f" ({sum(groups[r]['real'] for r in inside)} numbered), {len(asks)} overlap questions"
          f" ({sum(groups[r]['real'] for r, _, _ in asks)} about numbered pieces)")
    if a.gt:
        gt = json.load(open(a.gt))
        lab = {r: gt_label(gt, g["obs"]) for r, g in groups.items()}

        def answer(r, R):
            la, lR = lab[r], lab[R]
            if la is None:
                return "can't tell (ungraded)"
            if la.startswith("NF"):
                return FX.A_NOT_FACE
            return FX.A_SAME_FACE if la == lR else FX.A_SOMEONE_ELSE
        c = collections.Counter((answer(r, R), "numbered" if groups[r]["real"] else "grey") for r, R, _ in asks)
        print("answers from your grades:", dict(c))
        for d in decisions:
            if d["answer"] is None:
                ans = answer(d["about"]["piece"], d["about"]["person"])
                d["graded"] = None if ans.startswith("can't") else ans  # what the grades say; not an answer
        ci = collections.Counter((FX.A_NOT_FACE if (lab[r] or "").startswith("NF") else "ungraded" if lab[r] is None
                                  else FX.A_SAME_FACE if lab[r] == lab[R] else FX.A_SOMEONE_ELSE) for r, R in inside.items())
        print("dropped inside a person's box, by grade:", dict(ci))
        for r, R, why in asks[:60]:
            print(f"   {r:>5} on #{R:<5} {answer(r, R):22} {lab[r]!s:12} {why}")
    save_records(os.path.join(proj, FILE), decisions)
    print(f"decisions: {os.path.join(proj, FILE)}")
    if a.sheet:
        sheet(proj, groups, asks, a.sheet)


def sheet(proj, groups, asks, path, n=40):
    """Rows: the piece's box (green) and the person's (light grey), white-edged, on a crop around both, at three of the piece's frames."""
    import cv2
    rows = []
    for r, R, why in asks[:n]:
        ts = sorted(groups[r]["obs"])
        cells = []
        for t in (ts[0], ts[len(ts) // 2], ts[-1]):
            img = cv2.imread(f"{proj}/frames/{t:05d}.jpg")
            bs = [(groups[r]["obs"][t], (95, 220, 61))]
            if t in groups[R]["obs"]:
                bs.insert(0, (groups[R]["obs"][t], (196, 196, 196)))
            xs = np.array([T.from_cs(b) for b, _ in bs])
            cx, cy = (xs[:, 0].min() + xs[:, 2].max()) / 2, (xs[:, 1].min() + xs[:, 3].max()) / 2
            h = max(xs[:, 2].max() - xs[:, 0].min(), xs[:, 3].max() - xs[:, 1].min()) * 0.9 + 10
            x0, y0 = int(max(0, cx - h)), int(max(0, cy - h))
            for b, col in bs:
                q = T.from_cs(b)
                cv2.rectangle(img, (int(q[0]), int(q[1])), (int(q[2]), int(q[3])), col, 2)
                cv2.rectangle(img, (int(q[0]) - 2, int(q[1]) - 2), (int(q[2]) + 2, int(q[3]) + 2), (255, 255, 255), 1)
            cr = img[y0:int(cy + h), x0:int(cx + h)]
            cr = cv2.resize(cr, (180, 180)) if cr.size else np.zeros((180, 180, 3), np.uint8)
            cv2.putText(cr, str(t), (3, 16), 0, 0.5, (255, 255, 255), 2)
            cells.append(cr)
        row = np.hstack(cells)
        bar = np.zeros((22, row.shape[1], 3), np.uint8)
        cv2.putText(bar, f"{r} on #{R}: {why[:70]}", (4, 15), 0, 0.42, (255, 255, 255), 1)
        rows.append(np.vstack([bar, row]))
    if rows:
        cv2.imwrite(path, np.vstack(rows))
        print("sheet:", path)


if __name__ == "__main__":
    main()
