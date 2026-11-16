"""Split drafted truth into what can be accepted as is and what needs a person's focused check.

For every box in a drafted (unchecked) frame:
  easy  a confident detection (fused score >= EASY_SCORE) that the carry-forward linked to the same person in the
        previous labelled frame by strong overlap after camera motion (IoU >= EASY_IOU), with no decision by Claude,
        and -- where both views show a clear frontal face -- faces that agree (SFace >= FACE_OK, graph's
        'faces agree' line).
        A "not a head" box is easy when it is a weak detection (< 0.5) at a place already "not" before.
  hard  everything else, with the reasons: Claude decided it, a new person, a person seen again after a gap,
        no detection under the box, a weak detection, a face that disagrees with the person's earlier face.
A fixed random sample of AUDIT_N easy items is added to the checks to measure how often "easy" is wrong. Each drafted
frame also gets one "anything missing?" glance: a check of the whole frame, since a face nobody boxed can't
come up as an item.

    python tools/label/triage.py PROJECT NAME      -> out/truth/NAME_triage.json
"""
import json
import os
import random
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import draft as DR  # noqa: E402
from mojimask.scan import faces as FA  # noqa: E402
from mojimask.scan import graph  # noqa: E402
from mojimask import track as T  # noqa: E402

EASY_SCORE, EASY_IOU, AUDIT_N = 0.8, 0.5, 25
FACE_OK = graph.ASK_FACE  # the scan's own 'faces agree' line (SFace cosine)


def main(proj, name):
    D = DR.Draft(proj, name)
    drafted = sorted(D.d.get("drafted", []))
    labelled = D.labelled()
    rng = random.Random(0)
    imgs = {}

    def img(t):
        if t not in imgs:
            imgs[t] = cv2.imread(f"{proj}/frames/{t:05d}.jpg")
        return imgs[t]

    def face(t, b):
        f = FA.face(img(t), T.to_cs(b))
        return f if FA.frontal(f) else None
    qcache = {}

    def quality(t, b):
        """How clearly this view shows the face: the face detector's (RetinaFace) confidence, more for facing the camera; 0 if no face."""
        key = (t, tuple(round(v) for v in b))
        if key not in qcache:
            f = FA.face(img(t), T.to_cs(b))
            qcache[key] = 0.0 if f is None else f["score"] * (1.0 if abs(f["yaw"]) <= FA.FRONTAL["yaw"] else 0.6) * min(1.0, f["px"] / 30)
        return qcache[key]
    hist = {}  # person -> [(t, box)] of every labelled view so far

    def best_views(p, n):
        v = hist.get(p, [])
        ranked = sorted(v, key=lambda x: (quality(*x), x[0]), reverse=True)[:n]
        return [{"t": t, "b": [round(c, 1) for c in b]} for t, b in sorted(ranked)]
    items, easy_n, stats, easy_items = [], 0, {}, []
    last_seen = {}  # person -> (frame, box) of their latest labelled view before the current frame
    for t in labelled:
        prev_t = max([x for x in labelled if x < t], default=None)
        prev = {o["p"]: o for o in D.frame(prev_t)} if prev_t is not None else {}
        M = DR.motion(D.mats, prev_t, t) if prev_t is not None else None
        boxes = D.frame(t)
        if t in drafted:
            nots = []
            for i, o in enumerate(boxes):
                det = max([sc for b, sc in D.pre[t] if T.iou(b, o["b"]) > 0.5] or [0.0])
                if o["c"] == "not":  # checked in bulk, on the frame's card
                    nots.append({"i": i, "b": [round(v, 1) for v in o["b"]], "det": round(det, 2)})
                    continue
                why = []
                if o.get("by") == "claude":
                    why.append("Claude decided this")
                twin = [x for j, x in enumerate(boxes) if j != i and x["c"] != "not" and T.iou(x["b"], o["b"]) > 0.5]
                if twin:
                    why.append(f"another label (P{twin[0]['p']}) sits on this head too: one head with two numbers?")
                if o["src"] == "carried":
                    why.append("no detection under the box: carried from the frame before")
                elif det < EASY_SCORE:
                    why.append(f"weak detection ({det:.2f})")
                if o["p"] is not None:
                    if prev_t is None:
                        pass  # the first labelled frame: everyone is new, nothing to compare with
                    elif o["p"] not in prev:
                        why.append("new person" if o["p"] not in last_seen else f"seen again after a gap (last frame {last_seen[o['p']][0]})")
                    else:
                        moved = T.to_cs(DR.warp(prev[o["p"]]["b"], M))
                        here = T.to_cs(o["b"])
                        if np.hypot(*(moved[:2] - here[:2])) / here[2] > 0.6:
                            why.append("moved more than half a head since the frame before")
                    ref = last_seen.get(o["p"])
                    if ref and o["c"] == "face":
                        f0, f1 = face(ref[0], ref[1]), face(t, o["b"])
                        if f0 is not None and f1 is not None:
                            sim = float(f0["emb"] @ f1["emb"])
                            if sim < FACE_OK:
                                why.append(f"face doesn't match P{o['p']}'s last clear face ({sim:.2f})")
                ref = last_seen.get(o["p"]) if o["p"] is not None else None
                item = {"t": t, "i": i, "b": [round(v, 1) for v in o["b"]], "c": o["c"], "p": o["p"],
                        "ref": {"t": ref[0], "b": [round(v, 1) for v in ref[1]]} if ref else None,
                        "views": best_views(o["p"], 3) if o["p"] is not None else []}
                only_weak = why and all(w.startswith("weak detection") for w in why) and o["p"] in prev
                if why and not only_weak:
                    # people who could be this one: last seen near here (moved by the camera), nearest first
                    here = T.to_cs(o["b"])
                    near = []
                    for q, (tq, bq) in last_seen.items():
                        if q == o["p"] or any(x["p"] == q for x in boxes if x is not o):
                            continue
                        c = T.to_cs(DR.warp(bq, DR.motion(D.mats, tq, t)))
                        d = float(np.hypot(*(c[:2] - here[:2])) / here[2])
                        if d < 2.0:
                            near.append((d, q, tq, bq))
                    item["near"] = [{"p": q, "t": tq, "b": [round(v, 1) for v in bq], "views": best_views(q, 2)} for d, q, tq, bq in sorted(near)[:4]]
                    items.append(dict(item, kind="who", why=why))
                else:  # a confident (or merely weak but continuing, same person) box: accepted
                    easy_n += 1
                    easy_items.append(dict(item, kind="audit", why=["random check of a box accepted automatically"]))
            items.append({"t": t, "kind": "frame", "nots": nots,
                          "why": ["anything missing (a head with no box)? any of the 'not a head' boxes actually a head?"]})
        for o in boxes:
            if o["p"] is not None and o["c"] != "not":
                last_seen[o["p"]] = (t, o["b"])
                hist.setdefault(o["p"], []).append((t, o["b"]))
    for a in rng.sample(easy_items, min(AUDIT_N, len(easy_items))):  # a fixed-size audit, placed before its frame's card
        at = next(i for i, x in enumerate(items) if x["t"] == a["t"] and x["kind"] == "frame")
        items.insert(at, a)
    stats = {"drafted frames": len(drafted), "boxes": sum(len(D.frame(t)) for t in drafted), "easy (accepted)": easy_n,
             "who is this? cards": sum(1 for x in items if x["kind"] == "who"), "audit cards": sum(1 for x in items if x["kind"] == "audit"),
             "frame cards": sum(1 for x in items if x["kind"] == "frame"),
             "not-a-head boxes on frame cards": sum(len(x["nots"]) for x in items if x["kind"] == "frame")}
    out = {"name": name, "stats": stats, "items": items}
    json.dump(out, open(os.path.join(ROOT, "out", "truth", f"{name}_triage.json"), "w"))
    return out


if __name__ == "__main__":
    o = main(sys.argv[1], sys.argv[2])
    print(o["stats"])
    import collections
    c = collections.Counter(w.split(" (")[0].split(":")[0] for x in o["items"] if x["kind"] == "who" for w in x["why"])
    for k, v in c.most_common():
        print(f"  {v:>4}  {k}")
