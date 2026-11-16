"""Run the scan's graph and questions on a clip and prepare the result for a person to grade (tools/label/grade.html).

  people     every real person the graph found (>= 6 detections, a clear face at least once): up to 10 pictures
             spread over their time on screen, each the clearest view in its stretch -- "click any that isn't
             this person"
  questions  the graph's own questions (questions.py scoring, no answers simulated), as pictures of both sides
  glance     every 16th frame with every box: "is any face missing a box?"
Pieces that never show a clear face stay covered and aren't graded here.

    python experiments/proto/grade_prep.py PROJECT   -> out/proto/grade_<project>.json
"""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mojimask import track as T  # noqa: E402
from mojimask.scan import graph as G, questions as Qs  # noqa: E402


def main():
    proj = sys.argv[1]
    name = os.path.basename(proj.rstrip("/"))
    res, st = G.run(proj, None, state=True)
    gd = Qs.groups_data(proj, st)
    F, chains, tl, info, apart = st["F"], st["chains"], st["tl"], st["info"], st["apart"]

    def quality(r):
        q = {}
        for c in gd[r]["ms"]:
            for k in chains[c]:
                for t in tl[k]["obs"]:
                    f = F.get((k, t))
                    q[t] = 0.0 if f is None else f["score"] * (1.0 if abs(f["yaw"]) <= 0.25 else 0.6) * min(1.0, f["px"] / 30)
        return q

    def box(c):
        return [round(v, 1) for v in T.from_cs(c)]

    def pictures(r, n):
        ts, q = gd[r]["ts"], quality(r)
        bins = np.array_split(np.array(ts), min(n, len(ts)))
        return [{"t": int(t), "b": box(gd[r]["obs"][t])} for t in (max(b, key=lambda t: (q.get(int(t), 0), -abs(int(t) - b[len(b) // 2]))) for b in bins if len(b))]

    people = [{"id": int(r), "n": len(g["ts"]), "first": g["ts"][0], "last": g["ts"][-1], "pictures": pictures(r, 10),
               "all": {str(t): box(c) for t, c in g["obs"].items()}} for r, g in sorted(gd.items(), key=lambda x: x[1]["ts"][0])]
    qs = []
    cands = []
    for r1 in gd:
        for r2 in gd:
            if r1 >= r2:
                continue
            A, B = gd[r1], gd[r2]
            sc = Qs.score(A, B, any(apart(x, y) for x in A["ms"] for y in B["ms"]))
            if sc and sc[0] > 0:
                cands.append((sc[0] * min(len(A["ts"]), len(B["ts"])), r1, r2, sc[1]))
    for v, r1, r2, why in sorted(cands, reverse=True)[:15]:
        qs.append({"a": int(r1), "b": int(r2), "why": why, "a_pics": pictures(r1, 3), "b_pics": pictures(r2, 3)})
    n = len(st["chains"])
    nframes = max(t for g in gd.values() for t in g["ts"]) + 1
    glance = []
    for t in range(0, nframes, 16):
        bx = [{"b": box(g["obs"][t]), "g": int(r)} for r, g in gd.items() if t in g["obs"]]
        other = [{"b": box(c["obs"][t])} for k, c in info.items() if t in c["obs"] and not any(k in g["ms"] for g in gd.values())]
        glance.append({"t": t, "people": bx, "other": other})
    out = {"project": name, "people": people, "questions": qs, "glance": glance,
           "stats": {"chains": n, "people found (all groups)": res["people found"], "real people": len(people), "questions": len(qs)}}
    json.dump(out, open(os.path.join(ROOT, "out", "proto", f"grade_{name}.json"), "w"))
    print(out["stats"])


if __name__ == "__main__":
    main()
