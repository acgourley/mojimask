"""The scan's decide step, end to end, on a scanned project (pipeline.py runs it after detection):

  join      tracklets joined by continuity (join.py) -> PROJECT/proto/chains.json
  decide    graph, attachment, overlap and blip rules, with any answers already in the project's decision records
            (cover.py) -> PROJECT/decisions_<kind>.json
  cover     a cover per person per frame (cover.py)
  tracks    PROJECT/proto_tracks.json: one track per group, its covers made; the review server loads it in place of
            its own automatic tracks. Pieces a decision dropped are included, hidden (edits.not_face = "decision"),
            so answering them otherwise in Decisions shows them again at once.

    python -m mojimask.scan.newalgo PROJECT
"""
import json
import os
import sys
import threading
import time

from mojimask.scan import cover as CV  # noqa: E402
from mojimask.scan import graph as G  # noqa: E402
from mojimask.scan import store  # noqa: E402

# The scan's version. Box ids, and so every fact a person records about them (facts.py), belong to the scan that made
# them: a rescan throws the project's facts away. Raise this whenever a change to the scan could renumber or reshape
# boxes; a project made with another version is rescanned (losing its answers), not patched.
SCAN_VERSION = 2  # 2: graph.py's position rules need heads of about the same size (SPOT_SIZE)

EDITS0 = {"deleted": [], "trims": {}, "merges": {}, "manual": {}, "emoji": {}, "dismissed": [], "shown": [], "people": {},
          "cuts": {}, "cuts_frozen": 1}  # automatic splits frozen empty: the app doesn't cut the new tracks again


def _load(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def group_obs(info, ms):
    """{t: box} of the chains ms."""
    return {t: x for c in ms for t, x in info[c]["obs"].items()}


def tracks_for(proj):
    """(tracks, hidden, depth): the project's groups as app tracks, the ids of those a decision hides, and the depth
    model the covers were made with (MOJIMASK_DEPTH) or None."""
    from mojimask.scan import questions as Qs
    from mojimask.scan import same as SM
    b = CV.build(proj)
    covers, behind, flow = b["covers"], b["behind"], b["flow"]
    st, members, members0 = b["st"], b["members"], b["members0"]
    info, F, chains, tl = st["info"], st["F"], st["chains"], st["tl"]
    rnd = lambda d: {str(t): [round(float(v), 2) for v in o] for t, o in d.items()}  # noqa: E731
    photo = bool((_load(os.path.join(proj, "meta.json")) or {}).get("image"))  # a photo: every face is seen once (blips.py)
    tracks = {}
    for r, ms in members.items():
        obs = group_obs(info, ms)
        # one of the scan's people: in a video, seen enough and clearly; in a photo, every box it keeps
        tracks[str(r)] = {"real": photo or Qs.is_real(len(obs), G.clear_face(F, chains, tl, ms)), "obs_c": rnd(obs),
                          "cover": rnd(covers[r]),
                          **({"behind": behind[r]} if r in behind else {}),  # hidden behind someone
                          **({"flow": {str(t): m for t, m in flow[r].items()}} if flow.get(r) else {})}
    ov = _load(os.path.join(proj, "decisions_overlap.json"))
    hidden = []
    for r, ms in members0.items():
        if r in members:
            continue
        obs = group_obs(info, ms)
        tracks[str(r)] = {"real": False, "dropped": True, "obs_c": rnd(obs), "cover": rnd(obs)}
        hidden.append(str(r))
    hidden += settle(proj, tracks, hidden, ov)
    SM.find(proj, st)  # same-person questions between the people as they now stand
    return tracks, hidden, b["depth"]


TINY_PX = 24  # head box at its biggest (90th percentile of detections): under this nobody is recognisable (party clip,
# reviewer 2026-11-11: nothing under 24 px was a readable face) -- kept covered, not asked
INSIDE = 0.8  # covered by another mask: at least this much of the box inside another covered box in every frame
COVERED_BY_ANOTHER = "covered by another box"


def settle(proj, tracks, hidden, overlap=None):
    """Last pass, once covers exist: open questions whose answer can't change the video are settled by a rule.
      too small  the box is under TINY_PX: kept covered (no one could tell from it anyway)
      inside     every frame of the box is >= INSIDE within another box that stays covered: hidden (that box
                 covers it; as with overlaps.py's inside rule, showing that person must cover it again)
      asked twice  a box with an open overlap question isn't also asked as a track: the overlap question (part of
                 their face / not a face / someone else) answers it
    Your answers are never changed. overlap: PROJECT/decisions_overlap.json as already loaded, if it was. Returns the
    ids newly hidden."""
    import numpy as np
    from mojimask.scan import overlaps as OV
    hidden = set(hidden)
    covered = {k: t["cover"] for k, t in tracks.items() if k not in hidden}
    newly = []
    if overlap is None:
        overlap = _load(os.path.join(proj, "decisions_overlap.json"))
    asked_overlap = {str(d["about"]["piece"]) for d in overlap["decisions"] if d.get("by") is None} if overlap else set()
    for kind in ("track", "overlap", "blip"):
        path = os.path.join(proj, f"decisions_{kind}.json")
        doc = overlap if kind == "overlap" else _load(path)
        if doc is None:
            continue
        for d in doc["decisions"]:
            if d.get("by") is not None:
                continue
            k = str(d["about"]["piece"])
            t = tracks.get(k)
            if t is None or k in hidden:
                continue
            p90 = float(np.percentile([o[2] for o in t["obs_c"].values()], 90))
            inside = all(max([OV.covered(o, cv[ts]) for kk, cv in covered.items() if kk != k and ts in cv] or [0]) >= INSIDE
                         for ts, o in t["cover"].items())
            if kind == "track" and k in asked_overlap:
                d["answer"], d["by"] = "kept: covered", "rule: asked as an overlap question instead"
            elif inside:
                d["answer"], d["by"] = COVERED_BY_ANOTHER, f"rule: inside another covered box (>= {INSIDE:.0%} in every frame)"
                if k not in newly:
                    newly.append(k)
                covered.pop(k, None)  # hidden now, so it covers nothing: of two boxes each inside the other, one stays
            elif p90 < TINY_PX:
                d["answer"], d["by"] = "kept: covered", f"rule: too small to recognise anyone (under {TINY_PX} px)"
            else:
                continue
            d["options"] = d["options"] + [o for o in (COVERED_BY_ANOTHER, "kept: covered") if o not in d["options"]]
            d["rule_answer"], d["rule_by"] = d["answer"], d["by"]
        store.write_json(path, doc, indent=1, default=str)
    return newly


def write(dst, tracks, hidden, source=None, extra_edits=None, depth=None):
    """PROJECT/proto_tracks.json and a fresh PROJECT/edits.json. depth: the depth model the covers used (MOJIMASK_DEPTH)."""
    store.write_json(os.path.join(dst, "proto_tracks.json"),
                     {"source": source, "scan_version": SCAN_VERSION, "hidden": sorted(hidden), "tracks": tracks,
                      **({"depth": depth} if depth else {})})
    store.write_json(os.path.join(dst, "edits.json"),
                     {**EDITS0, **(extra_edits or {}), "not_face": {p: "decision" for p in hidden}})


# How far along the decide step is, from the events (viz.py): share of the step's time each stage takes (party_clip_c)
STAGES = {"follow": (0.02, 0.62), "faces": (0.62, 0.86), "cover": (0.88, 0.98)}


# One decide step at a time in a process: the app scans each project in its own thread, and the step's events go
# through one global sink (viz.sink) and module state shared by the rules. A second scan waits here.
_RUN_LOCK = threading.Lock()


def run(proj, report=lambda *a: None, events=None):
    """The scan's "decide" step. report(fraction, info); events(dict): what it is doing, for the scan screen
    (viz.py). The events are also kept in PROJECT/proto/decide_viz.json (a scan's replay)."""
    if not _RUN_LOCK.acquire(blocking=False):
        report(0.0, {"note": "waiting for another scan to finish deciding"})
        _RUN_LOCK.acquire()
    try:
        return _run_locked(proj, report, events)
    finally:
        _RUN_LOCK.release()


def _run_locked(proj, report, events):
    from mojimask.scan import viz
    proj = proj.rstrip("/")
    log, n = [], {}

    def sink(ev):
        ev["ms"] = int((time.time() - t0) * 1000)
        log.append(ev)
        if events:
            events(ev)
        e = ev["e"]
        if e == "pieces":
            n["follow"] = len(ev["pieces"])
        elif e == "faces_start":
            n["faces"] = ev["frames"]
        elif e == "people":
            n["cover"] = len(ev["people"])
        if e in STAGES and n.get(e):
            n[e + "_done"] = n.get(e + "_done", 0) + 1
            a, b = STAGES[e]
            report(a + (b - a) * min(1, n[e + "_done"] / n[e]), {})
    t0 = time.time()
    viz.sink = sink
    try:
        return _run(proj, report)
    finally:
        viz.sink = None
        try:
            store.write_json(store.path(proj, "decide_viz", "json", write=True), log)
        except OSError:
            pass


def _run(proj, report):
    from mojimask.scan import join as J
    t0 = time.time()
    report(0.02, {"note": "joining tracks"})
    # a rescan starts over: the old facts name the old scan's boxes (and the rules below mustn't read old answers)
    for f in ("facts.json", *(n for n in os.listdir(proj) if n.startswith("decisions_") and n.endswith(".json"))):
        if os.path.exists(os.path.join(proj, f)):
            os.remove(os.path.join(proj, f))
    _, chains, _ = J.run(proj)
    J.save(proj, chains)
    report(0.62, {"note": "deciding who's who"})
    tracks, hidden, depth = tracks_for(proj)
    # a new project opens on its finished video, with "Tilt & turn with the head" on (off again in Edit, per project)
    write(proj, tracks, hidden, extra_edits={"result": "pending", "head_pose": True}, depth=depth)
    n_real = sum(t["real"] for t in tracks.values())
    report(1.0, {"people": n_real, "seconds_decide": round(time.time() - t0)})
    return {"tracks": len(tracks) - len(hidden), "people": n_real, "hidden": len(hidden)}


if __name__ == "__main__":
    print(run(sys.argv[1], lambda f, i: print(f"{f:.0%}", i)))
