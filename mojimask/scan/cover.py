"""The scan's covers: one box per person per frame they're on screen, detected or not (the decide step's last stage,
called by newalgo.tracks_for).

For every group the graph found (people and everything else it covers):
  detected  the detections themselves
  gaps      for real people only (as for ends): between two detections at most attach.MAX_GAP frames apart, follow forward
            from the earlier one and backward from the later one (scan/follow.py), blended by time
            (nearer side counts more)
  ends      for real people only (>= MIN_DET detections, a clear face at least once): up to END frames past the
            first and last detection, followed (a cover stays a moment too long rather than ending early), stopping
            when the box leaves the picture, or after END_HOLDS frames of holding still with no evidence of the face
            (holding is right mid-track, for someone hidden; at a track's start a held box sat on twirl's father's
            legs waiting for the girl to appear).
  behind    a held frame on a nearer person's body is a face hidden behind them: not covered (behind.py), so the
            hidden person's emoji isn't drawn over whoever is in front. Kept per person in build()'s "behind".
  lost      gaps and ends stop where following loses the face: its corners stop tracking while the rest of the picture
            tracks fine (follow.face_motion) -- someone passed in front. Frames neither side reaches are uncovered and
            go in "behind" too, so the app doesn't bridge over them.
  blips     blips and never-a-face pieces are covered where detected, plus BLIP_HOLD frames either side carried by
            the camera (not followed); extending them by following smeared covers over nothing (jazz: 934 vs 62).

    python -m mojimask.scan.cover PROJECT [--truth NAME]
Prints coverage of the labelled faces (census frames) and covers on nothing, against detections alone.
"""
import argparse
import collections
import json
import os

import numpy as np

from mojimask.paths import ROOT
from mojimask.scan import behind as BH  # noqa: E402
from mojimask import depth as DP  # noqa: E402
from mojimask import facts as FX  # noqa: E402
from mojimask.scan import follow as FO  # noqa: E402
from mojimask.scan import graph as G  # noqa: E402
from mojimask import track as T  # noqa: E402
from mojimask.scan import viz  # noqa: E402

END, END_HOLDS = 12, 2  # (gaps are followed up to attach.MAX_GAP frames, the same reach attachment uses)
MIN_DET = G.MIN_DET
LOST_MIN = 3  # a gap's frames neither side reaches are uncovered only if there are at least this many
BLIP_HOLD = 4  # frames a confident blip is held either side
BLIP_SCORE = 0.6


def decide_groups(proj, write=True):
    """The groups after the graph, attachment and the overlap and blip rules: (st, members, members0). members0: the
    groups before the overlap and blip rules dropped any. write: the rules save their records (PROJECT/decisions_*.json,
    for the Decisions page); False leaves the project's files as they are."""
    # local imports: the rule modules load when the decide step runs, not whenever newalgo or cover is imported
    from mojimask.scan import attach as AT
    from mojimask.scan import blips as BL
    from mojimask.scan import overlaps as OV
    from mojimask.scan import questions as Qs
    _, st = G.run(proj, None, state=True)
    info, members = st["info"], st["members"]
    # fragments of a real person (a fast mover breaks into them) join that person when their bodies match
    gd = Qs.groups_data(proj, st)
    att = AT.attach(proj, st, gd)
    for f, r in att.items():
        members[r] |= members.pop(f)
    print(f"attached {len(att)} fragments to people")
    members0 = {r: set(ms) for r, ms in members.items()}  # before the overlap and blip decisions drop any
    # pieces sitting inside a person's box in every frame they were detected: that person's cover already blurs them
    # (PROJECT/decisions_overlap.json)
    dec, _ = OV.auto(proj, st, members, write=write)  # (no pieces join a person here: answers are facts, applied later)
    print(f"dropped {sum((d['by'] or '').startswith('rule: inside') for d in dec)} pieces inside a person's box")
    bd = BL.auto(proj, st, members, write=write)  # weak 1-2 frame blips aren't covered (PROJECT/decisions_blip.json)
    print(f"dropped {sum(d['answer'] in (FX.A_NOT_FACE, 'dropped: too weak') for d in bd)} of {len(bd)} blips")
    td = BL.auto_tracks(proj, st, members, write=write)  # longer tracks with weak face evidence: asked, covered until answered
    print(f"tracks: {sum(d['answer'] is None for d in td)} asked, {sum(d['answer'] == FX.A_NOT_FACE for d in td)} answered not a face")
    viz.emit("people", people={int(r): sorted(int(c) for c in ms) for r, ms in members.items()},
             dropped=sorted(int(c) for r, ms in members0.items() if r not in members for c in ms),
             attached=len(att),
             real=[int(r) for r, ms in members.items()  # as below: seen long enough, a clear face at least once
                   if sum(len(info[c]["obs"]) for c in ms) >= MIN_DET and G.clear_face(st["F"], st["chains"], st["tl"], ms)])
    return st, members, members0


def cover_blip(clip, obs, cv, info, ms, how):
    """A blip, or never a face: covered where detected; a confident one also holds BLIP_HOLD frames either side,
    carried by the camera -- a fast mover (twirl's spinning girl) breaks into such blips. Adds to cv."""
    ts = sorted(obs)
    for t in ts:
        if len(ts) > 2 and max(info[c].get("score", {}).get(t, 0) for c in ms) < BLIP_SCORE:  # 1-2 detection
            # pieces here survived blips.py's evidence rule: all held, so a kept blip doesn't flicker
            continue
        for d in (1, -1):
            c = np.array(obs[t], float)
            for k in range(1, BLIP_HOLD + 1):
                u = t + d * k
                if not 0 <= u < clip.n:
                    break
                c = T.warp_cs(c, clip.camera(u - d, u))
                if u not in cv:
                    cv[u] = c.copy()
                    how["blip hold"] += 1


def cover_person(clip, obs, cv, seen, how):
    """A real person: gaps followed from both sides, ends followed outward. Adds to cv; returns the frames where
    following lost the face from both sides (someone passed in front)."""
    from mojimask.scan.attach import MAX_GAP  # local, as in decide_groups: the rule modules load when the decide step runs
    ts = sorted(obs)
    lost = set()
    for a, b in zip(ts, ts[1:]):
        if b - a <= 1 or b - a > MAX_GAP:
            continue
        fp = FO.follow(clip, a, obs[a], 1, b - a - 1, lost=True, seen=seen)
        bp = FO.follow(clip, b, obs[b], -1, b - a - 1, lost=True, seen=seen)
        fw, bw = fp.boxes, bp.boxes
        if "lost" in (fp.why, bp.why):
            miss = {u for u in range(a + 1, b) if u not in fw and u not in bw}
            if len(miss) < LOST_MIN:  # lost for a frame or two: a blink, and likely a blurred face still there
                fw = FO.follow(clip, a, obs[a], 1, b - a - 1).boxes
                bw = FO.follow(clip, b, obs[b], -1, b - a - 1).boxes
            else:
                lost |= miss
        for u in range(a + 1, b):
            w = (u - a) / (b - a)
            if u in fw and u in bw:
                cv[u] = (1 - w) * fw[u] + w * bw[u]
            elif u in fw or u in bw:
                cv[u] = fw.get(u, bw.get(u))
            else:
                continue
            how["gap"] += 1
    for start, d in ((ts[-1], 1), (ts[0], -1)):
        p = FO.follow(clip, start, obs[start], d, END, lost=True, seen=seen)
        if p.why == "lost":  # the rest of the end isn't covered: the app mustn't bridge to another box of theirs
            how["end lost"] += 1
            lost |= {u for u in range(start + d * (len(p.boxes) + 1), start + d * (END + 1), d) if 0 <= u < clip.n}
        holds = 0
        for u in sorted(p.boxes, key=lambda u: d * u):  # outward from the detection
            if p.how[u] == "hold":
                holds += 1
                if holds > END_HOLDS:
                    break  # no evidence the person is here: at a track's end they may simply not be there yet
            if u not in cv:
                cv[u] = p.boxes[u]
                how["end"] += 1
    return lost


def build(proj, write=True):
    """The decide step's groups and their covers (write: as for decide_groups): a dict of
      covers    {group: {t: box}}           how       Counter of how the cover frames were made
      behind    {group: [t]} not covered there (behind someone, or following lost them): the app mustn't bridge
      flow      {group: {t: motion}} (flows)  depth   the depth model used (MOJIMASK_DEPTH) or None
      st, members, members0   the graph's state and the groups after / before the overlap and blip rules"""
    from mojimask import pipeline  # local: pipeline imports newalgo, which imports this module
    st, members, members0 = decide_groups(proj, write)
    info = st["info"]
    clip = FO.Clip(proj)
    W = float(clip.W)
    pp = os.path.join(proj, "dets", "pose_full.json")
    poses = []
    if os.path.exists(pp):
        with open(pp) as f:
            poses = json.load(f)["frames"]
    low = T.fused_frames(os.path.join(proj, "dets"), [(d.split(":")[0], T.WEAK_DET) for d in pipeline.CORE_DETS])
    covers, how, behind = {}, collections.Counter(), {}
    seen = lambda u, c: u < len(low) and BH.face_there(c, low[u])  # noqa: E731 -- a weak detection on box c at frame u
    dep = DP.Depth.for_scan(proj)  # MOJIMASK_DEPTH (off by default): depth as the "in front" cue (behind.py)
    if dep:
        print(f"depth cue on: {dep.name} ({DP.MODELS[dep.name]['licence']})")
    for r, ms in members.items():
        obs = {}
        for c in ms:
            obs.update(info[c]["obs"])
        cv = {t: np.array(obs[t], float) for t in sorted(obs)}
        how["detected"] += len(cv)
        if len(obs) < MIN_DET or not G.clear_face(st["F"], st["chains"], st["tl"], ms):
            cover_blip(clip, obs, cv, info, ms, how)
            lost = set()
        else:
            lost = cover_person(clip, obs, cv, seen, how)
        gone = BH.behind_frames(cv, obs, poses, low, dep)  # a hold on someone nearer's body: not covered
        for u in gone:
            del cv[u]
        lost -= set(cv)
        how["behind"] += len(gone)
        how["lost"] += len(lost)
        if gone or lost:
            behind[r] = sorted(int(u) for u in gone | lost)  # not covered there; the app mustn't bridge over them
        covers[r] = cv
        viz.emit("cover", r=int(r), fill=[[int(t), round(float(cv[t][0]) / W, 4)] for t in sorted(cv) if t not in obs])
    return {"covers": covers, "how": how, "behind": behind, "flow": {r: flows(clip, cv) for r, cv in covers.items()},
            "depth": dep.name if dep else None, "st": st, "members": members, "members0": members0}


# How each cover's head moved to the next frame (follow.face_motion), for the app's steady emoji (track.steady): it
# smooths a person's boxes after carrying them along this, so a fast real move (jazz: the dancer bobbing a head's
# height every 8 frames, halved by smoothing that knew only the camera) is followed, and only the detector's wobble
# is averaged away. Kept where it agrees with the next box (within FLOW_AGREE head sizes, scale within FLOW_SCALE).
FLOW_AGREE, FLOW_SCALE = 0.5, (0.8, 1.25)


def flows(clip, cv):
    out = {}
    for t in sorted(cv):
        if t + 1 not in cv:
            continue
        c, n = cv[t], cv[t + 1]
        M, kind = FO.face_motion(clip.gray(t), clip.gray(t + 1), c)
        if M is None:
            continue
        q = M @ [c[0], c[1], 1.0]
        k = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
        if np.hypot(q[0] - n[0], q[1] - n[1]) <= FLOW_AGREE * c[2] and FLOW_SCALE[0] <= k <= FLOW_SCALE[1]:
            out[t] = [round(float(M[0, 0]), 4), round(float(M[1, 0]), 4), round(float(M[0, 2]), 2), round(float(M[1, 2]), 2)]
    return out


def score(per_frame, truth):
    """per_frame: {t: [xyxy]}. Labelled faces covered (IoU >= 0.3 with some cover), by class; covers on nothing."""
    cov, tot, on_not, on_nothing, n = collections.Counter(), collections.Counter(), 0, 0, 0
    for t, G_ in truth.items():
        boxes = per_frame.get(t, [])
        for g in G_:
            if g["c"] == "not":
                continue
            k = g["c"] + (" hidden" if g.get("o") else "")
            tot[k] += 1
            cov[k] += any(T.iou(b, g["b"]) >= 0.3 for b in boxes)
        for b in boxes:
            n += 1
            if any(T.iou(b, g["b"]) >= 0.3 for g in G_ if g["c"] != "not"):
                continue
            if any(T.iou(b, g["b"]) >= 0.3 for g in G_ if g["c"] == "not"):
                on_not += 1
            else:
                on_nothing += 1
    return {k: f"{cov[k]}/{tot[k]} = {cov[k] / tot[k]:.3f}" for k in sorted(tot)}, {"covers": n, "on a 'not a head'": on_not, "on nothing labelled": on_nothing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--truth")
    a = ap.parse_args()
    b = build(a.project)
    covers, st = b["covers"], b["st"]
    print("cover frames:", dict(b["how"]))
    name = os.path.basename(a.project.rstrip("/"))
    os.makedirs(os.path.join(ROOT, "out", "proto"), exist_ok=True)
    with open(os.path.join(ROOT, "out", "proto", f"covers_{name}.json"), "w") as f:
        json.dump({str(r): {str(t): [round(float(v), 1) for v in c] for t, c in cv.items()} for r, cv in covers.items()}, f)
    if not a.truth:
        return
    from experiments.layers.l2_detect import load_truth
    truth, W, H = load_truth(a.truth)
    det = collections.defaultdict(list)
    new = collections.defaultdict(list)
    for c in st["info"].values():
        for t, x in c["obs"].items():
            det[t].append(T.from_cs(x))
    for cv in covers.values():
        for t, x in cv.items():
            new[t].append(T.from_cs(x))
    print("detections only:", *score(det, truth))
    print("new covers:     ", *score(new, truth))


if __name__ == "__main__":
    main()
