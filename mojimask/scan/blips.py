"""Blips: pieces seen in only 1-2 frames (<= MAX_DET detections). In the app each one is a mask that pops for a frame
or two. Never asked in a video. The rule, in the order the code applies it:
  1. big, no face   the box is BIG_PX or more and the face check never reaches FACE_FAILS: dropped (in a photo, asked
                    unless step 3 applies)
  2. sure           detector >= SURE_SCORE, face check >= SURE_FACE, or two kinds of evidence (detector >= MIN_SCORE,
                    face check >= FACE_OK, a body pose estimate with its head on the box: pose_util.pose_for_head).
                    In a video, then followed (FOLLOW_CHECK): runs into a tracked person ("met") or finds no face
                    ("nothing") -> dropped; lands on a face nobody tracks -> kept. In a photo, kept.
  3. weak           no kind of evidence at all: dropped
  4. borderline     one weak kind of evidence: dropped in a video, asked in a photo (PHOTO_ASK)
A kept blip is covered and held BLIP_HOLD frames either side (cover.py).

Evidence (2026-11-12): borderline blips used to be asked, and almost never were faces: across every answered project
152 "not a face" to 7 faces, with no evidence telling the two apart, and they were most of the questions left (jazz:
32 of 88). So a face seen only as a borderline blip goes uncovered for that frame or two; the record stays (Rules)
and "a face" there covers it. Real people's detections score median 1.00 (face check median 0.69); blips score median
~0.4 with face check 0, on nothing labelled or on a hand, neck, hair or arm beside a face. The skeleton matters: among
blips dropped without it, party 56, jazz 16 and twirl 5 had a skeleton head on them and no other cover, and by eye
many were blurred real faces or heads. Real people's own detections have a skeleton head 64-86% of the time; a "sits
on a torso" rule was rejected (16% of jazz's real faces sit on someone else's torso). The face-check numbers here
were measured with SCRFD's face check; the thresholds were kept unchanged for RetinaFace's after a four-clip bench
(2026-11-14), which passes more faces (safer).

Every blip is a decision record (overlaps.decision_record's shape, kind "blip") in PROJECT/decisions_blip.json; the
app's decisions page shows them. The rule's answer is "dropped: too weak" or "kept: covered" (it can't tell a face, only
how strong the evidence is); your answer ("a face" / "not a face") is a fact (facts.py) and overrides it.
"""
import collections
import json
import os
import sys

import numpy as np

from mojimask import facts as FX  # noqa: E402
from mojimask.scan import overlaps as OV  # noqa: E402
from mojimask import track as T  # noqa: E402
from mojimask.scan.questions import CLEAR_FACE  # noqa: E402

MAX_DET, MIN_SCORE = 2, 0.6
FACE_OK = CLEAR_FACE  # face check: a clear face (graph.CLEAR_FACE, 0.5)
# A blip whose box is BIG_PX or more and never comes near passing the face check (best < FACE_FAILS) is dropped, whatever
# the detector or skeleton say. At that size the check is decisive. Jazz census (2026-11-13), per detection: real faces
# >= 40 px pass 85-100%, non-faces 0% (under 30 px faces pass only 21%, so small boxes are left alone). On jazz it
# drops 6 kept blips: a dancer's back (82 px), shirt fronts, hands, a back of a head; no face. (Measured with SCRFD's
# face check; kept unchanged for RetinaFace's after a four-clip bench, 2026-11-14.)
BIG_PX, FACE_FAILS = 40, 0.3
# The follow check (FOLLOW_CHECK), for a blip the rule above would keep. Kept blips were kept for the passing face: a
# person crossing the picture whom no track follows. So test that directly. Follow it FOLLOW_FRAMES either way: if it
# runs into a tracked person's box, it's a stray piece of someone covered anyway; if it never lands on a face (face
# check), it's junk; only "a face nobody tracks" keeps it. Jazz census (2026-11-13), all 129 blips followed: 0 lead to
# an untracked face; the 9 face blips are 8 strays of tracked people + 1 lost at the frame edge (already dropped);
# of 116 not-faces, 26 run into a person and 90 find nothing. On the kept ones, needless covers 14 -> 4, nothing new
# uncovered. Real people's smallest heads (27-30 px) are still detected 9-50 times in proper tracks.
FOLLOW_CHECK, FOLLOW_FRAMES, FOLLOW_FACES = True, 12, 2
SURE_SCORE, SURE_FACE = OV.SURE_SCORE, 0.6  # kept without asking: one strong kind of evidence, or two kinds together
# (SURE_SCORE, 0.8: the same detector bar as the overlap and track face rules)
# A photo (a clip of one frame, video_io) sees every face once, so everything in it is a "blip", and the video priors
# don't hold: there a face is a long track and a one-frame box is almost never one; here every face is a one-frame box.
# And nothing moves, so there's nothing to follow. In a photo the clear boxes are kept without the follow check, those
# nothing speaks for are dropped as in a video, and the borderline ones are asked (PHOTO_ASK) rather than dropped: a
# face missed in a photo is in the whole result, and a photo has few questions. Measured on jazz's census frames as
# photos (photo_bench.py, 2026-11-14).
PHOTO_ASK = True
FILE = "decisions_blip.json"
OPTIONS = [FX.A_FACE, FX.A_NOT_FACE]


def find(proj, st, members):
    """The decision records, for the groups (after attachment and overlaps)."""
    info, F, chains, tl = st["info"], st["F"], st["chains"], st["tl"]
    from mojimask import pose_util as PU
    pp = os.path.join(proj, "dets", "pose_full.json")
    poses = json.load(open(pp))["frames"] if os.path.exists(pp) else None
    decisions = []
    mp = os.path.join(proj, "meta.json")
    photo = os.path.exists(mp) and bool(json.load(open(mp)).get("image"))
    clip, at = None, collections.defaultdict(list)  # tracked boxes (3+ detections) by frame, for the follow check
    if FOLLOW_CHECK and not photo:
        from mojimask.scan import follow as FO
        clip = FO.Clip(proj)
        for r, ms in members.items():
            obs = {t: o for c in ms for t, o in info[c]["obs"].items()}
            if len(obs) > MAX_DET:
                for t, o in obs.items():
                    at[t].append((r, o))
    for r, ms in members.items():
        obs, score = {}, {}
        for c in ms:
            obs.update(info[c]["obs"])
            score.update(info[c]["score"])
        if len(obs) > MAX_DET:
            continue
        best = max(score.values())
        face = max([(F.get((k, t)) or {}).get("score", 0) for c in ms for k in chains[c] for t in tl[k]["obs"]] or [0])
        # a body pose estimate whose head is on the box: evidence of a person, as a few connected detections are
        head = bool(poses) and any(t < len(poses) and PU.pose_for_head(poses[t], o) is not None for t, o in obs.items())
        kinds = (best >= MIN_SCORE) + (face >= FACE_OK) + head
        weak = kinds == 0
        size = float(np.median([o[2] for o in obs.values()]))
        seen_big = size >= BIG_PX and face < FACE_FAILS  # big enough for the face check to see a face, and it saw none
        sure = not seen_big and (best >= SURE_SCORE or face >= SURE_FACE or kinds >= 2)
        led = follows(clip, at, r, obs) if sure and FOLLOW_CHECK and not photo else None
        if led is not None and led != "untracked face":
            sure = False
        ask = photo and PHOTO_ASK and not sure and not weak
        ts = sorted(obs)
        decisions.append({
            "id": f"blip:{r}", "kind": "blip", "about": {"piece": r},
            "question": "Is the green box a face?", "options": OPTIONS + ["dropped: too weak", "kept: covered"],
            "evidence": {"frames": len(ts), "first": ts[0], "last": ts[-1], "detector_score": round(float(best), 2), "skeleton_head": head,
                         "face_check": round(float(face), 2), "size": round(float(obs[ts[0]][2]))},
            "why": f"seen in {len(ts)} frame{'s' if len(ts) > 1 else ''}, detector {best:.2f}, face check {face:.2f}, "
                   f"{'a' if head else 'no'} skeleton head on it",
            "show": [{"t": t, "piece": [round(float(v), 1) for v in T.from_cs(obs[t])], "person": None} for t in ts],
            "answer": "kept: covered" if sure else None if ask else "dropped: too weak",
            "by": (None if ask
                   else "rule: clear, and following it finds a face nobody else covers" if sure and led
                   else "rule: following it runs into a tracked person (that person's cover is there)" if led == "met"
                   else "rule: following it finds no face" if led == "nothing"
                   else f"rule: clear (detector >= {SURE_SCORE}, face check >= {SURE_FACE}, or two kinds of evidence)" if sure
                   else f"rule: big enough to check ({size:.0f} px), and no face found in it" if seen_big
                   else f"rule: nothing speaks for it (detector < {MIN_SCORE}, face check < {FACE_OK}, no skeleton head)" if weak
                   else "rule: one weak kind of evidence, seen once or twice (almost never a face; not asked)"),
        })
        decisions[-1]["rule_answer"], decisions[-1]["rule_by"] = decisions[-1]["answer"], decisions[-1]["by"]
    decisions.sort(key=lambda d: (d["answer"] is not None, -d["evidence"]["detector_score"]))
    return decisions


def follows(clip, at, r, obs):
    """Follow a blip FOLLOW_FRAMES either way with the checked follower (follow.py): "met" if it runs into a tracked
    person's box (>= half of one inside the other), else "untracked face" if it lands on a face (face check) in
    >= FOLLOW_FACES steps, else "nothing"."""
    from mojimask.scan import follow as FO

    def meet(u, c):
        return next((q for q, o in at.get(u, []) if q != r and max(OV.covered(c, o), OV.covered(o, c)) >= 0.5), None)
    ts = sorted(obs)
    faces = 0
    for direction, t0 in ((-1, ts[0]), (1, ts[-1])):
        p = FO.follow(clip, t0, obs[t0], direction, limit=FOLLOW_FRAMES, meet=meet, lost=True)
        if p.met:
            return "met"
        faces += sum(1 for h in p.how.values() if h.startswith("face"))
    return "untracked face" if faces >= FOLLOW_FACES else "nothing"


def auto(proj, st, members, write=True):
    """Drop the blips the rule says aren't faces from the graph's groups; save the records (unless write is False).
    Returns them. (Your answers are facts, facts.py: the Decisions page and the editor apply them over these.)"""
    decisions = find(proj, st, members)
    if write:
        OV.save_records(os.path.join(proj, FILE), decisions)
    for d in decisions:
        if d["answer"] in (FX.A_NOT_FACE, "dropped: too weak"):
            members.pop(d["about"]["piece"], None)
    return decisions


# ---- longer tracks: kept by rule when the face evidence is strong (a clear face in CLEAR_FRAMES frames or more, or a
# median detector score of SURE_SCORE), dropped by rule when they never show a clear face (NEVER_CLEAR, below);
# otherwise asked, "Is the green box a face?", and covered until answered. (Party clip, 2026-11-12: a lamp counted as a person -- 17 detections, median detector
# 0.49, the face check passed once -- with no record to explain it or answer it.)
CLEAR_FRAMES = OV.CLEAR_FRAMES  # 3: the same as the overlap face rules
TRACK_FILE = "decisions_track.json"
# A track that never shows one clear face (face check >= FACE_OK) and whose detector median is under SURE_SCORE is
# dropped by rule, like a weak blip (overridable: "a face" covers it). Jazz census (2026-11-13): all 27 such tracks
# were hands, a sousaphone bell, shirt fronts or a violin; real faces of the same size (24-50 px) show a clear face
# in dozens of frames, and the 2 faces with none are kept by their detector median. The party clip's lamp passed the
# face check once, so it is still asked.
NEVER_CLEAR = True
NEVER_RULE = f"rule: never a clear face (none of its frames passes the face check; median detector under {SURE_SCORE})"


def find_tracks(st, members):
    info, F, chains, tl = st["info"], st["F"], st["chains"], st["tl"]
    out = []
    for r, ms in members.items():
        obs, score = {}, {}
        for c in ms:
            obs.update(info[c]["obs"])
            score.update(info[c]["score"])
        if len(obs) <= MAX_DET:
            continue
        fs = [(F.get((k, t)) or {}).get("score", 0) for c in ms for k in chains[c] for t in tl[k]["obs"]]
        clear = sum(f >= FACE_OK for f in fs)
        med = float(np.median(list(score.values())))
        sure = clear >= CLEAR_FRAMES or med >= SURE_SCORE
        never = NEVER_CLEAR and not clear and not sure  # not one clear face in 3+ frames: dropped, not asked
        ts = sorted(obs)
        best = max(ts, key=lambda t: score.get(t, 0))
        show = sorted({ts[0], best, ts[-1]})
        out.append({
            "id": f"track:{r}", "kind": "track", "about": {"piece": r},
            "question": "Is the green box a face?", "options": OPTIONS + ["kept: covered", "dropped: too weak"],
            "evidence": {"frames": len(ts), "first": ts[0], "last": ts[-1], "clear_face_frames": clear,
                         "median_detector": round(med, 2), "best_face_check": round(float(max(fs or [0])), 2),
                         "size": round(float(np.median([o[2] for o in obs.values()])))},
            "why": f"{len(ts)} frames, a clear face in {clear}, median detector {med:.2f}",
            "show": [{"t": t, "piece": [round(float(v), 1) for v in T.from_cs(obs[t])], "person": None} for t in show],
            "answer": "kept: covered" if sure else "dropped: too weak" if never else None,
            "by": f"rule: clear (a clear face in >= {CLEAR_FRAMES} frames, or median detector >= {SURE_SCORE})" if sure
            else NEVER_RULE if never else None,
        })
        out[-1]["rule_answer"], out[-1]["rule_by"] = out[-1]["answer"], out[-1]["by"]
    out.sort(key=lambda d: (d["answer"] is not None, -d["evidence"]["frames"]))
    return out


def auto_tracks(proj, st, members, write=True):
    """Save the track records (unless write is False); drop the tracks the rule says aren't faces. Returns them."""
    decisions = find_tracks(st, members)
    if write:
        OV.save_records(os.path.join(proj, TRACK_FILE), decisions)
    for d in decisions:
        if d["answer"] in (FX.A_NOT_FACE, "dropped: too weak"):
            members.pop(d["about"]["piece"], None)
    return decisions


def main():
    from mojimask.scan import attach as AT
    from mojimask.scan import graph as G
    from mojimask.scan import questions as Qs
    proj = sys.argv[1].rstrip("/")
    res, st = G.run(proj, None, state=True)
    members = st["members"]
    for f, r in AT.attach(proj, st, Qs.groups_data(proj, st)).items():
        members[r] |= members.pop(f)
    OV.auto(proj, st, members)
    dec = auto(proj, st, members)
    n = sum(d["answer"] in (FX.A_NOT_FACE, "dropped: too weak") for d in dec)
    print(f"{len(dec)} blips: {n} dropped, {len(dec) - n} kept")


if __name__ == "__main__":
    main()
