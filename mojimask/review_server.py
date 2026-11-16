"""A project's server: the editor (review.html), Decisions, People and export for one scanned project.

    python review_server.py VIDEO DETS --workdir PROJECT [--port N]   (app_server.py starts one per project)

The scan (pipeline.py, its decide step scan/newalgo.py) leaves PROJECT/proto_tracks.json: the people it
found, their covers made. What people say about the boxes is a log of facts (facts.py, PROJECT/facts.json); the
edits (PROJECT/edits.json) hold the rest (masks drawn, splits, trims, style) and the fields derived from the facts.
"""
import argparse
import base64
import copy
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

import cv2
import numpy as np

from mojimask import coords
from mojimask import depth as DP
from mojimask import facts as FX
from mojimask import http_guard
from mojimask import lighting
from mojimask import palettes
from mojimask import portrait
from mojimask import pose_util as PU
from mojimask import track as T
from mojimask import video_io

from mojimask.paths import ROOT as HERE, WEB
DEV = os.environ.get("MOJIMASK_DEV") == "1"  # developer mode: Export labels (see app_server.py); the pose overlay is for everyone
DEV_ONLY = ("/api/export_labels", "/api/labels_export")
# The redesign's decision records (scan/, written per project by overlaps.py): every decision a rule made
# and every question still open, in one shape. A developer page lists them and takes answers (by: "user").
DECISIONS = "decisions_"  # decisions_<kind>.json: decisions_overlap.json, decisions_blip.json, ...


# ---- who counts as a person: one definition, used by People, the questions and anything else that asks.
# Every identity that's covered (drawn, not hidden) is a person, with a kind from the boxes it's made of:
#   scan   the scan's numbered people (enough detections and a clear face once: questions.is_real)
#   yours  a box you made -- a mask you drew ("m1") or a part you split off ("88~140"): you've said it's a face
#   short  a stretch too brief or unclear to count on its own (covered all the same)
# (party_clip_c: a mask joined to an 8-frame fragment was on no list and asked about by nothing.)
PERSON_KINDS = ("scan", "yours", "short")


def piece_kind(piece):
    p = str(piece)
    if p.startswith("m") or "~" in p:
        return "yours"
    return "scan" if p in PROTO_PEOPLE_SET else "short"


def people_of(pf):
    """[{tid, pieces, frames, kind}] for every person drawn in pf (resolve's per-frame boxes), most important first."""
    by = {}
    for f in pf:
        for x in f:
            tid, piece = x[0], x[-1] if len(x) == 4 else x[5]
            P = by.setdefault(tid, {"tid": tid, "pieces": set(), "frames": 0, "kind": "short"})
            P["pieces"].add(str(piece)); P["frames"] += 1
            k = piece_kind(piece)
            if PERSON_KINDS.index(k) < PERSON_KINDS.index(P["kind"]):
                P["kind"] = k
    return [{**P, "pieces": sorted(P["pieces"], key=lambda q: (len(q), q))}
            for P in sorted(by.values(), key=lambda P: (PERSON_KINDS.index(P["kind"]), -P["frames"]))]


# A box you make -- a mask you draw, a part you split off -- is a person of its own until you say whose face it is:
# one question per box, "Whose face is this?", with every person as an answer (and "someone new"), best match first.
# Best = looks most alike: faces (SFace on near-frontal views, as the scan compares them) and bodies (OSNet on the
# person's own skeleton box -- a whip-pan blur defeats the face but rarely the clothes); then nearest in time and
# place as the tiebreak. Worked out from the edits every time; the answer is a fact (same: your box + that person).
# A split-off part isn't offered the person it was split from; people you said it isn't aren't offered either.
# (party_clip_c: P2's mask at 186-212 was P1, last seen at frame 21 -- six seconds off, never asked.)
YOURS_FRAMES_PER_HEAD, LOOK_SAMPLES = 30, 8
_LOOKS, _POSES = {}, []


def _poses():
    """Skeleton poses per frame (dets/pose_full.json; [] without it). Whose-face likeness crops each body by its own
    skeleton, as the scan's same-person questions do, a box you draw is followed onto a skeleton's head, and tilt &
    turn and the pose overlay read it too. (Until 2026-11-12 an old duplicate of this function always returned [], so
    neither had ever used poses.)"""
    if not _POSES:
        with SETUP_LOCK:
            if not _POSES:
                p = os.path.join(A.detdir, "pose_full.json")
                frames = []
                if os.path.exists(p):
                    with open(p) as f:
                        frames = json.load(f)["frames"]
                _POSES.append(frames)
    return _POSES[0]


def look(piece, boxes):
    """(face, body): mean SFace embedding of the near-frontal views and mean OSNet body feature over up to LOOK_SAMPLES of
    this box's frames ({t: [cx, cy, s]}), each None when there's nothing to compare. Cached per box and frames."""
    key = (piece, tuple(sorted(boxes)))
    if key in _LOOKS:
        return _LOOKS[key]
    from mojimask.scan import faces as FA
    from mojimask import appearance as AP
    ts = sorted(boxes)
    ts = ts[::max(1, len(ts) // LOOK_SAMPLES)][:LOOK_SAMPLES]
    poses, fs, crops = _poses(), [], []
    for t in ts:
        img = cv2.imread(f"{FRAMES_DIR}/{t:05d}.jpg")
        if img is None:
            continue
        f = FA.face(img, boxes[t])
        if f and f["emb"] is not None:
            fs.append(f["emb"])
        sk = PU.pose_for_head(poses[t], boxes[t]) if poses and t < len(poses) else None
        cr = AP.crop(img, AP.body_box(boxes[t], sk))
        if cr is not None and min(cr.shape[:2]) >= 16:
            crops.append(cr)
    unit = lambda v: v / np.linalg.norm(v)  # noqa: E731
    out = (unit(np.mean(fs, 0)) if fs else None, unit(AP.reid_embed(crops).mean(0)) if crops else None)
    _LOOKS[key] = out
    return out


def likeness(a, b):
    """(score, words) for two looks, on the scan's scale (questions.score: face_term + body_term)."""
    from mojimask.scan import questions as Q
    s, parts = 0.0, []
    if a[0] is not None and b[0] is not None:
        sim = float(a[0] @ b[0]); s += Q.face_term(sim); parts.append(f"faces {sim:.2f}")  # SFace
    if a[1] is not None and b[1] is not None:
        sim = float(a[1] @ b[1]); s += Q.body_term(sim); parts.append(f"bodies {sim:.2f}")
    return s, parts


def your_box_questions(edits):
    from mojimask.scan import graph as G
    pf = T.resolve(AUTO, with_cuts(edits), MATS, N, links_for(edits))
    gone = set(edits.get("not_face") or {})
    manual = edits.get("manual") or {}
    at = defaultdict(dict)  # piece -> {t: [cx, cy, s]} (its own detections, or your box)
    tid_of = {}
    for t, f in enumerate(pf):
        for tid, c, kind, piece in f:
            tid_of[piece] = tid
            if kind in ("obs", "key") or piece in manual:
                at[piece][t] = c
    yours = [p for p in at if piece_kind(p) == "yours" and p not in gone]
    if not yours:
        return []
    inv = {}  # graph.to_room's cache of inverse camera motions
    facts = FACTS["facts"]
    comp, kept = FX.partition(facts)
    c_ = lambda q: comp.get(str(q), str(q))  # noqa: E731
    names = {"@" + k: P.get("name", "") for k, P in (edits.get("people") or {}).items()}
    persons = {P["tid"]: P for P in people_of(pf) if P["kind"] in ("scan", "yours")}
    box = lambda c: [round(float(v), 1) for v in T.from_cs(c)]  # noqa: E731
    out = []
    for mid in yours:
        mine, split = at[mid], "~" in mid
        ts = sorted(mine)
        if mid in manual:  # where you drew it or it was tracked, not its held ends (a pan carries those off)
            m_ = manual[mid]
            drawn = sorted(t for t in {int(k) for k in m_.get("keys", {})} | {int(k) for k in m_.get("prop", {})} if t in mine) or ts
        else:
            drawn = ts
        did = f"whose:{mid}"
        chosen = [f for f in facts if f.get("decision") == did]
        chosen = chosen[-1].get("answer") if chosen else None
        base = mid.split("~")[0]
        not_them = {tid_of[q] for q in at if split and q.split("~")[0] == base and q != mid and q in tid_of} - {tid_of.get(mid)}
        # its look: your box where you drew it, plus whatever it's already joined to (a detected fragment shows the face
        # the blur hid)
        own = {t: mine[t] for t in drawn}
        for q in at:
            if q != mid and tid_of.get(q) == tid_of.get(mid):
                own.update(at[q])
        me = look(tid_of.get(mid, mid), own)
        cands = []
        for tid, P in persons.items():
            ps = [q for q in P["pieces"] if q in at and q not in gone and q != mid]
            if not ps:
                continue
            is_chosen = chosen in ps
            if not is_chosen and (tid == tid_of.get(mid) or tid in not_them or frozenset((c_(mid), c_(ps[0]))) in kept):
                continue
            obs = {t: c for q in ps for t, c in at[q].items()}
            near = None  # nearest sighting in time and place
            for tm in drawn:
                to = min(obs, key=lambda u: abs(u - tm))
                (p0, s0), (p1, s1) = G.to_room(MATS, tm, mine[tm], inv), G.to_room(MATS, to, obs[to], inv)
                raw = float(np.hypot(*(p0 - p1)) / ((s0 + s1) / 2))
                cost = raw + abs(to - tm) / YOURS_FRAMES_PER_HEAD
                if near is None or cost < near[0]:
                    near = (cost, abs(to - tm), raw)
            sim, words = likeness(me, look(tid, obs))
            piece = max(ps, key=lambda q: len(at[q]))  # the answer joins your box to this one (any of theirs would do)
            if is_chosen:
                piece = chosen
            tb = max(obs, key=lambda u: obs[u][2])  # their biggest view, for the picture
            cands.append({"piece": piece, "tid": tid, "name": names.get(tid, ""), "t": tb, "box": box(obs[tb]),
                          "rank": sim - near[0] / 10, "likeness": round(sim, 2),
                          "why": ", ".join(words + [f"{near[1]} frames away"]) if words else f"{near[1]} frames away, nothing to compare"})
        cands.sort(key=lambda c: -c["rank"])
        keys = sorted(int(k) for k in (manual[mid].get("keys", {}) if mid in manual else {}) if int(k) in mine) or [drawn[len(drawn) // 2]]
        show = [{"t": t, "piece": box(mine[t]), "person": None} for t in dict.fromkeys((keys[0], keys[-1]))]
        what = "a part you split off" if split else "a mask you added"
        out.append({"id": did, "kind": "whose", "about": {"piece": mid},
                    "question": f"Whose face is this ({what})?", "options": [c["piece"] for c in cands] + [FX.A_SOMEONE_NEW],
                    "candidates": cands,
                    "evidence": {"manual": not split, "split": split, "frames": len(ts), "first": ts[0], "last": ts[-1],
                                 "candidates": len(cands)},
                    "why": f"{what}; everyone is offered, the most alike first (faces, then bodies, then nearest in time and place)",
                    "show": show, "answer": None, "by": None, "rule_answer": None, "rule_by": None})
    return out


_YOURS = {"key": None, "out": None}


def your_box_questions_now():
    """your_box_questions for the current edits, kept until the edits or the facts change (Decisions asks often)."""
    with DOC_LOCK:
        edits, key = copy.deepcopy(DOC["edits"]), (DOC["rev"], tuple(f["id"] for f in FACTS["facts"]))
    if _YOURS["key"] != key:
        out = your_box_questions(edits)
        _YOURS.update(key=key, out=out)
        return out
    return _YOURS["out"]


def _decision_files():
    return sorted(os.path.join(WORK, f) for f in os.listdir(WORK) if f.startswith(DECISIONS) and f.endswith(".json"))


ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("detdir")
ap.add_argument("--audio", default=None)
ap.add_argument("--original", default=None,
                help="original camera file: export decodes frames + audio + metadata from it (one encode generation)")
ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))
ap.add_argument("--workdir", required=True, help="the project's folder")
T.add_args(ap)
A = ap.parse_args()

WORK = A.workdir
os.makedirs(WORK, exist_ok=True)
FRAMES_DIR = f"{WORK}/frames"
EDITS = f"{WORK}/edits.json"
# One server per project: two servers would each keep their own copy of the edits. The lock is held for this
# process's life (the OS drops it when the process ends, even on a crash); the file says who holds it, so the
# app can reuse a server that is already running instead of starting a second one.
_LOCK = open(f"{WORK}/.server.lock", "a+")
try:
    fcntl.flock(_LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    _LOCK.seek(0)
    raise SystemExit(f"another review server already has this project open: {_LOCK.read().strip()}")
_LOCK.seek(0); _LOCK.truncate(); _LOCK.write(json.dumps({"pid": os.getpid(), "port": A.port})); _LOCK.flush()
os.makedirs(FRAMES_DIR, exist_ok=True)
print(f"workdir={WORK} dets={A.dets}")

# ------------------------------------------------------------------ startup
_saved = f"{WORK}/meta.json"  # the project's timeline, fixed when it was scanned
VMETA = video_io.probe(A.video, keep=json.load(open(_saved)) if os.path.exists(_saved) and os.path.exists(f"{FRAMES_DIR}/00000.jpg") else None)  # one decoder for frames, motion and export (see video_io.py)
FPS, W, H = VMETA["fps"], VMETA["width"], VMETA["height"]
# the saved positions (detections, camera motion, answers) in the pixels the decoder gives now: converted if
# they were saved in another space, which drops the frames and caches to be built again below (coords.py)
_SPACE_NOTE = coords.ensure(WORK, VMETA) if os.path.exists(_saved) else None
if not os.path.exists(f"{FRAMES_DIR}/00000.jpg"):
    print("extracting frames ...")
    video_io.extract_jpegs(A.video, FRAMES_DIR, VMETA)

spec = [d.split(":") for d in A.dets]
mats_path = f"{WORK}/mats_v2.npy"  # v2: camera motion with outlier/zoom guards
if os.path.exists(mats_path):
    MATS = list(np.load(mats_path))
else:
    print("estimating camera motion ...")
    MATS = T.camera_motion(A.video, meta=VMETA)
    np.save(mats_path, np.array(MATS))
A.frames_dir = FRAMES_DIR


# The scan's tracks (proto_tracks.json, from its decide step): one per group it found, its covers already made.
# A project without it was scanned before the current algorithm: it can't be opened (scan the video again).
PROTO_TRACKS = f"{WORK}/proto_tracks.json"
if not os.path.exists(PROTO_TRACKS):
    sys.exit(f"{WORK}: no proto_tracks.json -- scanned before the current algorithm; scan the video again")
_p = json.load(open(PROTO_TRACKS))
AUTO = {tid: {"boxes": {int(t): v for t, v in tr["cover"].items()}, "obs": sorted(int(t) for t in tr["obs_c"]),
              "obs_c": {int(t): v for t, v in tr["obs_c"].items()}, "cover": {int(t): v for t, v in tr["cover"].items()},
              "behind": set(tr.get("behind", [])),  # frames hidden behind someone nearer: never drawn (track.resolve)
              "flow": {int(t): m for t, m in (tr.get("flow") or {}).items()}}  # the head's own motion (track.steady)
        for tid, tr in _p["tracks"].items()}
PROTO_PEOPLE_SET = {tid for tid, tr in _p["tracks"].items() if tr.get("real")}  # its numbered people
# everything any detector half-believes: evidence a cover being followed is still on a face (face_evidence)
LOW = T.fused_frames(A.detdir, [(n, T.WEAK_DET) for n, _ in spec])
N = len(LOW)
print(f"{len(AUTO)} auto tracks over {N} frames")


def _read_edits():
    blank = {"deleted": [], "trims": {}, "merges": {}, "manual": {}, "emoji": {}, "dismissed": [], "shown": [],
             "people": {}}
    if os.path.exists(EDITS):
        return {**blank, **json.load(open(EDITS))}  # a partial file (e.g. just a starting palette) fills in the rest
    return {**blank, "head_pose": True}  # no edits yet (a fresh test copy): as a new project starts, tilt & turn on


def _write_json(path, obj):
    """Replace a file whole: write a temporary file, then rename it over the old one. A crash mid-save never
    leaves a half-written file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


# Notes shown once in the editor (a project whose frames were re-measured: coords.py)
NOTICES = [_SPACE_NOTE] if _SPACE_NOTE else []
# (Box ids are the scan's and never change for a project -- facts and edits name them -- so nothing is re-matched when
# a refresh changes covers: tools/bench/refresh.py checks every box's detections are unchanged first.)


# ------------------------------------------------------------------ the edits document
# The project's edits have ONE owner: this process. Pages (Review, the editor, a second window) keep a working
# copy and send only what they changed, as small operations on paths (edits_sync.js computes them):
#   {"op": "set", "path": ["not_face", "142"], "value": "you"}     set a value (objects are created on the way)
#   {"op": "del", "path": ["trims", "79"]}                          remove a key
#   {"op": "add" / "remove", "path": ["not_real"], "value": "150"}  add to / take from a set-like list
# so two pages can never overwrite each other's answers with a stale copy (the old whole-file save let the
# last writer win). Every change is saved at once (write + rename) and is one step on a shared undo history.
EDITS_VERSION = 2
SET_PATHS = {("deleted",), ("not_real",), ("dismissed",), ("shown",), ("not_same",)}  # + people.<pid>.pieces
DOC_LOCK = threading.RLock()
# rev starts from the clock, so a page that saw an earlier server's revisions can't mistake this one's for its own
DOC = {"edits": _read_edits(), "rev": int(time.time() * 1000), "undo": [], "redo": []}
DOC["edits"]["version"] = EDITS_VERSION


def load_edits():
    """A private copy of the current edits (callers may change it freely)."""
    with DOC_LOCK:
        return copy.deepcopy(DOC["edits"])


def _is_set(path):
    return tuple(path) in SET_PATHS or (len(path) == 3 and path[0] == "people" and path[2] == "pieces")


def _apply_op(doc, op):
    """Apply one op to doc in place; returns the ops that undo it ([] if it changed nothing)."""
    kind, path = op["op"], [str(p) for p in op["path"]]
    if not path or kind not in ("set", "del", "add", "remove"):
        raise ValueError(f"bad op {op}")
    parent = doc
    for k in path[:-1]:
        if not isinstance(parent.get(k), dict):
            if kind == "del" or kind == "remove":
                return []
            parent[k] = {}
        parent = parent[k]
    k = path[-1]
    if kind in ("add", "remove"):
        if not _is_set(path):
            raise ValueError(f"not a set: {path}")
        lst = parent.get(k) if isinstance(parent.get(k), list) else []
        key = json.dumps(op["value"], sort_keys=True)
        have = any(json.dumps(x, sort_keys=True) == key for x in lst)
        if kind == "add" and not have:
            parent[k] = lst + [op["value"]]
            return [{"op": "remove", "path": path, "value": op["value"]}]
        if kind == "remove" and have:
            parent[k] = [x for x in lst if json.dumps(x, sort_keys=True) != key]
            return [{"op": "add", "path": path, "value": op["value"]}]
        return []
    had, old = k in parent, copy.deepcopy(parent.get(k))
    if kind == "set":
        if had and json.dumps(old, sort_keys=True) == json.dumps(op["value"], sort_keys=True):
            return []
        parent[k] = op["value"]
    else:
        if not had:
            return []
        del parent[k]
    return [{"op": "set", "path": path, "value": old}] if had else [{"op": "del", "path": path}]


def _remember(step):
    """A new change: one step on the undo history; whatever was undone before it can no longer be redone."""
    DOC["undo"].append(step)
    del DOC["undo"][:-200]
    DOC["redo"].clear()


def apply_ops(ops, label="", undoable=True, inverse_out=None):
    """Apply a batch of ops as one change: saved at once, one undo step. Returns the new revision.
    inverse_out (a list): gets the ops that would take this change back (undo and redo use it)."""
    with DOC_LOCK:
        work = copy.deepcopy(DOC["edits"])  # all or nothing: a bad op leaves the document as it was
        inverse = []
        for op in ops:
            inverse = _apply_op(work, op) + inverse
        if inverse_out is not None:
            inverse_out.extend(inverse)
        if not inverse:
            return DOC["rev"]
        DOC["edits"] = work
        DOC["rev"] += 1
        if undoable:
            _remember({"ops": inverse, "label": label})
        _write_json(EDITS, work)
        return DOC["rev"]


def _take(step):
    """Apply a history step (an undo or a redo); returns the step that takes it back."""
    if "facts" in step:
        remove, restore = step["facts"]["remove"], step["facts"]["restore"]
        gone = [f for f in FACTS["facts"] if f["id"] in set(remove)]
        FX.retract(FACTS, remove)
        FACTS["facts"] = sorted(FACTS["facts"] + restore, key=lambda f: f["id"])
        FX.save(WORK, FACTS)
        sync_facts()
        return {"facts": {"remove": [f["id"] for f in restore], "restore": gone}, "label": step["label"]}
    back = []
    apply_ops(step["ops"], undoable=False, inverse_out=back)
    if any((o.get("path") or [""])[0] == "palette" for o in step["ops"]):
        sync_facts()  # (the preset came back: so do the picks made under it)
    return {"ops": back, "label": step["label"]}


def undo_last(redo=False):
    """Undo the last change (from any page), or redo the last undone one. Returns its label, or None if there is
    nothing to undo (redo)."""
    take, put = ("redo", "undo") if redo else ("undo", "redo")
    with DOC_LOCK:
        if not DOC[take]:
            return None
        step = DOC[take].pop()
        DOC[put].append(_take(step))
        return step["label"] or "last change"


# ---- facts (the new algorithm's projects): who's who and what's a face is one log, PROJECT/facts.json (facts.py).
# The edits fields below are derived from it after every change and never written by pages; Decisions' answers are
# read from it. Undo takes the last facts back out of the log, like any other change.
DERIVED = ("people", "emoji", "not_face", "part_of", "merges", "not_same", "shown", "attach")
FACTS = FX.load(WORK, _p.get("scan_version", 0))
HIDDEN = _p.get("hidden")
if HIDDEN is None:  # a scan from before the list was kept: its hidden boxes are the edits' "decision" ones
    HIDDEN = sorted(p for p, v in (DOC["edits"].get("not_face") or {}).items() if v == "decision")


def sync_facts():
    """Write the facts' consequences into the edits (not an undo step of its own)."""
    preset = DOC["edits"].get("palette")  # hand-picked emoji belong to the preset they were picked under
    apply_ops([{"op": "set", "path": [k], "value": v} for k, v in FX.derive(FACTS["facts"], HIDDEN, preset).items()], undoable=False)


sync_facts()  # at start: the edits as the facts stand (a log set aside for another scan version starts empty)


def change_facts(add=(), remove=(), label="", page="", decision=None, via=None):
    """Retract facts (ids) and add new ones as one change and one undo step. Returns the new facts' ids."""
    with DOC_LOCK:
        gone = [f for f in FACTS["facts"] if f["id"] in set(remove)]
        FX.retract(FACTS, remove)
        add = [{**f, "preset": DOC["edits"].get("palette") or FX.EMOJI_PRESET} if f.get("kind") == "emoji" and not f.get("preset") else f
               for f in add]  # an emoji pick is for the preset in use
        ids = FX.append(FACTS, add, page, decision, via)
        FX.save(WORK, FACTS)
        if ids or gone:
            _remember({"facts": {"remove": ids, "restore": gone}, "label": label})
        sync_facts()
        return ids


def doc_reply(extra=None):
    """What a page needs after a change: the new edits and revision, and what they resolve to."""
    with DOC_LOCK:
        edits, rev = copy.deepcopy(DOC["edits"]), DOC["rev"]
    return {"rev": rev, "edits": edits, "can_undo": bool(DOC["undo"]), "can_redo": bool(DOC["redo"]), **(extra or {}), **state(edits)}


_LINKS_CACHE = {}


# One lock for the shared caches below (manual embeddings, identity graphs, links, the preview's resolved
# tracks, head pose). The server is threaded and the editor fires dozens of /api/frame requests at once while
# scrubbing: without it, every request rebuilt the same caches at the same time after an edit, and they
# crashed each other ("dictionary changed size during iteration") -> frames without emoji. Re-entrant
# because these call each other; the build happens once and the other requests wait for it.
SETUP_LOCK = threading.RLock()


def with_cuts(edits):
    """edits with edits.cuts always present (the splits you made; the scan never cuts its tracks)."""
    return edits if "cuts" in edits else {**edits, "cuts": {}}


def links_for(edits):
    """Same-person links for these answers: {piece: root}, fed to track.resolve as the automatic links
    (the reviewer's own merges and People still override them there)."""
    with SETUP_LOCK:
        return _links_for(edits)


def _links_for(edits):
    people = edits.get("people") or {}
    key = json.dumps([with_cuts(edits)["cuts"], {k: m.get("keys") for k, m in (edits.get("manual") or {}).items()}, edits.get("merges") or {}, edits.get("attach") or {}, {k: P.get("pieces", []) for k, P in people.items()},
                      edits.get("not_same") or [], sorted(edits.get("deleted") or []),
                      sorted(edits.get("not_face") or {}), sorted(edits.get("not_real") or [])], sort_keys=True)
    if key not in _LINKS_CACHE:
        must = [(a, b) for a, b in (edits.get("merges") or {}).items() if a != b]
        # attach {piece: piece}: a box answered Not a face stays with the person it belongs to (no evidence,
        # and unlike a merge it doesn't make that group one a reviewer has defined)
        must += [(a, b) for a, b in (edits.get("attach") or {}).items() if a != b]
        for P in people.values():
            ps = P.get("pieces", [])
            must += list(zip(ps, ps[1:]))
        if len(_LINKS_CACHE) > 50:
            _LINKS_CACHE.clear()
        # the scan's tracks are already its people: no automatic links on top, only the joins in the edits
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for a, b in must:
            parent[find(a)] = find(b)
        groups = defaultdict(list)
        for x in list(parent):
            groups[find(x)].append(x)
        _LINKS_CACHE[key] = {x: min(g, key=lambda q: (len(str(q)), str(q))) for g in groups.values() if len(g) > 1 for x in g}
    return _LINKS_CACHE[key]


links_for({})  # warm the cache

# Scene-matched emoji lighting (lighting.py): estimated once per set of auto tracks (~30 s, cached in
# WORK), in the background so the editor opens immediately. User settings live in edits["light"].
LIGHT = {"obj": None, "json": None, "progress": 0.0, "error": None}


def _estimate_light():
    try:
        L = lighting.Lighting({tid: tr["boxes"] for tid, tr in AUTO.items()}, FRAMES_DIR, WORK,
                              progress=lambda p: LIGHT.update(progress=p))
        LIGHT.update(obj=L, json=json.dumps(L.to_json()).encode(), progress=1.0)
    except Exception as e:  # lighting is optional: the editor and export fall back to flat emoji
        LIGHT.update(error=str(e))
        print(f"lighting estimate failed: {e}")


LIGHT_THREAD = threading.Thread(target=_estimate_light, daemon=True)
LIGHT_THREAD.start()


def state(edits):
    pf = T.resolve(AUTO, with_cuts(edits), MATS, N, links_for(edits))
    shown = T.shown_pieces(edits)
    resolved = [[[tid, round(c[0], 1), round(c[1], 1), round(c[2], 1), kind, piece, piece in shown]
                 for tid, c, kind, piece in f] for f in pf]
    # boxes answered Not a face / Not a real person: not drawn, but Review shows them greyed in their group
    hidden = []
    gone = set(edits.get("not_face") or {}) | set(edits.get("not_real") or [])
    if gone:
        # without the overlap clean-up: two answered boxes on one spot must both be listed, or one would
        # vanish from Review and the editor, and Review's next save would drop its answer
        pf2 = T.resolve(AUTO, with_cuts({**edits, "not_face": {}, "not_real": []}), MATS, N, links_for(edits), dedupe=False)
        hidden = [[t, tid, round(c[0], 1), round(c[1], 1), round(c[2], 1), kind, piece]
                  for t, f in enumerate(pf2) for tid, c, kind, piece in f if piece in gone]
    return {"resolved": resolved, "hidden": hidden, "persons": people_of(pf)}


# ------------------------------------------------------------------ propagation (median flow)
@lru_cache(maxsize=400)
def gray(t):
    return cv2.imread(f"{FRAMES_DIR}/{t:05d}.jpg", cv2.IMREAD_GRAYSCALE)


def flow_step(g0, g1, c):
    """Move square box c=[cx,cy,s] from g0 to g1 by median optical flow of points inside it.
    Returns new c or None when tracking is unreliable."""
    x1, y1, x2, y2 = [int(round(v)) for v in T.from_cs(c)]
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W - 1), min(y2, H - 1)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    p0 = T.box_corners(g0, x1, y1, x2, y2)
    if p0 is None or len(p0) < 6:
        return None
    p1, good = T.lk_fb(g0, g1, p0)  # (on a crop around the box: see track.lk_fb)
    if good.sum() < 5:
        return None
    a, b = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
    d = np.median(b - a, axis=0)
    i, j = np.triu_indices(len(a), 1)
    da, db = np.linalg.norm(a[i] - a[j], axis=1), np.linalg.norm(b[i] - b[j], axis=1)
    ok = da > 2
    sc = float(np.clip(np.median(db[ok] / da[ok]), 0.9, 1.1)) if ok.sum() > 3 else 1.0
    return [c[0] + float(d[0]), c[1] + float(d[1]), c[2] * sc]


PROP_VERSION = 3   # edits.manual[id].pv: covers followed by this version of propagate()
PROP_SNAP = 0.5    # box sizes: a weak detection or skeleton head this close is the face -> snap to it
PROP_BLIND_S = 0.5  # seconds a cover may run on without any such evidence before it stops


_OBS = {tid: {int(t) for t in tr["obs"]} for tid, tr in AUTO.items()}  # frames each automatic box was detected


def _someone_else(u, step, prev, bc):
    """Is the detection bc at frame u the face of another tracked box -- one detected there that, the frame
    before, was more than a third of a box away from where this cover was (prev)? Then it's a neighbour's face,
    not this one moving: twirl.mp4, a cover on the bearded man hopped onto the dad's face beside him the frame
    the man's own detection dropped out."""
    for tid, tr in AUTO.items():
        ab = tr["boxes"].get(u)
        if ab is None or u not in _OBS[tid] or np.hypot(ab[0] - bc[0], ab[1] - bc[1]) > 0.25 * ab[2]:
            continue
        pb = tr["boxes"].get(u - step)
        if pb is not None and np.hypot(pb[0] - prev[0], pb[1] - prev[1]) > prev[2] / 3:
            return True
    return False


def face_evidence(u, c, prev=None, step=1):
    """Where frame u shows a face for a box near c=[cx,cy,s]: the closest low-threshold detection of about
    that size, or a confident skeleton head, within PROP_SNAP box sizes -> [cx,cy,s] or None. With prev (the
    box in the frame before): never a detection that is a neighbour's tracked face (_someone_else)."""
    best = None
    for b, s, srcs in LOW[u]:
        bc = T.to_cs(b)
        d = np.hypot(bc[0] - c[0], bc[1] - c[1]) / c[2]
        if d < PROP_SNAP and 0.6 < bc[2] / c[2] < 1.6 and (best is None or d < best[0]):
            if prev is not None and _someone_else(u, step, prev, bc):
                continue
            best = (d, [bc[0], bc[1], bc[2]])
    for p in (_poses()[u] if u < len(_poses()) else []):
        if p[4] < 0.25:
            continue
        xy, q = PU.pose_head(p, c[2])
        d = np.hypot(xy[0] - c[0], xy[1] - c[1]) / c[2]
        if q >= 0.5 and d < PROP_SNAP and (best is None or d < best[0]):
            best = (d, [float(xy[0]), float(xy[1]), c[2]])
    return best and best[1]


def propagate(t, c, limit=60):
    """Follow a box you placed (a cover) forward and backward from frame t -> (prop {frame: box}, ev [frames]).
    Each frame: local flow moves it; a weak detection or skeleton head within half a box is taken as the face
    and the box snaps to it (those frames are the evidence, ev). It stops when flow fails, when it has gone
    PROP_BLIND_S without evidence, or when an automatic emoji already covers it for 3 frames (handoff).
    Plain flow used to run the full 60 frames each way: on a child being spun it slid onto the dress, the
    furniture and the dad, and track.resolve drew it instead of her detected face (twirl.mp4, 2026-11-10)."""
    prop, ev = {}, []
    blind_max = max(3, round(PROP_BLIND_S * FPS))
    for step in (1, -1):
        cur, covered, blind = list(c), 0, 0
        for k in range(1, limit + 1):
            u = t + step * k
            if not 0 <= u < N:
                break
            nxt = flow_step(gray(u - step), gray(u), cur)
            if nxt is None or not (0 <= nxt[0] < W and 0 <= nxt[1] < H):
                break
            seen = face_evidence(u, nxt, cur, step)
            if seen:
                nxt, blind = seen, 0
                ev.append(u)
            else:
                blind += 1
                if blind > blind_max:
                    break
            nxt[2] = float(np.clip(nxt[2], 0.67 * c[2], 1.5 * c[2]))  # size stays near the placed box's
            cur = nxt
            prop[u] = [round(v, 1) for v in cur]
            b = T.from_cs(cur)
            auto_here = [tr["boxes"].get(u) for tr in AUTO.values()]
            # an automatic box on this face: overlapping, or centred on it (a cover can be bigger than the
            # face, e.g. seeded from a box that drifted off a nearer, bigger face, so overlap alone misses it)
            covered = covered + 1 if any(ab is not None and (T.iou(T.from_cs(ab), b) > 0.4 or
                                         np.hypot(ab[0] - cur[0], ab[1] - cur[1]) < 0.25 * min(ab[2], cur[2]))
                                         for ab in auto_here) else 0
            if covered >= 3:
                break
    return prop, sorted(ev)


def split_cover(mid, t):
    """Split cover mid (edits.manual) at frame t: the part before t stays mid, the rest becomes a new cover,
    each with explicit first/last frames so neither carries on over the other. A half without a keyframe gets
    one where its path is, at its first (or last) frame. The new cover joins mid's card (its named Person, else
    a link to mid), like a split track; you then move whichever half is wrong. One step on undo. -> new id."""
    e = load_edits()
    m = (e.get("manual") or {}).get(mid)
    if not m:
        raise ValueError(f"no cover {mid}")
    bx = T.manual_boxes(m, MATS, N)
    ts = sorted(u for u in bx if 0 <= u < N)
    if not ts or not ts[0] < t <= ts[-1]:
        raise ValueError(f"cover {mid} runs {ts[0] if ts else '-'}..{ts[-1] if ts else '-'}: can't split at {t}")
    rnd = lambda c: [round(float(v), 1) for v in c]
    halves = []
    for lo, hi in ((ts[0], t - 1), (t, ts[-1])):
        part = lambda d: {k: v for k, v in d.items() if lo <= int(k) <= hi}
        keys, prop = part(m.get("keys") or {}), part(m.get("prop") or {})
        if not keys:
            at = lo if lo == t else hi
            keys = {str(at): rnd(bx[at])}
            prop.pop(str(at), None)
        h = {"keys": keys, "prop": prop, "ev": [u for u in m.get("ev") or [] if lo <= u <= hi], "from": lo, "to": hi}
        if "pv" in m:
            h["pv"] = m["pv"]
        halves.append(h)
    new = "m" + str(max([int(k[1:]) for k in e["manual"] if k[1:].isdigit()] + [0]) + 1)
    ops = [{"op": "set", "path": ["manual", mid], "value": halves[0]},
           {"op": "set", "path": ["manual", new], "value": halves[1]}]
    pid = next((p for p, P in (e.get("people") or {}).items() if mid in (P.get("pieces") or [])), None)
    if pid:
        ops.append({"op": "add", "path": ["people", pid, "pieces"], "value": new})
    else:
        ops.append({"op": "set", "path": ["merges", new], "value": mid})
    why = (e.get("not_face") or {}).get(mid)
    if why:  # a box marked not a face: so is its other half, until you say otherwise
        ops.append({"op": "set", "path": ["not_face", new], "value": why})
    apply_ops(ops, f"split {mid} at {t}")
    return new


def undo_correction(mid, t):
    """Take back where you placed cover mid on frame t (a keyframe: drawn there, or dragged there to correct
    it) and re-follow the cover from your other placements, so the path that placement made goes too, not
    just the one frame. The cover's only placement can't go (use Not a face, or end / start the box). One
    undo step."""
    e = load_edits()
    m = (e.get("manual") or {}).get(mid)
    if not m:
        raise ValueError(f"no cover {mid}")
    keys = {int(k): v for k, v in (m.get("keys") or {}).items()}
    if t not in keys:
        raise ValueError(f"you didn't place {mid} on frame {t}")
    if len(keys) == 1:
        raise ValueError("that's the only place this box was set: use Not a face, or Start / End box here")
    rest = {k: v for k, v in keys.items() if k != t}
    prop, ev = {}, set()
    for k, c in rest.items():  # in the order they were placed; the first to reach a frame sets it
        p, e_ = propagate(k, [float(v) for v in c])
        for u, v in p.items():
            if u not in rest and str(u) not in prop:
                prop[str(u)] = v
        ev |= set(e_)
    apply_ops([{"op": "set", "path": ["manual", mid, "keys"], "value": {str(k): v for k, v in rest.items()}},
               {"op": "set", "path": ["manual", mid, "prop"], "value": prop},
               {"op": "set", "path": ["manual", mid, "ev"], "value": sorted(ev)},
               {"op": "set", "path": ["manual", mid, "pv"], "value": PROP_VERSION}], f"undo correction on {mid} at {t}")


def refollow_covers():
    """Once per cover: re-follow covers made by an older propagate() (no edits.manual[id].pv) from their own
    keyframes, so a project made before the change gets the shorter, evidence-snapped paths too. Like a page
    building a cover (app.html coverFor), each keyframe is followed in turn, first one first, and the
    first keyframe to reach a frame sets it. Saved as one change that undo skips: it only re-derives paths
    from your keyframes and changes no answer."""
    with DOC_LOCK:
        manual = copy.deepcopy(DOC["edits"].get("manual") or {})
    ops = []
    for mid, m in manual.items():
        if m.get("pv") == PROP_VERSION or not m.get("keys"):
            continue
        prop, ev = {}, set()
        for t, c in m["keys"].items():
            p, e = propagate(int(t), [float(v) for v in c])
            for u, v in p.items():
                if str(u) not in m["keys"] and str(u) not in prop:
                    prop[str(u)] = v
            ev |= set(e)
        ops += [{"op": "set", "path": ["manual", mid, "prop"], "value": prop},
                {"op": "set", "path": ["manual", mid, "ev"], "value": sorted(ev)},
                {"op": "set", "path": ["manual", mid, "pv"], "value": PROP_VERSION}]
    if not ops:
        return
    with DOC_LOCK:  # skip any cover that was deleted or re-followed by a page meanwhile
        cur = DOC["edits"].get("manual") or {}
        ops = [o for o in ops if o["path"][1] in cur and cur[o["path"][1]].get("pv") != PROP_VERSION]
        if ops:
            apply_ops(ops, undoable=False)
            print(f"re-followed {len(ops) // 3} covers (propagate v{PROP_VERSION})")


# ------------------------------------------------------------------ rendering
RENDER = {"status": "idle", "progress": 0.0, "file": None, "path": None, "error": None}
RENDER_LOCK = threading.Lock()  # one export at a time: checked and marked started together (POST /api/render)

# Where an export goes, asked before it starts. The Mac app asks with its own Save dialog (desktop.py); in a browser
# on this Mac (localhost) the server asks macOS for one instead (AppleScript's choose file name), so both pick a place
# first and both can show the file in Finder afterwards.
CHOOSE_OUTPUT = """on run argv
  activate
  set f to choose file name with prompt (item 3 of argv) default name (item 1 of argv) default location (POSIX file (item 2 of argv))
  return POSIX path of f
end run"""


VIDEO_OUT, PHOTO_OUT = (".mp4", ".m4v", ".mov"), (".jpg", ".jpeg", ".png")  # what an export can be saved as


def out_types():
    """(extensions an export can have, the default one): a photo exports as a photo."""
    return (PHOTO_OUT, ".jpg") if VMETA.get("image") else (VIDEO_OUT, ".mp4")


def choose_output(name, directory):
    """{"path": ...}, {"cancel": true} or {"unsupported": true} (not macOS)."""
    if sys.platform != "darwin":
        return {"unsupported": True}
    photo = bool(VMETA.get("image"))
    if not directory or not os.path.isdir(directory):
        directory = os.path.expanduser("~/Pictures" if photo else "~/Movies")
    r = subprocess.run(["osascript", "-e", CHOOSE_OUTPUT, name, directory,
                        "Save the anonymized photo" if photo else "Save the anonymized video"],
                       capture_output=True, text=True, timeout=600)
    path = r.stdout.strip()
    if r.returncode or not path:
        return {"cancel": True}
    exts, default = out_types()
    return {"path": path if path.lower().endswith(exts) else path + default}


def reveal_export():
    p = RENDER.get("path")
    if sys.platform == "darwin" and p and os.path.exists(p):
        subprocess.run(["open", "-R", p])
        return True
    return False


HEAD_POSE = {"key": None, "angles": None}


def head_pose_angles(pf):
    """Tilt/turn per cover (head_pose.track_angles) from the scan's pose skeletons; cached per resolved tracks."""
    from mojimask import head_pose as HP
    poses = _poses()
    if not poses:
        return None
    with SETUP_LOCK:
        key = hashlib.md5(json.dumps(pf).encode()).hexdigest()
        if HEAD_POSE["key"] != key:
            HEAD_POSE.update(key=key, angles=HP.track_angles(pf, poses))
        return HEAD_POSE["angles"]


def render_setup(edits, opts=None, wait_light=True):
    """Everything that decides how covers look, shared by the export AND the live preview
    (/api/frame) so the two can never drift apart. opts: style in track.STYLES, strict (bool: whole-head
    solid covers). Returns (per_frame, draw_frame kwargs)."""
    opts = opts or {}
    style = opts.get("style") if opts.get("style") in T.STYLES else "emoji"
    strict = str(opts.get("strict", "")).lower() in ("1", "true", "yes")
    if strict:
        style = "solid"
    scale = A.scale * (1.3 if strict else 1.0)  # strict: cover whole heads (hair, hats, head shape)
    pf = T.resolve(AUTO, with_cuts(edits), MATS, N, links_for(edits))
    st = lighting.settings(edits)
    if style not in ("emoji", "portrait"):
        st = {**st, "enabled": False}  # lighting only shades pasted pictures (emoji, the portrait photo)
    if st["enabled"] and wait_light:
        LIGHT_THREAD.join()  # export waits for the lighting estimate rather than silently going flat
    light = lighting.FrameLighter(LIGHT["obj"], st, pf) if st["enabled"] and LIGHT["obj"] else None
    kw = dict(scale=scale, overrides=palettes.assignment(pf, edits),
              shown=T.shown_pieces(edits), light=light, style=style,
              head_pose=head_pose_angles(pf) if style == "portrait" or style == "emoji" and edits.get("head_pose")
              else None,  # the portrait is always matched to the head; for emoji it's the 'tilt & turn' option
              portrait=portrait.key_for(portrait.project_photo(WORK, edits)) if style == "portrait" else None,
              saturation=lighting.film_settings(1.0, 0.0, lighting.settings(edits))[0],  # used where lighting isn't
              occlusion=occlusion())
    return pf, kw


# Partial occlusion (depth.py; MOJIMASK_DEPTH on and this project's depth maps computed by a scan or refresh.py with it on):
# the part of a cover behind something nearer is drawn as the video. Built once, on first use.
OCCL = {}


def _depth_setting():
    """MOJIMASK_DEPTH's model, read once at start; a name depth.py doesn't know is logged and treated as off."""
    try:
        return DP.setting()
    except ValueError as e:
        print(f"{e}: depth off, covers drawn whole", flush=True)
        return None


DEPTH = _depth_setting()


def occlusion():
    if "obj" not in OCCL:
        name, obj = DEPTH, None
        if name and DP.cached(WORK, name):
            obj = DP.Occlusion(DP.Depth(WORK, name), {k: tr["obs_c"] for k, tr in AUTO.items()})
            print(f"partial occlusion on: {name}", flush=True)
        elif name:
            print(f"MOJIMASK_DEPTH={name}, but {WORK}/dets/depth_{name}.mp4 isn't computed (scan or refresh.py with it on): "
                  "covers drawn whole", flush=True)
        OCCL["obj"] = obj
    return OCCL["obj"]


FRAMING = ("preset", "fit", "start", "end", "out", "mark")  # export/preview framing options (not cover settings)
_yes = lambda v: str(v).lower() in ("1", "true", "yes")


def export_settings(opts):
    from mojimask import formats
    opts = opts or {}
    preset = opts.get("preset") if opts.get("preset") in formats.BY_ID else "original"
    fit = opts.get("fit") if opts.get("fit") in formats.FITS else "fill"
    num = lambda v: float(v) if v not in (None, "") else None
    return formats.settings(VMETA, preset, fit, num(opts.get("start")), num(opts.get("end")), frames=N)


def do_render(edits, opts=None):
    """Export. opts: cover settings (style, strict) + framing (preset, fit, start, end, mark: the corner mark) + out (absolute
    .mp4 path chosen in the app's Save dialog -- .jpg or .png for a photo; default: the project's final.mp4 / .jpg)."""
    opts = opts or {}
    try:  # (RENDER was set to rendering by POST /api/render before this thread started)
        exts, default = out_types()
        final = os.path.abspath(f"{WORK}/final{default}")
        out = os.path.abspath(os.path.expanduser(opts.get("out") or final))
        if not out.lower().endswith(exts) or not os.path.isdir(os.path.dirname(out)):
            raise ValueError(f"can't save to {out}")
        pf, kw = render_setup(edits, opts)
        src = A.original or A.video
        T.render(src, pf, out, A.audio or A.original, kw.pop("scale"), kw.pop("overrides"),
                 progress=lambda p: RENDER.update(progress=p), output=export_settings(opts), meta=VMETA,
                 mark=_yes(opts.get("mark")), **kw)
        RENDER.update(status="done", progress=1.0, path=out,
                      file="api/download" if out == final else None)
    except Exception as e:  # surface to UI
        RENDER.update(status="error", error=str(e))


PREVIEW = {"key": None, "pf": None, "kw": None, "sprites": {}}


def preview_frame(t, opts):
    """One frame through the export's own drawing code with the current edits and options -> JPEG bytes (None if
    the frame is missing). The resolved tracks are cached until the edits or options change."""
    from mojimask import formats
    edits = load_edits()
    cover = {k: v for k, v in opts.items() if k not in FRAMING and k != "v"}
    key = hashlib.md5(json.dumps([edits, cover, LIGHT["obj"] is not None], sort_keys=True).encode()).hexdigest()
    with SETUP_LOCK:  # the first request after a change builds this; concurrent ones wait and reuse it
        if PREVIEW["key"] != key:
            pf, kw = render_setup(edits, cover, wait_light=False)
            PREVIEW.update(key=key, pf=pf, kw=kw)
        pf, kw = PREVIEW["pf"], PREVIEW["kw"]  # one consistent pair even if another edit lands mid-draw
    f = cv2.imread(f"{FRAMES_DIR}/{t:05d}.jpg")
    if f is None:
        return None
    out = T.draw_frame(f, t, pf[t], PREVIEW["sprites"], **kw)
    if opts.get("preset") and opts.get("preset") != "original":  # show the export's framing
        st = export_settings(opts)
        k = min(1.0, 960 / max(st["width"], st["height"]))  # preview size
        out = formats.fit_frame(out, max(2, round(st["width"] * k)), max(2, round(st["height"] * k)), st["fit"])
    if _yes(opts.get("mark")):
        out = T.watermark(out)
    return cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def depth_frame(t):
    """The editor's depth view for frame t: what depth.Occlusion decided for each cover drawn there."""
    with SETUP_LOCK:
        if PREVIEW["key"] is None:
            pf, kw = render_setup(load_edits(), {}, wait_light=False)
            PREVIEW.update(key="depth-view", pf=pf, kw=kw)
        pf, kw = PREVIEW["pf"], PREVIEW["kw"]
    shown = kw.get("shown") or set()
    items = [x for x in T.draw_order(pf[t], t, occlusion()) if not (x[3:] and x[3] in shown)]
    return {"covers": occlusion().explain(t, items, kw.get("scale", A.scale), (H, W)), "model": occlusion().D.name,
            "face_zone": DP.FACE_ZONE[0], "zone_dy": T.FACE_DY}  # the face zone's geometry, for depth_ui.js to draw


# ------------------------------------------------------------------ pose debug overlay
def pose_frame(t):
    """Skeletons for the editor's pose overlay: [{body, facing, head:[cx,cy,hw]|None, score, kps:[51]}].
    body: always None (a body id came from an old tracker; pose_ui.js still reads it)."""
    poses = _poses()
    out = []
    for p in (poses[t] if 0 <= t < len(poses) else []):
        if p[4] < 0.25:
            continue
        h = PU.head(p)
        out.append({"body": None, "facing": PU.facing(p),
                    "head": [round(v, 1) for v in h[:3]] if h else None, "score": round(p[4], 2),
                    "kps": [round(v, 1) for v in p[5:56]]})
    return out


# ------------------------------------------------------------------ training labels
def export_labels(edits):
    """Clean labels for training/eval, independent of render choices:
      faces       every tracked face after review (incl. people shown unmasked: they are still faces)
      not_faces   boxes of auto tracks the reviewer deleted, on frames where a detector fired
    Boxes are square head boxes [x1,y1,x2,y2] in source pixels."""
    pf = T.resolve(AUTO, with_cuts(edits), MATS, N, links_for(edits))
    shown = T.shown_pieces(edits)
    people = edits.get("people") or {}
    deleted = [d for d in [*edits.get("deleted", []), *(edits.get("not_face") or {})] if d in AUTO]
    frames = []
    for t in range(N):
        faces = [{"person": tid, "name": (people.get(tid[1:], {}).get("name") if tid.startswith("@") else None),
                  "piece": piece, "box": [round(v, 1) for v in T.from_cs(c)],
                  "kind": kind, "shown": piece in shown} for tid, c, kind, piece in pf[t]]
        nf = [{"piece": d, "box": [round(v, 1) for v in T.from_cs(AUTO[d]["boxes"][t])]}
              for d in deleted if t in AUTO[d]["obs"] and t in AUTO[d]["boxes"]]
        frames.append({"t": t, "faces": faces, "not_faces": nf})
    _write_json(f"{WORK}/labels_export.json",
                {"video": A.original or A.video, "w": W, "h": H, "fps": FPS, "frames": frames})
    return {"file": "api/labels_export", "faces": sum(len(f["faces"]) for f in frames),
            "not_faces": sum(len(f["not_faces"]) for f in frames)}


# ------------------------------------------------------------------ http


class BadRequest(ValueError):
    """A request the server can't make sense of (answered 400)."""


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        raise BadRequest(f"not a number: {v!r}") from None


def _load_json(path):
    with open(path) as f:
        return json.load(f)


class Handler(http_guard.GuardMixin, SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=WEB, **k)

    def log_message(self, *a):
        pass

    def end_headers(self):
        if http_guard.own_code(self.path):
            self.send_header("Cache-Control", "no-store")  # always load the current editor code
        self._sent = True
        super().end_headers()

    def _send(self, data, ctype, cache=None, extra=()):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        if cache:
            self.send_header("Cache-Control", cache)
        for k, v in extra:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except ValueError:
            raise BadRequest("the request isn't JSON") from None
        if not isinstance(body, dict):
            raise BadRequest("the request isn't a JSON object")
        return body

    def _guarded(self, handle):
        """Run a request: bad input answers 400, anything else that goes wrong 500 (logged), not a dropped connection."""
        self._sent = False
        try:
            handle()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the page stopped waiting (scrubbing drops frames it no longer wants)
        except BadRequest as e:
            if not self._sent:
                self._json({"error": str(e)}, 400)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            if not self._sent:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_GET(self):
        self._guarded(self._get)

    def do_POST(self):
        self._guarded(self._post)

    def _get(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html", "/review.html") and "app=" not in self.path:
            # the editor only runs inside the app (its Edit step); the old standalone editor is gone
            from mojimask import about
            body = (f"<!doctype html><meta charset=utf-8><title>{about.NAME}</title><body style='font:15px -apple-system,"
                    "sans-serif;padding:40px;background:#0f1012;color:#ececec'>This project's editor opens inside the app: "
                    "open the app, choose the project under Projects, then Edit.").encode()
            return self._send(body, "text/html; charset=utf-8")
        if path in ("/", "/index.html"):
            self.path = path = "/review.html"
        if not DEV and path.startswith(DEV_ONLY):
            return self._json({"error": "not found"}, 404)
        if self.path == "/api/init":
            return self._json(doc_reply({"n": N, "w": W, "h": H, "fps": FPS, "scale": A.scale, "frames": "/api/frames",
                                         "photo": bool(VMETA.get("image")),
                                         "notices": NOTICES, "dev": DEV,
                                         "depth": occlusion() is not None}))
        if self.path == "/api/edits":  # the current edits and what they resolve to (a page catching up)
            return self._json(doc_reply())
        if self.path == "/api/rev":  # cheap check whether a page's copy is current
            with DOC_LOCK:
                return self._json({"rev": DOC["rev"]})
        if self.path == "/api/render":
            return self._json(RENDER)
        if self.path == "/api/light":
            if LIGHT["json"] is None:
                return self._json({"ready": False, "progress": LIGHT["progress"], "error": LIGHT["error"]})
            return self._send(LIGHT["json"], "application/json")
        if self.path.startswith("/api/frame/"):
            u = urlsplit(self.path)
            t = max(0, min(N - 1, _int(u.path.rsplit("/", 1)[1])))
            data = preview_frame(t, {k: v[0] for k, v in parse_qs(u.query).items()})
            if data is None:
                return self._json({"error": "no such frame"}, 404)
            return self._send(data, "image/jpeg", "no-store")
        if path == "/api/portrait":  # Portrait style: the photo as drawn (thumbnail, stand-in)
            sp = T.emoji_sprite(portrait.key_for(portrait.project_photo(WORK, load_edits())), 120)
            return self._send(cv2.imencode(".png", cv2.cvtColor(sp, cv2.COLOR_RGBA2BGRA))[1].tobytes(), "image/png", "no-store")
        if self.path.startswith("/api/sprite/"):  # the export's own emoji picture, so the editor's quick preview matches it
            ch = unquote(path.rsplit("/", 1)[1])
            if not 0 < len(ch) <= 16 or ch.startswith(portrait.PREFIX):  # (a portrait key names a file: not from a URL)
                return self._json({"error": "no such emoji"}, 404)
            sp = PREVIEW["sprites"].get(ch)
            if sp is None:
                sp = PREVIEW["sprites"][ch] = T.emoji_sprite(ch)
            return self._send(cv2.imencode(".png", cv2.cvtColor(sp, cv2.COLOR_RGBA2BGRA))[1].tobytes(), "image/png", "max-age=86400")
        if self.path.startswith("/api/frames/"):
            name = os.path.basename(path)
            fp = os.path.join(FRAMES_DIR, name)
            if not re.fullmatch(r"\d{5}\.jpg", name) or not os.path.exists(fp):
                return self._json({"error": "no such frame"}, 404)
            with open(fp, "rb") as f:
                return self._send(f.read(), "image/jpeg")
        if self.path == "/api/formats":
            from mojimask import formats
            sizes = {p["id"]: {f: formats.output_size(p["id"], W, H, f, photo=bool(VMETA.get("image"))) for f in formats.FITS}
                     for p in formats.PRESETS}
            return self._json({**formats.describe(), "sizes": sizes, "fps": FPS, "frames": N, "photo": bool(VMETA.get("image")),
                               "seconds": N / FPS, "name": os.path.splitext(os.path.basename(A.original or A.video))[0],
                               "video": os.path.abspath(A.original or A.video),
                               # where an export is suggested: next to the original, not inside the app's own copy
                               "save_dir": "" if os.path.abspath(A.original or A.video).startswith(os.path.abspath(WORK) + os.sep)
                               else os.path.dirname(os.path.abspath(A.original or A.video))})
        if self.path.startswith("/api/pose/"):
            return self._json(pose_frame(max(0, min(N - 1, _int(path.rsplit("/", 1)[1])))))
        if self.path.startswith("/api/depth/"):  # the editor's depth view (depth_ui.js, MOJIMASK_DEPTH on)
            name = path.rsplit("/", 1)[1]
            occ = occlusion()
            if occ is None:
                return self._json({"error": "no depth"}, 404)
            t = max(0, min(N - 1, _int(name.split(".")[0])))
            if name.endswith(".jpg"):
                return self._send(cv2.imencode(".jpg", occ.picture(t), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes(), "image/jpeg")
            return self._json(depth_frame(t))
        if self.path.startswith("/api/boxes?"):  # Decisions' looping pictures: these boxes from their first detection to
            # their last (the cover's frames in between: gaps filled in). Not the frames a cover is held past either end:
            # those are the box carried on without a detection, and played back they slide off the person when the camera pans.
            ids = parse_qs(urlsplit(self.path).query).get("ids", [""])[0].split(",")
            return self._json({k: {t: [round(v, 1) for v in T.from_cs(c)] for t, c in AUTO[k]["cover"].items()
                                   if AUTO[k]["obs"] and AUTO[k]["obs"][0] <= t <= AUTO[k]["obs"][-1]}
                               for k in ids if k in AUTO})
        if self.path == "/api/decisions":
            files = _decision_files()
            if not files:
                return self._json({"decisions": [], "missing": True})
            ds = [d for f in files for d in _load_json(f)["decisions"]] + your_box_questions_now()
            return self._json({"decisions": FX.answers(ds, FACTS["facts"], HIDDEN)})
        if self.path == "/api/export_labels":
            return self._json(export_labels(load_edits()))
        if path == "/api/labels_export":  # the file Export labels wrote (a link on the editor's page)
            fp = f"{WORK}/labels_export.json"
            if not os.path.exists(fp):
                return self._json({"error": "no labels exported yet"}, 404)
            with open(fp, "rb") as f:
                return self._send(f.read(), "application/json", "no-store",
                                  [("Content-Disposition", 'attachment; filename="labels_export.json"')])
        if http_guard.static_ok(path):  # the pages and what they load; never projects or code
            return super().do_GET()
        return self._json({"error": "not found"}, 404)

    def _post(self):
        body = self._body()
        if self.path == "/api/ops":  # a page's changes (see "the edits document")
            try:
                ops = body.get("ops") or []
                # who's who comes from the facts: pages say it with /api/facts
                stray = [o for o in ops if str((o.get("path") or [""])[0]) in DERIVED]
                if stray:
                    print(f"ignored {len(stray)} edit(s) to fields the facts own: {[o['path'] for o in stray][:5]}")
                ops = [o for o in ops if o not in stray]
                apply_ops(ops, body.get("label", ""), undoable=body.get("undoable", True))  # bookkeeping (edits.result) isn't ⌘Z-able
                if any(str((o.get("path") or [""])[0]) == "palette" for o in ops):
                    sync_facts()  # another emoji preset: its own picks take the place of the last one's
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                return self._json({"error": str(e)}, 400)
            return self._json(doc_reply())
        if self.path == "/api/facts":  # {facts: [{kind, pieces, value}], label, page}: what a page says about boxes
            try:
                change_facts(body.get("facts") or [], label=body.get("label", ""), page=body.get("page", ""))
            except (ValueError, KeyError, TypeError) as e:
                return self._json({"error": str(e)}, 400)
            return self._json(doc_reply())
        if self.path == "/api/decisions":  # {id, answer[, via]}: answer (or re-answer, or clear) a decision -- as facts
            via = body.get("via")  # "agent": answered by a program for you (decisions_cli.py)
            if via not in (None, "agent"):
                return self._json({"error": "via must be \"agent\""}, 400)
            for fp in [*_decision_files(), None]:
                src = _load_json(fp)["decisions"] if fp else your_box_questions_now()
                d = next((x for x in src if x["id"] == body.get("id")), None)
                if d is None:
                    continue
                ans = body.get("answer")
                if ans is not None and ans not in d["options"]:
                    return self._json({"error": "not an option"}, 400)
                old = [f["id"] for f in FACTS["facts"] if f.get("decision") == d["id"]]
                change_facts(FX.facts_for_answer(d, ans) if ans is not None else [], old,
                             label=f"Answer: {ans}" if ans else "Clear answer", page="decisions", decision=d["id"], via=via)
                # with the new edits and what they resolve to, so the page needn't fetch them again
                return self._json({**doc_reply(), **FX.answers([d], FACTS["facts"], HIDDEN)[0]})
            return self._json({"error": "no such decision"}, 404)
        if self.path == "/api/undo":
            return self._json(doc_reply({"undone": undo_last()}))
        if self.path == "/api/redo":
            return self._json(doc_reply({"redone": undo_last(redo=True)}))
        if self.path == "/api/portrait":  # Portrait style: a chosen photo -> the project's cut-out (the page then
            try:                          # sets edits.portrait to the returned file name, like any other edit)
                data = base64.b64decode(str(body.get("data", "")).split(",", 1)[-1])
                return self._json({"file": portrait.prepare(data, WORK)})
            except ValueError as e:
                return self._json({"error": str(e)}, 400)
        if self.path == "/api/undo_correction":  # take back a placement of a cover and re-follow it
            try:
                undo_correction(str(body["piece"]), int(body["t"]))
            except (ValueError, KeyError, TypeError) as e:
                return self._json({"error": str(e)}, 400)
            return self._json(doc_reply())
        if self.path == "/api/split_cover":  # split a cover (edits.manual) in two at a frame
            try:
                new = split_cover(str(body["piece"]), int(body["t"]))
            except (ValueError, KeyError, TypeError) as e:
                return self._json({"error": str(e)}, 400)
            return self._json(doc_reply({"new": new}))
        if self.path == "/api/propagate":
            t = _int(body.get("t"))
            try:
                c = [float(v) for v in body["c"]]
            except (KeyError, TypeError, ValueError):
                raise BadRequest("c: [cx, cy, s]") from None
            if not 0 <= t < N or len(c) != 3 or c[2] <= 0:
                raise BadRequest("t: a frame, c: [cx, cy, s]")
            prop, ev = propagate(t, c)
            return self._json({"prop": prop, "ev": ev, "pv": PROP_VERSION})
        if self.path == "/api/choose_output":  # {name, dir}: this Mac's Save dialog (localhost; the app has its own)
            return self._json(choose_output(str(body.get("name") or "video.mp4"), str(body.get("dir") or "")))
        if self.path == "/api/thumb":  # {t, if_missing}: the project's picture on the Projects page: frame t with its
            # faces covered as they are now, small, kept with the project (thumb.jpg)
            out = os.path.join(WORK, "thumb.jpg")
            if body.get("if_missing") and os.path.exists(out):
                return self._json({"ok": True})
            t = max(0, min(N - 1, _int(body.get("t") or 0)))
            data = preview_frame(t, {"style": load_edits().get("cover_style") or "emoji"})
            if data is None:
                return self._json({"error": "no such frame"}, 404)
            im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            k = 400 / max(im.shape[:2])
            if k < 1:
                im = cv2.resize(im, (round(im.shape[1] * k), round(im.shape[0] * k)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(out + ".tmp.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
            os.replace(out + ".tmp.jpg", out)
            return self._json({"ok": True})
        if self.path == "/api/reveal":  # the last export, selected in Finder
            return self._json({"ok": reveal_export()})
        if self.path == "/api/render":
            with RENDER_LOCK:  # checked and marked together, so two clicks can't start two exports
                if RENDER["status"] != "rendering":
                    RENDER.update(status="rendering", progress=0.0, file=None, path=None, error=None)
                    threading.Thread(target=do_render, args=(load_edits(), body), daemon=True).start()
            return self._json(RENDER)
        self._json({"error": "not found"}, 404)


if __name__ == "__main__":
    os.chdir(HERE)
    # Warm up before answering: the first state() resolves the tracks and fills the links cache (it can take a while
    # on a long clip). Doing it here keeps it inside the app's "Linking people over time" step instead of a silent
    # wait after the scan says done.
    t0 = time.time()
    state(load_edits())
    print(f"warm-up {time.time() - t0:.0f}s")
    threading.Thread(target=refollow_covers, daemon=True).start()
    print(f"review tool: http://localhost:{A.port}")
    ThreadingHTTPServer(("127.0.0.1", A.port), Handler).serve_forever()
