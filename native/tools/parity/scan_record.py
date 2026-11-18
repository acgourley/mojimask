"""Record a full Python decide run (mojimask/scan/newalgo.run, as pipeline.py runs it) on a copy of jazz, for mm-scan's
parity tests. Python itself is not edited: the recorder monkeypatches the boundaries mm-scan puts behind traits.

    .venv/bin/python native/tools/parity/scan_record.py [--clip jazz] [--depth anything-small] [--name NAME]

Writes out/native/scan/<name>/:
  input/      the project as it was before the decide step (Rust runs on a copy of it)
  py/         the project after Python's decide step (every file the step writes)
  rec/        the recording:
    calls_<fn>.json   every call across a boundary, {key: result}:
                        flow     track._flow_measure(g0, g1, c, core)   key [t0, t1, cx, cy, s]
                        motion   follow.face_motion(g0, g1, c)          key [t0, t1, cx, cy, s]
                        trackable  Clip.trackable(a, b)                 key [a, b]
                        looks    follow.ncc(patch(g_t, c), patch(g_u, d))  key [t, cx, cy, s, u, dx, dy, ds]
                        face_near  follow.face_near(img_t, c)           key [t, cx, cy, s]
                        face     faces.face(img_t, c)                   key [t, cx, cy, s]
                        body     appearance.reid_embed rows per crop     key [t, x1, y1, x2, y2] (appearance.crop's box)
                        depth    Depth.near(t, c, f)                    key [t, cx, cy, s, f]
    stage_*.json      what each stage produced (tracklets, chains, faces, graph, attach, members, build, tracks)
    records_<n>_<file>  every decisions_*.json as each rule wrote it, in order
    events.json       the viz events (without "ms")
Keys are the arguments' float64 values (repr, so they round-trip exactly); Rust's replay looks them up bit for bit
and fails loudly on a call that wasn't recorded.
"""
import argparse
import json
import os
import re
import shutil
import sys
import threading
import weakref

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--clip", default="jazz")
ap.add_argument("--depth", default="off")
ap.add_argument("--name")
ap.add_argument("--project", help="record on this project instead (a scanned project under out/native/)")
a = ap.parse_args()
os.environ["MOJIMASK_DEPTH"] = a.depth
name = a.name or (a.clip + ("" if a.depth == "off" else "_depth"))
BASE = os.path.join(ROOT, "out", "native", "scan", name)
SRC = os.path.join(ROOT, "out", "poc", "projects", a.clip)
assert a.clip in ("jazz", "jazz_4k"), "privacy: jazz and its copies only"
if a.project:
    SRC = os.path.abspath(a.project)
    assert SRC.startswith(os.path.join(ROOT, "out", "native") + os.sep), "privacy: only projects made from jazz, under out/native/"

if os.path.exists(BASE):
    shutil.rmtree(BASE)
INPUT, PY, REC = (os.path.join(BASE, d) for d in ("input", "py", "rec"))


def copy_input(dst):
    """The project without anything the decide step makes (and without stale caches)."""
    skip = {"facts.json", "edits.json", "proto_tracks.json", ".server.lock", "review_server.log"}
    os.makedirs(dst)
    for f in os.listdir(SRC):
        p = os.path.join(SRC, f)
        if f in skip or f.startswith("decisions_") or f.startswith("film_est") or f.startswith("light_est"):
            continue
        if f == "proto":
            continue  # chains, decide_viz and the face cache are the step's own
        if os.path.isdir(p):
            shutil.copytree(p, os.path.join(dst, f))
        else:
            shutil.copy2(p, os.path.join(dst, f))


copy_input(INPUT)
shutil.copytree(INPUT, PY)
os.makedirs(REC)

import cv2  # noqa: E402
from mojimask import track as T  # noqa: E402
from mojimask import appearance as AP  # noqa: E402
from mojimask import depth as DP  # noqa: E402
from mojimask.scan import follow as FO  # noqa: E402
from mojimask.scan import faces as FA  # noqa: E402
from mojimask.scan import tracklets as TL  # noqa: E402
from mojimask.scan import join as J  # noqa: E402
from mojimask.scan import graph as G  # noqa: E402
from mojimask.scan import attach as AT  # noqa: E402
from mojimask.scan import overlaps as OV  # noqa: E402
from mojimask.scan import cover as CV  # noqa: E402
from mojimask.scan import newalgo as NA  # noqa: E402
from mojimask.scan import same as SM  # noqa: E402

LOCK = threading.Lock()
CALLS = {k: {} for k in ("flow", "motion", "trackable", "looks", "face_near", "face", "body", "depth")}
CONFLICTS = []


def fl(x):
    return [float(v) for v in np.asarray(x, float).ravel()]


def key(*parts):
    return json.dumps([float(v) for p in parts for v in (np.asarray(p, float).ravel() if not np.isscalar(p) else [p])])


def put(fn, k, v):
    with LOCK:
        d = CALLS[fn]
        if k in d and json.dumps(d[k]) != json.dumps(v):
            CONFLICTS.append((fn, k))
        d.setdefault(k, v)


# ---- tags: which frame an image (or a 32x32 patch) came from, by object id, while it lives
TAGS = {}


def tag(arr, v):
    with LOCK:
        TAGS[id(arr)] = v
    weakref.finalize(arr, lambda i=id(arr): TAGS.pop(i, None))
    return arr


def tag_of(arr):
    return TAGS[id(arr)]


_imread = cv2.imread
FRAME = re.compile(r"/frames/(\d{5})\.jpg$")


def imread(path, *args):
    im = _imread(path, *args)
    m = FRAME.search(str(path))
    if im is not None and m:
        tag(im, int(m.group(1)))
    return im


cv2.imread = imread

_gray = FO.Clip.gray


def gray(self, t):
    g = _gray(self, t)
    with LOCK:
        known = id(g) in TAGS
    if not known:
        tag(g, int(t))
    return g


FO.Clip.gray = gray

_flow = T._flow_measure


def flow_measure(g0, g1, c, core):
    assert tuple(core) == FO.CORE, core
    m = _flow(g0, g1, c, core)
    put("flow", key(tag_of(g0), tag_of(g1), c), None if m is None else [float(m[0][0]), float(m[0][1]), int(m[1]), float(m[2])])
    return m


T._flow_measure = flow_measure

_motion = FO.face_motion


def face_motion(g0, g1, c):
    M, kind = _motion(g0, g1, c)
    put("motion", key(tag_of(g0), tag_of(g1), c), [None if M is None else fl(M), kind])
    return M, kind


FO.face_motion = face_motion

_trk = FO.Clip.trackable


def trackable(self, a_, b_):
    v = _trk(self, a_, b_)
    put("trackable", key(a_, b_), float(v))
    return v


FO.Clip.trackable = trackable

_patch, _ncc = FO.patch, FO.ncc


def patch(g, c):
    p = _patch(g, c)
    return tag(p, (tag_of(g), fl(c)))


def ncc(x, y):
    v = _ncc(x, y)
    (t0, c0), (t1, c1) = tag_of(x), tag_of(y)
    put("looks", key(t0, c0, t1, c1), float(v))
    return v


FO.patch, FO.ncc = patch, ncc

_face_near = FO.face_near


def face_near(img, c):
    s, head = _face_near(img, c)
    put("face_near", key(tag_of(img), c), [float(s), None if head is None else fl(head)])
    return s, head


FO.face_near = face_near

_face = FA.face


def face(img, c):
    f = _face(img, c)
    put("face", key(tag_of(img), c), None if f is None else
        {"score": float(f["score"]), "yaw": float(f["yaw"]), "eyes": float(f["eyes"]), "px": float(f["px"]),
         "emb": None if f["emb"] is None else [float(v) for v in f["emb"]]})
    return f


FA.face = face

_crop, _embed = AP.crop, AP.reid_embed


def crop(frame, box, pad=0.0):
    c = _crop(frame, box, pad)
    if c is not None:
        x1, y1, x2, y2 = box
        tag(c, [tag_of(frame), float(x1), float(y1), float(x2), float(y2)])
    return c


def reid_embed(crops):
    v = _embed(crops)
    for c, row in zip(crops, v):
        put("body", key(*tag_of(c)), [float(x) for x in row])
    return v


AP.crop, AP.reid_embed = crop, reid_embed

_near = DP.Depth.near


def near(self, t, c, f=0.25):
    v = _near(self, t, c, f)
    put("depth", key(t, c, f), float(v))
    return v


DP.Depth.near = near


# ---- what each stage produced
def save(name_, obj):
    with open(os.path.join(REC, f"stage_{name_}.json"), "w") as fh:
        json.dump(obj, fh)


def tl_json(tl):
    return {str(k): {"obs": {str(t): fl(c) for t, c in v["obs"].items()}, "score": {str(t): float(s) for t, s in v["score"].items()},
                     "v": fl(v["v"]), "why": v["why"]} for k, v in tl.items()}


_build = TL.build
_tl_saved = []


def build(dets, mats, r=TL.R):
    tl = _build(dets, mats, r)
    if not _tl_saved:
        _tl_saved.append(1)
        save("dets", [[[fl(c), float(s)] for c, s in D] for D in dets])
        save("tracklets", tl_json(tl))
    return tl


TL.build = build

_jrun = J.run


def jrun(proj, limit=J.LIMIT):
    tl, chains, info = _jrun(proj, limit)
    save("chains", {"chains": {str(c): [int(k) for k in seq] for c, seq in chains.items()}, "info": {k: (v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()}) for k, v in info.items()}})
    return tl, chains, info


J.run = jrun

_measure = FA.measure


def measure(proj, tl):
    F = _measure(proj, tl)
    save("faces", [[int(k), int(t), None if f is None else {"score": float(f["score"]), "yaw": float(f["yaw"]), "eyes": float(f["eyes"]),
                                                            "px": float(f["px"]), "emb": None if f["emb"] is None else [float(v) for v in f["emb"]]}]
                   for (k, t), f in F.items()])
    return F


FA.measure = measure


def members_json(members):
    return [[int(r), [int(c) for c in ms]] for r, ms in members.items()]  # in dict and set iteration order


_grun = G.run


def grun(proj, name_truth=None, state=False):
    res, st = _grun(proj, name_truth, state=True)
    info = st["info"]
    save("graph", {"res": res, "members": members_json(st["members"]),
                   "info": {str(c): {"a": i["a"], "b": i["b"], "size": i["size"], "ctr": fl(i["ctr"]), "end": fl(i["end"]),
                                     "start": fl(i["start"]), "still": bool(i["still"]), "spread": i["spread"],
                                     "face": None if i["face"] is None else fl(i["face"]), "nfaces": i["nfaces"]}
                            for c, i in info.items()}})
    return (res, st) if state else res


G.run = grun

_attach = AT.attach


def attach(proj, st, gd):
    out = _attach(proj, st, gd)
    save("attach", {"att": [[int(f), int(r)] for f, r in out.items()],
                    "gd": {str(r): {"face": None if g["face"] is None else fl(g["face"]), "body": None if g["body"] is None else fl(g["body"]),
                                    "ms": [int(c) for c in g["ms"]]} for r, g in gd.items()}})
    return out


AT.attach = attach

_saves = []
_save_records = OV.save_records


def save_records(path, decisions):
    _save_records(path, decisions)
    _saves.append(1)
    shutil.copy(path, os.path.join(REC, f"records_{len(_saves):02d}_{os.path.basename(path)}"))


OV.save_records = save_records

_cvbuild = CV.build


def cvbuild(proj, write=True):
    b = _cvbuild(proj, write)
    save("members", {"members": members_json(b["members"]), "members0": members_json(b["members0"])})
    save("build", {"covers": {str(r): {str(t): fl(c) for t, c in cv.items()} for r, cv in b["covers"].items()},
                   "behind": {str(r): v for r, v in b["behind"].items()},
                   "flow": {str(r): {str(t): m for t, m in f.items()} for r, f in b["flow"].items()},
                   "how": dict(b["how"]), "depth": b["depth"]})
    return b


CV.build = cvbuild

_settle = NA.settle


def settle(proj, tracks, hidden, overlap=None):
    newly = _settle(proj, tracks, hidden, overlap)
    save("settle", {"hidden_in": list(hidden), "newly": newly})
    return newly


NA.settle = settle

events = []
res = NA.run(PY, events=lambda ev: events.append({k: v for k, v in ev.items() if k != "ms"}))
with open(os.path.join(REC, "events.json"), "w") as fh:
    json.dump(events, fh)
for fn, d in CALLS.items():
    with open(os.path.join(REC, f"calls_{fn}.json"), "w") as fh:
        json.dump(d, fh)
summary = {"result": res, "calls": {k: len(v) for k, v in CALLS.items()}, "conflicts": CONFLICTS[:20], "n_conflicts": len(CONFLICTS),
           "depth": a.depth, "base": BASE}
with open(os.path.join(REC, "summary.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print(json.dumps(summary))
