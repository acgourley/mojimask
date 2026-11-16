"""The scan's face measurements: one per detection, with how comparable it is. The graph (graph.py) and the
same-person questions (questions.py) take their face evidence from these, and the "clear face" test (graph.CLEAR_FACE)
is their score.

For every detection of every tracklet: RetinaFace (reid.landmarks) on the crop around it (reid.face_crop) finds the
most confident face within NEAR (0.35) head sizes of the box centre, its 5 landmarks, and SFace (reid.embed) on the
aligned face. Comparability comes from the landmarks:
  yaw    (nose x - midpoint of the eyes x) / eye distance: ~0 facing the camera, larger as the head turns
  eyes   eye distance / face box width: shrinks as the head turns to profile
  px     face box height in pixels
A face is "frontal" (comparable) when the landmark detector is confident, it is facing the camera and big enough. Face
evidence is only taken between frontal faces: a profile against a frontal face scored as low as two
strangers on twirl.

    measure(proj, tracklets) -> {(tracklet id, t): {"score", "yaw", "eyes", "px", "emb"}}  (cached per project)
"""
import os
import pickle

import cv2
import numpy as np

from mojimask import reid  # noqa: E402

NEAR = 0.35
CACHE = "faces_retina_sface"  # the cache is per model pair: measurements by other models (proto/faces.pkl: SCRFD +
# ArcFace, 512-d) never mix with these (128-d); a project scanned before the switch measures its faces again
CACHE_VERSION = 2  # 2: each entry keeps the box it measured
FRONTAL = dict(score=0.6, yaw=0.25, px=24)


def nearest(boxes, k, x0, y0, c):
    """Index of the most confident face box (crop coordinates, from reid.face_crop) whose centre is within NEAR head sizes
    of box c's centre, or None."""
    best = None
    for i, b in enumerate(boxes):
        d = np.hypot((b[0] + b[2]) / 2 / k + x0 - c[0], (b[1] + b[3]) / 2 / k + y0 - c[1])
        if d <= NEAR * c[2] and (best is None or b[4] > boxes[best][4]):
            best = i
    return best


def face(img, c):
    """The face measurement for detection c (cx, cy, s) in img: the most confident face within NEAR head sizes of its
    centre, or None."""
    cr = reid.face_crop(img, c)
    if cr is None:
        return None
    crop, k, x0, y0 = cr
    boxes, kps = reid.landmarks(crop)
    i = nearest(boxes, k, x0, y0, c)
    if i is None:
        return None
    b, kp = boxes[i], kps[i]
    eye_l, eye_r, nose = kp[0], kp[1], kp[2]
    ed = float(np.hypot(*(eye_r - eye_l))) + 1e-6
    yaw = float((nose[0] - (eye_l[0] + eye_r[0]) / 2) / ed)
    out = {"score": float(b[4]), "yaw": yaw, "eyes": ed / max(1.0, b[2] - b[0]), "px": float((b[3] - b[1]) / k), "emb": None}
    if frontal(out):  # the embedding is only ever compared for frontal faces (graph, questions): half the time saved
        e = reid.embed(reid.align(crop, kp)).flatten()
        out["emb"] = (e / np.linalg.norm(e)).astype(np.float32)
    return out


def frontal(f):
    return f is not None and f["score"] >= FRONTAL["score"] and abs(f["yaw"]) <= FRONTAL["yaw"] and f["px"] >= FRONTAL["px"]


def measure(proj, tl):
    """Cached per project, each entry with the box it measured: an entry whose tracklet box has changed (another
    detection spec or tracklet rule numbers them differently) is measured again."""
    from mojimask.scan import store
    cache = store.path(proj, CACHE, "pkl")
    doc = pickle.load(open(cache, "rb")) if os.path.exists(cache) else {}  # only ever written by this function
    if doc.get("version") != CACHE_VERSION:
        doc = {}  # an older cache, without the boxes: can't be checked, so measured again
    key = lambda c: tuple(round(float(v), 1) for v in c[:3])  # noqa: E731
    have = doc.get("faces", {})
    out = {(k, t): have[(k, t)][1] for k, v in tl.items() for t, c in v["obs"].items()
           if (k, t) in have and have[(k, t)][0] == key(c)}
    todo = [(k, t, c) for k, v in tl.items() for t, c in v["obs"].items() if (k, t) not in out]
    by_frame = {}
    for k, t, c in todo:
        by_frame.setdefault(t, []).append((k, c))

    def frame(t):  # one frame's detections; frames are independent, so they run in threads (ONNX Runtime releases
        img = cv2.imread(f"{proj}/frames/{t:05d}.jpg")  # Python's lock): same results, ~6x faster on 8 cores
        return [((k, t), face(img, c)) for k, c in by_frame[t]]
    from mojimask.scan import viz
    viz.emit("faces_start", frames=len(by_frame), detections=len(todo))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        for res in ex.map(frame, sorted(by_frame)):
            out.update(res)
            viz.emit("faces", t=int(res[0][0][1]), d=[[int(k), 2 if frontal(f) else 1 if f else 0] for (k, _), f in res])
    if todo or len(out) != len(have):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        boxes = {(k, t): key(c) for k, v in tl.items() for t, c in v["obs"].items()}
        tmp = cache + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"version": CACHE_VERSION, "faces": {kt: (boxes[kt], m) for kt, m in out.items()}}, f)
        os.replace(tmp, cache)
    return out
