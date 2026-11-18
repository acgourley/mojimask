"""End-to-end scan for one video (the attendee app's "Scan" step).

    python pipeline.py VIDEO PROJECT_DIR        # CLI; the app server calls run() in a thread

Steps (each reports progress):
  prepare   read size/fps/rotation, extract upright frames (review tool + previews use them)
  detect    one pass over the frames running every model in the profile; writes dets/<name>.json and a
            live preview (frame + a box on each face) for the scan screen
  motion    camera motion between frames
  decide    the new algorithm (scan/newalgo.py): its tracks, people and decision records
  done      the review server loads them (proto_tracks.json)

Profile "core": YOLOv5-Face (tiled) + RetinaFace-R50 + pose (full keypoints -> head boxes) -- the
strongest small detector set of those compared; tracker filters tuned for it in
CORE_TRACKER.
Detection runs on ONNX Runtime via onnx_core.Core (CoreML on Macs); no PyTorch needed for a scan.
"""
import json
import os
import sys
import time

import cv2
import numpy as np

from mojimask import pose_util as PU
from mojimask import video_io
from mojimask.coords import _write as write_json  # tmp file + rename: the app reads meta.json while a scan writes it

CORE = ["yolov5m_face_tiled", "retinaface_r50_2x", "pose_full"]
CORE_DETS = ["yolov5m_face_tiled:0.3", "retinaface_r50_2x:0.3", "pose_head:0.35"]
CORE_TRACKER = {"min_peak": 0.6, "min_score": 0.45}
PREVIEW_MIN_SCORE = 0.3  # boxes drawn on the scan screen's picture
PREVIEW_COLOR = (63, 210, 255)  # BGR: the app's yellow
HIT_SCORE = 0.5  # a face detection counted in the scan screen's "faces found"
PERSON_SCORE = 0.3  # a skeleton counted in its "people now"
KEEP_META = ("copied", "space", "label")  # meta.json keys a rescan keeps: the app's own notes and the frames' space


def draw_preview(frame, found):
    """The scan screen's picture: the frame with a box on each face found (the app's yellow). No emoji: who is who
    isn't known until the scan's last steps."""
    f = frame.copy()
    w = max(1, round(max(f.shape[:2]) / 640))
    drawn = []  # one box a head: the face detector's and the skeleton's often land on the same one
    for name, boxes in found.items():
        if name == "pose_full":
            continue
        for x1, y1, x2, y2, s in boxes:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if s < PREVIEW_MIN_SCORE or any(a <= cx <= c and b <= cy <= d for a, b, c, d in drawn):
                continue
            drawn.append((x1, y1, x2, y2))
            cv2.rectangle(f, (int(x1), int(y1)), (int(x2), int(y2)), PREVIEW_COLOR, w, cv2.LINE_AA)
    return f


def run(video, proj, report=print, profile=CORE, events=None, engine_cache=None):
    """report(step, fraction, info dict); events(dict): what the decide step is doing (scan/viz.py). Writes proj/frames, proj/dets, proj/mats_v2.npy, proj/meta.json,
    proj/preview.jpg. engine_cache: where the timed engine plan is remembered (default: engine.json beside the
    projects folder)."""
    scan_start = time.time()
    os.makedirs(f"{proj}/frames", exist_ok=True)
    os.makedirs(f"{proj}/dets", exist_ok=True)
    old = json.load(open(f"{proj}/meta.json")) if os.path.exists(f"{proj}/meta.json") else {}
    have_frames = os.path.exists(f"{proj}/frames/00000.jpg")
    meta = {**video_io.probe(video, keep=old if have_frames else None),  # an existing project keeps its timeline
            **{k: old[k] for k in KEEP_META if k in old}}
    meta["video"] = os.path.abspath(video)
    if not have_frames:  # what every position this scan saves is measured in (coords.py)
        meta["space"] = video_io.space(meta)
    write_json(f"{proj}/meta.json", meta)
    report("prepare", 0.0, meta)
    if not os.path.exists(f"{proj}/frames/00000.jpg"):
        expect = max(1, round((meta.get("video_duration") or meta.get("duration") or 0) * (meta.get("fps") or 30)))
        video_io.extract_jpegs(video, f"{proj}/frames", meta,  # a 4K clip takes minutes here: say how far it's got
                               progress=lambda k: report("prepare", min(k / expect, 0.99), {"frame": k, "frames": expect}))
    n = len([x for x in os.listdir(f"{proj}/frames") if x.endswith(".jpg")])
    meta["frames"] = n
    write_json(f"{proj}/meta.json", meta)
    report("prepare", 1.0, meta)

    # ---- detect: one pass, all models (ONNX Runtime, no PyTorch). Frames come straight from the decoder
    # (lossless, upright), not from the JPEGs, which shift borderline detections.
    from mojimask import onnx_core
    report("detect", 0.0, {"note": "loading models"})
    sample = []
    for f in video_io.frames(video, meta):  # a few real frames to time the engine plans on
        sample.append(f)
        if len(sample) >= 5:
            break
    cache = engine_cache or os.path.join(os.path.dirname(os.path.abspath(proj)), os.pardir, "engine.json")
    core, engine = onnx_core.fastest_core(sample, cache)
    meta["engine"] = engine
    write_json(f"{proj}/meta.json", meta)
    report("detect", 0.0, {"engine": engine["plan"]})
    out = {name: [] for name in profile}
    H, W = meta["height"], meta["width"]
    dec = video_io.frames(video, meta)
    t0 = time.time()
    faces_seen = 0
    try:
        for t in range(n):
            frame = next(dec, None)
            if frame is None:  # decoder gave fewer frames than the JPEG pass: pad with empty results
                for name in profile:
                    out[name] += [[] for _ in range(n - t)]
                break
            found = core(frame)
            for name in profile:
                out[name].append(found[name])
            faces_seen += sum(1 for name in profile if name != "pose_full" for b in found[name] if b[4] >= HIT_SCORE)
            if t % 8 == 0 or t == n - 1:
                cv2.imwrite(f"{proj}/preview.tmp.jpg", draw_preview(frame, found), [cv2.IMWRITE_JPEG_QUALITY, 80])
                os.replace(f"{proj}/preview.tmp.jpg", f"{proj}/preview.jpg")
                el = time.time() - t0
                report("detect", (t + 1) / n, {"frame": t + 1, "frames": n, "fps": round((t + 1) / el, 1),
                                              "eta_s": round(el / (t + 1) * (n - t - 1)),
                                              "face_hits": faces_seen,
                                              "people_now": sum(1 for p in found.get("pose_full", [])
                                                                if p[4] >= PERSON_SCORE)})
    finally:
        dec.close()
    for name, frames in out.items():
        json.dump({"w": W, "h": H, "fps": meta["fps"], "space": meta.get("space"), "frames": frames}, open(f"{proj}/dets/{name}.json", "w"))
    if "pose_full" in out:  # derived head boxes for the tracker fusion
        heads = [[h for h in (PU.head_box(p) for p in fr) if h] for fr in out["pose_full"]]
        json.dump({"w": W, "h": H, "fps": meta["fps"], "space": meta.get("space"), "frames": heads}, open(f"{proj}/dets/pose_head.json", "w"))

    # ---- camera motion
    report("motion", 0.0, {})
    from mojimask import track as T
    mats = T.camera_motion(video, meta=meta)
    np.save(f"{proj}/mats_v2.npy", np.array(mats))
    report("motion", 1.0, {})
    # the new algorithm: tracks, people and its decision records (scan/newalgo.py)
    report("decide", 0.0, {})
    from mojimask.scan import newalgo
    newalgo.run(proj, lambda f, info: report("decide", f, info), events=events)
    report("done", 1.0, {"seconds": round(time.time() - scan_start)})
    return meta


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], report=lambda s, f, i: print(f"{s:8s} {f:5.0%} {i}"))
