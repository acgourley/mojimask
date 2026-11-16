"""Which pixel space a project's saved files are in, checked when it opens.

Every position a project saves -- detections (dets/*.json), camera motion (mats_v2.npy), the scan's tracks
(proto_tracks.json) and the answers that hold positions (edits.json: covers drawn by hand) -- is in the pixels of the
frames the scan decoded. That space is recorded in meta.json["space"]:

    {"w": 643, "h": 476, "rot": 0, "v": 2}

  w, h  frame size in pixels, upright (after rotation)
  rot   the video's rotation (frames are stored upright)
  v     how frames were decoded: 2 = at the display size, square pixels (video_io._frames_avf)

If the decoder now gives another space, the project can't be used as it is: scan the video again. (Converting
older projects' positions was removed with the old pipeline, 2026-11-12: it predated proto_tracks.json and facts.)
"""
import json
import os


def saved_space(work, meta):
    """The space a project's files are in: recorded, else read off its first frame image (decoder v1), else
    None (nothing scanned yet)."""
    if meta.get("space"):
        return meta["space"]
    jpg = os.path.join(work, "frames", "00000.jpg")
    if not os.path.exists(jpg):
        return None
    import cv2
    img = cv2.imread(jpg)
    if img is None:  # unreadable: no space to check against, as if nothing were scanned
        return None
    h, w = img.shape[:2]
    return {"w": w, "h": h, "rot": meta.get("rotation"), "v": 1}  # rotation None: not recorded (frames were upright)


def _rot(sp):
    return sp.get("rot") or 0


def _same_rot(a, b):
    """Older projects didn't record the rotation (None); their frames were stored upright like now, so an
    unknown rotation matches any."""
    return a.get("rot") is None or b.get("rot") is None or _rot(a) == _rot(b)


def same(a, b):
    return a is not None and b is not None and (a["w"], a["h"]) == (b["w"], b["h"]) and _same_rot(a, b)


def ensure(work, video_meta, log=print):
    """Called when a project opens: are its saved positions in the space the decoder gives now? Returns a note
    for the reviewer when they aren't, else None."""
    mp = os.path.join(work, "meta.json")
    meta = json.load(open(mp)) if os.path.exists(mp) else {}
    from mojimask import video_io
    now = video_io.space(video_meta)
    src = saved_space(work, meta)
    if src is None:
        return None
    if same(src, now):
        if meta.get("space") != now:  # recorded for the first time (or the decoder version moved on, same pixels)
            meta["space"] = now
            _write(mp, meta)
        return None
    return (f"This project was scanned from frames of another shape ({src['w']}x{src['h']}, {_rot(src)} degrees; "
            f"the video now gives {now['w']}x{now['h']}, {now['rot']} degrees): import the video again for a new scan.")


def _write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)
