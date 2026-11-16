"""Portrait covers: one headshot drawn over every head instead of an emoji (cover style "portrait").

The photo is a sprite like an emoji: its key is "portrait:<path to a PNG>" (key_for), and track.emoji_sprite
returns the photo for such keys, so the export, the server preview and the editor's /api/sprite all draw it
with no other changes. The photo is expected to have its background removed already. Each project keeps its
chosen photo in <workdir>/portrait/ (edits.portrait = file name, see prepare()); without one, the Mona Lisa's head
(web/assets/portrait_mona.png, cut from the start page's painting by tools/mona_portrait.py) is used. The photo is always
matched to each head (head_pose.py):
  tilt    the head's tilt minus the photo's own tilt (so a tilted selfie stands upright on an upright head)
  facing  a photo turned to one side is mirrored when the head faces the other way
  turn    the card-turn effect only for the turn the photo doesn't already show
The photo's own pose comes from RetinaFace's 5 face points (eyes, nose, mouth corners), measured once and
cached next to the image (<image>.pose.json).
"""
import hashlib
import io
import json
import math
import os
import threading

import numpy as np

from mojimask.paths import WEB as HERE
DEFAULT = os.path.join(HERE, "assets", "portrait_mona.png")  # until a photo is chosen (public domain, shipped)
STAND_IN = "👤"  # only if DEFAULT is missing
PORTRAIT_KEY = "🖼️"  # no photo chosen yet; also what the editor's quick stand-in asks /api/sprite for
PREFIX = "portrait:"
FRONT = 0.15  # |yaw| below this: the photo counts as front-facing and is never mirrored


def key_for(path=None):
    return PREFIX + path if path else PORTRAIT_KEY


def is_portrait(ch):
    return ch == PORTRAIT_KEY or ch.startswith(PREFIX)


def _path(ch):
    """The photo file for ch, or None (then the 👤 stand-in). With no photo chosen, DEFAULT.
    A "portrait:" key names a file only if it is a .png in a project's portrait/ folder (prepare() puts it there):
    keys can arrive from a URL (/api/sprite/), and the photo's pose is cached next to it."""
    if ch.startswith(PREFIX):
        p = os.path.realpath(ch[len(PREFIX):])
        ok = p.endswith(".png") and os.path.basename(os.path.dirname(p)) == "portrait"
        return p if ok else None
    return DEFAULT if os.path.isfile(DEFAULT) else None


def project_photo(workdir, edits):
    """The project's chosen photo (absolute path) or None for the default."""
    name = (edits or {}).get("portrait")
    if not name or os.path.basename(name) != name:
        return None
    p = os.path.join(workdir, "portrait", name)
    return p if os.path.isfile(p) else None


def sprite(ch, size=160):
    """The photo cropped to its cut-out, centred on a transparent square, size x size RGBA (RGB order)."""
    from PIL import Image
    if _path(ch) is None:
        from mojimask import track as T
        return T.emoji_sprite(STAND_IN, size)
    im = Image.open(_path(ch)).convert("RGBA")
    im = im.crop(im.getchannel("A").point(lambda a: 255 if a > 24 else 0).getbbox())
    s = max(im.size)
    sq = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    sq.paste(im, ((s - im.size[0]) // 2, (s - im.size[1]) // 2))
    return np.array(sq.resize((size, size), Image.LANCZOS))


_RF = None
_RF_LOCK = threading.Lock()


def _retinaface():
    """One CPU RetinaFace, made on first use (loading the model each time cost more than the face itself)."""
    global _RF
    with _RF_LOCK:
        if _RF is None:
            from mojimask import onnx_core as OC
            _RF = OC.RetinaFace(engine="cpu")
        return _RF


def _face(rgb):
    """(box [x1,y1,x2,y2], 5 face points) of the most confident face in an RGB image (RetinaFace), or None."""
    import cv2
    return _retinaface().best_face(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), min_score=0.5)


_POSES = {}  # (photo path, its mtime) -> (roll, yaw): own_pose is asked for every cover on every frame


def own_pose(ch):
    """(roll_deg, yaw) of the photo itself, same conventions as head_pose (yaw + = facing image right)."""
    src = _path(ch)
    if src is None:
        return 0.0, 0.0  # the stand-in faces front
    mkey = (src, os.path.getmtime(src))
    if mkey in _POSES:
        return _POSES[mkey]
    path = src + ".pose.json"
    if os.path.exists(path) and os.path.getmtime(path) >= mkey[1]:
        try:
            with open(path) as f:
                d = json.load(f)
            _POSES[mkey] = d["roll"], d["yaw"]
            return _POSES[mkey]
        except (OSError, ValueError, KeyError):
            pass  # unreadable: measure again
    a = sprite(ch, 480)
    al = a[..., 3:4] / 255.0
    rgb = (a[..., :3] * al + 128 * (1 - al)).astype(np.uint8)  # on grey, like a photo
    r = _face(rgb)
    roll = yaw = 0.0
    if r is not None:
        p = r[1]
        le, re, nose = p[0], p[1], p[2]  # image-left eye, image-right eye, nose
        v = re - le
        d = float(np.linalg.norm(v))
        roll = math.degrees(math.atan2(v[1], v[0]))
        yaw = float(np.clip(np.dot(nose - (le + re) / 2, v / d) / d / 0.45, -1, 1))
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"roll": roll, "yaw": yaw, "found": r is not None}, f)
        os.replace(tmp, path)
    except OSError:
        pass  # not cached on disk: measured again next run
    _POSES[mkey] = roll, yaw
    return roll, yaw


def match(ch, roll, yaw, side):
    """Head pose -> (roll, yaw, mirror) to draw photo ch with (see module doc)."""
    r0, y0 = own_pose(ch)
    facing = 0 if abs(y0) < FRONT else (1 if y0 > 0 else -1)
    mirror = bool(facing and side and side != facing)
    shown = -y0 if mirror else y0  # how turned the photo already looks as drawn
    return roll - (-r0 if mirror else r0), float(np.clip(yaw - shown, -1, 1)), mirror


def prepare(data, workdir):
    """An uploaded photo (any image file's bytes) -> the project's copy as a PNG; returns its file name.
    The photo is expected to be a cut-out already (background removed, transparent around the person): it is
    used as it is, cropped to its picture when drawn. Raises ValueError if it isn't a picture or has no face."""
    from PIL import Image, ImageOps
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGBA")
    except Exception:
        raise ValueError("that file isn't a picture this app can read")
    im.thumbnail((1600, 1600), Image.LANCZOS)
    a = np.array(im)
    al = a[..., 3:4] / 255.0
    if _face((a[..., :3] * al + 128 * (1 - al)).astype(np.uint8)) is None:
        raise ValueError("no face found in that photo: try a clearer, front-on headshot")
    out = io.BytesIO()
    im.save(out, "PNG")
    png = out.getvalue()
    name = hashlib.md5(png).hexdigest()[:12] + ".png"
    os.makedirs(os.path.join(workdir, "portrait"), exist_ok=True)
    with open(os.path.join(workdir, "portrait", name), "wb") as f:
        f.write(png)
    own_pose(key_for(os.path.join(workdir, "portrait", name)))  # measure now, so the first frame isn't slow
    return name
