"""Head pose for covers (prototype): tilt (roll) and left/right turn (yaw) per covered head, so the emoji
can lean and turn with the face.

Source: the COCO-17 pose skeletons the scan already saves (pose_full.json) -- eyes, nose, ears. No new model.
  roll  angle of the eye line (subject's right eye -> left eye, i.e. image left -> right when facing camera);
        ears as fallback when the eyes aren't both seen.
  yaw   where the nose sits along the eye line: 0 = centred (facing camera), +-1 = at/after an eye (~profile).
        Sign is in IMAGE terms: + = face pointing towards image right. One-eye profiles use nose vs visible ear.
Per track piece the raw angles are smoothed over time (confidence-weighted Gaussian) with a weak pull to
neutral, so missing or noisy frames drift back to an upright, front-facing emoji instead of jittering.

posed_sprite() draws any RGBA sprite (emoji, or a cut-out photo) rotated by roll and turned by yaw (a
perspective "card" turn: narrower, far edge shorter). It never shrinks the cover below MIN_W of its width.
"""
import collections
import math
import threading
import weakref

import cv2
import numpy as np

from mojimask import pose_util as PU
from mojimask.pose_util import NOSE, LEYE, REYE, LEAR, REAR

POSE_ANGLE_VIS = 0.4  # keypoints trusted for angles: stricter than pose_util.VIS (0.3) on purpose
MAX_ROLL = 40.0  # degrees
SHIFT = 0.12     # sideways move towards the facing direction, in half-widths
MIN_W = 0.8      # turned card keeps at least this fraction of its width (privacy: coverage)


def raw_angles(p):
    """(roll_deg, yaw, roll_conf, yaw_conf) from one pose_full row; confs 0 when not measurable."""
    k = PU.kps(p)
    xy, c = k[:, :2], k[:, 2]
    roll = yaw = 0.0
    rc = yc = 0.0
    if c[LEYE] > POSE_ANGLE_VIS and c[REYE] > POSE_ANGLE_VIS:
        v = xy[LEYE] - xy[REYE]  # image left -> right when facing the camera
        d = np.linalg.norm(v)
        if d > 1.5 and v[0] > 0:
            roll = math.degrees(math.atan2(v[1], v[0]))
            rc = float(min(c[LEYE], c[REYE]))
            if c[NOSE] > POSE_ANGLE_VIS:
                u = v / d
                r = float(np.dot(xy[NOSE] - (xy[LEYE] + xy[REYE]) / 2, u)) / d
                yaw = float(np.clip(r / 0.45, -1, 1))
                yc = float(min(rc, c[NOSE]))
    elif c[LEAR] > POSE_ANGLE_VIS and c[REAR] > POSE_ANGLE_VIS and xy[LEAR, 0] > xy[REAR, 0]:
        v = xy[LEAR] - xy[REAR]
        roll = math.degrees(math.atan2(v[1], v[0]))
        rc = 0.5 * float(min(c[LEAR], c[REAR]))
    if yc == 0 and c[NOSE] > POSE_ANGLE_VIS:  # profile: nose vs the one visible ear
        ears = [e for e in (LEAR, REAR) if c[e] > POSE_ANGLE_VIS]
        if len(ears) == 1:
            dx = xy[NOSE, 0] - xy[ears[0], 0]
            if abs(dx) > 1:
                yaw, yc = float(np.sign(dx)), 0.6 * float(min(c[NOSE], c[ears[0]]))
    if abs(roll) > MAX_ROLL:
        rc = 0.0
    return roll, yaw, rc, yc


# Emoji drawn in profile (Apple Color Emoji), by the way the head points: -1 = image left, +1 = image right.
# Checked by eye against the rendered glyphs; front-facing ones (faces, 🦉 🦦 🐥 ...) and things with no
# head (food, planets) aren't listed and never flip.
FACING = {**{e: -1 for e in "🦄🐧🐳🐴🐎🦓🐕🐩🐈🐖🐄🐂🐑🐏🐐🐃🦬🦒🐘🦣🦏🦛🐪🐫🦙🐅🐆🦍🦮🐓🦃🦆🦢🦤🦅🐦🐤🦜🦩🐊🐢🐍🐉🦖🦕🐬🐟🐠🐡🦈🐋🦭🐁🐀🐿🦫🦔🐇🦘🐛🐜🐝"},
          **{e: -1 for e in ("🐈‍⬛", "🐕‍🦺", "🕊️", "🕊")},  # (more than one code point each: not for the strings above)
          **{e: +1 for e in "🦌🦨🦡🐌🦎"}}
SIDE_ON, SIDE_OFF = 0.35, 0.15  # flip hysteresis on the smoothed yaw: commit to a side past 0.35, hold it until
                                # the head has clearly come back past 0.15 the other way (no flicker near front)


def mirror_for(ch, side):
    """Draw emoji ch mirrored? side = the head's held facing (-1 left, +1 right, 0 undecided)."""
    f = FACING.get(ch)
    return bool(f and side and side != f)


def sprite_pose(ch, roll, yaw, side):
    """Head pose -> (roll, yaw, mirror) to draw cover ch with: emoji tilt/turn as the head does (profile emoji
    mirrored to face the same way); a portrait photo also corrects for its own pose (portrait.match)."""
    from mojimask import portrait
    if portrait.is_portrait(ch):
        return portrait.match(ch, roll, yaw, side)
    return roll, yaw, mirror_for(ch, side)


def track_angles(per_frame, poses, sigma=4.0, prior=0.15):
    """{piece: {t: (roll_deg, yaw, side)}} for every cover in per_frame (resolve() output), smoothed per piece.
    side = which way the head is held to face (-1 image left, +1 image right, 0 not yet clear), with
    hysteresis: decides whether a profile emoji (FACING) is mirrored."""
    samples, present = {}, {}
    for t, items in enumerate(per_frame):
        rows = poses[t] if poses and t < len(poses) else []
        for tid, c, kind, *rest in items:
            piece = rest[0] if rest else tid
            present.setdefault(piece, []).append(t)
            p = PU.pose_for_head(rows, c) if rows else None
            if p is not None:
                samples.setdefault(piece, []).append((t, *raw_angles(p)))
    out = {}
    R = int(3 * sigma)
    for piece, ts in present.items():
        s = np.array(samples.get(piece, []), float).reshape(-1, 5)
        res, side = {}, 0
        for t in ts:
            if len(s):
                m = np.abs(s[:, 0] - t) <= R
                g = np.exp(-0.5 * ((s[m, 0] - t) / sigma) ** 2)
                wr, wy = g * s[m, 3], g * s[m, 4]
                roll = float((wr * s[m, 1]).sum() / (wr.sum() + prior))
                yaw = float((wy * s[m, 2]).sum() / (wy.sum() + prior))
            else:
                roll = yaw = 0.0
            if yaw >= SIDE_ON:
                side = 1
            elif yaw <= -SIDE_ON:
                side = -1
            elif side == 1 and yaw < -SIDE_OFF or side == -1 and yaw > SIDE_OFF:
                side = 0  # clearly turned back the other way but not yet committed: back to the emoji's own way
            res[t] = (roll, yaw, side)
        out[piece] = res
    return out


CANVAS = 1.5  # posed sprite canvas, in sprite sizes (room for a 40 deg tilt of the turned card)
CACHE_BYTES = 256 << 20  # posed canvases kept (least recently used dropped first); a portrait's is ~1 MB
_CACHE = collections.OrderedDict()  # (id(sprite), roll, yaw, mirror) -> (weakref to sprite, out, bytes)
_CACHE_LOCK = threading.Lock()  # the editor's server draws frames on several threads
_cache_bytes = 0


def posed_sprite(sprite, roll=0.0, yaw=0.0, mirror=False):
    """The sprite tilted by roll (deg) and turned by yaw, drawn centred on a transparent square canvas
    CANVAS times its size -> (canvas RGBA, (dx, dy)): paste the canvas at the head centre moved by
    (dx, dy) head sizes, CANVAS times bigger. Quantised + cached (1 deg, 0.05 yaw). The cache is keyed by the
    sprite's id and checked against a weak reference to it, so a new sprite that reuses a freed one's id misses."""
    global _cache_bytes
    roll, yaw = round(roll), round(yaw * 20) / 20
    key = (id(sprite), roll, yaw, mirror)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None and hit[0]() is sprite:
            _CACHE.move_to_end(key)
            return hit[1]
    sp = np.ascontiguousarray(sprite[:, ::-1] if mirror else sprite)
    h0, w0 = sp.shape[:2]
    S = int(round(w0 * CANVAS))
    a = abs(yaw)
    w = max(MIN_W, 1 - 0.25 * a) * w0 / 2           # half width after the turn
    near, far = (1 + 0.06 * a) * w0 / 2, (1 - 0.18 * a) * w0 / 2  # half heights of near / far edge
    lh, rh = (far, near) if yaw > 0 else (near, far)  # facing image-right: right edge recedes
    pts = np.array([[-w, -lh], [w, -rh], [w, rh], [-w, lh]], float)
    th = math.radians(roll)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    pts = pts @ R.T + S / 2
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w0, 0], [w0, h0], [0, h0]]), np.float32(pts))
    canvas = cv2.warpPerspective(sp, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
    shift = SHIFT * yaw / 2  # the face sits towards where it points; the back of the head is not identifying
    out = (canvas, (shift * math.cos(th), shift * math.sin(th)))
    with _CACHE_LOCK:
        old = _CACHE.pop(key, None)
        if old is not None:
            _cache_bytes -= old[2]
        _CACHE[key] = (weakref.ref(sprite), out, canvas.nbytes)
        _cache_bytes += canvas.nbytes
        while _cache_bytes > CACHE_BYTES and len(_CACHE) > 1:
            _cache_bytes -= _CACHE.popitem(last=False)[1][2]
    return out


def posed_box(c, off):
    """Head box c=[cx,cy,s] -> the box to paste a posed_sprite canvas into."""
    return [c[0] + off[0] * c[2], c[1] + off[1] * c[2], c[2] * CANVAS]
