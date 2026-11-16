"""Pose helpers: head geometry and facing state from COCO-17 keypoints.

Keypoints: 0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows, 9/10 wrists, 11/12 hips, ...
A pose entry from experiments/detect.py "pose_full": [x1,y1,x2,y2,score, k0x,k0y,k0c, ..., k16x,k16y,k16c].
"""
import numpy as np

NOSE, LEYE, REYE, LEAR, REAR, LSH, RSH, LEL, REL, LWR, RWR = range(11)
VIS = 0.3


def kps(p):
    return np.array(p[5:5 + 51], float).reshape(17, 3)


def head(p):
    """Head estimate from one pose entry -> (cx, cy, head_width, confidence) or None.
    Head width is in 'head widths' via anthropometric ratios (shoulders ~2.2, ears ~0.8,
    eyes ~0.42 head widths); callers calibrate per person against face boxes."""
    k = kps(p)
    c = k[:, 2]
    headpts = [i for i in (NOSE, LEYE, REYE, LEAR, REAR) if c[i] > VIS]
    widths = []
    if c[LSH] > VIS and c[RSH] > VIS:
        widths.append(np.linalg.norm(k[LSH, :2] - k[RSH, :2]) / 2.2)
    if c[LEAR] > VIS and c[REAR] > VIS:
        widths.append(np.linalg.norm(k[LEAR, :2] - k[REAR, :2]) / 0.8)
    if c[LEYE] > VIS and c[REYE] > VIS:
        widths.append(np.linalg.norm(k[LEYE, :2] - k[REYE, :2]) / 0.42)
    if not widths:
        return None
    hw = float(np.median(widths))
    if len(headpts) >= 2:
        cx, cy = k[headpts, :2].mean(0)
        conf = float(c[headpts].mean())
    elif c[LSH] > VIS and c[RSH] > VIS:  # back of head / occluded face: above the shoulders
        mid = k[[LSH, RSH], :2].mean(0)
        cx, cy = mid[0], mid[1] - 1.1 * hw
        conf = float(c[[LSH, RSH]].mean()) * 0.5
    else:
        return None
    return float(cx), float(cy), hw, conf


def facing(p):
    """'front', 'profile', 'away' or 'unknown' from face keypoint visibility."""
    c = kps(p)[:, 2]
    eyes = (c[LEYE] > 0.5) + (c[REYE] > 0.5)
    if c[NOSE] > 0.5 and eyes == 2:
        return "front"
    if c[NOSE] > 0.4 or eyes >= 1:
        return "profile"
    if (c[LEAR] > 0.4 or c[REAR] > 0.4 or (c[LSH] > 0.5 and c[RSH] > 0.5)) and c[NOSE] < 0.2 and eyes == 0:
        return "away"
    return "unknown"


def head_from_pose(k, box):
    """Head box [x1, y1, x2, y2, score] from one skeleton's COCO keypoints k (17x3: x, y, confidence) and its person
    box [x1, y1, x2, y2] (experiments/detect.py's "pose_head"): around the visible head points (nose, eyes, ears)
    when at least two are seen; one head point sized by the shoulders; else above the shoulders (back of the head)."""
    xy, c = k[:, :2], k[:, 2]
    head = [i for i in range(NOSE, REAR + 1) if c[i] > VIS]
    sh = [i for i in (LSH, RSH) if c[i] > VIS]
    shw = np.linalg.norm(xy[LSH] - xy[RSH]) if len(sh) == 2 else 0
    if len(head) >= 2:
        pts = xy[head]
        cx, cy = pts.mean(0)
        span = max(np.ptp(pts[:, 0]), 0.55 * shw, 0.10 * (box[3] - box[1]), 8)
        s, score = span * 1.6, float(c[head].mean())
    elif len(head) == 1 and shw:
        (cx, cy), s, score = xy[head[0]], 0.8 * shw, float(c[head[0]]) * 0.7
    elif shw:  # back of head / occluded face: extrapolate above the shoulders
        mid = xy[sh].mean(0)
        cx, cy, s, score = mid[0], mid[1] - 0.75 * shw, 0.8 * shw, float(c[sh].mean()) * 0.4
    else:
        return None
    return [float(cx - s / 2), float(cy - s / 2), float(cx + s / 2), float(cy + s / 2), score]


def head_box(entry):
    """pose_full entry -> head box [x1,y1,x2,y2,score] (pose_head rule, score scaled by sqrt(person score))."""
    h = head_from_pose(kps(entry), entry[:4])
    if not h:
        return None
    h[4] *= float(entry[4]) ** 0.5
    return [float(v) for v in h]


# A head-box centre from a skeleton (measured on detected faces with a skeleton): head keypoints (nose/eyes/ears mean)
# -> centre +0.02 up (MAD 0.02); shoulder midpoint -> centre 0.75 up, 0.06 left (MAD 0.08 / 0.19; in shoulder widths
# it varies ~3x with body turn, so box size is the unit).
HEAD_FROM_KP, HEAD_FROM_SHOULDERS = (0.0, -0.023), (-0.056, -0.746)


def pose_head(p, s):
    """(head-box centre, confidence q in [0,1]) from a pose entry p for a head of box size s: confidence-weighted
    head keypoints, or the shoulders (0.75 box sizes above their midpoint) at half their confidence -- whichever is
    more confident. (The editor snaps a followed box to it.)"""
    k = kps(p)
    h = k[NOSE:REAR + 1]
    q_h = float(np.sort(h[:, 2])[-2:].mean())
    xy_h = (h[:, :2] * h[:, 2:3]).sum(0) / max(h[:, 2].sum(), 1e-6) + np.array(HEAD_FROM_KP) * s
    q_s = 0.5 * float(min(k[LSH, 2], k[RSH, 2]))
    xy_s = (k[LSH, :2] + k[RSH, :2]) / 2 + np.array(HEAD_FROM_SHOULDERS) * s
    return (xy_h, q_h) if q_h >= q_s else (xy_s, q_s)


OWN_POSE_TOL = 0.5  # head sizes


def pose_for_head(entries, c, min_score=0.25):
    """The pose_full entry (one frame's list) whose head estimate is nearest head box c = [cx, cy, s], or None.
    Only a skeleton whose head is within OWN_POSE_TOL head sizes counts as this head's own. On jazz census truth
    a skeleton's real owner is within 0.31 head sizes 9 times in 10 and the
    next-nearest head is beyond 0.65 9 times in 10; the old 0.8 let 18 of 131 heads in a crowd take a
    neighbour's skeleton, giving two people the same body picture (body similarity 1.0 between strangers)."""
    best = None
    for p in entries:
        if p[4] < min_score:
            continue
        h = head(p)
        if h:
            d = np.hypot(h[0] - c[0], h[1] - c[1])
            if d < OWN_POSE_TOL * max(c[2], h[2] * 1.3) and (best is None or d < best[0]):
                best = (d, p)
    return best[1] if best else None
