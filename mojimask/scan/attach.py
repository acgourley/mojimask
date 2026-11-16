"""After the graph: attach fragments (groups that aren't real people -- a few detections, or never a clear face)
to the real person they belong to. A fast mover (twirl's spinning girl) breaks into such fragments; they are
covered either way, but without this they belong to nobody (no person, no consistent emoji).

A fragment F goes to real person R when
  - F and R are never on screen at the same time as different heads;
  - motion fits: R's nearest sighting before F and/or after F (within MAX_GAP frames) is reachable from F
    (<= 1 head size + V head sizes per frame of the gap), on every side where R has one;
  - and F continues R: within CONT_GAP frames of where R left off (or leads into R), a detection >= CONT_SCORE,
    within CONT_CLOSE of the reachable distance, a face size within CONT_SIZE of R's, no third head within CROWD
    head sizes of F around then (a stray second detection on F's or R's own head doesn't count), and no other
    person fits.
Body matches (OSNet) and "on the path" (motion alone) were tried and removed: under the party clip's coloured light
body matches attached fragments to the wrong person even at >= 0.92 (jazz strangers' p99 is 0.84; that bar doesn't
carry over), and motion alone attached a wrong person and junk on jazz. Graded (2026-11-12): party 0 attached; jazz 0;
twirl 1 (the spinning girl at 175).
A fragment that fits nobody stays its own covered piece -- covered either way; this only gives it a person.

    attach(proj, st, gd) -> {fragment group id: person group id}
"""
import numpy as np

from mojimask.scan import graph as G  # noqa: E402
from mojimask.scan.questions import V  # noqa: E402  (head sizes per frame a person can move: the same as the questions')

MAX_GAP = 24
CROWD = 1.5  # head sizes: continuation only where no other head is this near the fragment (party: jumps to a neighbour)
# continuation: F picks up within CONT_GAP frames of where R left off (or leads into R) and no other person could
# explain it -- motion alone, but only over a short gap and only when it is unambiguous
CONT_GAP = 3  # jazz: 0 wrong-person attachments at 3, 6 and 10
CONT_SCORE = 0.6
CONT_CLOSE = 0.5  # within half the reachable distance
CONT_SIZE = 1.3  # face sizes within this ratio


def attach(proj, st, gd):
    """gd: questions.groups_data(proj, st) -- the real people (with "obs", "ts", "ms")."""
    info, members, apart, mats = st["info"], st["members"], st["apart"], st["mats"]
    inv = {}
    out = {}
    at = {}  # frame -> [(group, xy, size)] in picture coordinates
    for g, gms in members.items():
        for c in gms:
            for t, x in info[c]["obs"].items():
                at.setdefault(t, []).append((g, x))

    def near(a, b, k):
        return np.hypot(a[0] - b[0], a[1] - b[1]) < k * (a[2] + b[2]) / 2

    def alone(f, r, t, c):
        """No third head (any group, blips included) near this detection within CONT_GAP frames -- not counting a
        detection sitting on F's or R's own head in that frame (a stray second detection of the same head)."""
        for u in range(t - CONT_GAP, t + CONT_GAP + 1):
            here = at.get(u, ())
            own = [x for g, x in here if g in (f, r)]
            for g, x in here:
                if g in (f, r) or not near(x, c, CROWD) or any(near(x, o, 1.0) for o in own):
                    continue
                return False
        return True

    for f, ms in members.items():
        if f in gd:
            continue
        obs = {}
        for c in ms:
            obs.update(info[c]["obs"])
        fts = sorted(obs)
        conf = max(info[c].get("score", {}).get(t, 0) for c in ms for t in info[c]["obs"])
        conts = []
        for r, R in gd.items():
            if any(apart(x, y) for x in ms for y in R["ms"]):
                continue
            ok, close = True, 9.0
            sides, gap, ratio = 0, 99, 1.0
            for t_edge, d in ((fts[0], -1), (fts[-1], 1)):
                cand = [u for u in R["ts"] if 0 < d * (u - t_edge) <= MAX_GAP]
                if not cand:
                    continue
                u = min(cand, key=lambda u: abs(u - t_edge))
                (pf, sf), (pr, sr) = G.to_room(mats, t_edge, obs[t_edge], inv), G.to_room(mats, u, R["obs"][u], inv)
                dist = float(np.hypot(*(pf - pr)) / ((sf + sr) / 2))
                if dist > 1.0 + V * abs(u - t_edge):
                    ok = False
                    break
                close = min(close, dist / (1.0 + V * abs(u - t_edge)))
                sides += 1
                if abs(u - t_edge) < gap:
                    gap, ratio = abs(u - t_edge), max(sf, sr) / min(sf, sr)
            if not ok or sides == 0:
                continue
            if gap <= CONT_GAP and conf >= CONT_SCORE and close <= CONT_CLOSE and ratio <= CONT_SIZE and \
                    all(alone(f, r, t, obs[t]) for t in fts):
                conts.append(r)
        if len(conts) == 1:
            out[f] = conts[0]
    return out
