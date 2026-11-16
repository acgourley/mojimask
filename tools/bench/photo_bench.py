"""The decisions benchmark for photos: each checked frame of a clip's census, saved as a photo and scanned on its own.

A photo is a clip of one frame (video_io), so every face in it is seen once. This scores the scan's rules on such
one-frame projects the way decision_bench.py scores a clip: the open questions are answered by the census (in the
Decisions page's order), and what's left uncovered, covered needlessly or asked is counted, summed over the photos.

Photos and their scratch projects: out/bench/photos/<clip>_<t>.png and out/bench/photos/<clip>_<t>/ (detection is
cached there; the decide step always runs again with today's code).

    python tools/bench/photo_bench.py jazz [--note TEXT] [--every K] [--sheets]
    python tools/bench/photo_bench.py --history

Each run appends a line to tools/bench/data/photos_history.jsonl (numbers only).
"""
import argparse
import collections
import json
import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import decision_bench as B  # noqa: E402
from mojimask import paths  # noqa: E402

HISTORY = os.path.join(HERE, "data", "photos_history.jsonl")


class PhotoTruth(B.Truth):
    """The census boxes of one checked frame, as the truth of frame 0 of a one-photo project."""
    def __init__(self, proj, boxes):
        meta = json.load(open(os.path.join(proj, "meta.json")))
        self.W, self.H = meta["width"], meta["height"]
        self.pt = json.load(open(os.path.join(proj, "proto_tracks.json")))
        self.frames, self.scored, self.source, self._lab = {0: boxes}, {0}, "census", {}


def photos(clip, every=1):
    """[(t, photo path)] for the census's checked frames, writing the photos the first time."""
    from mojimask import video_io
    video = os.path.join(ROOT, "test_clips", f"{clip}.mp4")
    census = json.load(open(paths.truth(clip, ROOT)))
    checked = sorted(int(t) for t in census["checked"])[::every]
    d = os.path.join(ROOT, "out", "bench", "photos")
    os.makedirs(d, exist_ok=True)
    out = [(t, os.path.join(d, f"{clip}_{t}.png")) for t in checked]
    if not all(os.path.exists(p) for _, p in out):
        want = dict(out)
        for t, f in enumerate(video_io.frames(video, None, color="display")):
            if t in want:
                cv2.imwrite(want[t], f)
            if t >= checked[-1]:
                break
    return out


def run(clip, every=1, note=""):
    census, _ = B.census_frames(paths.truth(clip, ROOT))
    asked, harms, res = collections.Counter(), collections.Counter(), collections.Counter()
    per, t0 = [], time.time()
    for t, photo in photos(clip, every):
        proj = photo[:-4]
        B.scan_into(photo, proj)
        truth = PhotoTruth(proj, census[t])
        ds = B.decisions(proj)
        facts, a, log, final = B.replay(ds, truth.pt.get("hidden", []), truth)
        cov = B.Covers(truth, facts)
        rules, wrong = B.rule_errors(final, truth, truth, cov)
        r = B.result(truth, cov)
        asked.update(a); harms.update(w["harm"] for w in wrong)
        res.update({k: r[k] for k in ("face_frames", "not_found_by_scan", "uncovered_frames")})
        per.append({"t": t, "asked": dict(a), "wrong": [{k: w[k] for k in ("id", "rule", "said", "truth", "harm")} for w in wrong],
                    "result": {k: r[k] for k in ("face_frames", "not_found_by_scan", "uncovered_frames")}})
    rec = {"at": time.strftime("%Y-%m-%d %H:%M"), "clip": clip, "photos": len(per), "code": B.fingerprint(), "note": note,
           "seconds": round(time.time() - t0), "asked": sum(asked.values()), "asked_by_kind": dict(asked),
           "rule_wrong": dict(harms), "result": dict(res)}
    os.makedirs(os.path.dirname(HISTORY), exist_ok=True)
    with open(HISTORY, "a") as f:
        f.write(json.dumps(rec) + "\n")
    json.dump({"record": rec, "photos": per}, open(os.path.join(ROOT, "out", "bench", "photos", f"{clip}_last.json"), "w"), indent=1)
    return rec, per


def summary(r):
    w, res, k = r["rule_wrong"], r["result"], r["asked_by_kind"]
    kinds = " ".join(f"{n} {k[n]}" for n in ("track", "blip", "overlap", "same", "whose") if k.get(n))
    return (f"{r['at']}  {r['clip']} x{r['photos']} photos  asked {r['asked']:3d} ({kinds})  "
            f"needless covers {w.get('needless cover', 0) + w.get('cover, unlabelled', 0):2d}  "
            f"faces uncovered by rules {w.get('face uncovered', 0)}  uncovered {res.get('uncovered_frames', 0)}/{res.get('face_frames', 0)}  "
            f"never boxed {res.get('not_found_by_scan', 0)}" + (f"  | {r['note']}" if r["note"] else ""))


def sheets(clip, per):
    """One picture per photo: green = covered, red = a census face left uncovered, orange = a needless cover,
    blue = an open question's box (answered by the census in the replay). Only for footage that may be looked at
    (decision_bench.VIEWABLE)."""
    out = os.path.join(ROOT, "out", "bench", "photos", "sheets")
    os.makedirs(out, exist_ok=True)
    census, _ = B.census_frames(paths.truth(clip, ROOT))
    from mojimask import facts as FX
    for p in per:
        proj = os.path.join(ROOT, "out", "bench", "photos", f"{clip}_{p['t']}")
        im = cv2.imread(proj + ".png")
        H, W = im.shape[:2]
        truth = PhotoTruth(proj, census[p["t"]])
        hidden = truth.pt.get("hidden", [])
        open_ = {str(d["about"]["piece"]) for d in FX.answers(B.decisions(proj), [], hidden) if d["answer"] is None}
        kept = [(k, truth.frac(tr["cover"]["0"])) for k, tr in truth.pt["tracks"].items() if k not in hidden and "0" in tr["cover"]]
        px = lambda b: ((int(b[0] * W), int(b[1] * H)), (int(b[2] * W), int(b[3] * H)))  # noqa: E731
        for g in census[p["t"]]:
            if g[4][0] == "P":
                col = (60, 200, 60) if B.Covers.hit(g[:4], [b for _, b in kept]) else (40, 40, 230)
                cv2.rectangle(im, *px(g), col, 2)
        for k, b in kept:
            m = truth.match(0, b)
            if k in open_:
                cv2.rectangle(im, *px(b), (230, 140, 40), 1)
            elif not (m and m[0] in "PH"):
                cv2.rectangle(im, *px(b), (0, 150, 255), 1)
        cv2.putText(im, f"t{p['t']} asked {sum(p['asked'].values())}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(out, f"{clip}_{p['t']}.jpg"), im)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?", default="jazz")
    ap.add_argument("--every", type=int, default=1, help="every Kth checked frame")
    ap.add_argument("--note", default="")
    ap.add_argument("--sheets", action="store_true")
    ap.add_argument("--viewable", action="store_true", help="this clip's frames may be looked at (public footage, no guests)")
    ap.add_argument("--history", action="store_true")
    a = ap.parse_args()
    if a.sheets and a.clip not in B.VIEWABLE and not a.viewable:
        ap.error(f"--sheets draws the clip's faces: only for {', '.join(sorted(B.VIEWABLE))} (or say --viewable)")
    if a.history:
        for line in open(HISTORY) if os.path.exists(HISTORY) else []:
            print(summary(json.loads(line)))
        raise SystemExit
    rec, per = run(a.clip, a.every, a.note)
    print(summary(rec))
    if a.sheets:
        print("sheets:", sheets(a.clip, per))
