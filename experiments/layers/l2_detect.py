"""Layers 2 and 3 on labelled truth: where are the faces, and is each detection a real one?

Truth: the census (paths.truth, made with tools/label), only frames confirmed as complete ("checked").
Detections: the project's dets/*.json, face boxes turned into head boxes (track.face_to_head) like the app.

Layer 2, per detector and for the app's fused mix, over a threshold sweep:
  recall of truth heads by class (face clear / face mostly hidden / head / tiny),
  false detections per frame (unmatched boxes; a confirmed frame lists every head, so anything else is false),
  how well matched boxes sit (IoU with the truth box).
Layer 3, for every fused detection at the labelling pre-fill's low threshold (0.15): how well each signal
separates real heads from false ones (ROC AUC): fused score, number of detectors agreeing, best single score,
size, pose-head support.

    python experiments/layers/l2_detect.py jazz out/projects/jazz_20261111-160938 [--viewable]

Writes out/layers/l2_<name>.json and sheets of misses and false detections (out/layers/l2_<name>_*.jpg). The
sheets show faces, so only for clips that may be looked at (VIEWABLE; jazz: a public-domain film); any other clip
gets numbers only unless --viewable says it may be looked at.
"""
import collections
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mojimask import pipeline  # noqa: E402  (the scan's detector mix and tracker thresholds)
from mojimask import track as T  # noqa: E402

OUT = os.path.join(ROOT, "out", "layers")
DETS = ["yolov5m_face_tiled", "retinaface_r50_2x", "pose_head"]
APP = [(n, float(th)) for n, th in (d.split(":") for d in pipeline.CORE_DETS)]
MIN_SCORE, MIN_PEAK = pipeline.CORE_TRACKER["min_score"], pipeline.CORE_TRACKER["min_peak"]
MIN_IOU = 0.3
VIEWABLE = {"jazz"}  # (decision_bench.VIEWABLE: clips whose faces may be drawn; party clips are real guests)


def load_truth(name):
    d = json.load(open(__import__("mojimask.paths", fromlist=["truth"]).truth(name)))
    W, H = d["w"], d["h"]
    out = {}
    for t in d["checked"]:
        out[int(t)] = [dict(b, b=[b["b"][0] * W, b["b"][1] * H, b["b"][2] * W, b["b"][3] * H]) for b in d["frames"][str(t)]]
    return out, W, H


def kind(g):
    if g["c"] == "face":
        return "face hidden" if g.get("o") else "face"
    return g["c"]


def match(dets, truth):
    """Greedy by IoU; returns {det index: truth index} for pairs with IoU >= MIN_IOU, plus the IoUs."""
    pairs = sorted(((T.iou(d, g["b"]), i, j) for i, d in enumerate(dets) for j, g in enumerate(truth)), reverse=True)
    used_d, used_g, m = set(), set(), {}
    for o, i, j in pairs:
        if o < MIN_IOU:
            break
        if i in used_d or j in used_g:
            continue
        used_d.add(i)
        used_g.add(j)
        m[i] = (j, o)
    return m


def raw(proj, name):
    d = json.load(open(f"{proj}/dets/{name}.json"))["frames"]
    conv = T.face_to_head if name in T.FACE_DETS else (lambda b: b)
    return [[(conv(b[:4]), b[4]) for b in f] for f in d]


def evaluate(truth, frames_dets, ths):
    """frames_dets(t, th) -> list of boxes. Recall per class, false detections per frame, IoU of matches."""
    rows = []
    for th in ths:
        hit, tot, fp, fp_not, ious = collections.Counter(), collections.Counter(), 0, 0, []
        for t, G in truth.items():
            dets = frames_dets(t, th)
            heads = [g for g in G if g["c"] != "not"]
            nots = [g for g in G if g["c"] == "not"]
            m = match(dets, heads)
            for j, g in enumerate(heads):
                tot[kind(g)] += 1
            for i, (j, o) in m.items():
                hit[kind(heads[j])] += 1
                ious.append(o)
            un = [d for i, d in enumerate(dets) if i not in m]
            fp += len(un)
            fp_not += sum(any(T.iou(d, g["b"]) >= MIN_IOU for g in nots) for d in un)
        rows.append({"th": th, "recall": {k: round(hit[k] / tot[k], 3) for k in tot}, "n": dict(tot),
                     "false_per_frame": round(fp / len(truth), 2), "false_on_marked_not": fp_not, "false_elsewhere": fp - fp_not,
                     "iou_median": round(float(np.median(ious)), 2) if ious else None})
    return rows


def auc(pos, neg):
    if not pos or not neg:
        return None
    s = np.array(pos + neg)
    r = s.argsort().argsort() + 1.0
    return round(float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 3)


def main(name, proj, pictures=False):
    truth, W, H = load_truth(name)
    R = {n: raw(proj, n) for n in DETS if os.path.exists(f"{proj}/dets/{n}.json")}
    ths = [0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    res = {"name": name, "frames": sorted(truth), "per_detector": {}, "app_mix": None}
    for n, fr in R.items():
        res["per_detector"][n] = evaluate(truth, lambda t, th, fr=fr: [b for b, s in fr[t] if s >= th], ths)

    def mix(t, th):
        boxes = [(b, s, n) for n, th0 in APP for b, s in R[n][t] if s >= th0]
        return [b for b, s, src in T.fuse(boxes) if s >= th]
    # the app: detectors at their CORE_DETS thresholds, fused; a track starts at fused score >= min_score
    res["app_mix"] = evaluate(truth, mix, [0.0, MIN_SCORE, MIN_PEAK])
    res["app_mix_note"] = (f"th 0.0 = every fused box at the app's per-detector thresholds; {MIN_SCORE} = may start a track; "
                           f"{MIN_PEAK} = min_peak")

    # layer 3: every fused detection at the pre-fill threshold, labelled by truth
    feats = []
    for t, G in truth.items():
        boxes = [(b, s, n) for n in R for b, s in R[n][t] if s >= 0.15]
        fused = T.fuse(boxes)
        heads = [g for g in G if g["c"] != "not"]
        m = match([b for b, _, _ in fused], heads)
        for i, (b, s, srcs) in enumerate(fused):
            best = {n: max([sc for bb, sc in R[n][t] if sc >= 0.15 and T.iou(bb, b) > 0.3] or [0]) for n in R}
            feats.append({"t": t, "box": [round(v, 1) for v in b], "real": i in m, "kind": kind(heads[m[i][0]]) if i in m else "false",
                          "fused": s, "agree": len(srcs), "best": max(best.values()), "size": max(b[2] - b[0], b[3] - b[1]),
                          "face_det": max(best.get("yolov5m_face_tiled", 0), best.get("retinaface_r50_2x", 0)),
                          "pose": best.get("pose_head", 0)})
    pos = [f for f in feats if f["real"]]
    neg = [f for f in feats if not f["real"]]
    res["layer3"] = {"real": len(pos), "false": len(neg),
                     "auc": {k: auc([f[k] for f in pos], [f[k] for f in neg]) for k in ("fused", "agree", "best", "size", "face_det", "pose")}}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, f"l2_{name}.json"), "w"), indent=1)
    json.dump(feats, open(os.path.join(OUT, f"l3_{name}_feats.json"), "w"))
    if pictures:
        sheets(name, proj, truth, R, mix, feats)
    return res


def crop(img, b, size=96, color=(0, 220, 0), text=""):
    cx, cy, s = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(b[2] - b[0], b[3] - b[1]) * 2.2
    x0, y0 = int(cx - s / 2), int(cy - s / 2)
    pad = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT)
    c = pad[y0 + 200:y0 + 200 + int(s), x0 + 200:x0 + 200 + int(s)]
    if c.size == 0:
        c = np.zeros((size, size, 3), np.uint8)
    k = size / max(1, c.shape[0])
    c = cv2.resize(c, (size, size))
    cv2.rectangle(c, (int((b[0] - x0) * k), int((b[1] - y0) * k)), (int((b[2] - x0) * k), int((b[3] - y0) * k)), color, 1)
    if text:
        cv2.putText(c, text, (2, size - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1)
    return c


def grid(tiles, cols=10, title=""):
    if not tiles:
        return None
    rows = [np.hstack(tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols]))) for i in range(0, len(tiles), cols)]
    g = np.vstack(rows)
    bar = np.zeros((22, g.shape[1], 3), np.uint8)
    cv2.putText(bar, title, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return np.vstack([bar, g])


def sheets(name, proj, truth, R, mix, feats):
    imgs = {t: cv2.imread(f"{proj}/frames/{t:05d}.jpg") for t in truth}
    missed, false_app = [], []
    for t, G in truth.items():
        heads = [g for g in G if g["c"] != "not"]
        dets = mix(t, MIN_SCORE)
        m = match(dets, heads)
        got = {j for j, _ in m.values()}
        missed += [crop(imgs[t], g["b"], color=(0, 0, 255), text=f"{t} P{g['p']} {kind(g)}") for j, g in enumerate(heads) if j not in got]
        false_app += [crop(imgs[t], d, color=(0, 160, 255), text=f"{t}") for i, d in enumerate(dets) if i not in m]
    low_false = [crop(imgs[f["t"]], f["box"], color=(200, 200, 200), text=f"{f['t']} {f['fused']:.2f}") for f in feats if not f["real"]]
    for tag, tiles, title in (("missed", missed, f"truth heads the app misses (fused >= {MIN_SCORE})"),
                              ("false_app", false_app, f"app detections (fused >= {MIN_SCORE}) on no truth head"),
                              ("false_low", low_false, "every pre-fill detection (>= 0.15) on no truth head")):
        g = grid(tiles, title=f"{name}: {title} ({len(tiles)})")
        if g is not None:
            cv2.imwrite(os.path.join(OUT, f"l2_{name}_{tag}.jpg"), g)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--viewable"]
    viewable = args[0] in VIEWABLE or "--viewable" in sys.argv
    if not viewable:
        print(f"no sheets: they draw the clip's faces, only for {', '.join(sorted(VIEWABLE))} (or say --viewable)",
              file=sys.stderr)
    r = main(args[0], args[1], pictures=viewable)
    print(json.dumps({k: v for k, v in r.items() if k != "per_detector"}, indent=1))
    for n, rows in r["per_detector"].items():
        print(n)
        for row in rows:
            print("  ", row)
