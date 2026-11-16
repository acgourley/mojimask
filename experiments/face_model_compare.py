"""Compare the face landmark + face likeness models on the jazz census (2026-11-14).

Then:       SCRFD-10G (landmarks) + ArcFace R50 w600k (likeness), InsightFace buffalo_l: non-commercial only.
            (The app's pair when this was written; it has since moved to RetinaFace + SFace, reid.py.)
Candidates: YuNet 2023mar (MIT) + SFace 2021dec (Apache-2.0), OpenCV Zoo, in models/candidates/.

Every combination (detector x recognizer) on the census faces of tools/truth/jazz.json (538 faces, 33 people, every
4th frame), with the scan's own crop, nearest-face rule and frontal gate (scan/faces.py) and the same
112x112 alignment. Measures:
  found / frontal   how many census faces get landmarks, and pass the gate the scan compares faces behind
  pairs             single frontal faces, pairs from different frames: AUC, true-same rate at 1% false-same
  groups            means of 3 consecutive frontal faces of a person (as graph.py averages a chain's faces):
                    AUC; true-same at <=1% false-same (the "same person" side); true-different at <=1% same
                    pairs called different (the "different people" side); ArcFace also at its former app
                    thresholds (SAME_FACE 0.55, DIFF_FACE 0.25)
  rank-1            each frontal face's most similar face >= 2 s away: same person?
Doesn't touch the app: the old pair comes from experiments/insightface_models.py, the candidates run here on ONNX
Runtime.

    .venv/bin/python experiments/face_model_compare.py [--sheet out/face_compare/disagree.jpg]
    COMMON=1 ...      likeness on the faces frontal under both detectors (like for like); YUNET_FIT=224 ... crop size
"""
import argparse
import itertools
import json
import os
import sys
import threading
import time

import cv2
import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import insightface_models as reid  # noqa: E402  (SCRFD + ArcFace: the app's pair when this was written)
from mojimask.scan import faces as FA  # noqa: E402  (NEAR, FRONTAL: the scan's rules)

CAND = os.path.join(ROOT, "models", "candidates")
TRUTH = __import__("mojimask.paths", fromlist=["truth"]).truth("jazz")
FRAMES = os.path.join(ROOT, "out", "bench", "jazz", "frames")
HEAD_PER_FACE = 1.15  # census boxes are tight faces; the scan's c[2] is a head box, a little bigger


def _sess(name):
    return ort.InferenceSession(os.path.join(CAND, name), providers=["CPUExecutionProvider"])


YUNET = SFACE = None
_LAZY = threading.Lock()  # swap_bench calls these from the scan's threads: one session each, made once


YUNET_FIT = int(os.environ.get("YUNET_FIT", 160))  # the crop's long side on the 640 canvas (the rest is padding): 160 best on jazz; 640 (filling it) loses 40% of frontal faces


def yunet(img, size=640, thresh=0.3, iou=0.4, **_):
    """Faces in img (BGR) -> (boxes [n,5], kps [n,5,2]), decoded as OpenCV's FaceDetectorYN does (2023mar heads:
    score = sqrt(cls * obj); centre offsets and log sizes per stride; raw BGR 0-255 in). The ONNX input is fixed at
    640x640, so the image is scaled to fit and padded (bottom/right)."""
    global YUNET
    if YUNET is None:
        with _LAZY:
            YUNET = YUNET or _sess("face_detection_yunet_2023mar.onnx")
    h, w = img.shape[:2]
    k = min(size, YUNET_FIT) / max(h, w)
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:int(round(h * k)), :int(round(w * k))] = cv2.resize(img, (int(round(w * k)), int(round(h * k))))
    out = dict(zip([o.name for o in YUNET.get_outputs()],
                   YUNET.run(None, {"input": canvas.transpose(2, 0, 1)[None].astype(np.float32)})))
    B, K = [], []
    for s in (8, 16, 32):
        cols = size // s
        idx = np.arange(cols * cols)
        cc, rr = (idx % cols).astype(np.float32), (idx // cols).astype(np.float32)
        sc = np.sqrt(np.clip(out[f"cls_{s}"][0, :, 0], 0, 1) * np.clip(out[f"obj_{s}"][0, :, 0], 0, 1))
        m = sc >= thresh
        bb, kp = out[f"bbox_{s}"][0][m], out[f"kps_{s}"][0][m]
        cx, cy = (cc[m] + bb[:, 0]) * s, (rr[m] + bb[:, 1]) * s
        bw, bh = np.exp(bb[:, 2]) * s, np.exp(bb[:, 3]) * s
        B.append(np.c_[cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, sc[m]])
        K.append(np.stack([(kp[:, 0::2] + cc[m][:, None]) * s, (kp[:, 1::2] + rr[m][:, None]) * s], -1))
    B, K = np.concatenate(B), np.concatenate(K)
    if not len(B):
        return np.zeros((0, 5)), np.zeros((0, 5, 2))
    keep = __import__("mojimask.onnx_core", fromlist=["nms"]).nms(B[:, :4], B[:, 4], iou)
    B, K = B[keep], K[keep]
    B[:, :4] /= k
    return B, K / k


def sface(aligned):
    """112x112 aligned face (BGR) -> 128-d, as OpenCV's FaceRecognizerSF: RGB, 0-255, no mean or scale."""
    global SFACE
    if SFACE is None:
        with _LAZY:
            SFACE = SFACE or _sess("face_recognition_sface_2021dec.onnx")
    blob = aligned[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None].copy()
    return SFACE.run(None, {"data": blob})[0][0]


RETINA_SIZE = int(os.environ.get("RETINA_SIZE", 192))   # crops resized to this square: a fixed shape for CoreML
RETINA_ENGINE = os.environ.get("RETINA_ENGINE", "ane")   # as the app runs it (onnx_core.ENGINES); "cpu" to compare
_RETINA = threading.local()  # one session per thread (the scan measures faces in threads)


def retina(img, thresh=0.3, iou=0.4, **_):
    """Faces in img (BGR) -> (boxes [n,5], kps [n,5,2]) from the app's own RetinaFace-R50 (models/onnx, MIT code),
    with its landmark head decoded too (onnx_core decodes only boxes): BGR minus (104, 117, 123), priors from
    onnx_core.anchors, variances 0.1 / 0.2 as biubug6. The image is resized to RETINA_SIZE square and run on
    RETINA_ENGINE (Neural Engine: ~3.4 ms at 192 against ~100+ ms on the CPU)."""
    from mojimask import onnx_core as OC
    S = RETINA_SIZE
    sess = getattr(_RETINA, "s", None)
    if sess is None:
        sess = _RETINA.s = OC.session(os.path.join(OC.MODELS, "retinaface-r50.onnx"), RETINA_ENGINE,
                                      {"batch_size": 1, "height": S, "width": S})
    H, W = img.shape[:2]
    sq = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA if max(H, W) > S else cv2.INTER_LINEAR)
    x = np.ascontiguousarray((sq.astype(np.float32) - np.array([104, 117, 123], np.float32)).transpose(2, 0, 1)[None])
    loc, conf, lm = sess.run(None, {sess.get_inputs()[0].name: x})
    pri = OC.anchors(S, S)
    loc, sc, lm = loc[0], conf[0][:, 1], lm[0]
    m = sc >= thresh
    loc, sc, lm, pr = loc[m], sc[m], lm[m], pri[m]
    if not len(sc):
        return np.zeros((0, 5)), np.zeros((0, 5, 2))
    c = pr[:, :2] + loc[:, :2] * 0.1 * pr[:, 2:]
    wh = pr[:, 2:] * np.exp(loc[:, 2:] * 0.2)
    b = np.c_[c - wh / 2, c + wh / 2, sc] * np.array([W, H, W, H, 1], np.float32)  # priors are fractions: back to img
    k = (pr[:, None, :2] + lm.reshape(-1, 5, 2) * 0.1 * pr[:, None, 2:]) * np.array([W, H], np.float32)
    keep = OC.nms(b[:, :4], sc, iou)
    return b[keep], k[keep]


DETECTORS = {"scrfd": reid.scrfd, "yunet": yunet, "retina": retina}
if os.environ.get("DETECTORS"):  # e.g. DETECTORS=scrfd,retina
    DETECTORS = {k: DETECTORS[k] for k in os.environ["DETECTORS"].split(",")}
COMMON = os.environ.get("COMMON") == "1"  # likeness only on faces frontal under both detectors (a like-for-like set)
RECOGNIZERS = {"arcface": lambda a: reid.arcface(a).flatten(), "sface": sface}


def crop_of(img, c):
    """faces.face's crop: 0.9 head sizes around the centre, padded, small heads upscaled to 224."""
    H, W = img.shape[:2]
    r = c[2] * 0.9
    x1, y1, x2, y2 = int(c[0] - r), int(c[1] - r), int(c[0] + r), int(c[1] + r)
    pad = np.pad(img, ((max(-y1, 0), max(y2 - H, 0)), (max(-x1, 0), max(x2 - W, 0)), (0, 0)))
    crop = pad[y1 + max(-y1, 0):y2 + max(-y1, 0), x1 + max(-x1, 0):x2 + max(-x1, 0)]
    k = max(1.0, 224 / crop.shape[0])
    return cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC), k, x1, y1


def measure(det, crop, k, x1, y1, c):
    """faces.face's rule: the most confident face whose centre is within NEAR head sizes of the box centre."""
    boxes, kps = DETECTORS[det](crop)
    best = None
    for b, kp in zip(boxes, kps):
        d = np.hypot((b[0] + b[2]) / 2 / k + x1 - c[0], (b[1] + b[3]) / 2 / k + y1 - c[1])
        if d <= FA.NEAR * c[2] and (best is None or b[4] > best[0][4]):
            best = (b, kp)
    if best is None:
        return None
    b, kp = best
    ed = float(np.hypot(*(kp[1] - kp[0]))) + 1e-6
    f = {"score": float(b[4]), "yaw": float((kp[2][0] - (kp[0][0] + kp[1][0]) / 2) / ed),
         "eyes": ed / max(1.0, b[2] - b[0]), "px": float((b[3] - b[1]) / k), "kp": kp}
    f["frontal"] = FA.frontal(f)
    return f


def auc(pos, neg):
    """P(same-person sim > different-person sim), ties half."""
    pos, neg = np.sort(pos), np.sort(neg)
    lo = np.searchsorted(neg, pos, "left")
    hi = np.searchsorted(neg, pos, "right")
    return float(((lo + hi) / 2).sum() / (len(pos) * len(neg)))


def tar_at_far(pos, neg, far):
    t = np.quantile(neg, 1 - far)
    return float((pos > t).mean()), float(t)


def tnr_at_frr(pos, neg, frr):
    t = np.quantile(pos, frr)
    return float((neg < t).mean()), float(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", help="contact sheet of faces where the two detectors disagree")
    ap.add_argument("--dump", help="write the group similarities (same/different) per combination to this .npz")
    a = ap.parse_args()
    truth = json.load(open(TRUTH))
    W, H = truth["w"], truth["h"]
    items = []  # (t, person, c)
    for t, bs in truth["frames"].items():
        for b in bs:
            if b["c"] == "face" and b.get("p") is not None:
                x1, y1, x2, y2 = b["b"][0] * W, b["b"][1] * H, b["b"][2] * W, b["b"][3] * H
                items.append((int(t), b["p"], [(x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * HEAD_PER_FACE]))
    print(f"census: {len(items)} faces, {len({p for _, p, _ in items})} people, "
          f"{len({t for t, _, _ in items})} frames")

    recs, timing = [], {k: 0.0 for k in list(DETECTORS) + list(RECOGNIZERS)}
    cache = {}
    for t, p, c in items:
        img = cache.get(t)
        if img is None:
            img = cache[t] = cv2.imread(os.path.join(FRAMES, f"{t:05d}.jpg"))
        crop, k, x1, y1 = crop_of(img, c)
        r = {"t": t, "p": p, "c": c}
        for det in DETECTORS:
            t0 = time.perf_counter()
            f = measure(det, crop, k, x1, y1, c)
            timing[det] += time.perf_counter() - t0
            r[det] = f
            if f is None:
                continue
            aligned = cv2.warpAffine(crop, reid.similarity(f["kp"].astype(np.float64), reid.ARCFACE_DST.astype(np.float64)),
                                     (112, 112), borderValue=0.0)
            f["aligned"] = aligned
            for rec in RECOGNIZERS:
                t0 = time.perf_counter()
                e = RECOGNIZERS[rec](aligned).astype(np.float64)
                timing[rec] += time.perf_counter() - t0
                f[rec] = e / np.linalg.norm(e)
        recs.append(r)
    n = len(recs)

    print("\nlandmarks (the scan's nearest-face rule and frontal gate: score>=%.1f, |yaw|<=%.2f, px>=%d)"
          % (FA.FRONTAL["score"], FA.FRONTAL["yaw"], FA.FRONTAL["px"]))
    for det in DETECTORS:
        found = [r for r in recs if r[det]]
        fr = [r for r in found if r[det]["frontal"]]
        sc = np.array([r[det]["score"] for r in found])
        print(f"  {det:6} found {len(found):3}/{n} ({len(found) / n:.0%})  frontal {len(fr):3} ({len(fr) / n:.0%})  "
              f"score median {np.median(sc):.2f}  {timing[det] / n * 1000:.1f} ms/face")
    for d in DETECTORS:
        if d != "scrfd" and "scrfd" in DETECTORS:
            print(f"  scrfd and {d}: both found {sum(1 for r in recs if r['scrfd'] and r[d])}; only scrfd "
                  f"{sum(1 for r in recs if r['scrfd'] and not r[d])}, only {d} {sum(1 for r in recs if r[d] and not r['scrfd'])}")
    for rec in RECOGNIZERS:
        print(f"  {rec} {timing[rec] / sum(1 for r in recs for d in DETECTORS if r[d]) * 1000:.1f} ms/face")

    print("\nlikeness (frontal faces only; pairs from different frames)")
    hdr = f"  {'combo':16} {'faces':>5} | {'AUC [90% CI]':>20} {'TAR@1%':>7} | {'grpAUC':>6} {'same@1%':>8} {'diff@1%':>8} | {'rank1':>6}"
    print(hdr)
    rows, dumps = {}, {}
    for det, rec in itertools.product(DETECTORS, RECOGNIZERS):
        ok = (lambda r: all(r[d] and r[d]["frontal"] for d in DETECTORS)) if COMMON else (lambda r: r[det] and r[det]["frontal"])
        fr = sorted([r for r in recs if ok(r)], key=lambda r: (r["p"], r["t"]))
        E = np.array([r[det][rec] for r in fr])
        P = np.array([r["p"] for r in fr])
        T = np.array([r["t"] for r in fr])
        S = E @ E.T
        iu = np.triu_indices(len(fr), 1)
        cross = T[iu[0]] != T[iu[1]]
        same = (P[iu[0]] == P[iu[1]])[cross]
        sims = S[iu][cross]
        pos, neg = sims[same], sims[~same]
        tar, _ = tar_at_far(pos, neg, 0.01)
        rng, ppl, boots = np.random.default_rng(0), np.unique(P), []
        pa, pb = P[iu[0]][cross], P[iu[1]][cross]
        for _ in range(200):  # resample people: a person drawn twice counts twice
            w = dict(zip(*np.unique(rng.choice(ppl, len(ppl)), return_counts=True)))
            wt = np.array([w.get(x, 0) * w.get(y, 0) for x, y in zip(pa, pb)])
            keep = np.repeat(np.arange(len(sims)), wt)
            sp, sn = sims[keep][same[keep]], sims[keep][~same[keep]]
            if len(sp) and len(sn):
                boots.append(auc(sp, sn))
        lo, hi = np.percentile(boots, [5, 95])
        # groups: means of 3 consecutive frontal faces per person (a chain's face, as graph.py)
        G, GP = [], []
        for p in np.unique(P):
            idx = np.where(P == p)[0]
            for i in range(0, len(idx) - 2, 3):
                m = E[idx[i:i + 3]].mean(0)
                G.append(m / np.linalg.norm(m))
                GP.append(p)
        G, GP = np.array(G), np.array(GP)
        gs = G @ G.T
        gi = np.triu_indices(len(G), 1)
        gsame = GP[gi[0]] == GP[gi[1]]
        gpos, gneg = gs[gi][gsame], gs[gi][~gsame]
        gtar, gt_same = tar_at_far(gpos, gneg, 0.01)
        gtnr, gt_diff = tnr_at_frr(gpos, gneg, 0.01)
        # rank-1 over >= 48 frames (2 s) apart
        hits = tot = 0
        for i in range(len(fr)):
            far_ = np.abs(T - T[i]) >= 48
            if far_.any():
                j = np.where(far_)[0][np.argmax(S[i, far_])]
                hits += P[j] == P[i]
                tot += 1
        dumps[f"{det}+{rec}:same"], dumps[f"{det}+{rec}:diff"] = gpos, gneg
        rows[(det, rec)] = dict(n=len(fr), auc=auc(pos, neg), tar=tar, gauc=auc(gpos, gneg), gtar=gtar, gtnr=gtnr,
                                gt_same=gt_same, gt_diff=gt_diff, rank1=hits / max(tot, 1), groups=len(G),
                                app_same=float((gpos >= 0.55).mean()), app_false_same=float((gneg >= 0.55).mean()),
                                app_diff=float((gneg < 0.25).mean()), app_false_diff=float((gpos < 0.25).mean()))
        x = rows[(det, rec)]
        x["ci"] = (lo, hi)
        print(f"  {det + '+' + rec:16} {x['n']:5} | {x['auc']:6.3f} [{lo:.3f}-{hi:.3f}] {x['tar']:7.1%} | {x['gauc']:6.3f} "
              f"{x['gtar']:8.1%} {x['gtnr']:8.1%} | {x['rank1']:6.1%}")
    print("\n  groups: same@1% = same-person groups above the threshold that lets 1% of different-people groups "
          "through;\n          diff@1% = different-people groups below the threshold that 1% of same-person groups fall under")
    for (det, rec), x in rows.items():
        print(f"  {det + '+' + rec:16} {x['groups']} groups; thresholds same {x['gt_same']:.2f}, diff {x['gt_diff']:.2f}"
              + (f"; at ArcFace's former app 0.55/0.25: same {x['app_same']:.0%} (false {x['app_false_same']:.1%}), "
                 f"different {x['app_diff']:.0%} (false {x['app_false_diff']:.1%})" if rec == "arcface" else ""))

    if a.dump:
        np.savez(a.dump, **dumps)
    if a.sheet:  # faces one detector found and the other didn't, with both detectors' landmarks
        dets = list(DETECTORS)
        odd = [r for r in recs if len({bool(r[d]) for d in dets}) > 1
               or len({bool(r[d] and r[d]["frontal"]) for d in dets}) > 1][:60]
        tiles = []
        for r in odd:
            crop, k, x1, y1 = crop_of(cache[r["t"]], r["c"])
            v = cv2.resize(crop, (160, 160))
            s = 160 / crop.shape[0]
            for det, col in [x for x in (("scrfd", (0, 255, 0)), ("yunet", (0, 140, 255)), ("retina", (255, 80, 255))) if x[0] in DETECTORS]:
                f = r[det]
                if f:
                    for x, y in f["kp"]:
                        cv2.circle(v, (int(x * s), int(y * s)), 2, col, -1)
            lab = "".join(d[0].upper() if r[d] and r[d]["frontal"] else (d[0] if r[d] else "-") for d in DETECTORS)
            cv2.putText(v, f"{r['t']} p{r['p']} {lab}", (3, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            tiles.append(v)
        while len(tiles) % 10:
            tiles.append(np.zeros((160, 160, 3), np.uint8))
        if tiles:
            cv2.imwrite(a.sheet, np.vstack([np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]))
            print(f"\nsheet: {a.sheet} ({len(odd)} faces; label: S/Y frontal, s/y found, - missed; green scrfd, orange yunet, magenta retina)")


if __name__ == "__main__":
    main()
