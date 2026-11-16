"""The InsightFace pair the app used until 2026-11-14 (SCRFD-10G landmarks + ArcFace R50 w600k, buffalo_l pack:
non-commercial research licence), kept for comparisons only (experiments/face_model_compare.py and friends).
This is reid.py as it was before the switch to RetinaFace + SFace; the model files stay in models/onnx/ locally
and are no longer packaged.
"""
import os

import cv2
import numpy as np

from mojimask import onnx_core as OC


ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)  # 112x112 alignment template
_models = {}
import threading
_LOCK = threading.Lock()
_slot = {}  # thread -> which copy it uses


ENGINE = "gpu"  # where SCRFD and ArcFace run (onnx_core.ENGINES). On jazz's 2,726 detections the graphics chip gives the
# same faces, scores and embeddings as the CPU, 4.7x faster (72.6 -> 15.4 s for the scan's face stage); the neural
# chip is faster again (6.3 s) but not the same (28 faces change between frontal and not), so it isn't used.


POOL = int(os.environ.get("MOJIMASK_FACE_POOL", 3))  # copies of each model: one copy on the graphics chip runs one call at
# a time however many threads ask (380 calls/s at 160 px with 1 or 8 callers); three copies ran 905/s. Same results.


def models(size=320, engine=None):
    """SCRFD-10G with keypoints (finds the 5 landmarks) at input `size`, + ArcFace R50 (w600k), both from InsightFace's
    buffalo_l pack (non-commercial research licence), run directly on ONNX Runtime. Off the CPU the models need
    fixed input sizes, so there are SCRFD sessions per size used (320: face measuring; 160: the follow check).
    Returns this thread's copy of each (POOL copies, shared out by thread)."""
    with _LOCK:  # background passes load these at the same time
        engine = engine or ENGINE
        if ("det", size, engine) not in _models:
            _models["det", size, engine] = [OC.session(os.path.join(OC.MODELS, "scrfd-10g-kps.onnx"), engine, {"?": size}) for _ in range(POOL)]
        if "rec" not in _models:
            _models["rec"] = [OC.session(os.path.join(OC.MODELS, "arcface-w600k-r50.onnx"), ENGINE, {"None": 1}) for _ in range(POOL)]
        if threading.get_ident() not in _slot:
            _slot[threading.get_ident()] = len(_slot) % POOL
    i = _slot[threading.get_ident()]
    return _models["det", size, engine][i], _models["rec"][i]


def scrfd(img, size=320, thresh=0.3, iou=0.4, engine=None):
    """Faces in img -> (boxes [n,5] x1,y1,x2,y2,score; kps [n,5,2]). Same maths as insightface's SCRFD.detect."""
    det, _ = models(size, engine)  # engine: another of onnx_core.ENGINES for this call's model (the follow check)
    h, w = img.shape[:2]
    if h / w > 1:
        nh, nw = size, int(size / (h / w))
    else:
        nw, nh = size, int(size * h / w)
    scale = nh / h
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
    blob = OC.blob(canvas, 1 / 128.0, 127.5)  # canvas is already size x size
    outs = det.run(None, {det.get_inputs()[0].name: blob})
    B, S, K = [], [], []
    for i, stride in enumerate((8, 16, 32)):
        sc, bb, kp = outs[i][:, 0], outs[i + 3] * stride, outs[i + 6] * stride
        g = size // stride
        ctr = (np.stack(np.mgrid[:g, :g][::-1], -1).astype(np.float32) * stride).reshape(-1, 2)
        ctr = np.repeat(ctr, 2, axis=0)  # 2 anchors per cell
        m = sc >= thresh
        c = ctr[m]
        B.append(np.c_[c - bb[m][:, :2], c + bb[m][:, 2:]])
        K.append(c[:, None, :] + kp[m].reshape(-1, 5, 2))
        S.append(sc[m])
    S = np.concatenate(S)
    if not len(S):
        return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
    B, K = np.concatenate(B) / scale, np.concatenate(K) / scale
    order = np.argsort(-S, kind="stable")
    B, K, S = B[order], K[order], S[order]
    keep, idx = [], np.arange(len(S))  # greedy NMS with insightface's +1 pixel areas
    area = (B[:, 2] - B[:, 0] + 1) * (B[:, 3] - B[:, 1] + 1)
    while len(idx):
        i, rest = idx[0], idx[1:]
        keep.append(i)
        ww = np.maximum(0, np.minimum(B[i, 2], B[rest, 2]) - np.maximum(B[i, 0], B[rest, 0]) + 1)
        hh = np.maximum(0, np.minimum(B[i, 3], B[rest, 3]) - np.maximum(B[i, 1], B[rest, 1]) + 1)
        inter = ww * hh
        idx = rest[inter / (area[i] + area[rest] - inter) <= iou]
    return np.c_[B[keep], S[keep]], K[keep]


def similarity(src, dst):
    """Least-squares similarity transform src -> dst (Umeyama; what skimage's SimilarityTransform does)."""
    ms, md = src.mean(0), dst.mean(0)
    sd, dd = src - ms, dst - md
    U, S, Vt = np.linalg.svd(dd.T @ sd / len(src))
    d = np.array([1.0, -1.0 if np.linalg.det(dd.T @ sd) < 0 else 1.0])
    R = U @ np.diag(d) @ Vt
    k = (S @ d) / sd.var(0).sum()
    return np.c_[k * R, md - k * R @ ms]


def arcface(aligned):
    _, rec = models()
    blob = OC.blob(aligned, 1 / 127.5, 127.5)  # aligned is 112 x 112
    return rec.run(None, {rec.get_inputs()[0].name: blob})[0][0]


def embed_crop(frame, c, max_off=None):
    """ArcFace embedding of the face inside head box c=[cx,cy,s], or None.
    Crop generously, upscale small heads, find landmarks with SCRFD, align, embed.
    max_off: only a face whose centre is within max_off head sizes of the box centre counts -- where two
    heads are close (a hand-over), the nearest face in the crop can be the neighbour's."""
    H, W = frame.shape[:2]
    r = c[2] * 0.9
    x1, y1, x2, y2 = int(c[0] - r), int(c[1] - r), int(c[0] + r), int(c[1] + r)
    pad = np.pad(frame, ((max(-y1, 0), max(y2 - H, 0)), (max(-x1, 0), max(x2 - W, 0)), (0, 0)))
    crop = pad[y1 + max(-y1, 0):y2 + max(-y1, 0), x1 + max(-x1, 0):x2 + max(-x1, 0)]
    if crop.size == 0:
        return None, 0.0
    k = max(1.0, 224 / crop.shape[0])
    crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    boxes, kpss = scrfd(crop)
    if len(boxes) == 0:
        return None, 0.0
    # take the face closest to the crop centre
    ctr = np.array(crop.shape[1::-1]) / 2
    d = [np.linalg.norm((b[:2] + b[2:4]) / 2 - ctr) for b in boxes]
    i = int(np.argmin(d))
    if max_off is not None and d[i] / k > max_off * c[2]:
        return None, 0.0
    aligned = cv2.warpAffine(crop, similarity(kpss[i].astype(np.float64), ARCFACE_DST.astype(np.float64)),
                             (112, 112), borderValue=0.0)
    e = arcface(aligned).flatten()
    return e / np.linalg.norm(e), float(boxes[i][4])
