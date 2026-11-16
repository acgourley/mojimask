"""Faces for comparing people: RetinaFace finds a face and its 5 landmarks, SFace embeds the aligned face (the
scan's face check and face likeness, scan/faces.py, and its follow check, follow.py, use these).

Both are permissively licensed: RetinaFace-R50 is the app's own face detector (biubug6/Pytorch_Retinaface, MIT;
the scan's detector and this share onnx_core.retina_decode; this decodes its landmark head too), SFace is OpenCV Zoo's
face_recognition_sface_2021dec (Apache-2.0). They replaced InsightFace's SCRFD + ArcFace (non-commercial only) on
2026-11-14, measured: a better
face check on every clip tried, the same safety on the decisions bench, ~10% more questions. The old pair is kept
for comparisons in experiments/insightface_models.py.
"""
import itertools
import os
import threading

import cv2
import numpy as np

from mojimask import onnx_core as OC


ALIGN_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                      [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)  # 112x112 alignment template (SFace's too)

LANDMARK_SIZE = 192  # RetinaFace input: crops are resized to this square (CoreML needs one fixed shape)
LANDMARK_ENGINE = "ane"  # 3.4 ms a face at 192 on the neural chip (CPU: ~100-240 ms); measured on this engine
ENGINE = "gpu"  # where SFace runs (onnx_core.ENGINES)
POOL = int(os.environ.get("MOJIMASK_FACE_POOL", 3))  # copies of each model: one copy runs one call at a time however
# many threads ask; the scan measures faces in threads, so they're shared out by thread

_models = {}
_LOCK = threading.Lock()
_slot = threading.local()  # .i: which copy this thread uses, handed out in turn
_next = itertools.count()


def _copies(key, make):
    with _LOCK:  # background passes load these at the same time
        if key not in _models:
            _models[key] = [make() for _ in range(POOL)]
        if not hasattr(_slot, "i"):
            _slot.i = next(_next) % POOL
    return _models[key][_slot.i]


def landmarks(img, size=LANDMARK_SIZE, thresh=0.3, iou=0.4, engine=None):
    """Faces in img (BGR) -> (boxes [n,5] x1,y1,x2,y2,score; kps [n,5,2]: eyes, nose, mouth corners, image left first).
    RetinaFace-R50 on img resized to size x size: BGR minus (104, 117, 123), priors as onnx_core.anchors, variances
    0.1 / 0.2 (biubug6). Its scores run higher than SCRFD's did (real faces ~0.99): the scan's thresholds were kept
    as they were, which passes more faces (safer; see the module docstring)."""
    engine = engine or LANDMARK_ENGINE
    det = _copies(("det", size, engine), lambda: OC.session(os.path.join(OC.MODELS, "retinaface-r50.onnx"), engine,
                                                            {"batch_size": 1, "height": size, "width": size}))
    H, W = img.shape[:2]
    sq = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA if max(H, W) > size else cv2.INTER_LINEAR)
    loc, conf, lm = det.run(None, {det.get_inputs()[0].name: OC.retina_input(sq)})
    pri = _priors(size)
    loc, sc, lm = loc[0], conf[0][:, 1], lm[0]
    m = sc >= thresh
    loc, sc, lm, pr = loc[m], sc[m], lm[m], pri[m]
    if not len(sc):
        return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
    b, k = OC.retina_decode(loc, pr, W, H, lm)
    b = np.c_[b, sc]
    keep = OC.nms(b[:, :4], sc, iou)
    return b[keep], k[keep]


_PRIORS = {}


def _priors(size):
    if size not in _PRIORS:
        _PRIORS[size] = OC.anchors(size, size)
    return _PRIORS[size]


def similarity(src, dst):
    """Least-squares similarity transform src -> dst (Umeyama; what skimage's SimilarityTransform does)."""
    ms, md = src.mean(0), dst.mean(0)
    sd, dd = src - ms, dst - md
    U, S, Vt = np.linalg.svd(dd.T @ sd / len(src))
    d = np.array([1.0, -1.0 if np.linalg.det(dd.T @ sd) < 0 else 1.0])
    R = U @ np.diag(d) @ Vt
    k = (S @ d) / sd.var(0).sum()
    return np.c_[k * R, md - k * R @ ms]


def align(img, kps):
    """The 112x112 face aligned from its 5 landmarks."""
    return cv2.warpAffine(img, similarity(np.asarray(kps, np.float64), ALIGN_DST.astype(np.float64)), (112, 112),
                          borderValue=0.0)


def embed(aligned):
    """112x112 aligned face (BGR) -> 128-d SFace embedding (not normalised), as OpenCV's FaceRecognizerSF: RGB, 0-255."""
    rec = _copies("rec", lambda: OC.session(os.path.join(OC.MODELS, "sface-2021dec.onnx"), ENGINE))
    blob = np.ascontiguousarray(aligned[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None])
    return rec.run(None, {rec.get_inputs()[0].name: blob})[0][0]


def face_crop(img, c):
    """(crop, k, x0, y0): the picture around head box c=[cx,cy,s], 0.9 head sizes each way, padded black past the
    edges and scaled up by k to at least 224 px tall; x0, y0 its top left in the picture. None if nothing is left.
    (landmarks() resizes it again to LANDMARK_SIZE: the 224 is older than that and the face check was measured with
    it, so it stays until a benchmark says otherwise.)"""
    H, W = img.shape[:2]
    r = c[2] * 0.9
    x1, y1, x2, y2 = int(c[0] - r), int(c[1] - r), int(c[0] + r), int(c[1] + r)
    pad = np.pad(img, ((max(-y1, 0), max(y2 - H, 0)), (max(-x1, 0), max(x2 - W, 0)), (0, 0)))
    crop = pad[y1 + max(-y1, 0):y2 + max(-y1, 0), x1 + max(-x1, 0):x2 + max(-x1, 0)]
    if crop.size == 0:
        return None
    k = max(1.0, 224 / crop.shape[0])
    return cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC), k, x1, y1
