"""Body appearance for re-joining people across long gaps, when faces don't show clearly.

Person re-ID (OSNet, at the end: crop, body_box, reid_embed), used by the identity questions
(scan/questions.py) and the review server. On ONNX Runtime (onnx_core) -- no PyTorch. The DINOv2-small
embedder it was compared against lives on in experiments/dinov2_appearance.py, for the finished experiments only.
"""
import os
import threading

import cv2
import numpy as np

from mojimask import onnx_core as OC

_M = {}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # ImageNet normalisation (reid_embed)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def crop(frame, box, pad=0.0):
    """Crop [x1,y1,x2,y2] (+pad fraction) with black fill outside the frame."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, y1, x2, y2 = int(x1 - pad * w), int(y1 - pad * h), int(x2 + pad * w), int(y2 + pad * h)
    H, W = frame.shape[:2]
    P = max(0, -x1, -y1, x2 - W, y2 - H)
    f = np.pad(frame, ((P, P), (P, P), (0, 0))) if P else frame
    c = f[y1 + P:y2 + P, x1 + P:x2 + P]
    return c if c.size else None


# ---- person re-identification (the body measurement the identity graph uses)
# OSNet-AIN x1.0 trained on MSMT17 (torchreid, MIT code; Kaiyang Zhou et al.), exported to ONNX:
# models/onnx/osnet-ain-x1_0-msmt17.onnx, 256x128 RGB in, 512-d out. Made to match the same person across
# cameras by clothing and build; the "AIN" variant normalises away lighting. On the test clip's split-track
# check it ranks a person's other half first 48% of the time and
# in the top five 75% (DINOv2 on the head + upper body: 34% / 56%), on the WHOLE body as the pose skeleton frames it.
_RLOCK = threading.Lock()


def _reid_session():
    with _RLOCK:  # background passes load it at the same time
        if "reid" not in _M:
            _M["reid"] = OC.session(os.path.join(OC.MODELS, "osnet-ain-x1_0-msmt17.onnx"), "cpu")
    return _M["reid"]


def body_box(c, pose=None):
    """Whole-body box for a head c=[cx, cy, s]: the detection box of the person's own pose skeleton when
    there is one, else a head-relative guess (shoulders ~3 heads wide, body ~6 heads tall)."""
    if pose is not None:
        return [float(v) for v in pose[:4]]
    return [c[0] - 1.6 * c[2], c[1] - 0.8 * c[2], c[0] + 1.6 * c[2], c[1] + 5.5 * c[2]]


def reid_embed(crops):
    """crops: whole-body BGR images (unmasked) -> (n, 512) L2-normalised re-ID features."""
    if not crops:
        return np.zeros((0, 512), np.float32)
    xs = []
    for c in crops:
        rgb = cv2.cvtColor(cv2.resize(c, (128, 256), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        xs.append(((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))
    s = _reid_session()
    v = s.run(None, {"images": np.stack(xs).astype(np.float32)})[0]
    return v / np.linalg.norm(v, axis=1, keepdims=True)

