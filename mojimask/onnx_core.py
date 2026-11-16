"""Torch-free detection core: the three core models on ONNX Runtime only (numpy + opencv + onnxruntime).

    from mojimask.onnx_core import Core
    core = Core()                          # default engine plan; Core(plan="cpu") for any machine
    found = core(frame_bgr)                # {"yolov5m_face_tiled": [...], "retinaface_r50_2x": [...], "pose_full": [...]}

Output formats match experiments/detect.py exactly, so tracking / review / rendering are unchanged:
  face detectors: [x1, y1, x2, y2, score] in source pixels
  pose_full:      [x1, y1, x2, y2, score, k0x, k0y, k0c, ..., k16x, k16y, k16c] (COCO-17)

Working canvas: every frame is resized so its long side is CANVAS (1280) px before the face models see it,
so the cost per frame is the same for a 360p clip and a 4K one (the 360x640 test clip is upscaled 2x, exactly
what experiments/detect.py did). Model input shapes are pinned to that canvas when a session opens; CoreML needs static
shapes (RetinaFace with a free input size gives wrong boxes on the GPU path and is 3x slower).

Engines (Apple; other platforms use "cpu" until DirectML/CUDA are wired in):
  "gpu"  CoreML ML Program, CPU+GPU, fp32       fastest on big GPUs (M-series Pro/Max/Ultra)
  "ane"  CoreML NeuralNetwork, CPU+Neural Engine (fp16)   about the same speed on every M-series chip
  "cpu"  plain ONNX Runtime CPU
The models run concurrently (ONNX Runtime releases the GIL), so a plan that splits them across GPU and
Neural Engine uses both at once.

Scan models (models/onnx/):
  yolov5m-face.onnx     YOLOv5-Face-M (deepcam-cn, GPL-3.0; ONNX export from yakhyo/uniface), fixed 640x640
  retinaface-r50.onnx   RetinaFace ResNet-50 (biubug6 / yakhyo, MIT), free input size
  yolo11x-pose.onnx     YOLO11x-pose (Ultralytics, AGPL-3.0), free input size, run at 1280 rect
  yolo11m-pose.onnx     YOLO11m-pose, same (faster, less accurate candidate)
session() also opens the app's other models (reid.py, appearance.py, depth.py): every file in models/onnx/, what
uses it and its licence are listed in models/README.md.
Pre/post-processing follows uniface (face models) and Ultralytics (pose) so results match the PyTorch path.
"""
import itertools
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import onnxruntime as ort

from mojimask.paths import ROOT as HERE
MODELS = os.path.join(HERE, "models", "onnx")
CANVAS = 1280
ENGINES = {
    "gpu": {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU"},
    "ane": {"ModelFormat": "NeuralNetwork", "MLComputeUnits": "CPUAndNeuralEngine"},
    "all": {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"},  # no plan uses it; kept for experiments
    "cpu": None,
}
PLANS = {  # model -> engine
    "default": {"yolov5m_face_tiled": "gpu", "retinaface_r50_2x": "ane", "pose_full": "gpu"},
    "gpu": {"yolov5m_face_tiled": "gpu", "retinaface_r50_2x": "gpu", "pose_full": "gpu"},
    "ane": {"yolov5m_face_tiled": "ane", "retinaface_r50_2x": "ane", "pose_full": "ane"},
    "cpu": {"yolov5m_face_tiled": "cpu", "retinaface_r50_2x": "cpu", "pose_full": "cpu"},
}
PRUNE_DAYS = 14  # CoreML compiles not used for this long are deleted (_coreml_cache)


def session(path, engine="gpu", dims=None):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    for name, value in (dims or {}).items():  # pin free input dims (CoreML wants static shapes)
        so.add_free_dimension_override_by_name(name, int(value))
    opts = ENGINES.get(engine)
    if not (opts and "CoreMLExecutionProvider" in ort.get_available_providers()):
        return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    cache = _coreml_cache(path, opts, dims)
    try:
        return ort.InferenceSession(path, sess_options=so, providers=[
            ("CoreMLExecutionProvider", {**opts, "ModelCacheDirectory": cache}), "CPUExecutionProvider"])
    except Exception as e:  # a damaged cache must never stop the app: compile afresh, as before caching
        import shutil
        print(f"onnx_core: {os.path.basename(path)} would not load with its CoreML cache ({e!r}); compiling afresh",
              file=sys.stderr, flush=True)
        shutil.rmtree(cache, ignore_errors=True)
        return ort.InferenceSession(path, sess_options=so,
                                    providers=[("CoreMLExecutionProvider", dict(opts)), "CPUExecutionProvider"])


def _coreml_cache(path, opts, dims):
    """Folder for CoreML's compiled copy of a model. Compiling takes ~25 s for DINOv2 (every time the
    review server started); a cached copy loads in ~6 s. Keyed by the model file (name, size, date), its
    pinned input sizes and the engine, so a changed model never loads a stale compile. Regenerable:
    ~/Library/Caches/org.mojimask.app (override with MOJIMASK_CACHE). No spaces in this path: ONNX Runtime's
    CoreML cache fails on them ("Failed to create model URL from path").

    The model is identified by its size and its first and last MiB, not its file date: every build copies
    the models into the app with new dates, and keying on those left a fresh ~450 MB compile per model per
    build. Compiles not used for PRUNE_DAYS are deleted (each use marks its folder)."""
    import hashlib
    import shutil
    import time
    st = os.stat(path)
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        f.seek(max(0, st.st_size - (1 << 20)))
        h.update(f.read())
    key = hashlib.md5(json.dumps([os.path.basename(path), st.st_size, h.hexdigest(), sorted((dims or {}).items()),
                                  sorted(opts.items()), ort.__version__]).encode()).hexdigest()[:16]
    new = os.path.expanduser("~/Library/Caches/org.mojimask.app")  # = the app's bundle id (packaging/build_mac.sh)
    old = os.path.expanduser("~/Library/Caches/org.partyanonymous.app")  # the app's name until 2026-11-14
    if not os.environ.get("MOJIMASK_CACHE") and os.path.isdir(old) and not os.path.exists(new):
        try:
            os.rename(old, new)  # keep the compiled models (a move)
        except OSError:
            pass
    base = os.path.join(os.environ.get("MOJIMASK_CACHE") or new, "coreml")
    d = os.path.join(base, f"{os.path.splitext(os.path.basename(path))[0]}-{key}")
    os.makedirs(d, exist_ok=True)
    os.utime(d)
    old = time.time() - PRUNE_DAYS * 86400
    for name in os.listdir(base):
        p = os.path.join(base, name)
        if p != d and os.path.isdir(p) and os.path.getmtime(p) < old:
            shutil.rmtree(p, ignore_errors=True)
    return d


def nms(boxes, scores, iou):
    """Indices of boxes [n,4+] (x1,y1,x2,y2) kept by greedy non-maximum suppression, best score first. Same result
    as cv2.dnn.NMSBoxes(score_threshold=0) (checked on 3,000 random cases), so OpenCV's dnn module isn't needed."""
    if not len(boxes):
        return np.array([], int)
    b = np.asarray(boxes, np.float64)[:, :4]
    s = np.asarray(scores, np.float64)
    order = [i for i in np.argsort(-s, kind="stable") if s[i] > 0.0]
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    keep = []
    for i in order:
        if keep:
            k = np.array(keep)
            iw = np.minimum(b[i, 2], b[k, 2]) - np.maximum(b[i, 0], b[k, 0])
            ih = np.minimum(b[i, 3], b[k, 3]) - np.maximum(b[i, 1], b[k, 1])
            inter = np.where((iw > 0) & (ih > 0), iw * ih, 0.0)
            if (inter / (area[i] + area[k] - inter) > iou).any():
                continue
        keep.append(i)
    return np.array(keep, int)


def blob(img, scale, mean):
    """BGR uint8 image -> 1x3xHxW float32 RGB tensor of (pixel - mean) * scale (= cv2.dnn.blobFromImage, swapRB)."""
    return ((img[:, :, ::-1].astype(np.float32) - np.float32(mean)) * np.float32(scale)).transpose(2, 0, 1)[None].copy()


def to_canvas(frame):
    """Resize so the long side is CANVAS px. Returns (canvas image, scale k = canvas px per source px)."""
    h, w = frame.shape[:2]
    k = CANVAS / max(h, w)
    if abs(k - 1) < 1e-6:
        return frame, 1.0
    interp = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    return cv2.resize(frame, (round(w * k), round(h * k)), interpolation=interp), k


def letterbox(img, size, fill=114, rect=False, stride=32):
    """Resize to fit size (keep aspect), pad centred. rect=False: pad to size x size (uniface);
    rect=True: pad only to a multiple of stride (Ultralytics). Returns NCHW RGB float, scale, (pad_w, pad_h)."""
    h, w = img.shape[:2]
    s = min(size / h, size / w)
    if rect:
        nh, nw = round(h * s), round(w * s)
        H, W = nh + (size - nh) % stride, nw + (size - nw) % stride
        ph, pw = round((H - nh) / 2 - 0.1), round((W - nw) / 2 - 0.1)
    else:
        nh, nw = int(h * s), int(w * s)
        H = W = size
        ph, pw = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((H, W, 3), fill, np.uint8)
    canvas[ph:ph + nh, pw:pw + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(x), s, (pw, ph)


class _Shaped:
    """Holds one ONNX session per input shape (a video has one shape, so normally just one)."""

    def __init__(self, path, engine):
        self.path, self.engine, self._s = path, engine, {}

    def run(self, x, dims=None):
        key = tuple(x.shape)
        if key not in self._s:
            self._s[key] = session(self.path, self.engine, dims)
        s = self._s[key]
        return s.run(None, {s.get_inputs()[0].name: x})


# ------------------------------------------------------------------ YOLOv5-Face
class YoloFace:
    def __init__(self, engine="gpu", conf=0.05, iou=0.4):
        self.m = _Shaped(os.path.join(MODELS, "yolov5m-face.onnx"), engine)
        self.conf, self.iou = conf, iou

    def detect(self, img):
        x, s, (pw, ph) = letterbox(img, 640)
        p = self.m.run(x)[0][0]  # (25200, 16): cx, cy, w, h, obj, 10 landmarks, cls
        p = p[p[:, 4] >= self.conf]
        if not len(p):
            return []
        b = np.c_[p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2, p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2]
        k = nms(b, p[:, 4], self.iou)
        b, sc = b[k], p[k, 4]
        b[:, [0, 2]] = (b[:, [0, 2]] - pw) / s
        b[:, [1, 3]] = (b[:, [1, 3]] - ph) / s
        return [[*map(float, bb), float(c)] for bb, c in zip(b, sc)]

    def __call__(self, frame, big=None, k=None):
        """Whole frame (catches big faces) + overlapping 640 tiles over the working canvas (same as experiments/detect.py)."""
        if big is None:
            big, k = to_canvas(frame)
        H, W = big.shape[:2]
        out = self.detect(frame)
        T = 640
        ys = sorted({0, *range(0, max(H - T, 0), T // 2), max(H - T, 0)})
        xs = sorted({0, *range(0, max(W - T, 0), T // 2), max(W - T, 0)})
        for y in ys:
            for x in xs:
                out += [[(b[0] + x) / k, (b[1] + y) / k, (b[2] + x) / k, (b[3] + y) / k, b[4]]
                        for b in self.detect(big[y:y + T, x:x + T])]
        if not out:
            return []
        a = np.array(out)
        return [out[i] for i in nms(a[:, :4], a[:, 4], 0.5)]  # same merge threshold as experiments/detect.py


# ------------------------------------------------------------------ RetinaFace-R50
def anchors(h, w):
    steps, min_sizes = [8, 16, 32], [[16, 32], [64, 128], [256, 512]]
    out = []
    for step, ms in zip(steps, min_sizes):
        fh, fw = math.ceil(h / step), math.ceil(w / step)
        for i, j in itertools.product(range(fh), range(fw)):
            for m in ms:
                out += [(j + 0.5) * step / w, (i + 0.5) * step / h, m / w, m / h]
    return np.array(out, np.float32).reshape(-1, 4)


def retina_input(img):
    """BGR uint8 image -> RetinaFace-R50's input: 1x3xHxW float32, BGR minus (104, 117, 123) (biubug6)."""
    return np.ascontiguousarray((img.astype(np.float32) - np.array([104, 117, 123], np.float32)).transpose(2, 0, 1)[None])


def retina_decode(loc, pri, W, H, lm=None):
    """RetinaFace outputs for some priors -> boxes [n,4] x1,y1,x2,y2 in a W x H image (variances 0.1 / 0.2, biubug6);
    with lm (its landmark head) also kps [n,5,2]. Thresholds, caps and NMS are the caller's."""
    c = pri[:, :2] + loc[:, :2] * 0.1 * pri[:, 2:]
    wh = pri[:, 2:] * np.exp(loc[:, 2:] * 0.2)
    b = np.c_[c - wh / 2, c + wh / 2] * np.array([W, H, W, H], np.float32)  # priors are fractions: to px
    if lm is None:
        return b
    return b, (pri[:, None, :2] + lm.reshape(-1, 5, 2) * 0.1 * pri[:, None, 2:]) * np.array([W, H], np.float32)


class RetinaFace:
    def __init__(self, engine="ane", conf=0.05, iou=0.4):
        self.m = _Shaped(os.path.join(MODELS, "retinaface-r50.onnx"), engine)
        self.conf, self.iou = conf, iou
        self._pri = {}

    def __call__(self, frame, big=None, k=None):
        if big is None:
            big, k = to_canvas(frame)
        H, W = big.shape[:2]
        loc, conf, _ = self.m.run(retina_input(big), {"batch_size": 1, "height": H, "width": W})
        pri = self._pri.setdefault((H, W), anchors(H, W))
        loc, sc = loc[0], conf[0][:, 1]
        m = sc > self.conf
        loc, sc, pr = loc[m], sc[m], pri[m]
        if not len(sc):
            return []
        b = retina_decode(loc, pr, W, H)
        order = sc.argsort()[::-1][:5000]
        b, sc = b[order], sc[order]
        keep = nms(b, sc, self.iou)[:750]
        return [[*map(float, bb / k), float(s)] for bb, s in zip(b[keep], sc[keep])]

    def best_face(self, img_bgr, min_score=0.5):
        """(box [x1,y1,x2,y2], 5 face points [5,2]) of the most confident face in a BGR image, in its pixels, or
        None below min_score. One face, no NMS (portrait.py: the chosen photo's own face)."""
        big, k = to_canvas(img_bgr)
        H, W = big.shape[:2]
        loc, conf, lm = self.m.run(retina_input(big), {"batch_size": 1, "height": H, "width": W})
        pri = self._pri.setdefault((H, W), anchors(H, W))
        sc = conf[0][:, 1]
        i = int(sc.argmax())
        if sc[i] < min_score:
            return None
        b, pts = retina_decode(loc[0][i:i + 1], pri[i:i + 1], W, H, lm[0][i:i + 1])
        return b[0] / k, pts[0] / k


# ------------------------------------------------------------------ YOLO11-pose
class Pose:
    def __init__(self, engine="gpu", name="yolo11x-pose", imgsz=1280, conf=0.1, iou=0.5):  # Ultralytics predict default IoU
        self.m = _Shaped(os.path.join(MODELS, f"{name}.onnx"), engine)
        self.imgsz, self.conf, self.iou = imgsz, conf, iou

    def __call__(self, frame, big=None, k=None):
        x, s, (pw, ph) = letterbox(frame, self.imgsz, rect=True)  # Ultralytics rect inference
        p = self.m.run(x, {"batch": 1, "height": x.shape[2], "width": x.shape[3]})[0][0].T  # (A, 56)
        p = p[p[:, 4] >= self.conf]
        if not len(p):
            return []
        b = np.c_[p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2, p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2]
        keep = nms(b, p[:, 4], self.iou)
        b, p = b[keep], p[keep]
        b[:, [0, 2]] = (b[:, [0, 2]] - pw) / s
        b[:, [1, 3]] = (b[:, [1, 3]] - ph) / s
        kp = p[:, 5:].reshape(-1, 17, 3).copy()
        kp[:, :, 0] = (kp[:, :, 0] - pw) / s
        kp[:, :, 1] = (kp[:, :, 1] - ph) / s
        return [[*map(float, bb), float(c), *[round(float(v), 2) for v in kk.ravel()]]
                for bb, c, kk in zip(b, p[:, 4], kp)]


def default_plan():
    if "CoreMLExecutionProvider" not in ort.get_available_providers():
        return "cpu"
    return os.environ.get("MOJIMASK_PLAN", "default")


class Core:
    """The default 'core' profile: YOLOv5-Face (tiled) + RetinaFace-R50 + pose (full keypoints).
    plan: a PLANS name or a {model: engine} dict."""

    def __init__(self, plan=None, pose="yolo11x-pose", parallel=True):
        plan = PLANS[plan or default_plan()] if not isinstance(plan, dict) else plan
        self.plan = plan
        self.models = {"yolov5m_face_tiled": YoloFace(plan["yolov5m_face_tiled"]),
                       "retinaface_r50_2x": RetinaFace(plan["retinaface_r50_2x"]),
                       "pose_full": Pose(plan["pose_full"], pose)}
        self.pool = ThreadPoolExecutor(len(self.models)) if parallel else None

    def __call__(self, frame):
        big, k = to_canvas(frame)
        if self.pool is None:
            return {name: m(frame, big, k) for name, m in self.models.items()}
        futs = {name: self.pool.submit(m, frame, big, k) for name, m in self.models.items()}
        return {name: f.result() for name, f in futs.items()}


def machine_model():
    import subprocess
    try:
        return subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True, text=True).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def fastest_core(sample_frames, cache=None, plans=("default", "ane")):
    """Time each engine plan on a few real frames and keep the fastest. Big GPUs (Pro/Max/Ultra) favour the
    GPU-heavy default plan; MacBook Air-class chips have a small GPU but the same Neural Engine, so the
    Neural Engine plan should win there. The choice is remembered per Mac model (and ONNX Runtime version and
    plan table) in `cache` (JSON). Too few frames to time (a photo): the remembered plan, else the default,
    and nothing is remembered. Returns (Core, info)."""
    import hashlib
    import time
    if "CoreMLExecutionProvider" not in ort.get_available_providers():
        return Core("cpu"), {"plan": "cpu"}
    key = {"machine": machine_model(), "ort": ort.__version__,
           "plans": hashlib.md5(json.dumps(PLANS, sort_keys=True).encode()).hexdigest()[:12]}
    if cache and os.path.exists(cache):
        try:
            with open(cache) as f:
                c = json.load(f)
            if all(c.get(k) == v for k, v in key.items()) and c.get("plan") in PLANS:
                return Core(c["plan"]), {**c, "cached": True}
        except (OSError, ValueError, AttributeError):
            pass
    if len(sample_frames) < 3:  # one load frame + at least two timed ones, or the timing means nothing
        plan = default_plan()
        return Core(plan), {"plan": plan, "measured": False}
    best, fps = None, {}
    for plan in plans:
        core = Core(plan)
        core(sample_frames[0])  # load + compile
        t0 = time.perf_counter()
        for f in sample_frames[1:]:
            core(f)
        dt = time.perf_counter() - t0
        fps[plan] = round((len(sample_frames) - 1) / dt, 2) if dt > 0 else 0.0
        if best is None or fps[plan] > fps[best[1]]:
            best = (core, plan)
    info = {"plan": best[1], "fps": fps, **key}
    if cache:
        try:
            tmp = cache + ".tmp"
            with open(tmp, "w") as f:
                json.dump(info, f)
            os.replace(tmp, cache)
        except OSError:
            pass
    return best[0], info
