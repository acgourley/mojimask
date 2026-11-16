"""Run face/head detectors on every frame; write one JSON per detector.

Usage: python detect.py VIDEO OUTDIR [detector ...]
Output: OUTDIR/<detector>.json = {"w","h","fps","frames": [[[x1,y1,x2,y2,score], ...], ...]}
Boxes are in source-pixel coords. Thresholds are deliberately very low; filtering happens later.

Finished research (the app runs its detectors through ONNX Runtime, onnx_core.py). Weights under models/ must already
be there (models/README.md); the script stops rather than let ultralytics fetch a missing one. But scrfd10g_2x
(insightface's buffalo_l), retinaface_r50_2x and yolov5m_face_tiled (uniface) download their own weights on first
use: run those only with the go-ahead for that download. DETECT_DEVICE picks the torch device (default mps).
"""
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = os.environ.get("DETECT_DEVICE", "mps")


def weights(rel):
    """A file or folder under the repo's models/, which must already exist: never fetched."""
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        raise SystemExit(f"Missing {rel}: see models/README.md")
    return path


# ---------------------------------------------------------------- detectors
def make_yolo_face(path):
    from ultralytics import YOLO
    m = YOLO(weights(path))

    def run(frame):
        r = m.predict(frame, imgsz=1280, conf=0.05, device=DEV, verbose=False)[0]
        return [b + [s] for b, s in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist())]
    return run


def make_scrfd():
    import insightface
    fa = insightface.app.FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                                      providers=["CPUExecutionProvider"])
    fa.prepare(ctx_id=-1, det_size=(1280, 1280), det_thresh=0.1)
    det = fa.det_model

    def run(frame):
        big = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        b, _ = det.detect(big, input_size=(1280, 1280))
        return [[*(np.array(d[:4]) / 2).tolist(), float(d[4])] for d in b]
    return run


def _faces(fs, scale=1.0, dx=0, dy=0):
    return [[float(f.bbox[0]) / scale + dx, float(f.bbox[1]) / scale + dy,
             float(f.bbox[2]) / scale + dx, float(f.bbox[3]) / scale + dy, float(f.confidence)] for f in fs]


def make_retinaface():
    from uniface.detection import RetinaFace
    from uniface.constants import RetinaFaceWeights
    det = RetinaFace(model_name=RetinaFaceWeights.RESNET50, confidence_threshold=0.05,
                     nms_threshold=0.4, dynamic_size=True)

    def run(frame):
        big = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        return _faces(det.detect(big), scale=2)
    return run


def nms(boxes, iou=0.5):
    if not boxes:
        return boxes
    b = np.array(boxes)
    keep = cv2.dnn.NMSBoxes(np.c_[b[:, :2], b[:, 2:4] - b[:, :2]].tolist(), b[:, 4].tolist(), 0.0, iou)
    return [boxes[i] for i in np.array(keep).flatten()]


def make_yolov5face():
    """YOLOv5-Face-M has a fixed 640 input, so upscale 2x and run overlapping 640 tiles."""
    from uniface.detection import YOLOv5Face
    from uniface.constants import YOLOv5FaceWeights
    det = YOLOv5Face(model_name=YOLOv5FaceWeights.YOLOV5M, confidence_threshold=0.05, nms_threshold=0.4)

    def run(frame):
        big = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        H, W = big.shape[:2]
        out = _faces(det.detect(frame))  # whole frame at native scale catches big faces
        T = 640
        ys = sorted({0, *range(0, max(H - T, 0), T // 2), max(H - T, 0)})
        xs = sorted({0, *range(0, max(W - T, 0), T // 2), max(W - T, 0)})
        for y in ys:
            for x in xs:
                tile = big[y:y + T, x:x + T]
                out += _faces(det.detect(tile), scale=2, dx=x / 2, dy=y / 2)
        return nms(out)
    return run


def make_apple_vision():
    import Vision
    from Foundation import NSData

    def run(frame):
        h, w = frame.shape[:2]
        ok, buf = cv2.imencode(".png", frame)
        data = NSData.dataWithBytes_length_(buf.tobytes(), len(buf))
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
        req = Vision.VNDetectFaceRectanglesRequest.alloc().init()
        handler.performRequests_error_([req], None)
        out = []
        for o in req.results() or []:
            bb = o.boundingBox()  # normalized, origin bottom-left
            x, y, bw, bh = bb.origin.x, bb.origin.y, bb.size.width, bb.size.height
            out.append([x * w, (1 - y - bh) * h, (x + bw) * w, (1 - y) * h, float(o.confidence())])
        return out
    return run


sys.path.insert(0, ROOT)  # the app's modules (one folder up)
from mojimask.pose_util import head_from_pose  # noqa: E402  (shared with the torch-free engine)


def make_pose_head():
    from ultralytics import YOLO
    m = YOLO(weights("models/yolo11x-pose.pt"))

    def run(frame):
        r = m.predict(frame, imgsz=1280, conf=0.1, device=DEV, verbose=False)[0]
        out = []
        if r.keypoints is not None:
            for box, conf, kp in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                     r.keypoints.data.cpu().numpy()):
                h = head_from_pose(kp, box)
                if h:
                    h[4] *= float(conf) ** 0.5
                    out.append(h)
        return out
    return run


def make_egoblur():
    """Meta EgoBlur Gen2 face model (detectron2 Faster R-CNN, TorchScript) via Meta's predictor code.
    Uses Meta's resize (short edge 1200, long edge capped 1200) and a low score threshold."""
    import contextlib
    import torch
    sys.path.insert(0, weights("models/egoblur/pkg"))
    from gen2.script.detectron2.export.torchscript_patch import patch_instances
    from gen2.script.predictor import ClassID, EgoblurDetector, PATCH_INSTANCES_FIELDS
    stack = contextlib.ExitStack()
    stack.enter_context(patch_instances(fields=PATCH_INSTANCES_FIELDS))  # must stay active while running
    det = EgoblurDetector(weights("models/egoblur/ego_blur_face_gen2.jit"), DEV, ClassID.FACE, 0.05, 0.3,
                          resize_aug={"min_size_test": 1200, "max_size_test": 1200}, use_gpu_resize=True)
    stack.pop_all()  # never exit the patch: its cleanup raises a KeyError at interpreter shutdown

    def run(frame):
        x = torch.from_numpy(frame.transpose(2, 0, 1).copy()).to(DEV).unsqueeze(0)  # BGR uint8
        batch, orig_hw, in_hw = det.pre_process(x)
        preds = det.inference(batch)
        boxes, scores = det._post_process(preds, model_input_hw_list=in_hw, target_img_hw_list=orig_hw)
        return [[*map(float, b), float(s)] for b, s in zip(boxes[0], scores[0])]
    return run


def make_sam3(prompt="face"):
    """Meta SAM 3 (Transformers), open-vocabulary text prompt -> instance boxes (masks dropped here)."""
    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor
    proc = Sam3Processor.from_pretrained(weights("models/sam3"))
    model = Sam3Model.from_pretrained(weights("models/sam3")).to(DEV).eval()

    def run(frame):
        inputs = proc(images=Image.fromarray(frame[:, :, ::-1]), text=prompt, return_tensors="pt").to(DEV)
        with torch.no_grad():
            out = model(**inputs)
        res = proc.post_process_instance_segmentation(
            out, threshold=0.1, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist())[0]
        return [[*map(float, b), float(s)] for b, s in zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy())]
    return run


def make_pose_full(path="models/yolo11x-pose.pt", imgsz=1280):
    """Whole-body pose: one entry per person = [x1,y1,x2,y2,score, k0x,k0y,k0c, ..., k16x,k16y,k16c] (COCO-17)."""
    from ultralytics import YOLO
    m = YOLO(weights(path))

    def run(frame):
        r = m.predict(frame, imgsz=imgsz, conf=0.1, device=DEV, verbose=False)[0]
        out = []
        if r.keypoints is not None:
            for box, conf, kp in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                     r.keypoints.data.cpu().numpy()):
                out.append([*map(float, box), float(conf), *[round(float(v), 2) for v in kp.ravel()]])
        return out
    return run


DETECTORS = {
    "pose_full": make_pose_full,
    "sam3_face": make_sam3,
    "egoblur_gen2": make_egoblur,
    "yolo11l_face": lambda: make_yolo_face("models/yolov11l-face.pt"),
    "yolo12l_face": lambda: make_yolo_face("models/yolov12l-face.pt"),
    "scrfd10g_2x": make_scrfd,
    "retinaface_r50_2x": make_retinaface,
    "yolov5m_face_tiled": make_yolov5face,
    "apple_vision": make_apple_vision,
    "pose_head": make_pose_head,
}


def main():
    src, outdir, *names = sys.argv[1:]
    names = names or list(DETECTORS)
    os.makedirs(outdir, exist_ok=True)
    for name in names:
        run = DETECTORS[name]()
        cap = cv2.VideoCapture(src)
        res = {"w": int(cap.get(3)), "h": int(cap.get(4)), "fps": cap.get(cv2.CAP_PROP_FPS), "frames": []}
        t0 = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res["frames"].append(run(frame))
        n = len(res["frames"])
        dt = time.time() - t0
        json.dump(res, open(f"{outdir}/{name}.json", "w"))
        avg = sum(len(f) for f in res["frames"]) / max(n, 1)
        print(f"{name}: {n} frames, {n / dt:.1f} fps, {avg:.1f} boxes/frame", flush=True)


if __name__ == "__main__":
    main()
