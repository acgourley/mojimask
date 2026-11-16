# Models

The weight files aren't in git: several are over 100 MB, and their licences differ. The app needs the
five files marked **app** in `models/onnx/`, plus the depth model `models/depth-anything-v2-small/model.onnx`;
`packaging/build_mac.sh` copies them into the build. Check a
copy against the SHA-256 below (`shasum -a 256 models/onnx/*.onnx`).

| file | used for | source | licence (code / weights) | SHA-256 |
|---|---|---|---|---|
| `yolov5m-face.onnx` | **app**: face detection (scan) | YOLOv5-Face-M (deepcam-cn), ONNX export from yakhyo/uniface; fixed 640×640 | GPL-3.0; trained on WIDER FACE | `04302ce27a15bde3e20945691b688e2dd018a10e92dd8932146bede6a49207b2` |
| `retinaface-r50.onnx` | **app**: face detection (scan); also face landmarks and the face check (`reid.landmarks`) | RetinaFace ResNet-50 (biubug6, ONNX via yakhyo/uniface) | MIT; trained on WIDER FACE | `905eac744dc165dbcd62960938a191de3fecea6bef198fb1c5e13e690bc6a828` |
| `yolo11x-pose.onnx` | **app**: body pose, head boxes (scan) | Ultralytics YOLO11x-pose, `yolo export format=onnx` | AGPL-3.0 | `0e1fb014f4e53bffd53fcf68e48155fa65ce1531e00bc9866b9f03c39adc9801` |
| `scrfd-10g-kps.onnx` | replaced 2026-11-14 by RetinaFace's landmarks; comparisons only (`experiments/insightface_models.py`), never packaged | InsightFace `buffalo_l` pack | InsightFace: non-commercial research only | `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` |
| `arcface-w600k-r50.onnx` | replaced 2026-11-14 by SFace; comparisons only, never packaged | InsightFace `buffalo_l` pack | InsightFace: non-commercial research only | `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` |
| `sface-2021dec.onnx` | **app**: face likeness (same person, same emoji; `reid.embed`) | OpenCV Zoo face_recognition_sface_2021dec (huggingface.co/opencv/face_recognition_sface), 112×112 RGB 0-255 in, 128-d out | Apache-2.0 (code and weights); training data not documented | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |
| `dinov2-small.onnx` | not used by the app now (`appearance.embed` has no callers; not packaged) | facebook/dinov2-small, exported with CLS + mean-patch pooling built in | Apache-2.0 | `eb24f2bb5a1db726395c56c418b50c971f51fc9294e34418ab660a40951b58c8` |
| `osnet-ain-x1_0-msmt17.onnx` | **app**: body re-ID | OSNet-AIN x1.0 on MSMT17 (torchreid, Kaiyang Zhou et al.), 256×128 in, 512-d out | MIT code; MSMT17 is research-only data | `20865d9ddeade5929cb8f0e5d2adb2ec15013ac3c0531e1652b6d09c7b0590f3` |
| `../depth-anything-v2-small/model.onnx` | `MOJIMASK_DEPTH=anything-small` (depth.py; **app**: bundled, and on in the Mac app; off from a checkout unless set) | onnx-community/depth-anything-v2-small `onnx/model.onnx` (Depth Anything V2 Small, fp32, 99,060,839 bytes) | Apache-2.0 (only the Small size; Base/Large are CC-BY-NC-4.0) | `afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c` |
| `../depth-anything-v2-base/model.onnx` | research only: `MOJIMASK_DEPTH=anything-base` (depth.py) | onnx-community/depth-anything-v2-base `onnx/model.onnx` (fp32, 388,917,010 bytes) | CC-BY-NC-4.0: non-commercial, never packaged | `dab40f67d5ead5e0ed52acfae41a717a708b02bc57f6cb761c5de828f26a7289` |
| `../depth-pro/` | research only: `MOJIMASK_DEPTH=pro` (depth.py; torch + transformers) | apple/DepthPro-hf (`model.safetensors` 1,904,185,492 bytes, `config.json`, `preprocessor_config.json`) | code Apple sample code licence; weights Apple ML Research Model licence (apple-amlr): research only, never packaged | `9c6811e3165485b9a94a204329860cb333a79877e757eb795a179a4ea34bbcf7` (model.safetensors) |
| `yolo11m-pose.onnx` | research only | Ultralytics YOLO11m-pose (faster, less accurate candidate) | AGPL-3.0 | `b945b9810c58fe187a46a1f7255777df85b2794c30649e3d4197124d963f49c7` |

The research tools in `experiments/` also use PyTorch weights kept loose in `models/` (YOLO11/12 face and
pose `.pt`, `sam3/`, `egoblur/`, `dinov2-small/`); `experiments/detect.py` says which one it loads where.

