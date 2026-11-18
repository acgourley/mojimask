# Models

The weight files aren't in git: several are over 100 MB, and their licences differ. The app needs the
five files marked **app** in `models/onnx/`, plus the depth model `models/depth-anything-v2-small/model.onnx`;
`native/packaging/make_app.sh` copies them into the Mac app. Check a
copy against the SHA-256 below (`shasum -a 256 models/onnx/*.onnx`).

| file | used for | source | licence (code / weights) | SHA-256 |
|---|---|---|---|---|
| `yolov5m-face.onnx` | **app**: face detection (scan) | YOLOv5-Face-M (deepcam-cn), ONNX export from yakhyo/uniface; fixed 640×640 | GPL-3.0; trained on WIDER FACE | `04302ce27a15bde3e20945691b688e2dd018a10e92dd8932146bede6a49207b2` |
| `retinaface-r50.onnx` | **app**: face detection (scan); also face landmarks and the face check | RetinaFace ResNet-50 (biubug6, ONNX via yakhyo/uniface) | MIT; trained on WIDER FACE | `905eac744dc165dbcd62960938a191de3fecea6bef198fb1c5e13e690bc6a828` |
| `yolo11x-pose.onnx` | **app**: body pose, head boxes (scan) | Ultralytics YOLO11x-pose, `yolo export format=onnx` | AGPL-3.0 | `0e1fb014f4e53bffd53fcf68e48155fa65ce1531e00bc9866b9f03c39adc9801` |
| `sface-2021dec.onnx` | **app**: face likeness (same person, same emoji) | OpenCV Zoo face_recognition_sface_2021dec (huggingface.co/opencv/face_recognition_sface), 112×112 RGB 0-255 in, 128-d out | Apache-2.0 (code and weights); training data not documented | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |
| `osnet-ain-x1_0-msmt17.onnx` | **app**: body re-ID | OSNet-AIN x1.0 on MSMT17 (torchreid, Kaiyang Zhou et al.), 256×128 in, 512-d out | MIT code; MSMT17 is research-only data | `20865d9ddeade5929cb8f0e5d2adb2ec15013ac3c0531e1652b6d09c7b0590f3` |
| `../depth-anything-v2-small/model.onnx` | **app**: depth (the scan's depth rule, covers partly behind things, the Editor's Depth view); on whenever it is installed (`MOJIMASK_DEPTH=off` turns it off) | onnx-community/depth-anything-v2-small `onnx/model.onnx` (Depth Anything V2 Small, fp32, 99,060,839 bytes) | Apache-2.0 (only the Small size; Base/Large are CC-BY-NC-4.0) | `afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c` |
