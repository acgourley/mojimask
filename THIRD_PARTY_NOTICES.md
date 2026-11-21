# Third-party notices

mojimask's Mac app bundles the software, models, fonts and pictures below. Each keeps its own licence; the full texts
ship in the app's `licenses/` folder (`Contents/Resources/licenses`, collected when the app is built by `mm-pack
notices` (`native/crates/mm-pack`)), and mojimask's own `LICENSE` beside this file. Where a licence is copyleft
(GPL-3.0, AGPL-3.0), the corresponding source is this repository and the sources linked below.

## Models (`models/`, details and checksums in `models/README.md`)

| model | what it does in the app | licence | text in `licenses/` |
|---|---|---|---|
| YOLOv5-Face-M (deepcam-cn; ONNX export via yakhyo/uniface, modified: see below) | finds faces | GPL-3.0; trained on WIDER FACE | `GPL-3.0.txt` |
| RetinaFace ResNet-50 (biubug6/Pytorch_Retinaface; ONNX via yakhyo/uniface) | finds faces, face landmarks | MIT; trained on WIDER FACE | `MIT-RetinaFace.txt` |
| YOLO11x-pose (Ultralytics) | body poses, head boxes | AGPL-3.0 | `AGPL-3.0.txt` |
| SFace 2021dec (OpenCV Zoo) | face likeness | Apache-2.0 | `Apache-2.0.txt` |
| OSNet-AIN x1.0, MSMT17 (Kaiyang Zhou et al., torchreid) | body likeness | MIT code; trained on MSMT17, research-only data (an open question: see models/README.md) | `MIT-OSNet.txt` |
| Depth Anything V2 Small (onnx-community export) | depth, for partial covers | Apache-2.0 (the Small size only) | `Apache-2.0.txt` |

The YOLOv5-Face-M file in the app (`yolov5m-face-batch.onnx`) is a modified copy of the uniface export, changed on
2026-11-21 by `native/tools/free_batch.py`: the output's declared batch size of 1 takes the input's batch dimension,
and the recorded intermediate shapes are dropped, so the frame and its tiles run as one batch. No weight or node is
changed. That script is the modification's source.

WIDER FACE (the face detectors' training data) is published for research; its terms say nothing about trained
weights.

## Software

The app is Rust (the crates in `native/crates/`) with a Qt Quick window. What it links and bundles, each with its
licence and its texts, is listed in the app's own `THIRD_PARTY_NOTICES.md` (`Contents/Resources`), generated from the
build itself by `mm-pack notices` (`native/crates/mm-pack`): every Rust crate it is built from, ONNX Runtime (MIT,
with its own third-party notices), Qt 6 (LGPL-3.0, linked dynamically), GStreamer and GLib (LGPL-2.1), and the other
libraries they load. It fails the build if a library's licence isn't known.

## Fonts and pictures

| item | licence |
|---|---|
| Nunito, Inter (`native/web/fonts/`) | SIL Open Font License 1.1: `OFL-1.1-Nunito.txt`, `OFL-1.1-Inter.txt` |
| The start page's paintings (`native/web/assets/gallery`): *Mona Lisa*, Leonardo da Vinci, c. 1503–1519; *The Laughing Cavalier*, Frans Hals, 1624; *Girl with a Pearl Earring*, Johannes Vermeer, c. 1665 | public domain |
| The default portrait (`native/web/assets/portrait_mona.png`): the head of the *Mona Lisa* above, cut out | public domain |
| The app icon and opening page (`native/web/assets/icon_art.jpg`): *The Laughing Cavalier*, Frans Hals, 1624 | public domain |
| The sample clip (`test_clips/jazz.mp4`): a few seconds of *Our Dancing Daughters* (Harry Beaumont, MGM, 1928) | public domain in the US (1928 works since 1 January 2024); elsewhere terms may run longer |

Photographs of these two-dimensional public-domain paintings add no copyright of their own in the US (Bridgeman v.
Corel, 1999); some other countries see it differently.

## mojimask's own licence

mojimask is free software under the GNU Affero General Public License 3.0 (`LICENSE`), with no warranty. Because the
app ships YOLO11-pose (AGPL-3.0), the app as a whole is distributed on AGPL-3.0 terms: anyone who has it may get its
complete source (this repository, with `native/packaging/` for building it and `models/README.md` for the model files)
and change and share it under the same terms. The app links to these notices from its Projects page ("Licences").
