//! mm-ml: the engine's models from Rust through `ort` (ONNX Runtime 1.31).
//!
//! - [`rt`]: sessions and execution providers (CPU with PreferCPU, CoreML GPU/ANE, WebGPU with `--features webgpu`).
//! - [`pre`]: preprocessing with cv2.resize's maths reproduced (bit-exact INTER_LINEAR), letterbox, blobs.
//! - [`post`]: NMS, YOLO-face and pose decoding, RetinaFace anchors and boxes.
//! - [`npy`]: .npy files (the parity data from poc/ml/dump_python.py in the unpublished PoC).
//!
//! Parity: `mm-ml-parity out/poc/results/ml/ref` reruns poc/ml's check (raw outputs vs Python's, same EP).
pub mod npsort;
pub mod npy;
pub mod post;
pub mod pre;
pub mod rt;
