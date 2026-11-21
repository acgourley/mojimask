# /// script
# dependencies = ["onnx"]
# ///
"""Make models/onnx/yolov5m-face-batch.onnx from the uniface export of YOLOv5-Face-M (models/README.md).

    uv run native/tools/free_batch.py models/onnx/yolov5m-face.onnx models/onnx/yolov5m-face-batch.onnx

The export's input already has a free batch dimension, but its output is declared as a batch of one, and so are the
shapes it records for the values in between: ONNX Runtime refuses a batch. This frees the output's batch dimension
and drops the recorded shapes (ONNX Runtime infers them again). No weight and no node changes: a batch of one gives
the same numbers as the export. The detect step (mm-engine detect.rs) runs a frame's tiles as one batch.
"""
import sys

import onnx

src, dst = sys.argv[1:3]
m = onnx.load(src)
g = m.graph
batch = g.input[0].type.tensor_type.shape.dim[0].dim_param
assert batch, "the input's batch dimension is fixed: this needs a new export, not this script"
for o in g.output:
    o.type.tensor_type.shape.dim[0].dim_param = batch
del g.value_info[:]
onnx.save(m, dst)
print(f"{dst}: batch dimension {batch!r}")
