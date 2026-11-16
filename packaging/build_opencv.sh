#!/bin/bash
# Build the small OpenCV the app uses and install it into .venv as one file (cv2.cpython-311-darwin.so).
#
#   ./packaging/build_opencv.sh        (~5 minutes; needs cmake + ninja, e.g. `brew install cmake ninja`)
#
# Why not `pip install opencv-python`: that wheel links ~99 libraries the app never calls, among them a GPL
# build of FFmpeg with the x264/x265 encoders (its README says LGPL; the macOS arm64 wheel's own libavutil
# says "GPL version 3 or later"). Shipping those would mean publishing the exact source of every one.
# The app decodes and encodes video with macOS's AVFoundation and reads photos with CoreGraphics; from
# OpenCV it needs image maths, optical flow, affine fits, and JPEG/PNG for its own frame files. So this
# builds only those modules, statically, with OpenCV's own bundled libjpeg-turbo, libpng and zlib: the
# result links nothing but macOS system frameworks.
#
# Source: the opencv-python sdist below (OpenCV 5.0.0, Apache-2.0), checked against its SHA-256.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY="$ROOT/.venv/bin/python"
VER=5.0.0.93
URL="https://files.pythonhosted.org/packages/79/4c/a438d23e09ce2033c09f7b784ad2fbdb0adf529e434101ed28f142226f98/opencv_python-$VER.tar.gz"
SHA=66aac3e5b5faa48d4025816592f3af19e4bfc2c68dec067bae2dbb4ca10aa9e2
W="$ROOT/build/opencv"
mkdir -p "$W"
[ -x "$PY" ] || { echo "No .venv: see requirements.txt for the one-line setup." >&2; exit 1; }
# Checked now, not after the 5-minute compile: removing pip's opencv-python at the end needs uv or pip.
command -v uv >/dev/null || "$PY" -m pip --version >/dev/null 2>&1 \
  || { echo "Needs uv (or pip in .venv) to remove pip's opencv-python afterwards." >&2; exit 1; }

TGZ="$W/opencv_python-$VER.tar.gz"
[ -f "$TGZ" ] || curl -fL -o "$TGZ" "$URL"
echo "$SHA  $TGZ" | shasum -a 256 -c -
# Source and CMake cache are per version, so changing VER builds the new one. Extracted beside, moved into place
# only once tar has finished, so an interrupted run doesn't leave half a tree that is reused next time.
SRC="$W/src-$VER"
if [ ! -d "$SRC" ]; then
  rm -rf "$SRC.part" && mkdir -p "$SRC.part"
  tar xzf "$TGZ" -C "$SRC.part" --strip-components 1
  mv "$SRC.part" "$SRC"
fi

SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["platlib"])')"
NUMPY_INC="$("$PY" -c 'import numpy; print(numpy.get_include())')"

# OFF: everything that would pull in a third-party library, a download, or a module the app doesn't use.
cmake -S "$SRC/opencv" -B "$W/cmake-$VER" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_IGNORE_PREFIX_PATH="/opt/homebrew;/usr/local" -DCMAKE_FIND_FRAMEWORK=LAST \
  -DBUILD_SHARED_LIBS=OFF -DBUILD_LIST=core,imgproc,imgcodecs,video,geometry,features,python3 \
  -DBUILD_JPEG=ON -DBUILD_PNG=ON -DBUILD_ZLIB=ON -DWITH_JPEG=ON -DWITH_PNG=ON -DWITH_SPNG=OFF \
  -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF -DWITH_OPENEXR=OFF -DWITH_AVIF=OFF \
  -DWITH_JPEGXL=OFF -DWITH_GDAL=OFF -DWITH_GDCM=OFF -DWITH_IMGCODEC_HDR=OFF -DWITH_IMGCODEC_SUNRASTER=OFF \
  -DWITH_IMGCODEC_PXM=OFF -DWITH_IMGCODEC_PFM=OFF -DWITH_IMGCODEC_GIF=OFF \
  -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_AVFOUNDATION=OFF -DWITH_OPENCL=OFF -DWITH_OPENGL=OFF \
  -DWITH_QT=OFF -DWITH_GTK=OFF -DWITH_VTK=OFF -DWITH_TBB=OFF -DWITH_OPENMP=OFF -DWITH_EIGEN=OFF \
  -DWITH_PROTOBUF=OFF -DWITH_QUIRC=OFF -DWITH_ITT=OFF -DWITH_IPP=OFF -DWITH_KLEIDICV=OFF \
  -DWITH_ONNX=OFF -DWITH_OPENVINO=OFF -DWITH_CUDA=OFF -DWITH_VULKAN=OFF -DWITH_ADE=OFF \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF -DBUILD_JAVA=OFF \
  -DBUILD_opencv_apps=OFF -DOPENCV_GENERATE_PKGCONFIG=OFF -DINSTALL_PYTHON_EXAMPLES=OFF \
  -DOPENCV_SKIP_PYTHON_LOADER=ON -DOPENCV_PYTHON3_INSTALL_PATH="$SITE" \
  -DPYTHON3_EXECUTABLE="$PY" -DPYTHON3_NUMPY_INCLUDE_DIRS="$NUMPY_INC" \
  -DCMAKE_INSTALL_PREFIX="$W/prefix" | tee "$W/configure.log" | grep -E "^--   (To be built|Disabled|Video I/O|  FFMPEG|    JPEG|    PNG)" || true
cmake --build "$W/cmake-$VER" --target opencv_python3
cmake --install "$W/cmake-$VER" --component python

# pip's opencv-python (if any) is a cv2/ package and would shadow the single-file module.
if "$PY" -c 'import importlib.metadata as m; m.version("opencv-python")' 2>/dev/null; then
  if command -v uv >/dev/null; then uv pip uninstall --python "$PY" opencv-python
  else "$PY" -m pip uninstall -y opencv-python; fi
fi

# Check: imports, has no FFmpeg, links only system libraries.
SO="$("$PY" -c 'import cv2; print(cv2.__file__)')"
"$PY" -c 'import cv2; b = cv2.getBuildInformation(); assert "FFMPEG" not in b or "FFMPEG:                      NO" in b; print("cv2", cv2.__version__, "ok")'
if otool -L "$SO" | tail -n +2 | grep -vE "^\s*(/System/Library/|/usr/lib/)"; then
  echo "cv2 links a non-system library (above)" >&2; exit 1
fi
du -h "$SO"
