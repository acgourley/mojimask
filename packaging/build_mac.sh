#!/bin/bash
# Build "Mojimask.app" (Apple Silicon, macOS 13+) and a disk image to send.
#
#   ./packaging/build_mac.sh          -> dist/Mojimask.app, dist/Mojimask.dmg
#
# Uses the project's .venv (Python 3.11 + numpy, opencv, onnxruntime, pyobjc, pywebview, PyInstaller).
# Video is decoded and encoded by macOS itself (AVFoundation); no ffmpeg is bundled or needed.
# The app is ad-hoc signed. To ship it without the "unidentified developer" step, sign it with a
# Developer ID and notarize it (see the commented lines at the end).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY=.venv/bin/python
NAME="Mojimask"
VERSION="0.1.0"
mkdir -p build
[ -x "$PY" ] || { echo "No .venv: see requirements.txt for the one-line setup." >&2; exit 1; }

# Model weights aren't in git (models/README.md says where each comes from): stop before a 2-minute build
# that would ship without one. The list is desktop.py's CHECK_MODELS, which `--check` also tests the built app with.
MODELS=($("$PY" -c 'import desktop; print(*desktop.CHECK_MODELS)'))
[ ${#MODELS[@]} -gt 0 ] || { echo "No model list from desktop.CHECK_MODELS" >&2; exit 1; }
for m in "${MODELS[@]}"; do
  [ -f "models/onnx/$m.onnx" ] || { echo "Missing models/onnx/$m.onnx (see models/README.md)" >&2; exit 1; }
done
# The depth model the app turns on (desktop.APP_DEPTH): its file, at the same relative path in the bundle.
DEPTH_MODEL=$("$PY" -c 'import desktop; from mojimask import depth; print(depth.MODELS[desktop.APP_DEPTH]["path"])')
[ -f "$DEPTH_MODEL" ] || { echo "Missing $DEPTH_MODEL (see models/README.md)" >&2; exit 1; }

# OpenCV must be the small build from packaging/build_opencv.sh: pip's opencv-python wheel brings ~99 libraries,
# a GPL FFmpeg with x264/x265 among them, that would all ship (and need their source published) for nothing.
# (pip's is also a cv2/ package; ours is the one file cv2.cpython-311-darwin.so.)
"$PY" -c 'import cv2, os, re, sys
sys.exit(bool(re.search(r"FFMPEG:\s+YES", cv2.getBuildInformation()))
         or os.path.basename(os.path.dirname(cv2.__file__)) == "cv2")' \
  || { echo "cv2 is pip's full opencv-python: run ./packaging/build_opencv.sh first" >&2; exit 1; }

$PY packaging/make_icon.py build/AppIcon.icns

# Pages and scripts the two servers hand to the window: every .html, .js, .css and .json in web/, so a new page
# can't be left out. (Not all of web/assets/: its test portrait is a real person's photo, for developer mode only.)
DATA=()
for f in "$ROOT"/web/*.html "$ROOT"/web/*.js "$ROOT"/web/*.css "$ROOT"/web/*.json; do DATA+=(--add-data "$f:web"); done
# The pictures the app itself shows (old paintings, public domain): the start page's three, the icon's painting (the opening page shows it) and the default portrait.
DATA+=(--add-data "$ROOT/web/assets/gallery:web/assets/gallery" --add-data "$ROOT/web/assets/icon_art.jpg:web/assets" --add-data "$ROOT/web/assets/portrait_mona.png:web/assets")
# The app's two fonts (Nunito, Inter) with their Open Font License notices (web/fonts/README.md).
DATA+=(--add-data "$ROOT/web/fonts:web/fonts")
for m in "${MODELS[@]}"; do DATA+=(--add-data "$ROOT/models/onnx/$m.onnx:models/onnx"); done
DATA+=(--add-data "$ROOT/$DEPTH_MODEL:$(dirname "$DEPTH_MODEL")")
# The start screen's "Use the sample clip" (app_server.SAMPLE): same relative path as in a checkout.
DATA+=(--add-data "$ROOT/test_clips/jazz.mp4:test_clips")
# What the app bundles and under which licences (THIRD_PARTY_NOTICES.md), with the licence texts beside it.
$PY packaging/licenses.py "$ROOT/build/licenses"
DATA+=(--add-data "$ROOT/LICENSE:." --add-data "$ROOT/THIRD_PARTY_NOTICES.md:." --add-data "$ROOT/build/licenses:licenses")

# The scan's track-and-decide step and the editor's re-ID live in the scan package (mojimask.scan.newalgo, .faces and
# what they import); desktop.py --check imports both by name, so PyInstaller is told about them. Everything else is
# found by following imports from desktop.py; review_server is started by name.
HIDDEN=()
for m in mojimask.review_server mojimask.scan.newalgo mojimask.scan.faces AVFoundation CoreMedia Quartz AppKit Foundation WebKit objc \
         webview.platforms.cocoa; do HIDDEN+=(--hidden-import "$m"); done

# (experiments, tools: developer scripts, never imported by the app)
EXCLUDE=()
# (psutil, yaml, jinja2...: optional imports of onnxruntime and bottle that the app never reaches)
for m in torch torchvision transformers ultralytics insightface uniface matplotlib pandas scipy sklearn \
         tkinter IPython jupyter notebook sympy experiments tools \
         psutil yaml jinja2 markupsafe charset_normalizer requests urllib3 certifi; do EXCLUDE+=(--exclude-module "$m"); done

# (The bundle id is also the CoreML cache folder's name, ~/Library/Caches/org.mojimask.app, in onnx_core.py.)
$PY -m PyInstaller --noconfirm --clean --windowed --log-level WARN --name "$NAME" \
  --osx-bundle-identifier org.mojimask.app \
  --icon "$ROOT/build/AppIcon.icns" \
  --collect-data webview --collect-binaries onnxruntime \
  "${DATA[@]}" "${HIDDEN[@]}" "${EXCLUDE[@]}" \
  --distpath "$ROOT/dist" --workpath "$ROOT/build/pyinstaller" --specpath "$ROOT/build" \
  --paths "$ROOT" \
  "$ROOT/desktop.py" 2>&1 | tee build/build.log

APP="dist/$NAME.app"
PL="$APP/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VERSION" "$PL"
/usr/libexec/PlistBuddy -c "Add :CFBundleVersion string $VERSION" "$PL" 2>/dev/null || /usr/libexec/PlistBuddy -c "Set :CFBundleVersion $VERSION" "$PL"
/usr/libexec/PlistBuddy -c "Add :LSMinimumSystemVersion string 13.0" "$PL" 2>/dev/null || true
/usr/libexec/PlistBuddy -c "Add :LSApplicationCategoryType string public.app-category.video" "$PL" 2>/dev/null || true
/usr/libexec/PlistBuddy -c "Add :NSHumanReadableCopyright string AGPL-3.0 open source" "$PL" 2>/dev/null || true
/usr/libexec/PlistBuddy -c "Add :NSAppTransportSecurity dict" "$PL" 2>/dev/null || true
/usr/libexec/PlistBuddy -c "Add :NSAppTransportSecurity:NSAllowsLocalNetworking bool true" "$PL" 2>/dev/null || true

# PyInstaller has already signed every library inside; the Info.plist edits above only need the bundle
# itself re-signed (signing with --deep is deprecated by Apple).
codesign --force --sign - "$APP"
codesign --verify --deep --strict "$APP" || { echo "signature check failed" >&2; exit 1; }
echo "signature ok (ad-hoc)"

# The built app loads every engine module, page and model it needs, or the build fails here.
"$APP/Contents/MacOS/$NAME" --check

# the disk image: drag to Applications, with the first-launch steps drawn in its window (make_dmg.sh)
./packaging/make_dmg.sh "$APP"
du -sh "$APP" "dist/$NAME.dmg"

# With an Apple Developer ID (removes the Gatekeeper step for recipients). Untested. Sign inside-out, not with
# --deep: every nested library first, then the bundle with the entitlements.
#   ID="Developer ID Application: <you>"
#   find "$APP/Contents" \( -name '*.so' -o -name '*.dylib' \) -exec codesign --force --options runtime --timestamp --sign "$ID" {} +
#   codesign --force --options runtime --timestamp --entitlements packaging/entitlements.plist --sign "$ID" "$APP"
#   ditto -c -k --keepParent "$APP" app.zip && xcrun notarytool submit app.zip --keychain-profile <profile> --wait
#   xcrun stapler staple "$APP"
