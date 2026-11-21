#!/bin/bash
# Make the native app a self-contained Mac bundle: native/dist/Mojimask.app (Apple Silicon, macOS 15+).
#
#   native/packaging/make_app.sh [--no-build] [--no-check] [--dmg]
#
# 1. Builds the release host (native/build.sh --release: the Rust store linked statically, ONNX Runtime with it) and
#    mm-cli. If the host fails to link there, it is linked again in native/target/package/host with the system
#    frameworks the store needs (AVFoundation, Accelerate, CoreML, CoreText, ...), until app/CMakeLists.txt lists them.
# 2. Lays out the bundle:
#    MacOS/      Mojimask (launcher.c: sets GST_*, MM_ROOT, MOJIMASK_MODELS, then execs), mojimask-app (the Qt host),
#                mm-cli, gst-plugin-scanner
#    PlugIns/    Qt: platforms/libqcocoa, imageformats/libqjpeg; gstreamer/: the 8 plugins mm-media uses
#    Frameworks/ every Homebrew library those load (Qt frameworks without headers, GStreamer, GLib, ICU, ...)
#    Resources/  qml/ (only the QML modules app/qml imports, Controls with the Basic style only), qt.conf, web/ (fonts,
#                strings.json, icons.json, brand.json, palettes.json, emoji_eyes.json, the pictures), models/ (desktop.py's
#                CHECK_MODELS + depth, laid out as in a checkout), test_clips/jazz.mp4, check.sh,
#                THIRD_PARTY_NOTICES.md + licenses/, LICENSE
# 3. mm-pack relocate (crates/mm-pack): copies and rewrites so no load command or rpath names /opt/homebrew (it fails otherwise).
# 4. Signs inside-out: ad hoc, or with MOJIMASK_SIGN_ID (a "Developer ID Application" identity) for distribution.
#    With --dmg and MOJIMASK_NOTARY_PROFILE (an `xcrun notarytool store-credentials` profile), make_dmg.sh also signs,
#    notarizes and staples the disk image.
# 5. `Mojimask --check`: scan + export jazz from the bundle (check.sh), with PATH stripped and /opt/homebrew
#    unreadable (sandbox-exec), so a pass proves the bundle is self-contained. --no-check skips it.
#
# Instead of the launcher, main.cpp could set the same variables before QGuiApplication (the report has the patch);
# Qt finds its plugins and QML through Resources/qt.conf either way.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"      # native/packaging
NATIVE="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$NATIVE/.." && pwd)"
# the main checkout (a worktree has no venv or models of its own): the parent of git's common directory, else ROOT
G="$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"; MAIN="${G:+$(dirname "$G")}"; MAIN="${MAIN:-$ROOT}"
MODELS_SRC="${MOJIMASK_MODELS_SRC:-$ROOT/models}"
[ -d "$MODELS_SRC/onnx" ] || MODELS_SRC="$MAIN/models"
BREW=/opt/homebrew
QT=$BREW/share/qt
NAME=Mojimask
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$NATIVE/Cargo.toml" | head -1)"   # the workspace's version
DIST="$NATIVE/dist"
APP="$DIST/$NAME.app"
C="$APP/Contents"
BUILD=1; CHECK=1; DMG=0
for a in "$@"; do case "$a" in --no-build) BUILD=0;; --no-check) CHECK=0;; --dmg) DMG=1;; *) echo "unknown: $a" >&2; exit 2;; esac; done
T0=$(date +%s)

# -- 1. build ---------------------------------------------------------------------------------------------------
HOST="$NATIVE/app/build-release/$NAME.app"
mkdir -p "$DIST"
if [ $BUILD = 1 ]; then
  if ! "$NATIVE/build.sh" --release >"$DIST/build.log" 2>&1; then
    echo "build.sh --release failed (see $DIST/build.log); linking the host in target/package/host with the store's frameworks" >&2
    B="$NATIVE/target/package/host"
    [ -f "$B/build.ninja" ] || cmake -S "$NATIVE/app" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$BREW -DMM_STATIC=ON \
      -DCMAKE_EXE_LINKER_FLAGS="-framework AVFoundation -framework Accelerate -framework CoreML -framework CoreText -framework CoreImage -framework VideoToolbox -framework MediaToolbox -framework CoreAudio" >/dev/null
    cmake --build "$B" >>"$DIST/build.log" 2>&1 || { tail -30 "$DIST/build.log" >&2; exit 1; }
    HOST="$B/$NAME.app"
  fi
  # the fallback's output is newer when build.sh failed this time
  [ -x "$NATIVE/target/package/host/$NAME.app/Contents/MacOS/$NAME" ] && \
    [ "$NATIVE/target/package/host/$NAME.app/Contents/MacOS/$NAME" -nt "$HOST/Contents/MacOS/$NAME" ] && HOST="$NATIVE/target/package/host/$NAME.app"
else
  [ -x "$NATIVE/target/package/host/$NAME.app/Contents/MacOS/$NAME" ] && \
    { [ ! -x "$HOST/Contents/MacOS/$NAME" ] || [ "$NATIVE/target/package/host/$NAME.app/Contents/MacOS/$NAME" -nt "$HOST/Contents/MacOS/$NAME" ]; } && \
    HOST="$NATIVE/target/package/host/$NAME.app"
fi
CLI="$NATIVE/target/release/mm-cli"
# the packaging steps below (relocate, ICU trim, notices, sizes): crates/mm-pack
cargo build -q --release -p mm-pack --manifest-path "$NATIVE/Cargo.toml"
PACK="$NATIVE/target/release/mm-pack"
[ -x "$HOST/Contents/MacOS/$NAME" ] && [ -x "$CLI" ] || { echo "no release host or mm-cli: run without --no-build" >&2; exit 1; }
echo "host: $HOST"

# models: the detectors, pose, face and body re-ID the scan runs (mm-engine), plus the depth model the app turns on
MODELS=(yolov5m-face-batch retinaface-r50 yolo11x-pose sface-2021dec osnet-ain-x1_0-msmt17)
DEPTH_REL=depth-anything-v2-small/model.onnx      # mm-analysis depth.rs: models/depth-anything-v2-small/model.onnx
for m in "${MODELS[@]}"; do [ -f "$MODELS_SRC/onnx/$m.onnx" ] || { echo "missing $MODELS_SRC/onnx/$m.onnx" >&2; exit 1; }; done
[ -f "$MODELS_SRC/$DEPTH_REL" ] || { echo "missing $MODELS_SRC/$DEPTH_REL" >&2; exit 1; }

# -- 2. lay out -------------------------------------------------------------------------------------------------
rm -rf "$APP"
mkdir -p "$DIST"
ditto "$HOST" "$APP"
mv "$C/MacOS/$NAME" "$C/MacOS/mojimask-app"
clang -O2 -Wall -mmacosx-version-min=15.0 -o "$C/MacOS/$NAME" "$HERE/launcher.c"
cp "$CLI" "$C/MacOS/mm-cli"
cp "$BREW/opt/gstreamer/libexec/gstreamer-1.0/gst-plugin-scanner" "$C/MacOS/"
chmod 755 "$C/MacOS/"*
# local symbols out (-x keeps the dynamic ones): -15 MB app, -10 MB mm-cli
strip -x "$C/MacOS/mojimask-app" "$C/MacOS/mm-cli" 2>/dev/null

# Qt plugins: the Cocoa platform, JPEG (FramedArt's gallery pictures; PNG is built into QtGui) and SVG (Icon.qml's
# icons are SVG data URLs from native/web/icons.json)
mkdir -p "$C/PlugIns/platforms" "$C/PlugIns/imageformats"
cp -L "$QT/plugins/platforms/libqcocoa.dylib" "$C/PlugIns/platforms/"
cp -L "$QT/plugins/imageformats/libqjpeg.dylib" "$QT/plugins/imageformats/libqsvg.dylib" "$C/PlugIns/imageformats/"

# GStreamer: mm-media's elements. filesrc queue capsfilter appsink->app, filesink fakesink (coreelements); qtdemux
# mp4mux (isomp4); parsebin (playback) with typefind (typefindfunctions); h264parse h265parse (videoparsersbad);
# vtdec_hw vtenc_h264_hw (applemedia); videotestsrc (cv.rs registers GstCoreMediaMeta with one test frame)
mkdir -p "$C/PlugIns/gstreamer"
for p in coreelements app isomp4 playback typefindfunctions videoparsersbad applemedia videotestsrc; do
  cp -L "$BREW/lib/gstreamer-1.0/libgst$p.dylib" "$C/PlugIns/gstreamer/"
done

# QML: only what app/qml imports (qmlimportscanner), and of QtQuick.Controls only the Basic style
R="$C/Resources"
Q="$R/qml"
mkdir -p "$Q"
top() { mkdir -p "$Q/$1"; find -L "$QT/qml/$1" -maxdepth 1 -type f -exec cp -L {} "$Q/$1/" \; ; }   # a module's own files
all() { mkdir -p "$(dirname "$Q/$1")"; cp -RL "$QT/qml/$1" "$Q/$1"; }                               # with its sub-folders
top QtQml; all QtQml/Models; all QtQml/WorkerScript
top QtQuick; all QtQuick/Window; all QtQuick/Layouts; all QtQuick/Shapes; all QtQuick/Effects; all QtQuick/Templates
top QtQuick/Controls; all QtQuick/Controls/impl; all QtQuick/Controls/Basic
top QtQuick/Dialogs; all QtQuick/Dialogs/quickimpl
all Qt/labs/folderlistmodel
find "$Q" -name '*.qmltypes' -delete; find "$Q" -name designer -type d -prune -exec rm -rf {} +   # tooling only
cat >"$R/qt.conf" <<EOF
[Paths]
Plugins = PlugIns
QmlImports = Resources/qml
Qml2Imports = Resources/qml
EOF

# what the app and the engine read from web/ (mm-app text.rs, library.rs; mm-records' portrait; the QML's fonts
# and pictures)
W="$R/web"
mkdir -p "$W/assets"
cp -R "$NATIVE/web/fonts" "$W/fonts"
for f in strings.json icons.json brand.json palettes.json emoji_eyes.json; do cp "$NATIVE/web/$f" "$W/"; done
cp -R "$NATIVE/web/assets/gallery" "$W/assets/gallery"
cp "$NATIVE/web/assets/icon_art.jpg" "$NATIVE/web/assets/portrait_mona.png" "$W/assets/"
mkdir -p "$R/models/onnx" "$R/models/$(dirname $DEPTH_REL)" "$R/test_clips"
for m in "${MODELS[@]}"; do cp "$MODELS_SRC/onnx/$m.onnx" "$R/models/onnx/"; done
cp "$MODELS_SRC/$DEPTH_REL" "$R/models/$DEPTH_REL"
cp "$ROOT/test_clips/jazz.mp4" "$R/test_clips/"
cp "$HERE/check.sh" "$R/check.sh"; chmod 755 "$R/check.sh"
cp "$ROOT/LICENSE" "$R/LICENSE"

# the icon, drawn now with AppKit (make_icon.swift: Apple's emoji isn't kept in git)
swift "$HERE/make_icon.swift" "$NATIVE/web/assets/icon_art.jpg" "$DIST/AppIcon.icns" >/dev/null
cp "$DIST/AppIcon.icns" "$R/AppIcon.icns"

PL="$C/Info.plist"
chmod u+w "$PL"   # the build's copy is read-only, and PlistBuddy fails quietly on it
pb() { /usr/libexec/PlistBuddy -c "$1" "$PL" 2>/dev/null || true; }
pb "Set :CFBundleExecutable $NAME"
pb "Add :CFBundleShortVersionString string $VERSION"; pb "Set :CFBundleShortVersionString $VERSION"
pb "Add :CFBundleVersion string $VERSION"; pb "Set :CFBundleVersion $VERSION"
pb "Add :LSMinimumSystemVersion string 15.0"; pb "Set :LSMinimumSystemVersion 15.0"
pb "Add :LSApplicationCategoryType string public.app-category.video"
pb "Add :NSHumanReadableCopyright string AGPL-3.0 open source"; pb "Set :NSHumanReadableCopyright AGPL-3.0 open source"
pb "Add :NSHighResolutionCapable bool true"
[ -f "$R/AppIcon.icns" ] && { pb "Add :CFBundleIconFile string AppIcon"; pb "Set :CFBundleIconFile AppIcon"; }

# -- 3. relocate ------------------------------------------------------------------------------------------------
"$PACK" relocate "$APP" >"$DIST/relocate.json" || { cat "$DIST/relocate.json"; exit 1; }
grep -E '"(machos|copied|ok)"' "$DIST/relocate.json" | tr -d '\n'; echo
# ICU's data, cut to what Qt uses (34 -> ~10 MB); MM_FULL_ICU=1 keeps Homebrew's
[ "${MM_FULL_ICU:-0}" = 1 ] || "$PACK" trim-icu "$APP"

# licences: crates (cargo metadata), Qt, GStreamer and the other Homebrew libraries, models, fonts
"$PACK" notices "$APP" "$ROOT" "$NATIVE"

# -- 4. sign (inside-out) -----------------------------------------------------------------------------------------
# ad hoc by default; MOJIMASK_SIGN_ID="Developer ID Application: ..." signs for distribution: hardened runtime, a secure
# timestamp (asks Apple's timestamp server), entitlements.plist on the executables
if [ -n "${MOJIMASK_SIGN_ID:-}" ]; then
  SIGN=(--force --sign "$MOJIMASK_SIGN_ID" --options runtime --timestamp)
  ENT=(--entitlements "$HERE/entitlements.plist")
else
  SIGN=(--force --sign - --timestamp=none); ENT=()
fi
find "$C/Frameworks" "$C/PlugIns" "$C/Resources/qml" -type f \( -name '*.dylib' -o -perm -u+x \) ! -name '*.qml' ! -name qmldir -print0 |
  xargs -0 -n 50 codesign "${SIGN[@]}" 2>&1 | grep -v "replacing existing signature" || true
for f in "$C"/Frameworks/*.framework; do codesign "${SIGN[@]}" "$f" 2>&1 | grep -v "replacing existing" || true; done
for f in mojimask-app mm-cli gst-plugin-scanner; do codesign "${SIGN[@]}" ${ENT[@]+"${ENT[@]}"} "$C/MacOS/$f" 2>&1 | grep -v "replacing existing" || true; done
codesign "${SIGN[@]}" ${ENT[@]+"${ENT[@]}"} "$APP" 2>&1 | grep -v "replacing existing" || true
codesign --verify --deep --strict "$APP" || { echo "signature check failed" >&2; exit 1; }
echo "signature ok (${MOJIMASK_SIGN_ID:-ad hoc})"

# -- 5. sizes, then the self-test -------------------------------------------------------------------------------
echo "bundle $(du -sh "$APP" | cut -f1), built in $(( $(date +%s) - T0 )) s"
"$PACK" sizes "$APP" | tee "$DIST/sizes.json"
if [ $CHECK = 1 ]; then
  "$HERE/run_isolated.sh" "$APP/Contents/MacOS/$NAME" --check | tee "$DIST/check.json"
  tail -n 1 "$DIST/check.json" | grep -q '"ok": true}' || { echo "--check failed" >&2; exit 1; }
fi
if [ $DMG = 1 ]; then "$HERE/make_dmg.sh" "$APP"; fi
