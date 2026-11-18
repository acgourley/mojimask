#!/bin/bash
# Build the native app. Dev (default): the store dylib + mm-cli (cargo, debug) and the Qt host (cmake + ninja), which
# loads native/target/debug/libmm_app.dylib at start. --release: the store as a static library linked into the host
# (native/app/build-release/Mojimask.app). --dist [--no-check] [--dmg]: the release build made a self-contained bundle,
# native/dist/Mojimask.app (packaging/make_app.sh).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
if [ "${1:-}" = "--dist" ]; then shift; exec "$HERE/packaging/make_app.sh" "$@"; fi
if [ "${1:-}" = "--release" ]; then
    cargo build --release -p mm-app --lib -p mm-cli
    B=app/build-release
    [ -f "$B/build.ninja" ] || cmake -S app -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/homebrew -DMM_STATIC=ON
else
    cargo build -p mm-app --lib -p mm-cli
    B=app/build
    [ -f "$B/build.ninja" ] || cmake -S app -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=/opt/homebrew
fi
cmake --build "$B"
echo "$HERE/$B/Mojimask.app"
