#!/bin/sh
# The bundle's self-test (`Mojimask --check [DIR]`; the launcher sets the GStreamer and MM_ROOT environment first):
# scan the sample clip with the bundled engine, models and GStreamer, export it, and print one JSON report whose
# last key is "ok". Exit 0 only when it is true. The build fails without it (make_app.sh runs it).
#
# Uses only the bundle and /bin/sh tools: mm-cli for the scan (the native engine: motion native, depth on, the
# timed detection plan) and for the export (GStreamer: vtenc + mp4mux). Work goes to DIR (default: a new temp folder,
# removed afterwards).
#
# What it doesn't cover yet: the export draws no covers. mm-cli's store opens a video plus a PoC dump, not a scanned
# project (open_project goes through the Python sidecar until the native engine lands in mm-app). When the store can
# open a native project, export the scanned one instead (see the report's request to appnative).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"          # Contents/Resources
MACOS="$(cd "$HERE/../MacOS" && pwd)"
CLI="$MACOS/mm-cli"
CLIP="$HERE/test_clips/jazz.mp4"
KEEP=1
if [ $# -ge 1 ] && [ -n "$1" ]; then DIR="$1"; mkdir -p "$DIR"; else DIR="$(mktemp -d)"; KEEP=0; fi
# the store writes its state under its root's out/: give it a writable root that sees the bundle's web/ and
# test_clips/ (the bundle itself is read-only once signed)
ROOT="$DIR/root"
mkdir -p "$ROOT"
for d in web test_clips models; do [ -e "$ROOT/$d" ] || ln -s "$HERE/$d" "$ROOT/$d"; done

now() { /usr/bin/perl -MTime::HiRes=time -e 'printf "%.2f", time'; }
json_ok() { case "$1" in *'"ok":true'*|*'"ok": true'*) return 0;; *) return 1;; esac; }
t0=$(now)

# 1. the scan: detection (ONNX Runtime, CoreML), analysis, depth, motion, decide: every model the app loads
SCAN=$(MM_SCAN_QUIET=1 "$CLI" scan "$CLIP" "$DIR/project" --motion native --depth anything-small \
        --models "$HERE/models" --cache "$DIR/cache" --engine-cache "$DIR/engine.json" --root "$ROOT" 2>"$DIR/scan.log" | tail -n 1)
t1=$(now)
scan_ok=false; json_ok "$SCAN" && scan_ok=true
[ -n "$SCAN" ] || SCAN='{"ok": false, "error": "no output (see scan.log)"}'

# 2. the export: decode (vtdec), draw, encode (vtenc_h264_hw), mux (mp4mux) with the source's audio
OUT="$DIR/jazz_export.mp4"
EXP=$("$CLI" --root "$ROOT" --video "$ROOT/test_clips/jazz.mp4" call export "{\"out\": \"$OUT\"}" wait-export 2>"$DIR/export.log" | tail -n 1)
t2=$(now)
exp_ok=false
case "$EXP" in *'"state":"done"'*|*'"state": "done"'*) [ -s "$OUT" ] && exp_ok=true;; esac
[ -n "$EXP" ] || EXP='{"error": "no output (see export.log)"}'
BYTES=$( [ -f "$OUT" ] && stat -f %z "$OUT" || echo 0)

ok=false; $scan_ok && $exp_ok && ok=true
printf '{"bundle": "%s", "scan": %s, "scan_seconds": %s, "export": %s, "export_bytes": %s, "export_seconds": %s, "dir": "%s", "ok": %s}\n' \
  "$(cd "$HERE/../.." && pwd)" "$SCAN" "$(printf "%.2f" "$(echo "$t1 - $t0" | bc)")" "$EXP" "$BYTES" "$(printf "%.2f" "$(echo "$t2 - $t1" | bc)")" \
  "$([ $KEEP = 1 ] && echo "$DIR" || echo removed)" "$ok"
[ $KEEP = 1 ] || rm -rf "$DIR"
$ok
