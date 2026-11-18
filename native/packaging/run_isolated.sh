#!/bin/bash
# Run a bundled program the way a clean Mac would: an empty environment (PATH=/usr/bin:/bin, HOME kept), cwd /, and
# sandbox-exec denying reads of everything a build machine has that a user's Mac doesn't: Homebrew (/opt/homebrew,
# /usr/local), this checkout's web/, models/, test_clips/, .venv and native/target, and the main checkout's models.
# Anything the bundle still loads from outside itself then fails.
#
#   native/packaging/run_isolated.sh native/dist/Mojimask.app/Contents/MacOS/Mojimask --check
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NATIVE="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$NATIVE/.." && pwd)"
# the main checkout (a worktree has no venv or models of its own): the parent of git's common directory, else ROOT
G="$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"; MAIN="${G:+$(dirname "$G")}"; MAIN="${MAIN:-$ROOT}"
deny=""
for p in /opt/homebrew /usr/local "$ROOT/web" "$NATIVE/web" "$ROOT/models" "$ROOT/test_clips" "$ROOT/.venv" "$NATIVE/target" "$NATIVE/app" \
         "$MAIN/models" "$MAIN/.venv" "$MAIN/web"; do
  deny="$deny (subpath \"$p\")"
done
PROG="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; shift
PROFILE="(version 1) (allow default) (deny file-read* $deny)"
cd /
exec env -i HOME="$HOME" PATH=/usr/bin:/bin TMPDIR="${TMPDIR:-/tmp}" sandbox-exec -p "$PROFILE" "$PROG" "$@"
