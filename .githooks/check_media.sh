#!/bin/bash
# Refuse pictures, videos and scan output in git: people's faces never go in the repo (AGENTS.md, Privacy).
#
#     .githooks/check_media.sh          # staged files (the pre-commit hook runs this)
#     .githooks/check_media.sh --all    # every tracked file (CI runs this)
#
# .gitignore already leaves these out, but `git add -f` gets past it; this check doesn't. To allow a new picture
# (public domain or our own, never a person from real footage), add its path to allowed() and say why in the commit.
set -euo pipefail

allowed() {
  case "$1" in
    native/web/assets/gallery/*.jpg) ;;        # old paintings, public domain (THIRD_PARTY_NOTICES.md)
    native/web/assets/icon_art.jpg) ;;         # the app icon's painting, public domain
    native/web/assets/portrait_mona.png) ;;    # the default portrait: the Mona Lisa's head, public domain
    test_clips/jazz.mp4) ;;                    # Our Dancing Daughters (1928), US public domain: the sample clip
    docs/media/jazz-walkthrough.mp4) ;;        # the website's screen recording of the app on jazz.mp4
    docs/media/jazz-walkthrough-poster.jpg) ;; # its poster frame
    docs/img/*.webp) ;;                        # the website's pictures: public-domain paintings with emoji
    docs/img/social-card.jpg) ;;               # the website's link-preview picture: a public-domain painting
    docs/img/favicon.png|docs/img/apple-touch-icon.png) ;;  # the app icon (a public-domain painting with an emoji)
    *) return 1 ;;
  esac
}

refused() {
  allowed "$1" && return 1
  case "$1" in out/*|test_clips/*) return 0 ;; esac   # projects, frames, crops, census of real footage, other clips
  local name="${1##*/}"
  [[ "$name" == *.* ]] || return 1
  case "$(printf '%s' "${name##*.}" | tr '[:upper:]' '[:lower:]')" in
    mp4|mov|m4v|avi|mkv|webm|heic|heif|jpg|jpeg|png|webp|gif|bmp|tif|tiff|npy|npz) return 0 ;;
  esac
  return 1
}

if [ "${1:-}" = "--all" ]; then list=(git ls-files); else list=(git diff --cached --name-only --diff-filter=ACMR); fi
bad=()
while IFS= read -r p; do refused "$p" && bad+=("$p"); done < <("${list[@]}")
if [ ${#bad[@]} -gt 0 ]; then
  echo "Refused: pictures, videos or scan output (people's faces never go in git; see AGENTS.md, Privacy):"
  printf '  %s\n' "${bad[@]}"
  echo "Unstage them with \`git restore --staged <file>\`. If one is public domain, add it to allowed() in .githooks/check_media.sh."
  exit 1
fi
