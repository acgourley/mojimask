#!/usr/bin/env python3
"""Refuse pictures, videos and scan output in git: people's faces never go in the repo (AGENTS.md, Privacy).

    python3 .githooks/check_media.py            # staged files (the pre-commit hook runs this)
    python3 .githooks/check_media.py --all      # every tracked file (CI runs this)

.gitignore already leaves these out, but `git add -f` gets past it; this check doesn't. To allow a new picture
(public domain or our own, never a person from real footage), add its path to ALLOWED and say why in the commit.
"""
import fnmatch
import subprocess
import sys

MEDIA = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".heic", ".heif", ".jpg", ".jpeg", ".png", ".webp",
         ".gif", ".bmp", ".tif", ".tiff", ".npy", ".npz"}
ALLOWED = [
    "web/assets/gallery/*.jpg",   # old paintings, public domain (THIRD_PARTY_NOTICES.md)
    "web/assets/icon_art.jpg",    # the app icon's painting, public domain
    "web/assets/portrait_mona.png",  # the default portrait: the Mona Lisa's head, public domain
    "test_clips/jazz.mp4",    # Our Dancing Daughters (1928), US public domain: the sample clip
    "docs/media/jazz-walkthrough.mp4",         # the website's screen recording of the app on jazz.mp4
    "docs/media/jazz-walkthrough-poster.jpg",  # its poster frame
    "docs/img/*.webp",        # the website's pictures: public-domain paintings with emoji, the app shown on them
    "docs/img/social-card.jpg",       # the website's link-preview picture: a public-domain painting with emoji
    "docs/img/favicon.png",           # the app icon (a public-domain painting with an emoji), small
    "docs/img/apple-touch-icon.png",  # the same, for iOS home screens
]
NEVER = ["out/*", "test_clips/*"]  # projects, frames, crops, census of real footage, other clips


def files(all_tracked):
    cmd = ["git", "ls-files"] if all_tracked else ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.splitlines()


def refused(path):
    if any(fnmatch.fnmatch(path, a) for a in ALLOWED):
        return False
    ext = path[path.rfind("."):].lower() if "." in path.rsplit("/", 1)[-1] else ""
    return ext in MEDIA or any(fnmatch.fnmatch(path, n) for n in NEVER)


bad = [p for p in files("--all" in sys.argv) if refused(p)]
if bad:
    print("Refused: pictures, videos or scan output (people's faces never go in git; see AGENTS.md, Privacy):")
    for p in bad:
        print("  " + p)
    print("Unstage them with `git restore --staged <file>`. If one is public domain, add it to ALLOWED in "
          ".githooks/check_media.py.")
    sys.exit(1)
