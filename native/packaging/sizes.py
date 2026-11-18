#!/usr/bin/env python3
"""The bundle's size by part, in MB (on-disk bytes of regular files; symlinks not counted), as JSON.

    sizes.py native/dist/Mojimask.app
"""
import json
import os
import re
import sys


def size(path):
    if os.path.isfile(path) and not os.path.islink(path):
        return os.path.getsize(path)
    t = 0
    for d, _, fs in os.walk(path):
        for f in fs:
            p = os.path.join(d, f)
            if not os.path.islink(p):
                t += os.path.getsize(p)
    return t


def mb(b):
    return round(b / 1e6, 1)


def main():
    c = os.path.join(sys.argv[1], "Contents")
    fw = os.path.join(c, "Frameworks")
    groups = {"qt": 0, "icu": 0, "gstreamer": 0, "glib/gettext/ffi": 0, "other libs": 0}
    per = {}
    for f in sorted(os.listdir(fw)):
        s = size(os.path.join(fw, f))
        per[f] = mb(s)
        if f.startswith("Qt"):
            groups["qt"] += s
        elif f.startswith("libicu"):
            groups["icu"] += s
        elif f.startswith("libgst") or f.startswith("liborc"):
            groups["gstreamer"] += s
        elif re.match(r"lib(glib|gobject|gio|gmodule|gthread|intl|ffi|pcre2-8)", f):
            groups["glib/gettext/ffi"] += s
        else:
            groups["other libs"] += s
    r = os.path.join(c, "Resources")
    out = {
        "total": mb(size(c)),
        "MacOS": {f: mb(size(os.path.join(c, "MacOS", f))) for f in sorted(os.listdir(os.path.join(c, "MacOS")))},
        "Frameworks": {k: mb(v) for k, v in groups.items()},
        "PlugIns": {f: mb(size(os.path.join(c, "PlugIns", f))) for f in sorted(os.listdir(os.path.join(c, "PlugIns")))},
        "Resources": {f: mb(size(os.path.join(r, f))) for f in sorted(os.listdir(r))},
        "largest_frameworks": dict(sorted(per.items(), key=lambda kv: -kv[1])[:15]),
    }
    out["code (all but models, clip)"] = mb(size(c) - size(os.path.join(r, "models")) - size(os.path.join(r, "test_clips")))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
