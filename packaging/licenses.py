"""Collect the licence texts of what the Mac app bundles into one folder (build/licenses/), for build_mac.sh to ship
beside THIRD_PARTY_NOTICES.md. Everything comes from files on this machine (packaging/GPL-3.0.txt is kept in the repo, since no package ships it): each Python package's own
licence files (its .dist-info), Python's, the fonts' OFL texts, and one copy of
each licence the models and OpenCV use. Nothing is downloaded.

    .venv/bin/python packaging/licenses.py [OUT (default build/licenses)]

Fails (exit 1) when a text it needs isn't found, so a build can't ship without it.
"""
import glob
import importlib.metadata as md
import os
import shutil
import sys
import sysconfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The third-party Python packages PyInstaller puts in the app (build/pyinstaller/Mojimask/PYZ-00.toc lists them).
PACKAGES = ["numpy", "onnxruntime", "pillow", "pywebview", "bottle", "proxy_tools", "typing_extensions",
            "pyobjc-core", "pyobjc-framework-Cocoa", "pyobjc-framework-AVFoundation",
            "pyobjc-framework-CoreAudio", "pyobjc-framework-CoreMedia", "pyobjc-framework-Metal",
            "pyobjc-framework-Quartz", "pyobjc-framework-UniformTypeIdentifiers", "pyobjc-framework-WebKit"]
MIT = """MIT License

Copyright (c) {who}

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


def licence_files(dist):
    """A package's own licence files: in its dist-info (PEP 639 licenses/ folder, or LICENSE*/COPYING*/NOTICE*),
    else at the top of its package folder (onnxruntime keeps them there)."""
    d = md.distribution(dist)
    named = lambda p: os.path.basename(str(p)).upper().startswith(("LICEN", "COPYING", "NOTICE", "THIRDPARTYNOTICE"))
    files = [p for p in (d.files or []) if os.path.isfile(d.locate_file(p))]
    out = [p for p in files if ".dist-info/" in str(p) and ("/licenses/" in str(p) or named(p))]
    out = out or [p for p in files if str(p).count("/") == 1 and named(p)]
    return [d.locate_file(p) for p in out]


def stated_mit(dist):
    """The MIT text for a package whose metadata says MIT but ships no licence file (PyObjC, proxy_tools), with the
    author its metadata names; None when it doesn't say MIT."""
    m = md.metadata(dist)
    lic = " ".join(filter(None, [m.get("License-Expression"), m.get("License")] + (m.get_all("Classifier") or [])))
    if "MIT" not in lic:
        return None
    who = m.get("Author") or m.get("Author-email") or m.get("Maintainer") or dist
    return MIT.format(who=f"{who} ({dist} {m.get('Version')}; licence stated in the package's metadata)")


def first(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


def main(out):
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    missing = []
    for dist in PACKAGES:
        files = licence_files(dist)
        dest = os.path.join(out, "python-packages", dist)
        os.makedirs(dest, exist_ok=True)
        for f in files:
            shutil.copy2(f, os.path.join(dest, os.path.basename(str(f))))
        if not files:
            text = stated_mit(dist)
            if text is None:
                missing.append(f"{dist}: no licence file, and its metadata doesn't say MIT")
            else:
                with open(os.path.join(dest, "LICENSE-MIT.txt"), "w") as f:
                    f.write(text)
    site = sysconfig.get_paths()["purelib"]
    py = os.path.join(sys.base_prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "LICENSE.txt")
    texts = {
        "Python-PSF-2.0.txt": py,
        # one copy of each licence the models and OpenCV use (same standard texts, from packages that ship them)
        "Apache-2.0.txt": first(os.path.join(site, "onnx-*.dist-info", "licenses", "LICENSE")),
        "AGPL-3.0.txt": first(os.path.join(site, "ultralytics-*.dist-info", "licenses", "LICENSE")),
        "GPL-3.0.txt": os.path.join(ROOT, "packaging", "GPL-3.0.txt"),  # YOLOv5-Face's (SPDX copy, 2026-11-15)
        "OFL-1.1-Nunito.txt": os.path.join(ROOT, "web", "fonts", "Nunito-OFL.txt"),
        "OFL-1.1-Inter.txt": os.path.join(ROOT, "web", "fonts", "Inter-OFL.txt"),
    }
    for name, src in texts.items():
        if src and os.path.isfile(src):
            shutil.copy2(src, os.path.join(out, name))
        else:
            missing.append(f"{name}: not found ({src})")
    for name, who in (("MIT-RetinaFace.txt", "2019 biubug6 (Pytorch_Retinaface)"),
                      ("MIT-OSNet.txt", "2018 Kaiyang Zhou (deep-person-reid / torchreid)")):
        with open(os.path.join(out, name), "w") as f:
            f.write(MIT.format(who=who))
    if missing:
        sys.exit("licences missing:\n  " + "\n  ".join(missing))
    print(f"{out}: {sum(len(fs) for _, _, fs in os.walk(out))} licence files")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "licenses"))
