#!/usr/bin/env python3
"""THIRD_PARTY_NOTICES.md and licenses/ for the native bundle (Contents/Resources), from what is actually in it.

    notices.py APP ROOT NATIVE

- Rust crates: the normal (non-dev, non-build) dependency closure of mm-app and mm-cli for aarch64-apple-darwin
  (cargo metadata, offline), each crate's licence expression, and its own LICENSE*/COPYING*/NOTICE* files copied
  from the cargo registry to licenses/crates/<name>-<version>/. mojimask's own crates (mm-*) are AGPL-3.0 (LICENSE).
- ONNX Runtime (linked statically by ort): MIT, with its ThirdPartyNotices (the onnxruntime Python package's copy;
  the same upstream project).
- Native libraries: every file relocate.py copied into Contents/Frameworks plus the Qt and GStreamer plugins in
  PlugIns/, grouped by the Homebrew formula they came from (dist/relocate.json's sources), with the formula's
  licence (FORMULA_LICENCES below: Homebrew's own metadata, written down so no network is needed) and its licence
  files from the Cellar. Qt and GLib ship none there: their LGPL texts are in licenses/ (LGPL-2.1, LGPL-3.0 with
  GPL-3.0, which LGPL-3.0 builds on).
- Models, fonts, pictures and the sample clip: the same rows as the Python app's THIRD_PARTY_NOTICES.md (copied from
  ROOT/THIRD_PARTY_NOTICES.md, so the two stay one source), with packaging/licenses.py's MIT texts for RetinaFace and
  OSNet.
Fails (exit 1) on a library whose formula has no licence entry, so a new dependency can't ship unlisted.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys

# the main checkout (a worktree has no venv or models of its own): the parent of git's common directory
MAIN_CHECKOUT = os.path.dirname(subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse",
                                                "--path-format=absolute", "--git-common-dir"],
                                               capture_output=True, text=True).stdout.strip())

FORMULA_LICENCES = {
    "qtbase": "LGPL-3.0-only (or GPL-2.0/3.0); dynamically linked, replaceable in Contents/Frameworks",
    "qtdeclarative": "LGPL-3.0-only (or GPL-2.0/3.0); dynamically linked",
    "qthttpserver": "LGPL-3.0-only (or GPL-2.0/3.0); dynamically linked",
    "qtwebsockets": "LGPL-3.0-only (or GPL-2.0/3.0); dynamically linked",
    "qtsvg": "LGPL-3.0-only (or GPL-2.0/3.0); dynamically linked",
    "gstreamer": "LGPL-2.1-or-later (core, base, good, bad plugins used here); dynamically linked",
    "glib": "LGPL-2.1-or-later",
    "gettext": "LGPL-2.1-or-later (libintl)",
    "orc": "BSD-2-Clause",
    "icu4c": "Unicode-3.0 (ICU licence); its data trimmed (trim_icu.py)",
    "pcre2": "BSD-3-Clause WITH PCRE2-exception",
    "openssl": "Apache-2.0",
    "zstd": "BSD-3-Clause OR GPL-2.0-only",
    "double-conversion": "BSD-3-Clause",
    "libb2": "CC0-1.0",
    "brotli": "MIT",
    "dbus": "AFL-2.1 OR GPL-2.0-or-later",
    "libpng": "Libpng-2.0",
    "harfbuzz": "MIT",
    "md4c": "MIT",
    "freetype": "FTL OR GPL-2.0-or-later",
    "graphite2": "LGPL-2.1-or-later OR MPL-2.0 OR GPL-2.0-or-later",
    "libx11": "MIT and X11-style",
    "libxcb": "X11",
    "libxau": "MIT",
    "libxdmcp": "MIT",
    "jpeg-turbo": "IJG AND BSD-3-Clause AND Zlib",
    "libffi": "MIT",
    "pcre": "BSD-3-Clause",
}
MAIN = MAIN_CHECKOUT


def formula(real):
    m = re.search(r"/Cellar/([^/]+)/([^/]+)/", real)
    if not m:
        return None, None
    name = re.sub(r"@\d+$", "", m.group(1))
    return name, (m.group(1), m.group(2))


def section(md, start, end):
    i = md.index(start)
    j = md.index(end, i)
    return md[i:j].rstrip() + "\n"


def main():
    app, root, native = sys.argv[1:4]
    res = os.path.join(app, "Contents", "Resources")
    lic = os.path.join(res, "licenses")
    shutil.rmtree(lic, ignore_errors=True)
    os.makedirs(lic)
    missing = []

    # -- standard texts
    std = {
        "AGPL-3.0.txt": os.path.join(root, "LICENSE"),
        "GPL-3.0.txt": os.path.join(root, "packaging", "GPL-3.0.txt"),
        "LGPL-2.1.txt": glob.glob("/opt/homebrew/Cellar/gstreamer/*/LICENSE")[:1],
        "LGPL-3.0.txt": glob.glob("/opt/homebrew/Cellar/gmp/*/COPYING.LESSERv3")[:1],
        "OFL-1.1-Nunito.txt": os.path.join(root, "web", "fonts", "Nunito-OFL.txt"),
        "OFL-1.1-Inter.txt": os.path.join(root, "web", "fonts", "Inter-OFL.txt"),
    }
    for name, src in std.items():
        src = src[0] if isinstance(src, list) and src else src
        if src and isinstance(src, str) and os.path.isfile(src):
            shutil.copy2(src, os.path.join(lic, name))
        else:
            missing.append(f"{name}: not found")
    # MIT texts for the models whose repos state MIT (as packaging/licenses.py writes them)
    sys.path.insert(0, os.path.join(root, "packaging"))
    try:
        import licenses as pylic  # noqa: E402  (its MIT template; it imports only the standard library)
        for name, who in (("MIT-RetinaFace.txt", "2019 biubug6 (Pytorch_Retinaface)"),
                          ("MIT-OSNet.txt", "2018 Kaiyang Zhou (deep-person-reid / torchreid)")):
            open(os.path.join(lic, name), "w").write(pylic.MIT.format(who=who))
    except Exception as e:  # pragma: no cover
        missing.append(f"packaging/licenses.py: {e}")

    # -- crates
    meta = json.loads(subprocess.run(["cargo", "metadata", "--format-version", "1", "--offline",
                                      "--filter-platform", "aarch64-apple-darwin"], cwd=native,
                                     capture_output=True, text=True, check=True).stdout)
    pk = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    seen, st = set(), [i for i, p in pk.items() if p["name"] in ("mm-app", "mm-cli")]
    while st:
        i = st.pop()
        if i in seen:
            continue
        seen.add(i)
        st += [d["pkg"] for d in nodes[i]["deps"] if any(k["kind"] is None for k in d["dep_kinds"])]
    crates = sorted((pk[i] for i in seen), key=lambda p: (p["name"], p["version"]))
    ours = [p for p in crates if p["name"].startswith("mm-")]
    theirs = [p for p in crates if not p["name"].startswith("mm-")]
    crate_rows = []
    for p in theirs:
        d = os.path.dirname(p["manifest_path"])
        files = [f for f in os.listdir(d) if re.match(r"(?i)(licen[cs]e|copying|notice|unlicense)", f) and os.path.isfile(os.path.join(d, f))]
        dest = os.path.join(lic, "crates", f"{p['name']}-{p['version']}")
        if files:
            os.makedirs(dest, exist_ok=True)
            for f in files:
                shutil.copy2(os.path.join(d, f), os.path.join(dest, f))
        if not p.get("license") and not p.get("license_file"):
            missing.append(f"crate {p['name']} {p['version']}: no licence")
        crate_rows.append(f"| {p['name']} | {p['version']} | {p.get('license') or p.get('license_file')} | "
                          f"{'`crates/' + p['name'] + '-' + p['version'] + '/`' if files else 'standard text (SPDX)'} |")

    # the Apache-2.0 text (SFace, Depth Anything): one crate's copy of the standard text
    apache = sorted(glob.glob(os.path.join(lic, "crates", "*", "LICENSE-APACHE")))
    if apache:
        shutil.copy2(apache[0], os.path.join(lic, "Apache-2.0.txt"))
    else:
        missing.append("Apache-2.0.txt: no crate ships LICENSE-APACHE")

    # -- ONNX Runtime (static, through ort-sys)
    ort = os.path.join(MAIN, ".venv/lib/python3.11/site-packages/onnxruntime")
    os.makedirs(os.path.join(lic, "onnxruntime"))
    for f in ("LICENSE", "ThirdPartyNotices.txt"):
        if os.path.isfile(os.path.join(ort, f)):
            shutil.copy2(os.path.join(ort, f), os.path.join(lic, "onnxruntime", f))
        else:
            missing.append(f"onnxruntime {f}")
    ort_ver = next((p["version"] for p in crates if p["name"] == "ort-sys"), "?")

    # -- native libraries
    reloc = json.load(open(os.path.join(native, "dist", "relocate.json")))
    by_formula = {}
    for rel, real in reloc["sources"].items():
        name, cell = formula(real)
        by_formula.setdefault((name, cell), []).append(rel.replace("Frameworks/", ""))
    for kind, d in (("gstreamer", "PlugIns/gstreamer"), ("qtbase", "PlugIns/platforms"), ("qtbase", "PlugIns/imageformats")):
        full = os.path.join(app, "Contents", d)
        for f in sorted(os.listdir(full)):
            real = os.path.realpath(glob.glob(f"/opt/homebrew/share/qt/plugins/*/{f}")[0]) if kind == "qtbase" \
                else os.path.realpath(f"/opt/homebrew/lib/gstreamer-1.0/{f}")
            name, cell = formula(real)
            by_formula.setdefault((name, cell), []).append(f"{d.replace('PlugIns/', 'PlugIns/')}/{f}")
    lib_rows = []
    for (name, cell), files in sorted(by_formula.items(), key=lambda kv: kv[0][0] or ""):
        if name not in FORMULA_LICENCES:
            missing.append(f"formula {name}: no licence entry ({files[:3]})")
            continue
        cdir = f"/opt/homebrew/Cellar/{cell[0]}/{cell[1]}"
        texts = [f for f in os.listdir(cdir) if re.match(r"(?i)(licen|copying|notice|readme\.ijg)", f) and os.path.isfile(os.path.join(cdir, f))]
        if texts:
            dest = os.path.join(lic, "libraries", name)
            os.makedirs(dest, exist_ok=True)
            for f in texts:
                shutil.copy2(os.path.join(cdir, f), os.path.join(dest, f))
        where = f"`libraries/{name}/`" if texts else ("`LGPL-3.0.txt`, `GPL-3.0.txt`" if name.startswith("qt") else "`LGPL-2.1.txt`")
        shown = ", ".join(sorted(set(os.path.basename(f.split(".framework")[0]) if ".framework" in f else os.path.basename(f) for f in files)))
        lib_rows.append(f"| {name} {cell[1]} | {FORMULA_LICENCES[name]} | {shown} | {where} |")

    # -- models, fonts, pictures: the Python app's rows
    py = open(os.path.join(root, "THIRD_PARTY_NOTICES.md")).read()
    models = section(py, "## Models", "## Software")
    pictures = section(py, "## Fonts and pictures", "## mojimask's own licence")

    md = f"""# Third-party notices

mojimask's native Mac app bundles the software, models, fonts and pictures below. Each keeps its own licence; the
texts are in `licenses/` beside this file (`Contents/Resources/licenses`), and mojimask's own licence in `LICENSE`.
Generated by `native/packaging/notices.py` from what the bundle contains.

{models}
## Software: native libraries (`Contents/Frameworks`, `Contents/PlugIns`)

Built by Homebrew; `Contents/MacOS/gst-plugin-scanner` is GStreamer's too. Every one is a separate dynamic library that can be replaced with a compatible build (what the
LGPL asks). Their sources: the Homebrew formula of the same name and version.

| library | licence | files | text in `licenses/` |
|---|---|---|---|
{chr(10).join(lib_rows)}

## Software: linked into the app (`Contents/MacOS`)

| component | licence | text in `licenses/` |
|---|---|---|
| ONNX Runtime (Microsoft; prebuilt static library through the `ort` crate, ort-sys {ort_ver}), with its own third-party notices | MIT | `onnxruntime/` |
| mojimask's own crates ({", ".join(p["name"] for p in ours)}) | AGPL-3.0-or-later | `AGPL-3.0.txt` |

### Rust crates ({len(theirs)})

| crate | version | licence | text in `licenses/` |
|---|---|---|---|
{chr(10).join(crate_rows)}

{pictures}
## mojimask's own licence

mojimask is free software under the GNU Affero General Public License 3.0 (`LICENSE`), with no warranty. The app as a
whole is distributed on AGPL-3.0 terms: anyone who has it may get its complete source (this repository, with
`native/packaging/` for building it and `models/README.md` for the model files) and change and share it under the
same terms.
"""
    open(os.path.join(res, "THIRD_PARTY_NOTICES.md"), "w").write(md)
    n = sum(len(fs) for _, _, fs in os.walk(lic))
    print(json.dumps({"crates": len(theirs), "library_groups": len(lib_rows), "licence_files": n, "missing": missing}))
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
