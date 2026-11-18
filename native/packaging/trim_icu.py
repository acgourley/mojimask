#!/usr/bin/env python3
"""Replace the bundle's ICU data library (Homebrew's libicudata, 34 MB) with one holding only what Qt uses (~10 MB).

    trim_icu.py APP

Why ICU is there at all: Homebrew's QtCore links ICU, so it can't be left out without a Qt built with -no-icu. What
QtCore calls (nm -u QtCore, Qt 6.12): ucnv_* (QStringConverter's non-UTF encodings), ucol_* (QCollator) and
u_strToLower/Upper (QLocale's case mapping); nothing else. Qt keeps its own CLDR tables for QLocale, so ICU's
display names (lang/, region/, curr/, unit/, zone/), number spell-out (rbnf/), break iterators (brkitr/),
transliterators and the per-locale .res files (all but root and en) are never read.

Kept: every converter (*.cnv, cnvalias.icu), every collation tailoring (coll/), normalization (*.nrm, *.spp),
root/en and the structural tables (res_index, pool, supplementalData, likelySubtags, metadata, keyTypeData, ...).
Checked: ucol_open/strcoll (root, de), ucnv_open (UTF-8, Shift_JIS, windows-1252) and u_strToUpper("tr") give the
same results as with the full data (2026-11-17).

How: the data is one symbol (_icudtNN_dat) in libicudata's __TEXT,__const; it is cut out, trimmed with icupkg -r
(Homebrew's icu4c, build time only), and assembled back into a dylib with the same install name and versions.
"""
import glob
import os
import re
import subprocess
import sys
import tempfile

KEEP_TOP = {"root.res", "res_index.res", "pool.res", "supplementalData.res", "likelySubtags.res", "metadata.res",
            "keyTypeData.res", "en.res", "en_US.res", "en_US_POSIX.res", "icustd.res", "icuver.res", "plurals.res",
            "grammaticalFeatures.res", "zoneinfo64.res", "timezoneTypes.res", "windowsZones.res", "numberingSystems.res"}
DROP_DIRS = ("curr", "zone", "region", "lang", "unit", "rbnf", "brkitr", "translit")


def main():
    app = sys.argv[1]
    fw = os.path.join(app, "Contents", "Frameworks")
    libs = glob.glob(os.path.join(fw, "libicudata.*.dylib"))
    if not libs:
        print('{"icu": "none in bundle"}')
        return
    lib = libs[0]
    ver = re.search(r"libicudata\.(\d+)\.dylib", lib).group(1)
    icu = sorted(glob.glob(f"/opt/homebrew/opt/icu4c@{ver}/sbin/icupkg") + glob.glob("/opt/homebrew/opt/icu4c/sbin/icupkg"))
    if not icu:
        sys.exit(f"no icupkg for ICU {ver}")
    sym = f"_icudt{ver}_dat"
    nm = subprocess.run(["nm", "-m", lib], capture_output=True, text=True, check=True).stdout
    if sym not in nm:
        sys.exit(f"{lib}: no {sym}")
    sect = subprocess.run(["otool", "-l", lib], capture_output=True, text=True, check=True).stdout
    m = re.search(r"sectname __const\n\s+segname __TEXT\n\s+addr \S+\n\s+size (0x[0-9a-f]+)\n\s+offset (\d+)", sect)
    size, off = int(m.group(1), 16), int(m.group(2))
    before = os.path.getsize(lib)
    with tempfile.TemporaryDirectory() as t:
        name = f"icudt{ver}l"
        dat = os.path.join(t, f"{name}.dat")
        with open(lib, "rb") as f:
            f.seek(off)
            open(dat, "wb").write(f.read(size))
        items = subprocess.run([icu[0], "-l", dat], capture_output=True, text=True, check=True).stdout.split()
        drop = [i for i in items if ("/" in i and i.split("/")[0] in DROP_DIRS)
                or ("/" not in i and i.endswith(".res") and i not in KEEP_TOP)]
        open(os.path.join(t, "drop.txt"), "w").write("\n".join(drop) + "\n")
        os.makedirs(os.path.join(t, "out"))
        out = os.path.join(t, "out", f"{name}.dat")
        subprocess.run([icu[0], "-r", os.path.join(t, "drop.txt"), dat, out], check=True)
        asm = os.path.join(t, "data.s")
        open(asm, "w").write(f'.section __TEXT,__const\n.globl {sym}\n.p2align 4\n{sym}:\n.incbin "{out}"\n')
        idname = subprocess.run(["otool", "-D", lib], capture_output=True, text=True).stdout.splitlines()[1].strip()
        new = os.path.join(t, "lib.dylib")
        subprocess.run(["clang", "-dynamiclib", "-arch", "arm64", "-mmacosx-version-min=15.0", "-install_name", idname,
                        "-compatibility_version", f"{ver}.0.0", "-current_version", f"{ver}.1.0", "-o", new, asm], check=True)
        os.replace(new, lib)
    print(f'{{"icu": "{os.path.basename(lib)}", "items_dropped": {len(drop)}, "items_kept": {len(items) - len(drop)}, '
          f'"mb_before": {before / 1e6:.1f}, "mb_after": {os.path.getsize(lib) / 1e6:.1f}}}')


if __name__ == "__main__":
    main()
