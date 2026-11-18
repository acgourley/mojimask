#!/usr/bin/env python3
"""Make a .app self-contained: copy every Homebrew library its Mach-O files load into Contents/Frameworks and point
them at @rpath, so nothing under /opt/homebrew is read at run time.

    relocate.py APP [--prefix /opt/homebrew]

Walks every Mach-O file already in the bundle (the executables in MacOS/, the Qt and GStreamer plugins make_app.sh
copied into PlugIns/, the QML plugins in Resources/qml/) and, for each load command naming a file under PREFIX
(directly, or through @rpath / @loader_path resolved against the file's own LC_RPATHs):
- copies that file into Contents/Frameworks once, keyed by its real path (Homebrew names one library several ways:
  /opt/homebrew/opt/gstreamer/lib/..., /opt/homebrew/Cellar/gstreamer/1.28.9/lib/..., @rpath/...). A Qt framework
  is copied as a minimal framework bundle (Versions/A/<binary> + Resources/Info.plist, no headers);
- rewrites the load command to @rpath/<name> (or @rpath/QtX.framework/Versions/A/QtX), sets each copied library's
  own id the same way;
- replaces the file's LC_RPATHs with one @loader_path- or @executable_path-relative path to Contents/Frameworks.
Then it repeats for the copies until nothing new is found, and checks with otool that no PREFIX path remains
(exit 1 if one does). Signing is make_app.sh's job (install_name_tool breaks the signatures; it re-signs after).

Only the standard library; prints a JSON summary.
"""
import json
import os
import re
import shutil
import subprocess
import sys

MAGICS = {b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xfe\xed\xfa\xcf"}


def is_macho(p):
    if os.path.islink(p) or not os.path.isfile(p):
        return False
    with open(p, "rb") as f:
        return f.read(4) in MAGICS


def otool_deps(p):
    """The libraries p loads (otool -L lists a dylib's own id first: left out)."""
    out = subprocess.run(["otool", "-L", p], capture_output=True, text=True, check=True).stdout.splitlines()[1:]
    deps = [l.strip().split(" (compatibility")[0] for l in out if l.strip()]
    i = own_id(p)
    if i and deps and deps[0] == i:
        deps = deps[1:]
    return deps


def own_id(p):
    out = subprocess.run(["otool", "-D", p], capture_output=True, text=True, check=True).stdout.splitlines()
    return out[1].strip() if len(out) > 1 else None


def rpaths(p):
    out = subprocess.run(["otool", "-l", p], capture_output=True, text=True, check=True).stdout
    return re.findall(r"cmd LC_RPATH\n\s+cmdsize \d+\n\s+path (.+?) \(offset", out)


def resolve(dep, src_path, src_rpaths, exe_dir):
    """The file a load command names, on this machine (None when it can't be found)."""
    here = os.path.dirname(src_path)
    if dep.startswith("@rpath/"):
        for rp in src_rpaths:
            base = rp.replace("@loader_path", here).replace("@executable_path", exe_dir)
            c = os.path.join(base, dep[len("@rpath/"):])
            if os.path.exists(c):
                return os.path.realpath(c)
        return None
    if dep.startswith("@loader_path/"):
        c = os.path.join(here, dep[len("@loader_path/"):])
        return os.path.realpath(c) if os.path.exists(c) else None
    if dep.startswith("@executable_path/"):
        c = os.path.join(exe_dir, dep[len("@executable_path/"):])
        return os.path.realpath(c) if os.path.exists(c) else None
    return os.path.realpath(dep) if os.path.exists(dep) else None


def framework_parts(real):
    """(QtX.framework, 'QtX.framework/Versions/A/QtX') for a framework binary, else None."""
    m = re.search(r"/([^/]+\.framework)/Versions/([^/]+)/([^/]+)$", real)
    if not m:
        return None
    return m.group(1), f"{m.group(1)}/Versions/{m.group(2)}/{m.group(3)}"


def main():
    app = os.path.abspath(sys.argv[1])
    prefix = "/opt/homebrew"
    if "--prefix" in sys.argv:
        prefix = sys.argv[sys.argv.index("--prefix") + 1]
    contents = os.path.join(app, "Contents")
    fw = os.path.join(contents, "Frameworks")
    exe_dir = os.path.join(contents, "MacOS")
    os.makedirs(fw, exist_ok=True)

    copied = {}  # real source path -> (bundle path of the binary, @rpath name)
    todo = []
    for d, _, fs in os.walk(contents):
        for f in fs:
            p = os.path.join(d, f)
            if is_macho(p):
                todo.append(p)
    # remember where each file came from (its original rpaths resolve its @rpath deps), before rewriting anything
    origin = {p: p for p in todo}
    orig_rpaths = {p: rpaths(p) for p in todo}

    def bring(real):
        if real in copied:
            return copied[real][1]
        fp = framework_parts(real)
        if fp:
            fwname, rel = fp
            dest = os.path.join(fw, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(real, dest)
            os.chmod(dest, 0o755)
            # a framework bundle needs Resources/Info.plist to be signed; nothing else (no headers)
            src_fw = real[: real.index(fwname) + len(fwname)]
            ver = rel.split("/")[2]
            plist = os.path.join(src_fw, "Versions", ver, "Resources", "Info.plist")
            if os.path.exists(plist):
                os.makedirs(os.path.join(fw, fwname, "Versions", ver, "Resources"), exist_ok=True)
                shutil.copy2(plist, os.path.join(fw, fwname, "Versions", ver, "Resources", "Info.plist"))
            bin_name = os.path.basename(rel)
            for link, target in (("Current", ver),):
                lp = os.path.join(fw, fwname, "Versions", link)
                if not os.path.lexists(lp):
                    os.symlink(target, lp)
            for name in (bin_name, "Resources"):
                lp = os.path.join(fw, fwname, name)
                if not os.path.lexists(lp) and os.path.exists(os.path.join(fw, fwname, "Versions", "Current", name)):
                    os.symlink(f"Versions/Current/{name}", lp)
            name = "@rpath/" + rel
        else:
            dest = os.path.join(fw, os.path.basename(real))
            # Homebrew's file names carry the full version (libfoo.1.2.3.dylib) while load commands name the
            # compatibility one (libfoo.1.dylib): keep the name the load commands use, decided by the caller
            shutil.copy2(real, dest)
            os.chmod(dest, 0o755)
            name = "@rpath/" + os.path.basename(real)
        copied[real] = (dest, name)
        origin[dest] = real
        orig_rpaths[dest] = rpaths(real)
        todo.append(dest)
        return name

    # Prefer the name a load command uses for a plain dylib (libglib-2.0.0.dylib, not the versioned real file).
    def bring_dep(dep, real):
        if real in copied:
            return copied[real][1]
        if not framework_parts(real) and not dep.startswith("@"):
            # copy under the load command's basename
            base = os.path.basename(dep)
            dest = os.path.join(fw, base)
            if any(c[0] == dest for c in copied.values()):
                raise SystemExit(f"two different libraries named {base}: {real}")
            shutil.copy2(real, dest)
            os.chmod(dest, 0o755)
            name = "@rpath/" + base
            copied[real] = (dest, name)
            origin[dest] = real
            orig_rpaths[dest] = rpaths(real)
            todo.append(dest)
            return name
        return bring(real)

    done = set()
    changes = 0
    while todo:
        p = todo.pop()
        if p in done:
            continue
        done.add(p)
        src = origin.get(p, p)
        args = []
        for dep in otool_deps(p):
            if dep.startswith("/System/") or dep.startswith("/usr/lib/"):
                continue
            real = resolve(dep, src, orig_rpaths.get(p, []), exe_dir)
            if real is None and dep.startswith("@rpath/") and not os.path.exists(os.path.join(fw, dep[7:])):
                # a file copied out of Homebrew (a QML plugin): its rpaths pointed into Homebrew's tree
                c = os.path.join(prefix, "lib", dep[7:])
                real = os.path.realpath(c) if os.path.exists(c) else None
            if real is None:
                if dep.startswith("@rpath/") or dep.startswith("@loader_path"):
                    # already relative (one of ours, or resolved by a later copy); check at the end
                    continue
                raise SystemExit(f"{p}: can't find {dep}")
            if not real.startswith(prefix) and not real.startswith(os.path.realpath(prefix)):
                continue
            new = bring_dep(dep, real)
            if new != dep:
                args += ["-change", dep, new]
        # own id
        i = own_id(p)
        if i and p.startswith(fw + "/") and (i.startswith(prefix) or i.startswith("@rpath")):
            args += ["-id", "@rpath/" + os.path.relpath(p, fw)]
        elif i and i.startswith(prefix):
            # a plugin (dlopen()ed by path): its id only has to stop naming Homebrew
            args += ["-id", "@loader_path/" + os.path.basename(p)]
        # rpaths: drop all, add the one to Frameworks (only for files that load something through @rpath)
        old_rp = rpaths(p)
        uses_rpath = bool(args) or any(d.startswith("@rpath/") for d in otool_deps(p))
        if not uses_rpath and not old_rp:
            continue
        for rp in old_rp:
            args += ["-delete_rpath", rp]
        here = os.path.dirname(p)
        rel = os.path.relpath(fw, here)
        args += ["-add_rpath", "@loader_path/" + rel if rel != "." else "@loader_path"]
        r = subprocess.run(["install_name_tool", *args, p], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"install_name_tool {p}: {r.stderr}")
        changes += 1

    # every Mach-O in the bundle: no PREFIX left in its load commands or rpaths
    bad = []
    n = 0
    for d, _, fs in os.walk(contents):
        for f in fs:
            p = os.path.join(d, f)
            if not is_macho(p):
                continue
            n += 1
            txt = subprocess.run(["otool", "-l", p], capture_output=True, text=True).stdout
            if prefix in txt:
                bad.append(os.path.relpath(p, contents))
    sources = {os.path.relpath(dest, contents): real for real, (dest, _) in copied.items()}
    print(json.dumps({"machos": n, "rewritten": changes, "copied": len(copied), "sources": sources,
                      "prefix_left": bad, "ok": not bad}, indent=1))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
