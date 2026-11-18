"""Shared by the model parity tests (model_*.py): paths, project copies, the Rust binary, fingerprints.

Only jazz (public domain): out/poc/projects/jazz is copied under out/native/model/. Never out/projects/.
"""
import json
import os
import shutil
import subprocess
import sys

# the main checkout (a worktree has no venv or models of its own): the parent of git's common directory
MAIN_CHECKOUT = os.path.dirname(subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse",
                                                "--path-format=absolute", "--git-common-dir"],
                                               capture_output=True, text=True).stdout.strip())

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(NATIVE)
SRC = os.path.join(ROOT, "out", "poc", "projects", "jazz")
OUT = os.path.join(ROOT, "out", "native", "model")
VIDEO = os.path.join(MAIN_CHECKOUT, "test_clips/jazz.mp4")
PY = os.path.join(MAIN_CHECKOUT, ".venv/bin/python")
SKIP = {".server.lock", "review_server.log"}


def copy_project(dst, src=SRC):
    """A fresh copy of the jazz project at dst (the frames are linked, not copied: nothing here writes them)."""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for name in os.listdir(src):
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if name in SKIP:
            continue
        if name == "frames":
            os.symlink(s, d)
        elif os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
    return dst


def rust_bin():
    """mm-cli (its `decisions` subcommand). Built if needed; MM_DECISIONS_BIN overrides (a binary taking the same
    arguments after `decisions`)."""
    if os.environ.get("MM_DECISIONS_BIN"):
        return [os.environ["MM_DECISIONS_BIN"], "decisions"]
    r = subprocess.run(["cargo", "build", "-q", "-p", "mm-cli"], cwd=NATIVE, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("cargo build -p mm-cli failed (set MM_DECISIONS_BIN to a binary built from decisions.rs):\n" + r.stderr[-3000:])
    return [os.path.join(NATIVE, "target", "debug", "mm-cli"), "decisions"]


def fnv(b):
    """FNV-1a 64 (decisions.rs computes the same)."""
    h = 0xcbf29ce484222325
    for x in b:
        h ^= x
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def file_hash(p):
    try:
        with open(p, "rb") as f:
            return fnv(f.read())
    except OSError:
        return "missing"


def load(p):
    with open(p) as f:
        return json.load(f)


def without(obj, keys):
    """obj with these keys dropped at every level (for comparisons that ignore a field, e.g. "at")."""
    if isinstance(obj, dict):
        return {k: without(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [without(v, keys) for v in obj]
    return obj


def report(name, ok, **info):
    out = {"test": name, **info, "pass": bool(ok)}
    print(json.dumps(out, indent=1))
    print(f"{name}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1
