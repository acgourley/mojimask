"""Where projects live, and which project server has one open. Shared by app_server.py and decisions_cli.py.

Standard library only: desktop.py and the command line import it without the engine.
"""
import fcntl
import json
import os
import sys

from mojimask.paths import ROOT as HERE
LIBRARY = os.path.expanduser("~/Library/Application Support/Mojimask/Projects")  # the Mac app's projects
CHECKOUT = os.path.join(HERE, "out", "projects")  # run from a checkout


def fix_folder_case(path):
    """Give an existing folder the name's current case (the app was "MojiMask" until 2026-11-15, now "Mojimask").
    This Mac's disk ignores case, so the old folder still works under the new name, but would keep showing the old one."""
    parent, name = os.path.split(path)
    try:
        for entry in os.listdir(parent):
            if entry != name and entry.lower() == name.lower():
                os.rename(os.path.join(parent, entry), path)
    except OSError:
        pass


def roots(all=False):
    """Project folders. MOJIMASK_PROJECTS when set. Otherwise [the one this app uses]: the library in the Mac app
    (frozen, or MOJIMASK_APP=1), else this checkout's out/projects. all=True: every place projects can be (the library
    and, from a checkout, out/projects) that exists, for tools that look for a project wherever it is."""
    if os.environ.get("MOJIMASK_PROJECTS"):
        return [os.environ["MOJIMASK_PROJECTS"]]
    frozen = bool(getattr(sys, "frozen", False))
    if not all:
        return [LIBRARY if frozen or os.environ.get("MOJIMASK_APP") == "1" else CHECKOUT]
    return [r for r in [LIBRARY] + ([] if frozen else [CHECKOUT]) if os.path.isdir(r)]


def server_program(review_server_py):
    """Which program a project server runs, written in its lock: the app's executable when bundled, else
    review_server.py, each with its modification time. The app only reuses a running server whose program is the
    one it would start itself, so a server left running by an older build (or another checkout) is never
    shown with this build's pages around it."""
    path = os.path.realpath(sys.executable if getattr(sys, "frozen", False) else review_server_py)
    try:
        return {"program": path, "built": int(os.path.getmtime(path))}
    except OSError:
        return {"program": path}


def lock_holder(proj):
    """The project server holding this project's lock ({"pid", "port"}), or None if no server has it open.
    review_server.py takes the lock at start and the OS releases it when that process ends."""
    path = os.path.join(proj, ".server.lock")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.seek(0)
            try:
                return json.loads(f.read() or "{}") or None
            except ValueError:
                return None
        fcntl.flock(f, fcntl.LOCK_UN)
        return None
