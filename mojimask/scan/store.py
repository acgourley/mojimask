"""Where the scan keeps a project's intermediate files: PROJECT/proto/<kind>.<ext>, inside the project (so a project
needs nothing outside its folder).
"""
import json
import os


def path(proj, kind, ext, write=False):
    p = os.path.join(proj.rstrip("/"), "proto", f"{kind}.{ext}")
    if write:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def write_json(path, obj, **kw):
    """json.dump through a tmp file and os.replace: a crash mid-write leaves the old file, never half a new one."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, **kw)
    os.replace(tmp, path)
