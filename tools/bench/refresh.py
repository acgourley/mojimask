"""Bring a scanned project up to today's decide step (covers, hidden boxes, every question), keeping your answers.

Facts (facts.py) name box ids, so a project can take a newer decide step without a rescan as long as its boxes come
out with exactly the same detections (the grouping unchanged; covers, hidden boxes and questions may change). This
re-runs the decide step in a scratch copy that has none of the project's answers (as the scan itself saw it), checks
every track id and detection matches proto_tracks.json, and only then replaces PROJECT/proto_tracks.json (keeping its
scan version) and PROJECT/decisions_*.json. Your answers live in facts.json and still apply. Then restart the app
server: the Decisions page reads the new questions at once, but the project's server loaded the covers when it started.

    python tools/bench/refresh.py PROJECT
    MOJIMASK_DEPTH=anything-small python tools/bench/refresh.py PROJECT    (with a depth cue: depth.py)
"""
import fcntl
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mojimask.scan import newalgo as NA  # noqa: E402
from mojimask.scan import store  # noqa: E402

SKIP = {"facts.json", "edits.json", "proto_tracks.json", ".server.lock", "review_server.log"}  # and facts.discarded*


def server_running(proj):
    """A review server has this project open (it holds PROJECT/.server.lock while it runs)."""
    path = os.path.join(proj, ".server.lock")
    if not os.path.exists(path):
        return False
    with open(path) as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
    return False


def refresh(proj):
    proj = os.path.abspath(proj.rstrip("/"))
    with open(os.path.join(proj, "proto_tracks.json")) as f:
        old = json.load(f)
    # the scratch copy goes in the system's temp folder, not beside the project: the app lists every folder in the
    # projects folder that has a meta.json, and would offer this one as a project
    tmp = tempfile.mkdtemp(prefix="mojimask_refresh_")
    try:
        for name in os.listdir(proj):
            if name in SKIP or name.startswith("facts.discarded") or (name.startswith("decisions_") and name.endswith(".json")):
                continue
            os.symlink(os.path.join(proj, name), os.path.join(tmp, name))
        tracks, hidden, depth = NA.tracks_for(tmp)
        a = {k: t["obs_c"] for k, t in old["tracks"].items()}
        b = {k: t["obs_c"] for k, t in tracks.items()}
        if a != b:
            diff = sorted(set(a) ^ set(b))[:10] or [k for k in a if a[k] != b.get(k)][:10]
            raise SystemExit(f"the boxes came out differently (e.g. {diff}): not refreshed; this needs a rescan")
        doc = {**old, "tracks": tracks, "hidden": sorted(hidden)}
        doc.pop("depth", None)
        if depth:
            doc["depth"] = depth  # the depth model the covers were made with (MOJIMASK_DEPTH)
        store.write_json(os.path.join(proj, "proto_tracks.json"), doc)
        for name in os.listdir(tmp):
            if name.startswith("decisions_") and name.endswith(".json") and not os.path.islink(os.path.join(tmp, name)):
                shutil.copy(os.path.join(tmp, name), os.path.join(proj, name))
        held = lambda d: sum(len(t["cover"]) - len(t["obs_c"]) for t in d.values())  # noqa: E731
        return {"held_frames": [held(old["tracks"]), held(tracks)],
                "behind_frames": sum(len(t.get("behind", [])) for t in tracks.values()),
                "hidden": [len(old.get("hidden", [])), len(hidden)],
                **({"note": "the project's server is running: restart the app server to load the new covers"}
                   if server_running(proj) else {})}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print(refresh(sys.argv[1]))
