"""Attendee app backend (app.html): import a video, scan it, then hand the project to a per-project
review server (same engine and edit format as the full editor) behind a same-origin proxy.

    .venv/bin/python app_server.py [--port 8800] [--dev]   # development, in a browser
    (the Mac app, desktop.py, imports this module and serves it inside a native window)

Projects: ~/Library/Application Support/Mojimask/Projects/<id>/ in the app (override with
MOJIMASK_PROJECTS); out/projects/ when run from a checkout. Each project folder holds what's worth keeping
(meta.json, edits.json, dets/, mats_v2.npy) plus regenerable cache (frames/, *.npz caches, previews) that
"Clear cache" removes; it's rebuilt when the project is opened again.

API
  POST /api/import            raw video bytes (header X-Filename) -> {id}   (the video is copied in)
  POST /api/import_path       {"path": "/abs/video.mp4"} -> {id}           (the video stays where it is)
                              (a photo works the same way: a clip of one frame, see video_io.py)
  GET  /api/projects          projects with sizes: kept / cache / copy of the video
  GET  /api/status/<id>       scan progress, counters, review server readiness
  GET  /api/projects/<id>/preview.jpg   live scan preview
  GET  /api/projects/<id>/thumb.jpg     the project's picture on the Projects page (made by its review server)
  GET  /api/projects/<id>/head.jpg?t=&x=&y=&s=[&px=]   one head from a scanned frame (the decide step's picture)
  GET  /api/projects/<id>/decide_viz?since=N           what the decide step is doing (live, else the last scan's)
  POST /api/projects/<id>/cancel  stop a scan under way (the project goes to the Trash)
  POST /api/projects/<id>/clear_cache | /delete (to the Trash) | /relink {"path"}
  POST /api/projects/<id>/clone_fresh  developer mode only: a copy with the same scan and cache but none of
                              your answers (no edits.json), so it opens on the result screen like a new project
  GET  /api/app               where projects live, whether this is the Mac app, whether developer mode is on
  GET  /splash                developer mode only: the Mac app's opening page (splash.py)
  *    /p/<id>/<path>         proxied to the project's review server (state, frames, render, ...)

Developer mode (MOJIMASK_DEV=1, or --dev): shows testing tools that people using the app don't need -- Clone
fresh on the Projects list, and in each project's editor the pose overlay and Export labels
(review_server.py reads the same variable; project servers inherit it). Works in the
built app too: MOJIMASK_DEV=1 "/Applications/Mojimask.app/Contents/MacOS/Mojimask".
"""
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from mojimask import http_guard
from mojimask import projects

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")
FROZEN = bool(getattr(sys, "frozen", False))
APP = FROZEN or os.environ.get("MOJIMASK_APP") == "1"
DEV = os.environ.get("MOJIMASK_DEV") == "1"  # developer mode (see the docstring)
STARTED = time.time()
SAMPLE = os.path.join(HERE, "test_clips", "jazz.mp4")  # bundled by packaging/build_mac.sh; checked by desktop.py --check
SAMPLE_EDITS = {"light": {"saturation": 0.0}}  # a black-and-white clip: emoji start gray like the footage
OLD_ROOT = os.path.expanduser("~/Library/Application Support/Party Anonymous")  # the app's name until 2026-11-14
PROJ_ROOT = projects.roots()[0]  # MOJIMASK_PROJECTS, else the app's library (APP) or this checkout's out/projects
from mojimask import pipeline  # noqa: E402

STATE = {}   # id -> {"steps": {...}, "info": {...}, "error": str|None, "review_port": int|None}
REVIEW = {}  # id -> subprocess.Popen
# everything else is cache/copies. The new algorithm's output (proto_tracks.json, decisions_*.json, proto/) is kept too:
# it's regenerable only by a rescan, and a rescan throws away facts.json (your answers and choices, facts.py)
KEEP = {"meta.json", "edits.json", "dets", "mats_v2.npy", ".server.lock",
        "facts.json", "proto_tracks.json", "proto", "thumb.jpg"}
kept = lambda name: name in KEEP or (name.startswith("decisions_") and name.endswith(".json"))  # noqa: E731


def init_projects():
    """Make sure the projects folder exists (serve() calls it, not importing this module)."""
    if APP and not os.environ.get("MOJIMASK_PROJECTS") and os.path.isdir(OLD_ROOT) and not os.path.exists(os.path.dirname(PROJ_ROOT)):
        os.rename(OLD_ROOT, os.path.dirname(PROJ_ROOT))  # projects made under the old name come along (a move, nothing copied)
    if APP and not os.environ.get("MOJIMASK_PROJECTS"):
        projects.fix_folder_case(os.path.dirname(PROJ_ROOT))
    os.makedirs(PROJ_ROOT, exist_ok=True)


def code_changed_since_start():
    """Developer mode: has any of the app's Python changed since this server started? A running server (and the
    project servers it starts) keeps the code it loaded, so a page can look "old" after an edit."""
    newest = 0.0
    for d in (HERE, os.path.join(HERE, "mojimask"), os.path.join(HERE, "mojimask", "scan")):
        for f in os.listdir(d) if os.path.isdir(d) else ():
            if f.endswith(".py"):
                newest = max(newest, os.path.getmtime(os.path.join(d, f)))
    return newest > STARTED


def write_json(path, obj):
    """Replace a file whole (temporary file, then rename): a crash mid-save never leaves half a file."""
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def slug(name):
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.splitext(os.path.basename(name))[0])[:40] or "video"
    return f"{base}_{time.strftime('%Y%m%d-%H%M%S')}"


class BadProject(ValueError):
    """A project id that can't name a project folder (answered 400)."""


def proj_dir(pid):
    if not re.fullmatch(r"[A-Za-z0-9._-]+", pid or "") or pid in (".", ".."):
        raise BadProject("bad project id")
    return os.path.join(PROJ_ROOT, pid)


def scanned(pid):
    d = proj_dir(pid)
    return all(os.path.exists(f"{d}/{f}") for f in ("mats_v2.npy", "dets/pose_head.json", "proto_tracks.json"))


def du(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def sizes(pid):
    """Bytes kept (answers + scan), cache (rebuilt on open) and copy (a video copied in on import)."""
    d = proj_dir(pid)
    meta = _meta(pid)
    copy = meta.get("video") if meta.get("copied") else None
    out = {"kept": 0, "cache": 0, "copy": 0}
    for name in os.listdir(d):
        p = os.path.join(d, name)
        if copy and os.path.abspath(p) == os.path.abspath(copy):
            out["copy"] += du(p)
        elif kept(name):
            out["kept"] += du(p)
        else:
            out["cache"] += du(p)
    return out


def _meta(pid):
    p = os.path.join(proj_dir(pid), "meta.json")
    return read_json(p) if os.path.exists(p) else {}


def clone_fresh(pid):
    """A test copy of a scanned project, without your answers (projects.copy_scan), so it opens like a project just
    scanned."""
    if not scanned(pid):
        return {"error": "only a scanned project can be cloned"}
    meta = _meta(pid)
    new = f"{pid.split('_fresh')[0]}_fresh{time.strftime('%H%M%S')}"
    projects.copy_scan(proj_dir(pid), proj_dir(new),
                       f"{(meta.get('label') or os.path.basename(meta.get('video', pid))).removesuffix(' (fresh test copy)')} (fresh test copy)")
    return {"id": new}


def review_command(args):
    """The review server runs as a child process: `python -m mojimask.review_server ...` from a checkout, or the
    app's own executable with --review-server when bundled."""
    if FROZEN:
        return [sys.executable, "--review-server", *args]
    return [sys.executable, "-m", "mojimask.review_server", *args]


class Adopted:
    """A review server for this project that is already running (started by an earlier or another app
    process): used instead of starting a second one, which would keep its own copy of the edits."""
    adopted = True

    def __init__(self, proj, pid_):
        self.proj, self.pid = proj, pid_

    def poll(self):
        return None if projects.lock_holder(self.proj) else 0

    def terminate(self):
        import signal
        try:
            os.kill(self.pid, signal.SIGTERM)
        except OSError:
            pass

    kill = terminate

    def wait(self, timeout=None):
        for _ in range(int((timeout or 5) * 10)):
            if self.poll() is not None:
                return 0
            time.sleep(0.1)
        raise subprocess.TimeoutExpired("review_server", timeout)


def stop_review(pid):
    p = REVIEW.pop(pid, None)
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            p.kill()
    if pid in STATE:
        STATE[pid]["review_port"] = None


def shutdown():
    for pid in list(REVIEW):
        if not getattr(REVIEW[pid], "adopted", False):  # a server this process didn't start keeps running
            stop_review(pid)


_START_LOCKS, _START_LOCK = {}, threading.Lock()
_STATUS_LOCK = threading.Lock()  # /api/status: whether a project needs its scan or server started


def start_review(pid):
    """Launch the per-project review server (if not running) and wait until it answers. Requests arrive
    in parallel (a page loading many pictures), so only one of them may start it; the rest wait."""
    with _START_LOCK:
        lock = _START_LOCKS.setdefault(pid, threading.Lock())
    with lock:
        _start_review(pid)


def _same_server(held, proj):
    """Whether a running project server is one this app would have started: the same program, unchanged since,
    for this folder. (Locks written before 2026-11-14 say neither, so their servers are never reused.)"""
    want = projects.server_program(os.path.join(HERE, "mojimask", "review_server.py"))
    return all(held.get(k) == v for k, v in want.items()) and held.get("workdir") == os.path.realpath(proj)


def _stop_stale(proj, held):
    """Stop a project server this app can't use, and wait for the OS to release its lock. Its edits are already
    on disk (every change is saved as it's made), so nothing is lost."""
    import signal
    print(f"stopping a project server from another build (pid {held.get('pid')}, {held.get('program', 'unknown program')})")
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.kill(int(held.get("pid")), sig)
        except (OSError, TypeError, ValueError):
            return
        for _ in range(50):
            if not projects.lock_holder(proj):
                return
            time.sleep(0.1)


def _start_review(pid):
    st = STATE.setdefault(pid, {"steps": {}, "info": {}, "error": None, "review_port": None})
    if pid in REVIEW and REVIEW[pid].poll() is None:
        return
    proj = proj_dir(pid)
    meta = _meta(pid)
    if not os.path.isfile(meta.get("video", "")):
        st["error"] = "missing_video"
        return
    st["error"] = None
    held = projects.lock_holder(proj)
    if held and not _same_server(held, proj):
        _stop_stale(proj, held)  # an older build's server (or another checkout's): this build's pages need its own
        held = projects.lock_holder(proj)
    # (the server is kept in a local too: relink, clear cache or delete may take REVIEW[pid] away meanwhile)
    if held and held.get("port"):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{held['port']}/api/render", timeout=5)
            REVIEW[pid] = Adopted(proj, held.get("pid"))
            st["review_port"] = held["port"]; st["steps"]["review"] = 1.0; return
        except Exception:
            pass  # still starting (it holds the lock before it serves): wait for it below
        srv = REVIEW[pid] = Adopted(proj, held.get("pid"))
        for _ in range(900):
            if srv.poll() is not None:
                break
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{held['port']}/api/render", timeout=2)
                st["review_port"] = held["port"]; st["steps"]["review"] = 1.0; return
            except Exception:
                time.sleep(1)
        if REVIEW.get(pid) is srv:
            REVIEW.pop(pid)
    port = free_port()
    with open(f"{proj}/review_server.log", "w") as log:  # (the child keeps its own copy of the file)
        srv = REVIEW[pid] = subprocess.Popen(
            review_command([meta["video"], f"{proj}/dets", "--workdir", proj, "--original", meta["video"],
                            "--port", str(port), "--dets", *pipeline.CORE_DETS]),
            cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
    st["steps"]["review"] = 0.0
    for _ in range(6000):  # ~15 min: building tracks, people links and uncertain spots can take a minute or two
        if srv.poll() is not None:
            st["error"] = f"review server exited (see {proj}/review_server.log)"; return
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/render", timeout=2)
            st["review_port"] = port; st["steps"]["review"] = 1.0; return
        except Exception:
            time.sleep(0.15)  # it starts in ~1 s from a warm cache: don't add a second on top
    st["error"] = "review server did not start"


class ScanCancelled(Exception):
    pass


def scan(pid, video, start_edits=None):
    """start_edits: settings a new project starts with (the sample clip's), added once the scan has written its
    fresh edits.json -- the scan always starts the edits from scratch, so they can't be written before it."""
    proj = proj_dir(pid)
    st = STATE[pid]

    def report(step, frac, info):
        if st.get("cancel"):
            raise ScanCancelled()
        st["steps"][step] = frac
        st["info"].update(info or {})
        st["current"] = step
        st.setdefault("started", {}).setdefault(step, time.time())
    try:
        st["viz"] = []
        pipeline.run(video, proj, report, events=st["viz"].append)
        if start_edits:
            path = os.path.join(proj, "edits.json")
            ed = read_json(path) if os.path.exists(path) else {}
            write_json(path, {**ed, **start_edits})
        if st.get("cancel"):
            raise ScanCancelled()
        st["current"] = "review"
        st["started"].setdefault("review", time.time())
        start_review(pid)
        learn_tail(st)
    except ScanCancelled:  # (the Scan page's Cancel: nothing of the project is worth keeping)
        trash(proj)
        STATE.pop(pid, None)
    except Exception as e:
        try:  # the whole story for whoever looks into it; the page shows the message
            with open(os.path.join(proj, "scan.log"), "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n{traceback.format_exc()}\n")
        except OSError:
            pass
        traceback.print_exc()
        st["error"] = str(e) or type(e).__name__


# ---- time left. After finding faces come following the camera, deciding who's who (face and body likeness for every
# pair) and starting the editor: on party_clip_c (240 frames) 41 s of finding faces, then 1 + 83 + a few s. That tail
# scales with the clip and the machine much as finding faces does, so it's estimated as TAIL_RATIO x the time finding
# faces takes, the ratio learnt from each scan finished on this Mac (scan_timings.json next to the projects folder).
TAIL_RATIO = 2.2
TIMINGS = os.path.join(os.path.dirname(os.path.abspath(PROJ_ROOT)), "scan_timings.json")


TAIL_RANGE = (1.0, 3.5)  # a short clip's tail is mostly fixed start-up time: one learnt 4.2 from a few seconds of faces


def tail_ratio():
    try:
        return min(max(float(read_json(TIMINGS)["tail_ratio"]), TAIL_RANGE[0]), TAIL_RANGE[1])
    except Exception:
        return TAIL_RATIO


def learn_tail(st):
    s = st.get("started", {})
    if not st.get("review_port") or "detect" not in s or "motion" not in s:
        return
    detect, tail = s["motion"] - s["detect"], time.time() - s["motion"]
    if detect < 20:
        return  # too short to say anything (the tail's fixed start-up time would swamp the ratio)
    r = 0.5 * tail_ratio() + 0.5 * min(max(tail / detect, TAIL_RANGE[0]), TAIL_RANGE[1])  # a running average
    try:
        write_json(TIMINGS, {"tail_ratio": round(r, 3)})
    except OSError:
        pass


def time_left(st):
    """Seconds until the project opens (None once past the estimate: "finishing up")."""
    s, i, now = st.get("started", {}), st.get("info", {}), time.time()
    if st.get("review_port") or "detect" not in s:
        return None
    if "motion" not in s:  # finding faces: what's left of it, plus the tail for a scan this long
        if i.get("eta_s") is None:
            return None
        return round(i["eta_s"] + tail_ratio() * (now - s["detect"] + i["eta_s"]))
    left = tail_ratio() * (s["motion"] - s["detect"]) - (now - s["motion"])
    return round(left) if left > 3 else None


def new_project(video_path, name, pid=None, copied=False, edits=None):
    pid = pid or slug(name)
    os.makedirs(proj_dir(pid), exist_ok=True)
    if copied:  # remembered so the storage view can show it as "copy of the video"
        write_json(os.path.join(proj_dir(pid), "meta.json"), {"video": video_path, "copied": True})
    STATE[pid] = {"steps": {}, "info": {}, "error": None, "review_port": None, "current": "prepare", "name": name}
    threading.Thread(target=scan, args=(pid, video_path, edits), daemon=True).start()  # edits: starting settings (sample)
    return pid


def trash(path):
    """Move to the Trash (recoverable), like Finder; plain delete off macOS."""
    try:
        from Foundation import NSFileManager, NSURL
        ok, _, err = NSFileManager.defaultManager().trashItemAtURL_resultingItemURL_error_(
            NSURL.fileURLWithPath_(path), None, None)
        if ok:
            return
    except ImportError:
        pass
    shutil.rmtree(path, ignore_errors=True)


class Handler(http_guard.GuardMixin, SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=WEB, **k)

    def translate_path(self, path):
        """Pages come from web/; the licence texts the Licences page shows are at the root."""
        if path.split("?")[0] in ("/LICENSE", "/THIRD_PARTY_NOTICES.md"):
            return os.path.join(HERE, path.split("?")[0].lstrip("/"))
        return super().translate_path(path)

    def log_message(self, *a):
        pass

    def end_headers(self):
        if http_guard.own_code(self.path) or "/preview.jpg" in self.path or "/thumb.jpg" in self.path:
            self.send_header("Cache-Control", "no-store")
        self._sent = True
        super().end_headers()

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def _body(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

    def _proxy(self, method):
        m = re.match(r"^/p/([^/]+)(/.*)$", self.path)
        if not m:
            return self._json({"error": "not found"}, 404)
        pid, rest = m.group(1), m.group(2)
        if pid not in REVIEW or REVIEW[pid].poll() is not None:
            if not scanned(pid):
                return self._json({"error": "project not scanned"}, 409)
            if (STATE.get(pid) or {}).get("current") in ("prepare", "detect", "motion", "decide"):
                # still scanning: the editor server must not start before the last step has written its tracks
                return self._json({"error": "not ready"}, 409)
            start_review(pid)
        st = STATE.get(pid)  # (gone if the project was relinked, cleared or deleted meanwhile)
        port = st and st.get("review_port")
        if not port:
            return self._json({"error": (st or {}).get("error") or "not ready"}, 409)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0))) if method == "POST" else None
        req = urllib.request.Request(f"http://127.0.0.1:{port}{rest}", data=body, method=method,
                                     headers={"Content-Type": self.headers.get("Content-Type", "application/json")})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                status, data = r.status, r.read()
                headers = [(h, r.headers[h]) for h in ("Content-Type", "Content-Disposition", "Cache-Control") if r.headers.get(h)]
        except urllib.error.HTTPError as e:
            try:  # pass the review server's own message on ({"error": "..."}), not just the status's name
                return self._json(json.loads(e.read()), e.code)
            except ValueError:
                return self._json({"error": e.reason}, e.code)
        except OSError as e:  # (URLError too) the project's server went away or didn't answer in time
            return self._json({"error": f"the project's server didn't answer: {getattr(e, 'reason', e)}"}, 502)
        self.send_response(status)
        for h, v in headers:
            self.send_header(h, v)
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def _guarded(self, handle):
        """Run a request: a bad project id answers 400, anything else that goes wrong 500 (logged)."""
        self._sent = False
        try:
            handle()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the page stopped waiting
        except BadProject as e:
            if not self._sent:
                self._json({"error": str(e)}, 400)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            if not self._sent:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_GET(self):
        self._guarded(self._get)

    def do_POST(self):
        self._guarded(self._post)

    def _get(self):
        if self.path.split("?")[0] in ("/", "/index.html"):
            self.path = "/app.html"
        path = self.path.split("?")[0]
        if self.path.startswith("/p/"):
            return self._proxy("GET")
        if path == "/splash" and DEV:  # the Mac app's opening page, to look at in a browser
            from mojimask import splash
            body = splash.html().encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body))); self.end_headers()
            return self.wfile.write(body)
        if path == "/api/app":
            info = {"app": APP, "dev": DEV, "projects_dir": PROJ_ROOT}
            if DEV:
                info.update(depth=os.environ.get("MOJIMASK_DEPTH") or "off", started=STARTED, stale=code_changed_since_start())
            return self._json(info)
        if path == "/api/projects":
            out = []
            for pid in sorted(os.listdir(PROJ_ROOT)):
                d = os.path.join(PROJ_ROOT, pid)
                if not os.path.exists(f"{d}/meta.json") or (STATE.get(pid) or {}).get("cancel"):
                    continue
                meta = _meta(pid)
                used = max((os.path.getmtime(os.path.join(d, f)) for f in ("edits.json", "meta.json")
                            if os.path.exists(os.path.join(d, f))), default=0)
                out.append({"id": pid, "name": meta.get("label") or os.path.basename(meta.get("video", pid)), "scanned": scanned(pid),
                            "duration": meta.get("duration"), "width": meta.get("width"), "height": meta.get("height"),
                            "photo": bool(meta.get("image")),
                            "video": meta.get("video"), "video_found": os.path.isfile(meta.get("video", "")),
                            "copied": bool(meta.get("copied")), "last_used": used, "sizes": sizes(pid),
                            "path": d, "thumb": int(os.path.getmtime(f"{d}/thumb.jpg")) if os.path.exists(f"{d}/thumb.jpg") else 0})
            out.sort(key=lambda p: -p["last_used"])
            return self._json(out)
        m = re.match(r"^/api/projects/([^/]+)/(preview|thumb).jpg$", path)
        if m:  # the scan's live preview; the project's picture on the Projects page (review_server /api/thumb)
            p = os.path.join(proj_dir(m.group(1)), m.group(2) + ".jpg")
            if not os.path.exists(p):
                return self._json({"error": "no picture yet"}, 404)
            with open(p, "rb") as f:
                data = f.read()
            self.send_response(200); self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data))); self.end_headers()
            return self.wfile.write(data)
        m = re.match(r"^/api/projects/([^/]+)/head.jpg$", path)
        if m:  # one head from a scanned frame, for the decide step's picture: ?t=frame&x=&y=&s= (fractions of the
            # frame's width, as in scan/viz.py), a square `px` wide
            import cv2
            import numpy as np
            q = {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).items()}
            try:
                t, x, y, s, px = int(q["t"]), float(q["x"]), float(q["y"]), float(q["s"]), min(max(int(q.get("px", 96)), 16), 256)
            except (KeyError, ValueError):
                return self._json({"error": "t, x, y, s"}, 400)
            im = cv2.imread(os.path.join(proj_dir(m.group(1)), "frames", f"{t:05d}.jpg"))
            if im is None:
                return self._json({"error": "no frame"}, 404)
            W = im.shape[1]
            r = max(4, s * W * 0.7)
            M = np.float32([[px / (2 * r), 0, px / 2 - x * W * px / (2 * r)], [0, px / (2 * r), px / 2 - y * W * px / (2 * r)]])
            ok, buf = cv2.imencode(".jpg", cv2.warpAffine(im, M, (px, px), flags=cv2.INTER_AREA, borderValue=(0, 0, 0)),
                                   [cv2.IMWRITE_JPEG_QUALITY, 82])
            data = buf.tobytes()
            self.send_response(200); self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "max-age=600"); self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return self.wfile.write(data)
        m = re.match(r"^/api/projects/([^/]+)/decide_viz$", path)
        if m:  # what the decide step is doing (scan/viz.py): this scan's events from `since`, else the
            # last scan's, recorded (a replay)
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            since = int((q.get("since") or ["0"])[0])
            live = (STATE.get(m.group(1)) or {}).get("viz")
            if live is not None:
                return self._json({"events": live[since:], "live": True})
            f = os.path.join(proj_dir(m.group(1)), "proto", "decide_viz.json")
            return self._json({"events": read_json(f)[since:] if os.path.exists(f) else [], "live": False})
        if path.startswith("/api/status/"):
            pid = path.rsplit("/", 1)[1]
            with _STATUS_LOCK:  # checked and started together: two pages asking at once start one scan, not two
                if pid not in STATE and scanned(pid):
                    STATE[pid] = {"steps": {"prepare": 1, "detect": 1, "motion": 1, "done": 1}, "info": {}, "error": None,
                                  "review_port": None, "current": "review"}
                    threading.Thread(target=start_review, args=(pid,), daemon=True).start()
                if pid not in STATE and os.path.isfile(_meta(pid).get("video", "")):
                    # Resume: a scan that never finished (the app was closed, or its server restarted, part-way) and
                    # isn't running here. It starts again; what it had already worked out (frames, detections) is reused.
                    meta = _meta(pid)
                    if "frames" not in meta:  # stopped while still writing its frames: those start over
                        shutil.rmtree(os.path.join(proj_dir(pid), "frames"), ignore_errors=True)
                    STATE[pid] = {"steps": {}, "info": {}, "error": None, "review_port": None, "current": "prepare",
                                  "name": meta.get("label") or os.path.basename(meta["video"])}
                    threading.Thread(target=scan, args=(pid, meta["video"]), daemon=True).start()
            st = STATE.get(pid)
            if not st:
                return self._json({"error": "unknown project"}, 404)
            return self._json({**{k: v for k, v in st.items() if k != "viz"}, "viz_n": len(st.get("viz") or ()), "meta": _meta(pid), "ready": bool(st.get("review_port")), "time_left": time_left(st)})
        if http_guard.static_ok(path):  # the page and what it loads; never projects or code
            return super().do_GET()
        return self._json({"error": "not found"}, 404)

    def _post(self):
        if self.path.startswith("/p/"):
            return self._proxy("POST")
        if self.path == "/api/import":  # dropped into the window: no path, so the video is copied in
            name = os.path.basename(urllib.parse.unquote(self.headers.get("X-Filename", "video.mp4"))) or "video.mp4"  # (sent percent-encoded)
            remaining = int(self.headers.get("Content-Length", 0))
            if not remaining:
                return self._json({"error": "the file arrived empty."}, 400)
            pid = slug(name)
            os.makedirs(proj_dir(pid), exist_ok=True)
            dest = os.path.join(proj_dir(pid), re.sub(r"[^A-Za-z0-9._ -]+", "_", name))
            with open(dest, "wb") as f:
                while remaining:
                    chunk = self.rfile.read(min(remaining, 1 << 20))
                    if not chunk:  # the page went away part-way: nothing of it is kept
                        break
                    f.write(chunk); remaining -= len(chunk)
            if remaining:
                shutil.rmtree(proj_dir(pid), ignore_errors=True)
                return self._json({"error": "the copy was cut off."}, 400)
            return self._json({"id": new_project(dest, name, pid=pid, copied=True)})
        if self.path == "/api/import_path":
            body = self._body()
            if body.get("sample"):  # resolved here: in the app the working directory isn't the bundle
                return self._json({"id": new_project(SAMPLE, os.path.basename(SAMPLE), edits=SAMPLE_EDITS)})
            path = os.path.abspath(os.path.expanduser(body.get("path", "")))
            if not os.path.isfile(path):
                return self._json({"error": "file not found"}, 400)
            return self._json({"id": new_project(path, os.path.basename(path))})
        m = re.match(r"^/api/projects/([^/]+)/clone_fresh$", self.path)
        if m and DEV:
            return self._json(clone_fresh(m.group(1)))
        m = re.match(r"^/api/projects/([^/]+)/cancel$", self.path)
        if m:  # stop a scan under way: it ends at its next report and the project it was making goes to the Trash
            st = STATE.get(m.group(1))
            if not st or st.get("review_port") or scanned(m.group(1)):
                return self._json({"error": "no scan to cancel"}, 400)
            st["cancel"] = True
            return self._json({"ok": True})
        m = re.match(r"^/api/projects/([^/]+)/(clear_cache|delete|relink)$", self.path)
        if m:
            pid, action = m.group(1), m.group(2)
            d = proj_dir(pid)
            if not os.path.isdir(d):
                return self._json({"error": "unknown project"}, 404)
            if action == "relink":
                path = os.path.abspath(os.path.expanduser(self._body().get("path", "")))
                if not os.path.isfile(path):
                    return self._json({"error": "file not found"}, 400)
                meta = _meta(pid); meta["video"] = path; meta.pop("copied", None)
                write_json(os.path.join(d, "meta.json"), meta)
                stop_review(pid)  # (its server has the old video open; the next visit starts one on the new)
                STATE.pop(pid, None)
                return self._json({"ok": True})
            stop_review(pid)
            STATE.pop(pid, None)
            if action == "delete":
                trash(d)
                return self._json({"ok": True})
            meta = _meta(pid)
            copy = os.path.abspath(meta["video"]) if meta.get("copied") else None
            for name in os.listdir(d):
                p = os.path.join(d, name)
                if kept(name) or name.startswith(("before_space", "edits_before")) or os.path.abspath(p) == copy:  # backups are work, not cache
                    continue
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
            return self._json({"ok": True, "sizes": sizes(pid)})
        self._json({"error": "not found"}, 404)


def announce(port):
    """Say where this app is listening (<projects>/.app.json), so decisions_cli.py can find it: the Mac app picks a
    free port each run. Not a project (no meta.json), so the Projects list skips it."""
    try:
        path = os.path.join(PROJ_ROOT, ".app.json")
        with open(path + ".tmp", "w") as f:
            json.dump({"pid": os.getpid(), "port": port}, f)
        os.replace(path + ".tmp", path)
    except OSError as e:
        print(f"couldn't write .app.json: {e}")


def serve(port=0):
    """Start the server on 127.0.0.1 in a background thread; returns (server, port)."""
    init_projects()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    announce(srv.server_address[1])
    return srv, srv.server_address[1]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8800)))
    ap.add_argument("--dev", action="store_true", help="developer mode (same as MOJIMASK_DEV=1)")
    A = ap.parse_args()
    if A.dev:
        DEV = True
        os.environ["MOJIMASK_DEV"] = "1"  # project servers (child processes) inherit it
    from mojimask import about
    print(f"{about.NAME} app: http://localhost:{A.port}  (projects: {PROJ_ROOT})")
    srv, _ = serve(A.port)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()
        srv.shutdown()
