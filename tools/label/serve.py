"""Truth labelling: the census of a clip, every face boxed by hand.

Labels every head in a project's own frames (frames/*.jpg: the app's numbering, see layer 0) as points:
in frame t, this box is a face / head / tiny / not, of person P. Truth doesn't depend on any tracker.

    python tools/label/serve.py out/projects/<project> [--name jazz] [--every 4] [--port 8830]
    open http://127.0.0.1:8830          (label every head)
    open http://127.0.0.1:8830/check    (focused check of Claude's drafted frames: tools/label/triage.py first)
    open http://127.0.0.1:8830/grade    (grade the redesign's result: experiments/proto/grade_prep.py first)

Pre-fill: every face detector's boxes (as head boxes) and the pose model's heads, fused, at a low threshold,
so most of the work is keep / fix / delete. Saved to out/truth/<name>.json (gitignored):

    {"from": project, "space": {...}, "n": frames, "fps": ..., "w": W, "h": H,
     "frames": {t: [{"b": [x0, y0, x1, y1] (fractions of the frame), "c": "face"|"head"|"tiny"|"not",
                     "p": person number or null, "src": "prefill"|"drawn"|"edited"|"carried",
                     "o": 1 if mostly hidden (blocked by someone or something)}]},
     "checked": [t, ...]}   # frames you went through and confirmed nothing is missing
"""
import argparse
import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mojimask import paths  # noqa: E402
from mojimask import track as T  # noqa: E402
from mojimask.http_guard import GuardMixin  # noqa: E402  (Host and Origin checks: no other page may read frames or save truth)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from draft import PREFILL  # noqa: E402  (one copy of the pre-fill: draft.py drafts frames the way this tool would)

ap = argparse.ArgumentParser()
ap.add_argument("project")
ap.add_argument("--name")
ap.add_argument("--port", type=int, default=8830)
ap.add_argument("--every", type=int, default=4, help="label every Nth frame (a census); 1 = every frame")
A = ap.parse_args()
PROJ = os.path.abspath(A.project)
NAME = A.name or os.path.basename(PROJ).split("_")[0]
TRUTH = paths.truth(NAME)
meta = json.load(open(f"{PROJ}/meta.json"))
frames = sorted(f for f in os.listdir(f"{PROJ}/frames") if f.endswith(".jpg"))
import cv2  # noqa: E402

H, W = cv2.imread(f"{PROJ}/frames/{frames[0]}").shape[:2]
spec = [(n, th) for n, th in PREFILL if os.path.exists(f"{PROJ}/dets/{n}.json")]
fused = T.fused_frames(f"{PROJ}/dets", spec)
import numpy as np  # noqa: E402
mats = np.load(f"{PROJ}/mats_v2.npy").round(5).tolist() if os.path.exists(f"{PROJ}/mats_v2.npy") else None
prefill = [[[round(v, 1) for v in b] + [round(s, 2), len(src)] for b, s, src in f] for f in fused]


def write_atomic(path, data):
    """Write bytes or a JSON-able value to path through a .tmp file, flushed and synced before the rename."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        f.write(data if isinstance(data, bytes) else json.dumps(data).encode())
        f.flush()
        os.fsync(f.fileno())
    os.replace(path + ".tmp", path)


class Handler(GuardMixin, SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json", code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            return self._send(open(os.path.join(HERE, "label.html"), "rb").read(), "text/html; charset=utf-8")
        if p.startswith("/frame/"):
            t = int(p.split("/")[-1].split(".")[0])
            return self._send(open(f"{PROJ}/frames/{frames[t]}", "rb").read(), "image/jpeg")
        if p in ("/grade", "/grade.html"):
            return self._send(open(os.path.join(HERE, "grade.html"), "rb").read(), "text/html; charset=utf-8")
        if p == "/api/grade_data":
            return self._send(open(os.path.join(ROOT, "out", "proto", f"grade_{os.path.basename(PROJ)}.json"), "rb").read())
        if p == "/api/grade_load":
            gp = os.path.join(ROOT, "out", "truth", f"{NAME}_grading.json")
            return self._send(open(gp, "rb").read() if os.path.exists(gp) else b"{}")
        if p in ("/check", "/check.html"):
            return self._send(open(os.path.join(HERE, "check.html"), "rb").read(), "text/html; charset=utf-8")
        if p == "/api/triage":
            tp = os.path.join(ROOT, "out", "truth", f"{NAME}_triage.json")
            return self._send(open(tp, "rb").read() if os.path.exists(tp) else b'{"items": []}')
        if p == "/api/init":
            truth = json.load(open(TRUTH)) if os.path.exists(TRUTH) else None
            return self._send(json.dumps({"name": NAME, "project": os.path.basename(PROJ), "n": len(frames), "w": W, "h": H,
                                          "fps": meta.get("fps"), "prefill": prefill, "truth": truth,
                                          "every": A.every, "mats": mats}).encode())
        self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        if self.path == "/api/grade_save":
            d = self.rfile.read(int(self.headers["Content-Length"]))
            gp = os.path.join(ROOT, "out", "truth", f"{NAME}_grading.json")
            write_atomic(gp, d)
            return self._send(b'{"ok": true}')
        if self.path == "/api/save":
            # merged into what's there, so keys this page doesn't send (check.html's audit) are kept
            d = json.load(open(TRUTH)) if os.path.exists(TRUTH) else {}
            d.update(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            d.update({"from": os.path.basename(PROJ), "space": meta.get("space") or {"w": W, "h": H, "rot": meta.get("rotation"), "v": 1},
                      "n": len(frames), "fps": meta.get("fps"), "w": W, "h": H})
            write_atomic(TRUTH, d)
            return self._send(b'{"ok": true}')
        self._send(b"not found", "text/plain", 404)


print(f"labelling {os.path.basename(PROJ)} ({len(frames)} frames, {W}x{H}) -> {TRUTH}")
print(f"open http://127.0.0.1:{A.port}")
ThreadingHTTPServer(("127.0.0.1", A.port), Handler).serve_forever()
