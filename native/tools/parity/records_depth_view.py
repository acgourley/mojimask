"""The editor's depth view, native against Python: review_server's /api/depth/<t> (depth_frame: what the depth model
decided for each cover) and /api/depth/<t>.jpg (Occlusion.picture, JPEG q85) against mm-records --depth-view.

The JSON must be equal (numbers exactly) and the JPEG byte for byte. Runs Python's project server on a copy of a
public-domain project scanned with depth (port 9350), on every 10th frame plus the last (`--every N`). Prints one JSON object; the
last line is {"pass": true|false}.

    PYTHONPATH=ROOT .venv/bin/python native/tools/parity/records_depth_view.py [--project out/poc/projects/jazz] [--every N]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
NATIVE = os.path.join(ROOT, "native")
OUT = os.path.join(ROOT, "out", "native", "depth_view")
PORT = 9350
DEPTH = "anything-small"


def get(path):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=120) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.path.join(ROOT, "out", "poc", "projects", "jazz"))
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--bin", default=os.path.join(NATIVE, "target", "debug", "mm-records"))
    a = ap.parse_args()
    proj = os.path.join(OUT, "project")
    shutil.rmtree(OUT, ignore_errors=True)
    shutil.copytree(a.project, proj, ignore=shutil.ignore_patterns(".lock", "review_server.log"))
    meta = json.load(open(os.path.join(proj, "meta.json")))
    video = meta["video"] if os.path.isabs(meta["video"]) else os.path.join(ROOT, meta["video"])
    n = len([f for f in os.listdir(os.path.join(proj, "frames")) if f.endswith(".jpg")])
    ts = sorted(set(list(range(0, n, a.every)) + [n - 1]))

    env = dict(os.environ, MOJIMASK_DEPTH=DEPTH, PYTHONPATH=ROOT)
    log = open(os.path.join(OUT, "review_server.log"), "w")
    srv = subprocess.Popen([sys.executable, "-m", "mojimask.review_server", video, os.path.join(proj, "dets"),
                            "--workdir", proj, "--original", video, "--port", str(PORT),
                            "--dets", "yolov5m_face_tiled:0.3", "retinaface_r50_2x:0.3", "pose_head:0.35"],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        while True:
            try:
                get("/api/init")
                break
            except Exception:
                if srv.poll() is not None or time.time() - t0 > 120:
                    raise SystemExit("review_server didn't start: see " + log.name)
                time.sleep(0.5)
        py = {t: (json.loads(get(f"/api/depth/{t}")), get(f"/api/depth/{t}.jpg")) for t in ts}
    finally:
        srv.terminate()
        srv.wait()

    rs_dir = os.path.join(OUT, "rust")
    subprocess.run([a.bin, proj, "--depth", DEPTH, "--depth-view", ",".join(map(str, ts)), "--out", rs_dir], check=True)
    json_bad, jpg_bad, covers, cut = [], [], 0, 0
    for t in ts:
        pj, pjpg = py[t]
        rj = json.load(open(os.path.join(rs_dir, f"{t}.json")))
        rjpg = open(os.path.join(rs_dir, f"{t}.jpg"), "rb").read()
        if pj != rj:
            json_bad.append({"t": t, "python": pj, "rust": rj})
        if pjpg != rjpg:
            jpg_bad.append(t)
        covers += len(pj["covers"])
        cut += sum(1 for c in pj["covers"] if c["cut"])
    ok = not json_bad and not jpg_bad
    print(json.dumps({"project": a.project, "frames": len(ts), "covers": covers, "cut": cut,
                      "json_differ": json_bad[:3], "json_differ_n": len(json_bad), "jpg_differ": jpg_bad}, indent=1))
    print(json.dumps({"pass": ok}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
