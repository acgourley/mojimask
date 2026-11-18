"""The app's whole journey on jazz through mmctl, with one engine: import, scan, answer every decision, edit, export.

    python3 native/tools/parity/appnative_journey.py --engine native|python [--motion python|native] [--out DIR] [--noexec]
    python3 native/tools/parity/appnative_journey.py --compare DIR_A DIR_B

The app (native/app/build) is started with a fresh projects folder (DIR/projects) and the engine asked for. The same
actions every time:
  1. import test_clips/jazz.mp4; wait for the scan (its seconds are recorded);
  2. Decisions: answer every open question with its first answer (slot 1), until none is left;
  3. Editor, frame 40: select the first scan box drawn there (by piece id), split it there (S); at frame 60 drag out a
     new cover (a box of your own, followed by propagate), then move it by (8, 4) px; the selected person's next
     emoji (E); lighting level 1.2 (saved on release); emoji preset "animals"; cover style pixelate, then emoji;
     the latency from each edit to the records drawn is logged;
  4. Decisions again (the "whose face is this?" questions the new boxes opened): answered with slot 1;
  5. Export (original, fill) to DIR/export.mp4.
facts.json and edits.json are copied to DIR, with screenshots of each page (DIR/shots).

--noexec: the app runs under sandbox-exec with every process launch denied but its own (DIR/noexec.sb): proof that the
engine runs no Python (use with --engine native --motion native). The app's child processes are listed after the scan
and after the export (journey.json "children").

--compare: facts.json (the "at" time stamps set aside) and edits.json byte for byte, and the exports frame by frame
(PSNR, decoded with the project's Python cv2). Prints JSON and a PASS/FAIL line (files identical, every frame >= 45 dB).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time


# the main checkout (a worktree has no venv or models of its own): the parent of git's common directory
MAIN_CHECKOUT = os.path.dirname(subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse",
                                                "--path-format=absolute", "--git-common-dir"],
                                               capture_output=True, text=True).stdout.strip())

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
M = os.path.join(ROOT, "native", "tools", "mmctl")
VENV = os.path.join(MAIN_CHECKOUT, ".venv/bin/python")


def mm(*a, check=True, timeout=600):
    r = subprocess.run([sys.executable, M, *a], capture_output=True, text=True, timeout=timeout)
    try:
        v = json.loads(r.stdout)
    except ValueError:
        v = {"ok": False, "raw": r.stdout + r.stderr}
    if check and not v.get("ok"):
        raise RuntimeError(f"mmctl {' '.join(a)}: {v}")
    return v


def call(cmd, args=None):
    return mm("call", cmd, json.dumps(args or {}))["result"]


def state(path):
    return mm("state", path)["result"]


def wait(js, timeout=600000):
    return mm("wait", js, "--timeout", str(timeout), timeout=timeout / 1000 + 30)


def settle(path, before, timeout=60):
    """Wait until state `path` differs from `before`."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = state(path)
        if v != before:
            return v
        time.sleep(0.1)
    raise RuntimeError(f"{path} stayed {before}")


def idle():
    """Wait for queued work: rev stable over 0.6 s."""
    last, same = None, 0
    while same < 3:
        cur = (state("project.can_undo"), json.dumps(state("decisions.count")), state("editor.records"))
        same = same + 1 if cur == last else 0
        last = cur
        time.sleep(0.2)


def goto(t):
    for _ in range(5):
        call("seek", {"t": t})
        if mm("wait", f"store.state.editor.shown === {t}", "--timeout", "4000", check=False).get("ok"):
            return
    raise RuntimeError(f"couldn't show frame {t}")


def answer_all(log):
    n = 0
    call("go", {"page": "decisions"})
    call("decisions_filter", {"filter": "open", "kind": "all"})
    time.sleep(0.5)
    while True:
        d = state("decisions")
        rows = d.get("rows") or []
        if not rows:
            break
        rid = rows[0]["id"]
        call("decisions_select", {"i": 0})
        time.sleep(0.2)
        det = state("decisions.detail")
        ans = det["answers"][0]["value"]
        before = d["count"]
        t0 = time.time()
        call("decisions_answer", {"id": rid, "answer": ans})
        settle("decisions.count", before)
        log.append({"answer": rid, "value": ans, "ms": round((time.time() - t0) * 1000)})
        n += 1
        if n > 200:
            raise RuntimeError("too many questions")
    return n


def shot(out, name):
    mm("grab", os.path.join(out, "shots", f"{name}.png"))


def edit_latency(log, what, fn):
    """Time from an edit command to its records being drawn (editor.records' rev in the logs)."""
    rev0 = state("project.can_undo"), state("editor.edits")
    t0 = time.time()
    fn()
    # records are fetched after the server's reply: wait for the log line that says they arrived
    since = log_since[0]
    while time.time() - t0 < 60:
        lines = mm("logs", "--since", str(since))["result"]
        hit = [x for x in lines.get("lines", []) if x["message"].startswith("records:")]
        if hit:
            log_since[0] = lines.get("next", since)
            ms = re.search(r"in (\d+) ms", hit[-1]["message"])
            log.append({"edit": what, "to_drawn_ms": round((time.time() - t0) * 1000), "records_ms": int(ms.group(1)) if ms else None})
            return
        time.sleep(0.02)
    log.append({"edit": what, "to_drawn_ms": None})
    del rev0


log_since = [0]


APP = os.path.join(ROOT, "native", "app", "build", "Mojimask.app", "Contents", "MacOS", "Mojimask")


def children():
    """The app's descendants (pid, command)."""
    pid = json.load(open(os.path.join(ROOT, "out", "native", "devtools.json")))["pid"]
    ps = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,comm="], capture_output=True, text=True).stdout.splitlines()
    rows = [(int(x.split()[0]), int(x.split()[1]), " ".join(x.split()[2:])) for x in ps if x.strip()]
    kids, todo = [], [pid]
    while todo:
        p = todo.pop()
        for c, pp, cmd in rows:
            if pp == p:
                kids.append([c, cmd])
                todo.append(c)
    return kids


def start_noexec(out, args):
    prof = os.path.join(out, "noexec.sb")
    with open(prof, "w") as f:
        f.write('(version 1)\n(allow default)\n(deny process-exec)\n(allow process-exec (literal "%s"))\n' % APP)
    log = open(os.path.join(out, "app_noexec.log"), "ab")
    subprocess.Popen(["sandbox-exec", "-f", prof, APP, "--dev", *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True)
    for _ in range(300):
        time.sleep(0.2)
        try:
            if mm("state", "app.page", check=False).get("ok"):
                return
        except Exception:  # noqa: BLE001
            pass
    raise RuntimeError("the sandboxed app didn't come up")


def journey(engine, motion, out, noexec=False):
    os.makedirs(os.path.join(out, "shots"), exist_ok=True)
    projects = os.path.join(out, "projects")
    shutil.rmtree(projects, ignore_errors=True)
    os.makedirs(projects)
    mm("quit", check=False)
    time.sleep(1)
    args = ["--projects", projects, "--engine", engine]
    if motion:
        args += ["--motion", motion]
    if noexec:
        start_noexec(out, args)
    else:
        mm("start", *args)
    wait("store.state.library.ready")
    shot(out, "1_projects_empty")
    log = []
    t0 = time.time()
    call("import", {"path": os.path.join(ROOT, "test_clips", "jazz.mp4")})
    wait('store.state.app.page === "scan"', 30000)
    time.sleep(8)
    shot(out, "2_scan")
    wait('store.state.app.page !== "scan" || store.state.scan.error !== ""')
    scan_s = time.time() - t0
    if state("scan.error"):
        raise RuntimeError(state("scan.error"))
    wait("store.state.editor.records > 0", 120000)
    time.sleep(1)
    kids = {"after_scan": children()}
    shot(out, "3_result")
    pid = state("project.id")
    work = os.path.join(projects, pid)
    call("go", {"page": "decisions"})
    time.sleep(1)
    shot(out, "4_decisions")
    n1 = answer_all(log)
    shot(out, "5_decisions_done")
    call("go", {"page": "people"})
    time.sleep(1.5)
    shot(out, "6_people")
    # the editor
    call("go", {"page": "editor"})
    time.sleep(1.5)
    goto(40)
    time.sleep(0.5)
    log_since[0] = mm("logs", "--since", "0")["result"].get("next", 0)
    boxes = [b for b in state("editor.view.boxes") if not b["manual"] and not b["hidden"] and b["kind"] == "obs"]
    piece = sorted(boxes, key=lambda b: (len(b["piece"]), b["piece"]))[0]["piece"]
    call("editor_select", {"piece": piece})
    edit_latency(log, f"split {piece} at 40", lambda: call("editor_act", {"what": "split"}))
    idle()
    goto(60)
    edit_latency(log, "draw a cover at 60", lambda: call("editor_draw", {"x0": 300, "y0": 330, "x1": 345, "y1": 375}))
    idle()
    sel = state("editor.view.sel") or {}
    mid = sel.get("piece") or "m1"
    edit_latency(log, f"move {mid}", lambda: call("editor_move", {"piece": mid, "dx": 8, "dy": 4}))
    idle()
    goto(40)
    call("editor_select", {"piece": piece})
    edit_latency(log, "next emoji", lambda: call("editor_act", {"what": "emoji"}))
    idle()
    call("editor_light", {"key": "level", "value": 1.2, "live": True})
    edit_latency(log, "lighting level", lambda: call("editor_light_end"))
    idle()
    edit_latency(log, "emoji preset", lambda: call("editor_palette", {"key": "animals"}))
    idle()
    edit_latency(log, "style pixelate", lambda: call("editor_style", {"style": "pixelate"}))
    idle()
    shot(out, "7_editor_pixelate")
    edit_latency(log, "style emoji", lambda: call("editor_style", {"style": "emoji"}))
    idle()
    time.sleep(1)
    shot(out, "8_editor")
    n2 = answer_all(log)
    # export
    call("go", {"page": "export"})
    time.sleep(1)
    shot(out, "9_export")
    dst = os.path.join(out, "export.mp4")
    if os.path.exists(dst):
        os.remove(dst)
    call("export_set", {"preset": "original", "fit": "fill", "mark": False})
    t1 = time.time()
    call("export_run", {"out": dst})
    wait('store.state.export.state !== "running"', 600000)
    export_s = time.time() - t1
    ex = state("export")
    kids["after_export"] = children()
    for f in ("facts.json", "edits.json"):
        shutil.copy(os.path.join(work, f), os.path.join(out, f))
    res = {"engine": engine, "motion": motion, "project": pid, "scan_s": round(scan_s, 1), "answered": [n1, n2],
           "export": ex.get("state"), "export_s": round(export_s, 1), "children": kids, "noexec": noexec, "log": log}
    with open(os.path.join(out, "journey.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def psnr_video(a, b):
    code = r"""
import json, sys, numpy as np
from mojimask import video_io
ga = video_io.frames(sys.argv[1], video_io.probe(sys.argv[1]), color="display")
gb = video_io.frames(sys.argv[2], video_io.probe(sys.argv[2]), color="display")
out = []
while True:
    fa, fb = next(ga, None), next(gb, None)
    if fa is None or fb is None:
        out.append({"end": [fa is not None, fb is not None]}); break
    mse = float(np.mean((fa.astype(np.float64) - fb.astype(np.float64)) ** 2))
    out.append(99.0 if mse == 0 else float(10 * np.log10(255 ** 2 / mse)))
print(json.dumps(out))
"""
    r = subprocess.run([VENV, "-c", code, a, b], capture_output=True, text=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": ROOT})
    return json.loads(r.stdout)


def compare(a, b):
    out = {}
    fa, fb = (json.load(open(os.path.join(d, "facts.json"))) for d in (a, b))
    strip = lambda d: {**d, "facts": [{k: v for k, v in f.items() if k != "at"} for f in d.get("facts", [])]}  # noqa: E731
    out["facts_identical_but_at"] = json.dumps(strip(fa)) == json.dumps(strip(fb))
    out["facts_n"] = [len(fa.get("facts", [])), len(fb.get("facts", []))]
    ea, eb = (open(os.path.join(d, "edits.json"), "rb").read() for d in (a, b))
    out["edits_identical"] = ea == eb
    if not out["edits_identical"]:
        ja, jb = json.loads(ea), json.loads(eb)
        out["edits_diff_keys"] = sorted(k for k in set(ja) | set(jb) if ja.get(k) != jb.get(k))
    p = psnr_video(os.path.join(a, "export.mp4"), os.path.join(b, "export.mp4"))
    fr = [x for x in p if isinstance(x, float)]
    out["export_frames"] = len(fr)
    out["export_end"] = p[-1]
    out["export_psnr_min"] = round(min(fr), 2) if fr else None
    out["export_psnr_mean"] = round(sum(fr) / len(fr), 2) if fr else None
    ok = out["facts_identical_but_at"] and out["edits_identical"] and fr and min(fr) >= 45 and p[-1]["end"] == [False, False]
    print(json.dumps(out, indent=1))
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="native")
    ap.add_argument("--motion", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--noexec", action="store_true")
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
    else:
        out = a.out or os.path.join(ROOT, "out", "native", "appnative", f"journey_{a.engine}" + (f"_motion_{a.motion}" if a.motion else "") + ("_noexec" if a.noexec else ""))
        os.makedirs(out, exist_ok=True)
        print(json.dumps(journey(a.engine, a.motion, out, a.noexec), indent=1))
