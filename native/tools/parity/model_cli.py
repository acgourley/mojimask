"""Parity of the command line: decisions_cli.py (through a running review_server, as the app does it) against
`mm-cli decisions` (on the project folder), on two copies of the jazz project.

    .venv/bin/python native/tools/parity/model_cli.py [--answers 60] [--seed 3] [--port 9401]

A seeded walk, from an empty facts log (jazz's own answers every question): list the open questions, answer the first one (a random allowed choice), now and then clear an
earlier answer or list with --all / --kind / --limit; the same commands on both sides. Compared: every command's
output (text, byte for byte), and at the end facts.json (parsed, without "at": the CLI stamps facts with the real
time) and edits.json (byte for byte).
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--answers", type=int, default=60)
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--port", type=int, default=9401)
A = ap.parse_args()
assert 9400 <= A.port <= 9499

WORK = os.path.join(C.OUT, "cli")
PYR, RSR = os.path.join(WORK, "py"), os.path.join(WORK, "rs")
PYP, RSP = C.copy_project(os.path.join(PYR, "jazz")), C.copy_project(os.path.join(RSR, "jazz"))
for p in (PYP, RSP):  # jazz's own log answers every question: start from none, as a new project does
    os.remove(os.path.join(p, "facts.json"))
RUST = C.rust_bin()

srv = subprocess.Popen([C.PY, "-m", "mojimask.review_server", C.VIDEO, os.path.join(PYP, "dets"), "--workdir", PYP,
                        "--port", str(A.port)], cwd=C.ROOT, env={**os.environ, "PYTHONPATH": C.ROOT},
                       stdout=open(os.path.join(WORK, "review_server.log"), "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{A.port}/api/rev", timeout=2)
            break
        except OSError:
            time.sleep(0.2)
    else:
        sys.exit("review_server didn't start (see out/native/model/cli/review_server.log)")

    def py(*args):
        r = subprocess.run([C.PY, os.path.join(C.ROOT, "decisions_cli.py"), *args], capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": C.ROOT, "MOJIMASK_PROJECTS": PYR})
        return r.returncode, r.stdout

    def rs(*args):
        r = subprocess.run(RUST + list(args), capture_output=True, text=True, env={**os.environ, "MOJIMASK_PROJECTS": RSR})
        return r.returncode, r.stdout

    rng = random.Random(A.seed)
    cmds, same, first = 0, 0, None
    answered = []

    def both(*args):
        global cmds, same, first
        a, b = py(*args), rs(*args)
        cmds += 1
        if a == b:
            same += 1
        elif first is None:
            first = {"command": args, "python": a, "rust": b}
        return a

    # projects: the same list, but the folder differs and Python's copy is open in its server
    pa, ra = json.loads(py("projects")[1]), json.loads(rs("projects")[1])
    projects_ok = [(x["project"], x["name"]) for x in pa] == [(x["project"], x["name"]) for x in ra]
    both("questions", "jazz")
    both("questions", "jazz", "--all")
    for i in range(A.answers):
        r = rng.random()
        if r < 0.15 and answered:
            both("clear", "jazz", answered.pop(rng.randrange(len(answered))))
        elif r < 0.22:
            both("questions", "jazz", "--all", "--kind", rng.choice(["track", "blip", "overlap", "same", "whose"]), "--limit", str(rng.randint(-3, 9)))
        elif r < 0.25:
            both("answer", "jazz", "track:5", "maybe")  # refused: not a choice
        else:
            code, out = both("questions", "jazz", "--limit", "1")
            qs = json.loads(out)["questions"] if code == 0 else []
            if not qs:
                break
            q = qs[0]
            both("answer", "jazz", q["id"], rng.choice(q["choices"]))
            answered.append(q["id"])
    both("questions", "jazz", "--all")
finally:
    srv.terminate()
    srv.wait()

fa, fb = C.load(os.path.join(PYP, "facts.json")), C.load(os.path.join(RSP, "facts.json"))
ea, eb = (open(os.path.join(p, "edits.json"), "rb").read() for p in (PYP, RSP))
facts_ok = C.without(fa, {"at"}) == C.without(fb, {"at"})
ok = first is None and projects_ok and facts_ok and ea == eb
sys.exit(C.report("model_cli", ok, seed=A.seed, commands=cmds, outputs_equal=f"{same}/{cmds}", projects_listing_equal=projects_ok,
                  facts_equal_but_at=facts_ok, facts=len(fa["facts"]), via_agent=sum(f.get("via") == "agent" for f in fa["facts"]),
                  edits_bytes_equal=ea == eb, first_difference=first))
