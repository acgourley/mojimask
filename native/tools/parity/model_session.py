"""Parity of the project model: a long seeded random session of page requests, through review_server (Python) and
through mm-core (Rust, `mm-cli decisions session`), on two copies of the jazz project.

    .venv/bin/python native/tools/parity/model_session.py [--steps 600] [--seed 7] [--port 9400]

Python: review_server.py imported in this process (its startup reads the copy as it would when the app opens it)
and served on a port in 9400-9499 for this test only; every step is a real HTTP request to its handlers
(/api/ops, /api/facts, /api/decisions, /api/undo, /api/redo, /api/split_cover, /api/undo_correction,
/api/propagate; and refollow_covers(), which a server runs once as it starts, called directly). The steps are
chosen as the session goes (seeded), from the state Python is in, and written to
out/native/model/session/steps.jsonl. Rust then replays that file in one process.

Stand-ins, so the two run the same code paths:
  - time: facts are stamped with a clock both sides set per step (facts.time patched here);
  - the optical flow and face evidence propagate() follows (vision, P5): Python's real calls are recorded and Rust
    replays the recording, checking each call's inputs (so propagate's own rules are compared, not cv2's flow);
  - "whose face is this?" questions (SFace/OSNet, P5): turned off on the Python side (your_box_questions_now -> []).

Compared after every step: the step's reply, and fingerprints (FNV-1a) of facts.json, edits.json and the decision
list (GET /api/decisions). At the end: the files byte for byte, and parsed.
"""
import argparse
import json
import os
import random
import sys
import threading
import types
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--port", type=int, default=9400)
A = ap.parse_args()
assert 9400 <= A.port <= 9499

WORK = os.path.join(C.OUT, "session")
PYP, RSP = os.path.join(WORK, "py", "jazz"), os.path.join(WORK, "rs", "jazz")
C.copy_project(PYP)
C.copy_project(RSP)
STEPS, REC = os.path.join(WORK, "steps.jsonl"), os.path.join(WORK, "follow.json")
T0 = 1_800_000_000  # seconds: the session's clock starts here, one second per step

# ---- Python: review_server in this process
sys.path.insert(0, C.ROOT)
from mojimask import facts as FX  # noqa: E402
from mojimask import palettes  # noqa: E402

CLOCK = {"s": T0}
FX.time = types.SimpleNamespace(time=lambda: CLOCK["s"])
sys.argv = ["review_server", C.VIDEO, os.path.join(PYP, "dets"), "--workdir", PYP, "--port", str(A.port)]
import mojimask.review_server as RS  # noqa: E402

RS.your_box_questions_now = lambda: []  # (vision: P5)
RECORD = []
_flow, _face, _gray = RS.flow_step, RS.face_evidence, RS.gray
RS.gray = lambda t: t  # flow_step's frames by number, so the recording can name them


def flow_step(g0, g1, c):
    out = _flow(_gray(g0), _gray(g1), c)
    RECORD.append({"fn": "flow", "from": int(g0), "to": int(g1), "c": [float(v) for v in c],
                   "out": None if out is None else [float(v) for v in out]})
    return out


def face_evidence(u, c, prev=None, step=1):
    out = _face(u, c, prev, step)
    RECORD.append({"fn": "face", "u": int(u), "c": [float(v) for v in c], "prev": [float(v) for v in prev], "step": step,
                   "out": None if not out else [float(v) for v in out]})
    return out


RS.flow_step, RS.face_evidence = flow_step, face_evidence
srv = RS.ThreadingHTTPServer(("127.0.0.1", A.port), RS.Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{A.port}/api"


def call(path, body=None):
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(),
                                 method="GET" if body is None else "POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return 200, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


DOC_KEYS = {"rev", "edits", "can_undo", "can_redo", "resolved", "hidden", "persons"}


def fingerprints():
    code, r = call("/decisions")
    return {"facts": C.file_hash(os.path.join(PYP, "facts.json")), "edits": C.file_hash(os.path.join(PYP, "edits.json")),
            "decisions": C.fnv(json.dumps(r["decisions"]).encode())}


# ---- the session: steps chosen from the state Python is in
rng = random.Random(A.seed)
RECS = [d for f in RS._decision_files() for d in RS._load_json(f)["decisions"]]
PIECES = sorted({str(d["about"]["piece"]) for d in RECS} | {str(d["about"].get("person")) for d in RECS if d["about"].get("person") is not None},
                key=lambda q: (len(q), q))
TRACKS = sorted(RS.AUTO, key=lambda q: (len(q), q))
PRESETS = list(palettes.PRESETS) + ["custom", "retired"]
EMOJI = ["😀", "🐸", "🦊", "🌝", "🍕", ""]
NAMES = ["Ann", "Bo", "", "Zoë", "guest 3"]


def box_at(tid=None):
    tid = tid or rng.choice(TRACKS)
    t = rng.choice(sorted(RS.AUTO[tid]["cover"]))
    c = RS.AUTO[tid]["cover"][t]
    return t, [round(float(v) + rng.uniform(-3, 3), 1) for v in c]


def edits():
    return RS.load_edits()


def covers():
    return sorted((edits().get("manual") or {}).items())


def random_op():
    k = rng.random()
    p = rng.choice(PIECES)
    if k < 0.08:
        return {"op": "set", "path": ["palette"], "value": rng.choice(PRESETS)}
    if k < 0.11:
        return {"op": "set", "path": ["palette_custom"], "value": rng.sample(EMOJI[:-1], rng.randint(1, 4))}
    if k < 0.16:
        return {"op": "set", "path": ["cover_style"], "value": rng.choice(["emoji", "blur", "pixelate", "solid"])}
    if k < 0.20:
        return {"op": "set", "path": ["light", "strength"], "value": rng.choice([0, 0.5, 1.0, 0.25, 1])}
    if k < 0.23:
        return {"op": "set", "path": ["head_pose"], "value": rng.random() < 0.5}
    if k < 0.33:
        return {"op": rng.choice(["add", "remove"]), "path": [rng.choice(["deleted", "not_real", "dismissed"])], "value": p}
    if k < 0.38:
        lo = rng.randint(0, 60)
        return rng.choice([{"op": "set", "path": ["trims", p], "value": [lo, lo + rng.randint(0, 60)]},
                           {"op": "del", "path": ["trims", p]}])
    if k < 0.53:
        n = 1 + max([int(m[1:]) for m, _ in covers() if m[1:].isdigit()] + [0])
        tid = rng.choice(TRACKS)
        keys = dict(box_at(tid) for _ in range(rng.randint(1, 3)))
        m = {"keys": {str(t): c for t, c in sorted(keys.items())}}
        if rng.random() < 0.5:
            m["prop"] = {str(t + 1): [round(c[0] + 1, 1), round(c[1], 1), c[2]] for t, c in keys.items() if str(t + 1) not in m["keys"]}
            m["ev"] = sorted(t + 1 for t in keys)
            m["pv"] = 3
        return {"op": "set", "path": ["manual", f"m{n}"], "value": m}
    if k < 0.60 and covers():
        mid = rng.choice(covers())[0]
        if rng.random() < 0.3:
            return {"op": "del", "path": ["manual", mid]}
        t, c = box_at()
        return {"op": "set", "path": ["manual", mid, "keys", str(t)], "value": c}
    if k < 0.66:  # fields the facts own: dropped by /api/ops
        return rng.choice([{"op": "set", "path": ["not_face", p], "value": "you"},
                           {"op": "set", "path": ["people", p, "name"], "value": "x"},
                           {"op": "add", "path": ["shown"], "value": p}])
    if k < 0.68:
        return {"op": "set", "path": ["cuts", rng.choice(TRACKS)], "value": [rng.randint(1, 120)]}
    if k < 0.70:
        return rng.choice([{"op": "move", "path": ["x"]}, {"op": "add", "path": ["trims"], "value": 1}])  # bad: the batch fails
    if k < 0.75:
        return {"op": "set", "path": ["result", ], "value": rng.choice(["ok", "done", ""])}
    return {"op": "set", "path": ["export", "name"], "value": rng.choice(["a.mp4", "b.mp4", 3, 3.0, None, {"x": [1, 2]}])}


def random_fact():
    k = rng.random()
    p, q = rng.choice(PIECES), rng.choice(PIECES)
    people = sorted((edits().get("people") or {}).items())
    if k < 0.10 and people:  # as People says it: about every piece of a person as they stand
        _, P = rng.choice(people)
        kind = rng.choice(["name", "emoji", "show"])
        return {"kind": kind, "pieces": list(P["pieces"]),
                "value": rng.choice({"name": NAMES, "emoji": EMOJI, "show": [True, False]}[kind])}
    if k < 0.25:
        f = {"kind": "face", "pieces": [p], "value": rng.random() < 0.5}
        if not f["value"] and rng.random() < 0.5:
            f["why"] = rng.choice(["you", "hand"])
        return f
    if k < 0.55:
        return {"kind": rng.choice(["same", "same", "different", "part"]), "pieces": [p, q]}
    if k < 0.70:
        return {"kind": "name", "pieces": [p], "value": rng.choice(NAMES)}
    if k < 0.85:
        f = {"kind": "emoji", "pieces": [p], "value": rng.choice(EMOJI)}
        if rng.random() < 0.3:
            f["preset"] = rng.choice(PRESETS[:3])
        return f
    if k < 0.97:
        return {"kind": "show", "pieces": [p], "value": rng.random() < 0.5}
    return {"kind": "nope", "pieces": [p]}  # bad: refused


def random_step():
    k = rng.random()
    if k < 0.22:
        ops = [random_op() for _ in range(rng.choice([1, 1, 1, 2, 3]))]
        body = {"ops": ops, "label": rng.choice(["", "edit", "style"])}
        if rng.random() < 0.1:
            body["undoable"] = False
        return "ops", body
    if k < 0.40:
        return "facts", {"facts": [random_fact() for _ in range(rng.choice([1, 1, 2]))], "label": "facts", "page": rng.choice(["people", "edit"])}
    if k < 0.70:
        if rng.random() < 0.02:
            return "decide", {"id": "track:999999", "answer": "a face"}
        d = rng.choice(RECS)
        r = rng.random()
        ans = None if r < 0.15 else "maybe" if r < 0.18 else rng.choice(d["options"])
        body = {"id": d["id"], "answer": ans}
        if rng.random() < 0.5:
            body["via"] = "agent"
        return "decide", body
    if k < 0.80:
        return "undo", {}
    if k < 0.86:
        return "redo", {}
    cs = covers()
    if k < 0.92 and cs:
        mid, m = rng.choice(cs)
        ks = sorted(int(t) for t in m.get("keys", {}))
        t = rng.randint(min(ks) - 12, max(ks) + 12) if ks else 5
        return "split", {"piece": mid, "t": t}
    if k < 0.96 and cs:
        mid, m = rng.choice(cs)
        ks = sorted(int(t) for t in m.get("keys", {})) or [0]
        return "undo_correction", {"piece": mid, "t": rng.choice(ks) if rng.random() < 0.85 else ks[0] + 1}
    if k < 0.97:
        return "refollow", {}
    t, c = box_at()
    return "propagate", {"t": t, "c": c}


PATH = {"ops": "/ops", "facts": "/facts", "decide": "/decisions", "undo": "/undo", "redo": "/redo",
        "split": "/split_cover", "undo_correction": "/undo_correction", "propagate": "/propagate"}


def reply(kind, code, r):
    if code != 200:
        return {"error": r.get("error"), "code": code}
    if kind in ("ops", "facts", "undo_correction", "refollow"):
        return {"ok": True}
    if kind == "decide":
        return {k: v for k, v in r.items() if k not in DOC_KEYS}
    if kind == "undo":
        return {"undone": r["undone"]}
    if kind == "redo":
        return {"redone": r["redone"]}
    if kind == "split":
        return {"new": r["new"]}
    return r  # propagate


py_lines = [{"i": -1, "result": None, "hash": fingerprints()}]
with open(STEPS, "w") as out:
    for i in range(A.steps):
        kind, body = random_step()
        CLOCK["s"] = T0 + i + 1
        if kind == "refollow":  # what a project server does once as it starts (not a request): called directly
            RS.refollow_covers()
            code, r = 200, {}
        else:
            code, r = call(PATH[kind], body)
        py_lines.append({"i": i, "result": reply(kind, code, r), "hash": fingerprints()})
        out.write(json.dumps({"at": (T0 + i + 1) * 1000, "do": kind, "body": body}) + "\n")
with open(REC, "w") as f:
    json.dump(RECORD, f)
srv.shutdown()

# ---- Rust: the same steps, in one process
import subprocess  # noqa: E402

r = subprocess.run(C.rust_bin() + ["session", RSP, STEPS, "--follow", REC], capture_output=True, text=True)
rs_lines = [json.loads(x) for x in r.stdout.splitlines() if x.strip()]
if r.returncode != 0 or not rs_lines or not rs_lines[-1].get("end"):
    sys.exit(f"rust session failed: {r.stdout[-2000:]}\n{r.stderr[-2000:]}")
end = rs_lines.pop()

steps = [json.loads(x) for x in open(STEPS)]
first_bad, kinds, n_err = None, {}, 0
same_results = same_hashes = 0
for py, rs in zip(py_lines, rs_lines):
    ok_r = json.dumps(py["result"]) == json.dumps(rs["result"])
    ok_h = py["hash"] == rs["hash"]
    same_results += ok_r
    same_hashes += ok_h
    if py["i"] >= 0:
        k = steps[py["i"]]["do"]
        kinds[k] = kinds.get(k, 0) + 1
        n_err += "error" in (py["result"] or {})
    if (not ok_r or not ok_h) and first_bad is None:
        first_bad = {"step": py["i"], "do": steps[py["i"]] if py["i"] >= 0 else None, "python": py, "rust": rs}

files = {}
for name in ("facts.json", "edits.json"):
    a, b = open(os.path.join(PYP, name), "rb").read(), open(os.path.join(RSP, name), "rb").read()
    files[name] = {"bytes_equal": a == b, "parsed_equal": json.loads(a) == json.loads(b), "size": len(a)}
ok = (first_bad is None and len(py_lines) == len(rs_lines) and all(f["bytes_equal"] for f in files.values())
      and not end["follow_mismatches"] and end["follow_calls"] == end["follow_recorded"])
sys.exit(C.report(
    "model_session", ok, seed=A.seed, steps=A.steps, kinds=kinds, refused_steps=n_err,
    results_equal=f"{same_results}/{len(py_lines)}", fingerprints_equal=f"{same_hashes}/{len(py_lines)}",
    files=files, facts_in_log=len(json.load(open(os.path.join(PYP, "facts.json")))["facts"]),
    follow={"recorded_calls": end["follow_recorded"], "replayed": end["follow_calls"], "mismatches": end["follow_mismatches"]},
    first_difference=first_bad))
