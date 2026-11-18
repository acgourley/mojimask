"""Parity of facts.derive and facts.answers: replay facts logs one fact at a time, in Python (mojimask/facts.py) and
Rust (`mm-cli decisions replay-facts`), and compare after every fact.

    .venv/bin/python native/tools/parity/model_facts.py

Logs replayed: the jazz project's own (out/poc/projects/jazz/facts.json), and those model_session.py left (its
Python copy's log, if it has run). For each prefix of the log: the derived edits fields (people, emoji, not_face,
part_of, ...) compared as text (json.dumps, byte for byte) and parsed; the decision list compared by fingerprint
(FNV-1a of json.dumps).
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_common as C  # noqa: E402

sys.path.insert(0, C.ROOT)
from mojimask import facts as FX  # noqa: E402

PROJ = C.copy_project(os.path.join(C.OUT, "facts", "jazz"))
proto = C.load(os.path.join(PROJ, "proto_tracks.json"))
edits = C.load(os.path.join(PROJ, "edits.json"))
HIDDEN = proto.get("hidden")
if HIDDEN is None:
    HIDDEN = sorted(p for p, v in (edits.get("not_face") or {}).items() if v == "decision")
RECS = [d for f in sorted(x for x in os.listdir(PROJ) if x.startswith("decisions_") and x.endswith(".json"))
        for d in C.load(os.path.join(PROJ, f))["decisions"]]

logs = {"jazz": os.path.join(C.SRC, "facts.json")}
s = os.path.join(C.OUT, "session", "py", "jazz", "facts.json")
if os.path.exists(s):
    logs["session"] = s

results, ok = {}, True
for name, path in logs.items():
    facts = C.load(path)["facts"]
    r = subprocess.run(C.rust_bin() + ["replay-facts", PROJ, path], capture_output=True, text=True)
    rs = [json.loads(x) for x in r.stdout.splitlines() if x.strip()]
    if r.returncode != 0 or len(rs) != len(facts) + 1:
        sys.exit(f"rust replay-facts failed: {r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    text = parsed = ans = 0
    first = None
    for k in range(len(facts) + 1):
        d = FX.derive(facts[:k], HIDDEN, edits.get("palette"))
        a = C.fnv(json.dumps(FX.answers(RECS, facts[:k], HIDDEN)).encode())
        t, p, h = json.dumps(d) == json.dumps(rs[k]["derive"]), d == rs[k]["derive"], a == rs[k]["answers"]
        text, parsed, ans = text + t, parsed + p, ans + h
        if not (t and p and h) and first is None:
            first = {"k": k, "fact": facts[k - 1] if k else None, "python": d, "rust": rs[k]["derive"], "answers_equal": h}
    results[name] = {"facts": len(facts), "derive_text_equal": f"{text}/{len(facts) + 1}",
                     "derive_parsed_equal": f"{parsed}/{len(facts) + 1}", "answers_equal": f"{ans}/{len(facts) + 1}",
                     "first_difference": first}
    ok = ok and first is None
sys.exit(C.report("model_facts", ok, logs=results))
