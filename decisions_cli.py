"""Answer a project's Decisions from a program (an agent, a script) instead of clicking. Not advertised in the app.

    .venv/bin/python decisions_cli.py projects
    .venv/bin/python decisions_cli.py questions <project> [--all] [--kind track|blip|overlap|same|whose] [--limit N]
    .venv/bin/python decisions_cli.py answer <project> <question id> "<choice>"
    .venv/bin/python decisions_cli.py clear <project> <question id>
    (the Mac app: Mojimask.app/Contents/MacOS/Mojimask --decisions <same arguments>)

Everything prints JSON. <project> is a project's folder name, or the start of one if only one matches.

It talks to the app while it runs (the app or `app_server.py`; the project needn't be open: the app opens it), so the
answers go through the same path as a click: saved to the project's facts log, undoable in the editor, shown on the
Decisions page with an "answered by an agent" note (facts carry via: "agent"). They count as your answers everywhere
else. A Decisions page already open shows them the next time you come back to that step.

No pictures: questions give the numbers the scan knows (frames, sizes, scores, box positions), never frames or
crops. People's faces don't leave the machine through here; what a program does with the numbers is up to it.

Questions come in the Decisions page's order: is it a face first (tracks, overlaps, blips), then who's who. Answering
one can settle or open others (the same-person questions are re-picked), so read the list again after answering.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

from mojimask import facts as FX
from mojimask import projects

# the answers a person can give, per kind (decisions.html's KINDS); the records' other options are the rules' own
CHOICES = {"track": [FX.A_FACE, FX.A_NOT_FACE], "blip": [FX.A_FACE, FX.A_NOT_FACE],
           "overlap": [FX.A_SAME_FACE, FX.A_NOT_FACE, FX.A_SOMEONE_ELSE], "same": [FX.A_SAME, FX.A_DIFFERENT]}
RANK = {"whose": 0, "track": 1, "overlap": 1, "blip": 1, "same": 2}
MEANING = {
    "track": "Is the green box (piece) a face? A face from any side counts.",
    "blip": "Is the green box (piece), seen only briefly, a face?",
    "overlap": "The green box (piece) overlaps the grey box (person, a covered face): part of that face, not a face "
               "at all (a hand, clothes, background), or someone else's face?",
    "same": "Are the green box (piece) and the grey box (person) the same person? Any part of the head counts.",
    "whose": f"A box the user drew: whose face is it? Answer with a person's piece id, or \"{FX.A_SOMEONE_NEW}\".",
}


def all_projects():
    for r in projects.roots(all=True):
        for name in sorted(os.listdir(r)):
            if os.path.exists(os.path.join(r, name, "meta.json")):
                yield r, name


def find(project):
    hits = [(r, n) for r, n in all_projects() if n == project] or [(r, n) for r, n in all_projects() if n.startswith(project)]
    if not hits:
        fail(f"no project {project!r} (see: projects)")
    if len(hits) > 1:
        fail(f"{project!r} matches {len(hits)} projects: " + ", ".join(n for _, n in hits))
    return hits[0]


def lock_holder(proj):
    """The port of the project server that has this project open, or None."""
    return (projects.lock_holder(proj) or {}).get("port")


def base_url(root, name):
    """The project's API: its own server if it's open, else through the app (which opens it)."""
    port = lock_holder(os.path.join(root, name))
    if port:
        return f"http://127.0.0.1:{port}/api"
    try:
        app = json.load(open(os.path.join(root, ".app.json")))
        urllib.request.urlopen(f"http://127.0.0.1:{app['port']}/api/app", timeout=3)
        return f"http://127.0.0.1:{app['port']}/p/{name}/api"
    except (OSError, ValueError, KeyError):
        from mojimask import about
        fail(f"{about.NAME} isn't running: open the app (or `.venv/bin/python app_server.py`) and try again")


def call(url, body=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 method="GET" if body is None else "POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            fail(json.loads(e.read()).get("error") or e.reason)
        except ValueError:
            fail(f"{e.code} {e.reason}")
    except OSError as e:
        fail(f"couldn't reach the project: {e}")


def fail(msg):
    print(json.dumps({"error": msg}))
    sys.exit(1)


def slim(d):
    """What a program needs to answer: no pictures."""
    out = {"id": d["id"], "kind": d["kind"], "meaning": MEANING.get(d["kind"], d.get("question")),
           "piece": d["about"]["piece"], "person": d["about"].get("person"),
           "choices": CHOICES.get(d["kind"], d.get("options")), "why": d.get("why"), "evidence": d.get("evidence"),
           "boxes": d.get("show"),  # [{t: frame, piece: [x0, y0, x1, y1], person: [...]}] in video pixels
           "answer": d.get("answer"), "answered_by": None if not d.get("by") else
           ("agent" if d.get("via") == "agent" else "user") if d["by"] == "user" else "rule"}
    if d.get("rule_answer") and d.get("rule_by"):
        out["rule"] = {"answer": d["rule_answer"], "why": d["rule_by"]}
    return out


def ordered(ds):
    # decisions.html's order: open first; faces before who's who; the weightiest, then the longest, first
    return sorted(ds, key=lambda d: (bool(d.get("by")), RANK.get(d["kind"], 3), -(d.get("weight") or 0),
                                     -((d.get("evidence") or {}).get("frames") or 0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("projects", help="the projects on this Mac")
    q = sub.add_parser("questions", help="a project's questions, open ones first")
    q.add_argument("project"); q.add_argument("--all", action="store_true", help="answered ones too")
    q.add_argument("--kind", choices=sorted(RANK)); q.add_argument("--limit", type=int)
    a = sub.add_parser("answer", help="answer a question (or change an answer)")
    a.add_argument("project"); a.add_argument("id"); a.add_argument("choice")
    c = sub.add_parser("clear", help="take an answer back (the question is open again)")
    c.add_argument("project"); c.add_argument("id")
    A = ap.parse_args()

    if A.cmd == "projects":
        out = []
        for r, n in all_projects():
            with open(os.path.join(r, n, "meta.json")) as f:
                meta = json.load(f)
            out.append({"project": n, "name": meta.get("label") or os.path.basename(meta.get("video", n)),
                        "open_in_app": bool(lock_holder(os.path.join(r, n))), "folder": os.path.join(r, n)})
        print(json.dumps(out, indent=1))
        return
    root, name = find(A.project)
    url = base_url(root, name)
    if A.cmd == "questions":
        r = call(f"{url}/decisions")
        if r.get("missing"):
            fail("this project has no decisions yet (still scanning, or scanned before decisions existed)")
        ds = ordered(r["decisions"])
        n_open = sum(not d.get("by") for d in ds)
        ds = [d for d in ds if (A.all or not d.get("by")) and (not A.kind or d["kind"] == A.kind)]
        print(json.dumps({"project": name, "open": n_open, "questions": [slim(d) for d in ds[:A.limit]]}, indent=1))
    elif A.cmd in ("answer", "clear"):
        choice = None
        if A.cmd == "answer":
            choice = A.choice
            kind = A.id.split(":")[0]
            if kind in CHOICES and choice not in CHOICES[kind]:
                fail(f"{kind} questions take one of: {', '.join(CHOICES[kind])}")
        print(json.dumps(slim(call(f"{url}/decisions", {"id": A.id, "answer": choice, "via": "agent"})), indent=1))


if __name__ == "__main__":
    main()
