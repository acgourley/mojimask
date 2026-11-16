"""Who's who and what's a face, as one log of facts (the new algorithm's projects).

Everything a person says about the boxes -- on the Decisions page, on People, in Edit -- is a fact appended to
PROJECT/facts.json. Nothing else stores it: the edits fields the pages draw from (people, emoji, not_face, ...) and
the answers the Decisions page shows are computed from the facts by derive() / answers(), every time, the same
way. So an answer, a join in one page and an undo in another can't disagree: there is one record, and undo takes
the last facts back out of it.

A fact: {"id", "at" (ms), "kind", "pieces": [...], "value", "by" ("user"), "page", "decision" (the question it
answers, if any), "answer" (the Decisions option it stands for, if any), "via" ("agent" when a program answered)}.
Kinds:
  same       pieces [a, b]   one person
  part       pieces [a, b]   a is part of b's face (a chin, neck or hair box): one person, and drawn as one face
  different  pieces [a, b]   not one person
  face       pieces [a]      value true: a face (masked); false: not a face (no emoji), "why": the reason when false
                             ("decision", "you", ...)
  show       pieces [...]    value true: unmasked; false: masked (the pieces of the person at the time)
  name       pieces [...]    value: their name ("" clears it)
  emoji      pieces [...]    value: their emoji ("" clears it), "preset": the emoji preset it was picked under
                             (palettes.json key, or "custom"). A pick belongs to its preset: another preset deals
                             everyone its own emoji, and coming back brings the picks back.

Precedence: newer beats older. Joining is processed newest first: a "same" joins two groups unless a newer
"different" already keeps them apart, so the latest word on any pair wins and contradictions resolve the same way
every time. The scan's own rules are the baseline underneath (its hidden boxes; its tracks are its people).

The facts belong to one scan: facts.json records the scan's version (newalgo.SCAN_VERSION) and a rescan throws
them away (box ids are the scan's).
"""
import collections
import json
import os
import time

from mojimask import palettes

FILE = "facts.json"
KINDS = ("same", "part", "different", "face", "show", "name", "emoji")
JOIN = ("same", "part")
ATTRS = ("show", "name", "emoji")
# The Decisions page's answers (decisions.json and the facts store them as these strings; decisions.html labels them)
A_FACE, A_NOT_FACE = "a face", "not a face"
A_SAME, A_DIFFERENT = "same person", "different people"
A_SAME_FACE, A_SOMEONE_ELSE, A_SOMEONE_NEW = "same face", "someone else", "someone new"
A_JOINED, A_KEPT_APART = "joined already", "kept apart"  # settled by your other answers
A_OPEN, A_WAITING = "open", "waiting"  # same-person questions: (answers() turns "open" into None, asked now)


def load(work, scan_version=None):
    """The log. A log made for another scan version is set aside (facts.discarded.v<its version>.json) and an empty
    one started."""
    path = os.path.join(work, FILE)
    doc = None
    if os.path.exists(path):
        with open(path) as f:
            doc = json.load(f)
    if doc is None or (scan_version is not None and doc.get("scan_version") != scan_version):
        if doc is not None:
            os.replace(path, os.path.join(work, f"facts.discarded.v{doc.get('scan_version')}.json"))
        doc = {"scan_version": scan_version, "facts": []}
        save(work, doc)
    return doc


def save(work, doc):
    path = os.path.join(work, FILE)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, path)


def append(doc, facts, page="", decision=None, via=None):
    """Add facts (dicts with kind, pieces, value[, why, answer]); returns their ids. via="agent": given by a program
    answering for you (decisions_cli.py), not by a click. They still count as your answers (by "user") everywhere."""
    n = max([f["id"] for f in doc["facts"]] + [doc.get("last_id", 0)])  # never reused, even after a retract
    now = int(time.time() * 1000)
    ids = []
    for f in facts:
        if f["kind"] not in KINDS:
            raise ValueError(f"unknown fact kind {f['kind']}")
        n += 1
        own = {k: v for k, v in f.items() if k not in ("id", "at", "by", "page", "decision", "via")}  # the log's to set
        doc["facts"].append({**own, "id": n, "at": now, "by": "user", "page": page,
                             **({"decision": decision} if decision else {}), **({"via": via} if via else {}),
                             "pieces": [str(p) for p in f["pieces"]]})
        ids.append(n)
    doc["last_id"] = n
    return ids


def retract(doc, ids):
    ids = set(ids)
    doc["facts"] = [f for f in doc["facts"] if f["id"] not in ids]


def _anchor(group):
    """A person's id: their scan box with the shortest id (a mask you drew, "m1", only when there is none), so joining
    your mask to someone keeps their id and emoji."""
    return min(group, key=lambda q: (q.startswith("m"), len(q), q))


def partition(facts):
    """{piece: anchor} for every piece a same/different fact names, and the set of anchor pairs kept apart."""
    parent, apart = {}, {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for f in sorted((f for f in facts if f["kind"] in (*JOIN, "different")), key=lambda f: -f["id"]):  # newest first
        a, b = (find(p) for p in f["pieces"][:2])
        if f["kind"] == "different":
            if a != b:
                apart.setdefault(a, set()).add(b); apart.setdefault(b, set()).add(a)
            continue
        if a == b or b in apart.get(a, ()):
            continue  # already one, or a newer "different" keeps them apart
        parent[a] = b
        apart[b] = apart.get(b, set()) | apart.pop(a, set())
        for x in apart[b]:
            apart[x] = {b if y == a else y for y in apart[x]}
    groups = {}
    for x in list(parent):
        groups.setdefault(find(x), []).append(x)
    anchor = {r: _anchor(g) for r, g in groups.items()}
    comp = {x: anchor[find(x)] for x in parent}
    kept = {frozenset((anchor[a], anchor[b])) for a, bs in apart.items() for b in bs if a in anchor and b in anchor}
    return comp, kept


EMOJI_PRESET = palettes.DEFAULT  # the preset a pick without one is taken to be from (palettes.json's "default")


def derive(facts, hidden=(), preset=None):
    """The edits fields that come from the facts: people, emoji (per person: the picks made under `preset`, the emoji
    preset in use), not_face, part_of {piece: piece} (boxes drawn as one face with their person's), and empty merges /
    not_same / shown / attach (people carries all of it). hidden: the scan's own hidden boxes (its baseline)."""
    preset = preset or EMOJI_PRESET
    comp, _ = partition(facts)
    latest = {}
    for f in sorted((f for f in facts if f["kind"] in (*JOIN, "different")), key=lambda f: f["id"]):
        latest[frozenset(f["pieces"][:2])] = f
    part_of = {f["pieces"][0]: f["pieces"][1] for f in latest.values()
               if f["kind"] == "part" and comp.get(f["pieces"][0]) == comp.get(f["pieces"][1])}
    groups = {}
    for x, a in comp.items():
        groups.setdefault(a, set()).add(x)
    attrs = {}  # anchor -> {kind: value}, newest fact touching the group wins
    for f in sorted((f for f in facts if f["kind"] in ATTRS), key=lambda f: f["id"]):
        for a in {comp.get(p, p) for p in f["pieces"]}:
            groups.setdefault(a, {a})
            if f["kind"] == "emoji" and (f.get("preset") or EMOJI_PRESET) != preset:
                attrs.setdefault(a, {}).setdefault("picked", True)  # under another preset: not worn now, but they stay a person (the same id in every preset)
                continue
            attrs.setdefault(a, {})[f["kind"]] = f["value"]
    people, emoji = {}, {}
    for a, g in groups.items():
        at = attrs.get(a, {})
        if len(g) < 2 and not at:
            continue
        people[a] = {"name": at.get("name", ""), "pieces": sorted(g, key=lambda q: (len(q), q)), "show": bool(at.get("show", False))}
        if at.get("emoji"):
            emoji["@" + a] = at["emoji"]
    return {"people": people, "emoji": emoji, "not_face": _not_face(facts, hidden), "part_of": part_of,
            "merges": {}, "not_same": [], "shown": [], "attach": {}}


def _not_face(facts, hidden=()):
    """{piece: why} for the boxes that aren't faces: the scan's hidden ones, then your face facts, newest last."""
    not_face = {str(p): "decision" for p in hidden}
    for f in sorted((f for f in facts if f["kind"] == "face"), key=lambda f: f["id"]):
        p = f["pieces"][0]
        if f["value"]:
            not_face.pop(p, None)
        else:
            not_face[p] = f.get("why") or "you"
    return not_face


# ---- the Decisions page: each question's answer, from the facts (whichever page they came from)
# Faces first: whether a box is a face at all is asked before who it is. On jazz (200 tracks) 47 of 58 "different
# people" answers paired someone with a box later answered "not a face" -- a question that answer settles anyway.
FACE_KINDS = ("track", "blip", "overlap")
# Same-person questions not worth asking (2026-11-13). The 50 s party clip asked 179, 169 answered "different": each
# person walks down their candidates best first, and person 42 got 14 "different" in a row (scores 1.24 down to
# 0.56). A pair left apart only means one person's emoji may change partway; they stay covered.
#   ASK_SCORE   below this match score a pair isn't asked. The lowest score any of your "same person" answers has
#               had: 0.82 (jazz), 0.87 (party_clip_c), 1.03 (the party clip).
#   STOP_AFTER  after this many "different people" answers in a row about one person (since their last join), their
#               weaker candidates aren't asked. Replayed on the party clip, jazz and party_clip_c, with both rules:
#               no join missed (2026-11-13).
ASK_SCORE, STOP_AFTER = 0.8, 4
WEAK_RULE = f"rule: weak match (score under {ASK_SCORE}), not asked"
STOP_RULE = f"rule: you said 'different' to this person's last {STOP_AFTER} best matches, so weaker ones aren't asked"
HIDE = (A_NOT_FACE, "dropped: too weak", "covered by the person's box (while they are)", "covered by another box")


def facts_for_answer(d, ans):
    """The facts a decision's answer stands for."""
    p, q = str(d["about"]["piece"]), d["about"].get("person")
    q = None if q is None else str(q)
    k = d["kind"]
    if k == "whose":  # a box you made: whose face it is (a person's box), or someone new (it stays its own person)
        if ans == A_SOMEONE_NEW:
            return [{"kind": "face", "pieces": [p], "value": True, "answer": ans}]
        return [{"kind": "same", "pieces": [p, str(ans)], "answer": ans}]
    if k == "same":
        return [{"kind": "same" if ans == A_SAME else "different", "pieces": [p, q], "answer": ans}]
    if ans == A_NOT_FACE:
        return [{"kind": "face", "pieces": [p], "value": False, "why": "decision", "answer": ans}]
    out = [{"kind": "face", "pieces": [p], "value": True, "answer": ans}]
    if k == "overlap" and ans == A_SAME_FACE:
        out.append({"kind": "part", "pieces": [p, q], "answer": ans})
    if k == "overlap" and ans == A_SOMEONE_ELSE:
        out.append({"kind": "different", "pieces": [p, q], "answer": ans})
    return out


def answers(decisions, facts, hidden=()):
    """The decision records with their answers as the facts stand: a question answered on the Decisions page shows
    that answer; one settled elsewhere (a box marked not a face in Edit, two people joined) shows how; the
    same-person questions are re-picked (one open question per person at a time)."""
    comp, kept = partition(facts)
    gone = set(_not_face(facts, hidden))
    c = lambda x: comp.get(str(x), str(x))  # noqa: E731
    latest = {}  # (kind, key) -> newest fact
    by_decision = collections.defaultdict(list)  # decision id -> its facts, in log order
    for f in facts:
        if f.get("decision") is not None:
            by_decision[f["decision"]].append(f)
    for f in sorted(facts, key=lambda f: f["id"]):
        if f["kind"] == "face":
            latest[("face", f["pieces"][0])] = f
        elif f["kind"] in (*JOIN, "different"):
            latest[("pair", frozenset(f["pieces"][:2]))] = f
    out = []
    for d in decisions:
        d = {**d, "answer": d.get("rule_answer", d.get("answer")), "by": d.get("rule_by", d.get("by"))}
        p, q = str(d["about"]["piece"]), d["about"].get("person")
        mine = by_decision.get(d["id"])
        if mine:
            d["answer"], d["by"] = mine[-1].get("answer"), "user"
            if mine[-1].get("via"):
                d["via"] = mine[-1]["via"]
        elif d["kind"] == "whose":
            pass  # only its own answer settles it
        elif d["kind"] == "same":
            pf = latest.get(("pair", frozenset((p, str(q)))))
            if pf:
                d["answer"], d["by"] = A_SAME if pf["kind"] in JOIN else A_DIFFERENT, "user"
        else:
            ff = latest.get(("face", p))
            pf = latest.get(("pair", frozenset((p, str(q))))) if q is not None else None
            if d["kind"] == "overlap" and pf:
                d["answer"], d["by"] = A_SAME_FACE if pf["kind"] in JOIN else A_SOMEONE_ELSE, "user"
            elif ff:
                d["answer"], d["by"] = (A_FACE if d["kind"] != "overlap" else d["answer"]) if ff["value"] else A_NOT_FACE, "user"
                if d["kind"] == "overlap" and ff["value"] and d["answer"] in HIDE:
                    d["answer"], d["by"] = None, None  # shown again, but whose face is still the question
        out.append(d)
    # same-person questions not answered directly: settled by the people as they stand, else one open per person,
    # and only once both boxes are known to be faces (faces first)
    face_open = {str(d["about"]["piece"]) for d in out if d["kind"] in FACE_KINDS and d["answer"] is None}
    busy = set()
    same = [d for d in out if d["kind"] == "same"]
    # person -> "different people" answers to questions about their candidates (they're the grey box) since a join
    run, about = collections.Counter(), {d["id"]: str(d["about"]["person"]) for d in same}
    for f in sorted(facts, key=lambda f: f["id"]):
        who = about.get(f.get("decision"))
        if who is not None and f["kind"] in (*JOIN, "different"):
            run[c(who)] = run[c(who)] + 1 if f["kind"] == "different" else 0
    for d in same:
        if d["by"] == "user":
            continue
        a, b = c(d["about"]["person"]), c(d["about"]["piece"])
        if d.get("depth_apart") and a != b:
            d["answer"], d["by"] = A_DIFFERENT, d["depth_apart"]  # the scan's depth rule (MOJIMASK_DEPTH)
            continue
        if {str(d["about"]["person"]), str(d["about"]["piece"])} & gone:
            d["answer"], d["by"] = A_NOT_FACE, "rule: one of the two isn't a face (no emoji), so nobody to join"
        elif a == b:
            d["answer"], d["by"] = A_JOINED, "rule: your other answers make them one person"
        elif frozenset((a, b)) in kept:
            d["answer"], d["by"] = A_KEPT_APART, "rule: you said these two are different people"
        elif d["evidence"].get("score", 0) < ASK_SCORE:
            d["answer"], d["by"] = A_DIFFERENT, WEAK_RULE
        elif run[a] >= STOP_AFTER:
            d["answer"], d["by"] = A_DIFFERENT, STOP_RULE
        else:
            d["answer"], d["by"] = A_OPEN, None
    pending = collections.Counter(str(x) for d in same if d["answer"] == A_OPEN for x in (d["about"]["person"], d["about"]["piece"]))
    for d in out:  # a face question's weight: the same-person questions its answer can settle (asked first)
        if d["kind"] in FACE_KINDS:
            d["weight"] = pending.get(str(d["about"]["piece"]), 0)
    for d in sorted((d for d in same if d["answer"] == A_OPEN), key=lambda d: -d["evidence"].get("score", 0)):
        a, b = c(d["about"]["person"]), c(d["about"]["piece"])
        if {str(d["about"]["person"]), str(d["about"]["piece"])} & face_open:
            d["answer"], d["by"] = A_WAITING, "rule: asked once we know both boxes are faces"
        elif a in busy or b in busy:
            d["answer"], d["by"] = A_WAITING, "rule: asked once the open question about this person is answered"
        else:
            d["answer"] = None
            busy |= {a, b}
    return out
