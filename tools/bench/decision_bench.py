"""How many decisions a clip asks of you, and how many the scan's rules get wrong -- with a log of past runs.

Scans a scratch copy of a test clip (out/bench/<clip>/; detection is cached there, the decide step always runs
again with today's code), then plays the Decisions page through: each open question, in the page's order, is
answered by an oracle and the next ones re-picked from the facts, as the page does. Answers change which questions
come next, so the count is only meaningful with the same oracle answering every run.

The truth: the clip's census when it has one (tools/truth/<clip>.json, else out/truth/<clip>.json; made with tools/label/: every face boxed
by hand on every 4th frame, checked; boxes slide between those frames), else the old position grade below. Faces
left uncovered are counted on the census's checked frames only.

Oracles:
  gt      the truth above. With only the old position grade (out/probe/gt_<clip>.json, exported from the old app;
          the tool that made it is retired): each box takes the label of the truth box it overlaps most (frame by
          frame, majority). "P<id>" a person, "NF:<why>" not a face, no match = not a face. Its identities are weak: the grader rarely joined pieces of one person (they didn't
          have to), so "different people" from it is often wrong. Its faces are weak too: it is the covers the user
          accepted in the old app, held through gaps and drawn by hand, and by eye (2026-11-13) many of its "face"
          boxes are hands, a sousaphone bell or blur. Trust the question counts; check rule errors by eye.
  answers the user's own answers in another project of the same clip (--answers PROJECT): the decide step must
          give the same boxes as that project (as refresh.py checks); questions they never answered fall back to
          gt, or are counted as unknown.

What is scored, against the oracle:
  asked        questions answered (by kind): the cost to you
  rule errors  questions the scan's rules settled without asking whose rule answer the oracle disagrees with, by
               rule, split by harm: a face left uncovered (the worst), a wrong join, a missed join, a needless cover
  result       after the replay: truth face-frames with no cover, people split (emoji changes), people mixed

    python tools/bench/decision_bench.py jazz [--oracle gt|answers] [--answers PROJECT] [--detect] [--note TEXT]
    python tools/bench/decision_bench.py jazz --no-scan       (score the last scan again: scoring changes)
    python tools/bench/decision_bench.py jazz --depth --sheets (with depth; picture sheets to look at)
    python tools/bench/decision_bench.py --history [clip]

Each run appends a line to tools/bench/data/decisions_history.jsonl (numbers only, no pictures).
"""
import argparse
import re
import collections
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(os.path.dirname(HERE))  # the checkout: its code is what fingerprint() hashes
ROOT = CODE_ROOT  # where the data is (test_clips/, out/): swap_bench points it elsewhere
sys.path.insert(0, CODE_ROOT); sys.path.insert(0, HERE)
from mojimask import facts as FX  # noqa: E402
from mojimask import paths  # noqa: E402

HISTORY = os.path.join(HERE, "data", "decisions_history.jsonl")
# clips with answers of the user's own to replay (a project of the same video; out/ is local to this machine)
# (party_clip_c / IMG_1043 have answers too, but no census yet: left out on purpose, 2026-11-13)
# (falls back to the newest out/projects/<clip>_* with answers when the named folder isn't on this machine)
ANSWERS = {"jazz": "out/projects/jazz_20261112-193341"}
# clips whose frames may be looked at (picture sheets): public footage only. The party clips are real guests.
VIEWABLE = {"jazz"}
REPLAY_CAP = 5000  # questions answered in one replay before it gives up (a loop: the same question keeps coming back)
IOU, INSIDE = 0.3, 0.7
FACE = ("track", "blip", "overlap")
RANK = {"whose": 0, "track": 1, "overlap": 1, "blip": 1, "same": 2}  # the Decisions page's order (decisions.html)


# ---------------------------------------------------------------- scan
def answers_project(clip):
    """The answered project to replay for a clip (relative to ROOT): ANSWERS[clip] when it's on this machine, else the
    newest out/projects/<clip>_* that has answers (facts.json), else None."""
    named = ANSWERS.get(clip)
    if named and os.path.isdir(os.path.join(ROOT, named)):
        return named
    pdir = os.path.join(ROOT, "out", "projects")
    found = sorted(f for f in (os.listdir(pdir) if os.path.isdir(pdir) else [])
                   if f.startswith(f"{clip}_") and os.path.exists(os.path.join(pdir, f, "facts.json")))
    if named:
        print(f"note: {named} isn't here; " + (f"using out/projects/{found[-1]}" if found else "no answered project found"),
              file=sys.stderr)
    return os.path.join("out", "projects", found[-1]) if found else None


def scan_into(src, proj):
    """Scan src (a video or a photo) into proj: detection only the first time (it's cached there), the decide step
    again with today's code; the old result and any answers are thrown away first."""
    for f in ("proto_tracks.json", "edits.json", "facts.json"):
        if os.path.exists(os.path.join(proj, f)):
            os.remove(os.path.join(proj, f))
    if not os.path.exists(os.path.join(proj, "dets", "pose_head.json")):
        from mojimask import pipeline
        pipeline.run(src, proj, report=lambda *a: None)
    else:
        from mojimask.scan import newalgo
        newalgo.run(proj)


def scan(clip, detect=False, rescan=True):
    proj = os.path.join(ROOT, "out", "bench", clip)
    video = next(v for v in (os.path.join(ROOT, "test_clips", f"{clip}.mp4"), os.path.join(ROOT, f"{clip}.mp4"))
                 if os.path.exists(v))
    if detect and os.path.isdir(proj):
        shutil.rmtree(proj)
    os.makedirs(proj, exist_ok=True)
    t0 = time.time()
    if not rescan and os.path.exists(os.path.join(proj, "proto_tracks.json")):
        return proj, 0.0
    scan_into(video, proj)
    return proj, round(time.time() - t0, 1)


# ---------------------------------------------------------------- truth
def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / u if u > 0 else 0.0


def inside(a, b):
    """Share of box a inside box b."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))


CENSUS_REACH = 2  # frames from a census frame a "not" box is carried (they're things: a violin, a hand, a bell)


def census_frames(path, n=None):
    """The careful truth (tools/label: tools/truth/CLIP.json or out/truth/CLIP.json, every face boxed on checked frames), as
    {t: [[x0, y0, x1, y1, label]]} for every frame (n frames; default the census's own count): "P<person>" a face, "H<person>" a head whose face isn't visible
    (no cover needed), "NF" not a head. Between two checked frames a person's box slides from one to the other; a
    person in only one of them, and "not" boxes, are carried CENSUS_REACH frames."""
    c = json.load(open(path))
    n = c["n"] if n is None else n
    checked = sorted(int(t) for t in c["checked"])
    lab = {"face": "P", "tiny": "P", "head": "H", "not": "NF"}
    at = {t: [(b["b"], lab[b["c"]] + ("" if b["c"] == "not" else str(b["p"]))) for b in c["frames"][str(t)]] for t in checked}
    out = {}
    for t in range(n):
        if t in at:
            out[t] = [[*b, l] for b, l in at[t]]
            continue
        a = max((u for u in checked if u < t), default=None)
        z = min((u for u in checked if u > t), default=None)
        boxes = []
        pa = {l: b for b, l in at[a] if l != "NF"} if a is not None else {}
        pz = {l: b for b, l in at[z] if l != "NF"} if z is not None else {}
        for l in set(pa) | set(pz):
            if l in pa and l in pz:
                w = (t - a) / (z - a)
                boxes.append([pa[l][i] * (1 - w) + pz[l][i] * w for i in range(4)] + [l])
            elif l in pa and t - a <= CENSUS_REACH:
                boxes.append([*pa[l], l])
            elif l in pz and z - t <= CENSUS_REACH:
                boxes.append([*pz[l], l])
        near = min((u for u in (a, z) if u is not None), key=lambda u: abs(u - t))
        if abs(near - t) <= CENSUS_REACH:
            boxes += [[*b, l] for b, l in at[near] if l == "NF"]
        out[t] = boxes
    return out, set(checked)


class Truth:
    """Which boxes are faces, and whose. The careful census (out/truth/CLIP.json) when the clip has one, else the
    old position grade (out/probe/gt_CLIP.json: covers accepted in the old app, a weak face truth)."""
    def __init__(self, clip, proj):
        meta = json.load(open(os.path.join(proj, "meta.json")))
        self.W, self.H = meta["width"], meta["height"]
        self.pt = json.load(open(os.path.join(proj, "proto_tracks.json")))
        census = paths.truth(clip, ROOT)
        if os.path.exists(census):
            self.frames, self.scored = census_frames(census, meta["frames"])
            self.source = "census"
        elif os.path.exists(os.path.join(ROOT, "out", "probe", f"gt_{clip}.json")):
            g = json.load(open(os.path.join(ROOT, "out", "probe", f"gt_{clip}.json")))
            self.frames = {int(t): f for t, f in g["frames"].items()}
            self.scored, self.source = set(self.frames), "old grade"
        else:  # no truth: only the user's answers (--oracle answers), and no face scores
            self.frames, self.scored, self.source = {}, set(), "none"
        self._lab = {}

    def frac(self, c):
        h = c[2] / 2
        return [(c[0] - h) / self.W, (c[1] - h) / self.H, (c[0] + h) / self.W, (c[1] + h) / self.H]

    def match(self, t, b):
        """The truth label of box b (fractions) at frame t: a truth face it overlaps (IoU >= IOU), else a head, else a
        box called not a face, else a truth face it sits inside (a chin or hair box of that face), else None. Faces
        come first: the old grade's "NF:dup" boxes are second boxes on a face that has its own."""
        gs = self.frames.get(t, [])
        for want in ("P", "H", "N"):
            best = max(((iou(b, g[:4]), g[4]) for g in gs if g[4][0] == want), default=(0, None))
            if best[0] >= IOU:
                return best[1]
        ins = max(((inside(b, g[:4]), g[4]) for g in gs if g[4][0] == "P"), default=(0, None))
        return ins[1] if ins[0] >= INSIDE else None

    def label(self, piece):
        """(label, share of frames): "P<id>", "NF", or None (no truth box there: not a face)."""
        piece = str(piece)
        if piece not in self._lab:
            tr = self.pt["tracks"].get(piece)
            c = collections.Counter()
            obs = tr["obs_c"] if tr else {}
            for t, o in obs.items():
                m = self.match(int(t), self.frac(o))
                c["NF" if m and m.startswith("NF") else m] += 1
            lab, n = c.most_common(1)[0] if c else (None, 0)
            self._lab[piece] = (lab, n / max(1, len(obs)))
        return self._lab[piece]

    def people_in(self, piece):
        """The truth people a scan track's own boxes land on, frame by frame ({id: frames}, at least 2 frames each):
        more than one is a track that is two people (the graph merged them by their faces). label() takes the
        majority, so this is the only place that shows it."""
        tr = self.pt["tracks"].get(str(piece)) or {}
        c = collections.Counter()
        for t, o in tr.get("obs_c", {}).items():
            m = self.match(int(t), self.frac(o))
            if m and m[0] == "P":
                c[m[1:]] += 1
        return {k: v for k, v in c.items() if v >= 2}

    def person(self, piece):
        """Whose head this box is (a face or the back of a head), or None."""
        lab = self.label(piece)[0]
        return lab[1:] if lab and lab[0] in "PH" else None

    def is_face(self, piece):
        lab = self.label(piece)[0]
        return bool(lab) and lab.startswith("P")

    def answer(self, d):
        p, q = d["about"]["piece"], d["about"].get("person")
        k = d["kind"]
        if k in ("track", "blip"):
            return FX.A_FACE if self.is_face(p) else FX.A_NOT_FACE
        if k == "overlap":
            if not self.is_face(p):
                return FX.A_NOT_FACE
            return FX.A_SAME_FACE if self.person(p) == self.person(q) else FX.A_SOMEONE_ELSE
        if k == "same":
            a = self.person(p)
            return FX.A_SAME if a is not None and a == self.person(q) else FX.A_DIFFERENT
        if k == "whose":
            return FX.A_SOMEONE_NEW
        raise ValueError(k)


class Answers:
    """The user's own answers in another scan of this clip, carried over by position: each box of this scan takes the
    answered project's box it overlaps most (mean IoU over its frames >= MAP_IOU), so a scan change that renumbers or
    reshapes boxes keeps most answers. Anything they never said falls back to `rest` (counted). (Letting the grade
    overrule their "not a face" was tried and undone, 2026-11-13: by eye, 7 of those 14 boxes were the sousaphone
    bell, hands or a shirt front; the grade's old covers had drifted there.)"""
    MAP_IOU = 0.5

    def __init__(self, src, truth):
        final = json.load(open(os.path.join(src, "facts.json")))["facts"]
        pt = json.load(open(os.path.join(src, "proto_tracks.json")))
        self.comp, self.kept = FX.partition(final)
        self.nf = set(FX.derive(final, pt.get("hidden", []))["not_face"])
        self.said = {f["pieces"][0] for f in final if f["kind"] == "face"}
        self.paired = {frozenset(f["pieces"][:2]) for f in final if f["kind"] in (*FX.JOIN, "different")}
        self.rest, self.fallbacks, self.used = truth, 0, 0
        theirs = collections.defaultdict(list)  # frame -> [(piece, box)]
        for k, tr in pt["tracks"].items():
            for t, o in tr["obs_c"].items():
                theirs[int(t)].append((k, truth.frac(o)))
        self.map = {}
        for k, tr in truth.pt["tracks"].items():
            sc = collections.Counter()
            for t, o in tr["obs_c"].items():
                b = truth.frac(o)
                for k2, b2 in theirs[int(t)]:
                    sc[k2] += iou(b, b2)
            if sc:
                k2, v = sc.most_common(1)[0]
                if v / len(tr["obs_c"]) >= self.MAP_IOU:
                    self.map[k] = k2
        self.same_boxes = all(self.map.get(k) == k for k in truth.pt["tracks"])

    def _answer(self, d):
        p, q = self.map.get(str(d["about"]["piece"])), d["about"].get("person")
        q = None if q is None else self.map.get(str(q))
        c = lambda x: self.comp.get(x, x)  # noqa: E731
        k = d["kind"]
        if p is None or (q is None and d["about"].get("person") is not None):
            return None
        gone = lambda x: x in self.nf and x in self.said  # noqa: E731  (they said: not a face)
        if k != "same" and gone(p):
            return FX.A_NOT_FACE
        if k == "same":
            if gone(p) or gone(q):
                return FX.A_DIFFERENT
            if c(p) == c(q) and (p in self.comp):
                return FX.A_SAME
            if frozenset((c(p), c(q))) in self.kept or frozenset((p, q)) in self.paired:
                return FX.A_DIFFERENT
            return None
        if k == "overlap":
            if frozenset((p, q)) in self.paired or (p in self.comp and q in self.comp):
                return FX.A_SAME_FACE if c(p) == c(q) else FX.A_SOMEONE_ELSE
            return None
        if p in self.said:
            return FX.A_FACE
        return None

    # with no truth to fall back on, a question they never answered gets the answer that keeps covers on
    CAUTIOUS = {"track": FX.A_FACE, "blip": FX.A_FACE, "overlap": FX.A_SOMEONE_ELSE, "same": FX.A_DIFFERENT, "whose": FX.A_SOMEONE_NEW}

    def known(self, d):
        return self._answer(d) is not None or self.rest.source != "none"

    def answer(self, d):
        a = self._answer(d)
        if a is None:
            self.fallbacks += 1
            return self.CAUTIOUS[d["kind"]] if self.rest.source == "none" else self.rest.answer(d)
        self.used += 1
        return a


# ---------------------------------------------------------------- replay
def decisions(proj):
    return [d for f in sorted(os.listdir(proj)) if f.startswith("decisions_") and f.endswith(".json")
            for d in json.load(open(os.path.join(proj, f)))["decisions"]]


def replay(ds, hidden, oracle):
    doc, asked, log = {"facts": []}, collections.Counter(), []
    while True:
        out = FX.answers(ds, doc["facts"], hidden)
        op = sorted((d for d in out if d["answer"] is None),
                    key=lambda d: (RANK[d["kind"]], -d.get("weight", 0), -d["evidence"].get("frames", 0)))
        if not op or sum(asked.values()) > REPLAY_CAP:
            if op:
                print(f"warning: replay stopped after {REPLAY_CAP} questions, {len(op)} still open", file=sys.stderr)
            return doc["facts"], asked, log, out
        d = op[0]
        a = oracle.answer(d)
        FX.append(doc, FX.facts_for_answer(d, a), "bench", d["id"])
        asked[d["kind"]] += 1
        log.append({"id": d["id"], "answer": a, "evidence": d["evidence"], "why": d.get("why")})


# ---------------------------------------------------------------- scoring
class Covers:
    """The covers after the replay: every box not hidden and not answered "not a face", per frame."""
    def __init__(self, truth, facts):
        pt = truth.pt
        self.nf = set(FX.derive(facts, pt.get("hidden", []))["not_face"])
        self.comp = FX.partition(facts)[0]
        self.at = collections.defaultdict(list)
        for k, tr in pt["tracks"].items():
            if k not in self.nf:
                for t, c in tr["cover"].items():
                    self.at[int(t)].append(truth.frac(c))
        self.any = collections.defaultdict(list)  # every box the scan made, hidden or not
        for tr in pt["tracks"].values():
            for t, c in tr["cover"].items():
                self.any[int(t)].append(truth.frac(c))

    @staticmethod
    def hit(g, boxes):
        return any(iou(g, b) >= IOU or inside(g, b) >= INSIDE for b in boxes)

    def uncovered(self, truth, piece):
        """Frames where this box's truth face has no cover at all."""
        tr = truth.pt["tracks"].get(str(piece)) or {"obs_c": {}}
        n = 0
        for t, o in tr["obs_c"].items():
            m = truth.match(int(t), truth.frac(o))
            if m and m[0] == "P":
                g = next(g[:4] for g in truth.frames[int(t)] if g[4] == m)
                n += not self.hit(g, self.at[int(t)])
        return n


# a rule's answer, and the truth answers that agree with it
AGREES = {"kept: covered": (FX.A_FACE, FX.A_SAME_FACE, FX.A_SOMEONE_ELSE), FX.A_FACE: (FX.A_FACE,),
          FX.A_SAME: (FX.A_SAME,), FX.A_DIFFERENT: (FX.A_DIFFERENT,),
          FX.A_NOT_FACE: (FX.A_NOT_FACE,), "dropped: too weak": (FX.A_NOT_FACE,),
          "covered by another box": (FX.A_NOT_FACE,), "covered by the person's box (while they are)": (FX.A_NOT_FACE, FX.A_SAME_FACE),
          "left as it is": (FX.A_SOMEONE_ELSE,), FX.A_SAME_FACE: (FX.A_SAME_FACE,), FX.A_SOMEONE_ELSE: (FX.A_SOMEONE_ELSE,)}
HIDES = (FX.A_NOT_FACE, "dropped: too weak", "covered by another box", "covered by the person's box (while they are)")


def rule_errors(final, truth, oracle, cov):
    """Questions the scan's rules settled without asking, against the oracle, by rule. A wrong answer's harm:
      face uncovered    hid a face, and nothing else covers it in some frames (the worst)
      hidden, covered   hid a face, but another cover hides it in every frame (no harm)
      needless cover    covered a box the truth calls not a face (or, with the census, any box on a checked frame
                        that no truth face matches)
      cover, unlabelled covered a box with no truth there (between census frames; likely not a face)
      double cover      left a part of someone's face as its own box (two emoji on one face)
      missed / wrong join   same-person rules"""
    by_rule, wrong = collections.defaultdict(collections.Counter), []
    for d in final:
        if d["by"] in (None, "user") or d["answer"] in (FX.A_WAITING, FX.A_JOINED, FX.A_KEPT_APART, FX.A_OPEN):
            continue
        if d["answer"] == FX.A_NOT_FACE and str(d["by"]).startswith("rule: one of the two"):
            continue  # follows from a face answer
        if hasattr(oracle, "known") and not oracle.known(d):
            continue  # nobody knows (no truth, never answered)
        o, said = oracle.answer(d), d["answer"]
        rule = re.sub(r"\d+", "N", str(d["by"]).split("(")[0].strip())[:60]
        if o in AGREES.get(said, (said,)):
            by_rule[rule]["right"] += 1
            continue
        frames = 0
        if d["kind"] == "same":
            h = "missed join" if o == FX.A_SAME else "wrong join"
        elif said in HIDES:
            frames = cov.uncovered(truth, d["about"]["piece"])
            h = "face uncovered" if frames else "hidden, covered"
        elif o == FX.A_NOT_FACE:  # the grader's NF box, or no truth box there at all (they may never have seen it)
            obs = (truth.pt["tracks"].get(str(d["about"]["piece"])) or {}).get("obs_c", {})
            checked = any(int(t) in truth.scored for t in obs)  # on a frame where every face is boxed
            h = "needless cover" if truth.label(d["about"]["piece"])[0] == "NF" or checked else "cover, unlabelled"
        elif said == "left as it is" and o == FX.A_SAME_FACE:
            h = "double cover"
        else:
            h = "other"
        by_rule[rule][h] += 1
        q = d["about"].get("person")
        wrong.append({"id": d["id"], "rule": d["by"], "said": said, "truth": o, "harm": h, "frames": frames,
                      "label": [truth.label(d["about"]["piece"])[0], truth.label(q)[0] if q is not None else None]})
    return {k: dict(v) for k, v in by_rule.items()}, wrong


def result(truth, cov):
    """After the replay: truth face-frames the scan never boxed, those it boxed but left uncovered, and people
    split (one truth person across several, so their emoji changes) or mixed."""
    total = missed = uncovered = 0
    who = collections.Counter()
    for t, gs in truth.frames.items():
        if t not in truth.scored:  # only frames a person checked (the census), not the slid boxes between
            continue
        for g in gs:
            if g[4][0] != "P":
                continue
            total += 1
            if not Covers.hit(g[:4], cov.any[t]):
                missed += 1
            elif not Covers.hit(g[:4], cov.at[t]):
                uncovered += 1; who[g[4]] += 1
    person = collections.defaultdict(set)  # final person -> truth labels
    for k in truth.pt["tracks"]:
        if k not in cov.nf and truth.is_face(k):
            person[cov.comp.get(k, k)].add(truth.person(k))
    of = collections.defaultdict(set)
    for p, labs in person.items():
        for lab in labs:
            of[lab].add(p)
    return {"face_frames": total, "not_found_by_scan": missed, "uncovered_frames": uncovered, "uncovered_people": len(who),
            "people_split": sum(len(v) - 1 for v in of.values()), "people_mixed": sum(1 for v in person.values() if len(v) > 1),
            "tracks_mixed": sum(1 for k in truth.pt["tracks"] if k not in cov.nf and len(truth.people_in(k)) > 1),
            "people": len(person), "truth_people": len(of)}


# ---------------------------------------------------------------- run / history
def fingerprint():
    h = hashlib.sha1()
    for d in (CODE_ROOT, os.path.join(CODE_ROOT, "mojimask"), os.path.join(CODE_ROOT, "mojimask", "scan"), HERE):
        for f in sorted(os.listdir(d)):
            if f.endswith(".py"):
                h.update(open(os.path.join(d, f), "rb").read())
    return h.hexdigest()[:10]


def run(clip, oracle_name="gt", answers=None, detect=False, note="", rescan=True):
    import fcntl
    os.makedirs(os.path.join(ROOT, "out", "bench"), exist_ok=True)
    lock = open(os.path.join(ROOT, "out", "bench", f"{clip}.lock"), "w")
    try:  # two runs at once would rescan the same scratch project under each other
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another decision_bench run on {clip} is going; wait for it")
    proj, secs = scan(clip, detect, rescan)
    truth = Truth(clip, proj)
    oracle = truth
    if oracle_name == "answers":
        oracle = Answers(answers, truth)
    hidden = truth.pt.get("hidden", [])
    ds = decisions(proj)
    facts, asked, log, final = replay(ds, hidden, oracle)
    cov = Covers(truth, facts)
    rules, wrong = rule_errors(final, truth, oracle, cov)
    harms = collections.Counter(w["harm"] for w in wrong)
    rec = {"at": time.strftime("%Y-%m-%d %H:%M"), "clip": clip, "oracle": oracle_name, "truth": truth.source, "code": fingerprint(),
           "depth": os.environ.get("MOJIMASK_DEPTH", "off"), "note": note, "scan_s": secs,
           "decisions": len(ds), "asked": sum(asked.values()), "asked_by_kind": dict(asked),
           "rule_wrong": dict(harms), "uncovered_by_rules": sum(w["frames"] for w in wrong), "result": result(truth, cov),
           **({"truncated": True} if sum(asked.values()) > REPLAY_CAP else {}),
           **({"answers": {"from": os.path.basename(answers), "same_boxes": oracle.same_boxes, "boxes_mapped": len(oracle.map),
                           "used": oracle.used, "fell_back_to_gt": oracle.fallbacks}} if oracle_name == "answers" else {})}
    if rescan:
        os.makedirs(os.path.dirname(HISTORY), exist_ok=True)
        with open(HISTORY, "a") as f:
            f.write(json.dumps(rec) + "\n")
    json.dump({"record": rec, "rules": rules, "wrong": wrong, "asked": log},
              open(os.path.join(proj, "bench_last.json"), "w"), indent=1, default=str)
    return rec, rules, wrong


def summary(r):
    """One line per run: when, settings, questions, harms, coverage, note."""
    res, w = r["result"], r["rule_wrong"]
    k = r["asked_by_kind"]
    kinds = " ".join(f"{n} {k[n]}" for n in ("track", "blip", "overlap", "same", "whose") if k.get(n))
    return (f"{r['at']}  {r['clip']:8s} {r.get('truth', 'old grade'):9s} depth {'on ' if r['depth'] != 'off' else 'off'}  "
            f"asked {r['asked']:3d} ({kinds})  needless covers {w.get('needless cover', 0) + w.get('cover, unlabelled', 0):2d}  "
            f"faces uncovered by rules {w.get('face uncovered', 0)}  uncovered {res['uncovered_frames']}/{res['face_frames']}  "
            f"never boxed {res['not_found_by_scan']}  people split {res['people_split']} mixed {res['people_mixed']} (tracks {res.get('tracks_mixed', 0)})"
            + (f"  | {r['note']}" if r["note"] else ""))


def history(clip=None):
    rows = [json.loads(x) for x in open(HISTORY)] if os.path.exists(HISTORY) else []
    for r in rows:
        if not clip or r["clip"] == clip:
            print(summary(r))


def sheets(clip):
    """Picture sheets of the last run, for checking by eye: every question it asked, and every rule answer the truth
    disagrees with (by harm), to out/bench/CLIP/sheets/. Each row is one decision: its box in green at a few frames,
    the person it's asked against in yellow, truth faces in blue and truth not-faces in red. Only for footage that may
    be looked at (VIEWABLE; jazz: a public-domain film); party clips are real guests."""
    import cv2
    proj = os.path.join(ROOT, "out", "bench", clip)
    out = os.path.join(proj, "sheets")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    truth = Truth(clip, proj)
    last = json.load(open(os.path.join(proj, "bench_last.json")))
    ds = {d["id"]: d for d in decisions(proj)}
    tile_px, rows_per = 230, 6

    def tile(t, c, other, cap):
        f = cv2.imread(os.path.join(proj, "frames", f"{t:05d}.jpg"))
        H, W = f.shape[:2]
        half = max(45, c[2] * 1.6)
        x0, y0 = int(max(0, c[0] - half)), int(max(0, c[1] - half))
        x1, y1 = int(min(W, c[0] + half)), int(min(H, c[1] + half))
        im = f.copy()
        for g in truth.frames.get(t, []):
            col = (255, 120, 0) if g[4][0] == "P" else (0, 0, 255) if g[4][0] == "N" else (200, 200, 200)
            cv2.rectangle(im, (int(g[0] * W), int(g[1] * H)), (int(g[2] * W), int(g[3] * H)), col, 1)
        for box, col in ((other, (0, 220, 255)), (c, (0, 255, 0))):
            if box is not None:
                cv2.rectangle(im, (int(box[0] - box[2] / 2), int(box[1] - box[2] / 2)),
                              (int(box[0] + box[2] / 2), int(box[1] + box[2] / 2)), col, 1)
        im = cv2.resize(im[y0:y1, x0:x1], (tile_px, tile_px), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(im, (0, 0), (tile_px, 14), (0, 0, 0), -1)
        cv2.putText(im, f"t{t} {cap}", (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
        return im

    def row(d, lines):
        tr = truth.pt["tracks"][str(d["about"]["piece"])]
        q = d["about"].get("person")
        qt = truth.pt["tracks"].get(str(q)) if q is not None else None
        ts = sorted(map(int, tr["obs_c"]))
        pick = [ts[i] for i in sorted(set(np.linspace(0, len(ts) - 1, min(4, len(ts))).astype(int)))]
        tiles = [tile(t, tr["obs_c"][str(t)], qt["cover"].get(str(t)) if qt else None,
                      str(truth.match(t, truth.frac(tr["obs_c"][str(t)])))) for t in pick]
        tiles += [np.zeros((tile_px, tile_px, 3), np.uint8)] * (4 - len(tiles))
        head = np.zeros((tile_px, 280, 3), np.uint8)
        e = d["evidence"]
        lines = lines + [f"{k}={e[k]}" for k in ("frames", "clear_face_frames", "median_detector", "best_face_check",
                                                  "detector_score", "face_check", "size", "score", "median_overlap") if k in e]
        for i, ln in enumerate(lines[:14]):
            cv2.putText(head, str(ln)[:46], (4, 16 + 15 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        return np.hstack([head] + tiles)

    groups = collections.defaultdict(list)
    for x in last["asked"]:
        d = ds[x["id"]]
        groups[f"asked_{d['kind']}"].append(row(d, [d["id"], f"truth answered: {x['answer']}"]))
    for w in last["wrong"]:
        d = ds[w["id"]]
        groups["wrong_" + re.sub(r"\W+", "_", w["harm"]).strip("_")].append(
            row(d, [d["id"], f"rule said: {w['said']}", f"truth: {w['truth']}", str(w["rule"])[:46], str(w["rule"])[46:92]]))
    for name, rows in groups.items():
        for i in range(0, len(rows), rows_per):
            cv2.imwrite(os.path.join(out, f"{name}_{i // rows_per}.jpg"), np.vstack(rows[i:i + rows_per]),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out, {k: len(v) for k, v in groups.items()}


def main(argv=None):
    """The command line (argv: its arguments, default sys.argv[1:])."""
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?")
    ap.add_argument("--oracle", choices=("gt", "answers"), help="default: gt (the truth) when the clip has a census, else "
                    "your answers when it has some (ANSWERS), else gt")
    ap.add_argument("--answers", help="the answered project (default: ANSWERS[clip])")
    ap.add_argument("--detect", action="store_true", help="scan from the video again (detection too)")
    ap.add_argument("--depth", action="store_true", help="with the depth model on (MOJIMASK_DEPTH=anything-small)")
    ap.add_argument("--note", default="")
    ap.add_argument("--no-scan", action="store_true", help="score the last scan again (no decide step; not logged)")
    ap.add_argument("--sheets", action="store_true", help="draw picture sheets of the run (out/bench/CLIP/sheets/); "
                    f"only for clips that may be looked at ({', '.join(sorted(VIEWABLE))}) unless --viewable")
    ap.add_argument("--viewable", action="store_true", help="this clip's frames may be looked at (public footage, no guests)")
    ap.add_argument("--json", action="store_true", help="print the full record and the per-rule table")
    ap.add_argument("--history", action="store_true")
    a = ap.parse_args(argv)
    if a.depth:
        os.environ["MOJIMASK_DEPTH"] = "anything-small"
    if a.history:
        history(a.clip)
    elif not a.clip:
        ap.error("which clip? (e.g. jazz)")
    else:
        if a.sheets and a.clip not in VIEWABLE and not a.viewable:
            ap.error(f"--sheets draws the clip's faces: only for {', '.join(sorted(VIEWABLE))} (or say --viewable)")
        src = a.answers or answers_project(a.clip)
        census = os.path.exists(paths.truth(a.clip, ROOT))
        oracle = a.oracle or ("gt" if census or not src else "answers")
        if oracle == "answers" and not src:
            ap.error(f"no answered project of {a.clip} on this machine: pass --answers PROJECT")
        rec, rules, wrong = run(a.clip, oracle, src and os.path.join(ROOT, src), a.detect, a.note, not a.no_scan)
        if a.json:
            print(json.dumps(rec, indent=1))
        print(summary(rec))
        print("rules (answers, and where the truth disagrees):")
        for k, v in sorted(rules.items(), key=lambda x: -sum(x[1].values())):
            bad = ", ".join(f"{h} {n}" for h, n in v.items() if h != "right")
            print(f"  {sum(v.values()):4d}  {k}" + (f"   <- {bad}" if bad else ""))
        if a.sheets:
            out, n = sheets(a.clip)
            print(f"sheets: {out}  {n}")


if __name__ == "__main__":
    main()
