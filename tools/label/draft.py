"""Claude drafts census frames the way the label tool would, for a person to check later.

Same pre-fill and carry-forward as label.html (fused detector heads at 0.15; class and person carried from the
last labelled frame by camera-moved overlap or nearness; "not" remembered from every earlier frame; people not
found again become dashed guesses). Claude then decides each open item by eye.

  python tools/label/draft.py PROJECT NAME build T     # draft frame T, draw it for review
  python tools/label/draft.py PROJECT NAME auto START END STEP   # rules only, no judgement (then triage.py)
  python tools/label/draft.py PROJECT NAME apply T '{"3": "x", "5": "p14", "7": "del", ...}'
       actions per item: f/h/t/x (class face/head/tiny/not), pN (person N), new (a new person), del,
       o (toggle mostly hidden), keep (accept as drafted). Faces/heads left without a person become new people.

Drafted frames are saved in out/truth/NAME.json under "drafted" (never "checked": only a person confirms), and
every box Claude decided is marked "by": "claude". Review images go to out/label_draft/.
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mojimask import paths  # noqa: E402
from mojimask import track as T  # noqa: E402

PREFILL = [("yolov5m_face_tiled", 0.15), ("retinaface_r50_2x", 0.15), ("pose_head", 0.15)]
OUT = os.path.join(ROOT, "out", "label_draft")


def motion(mats, a, b):
    M = np.eye(3)
    if b >= a:
        for u in range(a + 1, b + 1):
            M = np.vstack([mats[u], [0, 0, 1]]) @ M
        return M[:2]
    return np.linalg.inv(np.vstack([motion(mats, b, a), [0, 0, 1]]))[:2]


def warp(bb, M):
    cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    k = np.sqrt(abs(np.linalg.det(M[:, :2])))
    x, y = M @ np.array([cx, cy, 1.0])
    hw, hh = (bb[2] - bb[0]) / 2 * k, (bb[3] - bb[1]) / 2 * k
    return [x - hw, y - hh, x + hw, y + hh]


class Draft:
    def __init__(self, proj, name):
        self.proj, self.name = proj, name
        self.path = paths.truth(name)
        if os.path.exists(self.path):
            self.d = json.load(open(self.path))
        else:  # a clip with no labels yet
            import glob
            im = cv2.imread(sorted(glob.glob(f"{proj}/frames/*.jpg"))[0])
            meta = json.load(open(f"{proj}/meta.json"))
            self.d = {"w": im.shape[1], "h": im.shape[0], "n": len(glob.glob(f"{proj}/frames/*.jpg")), "fps": meta.get("fps"),
                      "from": os.path.basename(proj.rstrip("/")), "space": meta.get("space"), "frames": {}, "checked": [], "drafted": []}
        self.W, self.H = self.d["w"], self.d["h"]
        self.mats = np.load(f"{proj}/mats_v2.npy")
        spec = [(n, th) for n, th in PREFILL if os.path.exists(f"{proj}/dets/{n}.json")]
        self.pre = [[(list(b), s) for b, s, src in f] for f in T.fused_frames(f"{proj}/dets", spec)]

    def px(self, b):
        return [b[0] * self.W, b[1] * self.H, b[2] * self.W, b[3] * self.H]

    def labelled(self):
        return sorted(set(self.d["checked"]) | set(self.d.get("drafted", [])))

    def frame(self, t):
        return [dict(o, b=self.px(o["b"])) for o in self.d["frames"][str(t)]]

    def build(self, t):
        earlier = [x for x in self.labelled() if x < t]
        if not earlier:  # the first labelled frame: just the pre-fill
            return None, [{"b": b, "c": "face" if s >= 0.5 else "?", "p": None, "src": "prefill", "s": round(s, 2)} for b, s in self.pre[t]]
        prev_t = max(earlier)
        prev = self.frame(prev_t)
        M = motion(self.mats, prev_t, t)
        pre = [{"b": b, "c": "face" if s >= 0.5 else "?", "p": None, "src": "prefill", "s": round(s, 2)} for b, s in self.pre[t]]
        moved = [warp(a["b"], M) for a in prev]
        pairs = []
        for i, a in enumerate(prev):
            for j, b in enumerate(pre):
                o = T.iou(moved[i], b["b"])
                sz = max(moved[i][2] - moved[i][0], 8)
                dc = np.hypot((moved[i][0] + moved[i][2] - b["b"][0] - b["b"][2]) / 2, (moved[i][1] + moved[i][3] - b["b"][1] - b["b"][3]) / 2) / sz
                r = (b["b"][2] - b["b"][0]) / sz
                if (o > 0.1 or dc < 0.8) and 0.6 < r < 1.6:
                    pairs.append((o - 0.5 * dc, i, j))
        ui, uj = set(), set()
        for _, i, j in sorted(pairs, reverse=True):
            if i in ui or j in uj:
                continue
            ui.add(i)
            uj.add(j)
            pre[j].update(c=prev[i]["c"], p=prev[i]["p"], o=prev[i].get("o", 0))
            if prev[i]["c"] == "not" and pre[j]["s"] >= 0.8:  # a confident detection where a "not" was: decide again
                pre[j]["c"] = "?"
        for t0 in self.labelled():  # "not" remembered from every earlier frame
            if t0 >= t:
                continue
            M0 = motion(self.mats, t0, t)
            for a in self.frame(t0):
                if a["c"] == "not":
                    w = warp(a["b"], M0)
                    for j, b in enumerate(pre):
                        if j not in uj and b["c"] == "?" and T.iou(w, b["b"]) > 0.3:
                            b["c"] = "not"
                            uj.add(j)
        for i, a in enumerate(prev):
            if i not in ui and a["p"] is not None and a["c"] != "not":
                pre.append({"b": moved[i], "c": a["c"], "p": a["p"], "src": "carried", "o": a.get("o", 0)})
        # the person's own earlier decisions on this frame (saved but not confirmed): keep their class calls
        if str(t) in self.d["frames"] and t not in self.labelled():
            for o in self.frame(t):
                for b in pre:
                    if T.iou(o["b"], b["b"]) > 0.6 and o["c"] in ("not", "head", "tiny") and b["c"] in ("?", "face"):
                        b["c"] = o["c"]
                        b["from_person"] = True
        return prev_t, pre

    def render(self, t, items, prev_t):
        img = cv2.imread(f"{self.proj}/frames/{t:05d}.jpg")
        k = 2.0
        big = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        for i, b in enumerate(items):
            x0, y0, x1, y1 = [int(v * k) for v in b["b"]]
            col = {"?": (0, 165, 255), "not": (150, 150, 150)}.get(b["c"], (0, 220, 0))
            if b["src"] == "carried":
                col = (255, 255, 0)
            if b["c"] in ("face", "head") and b["p"] is None:
                col = (0, 0, 255)
            cv2.rectangle(big, (x0, y0), (x1, y1), col, 2 if b["c"] != "not" else 1)
            lab = f"{i}" + (f":P{b['p']}" if b["p"] is not None else "") + ("" if b["c"] == "face" else f" {b['c']}") + (" c" if b["src"] == "carried" else "")
            cv2.putText(big, lab, (x0, max(12, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(big, lab, (x0, max(12, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        os.makedirs(OUT, exist_ok=True)
        cv2.imwrite(os.path.join(OUT, f"{self.name}_{t:03d}_frame.jpg"), big)
        # close-ups of everything that needs a decision, and the previous frame's view of carried people
        if prev_t is None:  # the first labelled frame: nothing earlier to show
            prev_img, prev_by_p = None, {}
        else:
            prev_img = cv2.imread(f"{self.proj}/frames/{prev_t:05d}.jpg")
            prev_by_p = {o["p"]: o["b"] for o in self.frame(prev_t) if o["p"] is not None}
        tiles = []
        for i, b in enumerate(items):
            if not (b["c"] == "?" or b["src"] == "carried" or (b["c"] in ("face", "head") and b["p"] is None)):
                continue
            tiles.append(self.tile(img, b["b"], f"{i} now"))
            if b["p"] in prev_by_p:
                tiles.append(self.tile(prev_img, prev_by_p[b["p"]], f"P{b['p']} @{prev_t}"))
        if tiles:
            rows = [np.hstack(tiles[r:r + 8] + [np.zeros_like(tiles[0])] * (8 - len(tiles[r:r + 8]))) for r in range(0, len(tiles), 8)]
            cv2.imwrite(os.path.join(OUT, f"{self.name}_{t:03d}_items.jpg"), np.vstack(rows))
        return len(tiles)

    @staticmethod
    def tile(img, b, text, sz=130):
        c = T.to_cs(b)
        s = int(max(c[2], 8) * 1.8)
        pad = cv2.copyMakeBorder(img, s, s, s, s, cv2.BORDER_CONSTANT)
        y, x = max(0, int(c[1])), max(0, int(c[0]))
        cr = pad[y:y + 2 * s, x:x + 2 * s].copy()
        if cr.size == 0 or min(cr.shape[:2]) < 2:  # the box is outside the picture
            cr = np.zeros((sz, sz, 3), np.uint8)
        cr = cv2.resize(cr, (sz, sz), interpolation=cv2.INTER_CUBIC)
        h = c[2] / 2 * sz / (2 * s)
        cv2.rectangle(cr, (int(sz / 2 - h), int(sz / 2 - h)), (int(sz / 2 + h), int(sz / 2 + h)), (255, 255, 255), 1)
        cv2.putText(cr, text, (3, sz - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
        cv2.putText(cr, text, (3, sz - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        return cr

    def pending_path(self, t):
        return os.path.join(OUT, f"{self.name}_{t:03d}_pending.json")

    def apply(self, t, acts):
        items = json.load(open(self.pending_path(t)))["items"]
        people = [o["p"] for f in self.d["frames"].values() for o in f if o.get("p") is not None]
        nxt = max(people + [0]) + 1
        out = []
        for i, b in enumerate(items):
            acts_i = acts.get(str(i), "keep").split(",")
            if "del" in acts_i:
                continue
            b = dict(b)
            for a in acts_i:
                b = self._act(b, a, nxt)
                if a == "new":
                    nxt += 1
            if b["c"] == "?":
                raise SystemExit(f"item {i} still undecided")
            out.append(b)
        return self._finish(t, out, nxt)

    def _act(self, b, a, nxt):
        if a in ("f", "h", "t", "x"):
            b["c"] = {"f": "face", "h": "head", "t": "tiny", "x": "not"}[a]
            if b["c"] == "not":
                b["p"] = None
            b["by"] = "claude"
        elif a.startswith("p"):
            b["p"] = int(a[1:])
            if b["c"] in ("?", "not"):
                b["c"] = "face"
            b["by"] = "claude"
        elif a == "new":
            b["p"] = nxt
            if b["c"] in ("?", "not"):
                b["c"] = "face"
            b["by"] = "claude"
        elif a == "o":
            b["o"] = 0 if b.get("o") else 1
            b["by"] = "claude"
        return b

    def _finish(self, t, out, nxt):
        for b in sorted(out, key=lambda b: b["b"][0]):
            if b["c"] in ("face", "head", "tiny") and b["p"] is None:
                b["p"], nxt = nxt, nxt + 1
                b["by"] = "claude"
        self.d["frames"][str(t)] = [{"b": [round(b["b"][0] / self.W, 5), round(b["b"][1] / self.H, 5), round(b["b"][2] / self.W, 5), round(b["b"][3] / self.H, 5)],
                                     "c": b["c"], "p": b["p"], "src": b["src"], **({"o": 1} if b.get("o") else {}),
                                     **({"by": b["by"]} if b.get("by") else {})} for b in out]
        self.d["drafted"] = sorted(set(self.d.get("drafted", [])) | {t})
        tmp = self.path + ".tmp"
        json.dump(self.d, open(tmp, "w"))
        os.replace(tmp, self.path)
        return out


def auto(D, start, end, step):
    """Draft frames by rules only, no judgement: confident detections are faces (carried person or a new one),
    weak ones become "not a head" (the frame card's strip, where a person taps any real one), people not found
    again are kept as carried guesses. Every rule decision is marked "by": "rules"."""
    os.makedirs(OUT, exist_ok=True)
    for t in range(start, end + 1, step):
        if t in D.labelled():
            continue
        prev_t, items = D.build(t)
        json.dump({"prev": prev_t, "items": items}, open(D.pending_path(t), "w"))
        acts = {str(i): "x" for i, b in enumerate(items) if b["c"] == "?"}
        out = D.apply(t, acts)
        W, H = D.W, D.H
        earlier = [x for x in D.labelled() if x < t][-2:]
        keep = []
        for o in D.d["frames"][str(t)]:
            if o.get("by") == "claude":
                o["by"] = "rules"
            if o["src"] == "carried":
                bx = [o["b"][0] * W, o["b"][1] * H, o["b"][2] * W, o["b"][3] * H]
                if T.inside_frac(T.to_cs(bx), W, H) < 0.3:
                    continue  # left the picture
                carried_before = [any(x["p"] == o["p"] and x["src"] == "carried" for x in D.d["frames"][str(e)]) for e in earlier]
                if len(carried_before) == 2 and all(carried_before):
                    continue  # no detection for 3 census frames in a row: gone or hidden
            keep.append(o)
        D.d["frames"][str(t)] = keep
        print(f"frame {t}: {len(out)} boxes, {sum(1 for b in out if b['c'] == 'not')} not, people {len({b['p'] for b in out if b['p'] is not None})}")
    tmp = D.path + ".tmp"
    json.dump(D.d, open(tmp, "w"))
    os.replace(tmp, D.path)


def main():
    proj, name, cmd, t = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    D = Draft(proj, name)
    if cmd == "auto":
        return auto(D, t, int(sys.argv[5]), int(sys.argv[6]))
    if cmd == "build":
        prev_t, items = D.build(t)
        json.dump({"prev": prev_t, "items": items}, open(D.pending_path(t), "w"))
        n = D.render(t, items, prev_t)
        for i, b in enumerate(items):
            flag = "?" if b["c"] == "?" else ("carried" if b["src"] == "carried" else ("NEW" if b["c"] in ("face", "head") and b["p"] is None else ""))
            if flag or b.get("from_person"):
                print(f"  {i:>2} {b['c']:<5} P{b['p']} {flag} s={b.get('s')} box={[int(v) for v in b['b']]}" + (" (your class call)" if b.get("from_person") else ""))
        print(f"frame {t} from {prev_t}: {len(items)} boxes, {n} close-ups -> out/label_draft/{name}_{t:03d}_*.jpg")
    elif cmd == "apply":
        out = D.apply(t, json.loads(sys.argv[5]))
        print(f"frame {t} drafted: {len(out)} boxes, people {sorted(b['p'] for b in out if b['p'] is not None)}")


if __name__ == "__main__":
    main()
