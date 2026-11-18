"""A project server for the native app: mojimask/review_server.py, unchanged, plus the draw records the native app
draws from (native/docs/porting.md: the UI first, over the Python engine; drawing is local).

    python project_server.py VIDEO DETDIR --workdir PROJECT --original VIDEO --port N --dets ...   (review_server's args)

The native app's sidecar (crates/mm-engine/src/python/sidecar.py) starts this instead of `-m mojimask.review_server`,
so each open project still has exactly one server, the one that owns its edits.json and facts.json (.server.lock).
Everything review_server serves is served as before; added (GET, JSON):

  /api/native/setup              the drawing constants (gpu_draw's TEX, SLICES, SIGMA_S/R; depth.FACE_ZONE, track.FACE_DY,
                                 OCC_SOFT), where the emoji eyes table, the brand and its font are, the edits' revision
  /api/native/records?from=&to=&style=&wait=&strict=
                                 frames [from, to): one FrameRecord each (mm_core::cover::FrameRecord, the format of
                                 native/tools/parity/draw_ref.py), computed exactly as the export decides them:
                                 review_server.render_setup (resolve, emoji per person, shown people, lighting, head
                                 pose, portrait, saturation, depth occlusion), track.draw_order, Occlusion.plan.
                                 style: the edits' cover_style when not given. wait=1: wait for the lighting estimate
                                 (the export does; the preview draws flat until it is ready, as /api/frame does).
                                 Each record's "near" is the depth guide (float32 mh x mw, base64) when a cover is cut.
                                 "grain" (lighting's frame grain) is left 0: the app measures it on the frame it draws.
  /api/native/export_frame?t=&style=&mark=&preset=&fit=&strict=   (preset=none: not fitted)
                                 PNG: frame t as Python's export draws it (video_io's display frame, track.draw_frame with
                                 render_setup(wait_light=True), formats.fit_frame, track.watermark): the parity reference
                                 for the native Editor and export (native/tools/sidecar/parity.py)

Records are cached per (edits, style, light ready), like review_server's PREVIEW.
"""
import base64
import hashlib
import json
import os
import sys
import threading
import time

HERE_ = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE_)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.argv = ["review_server.py", *sys.argv[1:]]

from mojimask import review_server as RS  # noqa: E402  (parses the args, takes the project's lock, loads it)
from mojimask import track as T  # noqa: E402
from mojimask import depth as DP  # noqa: E402
from mojimask import lighting as LT  # noqa: E402
from mojimask import portrait as PT  # noqa: E402
from mojimask import head_pose as HP  # noqa: E402
from mojimask import gpu_draw as G  # noqa: E402
from urllib.parse import parse_qs, urlsplit  # noqa: E402

_CACHE = {"key": None, "pf": None, "kw": None, "style": None, "recs": {}}
_LOCK = threading.RLock()


def _setup(style, wait, strict=False):
    edits = RS.load_edits()
    style = style if style in T.STYLES else (edits.get("cover_style") or "emoji")
    if strict:
        style = "solid"  # (render_setup: higher-risk mode covers whole heads, solid, 1.3x)
    key = hashlib.md5(json.dumps([edits, style, strict, RS.LIGHT["obj"] is not None, bool(wait)], sort_keys=True).encode()).hexdigest()
    with _LOCK:
        if _CACHE["key"] != key:
            pf, kw = RS.render_setup(edits, {"style": style, "strict": "1" if strict else ""}, wait_light=bool(wait))
            _CACHE.update(key=key, pf=pf, kw=kw, style=style, recs={}, rev=RS.DOC["rev"])
        return _CACHE["pf"], _CACHE["kw"], _CACHE["style"], _CACHE["recs"], key


def record(t, pf, kw, style):
    """Frame t's record, as draw_ref.py writes it (without the Python-side check fields)."""
    occl = kw["occlusion"]
    light = kw["light"]
    items = [x for x in T.draw_order(pf[t], t, occl) if not (x[3:] and x[3] in kw["shown"])]
    near, occ = occl.plan(t, items, kw["scale"], (RS.H, RS.W, 3)) if occl is not None else (None, {})
    covers = []
    photos = {}
    for i, (tid, c, kind, *rest) in enumerate(items):
        piece = rest[0] if rest else tid
        rec = {"tid": str(tid), "piece": str(piece), "c": [float(v) for v in c[:3]]}
        if style in ("emoji", "portrait"):
            ch = (kw["portrait"] or PT.PORTRAIT_KEY) if style == "portrait" else T.emoji_for(tid, kw["overrides"])
            rec["key"] = ch
            hp = (kw["head_pose"] or {}).get(piece, {}).get(t)
            if hp is not None:
                rec["pose"] = [float(hp[0]), float(hp[1]), int(hp[2])]
            if style == "portrait":
                rec["own_pose"] = list(map(float, PT.own_pose(ch)))
                p = PT._path(ch)
                if p:
                    photos[ch] = p
            if light is not None:
                k, g, amb = light.L.auto(piece, t)
                rec["light_auto"] = {"key": list(map(float, k)), "grad": list(map(float, g)), "amb": list(map(float, amb))}
                rec["vel_raw"] = list(map(float, light.velocity(piece, t)))
        oc = occ.get(i)
        if oc is not None:
            rec["occ"] = {k2: float(v) for k2, v in oc.items()}
        covers.append(rec)
    fr = {"t": t, "style": style, "scale": float(kw["scale"]), "seed": float(t), "soft": float(DP.OCC_SOFT),
          "saturation": float(kw["saturation"]), "light": None, "mw": 1, "mh": 1, "near": None, "covers": covers,
          "photos": photos}
    if near is not None:
        fr["mw"], fr["mh"] = int(near.shape[1]), int(near.shape[0])
        import numpy as np
        fr["near"] = base64.b64encode(np.ascontiguousarray(near, np.float32).tobytes()).decode()
    if light is not None:
        fr["light"] = {"settings": {k2: v for k2, v in light.st.items()}, "film": list(map(float, light.L.film(t))),
                       "grain": 0.0}
    return fr


def records(lo, hi, style, wait, strict=False):
    pf, kw, style, cache, key = _setup(style, wait, strict)
    out = []
    for t in range(max(0, lo), min(RS.N, hi)):
        r = cache.get(t)
        if r is None:
            r = cache[t] = record(t, pf, kw, style)
        out.append(r)
    return {"key": key, "rev": RS.DOC["rev"], "style": style, "n": RS.N,
            "light_ready": RS.LIGHT["obj"] is not None, "records": out}


def setup():
    from mojimask import about
    edits = RS.load_edits()
    return {"TEX": G.TEX, "SLICES": G.SLICES, "SIGMA_S": G.SIGMA_S, "SIGMA_R": G.SIGMA_R,
            "FACE_ZONE": list(map(float, DP.FACE_ZONE)), "FACE_DY": float(T.FACE_DY), "OCC_SOFT": DP.OCC_SOFT,
            "CANVAS": HP.CANVAS, "eyes": os.path.join(RS.WEB, "emoji_eyes.json"), "brand": about.BRAND,
            "font_file": about.FONT_FILE, "name": about.NAME, "W": RS.W, "H": RS.H, "n": RS.N, "fps": RS.FPS,
            "rev": RS.DOC["rev"], "style": edits.get("cover_style") or "emoji", "depth": RS.occlusion() is not None,
            "light_ready": RS.LIGHT["obj"] is not None, "portrait_photo": PT._path(PT.PORTRAIT_KEY)}


def export_frame(t, style, mark, preset, fit, strict=False):
    """Frame t exactly as review_server.do_render -> track.render draws it (lossless)."""
    import cv2
    from mojimask import formats
    from mojimask import video_io
    edits = RS.load_edits()
    opts = {"style": style or edits.get("cover_style") or "emoji", "strict": "1" if strict else ""}
    pf, kw = RS.render_setup(edits, opts, wait_light=True)
    o = RS.export_settings({"preset": "original" if preset == "none" else preset, "fit": fit})
    scale, overrides = kw.pop("scale"), kw.pop("overrides")
    for u, f in enumerate(video_io.frames(RS.A.original or RS.A.video, RS.VMETA, color="display")):
        if u == t:
            f = T.draw_frame(f, t, pf[t], {}, scale, overrides, False, kw["shown"], kw["light"], kw["style"],
                             kw["head_pose"], kw["portrait"], kw["saturation"], kw["occlusion"])
            if preset != "none":  # none: the drawn frame itself (what the Editor shows), not fitted to an output size
                f = formats.fit_frame(f, o["width"], o["height"], o["fit"])
            if mark:
                f = T.watermark(f)
            return cv2.imencode(".png", f)[1].tobytes()
    return None


class Handler(RS.Handler):
    def _get(self):
        u = urlsplit(self.path)
        if u.path == "/api/native/setup":
            return self._json(setup())
        if u.path == "/api/native/records":
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            lo = RS._int(q.get("from", 0))
            hi = RS._int(q.get("to", RS.N))
            return self._json(records(lo, hi, q.get("style"), q.get("wait") in ("1", "true"), q.get("strict") in ("1", "true")))
        if u.path == "/api/native/export_frame":
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            png = export_frame(RS._int(q.get("t", 0)), q.get("style"), q.get("mark") in ("1", "true"),
                               q.get("preset") or "original", q.get("fit") or "fill", q.get("strict") in ("1", "true"))
            if png is None:
                return self._json({"error": "no such frame"}, 404)
            return self._send(png, "image/png")
        return super()._get()


if __name__ == "__main__":
    os.chdir(RS.HERE)
    t0 = time.time()
    RS.state(RS.load_edits())
    print(f"warm-up {time.time() - t0:.0f}s (native project server: draw records at /api/native/records)", flush=True)
    threading.Thread(target=RS.refollow_covers, daemon=True).start()
    print(f"review tool: http://localhost:{RS.A.port}", flush=True)
    RS.ThreadingHTTPServer(("127.0.0.1", RS.A.port), Handler).serve_forever()
